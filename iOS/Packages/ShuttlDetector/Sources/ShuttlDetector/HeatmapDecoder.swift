import Foundation

/// Sub-pixel refinement of a decoded heatmap peak (decode_heatmap ``refine``).
public enum PeakRefinement: String, Sendable, Codable {
    /// Weighted centroid of the peak's above-half-maximum blob (default).
    case centroid
    /// Separable 3×3 log-parabola; plateaus fall back to the centroid.
    case quadratic
    /// Integer peak pixel.
    case none
}

/// One decoded peak in heatmap px.
public struct HeatmapPeak: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var score: Double
    public init(x: Double, y: Double, score: Double) { self.x = x; self.y = y; self.score = score }
}

/// Port of detector.py ``decode_heatmap`` (+ ``_blob``, ``_plateau_extent``, ``_refine_quadratic``).
///
/// Top-k peaks of an (H, W) heatmap in [0,1] → [(x, y, score)] in heatmap px, score-sorted.
/// Peaks are local maxima (5×5) above `minScore`, taken greedily by score. A seed is suppressed
/// if it lies within `nmsRadius` of a kept seed or refined position, or inside a kept peak's
/// above-half-maximum blob — so a plateau yields one peak. score = peak value.
///
/// Holds reusable scratch buffers: create one per detector/thread and reuse it (not thread-safe).
public final class HeatmapDecoder {
    public var k: Int
    public var minScore: Double
    public var nmsRadius: Int
    public var refine: PeakRefinement
    public var refineRadius: Int

    // scratch
    private var peaks: [(idx: Int, v: Float)] = []
    private var wts: [Double] = []
    private var comp: [UInt8] = []          // 0 = not visited, 1 = in seed component, 2 = other
    private var stack: [Int] = []

    private struct Taken { var y0: Int; var x0: Int; var h: Int; var w: Int; var mask: [Bool] }

    public init(k: Int = 3, minScore: Double = 0.1, nmsRadius: Int = 6, refine: PeakRefinement = .centroid,
                refineRadius: Int = 4) {
        self.k = k; self.minScore = minScore; self.nmsRadius = nmsRadius
        self.refine = refine; self.refineRadius = refineRadius
    }

    public func decode(_ hm: [Float], width: Int, height: Int) -> [HeatmapPeak] {
        hm.withUnsafeBufferPointer { decode($0, width: width, height: height) }
    }

    public func decode(_ hm: UnsafeBufferPointer<Float>, width w: Int, height h: Int) -> [HeatmapPeak] {
        precondition(hm.count >= w * h)
        guard w > 0, h > 0, let p = hm.baseAddress else { return [] }
        var mx = -Float.infinity
        for i in 0..<(w * h) where p[i] > mx { mx = p[i] }
        if Double(mx) < minScore { return [] }
        // numpy compares the float32 heatmap with the threshold cast to float32.
        let thr = Float(minScore)
        peaks.removeAll(keepingCapacity: true)
        for y in 0..<h {
            let row = p + y * w
            for x in 0..<w {
                let v = row[x]
                guard v >= thr else { continue }
                // hm >= dilate(hm, 5x5) (image-clipped window)
                var isMax = true
                let ya = max(y - 2, 0), yb = min(y + 2, h - 1), xa = max(x - 2, 0), xb = min(x + 2, w - 1)
                outer: for yy in ya...yb {
                    let r = p + yy * w
                    for xx in xa...xb where r[xx] > v { isMax = false; break outer }
                }
                if isMax { peaks.append((y * w + x, v)) }
            }
        }
        if peaks.isEmpty { return [] }
        // argsort(-score, kind="stable"): descending score, ties in row-major order.
        peaks.sort { $0.v != $1.v ? $0.v > $1.v : $0.idx < $1.idx }

        var out: [HeatmapPeak] = []
        var kept: [(Double, Double)] = []
        var taken: [Taken] = []
        let r2 = Double(nmsRadius * nmsRadius)
        for (idx, sc) in peaks {
            let y = idx / w, x = idx % w
            var suppressed = false
            for (ky, kx) in kept where (Double(y) - ky) * (Double(y) - ky) + (Double(x) - kx) * (Double(x) - kx) <= r2 {
                suppressed = true; break
            }
            if !suppressed {
                for t in taken {
                    let dy = y - t.y0, dx = x - t.x0
                    if dy >= 0 && dy < t.h && dx >= 0 && dx < t.w && t.mask[dy * t.w + dx] { suppressed = true; break }
                }
            }
            if suppressed { continue }
            var ties = 0
            for yy in max(y - 1, 0)...min(y + 1, h - 1) {
                for xx in max(x - 1, 0)...min(x + 1, w - 1) where p[yy * w + xx] == sc { ties += 1 }
            }
            let plateau = ties > 1
            let radius = refineRadius + (plateau ? plateauExtent(p, w, h, y, x) : 0)
            let (by, bx, bh, bw) = blob(p, w, h, y, x, radius)
            var fx: Double, fy: Double
            if refine == .quadratic && !plateau {
                (fx, fy) = refineQuadratic(p, w, h, y, x)
            } else if refine == .none {
                (fx, fy) = (Double(x), Double(y))
            } else {
                var tot = 0.0, sx = 0.0, sy = 0.0
                for i in 0..<bh {
                    for j in 0..<bw {
                        let v = wts[i * bw + j]
                        tot += v; sx += v * Double(bx + j); sy += v * Double(by + i)
                    }
                }
                fx = sx / tot; fy = sy / tot
            }
            kept.append((Double(y), Double(x)))
            kept.append((fy, fx))
            taken.append(Taken(y0: by, x0: bx, h: bh, w: bw, mask: wts[0..<(bh * bw)].map { $0 > 0 }))
            out.append(HeatmapPeak(x: fx, y: fy, score: Double(sc)))
            if out.count >= k { break }
        }
        return out
    }

    /// `_blob`: weights (hm − peak/2)+ on the 8-connected component of {hm > peak/2} containing
    /// the seed, inside a (2r+1)² box re-centred on the blob centroid (at most twice) when the
    /// blob touches the box edge. Leaves the weights in `wts` (row-major bh × bw).
    private func blob(_ hm: UnsafePointer<Float>, _ w: Int, _ h: Int, _ seedY: Int, _ seedX: Int, _ r: Int)
        -> (y0: Int, x0: Int, bh: Int, bw: Int) {
        var cy = seedY, cx = seedX
        let half = 0.5 * Double(hm[cy * w + cx])
        var y0 = 0, x0 = 0, bh = 0, bw = 0
        for _ in 0..<3 {
            y0 = max(cy - r, 0); x0 = max(cx - r, 0)
            let y1 = min(cy + r + 1, h), x1 = min(cx + r + 1, w)
            bh = y1 - y0; bw = x1 - x0
            let n = bh * bw
            if wts.count < n { wts = [Double](repeating: 0, count: n); comp = [UInt8](repeating: 0, count: n) }
            for i in 0..<bh {
                for j in 0..<bw { wts[i * bw + j] = max(Double(hm[(y0 + i) * w + x0 + j]) - half, 0) }
            }
            keepSeedComponent(bh, bw, cy - y0, cx - x0)
            var touches = false
            if y0 > 0 { for j in 0..<bw where wts[j] > 0 { touches = true; break } }
            if !touches && y1 < h { for j in 0..<bw where wts[(bh - 1) * bw + j] > 0 { touches = true; break } }
            if !touches && x0 > 0 { for i in 0..<bh where wts[i * bw] > 0 { touches = true; break } }
            if !touches && x1 < w { for i in 0..<bh where wts[i * bw + bw - 1] > 0 { touches = true; break } }
            if !touches { break }
            var tot = 0.0, my = 0.0, mx = 0.0
            for i in 0..<bh {
                var rs = 0.0
                for j in 0..<bw { rs += wts[i * bw + j] }
                my += rs * Double(y0 + i)
                tot += rs
            }
            for j in 0..<bw {
                var cs = 0.0
                for i in 0..<bh { cs += wts[i * bw + j] }
                mx += cs * Double(x0 + j)
            }
            let ny = Int((my / tot).rounded(.toNearestOrEven)), nx = Int((mx / tot).rounded(.toNearestOrEven))
            if (ny == cy && nx == cx) || !(wts[(ny - y0) * bw + (nx - x0)] > 0) { break }
            cy = ny; cx = nx
        }
        return (y0, x0, bh, bw)
    }

    /// cv2.connectedComponents(8-conn) + `wts[lab != lab[seed]] = 0` when there are ≥ 2 blobs.
    private func keepSeedComponent(_ bh: Int, _ bw: Int, _ sy: Int, _ sx: Int) {
        let n = bh * bw
        for i in 0..<n { comp[i] = 0 }
        let seed = sy * bw + sx
        if wts[seed] > 0 {
            flood(seed, bh, bw, mark: 1)
            for i in 0..<n where comp[i] != 1 { wts[i] = 0 }
            return
        }
        // Seed is background (only possible with min_score <= 0): Python keeps the single blob,
        // or zeroes everything when there are several.
        var blobs = 0
        for i in 0..<n where wts[i] > 0 && comp[i] == 0 { flood(i, bh, bw, mark: 2); blobs += 1 }
        if blobs > 1 { for i in 0..<n { wts[i] = 0 } }
    }

    private func flood(_ start: Int, _ bh: Int, _ bw: Int, mark: UInt8) {
        stack.removeAll(keepingCapacity: true)
        stack.append(start)
        comp[start] = mark
        while let i = stack.popLast() {
            let y = i / bw, x = i % bw
            for yy in max(y - 1, 0)...min(y + 1, bh - 1) {
                for xx in max(x - 1, 0)...min(x + 1, bw - 1) {
                    let j = yy * bw + xx
                    if comp[j] == 0 && wts[j] > 0 { comp[j] = mark; stack.append(j) }
                }
            }
        }
    }

    /// `_plateau_extent`: Chebyshev extent (px) from the seed of the connected set of pixels tied with it.
    private func plateauExtent(_ hm: UnsafePointer<Float>, _ w: Int, _ h: Int, _ cy: Int, _ cx: Int, cap: Int = 16) -> Int {
        let y0 = max(cy - cap, 0), y1 = min(cy + cap + 1, h), x0 = max(cx - cap, 0), x1 = min(cx + cap + 1, w)
        let bh = y1 - y0, bw = x1 - x0, n = bh * bw
        let v = hm[cy * w + cx]
        if wts.count < n { wts = [Double](repeating: 0, count: n); comp = [UInt8](repeating: 0, count: n) }
        for i in 0..<bh { for j in 0..<bw { wts[i * bw + j] = hm[(y0 + i) * w + x0 + j] >= v ? 1 : 0 } }
        for i in 0..<n { comp[i] = 0 }
        flood((cy - y0) * bw + (cx - x0), bh, bw, mark: 1)
        var ext = 0
        for i in 0..<n where comp[i] == 1 {
            ext = max(ext, abs(i / bw + y0 - cy), abs(i % bw + x0 - cx))
        }
        return ext
    }

    /// `_refine_quadratic`: separable log-parabola (Gaussian) fit on the 3×3 neighbourhood.
    private func refineQuadratic(_ hm: UnsafePointer<Float>, _ w: Int, _ h: Int, _ cy: Int, _ cx: Int) -> (Double, Double) {
        guard 0 < cx && cx < w - 1 && 0 < cy && cy < h - 1 else { return (Double(cx), Double(cy)) }
        func lp(_ y: Int, _ x: Int) -> Double { log(max(Double(hm[y * w + x]), 1e-6)) }
        func off(_ a: Double, _ b: Double, _ c: Double) -> Double {
            let d = a - 2 * b + c
            return d >= 0 ? 0 : min(max(0.5 * (a - c) / d, -0.5), 0.5)
        }
        let c = lp(cy, cx)
        return (Double(cx) + off(lp(cy, cx - 1), c, lp(cy, cx + 1)), Double(cy) + off(lp(cy - 1, cx), c, lp(cy + 1, cx)))
    }
}

/// Convenience wrapper with decode_heatmap's signature (allocates a decoder).
public func decodeHeatmap(_ hm: [Float], width: Int, height: Int, k: Int = 3, minScore: Double = 0.1,
                          nmsRadius: Int = 6, refine: PeakRefinement = .centroid, refineRadius: Int = 4) -> [HeatmapPeak] {
    HeatmapDecoder(k: k, minScore: minScore, nmsRadius: nmsRadius, refine: refine, refineRadius: refineRadius)
        .decode(hm, width: width, height: height)
}
