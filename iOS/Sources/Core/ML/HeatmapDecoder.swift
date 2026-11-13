import Foundation
import CoreGraphics
import Accelerate

// MARK: - FrameGeometry
// Full-resolution upright frame ↔ network canvas, index-affine like WASB's cv2.warpAffine and
// backend detector.FrameGeometry(corner=True):   net = s · full + pad   (pixel indices).
// letterbox: s = min(netW/W, netH/H) on both axes, pad = (net − s·size)/2 (black borders);
// stretch:   independent sx, sy, no pad. For 16:9 frames the two coincide.
// A decoded heatmap peak maps back as  full = (peak + peakOffset − pad) / s,  exactly as the
// backend does with the model's centroid_offset.

struct FrameGeometry: Sendable, Equatable {
    enum Mode: String, Sendable { case letterbox, stretch }

    let width: Int
    let height: Int
    let netWidth: Int
    let netHeight: Int
    let mode: Mode
    let scaleX: Double
    let scaleY: Double
    let padX: Double
    let padY: Double
    /// Learned (content − peak) offset of the network, net px.
    let peakOffset: Double

    init(width: Int, height: Int, netWidth: Int = 512, netHeight: Int = 288,
         mode: Mode = .letterbox, peakOffset: Double = 0) {
        self.width = max(width, 1)
        self.height = max(height, 1)
        self.netWidth = max(netWidth, 1)
        self.netHeight = max(netHeight, 1)
        self.mode = mode
        self.peakOffset = peakOffset
        let fx = Double(self.netWidth) / Double(self.width), fy = Double(self.netHeight) / Double(self.height)
        switch mode {
        case .letterbox:
            let s = min(fx, fy)
            scaleX = s
            scaleY = s
            padX = (Double(self.netWidth) - Double(self.width) * s) / 2
            padY = (Double(self.netHeight) - Double(self.height) * s) / 2
        case .stretch:
            scaleX = fx
            scaleY = fy
            padX = 0
            padY = 0
        }
    }

    var fullSize: CGSize { CGSize(width: width, height: height) }

    /// Full-res pixel → net pixel (where the content lands in the canvas).
    func toNet(_ p: CGPoint) -> CGPoint {
        CGPoint(x: Double(p.x) * scaleX + padX, y: Double(p.y) * scaleY + padY)
    }

    /// Net pixel (content position) → full-res pixel.
    func toFull(_ p: CGPoint) -> CGPoint {
        CGPoint(x: (Double(p.x) - padX) / scaleX, y: (Double(p.y) - padY) / scaleY)
    }

    /// Decoded heatmap peak (net px) → full-res pixel, removing the network's learned offset.
    func peakToFull(x: Double, y: Double) -> CGPoint {
        toFull(CGPoint(x: x + peakOffset, y: y + peakOffset))
    }

    /// Full-res pixel → 0…1 of the frame (backend serialize convention u / w).
    func normalized(_ fullPixel: CGPoint) -> CGPoint {
        CGPoint(x: Double(fullPixel.x) / Double(width), y: Double(fullPixel.y) / Double(height))
    }

    /// Affine that places the frame in the canvas for a renderer sampling at pixel centres
    /// (Core Image, top-left origin): v = s·u + pad + ½ − ½s reproduces net = s·full + pad on
    /// pixel indices, i.e. the same content position WASB was trained with.
    var centreSampledTranslation: (x: Double, y: Double) {
        (padX + 0.5 - 0.5 * scaleX, padY + 0.5 - 0.5 * scaleY)
    }
}

// MARK: - HeatmapPeak

struct HeatmapPeak: Sendable, Equatable {
    /// Heatmap pixel coordinates (pixel-centre convention), sub-pixel.
    var x: Double
    var y: Double
    /// Peak probability, 0…1.
    var score: Float
}

// MARK: - HeatmapDecoder
// Top-k peaks with NMS, found with vDSP_maxvi (no per-pixel Swift loops over the map),
// then a separable 3×3 log-parabola fit (exact for Gaussian blobs). Saturated plateaus
// (all 4-neighbours ≥ 90 % of the peak, e.g. disk-trained heads) fall back to the
// half-max weighted centroid used by the backend (`detector._refine_centroid`).

struct HeatmapDecoder: Sendable {
    enum Refinement: Sendable { case quadratic, centroid, none }

    var minScore: Float = 0.3
    var maxPeaks: Int = 2
    /// Chebyshev radius (heatmap px) suppressed around an accepted peak.
    var nmsRadius: Int = 6
    var refinement: Refinement = .quadratic
    var centroidRadius: Int = 4

    /// Decodes a probability map (0…1), contiguous row-major `width × height`.
    /// The buffer is used as scratch: accepted neighbourhoods are overwritten.
    func decode(_ map: UnsafeMutableBufferPointer<Float>, width: Int, height: Int) -> [HeatmapPeak] {
        let n = width * height
        guard let base = map.baseAddress, map.count >= n, n > 0, maxPeaks > 0 else { return [] }
        var peaks: [HeatmapPeak] = []
        var accepted: [(x: Int, y: Int)] = []
        var attempts = 0
        while peaks.count < maxPeaks, attempts < maxPeaks + 8 {
            attempts += 1
            var value: Float = 0
            var index: vDSP_Length = 0
            vDSP_maxvi(base, 1, &value, &index, vDSP_Length(n))
            guard value.isFinite, value >= minScore else { break }
            let cx = Int(index) % width, cy = Int(index) / width
            // On the rim of an accepted peak's suppression box → its shoulder, not a new peak.
            if accepted.contains(where: { max(abs($0.x - cx), abs($0.y - cy)) <= nmsRadius + 1 }) {
                Self.suppress(base, width: width, height: height, cx: cx, cy: cy, radius: max(1, nmsRadius / 2))
                continue
            }
            let (fx, fy) = refine(base, width: width, height: height, cx: cx, cy: cy)
            peaks.append(HeatmapPeak(x: fx, y: fy, score: min(value, 1)))
            accepted.append((cx, cy))
            Self.suppress(base, width: width, height: height, cx: cx, cy: cy, radius: nmsRadius)
        }
        return peaks
    }

    private func refine(_ m: UnsafeMutablePointer<Float>, width w: Int, height h: Int,
                        cx: Int, cy: Int) -> (Double, Double) {
        guard refinement != .none, cx > 0, cy > 0, cx < w - 1, cy < h - 1 else { return (Double(cx), Double(cy)) }
        @inline(__always) func v(_ x: Int, _ y: Int) -> Double { Double(m[y * w + x]) }
        if refinement == .quadratic {
            func lg(_ x: Int, _ y: Int) -> Double { log(max(v(x, y), 1e-6)) }
            let c = lg(cx, cy)
            let l = lg(cx - 1, cy), r = lg(cx + 1, cy), u = lg(cx, cy - 1), d = lg(cx, cy + 1)
            let dxx = l - 2 * c + r, dyy = u - 2 * c + d
            // Log-curvature of a Gaussian is −1/σ²; |·| < 0.02 means σ > 7 px or a saturated
            // plateau — the 3×3 fit is ill-conditioned there, use the centroid instead.
            if dxx < -0.02, dyy < -0.02 {
                return (Double(cx) + min(0.5, max(-0.5, 0.5 * (l - r) / dxx)),
                        Double(cy) + min(0.5, max(-0.5, 0.5 * (u - d) / dyy)))
            }
        }
        return centroid(m, width: w, height: h, cx: cx, cy: cy)
    }

    private func centroid(_ m: UnsafeMutablePointer<Float>, width w: Int, height h: Int,
                          cx: Int, cy: Int) -> (Double, Double) {
        let r = centroidRadius
        let half = 0.5 * Double(m[cy * w + cx])
        var sw = 0.0, sx = 0.0, sy = 0.0
        for y in max(cy - r, 0)...min(cy + r, h - 1) {
            for x in max(cx - r, 0)...min(cx + r, w - 1) {
                let wt = Double(m[y * w + x]) - half
                guard wt > 0 else { continue }
                sw += wt; sx += wt * Double(x); sy += wt * Double(y)
            }
        }
        return sw > 0 ? (sx / sw, sy / sw) : (Double(cx), Double(cy))
    }

    private static func suppress(_ m: UnsafeMutablePointer<Float>, width w: Int, height h: Int,
                                 cx: Int, cy: Int, radius r: Int) {
        let x0 = max(cx - r, 0), x1 = min(cx + r, w - 1)
        var fill: Float = -1
        for y in max(cy - r, 0)...min(cy + r, h - 1) {
            vDSP_vfill(&fill, m + y * w + x0, 1, vDSP_Length(x1 - x0 + 1))
        }
    }

    // MARK: Activation

    /// True when the map cannot be probabilities (model exported without its sigmoid).
    static func looksLikeLogits(_ map: UnsafeBufferPointer<Float>) -> Bool {
        guard let base = map.baseAddress, !map.isEmpty else { return false }
        var lo: Float = 0, hi: Float = 0
        vDSP_minv(base, 1, &lo, vDSP_Length(map.count))
        vDSP_maxv(base, 1, &hi, vDSP_Length(map.count))
        return lo < -0.01 || hi > 1.01
    }

    /// In-place logistic sigmoid with vForce.
    static func applySigmoid(_ map: UnsafeMutableBufferPointer<Float>) {
        guard let base = map.baseAddress, !map.isEmpty else { return }
        var n = Int32(map.count)
        var lo: Float = -60, hi: Float = 60
        vDSP_vclip(base, 1, &lo, &hi, base, 1, vDSP_Length(map.count))
        vDSP_vneg(base, 1, base, 1, vDSP_Length(map.count))
        vvexpf(base, base, &n)
        var one: Float = 1
        vDSP_vsadd(base, 1, &one, base, 1, vDSP_Length(map.count))
        vvrecf(base, base, &n)
    }
}
