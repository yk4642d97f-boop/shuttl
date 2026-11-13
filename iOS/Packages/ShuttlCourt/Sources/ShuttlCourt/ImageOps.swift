/// cv2 / numpy image primitives used by court_calibration.py, re-implemented to match
/// OpenCV 4.11 (the backend's cv2) bit-for-bit where it matters:
///   • GaussianBlur(float32, ksize=(0,0), σ) — bit-exact kernel (getGaussianKernelBitExact),
///     separable RowFilter / SymmColumnFilter with NEON fused multiply-adds, BORDER_REFLECT_101;
///   • copyMakeBorder(BORDER_REFLECT) neighbour lookups of the ridge test;
///   • connectedComponentsWithStats(connectivity=8) — only bounding boxes are used;
///   • distanceTransform(DIST_L2, maskSize=3) — OpenCV's 3x3 CHAMFER (a=0.955, b=1.3693,
///     16-bit fixed point), which is what cv2 computes for mask 3 (not the exact EDT);
///   • remap(float32 maps, INTER_LINEAR, BORDER_CONSTANT 0) — coordinates quantised to
///     1/32 px (INTER_TAB_SIZE) with the float bilinear weight table.
import Foundation
import ShuttlCore

/// Single-channel row-major plane.
struct Plane<T: Sendable>: Sendable {
    let w: Int
    let h: Int
    var data: [T]
    init(w: Int, h: Int, data: [T]) { precondition(data.count == w * h); self.w = w; self.h = h; self.data = data }
    init(w: Int, h: Int, fill: T) { self.w = w; self.h = h; data = [T](repeating: fill, count: w * h) }
    @inline(__always) subscript(x: Int, y: Int) -> T {
        get { data[y * w + x] }
        set { data[y * w + x] = newValue }
    }
}

typealias PlaneF = Plane<Float>
typealias PlaneU8 = Plane<UInt8>

enum ImageOps {

    @inline(__always) static func reflect101(_ i: Int, _ n: Int) -> Int {
        if n == 1 { return 0 }
        var i = i
        while i < 0 || i >= n { i = i < 0 ? -i : 2 * n - 2 - i }
        return i
    }

    @inline(__always) static func reflect(_ i: Int, _ n: Int) -> Int {
        var i = i
        while i < 0 || i >= n { i = i < 0 ? -i - 1 : 2 * n - i - 1 }
        return i
    }

    // MARK: whiteness

    /// `_whiteness`: min channel minus half the channel spread (float32; exact half-integers).
    /// Other paint colours (`LineColour`): min(S) − ½·(max(S) − min(S)) − ½·max(others).
    static func whiteness(_ img: ImageU8, colour: LineColour = .white) -> PlaneF {
        let n = img.width * img.height, c = img.channels
        var out = [Float](repeating: 0, count: n)
        img.data.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                if c == 1 {
                    for i in 0..<n { dst[i] = Float(src[i]) }
                    return
                }
                if colour != .white {
                    precondition(c == 3, "colour line models need an RGB image")
                    let S = colour.rgbChannels, O = (0..<3).filter { !S.contains($0) }
                    for i in 0..<n {
                        var mn = src[i * 3 + S[0]], mx = mn
                        for k in S.dropFirst() { let v = src[i * 3 + k]; mn = min(mn, v); mx = max(mx, v) }
                        var mo = src[i * 3 + O[0]]
                        for k in O.dropFirst() { mo = max(mo, src[i * 3 + k]) }
                        let fmn = Float(mn), fmx = Float(mx)
                        dst[i] = (fmn - 0.5 * (fmx - fmn)) - 0.5 * Float(mo)
                    }
                    return
                }
                for i in 0..<n {
                    var mn = src[i * c], mx = mn
                    for k in 1..<c { let v = src[i * c + k]; mn = min(mn, v); mx = max(mx, v) }
                    let fmn = Float(mn), fmx = Float(mx)
                    dst[i] = fmn - 0.5 * (fmx - fmn)
                }
            }
        }
        return PlaneF(w: img.width, h: img.height, data: out)
    }

    // MARK: Gaussian blur

    /// cv::getGaussianKernel(n, sigma, CV_32F) (4.x: getGaussianKernelBitExact, cast to float).
    static func gaussianKernel(n: Int, sigma: Double) -> [Float] {
        precondition(n % 2 == 1 && sigma > 0)
        let scale2X = -0.125 / (sigma * sigma)
        let n2 = (n - 1) / 2
        var values = [Double](repeating: 0, count: n2)
        var sum = 0.0
        var x = 1 - n
        for i in 0..<n2 {
            let t = exp(Double(x * x) * scale2X)
            values[i] = t
            sum += t
            x += 2
        }
        sum *= 2
        sum += 1
        let mul1 = 1 / sum
        var res = [Float](repeating: 0, count: n)
        for i in 0..<n2 {
            let t = values[i] * mul1
            res[i] = Float(t); res[n - 1 - i] = Float(t)
        }
        res[n2] = Float(mul1)
        return res
    }

    /// cv2.GaussianBlur(src, (0, 0), sigma) for CV_32F (ksize = round(8σ+1)|1).
    static func gaussianBlur(_ src: PlaneF, sigma: Double) -> PlaneF {
        let ksize = Int((sigma * 4 * 2 + 1).rounded(.toNearestOrEven)) | 1
        let k = gaussianKernel(n: ksize, sigma: sigma)
        let r = ksize / 2
        let w = src.w, h = src.h
        var tmp = [Float](repeating: 0, count: w * h)
        var ext = [Float](repeating: 0, count: w + 2 * r)
        let xmap = (0..<(w + 2 * r)).map { reflect101($0 - r, w) }
        src.data.withUnsafeBufferPointer { s in
            tmp.withUnsafeMutableBufferPointer { t in
                ext.withUnsafeMutableBufferPointer { e in
                    for y in 0..<h {
                        let row = y * w
                        for i in 0..<(w + 2 * r) { e[i] = s[row + xmap[i]] }
                        for x in 0..<w {
                            var acc = e[x] * k[0]
                            for j in 1..<ksize { acc = acc.addingProduct(e[x + j], k[j]) }   // vfmaq_f32
                            t[row + x] = acc
                        }
                    }
                }
            }
        }
        var out = [Float](repeating: 0, count: w * h)
        let kc = k[r]
        tmp.withUnsafeBufferPointer { t in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<h {
                    let row = y * w
                    let rowsBelow = (1...max(r, 1)).map { reflect101(y + $0, h) * w }
                    let rowsAbove = (1...max(r, 1)).map { reflect101(y - $0, h) * w }
                    for x in 0..<w {
                        var acc = t[row + x] * kc
                        for d in 1...r { acc = acc.addingProduct(t[rowsBelow[d - 1] + x] + t[rowsAbove[d - 1] + x], k[r + d]) }
                        o[row + x] = acc
                    }
                }
            }
        }
        return PlaneF(w: w, h: h, data: out)
    }

    // MARK: ridge strength

    /// `sorted({int(s) for s in np.round(np.geomspace(2, smax, 6))})`
    static func ridgeScales(D: Double) -> [Int] {
        let smax = max(4, Int((0.012 * D).rounded(.toNearestOrEven)))
        let num = 6
        let ls = log10(2.0), le = log10(Double(smax))
        let step = (le - ls) / Double(num - 1)
        var vals = (0..<num).map { pow(10.0, Double($0) * step + ls) }
        vals[0] = 2; vals[num - 1] = Double(smax)
        return Array(Set(vals.map { Int($0.rounded(.toNearestOrEven)) })).sorted()
    }

    /// `_ridge_strength` → (strength_h_test, strength_v_test).
    static func ridgeStrength(_ img: ImageU8, colour: LineColour = .white) -> (PlaneF, PlaneF) {
        let W = gaussianBlur(whiteness(img, colour: colour), sigma: 0.8)
        let w = W.w, h = W.h
        let scales = ridgeScales(D: Double(max(w, h)))
        var bh = [Float](repeating: 0, count: w * h)
        var bv = [Float](repeating: 0, count: w * h)
        W.data.withUnsafeBufferPointer { Wp in
            bh.withUnsafeMutableBufferPointer { BH in
                bv.withUnsafeMutableBufferPointer { BV in
                    for s in scales {
                        let xl = (0..<w).map { reflect($0 - s, w) }, xr = (0..<w).map { reflect($0 + s, w) }
                        for y in 0..<h {
                            let row = y * w
                            let ru = reflect(y - s, h) * w, rd = reflect(y + s, h) * w
                            for x in 0..<w {
                                let c = Wp[row + x]
                                let rh = min(c - Wp[row + xl[x]], c - Wp[row + xr[x]])
                                let rv = min(c - Wp[ru + x], c - Wp[rd + x])
                                if rh > BH[row + x] { BH[row + x] = rh }
                                if rv > BV[row + x] { BV[row + x] = rv }
                            }
                        }
                    }
                }
            }
        }
        return (PlaneF(w: w, h: h, data: bh), PlaneF(w: w, h: h, data: bv))
    }

    // MARK: connected components

    /// `_mask_from_strength`: (strength > minContrast) minus 8-connected components whose
    /// bounding-box extent max(width, height) < minLen. Returns a 0/255 mask.
    static func maskFromStrength(_ strength: PlaneF, minLen: Double, minContrast: Float) -> PlaneU8 {
        let w = strength.w, h = strength.h, n = w * h
        var label = [Int32](repeating: -1, count: n)
        var parent: [Int32] = []
        func find(_ a: Int32) -> Int32 {
            var a = a
            while parent[Int(a)] != a { parent[Int(a)] = parent[Int(parent[Int(a)])]; a = parent[Int(a)] }
            return a
        }
        func union(_ a: Int32, _ b: Int32) {
            let ra = find(a), rb = find(b)
            if ra != rb { if ra < rb { parent[Int(rb)] = ra } else { parent[Int(ra)] = rb } }
        }
        strength.data.withUnsafeBufferPointer { s in
            for y in 0..<h {
                for x in 0..<w where s[y * w + x] > minContrast {
                    var l: Int32 = -1
                    // already-visited 8-neighbours: W, NW, N, NE
                    let nbrs = [(x - 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1)]
                    for (nx, ny) in nbrs where nx >= 0 && ny >= 0 && nx < w {
                        let q = label[ny * w + nx]
                        if q >= 0 { if l < 0 { l = q } else if q != l { union(l, q) } }
                    }
                    if l < 0 { l = Int32(parent.count); parent.append(l) }
                    label[y * w + x] = l
                }
            }
        }
        let nl = parent.count
        var x0 = [Int](repeating: Int.max, count: nl), x1 = [Int](repeating: -1, count: nl)
        var y0 = [Int](repeating: Int.max, count: nl), y1 = [Int](repeating: -1, count: nl)
        for i in 0..<nl { parent[i] = find(Int32(i)) }
        for y in 0..<h {
            for x in 0..<w {
                let l = label[y * w + x]
                if l < 0 { continue }
                let r = Int(parent[Int(l)])
                x0[r] = min(x0[r], x); x1[r] = max(x1[r], x)
                y0[r] = min(y0[r], y); y1[r] = max(y1[r], y)
            }
        }
        var out = PlaneU8(w: w, h: h, fill: 0)
        for i in 0..<n {
            let l = label[i]
            if l < 0 { continue }
            let r = Int(parent[Int(l)])
            let ext = max(x1[r] - x0[r] + 1, y1[r] - y0[r] + 1)
            if Double(ext) >= minLen { out.data[i] = 255 }
        }
        return out
    }

    // MARK: distance transform

    /// cv2.distanceTransform(src, DIST_L2, 3): distance of every non-zero pixel to the
    /// nearest zero pixel with OpenCV's 3x3 chamfer metric (distanceTransform_3x3).
    static func distanceTransform3x3(_ nonzero: (Int) -> Bool, w: Int, h: Int) -> PlaneF {
        let shift = 16
        let hv = UInt32((Float(0.955) * Float(1 << shift)).rounded(.toNearestOrEven))
        let diag = UInt32((Float(1.3693) * Float(1 << shift)).rounded(.toNearestOrEven))
        let initDist: UInt32 = UInt32(Int32.max)
        let distMax: UInt32 = UInt32(Int32.max >> 2)
        let scale: Float = 1 / Float(1 << shift)
        let step = w + 2
        var tmp = [UInt32](repeating: initDist, count: (h + 2) * step)
        var out = [Float](repeating: 0, count: w * h)
        tmp.withUnsafeMutableBufferPointer { t in
            // forward pass
            for i in 0..<h {
                let base = (i + 1) * step + 1
                t[base - 1] = initDist; t[base + w] = initDist
                for j in 0..<w {
                    let p = base + j
                    if !nonzero(i * w + j) { t[p] = 0; continue }
                    var t0 = t[p - step - 1] &+ diag
                    var tt = t[p - step] &+ hv
                    if t0 > tt { t0 = tt }
                    tt = t[p - step + 1] &+ diag
                    if t0 > tt { t0 = tt }
                    tt = t[p - 1] &+ hv
                    if t0 > tt { t0 = tt }
                    t[p] = t0
                }
            }
            // backward pass
            out.withUnsafeMutableBufferPointer { d in
                for i in stride(from: h - 1, through: 0, by: -1) {
                    let base = (i + 1) * step + 1
                    for j in stride(from: w - 1, through: 0, by: -1) {
                        let p = base + j
                        var t0 = t[p]
                        if t0 > hv {
                            var tt = t[p + step + 1] &+ diag
                            if t0 > tt { t0 = tt }
                            tt = t[p + step] &+ hv
                            if t0 > tt { t0 = tt }
                            tt = t[p + step - 1] &+ diag
                            if t0 > tt { t0 = tt }
                            tt = t[p + 1] &+ hv
                            if t0 > tt { t0 = tt }
                            t[p] = t0
                        }
                        t0 = t0 > distMax ? distMax : t0
                        d[i * w + j] = Float(t0) * scale
                    }
                }
            }
        }
        return PlaneF(w: w, h: h, data: out)
    }

    // MARK: remap

    /// cv2.remap(src, X, Y, INTER_LINEAR, BORDER_CONSTANT, 0) for float32 src and maps,
    /// one sample per (X[i], Y[i]).
    @inline(__always)
    static func remapSample(_ s: UnsafeBufferPointer<Float>, w: Int, h: Int, _ X: Float, _ Y: Float) -> Float {
        let ix = Int32(clamping: Int((X * 32).rounded(.toNearestOrEven)))
        let iy = Int32(clamping: Int((Y * 32).rounded(.toNearestOrEven)))
        let sx = Int(Int16(clamping: ix >> 5)), sy = Int(Int16(clamping: iy >> 5))
        let fx = Float(ix & 31) * (1 / 32), fy = Float(iy & 31) * (1 / 32)
        let w0 = (1 - fy) * (1 - fx), w1 = (1 - fy) * fx, w2 = fy * (1 - fx), w3 = fy * fx
        if sx >= 0 && sx < w - 1 && sy >= 0 && sy < h - 1 {
            let p = sy * w + sx
            return bilinear(s[p], s[p + 1], s[p + w], s[p + w + 1], w0, w1, w2, w3)
        }
        if sx >= w || sx + 1 < 0 || sy >= h || sy + 1 < 0 { return 0 }
        @inline(__always) func v(_ x: Int, _ y: Int) -> Float { x >= 0 && x < w && y >= 0 && y < h ? s[y * w + x] : 0 }
        return bilinear(v(sx, sy), v(sx + 1, sy), v(sx, sy + 1), v(sx + 1, sy + 1), w0, w1, w2, w3)
    }

    /// `S0*w0 + S1*w1 + S2*w2 + S3*w3` as clang contracts it on arm64 (-ffp-contract=on).
    @inline(__always)
    static func bilinear(_ a: Float, _ b: Float, _ c: Float, _ d: Float,
                         _ w0: Float, _ w1: Float, _ w2: Float, _ w3: Float) -> Float {
        var t = (b * w1).addingProduct(a, w0)
        t = t.addingProduct(c, w2)
        return t.addingProduct(d, w3)
    }
}
