import Foundation
import ShuttlCore

/// The learned hit detector's 51 per-frame features: a transcription of
/// `research/hitnet/common.py`'s `derive(raw, fps)`, pinned against the golden fixtures by
/// `HitNetParityTests`.
///
/// The Python builds these from a (T,16) float64 matrix whose NaN entries mean "missing"
/// (never 0 — 0 is a legal shuttle x and a legal box width). The Swift side takes the same
/// (T,16) matrix, flattened row-major, so a caller can feed it from a fixture or from the
/// tracking stages without a conversion in between.
///
/// Every `NaN` decision here is deliberate and listed in `coreml/README.md`'s NaN table; a port
/// that fills a NaN with 0 early gets plausible-looking features and a silently wrong detector,
/// which is why the parity test checks the matrix before it ever looks at a logit.
public enum HitFeatures {
    /// Raw column layout of the (T,16) matrix (README "raw columns").
    public enum Raw {
        public static let count = 16
        public static let shuttleX = 0, shuttleY = 1, shuttleConf = 2, shuttleInterp = 3
        public static let box0 = 4, box1 = 8
        public static let foot0 = 12, foot1 = 14
    }

    /// Channel count of the derived matrix (asserted by the parity test against the fixture).
    public static let count = 51

    /// Court half length / half width in metres (common.py HALF_L, HALF_W).
    static let halfLen = 6.7
    static let halfWid = 3.05
    /// 1280 / 720 — the extract frame's aspect used by the shuttle-to-box offset (common.py ASPECT).
    static let aspect = 1280.0 / 720.0

    // MARK: - numpy difference helpers (common.py _fwd/_bwd/_cen/_acc)

    /// `_fwd(x)`: x[i+1] - x[i]; the last entry is NaN (one-sided difference has no right neighbour).
    static func fwd(_ x: [Double]) -> [Double] {
        var o = [Double](repeating: .nan, count: x.count)
        for i in 0..<(x.count - 1) { o[i] = x[i + 1] - x[i] }
        return o
    }

    /// `_bwd(x)`: x[i] - x[i-1]; the first entry is NaN.
    static func bwd(_ x: [Double]) -> [Double] {
        var o = [Double](repeating: .nan, count: x.count)
        for i in 1..<x.count { o[i] = x[i] - x[i - 1] }
        return o
    }

    /// `_cen(x)`: (x[i+1] - x[i-1]) / 2; the first and last entries are NaN.
    static func cen(_ x: [Double]) -> [Double] {
        var o = [Double](repeating: .nan, count: x.count)
        guard x.count >= 3 else { return o }
        for i in 1..<(x.count - 1) { o[i] = (x[i + 1] - x[i - 1]) / 2 }
        return o
    }

    /// `_acc(x)`: x[i+1] - 2 x[i] + x[i-1]; the first and last entries are NaN.
    static func acc(_ x: [Double]) -> [Double] {
        var o = [Double](repeating: .nan, count: x.count)
        guard x.count >= 3 else { return o }
        for i in 1..<(x.count - 1) { o[i] = x[i + 1] - 2 * x[i] + x[i - 1] }
        return o
    }

    /// `_z(x)` — numpy `nan_to_num(x, nan=0.0)`. The ONLY place "missing" turns into a number,
    /// and it always happens next to an explicit validity flag.
    @inline(__always) static func z(_ x: Double) -> Double { x.isFinite ? x : 0 }

    /// `c(a, k)` = `clip(nan_to_num(a) * k, -6, 6)` (common.py's lambda).
    @inline(__always) static func c(_ a: Double, _ k: Double) -> Double {
        min(max(z(a) * k, -6), 6)
    }

    /// `_dist_to_vis(vis)`: (frames since / until a visible shuttle) / 30, saturating at 1.0 so a
    /// long blackout reads as "far away" instead of growing without bound.
    static func distToVis(_ vis: [Bool]) -> ([Double], [Double]) {
        let T = vis.count
        var since = [Double](repeating: 1.0, count: T)
        var until = [Double](repeating: 1.0, count: T)
        var last = -Int(1e6)
        for i in 0..<T {
            if vis[i] { last = i }
            since[i] = Double(min(i - last, 30)) / 30.0
        }
        var next = Int(1e6)
        for i in stride(from: T - 1, through: 0, by: -1) {
            if vis[i] { next = i }
            until[i] = Double(min(next - i, 30)) / 30.0
        }
        return (since, until)
    }

    // MARK: - derive

    /// `derive(raw, fps)`: (T,16) row-major raw + the match fps -> (T*51) row-major float32.
    ///
    /// All the intermediate arithmetic is float64 like numpy's, and the float32 cast happens per
    /// channel at the end, which is what makes the port agree with the fixture to fp32 round-off.
    /// Returns an empty array for T == 0.
    public static func derive(raw: [Float], fps: Double) -> [Float] {
        precondition(raw.count % Raw.count == 0, "raw must be (T,16) row-major, got \(raw.count) floats")
        let T = raw.count / Raw.count
        guard T > 0 else { return [] }
        func col(_ c: Int) -> [Double] {
            var o = [Double](repeating: .nan, count: T)
            for i in 0..<T { o[i] = Double(raw[i * Raw.count + c]) }
            return o
        }
        return derive(columns: (0..<Raw.count).map(col), fps: fps)
    }

    /// The same, for a matrix that is already float64 (a player/track build, or a fixture slice).
    public static func derive(raw: [Double], nFrames T: Int, fps: Double) -> [Float] {
        guard T > 0, raw.count == T * Raw.count else { return [] }
        func col(_ c: Int) -> [Double] {
            var o = [Double](repeating: .nan, count: T)
            for i in 0..<T { o[i] = raw[i * Raw.count + c] }
            return o
        }
        return derive(columns: (0..<Raw.count).map(col), fps: fps)
    }

    /// The body of `derive`, over the 16 named columns.
    static func derive(columns cols: [[Double]], fps: Double) -> [Float] {
        let T = cols[Raw.shuttleX].count
        guard T > 0 else { return [] }
        // velocities are per 1/30 s so 25 and 30 fps rallies look alike to the net
        let ps = fps / 30.0
        let k1 = 30.0 * ps                    // position differences
        let k2 = 100.0 * ps * ps              // second differences
        let sx = cols[Raw.shuttleX], sy = cols[Raw.shuttleY]
        let vis = sx.map(\.isFinite)

        var f: [Float] = []
        f.reserveCapacity(T * count)

        func add(_ v: [Double]) { for x in v { f.append(Float(x)) } }
        func add(_ v: [Bool]) { for x in v { f.append(x ? 1 : 0) } }
        func add(_ v: Double) { for _ in 0..<T { f.append(Float(v)) } }

        // 0-4 shuttle position / confidence / interp (missing -> 0, the `visible` flag carries it)
        add(sx.map(z))
        add(sy.map(z))
        add(vis)
        add(cols[Raw.shuttleInterp].map(z))
        add(cols[Raw.shuttleConf].map(z))

        // 5-14 shuttle kinematics
        let vfx = fwd(sx), vfy = fwd(sy)
        let vbx = bwd(sx), vby = bwd(sy)
        let vcx = cen(sx), vcy = cen(sy)
        let ax = acc(sx), ay = acc(sy)
        let sp = (0..<T).map { (vcx[$0] * vcx[$0] + vcy[$0] * vcy[$0]).squareRoot() }
        // cosang: the angle between the incoming and outgoing step. NaN where either step is
        // missing (the norms are NaN, so the product is NaN and the `> 1e-8` test is false).
        let cosang: [Double] = (0..<T).map { i in
            let dot = vbx[i] * vfx[i] + vby[i] * vfy[i]
            let nb = (vbx[i] * vbx[i] + vby[i] * vby[i]).squareRoot()
            let nf = (vfx[i] * vfx[i] + vfy[i] * vfy[i]).squareRoot()
            let prod = nb * nf
            if prod.isFinite && prod > 1e-8 { return dot / max(prod, 1e-6) }
            return .nan
        }
        add((0..<T).map { c(vfx[$0], k1) })
        add((0..<T).map { c(vfy[$0], k1) })
        add((0..<T).map { c(vbx[$0], k1) })
        add((0..<T).map { c(vby[$0], k1) })
        add((0..<T).map { c(vcx[$0], k1) })
        add((0..<T).map { c(vcy[$0], k1) })
        add((0..<T).map { c(ax[$0], k2) })
        add((0..<T).map { c(ay[$0], k2) })
        add((0..<T).map { c(sp[$0], k1) })
        add(cosang.map(z))                     // `_z(cosang)`: NaN (no angle this frame) -> 0
        // 15-17 validity flags
        add(vfx.map(\.isFinite))
        add(vbx.map(\.isFinite))
        add(ax.map(\.isFinite))
        // 18-19 frames since / until a visible shuttle
        let (d0, d1) = distToVis(vis)
        add(d0)
        add(d1)

        // 20-49 the two players: box, box velocity, foot world position, foot velocity, offsets
        for p in 0..<2 {
            let o = Raw.box0 + 4 * p
            let cx = cols[o], cy = cols[o + 1], w = cols[o + 2], h = cols[o + 3]
            let okb = cx.map(\.isFinite)
            add(cx.map(z))
            add(cy.map(z))
            add(w.map { z($0) * 4 })
            add(h.map { z($0) * 2 })
            add(okb)
            add((0..<T).map { c(cen(cx)[$0], 100 * ps) })
            add((0..<T).map { c(cen(cy)[$0], 100 * ps) })
            let fx = cols[Raw.foot0 + 2 * p], fy = cols[Raw.foot0 + 2 * p + 1]
            let okf = fx.map(\.isFinite)
            add(fx.map { z($0) / halfLen })
            add(fy.map { z($0) / halfWid })
            add(okf)
            add((0..<T).map { c(cen(fx)[$0], 3 * ps) })
            add((0..<T).map { c(cen(fy)[$0], 3 * ps) })
            let (fxc, fyc) = (cen(fx), cen(fy))
            // The offset is in box-height units and only defined when the shuttle AND this box
            // are both resolved; otherwise it is exactly 0 (not NaN), flagged by the next channel.
            var dx = [Double](repeating: 0, count: T)
            var dy = [Double](repeating: 0, count: T)
            var okr = [Bool](repeating: false, count: T)
            for i in 0..<T {
                guard okb[i] && vis[i] else { continue }
                okr[i] = true
                let d = max(h[i], 1e-3)
                dx[i] = (sx[i] - cx[i]) * aspect / d
                dy[i] = (sy[i] - cy[i]) / d
            }
            _ = (fxc, fyc)
            add(dx.map { min(max($0, -6), 6) / 3 })   // clip BEFORE /3, per the README
            add(dy.map { min(max($0, -6), 6) / 3 })
            add(okr)
        }
        // 50 fps/30 (constant)
        add(fps / 30.0)

        // `f` is one contiguous run per channel (the Python appends columns and stacks them at
        // the end); the model's input is (T,51) row-major, so transpose once here. Delegating the
        // layout to one place is why every caller can read `f[i * 51 + c]`.
        var out = [Float](repeating: 0, count: T * count)
        precondition(f.count == T * count, "derived \(f.count) values for T = \(T)")
        for c in 0..<count {
            let base = c * T
            for i in 0..<T { out[i * count + c] = f[base + i] }
        }
        return out
    }
}
