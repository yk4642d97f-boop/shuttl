/// Image line extraction: Hough segments → greedy collinear merge.
/// Port of court_calibration._segments, _Line, _merge_segments.
import Foundation
import simd

/// `_Line`: infinite line l = (a, b, c) with a² + b² = 1, its supported extent and weight.
struct ImageLine: Sendable {
    var l: SIMD3<Double>
    var p0: SIMD2<Double>
    var p1: SIMD2<Double>
    var weight: Double

    /// Weighted total-least-squares fit (numpy eigh → LAPACK eigenvector signs).
    init(pts: [SIMD2<Double>], wts: [Double], weight: Double) {
        var sx = 0.0, sy = 0.0
        for (p, w) in zip(pts, wts) { sx += p.x * w; sy += p.y * w }
        let ws = NumpyCompat.sum(wts)
        let c = SIMD2(sx / ws, sy / ws)
        var cxx = 0.0, cxy = 0.0, cyy = 0.0
        for (p, w) in zip(pts, wts) {
            let d = p - c
            cxx += d.x * w * d.x; cxy += d.x * w * d.y; cyy += d.y * w * d.y
        }
        let u = Lapack.eigh2(cxx, cxy, cyy).v1          // direction = largest eigenvector
        let n = SIMD2(-u.y, u.x)
        l = SIMD3(n.x, n.y, -(n.x * c.x + n.y * c.y))
        var smin = Double.infinity, smax = -Double.infinity
        for p in pts {
            let d = p - c
            let s = d.x * u.x + d.y * u.y
            smin = min(smin, s); smax = max(smax, s)
        }
        p0 = c + smin * u; p1 = c + smax * u
        self.weight = weight
    }

    var mid: SIMD2<Double> { 0.5 * (p0 + p1) }
    var direction: SIMD2<Double> { let d = p1 - p0; return d / (norm2(d) + 1e-12) }
    var length: Double { norm2(p1 - p0) }
}

enum LineExtraction {
    /// `_segments`: HoughLinesP on centre-line pixels, longest `maxSegments` kept.
    static func segments(_ feat: Features, maxSegments: Int) -> [SIMD4<Double>] {
        let D = feat.D
        let raw = Hough.linesP(feat.thin, rho: 1, theta: Double.pi / 720, threshold: max(12, Int(0.012 * D)),
                               minLineLength: 0.03 * D, maxLineGap: 0.006 * D)
        var segs = raw.map { SIMD4<Double>(Double($0.x), Double($0.y), Double($0.z), Double($0.w)) }
        if segs.count > maxSegments {
            let negL = segs.map { -hypot($0.z - $0.x, $0.w - $0.y) }
            segs = NumpyCompat.argsortStable(negL).prefix(maxSegments).map { segs[$0] }
        }
        return segs
    }
}
extension LineExtraction {
    /// `_merge_segments`: greedy collinear merge (longest first) → lines with support weight.
    static func merge(_ segs: [SIMD4<Double>], D: Double, maxLines: Int = 40) -> [ImageLine] {
        guard !segs.isEmpty else { return [] }
        let L = segs.map { hypot($0.z - $0.x, $0.w - $0.y) }
        let tolD = max(1.5, 0.0015 * D)
        let tolA = sin(1.0 * Double.pi / 180)
        var groups: [[Int]] = []
        var glines: [ImageLine] = []
        for i in NumpyCompat.argsortQuick(L.map { -$0 }) {
            let p = SIMD2(segs[i].x, segs[i].y), q = SIMD2(segs[i].z, segs[i].w)
            let u = (q - p) / (L[i] + 1e-12)
            var merged = false
            for g in 0..<glines.count {
                let gl = glines[g].l
                if abs(gl.x * u.x + gl.y * u.y) > tolA { continue }
                if abs(gl.x * p.x + gl.y * p.y + gl.z) > tolD || abs(gl.x * q.x + gl.y * q.y + gl.z) > tolD { continue }
                groups[g].append(i)
                let idx = groups[g]
                let pts = idx.map { SIMD2(segs[$0].x, segs[$0].y) } + idx.map { SIMD2(segs[$0].z, segs[$0].w) }
                let lw = idx.map { L[$0] }
                glines[g] = ImageLine(pts: pts, wts: lw + lw, weight: NumpyCompat.sum(lw))
                merged = true
                break
            }
            if !merged {
                groups.append([i])
                glines.append(ImageLine(pts: [p, q], wts: [L[i], L[i]], weight: L[i]))
            }
        }
        // Python list.sort(key=-weight) is stable
        let order = NumpyCompat.argsortStable(glines.map { -$0.weight })
        return order.prefix(maxLines).map { glines[$0] }
    }
}