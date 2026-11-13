import Foundation
import ShuttlCore
import simd

// Port of SegmentPrior, _prior_vec, _viol, _resid_jac, _robust_cost, _chol_solve,
// _lm_kernel and _Problem from backend/app/services/tracking/reconstruct3d.py.

/// Soft priors for one flight. Times are absolute seconds (same clock as t).
///
/// start_*: at the contact time `startT` (default: first observed time); the reported
/// p0/v0 are at startT when it is given. end = .floor: the shuttle was airborne at the
/// last observation and down by `endT` (e.g. first frame it rests on the floor);
/// `endXYWorld` = rest point on the floor if seen. end = .hit: contact at `endT`, near
/// `endXYWorld` (receiver's feet) with height in `endHeight`.
public struct SegmentPrior: Sendable {
    public enum End: String, Sendable { case floor, hit }

    public var startT: Double?
    public var startXYWorld: SIMD2<Double>?
    public var startXYSigma: Double = 0.5
    public var startHeight: (lo: Double, hi: Double)? = (0.05, 3.5)
    /// Hitter's half. Only .near/.far act as priors (Python: 'near' | 'far'); any other
    /// value (e.g. .unknown) behaves like Python's other truthy strings.
    public var expectedSide: CourtSide?
    public var end: End?
    public var endT: Double?
    public var endXYWorld: SIMD2<Double>?
    public var endXYSigma: Double = 0.5
    public var endHeight: (lo: Double, hi: Double)? = (0.05, 3.5)
    /// Where the shuttle lay afterwards (net test only).
    public var restXYWorld: SIMD2<Double>?
    /// Observations stop at a hole in the video clock: how the flight ended was never
    /// seen (no landing, no net call). Python `end_at_gap`.
    public var endAtGap: Bool = false
    public var heightSigma: Double = 0.1
    public var dragLength: Double = Physics.dragLength
    public var dragLogSigma: Double = 0.08

    public init(startT: Double? = nil, startXYWorld: SIMD2<Double>? = nil, startXYSigma: Double = 0.5,
                startHeight: (lo: Double, hi: Double)? = (0.05, 3.5), expectedSide: CourtSide? = nil,
                end: End? = nil, endT: Double? = nil, endXYWorld: SIMD2<Double>? = nil,
                endXYSigma: Double = 0.5, endHeight: (lo: Double, hi: Double)? = (0.05, 3.5),
                restXYWorld: SIMD2<Double>? = nil, heightSigma: Double = 0.1,
                dragLength: Double = Physics.dragLength, dragLogSigma: Double = 0.08) {
        self.startT = startT; self.startXYWorld = startXYWorld; self.startXYSigma = startXYSigma
        self.startHeight = startHeight; self.expectedSide = expectedSide; self.end = end
        self.endT = endT; self.endXYWorld = endXYWorld; self.endXYSigma = endXYSigma
        self.endHeight = endHeight; self.restXYWorld = restXYWorld; self.heightSigma = heightSigma
        self.dragLength = dragLength; self.dragLogSigma = dragLogSigma
    }

    /// `dataclasses.replace(prior, end=None, end_t=None, end_xy_world=None)`.
    var withoutEnd: SegmentPrior {
        var p = self
        p.end = nil; p.endT = nil; p.endXYWorld = nil
        return p
    }
}

// prior float vector layout (NaN = inactive)
let PS_X = 0, PS_Y = 1, PS_SIG = 2, PS_HLO = 3, PS_HHI = 4, PS_SIDE = 5, PS_HSIG = 6
let PE_X = 7, PE_Y = 8, PE_SIG = 9, PE_HLO = 10, PE_HHI = 11, PE_HSIG = 12
let P_LOGL = 13, P_LOGSIG = 14, P_NPRI = 15

/// `_prior_vec`.
func priorVec(_ pr: SegmentPrior, endIsFloor: Bool) -> [Double] {
    var v = [Double](repeating: .nan, count: P_NPRI)
    if let s = pr.startXYWorld {
        v[PS_X] = s.x; v[PS_Y] = s.y
        v[PS_SIG] = pr.startXYSigma
    }
    if let h = pr.startHeight { v[PS_HLO] = h.lo; v[PS_HHI] = h.hi }
    v[PS_HSIG] = pr.heightSigma
    if pr.expectedSide == .near || pr.expectedSide == .far {
        v[PS_SIDE] = pr.expectedSide == .near ? -1.0 : 1.0
    }
    if let e = pr.endXYWorld {
        v[PE_X] = e.x; v[PE_Y] = e.y
        v[PE_SIG] = pr.endXYSigma
    }
    if endIsFloor {
        v[PE_HHI] = 0.0; v[PE_HSIG] = 0.03          // down by end_t
    } else {
        if let h = pr.endHeight { v[PE_HLO] = h.lo; v[PE_HHI] = h.hi }
        v[PE_HSIG] = pr.heightSigma
    }
    v[P_LOGL] = log(pr.dragLength)
    v[P_LOGSIG] = pr.dragLogSigma
    return v
}

/// `_viol`: signed violation of [lo, hi]; NaN bound = open.
@inline(__always) func viol(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
    if lo == lo && x < lo { return x - lo }
    if hi == hi && x > hi { return x - hi }
    return 0.0
}

/// Reprojection + prior residuals for θ = (p0, v0[, log L]) at `tRef` (port of `_Problem`).
///
/// Residual vector, all in noise-sigma units, fixed layout (inactive hinge rows are
/// exactly 0 with a zero gradient):
///   [0, 2n)     reprojection (u, v) per observation      — soft-L1
///   [2n, 3n)    z >= -2 cm at each observation           — L2 below
///   [3n, 4n)    camera depth >= 1 m at each observation
///   next 3      loose box on p0, 1 speed cap, 1 log L prior
///   start: 2 xy + 1 height hinge + 1 side hinge; end: 2 xy + 1 height/floor hinge
final class Problem {
    let cam: CameraModel
    let prior: SegmentPrior
    let tRef: Double
    let tr: [Double]
    let uv: [SIMD2<Double>]
    let uvIdeal: [SIMD2<Double>]
    let sw: [Double]
    let nObs: Int
    let fitDrag: Bool
    let npar: Int
    let LFixed: Double
    let iStart: Int
    let iEnd: Int
    let endIsFloor: Bool
    let inv: [Int]
    let timesSorted: [Double]
    let nT: Int
    let pri: [Double]

    // camera (Python row-major element names)
    let r00, r01, r02, r10, r11, r12, r20, r21, r22: Double
    let t0, t1, t2: Double
    let fx, fy, skew, cx, cy: Double

    // propagation scratch (reused by every residual evaluation)
    let ws = ODEWorkspace()
    let P: UnsafeMutablePointer<SIMD3<Double>>
    let V: UnsafeMutablePointer<SIMD3<Double>>
    let S: UnsafeMutablePointer<Double>

    init(t: [Double], uv: [SIMD2<Double>], w: [Double], camera: CameraModel,
         prior: SegmentPrior?, fitDrag: Bool, tRef: Double? = nil) {
        cam = camera
        let pr = prior ?? SegmentPrior()
        self.prior = pr
        let tRef = tRef ?? t[0]                                   // θ epoch
        self.tRef = tRef
        tr = t.map { $0 - tRef }
        self.uv = uv
        sw = w.map { max($0, 0.0).squareRoot() }
        nObs = t.count
        self.fitDrag = fitDrag
        npar = fitDrag ? 7 : 6
        LFixed = pr.dragLength
        let K = camera.K
        if camera.isDistorted {                                   // residuals are pinhole: undistort once
            uvIdeal = uv.map { q in
                let d = camera.ray(q).direction
                let dc = camera.R * d
                let x = dc.x / dc.z, y = dc.y / dc.z
                return SIMD2(x * K[0, 0] + y * K[1, 0] + K[2, 0], x * K[0, 1] + y * K[1, 1] + K[2, 1])
            }
        } else {
            uvIdeal = uv
        }
        var extra: [Double] = []
        var iS = -1, iE = -1
        if let st = pr.startT {
            iS = t.count + extra.count
            extra.append(st - tRef)
        }
        endIsFloor = pr.end == .floor
        if pr.end != nil, let et = pr.endT {
            iE = t.count + extra.count
            extra.append(et - tRef)
        }
        iStart = iS; iEnd = iE
        let times = tr + extra
        let order = Num.argsort(times)
        var invv = [Int](repeating: 0, count: times.count)
        for (j, o) in order.enumerated() { invv[o] = j }
        inv = invv
        timesSorted = order.map { times[$0] }
        nT = times.count
        pri = priorVec(pr, endIsFloor: endIsFloor)
        let R = camera.R                                          // simd: R[col, row]
        r00 = R[0, 0]; r01 = R[1, 0]; r02 = R[2, 0]
        r10 = R[0, 1]; r11 = R[1, 1]; r12 = R[2, 1]
        r20 = R[0, 2]; r21 = R[1, 2]; r22 = R[2, 2]
        t0 = camera.t.x; t1 = camera.t.y; t2 = camera.t.z
        fx = K[0, 0]; fy = K[1, 1]; skew = K[1, 0]; cx = K[2, 0]; cy = K[2, 1]
        P = .allocate(capacity: max(nT, 1))
        P.initialize(repeating: .zero, count: max(nT, 1))
        V = .allocate(capacity: max(nT, 1))
        V.initialize(repeating: .zero, count: max(nT, 1))
        S = .allocate(capacity: max(nT, 1) * 21)
        S.initialize(repeating: 0, count: max(nT, 1) * 21)
    }

    deinit {
        P.deallocate(); V.deallocate(); S.deallocate()
    }

    var nRes: Int { 4 * nObs + R3.nFixed }

    func L_of(_ th: [Double]) -> Double { th.count > 6 ? exp(th[6]) : LFixed }

    /// `_resid_jac`: fill r (and J, row-major m × npar, if wantJac) for θ.
    func residJac(_ th: UnsafePointer<Double>, fitDrag fd: Bool, _ r: UnsafeMutablePointer<Double>,
                  _ J: UnsafeMutablePointer<Double>?, wantJac: Bool) {
        let npar = fd ? 7 : 6
        let L = fd ? exp(th[6]) : LFixed
        let ncol = wantJac ? (fd ? R3.ncol : R3.ncol - 1) : 1
        timesSorted.withUnsafeBufferPointer { tb in
            ws.propagate(SIMD3(th[0], th[1], th[2]), SIMD3(th[3], th[4], th[5]), L,
                         tb.baseAddress!, nT, ncol, P, V, wantJac ? S : nil)
        }
        let n = nObs
        let m = 4 * n + R3.nFixed
        r.update(repeating: 0, count: m)
        if wantJac, let J { J.update(repeating: 0, count: m * npar) }
        let sObs = 1.0 / R3.sigmaPx
        let (a00, a01, a02, a10, a11, a12, a20, a21, a22) = (r00, r01, r02, r10, r11, r12, r20, r21, r22)
        let (fx, fy, skew, cx, cy) = (self.fx, self.fy, self.skew, self.cx, self.cy)
        inv.withUnsafeBufferPointer { invb in
        uvIdeal.withUnsafeBufferPointer { uvb in
        sw.withUnsafeBufferPointer { swb in
            for i in 0..<n {
                let k = invb[i]
                let pk = P[k]
                let px = pk.x, py = pk.y, pz = pk.z
                let X = a00 * px + a01 * py + a02 * pz + t0
                let Y = a10 * px + a11 * py + a12 * pz + t1
                let Z = a20 * px + a21 * py + a22 * pz + t2
                let Zc = max(Z, 0.05)
                let xn = X / Zc, yn = Y / Zc
                let wi = swb[i] * sObs
                let q = uvb[i]
                r[2 * i] = (fx * xn + skew * yn + cx - q.x) * wi
                r[2 * i + 1] = (fy * yn + cy - q.y) * wi
                let zv = viol(pz, -0.02, .nan)
                r[2 * n + i] = zv / 0.05
                let dv = viol(Z, 1.0, .nan)
                r[3 * n + i] = dv / 0.1
                if wantJac, let J {
                    let iz = 1.0 / Zc
                    let Sk = S + k * 21
                    let Ju = J + 2 * i * npar, Jv = J + (2 * i + 1) * npar
                    for c in 0..<npar {
                        let s0 = Sk[3 * c], s1 = Sk[3 * c + 1], s2 = Sk[3 * c + 2]
                        let dX = a00 * s0 + a01 * s1 + a02 * s2
                        let dY = a10 * s0 + a11 * s1 + a12 * s2
                        let dZ = a20 * s0 + a21 * s1 + a22 * s2
                        let dxn = (dX - xn * dZ) * iz
                        let dyn = (dY - yn * dZ) * iz
                        Ju[c] = (fx * dxn + skew * dyn) * wi
                        Jv[c] = fy * dyn * wi
                        if zv != 0.0 { J[(2 * n + i) * npar + c] = s2 / 0.05 }
                        if dv != 0.0 { J[(3 * n + i) * npar + c] = dZ / 0.1 }
                    }
                }
            }
        }
        }
        }
        var row = 4 * n
        // loose box on p0
        let lo = [R3.boxLo.0, R3.boxLo.1, R3.boxLo.2], hi = [R3.boxHi.0, R3.boxHi.1, R3.boxHi.2]
        for a in 0..<3 {
            let bv = viol(th[a], lo[a], hi[a])
            r[row + a] = bv / 0.2
            if wantJac && bv != 0.0, let J { J[(row + a) * npar + a] = 1.0 / 0.2 }
        }
        row += 3
        let sp0 = (th[3] * th[3] + th[4] * th[4] + th[5] * th[5]).squareRoot()
        if sp0 > R3.maxSpeed {
            r[row] = (sp0 - R3.maxSpeed) / 2.0
            if wantJac, let J {
                for a in 0..<3 { J[row * npar + 3 + a] = th[3 + a] / sp0 / 2.0 }
            }
        }
        row += 1
        if fd {
            r[row] = (th[6] - pri[P_LOGL]) / pri[P_LOGSIG]
            if wantJac, let J { J[row * npar + 6] = 1.0 / pri[P_LOGSIG] }
        }
        row += 1
        // start constraints (at combined-time index iStart)
        if iStart >= 0 {
            let k = inv[iStart]
            let pk = P[k]
            let Sk = S + k * 21
            let sig = pri[PS_SIG]
            if sig == sig {
                r[row] = (pk.x - pri[PS_X]) / sig
                r[row + 1] = (pk.y - pri[PS_Y]) / sig
                if wantJac, let J {
                    for c in 0..<npar {
                        J[row * npar + c] = Sk[3 * c] / sig
                        J[(row + 1) * npar + c] = Sk[3 * c + 1] / sig
                    }
                }
            }
            let hz = viol(pk.z, pri[PS_HLO], pri[PS_HHI])
            r[row + 2] = hz / pri[PS_HSIG]
            let side = pri[PS_SIDE]
            var hx = 0.0
            if side == side {
                hx = side < 0 ? viol(pk.x, .nan, 0.3) : viol(pk.x, -0.3, .nan)
            }
            r[row + 3] = hx / 0.1
            if wantJac, let J {
                for c in 0..<npar {
                    if hz != 0.0 { J[(row + 2) * npar + c] = Sk[3 * c + 2] / pri[PS_HSIG] }
                    if hx != 0.0 { J[(row + 3) * npar + c] = Sk[3 * c] / 0.1 }
                }
            }
        }
        row += 4
        if iEnd >= 0 {
            let k = inv[iEnd]
            let pk = P[k]
            let Sk = S + k * 21
            let sig = pri[PE_SIG]
            if sig == sig {
                r[row] = (pk.x - pri[PE_X]) / sig
                r[row + 1] = (pk.y - pri[PE_Y]) / sig
                if wantJac, let J {
                    for c in 0..<npar {
                        J[row * npar + c] = Sk[3 * c] / sig
                        J[(row + 1) * npar + c] = Sk[3 * c + 1] / sig
                    }
                }
            }
            let hz = viol(pk.z, pri[PE_HLO], pri[PE_HHI])
            r[row + 2] = hz / pri[PE_HSIG]
            if wantJac && hz != 0.0, let J {
                for c in 0..<npar { J[(row + 2) * npar + c] = Sk[3 * c + 2] / pri[PE_HSIG] }
            }
        }
    }

    /// Residuals (noise-sigma units) and the analytic Jacobian (row-major m × npar) or nil.
    func residuals(_ th: [Double], jac: Bool, fitDrag fd: Bool? = nil) -> (r: [Double], J: [Double]?) {
        let fd = fd ?? fitDrag
        let npar = fd ? 7 : 6
        let m = nRes
        var r = [Double](repeating: 0, count: m)
        var J = [Double](repeating: 0, count: jac ? m * npar : 0)
        th.withUnsafeBufferPointer { tb in
            r.withUnsafeMutableBufferPointer { rb in
                J.withUnsafeMutableBufferPointer { jb in
                    residJac(tb.baseAddress!, fitDrag: fd, rb.baseAddress!, jac ? jb.baseAddress! : nil, wantJac: jac)
                }
            }
        }
        return (r, jac ? J : nil)
    }

    /// Positions/velocities at [obs times..., extra times...] (sensitivities not needed by
    /// any caller of the port; `_Problem.simulate(th, sens=False)`).
    func simulate(_ th: [Double]) -> (P: [SIMD3<Double>], V: [SIMD3<Double>]) {
        var Ps = [SIMD3<Double>](repeating: .zero, count: nT)
        var Vs = Ps
        timesSorted.withUnsafeBufferPointer { tb in
            Ps.withUnsafeMutableBufferPointer { pb in
                Vs.withUnsafeMutableBufferPointer { vb in
                    ws.propagate(SIMD3(th[0], th[1], th[2]), SIMD3(th[3], th[4], th[5]), L_of(th),
                                 tb.baseAddress!, nT, 1, pb.baseAddress!, vb.baseAddress!, nil)
                }
            }
        }
        var P = Ps, V = Vs
        for i in 0..<nT { P[i] = Ps[inv[i]]; V[i] = Vs[inv[i]] }
        return (P, V)
    }
}

/// `_robust_cost`: soft-L1 per (u, v) pair on the first 2n rows, L2 elsewhere; IRLS weights.
@inline(__always)
func robustCost(_ r: UnsafePointer<Double>, _ m: Int, _ n: Int, _ wts: UnsafeMutablePointer<Double>) -> Double {
    let fs2 = R3.fScale * R3.fScale
    var cost = 0.0
    for i in 0..<n {
        let z = (r[2 * i] * r[2 * i] + r[2 * i + 1] * r[2 * i + 1]) / fs2
        let root = (1.0 + z).squareRoot()
        cost += 2.0 * fs2 * (root - 1.0)
        wts[2 * i] = 1.0 / root
        wts[2 * i + 1] = 1.0 / root
    }
    var i = 2 * n
    while i < m {
        cost += r[i] * r[i]
        wts[i] = 1.0
        i += 1
    }
    return cost
}

/// `_robust`: (cost, IRLS weights).
func robust(_ r: [Double], _ nObs: Int) -> (cost: Double, wts: [Double]) {
    var w = [Double](repeating: 1, count: r.count)
    let c = r.withUnsafeBufferPointer { rb in
        w.withUnsafeMutableBufferPointer { wb in robustCost(rb.baseAddress!, r.count, nObs, wb.baseAddress!) }
    }
    return (c, w)
}

/// `_chol_solve`: solve SPD A x = b (m × m, row-major). False if not SPD (a NaN pivot is
/// not `<= 0`, so NaN propagates exactly like the numba kernel).
@inline(__always)
func cholSolve(_ A: UnsafePointer<Double>, _ m: Int, _ b: UnsafePointer<Double>,
               _ out: UnsafeMutablePointer<Double>, _ Lc: UnsafeMutablePointer<Double>,
               _ y: UnsafeMutablePointer<Double>) -> Bool {
    Lc.update(repeating: 0, count: m * m)
    for i in 0..<m {
        for j in 0...i {
            var s = A[i * m + j]
            for k in 0..<j { s -= Lc[i * m + k] * Lc[j * m + k] }
            if i == j {
                if s <= 0.0 { return false }
                Lc[i * m + i] = s.squareRoot()
            } else {
                Lc[i * m + j] = s / Lc[j * m + j]
            }
        }
    }
    for i in 0..<m {
        var s = b[i]
        for k in 0..<i { s -= Lc[i * m + k] * y[k] }
        y[i] = s / Lc[i * m + i]
    }
    var i = m - 1
    while i >= 0 {
        var s = y[i]
        var k = i + 1
        while k < m { s -= Lc[k * m + i] * out[k]; k += 1 }
        out[i] = s / Lc[i * m + i]
        i -= 1
    }
    return true
}

/// Array convenience of `cholSolve`; nil if not SPD.
func cholSolve(_ A: [Double], _ b: [Double]) -> [Double]? {
    let m = b.count
    var out = [Double](repeating: 0, count: m)
    var Lc = [Double](repeating: 0, count: m * m)
    var y = [Double](repeating: 0, count: m)
    let ok = A.withUnsafeBufferPointer { ab in b.withUnsafeBufferPointer { bb in
        out.withUnsafeMutableBufferPointer { ob in Lc.withUnsafeMutableBufferPointer { lb in
            y.withUnsafeMutableBufferPointer { yb in
                cholSolve(ab.baseAddress!, m, bb.baseAddress!, ob.baseAddress!, lb.baseAddress!, yb.baseAddress!)
            } } } } }
    return ok ? out : nil
}

/// Result of one robust LM run (`_Fit`).
struct Fit {
    var th: [Double]
    var cost: Double
    var nIter: Int
}

/// `_lm_kernel` / `_lm`: robust (IRLS soft-L1) Levenberg–Marquardt from θ0. Same damping
/// schedule (λ0 = 1e-3, ×10 on a non-SPD system, ×8 per rejected step, up to 8 tries,
/// ÷5 after an accepted step, floor 1e-7) and the same stopping tests.
func lm(_ prob: Problem, _ th0: [Double], maxIter: Int = 40, tol: Double = 1e-7) -> Fit {
    let fitDrag = prob.fitDrag
    let npar = fitDrag ? 7 : 6
    let n = prob.nObs
    let m = prob.nRes
    // one scratch block: r, J, r2, wts, wts2, A, g, Ad, step, negG, Lc, y, th, thNew
    let total = m + m * npar + m + m + m + npar * npar + npar + npar * npar + npar + npar
        + npar * npar + npar + npar + npar
    let buf = UnsafeMutablePointer<Double>.allocate(capacity: total)
    buf.initialize(repeating: 0, count: total)
    defer { buf.deallocate() }
    let r = buf
    let J = r + m
    let r2 = J + m * npar
    let wts = r2 + m
    let wts2 = wts + m
    let A = wts2 + m
    let g = A + npar * npar
    let Ad = g + npar
    let step = Ad + npar * npar
    let negG = step + npar
    let Lc = negG + npar
    let yv = Lc + npar * npar
    let th = yv + npar
    let thNew = th + npar
    for a in 0..<npar { th[a] = th0[a] }
    prob.residJac(th, fitDrag: fitDrag, r, J, wantJac: true)
    var cost = robustCost(r, m, n, wts)
    var lam = 1e-3
    var it = 0
    var iter = 1
    while iter <= maxIter {
        it = iter
        for a in 0..<npar {
            g[a] = 0.0
            for b in 0..<npar { A[a * npar + b] = 0.0 }
        }
        for i in 0..<m {
            let w = wts[i]
            let ri = r[i] * w
            let Ji = J + i * npar
            for a in 0..<npar {
                let ja = Ji[a]
                if ja == 0.0 { continue }
                g[a] += ja * ri
                let jw = ja * w
                for b in 0...a { A[a * npar + b] += jw * Ji[b] }
            }
        }
        var dmax = 0.0
        for a in 0..<npar {
            for b in 0..<a { A[b * npar + a] = A[a * npar + b] }
            dmax = max(dmax, A[a * npar + a])
        }
        var improved = false
        var cNew = cost
        for a in 0..<npar { thNew[a] = th[a] }
        for _ in 0..<8 {
            for a in 0..<npar {
                for b in 0..<npar { Ad[a * npar + b] = A[a * npar + b] }
                Ad[a * npar + a] += lam * max(A[a * npar + a], 1e-9 * max(dmax, 1e-12))
            }
            for a in 0..<npar { negG[a] = -g[a] }
            if !cholSolve(Ad, npar, negG, step, Lc, yv) {
                lam *= 10.0
                continue
            }
            for a in 0..<npar { thNew[a] = th[a] + step[a] }
            if fitDrag {
                thNew[6] = min(max(thNew[6], R3.logL0 - 1.0), R3.logL0 + 1.0)
            }
            let v = (thNew[3] * thNew[3] + thNew[4] * thNew[4] + thNew[5] * thNew[5]).squareRoot()
            if v > 1.5 * R3.maxSpeed {                              // keep the ODE sane
                for a in 3..<6 { thNew[a] *= 1.5 * R3.maxSpeed / v }
            }
            prob.residJac(thNew, fitDrag: fitDrag, r2, nil, wantJac: false)
            cNew = robustCost(r2, m, n, wts2)
            if cNew == cNew && cNew < cost {
                improved = true
                break
            }
            lam *= 8.0
        }
        if !improved { break }
        let rel = (cost - cNew) / max(cost, 1e-12)
        var smax = 0.0
        for a in 0..<npar {
            smax = max(smax, abs(thNew[a] - th[a]))
            th[a] = thNew[a]
        }
        prob.residJac(th, fitDrag: fitDrag, r, J, wantJac: true)
        cost = robustCost(r, m, n, wts)
        lam = max(lam / 5.0, 1e-7)
        if rel < tol || smax < 1e-7 { break }
        iter += 1
    }
    return Fit(th: Array(UnsafeBufferPointer(start: th, count: npar)), cost: cost, nIter: it)
}
