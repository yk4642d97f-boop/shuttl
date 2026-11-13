import Foundation
import ShuttlCore
import simd

// Port of the numba kernels of backend/app/services/tracking/reconstruct3d.py
// (_rhs, _rk4, _hlim, _init_state, _propagate, _ground_time, _state_at), same
// operation order so the solver walks the same path as the Python reference.

/// Solver constants (module-level constants of reconstruct3d.py: physics / noise
/// model, not tuning knobs — `ReconstructConfig` holds the decision thresholds).
enum R3 {
    static let g = Physics.g
    static let dragLength = Physics.dragLength
    static let sigmaPx = 1.5                     // nominal detection noise (px)
    static let fScale = 1.6                      // soft-L1 scale in sigma units (~2.4 px)
    static let maxSpeed = 500.0 / 3.6            // m/s, plausibility bound
    static let logL0 = log(Physics.dragLength)
    static let hMax = 1.0 / 120.0                // adaptive RK4: h = min(H_MAX, C_STEP * L / |v|)
    static let cStep = 0.08                      // drag-timescale fraction; RK4 error ~ C_STEP^5
    static let ncol = 8                          // state column 0 + 7 sensitivity columns
    /// 3 box + 1 speed cap + 1 log L prior + start (2 xy + height + side) + end (2 xy + height)
    static let nFixed = 3 + 1 + 1 + 4 + 3
    static let boxLo: (Double, Double, Double) = (-9.5, -5.5, 0.0)
    static let boxHi: (Double, Double, Double) = (9.5, 5.5, 8.0)
    static let volumeLo = SIMD3(-CourtModel.halfLength - 2.0, -CourtModel.doublesHalfWidth - 2.0, 0.0)
    static let volumeHi = SIMD3(CourtModel.halfLength + 2.0, CourtModel.doublesHalfWidth + 2.0, 6.0)
    static let nDepths = 9
}

// MARK: - point integrator (ncol == 1: the flight only, no sensitivities)

@inline(__always)
func rhsPoint(_ v: SIMD3<Double>, _ L: Double) -> SIMD3<Double> {
    let sp = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    let k = sp / L
    return SIMD3(-k * v.x, -k * v.y, -R3.g - k * v.z)
}

/// `_rk4` restricted to column 0 (identical arithmetic to the general kernel).
@inline(__always)
func rk4Point(_ p: inout SIMD3<Double>, _ v: inout SIMD3<Double>, _ h: Double, _ L: Double) {
    let K1 = rhsPoint(v, L)
    var SY = v
    var Xt = v + 0.5 * h * K1
    SY += 2.0 * Xt
    let K2 = rhsPoint(Xt, L)
    Xt = v + 0.5 * h * K2
    SY += 2.0 * Xt
    let K3 = rhsPoint(Xt, L)
    Xt = v + h * K3
    SY += Xt
    let K4 = rhsPoint(Xt, L)
    p += h / 6.0 * SY
    v += h / 6.0 * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
}

/// `_hlim`.
@inline(__always)
func hlim(_ v: SIMD3<Double>, _ L: Double) -> Double {
    let sp = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    return min(R3.hMax, R3.cStep * L / max(sp, 1e-6))
}

// MARK: - augmented integrator (SIMD8 rows)

/// Augmented state held in registers, one SIMD8 per spatial row: lane c = column c of
/// Y = [p | dp/dθ] and X = [v | dv/dθ]. Every lane performs exactly the scalar
/// operations of `_rhs` / `_rk4` (Swift SIMD arithmetic is plain IEEE per lane, no
/// contraction), so results are bitwise those of the scalar kernel.
struct AugState {
    var y0, y1, y2: SIMD8<Double>
    var x0, x1, x2: SIMD8<Double>

    /// `_init_state` (ncol > 1: dp/dp0 = I in lanes 1…3, dv/dv0 = I in lanes 4…6).
    @inline(__always)
    init(_ p0: SIMD3<Double>, _ v0: SIMD3<Double>) {
        y0 = SIMD8(p0.x, 1, 0, 0, 0, 0, 0, 0)
        y1 = SIMD8(p0.y, 0, 1, 0, 0, 0, 0, 0)
        y2 = SIMD8(p0.z, 0, 0, 1, 0, 0, 0, 0)
        x0 = SIMD8(v0.x, 0, 0, 0, 1, 0, 0, 0)
        x1 = SIMD8(v0.y, 0, 0, 0, 0, 1, 0, 0)
        x2 = SIMD8(v0.z, 0, 0, 0, 0, 0, 1, 0)
    }

    /// `_rhs` for rows (a, b, c) of X; `dragCol` = the log-L column is active (ncol == 8).
    @inline(__always)
    static func rhs(_ a: SIMD8<Double>, _ b: SIMD8<Double>, _ c: SIMD8<Double>, _ L: Double, _ dragCol: Bool)
        -> (SIMD8<Double>, SIMD8<Double>, SIMD8<Double>) {
        let v0 = a[0], v1 = b[0], v2 = c[0]
        let sp = (v0 * v0 + v1 * v1 + v2 * v2).squareRoot()
        let k = sp / L
        let inv = sp > 1e-9 ? 1.0 / sp : 0.0
        // lanes >= 1: vs = (v · X_c) * inv; out_r = -(sp * X_rc + v_r * vs) / L
        let vs = (v0 * a + v1 * b + v2 * c) * inv
        let spv = SIMD8<Double>(repeating: sp), Lv = SIMD8<Double>(repeating: L)
        var o0 = -(spv * a + v0 * vs) / Lv
        var o1 = -(spv * b + v1 * vs) / Lv
        var o2 = -(spv * c + v2 * vs) / Lv
        // lane 0: the flight itself
        o0[0] = -k * v0
        o1[0] = -k * v1
        o2[0] = -R3.g - k * v2
        if dragCol {
            o0[7] += k * v0
            o1[7] += k * v1
            o2[7] += k * v2
        }
        return (o0, o1, o2)
    }

    /// `_rk4` (same stage order and expressions).
    @inline(__always)
    mutating func rk4(_ h: Double, _ L: Double, _ dragCol: Bool) {
        let hh = SIMD8<Double>(repeating: 0.5 * h), hv = SIMD8<Double>(repeating: h)
        let (k10, k11, k12) = AugState.rhs(x0, x1, x2, L, dragCol)
        var t0 = x0 + hh * k10, t1 = x1 + hh * k11, t2 = x2 + hh * k12
        var s0 = x0 + 2.0 * t0, s1 = x1 + 2.0 * t1, s2 = x2 + 2.0 * t2
        let (k20, k21, k22) = AugState.rhs(t0, t1, t2, L, dragCol)
        t0 = x0 + hh * k20; t1 = x1 + hh * k21; t2 = x2 + hh * k22
        s0 += 2.0 * t0; s1 += 2.0 * t1; s2 += 2.0 * t2
        let (k30, k31, k32) = AugState.rhs(t0, t1, t2, L, dragCol)
        t0 = x0 + hv * k30; t1 = x1 + hv * k31; t2 = x2 + hv * k32
        s0 += t0; s1 += t1; s2 += t2
        let (k40, k41, k42) = AugState.rhs(t0, t1, t2, L, dragCol)
        let w = SIMD8<Double>(repeating: h / 6.0)
        y0 += w * s0; y1 += w * s1; y2 += w * s2
        x0 += w * (k10 + 2.0 * k20 + 2.0 * k30 + k40)
        x1 += w * (k11 + 2.0 * k21 + 2.0 * k31 + k41)
        x2 += w * (k12 + 2.0 * k22 + 2.0 * k32 + k42)
    }
}

/// Host of `_propagate` (stateless now that the augmented state lives in registers; kept
/// as a type so call sites read like the Python module).
final class ODEWorkspace {
    init() {}

    /// `_propagate`: integrate to the sorted `times` (any sign, relative to the p0/v0
    /// epoch). Writes P, V and — when ncol > 1 — S (element (i, c, r) at i*21 + c*3 + r)
    /// = dp_r/dθ_c for the first ncol-1 columns, θ = (p0, v0, log L).
    func propagate(_ p0: SIMD3<Double>, _ v0: SIMD3<Double>, _ L: Double,
                   _ times: UnsafePointer<Double>, _ n: Int, _ ncol: Int,
                   _ P: UnsafeMutablePointer<SIMD3<Double>>, _ V: UnsafeMutablePointer<SIMD3<Double>>,
                   _ S: UnsafeMutablePointer<Double>?) {
        var firstPos = 0
        while firstPos < n && times[firstPos] < 0.0 { firstPos += 1 }
        if ncol == 1 {
            for direction in 0..<2 {
                var p = p0, v = v0
                var t = 0.0
                var i = direction == 0 ? firstPos : firstPos - 1
                while direction == 0 ? i < n : i >= 0 {
                    var rem = times[i] - t
                    let sgn = rem >= 0 ? 1.0 : -1.0
                    rem = abs(rem)
                    while rem > 1e-12 {
                        let h = min(rem, hlim(v, L))
                        rk4Point(&p, &v, sgn * h, L)
                        rem -= h
                    }
                    t = times[i]
                    P[i] = p
                    V[i] = v
                    i += direction == 0 ? 1 : -1
                }
            }
            return
        }
        let dragCol = ncol == R3.ncol
        let nsens = ncol - 1
        for direction in 0..<2 {
            var st = AugState(p0, v0)
            var t = 0.0
            var i = direction == 0 ? firstPos : firstPos - 1
            while direction == 0 ? i < n : i >= 0 {
                var rem = times[i] - t
                let sgn = rem >= 0 ? 1.0 : -1.0
                rem = abs(rem)
                while rem > 1e-12 {
                    let h = min(rem, hlim(SIMD3(st.x0[0], st.x1[0], st.x2[0]), L))
                    st.rk4(sgn * h, L, dragCol)
                    rem -= h
                }
                t = times[i]
                P[i] = SIMD3(st.y0[0], st.y1[0], st.y2[0])
                V[i] = SIMD3(st.x0[0], st.x1[0], st.x2[0])
                if let S {
                    let base = S + i * 21
                    for c in 0..<nsens {
                        base[3 * c] = st.y0[c + 1]
                        base[3 * c + 1] = st.y1[c + 1]
                        base[3 * c + 2] = st.y2[c + 1]
                    }
                }
                i += direction == 0 ? 1 : -1
            }
        }
    }
}

/// `_ground_time`: seconds (>= 0) until z first descends through zGround; -1 if never.
func groundTime(_ p0: SIMD3<Double>, _ v0: SIMD3<Double>, _ L: Double, tMax: Double,
                zGround: Double) -> Double {
    var p = p0, v = v0
    if p.z <= zGround { return 0.0 }
    var t = 0.0
    while t < tMax {
        let h = hlim(v, L)
        var pn = p, vn = v
        rk4Point(&pn, &vn, h, L)
        if pn.z <= zGround {
            var a = 0.0, b = h                  // bracket then secant/bisection on sub-step
            var za = p.z - zGround, zb = pn.z - zGround
            for _ in 0..<30 {
                var m = a + (b - a) * za / (za - zb)
                if !(a < m && m < b) { m = 0.5 * (a + b) }
                var pm = p, vm = v
                rk4Point(&pm, &vm, m, L)
                let zm = pm.z - zGround
                if abs(zm) < 1e-7 { return t + m }
                if zm > 0 { a = m; za = zm } else { b = m; zb = zm }
            }
            return t + 0.5 * (a + b)
        }
        p = pn; v = vn
        t += h
    }
    return -1.0
}

/// `_state_at`: positions/velocities at τ = `times` (any order/sign) relative to the
/// epoch of θ7 = (p0, v0, log L).
func stateAt(_ th7: [Double], _ times: [Double]) -> (P: [SIMD3<Double>], V: [SIMD3<Double>]) {
    let n = times.count
    if n == 0 { return ([], []) }
    let order = Num.argsort(times)
    var ts = [Double](repeating: 0, count: n)
    for (j, o) in order.enumerated() { ts[j] = times[o] }
    var Ps = [SIMD3<Double>](repeating: .zero, count: n)
    var Vs = Ps
    let ws = ODEWorkspace()
    ts.withUnsafeBufferPointer { tb in
        Ps.withUnsafeMutableBufferPointer { pb in
            Vs.withUnsafeMutableBufferPointer { vb in
                ws.propagate(SIMD3(th7[0], th7[1], th7[2]), SIMD3(th7[3], th7[4], th7[5]), exp(th7[6]),
                             tb.baseAddress!, n, 1, pb.baseAddress!, vb.baseAddress!, nil)
            }
        }
    }
    var sorted = true
    for (j, o) in order.enumerated() where j != o { sorted = false; break }
    if sorted { return (Ps, Vs) }
    var P = Ps, V = Vs
    for (j, o) in order.enumerated() { P[o] = Ps[j]; V[o] = Vs[j] }
    return (P, V)
}

/// `_state_at` at a single τ (== `_propagate` with one time: forward if τ >= 0, else backward).
@inline(__always)
func stateAt(_ th7: [Double], _ tau: Double) -> (p: SIMD3<Double>, v: SIMD3<Double>) {
    var p = SIMD3(th7[0], th7[1], th7[2]), v = SIMD3(th7[3], th7[4], th7[5])
    let L = exp(th7[6])
    let sgn = tau >= 0 ? 1.0 : -1.0
    var rem = abs(tau)
    while rem > 1e-12 {
        let h = min(rem, hlim(v, L))
        rk4Point(&p, &v, sgn * h, L)
        rem -= h
    }
    return (p, v)
}
