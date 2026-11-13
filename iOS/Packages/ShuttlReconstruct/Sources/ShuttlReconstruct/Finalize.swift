import Foundation
import ShuttlCore
import simd

// Port of the flight-end / uncertainty part of reconstruct3d.py: _landing, _landing_cov,
// _speed_sigma, net_height_at, _net_crossing, _calib_floor_cov, _finalize and the public
// helpers landing_covariance, landing_sigma, speed_profile, initial_speed_kmh,
// contact_speed_kmh.

/// `_landing`: first floor contact (xy, τ) after τ = tFrom, relative to the θ epoch.
func landing(_ th7: [Double], _ tFrom: Double) -> (xy: SIMD2<Double>, tau: Double)? {
    let (p, v) = stateAt(th7, tFrom)
    let tg = groundTime(p, v, exp(th7[6]), tMax: 8.0, zGround: 0.0)
    if tg < 0 { return nil }
    let P = stateAt(th7, tFrom + tg).p
    return (SIMD2(P.x, P.y), tFrom + tg)
}

/// 2×7 · 7×7 · 7×2 → 2×2 (row-major), numpy `Jl @ cov7 @ Jl.T` association.
func sandwich(_ Jl: [Double], _ rowsJ: Int, _ cov: [Double], _ n: Int) -> [Double] {
    var M = [Double](repeating: 0, count: rowsJ * n)
    for i in 0..<rowsJ { for j in 0..<n {
        var s = 0.0
        for k in 0..<n { s += Jl[i * n + k] * cov[k * n + j] }
        M[i * n + j] = s
    } }
    var out = [Double](repeating: 0, count: rowsJ * rowsJ)
    for i in 0..<rowsJ { for j in 0..<rowsJ {
        var s = 0.0
        for k in 0..<n { s += M[i * n + k] * Jl[j * n + k] }
        out[i * rowsJ + j] = s
    } }
    return out
}

/// `_landing_cov`: landing-point covariance by central differences of `_landing` through cov7.
func landingCovFD(_ th7: [Double], _ cov7: [Double], _ tFrom: Double) -> [Double]? {
    var Jl = [Double](repeating: 0, count: 14)
    for k in 0..<7 {
        let e = 1e-4
        var hp = th7, hm = th7
        hp[k] += e
        hm[k] -= e
        guard let lp = landing(hp, tFrom), let lmn = landing(hm, tFrom) else { return nil }
        Jl[k] = (lp.xy.x - lmn.xy.x) / (2 * e)
        Jl[7 + k] = (lp.xy.y - lmn.xy.y) / (2 * e)
    }
    return sandwich(Jl, 2, cov7, 7)
}

/// `_speed_sigma`: 1-σ of |v(τ)| (m/s) by central differences.
func speedSigma(_ th7: [Double], _ cov7: [Double], _ tau: Double) -> Double {
    var g = [Double](repeating: 0, count: 7)
    for k in 0..<7 {
        let e = 1e-4
        var hp = th7, hm = th7
        hp[k] += e
        hm[k] -= e
        let vp = norm3(stateAt(hp, tau).v)
        let vm = norm3(stateAt(hm, tau).v)
        g[k] = (vp - vm) / (2 * e)
    }
    var q = 0.0
    var gc = [Double](repeating: 0, count: 7)
    for j in 0..<7 { var s = 0.0; for k in 0..<7 { s += g[k] * cov7[k * 7 + j] }; gc[j] = s }
    for j in 0..<7 { q += gc[j] * g[j] }
    return max(q, 0.0).squareRoot()
}

/// `_net_crossing`: first crossing of x = 0 in [tau0, tau1]: (z, y, τ).
func netCrossing(_ th7: [Double], _ tau0: Double, _ tau1: Double) -> (z: Double, y: Double, tau: Double)? {
    if tau1 <= tau0 { return nil }
    let grid = arange(tau0, tau1 + 1e-9, 1.0 / 480.0)
    if grid.count < 2 { return nil }
    let P = stateAt(th7, grid).P
    @inline(__always) func sign(_ x: Double) -> Double { x.isNaN ? .nan : (x > 0 ? 1 : (x < 0 ? -1 : 0)) }
    for i in 0..<(P.count - 1) where sign(P[i].x) * sign(P[i + 1].x) < 0 {
        let a = P[i].x / (P[i].x - P[i + 1].x)
        let z = P[i].z + a * (P[i + 1].z - P[i].z)
        let y = P[i].y + a * (P[i + 1].y - P[i].y)
        return (z, y, grid[i] + a * (grid[i + 1] - grid[i]))
    }
    return nil
}

/// `_calib_floor_cov`: floor-point covariance (m², row-major 2×2) from the camera's
/// calibration reprojection RMSE.
func calibFloorCov(_ cam: CameraModel, _ xy: SIMD2<Double>) -> [Double] {
    let rms = cam.reprojRmsePx
    if !(rms == rms) || rms <= 0 { return [0, 0, 0, 0] }
    let uv = cam.project(SIMD3(xy.x, xy.y, 0.0))
    if !uv.isFinite { return [0, 0, 0, 0] }
    var Jf = [Double](repeating: 0, count: 4)
    for k in 0..<2 {
        var d = SIMD2<Double>(0, 0)
        d[k] = 1.0
        let a = cam.pixelToFloor(uv + d), b = cam.pixelToFloor(uv - d)
        Jf[k] = (a.x - b.x) / 2
        Jf[2 + k] = (a.y - b.y) / 2
    }
    if !Jf.allSatisfy(\.isFinite) { return [0, 0, 0, 0] }
    let r2 = rms * rms
    return [r2 * (Jf[0] * Jf[0] + Jf[1] * Jf[1]), r2 * (Jf[0] * Jf[2] + Jf[1] * Jf[3]),
            r2 * (Jf[2] * Jf[0] + Jf[3] * Jf[1]), r2 * (Jf[2] * Jf[2] + Jf[3] * Jf[3])]
}

/// Largest eigenvalue-based σ of a 2×2 covariance (`sqrt(max(eigvalsh(cov).max(), 0))`).
func maxSigma(_ cov: [Double]) -> Double {
    let ev = LinAlg.eigvalsh(Matrix(rows: 2, cols: 2, data: cov))
    return max(ev.max() ?? .nan, 0.0).squareRoot()
}

extension Engine {
    /// `_finalize`: covariance, contact state, how the flight ended, frames covered,
    /// confidence — the ShotSegment3D + extras.
    func finalize(_ prob: Problem, _ fit: Fit, _ tAll: [Double], _ framesObs: [Int], _ framesAll: [Int]) -> ReconstructedSegment {
        let th = th7(fit.th, prob.LFixed)
        let n = prob.nObs, pr = prob.prior, tRef = prob.tRef
        // covariance with log L always included, scaled by the robust reduced chi²
        let (r7, J7o) = prob.residuals(th, jac: true, fitDrag: true)
        let J7 = J7o!
        var e = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let s = max(prob.sw[i], 1e-9)
            let ex = r7[2 * i] * R3.sigmaPx / s, ey = r7[2 * i + 1] * R3.sigmaPx / s
            e[i] = (ex * ex + ey * ey).squareRoot()
        }
        let cut = max(3 * Num.median(e), 4.0)
        var inl = e.map { $0 < cut }
        if !inl.contains(true) { inl = [Bool](repeating: true, count: n) }
        var se2 = 0.0, nin = 0, ruv2 = 0.0
        for i in 0..<n where inl[i] {
            se2 += e[i] * e[i]; nin += 1
            ruv2 += r7[2 * i] * r7[2 * i] + r7[2 * i + 1] * r7[2 * i + 1]
        }
        let rmse = (se2 / Double(nin)).squareRoot()
        let w7 = robust(r7, n).wts
        let m = r7.count
        var A = [Double](repeating: 0, count: 49)
        for a in 0..<7 { for b in 0..<7 {
            var s = 0.0
            for i in 0..<m { s += J7[i * 7 + a] * (J7[i * 7 + b] * w7[i]) }
            A[a * 7 + b] = s
        } }
        let dof = max(2 * nin - prob.npar, 1)
        let s2 = max(1.0, ruv2 / Double(dof))
        let cov7 = invert(A, 7).map { $0.map { $0 * s2 } } ?? [Double](repeating: .nan, count: 49)
        let covOk = cov7.allSatisfy(\.isFinite)

        // contact epoch: report p0/v0 at the hit time when it is known and close
        var tauC = 0.0
        if let st = pr.startT {
            let tc = st - tRef
            if -cfg.maxBackExtrap <= tc && tc <= 0.05 { tauC = tc }
        }
        let (Pc, Vc) = stateAt(th, tauC)
        let tauLast = prob.tr[prob.tr.count - 1]
        let (PLast, VLast) = stateAt(th, tauLast)

        // how the flight ended
        var landingXY: SIMD2<Double>? = nil, landingT: Double? = nil
        var landingCov: [Double]? = nil, extrapXY: SIMD2<Double>? = nil
        var endReason = SegmentEndReason.unknown
        var lnd: (xy: SIMD2<Double>, tau: Double)? = nil
        let side = pr.expectedSide
        let hitterSgn = (side == .near || side == .far) ? (side == .near ? -1.0 : 1.0) : copysign(1.0, Pc.x)
        // where the shuttle came to rest afterwards, signed towards the hitter's half
        let restX: Double? = (pr.restXYWorld != nil && pr.end != .hit) ? hitterSgn * pr.restXYWorld!.x : nil
        var tauEnd: Double
        var tFrom = tauLast
        if pr.end == .hit {
            endReason = .hit
            tauEnd = pr.endT.map { $0 - tRef } ?? tauLast
        } else if pr.endAtGap {
            tauEnd = tauLast                                     // unseen end: no landing, no net call
        } else {
            if PLast.z <= 0.0 {                                  // fitted path already down: back up
                for back in [0.01, 0.03, 0.06, 0.12, 0.25] where stateAt(th, tauLast - back).p.z > 0 {
                    tFrom = tauLast - back
                    break
                }
            }
            lnd = landing(th, tFrom)
            tauEnd = lnd?.tau ?? tauLast
        }
        let nc = netCrossing(th, tauC, tauEnd)
        let netH = nc?.z
        if let lnd {
            // open end: descending and reaching the floor soon after the last sighting
            let floorish = pr.end == .floor || (pr.end == nil && VLast.z < 0 && lnd.tau - tauLast < cfg.openLandS)
            // a net fault needs the crossing below the tape AND no evidence it got over
            var over = nc != nil && tauLast > nc!.tau + cfg.netSeenS
            // never got across: the fitted flight lands on the hitter's own half
            var own = nc == nil && hitterSgn * lnd.xy.x > cfg.ownSideLandM
            if let restX {
                over = over || restX < -0.8
                own = own || restX > cfg.ownSideRestM
            }
            if !over && (own || (nc != nil && nc!.z < Reconstruct3D.netHeightAt(nc!.y))) {
                endReason = .net
            } else if floorish {
                endReason = .floor
                landingXY = lnd.xy
                landingT = tRef + lnd.tau
                if covOk, let lc = landingCovFD(th, cov7, tFrom) {
                    let cf = calibFloorCov(prob.cam, lnd.xy)
                    landingCov = zip(lc, cf).map { $0 + $1 }
                }
            } else {
                extrapXY = lnd.xy
            }
        } else if pr.end != .hit, !pr.endAtGap, let restX, restX > cfg.ownSideRestM {
            endReason = .net                                     // no landing, at rest on its own half
        }

        // frames covered: contact (or first obs) → landing / next contact / last obs
        let tauHi = (endReason == .floor || endReason == .hit) ? tauEnd : tauLast
        var covTaus: [Double] = [], frCov: [Int] = []
        let lo = min(tauC, 0.0) - 1e-9, hi = tauHi + 1e-9
        for (i, ta) in tAll.enumerated() {
            let tau = ta - tRef
            if tau.isFinite && tau >= lo && tau <= hi { covTaus.append(tau); frCov.append(framesAll[i]) }
        }
        let PCov = stateAt(th, covTaus).P

        var lsig: Double? = nil
        if let lc = landingCov, lc.allSatisfy(\.isFinite) { lsig = maxSigma(lc) }
        let spSig: Double? = covOk ? speedSigma(th, cov7, tauC) * 3.6 : nil
        let sp0 = norm3(Vc) * 3.6
        var conf = min(1.0, Double(n) / 15.0) / (1.0 + (rmse / 3.0) * (rmse / 3.0))
        if let lsig { conf /= 1.0 + lsig / 0.3 }
        if let spSig, sp0 > 1 { conf /= 1.0 + (spSig / sp0) / 0.1 }
        let seg = ShotSegment3D(
            startFrame: frCov.first ?? framesObs[0], endFrame: frCov.last ?? framesObs[framesObs.count - 1],
            t0: tRef + tauC, p0: Pc, v0: Vc, dragLength: exp(th[6]), rmsePx: rmse, frames: frCov,
            positions: PCov, landingXY: landingXY, landingT: landingT, netCrossingHeight: netH,
            confidence: conf, netCrossingY: nc?.y, endReason: endReason)
        return ReconstructedSegment(
            segment: seg, thetaRef: th, tRef: tRef, covTheta: cov7, landingCov: landingCov,
            landingSigmaM: lsig, speed0SigmaKmh: spSig, nObs: n,
            inlierFrac: Double(nin) / Double(n), landingExtrapolatedXY: extrapXY,
            fitDrag: prob.fitDrag, tEnd: tRef + tauHi)
    }
}
