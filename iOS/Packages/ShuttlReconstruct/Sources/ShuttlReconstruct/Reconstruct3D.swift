import Foundation
import ShuttlCore
import simd

/// Public entry points of the 3-D reconstruction stage (port of reconstruct3d.py).
public enum Reconstruct3D {
    public enum ServeSide: String, Sendable { case right, left }

    /// Net top height (m) at lateral position y (sags from posts to centre).
    public static func netHeightAt(_ y: Double) -> Double {
        let a = min(abs(y) / CourtModel.doublesHalfWidth, 1.0)
        return CourtModel.netHeightCentre + (CourtModel.netHeightPost - CourtModel.netHeightCentre) * a * a
    }

    /// `fit_segment`: fit one ODE flight to pixel observations (NaN / zero-weight rows
    /// skipped). Parametrised internally at the first observed time; the returned p0/v0/t0
    /// are at the contact time prior.startT when given (else the first observation).
    /// `frames` labels the rows of t (default 0..N-1); `initial` = (θ7, epoch) warm start.
    /// nil with < 5 valid points or no plausible solution.
    public static func fitSegment(t: [Double], uv: [SIMD2<Double>], w: [Double]? = nil, camera: CameraModel,
                                  prior: SegmentPrior? = nil, fitDrag: Bool = true, frames: [Int]? = nil,
                                  initial: (theta: [Double], epoch: Double)? = nil,
                                  config: ReconstructConfig = .init()) -> ReconstructedSegment? {
        Engine(cfg: config, camera: camera).fitSegment(t: t, uv: uv, w: w, prior: prior, fitDrag: fitDrag,
                                                       frames: frames, ini: initial)
    }

    // MARK: uncertainty / speed helpers

    /// 2×2 (row-major) landing-point covariance (m²): fit covariance propagated through the
    /// ODE plus calibration error; nil if the flight has no landing point.
    public static func landingCovariance(_ s: ReconstructedSegment) -> [Double]? {
        if s.landingCov != nil || s.segment.landingT == nil { return s.landingCov }
        return landingCovFD(s.thetaRef, s.covTheta, max(0.0, s.segment.landingT! - s.tRef - 0.05))
    }

    /// Largest 1-σ landing uncertainty (m), from `landingCovariance`.
    public static func landingSigma(_ s: ReconstructedSegment) -> Double? {
        guard let cov = landingCovariance(s), cov.allSatisfy(\.isFinite) else { return nil }
        return maxSigma(cov)
    }

    /// (t seconds, speed km/h) from contact (t0) to landing / segment end.
    public static func speedProfile(_ s: ReconstructedSegment, dt: Double = 1.0 / 120.0) -> (t: [Double], kmh: [Double]) {
        let t0 = s.segment.t0
        let tEnd = s.segment.landingT ?? s.tEnd
        let ts = arange(t0, max(tEnd, t0 + dt) + 1e-9, dt)
        let V = stateAt(s.thetaRef, ts.map { $0 - s.tRef }).V
        return (ts, V.map { norm3($0) * 3.6 })
    }

    /// Speed right after contact (the "smash speed"), km/h.
    public static func initialSpeedKmh(_ s: ShotSegment3D) -> Double { norm3(s.v0) * 3.6 }

    /// Speed (km/h) of the fitted flight at an arbitrary contact time.
    public static func contactSpeedKmh(_ s: ReconstructedSegment, tContact: Double) -> Double {
        norm3(stateAt(s.thetaRef, tContact - s.tRef).v) * 3.6
    }

    // MARK: in / out calls

    /// court_model.in_out_margin and the outward normal of the binding boundary.
    static func rallyMargin(_ x: Double, _ y: Double, _ doubles: Bool) -> (Double, SIMD2<Double>) {
        let halfW = doubles ? CourtModel.doublesHalfWidth : CourtModel.singlesHalfWidth
        let mx = CourtModel.halfLength + CourtModel.lineWidth / 2 - abs(x)
        let my = halfW + CourtModel.lineWidth / 2 - abs(y)
        if mx <= my { return (mx, SIMD2(copysign(1.0, x), 0.0)) }
        return (my, SIMD2(0.0, copysign(1.0, y)))
    }

    /// Service court (right | left, from the server's own view facing the net) of a server
    /// standing at serverXY. Near server faces +x, so its right is −y.
    public static func serveSideOf(_ serverXY: SIMD2<Double>, serverHalf: CourtSide? = nil) -> ServeSide {
        let half = serverHalf ?? CourtModel.sideOf(x: serverXY.x)
        return (serverXY.y < 0) == (half == .near) ? .right : .left
    }

    /// Legal receiving service court (xMin, xMax, yMin, yMax) on line centre-lines.
    public static func serveBox(serverHalf: CourtSide, serveSide: ServeSide, doubles: Bool)
        -> (xMin: Double, xMax: Double, yMin: Double, yMax: Double) {
        let tx = serverHalf == .near ? 1.0 : -1.0                         // target half
        let serverYs = (serveSide == .right ? -1.0 : 1.0) * (serverHalf == .near ? 1.0 : -1.0)
        let ty = -serverYs                                                 // diagonal
        let back = doubles ? CourtModel.doublesLongServiceX : CourtModel.halfLength
        let halfW = doubles ? CourtModel.doublesHalfWidth : CourtModel.singlesHalfWidth
        let xa = tx * CourtModel.shortServiceX, xb = tx * back
        let ya = 0.0, yb = ty * halfW
        return (min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb))
    }

    static func serveMarginNormal(_ x: Double, _ y: Double, _ half: CourtSide, _ side: ServeSide,
                                  _ doubles: Bool) -> (Double, SIMD2<Double>) {
        let b = serveBox(serverHalf: half, serveSide: side, doubles: doubles)
        let h = CourtModel.lineWidth / 2                                   // every bounding line counts as in
        let cands: [(Double, SIMD2<Double>)] = [(x - (b.xMin - h), SIMD2(-1.0, 0.0)), ((b.xMax + h) - x, SIMD2(1.0, 0.0)),
                                                (y - (b.yMin - h), SIMD2(0.0, -1.0)), ((b.yMax + h) - y, SIMD2(0.0, 1.0))]
        var best = cands[0]
        for c in cands.dropFirst() where c.0 < best.0 { best = c }
        return best
    }

    /// Signed distance (m) of a serve landing from its service-court boundary (+ = in).
    public static func serveMargin(x: Double, y: Double, serverHalf: CourtSide, serveSide: ServeSide,
                                   doubles: Bool) -> Double {
        serveMarginNormal(x, y, serverHalf, serveSide, doubles).0
    }

    /// IN/OUT call from the landing point and its covariance ('unsure' when |margin| < 2σ
    /// along the binding boundary normal, or no landing). See `call_in_out`.
    public static func callInOut(_ s: ReconstructedSegment?, doubles: Bool, isServe: Bool = false,
                                 serveSide: ServeSide? = nil, config: ReconstructConfig = .init()) -> InOutCall {
        guard let s else { return InOutCall(call: .unsure, marginM: nil, sigmaM: nil, pIn: nil) }
        return callInOut(s.segment, landingCov: landingCovariance(s), doubles: doubles, isServe: isServe,
                         serveSide: serveSide, config: config)
    }

    /// `call_in_out` for a contract segment with an explicit landing covariance (2×2
    /// row-major; nil → the default σ, like a segment without the fit extras).
    public static func callInOut(_ seg: ShotSegment3D?, landingCov: [Double]?, doubles: Bool, isServe: Bool = false,
                                 serveSide: ServeSide? = nil, config cfg: ReconstructConfig = .init()) -> InOutCall {
        guard let seg, let land = seg.landingXY else {
            return InOutCall(call: .unsure, marginM: nil, sigmaM: nil, pIn: nil)
        }
        let x = land.x, y = land.y
        var ambiguous = false
        var margin: Double, nrm: SIMD2<Double>
        if isServe {
            let half = CourtModel.sideOf(x: seg.p0.x)
            var side = serveSide
            if side == nil, let feet = seg.hitterFeetXY, abs(feet.y) > cfg.serveSideMinY {
                side = serveSideOf(feet, serverHalf: half)
            }
            if side == nil && abs(seg.p0.y) > cfg.serveSideP0Y {
                side = serveSideOf(SIMD2(seg.p0.x, seg.p0.y), serverHalf: half)
            }
            if let side {
                (margin, nrm) = serveMarginNormal(x, y, half, side, doubles)
            } else {                                                        // try both courts
                let (mr, nr) = serveMarginNormal(x, y, half, .right, doubles)
                let (ml, nl) = serveMarginNormal(x, y, half, .left, doubles)
                ambiguous = (mr >= 0) != (ml >= 0)
                (margin, nrm) = mr >= ml ? (mr, nr) : (ml, nl)
            }
        } else {
            (margin, nrm) = rallyMargin(x, y, doubles)
        }
        var sigma: Double
        if let cov = landingCov, cov.allSatisfy(\.isFinite) {
            let a = nrm.x * cov[0] + nrm.y * cov[2], b = nrm.x * cov[1] + nrm.y * cov[3]
            sigma = cfg.sigmaCal * max(a * nrm.x + b * nrm.y, 0.0).squareRoot()
        } else {
            sigma = cfg.sigmaDefaultM
        }
        sigma = hypot(sigma, cfg.sigmaFloorM)
        let call: InOutCall.Call = abs(margin) < 2 * sigma || ambiguous ? .unsure : (margin > 0 ? .in : .out)
        let pIn = 0.5 * (1.0 + erf(margin / (sigma * 2.0.squareRoot())))
        return InOutCall(call: call, marginM: margin, sigmaM: sigma, pIn: pIn)
    }
}
