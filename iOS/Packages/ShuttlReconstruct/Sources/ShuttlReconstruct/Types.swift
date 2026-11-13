import Foundation
import ShuttlCore
import simd

/// Decision thresholds of reconstruct3d.py (module constants), Python defaults.
public struct ReconstructConfig: Sendable, Equatable {
    /// s; contact state is extrapolated at most this far back (MAX_BACK_EXTRAP).
    public var maxBackExtrap = 0.35
    /// s; tracked this long past the net plane → it got over (NET_SEEN_S).
    public var netSeenS = 0.1
    /// s; open-ended flight: extrapolate a landing at most this far (OPEN_LAND_S).
    public var openLandS = 0.6
    /// fitted landing this far on the hitter's half → never got over (OWN_SIDE_LAND_M).
    public var ownSideLandM = 0.3
    /// at rest this far on the hitter's half → never got over (OWN_SIDE_REST_M).
    public var ownSideRestM = 0.15
    /// > this share of the last 40 % off by > thr → inconsistent (TAIL_BAD_FRAC).
    public var tailBadFrac = 0.25
    /// growth stops at 2 points off by > this × the clean-seed sigma (PREFIX_THR_MULT).
    public var prefixThrMult = 3.0
    /// px (PREFIX_MIN_THR).
    public var prefixMinThr = 4.0
    /// whole-arc / trimmed median error on the kept points (PREFIX_GAIN).
    public var prefixGain = 1.8
    /// model-error floor on the landing σ (SIGMA_FLOOR_M).
    public var sigmaFloorM = 0.01
    /// σ when no covariance is available (SIGMA_DEFAULT_M).
    public var sigmaDefaultM = 0.15
    /// calibration of the covariance σ along the boundary normal (SIGMA_CAL).
    public var sigmaCal = 0.85
    /// |feet y| (m) that decides the server's service court (SERVE_SIDE_MIN_Y).
    public var serveSideMinY = 0.05
    /// |contact y| (m) that decides it when no feet are known (SERVE_SIDE_P0_Y).
    public var serveSideP0Y = 0.25
    /// longest plausible flight (MAX_FLIGHT_S).
    public var maxFlightS = 2.8
    /// a longer detection gap ends a flight with no next hit (MAX_GAP_S).
    public var maxGapS = 1.2
    /// stationary this long → on the floor (REST_S).
    public var restS = 0.25
    /// MIN_OBS.
    public var minObs = 6
    /// A flight cut at a presentation-clock gap (no seen end, no end constraint) is kept only
    /// when σ(speed) ≤ this × speed (GAP_SPEED_REL_SIGMA).
    public var gapSpeedRelSigma = 0.15
    /// Worker threads for the independent per-piece fits of `reconstructSegments` (Swift
    /// only; 1 = serial). Results are identical for any value: every fit is self-contained.
    public var maxConcurrency = ProcessInfo.processInfo.activeProcessorCount
    /// feet → contact point (arm + racket reach) (FEET_SIGMA_M).
    public var feetSigmaM = 0.7
    /// FEET_SIGMA_RECV_M.
    public var feetSigmaRecvM = 0.9

    public init() {}
}

/// A fitted flight: the `ShotSegment3D` contract plus the non-contract extras the Python
/// module attaches to it (theta_ref, cov_theta, landing_cov, … — read via getattr there).
public struct ReconstructedSegment: Sendable {
    public var segment: ShotSegment3D
    /// (p0, v0, log L) at `tRef` (the first observed time or the contact epoch).
    public var thetaRef: [Double]
    public var tRef: Double
    /// 7×7 row-major covariance over θ at tRef (NaN-filled if the normal matrix was singular).
    public var covTheta: [Double]
    /// 2×2 row-major landing-point covariance (m²) incl. calibration error.
    public var landingCov: [Double]?
    public var landingSigmaM: Double?
    public var speed0SigmaKmh: Double?
    public var nObs: Int
    public var inlierFrac: Double
    public var landingExtrapolatedXY: SIMD2<Double>?
    public var fitDrag: Bool
    /// landing / next contact / last observation (absolute s).
    public var tEnd: Double
    /// false: p0/v0 at the first sighting (contact long before it).
    public var contactObserved: Bool = true
    public var contactGapS: Double = 0
    public var nTrimmed: Int = 0
    // set by reconstructSegments
    public var hitFrame: Int?
    public var hitTRefined: Double?
    public var restXY: SIMD2<Double>?
    /// The flight's observations stop at a presentation-clock gap: its end was never seen
    /// (end reason unknown, no landing). Python `ends_at_pts_gap`.
    public var endsAtPTSGap: Bool = false
    /// The hit was before a clock gap and the flight is first seen after it: p0/v0/speed are
    /// at the first sighting, not the contact. Python `starts_after_pts_gap`.
    public var startsAfterPTSGap: Bool = false

    public var endReason: SegmentEndReason { segment.endReason }
    public var landingXY: SIMD2<Double>? { segment.landingXY }
    public var p0: SIMD3<Double> { segment.p0 }
    public var v0: SIMD3<Double> { segment.v0 }
    public var t0: Double { segment.t0 }
}

/// Result of `callInOut` (Python's dict(call, margin_m, sigma_m, p_in)).
public struct InOutCall: Sendable, Equatable {
    public enum Call: String, Sendable { case `in`, out, unsure }
    public var call: Call
    public var marginM: Double?
    public var sigmaM: Double?
    public var pIn: Double?
}
