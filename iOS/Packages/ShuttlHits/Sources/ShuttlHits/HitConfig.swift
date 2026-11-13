import Foundation

/// Tunables of the hit detector — field-for-field port of `hits.HitConfig`
/// (same defaults). Speeds are normalised by the image diagonal per second.
public struct HitConfig: Sendable, Equatable, Codable {
    // change-point scan
    public var windowS: Double = 0.15          // nominal half-window
    public var maxWindowS: Double = 0.40       // half-window may grow to this to get min_side points
    public var minSide: Int = 3                // observations required on each side of tau
    public var wantSide: Int = 5               // windows grow (<= max_window_s) to hold this many
    public var maxSide: Int = 24               // cap (keeps 240 fps windows small)
    public var sigmaPx: Double? = nil          // detection noise; nil = estimate from track
    public var nmsS: Double = 0.20             // candidate NMS radius (decoder enforces min_sep_s)
    public var refineSteps: Int = 9            // sub-frame grid over +-1 frame for tau
    public var refineModel: String = "1"       // contact time model: '1' two quadratics | 'vc' cubic+kink | 'v'
    public var lamMin: Double = 12.0           // minimum Lambda to become a candidate
    public var lamMinWeak: Double = 4.0        // weaker kinks qualify only inside a player's reach
    public var weakReachMin: Double = 0.9      // reach term required for such weak candidates
    public var lamRef: Double = 15.0           // Lambda at which kinematic log-odds = 0
    // plausibility
    public var minSpeedOut: Double = 0.10      // slower after the change = carried / rolling
    public var staticPx: Double = 6.0          // spread below which the shuttle is at rest (per 720p)
    public var staticS: Double = 0.35          // look-ahead / -behind for rest tests
    public var settleS: Double = 0.8           // comes to rest within this after a change ...
    public var settleFrac: Double = 0.05       // ... never leaving this radius (x diag) = floor bounce
    public var settleSpeed: Double = 0.25      // ... and leaves slower than this (diag/s)
    public var deadTimeS: Double = 1.0         // gap before a flight start = new serve
    public var maxOutOfViewS: Double = 4.0     // longer gaps are dead time even at the border
    public var borderBand: Double = 0.06       // fraction of W/H: shuttle leaving/entering view
    public var borderPx: Double = 8.0          // meeting point this far outside image = not a hit
    // sequence decoder
    public var minSepS: Double = 0.22          // two contacts closer than this are impossible
    public var shortFlightS: Double = 0.28     // flight-duration prior plateau [short, long]
    public var longFlightS: Double = 1.7
    public var longFlightSd: Double = 0.45
    public var missedHitS: Double = 0.6        // same-side successor needs room for two flights
    public var logEpsSame: Double = -3.0       // log prob per missed contact between two accepted
    public var logEpsFastSame: Double = -9.0   // same-side within missed_hit_s (almost impossible)
    public var pairYWeight: Double = 1.0       // weight of the pairwise image-y side cue
    public var playerWeight: Double = 1.0
    public var cameraWeight: Double = 1.0
    public var minHitLogodds: Double = -4.0    // final gate on a decoded hit's own log-odds
    /// Python's `use_linker_kinks` calls linker.find_kinks (another stage). Here the
    /// caller passes those kink times (s) explicitly; nil/empty = the Python default (off).
    public var linkerKinkTimes: [Double]? = nil
    public var useRawXY: Bool = true           // score contacts on Track2D.raw_xy where present
    public var learnedBlend: Double = 0.5      // weight of a learned model's logit
    // hidden contact above a low phone's frame (hits.py _hidden_contacts)
    public var hiddenContacts: Bool = true
    public var hiddenGapS: Double = 2.0        // top-exit -> top-entry gap at least this long
    public var hiddenLogodds: Double = 2.0     // kinematic log-odds of such a candidate
    // contact right after a presentation-clock hole (hits.py _gap_contacts, robustness F4)
    public var gapContacts: Bool = true
    public var gapFirstObsS: Double = 0.1      // first observation after the gap within this of it
    public var gapSide: Int = 6                // observations used on each side of the gap
    public var gapLamRef: Double = 150.0       // Lambda (quadratic across vs split) at log-odds 0
    public var gapLogoddsMax: Double = 2.0     // cap of such a candidate's kinematic log-odds

    public init() {}

    enum CodingKeys: String, CodingKey {
        case windowS = "window_s", maxWindowS = "max_window_s", minSide = "min_side", wantSide = "want_side"
        case maxSide = "max_side", sigmaPx = "sigma_px", nmsS = "nms_s", refineSteps = "refine_steps"
        case refineModel = "refine_model", lamMin = "lam_min", lamMinWeak = "lam_min_weak"
        case weakReachMin = "weak_reach_min", lamRef = "lam_ref", minSpeedOut = "min_speed_out"
        case staticPx = "static_px", staticS = "static_s", settleS = "settle_s", settleFrac = "settle_frac"
        case settleSpeed = "settle_speed", deadTimeS = "dead_time_s", maxOutOfViewS = "max_out_of_view_s"
        case borderBand = "border_band", borderPx = "border_px", minSepS = "min_sep_s"
        case shortFlightS = "short_flight_s", longFlightS = "long_flight_s", longFlightSd = "long_flight_sd"
        case missedHitS = "missed_hit_s", logEpsSame = "log_eps_same", logEpsFastSame = "log_eps_fast_same"
        case pairYWeight = "pair_y_weight", playerWeight = "player_weight", cameraWeight = "camera_weight"
        case minHitLogodds = "min_hit_logodds", linkerKinkTimes = "linker_kink_times"
        case useRawXY = "use_raw_xy", learnedBlend = "learned_blend"
        case hiddenContacts = "hidden_contacts", hiddenGapS = "hidden_gap_s", hiddenLogodds = "hidden_logodds"
        case gapContacts = "gap_contacts", gapFirstObsS = "gap_first_obs_s", gapSide = "gap_side"
        case gapLamRef = "gap_lam_ref", gapLogoddsMax = "gap_logodds_max"
    }

    /// Decodes a (partial) Python `dataclasses.asdict(HitConfig(...))`; absent keys keep defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d(_ k: CodingKeys, _ v: inout Double) throws { if let x = try c.decodeIfPresent(Double.self, forKey: k) { v = x } }
        func i(_ k: CodingKeys, _ v: inout Int) throws { if let x = try c.decodeIfPresent(Int.self, forKey: k) { v = x } }
        try d(.windowS, &windowS); try d(.maxWindowS, &maxWindowS); try i(.minSide, &minSide)
        try i(.wantSide, &wantSide); try i(.maxSide, &maxSide)
        sigmaPx = try c.decodeIfPresent(Double.self, forKey: .sigmaPx)
        try d(.nmsS, &nmsS); try i(.refineSteps, &refineSteps)
        refineModel = try c.decodeIfPresent(String.self, forKey: .refineModel) ?? refineModel
        try d(.lamMin, &lamMin); try d(.lamMinWeak, &lamMinWeak); try d(.weakReachMin, &weakReachMin)
        try d(.lamRef, &lamRef); try d(.minSpeedOut, &minSpeedOut); try d(.staticPx, &staticPx)
        try d(.staticS, &staticS); try d(.settleS, &settleS); try d(.settleFrac, &settleFrac)
        try d(.settleSpeed, &settleSpeed); try d(.deadTimeS, &deadTimeS); try d(.maxOutOfViewS, &maxOutOfViewS)
        try d(.borderBand, &borderBand); try d(.borderPx, &borderPx); try d(.minSepS, &minSepS)
        try d(.shortFlightS, &shortFlightS); try d(.longFlightS, &longFlightS); try d(.longFlightSd, &longFlightSd)
        try d(.missedHitS, &missedHitS); try d(.logEpsSame, &logEpsSame); try d(.logEpsFastSame, &logEpsFastSame)
        try d(.pairYWeight, &pairYWeight); try d(.playerWeight, &playerWeight); try d(.cameraWeight, &cameraWeight)
        try d(.minHitLogodds, &minHitLogodds)
        linkerKinkTimes = try c.decodeIfPresent([Double].self, forKey: .linkerKinkTimes)
        useRawXY = try c.decodeIfPresent(Bool.self, forKey: .useRawXY) ?? useRawXY
        try d(.learnedBlend, &learnedBlend)
        hiddenContacts = try c.decodeIfPresent(Bool.self, forKey: .hiddenContacts) ?? hiddenContacts
        try d(.hiddenGapS, &hiddenGapS); try d(.hiddenLogodds, &hiddenLogodds)
        gapContacts = try c.decodeIfPresent(Bool.self, forKey: .gapContacts) ?? gapContacts
        try d(.gapFirstObsS, &gapFirstObsS); try i(.gapSide, &gapSide)
        try d(.gapLamRef, &gapLamRef); try d(.gapLogoddsMax, &gapLogoddsMax)
    }
}

/// Per-candidate feature names for a learned re-scorer, in Python's `HIT_FEATURES` order.
public let HIT_FEATURES: [String] = [
    "log_lam", "log_lam_v", "speed_in", "speed_out", "log_speed_ratio",
    "cos_turn", "vy_flip", "static_before", "static_after", "outside",
    "gap_s", "player_reach", "is_flight_start",
]

public enum HitConstants {
    public static let racketReachM = 0.65     // racket head beyond the wrist
    public static let playerHeightM = 1.75    // bbox height ~ standing height (px -> m scale)
    public static let leftWrist = 9, rightWrist = 10   // COCO-17
}
