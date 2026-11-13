import Foundation

/// Port of `LinkerConfig` (linker.py). All lengths are fractions of the image
/// diagonal D (1080p: D = 2203 px), all times in seconds, costs in nats.
/// Defaults are the Python defaults.
public struct LinkerConfig: Sendable, Equatable, Codable {
    // ── candidates ─────────────────────────────────────────────────────────
    public var maxCandidates: Int = 5          // top-k per frame kept (by score)
    public var minScore: Double = 0.05         // ignore weaker candidates
    // ── association DP ─────────────────────────────────────────────────────
    public var maxSpeed: Double = 8.0          // D/s hard gate
    public var maxLinkGapS: Double = 0.30      // longest burst of misses bridged by one link
    public var sigmaPx: Double = 1.2e-3        // detector noise (1080p: 2.6 px)
    public var accelSigma: Double = 1.5        // D/s² unmodelled acceleration
    public var dragLengthMin: Double = 0.12    // D, smallest image-space drag length
    public var dragLengthYoung: Double = 0.015 // D, same for the 1st velocity check after a hit
    public var speedUpMax: Double = 0.25       // along-track overshoot allowed (perspective)
    public var kinkCost: Double = 5.0          // cost of a kink (hit)
    public var kinkMaxGapS: Double = 0.10      // a kink link may skip at most this long
    public var firstCost: Double = 1.0         // motion cost of a link with no velocity yet
    public var birthCost: Double = 10.0        // starting a new chain
    public var missCost: Double = 0.35         // per skipped frame inside a chain
    public var detReward: Double = 2.0         // base reward for adopting a candidate
    public var scoreWeight: Double = 1.0       // × logit(score)
    public var minChainLen: Int = 4            // shorter chains are discarded
    // ── weak candidates ────────────────────────────────────────────────────
    public var strongScore: Double = 0.5       // detector decision threshold
    public var minStrong: Int = 2              // strong obs a flight piece needs …
    public var weakReachS: Double = 0.3        // … and weak obs must be this close to one
    public var weakFlySpeed: Double = 0.3      // D/s: a weak-only piece is kept only if it flies
    public var weakFlyMinS: Double = 0.25      // this fast (median) for this long
    public var weakBanRounds: Int = 3          // re-associate after banning rejected weak runs
    public var readoptGate: Double = 0.014     // D: a fill takes a strong detection this close
    // ── scene cuts ─────────────────────────────────────────────────────────
    public var cutDetect: Bool = true
    public var cutMinCells: Int = 3
    // ── static distractors ─────────────────────────────────────────────────
    public var staticCell: Double = 6e-3
    public var staticBlockS: Double = 20.0
    public var staticMinS: Double = 10.0
    public var staticOccLo: Double = 0.30
    public var staticOccHi: Double = 0.60
    public var staticFar: Double = 0.03
    public var staticMinPresentS: Double = 1.0
    public var staticLo: Double = 0.10
    public var staticHi: Double = 0.30
    public var staticCost: Double = 6.0
    // ── kinks (hits) ───────────────────────────────────────────────────────
    public var kinkWindowS: Double = 0.12
    public var kinkMinPts: Int = 3
    public var kinkMaxPts: Int = 12
    public var kinkDeg: Int = 3
    public var kinkZ: Double = 4.5
    public var kinkZRefine: Double = 3.5
    public var kinkMinDv: Double = 0.05
    public var kinkNmsS: Double = 0.15
    public var kinkLocS: Double = 0.04
    public var kinkLocPts: Int = 5
    public var kinkLocDeg: Int = 1
    public var kinkLocCoarse: Bool = true
    // ── smoothing / gap filling ────────────────────────────────────────────
    public var maxGapS: Double = 0.25
    public var maxExtrapFrames: Int = 3
    public var jerkFloor: Double = 60.0
    public var jerkDragLen: Double = 0.25
    public var jerkCorrS: Double = 0.03
    public var huberK: Double = 2.0
    public var outlierK: Double = 5.0
    public var irlsIters: Int = 4
    public var refineKinks: Bool = true
    public var refineRounds: Int = 3
    // ── repeated video frames ──────────────────────────────────────────────
    public var repeatDetect: Bool = true
    public var repeatMinMove: Double = 2e-3
    public var repeatMinCount: Int = 4
    public var kinkResidK: Double = 3.0

    public init() {}
}
