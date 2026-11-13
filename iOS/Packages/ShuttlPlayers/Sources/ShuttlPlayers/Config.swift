import Foundation
import ShuttlCore

/// Every tunable of players.py, defaults = the Python module constants.
public struct PlayersConfig: Sendable, Equatable {
    // detection / gating
    public var kpConf = 0.35                  // KP_CONF keypoint visibility threshold
    public var courtMargin = 1.5              // COURT_MARGIN m outside the doubles lines still eligible
    public var sideTol = 0.4                  // SIDE_TOL m a foot may stray over the net line
    public var ankleH = 0.08                  // ANKLE_H m, ankle joint above the floor
    public var personH = 1.8                  // PERSON_H m, scale prior without a camera
    public var wSit = 2.0                     // W_SIT cost of a fully seated pose
    // Viterbi
    public var vMax = 7.5                     // V_MAX m/s player speed gate
    public var lookbackS = 1.0                // LOOKBACK_S max gap bridged with a motion model
    public var cMiss = 2.5                    // C_MISS cost per sampled frame without the player
    public var cRestart = 6.0                 // C_RESTART re-acquire after a long gap
    // static people
    public var staticWinS = 6.0
    public var staticRM = 0.3
    public var staticMaxRef = 24
    public var wStatic = 3.0
    public var wStaticNoCam = 1.0
    // smoothing
    public var accSigma = 5.0                 // ACC_SIGMA m/s²
    public var maxInterpS = 0.5               // MAX_INTERP_S
    // doubles identity
    public var qLink = 225.0
    public var v0Link = 0.01
    public var kfTail = 16
    public var ambig = 2.0
    public var cNew = 4.0
    public var cCoast = 2.0
    public var trackletGapS = 1.0
    public var wApp = 8.0
    public var cDrop = 3.0
    public var stitchBeam = 400
    // movement stats
    public var speedWinS = 0.3
    public var speedCap = 9.0

    public init() {}
    public static let `default` = PlayersConfig()
}

/// COCO-17 indices used by players.py.
enum KP {
    static let lSh = 5, rSh = 6, lHip = 11, rHip = 12, lKnee = 13, rKnee = 14, lAnk = 15, rAnk = 16
}

/// Appearance histogram layout (APP_H × APP_S hue/sat bins + APP_V achromatic value bins).
public enum Appearance {
    public static let hBins = 8, sBins = 2, vBins = 3
    public static let region = hBins * sBins + vBins     // 19
    public static let dim = 2 * region                   // 38 (shirt, shorts)
}

/// Movement heat-map grid.
public enum Heat {
    public static let rows = 12, cols = 10
}

/// Pose sampling step for a video (pipeline_v2.player_step, PLAYER_HZ = 15).
public func playerStep(fps: Double?, override: Int? = nil, playerHz: Double = 15.0) -> Int {
    if let override { return max(1, override) }
    let f = (fps == nil || fps == 0) ? 30.0 : fps!          // `fps or 30.0`
    return max(1, Int((f / playerHz).rounded(.toNearestOrEven)))
}
