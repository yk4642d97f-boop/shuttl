import Foundation
import CoreGraphics

// MARK: - TrackerConfig
// Lengths are fractions of the image diagonal D (1080p: D = 2203 px), times in seconds —
// the same normalisation as backend linker.LinkerConfig, so one config fits any resolution,
// fps and variable frame rate. Costs are in nats.

struct TrackerConfig: Sendable {
    // Candidates
    var minScore: Float = 0.3
    var maxCandidates = 2
    // Motion model / gating
    var maxSpeed = 8.0               // D/s hard gate (≈ 400 km/h close to the camera)
    var sigmaMeasurement = 1.5e-3    // D, detector noise (1080p: 3.3 px)
    var initialSpeedSigma = 1.5      // D/s, velocity prior of a new track
    var jerkFloor = 60.0             // D/s³, baseline unmodelled jerk (perspective, gravity)
    var dragLength = 0.12            // D, image-space drag length: jerk prior grows as 2v³/L²
    var jerkCorrelation = 0.03       // s
    var gateChi2 = 16.0              // Mahalanobis² gate (2 dof)
    var acceptCost = 8.0             // max association cost
    var maxCoast = 0.30              // s of misses bridged before a track ends
    // Confirmation
    var confirmHits = 3
    var confirmMinTravel = 0.01      // D the candidate must move before it counts as a shuttle
    var tentativeMaxGap = 0.10       // s a tentative survives without a detection
    var maxTentatives = 6
    // Static distractors (spatial hit-count grid with exponential memory)
    var staticCell = 6e-3            // D (1080p: 13 px)
    var staticMemory = 4.0           // s
    var staticMinSeconds = 2.0       // occupancy denominator floor
    var staticOccLo = 0.30, staticOccHi = 0.60
    var staticFar = 0.03             // D, "the shuttle is elsewhere"
    var staticMinPresentSeconds = 1.0
    var staticElsewhereLo = 0.10, staticElsewhereHi = 0.30
    var staticCost = 10.0            // full penalty
    // Hits (kinks in the image path)
    var kinkWindow = 0.12            // s, line fit on each side
    var kinkMinPoints = 3
    var kinkMaxPoints = 16
    var kinkZ = 4.0                  // noise part of the Δv threshold (sigmas)
    var kinkAccel = 1.2              // D/s², deterministic allowance (gravity ⊥ v)
    var kinkMinDV = 0.10             // D/s floor on unexplained Δv
    var kinkRhoHi = 1.15             // v_out may be up to 1.15·v_in (perspective)
    var kinkNMS = 0.15               // s between hits
    // Output
    var trailDuration = 0.5          // s of path shown by the live overlay
    var speedHold = 3.0              // s an approx. flight speed stays on screen
    // Time
    var nominalFrameInterval = 1.0 / 30   // only when a frame has no timestamp
    var maxFrameGap = 1.0            // s; a longer gap (pause, seek) ends the track
}

// MARK: - Tracker I/O

struct TrackerCandidate: Sendable, Equatable {
    /// Full-resolution pixel in the upright frame.
    var pixel: CGPoint
    var score: Float
}

struct TrackPoint: Sendable, Equatable {
    var time: Double
    var pixel: CGPoint
    var score: Float
}

struct HitEvent: Sendable, Equatable {
    enum Source: Sendable { case kink, reacquired }
    var time: Double
    var pixel: CGPoint
    var source: Source
    /// Direction change between incoming and outgoing image velocity, degrees.
    var turnDegrees: Double?
}

/// Observed path between two hits (or until the track was lost).
struct FlightSegment: Sendable {
    enum End: Sendable { case hit, lost, reset }
    var points: [TrackPoint]
    var end: End
    /// Monocular 3-D estimate (calibrated cameras only). Approximate by construction.
    var approxSpeedKmh: Double?
    var duration: Double { (points.last?.time ?? 0) - (points.first?.time ?? 0) }
}

struct TrackerSnapshot: Sendable, Equatable {
    enum Status: Sendable { case searching, tracking, coasting }
    var status: Status = .searching
    /// Recent path, oldest → newest, normalised 0…1 of the upright frame.
    var trail: [CGPoint] = []
    var imageSize: CGSize = .zero
    var time: Double = 0
    var lastHit: HitEvent?
    /// Approximate speed of the last completed flight; nil unless calibrated and trustworthy.
    var approxSpeedKmh: Double?

    static let empty = TrackerSnapshot()
}
