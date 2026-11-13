import Foundation
import CoreGraphics
import CoreMedia

// MARK: - TrajectoryTracker
// Live, light port of backend linker.py. Per frame (in full-res pixels, real dt):
//   1. static-distractor cost from a spatial occupancy grid,
//   2. gated association of ≤ 2 candidates to the active track (constant-acceleration
//      Kalman per axis, Mahalanobis gate + hard speed gate, cost = d² + static − logit(score)),
//   3. tentative tracks from leftovers; a track is confirmed after ≥ 3 consistent frames that
//      actually move (a static distractor never confirms),
//   4. coasting over misses ≤ maxCoast, then termination; a confirmed tentative close to where a
//      coasting track could be is a re-acquisition after a hit,
//   5. hits = kinks in the image path (Δv not explained by drag/noise/gravity) → flight segments.
// Not thread-safe: own it from one actor / task.

final class TrajectoryTracker {
    let config: TrackerConfig
    /// When set, completed flights get a monocular 3-D speed estimate (approximate).
    var calibration: CourtCalibration?

    struct Track {
        var fx: AxisFilter
        var fy: AxisFilter
        var time: Double            // filter time
        var lastHit: Double         // last measurement time
        var points: [TrackPoint]
        var staticSum = 0.0
        var firstPixel: CGPoint
        var maxTravel = 0.0
    }

    // State
    private(set) var imageSize: CGSize = .zero
    private(set) var diag = 1.0
    private(set) var currentTime: Double?
    private(set) var frameInterval: Double
    var active: Track?
    var tentatives: [Track] = []
    var staticMap: StaticMap?
    var segmentStart = 0              // index into active.points
    var kinkCursor = 0
    var pendingKink: (index: Int, ratio: Double, turn: Double?)?
    // Mutated by the +Step / +Flights extensions (other files), hence internal setters.
    var recentTrail: [TrackPoint] = []   // points of ended tracks, for fading
    var hits: [HitEvent] = []
    var completedFlights: [FlightSegment] = []
    var lastSpeed: (kmh: Double, time: Double)?

    init(config: TrackerConfig = TrackerConfig()) {
        self.config = config
        frameInterval = config.nominalFrameInterval
    }

    // MARK: Public API

    /// Feeds one processed frame (detections may be empty). Uses the frame's presentation
    /// timestamp; frames without one advance by the running mean frame interval.
    @discardableResult
    func update(_ frame: FrameDetections) -> TrackerSnapshot {
        update(time: frame.seconds,
               candidates: frame.detections.map { TrackerCandidate(pixel: $0.pixel, score: $0.confidence) },
               imageSize: frame.imageSize)
    }

    @discardableResult
    func update(time: Double?, candidates: [TrackerCandidate], imageSize size: CGSize) -> TrackerSnapshot {
        guard size.width > 0, size.height > 0 else { return snapshot() }
        if size != imageSize { reset(); configure(for: size) }
        var t = time ?? ((currentTime ?? 0) + frameInterval)
        if let prev = currentTime {
            let dt = t - prev
            if time != nil, dt <= 1e-6 {
                if dt > -1e-6 { return snapshot() }      // duplicate frame
                resetTracks(); t = time!                  // time went backwards (new clip)
            } else if dt > config.maxFrameGap {
                resetTracks()
            } else {
                frameInterval += 0.1 * (min(dt, 0.2) - frameInterval)
            }
        }
        currentTime = t
        let cands = candidates
            .filter { $0.score >= config.minScore && $0.pixel.x.isFinite && $0.pixel.y.isFinite }
            .sorted { $0.score > $1.score }
            .prefix(config.maxCandidates)
        step(time: t, candidates: Array(cands))
        return snapshot()
    }

    func reset() {
        resetTracks()
        staticMap = nil
        imageSize = .zero
        currentTime = nil
        hits.removeAll()
        completedFlights.removeAll()
        lastSpeed = nil
        frameInterval = config.nominalFrameInterval
    }

    /// Observed points of the active track (oldest → newest).
    var activePoints: [TrackPoint] { active?.points ?? [] }
    var isTracking: Bool { active != nil }

    // MARK: Setup

    private func configure(for size: CGSize) {
        imageSize = size
        diag = max(hypot(Double(size.width), Double(size.height)), 1)
        staticMap = StaticMap(imageSize: size, cellSize: config.staticCell * diag)
    }

    func resetTracks() {
        if active != nil { endActive(.reset) }
        active = nil
        tentatives.removeAll()
        recentTrail.removeAll()
        pendingKink = nil
    }

    // Pixel-unit parameters
    var sigmaPx: Double { config.sigmaMeasurement * diag }
    var measurementVar: Double { sigmaPx * sigmaPx }
    var maxSpeedPx: Double { config.maxSpeed * diag }

    func jerkDensity(speedPx: Double) -> Double {
        let lPx = config.dragLength * diag
        let j = config.jerkFloor * diag + 2 * pow(min(speedPx, maxSpeedPx), 3) / (lPx * lPx)
        return j * j * config.jerkCorrelation
    }

    func newTrack(at c: TrackerCandidate, time t: Double, staticCost: Double) -> Track {
        let pv = measurementVar
        let vv = pow(config.initialSpeedSigma * diag, 2)
        let av = pow(4 * config.initialSpeedSigma * diag, 2)
        return Track(fx: AxisFilter(position: Double(c.pixel.x), positionVar: pv, velocityVar: vv, accelVar: av),
                     fy: AxisFilter(position: Double(c.pixel.y), positionVar: pv, velocityVar: vv, accelVar: av),
                     time: t, lastHit: t, points: [TrackPoint(time: t, pixel: c.pixel, score: c.score)],
                     staticSum: staticCost, firstPixel: c.pixel)
    }
}
