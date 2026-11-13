import XCTest
import CoreGraphics
import CoreMedia
import simd
@testable import Shuttl

final class TrajectoryTrackerTests: XCTestCase {

    private struct LCG {
        var state: UInt64
        mutating func uniform() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
        /// Approximately N(0, 1) (Irwin–Hall).
        mutating func normal() -> Double { (0..<12).reduce(0) { s, _ in s + uniform() } - 6 }
    }

    private let size = CGSize(width: 1920, height: 1080)

    /// Image-space lob: 900 px/s right, launched upward, 1500 px/s² "gravity" (down = +y).
    private func lob(_ t: Double) -> CGPoint {
        CGPoint(x: 300 + 900 * t, y: 800 - 900 * t + 750 * t * t)
    }

    func testFollowsParabolaThroughMissesAndIgnoresStaticDistractor() {
        let tracker = TrajectoryTracker()
        var rng = LCG(state: 3)
        let distractor = CGPoint(x: 1500, y: 300)
        let missing: Set<Int> = [10, 11, 12, 20, 27]
        var t = 0.0
        var truth: [(t: Double, p: CGPoint)] = []
        var snap = TrackerSnapshot.empty
        for i in 0..<30 {
            let p = lob(t)
            truth.append((t, p))
            var cands = [TrackerCandidate(pixel: CGPoint(x: distractor.x + rng.normal() * 0.5, y: distractor.y + rng.normal() * 0.5),
                                          score: 0.9)]
            if !missing.contains(i) {
                cands.append(TrackerCandidate(pixel: CGPoint(x: p.x + rng.normal(), y: p.y + rng.normal()), score: 0.7))
            }
            snap = tracker.update(time: t, candidates: cands, imageSize: size)
            t += 0.030 + 0.007 * rng.uniform()          // variable frame rate
        }
        XCTAssertEqual(snap.status, .tracking)
        let pts = tracker.activePoints
        XCTAssertGreaterThanOrEqual(pts.count, 30 - missing.count - 1, "track broke up: \(pts.count) points")
        for q in pts {
            guard let ref = truth.first(where: { abs($0.t - q.time) < 1e-9 }) else { return XCTFail("unknown time") }
            XCTAssertLessThan(hypot(Double(q.pixel.x - ref.p.x), Double(q.pixel.y - ref.p.y)), 6)
        }
        // Nothing from the distractor on screen, no fake hit on a smooth lob.
        let dn = CGPoint(x: distractor.x / size.width, y: distractor.y / size.height)
        XCTAssertFalse(snap.trail.isEmpty)
        XCTAssertTrue(snap.trail.allSatisfy { hypot(Double($0.x - dn.x), Double($0.y - dn.y)) > 0.02 })
        XCTAssertTrue(tracker.hits.isEmpty)
        XCTAssertNil(snap.approxSpeedKmh, "no calibration → no speed")
        // Trail head is the newest measurement, normalised by the frame size.
        if let head = snap.trail.last, let last = pts.last {
            XCTAssertEqual(Double(head.x), Double(last.pixel.x) / 1920, accuracy: 1e-9)
        }
    }

    func testStaticDistractorAloneNeverConfirms() {
        let tracker = TrajectoryTracker()
        var rng = LCG(state: 5)
        for i in 0..<90 {
            let c = TrackerCandidate(pixel: CGPoint(x: 640 + rng.normal(), y: 360 + rng.normal()), score: 0.95)
            let s = tracker.update(time: Double(i) / 30, candidates: [c], imageSize: size)
            XCTAssertEqual(s.status, .searching)
            XCTAssertTrue(s.trail.isEmpty)
        }
    }

    func testVelocityUsesTimestampsNotFrameCount() {
        // Same 600 px/s motion sampled at 60 fps, 30 fps and a jittery VFR clock (CMTime path).
        func run(_ times: [Double]) -> Double? {
            let tracker = TrajectoryTracker()
            for (i, t) in times.enumerated() {
                let det = ShuttleDetection(pixel: CGPoint(x: 200 + 600 * t, y: 500), position: .zero,
                                           confidence: 0.8, frameIndex: i,
                                           timestamp: CMTime(seconds: 12 + t, preferredTimescale: 90_000))
                tracker.update(FrameDetections(detections: [det], timestamp: det.timestamp, frameIndex: i, imageSize: size))
            }
            return tracker.active?.fx.velocity
        }
        var rng = LCG(state: 9)
        var vfr: [Double] = [0]
        while vfr.last! < 0.6 { vfr.append(vfr.last! + 0.022 + 0.03 * rng.uniform()) }
        for times in [stride(from: 0.0, to: 0.6, by: 1.0 / 60).map { $0 }, stride(from: 0.0, to: 0.6, by: 1.0 / 30).map { $0 }, vfr] {
            guard let v = run(times) else { XCTFail("no track"); continue }
            XCTAssertEqual(v, 600, accuracy: 15)
        }
    }

    func testDuplicateTimestampIsIgnoredAndLongGapRestarts() {
        let tracker = TrajectoryTracker()
        for i in 0..<10 {
            let t = Double(i) / 30
            tracker.update(time: t, candidates: [TrackerCandidate(pixel: CGPoint(x: 100 + 30 * Double(i), y: 400), score: 0.8)],
                           imageSize: size)
        }
        let n = tracker.activePoints.count
        tracker.update(time: 9.0 / 30, candidates: [TrackerCandidate(pixel: CGPoint(x: 900, y: 900), score: 0.9)], imageSize: size)
        XCTAssertEqual(tracker.activePoints.count, n)
        tracker.update(time: 5, candidates: [], imageSize: size)
        XCTAssertFalse(tracker.isTracking)
    }

    func testHitAtDirectionReversal() {
        let tracker = TrajectoryTracker()
        var rng = LCG(state: 21)
        let hitTime = 0.5
        let apex = CGPoint(x: 300 + 1500 * hitTime, y: 500)
        var t = 0.0
        while t < 1.0 {
            let p = t <= hitTime
                ? CGPoint(x: 300 + 1500 * t, y: 500)
                : CGPoint(x: apex.x - 1500 * (t - hitTime), y: apex.y + 300 * (t - hitTime))
            tracker.update(time: t, candidates: [TrackerCandidate(pixel: CGPoint(x: p.x + rng.normal(), y: p.y + rng.normal()),
                                                                  score: 0.8)], imageSize: size)
            t += 1.0 / 30
        }
        XCTAssertEqual(tracker.hits.count, 1, "hits: \(tracker.hits)")
        if let h = tracker.hits.first {
            XCTAssertEqual(h.time, hitTime, accuracy: 0.05)
            XCTAssertLessThan(hypot(Double(h.pixel.x - apex.x), Double(h.pixel.y - apex.y)), 60)
        }
        XCTAssertEqual(tracker.completedFlights.first?.end, .hit)
        XCTAssertTrue(tracker.isTracking, "the return flight continues the same track")
    }

    // MARK: Approximate flight speed (calibrated, monocular)

    func testFlightSpeedFromCalibratedCamera() throws {
        let cal = CourtCalibration.lookAt(eye: SIMD3(-10.5, 0, 3.0), target: SIMD3(0, 0, 1.0), focalPx: 1600,
                                          imageSize: .init(width: 1920, height: 1080))
        let v0 = SIMD3(17.0, -1.5, 7.0)
        let times = stride(from: 0.0, through: 0.9, by: 1.0 / 30).map { $0 }
        let world = FlightSpeedEstimator.simulate(p0: SIMD3(-5.2, 0.8, 2.4), v0: v0, times: times)
        var rng = LCG(state: 42)
        var pts: [TrackPoint] = []
        for (t, w) in zip(times, world) {
            let px = try XCTUnwrap(cal.project(w))
            pts.append(TrackPoint(time: 100 + t, pixel: CGPoint(x: px.x + 0.7 * rng.normal(), y: px.y + 0.7 * rng.normal()),
                                  score: 0.9))
        }
        let kmh = try XCTUnwrap(FlightSpeedEstimator.estimateKmh(points: pts, calibration: cal, imageSize: size))
        let truth = simd_length(v0) * 3.6
        XCTAssertEqual(kmh, truth, accuracy: 0.1 * truth)

        // Pure noise must not produce a number.
        let junk = (0..<20).map { i in
            TrackPoint(time: Double(i) / 30, pixel: CGPoint(x: 960 + 400 * rng.normal(), y: 540 + 300 * rng.normal()), score: 0.9)
        }
        XCTAssertNil(FlightSpeedEstimator.estimateKmh(points: junk, calibration: cal, imageSize: size))
    }

    func testOverlayAspectFillMapping() {
        // 1080×1920 portrait frame shown aspect-fill on a 390×844 screen: centre maps to centre.
        let pts = RecordingViewModel.aspectFill([CGPoint(x: 0.5, y: 0.5), CGPoint(x: 0, y: 0)],
                                                imageSize: CGSize(width: 1080, height: 1920),
                                                into: CGSize(width: 390, height: 844))
        XCTAssertEqual(Double(pts[0].x), 195, accuracy: 1e-9)
        XCTAssertEqual(Double(pts[0].y), 422, accuracy: 1e-9)
        let s = max(390.0 / 1080, 844.0 / 1920)
        XCTAssertEqual(Double(pts[1].x), (390 - 1080 * s) / 2, accuracy: 1e-9)
    }

    func testOverlayMirroredForFrontCamera() {
        // Front camera: preview mirrored, ML frames not → x flips about the frame centre, y unchanged.
        let image = CGSize(width: 1080, height: 1920), view = CGSize(width: 390, height: 844)
        let input = [CGPoint(x: 0.2, y: 0.3), CGPoint(x: 0.5, y: 0.9)]
        let plain = RecordingViewModel.aspectFill(input, imageSize: image, into: view)
        let mirrored = RecordingViewModel.aspectFill(input, imageSize: image, into: view, mirrored: true)
        for (p, m) in zip(plain, mirrored) {
            XCTAssertEqual(Double(m.x), Double(view.width - p.x), accuracy: 1e-9)
            XCTAssertEqual(Double(m.y), Double(p.y), accuracy: 1e-9)
        }
        XCTAssertEqual(Double(mirrored[1].x), 195, accuracy: 1e-9)
    }
}
