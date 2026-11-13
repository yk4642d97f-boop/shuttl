import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

final class CameraSteadinessTests: XCTestCase {
    /// A court-like scene: bright lines on a textured floor, with a "player" blob that moves.
    static func scene(shiftX: Int = 0, shiftY: Int = 0, player: Int = 0, width: Int = 640, height: Int = 360) -> ImageU8 {
        var img = ImageU8(width: width, height: height, channels: 3)
        for y in 0..<height {
            for x in 0..<width {
                let sx = x - shiftX, sy = y - shiftY
                var v = 60 + ((sx &* 7 &+ sy &* 13) & 15)                     // floor texture
                if sx % 80 < 3 || sy % 60 < 3 { v = 230 }                       // court lines
                let px = 100 + player * 17, py = 150
                if abs(x - px) < 18 && abs(y - py) < 40 { v = 20 }              // a moving player
                for c in 0..<3 { img.data[(y * width + x) * 3 + c] = UInt8(clamping: v) }
            }
        }
        return img
    }

    func testTripodWithMovingPlayersIsSteady() throws {
        let frames = (0..<12).map { Self.scene(player: $0) }
        let s = try XCTUnwrap(CameraSteadinessEstimator.estimate(frames))
        XCTAssertEqual(s.verdict(), .steady)
        XCTAssertEqual(s.shiftsPx.max(), 0)
        XCTAssertGreaterThan(s.correlations.min() ?? 0, 0.8)
    }

    func testHandHeldShakeIsCaught() throws {
        // A random walk of 6–40 px at 640 px wide, like a hand-held phone.
        var rng = SystemRandomNumberGenerator()
        let frames = (0..<12).map { i -> ImageU8 in
            i == 0 ? Self.scene() : Self.scene(shiftX: Int.random(in: 6...40, using: &rng) * (Bool.random() ? 1 : -1),
                                               shiftY: Int.random(in: 3...25, using: &rng), player: i)
        }
        let s = try XCTUnwrap(CameraSteadinessEstimator.estimate(frames))
        XCTAssertEqual(s.verdict(), .handHeld, "fraction \(s.unsteadyFraction()) shifts \(s.shiftsPx)")
    }

    func testOneBumpPartWayThroughIsAWarningNotARejection() throws {
        // 10 frames in place, 2 after the tripod was knocked 20 px.
        let frames = (0..<12).map { Self.scene(shiftX: $0 >= 10 ? 20 : 0, player: $0) }
        let s = try XCTUnwrap(CameraSteadinessEstimator.estimate(frames))
        XCTAssertEqual(s.verdict(), .moved, "fraction \(s.unsteadyFraction())")
        var a = MatchAnalysis(meta: VideoMeta(fps: 30, width: 640, height: 360, nFrames: 1, timestamps: [0]))
        a.diagnostics[CameraSteadiness.diagnosticsKey] = s.json
        XCTAssertNotNil(a.steadinessWarning)
        XCTAssertNotEqual(a.outcome, .cameraUnsteady)
    }

    func testHandHeldOutranksRalliesInTheOutcome() throws {
        let frames = (0..<12).map { i in Self.scene(shiftX: i.isMultiple(of: 2) ? 0 : 30, shiftY: i % 3 * 12, player: i) }
        let s = try XCTUnwrap(CameraSteadinessEstimator.estimate(frames))
        var a = MatchAnalysis(meta: VideoMeta(fps: 30, width: 640, height: 360, nFrames: 1, timestamps: [0]),
                              rallies: [RallyV2(startFrame: 0, endFrame: 10, shots: [], endReason: .inCourt, winner: .near)])
        a.diagnostics[CameraSteadiness.diagnosticsKey] = s.json
        XCTAssertEqual(a.outcome, .cameraUnsteady, "invented rallies from a shaking phone must not be shown")
        XCTAssertTrue(a.outcome.userMessage.contains("三脚架"))
        XCTAssertFalse(a.outcome.isRetryable)
    }

    func testTooFewFramesSaysNothing() {
        XCTAssertNil(CameraSteadinessEstimator.estimate([Self.scene(), Self.scene()]))
        var a = MatchAnalysis(meta: VideoMeta(fps: 30, width: 640, height: 360, nFrames: 1, timestamps: [0]))
        a.diagnostics["camera_motion"] = .object(["shifts_px": .array([]), "correlations": .array([]),
                                                  "image_width": .number(640)])
        XCTAssertNil(a.steadinessWarning)
    }

    func testJSONRoundTripsThroughTheAnalysis() throws {
        let s = CameraSteadiness(shiftsPx: [0, 0, 12.3], correlations: [0.97, 0.96, 0.5], imageWidth: 1920, searchPx: 48)
        var a = MatchAnalysis(meta: VideoMeta(fps: 30, width: 1920, height: 1080, nFrames: 1, timestamps: [0]))
        a.diagnostics[CameraSteadiness.diagnosticsKey] = s.json
        let back = try XCTUnwrap(a.cameraSteadiness)
        XCTAssertEqual(back.shiftsPx, [0, 0, 12.3])
        XCTAssertEqual(back.imageWidth, 1920)
        XCTAssertEqual(back.verdict(), s.verdict())
    }
}
