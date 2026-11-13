import XCTest
import ShuttlCore
@testable import ShuttlHitModel

final class PlayerOrderTests: XCTestCase {
    // The training extractor's contract is far = slot 0, near = slot 1
    // (research/extract/Sources/extract/RallyExtractor.swift). Start with semantic
    // observations, not an inverse of rawMatrix that could repeat its mistake.
    private let meta = VideoMeta(fps: 30, width: 1000, height: 600, nFrames: 1, timestamps: [0])
    private let track = Track2D(t: [0], xy: [SIMD2(500.0, 300.0)], conf: [1],
                                visible: [true], interpolated: [false])

    func testPlayerSlotsMatchTrainingExtractor() {
        let near = PlayerObservation(frame: 0, trackID: 8,
                                     bbox: BBox(x1: 600, y1: 300, x2: 900, y2: 500),
                                     score: 1, side: .near, footWorld: SIMD2(-4, -1))
        let far = PlayerObservation(frame: 0, trackID: 3,
                                    bbox: BBox(x1: 200, y1: 100, x2: 300, y2: 300),
                                    score: 1, side: .far, footWorld: SIMD2(4, 1))
        let raw = HitStage.rawMatrix(track: track, meta: meta, players: [near, far])
        XCTAssertEqual(raw[4], 0.25, accuracy: 1e-9, "first box belongs to far player")
        XCTAssertEqual(raw[8], 0.75, accuracy: 1e-9, "second box belongs to near player")
        XCTAssertEqual(Array(raw[12..<16]), [4, 1, -4, -1])
    }

    func testDuplicateFrameUsesLowestTrackIDLikeExtractor() {
        let distractor = PlayerObservation(frame: 0, trackID: 20,
                                          bbox: BBox(x1: 600, y1: 100, x2: 800, y2: 300),
                                          score: 1, side: .far)
        let selected = PlayerObservation(frame: 0, trackID: 2,
                                        bbox: BBox(x1: 200, y1: 100, x2: 300, y2: 300),
                                        score: 1, side: .far)
        let raw = HitStage.rawMatrix(track: track, meta: meta, players: [distractor, selected])
        XCTAssertEqual(raw[4], 0.25, accuracy: 1e-9)
    }
}
