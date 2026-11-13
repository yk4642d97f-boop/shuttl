import CoreVideo
import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// Repository root (…/华北五省), derived from this file's location.
let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

final class VideoTests: XCTestCase {
    func realCase(_ i: Int = 0) throws -> (PlayerCase, URL) {
        let pc = try PlayerCase(Fx.list(try Fx.load("real"))[i])
        let url = repoRoot.appendingPathComponent(pc.raw["relative_path"] as? String ?? "")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test clip not present: \(url.path)")
        return (pc, url)
    }

    /// track_players end to end: AVFoundation decode every `step` frames, the
    /// recorded YOLO detections replayed per frame index, appearance from the decoded
    /// pixels (unused in singles), assemble. Must equal Python's observations.
    func testTrackPlayersStreamsVideoWithReplayedDetections() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        XCTAssertEqual(src.meta.nFrames, pc.meta.nFrames)
        let byFrame = Dictionary(uniqueKeysWithValues: zip(pc.frames, pc.detections))
        final class Seen: @unchecked Sendable { var idx: [Int] = [] }
        let seen = Seen()
        let det = ClosureDetector { frames, idx in
            XCTAssertTrue(frames.allSatisfy { CVPixelBufferGetWidth($0) == pc.meta.width && CVPixelBufferGetHeight($0) == pc.meta.height })
            seen.idx += idx
            return idx.map { byFrame[$0] ?? .empty }
        }
        var prog: [Double] = []
        let step = Int(Fx.num(pc.raw["step"]))
        let tr = try PlayerTracker.trackPlayers(source: src, camera: pc.camera, step: step, doubles: nil,
                                                detector: det, batch: 4, progress: { prog.append($0) })
        XCTAssertEqual(seen.idx, Array(stride(from: 0, to: pc.meta.nFrames, by: step)))
        XCTAssertEqual(prog.last ?? 0, 1.0, accuracy: 0.05)
        XCTAssertEqual(tr.observations.map { [$0.frame, $0.trackID] }, tr.observations.map { [$0.frame, $0.trackID] }.sorted { $0.lexicographicallyPrecedes($1) })
        let exp = Fx.observations(pc.raw["expect"])
        XCTAssertEqual(tr.observations.count, exp.count)
        var worst = 0.0
        for (o, e) in zip(tr.observations, exp) {
            XCTAssertEqual(o.frame, Int(Fx.num(e["frame"]))); XCTAssertEqual(o.trackID, Int(Fx.num(e["track_id"])))
            XCTAssertEqual(o.side, Fx.side(e["side"]))
            let f = Fx.nums(e["foot_world"])
            worst = max(worst, abs(o.footWorld!.x - f[0]), abs(o.footWorld!.y - f[1]))
        }
        XCTAssertLessThanOrEqual(worst, 1e-6)
        // movement_stats on Python's (cv2 POS_MSEC) timestamps: exact. AVFoundation's differ by
        // a few ulps, which can flip a heat-map cell whose dwell is exactly 6 × 1/30 s = 0.2 s.
        let dts = zip(src.meta.timestamps, pc.meta.timestamps).map { abs($0 - $1) }.max() ?? 0
        print(String(format: "[video] max |t_AVFoundation - t_cv2| = %.3g s, max |dfoot| = %.3g m", dts, worst))
        XCTAssertLessThan(dts, 1e-9)
        let st = Fx.dict(pc.raw["expect_stats"])
        for side in [CourtSide.near, .far] {
            compareStats(Movement.stats(tr.observations, side: side, timestamps: pc.meta.timestamps, fps: pc.meta.fps),
                         Fx.dict(st[side.rawValue]), side.rawValue)
        }
    }

    /// Port of test_real_pose_model_on_bundled_image, with Apple Vision on a Test clip frame.
    func testVisionDetectorOnRealFrame() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        var frame: CVPixelBuffer?
        for (_, pb) in try src.frames(stop: 1) { frame = pb }
        let pb = try XCTUnwrap(frame)
        let roi = PlayerTracker.defaultFarROI(camera: pc.camera, meta: src.meta, farCrop: true)
        let det = VisionPoseDetector(options: .init(farROI: roi))
        let d = try det.detect(frame: pb)
        XCTAssertGreaterThanOrEqual(d.n, 2, "both players expected")
        let kps = KeypointArray(d.keypoints)
        let (foot, ank) = PlayerGeometry.footPixels(boxes: d.boxes, kps: kps)
        XCTAssertGreaterThanOrEqual(ank.filter { $0 }.count, 2, "standing players: ankles found")
        for (b, f) in zip(d.boxes, foot) { XCTAssertGreaterThan(f.y, b.y1 + 0.7 * (b.y2 - b.y1)) }
        // the tracker accepts them: two on-court candidates, one per half
        let c = Candidates.build(frame: 0, detections: d, camera: pc.camera, imageSize: src.meta.size)
        let good = (0..<c.n).filter { c.unary[$0] < 2 }
        XCTAssertTrue(good.contains { c.pos[$0].x < 0 } && good.contains { c.pos[$0].x > 0 }, "\(c.pos) \(c.unary)")
    }
}
