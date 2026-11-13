import CoreVideo
import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// trackPlayers with its default Apple Vision detector (default far-crop layout) on a real
/// Test clip, against Python's YOLOv8s-pose tracks; and the cancellation hook.
final class TrackPlayersVisionTests: XCTestCase {
    func realCase() throws -> (PlayerCase, URL) {
        let pc = try PlayerCase(Fx.list(try Fx.load("real"))[0])
        let url = repoRoot.appendingPathComponent(pc.raw["relative_path"] as? String ?? "")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test clip not present: \(url.path)")
        return (pc, url)
    }

    func testDefaultVisionTracksFollowYOLOTracks() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        let tr = try PlayerTracker.trackPlayers(source: src, camera: pc.camera, step: Int(Fx.num(pc.raw["step"])))
        XCTAssertEqual(PlayerTracker.defaultFarROIs(camera: pc.camera, meta: src.meta).count,
                       PlayerTracker.defaultFarLayout == .single ? 1 : 2)
        let exp = Fx.observations(pc.raw["expect"])
        for (tid, side) in [(1, CourtSide.near), (2, .far)] {
            let ref = exp.filter { Int(Fx.num($0["track_id"])) == tid }
            let mine = Dictionary(grouping: tr.observations.filter { $0.side == side }, by: \.frame)
            var errs: [Double] = []
            for e in ref {
                guard let o = mine[Int(Fx.num(e["frame"]))]?.first, let fw = o.footWorld else { continue }
                let f = Fx.nums(e["foot_world"])
                errs.append(hypot(fw.x - f[0], fw.y - f[1]))
            }
            XCTAssertGreaterThan(Double(errs.count), 0.9 * Double(ref.count), "\(side) coverage")
            XCTAssertLessThan(NP.median(errs), 0.15, "\(side) median foot distance to the YOLO track")
            XCTAssertEqual(Set(tr.observations.filter { $0.side == side }.map(\.trackID)), [tid], "\(side) one id")
        }
    }

    func testTrackPlayersHonoursCancellation() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        var polls = 0, detected = 0
        let det = ClosureDetector { frames, _ in detected += frames.count; return frames.map { _ in .empty } }
        XCTAssertThrowsError(try PlayerTracker.trackPlayers(source: src, camera: pc.camera, step: 2, detector: det,
                                                            isCancelled: { polls += 1; return polls > 3 })) {
            XCTAssertTrue($0 is CancellationError, "\($0)")
        }
        XCTAssertEqual(detected, 3, "stops before the 4th sampled frame")
    }
}
