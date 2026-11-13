import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlLive

/// A track whose per-frame arrays differ in length used to trap in ShuttlHits
/// (Observations.swift preferRaw, xy[i] = raw[i]) — on resume at every app launch.
final class TrackConsistencyTests: XCTestCase {
    private func track(_ n: Int) -> Track2D {
        Track2D(t: (0..<n).map { Double($0) / 60 }, xy: (0..<n).map { SIMD2(Double($0), 500) },
                conf: Array(repeating: 0.9, count: n), visible: Array(repeating: true, count: n),
                interpolated: Array(repeating: false, count: n), rawXY: (0..<n).map { SIMD2(Double($0), 501) })
    }

    func testConsistentTrackPasses() throws {
        XCTAssertNil(track(133).inconsistency)
        XCTAssertNoThrow(try LiveStages.requireConsistent(track(133)))
        XCTAssertNil(Track2D(t: [], xy: [], conf: [], visible: [], interpolated: []).inconsistency)
    }

    func testEveryShortArrayIsCaught() {
        let edits: [(String, (inout Track2D) -> Void)] = [
            ("xy", { $0.xy.removeLast() }), ("conf", { $0.conf.removeLast() }),
            ("visible", { $0.visible.removeLast() }), ("interpolated", { $0.interpolated.removeLast() }),
            ("raw_xy", { $0.rawXY!.removeLast() }), ("t", { $0.t.removeLast() }),
            ("xy longer", { $0.xy.append(SIMD2(1, 1)) }),
        ]
        for (name, edit) in edits {
            var tr = track(133)
            edit(&tr)
            XCTAssertNotNil(tr.inconsistency, name)
            XCTAssertThrowsError(try LiveStages.requireConsistent(tr), name)
        }
    }

    /// E's repro (tools/robustness/repro/track-len-mismatch): xy 132, everything else 133.
    func testDecodingAMismatchedTrackThrowsInsteadOfTrapping() throws {
        var tr = track(133)
        let good = try ShuttlJSON.encoder().encode(tr)
        XCTAssertEqual(try ShuttlJSON.decoder().decode(Track2D.self, from: good).n, 133)
        tr.xy.removeLast()
        let bad = try ShuttlJSON.encoder().encode(tr)
        XCTAssertThrowsError(try ShuttlJSON.decoder().decode(Track2D.self, from: bad)) { e in
            XCTAssertTrue(e is DecodingError, "\(e)")
        }
        var noRaw = track(10); noRaw.rawXY = nil
        XCTAssertNoThrow(try ShuttlJSON.decoder().decode(Track2D.self, from: ShuttlJSON.encoder().encode(noRaw)))
    }

    /// Through the checkpoint store: the damaged track is a cache miss, the stage re-runs.
    func testDamagedTrackCheckpointIsACacheMiss() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tlm-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        var tr = track(40)
        tr.conf.removeLast()
        store.save(tr, key: "track")
        XCTAssertNil(store.load(Track2D.self, key: "track"))
        store.save(track(40), key: "track")
        XCTAssertEqual(store.load(Track2D.self, key: "track")?.n, 40)
    }

    func testShotSegmentFramesAndPositionsMustMatch() throws {
        let seg = ShotSegment3D(startFrame: 0, endFrame: 2, t0: 0, p0: [0, 0, 1], v0: [10, 0, 5], dragLength: 3,
                                rmsePx: 1, frames: [0, 1, 2], positions: [[0, 0, 1], [1, 0, 1], [2, 0, 1]],
                                landingXY: nil, landingT: nil, netCrossingHeight: nil, confidence: 0.5,
                                netCrossingY: nil, endReason: .unknown)
        let data = try ShuttlJSON.encoder().encode(seg)
        XCTAssertEqual(try ShuttlJSON.decoder().decode(ShotSegment3D.self, from: data).frames.count, 3)
        var o = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        o["frames"] = [0, 1]
        let bad = try JSONSerialization.data(withJSONObject: o)
        XCTAssertThrowsError(try ShuttlJSON.decoder().decode(ShotSegment3D.self, from: bad))
    }
}
