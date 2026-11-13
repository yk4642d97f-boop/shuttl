import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

final class PlayersCodecTests: XCTestCase {
    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-codec-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    private func tempFile(_ name: String = "players.bin") -> URL { tempDir().appendingPathComponent(name) }

    /// Real ShuttlPlayers tracker output (copied from ShuttlPlayers' real.json fixture).
    private func realObservations() throws -> [PlayerObservation] {
        struct F: Decodable { let observations: [PlayerObservation] }
        let url = repoTestsDirectory.appendingPathComponent("Fixtures/players_real_sample.json")
        return try ShuttlJSON.load(F.self, from: url).observations
    }

    func testRealFixtureRoundTripsBitExactly() throws {
        let real = try realObservations()
        XCTAssertEqual(real.count, 300)
        XCTAssertGreaterThan(real.filter { $0.keypoints != nil }.count, 50)
        let url = tempFile()
        try PlayersCodec.write(real, to: url, fingerprint: "abc")
        let back = try XCTUnwrap(PlayersCodec.read(from: url, expectedFingerprint: "abc"))
        assertBitIdentical(back, real)
        XCTAssertEqual(back, real)
        // Same observations the JSON checkpoint path would give back.
        let viaJSON = try ShuttlJSON.decoder().decode([PlayerObservation].self, from: ShuttlJSON.encoder().encode(real))
        XCTAssertEqual(back, viaJSON)
    }

    func testEdgeCasesRoundTrip() throws {
        var obs = ScriptedStages.samplePlayers(count: 97)
        obs.append(PlayerObservation(frame: -1, trackID: Int(Int32.max) + 5, bbox: BBox(x1: -0.0, y1: .infinity, x2: -.infinity, y2: .nan),
                                     score: .nan, side: .unknown, footWorld: SIMD2(.nan, .nan), keypoints: []))
        obs.append(PlayerObservation(frame: 1 << 40, trackID: -3, bbox: BBox(x1: 1, y1: 2, x2: 3, y2: 4), score: 1,
                                     keypoints: (0..<5).map { Keypoint(x: Double($0), y: .nan, conf: 0) }))
        let url = tempFile()
        // Tiny chunks exercise buffer refills across record boundaries.
        try PlayersCodec.write(obs, to: url, fingerprint: "é-指纹", chunkBytes: 7)
        let back = try XCTUnwrap(PlayersCodec.read(from: url, chunkBytes: 5))
        assertBitIdentical(back, obs)
        XCTAssertEqual(PlayersCodec.fingerprint(at: url), "é-指纹")
        XCTAssertEqual(back[97].keypoints?.count, 0, "empty ≠ nil")
        XCTAssertNil(back[5].keypoints)
        XCTAssertNil(back[0].footWorld)
        XCTAssertEqual(back[98].frame, 1 << 40)
    }

    func testEmptyAndOverwrite() throws {
        let url = tempFile()
        try PlayersCodec.write([], to: url)
        XCTAssertEqual(try PlayersCodec.read(from: url), [])
        let obs = ScriptedStages.samplePlayers(count: 3)
        try PlayersCodec.write(obs, to: url)                    // replaces the existing file
        XCTAssertEqual(try PlayersCodec.read(from: url)?.count, 3)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path)
        XCTAssertEqual(leftovers, ["players.bin"], "no temp files left behind")
    }

    func testWrongFingerprintTruncationAndGarbage() throws {
        let url = tempFile()
        try PlayersCodec.write(ScriptedStages.samplePlayers(count: 20), to: url, fingerprint: "one")
        XCTAssertNil(try PlayersCodec.read(from: url, expectedFingerprint: "two"))
        let data = try Data(contentsOf: url)
        for cut in [data.count - 1, data.count - 12, data.count / 2, 10] {
            try data.prefix(cut).write(to: url)
            XCTAssertThrowsError(try PlayersCodec.read(from: url), "cut at \(cut)")
        }
        try Data("{\"fingerprint\":\"one\"}".utf8).write(to: url)
        XCTAssertThrowsError(try PlayersCodec.read(from: url)) { XCTAssertTrue("\($0)".contains("not a Shuttl players file")) }
    }

    func testSlimFormDropsOnlyKeypoints() throws {
        let obs = ScriptedStages.samplePlayers(count: 30)
        let url = tempFile()
        try PlayersCodec.write(obs, to: url, includeKeypoints: false)
        let back = try XCTUnwrap(PlayersCodec.read(from: url))
        var expect = obs
        for i in expect.indices { expect[i].keypoints = nil }
        assertBitIdentical(back, expect)
    }

    func testCheckpointStoreUsesBinaryForPlayersAndJSONForTheRest() throws {
        let dir = tempDir()
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp1")
        let obs = try realObservations()
        store.save(obs, key: "players")
        store.save([[ShuttleCandidate(x: 1, y: 2, score: 0.5)]], key: "candidates")
        store.save([PlayerObservation](), key: "empty_players")
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("players.bin").path))
        XCTAssertFalse(fm.fileExists(atPath: dir.appendingPathComponent("players.json").path))
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("candidates.json").path))
        XCTAssertEqual(store.load([PlayerObservation].self, key: "players"), obs)
        XCTAssertEqual(store.load([PlayerObservation].self, key: "empty_players"), [])
        XCTAssertEqual(store.load([[ShuttleCandidate]].self, key: "candidates")?.first?.first?.score, 0.5)
        XCTAssertNil(FileCheckpointStore(directory: dir, fingerprint: "fp2").load([PlayerObservation].self, key: "players"))

        // A JSON checkpoint without a digest (older builds) can't be verified: a miss, recomputed.
        let legacyDir = dir.appendingPathComponent("legacy")
        let legacy = FileCheckpointStore(directory: legacyDir, fingerprint: "fp1")
        try fm.createDirectory(at: legacyDir, withIntermediateDirectories: true)
        struct Envelope: Encodable { let fingerprint: String; let value: [PlayerObservation] }
        try ShuttlJSON.encoder().encode(Envelope(fingerprint: "fp1", value: Array(obs.prefix(5))))
            .write(to: legacyDir.appendingPathComponent("players.json"))
        XCTAssertNil(legacy.load([PlayerObservation].self, key: "players"))
        // Garbage in the binary file = no checkpoint, never a crash.
        try Data([1, 2, 3]).write(to: dir.appendingPathComponent("players.bin"))
        XCTAssertNil(store.load([PlayerObservation].self, key: "players"))
    }
}

/// Peak-memory checks for long matches. The 45-min ones are gated (`SHUTTL_BIG=1`) and
/// print their numbers; the small ones always run.
final class CheckpointMemoryTests: XCTestCase {
    private var big: Bool { ProcessInfo.processInfo.environment["SHUTTL_BIG"] == "1" }

    private func dir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-mem-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    /// 45-min singles at 60 fps, players every 4th frame, 2 players → 324k observations.
    func testPlayersCheckpoint324kUnder400MB() throws {
        let n = big ? 324_000 : 20_000
        let store = FileCheckpointStore(directory: dir(), fingerprint: "mem")
        var obs: [PlayerObservation]? = ScriptedStages.samplePlayers(count: n)
        var loadedCount = 0
        let save = FootprintProbe.measure { store.save(obs!, key: "players") }
        // Measured while the source array is still alive, so freed pages can't be reused:
        // an upper bound (decoded copy + transient).
        let load = FootprintProbe.measure {
            loadedCount = store.load([PlayerObservation].self, key: "players")?.count ?? -1
        }
        obs = nil
        XCTAssertEqual(loadedCount, n)
        let size = (try? FileManager.default.attributesOfItem(atPath: store.directory.appendingPathComponent("players.bin").path)[.size] as? Int) ?? 0
        print(String(format: "[mem] players checkpoint n=%d  save peak +%.1f MB  load peak +%.1f MB  file %.1f MB  (baseline %.0f MB)",
                     n, save.growth, load.growth, Double(size) / 1_048_576, save.baseline))
        XCTAssertLessThan(save.growth, 400)
        XCTAssertLessThan(load.growth, 400)
        if !big { XCTAssertLessThan(save.growth, 40, "streaming save must not scale with n") }
    }

    /// 90-min singles (E's base-90m: 613,828 observations): what the array itself costs vs. what
    /// saving and loading the checkpoint add on top. Every observation has 17 keypoints here.
    func testPlayersCheckpoint90MinBreakdown() throws {
        try XCTSkipUnless(big, "set SHUTTL_BIG=1")
        let n = 613_828
        let store = FileCheckpointStore(directory: dir(), fingerprint: "mem90")
        var obs: [PlayerObservation]?
        let build = FootprintProbe.measure {
            obs = (0..<n).map { i in
                var o = ScriptedStages.samplePlayer(i, keypoints: true)
                if o.keypoints == nil { o.keypoints = ScriptedStages.samplePlayer(0, keypoints: true).keypoints }
                return o
            }
        }
        let save = FootprintProbe.measure { store.save(obs!, key: "players") }
        var loaded: [PlayerObservation]?
        let load = FootprintProbe.measure { loaded = store.load([PlayerObservation].self, key: "players") }
        XCTAssertEqual(loaded?.count, n)
        let size = (try? FileManager.default.attributesOfItem(atPath: store.directory.appendingPathComponent("players.bin").path)[.size] as? Int) ?? 0
        print(String(format: "[mem90] n=%d  array build +%.1f MB (%.0f B/obs)  save +%.1f MB  load +%.1f MB  file %.1f MB",
                     n, build.growth, build.growth * 1_048_576 / Double(n), save.growth, load.growth, Double(size) / 1_048_576))
        XCTAssertLessThan(save.growth, 40, "streaming save must not scale with n")
        // Loading costs the decoded array, not array + file (autoreleased read chunks: +572 MB before).
        XCTAssertLessThan(load.growth, build.growth * 1.25 + 20)
        obs = nil; loaded = nil
    }

    /// What the old JSON checkpoint cost, for the log (per-observation cost × 324k).
    func testLegacyJSONCostForComparison() throws {
        try XCTSkipUnless(big, "set SHUTTL_BIG=1")
        let n = 20_000
        let obs = ScriptedStages.samplePlayers(count: n)
        var bytes = 0
        let m = try FootprintProbe.measure { bytes = try ShuttlJSON.encoder().encode(obs).count }
        print(String(format: "[mem] legacy JSON encode n=%d  peak +%.1f MB  (%.1f KB/obs → ×324k ≈ %.2f GB), %d bytes",
                     n, m.growth, m.growth * 1024 / Double(n), m.growth * 324_000 / Double(n) / 1024, bytes))
    }

    /// The other checkpoints of a 45-min 60 fps match: candidates + track (162k frames) as JSON.
    func testCandidatesAndTrackCheckpointsForALongMatch() throws {
        let frames = big ? 162_000 : 10_000
        let store = FileCheckpointStore(directory: dir(), fingerprint: "mem")
        var cands: [[ShuttleCandidate]]? = (0..<frames).map { f in
            (0..<(f % 3)).map { ShuttleCandidate(x: Double(f % 1920) + 0.123456, y: Double(f % 1080) * 0.987, score: 0.5 + Double($0) / 7) }
        }
        let t = (0..<frames).map { Double($0) / 60 }
        var track: Track2D? = Track2D(t: t, xy: t.map { SIMD2($0 * 13.3, $0.truncatingRemainder(dividingBy: 7) * 99) },
                                      conf: t.map { _ in 0.87654321 }, visible: t.map { Int($0 * 60) % 5 != 0 },
                                      interpolated: t.map { Int($0 * 60) % 11 == 0 }, rawXY: t.map { SIMD2($0, $0 * 2) })
        let saveC = FootprintProbe.measure { store.save(cands!, key: "candidates") }
        let saveT = FootprintProbe.measure { store.save(track!, key: "track") }
        var ok = false
        let load = FootprintProbe.measure {
            ok = store.load([[ShuttleCandidate]].self, key: "candidates")?.count == frames
                && store.load(Track2D.self, key: "track")?.n == frames
        }
        cands = nil; track = nil
        XCTAssertTrue(ok)
        print(String(format: "[mem] frames=%d  candidates save +%.1f MB  track save +%.1f MB  load both +%.1f MB",
                     frames, saveC.growth, saveT.growth, load.growth))
        XCTAssertLessThan(max(saveC.growth, saveT.growth, load.growth), 400)
    }
}
