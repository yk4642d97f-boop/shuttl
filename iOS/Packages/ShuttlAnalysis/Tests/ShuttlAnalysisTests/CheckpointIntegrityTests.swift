import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// A checkpoint that still decodes but whose content changed must be a cache miss
/// (robustness T10 finding #2: bit flips / valid-JSON edits were silently reused).
final class CheckpointIntegrityTests: XCTestCase {
    private func store(_ fp: String = "fp") -> FileCheckpointStore {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-integrity-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return FileCheckpointStore(directory: d, fingerprint: fp)
    }

    private func sampleTrack(_ n: Int = 40) -> Track2D {
        let t = (0..<n).map { Double($0) / 60 }
        return Track2D(t: t, xy: t.map { SIMD2($0 * 100 + 0.25, $0 * 50) }, conf: t.map { _ in 0.75 },
                       visible: t.map { Int($0 * 60) % 4 != 0 }, interpolated: t.map { _ in false },
                       rawXY: t.map { SIMD2($0, $0) })
    }

    private func file(_ s: FileCheckpointStore, _ name: String) -> URL { s.directory.appendingPathComponent(name) }

    /// Replaces the first occurrence of `from` with `to` (same length keeps the JSON valid).
    private func edit(_ url: URL, _ from: String, _ to: String) throws {
        var text = try String(contentsOf: url, encoding: .utf8)
        let r = try XCTUnwrap(text.range(of: from), "\(from) not in file")
        text.replaceSubrange(r, with: to)
        try Data(text.utf8).write(to: url)
    }

    func testRoundTripHitsForEveryCheckpointShape() throws {
        let s = store()
        let track = sampleTrack()
        let cands: [[ShuttleCandidate]] = [[], [ShuttleCandidate(x: 1.5, y: 1e-300, score: 0.1 + 0.2)], [ShuttleCandidate(x: 3, y: 4, score: 1)]]
        s.save(track, key: "track")
        s.save(cands, key: "candidates")
        s.save("é 指纹 \"quoted\"", key: "text")
        XCTAssertEqual(s.load(Track2D.self, key: "track")?.n, track.n)
        XCTAssertEqual(s.load(Track2D.self, key: "track")?.xy, track.xy)
        let back = try XCTUnwrap(s.load([[ShuttleCandidate]].self, key: "candidates"))
        XCTAssertEqual(back.count, 3)
        XCTAssertEqual(back[1][0], cands[1][0], "full precision survives")
        XCTAssertEqual(s.load(String.self, key: "text"), "é 指纹 \"quoted\"")
        // Still valid JSON with the documented fields.
        let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: file(s, "track.json"))) as? [String: Any]
        XCTAssertEqual(obj?["fingerprint"] as? String, "fp")
        XCTAssertEqual((obj?["sha256"] as? String)?.count, 64)
        XCTAssertNotNil(obj?["value"])
    }

    func testValidJSONValueEditIsAMiss() throws {
        let s = store()
        s.save(sampleTrack(), key: "track")
        let url = file(s, "track.json")
        let original = try Data(contentsOf: url)
        // Same-length edits keep the JSON decodable: exactly the case that used to be reused.
        try edit(url, "0.75", "0.95")
        XCTAssertNotNil(try? JSONSerialization.jsonObject(with: Data(contentsOf: url)))
        XCTAssertNil(s.load(Track2D.self, key: "track"))
        try original.write(to: url)
        try edit(url, "true", "fals")   // breaks JSON: also a miss
        XCTAssertNil(s.load(Track2D.self, key: "track"))
        try original.write(to: url)
        XCTAssertNotNil(s.load(Track2D.self, key: "track"), "untouched file hits again")
    }

    func testBitFlipsAnywhereAreMisses() throws {
        let s = store()
        s.save(sampleTrack(), key: "track")
        let url = file(s, "track.json")
        let original = try Data(contentsOf: url)
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<60 {
            var d = original
            let i = Int.random(in: 0..<d.count, using: &rng)
            d[i] ^= UInt8(1 << Int.random(in: 0..<7, using: &rng))
            try d.write(to: url)
            XCTAssertNil(s.load(Track2D.self, key: "track"), "flip at byte \(i) was accepted")
        }
    }

    func testTamperedDigestReformattedFileAndOtherFingerprintAreMisses() throws {
        let s = store()
        s.save(sampleTrack(), key: "track")
        let url = file(s, "track.json")
        let data = try Data(contentsOf: url)
        // Re-serialised by another JSON writer (key order / spacing): can't be verified.
        let obj = try JSONSerialization.jsonObject(with: data)
        try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]).write(to: url)
        XCTAssertNil(s.load(Track2D.self, key: "track"))
        // A digest that doesn't match.
        var text = String(decoding: data, as: UTF8.self)
        let r = try XCTUnwrap(text.range(of: #""sha256":""#))
        let at = text.index(r.upperBound, offsetBy: 0)
        text.replaceSubrange(at...at, with: text[at] == "0" ? "1" : "0")
        try Data(text.utf8).write(to: url)
        XCTAssertNil(s.load(Track2D.self, key: "track"))
        try data.write(to: url)
        XCTAssertNil(FileCheckpointStore(directory: s.directory, fingerprint: "other").load(Track2D.self, key: "track"))
        XCTAssertNil(FileCheckpointStore(directory: s.directory, fingerprint: "f").load(Track2D.self, key: "track"),
                     "a fingerprint that is a prefix of the real one must not match")
    }

    func testPlayersBinBitFlipInRecordsIsAMiss() throws {
        let s = store()
        let obs = ScriptedStages.samplePlayers(count: 50)
        s.save(obs, key: "players")
        let url = file(s, "players.bin")
        let original = try Data(contentsOf: url)
        assertBitIdentical(try XCTUnwrap(s.load([PlayerObservation].self, key: "players")), obs)
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<60 {
            var d = original
            let i = Int.random(in: 0..<d.count, using: &rng)
            d[i] ^= UInt8(1 << Int.random(in: 0..<8, using: &rng))
            try d.write(to: url)
            XCTAssertNil(s.load([PlayerObservation].self, key: "players"), "flip at byte \(i) was accepted")
        }
    }

    func testPlayersVersion1WithoutChecksumOnlyForArchives() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-v1-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("players.bin")
        let obs = ScriptedStages.samplePlayers(count: 12)
        try PlayersCodec.write(obs, to: url, fingerprint: "fp")
        // Turn it into what version 1 wrote: version field 1, no digest after the trailer.
        var d = try Data(contentsOf: url)
        d[d.startIndex + 4] = 1; d[d.startIndex + 5] = 0
        d.removeLast(32)
        try d.write(to: url)
        assertBitIdentical(try XCTUnwrap(PlayersCodec.read(from: url, expectedFingerprint: "fp")), obs)  // old archives stay readable
        XCTAssertThrowsError(try PlayersCodec.read(from: url, expectedFingerprint: "fp", requireChecksum: true))
        XCTAssertNil(FileCheckpointStore(directory: dir, fingerprint: "fp").load([PlayerObservation].self, key: "players"))
    }

    func testPlayersChecksumSurvivesTinyChunks() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-chunks-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("p.bin")
        let obs = ScriptedStages.samplePlayers(count: 97)
        for (w, r) in [(7, 5), (4096, 3), (1 << 20, 4097)] {
            try PlayersCodec.write(obs, to: url, fingerprint: "x", chunkBytes: w)
            assertBitIdentical(try XCTUnwrap(PlayersCodec.read(from: url, requireChecksum: true, chunkBytes: r), "chunks \(w)/\(r)"), obs)
        }
    }
}
