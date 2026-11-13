import XCTest
import ShuttlAnalysis
import ShuttlCore
import ShuttlRuntime

final class AnalysisArchiveTests: XCTestCase {
    private func makeArchive() -> (AnalysisArchive, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-archive-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let a = AnalysisArchive.standard(documents: root.appendingPathComponent("Documents"),
                                         applicationSupport: root.appendingPathComponent("Support"))
        return (a, root)
    }

    static func players(_ n: Int) -> [PlayerObservation] {
        (0..<n).map { i in
            let d = Double(i)
            let kps = (0..<17).map { k in Keypoint(x: d + Double(k), y: d * 0.5, conf: Double(k) / 17) }
            let foot: SIMD2<Double>? = i % 5 == 0 ? nil : SIMD2(-3 + d * 1e-4, i % 7 == 1 ? .nan : 0.25)
            return PlayerObservation(frame: i * 2, trackID: i % 4 + 1, bbox: BBox(x1: d, y1: 2, x2: d + 50, y2: 190.5),
                                     score: 0.25 + Double(i % 3) / 4, side: [CourtSide.near, .far, .unknown][i % 3],
                                     footWorld: foot, keypoints: i % 9 == 0 ? nil : kps)
        }
    }

    static func analysis(frames: Int, players n: Int) -> MatchAnalysis {
        let t = (0..<frames).map { Double($0) / 60 }
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: frames, timestamps: t)
        let track = Track2D(t: t, xy: t.map { SIMD2($0 * 10, 500) }, conf: t.map { _ in 0.9 },
                            visible: t.map { _ in true }, interpolated: t.map { _ in false })
        let hit = HitEvent(frame: 3, t: 0.05, xyPx: [10, 20], hitter: .near, confidence: 0.8)
        var a = MatchAnalysis(meta: meta, track: track, players: players(n), hits: [hit],
                              rallies: [RallyV2(startFrame: 0, endFrame: 30, shots: [ShotEventV2(hit: hit, segment: nil)])],
                              score: [.object(["score": .array([.number(1), .number(0)])])], processingTimeSeconds: 3.5)
        a.diagnostics["stages"] = .object(["detect": .object(["ok": .bool(true)])])
        return a
    }

    private func stripped(_ p: [PlayerObservation]) -> [PlayerObservation] {
        p.map { var o = $0; o.keypoints = nil; return o }
    }

    /// Equality that treats NaN == NaN (bit patterns).
    private func assertSame(_ a: [PlayerObservation], _ b: [PlayerObservation], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        func key(_ o: PlayerObservation) -> String {
            let f = o.footWorld.map { "\($0.x.bitPattern),\($0.y.bitPattern)" } ?? "-"
            let k = o.keypoints.map { $0.map { "\($0.x.bitPattern)/\($0.y.bitPattern)/\($0.conf.bitPattern)" }.joined(separator: ";") } ?? "-"
            return "\(o.frame)|\(o.trackID)|\(o.side)|\(o.bbox)|\(o.score.bitPattern)|\(f)|\(k)"
        }
        XCTAssertEqual(a.map(key), b.map(key), file: file, line: line)
    }

    func testRoundTripDropsOnlyKeypointsAndJSONStaysAMatchAnalysis() throws {
        let (archive, _) = makeArchive()
        let id = UUID(), original = Self.analysis(frames: 300, players: 200)
        let url = try archive.save(original, matchID: id)
        XCTAssertEqual(url.lastPathComponent, "\(id.uuidString.lowercased()).json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: archive.playersURL(matchID: id).path))
        XCTAssertTrue(archive.playersURL(matchID: id).lastPathComponent.hasSuffix(".players.bin"))

        let plain = try ShuttlJSON.load(MatchAnalysis.self, from: url)       // what an old loader sees
        XCTAssertTrue(plain.players.isEmpty)
        XCTAssertEqual(plain.rallies.count, 1)
        XCTAssertEqual(plain.diagnostics["archive"]?["players_count"]?.double, 200)

        let back = try archive.load(matchID: id)
        assertSame(back.players, stripped(original.players))
        XCTAssertNil(back.diagnostics["archive"], "bookkeeping key removed on load")
        XCTAssertEqual(back.diagnostics["stages"], original.diagnostics["stages"])
        XCTAssertEqual(back.track?.n, 300)
        XCTAssertEqual(back.hits, original.hits)
        XCTAssertEqual(back.score, original.score)
        XCTAssertEqual(back.processingTimeSeconds, 3.5)

        try archive.save(original, matchID: id, keepKeypoints: true)
        assertSame(try archive.load(matchID: id).players, original.players)
    }

    func testStaleSideFileIsIgnoredAndLegacyJSONLoads() throws {
        let (archive, _) = makeArchive()
        let id = UUID(), a = Self.analysis(frames: 10, players: 5)
        try archive.save(a, matchID: id)
        try PlayersCodec.write(Self.players(3), to: archive.playersURL(matchID: id), fingerprint: "someone-else")
        XCTAssertTrue(try archive.load(matchID: id).players.isEmpty, "never mix in another save's players")

        let legacy = UUID()
        try FileManager.default.createDirectory(at: archive.analysesDirectory, withIntermediateDirectories: true)
        try ShuttlJSON.encoder().encode(a).write(to: archive.analysisURL(matchID: legacy))
        assertSame(try archive.load(matchID: legacy).players, a.players)
    }

    func testDeleteAllRemovesOnlyThatMatch() throws {
        let (archive, root) = makeArchive()
        let fm = FileManager.default
        let gone = UUID(), kept = UUID()
        let video = root.appendingPathComponent("Documents/Matches/match_1.mov")
        try fm.createDirectory(at: video.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: video)
        for id in [gone, kept] {
            try archive.save(Self.analysis(frames: 5, players: 2), matchID: id)
            archive.checkpointStore(matchID: id, video: video, build: "1", options: PipelineOptions()).save([1, 2], key: "x")
        }
        let tmp = archive.analysesDirectory.appendingPathComponent(".\(gone.uuidString.lowercased()).players.bin.ABC.tmp")
        try Data([0]).write(to: tmp)

        let removed = archive.deleteAll(matchID: gone, video: video)
        XCTAssertEqual(Set(removed.map(\.lastPathComponent)),
                       ["\(gone.uuidString.lowercased()).json", "\(gone.uuidString.lowercased()).players.bin",
                        gone.uuidString.lowercased(), "match_1.mov", tmp.lastPathComponent])
        for u in [archive.analysisURL(matchID: gone), archive.playersURL(matchID: gone),
                  archive.checkpointDirectory(matchID: gone), video, tmp] {
            XCTAssertFalse(fm.fileExists(atPath: u.path), u.lastPathComponent)
        }
        for u in [archive.analysisURL(matchID: kept), archive.playersURL(matchID: kept), archive.checkpointDirectory(matchID: kept)] {
            XCTAssertTrue(fm.fileExists(atPath: u.path), "other match untouched: \(u.lastPathComponent)")
        }
        XCTAssertEqual(archive.deleteAll(matchID: gone, video: video), [], "idempotent")
        archive.removeCheckpoints(matchID: kept)
        XCTAssertFalse(fm.fileExists(atPath: archive.checkpointDirectory(matchID: kept).path))
        XCTAssertTrue(fm.fileExists(atPath: archive.analysisURL(matchID: kept).path))
    }

    func testCheckpointStoreFingerprintFollowsOptionsAndVideo() throws {
        let (archive, root) = makeArchive()
        let video = root.appendingPathComponent("v.mov")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(count: 100).write(to: video)
        let id = UUID()
        let corners: [String: SIMD2<Double>] = ["nb_ld": [0.7, 0.9], "nb_rd": [0.3, 0.9]]
        let optsA = PipelineOptions(manualCorners: corners), optsB = PipelineOptions(doubles: true, manualCorners: corners)
        archive.checkpointStore(matchID: id, video: video, build: "1.0-7", options: optsA).save([1.5], key: "camera")
        XCTAssertEqual(archive.checkpointStore(matchID: id, video: video, build: "1.0-7", options: optsA)
            .load([Double].self, key: "camera"), [1.5])
        XCTAssertNil(archive.checkpointStore(matchID: id, video: video, build: "1.0-7", options: optsB)
            .load([Double].self, key: "camera"), "different options → fresh run")
        XCTAssertNil(archive.checkpointStore(matchID: id, video: video, build: "1.0-8", options: optsA)
            .load([Double].self, key: "camera"), "different build")
        try Data(count: 101).write(to: video)
        XCTAssertNil(archive.checkpointStore(matchID: id, video: video, build: "1.0-7", options: optsA)
            .load([Double].self, key: "camera"), "different video file")
        XCTAssertEqual(archive.checkpointDirectory(matchID: id).deletingLastPathComponent().lastPathComponent, "AnalysisCheckpoints")
        XCTAssertEqual(archive.analysesDirectory.lastPathComponent, "Analyses")
    }

    /// 45-min singles: 324k observations + a 162k-frame track. Gated: SHUTTL_BIG=1.
    func testLongMatchSaveAndLoadStayUnder400MB() throws {
        let big = ProcessInfo.processInfo.environment["SHUTTL_BIG"] == "1"
        let (n, frames) = big ? (324_000, 162_000) : (20_000, 10_000)
        let (archive, _) = makeArchive()
        let id = UUID()
        let a = Self.analysis(frames: frames, players: n)
        let save = try Footprint.measure { try archive.save(a, matchID: id) }
        var count = 0
        let load = try Footprint.measure { count = try archive.load(matchID: id).players.count }
        XCTAssertEqual(count, n)
        let fm = FileManager.default
        let json = (try fm.attributesOfItem(atPath: archive.analysisURL(matchID: id).path)[.size] as? Int) ?? 0
        let side = (try fm.attributesOfItem(atPath: archive.playersURL(matchID: id).path)[.size] as? Int) ?? 0
        var legacy = -1.0
        if !big {   // the old AnalysisStore path, for comparison (only at the small size)
            legacy = try Footprint.measure { _ = try ShuttlJSON.encoder().encode(a) }
        }
        print(String(format: "[mem] archive n=%d frames=%d  save +%.1f MB  load +%.1f MB  json %.1f MB  players %.1f MB  legacy full-JSON encode +%.1f MB",
                     n, frames, save, load, Double(json) / 1_048_576, Double(side) / 1_048_576, legacy))
        XCTAssertLessThan(save, 400)
        XCTAssertLessThan(load, 400)
    }
}
