import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

final class DetectPartialStoreTests: XCTestCase {
    private func store(_ fp: String = "fp") -> FileCheckpointStore {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-dp-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return FileCheckpointStore(directory: d, fingerprint: fp)
    }

    private func cp(_ next: Int, bg: Data? = ResumableStages.background, name: String = "net") -> DetectCheckpoint {
        DetectCheckpoint(nextFrame: next, candidates: (0..<next).map(ResumableStages.candidates), background: bg, detectorName: name)
    }

    private func bytes(_ s: FileCheckpointStore) throws -> Data { try Data(contentsOf: s.detectPartialURL) }

    func testAppendOnlyDeltasAndRoundTrip() throws {
        let s = store()
        XCTAssertNil(s.loadDetectPartial())
        s.saveDetectPartial(cp(8))
        let b8 = try bytes(s)
        s.saveDetectPartial(cp(16))
        let b16 = try bytes(s)
        s.saveDetectPartial(cp(24))
        let b24 = try bytes(s)
        XCTAssertTrue(b16.starts(with: b8) && b24.starts(with: b16), "append-only")
        XCTAssertLessThan(b24.count - b16.count, ResumableStages.background.count, "background not repeated")
        XCTAssertEqual(s.loadDetectPartial(), cp(24))
        s.saveDetectPartial(cp(24))
        XCTAssertEqual(try bytes(s), b24, "same checkpoint again writes nothing")
        s.saveDetectPartial(cp(0, bg: nil))                     // shorter → restarted pass → rewrite
        XCTAssertEqual(s.loadDetectPartial(), cp(0, bg: nil))
    }

    func testBackgroundChangeIsAppendedAndTakesEffectWithItsDelta() throws {
        let s = store()
        s.saveDetectPartial(cp(8))
        let other = Data([9, 9, 9])
        s.saveDetectPartial(cp(16, bg: other))
        XCTAssertEqual(s.loadDetectPartial()?.background, other)
        s.saveDetectPartial(cp(24, bg: nil))
        XCTAssertNil(s.loadDetectPartial()?.background)
        XCTAssertEqual(s.loadDetectPartial()?.nextFrame, 24)
    }

    func testTornTailIsDroppedAndRepairedOnTheNextSave() throws {
        let s = store(), ref = store()
        for n in [8, 16, 24] { ref.saveDetectPartial(cp(n)) }
        s.saveDetectPartial(cp(8)); s.saveDetectPartial(cp(16))
        let b16 = try bytes(s)
        s.saveDetectPartial(cp(24))
        for cut in [1, 7, 30] {
            let full = try bytes(ref)
            try full.dropLast(cut).write(to: s.detectPartialURL)
            XCTAssertEqual(s.loadDetectPartial(), cp(16), "cut \(cut): back to the last complete record")
        }
        s.saveDetectPartial(cp(24))
        XCTAssertEqual(try bytes(s), try bytes(ref), "torn tail truncated, then the delta appended")
        XCTAssertTrue(try bytes(s).starts(with: b16))
        // A background record whose delta never made it is not applied.
        try b16.write(to: s.detectPartialURL)
        let h = try FileHandle(forWritingTo: s.detectPartialURL)
        _ = try h.seekToEnd()
        try h.write(contentsOf: DetectPartialFile.backgroundRecord(Data([1])))
        try h.close()
        XCTAssertEqual(s.loadDetectPartial(), cp(16))
    }

    func testForeignFingerprintDetectorOrPrefixMeansRewrite() throws {
        let s = store("fp1")
        s.saveDetectPartial(cp(16))
        XCTAssertNil(FileCheckpointStore(directory: s.directory, fingerprint: "fp2").loadDetectPartial(), "stale video/build/options")
        FileCheckpointStore(directory: s.directory, fingerprint: "fp2").saveDetectPartial(cp(8))
        XCTAssertNil(s.loadDetectPartial(), "fp2 replaced it")
        XCTAssertEqual(FileCheckpointStore(directory: s.directory, fingerprint: "fp2").loadDetectPartial(), cp(8))

        let t = store()
        t.saveDetectPartial(cp(16, name: "a"))
        t.saveDetectPartial(cp(24, name: "b"))
        XCTAssertEqual(t.loadDetectPartial(), cp(24, name: "b"))
        var changed = cp(24, name: "b")          // same detector, different prefix
        changed.candidates[3] = [ShuttleCandidate(x: 1, y: 1, score: 1)]
        t.saveDetectPartial(changed)
        XCTAssertEqual(t.loadDetectPartial(), changed, "a different prefix is never appended to")
        t.clearDetectPartial()
        XCTAssertNil(t.loadDetectPartial())
        t.clearDetectPartial()                                  // idempotent
    }

    func testClearWithTheWholeStore() {
        let s = store()
        s.saveDetectPartial(cp(8))
        s.clear()
        XCTAssertNil(s.loadDetectPartial())
    }
}
