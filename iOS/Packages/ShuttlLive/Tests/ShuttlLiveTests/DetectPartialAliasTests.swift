import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlDetector
@testable import ShuttlLive

/// Monitor hypothesis (bg-corrupt-after-resume): the loaded detect_partial Data (maybe a file
/// mapping) is later truncated + appended IN PLACE by saveDetectPartial. If any background-plan
/// pixels aliased those bytes, a later save could rewrite them. They must be private copies.
final class DetectPartialAliasTests: XCTestCase {
    /// A realistic plan: 23 net-size segments (≈ 10 MB, like the 45-min clip).
    private func plan(segments: Int = 23) -> BackgroundPlan {
        let imgs = (0..<segments).map { s -> NetImage in
            let img = NetImage()
            for i in 0..<img.count { img.pixels[i] = UInt8(truncatingIfNeeded: i &* 31 &+ s &* 97 &+ (i >> 9)) }
            return img
        }
        return BackgroundPlan(starts: (0..<segments).map { $0 * 7200 }, images: imgs)
    }

    private func candidates(_ n: Int, salt: Int = 0) -> [[ShuttleCandidate]] {
        (0..<n).map { i in i % 3 == 0 ? [] : [ShuttleCandidate(x: Double(i % 1920), y: Double((i + salt) % 1080), score: 0.5)] }
    }

    func testSavingWhileALoadedPlanIsAliveNeverChangesIt() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dpa-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        let bg = plan().serialized()
        store.saveDetectPartial(DetectCheckpoint(nextFrame: 43200, candidates: candidates(43200), background: bg,
                                                 detectorName: "tracknetv3"))
        let loaded = try XCTUnwrap(store.loadDetectPartial())
        XCTAssertEqual(loaded.background, bg)
        // The pipeline's resume path: deserialize (pre-fix plan) and seal (the detector's copy).
        let raw = try XCTUnwrap(BackgroundPlan.deserialize(try XCTUnwrap(loaded.background)))
        let sealed = raw.sealed()
        let rawDigests = raw.images.map(BackgroundPlan.digest)
        // No plan pixel may live inside the loaded Data's bytes.
        try XCTUnwrap(loaded.background).withUnsafeBytes { d in
            let lo = UInt(bitPattern: d.baseAddress), hi = lo + UInt(d.count)
            for img in raw.images + sealed.images {
                let a = UInt(bitPattern: img.pixels), b = a + UInt(img.count)
                XCTAssertTrue(b <= lo || a >= hi, "plan pixels alias the loaded checkpoint bytes")
            }
        }
        // Keep saving in place (append, re-append after a torn tail, a changed background forcing a
        // new record, and a full rewrite) while both plans are alive.
        var next = 43200
        for round in 0..<12 {
            next += 7200
            let changed = round == 5 ? plan(segments: 23).serialized().dropLast(0) + Data([1]) : bg
            store.saveDetectPartial(DetectCheckpoint(nextFrame: next, candidates: candidates(next), background: changed,
                                                     detectorName: "tracknetv3"))
            if round == 8 {                                         // a torn tail on disk
                let h = try FileHandle(forWritingTo: store.detectPartialURL)
                try h.seekToEnd(); try h.write(contentsOf: Data(repeating: 0xAB, count: 5000)); try h.close()
            }
            if round == 10 {                                        // candidates changed → full rewrite
                store.saveDetectPartial(DetectCheckpoint(nextFrame: next, candidates: candidates(next, salt: 1),
                                                         background: bg, detectorName: "tracknetv3"))
            }
            XCTAssertEqual(raw.images.map(BackgroundPlan.digest), rawDigests, "round \(round): loaded plan changed")
            XCTAssertEqual(sealed.changedSegments(), [], "round \(round): sealed plan changed")
        }
        XCTAssertEqual(sealed.serialized(), bg)
        XCTAssertEqual(store.loadDetectPartial()?.nextFrame, next)
    }

    /// A damaged in-memory plan is recoverable: the checkpoint still holds the good bytes.
    func testBackgroundChangedIsRetryable() {
        XCTAssertTrue(AnalysisErrorClassifier.isRetryable(DetectorError.backgroundChanged([15])))
    }

    /// A decode stalled by a busy hardware decoder (camera preview) is an interruption: retry.
    func testDecodeStalledIsRetryable() {
        XCTAssertTrue(AnalysisErrorClassifier.isRetryable(ShuttlCoreError.decodeStalled(seconds: 20, frame: 1200)))
    }
}
