import AVFoundation
import XCTest
@testable import ShuttlCore

/// Phone pass 09-30 (F-P2): with the record screen's camera preview live, the players stage's
/// AVAssetReader got no frame for 920 s before -11847. The watchdog must end such a read quickly
/// with an error the pipeline treats as an interruption (retryable), never as a clean end.
final class DecodeWatchdogTests: XCTestCase {
    private var clip: URL {
        repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4")
    }

    override func tearDown() {
        FrameSequence.copyNextHook = nil
        FrameSequence.stallTimeout = 20
        super.tearDown()
    }

    func testStalledDecoderEndsTheReadWithAnError() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split not downloaded")
        let src = try await VideoSource.open(clip)
        FrameSequence.stallTimeout = 0.5
        let hang = DispatchSemaphore(value: 0)
        var delivered = 0
        // 5 real frames, then a decoder that never answers (it is released only at the end).
        FrameSequence.copyNextHook = { output in
            if delivered < 5 { delivered += 1; return output.copyNextSampleBuffer() }
            _ = hang.wait(timeout: .now() + 1.5)
            return nil
        }
        let seq = try src.frames()
        let t0 = Date()
        var got = 0
        for _ in seq { got += 1 }
        let elapsed = Date().timeIntervalSince(t0)
        hang.signal()
        XCTAssertEqual(got, 5)
        XCTAssertLessThan(elapsed, 5, "the stall must end the read quickly")
        guard case .decodeStalled(let secs, let frame)? = seq.error as? ShuttlCoreError else {
            return XCTFail("expected decodeStalled, got \(String(describing: seq.error))")
        }
        XCTAssertEqual(secs, 0.5)
        XCTAssertEqual(frame, 4)
        XCTAssertTrue(String(describing: seq.error!).contains("interrupted"),
                      "worded as an interruption so AnalysisErrorClassifier retries it")
    }

    func testNormalDecodeIsUntouched() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split not downloaded")
        let src = try await VideoSource.open(clip)
        FrameSequence.stallTimeout = 2
        let seq = try src.frames()
        var n = 0
        for _ in seq { n += 1 }
        XCTAssertEqual(n, src.meta.nFrames)
        XCTAssertNil(seq.error)
    }
}
