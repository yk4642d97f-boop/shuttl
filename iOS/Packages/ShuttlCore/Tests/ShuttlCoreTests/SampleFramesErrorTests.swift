import AVFoundation
import XCTest
@testable import ShuttlCore

/// F-P3: `sampleFrames` (calibration frames) ignored a reader that failed or stalled and returned
/// the frames it had, so calibration solved and checkpointed a camera on an interrupted decoder.
final class SampleFramesErrorTests: XCTestCase {
    private var clip: URL {
        repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4")
    }

    override func tearDown() {
        FrameSequence.copyNextHook = nil
        FrameSequence.stallTimeout = 20
        super.tearDown()
    }

    /// One real sample, then a decoder that never answers.
    private func stallAfterOneSample() -> DispatchSemaphore {
        let hang = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var delivered = 0
        FrameSequence.stallTimeout = 0.3
        FrameSequence.copyNextHook = { output in
            let n: Int = lock.withLock { delivered += 1; return delivered }
            if n == 1 { return output.copyNextSampleBuffer() }
            _ = hang.wait(timeout: .now() + 1)
            return nil
        }
        return hang
    }

    private func assertStalled(_ body: () throws -> Void, file: StaticString = #filePath, line: UInt = #line) {
        do {
            try body()
            XCTFail("a stalled read must throw, not return fewer frames", file: file, line: line)
        } catch {
            guard case .decodeStalled? = error as? ShuttlCoreError else {
                return XCTFail("expected decodeStalled, got \(error)", file: file, line: line)
            }
        }
    }

    func testDenseSamplingThrowsWhenTheReaderStalls() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split not downloaded")
        let src = try await VideoSource.open(clip)
        let n = min(src.meta.nFrames, 30)
        try XCTSkipUnless(Double(src.meta.nFrames - 1) / Double(n - 1) < 120, "clip too long for the dense branch")
        let hang = stallAfterOneSample()
        defer { hang.signal() }
        assertStalled { _ = try src.sampleFrames(n) { _ in 0 } }
    }

    /// 22.9 s: two samples are ≥ 120 frames apart, so each one gets its own seeking reader.
    private var longClip: URL {
        repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_03.mp4")
    }

    func testSeekingSamplingThrowsWhenTheReaderStalls() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: longClip.path), "Test split not downloaded")
        let src = try await VideoSource.open(longClip)
        try XCTSkipUnless(src.meta.nFrames - 1 >= 120, "clip too short for the seek-per-frame branch")
        let hang = stallAfterOneSample()
        defer { hang.signal() }
        assertStalled { _ = try src.sampleFrames(2) { _ in 0 } }
    }

    func testCleanSamplingIsUnchanged() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split not downloaded")
        let src = try await VideoSource.open(clip)
        XCTAssertEqual(try src.sampleFrames(5) { _ in 0 }.count, 5)
        XCTAssertEqual(try src.sampleFrames(2) { _ in 0 }.count, 2)
    }
}
