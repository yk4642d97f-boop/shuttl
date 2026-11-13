import AVFoundation
import Accelerate
import CoreML
import XCTest
import ShuttlCore
@testable import ShuttlDetector

/// Phone-format streaming benchmark: a real Test clip upscaled to 1920x1080 @ 60 fps (H.264,
/// written with AVAssetWriter), then decode → letterbox → CoreML → fusion → decode per model.
/// Opt-in (writes a temp video):  SHUTTL_DET_BENCH=1 swift test -c release --filter BenchTests
/// Run alone for clean memory numbers (peak physical footprint while streaming, relative to the
/// footprint with the model loaded and warmed up).
final class BenchTests: XCTestCase {
    func testStreaming1080p60() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHUTTL_DET_BENCH"] == "1", "set SHUTTL_DET_BENCH=1")
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_05_02.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split not downloaded")
        let repeats = Int(ProcessInfo.processInfo.environment["SHUTTL_DET_BENCH_REPEATS"] ?? "4") ?? 4
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-bench-1080p60-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }
        let t0 = CFAbsoluteTimeGetCurrent()
        let written = try await writeUpscaled(clip, to: url, width: 1920, height: 1080, fps: 60, repeats: repeats)
        print(String(format: "[bench] wrote %d frames 1920x1080@60 in %.1f s", written, CFAbsoluteTimeGetCurrent() - t0))
        let video = try await VideoSource.open(url)
        XCTAssertEqual(video.meta.nFrames, written)
        XCTAssertEqual(video.meta.fps, 60, accuracy: 0.5)

        // decode-only baseline (AVAssetReader → upright BGRA)
        var t = CFAbsoluteTimeGetCurrent()
        var n = 0
        for _ in try video.frames() { n += 1 }
        let decodeMs = (CFAbsoluteTimeGetCurrent() - t) / Double(n) * 1e3
        print(String(format: "[bench] decode only: %.2f ms/frame (%.0f fps)", decodeMs, 1e3 / decodeMs))

        for kind in (ProcessInfo.processInfo.environment["SHUTTL_DET_MODELS"] ?? "tracknetv3,wasb").split(separator: ",") {
            let modelURL = try XCTUnwrap(HeatmapDetector.findModel(DetectorKind(rawValue: String(kind))!, in: modelsDir))
            let before = FootprintSampler.footprint()
            t = CFAbsoluteTimeGetCurrent()
            let det = try await HeatmapDetector.load(modelURL: modelURL, cacheDirectory: compileCache)
            let loadS = CFAbsoluteTimeGetCurrent() - t
            // warm-up (ANE program load) on a short prefix
            _ = try det.detectFrames(VideoFrameSource(video, stop: 3 * det.spec.seqLen), width: 1920, height: 1080, fps: 60,
                                     nFrames: 3 * det.spec.seqLen)
            let sampler = FootprintSampler()
            t = CFAbsoluteTimeGetCurrent()
            let cands = try await det.detectVideo(video)
            let wall = CFAbsoluteTimeGetCurrent() - t
            let mem = sampler.stop()
            XCTAssertEqual(cands.count, written)
            let s = det.lastStats, f = Double(written)
            print(String(format: "[bench] %@ 1080p60 %d frames: %.1f fps end-to-end (%.2f ms/frame) → 1 min of 60 fps video in %.0f s | net %.2f ms/frame, background pass %.2f ms/frame, letterbox %.2f ms/frame, heatmap decode %.2f ms/frame | load+compile %.2f s, model resident +%.0f MB, streaming peak +%.1f MB over loaded (abs %.0f MB)",
                         String(kind), written, f / wall, 1e3 * wall / f, 3600 * wall / f, 1e3 * s.networkSeconds / f,
                         1e3 * s.backgroundSeconds / f, 1e3 * s.preprocessSeconds / f, 1e3 * s.decodeSeconds / f, loadS,
                         Double(sampler.baseline - before) / 1e6, mem.deltaMB, mem.peakMB))
        }
    }

    /// Re-encodes `src` upscaled to width×height at `fps` (every source frame, `repeats` times).
    private func writeUpscaled(_ src: URL, to url: URL, width: Int, height: Int, fps: Int32, repeats: Int) async throws -> Int {
        let source = try await VideoSource.open(src)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 20_000_000],
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        var k = 0
        for _ in 0..<repeats {
            for (_, pb) in try source.frames() {
                while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
                var out: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, try XCTUnwrap(adaptor.pixelBufferPool), &out)
                let dst = try XCTUnwrap(out)
                CVPixelBufferLockBaseAddress(pb, .readOnly); CVPixelBufferLockBaseAddress(dst, [])
                var s = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(CVPixelBufferGetHeight(pb)),
                                      width: vImagePixelCount(CVPixelBufferGetWidth(pb)), rowBytes: CVPixelBufferGetBytesPerRow(pb))
                var d = vImage_Buffer(data: CVPixelBufferGetBaseAddress(dst), height: vImagePixelCount(height),
                                      width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(dst))
                vImageScale_ARGB8888(&s, &d, nil, vImage_Flags(kvImageNoFlags))
                CVPixelBufferUnlockBaseAddress(dst, []); CVPixelBufferUnlockBaseAddress(pb, .readOnly)
                XCTAssertTrue(adaptor.append(dst, withPresentationTime: CMTime(value: CMTimeValue(k), timescale: fps)))
                k += 1
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, writer.error?.localizedDescription ?? "")
        return k
    }
}
