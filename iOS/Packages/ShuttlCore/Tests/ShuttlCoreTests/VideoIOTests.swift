import AVFoundation
import CoreVideo
import XCTest
@testable import ShuttlCore

final class VideoIOTests: XCTestCase {

    /// Probe + decode a real Test-split clip and compare with the OpenCV reference.
    func testRealClipMatchesOpenCV() async throws {
        let ref = try fixture("video")
        let url = repoRoot.appendingPathComponent(try XCTUnwrap(ref["relative_path"]?.string))
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test split not downloaded")
        let src = try await VideoSource.open(url)
        XCTAssertEqual(src.meta.width, Int(ref["width"]!.double!))
        XCTAssertEqual(src.meta.height, Int(ref["height"]!.double!))
        XCTAssertEqual(src.meta.nFrames, Int(ref["n_frames"]!.double!))
        XCTAssertEqual(src.meta.fps, ref["fps"]!.double!, accuracy: 0.01)
        assertClose(Array(src.meta.timestamps.prefix(10)), ref["timestamps_head"]!.doubles, tol: 2e-3, "head ts")
        assertClose(Array(src.meta.timestamps.suffix(5)), ref["timestamps_tail"]!.doubles, tol: 2e-3, "tail ts")

        let samples = try XCTUnwrap(ref["samples"]?.array)
        let got = try src.sampleFrames(samples.count) { pb in
            try ImageU8.gray(from: pb).resized(width: 160, height: 90, method: .area)
        }
        XCTAssertEqual(got.map(\.index), samples.map { Int($0["index"]!.double!) })
        for (g, s) in zip(got, samples) {
            let want = s["gray_160x90"]!.doubles
            let signed = zip(g.value.data, want).map { Double($0) - $1 }
            let diffs = signed.map(abs)
            XCTAssertLessThan(abs(Num.mean(signed)), 1.5, "frame \(g.index) bias")
            // VideoToolbox vs FFmpeg: unbiased per-pixel noise (measured mean |d| ≈ 2.6, p99 8).
            XCTAssertLessThan(Num.mean(diffs), 3.5, "frame \(g.index) mean abs diff")
            XCTAssertLessThan(Num.percentile(diffs, 99), 12, "frame \(g.index) p99 diff")
        }
    }

    /// Rotation metadata must yield upright frames with swapped dimensions.
    func testRotatedVideoIsDeliveredUpright() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-rot-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }
        // Stored 64×48 landscape with a bright block in the stored top-left corner,
        // tagged +90° (how iPhone portrait video is stored).
        try await writeTestMovie(to: url, width: 64, height: 48, frames: 12, fps: 30,
                                 transform: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 48, ty: 0))
        let src = try await VideoSource.open(url)
        XCTAssertEqual(src.meta.rotationDeg, 90)
        XCTAssertEqual(src.meta.width, 48); XCTAssertEqual(src.meta.height, 64)
        XCTAssertEqual(src.meta.nFrames, 12)
        XCTAssertEqual(src.meta.timestamps[1], 1.0 / 30, accuracy: 1e-3)

        var seen = 0
        for (i, pb) in try src.frames(step: 3) {
            XCTAssertEqual(i % 3, 0)
            let g = try ImageU8.gray(from: pb)
            XCTAssertEqual(g.width, 48); XCTAssertEqual(g.height, 64)
            // Rotating the stored frame 90° clockwise moves its top-left block to the top-right.
            XCTAssertGreaterThan(Double(g[44, 4]), 180, "block should be top-right")
            XCTAssertLessThan(Double(g[4, 4]), 80, "top-left should be dark")
            XCTAssertLessThan(Double(g[44, 60]), 80, "bottom-right should be dark")
            seen += 1
        }
        XCTAssertEqual(seen, 4)

        let mid = try src.frames(start: 7, stop: 9).map(\.0)
        XCTAssertEqual(mid, [7, 8])
        let sampled = try src.sampleFrames(3) { _ in 1 }.map(\.index)
        XCTAssertEqual(sampled, [0, 5, 11])
    }

    private func writeTestMovie(to url: URL, width: Int, height: Int, frames: Int, fps: Int32,
                                transform: CGAffineTransform) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        input.transform = transform
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for f in 0..<frames {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, try XCTUnwrap(adaptor.pixelBufferPool), &pb)
            let buf = try XCTUnwrap(pb)
            CVPixelBufferLockBaseAddress(buf, [])
            let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(buf)
            for y in 0..<height {
                for x in 0..<width {
                    let v: UInt8 = (x < 16 && y < 16) ? 250 : 20
                    let p = base + y * stride + x * 4
                    p[0] = v; p[1] = v; p[2] = v; p[3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(buf, [])
            XCTAssertTrue(adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: fps)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, writer.error?.localizedDescription ?? "")
    }
}
