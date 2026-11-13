import AVFoundation
import CoreVideo
import XCTest
@testable import ShuttlCore

/// A file whose samples share presentation times (a broken muxer) decodes to fewer
/// frames than it has samples: AVAssetReader drops the duplicates. VideoSource must count
/// frames the way the decoder delivers them, or every stage that needs consecutive frame
/// indices (the detector) fails on such a file.
final class DuplicatePTSTests: XCTestCase {
    private let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-duppts-\(UUID().uuidString)")

    override func tearDown() { try? FileManager.default.removeItem(at: dir) }

    func testDuplicatePTSGivesConsecutiveFramesAndRealTimes() async throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let base = dir.appendingPathComponent("base.mov"), dup = dir.appendingPathComponent("dup.mov")
        try await writeMovie(base, frames: 30, fps: 30)
        try makeDuplicatePTS(base, dup, at: [10, 20])

        // The patched file really has duplicate PTS at the sample level.
        let asset = AVURLAsset(url: dup)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let reader = try AVAssetReader(asset: asset)
        let raw = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(raw); reader.startReading()
        var samplePTS: [Int64] = []
        while let sb = raw.copyNextSampleBuffer() {
            guard CMSampleBufferGetNumSamples(sb) > 0 else { continue }
            samplePTS.append(CMSampleBufferGetPresentationTimeStamp(sb).convertScale(3000, method: .default).value)
        }
        XCTAssertEqual(samplePTS.count, 30)
        XCTAssertEqual(Set(samplePTS).count, 28, "two pairs share a PTS")

        let src = try await VideoSource.open(dup)
        XCTAssertEqual(src.droppedDuplicateSamples, 2)
        XCTAssertEqual(src.meta.nFrames, 28)
        let t = src.meta.timestamps
        XCTAssertTrue(zip(t, t.dropFirst()).allSatisfy { $1 > $0 }, "strictly increasing")
        // Real times kept (not replaced by i/fps): the collapsed pair leaves a 2-frame gap.
        XCTAssertEqual(t[11] - t[10], 2.0 / 30, accuracy: 1e-6)
        XCTAssertEqual(t[1] - t[0], 1.0 / 30, accuracy: 1e-6)

        XCTAssertEqual(try src.frames().map(\.0), Array(0..<28), "every frame, no index gaps")
        XCTAssertEqual(try src.frames(start: 12).map(\.0), Array(12..<28))
        XCTAssertEqual(try src.frames(step: 3).map(\.0), Array(stride(from: 0, to: 28, by: 3)))
        let sampled = try src.sampleFrames(6) { _ in 0 }.map(\.index)
        XCTAssertEqual(sampled.count, 6)
        XCTAssertTrue(sampled.allSatisfy { $0 < 28 })

        // A normal file is untouched.
        let clean = try await VideoSource.open(base)
        XCTAssertEqual(clean.droppedDuplicateSamples, 0)
        XCTAssertEqual(clean.meta.nFrames, 30)
        XCTAssertEqual(try clean.frames().map(\.0), Array(0..<30))
    }

    /// H.264 without frame reordering, so there is no ctts and PTS == DTS.
    private func writeMovie(_ url: URL, frames: Int, fps: Int32) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 48,
            AVVideoCompressionPropertiesKey: [AVVideoAllowFrameReorderingKey: false],
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 48,
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
            memset(CVPixelBufferGetBaseAddress(buf), Int32(8 * f), CVPixelBufferGetDataSize(buf))
            CVPixelBufferUnlockBaseAddress(buf, [])
            XCTAssertTrue(adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: fps)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, writer.error?.localizedDescription ?? "")
    }
}
