import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlExport

final class OverlayExporterTests: XCTestCase {
    func testExportPreservesFramesAudioAndRendersOverlay() async throws {
        let input = try await makeMovie(frameCount: 24, width: 160, height: 120, fps: 12)
        defer { try? FileManager.default.removeItem(at: input) }
        let analysis = makeAnalysis(frames: 24, width: 160, height: 120, fps: 12)
        let output = try await OverlayExporter.export(video: input, analysis: analysis,
            options: ExportOptions(showScore: true, watermark: false, outputSize: .source))
        defer { try? FileManager.default.removeItem(at: output) }

        let info = try await inspect(output)
        XCTAssertEqual(info.frames, 24)
        XCTAssertEqual(info.duration, 2, accuracy: 1.0 / 12)
        XCTAssertGreaterThan(info.audioTracks, 0)
        XCTAssertGreaterThan(info.yellowPixels, 0, "expected a butter-colored shuttle trail")
        XCTAssertEqual(info.emptyRegionYellowPixels, 0, "must not draw a shuttle where the track is absent")
        try savePNG(info.firstOverlayFrame, to: URL(fileURLWithPath: "/tmp/shuttl_export_frame.png"))
    }

    func testRallyRangeExportsOnlyRallyFrames() async throws {
        let input = try await makeMovie(frameCount: 24, width: 160, height: 120, fps: 12)
        defer { try? FileManager.default.removeItem(at: input) }
        let analysis = makeAnalysis(frames: 24, width: 160, height: 120, fps: 12)
        let output = try await OverlayExporter.export(video: input, analysis: analysis,
            options: ExportOptions(range: .rally(index: 0), showScore: false, watermark: false))
        defer { try? FileManager.default.removeItem(at: output) }
        let info = try await inspect(output)
        XCTAssertEqual(info.frames, 7)
        XCTAssertEqual(info.duration, 7.0 / 12.0, accuracy: 1.0 / 12)
        XCTAssertGreaterThan(info.audioTracks, 0)
    }

    func testRelease1080pBenchmark() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHUTTL_EXPORT_BENCHMARK"] == "1")
        let frameCount = 60
        let fps = 60
        let input = try await makeMovie(frameCount: frameCount, width: 1920, height: 1080, fps: fps)
        defer { try? FileManager.default.removeItem(at: input) }
        let analysis = makeAnalysis(frames: frameCount, width: 1920, height: 1080, fps: Double(fps))
        let started = ContinuousClock.now
        let output = try await OverlayExporter.export(video: input, analysis: analysis,
            options: ExportOptions(showScore: false, watermark: true, outputSize: .source))
        let elapsed = started.duration(to: .now)
        defer { try? FileManager.default.removeItem(at: output) }
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        print("ShuttlExport benchmark: \(frameCount) frames in \(seconds)s = \(Double(frameCount) / seconds) fps")
        XCTAssertGreaterThan(seconds, 0)
    }

    func testCancellationDeletesPartialOutput() async throws {
        let input = try await makeMovie(frameCount: 36, width: 160, height: 120, fps: 12)
        defer { try? FileManager.default.removeItem(at: input) }
        let analysis = makeAnalysis(frames: 36, width: 160, height: 120, fps: 12)
        let tempDirectory = FileManager.default.temporaryDirectory
        let before = Set((try? FileManager.default.contentsOfDirectory(at: tempDirectory,
            includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix("shuttl-export-") }.map(\.lastPathComponent)) ?? [])
        let cancellation = CancellationBox()
        let task: Task<URL, Error> = Task {
            try await OverlayExporter.export(video: input, analysis: analysis,
                options: ExportOptions(showScore: false, watermark: false)) { value in
                    if value > 0.12 { cancellation.cancel() }
                }
        }
        cancellation.task = task
        do {
            let url = try await task.value
            try? FileManager.default.removeItem(at: url)
            XCTFail("Expected cancelled export")
        } catch {
            XCTAssertTrue(error is CancellationError || {
                if case .cancelled? = error as? OverlayExportError { return true }
                return false
            }())
        }
        let after = Set(try FileManager.default.contentsOfDirectory(at: tempDirectory,
            includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix("shuttl-export-") }.map(\.lastPathComponent))
        XCTAssertTrue(after.subtracting(before).isEmpty, "partial export files should be removed")
    }

    private func makeAnalysis(frames: Int, width: Int, height: Int, fps: Double) -> MatchAnalysis {
        let timestamps = (0..<frames).map { Double($0) / fps }
        let xy: [SIMD2<Double>] = (0..<frames).map { frame in
            if (6...12).contains(frame) { return SIMD2(35 + Double(frame - 6) * 5, 46 + Double(frame - 6) * 2) }
            return SIMD2(.nan, .nan)
        }
        let track = Track2D(t: timestamps, xy: xy, conf: Array(repeating: 1, count: frames),
                            visible: (0..<frames).map { (6...12).contains($0) },
                            interpolated: Array(repeating: false, count: frames))
        let hit = HitEvent(frame: 6, t: timestamps[6], xyPx: xy[6], hitter: .near)
        let segment = ShotSegment3D(startFrame: 6, endFrame: 12, t0: timestamps[6], p0: [0, 0, 2],
            v0: [10, 0, 0], rmsePx: 1, frames: [6, 7, 8, 9, 10, 11, 12],
            positions: Array(repeating: SIMD3<Double>(0, 0, 1), count: 7), landingXY: [2, 0],
            landingT: timestamps[12], confidence: 0.9)
        let shot = ShotEventV2(hit: hit, segment: segment, shotType: .smash)
        let rally = RallyV2(startFrame: 6, endFrame: 12, shots: [shot], endReason: .inCourt,
                            landingXY: [2, 0], winner: .near, confidence: 0.9)
        let score = JSONValue.object([
            "rally": .number(0), "start_frame": .number(6), "end_frame": .number(12), "game": .number(1),
            "score": .object(["near": .number(1), "far": .number(0)]),
        ])
        return MatchAnalysis(meta: VideoMeta(fps: fps, width: width, height: height,
            nFrames: frames, timestamps: timestamps), track: track, camera: .lookAt(
                eye: [0, -10, 7], target: [0, 0, 0], focalPx: 90, imageSize: ImageSize(width: width, height: height)),
            hits: [hit], segments: [segment], rallies: [rally], score: [score])
    }

    private func makeMovie(frameCount: Int, width: Int, height: Int, fps: Int) async throws -> URL {
        let directory = FileManager.default.temporaryDirectory
        let videoURL = directory.appendingPathComponent("shuttl-fixture-video-\(UUID()).mp4")
        let audioURL = directory.appendingPathComponent("shuttl-fixture-audio-\(UUID()).m4a")
        let outputURL = directory.appendingPathComponent("shuttl-fixture-\(UUID()).mp4")
        defer {
            try? FileManager.default.removeItem(at: videoURL)
            try? FileManager.default.removeItem(at: audioURL)
        }

        let videoWriter = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 400_000,
                AVVideoExpectedSourceFrameRateKey: fps, AVVideoMaxKeyFrameIntervalKey: fps],
        ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]])
        videoWriter.add(videoInput)
        guard videoWriter.startWriting() else { throw videoWriter.error ?? CocoaError(.fileWriteUnknown) }
        videoWriter.startSession(atSourceTime: .zero)
        for frame in 0..<frameCount {
            while !videoInput.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
            var pixel: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, adaptor.pixelBufferPool!, &pixel)
            guard status == kCVReturnSuccess, let buffer = pixel else { throw CocoaError(.fileWriteUnknown) }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, frame % 2 == 0 ? 34 : 45, CVPixelBufferGetBytesPerRow(buffer) * height)
                for offset in stride(from: 3, to: CVPixelBufferGetBytesPerRow(buffer) * height, by: 4) {
                    base.storeBytes(of: 255, toByteOffset: offset, as: UInt8.self)
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            guard adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(fps))) else {
                throw videoWriter.error ?? CocoaError(.fileWriteUnknown)
            }
        }
        videoInput.markAsFinished()
        await videoWriter.finishWriting()
        guard videoWriter.status == .completed else { throw videoWriter.error ?? CocoaError(.fileWriteUnknown) }

        let sampleRate = 48_000
        let audioWriter = try AVAssetWriter(outputURL: audioURL, fileType: .m4a)
        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 64_000,
        ])
        audioInput.expectsMediaDataInRealTime = false
        audioWriter.add(audioInput)
        guard audioWriter.startWriting() else { throw audioWriter.error ?? CocoaError(.fileWriteUnknown) }
        audioWriter.startSession(atSourceTime: .zero)
        let samples = Int64(Double(frameCount) / Double(fps) * Double(sampleRate))
        let pcm = try pcmSampleBuffer(sampleCount: samples, sampleRate: sampleRate)
        while !audioInput.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
        guard audioInput.append(pcm) else { throw audioWriter.error ?? CocoaError(.fileWriteUnknown) }
        audioInput.markAsFinished()
        await audioWriter.finishWriting()
        guard audioWriter.status == .completed else { throw audioWriter.error ?? CocoaError(.fileWriteUnknown) }

        let composition = AVMutableComposition()
        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)
        let videoTracks = try await videoAsset.loadTracks(withMediaType: .video)
        let audioTracks = try await audioAsset.loadTracks(withMediaType: .audio)
        let videoTrack = try XCTUnwrap(videoTracks.first)
        let audioTrack = try XCTUnwrap(audioTracks.first)
        let videoCompositionTrack = try XCTUnwrap(composition.addMutableTrack(withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid))
        let audioCompositionTrack = try XCTUnwrap(composition.addMutableTrack(withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid))
        let videoDuration = try await videoAsset.load(.duration)
        let audioDuration = try await audioAsset.load(.duration)
        try videoCompositionTrack.insertTimeRange(CMTimeRange(start: .zero, duration: videoDuration),
                                                   of: videoTrack, at: .zero)
        try audioCompositionTrack.insertTimeRange(CMTimeRange(start: .zero, duration: min(videoDuration, audioDuration)),
                                                   of: audioTrack, at: .zero)
        guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw CocoaError(.fileWriteUnknown)
        }
        exporter.outputURL = outputURL
        exporter.outputFileType = .mp4
        exporter.shouldOptimizeForNetworkUse = true
        await exporter.export()
        guard exporter.status == .completed else { throw exporter.error ?? CocoaError(.fileWriteUnknown) }
        return outputURL
    }

    private func pcmSampleBuffer(sampleCount: Int64, sampleRate: Int) throws -> CMSampleBuffer {
        let bytes = Int(sampleCount) * MemoryLayout<Int16>.size
        let memory = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: MemoryLayout<Int16>.alignment)
        memory.initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
        var block: CMBlockBuffer?
        let blockStatus = CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault,
            memoryBlock: memory, blockLength: bytes, blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil, offsetToData: 0, dataLength: bytes, flags: 0, blockBufferOut: &block)
        XCTAssertEqual(blockStatus, kCMBlockBufferNoErr)
        let asbd = AudioStreamBasicDescription(mSampleRate: Float64(sampleRate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2, mChannelsPerFrame: 1, mBitsPerChannel: 16,
            mReserved: 0)
        var format: CMAudioFormatDescription?
        let formatStatus = CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: [asbd],
            layoutSize: 0, layout: nil, magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        XCTAssertEqual(formatStatus, noErr)
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
            presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        var sampleSize = MemoryLayout<Int16>.size
        let sampleStatus = CMSampleBufferCreate(allocator: kCFAllocatorDefault, dataBuffer: block,
            dataReady: true, makeDataReadyCallback: nil, refcon: nil, formatDescription: format,
            sampleCount: Int(sampleCount), sampleTimingEntryCount: 1, sampleTimingArray: &timing,
            sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize, sampleBufferOut: &sample)
        XCTAssertEqual(sampleStatus, noErr)
        return try XCTUnwrap(sample)
    }

    private struct MovieInfo {
        let frames: Int
        let duration: Double
        let audioTracks: Int
        let yellowPixels: Int
        let emptyRegionYellowPixels: Int
        let firstOverlayFrame: CGImage
    }

    private func inspect(_ url: URL) async throws -> MovieInfo {
        let asset = AVURLAsset(url: url)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: try XCTUnwrap(videoTracks.first), outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(output)
        XCTAssertTrue(reader.startReading())
        var count = 0, yellow = 0, emptyYellow = 0
        var firstImage: CGImage?
        while let sample = output.copyNextSampleBuffer() {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            if count == 9 || (count == 3 && firstImage == nil) {
                firstImage = try pngImage(buffer)
                (yellow, emptyYellow) = countYellow(buffer)
            }
            count += 1
        }
        let duration = CMTimeGetSeconds(try await asset.load(.duration))
        return MovieInfo(frames: count, duration: duration, audioTracks: audioTracks.count,
                         yellowPixels: yellow, emptyRegionYellowPixels: emptyYellow,
                         firstOverlayFrame: try XCTUnwrap(firstImage))
    }

    private func countYellow(_ buffer: CVPixelBuffer) -> (Int, Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return (0, 0) }
        var total = 0, empty = 0
        for y in 0..<height { for x in 0..<width {
            let p = base.advanced(by: y * rowBytes + x * 4).assumingMemoryBound(to: UInt8.self)
            if p[2] > 170 && p[1] > 135 && p[0] < 150 {
                total += 1
                if x > 110 || y > 90 { empty += 1 }
            }
        } }
        return (total, empty)
    }

    private func pngImage(_ buffer: CVPixelBuffer) throws -> CGImage {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let context = try XCTUnwrap(CGContext(data: CVPixelBufferGetBaseAddress(buffer),
            width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue))
        return try XCTUnwrap(context.makeImage())
    }

    private func savePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}

private final class CancellationBox: @unchecked Sendable {
    var task: Task<URL, Error>?
    func cancel() { task?.cancel() }
}

