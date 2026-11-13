import Accelerate
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import ShuttlCore

public enum OverlayExporter {
    /// Exports a streamed MP4 with the analysis overlay. The returned URL points to a
    /// temporary file owned by the caller; cancellation removes any partial output.
    public static func export(
        video: URL,
        analysis: MatchAnalysis,
        options: ExportOptions = ExportOptions(),
        progress: @Sendable @escaping (Double) -> Void = { _ in }
    ) async throws -> URL {
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("shuttl-export-\(UUID().uuidString).mp4")
        let worker = Task.detached(priority: .userInitiated) {
            try await performExport(video: video, analysis: analysis, options: options,
                                    destination: destination, progress: progress)
        }
        return try await withTaskCancellationHandler {
            try await worker.value
        } onCancel: {
            worker.cancel()
        }
    }

    private static func performExport(
        video: URL,
        analysis: MatchAnalysis,
        options: ExportOptions,
        destination: URL,
        progress: @Sendable (Double) -> Void
    ) async throws -> URL {
        var completed = false
        defer { if !completed { removePartialOutput(destination) } }
        try Task.checkCancellation()

        let source = try await VideoSource.open(video)
        let bounds = try frameBounds(options.range, analysis: analysis, meta: source.meta)
        guard bounds.start < bounds.end else { throw OverlayExportError.noVideoFrames }

        let asset = AVURLAsset(url: video, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = videoTracks.first else { throw OverlayExportError.noVideoFrames }
        let transform = try await videoTrack.load(.preferredTransform)
        let sequence = try ExportFrameSequence(asset: asset, track: videoTrack,
                                               start: bounds.start, stop: bounds.end,
                                               rotation: exportRotationDegrees(transform))
        guard let firstFrame = sequence.next() else {
            throw OverlayExportError.noVideoFrames
        }
        let firstPTS = firstFrame.presentationTime
        let duration = frameRangeDuration(bounds, meta: source.meta)
        guard CMTimeCompare(duration, .zero) > 0 else { throw OverlayExportError.noVideoFrames }

        let outputSize = dimensions(for: source.meta, setting: options.outputSize)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let audioTrack = audioTracks.first
        let audioFormats = try await audioTrack?.load(.formatDescriptions) ?? []
        let audioFormat = audioFormats.first

        let writer: AVAssetWriter
        do { writer = try AVAssetWriter(outputURL: destination, fileType: .mp4) }
        catch { throw OverlayExportError.cannotCreateWriter(error.localizedDescription) }
        writer.shouldOptimizeForNetworkUse = true

        let fps = max(source.meta.fps, 1)
        let pixels = outputSize.width * outputSize.height
        let bitrate = min(10_000_000, max(2_500_000, Int(Double(pixels) * fps * 0.105)))
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: outputSize.width,
            AVVideoHeightKey: outputSize.height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoExpectedSourceFrameRateKey: Int(fps.rounded()),
                AVVideoMaxKeyFrameIntervalKey: max(1, Int(fps.rounded()) * 2),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoAllowFrameReorderingKey: true,
            ],
        ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outputSize.width,
            kCVPixelBufferHeightKey as String: outputSize.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ])
        guard writer.canAdd(videoInput) else {
            throw OverlayExportError.cannotCreateWriter("不支持视频编码设置")
        }
        writer.add(videoInput)

        var audioInput: AVAssetWriterInput?
        if audioTrack != nil, let audioFormat {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: audioFormat)
            input.expectsMediaDataInRealTime = false
            if writer.canAdd(input) {
                writer.add(input)
                audioInput = input
            }
        }

        guard writer.startWriting() else {
            throw OverlayExportError.writerFailed(writer.error?.localizedDescription ?? "startWriting")
        }
        writer.startSession(atSourceTime: .zero)

        var audioPump: AudioPump?
        if let audioTrack, let audioInput {
            let reader = try AVAssetReader(asset: asset)
            reader.timeRange = CMTimeRange(start: firstPTS, duration: duration)
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else {
                throw OverlayExportError.readerFailed("无法读取音轨")
            }
            reader.add(output)
            guard reader.startReading() else {
                throw OverlayExportError.readerFailed(reader.error?.localizedDescription ?? "audio startReading")
            }
            let pump = AudioPump(input: audioInput, reader: reader, output: output,
                                 sourceStart: firstPTS, sourceEnd: CMTimeAdd(firstPTS, duration),
                                 writer: writer)
            audioPump = pump
            pump.start()
        }

        let renderer = OverlayRenderer(analysis: analysis, options: options,
                                       width: outputSize.width, height: outputSize.height)
        let totalFrames = bounds.end - bounds.start
        var writtenFrames = 0
        var nextFrame: ExportFrame? = firstFrame
        do {
            while let frame = nextFrame {
                try Task.checkCancellation()
                guard writer.status == .writing else {
                    throw OverlayExportError.writerFailed(writer.error?.localizedDescription ?? "writer stopped")
                }
                let pts = CMTimeSubtract(frame.presentationTime, firstPTS)
                let outputBuffer = try makeOutputBuffer(frame.pixelBuffer, adaptor: adaptor,
                                                        width: outputSize.width, height: outputSize.height)
                renderer.draw(frame: frame.index, pixelBuffer: outputBuffer)
                try waitUntilReady(videoInput, writer: writer)
                guard adaptor.append(outputBuffer, withPresentationTime: pts) else {
                    throw OverlayExportError.writerFailed(writer.error?.localizedDescription ?? "append video frame")
                }
                writtenFrames += 1
                progress(Double(writtenFrames) / Double(max(totalFrames, 1)))
                nextFrame = sequence.next()
            }
            if let sequenceError = sequence.error {
                throw OverlayExportError.readerFailed(sequenceError.localizedDescription)
            }
            guard writtenFrames > 0 else { throw OverlayExportError.noVideoFrames }
            videoInput.markAsFinished()
            try await audioPump?.waitUntilFinished()
            writer.endSession(atSourceTime: duration)
            try await finish(writer)
            try Task.checkCancellation()
            completed = true
            progress(1)
            return destination
        } catch {
            audioPump?.cancel()
            writer.cancelWriting()
            throw error
        }
    }

    /// Removes a cancelled / failed export, including the safe-save sibling
    /// (`<name>.sb-XXXX`) AVAssetWriter can leave next to it after `cancelWriting()`.
    private static func removePartialOutput(_ destination: URL) {
        let fileManager = FileManager.default
        try? fileManager.removeItem(at: destination)
        let directory = destination.deletingLastPathComponent()
        let prefix = destination.lastPathComponent + ".sb-"
        let siblings = (try? fileManager.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in siblings where name.hasPrefix(prefix) {
            try? fileManager.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    private static func frameBounds(_ range: ExportOptions.Range, analysis: MatchAnalysis,
                                    meta: VideoMeta) throws -> (start: Int, end: Int) {
        func clamp(_ frame: Int) -> Int { min(max(frame, 0), meta.nFrames) }
        switch range {
        case .wholeVideo:
            return (0, meta.nFrames)
        case .rally(let index):
            guard analysis.rallies.indices.contains(index) else {
                throw OverlayExportError.invalidRange("回合序号 \(index) 不存在")
            }
            let rally = analysis.rallies[index]
            return (clamp(rally.startFrame), clamp(rally.endFrame + 1))
        case .timeRange(let start, let end):
            guard start.isFinite, end.isFinite, start >= 0, end > start, meta.nFrames > 0 else {
                throw OverlayExportError.invalidRange("起止时间不正确")
            }
            let first = nearestFrame(to: start, timestamps: meta.timestamps)
            let last = nearestFrame(to: end, timestamps: meta.timestamps)
            return (clamp(first), clamp(max(first + 1, last)))
        }
    }

    private static func nearestFrame(to time: Double, timestamps: [Double]) -> Int {
        guard !timestamps.isEmpty else { return 0 }
        var low = 0, high = timestamps.count - 1
        while low < high {
            let middle = (low + high) / 2
            if timestamps[middle] < time { low = middle + 1 } else { high = middle }
        }
        if low > 0, abs(timestamps[low - 1] - time) <= abs(timestamps[low] - time) { return low - 1 }
        return low
    }

    private static func dimensions(for meta: VideoMeta, setting: ExportOptions.OutputSize) -> (width: Int, height: Int) {
        let maxDimension: Int
        switch setting {
        case .source: maxDimension = max(meta.width, meta.height)
        case .p1080: maxDimension = 1920
        case .p720: maxDimension = 1280
        }
        let scale = min(1, Double(maxDimension) / Double(max(meta.width, meta.height, 1)))
        let width = max(2, Int((Double(meta.width) * scale / 2).rounded()) * 2)
        let height = max(2, Int((Double(meta.height) * scale / 2).rounded()) * 2)
        return (width, height)
    }

    private static func frameRangeDuration(_ bounds: (start: Int, end: Int), meta: VideoMeta) -> CMTime {
        let start = meta.time(ofFrame: bounds.start)
        let lastFrame = max(bounds.end - 1, bounds.start)
        let lastTime = meta.time(ofFrame: lastFrame)
        let previousTime = lastFrame > bounds.start ? meta.time(ofFrame: lastFrame - 1) : start
        let step = lastFrame > bounds.start ? max(lastTime - previousTime, 1 / max(meta.fps, 1)) : 1 / max(meta.fps, 1)
        let end = bounds.end < meta.nFrames ? meta.time(ofFrame: bounds.end) : lastTime + step
        return CMTime(seconds: max(end - start, step), preferredTimescale: 600)
    }

    private static func makeOutputBuffer(_ source: CVPixelBuffer, adaptor: AVAssetWriterInputPixelBufferAdaptor,
                                         width: Int, height: Int) throws -> CVPixelBuffer {
        guard let pool = adaptor.pixelBufferPool else {
            throw OverlayExportError.pixelBufferFailed("编码器没有提供像素缓冲池")
        }
        var output: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &output)
        guard status == kCVReturnSuccess, let output else {
            throw OverlayExportError.pixelBufferFailed("CVPixelBufferPoolCreatePixelBuffer: \(status)")
        }
        let sourceWidth = CVPixelBufferGetWidth(source), sourceHeight = CVPixelBufferGetHeight(source)
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        guard let src = CVPixelBufferGetBaseAddress(source), let dst = CVPixelBufferGetBaseAddress(output) else {
            throw OverlayExportError.pixelBufferFailed("像素缓冲区没有可用内存")
        }
        var srcBuffer = vImage_Buffer(data: src, height: vImagePixelCount(sourceHeight),
                                      width: vImagePixelCount(sourceWidth), rowBytes: CVPixelBufferGetBytesPerRow(source))
        var dstBuffer = vImage_Buffer(data: dst, height: vImagePixelCount(height),
                                      width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(output))
        let error = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        guard error == kvImageNoError else { throw OverlayExportError.pixelBufferFailed("vImageScale: \(error)") }
        return output
    }

    private static func appendAudio(_ sample: CMSampleBuffer, to input: AVAssetWriterInput?,
                                    offset: CMTime, writer: AVAssetWriter) throws {
        guard let input else { return }
        try Task.checkCancellation()
        try waitUntilReady(input, writer: writer)
        var count = 0
        CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: 0, arrayToFill: nil, entriesNeededOut: &count)
        var timing = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: count)
        CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: count, arrayToFill: &timing, entriesNeededOut: &count)
        for index in timing.indices {
            if timing[index].presentationTimeStamp.isValid {
                timing[index].presentationTimeStamp = CMTimeSubtract(timing[index].presentationTimeStamp, offset)
            }
            if timing[index].decodeTimeStamp.isValid {
                timing[index].decodeTimeStamp = CMTimeSubtract(timing[index].decodeTimeStamp, offset)
            }
        }
        var adjusted: CMSampleBuffer?
        let status = CMSampleBufferCreateCopyWithNewTiming(allocator: kCFAllocatorDefault, sampleBuffer: sample,
                                                           sampleTimingEntryCount: count, sampleTimingArray: &timing,
                                                           sampleBufferOut: &adjusted)
        guard status == noErr, let adjusted else {
            throw OverlayExportError.readerFailed("无法调整音频时间戳：\(status)")
        }
        guard input.append(adjusted) else {
            throw OverlayExportError.writerFailed(writer.error?.localizedDescription ?? "append audio")
        }
    }

    private static func waitUntilReady(_ input: AVAssetWriterInput, writer: AVAssetWriter) throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            guard writer.status == .writing else {
                throw OverlayExportError.writerFailed(writer.error?.localizedDescription ?? "writer stopped")
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.002))
        }
    }

    private static func finish(_ writer: AVAssetWriter) async throws {
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw OverlayExportError.writerFailed(writer.error?.localizedDescription ?? "finishWriting")
        }
    }
}
