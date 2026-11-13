import AVFoundation
import CoreMedia
import Foundation

/// Feeds compressed audio samples concurrently with the transformed video input.
/// AVAssetWriter applies back pressure per track, so audio cannot be drained from
/// the video loop without risking a multi-track deadlock.
final class AudioPump: @unchecked Sendable {
    private let input: AVAssetWriterInput
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private let sourceStart: CMTime
    private let sourceEnd: CMTime
    private let writer: AVAssetWriter
    private let queue = DispatchQueue(label: "com.shuttl.export.audio", qos: .userInitiated)
    private let stateQueue = DispatchQueue(label: "com.shuttl.export.audio.state")
    private var finished = false
    private var cancelled = false
    private var failure: String?

    init(input: AVAssetWriterInput, reader: AVAssetReader, output: AVAssetReaderTrackOutput,
         sourceStart: CMTime, sourceEnd: CMTime, writer: AVAssetWriter) {
        self.input = input
        self.reader = reader
        self.output = output
        self.sourceStart = sourceStart
        self.sourceEnd = sourceEnd
        self.writer = writer
    }

    func start() {
        input.requestMediaDataWhenReady(on: queue) { [weak self] in
            self?.pump()
        }
    }

    func cancel() {
        stateQueue.sync { cancelled = true }
        reader.cancelReading()
        input.markAsFinished()
    }

    func waitUntilFinished() async throws {
        while true {
            try Task.checkCancellation()
            let state = stateQueue.sync { (finished, failure) }
            if state.0 {
                if let failure = state.1 { throw OverlayExportError.readerFailed(failure) }
                return
            }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
    }

    private func pump() {
        while input.isReadyForMoreMediaData {
            let shouldStop = stateQueue.sync { cancelled }
            if shouldStop {
                finish()
                return
            }
            guard writer.status == .writing else {
                fail(writer.error?.localizedDescription ?? "writer stopped while writing audio")
                return
            }
            guard let sample = output.copyNextSampleBuffer() else {
                if reader.status == .failed {
                    fail(reader.error?.localizedDescription ?? "audio reader failed")
                } else {
                    finish()
                }
                return
            }
            do {
                if let adjusted = try adjustedSample(sample) {
                    guard input.append(adjusted) else {
                        fail(writer.error?.localizedDescription ?? "append audio")
                        return
                    }
                }
            } catch {
                fail(error.localizedDescription)
                return
            }
        }
    }

    private func adjustedSample(_ sample: CMSampleBuffer) throws -> CMSampleBuffer? {
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        let duration = CMSampleBufferGetDuration(sample)
        if pts.isValid, CMTimeCompare(pts, sourceEnd) >= 0 { return nil }
        if pts.isValid, duration.isValid,
           CMTimeCompare(CMTimeAdd(pts, duration), sourceStart) <= 0 { return nil }

        var count = 0
        CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: 0, arrayToFill: nil, entriesNeededOut: &count)
        guard count > 0 else { return sample }
        var timing = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: count)
        CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: count, arrayToFill: &timing, entriesNeededOut: &count)
        for index in timing.indices {
            if timing[index].presentationTimeStamp.isValid {
                timing[index].presentationTimeStamp = CMTimeSubtract(timing[index].presentationTimeStamp, sourceStart)
                if CMTimeCompare(timing[index].presentationTimeStamp, .zero) < 0 {
                    timing[index].presentationTimeStamp = .zero
                }
            }
            if timing[index].decodeTimeStamp.isValid {
                timing[index].decodeTimeStamp = CMTimeSubtract(timing[index].decodeTimeStamp, sourceStart)
                if CMTimeCompare(timing[index].decodeTimeStamp, .zero) < 0 {
                    timing[index].decodeTimeStamp = .invalid
                }
            }
        }
        var adjusted: CMSampleBuffer?
        let status = CMSampleBufferCreateCopyWithNewTiming(allocator: kCFAllocatorDefault,
            sampleBuffer: sample, sampleTimingEntryCount: count, sampleTimingArray: &timing,
            sampleBufferOut: &adjusted)
        guard status == noErr else { throw OverlayExportError.readerFailed("音频时间戳调整失败：\(status)") }
        return adjusted
    }

    private func fail(_ message: String) {
        stateQueue.sync {
            if !finished {
                failure = message
                finished = true
            }
        }
        reader.cancelReading()
        input.markAsFinished()
    }

    private func finish() {
        let shouldFinish = stateQueue.sync { () -> Bool in
            let shouldFinish = !finished
            finished = true
            return shouldFinish
        }
        if shouldFinish { input.markAsFinished() }
    }
}
