import Accelerate
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

struct ExportFrame {
    let index: Int
    let pixelBuffer: CVPixelBuffer
    let presentationTime: CMTime
}

/// Ordinal frame reader for export. VideoSource's metadata probe is intentionally
/// compressed-only; some assets with edit lists report a different time origin for
/// that probe than their decompressed samples. Export needs the decoded sample order
/// and PTS together, so this reader keeps both from the same AVAssetReader pass.
final class ExportFrameSequence: Sequence, IteratorProtocol {
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private let start: Int
    private let stop: Int
    private let rotation: Int
    private var nextIndex = 0
    private var rotatedPool: CVPixelBufferPool?

    init(asset: AVAsset, track: AVAssetTrack, start: Int, stop: Int, rotation: Int) throws {
        self.start = Swift.max(0, start)
        self.stop = Swift.max(start, stop)
        self.rotation = rotation
        reader = try AVAssetReader(asset: asset)
        output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw OverlayExportError.readerFailed("无法读取视频画面") }
        reader.add(output)
        guard reader.startReading() else {
            throw OverlayExportError.readerFailed(reader.error?.localizedDescription ?? "video startReading")
        }
    }

    func next() -> ExportFrame? {
        guard nextIndex < stop else {
            if reader.status == .reading { reader.cancelReading() }
            return nil
        }
        while let sample = output.copyNextSampleBuffer() {
            let index = nextIndex
            nextIndex += 1
            guard CMSampleBufferGetNumSamples(sample) > 0,
                  let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            if index < start { continue }
            guard let upright = uprightBuffer(buffer) else { continue }
            return ExportFrame(index: index, pixelBuffer: upright,
                               presentationTime: CMSampleBufferGetPresentationTimeStamp(sample))
        }
        return nil
    }

    var error: Error? { reader.status == .failed ? reader.error : nil }

    private func uprightBuffer(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard rotation != 0 else { return buffer }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let outputWidth = rotation == 180 ? width : height
        let outputHeight = rotation == 180 ? height : width
        if rotatedPool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: outputWidth,
                kCVPixelBufferHeightKey as String: outputHeight,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &rotatedPool)
        }
        guard let pool = rotatedPool else { return nil }
        var outputBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outputBuffer) == kCVReturnSuccess,
              let outputBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(outputBuffer, [])
        }
        var source = vImage_Buffer(data: CVPixelBufferGetBaseAddress(buffer),
                                   height: vImagePixelCount(height), width: vImagePixelCount(width),
                                   rowBytes: CVPixelBufferGetBytesPerRow(buffer))
        var destination = vImage_Buffer(data: CVPixelBufferGetBaseAddress(outputBuffer),
                                        height: vImagePixelCount(outputHeight), width: vImagePixelCount(outputWidth),
                                        rowBytes: CVPixelBufferGetBytesPerRow(outputBuffer))
        var background: [UInt8] = [0, 0, 0, 255]
        let turns: UInt8 = switch rotation {
        case 90: UInt8(kRotate270DegreesCounterClockwise)
        case 180: UInt8(kRotate180DegreesCounterClockwise)
        default: UInt8(kRotate90DegreesCounterClockwise)
        }
        let status = vImageRotate90_ARGB8888(&source, &destination, turns, &background,
                                             vImage_Flags(kvImageNoFlags))
        return status == kvImageNoError ? outputBuffer : nil
    }

    deinit {
        if reader.status == .reading { reader.cancelReading() }
    }
}

func exportRotationDegrees(_ transform: CGAffineTransform) -> Int {
    let degrees = Int((atan2(Double(transform.b), Double(transform.a)) * 180 / .pi).rounded())
    return ((degrees % 360) + 360) % 360 / 90 * 90
}
