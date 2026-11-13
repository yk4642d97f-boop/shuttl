import Foundation
import CoreImage
import CoreMedia
import CoreVideo
import ImageIO
import Accelerate

// MARK: - CameraFrame
// A captured frame handed from the capture queue to the inference actor. CVPixelBuffer is a
// thread-safe CF type and capture never writes to a delivered buffer; we only read it, once.

struct CameraFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    /// Presentation timestamp from the sample buffer (real, variable-rate timing).
    let timestamp: CMTime
    /// Rotation that makes the buffer upright (`.up` when the connection already rotates).
    var orientation: CGImagePropertyOrientation = .up

    init(pixelBuffer: CVPixelBuffer, timestamp: CMTime, orientation: CGImagePropertyOrientation = .up) {
        self.pixelBuffer = pixelBuffer
        self.timestamp = timestamp
        self.orientation = orientation
    }

    init?(sampleBuffer: CMSampleBuffer, orientation: CGImagePropertyOrientation = .up) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        self.init(pixelBuffer: pb, timestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
                  orientation: orientation)
    }

    var seconds: Double? {
        guard timestamp.isValid, !timestamp.isIndefinite else { return nil }
        let s = CMTimeGetSeconds(timestamp)
        return s.isFinite ? s : nil
    }

    /// Upright size in pixels.
    var uprightSize: (width: Int, height: Int) {
        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        switch orientation {
        case .left, .right, .leftMirrored, .rightMirrored: return (h, w)
        default: return (w, h)
        }
    }
}

// MARK: - FloatStorage
// Fixed-size float scratch owned by a reference (freed with its owner; no per-frame allocation).

final class FloatStorage: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
    let count: Int

    init(count: Int) {
        self.count = count
        pointer = .allocate(capacity: count)
        pointer.initialize(repeating: 0, count: count)
    }

    deinit { pointer.deallocate() }
}

// MARK: - FramePreprocessor
// Camera buffer (420f / 420v / BGRA, any size) → network canvas as planar float RGB, 0…1.
// One GPU pass in Core Image does YCbCr→RGB, orientation, the index-affine resize of
// FrameGeometry (bilinear, like WASB's cv2.warpAffine) and the black letterbox; vImage then
// de-interleaves BGRA8 into three float planes written straight into the caller's tensor
// memory. No per-pixel Swift loops, no per-frame allocations (one reused canvas buffer).
// Colour management is off: the network sees the same gamma-encoded RGB a video decoder
// produces, as in training.

final class FramePreprocessor {
    let netWidth: Int
    let netHeight: Int
    private let context: CIContext
    private let canvas: CVPixelBuffer
    private let alpha: UnsafeMutablePointer<Float>
    private let black: CIImage

    init?(netWidth: Int, netHeight: Int) {
        self.netWidth = netWidth
        self.netHeight = netHeight
        context = CIContext(options: [
            .workingColorSpace: NSNull(),
            .outputColorSpace: NSNull(),
            .cacheIntermediates: false,
        ])
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                     kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(nil, netWidth, netHeight, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        canvas = pb
        alpha = .allocate(capacity: netWidth * netHeight)
        black = CIImage(color: CIColor(red: 0, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: netWidth, height: netHeight))
    }

    deinit { alpha.deallocate() }

    var planeCount: Int { netWidth * netHeight }

    /// Writes 3 contiguous planes (`planeCount` floats each) at `dst`: R, G, B — or B, G, R when
    /// `bgr`. Returns false if rendering failed.
    func render(_ frame: CameraFrame, geometry g: FrameGeometry, into dst: UnsafeMutablePointer<Float>,
                bgr: Bool = false, imagenet: Bool = false) -> Bool {
        var image = CIImage(cvPixelBuffer: frame.pixelBuffer)
        if frame.orientation != .up { image = image.oriented(frame.orientation) }
        image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        return render(image, geometry: g, into: dst, bgr: bgr, imagenet: imagenet)
    }

    /// Same for an upright CIImage whose extent starts at the origin (tests).
    func render(_ image: CIImage, geometry g: FrameGeometry, into dst: UnsafeMutablePointer<Float>,
                bgr: Bool = false, imagenet: Bool = false) -> Bool {
        // Top-left net coords (pixel centres at +½): v = s·u + T. Core Image is bottom-left.
        let (tx, tyTop) = g.centreSampledTranslation
        let ty = Double(netHeight) - g.scaleY * Double(g.height) - tyTop
        let affine = CGAffineTransform(a: g.scaleX, b: 0, c: 0, d: g.scaleY, tx: tx, ty: ty)
        // Content rectangle in CI coordinates; outside it the canvas stays black (letterbox).
        let content = CGRect(x: g.padX, y: Double(netHeight) - g.padY - g.scaleY * Double(g.height),
                             width: g.scaleX * Double(g.width), height: g.scaleY * Double(g.height))
            .intersection(black.extent)
        let placed = image.clampedToExtent().transformed(by: affine).cropped(to: content).composited(over: black)
        context.render(placed, to: canvas, bounds: black.extent, colorSpace: nil)

        CVPixelBufferLockBaseAddress(canvas, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(canvas, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(canvas) else { return false }
        let n = planeCount
        let rowBytes = netWidth * MemoryLayout<Float>.stride
        var src = vImage_Buffer(data: base, height: vImagePixelCount(netHeight), width: vImagePixelCount(netWidth),
                                rowBytes: CVPixelBufferGetBytesPerRow(canvas))
        func plane(_ p: UnsafeMutablePointer<Float>) -> vImage_Buffer {
            vImage_Buffer(data: p, height: vImagePixelCount(netHeight), width: vImagePixelCount(netWidth), rowBytes: rowBytes)
        }
        // Memory order is B, G, R, A.
        var b = plane(bgr ? dst : dst + 2 * n)
        var gr = plane(dst + n)
        var r = plane(bgr ? dst + 2 * n : dst)
        var a = plane(alpha)
        let maxF: [Float] = [1, 1, 1, 1], minF: [Float] = [0, 0, 0, 0]
        let err = vImageConvert_ARGB8888toPlanarF(&src, &b, &gr, &r, &a, maxF, minF, vImage_Flags(kvImageNoFlags))
        guard err == kvImageNoError else { return false }
        if imagenet {
            // ImageNet statistics in R, G, B order.
            let mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]
            for c in 0..<3 {
                let rgb = bgr ? 2 - c : c
                var scale = 1 / std[rgb], offset = -mean[rgb] / std[rgb]
                vDSP_vsmsa(dst + c * n, 1, &scale, &offset, dst + c * n, 1, vDSP_Length(n))
            }
        }
        return true
    }
}

// MARK: - ApproximateMedianBackground
// Running per-pixel median approximation for models with uses_background = 1 (TrackNetV3):
// bg += clamp(frame − bg, ±step) — the sigma-delta / approximate-median filter. Moving objects
// barely move it; lighting changes are followed within a few seconds. vDSP only.

struct ApproximateMedianBackground {
    private(set) var values: [Float] = []
    private var scratch: [Float] = []
    var step: Float = 2.0 / 255
    var isEmpty: Bool { values.isEmpty }

    mutating func reset() {
        values.removeAll(keepingCapacity: true)
    }

    mutating func update(with frame: UnsafePointer<Float>, count n: Int) {
        if values.count != n {
            values = Array(UnsafeBufferPointer(start: frame, count: n))
            scratch = [Float](repeating: 0, count: n)
            return
        }
        var lo = -step, hi = step
        values.withUnsafeMutableBufferPointer { bg in
            scratch.withUnsafeMutableBufferPointer { d in
                guard let bgp = bg.baseAddress, let dp = d.baseAddress else { return }
                vDSP_vsub(bgp, 1, frame, 1, dp, 1, vDSP_Length(n))       // d = frame − bg
                vDSP_vclip(dp, 1, &lo, &hi, dp, 1, vDSP_Length(n))
                vDSP_vadd(bgp, 1, dp, 1, bgp, 1, vDSP_Length(n))
            }
        }
    }
}
