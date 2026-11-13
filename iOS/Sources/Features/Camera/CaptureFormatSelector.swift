import AVFoundation

/// Picks the capture format for match recording.
///
/// Why this exists: setting `activeFormat` overrides the session preset, and the
/// old code took the *first* format that could do 60 fps — on iPhones that is a
/// low-resolution (often binned) format listed before the 1080p ones, so the
/// "1080p" recording was really upscaled 640×480 / 1280×720 and looked blurred.
///
/// Rule: exactly 1920×1080, 420v, a frame-rate range that contains 60 fps; among
/// those prefer non-binned sensor readout (sharper), then the smallest max frame
/// rate (the plain 60 fps mode rather than a 120/240 fps slow-motion mode).
/// Fallback: the same at 30 fps. Nil means "keep the session preset".
enum CaptureFormatSelector {
    /// Plain-value mirror of the `AVCaptureDevice.Format` fields the rule looks at,
    /// so the rule can be unit-tested without a camera.
    struct Candidate: Equatable, Sendable {
        var width: Int32
        var height: Int32
        var pixelFormat: FourCharCode
        /// (min, max) fps of each supported range.
        var frameRateRanges: [ClosedRange<Double>]
        var isBinned: Bool
    }

    static let targetWidth: Int32 = 1920
    static let targetHeight: Int32 = 1080
    /// '420v' — kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, what the ML path consumes.
    static let pixelFormat: FourCharCode = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange

    /// Index into `candidates` of the chosen format and the fps to lock, or nil.
    static func select(from candidates: [Candidate], preferredFPS: [Double] = [60, 30]) -> (index: Int, fps: Double)? {
        for fps in preferredFPS {
            let matching = candidates.indices.filter { i in
                let c = candidates[i]
                return c.width == targetWidth && c.height == targetHeight
                    && c.pixelFormat == pixelFormat
                    && c.frameRateRanges.contains { $0.contains(fps) }
            }
            let best = matching.min { a, b in
                let ca = candidates[a], cb = candidates[b]
                if ca.isBinned != cb.isBinned { return !ca.isBinned }        // non-binned first
                return maxFPS(ca) < maxFPS(cb)                               // then the plain mode
            }
            if let best { return (best, fps) }
        }
        return nil
    }

    private static func maxFPS(_ c: Candidate) -> Double {
        c.frameRateRanges.map(\.upperBound).max() ?? 0
    }
}

extension CaptureFormatSelector.Candidate {
    init(_ format: AVCaptureDevice.Format) {
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        self.init(
            width: dims.width,
            height: dims.height,
            pixelFormat: CMFormatDescriptionGetMediaSubType(format.formatDescription),
            frameRateRanges: format.videoSupportedFrameRateRanges.map { $0.minFrameRate...$0.maxFrameRate },
            isBinned: format.isVideoBinned
        )
    }
}
