import Foundation
import CoreGraphics
import CoreMedia

// MARK: - ShuttleDetection

/// One heatmap peak mapped back to the camera frame.
struct ShuttleDetection: Sendable {
    /// Full-resolution pixel in the upright frame (pixel-centre convention), sub-pixel.
    var pixel: CGPoint
    /// Normalised position in the frame: x and y in [0, 1].
    var position: CGPoint
    /// Peak probability (0…1).
    var confidence: Float
    /// Index of the processed frame that produced this result.
    var frameIndex: Int
    /// Presentation timestamp of the source sample buffer (real, variable-rate timing).
    var timestamp: CMTime
    /// 0 = strongest peak of its frame.
    var rank: Int

    init(pixel: CGPoint, position: CGPoint, confidence: Float, frameIndex: Int,
         timestamp: CMTime = .invalid, rank: Int = 0) {
        self.pixel = pixel
        self.position = position
        self.confidence = confidence
        self.frameIndex = frameIndex
        self.timestamp = timestamp
        self.rank = rank
    }

    /// Normalised-only detection (tests, synthetic data): `pixel` = `position`.
    init(position: CGPoint, confidence: Float, frameIndex: Int, timestamp: CMTime = .invalid) {
        self.init(pixel: position, position: position, confidence: confidence,
                  frameIndex: frameIndex, timestamp: timestamp)
    }
}

// MARK: - FrameDetections

/// Everything the engine produced for one processed frame (possibly no detections).
struct FrameDetections: Sendable {
    var detections: [ShuttleDetection]
    var timestamp: CMTime
    var frameIndex: Int
    /// Upright full-resolution frame size the pixels refer to.
    var imageSize: CGSize
    /// Wall time spent in preprocessing + prediction + decoding, seconds.
    var latency: Double = 0

    /// Presentation time in seconds, nil when the buffer carried no valid timestamp.
    var seconds: Double? {
        guard timestamp.isValid, !timestamp.isIndefinite else { return nil }
        let s = CMTimeGetSeconds(timestamp)
        return s.isFinite ? s : nil
    }
}
