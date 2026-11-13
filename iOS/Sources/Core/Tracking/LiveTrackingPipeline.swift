import Foundation
import CoreGraphics
import CoreMedia

// MARK: - LiveTrackingPipeline
// Camera frames → MLInferenceEngine → TrajectoryTracker → TrackerSnapshot, off the main actor.
// The tracker is not thread-safe; this actor owns it. Frames arrive through `FrameFeed`, which
// keeps only the newest buffers when inference falls behind (drop, never queue).

actor LiveTrackingPipeline {
    private let engine: MLInferenceEngine
    private let tracker: TrajectoryTracker

    init(engine: MLInferenceEngine = .shared, config: TrackerConfig = TrackerConfig()) {
        self.engine = engine
        tracker = TrajectoryTracker(config: config)
    }

    /// Loads the model ahead of the first frame. Returns whether live detection is available.
    func prepare() async -> Bool {
        await engine.load()
    }

    func setCalibration(_ calibration: CourtCalibration?) {
        tracker.calibration = calibration
    }

    func reset() async {
        tracker.reset()
        await engine.reset()
    }

    /// Nil when this frame produced no new tracker state (window filling, thinned, no model).
    func process(_ frame: CameraFrame) async -> TrackerSnapshot? {
        let results = await engine.process(frame)
        var snapshot: TrackerSnapshot?
        for r in results { snapshot = tracker.update(r) }
        return snapshot
    }

    /// Detections from any source (replay, tests).
    func process(_ frame: FrameDetections) -> TrackerSnapshot {
        tracker.update(frame)
    }
}

// MARK: - FrameFeed
// Bridges the capture queue to the pipeline: `send` is called on the camera queue and returns
// immediately; one consumer task processes frames in order. The stream holds at most
// `bufferLimit` frames (newest win), so camera pool buffers are never hoarded.

final class FrameFeed: Sendable {
    private let continuation: AsyncStream<CameraFrame>.Continuation
    private let task: Task<Void, Never>

    init(pipeline: LiveTrackingPipeline, bufferLimit: Int = 2,
         onSnapshot: @escaping @Sendable (TrackerSnapshot) async -> Void) {
        let (stream, continuation) = AsyncStream.makeStream(of: CameraFrame.self,
                                                             bufferingPolicy: .bufferingNewest(bufferLimit))
        self.continuation = continuation
        task = Task.detached(priority: .userInitiated) {
            _ = await pipeline.prepare()
            for await frame in stream {
                if let snap = await pipeline.process(frame) { await onSnapshot(snap) }
            }
        }
    }

    func send(_ frame: CameraFrame) {
        continuation.yield(frame)
    }

    func send(_ sampleBuffer: CMSampleBuffer) {
        if let frame = CameraFrame(sampleBuffer: sampleBuffer) { continuation.yield(frame) }
    }

    func stop() {
        continuation.finish()
        task.cancel()
    }

    deinit { stop() }
}
