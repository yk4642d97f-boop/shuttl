import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ShuttlCore

/// Live "court locked" guide for the record screen.
///
/// Feed it the UPRIGHT frames of the capture video-data output (32BGRA or 420YpCbCr8BiPlanar
/// video/full range — what `CameraSession` delivers) from the capture queue. It samples ≤ 2 Hz,
/// keeps the last 8 frames downscaled to 0.75× (verified: same accept decision and ≤ 0.34 px
/// keypoints vs full resolution on 29 real clips), and runs ONE calibration attempt at a time on
/// a background queue (a full ShuttlCourt search first, then fast line re-fits to confirm and to
/// watch a lock); frames arriving while it is busy are only buffered, never queued. States are
/// published on `states()` streams (every subscriber gets the current state first).
///
/// ```swift
/// let lock = CourtLockEngine()
/// // capture queue:
/// lock.ingest(pixelBuffer: CMSampleBufferGetImageBuffer(sb)!, time: CMSampleBufferGetPresentationTimeStamp(sb))
/// // UI:
/// for await s in lock.states() { self.lockState = s }
/// // when recording starts:
/// options.cameraHint = lock.lockedCameraHint(recordingSize: ImageSize(width: 1920, height: 1080))
/// ```
public final class CourtLockEngine: @unchecked Sendable {
    public let configuration: CourtLockConfiguration
    private let calibrator: CourtLockCalibrator
    private let lock = NSLock()
    private var core: CourtLockCore
    private var busy = false
    private var converting = false
    private var paused = false
    private var continuations: [UUID: AsyncStream<CourtLockState>.Continuation] = [:]
    private var attemptTimes: [CourtLockPlan.Kind: [Double]] = [:]
    private let ingestQueue = DispatchQueue(label: "shuttl.courtlock.ingest", qos: .userInitiated)
    private let workQueue = DispatchQueue(label: "shuttl.courtlock.calibrate", qos: .userInitiated)
    private let idle = DispatchGroup()
    private let downsampler: FrameDownsampler     // ingestQueue only

    public init(configuration: CourtLockConfiguration = CourtLockConfiguration(),
                calibrator: CourtLockCalibrator = ShuttlCourtLockCalibrator()) {
        self.configuration = configuration
        self.calibrator = calibrator
        core = CourtLockCore(config: configuration)
        downsampler = FrameDownsampler()
    }

    // MARK: Output

    /// Current state (thread-safe).
    public var state: CourtLockState { lock.withLock { core.state } }

    /// A new stream of states; yields the current state immediately, then every change.
    public func states() -> AsyncStream<CourtLockState> {
        let id = UUID()
        return AsyncStream(bufferingPolicy: .bufferingNewest(8)) { cont in
            lock.withLock {
                continuations[id] = cont
                cont.yield(core.state)
            }
            cont.onTermination = { [weak self] _ in
                guard let self else { return }
                _ = self.lock.withLock { self.continuations.removeValue(forKey: id) }
            }
        }
    }

    /// The locked camera re-expressed at the recording resolution (same field of view, e.g. the
    /// 1920x1080 file when frames came from the same 1080p output), for `PipelineOptions.cameraHint`.
    /// nil when not locked or the aspect ratio differs (a crop, not a resize).
    public func lockedCameraHint(recordingSize: ImageSize? = nil) -> CameraModel? {
        guard let cam = lock.withLock({ core.lockedCamera }) else { return nil }
        guard let size = recordingSize else { return cam }
        return cam.rescaled(to: size)
    }

    public var stats: CourtLockStats { lock.withLock { core.stats } }

    /// Median wall-clock seconds per attempt kind so far (search / refine).
    public func medianAttemptSeconds(_ kind: CourtLockPlan.Kind) -> Double? {
        lock.withLock {
            guard let v = attemptTimes[kind], !v.isEmpty else { return nil }
            return v.sorted()[v.count / 2]
        }
    }

    // MARK: Control

    /// Stop sampling (e.g. while recording, so the analysis gets the CPU); the state is kept.
    public func pause() { lock.withLock { paused = true } }
    public func resume() { lock.withLock { paused = false } }

    /// Forget everything (new session / camera switched) and publish `.searching`.
    public func reset() {
        lock.withLock {
            core.reset(keepStats: false)
            publishLocked()
        }
    }

    // MARK: Input

    /// Capture-queue entry point. Cheap when the frame is not sampled (a lock + a compare).
    public func ingest(pixelBuffer: CVPixelBuffer, time: CMTime) {
        let t = CMTimeGetSeconds(time)
        guard t.isFinite, reserve(t) else { return }
        let box = PixelBufferBox(pixelBuffer)
        idle.enter()
        ingestQueue.async { [self] in
            defer { idle.leave() }
            let full = ImageSize(width: CVPixelBufferGetWidth(box.buffer), height: CVPixelBufferGetHeight(box.buffer))
            let frame = try? downsampler.frame(from: box.buffer, workingSize: configuration.workingSize(for: full), time: t)
            lock.withLock { converting = false }
            if let frame { accept(frame) }
        }
    }

    /// Test / offline entry point: an upright RGB frame at full resolution.
    public func ingest(image: ImageU8, time: Double) {
        guard reserve(time) else { return }
        idle.enter()
        ingestQueue.async { [self] in
            defer { idle.leave() }
            let frame = downsampler.frame(from: image, workingSize: configuration.workingSize(for: image.size), time: time)
            lock.withLock { converting = false }
            accept(frame)
        }
    }

    /// Blocks until no conversion or attempt is running (tests).
    public func waitUntilIdle() { idle.wait() }

    // MARK: Internals

    private func reserve(_ t: Double) -> Bool {
        lock.withLock {
            // Drop while paused, while the previous sample is still being converted, or too soon.
            guard !paused, !converting, core.reserveSample(at: t) else { return false }
            converting = true
            return true
        }
    }

    private func accept(_ frame: WorkingFrame) {
        lock.withLock {
            if core.offer(frame) { publishLocked() }
        }
        startAttemptIfIdle()
    }

    private func startAttemptIfIdle() {
        let plan: CourtLockPlan? = lock.withLock {
            guard !busy, let p = core.plan() else { return nil }
            busy = true
            return p
        }
        guard let plan else { return }
        idle.enter()
        workQueue.async { [self] in
            defer { idle.leave() }
            let t0 = DispatchTime.now().uptimeNanoseconds
            let attempt = plan.run(calibrator)
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            lock.withLock {
                attemptTimes[plan.kind, default: []].append(dt)
                if core.apply(plan, attempt) { publishLocked() }
                busy = false
            }
            startAttemptIfIdle()                     // a newer frame may be waiting
        }
    }

    /// Caller holds `lock`.
    private func publishLocked() {
        let s = core.state
        for c in continuations.values { c.yield(s) }
    }
}

/// CVPixelBuffer is a thread-safe CF object; the box only carries it to the ingest queue.
private struct PixelBufferBox: @unchecked Sendable {
    let buffer: CVPixelBuffer
    init(_ b: CVPixelBuffer) { buffer = b }
}

public extension CourtLockConfiguration {
    static func workingSize(for full: ImageSize, scale: Double) -> ImageSize {
        var c = CourtLockConfiguration()
        c.workingScale = scale
        return c.workingSize(for: full)
    }
}
