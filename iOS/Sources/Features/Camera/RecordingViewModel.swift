import Foundation
import SwiftData
import Combine
import AVFoundation
import CoreImage
import SwiftUI
import ShuttlCore
import ShuttlPresent
import ShuttlRuntime

@MainActor
@Observable
final class RecordingViewModel {
    private(set) var isRecording = false
    /// Live shuttle path, oldest → newest, normalised 0…1 of the upright camera frame.
    private(set) var recentTrajectoryPoints: [CGPoint] = []
    /// Upright camera frame size the points refer to (0 until the first processed frame).
    private(set) var trajectoryFrameSize: CGSize = .zero
    /// Approximate speed of the last completed flight — only with a court calibration and a
    /// tight monocular fit; nil otherwise (never shown as a made-up number).
    private(set) var approxSpeedKmh: Double?

    /// Feedback after pressing stop.
    enum SaveNotice: Equatable {
        case finishing                  // AVFoundation is still writing the file
        case saved                      // file complete, match created, upload started
        case failed(String)
    }
    private(set) var saveNotice: SaveNotice?

    let cameraSession = CameraSession()
    /// Injected by the view from the environment (app-lifetime, outlives this screen).
    var matchPipeline: MatchPipeline?
    /// Court type chosen on screen; captured when a take starts.
    var courtType: CourtType = .singles
    /// Front camera in use: the preview is mirrored, so the overlay must be too.
    private(set) var isFrontCamera = false
    private(set) var isSwitchingCamera = false
    /// Hardware has both lenses (false in the simulator).
    let hasBothCameras = CameraSession.canSwitchCameras
    private var isOnScreen = false

    /// The flip button is only live when idle.
    var canSwitchCamera: Bool {
        #if DEBUG
        if DemoMode.isOn { return !isRecording }
        #endif
        return hasBothCameras && !isRecording && !isSwitchingCamera
    }

    /// Live setup guide: court lines found in the preview frames (never gates recording).
    private(set) var courtLock: CourtLockDisplay = .searching
    private let courtLockEngine = CourtLockEngine()
    private var courtLockTask: Task<Void, Never>?

    private let pipeline = LiveTrackingPipeline()
    private var feed: FrameFeed?
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Camera power
    // The preview, live tracking and CourtLock heat the phone (on a charger it reached
    // "serious" and held every analysis back for minutes). So the camera runs only while
    // this screen is visible, the app is active and the camera is awake. It sleeps after
    // `idleTimeout` without a take, and at once while an analysis is running or waiting:
    // the live camera also blocks analysis decode (admission reason "camera").

    /// Why the camera is asleep; nil while it's awake (or the screen isn't showing).
    enum Sleep: Equatable {
        /// No take for `idleTimeout`.
        case idle
        /// An analysis is running or waiting: the camera would heat the phone and hold it back.
        case analysisPending
    }
    private(set) var sleep: Sleep?
    /// The last preview frame, shown desaturated behind the sleep card (nil until one arrived).
    private(set) var sleepStill: CGImage?

    static let idleTimeout: Duration = .seconds(90)
    private var isSceneActive = true
    private var idleTask: Task<Void, Never>?
    private var cameraRunning = false
    private let lastFrame = LastFrameBox()

    /// The screen appeared (or came back from another tab).
    func onAppear() async {
        isOnScreen = true
        #if DEBUG
        if DemoMode.showsSaveNotice { saveNotice = .saved }
        if let lock = DemoMode.courtLock { courtLock = lock; courtLockTask = Task {} }
        if DemoMode.fakesRecording { isRecording = true; return }
        if let sleep = DemoMode.cameraSleep { self.sleep = sleep; return }
        #endif
        refreshStorage()
        await cameraSession.requestPermissionAndConfigure()
        if matchPipeline?.hasLocalWork == true { goToSleep(.analysisPending) } else { wake() }
    }

    func onDisappear() {
        isOnScreen = false
        if isRecording { stopRecording() }
        stopCamera()
        sleep = nil
    }

    /// scenePhase: background / inactive stops the camera; coming back resumes it (or keeps it asleep).
    func sceneBecame(active: Bool) {
        guard active != isSceneActive else { return }
        isSceneActive = active
        guard isOnScreen else { return }
        if !active {
            if !isRecording { stopCamera() }
        } else if sleep == nil {
            if matchPipeline?.hasLocalWork == true { goToSleep(.analysisPending) } else { wake() }
        }
    }

    /// Tap on the sleep card, or the phone raised to landscape: the camera comes back now.
    /// Recording keeps its priority over analysis (a take pauses it with checkpoints).
    func wake() {
        #if DEBUG
        if DemoMode.cameraSleep != nil { sleep = nil; return }
        #endif
        sleep = nil
        guard isOnScreen, isSceneActive else { return }
        startCamera()
        armIdleTimer()
    }

    /// Any touch on the live screen restarts the idle clock.
    func noteActivity() {
        guard sleep == nil, !isRecording else { return }
        armIdleTimer()
    }

    private func goToSleep(_ reason: Sleep) {
        guard !isRecording else { return }
        idleTask?.cancel()
        idleTask = nil
        if let still = lastFrame.take() { sleepStill = still }
        stopCamera()
        withAnimation(AppAnimation.spring) { sleep = reason }
    }

    private func armIdleTimer() {
        idleTask?.cancel()
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: Self.idleTimeout)
            guard !Task.isCancelled, let self, !self.isRecording, self.sleep == nil else { return }
            self.goToSleep(self.matchPipeline?.hasLocalWork == true ? .analysisPending : .idle)
        }
    }

    private func startCamera() {
        guard !cameraRunning else { return }
        cameraRunning = true
        // Auto-lock mid-match would background the app and cut the recording.
        IdleTimerGuard.acquire("camera")
        cameraSession.startRunning()
        startLiveTracking()
        matchPipeline?.cameraDidStart()
    }

    private func stopCamera() {
        idleTask?.cancel()
        idleTask = nil
        guard cameraRunning else { return }
        cameraRunning = false
        IdleTimerGuard.release("camera")
        cameraSession.stopRunning()
        stopLiveTracking()
        matchPipeline?.cameraDidStop()
    }

    /// Keeps a downscaled copy of every 30th preview frame (capture queue), for the sleep card.
    final class LastFrameBox: @unchecked Sendable {
        private let lock = NSLock()
        private var buffer: CVPixelBuffer?
        private var frames = 0
        func update(from sample: CMSampleBuffer) {
            let due = lock.withLock { () -> Bool in frames += 1; return frames % 30 == 1 }
            guard due, let pixels = CMSampleBufferGetImageBuffer(sample) else { return }
            lock.withLock { buffer = pixels }
        }
        /// A desaturated still of the newest kept frame (main actor, once, when going to sleep).
        func take() -> CGImage? {
            guard let pixels = lock.withLock({ buffer }) else { return nil }
            lock.withLock { buffer = nil }
            let image = CIImage(cvPixelBuffer: pixels)
                .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.15, kCIInputContrastKey: 0.85])
            return CIContext(options: [.cacheIntermediates: false]).createCGImage(image, from: image.extent)
        }
    }

    func toggleRecording() {
        if isRecording { stopRecording() } else { startRecording() }
    }

    func dismissNotice() { saveNotice = nil }

    /// Front ⇄ back. The session is reconfigured on its queue; afterwards the live
    /// tracker starts from scratch (its state and any court calibration belonged to the other lens).
    func toggleCamera() {
        guard canSwitchCamera else { return }
        #if DEBUG
        if DemoMode.isOn { isFrontCamera.toggle(); return }
        #endif
        isSwitchingCamera = true
        cameraSession.switchCamera { [weak self] position in
            guard let self else { return }
            self.isFrontCamera = position == .front
            self.isSwitchingCamera = false
            self.restartLiveTracking()
        }
    }

    /// Court calibration (e.g. from four tapped corners) — enables the approximate speed.
    func setCourtCalibration(_ calibration: CourtCalibration?) {
        let pipeline = pipeline
        Task { await pipeline.setCalibration(calibration) }
    }

    /// Trail in the coordinates of a view showing the camera with `.resizeAspectFill`.
    func overlayPoints(in viewSize: CGSize) -> [CGPoint] {
        Self.aspectFill(recentTrajectoryPoints, imageSize: trajectoryFrameSize, into: viewSize, mirrored: isFrontCamera)
    }

    var durationText: String {
        let d = cameraSession.recordingDuration
        let minutes = Int(d) / 60
        let seconds = Int(d) % 60
        let tenths = Int((d - Double(Int(d))) * 10)
        return String(format: "%02d:%02d.%d", minutes, seconds, tenths)
    }

    /// - Parameter mirrored: the preview is mirrored (front camera) while ML frames are not,
    ///   so x is flipped in normalised space before mapping.
    nonisolated static func aspectFill(
        _ points: [CGPoint], imageSize: CGSize, into view: CGSize, mirrored: Bool = false
    ) -> [CGPoint] {
        let points = mirrored ? points.map { CGPoint(x: 1 - $0.x, y: $0.y) } : points
        guard imageSize.width > 0, imageSize.height > 0 else {
            return points.map { CGPoint(x: $0.x * view.width, y: $0.y * view.height) }
        }
        let s = max(view.width / imageSize.width, view.height / imageSize.height)
        let ox = (view.width - imageSize.width * s) / 2, oy = (view.height - imageSize.height * s) / 2
        return points.map { CGPoint(x: $0.x * imageSize.width * s + ox, y: $0.y * imageSize.height * s + oy) }
    }

    // MARK: - Live tracking

    private func startLiveTracking() {
        guard feed == nil else { return }
        let feed = FrameFeed(pipeline: pipeline) { [weak self] snapshot in
            await self?.apply(snapshot)
        }
        self.feed = feed
        // The subject fires on the capture queue; the sink only hands the buffer over.
        cameraSession.framePublisher
            .sink(receiveValue: Self.frameSink(feed))
            .store(in: &cancellables)
        startCourtLock()
    }

    private func startCourtLock() {
        guard courtLockTask == nil else { return }
        cameraSession.framePublisher
            .sink(receiveValue: Self.courtLockSink(courtLockEngine, focal: focal, last: lastFrame))
            .store(in: &cancellables)
        let engine = courtLockEngine
        courtLockTask = Task { [weak self] in
            for await state in engine.states() {
                guard let self else { return }
                let display = CourtLockDisplay(state, mirrored: self.isFrontCamera)
                if display != self.courtLock { self.courtLock = display }
            }
        }
    }

    private func stopCourtLock() {
        courtLockTask?.cancel()
        courtLockTask = nil
        courtLockEngine.reset()
        courtLock = .searching
    }

    /// Runs on the capture queue; the engine samples at 2 Hz and works on its own queues.
    private nonisolated static func courtLockSink(_ engine: CourtLockEngine, focal: FocalBox,
                                                  last: LastFrameBox) -> @Sendable (CMSampleBuffer) -> Void {
        { sample in
            focal.update(from: sample)
            last.update(from: sample)
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return }
            engine.ingest(pixelBuffer: buffer, time: CMSampleBufferGetPresentationTimeStamp(sample))
        }
    }

    /// The latest lens focal length (px at 1080p width) from the capture intrinsics.
    final class FocalBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Double?
        private var frames = 0
        var current: Double? { lock.withLock { value } }
        /// Every 30th frame: the value only changes when the lens refocuses.
        func update(from sample: CMSampleBuffer) {
            let due = lock.withLock { () -> Bool in frames += 1; return frames % 30 == 1 }
            guard due, let f = MatchInputs.focalPx(from: sample, videoLongSide: 1920) else { return }
            lock.withLock { value = f }
        }
    }
    private let focal = FocalBox()

    private func stopLiveTracking() {
        tearDownFeed()
        let pipeline = pipeline
        Task { await pipeline.reset() }
    }

    private func tearDownFeed() {
        cancellables.removeAll()
        stopCourtLock()
        feed?.stop()
        feed = nil
        recentTrajectoryPoints = []
        trajectoryFrameSize = .zero
        approxSpeedKmh = nil
    }

    /// After a lens switch: drop in-flight frames, reset tracker + calibration, then resume.
    private func restartLiveTracking() {
        tearDownFeed()
        let pipeline = pipeline
        Task { [weak self] in
            await pipeline.setCalibration(nil)
            await pipeline.reset()
            guard let self, self.isOnScreen else { return }
            self.startLiveTracking()
        }
    }

    private nonisolated static func frameSink(_ feed: FrameFeed) -> @Sendable (CMSampleBuffer) -> Void {
        { feed.send($0) }
    }

    private func apply(_ snapshot: TrackerSnapshot) {
        if recentTrajectoryPoints != snapshot.trail { recentTrajectoryPoints = snapshot.trail }
        if trajectoryFrameSize != snapshot.imageSize { trajectoryFrameSize = snapshot.imageSize }
        if approxSpeedKmh != snapshot.approxSpeedKmh { approxSpeedKmh = snapshot.approxSpeedKmh }
    }

    // MARK: - Private

    private func startRecording() {
        refreshStorage()
        if let storage, !storage.canStartRecording {
            saveNotice = .failed(storage.message ?? "手机空间不够了，先清理一些照片或视频再录")
            return
        }
        let url = VideoStore.newRecordingURL()
        let court = courtType
        // The court the setup lock found, at the recording's size (nil if it never locked).
        let frame = trajectoryFrameSize.width > 0 ? trajectoryFrameSize : CGSize(width: 1920, height: 1080)
        let focalPx = focal.current
        let hint = courtLockEngine.lockedCameraHint(
            recordingSize: ImageSize(width: Int(frame.width.rounded()), height: Int(frame.height.rounded())))
        // Strong pipeline capture: the take is ingested even if this screen is gone by then.
        let matchPipeline = matchPipeline
        let started = cameraSession.startRecording(to: url) { [weak self] take in
            let notice = Self.ingest(take, fileURL: url, court: court, cameraHint: hint, focalHintPx: focalPx,
                                     pipeline: matchPipeline)
            guard let self else { return }
            // The take can end on its own (disk full, interruption): don't leave a live stop button.
            if self.isRecording {
                self.isRecording = false
                self.recentTrajectoryPoints = []
                self.courtLockEngine.resume()
            }
            // After ingest, so the new take joins the queue behind the paused analysis.
            matchPipeline?.recordingDidStop()
            self.saveNotice = notice
        }
        guard started else {
            saveNotice = .failed("相机还没准备好，请稍后再试")
            return
        }
        saveNotice = nil
        isRecording = true
        idleTask?.cancel()
        idleTask = nil
        // Calibration searches would compete with 1080p60 encoding; the analysis re-calibrates from the file.
        courtLockEngine.pause()
        matchPipeline?.recordingDidStart()
        watchStorageWhileRecording()
    }

    /// Free space on the phone; nil until first checked (or when it can't be read).
    private(set) var storage: StorageStatus?

    /// A line for the record screen when space is short (nil when there's plenty).
    var storageMessage: String? { storage?.message }

    /// The storage line with how loud it should be (refusing to record or stopping is serious).
    var storageChip: SystemChipItem? {
        storage.flatMap(SystemChipItem.init(storage:))
    }

    func refreshStorage() { storage = StorageGuard().check() }

    /// Every 10 s while recording: stop and save before the disk fills up.
    private func watchStorageWhileRecording() {
        Task { [weak self] in
            while let self, self.isRecording {
                self.refreshStorage()
                if self.storage?.mustStopRecording == true { self.stopRecording(); return }
                try? await Task.sleep(for: .seconds(10))
            }
        }
    }

    private func stopRecording() {
        isRecording = false
        armIdleTimer()
        recentTrajectoryPoints = []
        courtLockEngine.resume()
        matchPipeline?.recordingDidStop()
        #if DEBUG
        if DemoMode.fakesRecording { saveNotice = .saved; return }
        #endif
        saveNotice = .finishing
        cameraSession.stopRecording()
        // Safety net: AVFoundation normally finalises within a second or two.
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(20))
            guard let self, self.saveNotice == .finishing else { return }
            self.saveNotice = .failed("保存录像超时，请到「比赛」查看是否已生成")
        }
    }

    /// Hands a finished file to the pipeline; returns what to tell the user.
    private static func ingest(
        _ take: CameraSession.FinishedTake, fileURL: URL, court: CourtType, cameraHint: CameraModel?, focalHintPx: Double?,
        pipeline: MatchPipeline?
    ) -> SaveNotice {
        if let message = take.errorMessage { return .failed(message) }
        guard let pipeline else { return .failed("录像已保存在本机，但上传服务未就绪") }
        do {
            try pipeline.ingestRecording(fileURL: fileURL, durationSeconds: take.durationSeconds, courtType: court,
                                         cameraHint: cameraHint, focalHintPx: focalHintPx)
            return .saved
        } catch {
            return .failed(MatchPipeline.failureReason(for: error))
        }
    }
}
