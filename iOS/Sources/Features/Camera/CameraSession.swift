import AVFoundation
import UIKit
import Combine

// AVCaptureSession wrapper — owns the session lifecycle and exposes
// a previewLayer plus a frame publisher for ML inference.
//
// Two halves, so no capture object is touched from two threads:
// - `CaptureGraph` owns the session, inputs and outputs and only touches them on its
//   serial queue (Apple's rule for AVCaptureSession).
// - `CameraSession` (main actor) holds the UI state, the preview layer and the timers,
//   and only talks to the graph by queueing work on it.
@MainActor
final class CameraSession: NSObject, ObservableObject {
    enum State { case idle, running, recording, error(String) }

    @Published private(set) var state: State = .idle
    @Published private(set) var recordingDuration: TimeInterval = 0
    /// The lens currently feeding preview, ML frames and the recorded file.
    @Published private(set) var position: AVCaptureDevice.Position = .back
    /// True while the input is being swapped; recording can't start meanwhile.
    @Published private(set) var isSwitching = false

    /// Whether this device has both a front and a back wide-angle camera.
    static var canSwitchCameras: Bool {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified
        )
        let positions = Set(discovery.devices.map(\.position))
        return positions.contains(.front) && positions.contains(.back)
    }

    let previewLayer = AVCaptureVideoPreviewLayer()

    /// Downstream consumers subscribe to receive sample buffers (sent on the capture queue).
    var framePublisher: PassthroughSubject<CMSampleBuffer, Never> { graph.frames }

    private let graph = CaptureGraph()
    private var durationTask: Task<Void, Never>?
    private var recordingStart: Date?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservations: [NSKeyValueObservation] = []

    /// A finished take, delivered only after AVFoundation has finalised the file.
    struct FinishedTake: Sendable {
        let url: URL
        let durationSeconds: Double
        /// nil on success; otherwise why the file is unusable.
        let errorMessage: String?
    }

    private var takeCompletions: [URL: @MainActor (FinishedTake) -> Void] = [:]
    /// Keeps the capture graph alive while a take is being finalised, even if the
    /// screen that owns this session goes away mid-recording (tab switch).
    private var keepAlive: CameraSession?
    private var switchCompletion: (@MainActor (AVCaptureDevice.Position) -> Void)?
    private var sessionObservers: [NSObjectProtocol] = []

    override init() {
        super.init()
        previewLayer.session = graph.session
        previewLayer.videoGravity = .resizeAspectFill
        graph.onTakeFinished = { [weak self] url, seconds, message in
            Task { @MainActor in self?.finishTake(url: url, reportedSeconds: seconds, errorMessage: message) }
        }
    }

    func requestPermissionAndConfigure() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .video)
        }
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            state = .error("需要相机权限，请在设置中开启")
            return
        }
        let graph = graph
        graph.queue.async { [weak self] in
            let configured = graph.configure()
            Task { @MainActor in self?.didConfigure(configured) }
        }
    }

    func startRunning() {
        let graph = graph
        graph.queue.async { [weak self] in
            guard graph.startIfNeeded() else { return }
            Task { @MainActor in self?.state = .running }
        }
    }

    func stopRunning() {
        let graph = graph
        graph.queue.async { [weak self] in
            graph.session.stopRunning()
            Task { @MainActor in self?.state = .idle }
        }
    }

    /// Starts a take. `completion` runs on the main actor once the file is fully
    /// written (or failed). Returns false if the camera isn't running yet.
    @discardableResult
    func startRecording(to url: URL, completion: @escaping @MainActor (FinishedTake) -> Void) -> Bool {
        guard case .running = state, !isSwitching else { return false }
        takeCompletions[url] = completion
        keepAlive = self
        let graph = graph
        graph.queue.async { [weak self] in
            graph.startRecording(to: url)
            Task { @MainActor in self?.didStartRecording() }
        }
        return true
    }

    func stopRecording() {
        let graph = graph
        graph.queue.async { graph.movieOutput.stopRecording() }
        durationTask?.cancel()
        durationTask = nil
    }

    /// Flips between the back and front camera. No-op while recording or mid-switch.
    /// `completion` receives the position actually in use afterwards.
    func switchCamera(completion: @escaping @MainActor (AVCaptureDevice.Position) -> Void) {
        guard case .running = state, !isSwitching, recordingStart == nil else {
            completion(position)
            return
        }
        let target: AVCaptureDevice.Position = position == .back ? .front : .back
        isSwitching = true
        switchCompletion = completion
        let graph = graph
        graph.queue.async { [weak self] in
            let switched = graph.reconfigureInput(for: target)
            Task { @MainActor in self?.finishSwitch(to: switched ? target : nil) }
        }
    }

    // MARK: - Private (main actor)

    private func didConfigure(_ configured: Bool) {
        guard configured else {
            state = .error("无法初始化摄像头")
            return
        }
        state = .running
        rebuildRotationTracking()
        applyPreviewMirroring()
        observeSessionLifecycle()
    }

    private func didStartRecording() {
        state = .recording
        let start = Date.now
        recordingStart = start
        durationTask?.cancel()
        durationTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard let self, !Task.isCancelled else { return }
                self.recordingDuration = Date.now.timeIntervalSince(start)
            }
        }
    }

    /// Brings the camera back after an interruption (another app took the camera, Control
    /// Centre, a media-services reset) so the screen never sits on a frozen preview.
    private func observeSessionLifecycle() {
        guard sessionObservers.isEmpty else { return }
        let center = NotificationCenter.default
        let graph = graph
        let restart: @Sendable (Notification) -> Void = { _ in
            graph.queue.async { _ = graph.startIfNeeded() }
        }
        sessionObservers = [
            center.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: graph.session,
                               queue: nil, using: restart),
            center.addObserver(forName: AVCaptureSession.runtimeErrorNotification, object: graph.session,
                               queue: nil, using: restart),
        ]
    }

    private func finishSwitch(to newPosition: AVCaptureDevice.Position?) {
        if let newPosition { position = newPosition }
        rebuildRotationTracking()
        applyPreviewMirroring()
        isSwitching = false
        let completion = switchCompletion
        switchCompletion = nil
        completion?(position)
    }

    /// Front camera preview behaves like a mirror (what people expect); the back camera doesn't.
    private func applyPreviewMirroring() {
        guard let connection = previewLayer.connection, connection.isVideoMirroringSupported else { return }
        connection.automaticallyAdjustsVideoMirroring = false
        connection.isVideoMirrored = position == .front
    }

    // Keeps preview, ML frames and the recorded file horizon-level for however the phone is held.
    // Rebuilt on every camera switch: the coordinator is bound to one device.
    private func rebuildRotationTracking() {
        rotationObservations.forEach { $0.invalidate() }
        rotationObservations = []
        rotationCoordinator = nil
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
        else { return }
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotationCoordinator = coordinator
        applyPreviewRotation(coordinator.videoRotationAngleForHorizonLevelPreview)
        applyCaptureRotation(coordinator.videoRotationAngleForHorizonLevelCapture)
        rotationObservations = [
            coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: .new) { [weak self] _, change in
                guard let angle = change.newValue else { return }
                Task { @MainActor in self?.applyPreviewRotation(angle) }
            },
            coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: .new) { [weak self] _, change in
                guard let angle = change.newValue else { return }
                Task { @MainActor in self?.applyCaptureRotation(angle) }
            },
        ]
    }

    private func applyPreviewRotation(_ angle: CGFloat) {
        guard let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(angle) else { return }
        connection.videoRotationAngle = angle
    }

    /// The output connections belong to the graph, so the change is made on its queue.
    private func applyCaptureRotation(_ angle: CGFloat) {
        let graph = graph
        graph.queue.async { graph.applyCaptureRotation(angle) }
    }

    private func finishTake(url: URL, reportedSeconds: Double, errorMessage: String?) {
        // Also reached when AVFoundation ended the take itself (disk-space floor, interruption).
        durationTask?.cancel()
        durationTask = nil
        let wallClock = recordingStart.map { Date.now.timeIntervalSince($0) } ?? recordingDuration
        let duration = reportedSeconds.isFinite && reportedSeconds > 0 ? reportedSeconds : wallClock
        recordingStart = nil
        recordingDuration = 0
        let running = graph.session.isRunning
        if let errorMessage {
            // Surface the failure but leave the camera usable for the next take.
            state = running ? .running : .idle
            print("[CameraSession] recording failed: \(errorMessage)")
        } else if case .recording = state {
            state = running ? .running : .idle
        }

        var completion = takeCompletions.removeValue(forKey: url)
        if completion == nil, takeCompletions.count == 1 {
            // URL normalisation differences (file:///private/var vs /var): only one take can be pending.
            completion = takeCompletions.removeValue(forKey: takeCompletions.keys.first!)
        }
        completion?(FinishedTake(url: url, durationSeconds: duration, errorMessage: errorMessage.map { "录制失败：\($0)" }))
        if takeCompletions.isEmpty { keepAlive = nil }
    }
}

// MARK: - CaptureGraph
// Everything AVFoundation, confined to one serial queue. The capture objects are not
// Sendable; this type is @unchecked Sendable because every method below that touches
// them runs on `queue` (the only exceptions are `session`, handed to the preview layer
// once at init, and `session.isRunning`, a thread-safe read).

final class CaptureGraph: NSObject, @unchecked Sendable {
    let queue = DispatchQueue(label: "com.shuttlevision.camera", qos: .userInteractive)
    let session = AVCaptureSession()
    let movieOutput = AVCaptureMovieFileOutput()
    private let videoOutput = AVCaptureVideoDataOutput()
    /// Subjects serialise `send` internally; frames are sent on `queue`.
    let frames = PassthroughSubject<CMSampleBuffer, Never>()
    /// url, recorded seconds, error message (nil = usable file). Set once, before any take.
    var onTakeFinished: (@Sendable (URL, Double, String?) -> Void)?

    /// Builds the graph once (back camera, 1080p60, ML frames + movie file) and starts it.
    /// Queue only. False when the camera can't be opened.
    func configure() -> Bool {
        dispatchPrecondition(condition: .onQueue(queue))
        guard session.inputs.isEmpty else { return true }   // already configured
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080

        // Video input — wide-angle back camera
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let videoInput = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(videoInput)
        else {
            session.commitConfiguration()
            return false
        }
        session.addInput(videoInput)
        // After addInput: an explicit activeFormat (1080p60) takes over from the preset.
        Self.configureDevice(device)

        // Frame output for ML — drop frames when the queue is busy rather than backing up
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        ]
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        // Movie file output. Video only: a microphone input would let every incoming call
        // or voice chat interrupt the capture session mid-match.
        if session.canAddOutput(movieOutput) { session.addOutput(movieOutput) }
        // AVFoundation stops the take itself (with a complete, playable file) below this.
        movieOutput.minFreeDiskSpaceLimit = 1_000_000_000
        configureOutputConnections()

        session.commitConfiguration()
        session.startRunning()
        return true
    }

    /// Queue only. True when the session is (now) running.
    func startIfNeeded() -> Bool {
        dispatchPrecondition(condition: .onQueue(queue))
        if !session.isRunning { session.startRunning() }
        return session.isRunning
    }

    func startRecording(to url: URL) {
        dispatchPrecondition(condition: .onQueue(queue))
        movieOutput.startRecording(to: url, recordingDelegate: self)
    }

    /// Swaps the camera input inside one configuration transaction (queue only).
    /// Rolls back to the previous input if the new one can't be added.
    func reconfigureInput(for target: AVCaptureDevice.Position) -> Bool {
        dispatchPrecondition(condition: .onQueue(queue))
        guard !movieOutput.isRecording,
              let newDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: target),
              let newInput = try? AVCaptureDeviceInput(device: newDevice) else { return false }
        let oldInputs = session.inputs.compactMap { $0 as? AVCaptureDeviceInput }.filter { $0.device.hasMediaType(.video) }

        session.beginConfiguration()
        oldInputs.forEach(session.removeInput)
        session.sessionPreset = .hd1920x1080
        guard session.canAddInput(newInput) else {
            oldInputs.forEach { if session.canAddInput($0) { session.addInput($0) } }
            session.commitConfiguration()
            return false
        }
        session.addInput(newInput)
        Self.configureDevice(newDevice)
        // New input → new connections: re-apply unmirrored, unstabilised capture.
        configureOutputConnections()
        session.commitConfiguration()
        return true
    }

    /// ML frames follow the phone; the file's orientation is fixed when recording starts,
    /// so rotating mid-take must not change it. Queue only.
    func applyCaptureRotation(_ angle: CGFloat) {
        dispatchPrecondition(condition: .onQueue(queue))
        if let connection = videoOutput.connection(with: .video), connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
        }
        guard !movieOutput.isRecording,
              let connection = movieOutput.connection(with: .video),
              connection.isVideoRotationAngleSupported(angle) else { return }
        connection.videoRotationAngle = angle
    }

    /// 1080p60 (fallback 1080p30, else the preset), continuous AF/AE. Call after the
    /// input is in the session and inside its configuration transaction.
    private static func configureDevice(_ device: AVCaptureDevice) {
        do { try device.lockForConfiguration() } catch { return }
        defer { device.unlockForConfiguration() }

        let formats = device.formats
        if let choice = CaptureFormatSelector.select(from: formats.map(CaptureFormatSelector.Candidate.init)) {
            let duration = CMTime(value: 1, timescale: CMTimeScale(choice.fps))
            device.activeFormat = formats[choice.index]
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
        }
        if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
        if device.isSmoothAutoFocusSupported { device.isSmoothAutoFocusEnabled = true }
        if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
    }

    /// The recorded file and the ML frames are never mirrored and never stabilised
    /// (stabilisation crops and warps the frame, so file pixels would stop matching
    /// what the tracker saw, and it adds latency).
    private func configureOutputConnections() {
        // The lens's real focal length rides along with each frame; court calibration uses it
        // to reject cameras with an impossible focal length.
        if let connection = videoOutput.connection(with: .video), connection.isCameraIntrinsicMatrixDeliverySupported {
            connection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
        for connection in [movieOutput.connection(with: .video), videoOutput.connection(with: .video)].compactMap({ $0 }) {
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CaptureGraph: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frames.send(sampleBuffer)
    }
}

// MARK: - AVCaptureFileOutputRecordingDelegate

extension CaptureGraph: AVCaptureFileOutputRecordingDelegate {
    /// Called by AVFoundation once the movie file is completely written — the
    /// only point at which it is safe to upload.
    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        // Some "errors" (disk nearly full, max duration) still leave a complete, playable file.
        var usable = error == nil
        if let error = error as NSError?,
           error.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool == true {
            usable = true
        }
        let message = usable ? nil : (error?.localizedDescription ?? "未知错误")
        onTakeFinished?(outputFileURL, output.recordedDuration.seconds, message)
    }
}
