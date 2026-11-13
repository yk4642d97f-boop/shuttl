import SwiftUI
import AVFoundation
import ShuttlCore
import SwiftData

// MARK: - ReplayViewModel

@MainActor
@Observable
final class ReplayViewModel {
    // MARK: Player state
    var player: AVPlayer?
    var isPlaying: Bool = false
    var currentTime: Double = 0
    var duration: Double = 0
    var playbackRate: Float = 1.0
    var nearbyShot: ShotPoint? = nil
    /// Upright presentation size of the video (zero until the item is ready).
    var videoSize: CGSize = .zero
    /// The match points at a file that isn't on this device (deleted, or never copied back).
    var videoMissing = false

    // MARK: Trajectory state
    var currentTrajectoryWindow: [TrajectoryPoint] = []

    // MARK: Private
    private var periodicObserver: Any?
    private var statusObserver: NSKeyValueObservation?
    private var allShotPoints: [ShotPoint] = []
    private var videoFPS: Double = 60.0
    private let playbackRates: [Float] = [0.25, 0.5, 1.0, 2.0]
    private var trajectoryPoints: [TrajectoryPoint] = []
    /// Rally spans in frames, for "只看回合" and the scrubber bands.
    private(set) var rallyWindows: [ClosedRange<Int>] = []
    /// Skip the dead time between rallies (picking up the shuttle, walking back).
    var ralliesOnly = false
    private var lastSkip: Date = .distantPast
    /// Frame rate of the loaded video (60 until the track reports its own).
    var fps: Double { videoFPS }

    var availableRates: [Float] { playbackRates }

    private let match: Match?
    /// Seek here once the video is ready (frame index at the video's own rate).
    private var pendingStartFrame: Int?

    init(match: Match? = nil, startFrame: Int? = nil) {
        self.match = match
        self.pendingStartFrame = startFrame
    }

    // MARK: Setup

    /// Loads the match's local video and shot points. No-op without a video file.
    func setup() {
        guard let match else { return }
        guard let url = match.localVideoURL else { videoMissing = true; return }
        #if DEBUG
        let isDemo = DemoMode.isOn
        #else
        let isDemo = false
        #endif
        if !isDemo && !FileManager.default.fileExists(atPath: url.path) {
            videoMissing = true
            return
        }
        rallyWindows = match.rallies
            .map { max(0, $0.startFrame - 30)...(max($0.startFrame, $0.endFrame) + 45) }
            .sorted { $0.lowerBound < $1.lowerBound }
        load(videoURL: url, shotPoints: match.rallies.flatMap(\.shots))
        loadTrack(from: match.analysisPath)
    }

    /// Where the video is drawn inside a container using `.resizeAspect`, so
    /// normalised (0–1 of the video frame) points land on the picture, not the letterbox.
    func videoRect(in container: CGSize) -> CGRect {
        guard videoSize.width > 0, videoSize.height > 0 else { return CGRect(origin: .zero, size: container) }
        let scale = min(container.width / videoSize.width, container.height / videoSize.height)
        let size = CGSize(width: videoSize.width * scale, height: videoSize.height * scale)
        return CGRect(x: (container.width - size.width) / 2, y: (container.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    func teardown() {
        cleanup()
    }

    func load(videoURL: URL, shotPoints: [ShotPoint]) {
        self.allShotPoints = shotPoints
        loadTrajectory(from: shotPoints)

        let item = AVPlayerItem(asset: AVURLAsset(url: videoURL))
        let newPlayer = AVPlayer(playerItem: item)
        self.player = newPlayer

        // KVO fires off the main thread: hand over only plain values.
        statusObserver = item.observe(\.status, options: [.new]) { [weak self] item, _ in
            guard item.status == .readyToPlay else { return }
            let seconds = item.duration.seconds
            let presentation = item.presentationSize
            Task { @MainActor in await self?.itemReady(seconds: seconds, presentation: presentation, url: videoURL) }
        }

        startPeriodicObserver(player: newPlayer)
    }

    private func itemReady(seconds: Double, presentation: CGSize, url: URL) async {
        duration = seconds
        videoSize = presentation
        videoFPS = await Self.nominalFPS(of: url)
        if let frame = pendingStartFrame {
            pendingStartFrame = nil
            // A little lead-in so the serve is visible, not just the contact.
            seek(to: max(0, Double(frame) / videoFPS - 0.8))
        }
    }

    /// Reads the frame rate from its own asset, so no AVAsset is shared across threads.
    private nonisolated static func nominalFPS(of url: URL) async -> Double {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let fps = try? await track.load(.nominalFrameRate), fps > 0 else { return 60 }
        return Double(fps)
    }

    /// The on-device analysis's per-frame shuttle track, normalised to the upright frame
    /// (NaN where the shuttle wasn't seen). Empty for server matches, which only have hits.
    private var trackFrames: [SIMD2<Float>] = []
    /// Trail length behind the shuttle.
    private let trailSeconds = 0.5
    /// A hole longer than this ends the trail, so it never bridges two separate flights.
    private let maxGapFrames = 3

    private func loadTrack(from storedPath: String?) {
        guard let storedPath else { return }
        Task { @MainActor [weak self] in
            let frames = await Task.detached(priority: .userInitiated) { () -> [SIMD2<Float>] in
                guard let analysis = AnalysisStore.load(storedPath), let track = analysis.track else { return [] }
                let w = Double(max(analysis.meta.width, 1)), h = Double(max(analysis.meta.height, 1))
                return track.xy.map { p in
                    p.x.isFinite && p.y.isFinite ? SIMD2(Float(p.x / w), Float(p.y / h)) : SIMD2(.nan, .nan)
                }
            }.value
            self?.trackFrames = frames
        }
    }

    /// The last `trailSeconds` of flight up to `frame`, oldest first, stopping at a gap.
    private func trackWindow(endingAt frame: Int) -> [TrajectoryPoint] {
        guard !trackFrames.isEmpty, frame >= 0 else { return [] }
        let last = min(frame, trackFrames.count - 1)
        let first = max(0, last - Int(trailSeconds * videoFPS))
        var out: [TrajectoryPoint] = []
        var gap = 0
        var i = last
        while i >= first {
            let p = trackFrames[i]
            if p.x.isFinite && p.y.isFinite {
                gap = 0
                out.append(TrajectoryPoint(position: CGPoint(x: Double(p.x), y: Double(p.y)), frameIndex: i, speedKmh: 0))
            } else {
                gap += 1
                if gap > maxGapFrames { break }
            }
            i -= 1
        }
        return out.reversed()
    }

    private func loadTrajectory(from shots: [ShotPoint]) {
        trajectoryPoints = shots.map { shot in
            TrajectoryPoint(
                position: CGPoint(x: Double(shot.normalizedX), y: Double(shot.normalizedY)),
                frameIndex: shot.frameIndex,
                speedKmh: shot.speedKmh
            )
        }.sorted { $0.frameIndex < $1.frameIndex }
    }

    private func startPeriodicObserver(player: AVPlayer) {
        let interval = CMTime(seconds: 1.0 / 60.0, preferredTimescale: 600)
        periodicObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            // Delivered on the main queue (asked for above); checked at run time.
            MainActor.assumeIsolated {
                guard let self else { return }
                self.currentTime = time.seconds
                self.updateTrajectoryWindow(for: time)
                self.updateNearbyShot(for: time)
                self.skipDeadTimeIfNeeded(at: time)
            }
        }
    }

    private func updateTrajectoryWindow(for time: CMTime) {
        let currentFrame = Int((time.seconds * videoFPS).rounded())
        if !trackFrames.isEmpty {
            currentTrajectoryWindow = trackWindow(endingAt: currentFrame)
            return
        }
        let windowStart = currentFrame - 30
        currentTrajectoryWindow = trajectoryPoints.filter {
            $0.frameIndex >= windowStart && $0.frameIndex <= currentFrame
        }
    }

    /// With "只看回合" on, jump from the end of one rally to the start of the next.
    private func skipDeadTimeIfNeeded(at time: CMTime) {
        guard ralliesOnly, isPlaying, !rallyWindows.isEmpty, Date().timeIntervalSince(lastSkip) > 0.6 else { return }
        let frame = Int(time.seconds * videoFPS)
        guard !rallyWindows.contains(where: { $0.contains(frame) }) else { return }
        if let next = rallyWindows.first(where: { $0.lowerBound > frame }) {
            lastSkip = Date()
            seek(to: Double(next.lowerBound) / videoFPS)
        } else if frame > rallyWindows.last!.upperBound {
            togglePlayback()
        }
    }

    private func updateNearbyShot(for time: CMTime) {
        let currentFrame = Int(time.seconds * videoFPS)
        nearbyShot = allShotPoints.first { abs($0.frameIndex - currentFrame) <= 5 }
    }

    // MARK: Playback controls

    func togglePlayback() {
        guard let player else { return }
        if isPlaying {
            player.pause()
            isPlaying = false
        } else {
            player.rate = playbackRate
            isPlaying = true
        }
    }

    func seek(to time: Double) {
        guard let player else { return }
        let cmTime = CMTime(seconds: time, preferredTimescale: 600)
        player.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func setRate(_ rate: Float) {
        playbackRate = rate
        if isPlaying { player?.rate = rate }
    }

    func cleanup() {
        if let observer = periodicObserver {
            player?.removeTimeObserver(observer)
        }
        statusObserver?.invalidate()
        player?.pause()
        player = nil
    }
}
