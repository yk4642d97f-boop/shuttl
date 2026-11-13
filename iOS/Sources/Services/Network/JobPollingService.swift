import Foundation
import SwiftData
import AVFoundation

/// Polls a backend analysis job until Celery reports SUCCESS or FAILURE.
/// Pure networking (no SwiftData), so it can run off the main actor; the
/// pipeline applies the result with `apply(_:to:frameSize:)` on the main actor.
struct JobPollingService: Sendable {
    struct Options: Sendable {
        var interval: Duration = .seconds(5)
        var maxBackoff: Duration = .seconds(60)
        /// Give up if the job never leaves PENDING — no worker picked it up
        /// (a single worker also queues jobs behind a running one, hence generous).
        var pendingTimeout: Duration = .seconds(30 * 60)
        /// Absolute cap once the job is running.
        var overallTimeout: Duration = .seconds(4 * 60 * 60)
        /// Network / 5xx failures in a row before giving up (with backoff ≈ 3 min).
        var maxConsecutiveFailures = 6

        static let standard = Options()
    }

    enum Outcome: Equatable, Sendable {
        case succeeded
        case failed(String?)
    }

    let backend: any MatchBackend
    let options: Options
    private let sleep: @Sendable (Duration) async throws -> Void

    init(
        backend: any MatchBackend = LiveMatchBackend(),
        options: Options = .standard,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.backend = backend
        self.options = options
        self.sleep = sleep
    }

    /// - Parameter onUpdate: called with every successfully fetched status (for progress UI).
    func waitForCompletion(
        matchID: UUID,
        jobID: String,
        onUpdate: @escaping @Sendable (JobStatusResponse) async -> Void = { _ in }
    ) async throws -> Outcome {
        var elapsed: Duration = .zero
        var pendingElapsed: Duration = .zero
        var consecutiveFailures = 0

        while true {
            try Task.checkCancellation()
            let status: JobStatusResponse
            do {
                status = try await backend.pollJob(matchID: matchID, jobID: jobID)
                consecutiveFailures = 0
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                guard Self.shouldRetry(error), consecutiveFailures < options.maxConsecutiveFailures else { throw error }
                consecutiveFailures += 1
                let backoff = min(options.interval * (1 << min(consecutiveFailures - 1, 10)), options.maxBackoff)
                try await sleep(backoff)
                elapsed += backoff
                continue
            }

            await onUpdate(status)
            switch status.state {
            case "SUCCESS":
                return .succeeded
            case "FAILURE", "REVOKED":
                return .failed(Self.cleanInfo(status.info))
            case "PENDING":
                pendingElapsed += options.interval
                if pendingElapsed > options.pendingTimeout { throw PipelineError.analysisStalled }
            default:
                // STARTED / PROGRESS / RETRY: the worker has it.
                pendingElapsed = .zero
            }
            if elapsed > options.overallTimeout { throw PipelineError.analysisStalled }
            try await sleep(options.interval)
            elapsed += options.interval
        }
    }

    private static func shouldRetry(_ error: Error) -> Bool {
        guard let error = error as? APIError else { return false }
        switch error {
        case .networkError, .timeout:
            return true
        case .serverError(let code, _):
            return code == 408 || code == 425 || code == 429 || code >= 500
        case .unauthorized, .decodingError, .uploadFailed:
            return false
        }
    }

    /// `info` is `str(result.info)`: a Python dict repr while running
    /// (`{'match_id': '…', 'percent': 42, 'stage': 'detect'}`), the exception text on failure.
    static func progressPercent(from info: String?) -> Int? {
        guard let info,
              let match = info.firstMatch(of: /['"]percent['"]\s*:\s*(\d+)/),
              let value = Int(match.1) else { return nil }
        return min(max(value, 0), 100)
    }

    private static func cleanInfo(_ info: String?) -> String? {
        guard let info = info?.trimmingCharacters(in: .whitespacesAndNewlines),
              !info.isEmpty, info != "None" else { return nil }
        return info
    }

    // MARK: - Applying results (main actor)

    /// Replaces the match's rallies with the server trajectory.
    /// - Parameter frameSize: upright video size, only used if the server sent
    ///   pixel coordinates instead of normalised ones.
    /// Legacy `/trajectory` rows are one per hit, so rallies are split on a
    /// hit-sized gap rather than a per-frame one.
    @MainActor
    static func apply(_ trajectory: TrajectoryResponse, to match: Match, frameSize: CGSize?) {
        clearRallies(of: match)
        for segment in segmentIntoRallies(trajectory.points, fps: trajectory.fps, maxGapSeconds: 4) {
            guard let first = segment.first, let last = segment.last else { continue }
            let rally = Rally(startFrame: first.frame, endFrame: last.frame)
            for p in segment {
                rally.shots.append(ShotPoint(
                    frameIndex: p.frame,
                    xPx: p.xPx,
                    yPx: p.yPx,
                    normalizedX: normalized(p.xPx, extent: frameSize?.width),
                    normalizedY: normalized(p.yPx, extent: frameSize?.height),
                    xMeters: p.xMeters,
                    yMeters: p.yMeters,
                    speedKmh: p.speedKmh,
                    shotType: ShotType(rawValue: p.shotType) ?? .unknown
                ))
            }
            match.rallies.append(rally)
        }
    }

    /// Preferred source: the v2 analysis document carries the real rally boundaries and winners.
    @MainActor
    static func apply(_ doc: AnalysisDocumentDTO, to match: Match) {
        clearRallies(of: match)
        for r in doc.rallies {
            let rally = Rally(startFrame: r.startFrame, endFrame: max(r.endFrame, r.startFrame),
                              winner: PlayerSide(rawValue: r.winner ?? "") ?? .unknown)
            for s in r.shots.sorted(by: { $0.frame < $1.frame }) {
                let nx = Float(component(s.xyNorm, 0) ?? 0.5), ny = Float(component(s.xyNorm, 1) ?? 0.5)
                let type = ShotType(rawValue: s.legacyShotType ?? s.shotType ?? "") ?? .unknown
                rally.shots.append(ShotPoint(
                    frameIndex: s.frame,
                    xPx: nx, yPx: ny,
                    normalizedX: normalized(nx, extent: nil),
                    normalizedY: normalized(ny, extent: nil),
                    xMeters: Float(component(s.hitterPos, 0) ?? 0),
                    yMeters: Float(component(s.hitterPos, 1) ?? 0),
                    speedKmh: Float(s.flight?.speed0Kmh ?? 0),
                    shotType: type
                ))
            }
            match.rallies.append(rally)
        }
    }

    private static func component(_ values: [Double?]?, _ i: Int) -> Double? {
        guard let values, i < values.count, let v = values[i], v.isFinite else { return nil }
        return v
    }

    /// Deletes (not just unlinks) existing rallies so a re-analysis leaves no orphans.
    @MainActor
    private static func clearRallies(of match: Match) {
        if let context = match.modelContext { match.rallies.forEach(context.delete) }
        match.rallies = []
    }

    /// The backend stores `x_px / width`, `y_px / height` of the upright frame
    /// (tasks.py `_write_results`), so values are normally already 0–1. Anything
    /// clearly larger is treated as pixels and divided by the frame extent.
    static func normalized(_ value: Float, extent: CGFloat?) -> Float {
        if value > 1.5, let extent, extent > 0 { return min(max(value / Float(extent), 0), 1) }
        return min(max(value, 0), 1)
    }

    /// Upright (display-orientation) size of a local video, or nil if unreadable.
    static func uprightVideoSize(of url: URL) async -> CGSize? {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let (size, transform) = try? await track.load(.naturalSize, .preferredTransform) else { return nil }
        let rect = CGRect(origin: .zero, size: size).applying(transform)
        return CGSize(width: abs(rect.width), height: abs(rect.height))
    }

    /// Splits the flat point list wherever consecutive frames are more than
    /// `maxGapSeconds` apart.
    static func segmentIntoRallies(
        _ points: [TrajectoryPointDTO], fps: Double, maxGapSeconds: Double = 0.3
    ) -> [[TrajectoryPointDTO]] {
        let maxGap = max(1, Int(fps * maxGapSeconds))
        var segments: [[TrajectoryPointDTO]] = []
        var current: [TrajectoryPointDTO] = []
        for p in points.sorted(by: { $0.frame < $1.frame }) {
            if let last = current.last, p.frame - last.frame > maxGap {
                segments.append(current)
                current = []
            }
            current.append(p)
        }
        if !current.isEmpty { segments.append(current) }
        return segments
    }
}
