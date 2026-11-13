import CoreMedia
import CoreVideo
import Foundation
import ShuttlAnalysis
import ShuttlCore
import ShuttlLive
import simd

/// The user's fixes to the AI's rally calls, kept next to the analysis JSON
/// (`<id>.corrections.json`) so the original result stays intact and a fix can be undone.
struct ScoreCorrections: Codable, Equatable, Sendable {
    enum Fix: String, Codable, Sendable {
        case nearWon, farWon
        /// Not a rally (a warm-up hit, a let): dropped from the rally list and the score.
        case notARally
    }

    /// Keyed by the rally's start frame, which stays stable across re-analysis of the same video.
    var fixes: [Int: Fix] = [:]

    var isEmpty: Bool { fixes.isEmpty }

    static func url(forAnalysisAt url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("corrections.json")
    }

    static func load(forAnalysisAt url: URL) -> ScoreCorrections {
        guard let data = try? Data(contentsOf: Self.url(forAnalysisAt: url)),
              let c = try? JSONDecoder().decode(ScoreCorrections.self, from: data) else { return ScoreCorrections() }
        return c
    }

    func save(forAnalysisAt url: URL) throws {
        let file = Self.url(forAnalysisAt: url)
        if isEmpty { try? FileManager.default.removeItem(at: file); return }
        try JSONEncoder().encode(self).write(to: file, options: .atomic)
    }

    /// The analysis as the user sees it: fixed winners (sure, source "user"), dropped
    /// non-rallies, and the score recomputed from the result.
    func applied(to analysis: MatchAnalysis) -> MatchAnalysis {
        guard !isEmpty else { return analysis }
        var out = analysis
        out.rallies = analysis.rallies.compactMap { rally in
            guard let fix = fixes[rally.startFrame] else { return rally }
            var r = rally
            switch fix {
            case .notARally: return nil
            case .nearWon: r.winner = .near
            case .farWon: r.winner = .far
            }
            r.outcomeSource = "user"
            r.outcomeConfidence = 1
            return r
        }
        out.score = PortedStages.score(out.rallies)
        return out
    }
}

/// What the user told us about a recording before analysis: the court camera the live
/// setup lock found, and corner taps from the calibration screen. Lives in Application
/// Support/AnalysisInputs/<id>.json and feeds `PipelineOptions`.
struct MatchInputs: Codable, Equatable, Sendable {
    /// Court keypoints by CourtModel name, 0…1 of the upright frame.
    var manualCorners: [String: SIMD2<Double>]?
    /// Camera from the record screen's court lock; used when the file's own calibration fails.
    var cameraHint: CameraModel?
    /// The lens's focal length in px at the recording size, from the capture intrinsics.
    var focalHintPx: Double?
    /// Stable across retries/relaunch so a corrected court creates exactly one cloud job.
    var cloudReanalysisID: UUID?

    static func directory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AnalysisInputs", isDirectory: true)
    }

    static func url(matchID: UUID, directory: URL = directory()) -> URL {
        directory.appendingPathComponent("\(matchID.uuidString.lowercased()).json")
    }

    static func load(matchID: UUID, directory: URL = directory()) -> MatchInputs {
        guard let data = try? Data(contentsOf: url(matchID: matchID, directory: directory)),
              let inputs = try? ShuttlJSON.decoder().decode(MatchInputs.self, from: data) else { return MatchInputs() }
        return inputs
    }

    func save(matchID: UUID, directory: URL = Self.directory()) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try ShuttlJSON.encoder().encode(self).write(to: Self.url(matchID: matchID, directory: directory), options: .atomic)
    }

    static func delete(matchID: UUID, directory: URL = directory()) {
        try? FileManager.default.removeItem(at: url(matchID: matchID, directory: directory))
    }

    func options(doubles: Bool?) -> PipelineOptions {
        PipelineOptions(doubles: doubles, manualCorners: manualCorners, cornersNormalized: true, cameraHint: cameraHint,
                        focalHintPx: focalHintPx)
    }

    /// fx from a frame's camera intrinsic matrix, in px of a video whose long side is
    /// `videoLongSide`. Pixels are square, so only the scale (long side to long side) matters,
    /// not whether the buffer is upright or sideways.
    static func focalPx(from sample: CMSampleBuffer, videoLongSide: Int) -> Double? {
        guard let data = CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
                                         attachmentModeOut: nil) as? Data,
              data.count >= MemoryLayout<matrix_float3x3>.size,
              let buffer = CMSampleBufferGetImageBuffer(sample) else { return nil }
        let k = data.withUnsafeBytes { $0.loadUnaligned(as: matrix_float3x3.self) }
        let fx = Double(k.columns.0.x)
        let long = Double(max(CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer)))
        guard fx.isFinite, fx > 0, long > 0 else { return nil }
        return fx * Double(videoLongSide) / long
    }
}
