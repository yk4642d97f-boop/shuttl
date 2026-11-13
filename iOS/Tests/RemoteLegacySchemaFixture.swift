import SwiftData
import Foundation
@testable import Shuttl

// Frozen pre-lifecycle schema, copied from the source backup. Same persistent entity names.
// Used to create a real old store before opening it with the current app schema.
enum LifecycleBeforeCloudSchema {
    
    // MARK: - Match
    
    enum MatchStatus: String, Codable {
        case recording
        case uploading
        case analyzing
        case done
        case failed
    
        var displayName: String {
            switch self {
            case .recording: return "录制中"
            case .uploading: return "上传中"
            case .analyzing: return "分析中"
            case .done:      return "已完成"
            case .failed:    return "分析失败"
            }
        }
    }
    
    /// Where the upload/analysis pipeline resumes for a match. Ordered: a later
    /// step implies the earlier ones succeeded on the server.
    enum PipelineStep: String, Codable, Comparable, Sendable {
        case create     // POST /matches (the server id becomes Match.id)
        case upload     // POST /matches/{id}/upload, then PUT the file
        case analyze    // POST /matches/{id}/analyze
        case poll       // GET job status until done, then fetch the trajectory
        case local      // analyse on this phone (no server involved)
    
        private var order: Int {
            switch self {
            case .create: return 0
            case .upload: return 1
            case .analyze, .local: return 2
            case .poll: return 3
            }
        }
    
        static func < (lhs: PipelineStep, rhs: PipelineStep) -> Bool { lhs.order < rhs.order }
    }
    
    @Model
    final class Match {
        /// Local id until the server creates the match, then the server's id.
        var id: UUID
        var createdAt: Date
        var durationSeconds: Double
        var status: MatchStatus
        /// Relative to the Documents directory ("Matches/match_123.mov"). Older rows
        /// may hold an absolute path; `localVideoURL` resolves both.
        var localVideoPath: String?
        var cloudJobID: String?
        var courtType: CourtType
        /// Human-readable reason shown when `status == .failed`.
        var failureReason: String?
        /// Whether "重试" could end differently: false for a clean verdict on the video itself
        /// (no shuttle, no complete rally, hand-held), which only a new recording changes.
        /// nil = not failed, or failed before this was recorded (treated as worth a retry).
        var failureRetryable: Bool?
        /// Raw `PipelineStep`; nil once the match is done (or before it enters the pipeline).
        var pipelineStepRaw: String?
        /// Full on-device analysis (MatchAnalysis JSON), relative to Documents.
        var analysisPath: String?
    
        @Relationship(deleteRule: .cascade)
        var rallies: [Rally]
    
        init(
            id: UUID = UUID(),
            createdAt: Date = .now,
            durationSeconds: Double = 0,
            status: MatchStatus = .recording,
            localVideoPath: String? = nil,
            cloudJobID: String? = nil,
            courtType: CourtType = .singles,
            failureReason: String? = nil,
            pipelineStep: PipelineStep? = nil
        ) {
            self.id = id
            self.createdAt = createdAt
            self.durationSeconds = durationSeconds
            self.status = status
            self.localVideoPath = localVideoPath
            self.cloudJobID = cloudJobID
            self.courtType = courtType
            self.failureReason = failureReason
            self.pipelineStepRaw = pipelineStep?.rawValue
            self.rallies = []
        }
    
        var localVideoURL: URL? { VideoStore.resolve(localVideoPath) }
    
        var pipelineStep: PipelineStep? {
            get { pipelineStepRaw.flatMap(PipelineStep.init(rawValue:)) }
            set { pipelineStepRaw = newValue?.rawValue }
        }
    }
    
    // MARK: - Rally
    
    enum PlayerSide: String, Codable {
        case near, far, unknown
    }
    
    @Model
    final class Rally {
        var startFrame: Int
        var endFrame: Int
        var winner: PlayerSide
    
        @Relationship(deleteRule: .cascade)
        var shots: [ShotPoint]
    
        init(startFrame: Int, endFrame: Int, winner: PlayerSide = .unknown) {
            self.startFrame = startFrame
            self.endFrame = endFrame
            self.winner = winner
            self.shots = []
        }
    
        var frameCount: Int { endFrame - startFrame }
    }
    
    // MARK: - ShotPoint
    
    enum ShotType: String, Codable {
        case smash      // 扣杀
        case clear      // 高远
        case drop       // 吊球
        case net        // 网前
        case drive      // 平抽
        case serve      // 发球
        case unknown
    }
    
    @Model
    final class ShotPoint {
        // frameIndex is the canonical frame counter (aliased as `frame` in persistence for brevity)
        var frameIndex: Int
        // Pixel coords in the original video frame
        var xPx: Float
        var yPx: Float
        // Normalized coords 0–1, for overlay rendering without knowing image dimensions at read time
        var normalizedX: Float
        var normalizedY: Float
        // Real-world court coords in meters (requires homography; 0 if unknown)
        var xMeters: Float
        var yMeters: Float
        var speedKmh: Float
        var shotType: ShotType
    
        init(
            frameIndex: Int,
            xPx: Float,
            yPx: Float,
            normalizedX: Float = 0,
            normalizedY: Float = 0,
            xMeters: Float = 0,
            yMeters: Float = 0,
            speedKmh: Float = 0,
            shotType: ShotType = .unknown
        ) {
            self.frameIndex = frameIndex
            self.xPx = xPx
            self.yPx = yPx
            self.normalizedX = normalizedX
            self.normalizedY = normalizedY
            self.xMeters = xMeters
            self.yMeters = yMeters
            self.speedKmh = speedKmh
            self.shotType = shotType
        }
    }
    
    // MARK: - Supporting enums
    
    enum CourtType: String, Codable {
        case singles, doubles
    }
}
