import Foundation
import ShuttlAnalysis
import ShuttlCore
import ShuttlLive
import ShuttlRuntime

/// Runs the whole tracking pipeline on this phone. The live implementation wraps
/// `AnalysisPipeline` with the stage packages; tests substitute a fake.
protocol OnDeviceAnalyzer: Sendable {
    func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                 progress: @escaping @Sendable (AnalysisProgress) -> Void,
                 stopOnRetryableFailure: Bool) async throws -> MatchAnalysis
}

struct PipelineAnalyzer: OnDeviceAnalyzer {
    let stages: AnalysisStages

    func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                 progress: @escaping @Sendable (AnalysisProgress) -> Void,
                 stopOnRetryableFailure: Bool) async throws -> MatchAnalysis {
        try await AnalysisPipeline(stages: stages, checkpoints: checkpoints, stopOnRetryableFailure: stopOnRetryableFailure)
            .analyze(video, options: options, progress: progress)
    }
}

/// Loads the CoreML detector on first use (compiling is async and costs ~1 s the
/// first time; the compiled model is cached) and reuses it for every match.
actor LiveAnalyzer: OnDeviceAnalyzer {
    private var stages: LiveStages?
    private let modelsDirectory: URL
    private let cacheDirectory: URL

    init(modelsDirectory: URL, cacheDirectory: URL) {
        self.modelsDirectory = modelsDirectory
        self.cacheDirectory = cacheDirectory
    }

    private func loadedStages() async throws -> LiveStages {
        if let stages { return stages }
        // The learned hit detector replaces the rule detector on the hit stage; loading it here
        // (once, compiled into the caches directory) keeps the first match from paying the
        // compile. A failure is not fatal: the stage falls back to the ported rules.
        _ = await LiveStages.loadHitModel(cacheDirectory: cacheDirectory)
        let s = try await LiveStages.make(modelsDirectory: modelsDirectory, cacheDirectory: cacheDirectory)
        stages = s
        return s
    }

    nonisolated func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                             progress: @escaping @Sendable (AnalysisProgress) -> Void,
                             stopOnRetryableFailure: Bool) async throws -> MatchAnalysis {
        let stages = try await loadedStages()
        // Per-stage peak memory and thermal state land in diagnostics["stages"], so a slow or
        // killed run on a user's phone can be explained from the saved analysis.
        return try await AnalysisPipeline(stages: stages, checkpoints: checkpoints, observer: ResourceSampler(),
                                          stopOnRetryableFailure: stopOnRetryableFailure)
            .analyze(video, options: options, progress: progress)
    }
}

enum OnDeviceAnalyzers {
    /// Settings key to force the path: true = always on this phone, false = always the server.
    static let preferenceKey = "shuttl.analyzeOnDevice"

    /// The analyzer the app uses, or nil to send matches to the server pipeline.
    /// A configured HTTPS deployment offloads work without loading the phone's ML models.
    static func makeDefault(defaults: UserDefaults = .standard) -> (any OnDeviceAnalyzer)? {
        if CloudConfiguration.isEnabled { return nil }
        #if DEBUG
        let wanted = defaults.object(forKey: preferenceKey) as? Bool ?? LiveStages.isComplete
        #else
        let wanted = true
        #endif
        guard wanted, let models = Bundle.main.resourceURL else { return nil }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return LiveAnalyzer(modelsDirectory: models, cacheDirectory: caches.appendingPathComponent("CompiledModels"))
    }
}

/// Full analyses live under Documents/Analyses (JSON + a compact players side file, see
/// `AnalysisArchive`); resumable checkpoints under Application Support/AnalysisCheckpoints.
/// `load` returns the analysis with the user's score corrections applied.
enum AnalysisStore {
    static let folderName = "Analyses"

    static func archive(documents: URL = VideoStore.documentsDirectory) -> AnalysisArchive {
        AnalysisArchive.standard(documents: documents)
    }

    /// Saves (keypoints dropped: nothing in the library reads them) and returns the
    /// Documents-relative path to persist on the match.
    static func save(_ analysis: MatchAnalysis, matchID: UUID, documents: URL = VideoStore.documentsDirectory) throws -> String {
        let url = try archive(documents: documents).save(analysis, matchID: matchID)
        return VideoStore.storedPath(for: url, documents: documents)
    }

    /// As analysed, without the user's corrections.
    static func loadOriginal(_ storedPath: String?, documents: URL = VideoStore.documentsDirectory) -> MatchAnalysis? {
        guard let url = VideoStore.resolve(storedPath, documents: documents) else { return nil }
        return try? AnalysisArchive.load(analysisAt: url)
    }

    static func load(_ storedPath: String?, documents: URL = VideoStore.documentsDirectory) -> MatchAnalysis? {
        guard let url = VideoStore.resolve(storedPath, documents: documents),
              let analysis = try? AnalysisArchive.load(analysisAt: url) else { return nil }
        return ScoreCorrections.load(forAnalysisAt: url).applied(to: analysis)
    }

    static func delete(_ storedPath: String?, documents: URL = VideoStore.documentsDirectory) {
        guard let url = VideoStore.resolve(storedPath, documents: documents) else { return }
        let fm = FileManager.default
        for file in [url, AnalysisArchive.playersURL(forAnalysis: url), ScoreCorrections.url(forAnalysisAt: url)] {
            try? fm.removeItem(at: file)
        }
    }

    static func checkpointDirectory(matchID: UUID) -> URL {
        archive().checkpointDirectory(matchID: matchID)
    }

    /// Keyed to the exact video file, app build and analysis options, so a new recording,
    /// an updated app or re-tapped court corners never reuse stale intermediate results.
    static func checkpoints(matchID: UUID, video: URL, options: PipelineOptions) -> CheckpointStore {
        archive().checkpointStore(matchID: matchID, video: video, build: AnalysisArchive.build(), options: options)
    }
}

/// Stage names from `AnalysisPipeline` → what the user reads.
enum AnalysisStageName {
    static func display(_ stage: String) -> String {
        switch stage {
        case "detect": return "识别羽毛球"
        case "link": return "连接飞行轨迹"
        case "calibrate": return "标定球场"
        case "players": return "追踪球员"
        case "hits": return "判断击球"
        case "reconstruct3d": return "计算落点和球速"
        case "rallies": return "划分回合"
        case "shots": return "识别球路"
        case "score": return "计分"
        case "done": return "整理结果"
        case "queued": return "排队中，前一场分析完就开始"
        case "recording": return "正在录像，录完接着分析"
        case "camera": return "相机开着 · 离开录像页或相机休息后就开始"
        case "cooling": return "降温中 · 手机凉下来就开始"
        case "cooling_camera": return "降温中 · 手机凉下来就开始 · 录像页相机也在发热"
        default: return "分析中"
        }
    }
}

/// MatchAnalysis → the app's SwiftData rows (same mapping as the server's /analysis
/// document: hit position normalised to the upright frame, hitter position in metres,
/// launch speed only when a 3-D flight was fitted).
enum MatchAnalysisMapper {
    @MainActor
    static func apply(_ analysis: MatchAnalysis, to match: Match) {
        if let context = match.modelContext { match.rallies.forEach(context.delete) }
        match.rallies = []
        let w = Double(max(analysis.meta.width, 1)), h = Double(max(analysis.meta.height, 1))
        for r in analysis.rallies {
            let rally = Rally(startFrame: r.startFrame, endFrame: max(r.endFrame, r.startFrame),
                              winner: PlayerSide(rawValue: r.winner.rawValue) ?? .unknown)
            for s in r.shots.sorted(by: { $0.hit.frame < $1.hit.frame }) {
                let nx = finite(s.hit.xyPx.x / w).map { Float(min(max($0, 0), 1)) } ?? 0.5
                let ny = finite(s.hit.xyPx.y / h).map { Float(min(max($0, 0), 1)) } ?? 0.5
                let pos = s.hitterPosWorld
                rally.shots.append(ShotPoint(
                    frameIndex: s.hit.frame,
                    xPx: nx, yPx: ny,
                    normalizedX: nx, normalizedY: ny,
                    xMeters: Float(pos.flatMap { finite($0.x) } ?? 0),
                    yMeters: Float(pos.flatMap { finite($0.y) } ?? 0),
                    speedKmh: Float(s.segment.flatMap { finite($0.speed0Kmh) } ?? 0),
                    shotType: ShotType(rawValue: s.shotType.legacy.rawValue) ?? .unknown
                ))
            }
            match.rallies.append(rally)
        }
    }

    private static func finite(_ x: Double) -> Double? { x.isFinite ? x : nil }
}
