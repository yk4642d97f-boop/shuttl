import SwiftUI
import ShuttlCore
import ShuttlAnalysis
import ShuttlResultsUI

// The only file that imports ShuttlResultsUI: that package has its own `ShotPoint`
// (a landing stamp), which would collide with the app's SwiftData `ShotPoint`.

/// A match's full on-device analysis, loaded once from Documents/Analyses.
@MainActor
@Observable
final class OnDeviceResultsModel {
    private(set) var viewModel: MatchResultViewModel?
    private(set) var analysis: MatchAnalysis?
    private(set) var loadedPath: String?
    /// Rallies whose call the user changed (or dropped).
    private(set) var correctionCount = 0
    var hasCorrections: Bool { correctionCount > 0 }

    var hasResults: Bool { viewModel != nil }
    var isCalibrated: Bool { analysis?.camera != nil }
    /// Why automatic calibration gave up, in one sentence (nil when calibrated or never tried).
    var calibrationAdvice: String? { analysis.flatMap(CalibrationAdvice.message(for:)) }
    /// "Results are shown, but…" lines, in reading order: the phone was bumped, some
    /// landings were withdrawn, the far end was seen too flat. Empty when all is well.
    nonisolated static func cautions(_ analysis: MatchAnalysis) -> [String] {
        [analysis.steadinessWarning, analysis.landingWarning, CalibrationAdvice.precisionNote(for: analysis)]
            .compactMap { $0 }
    }

    func load(_ storedPath: String?) async {
        guard storedPath != loadedPath else { return }
        await reload(storedPath)
    }

    /// Reads the analysis again (after a correction or a re-analysis saved under the same path).
    func reload(_ storedPath: String?) async {
        loadedPath = storedPath
        guard let storedPath else { viewModel = nil; analysis = nil; correctionCount = 0; return }
        let loaded = await Task.detached(priority: .userInitiated) { () -> (MatchAnalysis, MatchResultViewModel, Int)? in
            guard let analysis = AnalysisStore.load(storedPath) else { return nil }
            let fixes = VideoStore.resolve(storedPath).map { ScoreCorrections.load(forAnalysisAt: $0).fixes.count } ?? 0
            return (analysis, MatchResultViewModel(analysis: analysis, cautions: Self.cautions(analysis)), fixes)
        }.value
        analysis = loaded?.0
        viewModel = loaded?.1
        correctionCount = loaded?.2 ?? 0
    }
}

/// Every results section (overview, landings, rallies, movement) for the detail page.
struct OnDeviceResultsSection: View {
    let model: OnDeviceResultsModel
    /// Asks the host to open replay at a frame.
    let onReplay: (Int) -> Void
    /// The user's call for the rally starting at a frame (nil fix = restore the AI's call).
    let onCorrect: (Int, ScoreCorrections.Fix?) -> Void

    var body: some View {
        if let viewModel = model.viewModel {
            MatchResultSections(viewModel: viewModel, showsHero: false, initialTab: initialTab,
                                initialCorrection: initialCorrection, initialDetail: initialDetail,
                                onReplay: onReplay) { frame, call in
                onCorrect(frame, Self.fix(for: call))
            }
        }
    }

    private static func fix(for call: RallyCall) -> ScoreCorrections.Fix? {
        switch call {
        case .nearWon: return .nearWon
        case .farWon: return .farWon
        case .notARally: return .notARally
        case .restoreAI: return nil
        }
    }

    private var initialTab: MatchResultTab {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-UIDemoResultsTab"), i + 1 < args.count,
           let tab = MatchResultTab(rawValue: args[i + 1]) { return tab }
        #endif
        return .overview
    }

    /// Demo: open this rally number's detail sheet.
    private var initialDetail: Int? {
        #if DEBUG
        return DemoMode.openRallyDetail
        #else
        return nil
        #endif
    }

    /// Demo: open the correction slip for this rally number.
    private var initialCorrection: Int? {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-UIDemoCorrectRally"), i + 1 < args.count { return Int(args[i + 1]) }
        #endif
        return nil
    }
}

/// The hero figure for a match analysed on the phone: fastest smash, else fastest shot.
extension OnDeviceResultsModel {
    var heroSpeedKmh: Double? {
        guard let summary = viewModel?.summary else { return nil }
        return summary.maxSmashKmh ?? summary.maxSpeedKmh
    }
    var heroIsSmash: Bool { viewModel?.summary.maxSmashKmh != nil }
    var totalShots: Int { viewModel?.summary.totalShots ?? 0 }
}
