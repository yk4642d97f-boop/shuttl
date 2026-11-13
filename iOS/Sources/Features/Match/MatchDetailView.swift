import SwiftUI
import SwiftData

// MARK: - Hero block
// The single loud thing on the page: top speed printed huge on tomato over a faint
// court, or an in-progress / failure panel in its status ink.

private struct HeroBlock: View {
    let match: Match
    var activity: PipelineActivity?
    /// Speed from the on-device analysis when there is one (fastest smash, else fastest shot).
    var analysedSpeed: (kmh: Double, isSmash: Bool)?
    /// On-device result without a court: no speed exists, so the poster prints the shot count.
    var shotsWithoutSpeed: Int?
    /// Rallies the user re-called; shown as a seal on the poster.
    var correctionCount = 0
    var onRetry: (() -> Void)?
    /// Offered instead of 重试 when retrying would fail the same way.
    var onDelete: (() -> Void)?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        switch match.status {
        case .done:      speedPoster
        case .analyzing:
            if let waiting = MatchWaiting(localStage: activity?.localStage) {
                // Waiting its turn: say so, and why; no spinning / pulsing "busy" signals.
                working(title: "等一下再开始分析", note: waitingNote(waiting), still: waiting.symbol)
            } else if match.pipelineStep == .local {
                working(title: "正在这台手机上分析", note: "识别羽毛球、分割回合、计算球速，全程不联网。分析时请让 Shuttl 保持打开：锁屏或切到别的 App 会暂停，回来后会接着分析。")
            } else {
                working(title: "AI 正在逐帧追踪", note: "识别羽毛球、分割回合、计算球速。通常需要几分钟，可以先离开这个页面。")
            }
        case .uploading: working(title: "正在上传视频", note: "上传完成后会自动开始分析。可以切到后台；网络中断后可回来重试。")
        case .recording: working(title: "正在录制", note: "结束录制后会自动上传。")
        case .failed:    failed
        }
    }

    private var speedText: String {
        if let analysedSpeed { return String(Int(analysedSpeed.kmh.rounded())) }
        if let shotsWithoutSpeed { return "\(shotsWithoutSpeed)" }
        return match.maxSpeedText
    }

    private var speedTitle: String {
        if let analysedSpeed { return analysedSpeed.isSmash ? "本场最快扣杀" : "本场最快球速" }
        if shotsWithoutSpeed != nil { return "本场击球 · 标定后才有球速" }
        return "本场最高球速"
    }

    private var speedUnit: String { analysedSpeed == nil && shotsWithoutSpeed != nil ? "拍" : "km/h" }

    /// Without a court there is no speed to shout about: the poster goes quiet (paper, ink
    /// numerals) and the calibration card below becomes the page's one loud block.
    private var uncalibrated: Bool { analysedSpeed == nil && shotsWithoutSpeed != nil }

    private var speedPoster: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        return VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: AppSpacing.sm) {
                Text(uncalibrated ? "本场击球" : speedTitle)
                    .instrumentLabel(AppFont.label, color: .ink)
                if uncalibrated {
                    Sticker("标定后才有球速", symbol: "viewfinder", fill: .butter, tilt: -3)
                }
            }
            HStack(alignment: .firstTextBaseline, spacing: AppSpacing.sm) {
                if uncalibrated {
                    Text(speedText)
                        .font(AppFont.hero)
                        .foregroundStyle(Color.ink)
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)
                } else {
                    OverprintText(speedText, font: AppFont.hero, plate: .butter, offset: CGSize(width: 5, height: 4))
                }
                Text(speedUnit)
                    .font(AppFont.unit)
                    .foregroundStyle(Color.ink)
            }
            Text("\(match.courtLabel) · \(match.durationText) · \(match.rallies.count) 回合")
                .font(AppFont.callout.weight(.semibold))
                .foregroundStyle(Color.ink)
        }
        .padding(AppSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            ZStack {
                uncalibrated ? Color.card : Color.tomato
                CourtWatermark(color: .ink.opacity(uncalibrated ? 0.08 : 0.15))
                if !uncalibrated {
                    Halftone(color: .ink.opacity(0.2), spacing: 9, maxDot: 3.6, focus: .bottomTrailing)
                }
            }
            .clipShape(shape)
        }
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .overlay(alignment: .topTrailing) {
            if correctionCount > 0 {
                Seal(value: "\(correctionCount)", caption: "已改", fill: .butter, size: 74, tilt: 12, captionOnTop: true)
                    .offset(x: -14, y: -24)
                    .accessibilityLabel("你改判了 \(correctionCount) 分")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(speedTitle) \(speedText) \(speedUnit == "拍" ? "拍" : "公里每小时")，\(match.courtLabel)，\(match.rallies.count) 回合")
    }

    private func waitingNote(_ waiting: MatchWaiting) -> String {
        switch waiting {
        case .queued: return "一次只分析一场，前一场好了就轮到它。"
        case .cooling: return "手机太热时先不开始，凉下来会自动接着分析。"
        case .camera: return "录像页的相机和分析不能同时用，相机关掉就开始。"
        case .recording: return "正在录新的一场，录完会自动接着分析。"
        }
    }

    /// - still: a waiting hero shows this glyph, without the spin and pulse of real work.
    private func working(title: String, note: String, still: String? = nil) -> some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        return HStack(alignment: .top, spacing: AppSpacing.md) {
            ZStack {
                if let still {
                    Scallop(lobes: 12, depth: 0.12)
                        .fill(Color.card)
                        .overlay(Scallop(lobes: 12, depth: 0.12).strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                    Image(systemName: still)
                        .font(.system(size: 22, weight: .black))
                        .foregroundStyle(Color.ink)
                } else {
                    Scallop(lobes: 12, depth: 0.12)
                        .fill(Color.card)
                        .overlay(Scallop(lobes: 12, depth: 0.12).strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                        .slowSpin(period: 10)
                    Image(systemName: match.status.symbol)
                        .font(.system(size: 22, weight: .black))
                        .foregroundStyle(Color.ink)
                        .symbolEffect(.pulse, isActive: !reduceMotion)
                }
            }
            .frame(width: 64, height: 64)
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: AppSpacing.sm) {
                Text(title)
                    .font(AppFont.title2)
                    .foregroundStyle(Color.ink)
                Text(note)
                    .font(AppFont.callout)
                    .foregroundStyle(Color.ink)
                    .fixedSize(horizontal: false, vertical: true)
                progress
                    .padding(.top, AppSpacing.xs)
            }
        }
        .padding(AppSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            ZStack {
                match.status.ink
                CourtWatermark(color: .ink.opacity(0.1))
            }
            .clipShape(shape)
        }
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .accessibilityElement(children: .combine)
    }

    /// Live status line plus a determinate ink bar when the percentage is known.
    @ViewBuilder
    private var progress: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(match.progressText(activity))
                .instrumentLabel(AppFont.label, color: .ink)
                .contentTransition(.numericText())
            if let fraction = match.progressFraction(activity) {
                InkProgressBar(value: fraction, fill: .ink, track: .card, height: 12)
            } else if MatchWaiting(localStage: activity?.localStage) == nil {
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(.ink)
            }
        }
        .animation(AppAnimation.standard, value: activity)
    }

    private var failed: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        return VStack(alignment: .leading, spacing: AppSpacing.md) {
            VStack(alignment: .leading, spacing: AppSpacing.sm) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 28, weight: .bold))
                Text("这场没能分析出来")
                    .font(AppFont.title2)
                Text(match.failureReason ?? "常见原因：画面没拍全场地、光线太暗，或视频太短。")
                    .font(AppFont.callout)
                    .fixedSize(horizontal: false, vertical: true)
                if match.retryIsPointless {
                    Text("这段视频再分析一次结果也一样。照下面的方法重新拍一段就好。")
                        .font(AppFont.callout.weight(.semibold))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .accessibilityElement(children: .combine)

            if match.retryIsPointless {
                if let onDelete {
                    PillButton(title: "删除这场", icon: "trash", style: .secondary, action: onDelete)
                        .accessibilityHint("删除这段视频和它的记录，然后照提示重新拍")
                }
            } else if let onRetry {
                VStack(alignment: .leading, spacing: 6) {
                    PillButton(title: "重试", icon: "arrow.clockwise", style: .secondary, action: onRetry)
                    Text(match.retryHint)
                        .font(AppFont.caption)
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .foregroundStyle(Color.ink)
        .padding(AppSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.tomato, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .accessibilityElement(children: .contain)
    }
}

// MARK: - Rally scorecard row (server-analysed matches)

private struct RallyRow: View {
    let index: Int
    let rally: Rally

    private var winner: (label: String, fill: Color, text: Color) {
        switch rally.winner {
        case .near:    return ("近端得分", .cobalt, .onCobalt)
        case .far:     return ("远端得分", .flamingo, .onFlamingo)
        case .unknown: return ("未判定", .paperDeep, .inkSoft)
        }
    }

    private var orderedShots: [ShotPoint] { rally.shots.sorted { $0.frameIndex < $1.frameIndex } }

    var body: some View {
        HStack(spacing: AppSpacing.md) {
            Text(String(format: "%02d", index))
                .font(AppFont.numeral(28, relativeTo: .title2))
                .foregroundStyle(Color.ink)
                .monospacedDigit()
                .frame(width: 40, alignment: .leading)

            VStack(alignment: .leading, spacing: 6) {
                shotBeads
                HStack(spacing: AppSpacing.sm) {
                    Text("\(rally.shots.count) 拍")
                    if let top = rally.shots.map(\.speedKmh).max(), top > 0 {
                        Text("·")
                        Text("最快 \(Int(top)) km/h")
                    }
                }
                .font(AppFont.caption)
                .foregroundStyle(Color.inkSoft)
                .monospacedDigit()
            }

            Spacer(minLength: 0)

            Sticker(winner.label, fill: winner.fill, foreground: winner.text, tilt: 2)
        }
        .padding(.vertical, AppSpacing.sm + 2)
        .accessibilityElement(children: .combine)
    }

    /// One bead per shot, in its shot-type ink — the rally read left to right.
    private var shotBeads: some View {
        HStack(spacing: 3) {
            ForEach(Array(orderedShots.prefix(12).enumerated()), id: \.offset) { _, shot in
                Circle()
                    .fill(shot.shotType.ink)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1))
                    .frame(width: 11, height: 11)
            }
            if orderedShots.count > 12 {
                Text("+\(orderedShots.count - 12)")
                    .font(AppFont.labelSmall)
                    .foregroundStyle(Color.inkSoft)
            }
        }
        .accessibilityHidden(true)
    }
}

// MARK: - Shot legend
// Pictogram chips, so shot types read by shape as well as by ink.

struct ShotLegend: View {
    let shots: [ShotPoint]

    var body: some View {
        let entries = ShotType.allCases.compactMap { t -> (ShotType, Int)? in
            let n = shots.filter { $0.shotType == t }.count
            return n > 0 ? (t, n) : nil
        }
        FlowRow(spacing: AppSpacing.sm) {
            ForEach(entries, id: \.0) { type, count in
                HStack(spacing: 5) {
                    ShotChip(type.glyphKind, size: 22)
                    Text(type.displayName)
                        .font(AppFont.caption.weight(.semibold))
                        .foregroundStyle(Color.ink)
                    Text("\(count)")
                        .font(AppFont.labelSmall)
                        .foregroundStyle(Color.inkSoft)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }
}

// MARK: - MatchDetailView

struct MatchDetailView: View {
    let match: Match
    @Environment(MatchPipeline.self) private var pipeline: MatchPipeline?
    @Environment(AppChrome.self) private var chrome: AppChrome?
    @State private var results = OnDeviceResultsModel()
    @State private var replay: ReplayRoute?
    @State private var calibration: CalibrationFrame?
    @State private var loadingFrame = false
    @State private var frameError: String?
    @State private var correctionError: String?
    @State private var confirmingDelete = false
    @State private var lifecycleError: String?
    @State private var lifecycleBusy = false
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss

    private var smashCount: Int { match.allShots.filter { $0.shotType == .smash }.count }

    /// Analysed on this phone but the court wasn't found: no speeds or landings until corners are placed.
    private var needsCorners: Bool {
        match.status == .done && results.hasResults && !results.isCalibrated && match.localVideoURL != nil
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: AppSpacing.lg) {
                masthead

                HeroBlock(
                    match: match,
                    activity: pipeline?.activity(for: match),
                    analysedSpeed: results.heroSpeedKmh.map { ($0, results.heroIsSmash) },
                    shotsWithoutSpeed: results.hasResults && !results.isCalibrated ? results.totalShots : nil,
                    correctionCount: results.correctionCount,
                    onRetry: pipeline.map { pipeline in { withAnimation(AppAnimation.spring) { pipeline.retry(match) } } },
                    onDelete: { confirmingDelete = true }
                )

                if pipeline?.canCancelRemotely(match) == true {
                    Button("取消云端分析", role: .destructive) {
                        guard let pipeline else { return }
                        lifecycleBusy = true
                        Task {
                            defer { lifecycleBusy = false }
                            do { try await pipeline.cancelRemotely(match) }
                            catch { lifecycleError = MatchPipeline.failureReason(for: error) }
                        }
                    }.disabled(lifecycleBusy)
                }
                if match.status == .uploading || match.status == .analyzing {
                    Button("删除这场比赛", role: .destructive) { confirmingDelete = true }
                        .disabled(lifecycleBusy)
                }
                if needsCorners {
                    CalibrationPromptCard(isLoading: loadingFrame, advice: results.calibrationAdvice) { openCalibration() }
                    if let frameError {
                        Text(frameError).font(AppFont.caption).foregroundStyle(Color.tomatoText)
                    }
                }

                // Printed on paper, not a cobalt slab: the hero (or the calibration card) is
                // the page's one loud block.
                if match.canReplay {
                    NavigationLink(value: ReplayRoute(match: match)) {
                        HStack(spacing: AppSpacing.sm) {
                            Image(systemName: "play.fill")
                                .font(.system(size: 13, weight: .black))
                                .foregroundStyle(Color.onCobalt)
                                .frame(width: 28, height: 28)
                                .background(Color.cobalt, in: Circle())
                            Text("开始回放").font(DisplayFace.font(21, relativeTo: .headline))
                                .foregroundStyle(Color.ink)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 56)
                        .background(Color.card, in: Capsule())
                        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                    }
                    .buttonStyle(PrintButtonStyle(shape: Capsule(), plate: .ink, depth: 4))
                }

                if results.hasResults {
                    if let correctionError {
                        Text(correctionError).font(AppFont.caption).foregroundStyle(Color.tomatoText)
                    }
                    OnDeviceResultsSection(model: results, onReplay: { frame in
                        // Let a presenting sheet finish dismissing before pushing replay.
                        Task { @MainActor in
                            try? await Task.sleep(for: .milliseconds(350))
                            replay = ReplayRoute(match: match, startFrame: frame)
                        }
                    }, onCorrect: correct)
                } else if match.status == .analyzing || match.status == .uploading {
                    AnalysisStageLadder(match: match, activity: pipeline?.activity(for: match))
                    WhileYouWaitCard(onDevice: match.isOnDevice || pipeline?.analyzesOnDevice == true)
                } else if match.status == .failed {
                    ShootingTipsCard()
                } else {
                    legacySections
                }
            }
            .padding(.horizontal, AppSpacing.gutter)
            .padding(.bottom, AppSpacing.xl)
        }
        .tabBarClearance()
        .paperPage()
        #if DEBUG
        .demoScrollAnchor()
        #endif
        .navigationBarTitleDisplayMode(.inline)
        #if DEBUG
        .task {
            guard DemoMode.confirmsDelete else { return }
            try? await Task.sleep(for: .milliseconds(900))
            confirmingDelete = true
        }
        #endif
        .alert("操作未完成", isPresented: Binding(get: { lifecycleError != nil }, set: { if !$0 { lifecycleError = nil } })) {
            Button("知道了", role: .cancel) {}
        } message: { Text(lifecycleError ?? "") }
        .confirmationDialog("删除这场比赛？", isPresented: $confirmingDelete, titleVisibility: .visible) {
            Button("删除这场", role: .destructive) { deleteMatch() }
            Button("取消", role: .cancel) {}
        } message: {
            Text(pipeline?.hasCloudCopy(match) == true
                 ? "删除手机录像和记录，并请求云端删除。离线时将在联网并登录原账号后同步，云端文件随后清理。不能恢复。"
                 : "这段视频、记录和分析结果都会从这台手机上删掉，不能恢复。")
        }
        .toolbar {
            if results.hasResults && match.status == .done && match.localVideoURL != nil {
                ToolbarItem(placement: .topBarTrailing) { overflowMenu }
            }
        }
        .navigationDestination(item: $replay) { route in
            ReplayView(match: route.match, startFrame: route.startFrame)
        }
        .fullScreenCover(item: $calibration) { frame in
            CourtCalibrationSheet(frame: frame, onDone: { corners in
                calibration = nil
                pipeline?.reanalyze(match, corners: corners)
            }, onCancel: { calibration = nil })
        }
        .onChange(of: calibration != nil) { _, open in chrome?.calibrationOpen = open }
        .task(id: match.analysisPath) { await results.load(match.analysisPath) }
        // A re-analysis saves under the same path: read it again when the match settles.
        .onChange(of: match.status) { _, status in
            if status == .done { Task { await results.reload(match.analysisPath) } }
        }
        #if DEBUG
        .task {
            guard DemoMode.opensCalibration, calibration == nil else { return }
            try? await Task.sleep(for: .milliseconds(600))
            calibration = CalibrationFrame.demo()
        }
        #endif
    }

    private var overflowMenu: some View {
        Menu {
            Button { openCalibration() } label: {
                Label(results.isCalibrated ? "重新标定球场" : "手动标定球场", systemImage: "viewfinder")
            }
            NavigationLink { AcknowledgementsView() } label: {
                Label("关于与致谢", systemImage: "info.circle")
            }
            if results.hasCorrections {
                Button(role: .destructive) {
                    Task {
                        await pipeline?.clearCorrections(match)
                        await results.reload(match.analysisPath)
                    }
                } label: {
                    Label("恢复全部 AI 判断", systemImage: "arrow.uturn.backward")
                }
            }
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(Color.ink)
                .frame(width: 36, height: 36)
                .background(Color.card, in: Circle())
                .overlay(Circle().strokeBorder(Color.ink, lineWidth: AppStroke.fine))
        }
        .accessibilityLabel("更多")
    }

    private func correct(_ startFrame: Int, _ fix: ScoreCorrections.Fix?) {
        guard let pipeline else { return }
        Task {
            do {
                try await pipeline.correctRally(match, startFrame: startFrame, fix: fix)
                correctionError = nil
                await results.reload(match.analysisPath)
            } catch {
                correctionError = "没能保存这次改判：\(error.localizedDescription)"
            }
        }
    }

    /// Same as the list's swipe-to-delete: stop any work on it, then remove the row.
    private func deleteMatch() {
        guard let pipeline, !lifecycleBusy else { return }
        lifecycleBusy = true
        Task { @MainActor in
            defer { lifecycleBusy = false }
            do {
                try await pipeline.discard(match)
                dismiss()
                try? await Task.sleep(for: .milliseconds(350))
                modelContext.delete(match)
                try modelContext.save()
            } catch { lifecycleError = MatchPipeline.failureReason(for: error) }
        }
    }

    private func openCalibration() {
        guard let url = match.localVideoURL, !loadingFrame else { return }
        loadingFrame = true
        frameError = nil
        let initial = MatchInputs.load(matchID: match.id).cornerPoints
        Task {
            let still = await MatchPipeline.calibrationFrame(for: url)
            loadingFrame = false
            if let (image, _) = still {
                calibration = CalibrationFrame(image: image, initial: initial)
            } else {
                frameError = "读不出这段视频的画面，没法标定。视频可能已经被删掉了。"
            }
        }
    }

    private var masthead: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: AppSpacing.sm) {
                Text(match.createdAt.formatted(.dateTime.year().weekday(.wide).hour().minute()))
                    .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                if match.isOnDevice {
                    Sticker("手机端分析", symbol: "iphone", fill: .pistachio, tilt: -2)
                }
            }
            OverprintText(match.createdAt.formatted(.dateTime.month().day()),
                          font: AppFont.numeral(54, relativeTo: .largeTitle), plate: .cobalt,
                          offset: CGSize(width: 3.5, height: 3), blend: .multiply)
                .accessibilityAddTraits(.isHeader)
        }
        .padding(.top, AppSpacing.xs)
    }

    /// Sections for matches analysed on the server (no full analysis document).
    @ViewBuilder
    private var legacySections: some View {
        if match.status == .done {
            AppCard(padding: AppSpacing.md + 2) {
                StatRow(stats: [
                    .init(match.durationText, "时长"),
                    .init("\(match.rallies.count)", "回合"),
                    .init("\(match.allShots.count)", "击球"),
                    .init("\(smashCount)", "扣杀"),
                ], numeralFont: AppFont.numeral(28, relativeTo: .title2))
            }
        }

        if !match.allShots.isEmpty {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "击球构成", folio: "01")
                ShotMixBar(shots: match.allShots, height: 16)
                ShotLegend(shots: match.allShots)
            }
        }

        if !match.rallies.isEmpty {
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                SectionHeader(title: "回合记分卡", folio: match.allShots.isEmpty ? "01" : "02")
                    .padding(.bottom, AppSpacing.xs)
                ForEach(Array(match.rallies.enumerated()), id: \.element.id) { i, rally in
                    RallyRow(index: i + 1, rally: rally)
                    if i < match.rallies.count - 1 { AppDivider() }
                }
            }
        }
    }
}
