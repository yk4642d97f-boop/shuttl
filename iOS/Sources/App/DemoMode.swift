#if DEBUG
import SwiftUI
import SwiftData

/// Debug-only launch flags for visual QA without the backend:
///   -UIDemoMode              skip sign-in, use an in-memory store seeded with sample matches
///   -UIDemoTab <tab>         open on record | matches | statistics
///   -UIDemoEmpty             seed nothing (to review empty states)
///   -UIDemoOpenMatch         push the newest finished match on the matches tab
///   -UIDemoOpenReplay        …and then its replay screen (needs -UIDemoOpenMatch)
///   -UIDemoTrajectory        feed a synthetic shuttle trail into camera / replay overlays
///   -UIDemoRecording         show the camera screen in its recording state (no real capture)
///   -UIDemoLoginErrors       open sign-up with invalid input and a server error showing
///   -UIDemoScrollBottom      start scrolling screens at their bottom edge
///   -UIDemoOpenStatus <s>    with -UIDemoOpenMatch: open the first match in status s (failed | analyzing)
///   -UIDemoSaveNotice        show the "已保存，正在手机上分析" slip on the record screen
///   -UIDemoResultsTab <tab>  open on-device results on overview | court | rallies | movement
///   -UIDemoLandscape         rotate the camera / replay screens to landscape
///   -UIDemoReplayAt <s>      replay opens at s seconds (bands, ticks and the rally chip mid-match)
///   -UIDemoOpenAbout         with -UIDemoTab statistics: open 关于与致谢
///   -UIDemoUpright           camera sees the phone upright (shows the rotate prompt)
///   -UIDemoUncalibrated      the seeded analysis has no court camera (manual-corners card)
///   -UIDemoOpenCalibration   with -UIDemoOpenMatch: open the corner-drag screen
///   -UIDemoCorrectRally <n>  with -UIDemoResultsTab rallies: open the score-fix slip for rally n
///   -UIDemoSystemChips       show sample storage / heat chips
///   -UIDemoServerMatch       with -UIDemoOpenMatch: open a server-analysed match (legacy sections)
///   -UIDemoRallyDetail <n>   with -UIDemoResultsTab rallies: open rally n's detail sheet
///   -UIDemoImprecise         the seeded court was found, but seen too flat (one caution line)
///   -UIDemoNotRetryable      the failed demo match failed for good (no retry, re-record tips + delete)
///   -UIDemoConfirmDelete     with -UIDemoOpenMatch: open the delete confirmation
///   -UIDemoCautions          all three caution lines: phone bumped, landings withdrawn, far end flat
///   -UIDemoLaunchCheck       with -UIDemoTab record: apply the real launch-tab rule (opens 比赛 while analysing)
///   -UIDemoCameraSleep <r>   record screen with the camera asleep: idle | analysis
///   -UIDemoCourtLock <p>     record screen court lock in phase p (candidate | locked | lost)
enum DemoMode {
    private static let args = ProcessInfo.processInfo.arguments

    static var isOn: Bool { args.contains("-UIDemoMode") }
    static var isEmpty: Bool { args.contains("-UIDemoEmpty") }
    static var opensMatch: Bool { isOn && args.contains("-UIDemoOpenMatch") }
    static var opensReplay: Bool { opensMatch && args.contains("-UIDemoOpenReplay") }
    static var showsTrajectory: Bool { isOn && args.contains("-UIDemoTrajectory") }
    static var fakesRecording: Bool { isOn && args.contains("-UIDemoRecording") }
    static var showsLoginErrors: Bool { isOn && args.contains("-UIDemoLoginErrors") }
    static var showsSaveNotice: Bool { isOn && args.contains("-UIDemoSaveNotice") }
    static var forcesLandscape: Bool { isOn && args.contains("-UIDemoLandscape") }
    /// With -UIDemoTab statistics: open 关于与致谢.
    static var opensAbout: Bool { isOn && args.contains("-UIDemoOpenAbout") }
    /// Replay opens this many seconds in (the simulator has no real video to scrub).
    static var replayAtSeconds: Double? {
        guard isOn, let i = args.firstIndex(of: "-UIDemoReplayAt"), i + 1 < args.count else { return nil }
        return Double(args[i + 1])
    }
    /// Pretend the phone is held upright with motion available (the simulator has none).
    static var forcesUpright: Bool { isOn && args.contains("-UIDemoUpright") }
    /// The seeded on-device analysis has no court (shows the manual-corners card).
    static var uncalibrated: Bool { isOn && args.contains("-UIDemoUncalibrated") }
    /// Open the corner-drag screen on the match detail page.
    static var opensCalibration: Bool { opensMatch && args.contains("-UIDemoOpenCalibration") }
    /// With -UIDemoOpenMatch: open a finished match that was analysed on the server (no analysis file).
    static var opensServerMatch: Bool { opensMatch && args.contains("-UIDemoServerMatch") }
    /// With -UIDemoResultsTab rallies: open the detail sheet for this rally number.
    static var openRallyDetail: Int? {
        guard isOn, let i = args.firstIndex(of: "-UIDemoRallyDetail"), i + 1 < args.count else { return nil }
        return Int(args[i + 1])
    }
    /// The seeded court was found from a low / far tripod (far end seen too flat).
    static var imprecise: Bool { isOn && args.contains("-UIDemoImprecise") }
    /// Open the "删除这场比赛？" confirmation on the match detail page.
    static var confirmsDelete: Bool { opensMatch && args.contains("-UIDemoConfirmDelete") }
    /// The failed demo match can't be fixed by retrying.
    static var failureNotRetryable: Bool { isOn && args.contains("-UIDemoNotRetryable") }
    /// The seeded analysis carries every "results are shown, but…" condition.
    static var cautions: Bool { isOn && args.contains("-UIDemoCautions") }
    /// Record screen: the camera asleep, for this reason.
    static var cameraSleep: RecordingViewModel.Sleep? {
        guard isOn, let i = args.firstIndex(of: "-UIDemoCameraSleep"), i + 1 < args.count else { return nil }
        switch args[i + 1] {
        case "idle": return .idle
        case "analysis": return .analysisPending
        default: return nil
        }
    }
    /// Record screen: a found court outline in this phase (the simulator camera sees nothing).
    static var courtLock: CourtLockDisplay? {
        guard isOn, let i = args.firstIndex(of: "-UIDemoCourtLock"), i + 1 < args.count else { return nil }
        let outline = [CGPoint(x: 0.33, y: 0.36), CGPoint(x: 0.67, y: 0.36), CGPoint(x: 0.88, y: 0.8), CGPoint(x: 0.12, y: 0.8)]
        switch args[i + 1] {
        case "candidate":
            return CourtLockDisplay(phase: .candidate, title: "找到了，别动…", message: "再稳住两秒",
                                    outline: outline, imageSize: CGSize(width: 1920, height: 1080))
        case "locked":
            return CourtLockDisplay(phase: .locked, title: "球场锁定", message: "",
                                    outline: outline, imageSize: CGSize(width: 1920, height: 1080))
        case "lost":
            return CourtLockDisplay(phase: .lost, title: "球场线丢了", message: "手机动了？放回原位",
                                    outline: outline, imageSize: CGSize(width: 1920, height: 1080))
        default: return nil
        }
    }

    /// Sample storage / heat lines on the record HUD and the match list.
    static var showsSystemChips: Bool { isOn && args.contains("-UIDemoSystemChips") }

    static var openStatus: MatchStatus? {
        guard isOn, let i = args.firstIndex(of: "-UIDemoOpenStatus"), i + 1 < args.count else { return nil }
        return MatchStatus(rawValue: args[i + 1])
    }

    static var initialTab: AppTab? {
        guard isOn, let i = args.firstIndex(of: "-UIDemoTab"), i + 1 < args.count else { return nil }
        switch args[i + 1] {
        case "record":     return .record
        case "matches":    return .matches
        case "statistics": return .statistics
        default:           return nil
        }
    }

    @MainActor
    static func makeContainer(schema: Schema) -> ModelContainer {
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        do {
            let container = try ModelContainer(for: schema, configurations: [config])
            if !isEmpty { seed(into: container.mainContext) }
            return container
        } catch {
            fatalError("Demo 容器初始化失败: \(error)")
        }
    }

    /// Normalised points of a smash-like arc, oldest → newest.
    static func syntheticTrail(count: Int = 24) -> [CGPoint] {
        (0..<count).map { i in
            let t = Double(i) / Double(count - 1)
            return CGPoint(x: 0.18 + 0.58 * t, y: 0.62 - 0.34 * sin(t * .pi * 0.9))
        }
    }

    // MARK: - Seed data

    private struct SeededRNG: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    @MainActor
    private static func seed(into context: ModelContext) {
        var rng = SeededRNG(state: 20260926)
        // A fixed evening, so every launch (and every screenshot) shows the same dates and times.
        let now = Calendar.current.date(from: DateComponents(year: 2026, month: 9, day: 28, hour: 21, minute: 30)) ?? .now
        let plans: [(daysAgo: Double, minutes: Double, status: MatchStatus, court: CourtType, rallies: Int)] = [
            (0.1, 42, .analyzing, .singles, 0),
            (0.9, 38, .done,      .singles, 14),
            (2.2, 55, .done,      .doubles, 18),
            // Release only analyses on the phone, so the second live match is queued behind the first.
            (3.0, 12, .analyzing, .doubles, 0),
            (5.4, 47, .done,      .singles, 11),
            (8.0, 29, .failed,    .singles, 0),
        ]

        var seededAnalysis = false
        for plan in plans {
            let match = Match(
                createdAt: now.addingTimeInterval(-plan.daysAgo * 86_400),
                durationSeconds: plan.minutes * 60 + Double(Int.random(in: 0..<60, using: &rng)),
                status: plan.status,
                // Finished demo matches point at a placeholder file so the replay entry is reachable.
                localVideoPath: plan.status == .done ? NSTemporaryDirectory() + "demo.mov" : nil,
                courtType: plan.court,
                // The phone analyses on its own now, so a failure reads like one from ShuttlAnalysis.
                failureReason: plan.status == .failed
                    ? "没有看到羽毛球。请确认整片球场都在画面里，光线充足。" : nil,
                pipelineStep: plan.status == .failed || plan.status == .analyzing ? .local : nil
            )
            // "没看到羽毛球" is a verdict on the video: the real pipeline marks it not retryable.
            if plan.status == .failed { match.failureRetryable = !failureNotRetryable }
            context.insert(match)

            // The newest finished match carries a full on-device analysis, so the
            // results screens have real-shaped data to show.
            if plan.status == .done && !seededAnalysis {
                seededAnalysis = true
                var analysis = DemoAnalysis.make(rallies: plan.rallies)
                if (imprecise || cautions) && !uncalibrated {
                    analysis.diagnostics["calibration_auto"] = .object(["reason": .string("ok"), "far_m_per_px": .number(0.31)])
                }
                if cautions && !uncalibrated {
                    // Real shapes, read back by the same code as on the phone: 2 of 10 sampled
                    // frames moved (→ "moved"), and two rallies withdrawn by the landing guard.
                    analysis.diagnostics["camera_motion"] = .object([
                        "shifts_px": .array([0, 0, 0, 24, 0, 0, 31, 0, 0, 0].map { .number($0) }),
                        "correlations": .array([0.97, 0.98, 0.97, 0.52, 0.98, 0.97, 0.49, 0.98, 0.97, 0.98].map { .number($0) }),
                        "image_width": .number(1920), "search_px": .number(48),
                    ])
                    for r in [4, 9] where r < analysis.rallies.count {
                        analysis.rallies[r].outcomeSource = "implausible_landing"
                        analysis.rallies[r].winner = .unknown
                        analysis.rallies[r].landingXY = nil
                    }
                }
                if uncalibrated {
                    analysis.camera = nil
                    // No court, no 3-D flights: the real pipeline has no speeds or landings either.
                    for r in analysis.rallies.indices {
                        for k in analysis.rallies[r].shots.indices { analysis.rallies[r].shots[k].segment = nil }
                    }
                    analysis.diagnostics["calibration_auto"] = .object([
                        "reason": .string("evidence_gate"), "n_t": .number(2), "n_l": .number(5), "n_absent": .number(0),
                    ])
                }
                if let path = try? AnalysisStore.save(analysis, matchID: match.id) {
                    match.analysisPath = path
                    // The clip is as long as its analysis, so the replay timeline and the bands agree.
                    match.durationSeconds = Double(analysis.meta.nFrames) / max(analysis.meta.fps, 1)
                    // One rally already fixed by the user, so the "已改" state is on screen.
                    if let url = VideoStore.resolve(path), !analysis.rallies.isEmpty {
                        var fixes = ScoreCorrections()
                        fixes.fixes[analysis.rallies[0].startFrame] = analysis.rallies[0].winner == .near ? .farWon : .nearWon
                        try? fixes.save(forAnalysisAt: url)
                    }
                    MatchAnalysisMapper.apply(AnalysisStore.load(path) ?? analysis, to: match)
                    continue
                }
            }

            var frame = 0
            for _ in 0..<plan.rallies {
                let length = Int.random(in: 90...600, using: &rng)
                let winner: PlayerSide = [.near, .far, .near, .far, .unknown].randomElement(using: &rng)!
                let rally = Rally(startFrame: frame, endFrame: frame + length, winner: winner)
                match.rallies.append(rally)

                for s in 0..<Int.random(in: 3...12, using: &rng) {
                    let type: ShotType = s == 0
                        ? .serve
                        : [.smash, .clear, .clear, .drop, .net, .drive, .smash, .unknown].randomElement(using: &rng)!
                    let speed: Float = switch type {
                    case .smash:   .random(in: 180...352, using: &rng)
                    case .clear:   .random(in: 110...190, using: &rng)
                    case .drive:   .random(in: 90...160, using: &rng)
                    case .drop:    .random(in: 40...95, using: &rng)
                    case .net:     .random(in: 15...55, using: &rng)
                    case .serve:   .random(in: 30...80, using: &rng)
                    case .unknown: .random(in: 20...140, using: &rng)
                    }
                    let nx = Float.random(in: 0.08...0.92, using: &rng)
                    let ny = Float.random(in: 0.15...0.85, using: &rng)
                    rally.shots.append(ShotPoint(
                        frameIndex: frame + s * 24,
                        xPx: nx * 1920,
                        yPx: ny * 1080,
                        normalizedX: nx,
                        normalizedY: ny,
                        xMeters: (s.isMultiple(of: 2) ? -1 : 1) * Float.random(in: 0.6...6.4, using: &rng),
                        yMeters: Float.random(in: -2.8...2.8, using: &rng),
                        speedKmh: speed,
                        shotType: type
                    ))
                }
                frame += length + Int.random(in: 60...240, using: &rng)
            }
        }
        try? context.save()
    }
}
extension View {
    /// Starts scroll views at the bottom when `-UIDemoScrollBottom` is passed.
    @ViewBuilder func demoScrollAnchor() -> some View {
        if DemoMode.isOn && ProcessInfo.processInfo.arguments.contains("-UIDemoScrollBottom") {
            // Stay pinned while results load in after the first layout.
            if #available(iOS 18.0, *) {
                defaultScrollAnchor(.bottom).defaultScrollAnchor(.bottom, for: .sizeChanges)
            } else {
                defaultScrollAnchor(.bottom)
            }
        } else {
            self
        }
    }
}

// MARK: - Fake camera feed
// Simulators have no camera; this stands in for footage of an indoor court
// so overlay contrast can be judged against something realistic.

struct DemoCourtBackdrop: View {
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            ZStack {
                LinearGradient(colors: [Color(hex: 0x2A2630), Color(hex: 0x3B3440), Color(hex: 0x1E4A45)],
                               startPoint: .top, endPoint: .bottom)
                Path { p in
                    p.move(to: CGPoint(x: w * 0.24, y: h * 0.3))
                    p.addLine(to: CGPoint(x: w * 0.76, y: h * 0.3))
                    p.addLine(to: CGPoint(x: w * 1.02, y: h * 0.72))
                    p.addLine(to: CGPoint(x: w * -0.02, y: h * 0.72))
                    p.closeSubpath()
                }
                .fill(Color(hex: 0x2F6B5E))
                Path { p in
                    let tl = CGPoint(x: w * 0.26, y: h * 0.31), tr = CGPoint(x: w * 0.74, y: h * 0.31)
                    let bl = CGPoint(x: w * 0.02, y: h * 0.71), br = CGPoint(x: w * 0.98, y: h * 0.71)
                    p.move(to: tl); p.addLine(to: tr); p.addLine(to: br); p.addLine(to: bl); p.closeSubpath()
                    p.move(to: CGPoint(x: w * 0.14, y: h * 0.51)); p.addLine(to: CGPoint(x: w * 0.86, y: h * 0.51))
                    p.move(to: CGPoint(x: w * 0.5, y: h * 0.31)); p.addLine(to: CGPoint(x: w * 0.5, y: h * 0.71))
                }
                .stroke(Color.white.opacity(0.55), lineWidth: 2)
                Rectangle().fill(Color.white.opacity(0.25)).frame(width: w * 0.74, height: 36)
                    .position(x: w / 2, y: h * 0.49)
                RadialGradient(colors: [.clear, .black.opacity(0.55)], center: .center, startRadius: w * 0.3, endRadius: h * 0.7)
            }
            .blur(radius: 1.2)
        }
        .allowsHitTesting(false)
    }
}
#endif
