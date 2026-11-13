import SwiftUI
import CoreMotion
import ShuttlRuntime

// The record screen's interface stays portrait (like Camera.app); when the phone is
// on its side, the framing overlay, HUD and control glyphs counter-rotate so they
// read upright in the user's landscape view. Tilt comes from gravity, not the
// interface orientation, so it works with rotation lock on.

// MARK: - Device tilt

@MainActor
@Observable
final class DeviceTilt {
    enum Pose: Equatable { case portrait, landscapeLeft, landscapeRight }

    private(set) var pose: Pose = .portrait
    /// False on the simulator / without motion hardware: recording is then never gated.
    private(set) var isAvailable = false
    private let motion = CMMotionManager()

    var isLandscape: Bool { pose != .portrait }

    /// Rotation that makes content upright for the holder (landscapeLeft = home side right).
    var angle: Angle {
        switch pose {
        case .portrait: return .zero
        case .landscapeLeft: return .degrees(90)
        case .landscapeRight: return .degrees(-90)
        }
    }

    func start() {
        #if DEBUG
        if DemoMode.forcesLandscape { pose = .landscapeLeft; isAvailable = true; return }
        if DemoMode.forcesUpright { pose = .portrait; isAvailable = true; return }
        #endif
        guard motion.isDeviceMotionAvailable, !motion.isDeviceMotionActive else { return }
        isAvailable = true
        motion.deviceMotionUpdateInterval = 1.0 / 15
        motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
            guard let g = data?.gravity else { return }
            MainActor.assumeIsolated { self?.update(x: g.x, y: g.y) }
        }
    }

    func stop() { motion.stopDeviceMotionUpdates() }

    /// Hysteresis: only switch once gravity clearly points along the new axis; lying
    /// flat keeps the last pose.
    private func update(x: Double, y: Double) {
        let next: Pose
        if abs(x) > 0.72 && abs(x) > abs(y) { next = x < 0 ? .landscapeLeft : .landscapeRight }
        else if abs(y) > 0.72 && abs(y) > abs(x) { next = .portrait }
        else { return }
        guard next != pose else { return }
        withAnimation(AppAnimation.spring) { pose = next }
    }
}

// MARK: - Upright frame
// Lays content out in the holder's frame: in landscape it gets the screen's height
// as its width and is rotated into place, so "top" is the holder's top.

struct UprightFrame<Content: View>: View {
    let angle: Angle
    @ViewBuilder let content: (CGSize) -> Content

    var body: some View {
        GeometryReader { geo in
            let sideways = angle != .zero
            let size = sideways ? CGSize(width: geo.size.height, height: geo.size.width) : geo.size
            content(size)
                .frame(width: size.width, height: size.height)
                .rotationEffect(angle)
                .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
    }
}

// MARK: - Court lock (display model)
// Four states of setting up the tripod shot. The analysis side maps its engine state
// into this; the view only draws it.

struct CourtLockDisplay: Equatable {
    enum Phase: Equatable { case searching, candidate, locked, lost }

    var phase: Phase
    var title: String
    var message: String
    /// Court outline, 0…1 of the upright camera frame (candidate / locked).
    var outline: [CGPoint]?
    /// Size of the frames the outline was measured in (for aspect-fill mapping).
    var imageSize: CGSize = .zero

    static let searching = CourtLockDisplay(phase: .searching, title: "正在找球场线…",
                                            message: "把四个角都框进来，画面稳住别动")

    var ink: Color {
        switch phase {
        case .searching: return .butter
        case .candidate: return .lilac
        case .locked: return .pistachio
        case .lost: return .tomato
        }
    }

    var symbol: String {
        switch phase {
        case .searching: return "viewfinder"
        case .candidate: return "hand.raised.fill"
        case .locked: return "lock.fill"
        case .lost: return "exclamationmark.triangle.fill"
        }
    }
}

// MARK: - Framing overlay
// Before a court is found: a dashed target court with butter brackets. Candidate:
// the found outline, dashed lilac. Locked: solid pistachio lines on an ink keyline.

struct FramingOverlay: View {
    let lock: CourtLockDisplay
    /// Room taken by the control deck on the holder's leading / trailing side.
    var leadingInset: CGFloat = 0
    var trailingInset: CGFloat = 0
    /// Space taken at the top by the HUD and any system chips above the sticker.
    var topInset: CGFloat = 0
    /// Space taken at the bottom by the record deck (portrait only; landscape uses the sides).
    var bottomInset: CGFloat = 0

    var body: some View {
        GeometryReader { geo in
            let area = CGRect(x: leadingInset, y: 0,
                              width: geo.size.width - leadingInset - trailingInset, height: geo.size.height)
            ZStack {
                if let outline = lock.outline, outline.count >= 4 {
                    found(outline, in: geo.size)
                } else {
                    // Keeps its true perspective; only the part under the record deck is cut off.
                    target(in: area)
                        .mask(alignment: .top) {
                            Rectangle().frame(height: max(0, geo.size.height - bottomInset))
                        }
                }
                let stickerY = max(56 + topInset, geo.size.height * 0.18)
                statusSticker
                    .position(x: area.midX, y: stickerY)
                // Searching: the spec card sits inside the target court, under its far edge and
                // between the far brackets. Once an outline shows up it shrinks to a corner tag;
                // locked, the frame is left clear.
                switch lock.phase {
                case .searching:
                    let farY = area.height * Self.farEdge
                    let width = area.width * (1 - 2 * Self.farInset) - 2 * (Self.bracketArm + 12)
                    CameraSetupNote(compact: true, maxWidth: min(240, max(190, width)))
                        .padding(.top, max(farY + 12, stickerY + 44))
                        .frame(width: area.width, height: area.height, alignment: .top)
                        .position(x: area.midX, y: area.midY)
                        .transition(.scale(scale: 0.6, anchor: .bottomTrailing).combined(with: .opacity))
                case .candidate, .lost:
                    // Bottom-leading corner of the holder's view: away from the HUD (top) and
                    // the control deck (trailing in landscape, bottom in portrait).
                    CameraSetupTag()
                        .padding(.leading, AppSpacing.sm)
                        .padding(.bottom, bottomInset + AppSpacing.md)
                        .frame(width: area.width, height: area.height, alignment: .bottomLeading)
                        .position(x: area.midX, y: area.midY)
                        .transition(.scale(scale: 1.4, anchor: .bottomTrailing).combined(with: .opacity))
                case .locked:
                    EmptyView()
                }
            }
            .animation(AppAnimation.spring, value: lock.phase)
        }
        .allowsHitTesting(false)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(lock.title)。\(lock.message)")
    }

    private var statusSticker: some View {
        VStack(spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: lock.symbol).font(.system(size: 13, weight: .black))
                Text(lock.title).font(DisplayFace.font(19, relativeTo: .headline))
            }
            .foregroundStyle(Color.ink)
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(lock.ink, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            .rotationEffect(.degrees(-3))
            if !lock.message.isEmpty {
                Text(lock.message)
                    .instrumentLabel(AppFont.labelSmall, color: .paper.opacity(0.9))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.night.opacity(0.6), in: Capsule())
            }
        }
        .animation(AppAnimation.bouncy, value: lock.phase)
    }

    /// Target-court geometry, shared by the dashed guide and the setup note.
    private static let farEdge: CGFloat = 0.34
    private static let farInset: CGFloat = 0.27
    private static let bracketArm: CGFloat = 16

    private func target(in r: CGRect) -> some View {
        let tl = CGPoint(x: r.minX + r.width * Self.farInset, y: r.height * Self.farEdge)
        let tr = CGPoint(x: r.maxX - r.width * Self.farInset, y: r.height * Self.farEdge)
        let bl = CGPoint(x: r.minX + r.width * 0.06, y: r.height * 0.88)
        let br = CGPoint(x: r.maxX - r.width * 0.06, y: r.height * 0.88)
        return ZStack {
            Path { p in
                p.move(to: tl); p.addLine(to: tr); p.addLine(to: br); p.addLine(to: bl); p.closeSubpath()
                p.move(to: CGPoint(x: (tl.x + bl.x) / 2, y: (tl.y + bl.y) / 2))
                p.addLine(to: CGPoint(x: (tr.x + br.x) / 2, y: (tr.y + br.y) / 2))
            }
            .stroke(Color.paper.opacity(0.55), style: StrokeStyle(lineWidth: 1.5, dash: [6, 7]))
            ForEach(Array([(tl, 1.0, 1.0), (tr, -1.0, 1.0), (bl, 1.0, -1.0), (br, -1.0, -1.0)].enumerated()),
                    id: \.offset) { _, c in
                let (pt, dx, dy) = c
                let bracket = Path { p in
                    p.move(to: CGPoint(x: pt.x + dx * 16, y: pt.y))
                    p.addLine(to: pt)
                    p.addLine(to: CGPoint(x: pt.x, y: pt.y + dy * 16))
                }
                bracket.stroke(Color.ink, style: StrokeStyle(lineWidth: 7, lineCap: .round, lineJoin: .round))
                bracket.stroke(lock.ink, style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round))
            }
        }
    }

    private func found(_ outline: [CGPoint], in size: CGSize) -> some View {
        let pts = RecordingViewModel.aspectFill(outline, imageSize: lock.imageSize, into: size)
        let path = Path { p in p.addLines(pts); p.closeSubpath() }
        let locked = lock.phase == .locked
        return ZStack {
            path.stroke(Color.ink.opacity(locked ? 0.8 : 0), lineWidth: 6)
            path.stroke(lock.ink, style: StrokeStyle(lineWidth: locked ? 3 : 2, lineJoin: .round,
                                                     dash: locked ? [] : [8, 6]))
        }
    }
}

// MARK: - Recommended tripod setup
// Re-projected at iPhone main-camera FOV: at 1.6 m high, 5 m behind the near baseline the
// whole court fits with margin (near corners ~0.72 of the frame height). Closer than that
// the near corners leave the frame. The live court-lock hints ("往后退一步") only ever
// nudge relative to this, so they can't contradict it.

enum CameraSetup {
    static let why = "横着拍才装得下整片球场"
    static let recommended = "三脚架约 1.6 米高 · 底线后约 5 米 · 正中间"
    static let alternative = "后面没那么远？架高到约 2.5 米，退 4.5 米也行"
}

/// The setup advice on paper, set like a spec card: three short lines, each led by its
/// figure, then the fallback in the quieter label voice. Wraps at every Dynamic Type size.
struct CameraSetupNote: View {
    /// Under the live court-lock sticker: no "why" line, tighter padding.
    var compact = false
    var maxWidth: CGFloat = 330

    /// Each line as (figure, words after it); the figure is set heavier. "1.6 米高" reads as
    /// one word, the same as the corner tag, so no space between 米 and 高.
    static let lines: [(lead: String, figure: String, tail: String)] = [
        ("", "1.6 米", "高"), ("底线后 ", "5 米", ""), ("正中间", "", ""),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 3 : 5) {
            if !compact {
                Text(CameraSetup.why)
                    .font(.system(.footnote).weight(.semibold))
                    .foregroundStyle(Color.ink)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.bottom, 2)
            }
            HStack(spacing: 6) {
                Image(systemName: "camera.metering.center.weighted")
                    .font(.system(size: 11, weight: .bold))
                Text("三脚架").instrumentLabel(AppFont.labelSmall, color: .ink)
            }
            .foregroundStyle(Color.ink)
            ForEach(Array(Self.lines.enumerated()), id: \.offset) { i, line in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(i + 1)")
                        .font(.system(size: 10, weight: .black).monospacedDigit())
                        .foregroundStyle(Color.paper)
                        .frame(width: 15, height: 15)
                        .background(Color.ink, in: Circle())
                        .alignmentGuide(.firstTextBaseline) { $0[VerticalAlignment.center] + 4 }
                    Self.line(line.lead, line.figure, line.tail)
                }
            }
            // Two deliberate lines, so the figures never wrap apart.
            VStack(alignment: .leading, spacing: 0) {
                Text("没那么远？")
                Text("架到 2.5 米高、退 4.5 米").lineLimit(1).minimumScaleFactor(0.85)
            }
            .font(AppFont.caption)
            .foregroundStyle(Color.inkSoft)
            .padding(.top, 1)
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, compact ? 10 : 14)
        .padding(.vertical, compact ? 8 : 12)
        .frame(maxWidth: compact ? maxWidth : 260, alignment: .leading)
        .background(Color.paper, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.ink, lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(CameraSetup.recommended)。\(CameraSetup.alternative)")
    }

    /// Figures ("1.6 米", "5 米") heavier than the words around them.
    private static func line(_ lead: String, _ figure: String, _ tail: String) -> Text {
        let base = Font.system(.subheadline)
        return Text(lead).font(base.weight(.semibold))
            + Text(figure).font(base.weight(.heavy).monospacedDigit())
            + Text(tail).font(base.weight(.semibold))
    }
}

/// Once a court outline shows up the setup card gets out of the frame: a small paper tag
/// in the corner that still says the three numbers.
struct CameraSetupTag: View {
    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "camera.metering.center.weighted").font(.system(size: 10, weight: .bold))
            Text("1.6 米高 · 后 5 米 · 正中").font(.system(size: 11, weight: .bold)).monospacedDigit()
        }
        .foregroundStyle(Color.ink)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.paper.opacity(0.92), in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1))
        .fixedSize()
        .accessibilityLabel(CameraSetup.recommended)
    }
}

// MARK: - Camera asleep
// The idle record screen rests its camera (it heats the phone and holds analysis back).
// No dark scrim: the last frame, drained of colour, sits behind a paper card with a
// shut lens and a butter "Zz" stamp. The whole screen wakes it; so does raising the phone.

struct CameraSleepCard: View {
    let reason: RecordingViewModel.Sleep
    let onWake: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var drift = false

    private var title: String { reason == .idle ? "相机已休息" : "相机先休息" }
    private var line: String { reason == .idle ? "轻点继续" : "轻点录新的一场" }
    private var why: String {
        reason == .idle ? "90 秒没录像，先关掉相机，手机不发热" : "正在分析上一场；相机开着会发热，分析要等它关掉"
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        Button(action: onWake) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                HStack(alignment: .top, spacing: AppSpacing.md) {
                    SleepingLens()
                        .frame(width: 64, height: 64)
                        .overlay(alignment: .topTrailing) {
                            Text("Zz")
                                .font(DisplayFace.font(15, relativeTo: .caption))
                                .foregroundStyle(Color.ink)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(Color.butter, in: Capsule())
                                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
                                .rotationEffect(.degrees(12))
                                .offset(x: 12, y: drift ? -12 : -6)
                        }
                    VStack(alignment: .leading, spacing: 4) {
                        OverprintText(title, font: DisplayFace.font(28, relativeTo: .title2),
                                      plate: reason == .idle ? .lilac : .butter, offset: CGSize(width: 2.5, height: 2.5))
                        Text(why)
                            .font(AppFont.caption)
                            .foregroundStyle(Color.inkSoft)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                HStack(spacing: 8) {
                    Image(systemName: "hand.tap.fill").font(.system(size: 14, weight: .bold))
                    Text(line)
                        .font(DisplayFace.font(18, relativeTo: .headline))
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
                .foregroundStyle(Color.ink)
                .padding(.horizontal, 14)
                .frame(maxWidth: .infinity, minHeight: 46, alignment: .leading)
                .background(Color.card, in: Capsule())
                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            }
            .padding(AppSpacing.lg)
            .frame(maxWidth: 340, alignment: .leading)
            .background(Color.paper, in: shape)
            .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            .printShadow(shape, color: .ink, x: 0, y: 5)
            .rotationEffect(.degrees(-1.5))
        }
        .buttonStyle(ScaleButtonStyle())
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) { drift = true }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title)。\(why)")
        .accessibilityHint(reason == .idle ? "轻点打开相机" : "轻点打开相机，录新的一场")
        .accessibilityAddTraits(.isButton)
    }
}

/// A camera body with its lens shut: an ink ring, a closed-eye lid and lashes.
private struct SleepingLens: View {
    var body: some View {
        Canvas { ctx, size in
            let w = size.width, h = size.height
            let body = CGRect(x: 2, y: h * 0.22, width: w - 4, height: h * 0.7)
            let bump = CGRect(x: w * 0.3, y: h * 0.1, width: w * 0.4, height: h * 0.18)
            ctx.fill(Path(roundedRect: bump, cornerRadius: 4), with: .color(.lilac))
            ctx.stroke(Path(roundedRect: bump, cornerRadius: 4), with: .color(.ink), lineWidth: 1.5)
            ctx.fill(Path(roundedRect: body, cornerRadius: 10), with: .color(.card))
            ctx.stroke(Path(roundedRect: body, cornerRadius: 10), with: .color(.ink), lineWidth: 1.75)
            let c = CGPoint(x: body.midX, y: body.midY), r = h * 0.22
            let lens = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2))
            ctx.fill(lens, with: .color(.paperDeep))
            ctx.stroke(lens, with: .color(.ink), lineWidth: 1.75)
            var lid = Path()
            lid.move(to: CGPoint(x: c.x - r * 0.62, y: c.y - r * 0.05))
            lid.addQuadCurve(to: CGPoint(x: c.x + r * 0.62, y: c.y - r * 0.05), control: CGPoint(x: c.x, y: c.y + r * 0.55))
            ctx.stroke(lid, with: .color(.ink), style: StrokeStyle(lineWidth: 2, lineCap: .round))
            for k in [-0.4, 0.0, 0.4] {
                var lash = Path()
                let x = c.x + r * k
                let y = c.y + r * (0.22 - abs(k) * 0.2)
                lash.move(to: CGPoint(x: x, y: y))
                lash.addLine(to: CGPoint(x: x + r * k * 0.35, y: y + r * 0.3))
                ctx.stroke(lash, with: .color(.ink), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
            }
            let dot = CGRect(x: body.maxX - 12, y: body.minY + 6, width: 6, height: 6)
            ctx.fill(Path(ellipseIn: dot), with: .color(.tomato.opacity(0.35)))
        }
        .accessibilityHidden(true)
    }
}

// MARK: - Rotate prompt (phone held upright)

struct RotatePrompt: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var turned = false

    var body: some View {
        VStack(spacing: AppSpacing.md) {
            Image(systemName: "iphone.gen3")
                .font(.system(size: 64, weight: .regular))
                .foregroundStyle(Color.paper)
                .rotationEffect(.degrees(turned ? -90 : 0))
                .animation(reduceMotion ? nil : .easeInOut(duration: 0.9).delay(0.4).repeatForever(autoreverses: true),
                           value: turned)
            Text("把手机横过来")
                .font(DisplayFace.font(26, relativeTo: .title2))
                .foregroundStyle(Color.ink)
                .padding(.horizontal, 16)
                .padding(.vertical, 7)
                .background(Color.butter, in: Capsule())
                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                .rotationEffect(.degrees(-3))
            CameraSetupNote()
        }
        .onAppear { turned = true }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("请把手机横过来再录制。\(CameraSetup.why)。\(CameraSetup.recommended)。\(CameraSetup.alternative)")
    }
}

// MARK: - System chips
// Slips for things the phone itself is telling us: storage left, heat.
// They sit under the record HUD and the match-list masthead.
// Normal: no chip at all. Mild: a paper slip with a butter dot. Serious: a tomato slip.

enum SystemChipSeverity: Sendable, Equatable {
    case mild, serious

    init(storage level: StorageStatus.Level) {
        self = level >= .refuseStart ? .serious : .mild
    }

    /// `.hold` (finish this one, then rest) and Low Power Mode are mild; `.pause` is serious.
    init(thermal decision: ThermalDecision) {
        self = decision == .pause ? .serious : .mild
    }
}

/// A chip is one short line; the full sentence goes to VoiceOver.
struct SystemChipItem: Equatable {
    var text: String
    var severity: SystemChipSeverity
    var spoken: String? = nil

    init(text: String, severity: SystemChipSeverity, spoken: String? = nil) {
        self.text = text
        self.severity = severity
        self.spoken = spoken
    }

    init?(storage: StorageStatus) {
        guard let full = storage.message else { return nil }
        let short: String
        switch storage.level {
        case .ok: return nil
        case .low: short = "空间只够再录约 \(Int(storage.minutesLeft.rounded(.down))) 分钟"
        case .refuseStart: short = "空间不够了，先清理再录"
        case .stopRecording: short = "空间快满，已停录并保存"
        }
        self.init(text: short, severity: SystemChipSeverity(storage: storage.level), spoken: full)
    }

    init?(thermal decision: ThermalDecision, message: String?) {
        guard let full = message else { return nil }
        let short: String
        switch decision {
        case .pause: short = "手机有点热，先歇一会儿"
        case .hold: short = "有点热，这场分析完歇会儿"
        case .run: short = "低电量模式，分析会慢些"
        }
        self.init(text: short, severity: SystemChipSeverity(thermal: decision), spoken: full)
    }
}

struct SystemChip: View {
    let text: String
    let symbol: String
    var severity: SystemChipSeverity = .mild
    var spoken: String? = nil

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: 12, style: .continuous)
        let serious = severity == .serious
        HStack(alignment: .top, spacing: 6) {
            ZStack {
                if !serious {
                    Circle().fill(Color.butter)
                        .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1))
                        .frame(width: 9, height: 9)
                        .offset(x: 6, y: -5)
                }
                Image(systemName: symbol).font(.system(size: 11, weight: .black))
            }
            .frame(width: 16, height: 14)
            Text(text)
                .font(.system(size: 12, weight: .bold))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
        .foregroundStyle(Color.ink)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(serious ? Color.tomato : Color.paper, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: serious ? AppStroke.ink : 1.25))
        .printShadow(shape, color: .ink, x: 0, y: serious ? 3 : 2)
        .rotationEffect(.degrees(serious ? -2.5 : -1.5))
        .fixedSize()
        .accessibilityElement(children: .combine)
        .accessibilityLabel(serious ? "注意：\(spoken ?? text)" : spoken ?? text)
    }
}

struct SystemChips: View {
    var storage: SystemChipItem?
    var thermal: SystemChipItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let storage { SystemChip(text: storage.text, symbol: "externaldrive.fill", severity: storage.severity, spoken: storage.spoken) }
            if let thermal { SystemChip(text: thermal.text, symbol: "thermometer.medium", severity: thermal.severity, spoken: thermal.spoken) }
        }
        .transition(.scale(scale: 0.9, anchor: .top).combined(with: .opacity))
        .animation(AppAnimation.spring, value: storage)
        .animation(AppAnimation.spring, value: thermal)
    }
}

extension View {
    /// Demo text for the system chips when `-UIDemoSystemChips` is on.
    func demoSystemText(_ real: String?, _ sample: String) -> String? {
        #if DEBUG
        if DemoMode.showsSystemChips { return real ?? sample }
        #endif
        return real
    }

    /// Same as `demoSystemText`, for a chip that carries its severity.
    func demoSystemItem(_ real: SystemChipItem?, _ sample: String, _ severity: SystemChipSeverity) -> SystemChipItem? {
        #if DEBUG
        if DemoMode.showsSystemChips { return real ?? SystemChipItem(text: sample, severity: severity) }
        #endif
        return real
    }
}

extension MatchPipeline {
    /// The heat / Low Power line with how loud it should be (nil when all is well).
    var thermalChip: SystemChipItem? {
        SystemChipItem(thermal: thermalDecision, message: thermalMessage)
    }
}
