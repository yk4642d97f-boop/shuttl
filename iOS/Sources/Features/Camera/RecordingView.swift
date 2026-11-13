import SwiftUI
import AVFoundation

// MARK: - Camera preview wrapper

private struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = UIColor(Color.night)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            previewLayer.frame = uiView.bounds
        }
    }
}

// MARK: - Trajectory overlay
// Cool → hot trail: lilac tail, tomato body, butter head with an ink ring.

struct TrajectoryOverlayView: View {
    let points: [CGPoint]

    var body: some View {
        Canvas { ctx, _ in
            TrailPainter.paint(points, in: &ctx)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// The trail encodes age, not speed: ink for the older part, tomato for the newest, so butter
/// stays reserved for the court-lock sticker. A thin paper keyline keeps the ink visible on
/// dark footage.
enum TrailPainter {
    static func paint(_ points: [CGPoint], in ctx: inout GraphicsContext) {
        guard let head = points.last else { return }
        let count = points.count
        if count >= 2 {
            for i in 1..<count {
                let t = Double(i) / Double(count - 1)
                var seg = Path()
                seg.move(to: points[i - 1])
                seg.addLine(to: points[i])
                let width = 2 + 5 * t
                ctx.stroke(seg, with: .color(Color.paper.opacity(0.35 + 0.4 * t)),
                           style: StrokeStyle(lineWidth: width + 2, lineCap: .round))
                ctx.stroke(seg, with: .color(color(at: t)),
                           style: StrokeStyle(lineWidth: width, lineCap: .round))
            }
        }
        let ring = CGRect(x: head.x - 11, y: head.y - 11, width: 22, height: 22)
        ctx.fill(Path(ellipseIn: ring), with: .color(.paper))
        ctx.fill(Path(ellipseIn: ring.insetBy(dx: 2.5, dy: 2.5)), with: .color(.tomato))
        ctx.stroke(Path(ellipseIn: ring.insetBy(dx: 2.5, dy: 2.5)), with: .color(.ink), lineWidth: 1.5)
    }

    private static func color(at t: Double) -> Color {
        t < 0.6 ? Color.ink.opacity(0.25 + t * 1.2) : .tomato
    }
}

// MARK: - Top HUD

private struct RecordingHUDView: View {
    let viewModel: RecordingViewModel

    var body: some View {
        HStack(spacing: AppSpacing.sm) {
            if viewModel.isRecording {
                HStack(spacing: 6) {
                    Circle().fill(Color.ink).frame(width: 8, height: 8).pulse(maxScale: 1.3)
                    Text("REC").font(AppFont.labelSmall).kerning(1.5)
                }
                .foregroundStyle(Color.ink)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.tomato, in: Capsule())
                .transition(.scale.combined(with: .opacity))

                Text(viewModel.durationText)
                    .font(AppFont.timer)
                    .foregroundStyle(Color.butter)
                    .contentTransition(.numericText())
                    .transition(.opacity)

                // Monocular estimate of the last flight; only exists when calibrated.
                if let kmh = viewModel.approxSpeedKmh {
                    Text("≈\(Int(kmh.rounded())) km/h")
                        .font(AppFont.labelSmall).kerning(1)
                        .foregroundStyle(Color.paper.opacity(0.75))
                        .transition(.opacity)
                        .accessibilityLabel("上一拍球速约 \(Int(kmh.rounded())) 公里每小时，估算值")
                }
            } else {
                (Text("Shuttl").foregroundColor(.paper) + Text(".").foregroundColor(.tomato))
                    .font(DisplayFace.fixed(17))
                Rectangle().fill(Color.paper.opacity(0.3)).frame(width: 1, height: 12)
                Text(viewModel.sleep == nil ? "待机 · 1080p · 60fps" : "相机休息中")
                    .font(AppFont.labelSmall).kerning(1)
                    .foregroundStyle(Color.paper.opacity(0.75))
            }
        }
        .padding(.horizontal, AppSpacing.md - 2)
        .padding(.vertical, AppSpacing.sm)
        .background(Color.night.opacity(0.62), in: Capsule())
        .overlay(Capsule().strokeBorder(Color.paper.opacity(0.18), lineWidth: 1))
        .animation(AppAnimation.spring, value: viewModel.isRecording)
    }
}

// MARK: - Match type selector (sliding ink blob)

private struct MatchTypeSelectorView: View {
    @Binding var courtType: CourtType
    /// Counter-rotation for the labels when the phone is on its side.
    var glyphAngle: Angle = .zero
    @Namespace private var blob

    var body: some View {
        HStack(spacing: 0) {
            chip(label: "单打", type: .singles)
            chip(label: "双打", type: .doubles)
        }
        .padding(3)
        .background(Color.paperDeep, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
        .sensoryFeedback(.selection, trigger: courtType)
    }

    private func chip(label: String, type: CourtType) -> some View {
        let active = courtType == type
        return Button {
            withAnimation(AppAnimation.bouncy) { courtType = type }
        } label: {
            Text(label)
                .font(DisplayFace.font(17, relativeTo: .subheadline))
                .foregroundStyle(active ? Color.paper : Color.inkSoft)
                .fixedSize()
                .rotationEffect(glyphAngle)
                .frame(width: 48, height: 36)
                .background {
                    if active {
                        Capsule().fill(Color.ink).matchedGeometryEffect(id: "blob", in: blob)
                    }
                }
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(active ? [.isSelected] : [])
    }
}

// MARK: - Record button
// Ink ring on card stock around a tomato disc; the disc morphs to a stop square.

private struct RecordButtonView: View {
    let isRecording: Bool
    /// Phone is upright: the disc shows a rotate glyph and a tap only nudges.
    var needsLandscape = false
    /// The camera sleeps: no picture, so no take. The disc turns paper and a tap wakes the camera.
    var asleep = false
    var glyphAngle: Angle = .zero
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color.card)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 2))
                    .frame(width: 78, height: 78)
                // Stopwatch bezel: 24 ticks, every sixth one long.
                Canvas { ctx, size in
                    let c = CGPoint(x: size.width / 2, y: size.height / 2)
                    for i in 0..<24 {
                        let a = Double(i) / 24 * 2 * .pi
                        let long = i % 6 == 0
                        let r0: CGFloat = long ? 30.5 : 33, r1: CGFloat = 36
                        var tick = Path()
                        tick.move(to: CGPoint(x: c.x + cos(a) * r0, y: c.y + sin(a) * r0))
                        tick.addLine(to: CGPoint(x: c.x + cos(a) * r1, y: c.y + sin(a) * r1))
                        ctx.stroke(tick, with: .color(Color.ink.opacity(long ? 0.9 : 0.45)),
                                   style: StrokeStyle(lineWidth: long ? 2 : 1.2, lineCap: .round))
                    }
                }
                .frame(width: 78, height: 78)
                .opacity(isRecording ? 0 : 1)
                RoundedRectangle(cornerRadius: isRecording ? 8 : 30, style: .continuous)
                    .fill(asleep ? Color.paperDeep : Color.tomato)
                    .overlay(RoundedRectangle(cornerRadius: isRecording ? 8 : 30, style: .continuous)
                        .strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: isRecording ? 30 : 52, height: isRecording ? 30 : 52)
                    .opacity(needsLandscape && !asleep ? 0.35 : 1)
                if asleep {
                    Image(systemName: "power")
                        .font(.system(size: 22, weight: .black))
                        .foregroundStyle(Color.ink)
                        .rotationEffect(glyphAngle)
                        .transition(.scale.combined(with: .opacity))
                } else if needsLandscape {
                    Image(systemName: "rotate.right.fill")
                        .font(.system(size: 22, weight: .black))
                        .foregroundStyle(Color.ink)
                        .rotationEffect(glyphAngle)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .animation(AppAnimation.bouncy, value: isRecording)
            .animation(AppAnimation.bouncy, value: needsLandscape)
            .animation(AppAnimation.bouncy, value: asleep)
        }
        .buttonStyle(PrintButtonStyle(shape: Circle(), plate: .ink, depth: 4))
        .sensoryFeedback(.impact(weight: .heavy), trigger: isRecording)
        .accessibilityLabel(isRecording ? "停止录制" : asleep ? "相机休息中，轻点打开相机"
                            : needsLandscape ? "开始录制，请先把手机横过来" : "开始录制")
    }
}

// MARK: - Bottom control deck
// A slab of paper over the video, like the control panel of a field recorder.

private struct BottomControlsView: View {
    let viewModel: RecordingViewModel
    @Binding var courtType: CourtType
    var glyphAngle: Angle = .zero
    var needsLandscape = false
    var onBlockedRecord: () -> Void = {}

    var body: some View {
        HStack(alignment: .center, spacing: 0) {
            Group {
                if !viewModel.isRecording {
                    MatchTypeSelectorView(courtType: $courtType, glyphAngle: glyphAngle)
                        .transition(.opacity.combined(with: .scale(scale: 0.85)))
                } else {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("正在录制")
                            .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                        Text(courtType == .singles ? "单打" : "双打")
                            .font(AppFont.title2)
                            .foregroundStyle(Color.ink)
                    }
                    .fixedSize()
                    .rotationEffect(glyphAngle)
                    .transition(.opacity)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            RecordButtonView(isRecording: viewModel.isRecording,
                             needsLandscape: needsLandscape && !viewModel.isRecording,
                             asleep: viewModel.sleep != nil && !viewModel.isRecording,
                             glyphAngle: glyphAngle) {
                // Asleep: never record blind; the first tap only brings the camera back.
                if viewModel.sleep != nil && !viewModel.isRecording {
                    withAnimation(AppAnimation.spring) { viewModel.wake() }
                } else if needsLandscape && !viewModel.isRecording { onBlockedRecord() } else { viewModel.toggleRecording() }
            }

            Group {
                if !viewModel.isRecording {
                    IconButton(systemName: "arrow.triangle.2.circlepath.camera",
                               size: 44, iconSize: 18, style: .filled,
                               label: viewModel.isFrontCamera ? "切换到后置摄像头" : "切换到前置摄像头") {
                        viewModel.toggleCamera()
                    }
                    .disabled(!viewModel.canSwitchCamera)
                    .opacity(viewModel.canSwitchCamera ? 1 : 0.4)
                    .rotationEffect(glyphAngle)
                    .rotation3DEffect(.degrees(viewModel.isFrontCamera ? 180 : 0), axis: (x: 0, y: 1, z: 0))
                    .animation(AppAnimation.bouncy, value: viewModel.isFrontCamera)
                    .sensoryFeedback(.selection, trigger: viewModel.isFrontCamera)
                    .transition(.opacity.combined(with: .scale(scale: 0.85)))
                } else {
                    Color.clear.frame(width: 44, height: 44)
                }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.vertical, AppSpacing.md - 4)
        .background(Color.paper, in: RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
            .strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .printShadow(RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous), color: .tomato, y: 5)
        .padding(.horizontal, AppSpacing.md)
        .animation(AppAnimation.spring, value: viewModel.isRecording)
        .animation(AppAnimation.spring, value: glyphAngle)
    }
}

// MARK: - Save notice
// Paper slip above the control deck: what happened to the take you just stopped.

private struct SaveNoticeCard: View {
    let notice: RecordingViewModel.SaveNotice
    /// Analysis runs on this phone (nothing is uploaded).
    let onDevice: Bool
    let onOpenMatches: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.md, style: .continuous)
        HStack(spacing: AppSpacing.md) {
            badge
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(AppFont.headline)
                    .foregroundStyle(Color.ink)
                Text(subtitle)
                    .font(AppFont.caption)
                    .foregroundStyle(Color.inkSoft)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            action
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.vertical, AppSpacing.sm + 2)
        .background(Color.paper, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: 1.5))
        .printShadow(shape, color: .ink, y: 4)
    }

    private var title: String {
        switch notice {
        case .finishing: return "正在保存视频…"
        case .saved:     return onDevice ? "已保存，正在手机上分析" : "已保存，正在上传分析"
        case .failed:    return "这段录像没有保存成功"
        }
    }

    private var subtitle: String {
        switch notice {
        case .finishing:          return onDevice ? "请稍等，文件写完后自动开始分析" : "请稍等，文件写完后自动上传"
        case .saved:              return onDevice ? "可以在「比赛」里看分析进度" : "可以在「比赛」里看上传和分析进度"
        case .failed(let reason): return reason
        }
    }

    @ViewBuilder
    private var badge: some View {
        ZStack {
            Circle()
                .fill(badgeInk)
                .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
            if notice == .finishing {
                ProgressView().tint(.ink)
            } else {
                Image(systemName: notice == .saved ? "checkmark" : "exclamationmark")
                    .font(.system(size: 16, weight: .black))
                    .foregroundStyle(Color.ink)
            }
        }
        .frame(width: 40, height: 40)
        .accessibilityHidden(true)
    }

    private var badgeInk: Color {
        switch notice {
        case .finishing: return .butter
        case .saved:     return .pistachio
        case .failed:    return .tomato
        }
    }

    @ViewBuilder
    private var action: some View {
        switch notice {
        case .finishing:
            EmptyView()
        case .saved:
            Button(action: onOpenMatches) {
                Text("查看进度")
                    .font(DisplayFace.font(17, relativeTo: .subheadline))
                    .foregroundStyle(Color.onCobalt)
                    .padding(.horizontal, AppSpacing.md - 2)
                    .frame(minHeight: 40)
                    .background(Color.cobalt, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.5))
            }
            .buttonStyle(PrintButtonStyle(shape: Capsule(), plate: .ink, depth: 3))
            .accessibilityHint("切换到比赛页")
        case .failed:
            IconButton(systemName: "xmark", size: 36, iconSize: 13, style: .ghost, label: "关闭提示", action: onDismiss)
        }
    }
}

// MARK: - RecordingView

struct RecordingView: View {
    @State private var viewModel = RecordingViewModel()
    @State private var tilt = DeviceTilt()
    @State private var nudge = 0
    @Environment(MatchPipeline.self) private var pipeline: MatchPipeline?
    @Environment(AppChrome.self) private var chrome: AppChrome?
    @Environment(\.scenePhase) private var scenePhase

    /// The frozen, colour-drained last frame behind the sleep card (paper-tinted, no dark scrim).
    @ViewBuilder
    private var sleepStill: some View {
        if viewModel.sleep != nil {
            ZStack {
                if let still = viewModel.sleepStill {
                    Image(decorative: still, scale: 1).resizable().scaledToFill()
                }
                Color.paper.opacity(viewModel.sleepStill == nil ? 0.22 : 0.35).blendMode(.multiply)
            }
            .ignoresSafeArea()
            .transition(.opacity)
        }
    }

    /// Court setup state; the analysis side feeds real lock states in here.
    var courtLock: CourtLockDisplay = .searching

    /// Recording needs the phone on its side (a portrait clip can't hold the court).
    private var needsLandscape: Bool { tilt.isAvailable && !tilt.isLandscape }

    var body: some View {
        ZStack {
            Color.night.ignoresSafeArea()

            CameraPreviewView(previewLayer: viewModel.cameraSession.previewLayer)
                .ignoresSafeArea()
                .overlay { sleepStill }
                // Brief dip to night while the lens is swapped, instead of a frozen / jumping frame.
                .overlay { Color.night.opacity(viewModel.isSwitchingCamera ? 0.85 : 0).ignoresSafeArea() }
                .animation(AppAnimation.standard, value: viewModel.isSwitchingCamera)

            #if DEBUG
            if DemoMode.isOn {
                UprightFrame(angle: tilt.angle) { _ in DemoCourtBackdrop() }.ignoresSafeArea()
                    // Stands in for the drained still: the simulator has no camera frame.
                    .saturation(viewModel.sleep == nil ? 1 : 0.15)
                    .overlay { if viewModel.sleep != nil { Color.paper.opacity(0.35).blendMode(.multiply).ignoresSafeArea() } }
            }
            #endif

            scrims

            // Everything the holder reads is laid out in their frame.
            UprightFrame(angle: tilt.angle) { size in uprightLayer(size) }
                .ignoresSafeArea()

            VStack {
                Spacer()
                BottomControlsView(viewModel: viewModel, courtType: $viewModel.courtType,
                                   glyphAngle: tilt.angle, needsLandscape: needsLandscape) {
                    withAnimation(AppAnimation.bouncy) { nudge += 1 }
                }
                .padding(.bottom, AppSpacing.md)
            }
        }
        .animation(AppAnimation.standard, value: viewModel.isRecording)
        .animation(AppAnimation.spring, value: viewModel.saveNotice)
        .animation(AppAnimation.spring, value: tilt.pose)
        .sensoryFeedback(.warning, trigger: nudge)
        .statusBarHidden(true)
        .task {
            viewModel.matchPipeline = pipeline
            await viewModel.onAppear()
        }
        .task(id: viewModel.saveNotice) {
            // "Saved" steps aside on its own; failures stay until dismissed.
            guard viewModel.saveNotice == .saved else { return }
            try? await Task.sleep(for: .seconds(8))
            if !Task.isCancelled, viewModel.saveNotice == .saved { viewModel.dismissNotice() }
        }
        .onAppear { tilt.start(); chrome?.deviceLandscape = tilt.isLandscape }
        .onChange(of: tilt.isLandscape) { was, sideways in
            chrome?.deviceLandscape = sideways
            // Raising the phone onto its side is the "I'm about to record" gesture.
            if sideways && !was && viewModel.sleep != nil { withAnimation(AppAnimation.spring) { viewModel.wake() } }
        }
        .onChange(of: scenePhase, initial: true) { _, phase in viewModel.sceneBecame(active: phase == .active) }
        // Any touch on the live screen keeps the camera awake another 90 s.
        .simultaneousGesture(TapGesture().onEnded { viewModel.noteActivity() })
        .animation(AppAnimation.spring, value: viewModel.sleep)
        .onDisappear {
            tilt.stop()
            chrome?.deviceLandscape = false
            viewModel.onDisappear()
        }
    }

    /// Upright: the record deck and the tab bar take the bottom of the screen; the target
    /// guide stops above them so no bracket pokes out under the deck.
    static let portraitDeckClearance: CGFloat = 215

    /// Overlay, HUD, trail and notices in the holder's frame. On its side the deck sits at
    /// the holder's right (landscapeLeft) or left (landscapeRight); the Dynamic Island opposite.
    @ViewBuilder
    private func uprightLayer(_ size: CGSize) -> some View {
        let deckDepth: CGFloat = 132
        let deckOnTrailing = tilt.pose == .landscapeLeft
        let leading: CGFloat = tilt.isLandscape ? (deckOnTrailing ? 56 : deckDepth) : 0
        let trailing: CGFloat = tilt.isLandscape ? (deckOnTrailing ? deckDepth : 56) : 0
        ZStack {
            if let reason = viewModel.sleep {
                // Asleep: the setup guide and card stay hidden; one paper card, centred in the holder's frame.
                CameraSleepCard(reason: reason) { withAnimation(AppAnimation.spring) { viewModel.wake() } }
                    .padding(.leading, leading + AppSpacing.md)
                    .padding(.trailing, trailing + AppSpacing.md)
                    .padding(.bottom, tilt.isLandscape ? 0 : 150)
                    .transition(.scale(scale: 0.92).combined(with: .opacity))
            } else if !viewModel.isRecording {
                if tilt.isLandscape || !tilt.isAvailable {
                    FramingOverlay(lock: viewModel.courtLock, leadingInset: leading, trailingInset: trailing,
                                   topInset: systemChipCount == 0 ? 0 : CGFloat(systemChipCount) * 44 + 44,
                                   bottomInset: tilt.isLandscape ? 0 : Self.portraitDeckClearance)
                        .transition(.opacity)
                } else {
                    RotatePrompt()
                        .scaleEffect(nudge.isMultiple(of: 2) ? 1 : 1.06)
                        .padding(.bottom, 140)
                        .transition(.opacity)
                }
            }
            let pts = overlayPoints(in: size)
            if !pts.isEmpty { TrajectoryOverlayView(points: pts) }

            VStack(alignment: tilt.isLandscape ? .leading : .center) {
                RecordingHUDView(viewModel: viewModel)
                SystemChips(storage: demoSystemItem(viewModel.storageChip, "空间只够再录约 8 分钟", .mild),
                            thermal: demoSystemItem(pipeline?.thermalChip, "手机有点热，先歇一会儿", .serious))
                    .padding(.top, 4)
                Spacer()
                saveNotice
                    .frame(maxWidth: 440)
                    .padding(.bottom, tilt.isLandscape ? AppSpacing.md : 150)
            }
            .padding(.top, tilt.isLandscape ? AppSpacing.md : 60)
            .padding(.leading, tilt.isLandscape ? leading + AppSpacing.sm : AppSpacing.md)
            .padding(.trailing, tilt.isLandscape ? trailing + AppSpacing.sm : AppSpacing.md)
            .frame(maxWidth: .infinity, alignment: tilt.isLandscape ? .leading : .center)
        }
    }

    private var systemChipCount: Int {
        [demoSystemItem(viewModel.storageChip, "x", .mild), demoSystemItem(pipeline?.thermalChip, "x", .mild)]
            .compactMap { $0 }.count
    }

    /// Soft scrims so paper UI and the HUD read over any footage.
    private var scrims: some View {
        VStack(spacing: 0) {
            LinearGradient(colors: [Color.night.opacity(0.55), .clear], startPoint: .top, endPoint: .bottom)
                .frame(height: 160)
            Spacer()
            LinearGradient(colors: [.clear, Color.night.opacity(0.6)], startPoint: .top, endPoint: .bottom)
                .frame(height: 260)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private var saveNotice: some View {
        if let notice = viewModel.saveNotice {
            SaveNoticeCard(notice: notice,
                           onDevice: showsOnDeviceNotice,
                           onOpenMatches: openMatches,
                           onDismiss: viewModel.dismissNotice)
                .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    /// Release analyses on the phone; the demo pipeline has no analyzer, so it follows Release.
    private var showsOnDeviceNotice: Bool {
        #if DEBUG
        if DemoMode.isOn { return true }
        #endif
        return pipeline?.analyzesOnDevice ?? false
    }

    private func openMatches() {
        viewModel.dismissNotice()
        withAnimation(AppAnimation.bouncy) { chrome?.selectedTab = .matches }
    }

    private func overlayPoints(in size: CGSize) -> [CGPoint] {
        #if DEBUG
        if DemoMode.showsTrajectory {
            return DemoMode.syntheticTrail().map { CGPoint(x: $0.x * size.width, y: $0.y * size.height) }
        }
        #endif
        return viewModel.overlayPoints(in: size)
    }
}
