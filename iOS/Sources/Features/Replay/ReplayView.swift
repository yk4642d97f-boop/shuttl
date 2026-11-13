import SwiftUI
import AVFoundation
import SwiftData

// MARK: - AVPlayerView (UIViewRepresentable)

struct AVPlayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerUIView {
        let view = PlayerUIView()
        view.player = player
        return view
    }

    func updateUIView(_ uiView: PlayerUIView, context: Context) {
        uiView.player = player
    }
}

final class PlayerUIView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }

    var player: AVPlayer? {
        get { (layer as? AVPlayerLayer)?.player }
        set { (layer as? AVPlayerLayer)?.player = newValue }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        (layer as? AVPlayerLayer)?.videoGravity = .resizeAspect
    }

    required init?(coder: NSCoder) { fatalError() }
}

// MARK: - ReplayView
// Video sits on a night surround; everything you touch is printed on paper.

struct ReplayView: View {
    let match: Match
    @State private var viewModel: ReplayViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(AppChrome.self) private var chrome: AppChrome?
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    /// On its side the picture gets the height; controls fold into one strip.
    private var landscape: Bool { verticalSizeClass == .compact }

    init(match: Match, startFrame: Int? = nil) {
        self.match = match
        _viewModel = State(initialValue: ReplayViewModel(match: match, startFrame: startFrame))
    }

    var body: some View {
        ZStack {
            Color.night.ignoresSafeArea()

            #if DEBUG
            if DemoMode.isOn { DemoCourtBackdrop().ignoresSafeArea() }
            #endif

            if let player = viewModel.player {
                AVPlayerView(player: player).ignoresSafeArea()
            } else if viewModel.videoMissing {
                VStack(spacing: AppSpacing.sm) {
                    Image(systemName: "film.stack")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundStyle(Color.butter)
                    Text("这台设备上找不到这场的视频文件")
                        .font(AppFont.callout)
                        .foregroundStyle(Color.paper.opacity(0.8))
                }
                .accessibilityElement(children: .combine)
            } else {
                VStack(spacing: AppSpacing.sm) {
                    ProgressView().tint(.butter)
                    Text("加载视频…").font(AppFont.callout).foregroundStyle(Color.paper.opacity(0.7))
                }
            }

            GeometryReader { geo in
                ZStack {
                    TrajectoryOverlayView(points: overlayPoints(in: geo.size))
                    if let shot = viewModel.nearbyShot {
                        let rect = viewModel.videoRect(in: geo.size)
                        shotTypeLabel(shot)
                            .position(x: rect.minX + CGFloat(shot.normalizedX) * rect.width,
                                      y: max(60, rect.minY + CGFloat(shot.normalizedY) * rect.height - 36))
                            .transition(.scale.combined(with: .opacity))
                            .animation(AppAnimation.springFast, value: shot.frameIndex)
                    }
                }
            }
            .ignoresSafeArea()

            VStack {
                topBar
                Spacer()
                Group {
                    if landscape { compactControlBar } else { controlBar }
                }
                .padding(.horizontal, AppSpacing.md)
                .padding(.bottom, landscape ? AppSpacing.xs : AppSpacing.sm)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .statusBarHidden(true)
        .onAppear {
            chrome?.replayOpen = true
            viewModel.setup()
        }
        #if DEBUG
        .task {
            guard let at = DemoMode.replayAtSeconds else { return }
            try? await Task.sleep(for: .milliseconds(900))
            viewModel.currentTime = at
        }
        #endif
        .onDisappear {
            chrome?.replayOpen = false
            viewModel.teardown()
        }
    }

    private func overlayPoints(in size: CGSize) -> [CGPoint] {
        #if DEBUG
        if DemoMode.isOn && viewModel.currentTrajectoryWindow.isEmpty {
            return DemoMode.syntheticTrail().map { CGPoint(x: $0.x * size.width, y: $0.y * size.height) }
        }
        #endif
        // Same frame as the aspect-fit player, so map into the picture rect, not the whole screen.
        let rect = viewModel.videoRect(in: size)
        return viewModel.currentTrajectoryWindow.map {
            CGPoint(x: rect.minX + $0.position.x * rect.width, y: rect.minY + $0.position.y * rect.height)
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: AppSpacing.sm) {
            IconButton(systemName: "chevron.left", style: .filled, label: "返回") { dismiss() }

            VStack(alignment: .leading, spacing: 0) {
                Text("回放")
                    .font(AppFont.title3)
                    .foregroundStyle(Color.ink)
                    .padding(.top, 2)
                Text(match.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(AppFont.labelSmall)
                    .foregroundStyle(Color.inkSoft)
            }
            .padding(.horizontal, AppSpacing.md)
            .frame(height: 44)
            .background(Color.paper, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.5))

            if let chip = rallyChip {
                HStack(spacing: 6) {
                    Text("回合 \(chip.label)").font(AppFont.labelSmall).foregroundStyle(Color.inkSoft)
                    Text("近 \(chip.near)").font(DisplayFace.fixed(16)).foregroundStyle(Color.cobalt)
                    Text(":").font(DisplayFace.fixed(14)).foregroundStyle(Color.inkFaint)
                    Text("远 \(chip.far)").font(DisplayFace.fixed(16)).foregroundStyle(Color.ink)
                }
                .padding(.horizontal, 10)
                .frame(height: 32)
                .background(Color.paper, in: Capsule())
                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.fine))
                .contentTransition(.numericText())
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("第 \(chip.label) 回合，近端 \(chip.near) 比远端 \(chip.far)")
            }

            Spacer()

            if !viewModel.rallyWindows.isEmpty {
                Button {
                    withAnimation(AppAnimation.bouncy) { viewModel.ralliesOnly.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: viewModel.ralliesOnly ? "scissors" : "film")
                            .font(.system(size: 13, weight: .black))
                        Text(viewModel.ralliesOnly ? "只看回合 · 开" : "只看回合")
                            .font(DisplayFace.font(17, relativeTo: .subheadline))
                    }
                    .foregroundStyle(Color.ink)
                    .padding(.horizontal, 14)
                    .frame(height: 38)
                    .background(viewModel.ralliesOnly ? Color.butter : Color.paper, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                    .rotationEffect(.degrees(viewModel.ralliesOnly ? 3 : 0))
                }
                .buttonStyle(ScaleButtonStyle())
                .sensoryFeedback(.selection, trigger: viewModel.ralliesOnly)
                .accessibilityLabel("只看回合")
                .accessibilityValue(viewModel.ralliesOnly ? "开，自动跳过捡球时间" : "关")
            }
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, landscape ? AppSpacing.xs : AppSpacing.sm)
    }

    // MARK: - Control deck

    private var controlBar: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        return VStack(spacing: AppSpacing.md) {
            scrubber

            HStack(spacing: AppSpacing.sm) {
                Text(viewModel.replayTimeText)
                    .font(AppFont.timer)
                    .foregroundStyle(Color.ink)
                    .contentTransition(.numericText())
                    .frame(minWidth: 60, alignment: .leading)

                Spacer(minLength: 0)

                HStack(spacing: 4) {
                    ForEach(viewModel.availableRates, id: \.self) { rate in
                        let active = abs(viewModel.playbackRate - rate) < 0.01
                        Button { withAnimation(AppAnimation.bouncy) { viewModel.setRate(rate) } } label: {
                            Text(rateText(rate))
                                .font(AppFont.labelSmall)
                                .foregroundStyle(active ? Color.paper : Color.inkSoft)
                                .frame(minWidth: 40, minHeight: 32)
                                .background(active ? Color.ink : Color.clear, in: Capsule())
                                .contentShape(Capsule())
                        }
                        .buttonStyle(ScaleButtonStyle())
                        .accessibilityLabel("\(rateText(rate)) 速")
                        .accessibilityAddTraits(active ? [.isSelected] : [])
                    }
                }
                .padding(3)
                .background(Color.paperDeep, in: Capsule())
                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))

                Spacer(minLength: 0)

                Button { viewModel.togglePlayback() } label: {
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 20, weight: .black))
                        .foregroundStyle(Color.onCobalt)
                        .contentTransition(.symbolEffect(.replace))
                        .frame(width: 56, height: 56)
                        .background(Color.cobalt, in: Circle())
                        .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
                }
                .buttonStyle(PrintButtonStyle(shape: Circle(), plate: .ink, depth: 4))
                .accessibilityLabel(viewModel.isPlaying ? "暂停" : "播放")
            }
        }
        .padding(AppSpacing.md)
        .background(Color.paper, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: 1.5))
    }

    /// Landscape: play · time · scrubber · speed, on one paper strip.
    private var compactControlBar: some View {
        let shape = Capsule()
        return HStack(spacing: AppSpacing.md) {
            playButton(size: 46)
            Text(viewModel.replayTimeText)
                .font(AppFont.timer)
                .foregroundStyle(Color.ink)
                .contentTransition(.numericText())
                .frame(minWidth: 56, alignment: .leading)
            scrubber
            rateChips
        }
        .padding(.leading, 6)
        .padding(.trailing, AppSpacing.md)
        .padding(.vertical, 6)
        .background(Color.paper, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .printShadow(shape, color: .ink, y: 4)
        .frame(maxWidth: 760)
    }

    private var rateChips: some View {
        HStack(spacing: 4) {
            ForEach(viewModel.availableRates, id: \.self) { rate in
                let active = abs(viewModel.playbackRate - rate) < 0.01
                Button { withAnimation(AppAnimation.bouncy) { viewModel.setRate(rate) } } label: {
                    Text(rateText(rate))
                        .font(AppFont.labelSmall)
                        .foregroundStyle(active ? Color.paper : Color.inkSoft)
                        .frame(minWidth: 40, minHeight: 32)
                        .background(active ? Color.ink : Color.clear, in: Capsule())
                        .contentShape(Capsule())
                }
                .buttonStyle(ScaleButtonStyle())
                .accessibilityLabel("\(rateText(rate)) 速")
                .accessibilityAddTraits(active ? [.isSelected] : [])
            }
        }
        .padding(3)
        .background(Color.paperDeep, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
    }

    private func playButton(size: CGFloat) -> some View {
        Button { viewModel.togglePlayback() } label: {
            Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                .font(.system(size: size * 0.36, weight: .black))
                .foregroundStyle(Color.onCobalt)
                .contentTransition(.symbolEffect(.replace))
                .frame(width: size, height: size)
                .background(Color.cobalt, in: Circle())
                .overlay(Circle().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        }
        .buttonStyle(ScaleButtonStyle())
        .accessibilityLabel(viewModel.isPlaying ? "暂停" : "播放")
    }

    private var scrubber: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let x = w * min(1, max(0, viewModel.currentTime / max(timelineSeconds, 0.001)))
            ZStack(alignment: .leading) {
                Capsule().fill(Color.paperDeep)
                    .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
                    .frame(height: 12)
                Capsule().fill(Color.tomato)
                    .frame(width: max(12, x), height: 12)
                    .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
                Circle()
                    .fill(Color.butter)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: 24, height: 24)
                    .offset(x: min(max(0, x - 12), w - 24))
            }
            .frame(height: 24)
            // Where the play is, under the track: a full-width ink hairline, a 3 pt pistachio
            // segment per rally, and an ink tick at every hit.
            .overlay(alignment: .bottomLeading) {
                ZStack(alignment: .leading) {
                    Rectangle().fill(Color.ink.opacity(0.35)).frame(width: w, height: 1)
                    ForEach(Array(rallyBands.enumerated()), id: \.offset) { _, band in
                        Rectangle().fill(Color.pistachio.opacity(0.6))
                            .frame(width: max(3, w * (band.upperBound - band.lowerBound)), height: 3)
                            .offset(x: w * band.lowerBound)
                    }
                    // Hits sit below the band as short, light ticks: a rally has 5–15 of them, and
                    // full-height ink ticks would cover the band they belong to.
                    ForEach(Array(hitTicks.enumerated()), id: \.offset) { _, t in
                        Rectangle().fill(Color.ink.opacity(0.45)).frame(width: 1, height: 3).offset(x: w * t, y: 4)
                    }
                }
                .frame(width: w, height: 8, alignment: .leading)
                .offset(y: 12)
                .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { val in
                let p = max(0, min(1, val.location.x / w))
                viewModel.seek(to: p * timelineSeconds)
            })
        }
        .frame(height: 34)
        .accessibilityElement()
        .accessibilityLabel("播放进度")
        .accessibilityValue(viewModel.replayTimeText)
        .accessibilityAdjustableAction { dir in
            let step = max(1, viewModel.duration / 20)
            switch dir {
            case .increment: viewModel.seek(to: min(viewModel.duration, viewModel.currentTime + step))
            case .decrement: viewModel.seek(to: max(0, viewModel.currentTime - step))
            @unknown default: break
            }
        }
    }

    /// The timeline length: the player's once it knows it, else the recorded duration, so the
    /// rally bands are there from the first frame.
    private var timelineSeconds: Double {
        viewModel.duration > 0 && viewModel.duration.isFinite ? viewModel.duration : match.durationSeconds
    }

    /// Rally spans as fractions of the timeline, so the play is visible at a glance.
    private var rallyBands: [ClosedRange<Double>] {
        guard timelineSeconds > 0 else { return [] }
        let d = timelineSeconds * viewModel.fps
        return viewModel.rallyWindows.compactMap { r in
            let a = Double(r.lowerBound) / d, b = min(1, Double(r.upperBound) / d)
            return a < 1 ? a...max(a, b) : nil
        }
    }

    private var hitTicks: [Double] {
        guard timelineSeconds > 0 else { return [] }
        let d = timelineSeconds * viewModel.fps
        return match.allShots.map { Double($0.frameIndex) / d }.filter { $0 >= 0 && $0 < 1 }
    }

    /// The rally the playhead is in (or the one just before it) and the score after it.
    private var rallyChip: (label: String, near: Int, far: Int)? {
        let frame = Int(viewModel.currentTime * viewModel.fps)
        let rallies = match.rallies.sorted { $0.startFrame < $1.startFrame }
        guard !rallies.isEmpty else { return nil }
        let index = rallies.lastIndex { $0.startFrame - 30 <= frame } ?? 0
        let upTo = rallies.prefix(index + 1)
        return (String(format: "%02d", index + 1),
                upTo.filter { $0.winner == .near }.count, upTo.filter { $0.winner == .far }.count)
    }

    // MARK: - Shot label

    private func shotTypeLabel(_ shot: ShotPoint) -> some View {
        HStack(spacing: 5) {
            ShotGlyph(shot.shotType.glyphKind).frame(width: 16, height: 16)
            Text(shot.shotType.displayName).font(DisplayFace.font(16, relativeTo: .subheadline))
            if shot.speedKmh > 0 {
                Text("\(Int(shot.speedKmh))").font(AppFont.labelSmall)
            }
        }
        .foregroundStyle(Color.ink)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(shot.shotType.ink, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.5))
        .rotationEffect(.degrees(-4))
    }

    private func rateText(_ rate: Float) -> String {
        rate == 1.0 ? "1×" : String(format: "%.2g×", rate)
    }
}

// MARK: - ReplayViewModel convenience extensions

extension ReplayViewModel {
    var replayTimeText: String {
        let t = currentTime.isFinite ? currentTime : 0
        return String(format: "%d:%02d", Int(t) / 60, Int(t) % 60)
    }

    var replayProgress: Double {
        guard duration > 0, duration.isFinite else { return 0 }
        return min(1, max(0, currentTime / duration))
    }
}
