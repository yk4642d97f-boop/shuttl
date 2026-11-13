import CoreGraphics
import ShuttlCore
import ShuttlDesign
import SwiftUI
#if os(iOS)
import UIKit
#endif

public enum CourtCalibrationPreviewState: Sendable, Equatable {
    case normal
    case selectedWithLoupe(String)
    case invalid
}

/// Manual four-corner calibration over an upright video frame, in the app's
/// "Riso Court · Overprint" look: paper bars top and bottom, the frame on a night
/// surround, ink-ringed paper handles and the projected court printed in butter.
/// Pass `image: nil` (previews, tests) to get a drawn stand-in court instead of a photo.
@MainActor
public struct CourtCalibrationView: View {
    private let image: CGImage?
    private let frameSize: ImageSize
    private let onDone: (CourtCornerTaps) -> Void
    private let onCancel: () -> Void
    private let previewState: CourtCalibrationPreviewState

    @State private var taps: CourtCornerTaps
    @State private var selectedName: String?
    @State private var zoom: CGFloat = 1
    @State private var pan: CGSize = .zero
    @State private var activeDragName: String?
    @State private var undoStack: [CourtCornerTaps] = []

    @MainActor
    public init(
        image: CGImage?,
        initial: CourtCornerTaps? = nil,
        onDone: @escaping (CourtCornerTaps) -> Void,
        onCancel: @escaping () -> Void,
        previewState: CourtCalibrationPreviewState = .normal
    ) {
        DisplayFace.register()
        self.image = image
        self.onDone = onDone
        self.onCancel = onCancel
        self.previewState = previewState
        let size = image.map { ImageSize(width: $0.width, height: $0.height) }
            ?? initial?.imageSize ?? CalibrationDefaults.placeholderSize
        self.frameSize = size
        var value = (initial ?? CourtCalibrationStore.shared.load(for: size))?.normalized(to: size)
            ?? CalibrationDefaults.taps(size: size)
        if case .invalid = previewState {
            value["fb_ld"] = CGPoint(x: 0.62, y: 0.86)
            value["fb_rd"] = CGPoint(x: 0.38, y: 0.86)
        }
        _taps = State(initialValue: value)
        if case .selectedWithLoupe(let name) = previewState { _selectedName = State(initialValue: name) }
    }

    public var body: some View {
        GeometryReader { screen in
            let wide = screen.size.width > screen.size.height
            VStack(spacing: 0) {
                topBar(compact: wide)
                CalibrationStage(
                    image: image,
                    frameSize: frameSize,
                    taps: taps,
                    selectedName: selectedName,
                    selectCorner: { selectedName = $0 },
                    zoom: $zoom,
                    pan: $pan,
                    activeDragName: $activeDragName,
                    moveCorner: moveCorner,
                    beginCornerDrag: beginCornerDrag,
                    endCornerDrag: endCornerDrag,
                    previewState: previewState
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.night)
                .overlay(alignment: .topTrailing) {
                    if activeDragName == nil {
                        cornerSetToggle
                            .padding(AppSpacing.sm + 2)
                            .transition(.opacity)
                    }
                }
                .clipped()
                if wide { wideBottomBar } else { portraitBottomBar }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.night.ignoresSafeArea())
        }
        .preferredColorScheme(.light)
        .dynamicTypeSize(...DynamicTypeSize.accessibility3)
    }

    // MARK: Bars

    private func topBar(compact: Bool) -> some View {
        HStack(alignment: .center, spacing: AppSpacing.md) {
            IconButton(systemName: "xmark", size: 40, iconSize: 15, style: .filled, label: "取消标定", action: onCancel)
            VStack(alignment: .leading, spacing: compact ? 0 : AppSpacing.xxs) {
                OverprintText("标定球场", font: DisplayFace.font(compact ? 24 : 30, relativeTo: .title2),
                              plate: .butter, offset: CGSize(width: 2.5, height: 2), registers: false)
                    .accessibilityAddTraits(.isHeader)
                Text(taps.cornerSet == .baseline ? "把四个圆点拖到球场最外圈边线的四个角"
                                                 : "近处两点拖到前发球线和边线的交点，远处两点拖到远端底线两角")
                    .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                    .lineLimit(compact ? 1 : 2)
                    .minimumScaleFactor(0.8)
                    .fixedSize(horizontal: false, vertical: !compact)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, compact ? AppSpacing.xs : AppSpacing.sm)
        .padding(.bottom, compact ? AppSpacing.xs : AppSpacing.sm + 2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background { barBackground.ignoresSafeArea(edges: .top) }
        .overlay(alignment: .bottom) { Rectangle().fill(Color.ink).frame(height: AppStroke.ink) }
    }

    private var portraitBottomBar: some View {
        VStack(spacing: AppSpacing.sm + 4) {
            HStack(alignment: .bottom, spacing: AppSpacing.sm + 2) {
                undoButton
                resetButton
                Spacer(minLength: AppSpacing.xs)
                nudgePad
            }
            confirmButton
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, AppSpacing.sm + 4)
        .padding(.bottom, AppSpacing.xs)
        .background { barBackground.ignoresSafeArea(edges: .bottom) }
        .overlay(alignment: .top) { Rectangle().fill(Color.ink).frame(height: AppStroke.ink) }
    }

    private var wideBottomBar: some View {
        HStack(alignment: .center, spacing: AppSpacing.sm + 2) {
            undoButton
            resetButton
            nudgePad
            Spacer(minLength: AppSpacing.sm)
            confirmButton.frame(maxWidth: 220)
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, AppSpacing.sm)
        .padding(.bottom, AppSpacing.xxs)
        .background { barBackground.ignoresSafeArea(edges: .bottom) }
        .overlay(alignment: .top) { Rectangle().fill(Color.ink).frame(height: AppStroke.ink) }
    }

    private var barBackground: some View {
        Color.paper.paperGrain()
    }

    // MARK: Controls

    /// "近端底线角看不到？" — swaps the near baseline corners for the near short-service-line
    /// corners (4.72 m up the court), which stay in frame with a tripod close behind the baseline.
    private var cornerSetToggle: some View {
        let service = taps.cornerSet == .nearServiceLine
        return Button {
            undoStack.append(taps)
            withAnimation(AppAnimation.spring) {
                taps = taps.switched(to: service ? .baseline : .nearServiceLine)
                selectedName = nil
            }
            calibrationGrabFeedback()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: service ? "arrow.uturn.backward" : "arrow.up.to.line")
                    .font(.system(size: 11, weight: .black))
                Text(service ? "改回标底线四角" : "近端底线角看不到？改标发球线")
                    .font(.system(size: 12, weight: .bold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(Color.ink)
            .padding(.horizontal, 10)
            .frame(minHeight: 34)
            .background(service ? Color.butter : Color.paper, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
            .rotationEffect(.degrees(service ? 2 : 0))
        }
        .buttonStyle(ScaleButtonStyle())
        .accessibilityLabel(service ? "改回标近端底线四角" : "近端底线角看不到，改标近端发球线")
        .accessibilityHint("近端两点会移到对应位置，可以继续拖动")
    }

    private var undoButton: some View {
        let enabled = !undoStack.isEmpty
        return toolColumn("撤销") {
            IconButton(systemName: "arrow.uturn.backward", size: 44, iconSize: 16, style: .filled,
                       label: "撤销", action: undo)
        }
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.4)
        .help("撤销")
    }

    private var resetButton: some View {
        toolColumn("重置") {
            IconButton(systemName: "arrow.counterclockwise", size: 44, iconSize: 16, style: .filled,
                       label: "重置四角", action: reset)
        }
        .help("重置")
    }

    private func toolColumn<Content: View>(_ caption: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(spacing: AppSpacing.xs) {
            content()
            Text(caption)
                .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                .lineLimit(1)
                .fixedSize()
                .accessibilityHidden(true)
        }
    }

    /// Four one-pixel nudges printed as a single ink capsule, like a hardware d-pad.
    private var nudgePad: some View {
        VStack(alignment: .trailing, spacing: AppSpacing.xs) {
            HStack(spacing: 0) {
                nudgeButton("arrow.left", label: "向左微调", dx: -1, dy: 0)
                nudgeDivider
                nudgeButton("arrow.up", label: "向上微调", dx: 0, dy: -1)
                nudgeDivider
                nudgeButton("arrow.down", label: "向下微调", dx: 0, dy: 1)
                nudgeDivider
                nudgeButton("arrow.right", label: "向右微调", dx: 1, dy: 0)
            }
            .padding(.horizontal, 3)
            .background(Color.ink, in: Capsule())
            Text("微调 1 像素 · \(CornerShortLabel.text(for: selectedName ?? taps.cornerSet.names[0]))")
                .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                .lineLimit(1)
                .fixedSize()
                .accessibilityHidden(true)
        }
    }

    private var nudgeDivider: some View {
        Rectangle().fill(Color.paper.opacity(0.18)).frame(width: 1, height: 20)
    }

    private func nudgeButton(_ symbol: String, label: String, dx: CGFloat, dy: CGFloat) -> some View {
        Button { nudge(dx: dx, dy: dy) } label: {
            Image(systemName: symbol)
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(Color.paper)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(ScaleButtonStyle())
        .accessibilityLabel(label)
        .accessibilityHint("每次微调一像素")
        .help(label)
    }

    private var confirmButton: some View {
        let valid = CourtQuadValidator.isValid(taps)
        return PillButton(title: "完成标定", icon: valid ? "checkmark" : nil, style: .primary,
                          tint: valid ? nil : .paperDeep, tintText: .inkFaint, action: finish)
            .disabled(!valid)
            .saturation(valid ? 1 : 0.4)
            .accessibilityHint(valid ? "保存球场角点" : "四个圆点位置有效后可完成")
    }

    // MARK: Actions

    private func beginCornerDrag(_ name: String) {
        guard activeDragName != name else { return }
        undoStack.append(taps)
        selectedName = name
        activeDragName = name
        calibrationGrabFeedback()
    }

    private func moveCorner(_ name: String, to point: CGPoint) {
        guard taps[name] != point else { return }
        taps[name] = CGPoint(x: min(max(point.x, 0), 1), y: min(max(point.y, 0), 1))
    }

    private func endCornerDrag() {
        activeDragName = nil
        calibrationDropFeedback()
    }

    private func nudge(dx: CGFloat, dy: CGFloat) {
        let name = selectedName ?? taps.cornerSet.names[0]
        guard let point = taps[name] else { return }
        if selectedName == nil { selectedName = name }
        undoStack.append(taps)
        taps[name] = CGPoint(
            x: min(max(point.x + dx / CGFloat(taps.imageSize.width), 0), 1),
            y: min(max(point.y + dy / CGFloat(taps.imageSize.height), 0), 1)
        )
    }

    private func undo() {
        guard let previous = undoStack.popLast() else { return }
        taps = previous
    }

    private func reset() {
        undoStack.append(taps)
        let set = taps.cornerSet
        taps = CalibrationDefaults.taps(size: frameSize).switched(to: set)
        selectedName = nil
        zoom = 1
        pan = .zero
    }

    private func finish() {
        guard CourtQuadValidator.isValid(taps) else { return }
        // Only the active set goes on: a stale pair from the other set would be read as extra keypoints.
        let result = CourtCornerTaps(points: taps.activePoints, imageSize: taps.imageSize)
        try? CourtCalibrationStore.shared.save(result)
        onDone(result)
    }
}

/// Starting quad and stand-in frame size. Kept free of the main actor so the
/// placeholder court can draw the same default from inside a Canvas.
enum CalibrationDefaults {
    static let placeholderSize = ImageSize(width: 1920, height: 1080)

    static func taps(size: ImageSize) -> CourtCornerTaps {
        CourtCornerTaps(imageSize: size, points: [
            "nb_ld": CGPoint(x: 0.82, y: 0.82),
            "nb_rd": CGPoint(x: 0.18, y: 0.82),
            "fb_ld": CGPoint(x: 0.66, y: 0.25),
            "fb_rd": CGPoint(x: 0.34, y: 0.25),
        ])
    }
}

/// Two-character corner name for handle stickers ("近左", "远右"), derived from
/// `CourtCornerLabel.chinese` so the short and long forms can never disagree.
enum CornerShortLabel {
    static func text(for name: String) -> String {
        // Service-line corners must not read like the baseline ones ("近右").
        if name.hasPrefix("ns_") { return name.hasSuffix("ld") ? "发球线右" : "发球线左" }
        let full = Array(CourtCornerLabel.chinese(for: name))
        guard let first = full.first, let corner = full.lastIndex(of: "角"), corner > 0 else {
            return String(full.prefix(2))
        }
        return String([first, full[corner - 1]])
    }

    static func isFar(_ name: String) -> Bool { name.hasPrefix("f") }
}

#if os(iOS)
@MainActor
private func calibrationGrabFeedback() {
    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
}

@MainActor
private func calibrationDropFeedback() {
    UIImpactFeedbackGenerator(style: .light).impactOccurred()
}
#else
@MainActor private func calibrationGrabFeedback() {}
@MainActor private func calibrationDropFeedback() {}
#endif
