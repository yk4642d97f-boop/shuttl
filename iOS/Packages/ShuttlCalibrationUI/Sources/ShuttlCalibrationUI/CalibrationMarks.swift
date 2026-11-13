import CoreGraphics
import ShuttlCore
import ShuttlDesign
import SwiftUI

// MARK: - CornerHandle
// An ink-ringed paper disc with a small spot-ink core, and a tilted sticker naming
// the corner. Only the disc takes touches; the sticker is decoration.

struct CornerHandle: View {
    enum LabelPlacement { case above, below, leading, trailing }

    let name: String
    let selected: Bool
    let showsLabel: Bool
    let labelPlacement: LabelPlacement
    let position: CGPoint
    let onSelect: () -> Void
    let onDragStart: () -> Void
    let onDrag: (CGPoint) -> Void
    let onDragEnd: () -> Void
    let onAdjust: (AccessibilityAdjustmentDirection) -> Void

    /// Where the finger grabbed the handle relative to its centre, so the corner moves
    /// with the finger instead of jumping under it (off-centre grabs are the norm).
    @State private var grabOffset: CGSize?

    static let hitSize: CGFloat = 52
    private static let disc: CGFloat = 30

    var body: some View {
        ZStack {
            if selected {
                Circle()
                    .strokeBorder(Color.butter, style: StrokeStyle(lineWidth: 2, dash: [3, 3]))
                    .frame(width: Self.disc + 14, height: Self.disc + 14)
            }
            Circle()
                .fill(selected ? Color.butter : Color.paper)
                .overlay(Circle().strokeBorder(Color.ink, lineWidth: 3))
                .frame(width: Self.disc, height: Self.disc)
            Circle()
                .fill(selected ? Color.ink : Color.cobalt)
                .frame(width: 8, height: 8)
        }
        .frame(width: Self.hitSize, height: Self.hitSize)
        .contentShape(Circle())
        .overlay {
            if showsLabel {
                Sticker(CornerShortLabel.text(for: name), fill: selected ? .butter : .card, tilt: tilt)
                    .fixedSize()
                    .offset(labelOffset)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
        .scaleEffect(selected ? 1.1 : 1)
        .animation(AppAnimation.springFast, value: selected)
        .position(position)
        .onTapGesture(perform: onSelect)
        .gesture(DragGesture(minimumDistance: 3, coordinateSpace: .named("courtImageStage"))
            .onChanged { value in
                if grabOffset == nil {
                    grabOffset = CGSize(width: position.x - value.startLocation.x,
                                        height: position.y - value.startLocation.y)
                    onDragStart()
                }
                let offset = grabOffset ?? .zero
                onDrag(CGPoint(x: value.location.x + offset.width, y: value.location.y + offset.height))
            }
            .onEnded { _ in
                grabOffset = nil
                onDragEnd()
            })
        .accessibilityElement()
        .accessibilityLabel(CourtCornerLabel.chinese(for: name))
        .accessibilityValue(selected ? "已选中" : "未选中")
        .accessibilityHint("可拖动，或向左、向右微调")
        .accessibilityAddTraits(.isButton)
        .accessibilityAdjustableAction(onAdjust)
    }

    /// Centre-to-centre offset of the sticker (≈ 44×24 pt) from the disc, leaving a
    /// clear gap of about 8 pt so the disc edge and core stay visible.
    private var labelOffset: CGSize {
        let vertical = Self.disc / 2 + 8 + 12
        let horizontal = Self.disc / 2 + 8 + 22
        switch labelPlacement {
        case .above:    return CGSize(width: 0, height: -vertical)
        case .below:    return CGSize(width: 0, height: vertical)
        case .leading:  return CGSize(width: -horizontal, height: 0)
        case .trailing: return CGSize(width: horizontal, height: 0)
        }
    }

    private var tilt: Double {
        switch name {
        case "nb_rd": return -4
        case "nb_ld": return 3
        case "fb_rd": return 4
        default: return -3
        }
    }
}

// MARK: - LoupeView
// A printed magnifier: 2 pt card ring inside an ink outline, a gapped two-ink
// crosshair so the exact pixel stays visible, and a small tilted tag.

struct LoupeView: View {
    let image: CGImage?
    let frameSize: ImageSize
    let cornerName: String
    let normalizedPoint: CGPoint
    let rect: CGRect
    let zoom: CGFloat

    static let diameter: CGFloat = 128
    static let magnification: CGFloat = 2.5

    var body: some View {
        let d = Self.diameter
        let w = rect.width * zoom * Self.magnification
        let h = rect.height * zoom * Self.magnification
        ZStack {
            Color.night
            FrameContent(image: image, frameSize: frameSize)
                .frame(width: w, height: h)
                .offset(x: w * (0.5 - normalizedPoint.x), y: h * (0.5 - normalizedPoint.y))
            Crosshair(gap: 13)
                .stroke(Color.ink, style: StrokeStyle(lineWidth: 3, lineCap: .round))
            Crosshair(gap: 13)
                .stroke(Color.card, style: StrokeStyle(lineWidth: 1.25, lineCap: .round))
            Circle()
                .strokeBorder(Color.butter, lineWidth: 1.5)
                .frame(width: 11, height: 11)
        }
        .frame(width: d, height: d)
        .clipShape(Circle())
        .overlay(Circle().strokeBorder(Color.card, lineWidth: 2).padding(AppStroke.ink))
        .overlay(Circle().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .background(Circle().fill(Color.ink).offset(x: 3, y: 4))
        .overlay(alignment: .top) {
            Sticker("×2.5 · \(CornerShortLabel.text(for: cornerName))", fill: .butter, tilt: -4)
                .fixedSize()
                .offset(y: -10)
        }
        .accessibilityHidden(true)
    }
}

/// Two hairlines through the centre with a gap where they would cross.
struct Crosshair: Shape {
    var gap: CGFloat

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let c = CGPoint(x: rect.midX, y: rect.midY)
        p.move(to: CGPoint(x: rect.minX, y: c.y)); p.addLine(to: CGPoint(x: c.x - gap, y: c.y))
        p.move(to: CGPoint(x: c.x + gap, y: c.y)); p.addLine(to: CGPoint(x: rect.maxX, y: c.y))
        p.move(to: CGPoint(x: c.x, y: rect.minY)); p.addLine(to: CGPoint(x: c.x, y: c.y - gap))
        p.move(to: CGPoint(x: c.x, y: c.y + gap)); p.addLine(to: CGPoint(x: c.x, y: rect.maxY))
        return p
    }
}

// MARK: - CourtLinesOverlay
// The projected court printed in butter over a thin ink underlay (so it reads on
// bright floors too), with the net as a dashed tape. When the quad is invalid the
// four handles are joined by a dashed tomato outline instead, showing what is wrong.

struct CourtLinesOverlay: View {
    let taps: CourtCornerTaps
    let issue: CourtQuadIssue?
    /// Image pixels → stage points.
    let transform: (CGPoint) -> CGPoint
    /// Screen positions of the active corners, in `taps.cornerSet.names` order.
    let corners: [CGPoint]

    var body: some View {
        Canvas { context, _ in
            guard issue == nil else {
                drawInvalidQuad(in: &context)
                return
            }
            for (name, points) in CourtOverlayGeometry.projectedLines(for: taps, samplesPerLine: 24) {
                guard points.count > 1 else { continue }
                var path = Path()
                path.addLines(points.map(transform))
                let role = CourtLineRole(name)
                let width: CGFloat = role == .outer ? 2.5 : (role == .service ? 1.6 : 1.1)
                context.stroke(path, with: .color(Color.ink.opacity(0.55)),
                               style: StrokeStyle(lineWidth: width + 2, lineCap: .round, lineJoin: .round))
                context.stroke(path, with: .color(Color.butter.opacity(role == .inner ? 0.6 : 1)),
                               style: StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round))
            }
            let net = CourtOverlayGeometry.projectedNetLine(for: taps).map(transform)
            if net.count > 1 {
                var path = Path()
                path.addLines(net)
                context.stroke(path, with: .color(Color.ink.opacity(0.8)),
                               style: StrokeStyle(lineWidth: 5, lineCap: .round))
                context.stroke(path, with: .color(Color.butter),
                               style: StrokeStyle(lineWidth: 2.5, lineCap: .butt, dash: [7, 5]))
            }
        }
    }

    private func drawInvalidQuad(in context: inout GraphicsContext) {
        // cornerSet.names order is near-right, near-left, far-right, far-left; walk the perimeter.
        guard corners.count == 4 else { return }
        var path = Path()
        path.addLines([corners[0], corners[1], corners[3], corners[2]])
        path.closeSubpath()
        context.stroke(path, with: .color(Color.ink.opacity(0.7)),
                       style: StrokeStyle(lineWidth: 4.5, lineCap: .round, lineJoin: .round))
        context.stroke(path, with: .color(Color.tomato),
                       style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round, dash: [6, 5]))
    }
}

// MARK: - QuadIssueChip
// The "invalid quad" message as a tomato sticker: ink type on tomato (4.8:1),
// ink outline, an offset ink print shadow, stuck on slightly askew.

struct QuadIssueChip: View {
    let issue: CourtQuadIssue

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.sm + 2, style: .continuous)
        HStack(alignment: .center, spacing: AppSpacing.sm + 2) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 17, weight: .black))
            VStack(alignment: .leading, spacing: 1) {
                Text("四角位置无效")
                    .font(DisplayFace.font(19, relativeTo: .headline))
                Text(issue.hint)
                    .font(.system(.footnote).weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .foregroundStyle(Color.onTomato)
        .padding(.horizontal, AppSpacing.md - 2)
        .padding(.vertical, AppSpacing.sm + 1)
        .background(Color.tomato, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .printShadow(shape, color: .ink, x: 2, y: 3)
        .rotationEffect(.degrees(-1.5))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("四角位置无效，\(issue.hint)")
    }
}

extension CourtQuadIssue {
    /// What to do about it, in plain Chinese.
    var hint: String {
        switch self {
        case .missingCorner:         return "还缺一个角点，请把四个圆点都放好"
        case .pointOutOfBounds:      return "有圆点跑出了画面，请拖回画面内"
        case .nonConvex:             return "四条边交叉了，请检查左右两边是否对调"
        case .degenerate:            return "圆点靠得太近，请拉开到球场四角"
        case .farBaselineNotAbove:   return "远端底线要在近端底线上方"
        case .farBaselineNotShorter: return "远端底线应比近端底线短（透视）"
        }
    }
}

// MARK: - StageHint
// A small night tag in the stage corner: gesture help, or the zoom level.

struct StageHint: View {
    let zoom: CGFloat
    let isPlaceholder: Bool

    var body: some View {
        HStack(spacing: AppSpacing.xs + 1) {
            Image(systemName: zoom > 1.01 ? "plus.magnifyingglass" : "hand.draw")
                .font(.system(size: 10, weight: .bold))
            Text(text)
                .font(.system(.caption2, design: .monospaced).weight(.semibold))
                .lineLimit(1)
        }
        .foregroundStyle(Color.paper.opacity(0.82))
        .padding(.horizontal, AppSpacing.sm + 1)
        .padding(.vertical, AppSpacing.xs + 1)
        .background(Color.night.opacity(0.72), in: Capsule())
        .overlay(Capsule().strokeBorder(Color.paper.opacity(0.22), lineWidth: 1))
        .accessibilityHidden(true)
    }

    private var text: String {
        if zoom > 1.01 { return "放大 \(Int((zoom * 100).rounded()))%" }
        return isPlaceholder ? "示意画面 · 双指缩放、拖动画面" : "双指缩放 · 拖动画面"
    }
}
