import SwiftUI
import ShuttlCore
import ShuttlCalibrationUI

// Manual court corners: the fallback when the analysis couldn't find the court itself.
// The only file that imports ShuttlCalibrationUI (its geometry type names overlap the app's).

/// Card on the detail page of an uncalibrated match.
struct CalibrationPromptCard: View {
    let isLoading: Bool
    /// Why automatic calibration gave up (`CalibrationAdvice`); nil falls back to the generic line.
    var advice: String? = nil
    let action: () -> Void

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            HStack(alignment: .top, spacing: AppSpacing.md) {
                cornerGlyph
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    Text("没认出球场")
                        .font(AppFont.title2)
                        .foregroundStyle(Color.ink)
                    if let advice {
                        Text(advice)
                            .font(AppFont.callout.weight(.semibold))
                            .foregroundStyle(Color.ink)
                            .fixedSize(horizontal: false, vertical: true)
                        Text("没有球场就没有球速和落点。手动标一下四个角，大约 20 秒，会重新分析这一场。")
                            .font(AppFont.caption)
                            .foregroundStyle(Color.ink.opacity(0.78))
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        Text("所以没有球速和落点。手动标一下四个角，大约 20 秒，然后会重新分析这一场。")
                            .font(AppFont.callout)
                            .foregroundStyle(Color.ink)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            PillButton(title: "手动标四个角", icon: "viewfinder", isLoading: isLoading, action: action)
        }
        .padding(AppSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            ZStack {
                Color.butter
                Halftone(color: .ink.opacity(0.12), spacing: 9, maxDot: 3, focus: .bottomTrailing)
            }
            .clipShape(shape)
        }
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .printShadow(shape, color: .ink, x: 0, y: 5)
        .accessibilityElement(children: .combine)
    }

    /// A court in perspective with its four corners stamped tomato: what the user will do.
    private var cornerGlyph: some View {
        Canvas { ctx, size in
            let w = size.width, h = size.height
            let tl = CGPoint(x: w * 0.3, y: h * 0.22), tr = CGPoint(x: w * 0.7, y: h * 0.22)
            let bl = CGPoint(x: w * 0.06, y: h * 0.84), br = CGPoint(x: w * 0.94, y: h * 0.84)
            var court = Path()
            court.addLines([tl, tr, br, bl]); court.closeSubpath()
            ctx.fill(court, with: .color(.pistachio))
            ctx.stroke(court, with: .color(.ink), lineWidth: 1.5)
            var net = Path()
            net.move(to: CGPoint(x: (tl.x + bl.x) / 2, y: (tl.y + bl.y) / 2))
            net.addLine(to: CGPoint(x: (tr.x + br.x) / 2, y: (tr.y + br.y) / 2))
            ctx.stroke(net, with: .color(.ink), style: StrokeStyle(lineWidth: 2, dash: [2, 2]))
            for p in [tl, tr, bl, br] {
                let r: CGFloat = 5.5
                let dot = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
                ctx.fill(dot, with: .color(.tomato))
                ctx.stroke(dot, with: .color(.ink), lineWidth: 1.5)
            }
        }
        .frame(width: 64, height: 52)
        .padding(6)
        .background(Color.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .rotationEffect(.degrees(-4))
        .accessibilityHidden(true)
    }
}

/// A still from the match to tap corners on.
struct CalibrationFrame: Identifiable {
    let id = UUID()
    let image: CGImage
    /// Corners the user placed last time (0…1 of the upright frame), to start from.
    var initial: [String: CGPoint]?
}

/// Full-screen corner-drag screen; hands normalised CourtModel corners back.
struct CourtCalibrationSheet: View {
    let frame: CalibrationFrame
    let onDone: ([String: CGPoint]) -> Void
    let onCancel: () -> Void

    var body: some View {
        let size = ImageSize(width: frame.image.width, height: frame.image.height)
        CourtCalibrationView(image: frame.image,
                             initial: frame.initial.map { CourtCornerTaps(points: $0, imageSize: size) },
                             onDone: { taps in
                                 try? CourtCalibrationStore.shared.save(taps)
                                 onDone(taps.points)
                             },
                             onCancel: onCancel)
    }
}

extension MatchInputs {
    /// Saved manual corners as points, to prefill the calibration screen.
    var cornerPoints: [String: CGPoint]? {
        manualCorners.map { $0.mapValues { CGPoint(x: $0.x, y: $0.y) } }
    }
}

#if DEBUG
extension CalibrationFrame {
    /// Demo still: the fake court backdrop rendered at 1080p (the simulator has no match video).
    @MainActor static func demo() -> CalibrationFrame? {
        let renderer = ImageRenderer(content: DemoCourtBackdrop().frame(width: 1920, height: 1080))
        renderer.scale = 1
        return renderer.cgImage.map { CalibrationFrame(image: $0) }
    }
}
#endif
