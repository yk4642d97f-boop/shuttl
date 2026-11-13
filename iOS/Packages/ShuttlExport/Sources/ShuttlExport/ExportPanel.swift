import ShuttlDesign
import SwiftUI

/// The export control's four states in the app's printed look: an idle cobalt pill,
/// a progress card (lilac sticker + ink progress bar), a "done" card with share and
/// save, and a tomato error card with retry. Platform-neutral so it can be rendered
/// in tests; `ExportButton` supplies the live ShareLink and Photos saving on iOS.
enum ExportPhase: Equatable {
    case idle
    case exporting(Double)
    case done
    case failed(String)
}

enum ExportSaveStatus: Equatable {
    case none, saving, saved
    case failed(String)
}

struct ExportPanel<Share: View>: View {
    var phase: ExportPhase
    var saveStatus: ExportSaveStatus
    var start: () -> Void
    var save: () -> Void
    @ViewBuilder var share: () -> Share

    var body: some View {
        switch phase {
        case .idle: idle
        case .exporting(let value): exporting(value)
        case .done: done
        case .failed(let message): failed(message)
        }
    }

    private var idle: some View {
        VStack(alignment: .leading, spacing: AppSpacing.sm) {
            PillButton(title: "导出分享视频", icon: "film", style: .primary, action: start)
            Text("带轨迹、球速和比分的 MP4")
                .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                .padding(.leading, AppSpacing.xs)
        }
    }

    private func exporting(_ value: Double) -> some View {
        let percent = Int((min(max(value, 0), 1) * 100).rounded())
        return AppCard(style: .print(.ink)) {
            VStack(alignment: .leading, spacing: AppSpacing.sm + 2) {
                HStack(alignment: .firstTextBaseline) {
                    Sticker("导出中", symbol: "sparkle", fill: .lilac)
                    Spacer(minLength: AppSpacing.sm)
                    Text("\(percent)%")
                        .font(AppFont.numeral(34, relativeTo: .title2))
                        .monospacedDigit()
                        .foregroundStyle(Color.ink)
                        .contentTransition(.numericText())
                }
                InkProgressBar(value: value, fill: .butter, track: .paperDeep, height: 12)
                Text("正在把轨迹印进视频，请停留在此页面")
                    .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("正在导出视频 \(percent)%")
    }

    private var done: some View {
        AppCard(style: .print(.ink)) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                HStack(spacing: AppSpacing.sm) {
                    Sticker("视频已生成", symbol: "checkmark", fill: .pistachio)
                    Spacer(minLength: 0)
                }
                share()
                saveButton
                if case .failed(let message) = saveStatus {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .font(.system(.footnote).weight(.semibold))
                        .foregroundStyle(Color.tomatoText)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    @ViewBuilder private var saveButton: some View {
        switch saveStatus {
        case .saved:
            // A status, not an action: printed flat in the success ink, no press plate.
            HStack(spacing: AppSpacing.sm) {
                Image(systemName: "checkmark").font(.system(size: 16, weight: .bold))
                Text("已保存到相册").font(DisplayFace.font(21, relativeTo: .headline))
            }
            .foregroundStyle(Color.onPistachio)
            .frame(maxWidth: .infinity)
            .frame(height: 56)
            .background(Color.pistachio, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            .padding(.bottom, 4)
            .accessibilityElement(children: .combine)
        case .saving:
            PillButton(title: "保存中", isLoading: true, style: .secondary, action: {})
        default:
            PillButton(title: "保存到相册", icon: "square.and.arrow.down", style: .secondary, action: save)
        }
    }

    private func failed(_ message: String) -> some View {
        AppCard(style: .print(.ink)) {
            VStack(alignment: .leading, spacing: AppSpacing.sm + 4) {
                Sticker("导出失败", symbol: "exclamationmark.triangle.fill", fill: .tomato)
                Text(message)
                    .font(AppFont.callout)
                    .foregroundStyle(Color.inkSoft)
                    .fixedSize(horizontal: false, vertical: true)
                PillButton(title: "再试一次", icon: "arrow.clockwise", style: .primary, action: start)
            }
        }
    }
}

/// The share control's face: the same printed cobalt pill as `PillButton(.primary)`,
/// for wrapping in a `ShareLink` (which needs its own label).
struct ExportShareLabel: View {
    var body: some View {
        HStack(spacing: AppSpacing.sm) {
            Image(systemName: "square.and.arrow.up").font(.system(size: 16, weight: .bold))
            Text("分享视频").font(DisplayFace.font(21, relativeTo: .headline))
        }
        .foregroundStyle(Color.onCobalt)
        .frame(maxWidth: .infinity)
        .frame(height: 56)
        .background(Color.cobalt, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
    }
}
