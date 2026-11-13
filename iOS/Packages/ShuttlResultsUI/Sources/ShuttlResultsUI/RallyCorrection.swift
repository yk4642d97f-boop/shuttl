import SwiftUI
import ShuttlCore
import ShuttlDesign

// MARK: - Rally call (one-tap score fix)

/// The user's call for one rally. `restoreAI` drops their fix and brings the AI's call back.
public enum RallyCall: Sendable, Equatable {
    case nearWon, farWon, notARally, restoreAI
}

public extension RallySummary {
    /// The user changed this rally's outcome.
    var isUserCorrected: Bool { outcomeSource == "user" }

    /// The AI wasn't sure who won (and the user hasn't settled it).
    var isUncertain: Bool {
        !isUserCorrected && (winner == .unknown || outcomeConfidence < 0.6)
    }
}

// MARK: - RallyCorrectionSheet
// A small paper slip: "这一分是谁的？" and four printed choices. Big targets, because
// this gets used courtside between rallies.

public struct RallyCorrectionSheet: View {
    public let rally: RallySummary
    public let onCall: (RallyCall) -> Void

    public init(rally: RallySummary, onCall: @escaping (RallyCall) -> Void) {
        self.rally = rally
        self.onCall = onCall
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("回合 \(String(format: "%02d", rally.index)) · \(rally.shots.count) 拍")
                        .instrumentLabel(AppFont.labelSmall)
                    OverprintText("这一分是谁的？", font: AppFont.numeral(34, relativeTo: .title),
                                  plate: .butter, offset: CGSize(width: 2.5, height: 2.5))
                }
                Spacer(minLength: 0)
                if rally.isUserCorrected {
                    Sticker("已改", symbol: "pencil", fill: .ink, foreground: .paper, tilt: 4)
                } else {
                    Sticker(aiLabel, symbol: "sparkle", fill: .lilac, tilt: 4)
                }
            }

            HStack(spacing: AppSpacing.sm) {
                choice("近端得分", side: .near, call: .nearWon)
                choice("远端得分", side: .far, call: .farWon)
            }

            Button { onCall(.notARally) } label: {
                Label("这不算一分", systemImage: "nosign")
                    .font(DisplayFace.font(19, relativeTo: .headline))
                    .foregroundStyle(Color.ink)
                    .frame(maxWidth: .infinity, minHeight: 50)
                    .background(Color.card, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            }
            .buttonStyle(PrintButtonStyle(shape: Capsule(), plate: .ink, depth: 3))
            .accessibilityHint("热身球或重发，从回合和比分里去掉")

            if rally.isUserCorrected {
                Button { onCall(.restoreAI) } label: {
                    Label("恢复 AI 判断", systemImage: "arrow.uturn.backward")
                        .font(AppFont.callout.weight(.semibold))
                        .foregroundStyle(Color.cobalt)
                        .frame(maxWidth: .infinity, minHeight: 44)
                }
                .buttonStyle(ScaleButtonStyle())
            }
        }
        .padding(AppSpacing.gutter)
        .padding(.top, AppSpacing.sm)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        #if os(iOS)
        .presentationDetents([.height(rally.isUserCorrected ? 330 : 286)])
        .presentationDragIndicator(.visible)
        .presentationBackground { Color.paper.paperGrain() }
        .presentationCornerRadius(AppRadius.lg)
        #endif
    }

    private var aiLabel: String {
        rally.winner == .unknown ? "AI 没判出来" : "AI 判\(rally.winner.chineseName) · \(Int((rally.outcomeConfidence * 100).rounded()))%"
    }

    private func choice(_ title: String, side: CourtSide, call: RallyCall) -> some View {
        let current = rally.winner == side
        return Button { onCall(call) } label: {
            VStack(spacing: 4) {
                Text(title).font(DisplayFace.font(24, relativeTo: .title3))
                if current {
                    Text(rally.isUserCorrected ? "你改的" : "AI 的判断")
                        .font(.system(size: 11, weight: .bold))
                        .opacity(0.8)
                }
            }
            .foregroundStyle(side.onInk)
            .frame(maxWidth: .infinity, minHeight: 72)
            .background(side.ink, in: RoundedRectangle(cornerRadius: AppRadius.md, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: AppRadius.md, style: .continuous)
                .strokeBorder(Color.ink, lineWidth: current ? 3 : AppStroke.ink))
            .overlay(alignment: .topTrailing) {
                if current {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20, weight: .bold))
                        .foregroundStyle(Color.butter, Color.ink)
                        .offset(x: 6, y: -6)
                }
            }
        }
        .buttonStyle(PrintButtonStyle(shape: RoundedRectangle(cornerRadius: AppRadius.md, style: .continuous),
                                      plate: .ink, depth: 4))
        .accessibilityLabel(title)
        .accessibilityAddTraits(current ? .isSelected : [])
    }
}

// MARK: - VoiceOver calls
// The same four choices as the slip, straight from the rally row's actions rotor,
// so a VoiceOver user doesn't have to open the sheet and find the buttons.

struct RallyCallActions: ViewModifier {
    let rally: RallySummary
    let onCorrect: ((Int, RallyCall) -> Void)?
    let openSlip: () -> Void

    func body(content: Content) -> some View {
        if let onCorrect {
            content.accessibilityActions {
                if rally.winner != .near {
                    Button("判近端得分") { call(.nearWon, "已改为近端得分", onCorrect) }
                }
                if rally.winner != .far {
                    Button("判远端得分") { call(.farWon, "已改为远端得分", onCorrect) }
                }
                Button("这不算一分") { call(.notARally, "已去掉这一分", onCorrect) }
                if rally.isUserCorrected {
                    Button("恢复 AI 判断") { call(.restoreAI, "已恢复 AI 判断", onCorrect) }
                }
                Button("打开改判单", action: openSlip)
            }
        } else {
            content
        }
    }

    private func call(_ call: RallyCall, _ announcement: String, _ onCorrect: (Int, RallyCall) -> Void) {
        onCorrect(rally.startFrame, call)
        AccessibilityNotification.Announcement("回合 \(rally.index)，\(announcement)").post()
    }
}
