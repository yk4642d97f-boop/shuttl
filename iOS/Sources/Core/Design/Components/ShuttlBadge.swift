import SwiftUI

// MARK: - Match status inks (single source for every status visual)

extension MatchStatus {
    var ink: Color {
        switch self {
        case .recording: return .tomato
        case .uploading: return .butter
        case .analyzing: return .lilac
        case .done:      return .pistachio
        case .failed:    return .tomato
        }
    }

    var symbol: String {
        switch self {
        case .recording: return "circle.fill"
        case .uploading: return "arrow.up"
        case .analyzing: return "sparkle"
        case .done:      return "checkmark"
        case .failed:    return "exclamationmark"
        }
    }

    var shortLabel: String {
        switch self {
        case .recording: return "录制中"
        case .uploading: return "上传中"
        case .analyzing: return "分析中"
        case .done:      return "已完成"
        case .failed:    return "失败"
        }
    }

    var isWorking: Bool { self == .uploading || self == .analyzing }
}

// MARK: - ShuttlBadge
// A sticker: spot-ink capsule with an ink outline, stuck on slightly askew.
// In-progress states breathe so you can tell work is happening.

/// A live match that isn't being worked on right now: its turn hasn't come, or the phone
/// is cooling down. It reads quieter than "分析中": paper with a lilac outline, no sparkle.
enum MatchWaiting: Equatable {
    /// Another analysis holds the slot.
    case queued
    /// The phone is too warm (thermal admission hold).
    case cooling
    /// The record screen's camera is running (it shares the codec and heats the phone).
    case camera
    /// A take is being recorded.
    case recording

    init?(localStage: String?) {
        switch localStage {
        case "queued": self = .queued
        case "cooling", "cooling_camera": self = .cooling
        case "camera": self = .camera
        case "recording": self = .recording
        default: return nil
        }
    }

    var label: String {
        switch self {
        case .queued: return "排队中"
        case .cooling: return "降温中"
        case .camera: return "等相机"
        case .recording: return "等录完"
        }
    }

    var symbol: String {
        switch self {
        case .queued: return "hourglass"
        case .cooling: return "thermometer.medium"
        case .camera: return "camera.fill"
        case .recording: return "record.circle"
        }
    }
}

struct ShuttlBadge: View {
    let status: MatchStatus
    var tilt: Double = -3
    /// Set while an analysing match waits its turn (or for the phone to cool).
    var waiting: MatchWaiting? = nil

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var breathe = false

    var body: some View {
        if let waiting, status == .analyzing {
            waitingBody(waiting)
        } else {
            workingBody
        }
    }

    private func waitingBody(_ waiting: MatchWaiting) -> some View {
        HStack(spacing: 5) {
            Image(systemName: waiting.symbol).font(.system(size: 9, weight: .black))
            Text(waiting.label).font(.system(size: 12, weight: .bold))
        }
        .foregroundStyle(Color.lilacText)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.paper, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.lilac, lineWidth: 1.5))
        .rotationEffect(.degrees(tilt))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(waiting.label)
    }

    private var workingBody: some View {
        HStack(spacing: 5) {
            Image(systemName: status.symbol)
                .font(.system(size: 9, weight: .black))
                .symbolEffect(.pulse, isActive: status.isWorking && !reduceMotion)
            Text(status.shortLabel)
                .font(.system(size: 12, weight: .bold))
        }
        .foregroundStyle(Color.ink)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(status.ink, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
        .rotationEffect(.degrees(tilt))
        .scaleEffect(breathe ? 1.04 : 1)
        .onAppear { updateBreathing(status) }
        .onChange(of: status) { _, new in updateBreathing(new) }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(status.displayName)
    }

    private func updateBreathing(_ s: MatchStatus) {
        guard !reduceMotion, s.isWorking else {
            var t = Transaction(animation: nil)
            t.disablesAnimations = true
            withTransaction(t) { breathe = false }
            return
        }
        withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) { breathe = true }
    }
}

// MARK: - Preview

#Preview("ShuttlBadge") {
    VStack(alignment: .leading, spacing: AppSpacing.md) {
        ShuttlBadge(status: .recording)
        ShuttlBadge(status: .uploading)
        ShuttlBadge(status: .analyzing)
        ShuttlBadge(status: .analyzing, waiting: .queued)
        ShuttlBadge(status: .analyzing, waiting: .cooling)
        ShuttlBadge(status: .analyzing, waiting: .camera)
        ShuttlBadge(status: .analyzing, waiting: .recording)
        ShuttlBadge(status: .done)
        ShuttlBadge(status: .failed)
    }
    .padding(AppSpacing.lg)
    .paperPage()
}
