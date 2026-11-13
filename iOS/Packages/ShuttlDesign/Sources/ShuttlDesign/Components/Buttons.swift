import SwiftUI

// MARK: - PillButton
// .primary   — cobalt ink, card-white text, printed on an ink plate that it presses into
// .secondary — card stock with an ink outline, same press behaviour
// .ghost     — text only, for tertiary actions
// `tint` swaps the primary ink (e.g. tomato for destructive).

public struct PillButton: View {
    public let title: String
    public var icon: String?
    public var isLoading: Bool
    public var isDestructive: Bool
    public var style: Style
    public var tint: Color?
    /// Text colour on `tint`; defaults to ink, which suits the light spot inks.
    public var tintText: Color
    public let action: () -> Void

    public enum Style { case primary, secondary, ghost }

    public init(title: String, icon: String? = nil, isLoading: Bool = false, isDestructive: Bool = false,
                style: Style = .primary, tint: Color? = nil, tintText: Color = .ink, action: @escaping () -> Void) {
        self.title = title; self.icon = icon; self.isLoading = isLoading; self.isDestructive = isDestructive
        self.style = style; self.tint = tint; self.tintText = tintText; self.action = action
    }

    public var body: some View {
        switch style {
        case .ghost:
            Button(action: action) { label.frame(minHeight: 44) }
                .buttonStyle(ScaleButtonStyle())
                .disabled(isLoading)
                .accessibilityLabel(isLoading ? "加载中" : title)
        case .primary, .secondary:
            Button(action: action) {
                label
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .background(fill, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.5))
            }
            .buttonStyle(PrintButtonStyle(shape: Capsule(), plate: .ink, depth: 4))
            .disabled(isLoading)
            .accessibilityLabel(isLoading ? "加载中" : title)
        }
    }

    private var label: some View {
        HStack(spacing: AppSpacing.sm) {
            if isLoading {
                ProgressView()
                    .progressViewStyle(.circular)
                    .tint(foreground)
            } else {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 16, weight: .bold))
                }
                Text(title)
                    .font(DisplayFace.font(21, relativeTo: .headline))
            }
        }
        .foregroundStyle(foreground)
    }

    private var ink: Color {
        if let tint { return tint }
        return isDestructive ? .tomato : .cobalt
    }

    private var fill: Color {
        style == .primary ? ink : .card
    }

    private var foreground: Color {
        switch style {
        case .primary:
            if tint != nil { return tintText }
            return isDestructive ? .onTomato : .onCobalt
        case .secondary: return .ink
        case .ghost:     return isDestructive ? .tomatoText : .cobalt
        }
    }
}

// MARK: - ScaleButtonStyle
// Subtle press-down scale, no grey flash.

public struct ScaleButtonStyle: ButtonStyle {
    public init() {}
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.94 : 1.0)
            .animation(AppAnimation.springFast, value: configuration.isPressed)
    }
}

// MARK: - IconButton
// Round icon control.
// .ghost  — ink outline on paper      .filled — card stock with outline
// .accent — cobalt ink                .night  — translucent over video

public struct IconButton: View {
    public let systemName: String
    public var size: CGFloat
    public var iconSize: CGFloat
    public var style: Style
    public var label: String?
    public let action: () -> Void

    public enum Style { case ghost, filled, accent, night }

    public init(systemName: String, size: CGFloat = 44, iconSize: CGFloat = 18, style: Style = .ghost,
                label: String? = nil, action: @escaping () -> Void) {
        self.systemName = systemName; self.size = size; self.iconSize = iconSize
        self.style = style; self.label = label; self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: iconSize, weight: .bold))
                .foregroundStyle(iconColor)
                .frame(width: size, height: size)
                .background(bgColor, in: Circle())
                .overlay(Circle().strokeBorder(borderColor, lineWidth: 1.5))
        }
        .buttonStyle(ScaleButtonStyle())
        .accessibilityLabel(label ?? systemName)
    }

    private var iconColor: Color {
        switch style {
        case .ghost, .filled: return .ink
        case .accent:         return .onCobalt
        case .night:          return .paper
        }
    }

    private var bgColor: Color {
        switch style {
        case .ghost:  return .clear
        case .filled: return .card
        case .accent: return .cobalt
        case .night:  return Color.night.opacity(0.55)
        }
    }

    private var borderColor: Color {
        switch style {
        case .ghost, .filled, .accent: return .ink
        case .night:                   return Color.paper.opacity(0.22)
        }
    }
}

#Preview("Buttons") {
    VStack(spacing: 20) {
        PillButton(title: "开始录制", icon: "record.circle", style: .primary) {}
        PillButton(title: "上传中…", isLoading: true, style: .primary) {}
        PillButton(title: "查看回放", icon: "play.fill", style: .secondary) {}
        PillButton(title: "删除比赛", isDestructive: true, style: .primary) {}
        PillButton(title: "忘记密码", style: .ghost) {}
        HStack {
            IconButton(systemName: "xmark", style: .ghost) {}
            IconButton(systemName: "play.fill", style: .accent) {}
            IconButton(systemName: "arrow.clockwise", style: .filled) {}
        }
    }
    .padding(24)
    .paperPage()
}
