import SwiftUI

// MARK: - AppCard
// .sheet  — bright card stock with a hairline rule, the neutral default
// .block  — a flat field of one spot ink, for the single loud thing on a screen
// .print  — card stock, ink outline and an offset ink plate beneath (tappable things)

public struct AppCard<Content: View>: View {
    public var style: Style
    public var padding: CGFloat
    public var radius: CGFloat
    private let content: Content

    public enum Style {
        case sheet
        case block(Color)
        case print(Color)
    }

    public init(style: Style = .sheet, padding: CGFloat = AppSpacing.md, radius: CGFloat = AppRadius.md,
                @ViewBuilder content: () -> Content) {
        self.style = style; self.padding = padding; self.radius = radius; self.content = content()
    }

    public var body: some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        switch style {
        case .sheet:
            content
                .padding(padding)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.card, in: shape)
                .overlay(shape.strokeBorder(Color.rule, lineWidth: AppStroke.hair))
        case .block(let ink):
            content
                .padding(padding)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(ink, in: shape)
                .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        case .print(let plate):
            content
                .padding(padding)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.card, in: shape)
                .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                .printShadow(shape, color: plate, y: 5)
        }
    }
}

// MARK: - PrintButtonStyle
// The pressed state of a printed object: the top layer drops onto its plate.

public struct PrintButtonStyle<S: Shape>: ButtonStyle {
    public var shape: S
    public var plate: Color
    public var depth: CGFloat

    public init(shape: S, plate: Color = .ink, depth: CGFloat = 5) {
        self.shape = shape; self.plate = plate; self.depth = depth
    }

    public func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        configuration.label
            .offset(y: pressed ? depth - 1 : 0)
            .background(shape.fill(plate).offset(y: depth))
            .padding(.bottom, depth)
            .animation(AppAnimation.springFast, value: pressed)
            .sensoryFeedback(.impact(flexibility: .soft, intensity: 0.6), trigger: pressed) { _, new in new }
    }
}

// MARK: - SectionHeader
// Broadsheet section head: tomato folio number, display title, a rule that runs out
// and ends in a small net-post tick.

public struct SectionHeader: View {
    public let title: String
    public var folio: String?
    public var action: (title: String, handler: () -> Void)?

    public init(title: String, folio: String? = nil, action: (title: String, handler: () -> Void)? = nil) {
        self.title = title; self.folio = folio; self.action = action
    }

    public var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: AppSpacing.sm) {
            if let folio {
                Text(folio)
                    .font(AppFont.labelSmall)
                    .foregroundStyle(Color.tomatoText)
                    .fixedSize()
            }
            Text(title)
                .font(AppFont.title3)
                .foregroundStyle(Color.ink)
                .fixedSize()
            HStack(spacing: 0) {
                Rectangle().fill(Color.ink).frame(height: AppStroke.ink)
                Rectangle().fill(Color.ink).frame(width: 2.5, height: 9).offset(y: -3.75)
            }
            .alignmentGuide(.firstTextBaseline) { d in d[.bottom] + 4 }
            if let action {
                Button(action.title, action: action.handler)
                    .instrumentLabel(AppFont.labelSmall, color: .cobalt)
                    .fixedSize()
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

// MARK: - PageHeader
// Masthead for root screens: mono kicker, a big serif title in ink, and a squiggle
// in the section's ink.

public struct PageHeader<Trailing: View>: View {
    public let kicker: String
    public let title: String
    public var ink: Color
    private let trailing: Trailing

    @ScaledMetric(relativeTo: .largeTitle) private var titleSize: CGFloat = 46

    public init(kicker: String, title: String, ink: Color = .tomato, @ViewBuilder trailing: () -> Trailing) {
        self.kicker = kicker; self.title = title; self.ink = ink; self.trailing = trailing()
    }

    public var body: some View {
        HStack(alignment: .bottom) {
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text(kicker)
                    .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                OverprintText(title, font: .system(size: titleSize, weight: .black, design: .serif), plate: ink,
                              offset: CGSize(width: 3.5, height: 3), blend: .multiply)
                    .accessibilityAddTraits(.isHeader)
                Squiggle(waves: 5)
                    .stroke(ink, style: StrokeStyle(lineWidth: 3.5, lineCap: .round))
                    .frame(width: 96, height: 8)
                    .accessibilityHidden(true)
            }
            Spacer(minLength: AppSpacing.md)
            trailing
        }
        .padding(.horizontal, AppSpacing.gutter)
        .padding(.top, AppSpacing.md)
    }
}

public extension PageHeader where Trailing == EmptyView {
    init(kicker: String, title: String, ink: Color = .tomato) {
        self.init(kicker: kicker, title: title, ink: ink) { EmptyView() }
    }
}

// MARK: - Divider

public struct AppDivider: View {
    public init() {}
    public var body: some View {
        Rectangle()
            .fill(Color.rule)
            .frame(height: 1)
    }
}

// MARK: - FlowRow (wrapping HStack)

public struct FlowRow: Layout {
    public var spacing: CGFloat

    public init(spacing: CGFloat = 8) { self.spacing = spacing }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews: subviews)
        return CGSize(width: proposal.width ?? rows.width, height: rows.height)
    }

    public func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = arrange(width: bounds.width, subviews: subviews)
        for (i, origin) in rows.origins.enumerated() {
            subviews[i].place(at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y), proposal: .unspecified)
        }
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> (origins: [CGPoint], width: CGFloat, height: CGFloat) {
        var origins: [CGPoint] = []
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxW: CGFloat = 0
        for sv in subviews {
            let s = sv.sizeThatFits(.unspecified)
            if x > 0 && x + s.width > width { x = 0; y += rowH + spacing; rowH = 0 }
            origins.append(CGPoint(x: x, y: y))
            x += s.width + spacing
            rowH = max(rowH, s.height)
            maxW = max(maxW, x - spacing)
        }
        return (origins, maxW, y + rowH)
    }
}
