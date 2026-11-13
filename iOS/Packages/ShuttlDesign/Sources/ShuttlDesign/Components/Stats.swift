import SwiftUI

// MARK: - StatBubble
// One scoreboard figure: display-face numeral, mono unit, mono caption.
// Pass `fill` to print it on a block of spot ink (the hero figure).

public struct StatBubble: View {
    public let value: String
    public let label: String
    public var unit: String?
    public var fill: Color?
    public var numeralFont: Font

    public init(value: String, label: String, unit: String? = nil, fill: Color? = nil,
                numeralFont: Font = AppFont.statSmall) {
        self.value = value; self.label = label; self.unit = unit; self.fill = fill; self.numeralFont = numeralFont
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(numeralFont)
                    .foregroundStyle(Color.ink)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .contentTransition(.numericText())
                if let unit {
                    Text(unit)
                        .instrumentLabel(AppFont.labelSmall, color: fill == nil ? .inkSoft : .ink)
                        .fixedSize()
                }
            }
            Text(label)
                .instrumentLabel(AppFont.labelSmall, color: fill == nil ? .inkSoft : .ink)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, fill == nil ? 0 : AppSpacing.sm + 4)
        .padding(.vertical, fill == nil ? 0 : AppSpacing.sm + 2)
        .background {
            if let fill {
                RoundedRectangle(cornerRadius: AppRadius.sm + 4, style: .continuous)
                    .fill(fill)
                    .overlay(RoundedRectangle(cornerRadius: AppRadius.sm + 4, style: .continuous)
                        .strokeBorder(Color.ink, lineWidth: 1.5))
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label) \(value)\(unit.map { " \($0)" } ?? "")")
    }
}

// MARK: - StatRow
// A scoreboard strip: figures separated by hairline ink rules.

public struct StatRow: View {
    public struct Stat {
        public let value: String
        public let label: String
        public let unit: String?
        public let fill: Color?
        public init(_ value: String, _ label: String, unit: String? = nil, hero: Bool = false, ink: Color = .tomato) {
            self.value = value; self.label = label; self.unit = unit
            self.fill = hero ? ink : nil
        }
    }

    public let stats: [Stat]
    /// One size for every figure, so the row never scales numerals unevenly.
    public var numeralFont: Font

    public init(stats: [Stat], numeralFont: Font = AppFont.statSmall) {
        self.stats = stats; self.numeralFont = numeralFont
    }

    public var body: some View {
        HStack(alignment: .center, spacing: AppSpacing.md) {
            ForEach(stats.indices, id: \.self) { i in
                StatBubble(value: stats[i].value, label: stats[i].label,
                           unit: stats[i].unit, fill: stats[i].fill, numeralFont: numeralFont)
                if i < stats.count - 1 && stats[i].fill == nil && stats[i + 1].fill == nil {
                    Rectangle().fill(Color.rule).frame(width: 1, height: 36)
                }
            }
        }
    }
}

#Preview("Stats") {
    VStack(spacing: 24) {
        StatRow(stats: [
            .init("42:18", "时长"),
            .init("14", "回合"),
            .init("341", "最高球速", unit: "km/h", hero: true),
        ])
        StatBubble(value: "352", label: "最高球速", unit: "km/h", fill: .butter, numeralFont: AppFont.display)
    }
    .padding(24)
    .paperPage()
}
