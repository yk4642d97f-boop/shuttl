import SwiftUI
import ShuttlCore
import ShuttlDesign

// Results-specific pieces built on the shared ShuttlDesign system. Tokens, cards,
// headers and pictograms come from ShuttlDesign; nothing here redefines them.

extension ShotType {
    var glyphKind: ShotGlyphKind { ShotGlyphKind(raw: rawValue) }
    var ink: Color { glyphKind.ink }
}

extension CourtSide {
    /// Near is cobalt, far is pink — the same pairing as the export score bug.
    var ink: Color {
        switch self {
        case .near: return .cobalt
        case .far: return .flamingo
        case .unknown: return .paperDeep
        }
    }
    var onInk: Color { self == .near ? .onCobalt : .ink }
}

extension View {
    func resultPage() -> some View { paperPage() }
}

struct ResultsScrollView<Content: View>: View {
    @ViewBuilder let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        #if os(iOS)
        ScrollView { content }
        #else
        content
        #endif
    }
}

/// A stat in the results voice: display numeral, mono unit and caption,
/// optionally printed on a spot-ink tile.
struct ResultsStat: View {
    let value: String
    let label: String
    let unit: String?
    var fill: Color?

    var body: some View {
        StatBubble(value: value, label: label, unit: unit, fill: fill, numeralFont: AppFont.numeral(30, relativeTo: .title2))
    }
}

struct EmptyResultsView: View {
    let title: String
    let message: String

    var body: some View {
        AppCard(style: .print(.lilac), padding: AppSpacing.lg) {
            HStack(alignment: .top, spacing: AppSpacing.md) {
                ShotGlyph(.unknown, color: .ink)
                    .frame(width: 44, height: 44)
                    .padding(6)
                    .background(Color.lilac, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                    .rotationEffect(.degrees(-4))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    Text(title).font(AppFont.title2).foregroundStyle(Color.ink)
                    Text(message)
                        .font(AppFont.callout)
                        .foregroundStyle(Color.inkSoft)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// A notice on paper: a 4 pt rule in the notice's ink on the leading edge and an ink icon.
/// Butter marks calibration, lilac low confidence. It stays quiet next to the hero, so a
/// page still has one loud block.
struct ResultNoticeView: View {
    let notice: ResultsNotice

    var body: some View {
        let rule: Color = notice.kind == .calibration || notice.kind == .caution ? .butter : .lilac
        let shape = RoundedRectangle(cornerRadius: AppRadius.sm + 2, style: .continuous)
        HStack(alignment: .top, spacing: AppSpacing.sm) {
            Image(systemName: Self.symbol(notice.kind))
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(Color.ink)
                .frame(width: 24)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 3) {
                Text(notice.title).font(AppFont.headline).foregroundStyle(Color.ink)
                if notice.lines.isEmpty {
                    Text(notice.message)
                        .font(AppFont.callout)
                        .foregroundStyle(Color.inkSoft)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    ForEach(notice.lines, id: \.self) { line in
                        HStack(alignment: .firstTextBaseline, spacing: 7) {
                            Circle().fill(Color.butter)
                                .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1))
                                .frame(width: 8, height: 8)
                                .alignmentGuide(.firstTextBaseline) { $0[VerticalAlignment.center] + 4 }
                            Text(line)
                                .font(AppFont.callout)
                                .foregroundStyle(Color.inkSoft)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.top, 2)
                    }
                }
            }
        }
        .padding(AppSpacing.md)
        .padding(.leading, 4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.card, in: shape)
        .overlay(alignment: .leading) {
            Rectangle().fill(rule).frame(width: 4)
                .overlay(alignment: .trailing) { Rectangle().fill(Color.ink).frame(width: AppStroke.hair) }
        }
        .clipShape(shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.fine))
        .accessibilityElement(children: .combine)
    }
}

extension ResultNoticeView {
    static func symbol(_ kind: ResultsNotice.Kind) -> String {
        switch kind {
        case .calibration: return "scope"
        case .caution: return "eye"
        case .confidence, .empty: return "exclamationmark.triangle"
        }
    }
}

/// Filter chip with a shot pictogram (or text only).
struct ResultsFilterChip: View {
    let title: String
    var kind: ShotGlyphKind?
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                if let kind { ShotChip(kind, size: 20) }
                Text(title)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(selected ? Color.onCobalt : Color.ink)
            }
            .padding(.leading, kind == nil ? 12 : 5)
            .padding(.trailing, 12)
            .frame(minHeight: 36)
            .background(selected ? Color.cobalt : Color.card, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.ink, lineWidth: AppStroke.fine))
        }
        .buttonStyle(ScaleButtonStyle())
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
