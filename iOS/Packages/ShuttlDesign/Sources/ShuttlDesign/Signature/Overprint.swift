import SwiftUI

// MARK: - Overprint (retired)
// This used to print a second, off-register ink plate under the type. The user found the
// double image ugly, so it now prints one layer in `ink`. The API stays so call sites
// compile; `plate`, `offset`, `blend` and `registers` are ignored.

public enum OverprintBlend: Sendable { case normal, multiply }

public struct Overprint<Content: View>: View {
    private let ink: Color
    private let content: Content

    public init(ink: Color = .ink, plate: Color, offset: CGSize = CGSize(width: 3, height: 3),
                blend: OverprintBlend = .normal, registers: Bool = true, @ViewBuilder content: () -> Content) {
        self.ink = ink
        self.content = content()
    }

    public var body: some View {
        content.foregroundStyle(ink)
    }
}

// MARK: - OverprintText
// One string in one font and one ink.

public struct OverprintText: View {
    private let text: String
    private let font: Font
    private let ink: Color
    private let minimumScale: CGFloat

    public init(_ text: String, font: Font, ink: Color = .ink, plate: Color,
                offset: CGSize = CGSize(width: 3, height: 3), blend: OverprintBlend = .normal,
                registers: Bool = true, minimumScale: CGFloat = 0.5) {
        self.text = text; self.font = font; self.ink = ink; self.minimumScale = minimumScale
    }

    public var body: some View {
        Text(text)
            .font(font)
            .foregroundStyle(ink)
            .lineLimit(1)
            .minimumScaleFactor(minimumScale)
            .contentTransition(.numericText())
            // Display type is the point of the element: it claims its width before siblings
            // (units, stickers) and only shrinks when the row genuinely runs out of room.
            .layoutPriority(1)
            .accessibilityLabel(text)
    }
}

#Preview("Overprint") {
    VStack(alignment: .leading, spacing: 18) {
        OverprintText("比赛", font: AppFont.numeral(64, relativeTo: .largeTitle), plate: .cobalt)
        OverprintText("Shuttl.", font: AppFont.brand, ink: .cobalt, plate: .tomato,
                      offset: CGSize(width: 4, height: 3), blend: .multiply)
        OverprintText("341", font: AppFont.hero, plate: .butter, offset: CGSize(width: 5, height: 4))
            .padding(12)
            .background(Color.tomato)
    }
    .padding(24)
    .paperPage()
    .onAppear { DisplayFace.register() }
}
