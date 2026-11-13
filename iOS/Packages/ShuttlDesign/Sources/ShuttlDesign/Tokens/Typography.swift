import SwiftUI
import CoreText

// MARK: - Type system — "Scorecard Editorial"
// Three voices, deliberately mismatched like a sports broadsheet:
//   • New York (serif, black / heavy, italic for the wordmark and units) — headlines and
//     hero numerals. Editorial, human. Chinese falls back to PingFang heavy inside it.
//   • SF Mono (uppercase, tracked) — labels, units, tab names. The lab-instrument voice.
//   • SF Pro / PingFang — body copy and controls. Quiet and legible.
// Titles ride Dynamic Type text styles; hero numerals are fixed and shrink to fit.

public enum DisplayFace {
    /// Call sites were sized for a condensed face; New York is about twice as wide, so
    /// sizes passed in are drawn at this fraction (28 → 22, 40 → 32, 19 → 15, 21 → 17).
    static let scale: CGFloat = 0.8

    /// Kept so existing launch / renderer code still compiles; the display face is a
    /// system font now, so there is nothing to register.
    public static func register() {}
    public static var isAvailable: Bool { true }

    public static func font(_ size: CGFloat, relativeTo style: Font.TextStyle = .body) -> Font {
        .system(size: (size * scale).rounded(), weight: .black, design: .serif)
    }

    public static func fixed(_ size: CGFloat) -> Font {
        .system(size: (size * scale).rounded(), weight: .black, design: .serif)
    }

    /// Core Text face for off-screen rendering (video export): the heavy system face.
    public static func ctFont(size: CGFloat) -> CTFont {
        CTFontCreateUIFontForLanguage(.emphasizedSystem, size, nil)
            ?? CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
    }
}

public enum AppFont {
    // Hero numerals: speeds, scores. Use with .lineLimit(1).minimumScaleFactor(…).
    public static let hero      = Font.system(size: 108, weight: .black, design: .serif).monospacedDigit()
    public static let display   = Font.system(size: 64, weight: .black, design: .serif).monospacedDigit()
    public static let stat      = Font.system(size: 44, weight: .black, design: .serif).monospacedDigit()
    public static let statSmall = Font.system(size: 30, weight: .heavy, design: .serif).monospacedDigit()

    // Wordmark & screen titles
    public static let brand  = Font.system(size: 56, weight: .black, design: .serif).italic()
    public static let title1 = Font.system(.largeTitle, design: .serif).weight(.black)
    public static let title2 = Font.system(.title2, design: .serif).weight(.heavy)
    public static let title3 = Font.system(.title3, design: .serif).weight(.bold)
    /// "km/h" next to a hero numeral.
    public static let unit   = Font.system(size: 22, weight: .heavy, design: .serif).italic()

    // Body
    public static let headline = Font.system(.headline, design: .default).weight(.semibold)
    public static let body     = Font.system(.body, design: .default)
    public static let callout  = Font.system(.callout, design: .default)

    // Instrument labels — pair with .textCase(.uppercase) and AppFont.labelTracking
    public static let label      = Font.system(.caption, design: .monospaced).weight(.semibold)
    public static let labelSmall = Font.system(.caption2, design: .monospaced).weight(.bold)
    public static let labelTracking: CGFloat = 1.2

    // Legacy names kept for call sites that still use them
    public static let caption      = Font.system(.caption, design: .default).weight(.medium)
    public static let captionSmall = labelSmall

    // Recording / replay timecode
    public static let timer = Font.system(.title3, design: .monospaced).weight(.bold).monospacedDigit()

    /// Serif numeral at a condensed-era size (see `DisplayFace.scale`).
    public static func numeral(_ size: CGFloat, relativeTo style: Font.TextStyle = .title) -> Font {
        DisplayFace.font(size, relativeTo: style).monospacedDigit()
    }
}

// MARK: - Label helper

public extension View {
    /// Uppercase tracked mono label — the "specimen tag" voice.
    func instrumentLabel(_ font: Font = AppFont.label, color: Color = .inkSoft) -> some View {
        self.font(font)
            .textCase(.uppercase)
            .kerning(AppFont.labelTracking)
            .foregroundStyle(color)
    }
}
