import SwiftUI

// MARK: - "Riso Court" colour system
// Paper-and-ink palette modelled on risograph spot inks: every hue is a flat,
// unmixed ink laid on warm uncoated stock. Values are chosen in OKLCH so the
// loud inks sit on a few shared lightness steps, which is what lets an
// "odd" pairing like tomato + lilac + pistachio feel calm instead of clashing.
//
// Proportion (Itten, contrast of extension): ~70% paper, ~20% ink, ~10% colour.
// Colour is spent on one thing per screen; tomato is the deliberate
// "wrong" accent and is only ever a fill, never small text on paper.
//
// Overprint: two inks laid over each other multiply, the way riso drums do —
// cobalt × tomato ≈ ink, cobalt × butter ≈ deep green, tomato × butter ≈ vermilion.
// `OverprintText` uses that to make a third colour out of two (Albers).
//
// Contrast (WCAG 2.x, measured):
//   ink on paper 14.4 · inkSoft on paper 6.4 · inkFaint on paper 3.1 (≥18pt only)
//   ink on tomato 4.8 · card on cobalt 6.4 · ink on pistachio 10.2 · ink on lilac 9.0
//   paper on night 16.0 · pistachio on night 11.4 · tomato on night 5.4

public extension Color {
    // ── Stock ───────────────────────────────────────────────────────────────
    /// oklch(95% 0.016 83) — warm uncoated paper, the page
    static let paper     = Color(hex: 0xF4EEE3)
    /// oklch(91% 0.024 82) — a second, deeper sheet for wells and tracks
    static let paperDeep = Color(hex: 0xEAE1D1)
    /// oklch(99% 0.010 82) — bright card stock laid on the page
    static let card      = Color(hex: 0xFFFBF4)

    // ── Ink (aubergine black — never pure #000) ─────────────────────────────
    /// oklch(24% 0.037 301)
    static let ink      = Color(hex: 0x221B2E)
    /// oklch(46% 0.037 304) — secondary text, AA on paper
    static let inkSoft  = Color(hex: 0x5B5268)
    /// oklch(63% 0.031 306) — large or decorative text only
    static let inkFaint = Color(hex: 0x8C8397)
    /// Hairline rules
    static let rule     = Color(hex: 0x221B2E, opacity: 0.14)

    // ── Spot inks ───────────────────────────────────────────────────────────
    /// oklch(66% 0.20 35) — the unexpected red: record, hero numbers, alarms
    static let tomato    = Color(hex: 0xF2542D)
    /// oklch(50% 0.22 268) — primary action, links, "matches"
    static let cobalt    = Color(hex: 0x2F4BDE)
    /// oklch(83% 0.14 122) — success, "statistics", court surface
    static let pistachio = Color(hex: 0xB9D66A)
    /// oklch(81% 0.085 298) — soft secondary fills, analysis in progress
    static let lilac     = Color(hex: 0xC7B6F2)
    /// oklch(89% 0.15 96) — highlights, uploads, stickers
    static let butter    = Color(hex: 0xF7D95C)
    /// oklch(75% 0.17 354) — riso fluorescent pink, used sparingly
    static let flamingo      = Color(hex: 0xFF7AB6)

    // ── Ink-safe text versions of light spot inks (AA on paper) ─────────────
    static let tomatoText    = Color(hex: 0xBE3417)
    static let pistachioText = Color(hex: 0x4F6A14)
    static let lilacText     = Color(hex: 0x6C4FC4)

    // ── Foregrounds on each spot ink ────────────────────────────────────────
    static let onTomato    = Color.ink
    static let onCobalt    = Color.card
    static let onPistachio = Color.ink
    static let onLilac     = Color.ink
    static let onButter    = Color.ink
    static let onFlamingo      = Color.ink

    // ── Night (camera & replay, where video needs a dark surround) ──────────
    /// oklch(19% 0.02 298)
    static let night       = Color(hex: 0x15121C)
    static let nightRaised = Color(hex: 0x221D2B)
    /// Court felt seen at night — the calibration and replay surround.
    static let courtNight  = Color(hex: 0x1F4A3A)

    // ── Semantic ─────────────────────────────────────────────────────────────
    static let success = Color.pistachio
    static let warning = Color.butter
    static let danger  = Color.tomato
    static let info    = Color.cobalt
}

// MARK: - Hex init

public extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(
            .sRGB,
            red:   Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >>  8) & 0xFF) / 255,
            blue:  Double( hex        & 0xFF) / 255,
            opacity: opacity
        )
    }
}

// MARK: - ShapeStyle leading-dot access
// Lets call sites write `.foregroundStyle(.ink)` / `.fill(.paper)`.

public extension ShapeStyle where Self == Color {
    static var paper: Color { Color.paper }
    static var paperDeep: Color { Color.paperDeep }
    static var card: Color { Color.card }
    static var ink: Color { Color.ink }
    static var inkSoft: Color { Color.inkSoft }
    static var inkFaint: Color { Color.inkFaint }
    static var rule: Color { Color.rule }
    static var tomato: Color { Color.tomato }
    static var cobalt: Color { Color.cobalt }
    static var pistachio: Color { Color.pistachio }
    static var lilac: Color { Color.lilac }
    static var butter: Color { Color.butter }
    static var flamingo: Color { Color.flamingo }
    static var night: Color { Color.night }
}

// MARK: - Trajectory gradient
// Trail runs cool → hot: lilac tail, tomato body, butter head.

public extension Gradient {
    static let trajectory = Gradient(colors: [.lilac.opacity(0), .lilac, .tomato, .butter])
}
