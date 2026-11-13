import CoreGraphics
import CoreText
import Foundation
import ShuttlCore
import ShuttlDesign
import SwiftUI

/// Core Graphics view of the ShuttlDesign tokens for the video renderer.
///
/// Colours are not copied: each one is resolved from the SwiftUI token in
/// `ShuttlDesign/Tokens/Colors.swift` (`Color.resolve(in:)`, iOS 17 / macOS 14) and
/// re-encoded in sRGB, so the burnt-in overlay prints exactly the hex values the app
/// shows. The renderer draws into an sRGB context, so no colour conversion happens.
/// Type is the bundled display face (Smiley Sans Oblique 得意黑) via `DisplayFace`.
enum ExportTheme {
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    static let paper = cg(.paper)
    static let card = cg(.card)
    static let ink = cg(.ink)
    static let inkSoft = cg(.inkSoft)
    static let tomato = cg(.tomato)
    static let cobalt = cg(.cobalt)
    static let pistachio = cg(.pistachio)
    static let lilac = cg(.lilac)
    static let butter = cg(.butter)
    /// Riso fluorescent pink. Read through `ShotGlyphKind.serve.ink` because in a client
    /// module that also imports SwiftUI, `Color.pink` resolves to Apple's system pink
    /// (#FF2D55) rather than ShuttlDesign's token (#FF7AB6).
    static let pink = cg(ShotGlyphKind.serve.ink)
    static let night = cg(.night)

    /// A design token as an sRGB CGColor.
    static func cg(_ color: Color) -> CGColor {
        let resolved = color.resolve(in: EnvironmentValues()).cgColor
        return resolved.converted(to: sRGB, intent: .relativeColorimetric, options: nil) ?? resolved
    }

    /// The ink a shot type is charted in across the app (`ShotGlyphKind.ink`).
    static func shotColor(_ shot: String) -> CGColor {
        let kind = ShotGlyphKind(raw: shot)
        return kind == .unknown ? card : cg(kind.ink)
    }

    /// Readable type colour on a spot-ink fill (card on cobalt, ink on the light inks).
    static func onShotColor(_ shot: String) -> CGColor {
        switch ShotGlyphKind(raw: shot) {
        case .clear, .lift: return card
        default: return ink
        }
    }

    /// Chinese shot name shared with the rest of the app (`ShotGlyphKind.chineseName`).
    static func shotName(_ type: ShotType) -> String? {
        let kind = ShotGlyphKind(raw: type.rawValue)
        return kind == .unknown ? nil : kind.chineseName
    }

    static func displayFont(_ size: CGFloat) -> CTFont { DisplayFace.ctFont(size: size) }

    /// Mono label voice (`AppFont.label`), for the small tags on the score card.
    static func monoFont(_ size: CGFloat) -> CTFont {
        CTFontCreateUIFontForLanguage(.userFixedPitch, size, nil)
            ?? CTFontCreateWithName("Menlo-Bold" as CFString, size, nil)
    }
}

extension CGColor {
    /// The same colour at a new alpha, kept in its own colour space.
    func with(alpha: CGFloat) -> CGColor { copy(alpha: alpha) ?? self }
}
