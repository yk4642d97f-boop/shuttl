@_exported import ShuttlDesign
import SwiftUI
import UIKit

// The design system lives in the ShuttlDesign package (tokens, surfaces, components,
// overprint / court / shot-glyph signatures). This file re-exports it to the whole
// app target and adds the pieces that depend on app model types or UIKit chrome.

// MARK: - AppTheme
// Global paper-first style. Apply once at the root view.

struct AppTheme: ViewModifier {
    func body(content: Content) -> some View {
        content
            .tint(.cobalt)
            .preferredColorScheme(.light)
            .onAppear(perform: configureNavigationBar)
    }

    private func configureNavigationBar() {
        func display(_ size: CGFloat) -> UIFont {
            let base = UIFont.systemFont(ofSize: size, weight: .black)
            guard let serif = base.fontDescriptor.withDesign(.serif) else { return base }
            return UIFont(descriptor: serif, size: size)
        }
        let ink = UIColor(Color.ink)
        let titleAttrs: [NSAttributedString.Key: Any] = [.foregroundColor: ink, .font: display(20)]
        let largeAttrs: [NSAttributedString.Key: Any] = [.foregroundColor: ink, .font: display(40)]

        let scrolled = UINavigationBarAppearance()
        scrolled.configureWithOpaqueBackground()
        scrolled.backgroundColor = UIColor(Color.paper)
        scrolled.shadowColor = UIColor(Color.rule)
        scrolled.titleTextAttributes = titleAttrs
        scrolled.largeTitleTextAttributes = largeAttrs

        let atTop = UINavigationBarAppearance()
        atTop.configureWithTransparentBackground()
        atTop.titleTextAttributes = titleAttrs
        atTop.largeTitleTextAttributes = largeAttrs

        let bar = UINavigationBar.appearance()
        bar.standardAppearance   = scrolled
        bar.compactAppearance    = scrolled
        bar.scrollEdgeAppearance = atTop
        bar.tintColor = ink
    }
}

extension View {
    func appTheme() -> some View { modifier(AppTheme()) }
}

// MARK: - Shot-type inks and pictograms (app model → design system)

extension ShotType {
    var glyphKind: ShotGlyphKind { ShotGlyphKind(raw: rawValue) }
    var ink: Color { glyphKind.ink }
}
