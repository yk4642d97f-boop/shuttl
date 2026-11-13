import SwiftUI

// MARK: - AppIconView
// Bauhaus-poster mark: cobalt field, a butter "sun" rising off-centre, a lilac
// quarter-disc in the corner, and the shuttlecock mid-flight with three speed
// strokes behind it. Built from primitives so it stays crisp at every size.

struct AppIconView: View {
    var size: CGFloat = 1024
    /// Draw the ink outline around the tile (off for the real icon export).
    var outlined: Bool = true

    var body: some View {
        let u = size / 100
        ZStack {
            Color.cobalt

            Circle()
                .fill(Color.butter)
                .frame(width: 62 * u, height: 62 * u)
                .offset(x: 20 * u, y: -18 * u)

            Circle()
                .fill(Color.lilac)
                .frame(width: 56 * u, height: 56 * u)
                .offset(x: -50 * u, y: 50 * u)

            speedStrokes(u)
                .offset(x: -18 * u, y: 16 * u)

            ShuttlecockMark(skirt: .card, cork: .tomato, ink: .ink)
                .frame(width: 62 * u, height: 62 * u)
                .rotationEffect(.degrees(-128))
                .offset(x: 6 * u, y: -4 * u)
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.2237, style: .continuous))
        .overlay {
            if outlined {
                RoundedRectangle(cornerRadius: size * 0.2237, style: .continuous)
                    .strokeBorder(Color.ink, lineWidth: max(1.5, size * 0.02))
            }
        }
        .accessibilityHidden(true)
    }

    private func speedStrokes(_ u: CGFloat) -> some View {
        VStack(alignment: .trailing, spacing: 5 * u) {
            Capsule().fill(Color.paper).frame(width: 18 * u, height: 3.2 * u)
            Capsule().fill(Color.paper).frame(width: 26 * u, height: 3.2 * u)
            Capsule().fill(Color.paper).frame(width: 12 * u, height: 3.2 * u)
        }
        .rotationEffect(.degrees(-38))
    }
}

// MARK: - Preview

#Preview("App Icon — multiple sizes") {
    VStack(spacing: 32) {
        AppIconView(size: 180)
        HStack(spacing: 24) {
            AppIconView(size: 60)
            AppIconView(size: 88)
            AppIconView(size: 120)
        }
    }
    .padding(40)
    .paperPage()
}
