import SwiftUI

// MARK: - InkProgressBar
// Determinate progress as a printed capsule: a spot-ink fill on a paper track,
// both under one ink outline (same construction as ShotMixBar).

public struct InkProgressBar: View {
    /// 0…1
    public let value: Double
    public var fill: Color
    public var track: Color
    public var height: CGFloat

    public init(value: Double, fill: Color = .ink, track: Color = .paperDeep, height: CGFloat = 10) {
        self.value = value; self.fill = fill; self.track = track; self.height = height
    }

    public var body: some View {
        let clamped = min(max(value, 0), 1)
        GeometryReader { geo in
            Capsule()
                .fill(fill)
                .frame(width: max(height, geo.size.width * clamped))
                .opacity(clamped > 0 ? 1 : 0)
        }
        .frame(height: height)
        .background(track)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
        .animation(AppAnimation.standard, value: clamped)
        .accessibilityElement(children: .ignore)
        .accessibilityValue("\(Int((clamped * 100).rounded()))%")
    }
}

#Preview("InkProgressBar") {
    VStack(spacing: AppSpacing.md) {
        InkProgressBar(value: 0)
        InkProgressBar(value: 0.42, fill: .butter)
        InkProgressBar(value: 0.9, fill: .ink, track: .card, height: 12)
    }
    .padding(24)
    .paperPage()
}
