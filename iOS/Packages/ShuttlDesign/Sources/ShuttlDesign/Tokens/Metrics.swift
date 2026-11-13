import SwiftUI

// MARK: - Spacing (4pt base, 8pt rhythm)

public enum AppSpacing {
    public static let xxs: CGFloat =  2
    public static let xs:  CGFloat =  4
    public static let sm:  CGFloat =  8
    public static let md:  CGFloat = 16
    public static let lg:  CGFloat = 24
    public static let xl:  CGFloat = 32
    public static let xxl: CGFloat = 48
    public static let xxxl: CGFloat = 64

    /// Page gutter for all scrolling screens.
    public static let gutter: CGFloat = 20
}

// MARK: - Corner radii
// Soft, pebble-like corners. Inner radius of a nested element = outer − padding.

public enum AppRadius {
    public static let sm:   CGFloat = 10   // chip, small control
    public static let md:   CGFloat = 20   // card, input
    public static let lg:   CGFloat = 30   // hero block, sheet
    public static let pill: CGFloat = 999
}

// MARK: - Line weights
// One ink weight for every outline, one for hairlines — consistency is what makes
// the loud colour read as intentional rather than messy.

public enum AppStroke {
    public static let ink: CGFloat = 1.5
    public static let fine: CGFloat = 1.1
    public static let hair: CGFloat = 1
}

// MARK: - Print shadow
// A solid, offset copy of the shape in a flat ink — the way a risograph layer
// lands a hair off-register. Reads as tactile without blur-heavy "glass".

public struct PrintShadow<S: Shape>: ViewModifier {
    public var shape: S
    public var color: Color
    public var offset: CGSize

    public init(shape: S, color: Color, offset: CGSize) {
        self.shape = shape; self.color = color; self.offset = offset
    }

    public func body(content: Content) -> some View {
        content.background(
            shape
                .fill(color)
                .offset(offset)
                .accessibilityHidden(true)
        )
    }
}

public extension View {
    func printShadow<S: Shape>(_ shape: S, color: Color = .ink, x: CGFloat = 0, y: CGFloat = 5) -> some View {
        modifier(PrintShadow(shape: shape, color: color, offset: CGSize(width: x, height: y)))
    }

    /// Hairline ink rule around a shape — the default card edge.
    func inkBorder<S: InsettableShape>(_ shape: S, color: Color = .ink, width: CGFloat = AppStroke.ink) -> some View {
        overlay(shape.strokeBorder(color, lineWidth: width))
    }
}

// MARK: - Animation

public enum AppAnimation {
    public static let standard   = Animation.easeOut(duration: 0.2)
    public static let spring     = Animation.spring(response: 0.38, dampingFraction: 0.78)
    public static let springFast = Animation.spring(response: 0.24, dampingFraction: 0.8)
    /// Playful overshoot for selection changes.
    public static let bouncy     = Animation.spring(response: 0.42, dampingFraction: 0.62)
    public static let slow       = Animation.easeInOut(duration: 0.4)
    /// The overprint plate snapping into register.
    public static let register   = Animation.spring(response: 0.55, dampingFraction: 0.58)
}

// MARK: - Hit target enforcement

public extension View {
    /// Ensures a minimum 44×44pt tap target per accessibility guidelines.
    func minTapTarget() -> some View {
        self.frame(minWidth: 44, minHeight: 44)
            .contentShape(Rectangle())
    }
}
