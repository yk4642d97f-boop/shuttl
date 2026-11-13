import SwiftUI
import CoreGraphics

// MARK: - Paper grain
// A 128pt tile of sparse ink specks, rendered once and tiled. Gives flat fills the
// slightly toothy feel of uncoated / risograph stock without per-frame noise cost.

public enum PaperTexture {
    public static let grain: Image = {
        let scale: CGFloat = 3
        let side = 128
        let px = Int(CGFloat(side) * scale)
        guard let ctx = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return Image(systemName: "circle").resizable()
        }
        ctx.scaleBy(x: scale, y: scale)
        var state: UInt64 = 0x5EED_CAFE
        func rand() -> CGFloat {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return CGFloat(state >> 33) / CGFloat(UInt32.max >> 1)
        }
        for _ in 0..<700 {
            let x = rand() * CGFloat(side), y = rand() * CGFloat(side)
            let r = 0.4 + rand() * 0.7
            ctx.setFillColor(CGColor(gray: 0.1, alpha: 0.04 + rand() * 0.09))
            ctx.fillEllipse(in: CGRect(x: x, y: y, width: r, height: r))
        }
        guard let cg = ctx.makeImage() else { return Image(systemName: "circle").resizable() }
        return Image(decorative: cg, scale: scale).resizable(resizingMode: .tile)
    }()
}

public extension View {
    /// Overlays the tiled paper grain. Keep opacity low; it should be felt, not seen.
    func paperGrain(opacity: Double = 1) -> some View {
        overlay(
            PaperTexture.grain
                .opacity(opacity)
                .blendMode(.multiply)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        )
    }

    /// Standard page: paper stock with grain, edge to edge.
    func paperPage() -> some View {
        background {
            Color.paper
                .paperGrain()
                .ignoresSafeArea()
        }
    }
}

// MARK: - ShimmerModifier
// Sweeps a soft highlight across a placeholder — skeleton loading states.

public struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = -1.0
    public var color: Color = .card

    public init(color: Color = .card) { self.color = color }

    public func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geo in
                    let w = geo.size.width
                    LinearGradient(
                        stops: [
                            .init(color: .clear,              location: 0.0),
                            .init(color: color.opacity(0.55), location: 0.45),
                            .init(color: color.opacity(0.85), location: 0.5),
                            .init(color: color.opacity(0.55), location: 0.55),
                            .init(color: .clear,              location: 1.0),
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(width: w * 2)
                    .offset(x: phase * w * 2)
                }
            )
            .clipped()
            .onAppear {
                withAnimation(.linear(duration: 1.6).repeatForever(autoreverses: false)) {
                    phase = 1.0
                }
            }
    }
}

// MARK: - PulseModifier
// Repeating scale, used for the live recording dot and "analysing" stickers.

public struct PulseModifier: ViewModifier {
    @State private var scale: CGFloat = 1.0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public var maxScale: CGFloat = 1.06
    public var duration: Double  = 0.9

    public init(maxScale: CGFloat = 1.06, duration: Double = 0.9) {
        self.maxScale = maxScale; self.duration = duration
    }

    public func body(content: Content) -> some View {
        content
            .scaleEffect(scale)
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.easeInOut(duration: duration).repeatForever(autoreverses: true)) {
                    scale = maxScale
                }
            }
    }
}

// MARK: - Slow spin (for "analysing" seals)

public struct SlowSpin: ViewModifier {
    @State private var angle: Double = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    public var period: Double = 12

    public init(period: Double = 12) { self.period = period }

    public func body(content: Content) -> some View {
        content
            .rotationEffect(.degrees(angle))
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.linear(duration: period).repeatForever(autoreverses: false)) { angle = 360 }
            }
    }
}

public extension View {
    func shimmer(color: Color = .card) -> some View { modifier(ShimmerModifier(color: color)) }
    func pulse(maxScale: CGFloat = 1.06, duration: Double = 0.9) -> some View {
        modifier(PulseModifier(maxScale: maxScale, duration: duration))
    }
    func slowSpin(period: Double = 12) -> some View { modifier(SlowSpin(period: period)) }
}
