import SwiftUI

// MARK: - Sticker
// A spot-ink capsule with an ink outline, stuck on slightly askew.

public struct Sticker: View {
    public var text: String
    public var symbol: String?
    public var fill: Color
    public var foreground: Color
    public var tilt: Double

    public init(_ text: String, symbol: String? = nil, fill: Color = .butter, foreground: Color = .ink, tilt: Double = -3) {
        self.text = text; self.symbol = symbol; self.fill = fill; self.foreground = foreground; self.tilt = tilt
    }

    public var body: some View {
        HStack(spacing: 5) {
            if let symbol {
                Image(systemName: symbol).font(.system(size: 9, weight: .black))
            }
            Text(text).font(.system(size: 12, weight: .bold)).lineLimit(1)
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(fill, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
        .rotationEffect(.degrees(tilt))
    }
}

// MARK: - Seal
// A rubber-stamp rosette: a big display figure over a small caption.

public struct Seal: View {
    public var value: String
    public var caption: String?
    public var symbol: String?
    public var fill: Color
    public var size: CGFloat
    public var tilt: Double
    public var captionOnTop: Bool

    public init(value: String = "", caption: String? = nil, symbol: String? = nil, fill: Color = .butter,
                size: CGFloat = 76, tilt: Double = 10, captionOnTop: Bool = false) {
        self.value = value; self.caption = caption; self.symbol = symbol; self.fill = fill
        self.size = size; self.tilt = tilt; self.captionOnTop = captionOnTop
    }

    public var body: some View {
        let lobes = size > 70 ? 16 : 14
        ZStack {
            Scallop(lobes: lobes, depth: 0.09)
                .fill(fill)
                .overlay(Scallop(lobes: lobes, depth: 0.09).strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            Circle()
                .strokeBorder(Color.ink.opacity(0.35), style: StrokeStyle(lineWidth: 1, dash: [2, 2.5]))
                .padding(size * 0.14)
            VStack(spacing: -1) {
                if captionOnTop, let caption { captionText(caption) }
                if let symbol {
                    Image(systemName: symbol).font(.system(size: size * 0.21, weight: .black))
                }
                if !value.isEmpty {
                    Text(value)
                        .font(DisplayFace.fixed(size * 0.4))
                        .monospacedDigit()
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                }
                if !captionOnTop, let caption { captionText(caption) }
            }
            .foregroundStyle(Color.ink)
            .padding(size * 0.16)
        }
        .frame(width: size, height: size)
        .rotationEffect(.degrees(tilt))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel([caption, value].compactMap { $0 }.joined(separator: " "))
    }

    private func captionText(_ s: String) -> some View {
        Text(s).font(.system(size: max(9, size * 0.14), weight: .heavy)).lineLimit(1).minimumScaleFactor(0.7)
    }
}

// MARK: - InkSegmented
// A segmented control printed like the tab bar: ink track, a spot-ink blob that
// slides to the selection.

public struct InkSegmented<Value: Hashable>: View {
    private let values: [Value]
    @Binding private var selection: Value
    private let ink: Color
    private let onInk: Color
    private let title: (Value) -> String
    @Namespace private var blob

    public init(_ values: [Value], selection: Binding<Value>, ink: Color = .cobalt, onInk: Color = .onCobalt,
                title: @escaping (Value) -> String) {
        self.values = values; self._selection = selection; self.ink = ink; self.onInk = onInk; self.title = title
    }

    public var body: some View {
        HStack(spacing: 2) {
            ForEach(values, id: \.self) { value in
                let selected = value == selection
                Button {
                    withAnimation(AppAnimation.bouncy) { selection = value }
                } label: {
                    Text(title(value))
                        .font(DisplayFace.font(17, relativeTo: .subheadline))
                        .foregroundStyle(selected ? onInk : Color.paper.opacity(0.7))
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                        .frame(maxWidth: .infinity, minHeight: 40)
                        .background {
                            if selected {
                                Capsule().fill(ink).matchedGeometryEffect(id: "blob", in: blob)
                            }
                        }
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(3)
        .background(Color.ink, in: Capsule())
        .sensoryFeedback(.selection, trigger: selection)
    }
}

#Preview("Stickers") {
    struct Demo: View {
        @State var tab = "概览"
        var body: some View {
            VStack(spacing: 20) {
                Sticker("分析中", symbol: "sparkle", fill: .lilac)
                Seal(value: "6", caption: "场")
                Seal(caption: "已分析", symbol: "checkmark", size: 80, tilt: 12)
                InkSegmented(["概览", "落点", "回合", "移动"], selection: $tab) { $0 }
            }
            .padding(24)
            .paperPage()
        }
    }
    return Demo()
}
