import CoreGraphics
import ShuttlCore
import ShuttlDesign
import SwiftUI

/// The night-surround stage: the aspect-fitted frame (or a drawn stand-in), the
/// projected court, four corner handles, the loupe and the status chips. Zoom and pan
/// are applied about the stage centre, and every overlay maps through the same
/// transform, so handles stay glued to the picture while pinching.
struct CalibrationStage: View {
    let image: CGImage?
    let frameSize: ImageSize
    let taps: CourtCornerTaps
    let selectedName: String?
    let selectCorner: (String) -> Void
    @Binding var zoom: CGFloat
    @Binding var pan: CGSize
    @Binding var activeDragName: String?
    let moveCorner: (String, CGPoint) -> Void
    let beginCornerDrag: (String) -> Void
    let endCornerDrag: () -> Void
    let previewState: CourtCalibrationPreviewState

    @State private var dragStartPan: CGSize = .zero
    @State private var pinchStartZoom: CGFloat = 1
    @State private var isPanning = false
    @State private var isPinching = false

    var body: some View {
        GeometryReader { geometry in
            let stage = geometry.size
            let rect = fittedRect(in: stage)
            let issue = CourtQuadValidator.issue(for: taps)
            let loupeName = loupeCorner
            let cornerPoints = taps.cornerSet.names.compactMap { name in
                taps[name].map { screenPoint($0, rect: rect, stage: stage) }
            }
            ZStack {
                Color.night

                FrameContent(image: image, frameSize: frameSize)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                    .scaleEffect(zoom)
                    .offset(pan)
                    .gesture(panGesture)
                    .accessibilityHidden(true)

                CourtLinesOverlay(taps: taps, issue: issue, transform: { point in
                    screenPoint(CGPoint(x: point.x / CGFloat(taps.imageSize.width),
                                        y: point.y / CGFloat(taps.imageSize.height)), rect: rect, stage: stage)
                }, corners: cornerPoints)
                .allowsHitTesting(false)

                ForEach(taps.cornerSet.names, id: \.self) { name in
                    if let point = taps[name] {
                        let position = screenPoint(point, rect: rect, stage: stage)
                        CornerHandle(
                            name: name,
                            selected: selectedName == name,
                            showsLabel: loupeName != name,
                            labelPlacement: labelPlacement(for: name, at: position, corners: cornerPoints, stage: stage),
                            position: position,
                            onSelect: { selectCorner(name) },
                            onDragStart: { selectCorner(name); beginCornerDrag(name) },
                            onDrag: { location in
                                moveCorner(name, normalizedPoint(from: location, rect: rect, stage: stage))
                            },
                            onDragEnd: endCornerDrag,
                            onAdjust: { direction in adjust(name, direction: direction) }
                        )
                        .zIndex(selectedName == name ? 2 : 1)
                    }
                }

                if let name = loupeName, let point = taps[name] {
                    LoupeView(image: image, frameSize: frameSize, cornerName: name,
                              normalizedPoint: point, rect: rect, zoom: zoom)
                        .position(loupePosition(for: screenPoint(point, rect: rect, stage: stage),
                                                frame: visibleFrame(rect, stage: stage), in: stage))
                        .zIndex(5)
                        .allowsHitTesting(false)
                }

                if loupeName == nil {
                    StageHint(zoom: zoom, isPlaceholder: image == nil)
                        .padding(AppSpacing.sm + 2)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .allowsHitTesting(false)
                        .zIndex(3)
                }

                VStack {
                    Spacer(minLength: 0)
                    if let issue {
                        QuadIssueChip(issue: issue)
                            .transition(.scale(scale: 0.9).combined(with: .opacity))
                    }
                }
                .padding(.horizontal, AppSpacing.md)
                .padding(.bottom, AppSpacing.md)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .animation(AppAnimation.springFast, value: issue)
                .allowsHitTesting(false)
                .zIndex(4)
            }
            .coordinateSpace(name: "courtImageStage")
            .contentShape(Rectangle())
            .simultaneousGesture(magnifyGesture)
        }
    }

    // MARK: Gestures

    private var panGesture: some Gesture {
        DragGesture(minimumDistance: 2, coordinateSpace: .named("courtImageStage"))
            .onChanged { value in
                if !isPanning {
                    dragStartPan = pan
                    isPanning = true
                }
                pan = CGSize(width: dragStartPan.width + value.translation.width,
                             height: dragStartPan.height + value.translation.height)
            }
            .onEnded { _ in
                dragStartPan = .zero
                isPanning = false
            }
    }

    private var magnifyGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                if !isPinching {
                    pinchStartZoom = zoom
                    isPinching = true
                }
                zoom = min(max(pinchStartZoom * value.magnification, 1), 6)
            }
            .onEnded { _ in
                pinchStartZoom = 1
                isPinching = false
            }
    }

    // MARK: Geometry

    /// The corner whose loupe is showing: the one being dragged, or the preview's pick.
    private var loupeCorner: String? {
        guard let selectedName else { return nil }
        if activeDragName != nil { return selectedName }
        if case .selectedWithLoupe = previewState { return selectedName }
        return nil
    }

    private func fittedRect(in size: CGSize) -> CGRect {
        let w = CGFloat(max(frameSize.width, 1)), h = CGFloat(max(frameSize.height, 1))
        let scale = min(size.width / w, size.height / h)
        let width = w * scale, height = h * scale
        return CGRect(x: (size.width - width) / 2, y: (size.height - height) / 2, width: width, height: height)
    }

    private func screenPoint(_ point: CGPoint, rect: CGRect, stage: CGSize) -> CGPoint {
        let base = CGPoint(x: rect.minX + point.x * rect.width, y: rect.minY + point.y * rect.height)
        return CGPoint(x: stage.width / 2 + (base.x - stage.width / 2) * zoom + pan.width,
                       y: stage.height / 2 + (base.y - stage.height / 2) * zoom + pan.height)
    }

    private func normalizedPoint(from screen: CGPoint, rect: CGRect, stage: CGSize) -> CGPoint {
        let x = (screen.x - pan.width - stage.width / 2) / zoom + stage.width / 2
        let y = (screen.y - pan.height - stage.height / 2) / zoom + stage.height / 2
        return CGPoint(x: (x - rect.minX) / rect.width, y: (y - rect.minY) / rect.height)
    }

    /// The fitted frame as it currently sits on screen (after zoom and pan).
    private func visibleFrame(_ rect: CGRect, stage: CGSize) -> CGRect {
        let a = screenPoint(.zero, rect: rect, stage: stage)
        let b = screenPoint(CGPoint(x: 1, y: 1), rect: rect, stage: stage)
        return CGRect(x: a.x, y: a.y, width: b.x - a.x, height: b.y - a.y)
    }

    /// Parks the loupe in an empty letterbox band when there is one (above the frame in
    /// portrait, the nearer side in landscape) so it never covers the other handles;
    /// otherwise floats it above the finger, flipping below near the top edge.
    private func loupePosition(for handle: CGPoint, frame: CGRect, in size: CGSize) -> CGPoint {
        let d = LoupeView.diameter, r = d / 2
        let margin: CGFloat = 12, tag: CGFloat = 20       // the tilted tag pokes out above
        let clampedX = min(max(handle.x, r + margin), size.width - r - margin)
        let clampedY = min(max(handle.y, r + margin + tag), size.height - r - margin)
        let topBand = frame.minY
        if topBand >= d + tag + 2 * margin {
            return CGPoint(x: clampedX, y: (topBand + tag) / 2)
        }
        let leftBand = frame.minX, rightBand = size.width - frame.maxX
        if handle.x < size.width / 2, leftBand >= d + 2 * margin { return CGPoint(x: leftBand / 2, y: clampedY) }
        if handle.x >= size.width / 2, rightBand >= d + 2 * margin {
            return CGPoint(x: size.width - rightBand / 2, y: clampedY)
        }
        let lift = r + 44
        var y = handle.y - lift
        if y - r - tag < margin { y = handle.y + lift }
        return CGPoint(x: clampedX, y: min(max(y, r + margin + tag), size.height - r - margin))
    }

    /// Far stickers go above their disc and near ones below; when the stage edge is in
    /// the way the sticker tucks in beside the disc, on the outside of the court.
    private func labelPlacement(for name: String, at p: CGPoint, corners: [CGPoint],
                                stage: CGSize) -> CornerHandle.LabelPlacement {
        let reach = CornerHandle.hitSize / 2 + 30
        let far = CornerShortLabel.isFar(name)
        if far, p.y - reach >= 0 { return .above }
        if !far, p.y + reach <= stage.height { return .below }
        let centroidX = corners.isEmpty ? stage.width / 2 : corners.map(\.x).reduce(0, +) / CGFloat(corners.count)
        let side = CornerHandle.hitSize / 2 + 50
        if p.x < centroidX, p.x - side >= 0 { return .leading }
        if p.x >= centroidX, p.x + side <= stage.width { return .trailing }
        return far ? .below : .above
    }

    private func adjust(_ name: String, direction: AccessibilityAdjustmentDirection) {
        guard let point = taps[name] else { return }
        let horizontal = 1 / CGFloat(taps.imageSize.width)
        switch direction {
        case .increment:
            moveCorner(name, CGPoint(x: point.x + horizontal, y: point.y))
        case .decrement:
            moveCorner(name, CGPoint(x: point.x - horizontal, y: point.y))
        @unknown default:
            break
        }
    }
}

// MARK: - Frame content

/// The video frame, or — when there is none — a drawn stand-in court. Shared by the
/// stage and the loupe so both always magnify the same picture.
struct FrameContent: View {
    let image: CGImage?
    let frameSize: ImageSize

    var body: some View {
        if let image {
            Image(decorative: image, scale: 1, orientation: .up)
                .resizable()
                .interpolation(.high)
        } else {
            PlaceholderCourt(frameSize: frameSize)
        }
    }
}

/// Court felt at night with the default quad's lines chalked in faintly and a net
/// band, so previews read as "a court seen from the baseline" rather than an error.
struct PlaceholderCourt: View {
    let frameSize: ImageSize

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.courtNight))
            // A darker back wall above the far baseline.
            let wall = CGRect(x: 0, y: 0, width: size.width, height: size.height * 0.2)
            context.fill(Path(wall), with: .linearGradient(
                Gradient(colors: [Color.night.opacity(0.55), Color.night.opacity(0)]),
                startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 0, y: wall.maxY)))

            let taps = CalibrationDefaults.taps(size: frameSize)
            let sx = size.width / CGFloat(max(frameSize.width, 1))
            let sy = size.height / CGFloat(max(frameSize.height, 1))
            func map(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * sx, y: p.y * sy) }

            for (name, points) in CourtOverlayGeometry.projectedLines(for: taps, samplesPerLine: 2) {
                guard points.count > 1 else { continue }
                var path = Path()
                path.addLines(points.map(map))
                let outer = CourtLineRole(name) == .outer
                context.stroke(path, with: .color(Color.paper.opacity(outer ? 0.30 : 0.16)),
                               style: StrokeStyle(lineWidth: outer ? 2 : 1.2, lineCap: .round))
            }

            let net = CourtOverlayGeometry.projectedNetLine(for: taps, samples: 2).map(map)
            if net.count == 2 {
                let height = hypot(net[1].x - net[0].x, net[1].y - net[0].y) * 0.2
                let top = net.map { CGPoint(x: $0.x, y: $0.y - height) }
                var band = Path()
                band.addLines([net[0], net[1], top[1], top[0]])
                band.closeSubpath()
                context.fill(band, with: .color(Color.night.opacity(0.28)))
                var mesh = Path()
                for i in 1..<12 {
                    let t = CGFloat(i) / 12
                    mesh.move(to: CGPoint(x: net[0].x + (net[1].x - net[0].x) * t, y: net[0].y + (net[1].y - net[0].y) * t))
                    mesh.addLine(to: CGPoint(x: top[0].x + (top[1].x - top[0].x) * t, y: top[0].y + (top[1].y - top[0].y) * t))
                }
                context.stroke(mesh, with: .color(Color.paper.opacity(0.07)), lineWidth: 1)
                var tape = Path()
                tape.move(to: top[0]); tape.addLine(to: top[1])
                context.stroke(tape, with: .color(Color.paper.opacity(0.34)), lineWidth: 2)
                var posts = Path()
                for i in 0..<2 { posts.move(to: net[i]); posts.addLine(to: top[i]) }
                context.stroke(posts, with: .color(Color.paper.opacity(0.26)),
                               style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
            }
        }
        .overlay(Halftone(color: Color.pistachio.opacity(0.07), spacing: 10, maxDot: 3, focus: .bottom))
    }
}

/// Which printed weight a projected court line gets.
enum CourtLineRole: Equatable {
    case outer, service, inner

    init(_ name: String) {
        switch name {
        case "near_baseline", "far_baseline", "left_doubles_side", "right_doubles_side": self = .outer
        case let n where n.contains("service") || n.contains("centre"): self = .service
        default: self = .inner
        }
    }
}
