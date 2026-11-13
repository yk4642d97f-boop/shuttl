import CoreGraphics
import Foundation
import ImageIO
@testable import ShuttlCalibrationUI
import ShuttlCore
import ShuttlDesign
import SwiftUI
import XCTest

final class CalibrationTests: XCTestCase {
    private let size = ImageSize(width: 1920, height: 1080)

    func testCornerNamesMatchLookAtProjection() throws {
        let camera = CameraModel.lookAt(eye: [-11, 0, 3], target: [0, 0, 0], focalPx: 1200, imageSize: size)
        let world: [String: SIMD3<Double>] = [
            "nb_ld": [-CourtModel.halfLength, -CourtModel.doublesHalfWidth, 0],
            "nb_rd": [-CourtModel.halfLength, CourtModel.doublesHalfWidth, 0],
            "fb_ld": [CourtModel.halfLength, -CourtModel.doublesHalfWidth, 0],
            "fb_rd": [CourtModel.halfLength, CourtModel.doublesHalfWidth, 0],
        ]
        let projected = world.mapValues { point -> CGPoint in
            let pixel = camera.project(point)
            XCTAssertTrue(pixel.isFinite)
            return CGPoint(x: pixel.x, y: pixel.y)
        }

        XCTAssertLessThan(projected["nb_rd"]!.x, projected["nb_ld"]!.x, "+y camera-left puts rd at image-left on the near baseline")
        XCTAssertLessThan(projected["fb_rd"]!.x, projected["fb_ld"]!.x, "+y camera-left puts rd at image-left on the far baseline")
        XCTAssertLessThan(projected["fb_rd"]!.y, projected["nb_rd"]!.y, "far baseline must be above the near baseline")
        XCTAssertLessThan(projected["fb_ld"]!.y, projected["nb_ld"]!.y)

        let assignedImageCorners: [String: CGPoint] = [
            "nb_rd": projected["nb_rd"]!, "nb_ld": projected["nb_ld"]!,
            "fb_rd": projected["fb_rd"]!, "fb_ld": projected["fb_ld"]!,
        ]
        XCTAssertEqual(assignedImageCorners["nb_rd"], projected["nb_rd"])
        XCTAssertEqual(assignedImageCorners["nb_ld"], projected["nb_ld"])
        XCTAssertEqual(assignedImageCorners["fb_rd"], projected["fb_rd"])
        XCTAssertEqual(assignedImageCorners["fb_ld"], projected["fb_ld"])
        XCTAssertEqual(CourtCornerLabel.chinese(for: "nb_rd"), "近端底线左角")
        XCTAssertEqual(CourtCornerLabel.chinese(for: "nb_ld"), "近端底线右角")
        XCTAssertEqual(CourtCornerLabel.chinese(for: "fb_rd"), "远端底线左角")
        XCTAssertEqual(CourtCornerLabel.chinese(for: "fb_ld"), "远端底线右角")
    }

    func testHomographyAndProjectedCourtLinesMatchCameraGroundTruth() throws {
        let camera = CameraModel.lookAt(eye: [-11, 0, 3], target: [0, 0, 0], focalPx: 1200, imageSize: size)
        let names = CourtCornerTaps.outerNames
        let taps = CourtCornerTaps(imageSize: size, points: Dictionary(uniqueKeysWithValues: names.map { name in
            let world = worldPoint(for: name)
            let px = camera.project(world)
            return (name, CGPoint(x: px.x / Double(size.width), y: px.y / Double(size.height)))
        }))
        let overlay = CourtOverlayGeometry.projectedLines(for: taps, samplesPerLine: 31)
        XCTAssertEqual(overlay.count, CourtModel.lines.count)

        for line in CourtModel.lines {
            let actual = try XCTUnwrap(overlay[line.name])
            XCTAssertEqual(actual.count, 31)
            for (index, pixel) in actual.enumerated() {
                let fraction = Double(index) / Double(actual.count - 1)
                let world = line.a + fraction * (line.b - line.a)
                let expected = camera.project(SIMD3(world.x, world.y, 0))
                XCTAssertLessThan(hypot(Double(pixel.x) - expected.x, Double(pixel.y) - expected.y), 0.5,
                                  "\(line.name) sample \(index) deviated from pinhole projection")
            }
        }
    }

    func testQuadValidation() {
        let valid = sampleTaps()
        XCTAssertNil(CourtQuadValidator.issue(for: valid))
        XCTAssertTrue(CourtQuadValidator.isValid(valid))

        var missing = valid
        missing["fb_rd"] = nil
        XCTAssertEqual(CourtQuadValidator.issue(for: missing), .missingCorner)

        var nonConvex = valid
        nonConvex["fb_rd"] = CGPoint(x: 0.9, y: 0.75)
        XCTAssertEqual(CourtQuadValidator.issue(for: nonConvex), .nonConvex)

        var degenerate = valid
        degenerate["fb_ld"] = CGPoint(x: 0.50, y: 0.80)
        degenerate["fb_rd"] = CGPoint(x: 0.49, y: 0.80)
        XCTAssertEqual(CourtQuadValidator.issue(for: degenerate), .degenerate)

        var farBelow = valid
        farBelow["fb_ld"] = CGPoint(x: 0.70, y: 0.85)
        farBelow["fb_rd"] = CGPoint(x: 0.30, y: 0.85)
        XCTAssertEqual(CourtQuadValidator.issue(for: farBelow), .farBaselineNotAbove)

        var farWider = valid
        farWider["fb_ld"] = CGPoint(x: 0.99, y: 0.20)
        farWider["fb_rd"] = CGPoint(x: 0.01, y: 0.20)
        XCTAssertEqual(CourtQuadValidator.issue(for: farWider), .farBaselineNotShorter)
    }

    func testStoreRoundTripIsResolutionScoped() throws {
        let suiteName = "ShuttlCalibrationUI.tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = CourtCalibrationStore(defaults: defaults, keyPrefix: "test.taps")
        let taps = sampleTaps()
        try store.save(taps)
        XCTAssertEqual(store.load(for: size), taps)
        XCTAssertNil(store.load(for: ImageSize(width: 1280, height: 720)))
        store.remove(for: size)
        XCTAssertNil(store.load(for: size))
    }

    /// Tripod close behind the near baseline: the near corners are below the frame, so the user
    /// marks the near short-service-line corners instead. The quad must validate, the court
    /// overlay must still match the camera, and only the active four names may be handed on.
    func testNearServiceLineCornersStandInForAnOffFrameBaseline() throws {
        let camera = CameraModel.lookAt(eye: [-8.2, 0, 1.6], target: [0, 0, 0.3], focalPx: 1450, imageSize: size)
        func norm(_ name: String) -> CGPoint {
            let w = CourtModel.keypoints[name]!
            let px = camera.project(SIMD3(w.x, w.y, 0))
            return CGPoint(x: px.x / Double(size.width), y: px.y / Double(size.height))
        }
        XCTAssertGreaterThan(norm("nb_ld").y, 1, "fixture: the near baseline really is off the bottom of the frame")
        var taps = CourtCornerTaps(imageSize: size, points: ["ns_ld": norm("ns_ld"), "ns_rd": norm("ns_rd"),
                                                              "fb_ld": norm("fb_ld"), "fb_rd": norm("fb_rd")])
        XCTAssertEqual(taps.cornerSet, .nearServiceLine)
        XCTAssertTrue(taps.isComplete)
        XCTAssertNil(CourtQuadValidator.issue(for: taps))

        let homography = try XCTUnwrap(CourtFloorHomography(taps: taps))
        for name in ["nb_ld", "nb_rd", "fs_ld", "fl_rs"] {
            let w = CourtModel.keypoints[name]!
            let got = try XCTUnwrap(homography.project(w))
            let want = camera.project(SIMD3(w.x, w.y, 0))
            XCTAssertLessThan(hypot(got.x - want.x, got.y - want.y), 0.5, name)
        }

        // Switching to baseline projects the near corners (clamped into the frame) and drops ns_*.
        let baseline = taps.switched(to: .baseline)
        XCTAssertEqual(Set(baseline.points.keys), Set(CalibrationCornerSet.baseline.names))
        XCTAssertEqual(baseline["nb_ld"]!.y, 1, accuracy: 1e-9)
        // From a valid baseline quad (camera far enough back to see the whole court) the service
        // corners land exactly where that camera puts them.
        let wide = CameraModel.lookAt(eye: [-11, 0, 3], target: [0, 0, 0], focalPx: 1200, imageSize: size)
        func wideNorm(_ name: String) -> CGPoint {
            let w = CourtModel.keypoints[name]!
            let px = wide.project(SIMD3(w.x, w.y, 0))
            return CGPoint(x: px.x / Double(size.width), y: px.y / Double(size.height))
        }
        taps = CourtCornerTaps(imageSize: size, points: Dictionary(uniqueKeysWithValues:
            CalibrationCornerSet.baseline.names.map { ($0, wideNorm($0)) }))
        XCTAssertNil(CourtQuadValidator.issue(for: taps))
        let service = taps.switched(to: .nearServiceLine)
        XCTAssertEqual(Set(service.points.keys), Set(CalibrationCornerSet.nearServiceLine.names))
        for name in ["ns_ld", "ns_rd"] {
            XCTAssertEqual(service[name]!.x, wideNorm(name).x, accuracy: 1e-4, name)
            XCTAssertEqual(service[name]!.y, wideNorm(name).y, accuracy: 1e-4, name)
        }

        // A stale extra pair never reaches the solver; a complete baseline pair wins.
        var mixed = service
        mixed.points["nb_ld"] = wideNorm("nb_ld")
        mixed.points["nb_rd"] = wideNorm("nb_rd")
        XCTAssertEqual(mixed.cornerSet, .baseline)
        XCTAssertEqual(Set(mixed.activePoints.keys), Set(CalibrationCornerSet.baseline.names))
        XCTAssertEqual(Set(service.activePoints.keys), Set(CalibrationCornerSet.nearServiceLine.names))
        XCTAssertEqual(CornerShortLabel.text(for: "ns_ld"), "发球线右")
        XCTAssertEqual(CornerShortLabel.text(for: "ns_rd"), "发球线左")
    }

    func testCornerShortLabelsFollowTheLongNames() {
        XCTAssertEqual(CornerShortLabel.text(for: "nb_rd"), "近左")
        XCTAssertEqual(CornerShortLabel.text(for: "nb_ld"), "近右")
        XCTAssertEqual(CornerShortLabel.text(for: "fb_rd"), "远左")
        XCTAssertEqual(CornerShortLabel.text(for: "fb_ld"), "远右")
        XCTAssertTrue(CornerShortLabel.isFar("fb_ld"))
        XCTAssertFalse(CornerShortLabel.isFar("nb_rd"))
    }

    @MainActor
    func testImageRendererSnapshotsWritePNGFiles() throws {
        DisplayFace.register()
        let output = URL(fileURLWithPath: "/tmp/restyle_cal", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let frame = makeFrame(width: 960, height: 540)
        let frames: [(String, CGImage?, CourtCalibrationPreviewState, CGSize)] = [
            ("initial.png", frame, .normal, CGSize(width: 390, height: 844)),
            ("selected-loupe.png", frame, .selectedWithLoupe("fb_ld"), CGSize(width: 390, height: 844)),
            ("invalid.png", frame, .invalid, CGSize(width: 390, height: 844)),
            ("placeholder.png", nil, .normal, CGSize(width: 390, height: 844)),
            ("placeholder-loupe.png", nil, .selectedWithLoupe("nb_rd"), CGSize(width: 390, height: 844)),
            ("landscape.png", frame, .normal, CGSize(width: 844, height: 390)),
            ("large-type.png", frame, .invalid, CGSize(width: 390, height: 844)),
            ("service-line.png", frame, .normal, CGSize(width: 390, height: 844)),
            ("service-line-landscape.png", frame, .normal, CGSize(width: 844, height: 390)),
        ]

        for (name, image, state, size) in frames {
            var taps = image == nil ? nil : sampleTaps(width: 960, height: 540)
            if name.hasPrefix("service-line") {
                taps?.points["nb_ld"] = nil
                taps?.points["nb_rd"] = nil
            }
            let view = CourtCalibrationView(image: image, initial: taps, onDone: { _ in }, onCancel: {}, previewState: state)
                .frame(width: size.width, height: size.height)
                .environment(\.dynamicTypeSize, name == "large-type.png" ? .accessibility2 : .large)
            let renderer = ImageRenderer(content: view)
            renderer.scale = 2
            let cgImage = try XCTUnwrap(renderer.cgImage, "ImageRenderer produced no CGImage for \(name)")
            let url = output.appendingPathComponent(name)
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, cgImage, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            XCTAssertGreaterThan(try Data(contentsOf: url).count, 1000)
        }
    }

    private func sampleTaps(width: Int? = nil, height: Int? = nil) -> CourtCornerTaps {
        let size = ImageSize(width: width ?? self.size.width, height: height ?? self.size.height)
        return CourtCornerTaps(imageSize: size, points: [
            "nb_ld": CGPoint(x: 0.80, y: 0.84),
            "nb_rd": CGPoint(x: 0.20, y: 0.84),
            "fb_ld": CGPoint(x: 0.62, y: 0.22),
            "fb_rd": CGPoint(x: 0.38, y: 0.22),
            "ns_ld": CGPoint(x: 0.72, y: 0.58),
            "ns_rd": CGPoint(x: 0.28, y: 0.58),
        ])
    }

    private func worldPoint(for name: String) -> SIMD3<Double> {
        switch name {
        case "nb_ld": return [-CourtModel.halfLength, -CourtModel.doublesHalfWidth, 0]
        case "nb_rd": return [-CourtModel.halfLength, CourtModel.doublesHalfWidth, 0]
        case "fb_ld": return [CourtModel.halfLength, -CourtModel.doublesHalfWidth, 0]
        default: return [CourtModel.halfLength, CourtModel.doublesHalfWidth, 0]
        }
    }

    /// A synthetic "photo": hall wall, green floor, and court lines painted where the
    /// sample taps say they are, so snapshots show the overlay sitting on real lines.
    private func makeFrame(width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let h = CGFloat(height)
        context.setFillColor(red: 0.16, green: 0.36, blue: 0.27, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(red: 0.20, green: 0.21, blue: 0.26, alpha: 1)
        context.fill(CGRect(x: 0, y: h * 0.84, width: CGFloat(width), height: h * 0.16))
        context.setStrokeColor(red: 0.95, green: 0.92, blue: 0.82, alpha: 0.9)
        context.setLineWidth(3)
        let lines = CourtOverlayGeometry.projectedLines(for: sampleTaps(width: width, height: height), samplesPerLine: 2)
        for points in lines.values where points.count == 2 {
            context.move(to: CGPoint(x: points[0].x, y: h - points[0].y))
            context.addLine(to: CGPoint(x: points[1].x, y: h - points[1].y))
        }
        context.strokePath()
        return context.makeImage()!
    }
}
