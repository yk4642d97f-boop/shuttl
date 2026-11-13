import XCTest
import CoreGraphics
import simd
import ShuttlCore
import ShuttlPresent

final class CourtProjectionTests: XCTestCase {
    typealias F = LockFixtures

    func testProjectionOfAWellFramedCamera() {
        let cam = F.camera()
        let p = CourtProjection(camera: cam)
        XCTAssertEqual(p.lines.count, CourtModel.lines.count + 3)       // + net top + 2 posts
        XCTAssertEqual(Set(p.corners.keys), Set(CourtCorners.outerNames))
        for (_, c) in p.corners { XCTAssertTrue((0...1).contains(c.x) && (0...1).contains(c.y), "\(c)") }
        let o = try! XCTUnwrap(p.outline)
        XCTAssertLessThan(o[0].x, o[1].x, "near-left is left of near-right")
        XCTAssertGreaterThan(o[0].y, o[3].y, "near baseline is below the far one")
        XCTAssertEqual(p.framing.hints, [])
        XCTAssertEqual(p.framing.rollDegrees, 0, accuracy: 1e-9)
        XCTAssertGreaterThan(p.framing.courtAreaFraction, 0.1)
        // Every polyline point is the camera's projection, normalised.
        let nb = try! XCTUnwrap(p.line(named: "near_baseline")?.polylines.first)
        let a = cam.project([-6.7, -3.05, 0])
        XCTAssertEqual(Double(nb[0].x), a.x / 640, accuracy: 1e-9)
        XCTAssertEqual(Double(nb[0].y), a.y / 360, accuracy: 1e-9)
        // Net top is above the floor under it, and sags in the middle less than a pixel-ish.
        let net = try! XCTUnwrap(p.line(named: "net_top")?.polylines.first)
        XCTAssertLessThan(Double(net[net.count / 2].y), cam.project([0, 0, 0]).y / 360)
        XCTAssertEqual(p.line(named: "net_top")?.kind, .netTop)
    }

    func testRollSignAndTiltHint() {
        let cam = F.camera(roll: 6)
        XCTAssertEqual(cam.rollDegrees, 6, accuracy: 1e-6)
        let left = cam.project([6.7, 3.05, 0]), right = cam.project([6.7, -3.05, 0])
        XCTAssertGreaterThan(right.y, left.y, "positive roll: scene rotated clockwise (right end lower)")
        XCTAssertTrue(CourtProjection(camera: cam).framing.hints.contains(.tilted))
        XCTAssertFalse(CourtProjection(camera: F.camera(roll: 2)).framing.hints.contains(.tilted))
        XCTAssertEqual(F.camera(roll: -5).rollDegrees, -5, accuracy: 1e-6)
    }

    func testOutOfFrameHints() {
        // Aimed at the near service line with a long lens: far baseline above the top edge.
        var c = CameraModel.lookAt(eye: [-12.5, 0, 3.5], target: [-4, 0, 0], focalPx: 900, imageSize: F.size)
        var h = CourtProjection(camera: c).framing.hints
        XCTAssertTrue(h.contains(.farBaselineOutOfFrame), "\(h)")
        // Panned left (towards +y): the right side line leaves the right edge.
        c = CameraModel.lookAt(eye: [-12.5, 0, 3.5], target: [0.5, 2.8, 0], focalPx: 500, imageSize: F.size)
        h = CourtProjection(camera: c).framing.hints
        XCTAssertTrue(h.contains(.rightSideOutOfFrame), "\(h)")
        XCTAssertFalse(h.contains(.leftSideOutOfFrame))
        // Far away with a short lens: court too small.
        c = CameraModel.lookAt(eye: [-30, 0, 4], target: [0, 0, 0], focalPx: 300, imageSize: F.size)
        h = CourtProjection(camera: c).framing.hints
        XCTAssertTrue(h.contains(.courtTooSmall), "\(h)")
        // Right behind the near baseline, low: both side lines out, near baseline out.
        c = CameraModel.lookAt(eye: [-7.2, 0, 1.4], target: [0, 0, 0], focalPx: 700, imageSize: F.size)
        h = CourtProjection(camera: c).framing.hints
        XCTAssertEqual(h.first, .bothSidesOutOfFrame, "\(h)")
        XCTAssertTrue(h.contains(.nearBaselineOutOfFrame), "\(h)")
    }

    func testDistortionFoldIsCut() {
        var c = F.camera()
        c.dist = [-0.9, 0, 0, 0, 0]
        let p = CourtProjection(camera: c)
        for l in p.lines { for pl in l.polylines { for q in pl { XCTAssertTrue(q.x.isFinite && q.y.isFinite) } } }
    }

    func testBehindCameraPointsDropped() {
        // Camera standing ON the court looking at the far half: the near baseline is behind it.
        let c = CameraModel.lookAt(eye: [-2, 0, 1.7], target: [6, 0, 0], focalPx: 500, imageSize: F.size)
        let p = CourtProjection(camera: c)
        XCTAssertNil(p.outline)
        XCTAssertTrue(p.line(named: "near_baseline")!.polylines.isEmpty)
        XCTAssertFalse(p.line(named: "far_baseline")!.polylines.isEmpty)
    }

    func testRescaleIsExactPixelCentreMapping() throws {
        let c = F.camera(dx: 3, dy: -2)
        let big = try XCTUnwrap(c.rescaled(to: ImageSize(width: 1920, height: 1080)))
        for p in CameraAgreement.points {
            let u = c.project(p), v = big.project(p)
            guard u.x.isFinite else { continue }
            XCTAssertEqual(v.x, (u.x + 0.5) * 3 - 0.5, accuracy: 1e-9)
            XCTAssertEqual(v.y, (u.y + 0.5) * 3 - 0.5, accuracy: 1e-9)
        }
        XCTAssertNil(c.rescaled(to: ImageSize(width: 1440, height: 1080)), "4:3 is a crop of 16:9, not a resize")
        let back = try XCTUnwrap(big.rescaled(to: F.size))
        XCTAssertLessThan(CameraAgreement.maxDisagreementPx(c, back), 1e-9)
        XCTAssertEqual(CameraAgreement.maxDisagreementPx(c, F.camera(dx: 5, dy: -2)), 2, accuracy: 1e-9)
    }

    func testManualCorners() {
        let taps: [String: CGPoint] = ["nb_ld": CGPoint(x: 0.9, y: 0.95), "nb_rd": CGPoint(x: 0.1, y: 0.95),
                                       "fb_ld": CGPoint(x: 0.6, y: 0.3), "fb_rd": CGPoint(x: 0.4, y: 0.3),
                                       "bogus": CGPoint(x: 0.5, y: 0.5), "ns_ld": CGPoint(x: 1.2, y: 0.5)]
        let m = CourtCorners.manualCorners(taps)
        XCTAssertEqual(Set(m.keys), Set(CourtCorners.outerNames))
        XCTAssertEqual(m["fb_ld"]!.x, 0.6, accuracy: 1e-12)
        XCTAssertTrue(CourtCorners.isComplete(taps))
        let px = CourtCorners.manualCornersPixels(taps, size: ImageSize(width: 1920, height: 1080))
        XCTAssertEqual(px["nb_ld"]!.y, 0.95 * 1080, accuracy: 1e-9)
        let fromLock = CourtCorners.taps(from: CourtProjection(camera: F.camera()))
        XCTAssertEqual(fromLock.count, 4)
    }

    func testVideoGeometry() {
        let fit = VideoGeometry(videoSize: CGSize(width: 1920, height: 1080), in: CGSize(width: 390, height: 844), gravity: .fit)
        XCTAssertEqual(fit.videoRect.width, 390, accuracy: 1e-9)
        XCTAssertEqual(fit.point(CGPoint(x: 0.5, y: 0.5)).y, 422, accuracy: 1e-9)
        let fill = VideoGeometry(videoSize: CGSize(width: 1920, height: 1080), in: CGSize(width: 844, height: 390), gravity: .fill)
        XCTAssertEqual(fill.videoRect.width, 844, accuracy: 1e-9)
        let q = fill.normalized(fill.point(CGPoint(x: 0.2, y: 0.7)))
        XCTAssertEqual(q.x, 0.2, accuracy: 1e-12); XCTAssertEqual(q.y, 0.7, accuracy: 1e-12)
    }
}
