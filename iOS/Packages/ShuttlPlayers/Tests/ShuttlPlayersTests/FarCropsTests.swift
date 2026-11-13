import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// Far-crop layouts of the Vision detector (geometry only; recall is measured by
/// ShuttlPlayersBench --layouts).
final class FarCropsTests: XCTestCase {
    func broadcastCamera() throws -> CameraModel {
        let f = Fx.dict(Fx.dict(try Fx.load("misc"))["far_roi"])
        return try XCTUnwrap(try Fx.camera(f["camera"]))
    }

    /// iPhone main camera (70° HFOV, 1080p) on a tripod 1.4 m high, 1.5 m behind the baseline.
    func phoneCamera() -> CameraModel {
        let W = 1920.0, f = W / (2 * tan(35 * Double.pi / 180))
        return CameraModel.lookAt(eye: [-CourtModel.halfLength - 1.5, 0, 1.4], target: [0, 0, 0], focalPx: f,
                                  imageSize: ImageSize(width: 1920, height: 1080))
    }

    func testSingleLayoutIsPlayersPyFarROI() throws {
        for cam in [try broadcastCamera(), phoneCamera()] {
            let old = try XCTUnwrap(FarROI.compute(camera: cam, size: cam.imageSize)).rect
            XCTAssertEqual(FarCrops.rects(.single, camera: cam, size: cam.imageSize), [old])
            XCTAssertEqual(PlayerTracker.defaultFarROIs(camera: cam, meta: VideoMeta(fps: 30, width: cam.imageSize.width,
                           height: cam.imageSize.height, nFrames: 1, timestamps: [0]), layout: .single), [old])
        }
        XCTAssertEqual(FarCrops.rects(.none, camera: phoneCamera(), size: phoneCamera().imageSize), [])
        XCTAssertEqual(PlayerTracker.defaultFarROIs(camera: nil, meta: VideoMeta(fps: 30, width: 1920, height: 1080,
                       nFrames: 1, timestamps: [0])), [])
    }

    func testTwoCropLayoutsMagnifyAndOverlap() throws {
        for cam in [try broadcastCamera(), phoneCamera()] {
            let size = cam.imageSize
            let single = FarCrops.rects(.single, camera: cam, size: size)[0]
            for layout in [FarCropLayout.lateral, .depth] {
                let r = FarCrops.rects(layout, camera: cam, size: size)
                XCTAssertEqual(r.count, 2, "\(layout)")
                for c in r {
                    XCTAssertTrue(c.x1 >= 0 && c.y1 >= 0 && c.x2 <= size.width && c.y2 <= size.height)
                    // inside the single crop (same world extent, split)
                    XCTAssertTrue(c.x1 >= single.x1 && c.x2 <= single.x2 && c.y1 >= single.y1 && c.y2 <= single.y2, "\(layout) \(c)")
                }
                // the two crops overlap (a player on the split line is whole in one of them)
                XCTAssertLessThan(max(r[0].x1, r[1].x1), min(r[0].x2, r[1].x2))
                XCTAssertLessThan(max(r[0].y1, r[1].y1), min(r[0].y2, r[1].y2))
            }
            // lateral halves the long side of the phone's wide far-half crop
            let lat = FarCrops.rects(.lateral, camera: cam, size: size)
            XCTAssertLessThan(Double(max(lat[0].width, lat[0].height)), 0.7 * Double(max(single.width, single.height)))
        }
    }

    /// A far player on the centre line is whole inside both lateral crops.
    func testPlayerOnSplitLineIsInsideBothLateralCrops() {
        let cam = phoneCamera()
        let r = FarCrops.rects(.lateral, camera: cam, size: cam.imageSize)
        for x in [1.0, 3.5, 6.7] {
            for y in [-0.25, 0.0, 0.25] {
                let pts = [SIMD3(x, y - 0.3, 0), SIMD3(x, y + 0.3, 0), SIMD3(x, y, 1.8)].map { cam.project($0) }
                for c in r {
                    XCTAssertTrue(pts.allSatisfy { Double(c.x1) <= $0.x && $0.x <= Double(c.x2) }, "x \(x) y \(y) \(c)")
                }
            }
        }
    }
}
