import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// calibrate_from_points parity + the assertions of tests/test_court_calibration.py.
final class ManualCalibrationTests: XCTestCase {

    func testCalibrateFromPointsMatchesPython() throws {
        let cases = try XCTUnwrap(fixture("manual")["cases"]?.array)
        var worstPx = 0.0, worstF = 0.0
        for c in cases {
            let kind = c["kind"]!.string!
            let size = try c["size"]!.decode(ImageSize.self)
            let truth = try c["true"]!.decode(CameraModel.self)
            guard case .object(let po) = c["points"]! else { return XCTFail() }
            let pts = po.mapValues { v -> SIMD2<Double> in let d = v.doubles; return SIMD2(d[0], d[1]) }
            if let err = c["error"]?.string {
                XCTAssertThrowsError(try CourtCalibration.calibrateFromPoints(pts, size: size), "\(kind): \(err)")
                continue
            }
            let want = try c["camera"]!.decode(CameraModel.self)
            let got = try CourtCalibration.calibrateFromPoints(pts, size: size)
            let ag = keypointAgreement(want, got, margin: 200)
            let fRel = abs(got.focal / want.focal - 1)
            worstPx = max(worstPx, ag.max); worstF = max(worstF, fRel)
            XCTAssertLessThan(ag.max, 1e-3, "\(kind): keypoints vs Python")
            XCTAssertLessThan(fRel, 1e-6, "\(kind): focal vs Python")
            XCTAssertEqual(got.confidence, c["camera"]!["confidence"]!.double!, accuracy: 1e-6, kind)
            XCTAssertEqual(got.reprojRmsePx, c["camera"]!["reproj_rmse_px"]!.double!, accuracy: 1e-4, kind)
            XCTAssertLessThan(got.center.x, 0.5, kind)
            let e = calibErrors(truth, got)
            switch kind {
            case "exact":
                XCTAssertLessThan(e["world_max"]!, 1e-3, kind); XCTAssertLessThan(e["focal_rel"]!, 1e-3, kind)
                XCTAssertGreaterThan(got.confidence, 0.7); XCTAssertLessThan(got.reprojRmsePx, 0.01)
            case "noise":
                XCTAssertLessThan(e["reproj_med"]!, 2.5, kind); XCTAssertLessThan(e["focal_rel"]!, 0.03, kind)
            case "mirrored", "far_side", "extra":
                XCTAssertLessThan(e["world_max"]!, 1e-3, kind)
            default: break
            }
        }
        print("manual path vs Python: max keypoint diff \(worstPx) px, max focal rel diff \(worstF)")
    }

    func testErrors() {
        let size = ImageSize(width: 1920, height: 1080)
        XCTAssertThrowsError(try CourtCalibration.calibrateFromPoints(["nb_ld": [1, 2], "nb_rd": [5, 2], "fb_ld": [3, 1]], size: size)) {
            XCTAssertEqual($0 as? CourtCalibrationError, .insufficientKeypoints(3))
        }
        XCTAssertThrowsError(try CourtCalibration.calibrateFromPoints(["nb_ld": [.nan, 2], "nb_rd": [5, 2], "fb_ld": [3, 1], "fb_rd": [4, 4]], size: size))
    }

    func testHelpersRoundtrip() {
        let size = ImageSize(width: 1920, height: 1080)
        let truth = CameraModel.lookAt(eye: [-14, 0, 6], target: .zero, focalPx: 1500, imageSize: size)
        let P = CourtModel.keypointsArray().points
        let H = CourtCalibration.homographyCourtToImage(truth)
        for p in P {
            let q = H * SIMD3(p.x, p.y, 1)
            XCTAssertLessThan(simd_distance(SIMD2(q.x / q.z, q.y / q.z), truth.project(p)), 1e-6)
        }
        let back = CourtCalibration.imageToCourt(truth, truth.project(P))
        for (b, p) in zip(back, P) { XCTAssertLessThan(simd_distance(b, p), 1e-6) }
    }
}
