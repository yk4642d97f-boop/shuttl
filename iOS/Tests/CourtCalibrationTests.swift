import XCTest
import simd
@testable import Shuttl

final class CourtCalibrationTests: XCTestCase {

    private let hd = CourtCalibration.PixelSize(width: 1920, height: 1080)

    /// Broadcast-style: high behind the near baseline (like the TrackNet test clips).
    private var broadcastCam: CourtCalibration {
        .lookAt(eye: SIMD3(-15.5, 0.4, 6.5), target: SIMD3(0.5, 0, 0), focalPx: 1650, imageSize: hd, rollDegrees: 1.2)
    }

    /// Phone on a tripod, lower and off-centre.
    private var tripodCam: CourtCalibration {
        .lookAt(eye: SIMD3(-14.5, 1.0, 2.2), target: SIMD3(-0.5, -0.3, 0), focalPx: 1450, imageSize: hd)
    }

    // MARK: Helpers

    private struct LCG {
        var state: UInt64
        /// Uniform in [-1, 1].
        mutating func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53) * 2 - 1
        }
    }

    private func corners(of cam: CourtCalibration) throws -> [CourtCorner: CGPoint] {
        var out: [CourtCorner: CGPoint] = [:]
        for c in CourtCorner.allCases {
            let p = try XCTUnwrap(cam.project(c.world3))
            XCTAssert((0...CGFloat(cam.imageSize.width)).contains(p.x) && (0...CGFloat(cam.imageSize.height)).contains(p.y),
                      "\(c) off-frame at \(p)")
            out[c] = p
        }
        return out
    }

    private func sampleWorldPoints() -> [SIMD3<Double>] {
        var pts: [SIMD3<Double>] = []
        for x in stride(from: -6.7, through: 6.7, by: 1.675) {
            for y in stride(from: -3.05, through: 3.05, by: 1.525) {
                for z in [0.0, 1.55, 4.0] { pts.append(SIMD3(x, y, z)) }
            }
        }
        return pts
    }

    // MARK: Projection round trip

    func testProjectPixelToFloorRoundTrip() throws {
        for cam in [broadcastCam, tripodCam] {
            for p in sampleWorldPoints() {
                let px = try XCTUnwrap(cam.project(p))
                let back = try XCTUnwrap(cam.pixelToFloor(px, height: p.z))
                XCTAssertLessThan(simd_distance(back, p), 1e-6, "round trip failed for \(p)")
            }
        }
    }

    func testRoundTripWithDistortion() throws {
        var cam = broadcastCam
        cam.dist = [0.08, -0.02, 0.0005, -0.0003, 0.001]
        for p in sampleWorldPoints() {
            let px = try XCTUnwrap(cam.project(p))
            let back = try XCTUnwrap(cam.pixelToFloor(px, height: p.z))
            XCTAssertLessThan(simd_distance(back, p), 1e-6)
        }
    }

    func testPointsBehindCameraAreRejected() {
        let cam = broadcastCam
        XCTAssertNil(cam.project(cam.center - SIMD3(5, 0, 0)))
        // A pixel above the horizon never reaches the floor.
        XCTAssertNil(cam.pixelToFloor(CGPoint(x: 960, y: -4000)))
    }

    func testFloorHomographyMatchesProjection() throws {
        let cam = tripodCam
        let h = cam.floorHomography
        for p in sampleWorldPoints() where p.z == 0 {
            let q = h * SIMD3(p.x, p.y, 1)
            let px = try XCTUnwrap(cam.project(p))
            XCTAssertEqual(q.x / q.z, Double(px.x), accuracy: 1e-6)
            XCTAssertEqual(q.y / q.z, Double(px.y), accuracy: 1e-6)
        }
        let mapper = CourtMapper(calibration: cam)
        let px = try XCTUnwrap(cam.project(SIMD3(-3, 1.2, 0)))
        let m = try XCTUnwrap(mapper.imagePixelToMeters(px))
        XCTAssertEqual(m.x, -3, accuracy: 1e-6)
        XCTAssertEqual(m.y, 1.2, accuracy: 1e-6)
    }

    // MARK: Codable (backend CameraModel.to_dict)

    func testDecodesBackendJSON() throws {
        let json = """
        {"K": [[1650.0, 0.0, 960.0], [0.0, 1650.0, 540.0], [0.0, 0.0, 1.0]],
         "R": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
         "t": [0.1, 2.5, 14.0], "dist": [0.01, 0.0, 0.0, 0.0],
         "image_size": [1920, 1080], "reproj_rmse_px": null, "confidence": 0.8}
        """
        let cam = try JSONDecoder().decode(CourtCalibration.self, from: Data(json.utf8))
        XCTAssertEqual(cam.K[2, 0], 960)
        XCTAssertEqual(cam.K[2, 1], 540)
        XCTAssertEqual(cam.R[1, 2], 1)      // row 2, column 1
        XCTAssertEqual(cam.R[2, 1], -1)     // row 1, column 2
        XCTAssertEqual(cam.t, SIMD3(0.1, 2.5, 14.0))
        XCTAssertEqual(cam.dist, [0.01, 0, 0, 0, 0])
        XCTAssertEqual(cam.imageSize, .init(width: 1920, height: 1080))
        XCTAssertNil(cam.reprojRMSEPx)
        XCTAssertEqual(cam.confidence, 0.8)
    }

    func testDecodesMinimalAndColumnVectorJSON() throws {
        let json = """
        {"K": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "t": [[0], [0], [10]], "image_size": [1280, 720]}
        """
        let cam = try JSONDecoder().decode(CourtCalibration.self, from: Data(json.utf8))
        XCTAssertEqual(cam.t, SIMD3(0, 0, 10))
        XCTAssertEqual(cam.dist, [0, 0, 0, 0, 0])
        XCTAssertEqual(cam.confidence, 0)
    }

    func testEncodeUsesBackendKeysAndRoundTrips() throws {
        var cam = broadcastCam
        cam.reprojRMSEPx = 0.7
        cam.confidence = 0.9
        let data = try JSONEncoder().encode(cam)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(obj.keys), ["K", "R", "t", "dist", "image_size", "reproj_rmse_px", "confidence"])
        XCTAssertEqual(obj["image_size"] as? [Int], [1920, 1080])
        let k = try XCTUnwrap(obj["K"] as? [[Double]])
        XCTAssertEqual(k[0][2], 960, accuracy: 1e-12)   // row-major like numpy tolist()
        XCTAssertEqual(try JSONDecoder().decode(CourtCalibration.self, from: data), cam)

        cam.reprojRMSEPx = nil
        let obj2 = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(cam)) as? [String: Any])
        XCTAssertNil(obj2["reproj_rmse_px"], "unknown RMSE is omitted so from_dict defaults to NaN")
    }

    // MARK: Four-corner calibration

    func testFourCornerCalibrationIsExactWithoutNoise() throws {
        for truth in [broadcastCam, tripodCam] {
            let cal = try CourtCalibration.calibrate(corners: try corners(of: truth), imageSize: hd)
            XCTAssertEqual(cal.focalLengthPx, truth.focalLengthPx, accuracy: 1e-4 * truth.focalLengthPx)
            XCTAssertLessThan(simd_distance(cal.center, truth.center), 1e-4)
            for p in sampleWorldPoints() {
                let a = try XCTUnwrap(cal.project(p)), b = try XCTUnwrap(truth.project(p))
                XCTAssertLessThan(hypot(a.x - b.x, a.y - b.y), 1e-3)
            }
            XCTAssertLessThan(try XCTUnwrap(cal.reprojRMSEPx), 1e-4)
            XCTAssertGreaterThan(cal.confidence, 0.5)
        }
    }

    func testFourCornerCalibrationWithOnePixelNoiseNearHalf() throws {
        var rng = LCG(state: 42)
        let nearGrid: [SIMD2<Double>] = stride(from: -6.7, through: 0, by: 0.67).flatMap { x in
            stride(from: -3.05, through: 3.05, by: 0.61).map { SIMD2(x, $0) }
        }
        // Measured (200 trials): broadcast mean 0.7 cm / p95 1.6 cm / worst 3.5 cm;
        // low 2.2 m tripod mean 1.3 cm / p95 3.2 cm / worst 6.7 cm (grazing view of the far corners).
        for (truth, worstBound) in [(broadcastCam, 0.05), (tripodCam, 0.08)] {
            let clean = try corners(of: truth)
            var errors: [Double] = []
            for _ in 0..<200 {
                var noisy = clean
                for (k, p) in clean { noisy[k] = CGPoint(x: p.x + rng.next(), y: p.y + rng.next()) }
                let cal = try CourtCalibration.calibrate(corners: noisy, imageSize: hd)
                for g in nearGrid {
                    let px = try XCTUnwrap(truth.project(SIMD3(g.x, g.y, 0)))
                    let back = try XCTUnwrap(cal.pixelToFloor(px))
                    errors.append(simd_distance(SIMD2(back.x, back.y), g))
                }
            }
            errors.sort()
            let mean = errors.reduce(0, +) / Double(errors.count)
            let p95 = errors[Int(Double(errors.count) * 0.95)]
            XCTAssertLessThan(mean, 0.025, "mean near-half floor error \(mean) m")
            XCTAssertLessThan(p95, 0.05, "p95 near-half floor error \(p95) m")
            XCTAssertLessThan(errors.last!, worstBound, "worst near-half floor error \(errors.last!) m")
        }
    }

    func testMirroredCornersAreRejected() throws {
        let c = try corners(of: broadcastCam)
        // Swap left/right on both baselines: the only consistent camera is under the floor.
        let mirrored: [CourtCorner: CGPoint] = [.nbLd: c[.nbRd]!, .nbRd: c[.nbLd]!, .fbLd: c[.fbRd]!, .fbRd: c[.fbLd]!]
        XCTAssertThrowsError(try CourtCalibration.calibrate(corners: mirrored, imageSize: hd))
    }

    func testCornerAssignmentFromImagePositions() throws {
        let c = try corners(of: tripodCam)
        let assigned = try XCTUnwrap(CourtCorner.assign(tapped: [c[.fbLd]!, c[.nbRd]!, c[.nbLd]!, c[.fbRd]!]))
        for k in CourtCorner.allCases { XCTAssertEqual(assigned[k], c[k], "\(k)") }
    }

    func testNotEnoughPoints() {
        XCTAssertThrowsError(try CourtCalibration.calibrate(corners: [.nbLd: .zero], imageSize: hd)) { err in
            XCTAssertEqual(err as? CourtCalibration.CalibrationError, .notEnoughFloorPoints)
        }
    }

    func testScaledCalibrationMatchesResizedPixels() throws {
        let cam = broadcastCam
        let half = cam.scaled(to: .init(width: 960, height: 540))
        let p = SIMD3(2.0, -1.0, 1.2)
        let a = try XCTUnwrap(cam.project(p)), b = try XCTUnwrap(half.project(p))
        XCTAssertEqual(b.x, a.x / 2, accuracy: 1e-9)
        XCTAssertEqual(b.y, a.y / 2, accuracy: 1e-9)
    }
}
