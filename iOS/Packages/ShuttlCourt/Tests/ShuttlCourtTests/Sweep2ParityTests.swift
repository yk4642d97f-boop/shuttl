import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// Exact-input parity on the calib_sweep2 subset (court.py section "sweep2", >= 60 median
/// images covering every grid, hard-case kind and reject reason; TestData/sweep2, kept out
/// of the test bundle for size). Swift calibrateImage on the SAME median image must give
/// Python's reason and accept/reject, and the camera. Runs with SHUTTL_COURT_REAL=1.
final class Sweep2ParityTests: XCTestCase {
    let dir = repoRoot.appendingPathComponent("iOS/Packages/ShuttlCourt/TestData/sweep2")

    func planes(_ name: String, _ size: ImageSize) throws -> ImageU8 {
        let raw = try Data(contentsOf: dir.appendingPathComponent(name)) as NSData
        let data = try raw.decompressed(using: .zlib) as Data
        XCTAssertEqual(data.count, size.width * size.height * 2, name)
        return ImageU8(width: size.width, height: size.height, channels: 2, data: [UInt8](data))
    }

    func testSweep2MedianParity() throws {
        guard ProcessInfo.processInfo.environment["SHUTTL_COURT_REAL"] == "1" else {
            throw XCTSkip("set SHUTTL_COURT_REAL=1 to run the sweep2 parity set")
        }
        let man = dir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: man.path) else {
            throw XCTSkip("run `.venv/bin/python -m scripts.swift_fixtures.court sweep2` first")
        }
        let cases = try XCTUnwrap(try ShuttlJSON.load(JSONValue.self, from: man)["cases"]?.array)
        XCTAssertGreaterThanOrEqual(cases.count, 60)
        var agree = 0, reasonAgree = 0, kpMax = 0.0
        var flips: [String] = []
        for c in cases {
            let id = c["id"]!.int!
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            let img = try planes(c["planes"]!.string!, size)
            var d = CourtCalibrationDiagnostics()
            let cam = AutoCalibration.calibrateImage(img, size: size, config: .default, diag: &d)
            let want = c["median"]!
            let wcam = want["camera"]?.cameraModel
            let label = "#\(id) \(c["grid"]!.string!)/\(c["kind"]!.string!)"
            if (cam == nil) == (wcam == nil) { agree += 1 } else { flips.append("\(label) py \(want["reason"]!.string!) swift \(d.reason.rawValue) margin \(d.margin ?? -1)") }
            if d.reason.rawValue == want["reason"]!.string! { reasonAgree += 1 }
            if let cam, let wcam {
                let a = keypointAgreement(wcam, cam)
                kpMax = max(kpMax, a.max)
                if a.max > 1e-4 {
                    print("PARITY sweep2 \(label): kp \(a.max) px; f swift \(mget(cam.K, 0, 0)) py \(mget(wcam.K, 0, 0)); "
                          + "k1 swift \(cam.dist[0]) py \(wcam.dist[0]); score \(d.score ?? -1) margin \(d.margin ?? -1)")
                }
                // VP-sign twin: the same hypothesis enters refinement from a start differing
                // at ~1e-12 and stops (scipy ftol) at a slightly different point; with k1 fitted
                // the objective is flatter (R7 hard_lights_s23: 2.7e-3 px, focal rel 1e-6).
                let bound = wcam.dist[0] != 0 ? 1e-2 : 1e-3
                XCTAssertLessThan(a.max, bound, "\(label) camera px (VP-sign twin tolerance)")
            }
        }
        print("PARITY sweep2: \(cases.count) cases, accept/reject agree \(agree), reason agree \(reasonAgree), "
              + "kp max \(kpMax) px; flips \(flips)")
        XCTAssertEqual(agree, cases.count, "accept/reject flips: \(flips)")
        XCTAssertEqual(reasonAgree, cases.count, "reason mismatches")
    }
}
