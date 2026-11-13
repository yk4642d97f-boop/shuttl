import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// Single-frame fallback parity (court.py section "fallback"): per-frame results with the
/// dominated-winner veto, the ">= 2 different frames agree within 3 px" rule and the final
/// reason, on the same median / frame images as Python.
final class FallbackTests: XCTestCase {
    let config = CourtCalibrationConfig.default

    func testFallbackMatchesPython() throws {
        for c in try XCTUnwrap(fixture("fallback")["cases"]?.array) {
            let name = c["name"]!.string!
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            let truth = try XCTUnwrap(c["true"]?.cameraModel)
            var median = CourtCalibrationDiagnostics()
            if c["forced_median_fail"]?.bool == true {
                median.reason = .evidenceGate                   // Python forces the median to fail
            } else {
                let med = try ImageFixtures.planes(c["median_planes"]!.string!, size: size)
                XCTAssertNil(AutoCalibration.calibrateImage(med, size: size, config: config, diag: &median), name)
            }
            XCTAssertEqual(median.reason.rawValue, c["median"]!["reason"]!.string!, "\(name) median reason")
            let frames = c["frames"]!.array!
            var byIndex: [Int: JSONValue] = [:]
            for f in frames { byIndex[f["i"]!.int!] = f }
            let res = AutoFallback.finish(median: median, n: c["n_frames"]!.int!) { i in
                var d = CourtCalibrationDiagnostics()
                d.source = "frame\(i)"
                guard let f = byIndex[i], let img = try? ImageFixtures.planes(f["planes"]!.string!, size: size) else {
                    XCTFail("\(name): Swift tried frame \(i), which Python did not export"); return (nil, d)
                }
                let cam = AutoCalibration.calibrateImage(img, size: size, config: self.config, vetoDominated: true, diag: &d)
                XCTAssertEqual(d.reason.rawValue, f["reason"]!.string!, "\(name) frame \(i) reason")
                XCTAssertEqual(cam == nil, f["camera"]?.cameraModel == nil, "\(name) frame \(i) accept")
                if let cam, let w = f["camera"]?.cameraModel {
                    XCTAssertLessThan(keypointAgreement(w, cam).max, 1e-3, "\(name) frame \(i) camera px")
                }
                return (cam, d)
            }
            let want = c["final"]!
            XCTAssertEqual(res.diagnostics.reason.rawValue, want["reason"]!.string!, "\(name) final reason")
            XCTAssertEqual(res.camera == nil, want["camera"]?.cameraModel == nil, "\(name) final accept")
            XCTAssertEqual(res.diagnostics.nFrameAccepts, want["n_frame_accepts"]?.int, "\(name) n_frame_accepts")
            if let ft = want["frames_tried"]?.array { XCTAssertEqual(res.diagnostics.framesTried, ft.map { $0.int! }, name) }
            if let fa = want["frames_agree"]?.array { XCTAssertEqual(res.diagnostics.framesAgree, fa.map { $0.int! }, name) }
            if let cam = res.camera, let w = want["camera"]?.cameraModel {
                let a = keypointAgreement(w, cam)
                print("PARITY fallback \(name): kp max \(a.max) px, reason \(res.diagnostics.reason.rawValue)")
                XCTAssertLessThan(a.max, 1e-3, "\(name) camera px")
                XCTAssertLessThan(calibErrors(truth, cam)["world_max"]!, 0.05, "\(name) vs truth")
            }
        }
    }

    func testAgreementAndIndices() {
        XCTAssertEqual(AutoFallback.indices(24), [12, 6, 18, 3, 21])
        XCTAssertEqual(AutoFallback.indices(3), [1, 0, 2])
        XCTAssertEqual(AutoFallback.indices(5), [2, 1, 3, 0, 4])
        let a = CameraModel.lookAt(eye: SIMD3(-9.7, 0, 1.6), target: .zero, focalPx: 1371,
                                   imageSize: ImageSize(width: 1920, height: 1080))
        var b = a
        b.t += a.R * SIMD3(0.02, 0, 0)
        XCTAssertEqual(AutoFallback.agreementPx(a, a), 0)
        XCTAssertGreaterThan(AutoFallback.agreementPx(a, b), 0)
        XCTAssertEqual(AutoFallback.agreementPx(a, b), AutoFallback.agreementPx(b, a))
        XCTAssertEqual(CalibrationFailure.inconsistentFrames.rawValue, "inconsistent_frames")
    }
}
