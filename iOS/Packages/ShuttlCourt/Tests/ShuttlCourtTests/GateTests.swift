import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// Evidence-gate parity (court.py section "gates"): the n_t == 2 exception (venue sweep
/// cases 1 / 3, 0.8 m high 0.5 m behind), the optional focal hint and the reason strings,
/// against Python on the same median images.
final class GateTests: XCTestCase {
    let config = CourtCalibrationConfig.default

    struct GateCase { let name: String; let size: ImageSize; let image: ImageU8; let json: JSONValue }

    func cases() throws -> [GateCase] {
        try XCTUnwrap(fixture("gates")["cases"]?.array).map { c in
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            return GateCase(name: c["name"]!.string!, size: size,
                            image: try ImageFixtures.planes(c["planes"]!.string!, size: size), json: c)
        }
    }

    /// `_calibrate_image` on Python's exact lines / families (numpy VP sign).
    func calibrate(_ c: GateCase, focalHint: Double?) -> (CameraModel?, CourtCalibrationDiagnostics) {
        var d = CourtCalibrationDiagnostics()
        let tr = c.json["auto"]!["trace"]!
        let feat = Features(c.image, config: config)
        let fams = AutoTraceTests.pythonFamilies(tr)
        guard fams.count >= 2 else { d.reason = .noLineFamilies; return (nil, d) }
        let (hs, _) = Hypotheses.all(feat, lines: AutoTraceTests.pythonLines(tr), families: fams)
        guard !hs.H.isEmpty else { d.reason = .noHypotheses; return (nil, d) }
        let (H, ref, _, chosen) = AutoCalibration.choose(feat, hs)
        let mt = Matcher(feat: feat, config: config)
        let cands = AutoCalibration.candidates(mt, size: c.size, H: H, ref: ref, chosen: chosen)
        guard !cands.isEmpty else { d.reason = .noCandidates; return (nil, d) }
        for (cd, w) in zip(cands, tr["cands"]?.array ?? []) {
            XCTAssertEqual(cd.stats.tStraddle, w["stats"]!["t_straddle"]!.bool, "\(c.name) cand t_straddle")
            XCTAssertEqual(cd.stats.nT, w["stats"]!["n_t"]!.int, "\(c.name) cand n_t")
        }
        return (AutoCalibration.decide(mt, cands, focalHint: focalHint, diag: &d), d)
    }

    func check(_ label: String, _ cam: CameraModel?, _ reason: CalibrationFailure, want: JSONValue?,
               wantReason: String, tol: Double) {
        XCTAssertEqual(reason.rawValue, wantReason, "\(label) reason")
        let w = want?.cameraModel
        XCTAssertEqual(cam == nil, w == nil, "\(label) accept/reject")
        if let cam, let w {
            let a = keypointAgreement(w, cam)
            print("PARITY gates \(label): kp max \(a.max) px, reason \(reason.rawValue)")
            XCTAssertLessThan(a.max, tol, "\(label) camera keypoints px")
        }
    }

    func testGatesExactInput() throws {
        for c in try cases() {
            let (cam, d) = calibrate(c, focalHint: nil)
            check("\(c.name) exact", cam, d.reason, want: c.json["auto"]!["camera"],
                  wantReason: c.json["auto"]!["diag"]!["reason"]!.string!, tol: 1e-6)
            if let wd = c.json["auto"]?["diag"], let m = wd["margin"]?.double {
                // margin on net evidence (2e − e/explained_frac): the camera's ~3e-7 px parity
                // difference reaches ~2e-9 here (gate threshold 0.05, sweep3 margin 0.216)
                XCTAssertEqual(d.margin ?? -1, m, accuracy: 1e-8, "\(c.name) margin")
                XCTAssertEqual(d.nValid, wd["n_valid"]!.int, "\(c.name) n_valid")
            }
            for h in c.json["hinted"]?.array ?? [] {
                let f = h["focal_hint"]!.double!
                let (hc, hd) = calibrate(c, focalHint: f)
                check("\(c.name) hint \(Int(f))", hc, hd.reason, want: h["camera"],
                      wantReason: h["reason"]!.string!, tol: 1e-6)
            }
        }
    }

    /// Whole calibrateImage on Swift's own lines/families (own VP sign → twin tolerance).
    func testGatesEndToEnd() throws {
        for c in try cases() {
            var d = CourtCalibrationDiagnostics()
            let cam = AutoCalibration.calibrateImage(c.image, size: c.size, config: config, diag: &d)
            check("\(c.name) e2e", cam, d.reason, want: c.json["auto"]!["camera"],
                  wantReason: c.json["auto"]!["diag"]!["reason"]!.string!, tol: 1e-3)
            if let t = c.json["true"]?.cameraModel, let cam {
                let e = calibErrors(t, cam)
                XCTAssertLessThan(e["world_max"]!, 0.15, "\(c.name) floor error vs truth")
            }
        }
    }

    /// CalibrationFailure raw values == court_calibration.REASONS (read from the Python source).
    func testReasonStringsMatchPython() throws {
        let src = try String(contentsOf: repoRoot.appendingPathComponent("backend/app/services/tracking/court_calibration.py"),
                             encoding: .utf8)
        let start = try XCTUnwrap(src.range(of: "REASONS = ("))
        let end = try XCTUnwrap(src.range(of: ")", range: start.upperBound..<src.endIndex))
        let py = src[start.upperBound..<end.lowerBound].split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\"")) }
            .filter { !$0.isEmpty }
        XCTAssertEqual(CalibrationFailure.allCases.map(\.rawValue), py)
        XCTAssertEqual(CourtCalibration.calibrateFromFramesDetailed([], size: ImageSize(width: 4, height: 4))
            .diagnostics.reason, .noFrames)
    }
}
