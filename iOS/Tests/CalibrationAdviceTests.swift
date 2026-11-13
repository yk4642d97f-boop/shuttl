import XCTest
import ShuttlCore
@testable import Shuttl

final class CalibrationAdviceTests: XCTestCase {
    private let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 10, timestamps: (0..<10).map { Double($0) / 60 })

    private func analysis(_ auto: [String: JSONValue]?, camera: CameraModel? = nil) -> MatchAnalysis {
        var a = MatchAnalysis(meta: meta, camera: camera)
        if let auto { a.diagnostics["calibration_auto"] = .object(auto) }
        return a
    }

    func testEveryKnownReasonHasItsOwnPlainSentence() {
        let reasons = ["no_frames", "no_line_families", "no_hypotheses", "no_candidates", "evidence_gate",
                       "focal_mismatch", "low_confidence", "ambiguous", "low_coverage", "implausible_focal",
                       "high_rmse", "inconsistent_frames"]
        for r in reasons {
            let m = CalibrationAdvice.message(reason: r, transverse: nil, longitudinal: nil, absent: nil)
            XCTAssertFalse(m.isEmpty, r)
            XCTAssertLessThanOrEqual(m.count, 60, "fits the card: \(r)")
            XCTAssertFalse(m.contains("_"), "no internal codes leak: \(r)")
        }
        XCTAssertTrue(CalibrationAdvice.message(reason: "something_new", transverse: nil, longitudinal: nil, absent: nil)
                        .contains("1.6 米"), "an unknown reason still gives the setup advice")
    }

    func testEvidenceGateSaysWhichLinesWereMissing() {
        XCTAssertTrue(CalibrationAdvice.message(reason: "evidence_gate", transverse: 2, longitudinal: 5, absent: 0)
                        .contains("底线和发球线"))
        XCTAssertTrue(CalibrationAdvice.message(reason: "evidence_gate", transverse: 5, longitudinal: 3, absent: 0)
                        .contains("边线"))
        XCTAssertTrue(CalibrationAdvice.message(reason: "evidence_gate", transverse: 5, longitudinal: 5, absent: 2)
                        .contains("挡着"))
    }

    func testReadsTheAnalysisDiagnostics() {
        let a = analysis(["reason": .string("evidence_gate"), "n_t": .number(2), "n_l": .number(5), "n_absent": .number(0)])
        XCTAssertEqual(CalibrationAdvice.message(for: a),
                       CalibrationAdvice.message(reason: "evidence_gate", transverse: 2, longitudinal: 5, absent: 0))
    }

    func testNothingToSayWhenCalibratedOrManual() {
        let cam = CameraModel.lookAt(eye: [-11, 0, 1.6], target: [0, 0, 0], focalPx: 1370,
                                     imageSize: ImageSize(width: 1920, height: 1080))
        XCTAssertNil(CalibrationAdvice.message(for: analysis(["reason": .string("ok")], camera: cam)))
        XCTAssertNil(CalibrationAdvice.message(for: analysis(nil)), "manual corners: no automatic attempt to explain")
    }

    func testFarPrecisionNoteOnlyForAFoundButFlatCourt() {
        let cam = CameraModel.lookAt(eye: [-11, 0, 0.8], target: [0, 0, 0], focalPx: 1370,
                                     imageSize: ImageSize(width: 1920, height: 1080))
        let flat = analysis(["reason": .string("ok"), "far_m_per_px": .number(0.31)], camera: cam)
        XCTAssertEqual(CalibrationAdvice.precisionNote(for: flat), CalibrationAdvice.farPrecisionNote)
        XCTAssertNil(CalibrationAdvice.message(for: flat), "the court was found: no failure sentence")
        // The recommended setup (sweep2: 0.117-0.164) and the threshold itself stay silent.
        for v in [0.117, 0.164, 0.22] {
            XCTAssertNil(CalibrationAdvice.precisionNote(for: analysis(["reason": .string("ok"), "far_m_per_px": .number(v)], camera: cam)), "\(v)")
        }
        XCTAssertNil(CalibrationAdvice.precisionNote(for: analysis(["reason": .string("ok")], camera: cam)), "older analyses: unknown")
        XCTAssertNil(CalibrationAdvice.precisionNote(for: analysis(["reason": .string("evidence_gate"), "far_m_per_px": .number(0.4)])),
                     "no camera: the failure sentence applies instead")
        XCTAssertNil(CalibrationAdvice.precisionNote(farMPerPx: .infinity))
        XCTAssertLessThanOrEqual(CalibrationAdvice.farPrecisionNote.count, 60, "fits the card")
    }
}
