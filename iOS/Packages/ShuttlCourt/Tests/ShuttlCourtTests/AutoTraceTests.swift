import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// Exact-input parity of the automatic path, stage by stage, against Python's trace of
/// `_calibrate_image` on the same median images (images.json, court.py section "images").
final class AutoTraceTests: XCTestCase {
    let config = CourtCalibrationConfig.default

    func testFeaturesSegmentsLines() throws {
        for c in try ImageFixtures.cases() {
            let tr = c.trace
            let feat = Features(c.image, config: config)
            let fx = tr["features"]!
            XCTAssertEqual(feat.mask.data.filter { $0 > 0 }.count, fx["mask_n"]!.int, "\(c.name) mask")
            XCTAssertEqual(feat.thin.data.filter { $0 > 0 }.count, fx["thin_n"]!.int, "\(c.name) thin")
            let segs = LineExtraction.segments(feat, maxSegments: config.maxSegments)
            let want = tr["segments"]!.rows
            XCTAssertEqual(segs.count, want.count, "\(c.name) n segments")
            for (s, w) in zip(segs, want) {
                XCTAssertEqual([s.x, s.y, s.z, s.w], w, "\(c.name) segment")
            }
            let lines = LineExtraction.merge(segs, D: feat.D)
            let wl = tr["lines"]!.array!
            XCTAssertEqual(lines.count, wl.count, "\(c.name) n lines")
            for (ln, w) in zip(lines, wl) {
                XCTAssertEqual(simd_length(ln.l - w["l"]!.vec3), 0, accuracy: 1e-9, "\(c.name) line l")
                XCTAssertEqual(simd_distance(ln.p0, w["p0"]!.vec2), 0, accuracy: 1e-7, "\(c.name) p0")
                XCTAssertEqual(ln.weight, w["weight"]!.double!, accuracy: 1e-9, "\(c.name) weight")
            }
        }
    }
}
extension AutoTraceTests {
    /// Python's merged lines (exact above) → Swift families. VP compared up to sign.
    static func pythonLines(_ tr: JSONValue) -> [ImageLine] {
        tr["lines"]!.array!.map { w in
            var ln = ImageLine(pts: [w["p0"]!.vec2, w["p1"]!.vec2], wts: [1, 1], weight: w["weight"]!.double!)
            ln.l = w["l"]!.vec3; ln.p0 = w["p0"]!.vec2; ln.p1 = w["p1"]!.vec2
            return ln
        }
    }

    func testFamilies() throws {
        for c in try ImageFixtures.cases() {
            let tr = c.trace
            let fams = VanishingPoints.families(Self.pythonLines(tr))
            let want = tr["families"]!.array!
            XCTAssertEqual(fams.count, want.count, "\(c.name) n families")
            for (f, w) in zip(fams, want) {
                XCTAssertEqual(f.members, w["members"]!.ints, "\(c.name) members")
                let wv = w["v"]!.vec3
                let d = min(simd_length(f.v - wv), simd_length(f.v + wv))
                XCTAssertLessThan(d, 1e-9, "\(c.name) VP")
            }
        }
    }
}
extension AutoTraceTests {
    /// Python's families (with numpy's VP sign) over Python's lines, >= 2 members.
    static func pythonFamilies(_ tr: JSONValue) -> [VPFamily] {
        tr["families"]!.array!.map { VPFamily(v: $0["v"]!.vec3, members: $0["members"]!.ints) }
            .filter { $0.members.count >= 2 }
    }

    /// Hypothesis counts per family pair and the top-50 scores (float32, exact).
    func testHypotheses() throws {
        for c in try ImageFixtures.cases() {
            let tr = c.trace
            let feat = Features(c.image, config: config)
            let (hs, perPair) = Hypotheses.all(feat, lines: Self.pythonLines(tr), families: Self.pythonFamilies(tr))
            XCTAssertEqual(perPair, tr["n_hyp_per_pair"]!.ints, "\(c.name) n_hyp per pair")
            XCTAssertEqual(hs.H.count, tr["n_hyp"]!.int, "\(c.name) n_hyp")
            let top = NumpyCompat.argsortStable(hs.score.map { -$0 }).prefix(50).map { Double(hs.score[$0]) }
            XCTAssertEqual(Array(top), tr["top_scores"]!.doubles, "\(c.name) top scores")
        }
    }
}
extension AutoTraceTests {
    /// Chosen H, quick-refined candidates and the final accept/reject + camera, from
    /// Python's exact lines/families (so numpy's VP sign is used).
    func testChosenCandidatesCamera() throws {
        var worstH = 0.0, worstCand = 0.0, worstCam = 0.0, worstF = 0.0
        for c in try ImageFixtures.cases() {
            let tr = c.trace
            let feat = Features(c.image, config: config)
            let (hs, _) = Hypotheses.all(feat, lines: Self.pythonLines(tr), families: Self.pythonFamilies(tr))
            let (H, ref, ls, chosen) = AutoCalibration.choose(feat, hs)
            let want = tr["chosen"]!.array!
            XCTAssertEqual(chosen.count, want.count, "\(c.name) n chosen")
            for (i, w) in zip(chosen, want) {
                let dh = homographyDiff(AutoCalibration.simd(H[i]), w["H"]!.mat3)
                worstH = max(worstH, dh)
                XCTAssertLessThan(dh, 1e-9, "\(c.name) chosen H")
                XCTAssertEqual(ref[i], w["ref"]!.vec2, "\(c.name) ref")
                XCTAssertEqual(Double(ls[i]), w["ls"]!.double!, "\(c.name) ls")
            }
            let mt = Matcher(feat: feat, config: config)
            let cands = AutoCalibration.candidates(mt, size: c.size, H: H, ref: ref, chosen: chosen)
            let wc = tr["cands"]!.array!
            XCTAssertEqual(cands.count, wc.count, "\(c.name) n cands")
            for (cd, w) in zip(cands, wc) {
                let a = keypointAgreement(try XCTUnwrap(w["camera"]!.cameraModel), cd.camera, margin: 0)
                worstCand = max(worstCand, a.max)
                let ws = w["stats"]!
                XCTAssertEqual(cd.stats.nT, ws["n_t"]!.int, "\(c.name) cand n_t")
                XCTAssertEqual(cd.stats.nL, ws["n_l"]!.int, "\(c.name) cand n_l")
                XCTAssertEqual(cd.stats.nAbsent, ws["n_absent"]!.int, "\(c.name) cand n_absent")
                XCTAssertEqual(cd.explained, w["explained"]!.double!, accuracy: 1e-6 * cd.explained, "\(c.name) explained")
            }
            var d = CourtCalibrationDiagnostics()
            let cam = AutoCalibration.decide(mt, cands, diag: &d)
            let wantCam = c.auto["camera"]!.cameraModel
            XCTAssertEqual(cam == nil, wantCam == nil, "\(c.name) accept/reject")
            let wd = c.auto["diag"]!
            XCTAssertEqual(d.nValid, wd["n_valid"]!.int, "\(c.name) n_valid")
            XCTAssertEqual(d.margin ?? -1, wd["margin"]!.double!, accuracy: 1e-9, "\(c.name) margin")
            if let cam, let wantCam {
                let a = keypointAgreement(wantCam, cam)
                worstCam = max(worstCam, a.max)
                worstF = max(worstF, abs(cam.focal / wantCam.focal - 1))
                XCTAssertLessThan(a.max, 1e-6, "\(c.name) camera keypoints px")
                XCTAssertEqual(cam.confidence, wantCam.confidence, accuracy: 1e-9, "\(c.name) confidence")
            }
        }
        print("PARITY exact-input: chosen H rel \(worstH), cand kp px \(worstCand), camera kp px \(worstCam), focal rel \(worstF)")
    }

    /// Whole `_calibrate_image` on Swift's own lines/families (own VP sign convention).
    func testCalibrateImageEndToEnd() throws {
        for c in try ImageFixtures.cases() {
            var d = CourtCalibrationDiagnostics()
            let cam = AutoCalibration.calibrateImage(c.image, size: c.size, config: config, diag: &d)
            let wantCam = c.auto["camera"]!.cameraModel
            let wd = c.auto["diag"]!
            XCTAssertEqual(d.nHyp, wd["n_hyp"]!.int, "\(c.name) n_hyp")
            XCTAssertEqual(d.nLines, wd["n_lines"]!.int, "\(c.name) n_lines")
            XCTAssertEqual(cam == nil, wantCam == nil, "\(c.name) accept/reject")
            if let r = wd["reason"]?.string { XCTAssertEqual(d.reason.rawValue, r, "\(c.name) reason") }
            if let cam, let wantCam {
                let a = keypointAgreement(wantCam, cam)
                print("PARITY e2e \(c.name): kp max \(a.max) px, focal rel \(abs(cam.focal / wantCam.focal - 1)), conf \(cam.confidence) vs \(wantCam.confidence)")
                // 1e-6 px with numpy's VP sign (testChosenCandidatesCamera). Swift's own sign
                // convention can enumerate the 180°-twin hypothesis set, whose refinement stops
                // (scipy ftol) at a point up to ~6e-4 px away (match1) — the same camera.
                XCTAssertLessThan(a.max, 1e-3, "\(c.name) camera keypoints px")
                XCTAssertEqual(cam.confidence, wantCam.confidence, accuracy: 1e-4, "\(c.name) confidence")
            }
        }
    }
}
