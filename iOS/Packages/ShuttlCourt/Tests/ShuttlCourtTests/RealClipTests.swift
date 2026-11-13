import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// calibrateFromFrames on the 29 TrackNetV2 Test clips vs Python (court.py section "real").
/// Slow: runs only with SHUTTL_COURT_REAL=1 and when the Test split is present. Frames come
/// from AVFoundation (ShuttlCore.VideoSource), Python's from FFmpeg: decoded pixels differ by
/// unbiased noise (mean |d| ≈ 2.6 grey levels), so cameras agree closely but not bit-exactly.
final class RealClipTests: XCTestCase {
    func testRealClips() async throws {
        guard ProcessInfo.processInfo.environment["SHUTTL_COURT_REAL"] == "1" else {
            throw XCTSkip("set SHUTTL_COURT_REAL=1 to run the 29-clip real test")
        }
        let split = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test")
        guard FileManager.default.fileExists(atPath: split.path) else { throw XCTSkip("Test split missing") }
        guard fixtureURL("real.json") != nil else { throw XCTSkip("run court.py real first") }
        let clips = try XCTUnwrap(fixture("real")["clips"]?.array)
        var agree = 0, kpMax: [Double] = [], kpMed: [Double] = [], fRel: [Double] = [], times: [Double] = []
        for c in clips {
            let rel = c["clip"]!.string!
            let src = try await VideoSource.open(repoRoot.appendingPathComponent(rel))
            let sampled = try src.sampleFrames(24) { try ImageU8.rgb(from: $0) }
            XCTAssertEqual(sampled.map(\.index), c["indices"]!.ints, "\(rel) sampled indices")
            let size = ImageSize(width: src.meta.width, height: src.meta.height)
            let t0 = Date()
            let res = CourtCalibration.calibrateFromFramesDetailed(sampled.map(\.value), size: size)
            let dt = Date().timeIntervalSince(t0)
            times.append(dt)
            let want = c["camera"]!.cameraModel
            let d = res.diagnostics, pd = c["diag"]!
            let tag = rel.replacingOccurrences(of: "backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/", with: "")
            if (res.camera == nil) == (want == nil) { agree += 1 }
            if let r = pd["reason"]?.string { XCTAssertEqual(d.reason.rawValue, r, "\(rel) reason") }
            var line = "REAL \(tag): swift \(res.camera == nil ? "reject" : "accept") py \(want == nil ? "reject" : "accept")"
            line += String(format: " %.2fs", dt)
            if let cam = res.camera, let want {
                let a = keypointAgreement(want, cam)
                kpMax.append(a.max); kpMed.append(a.median)
                fRel.append(abs(cam.focal / want.focal - 1))
                line += String(format: " kp max %.3f med %.3f px, focal rel %.2e, k1 %.4f/%.4f", a.max, a.median,
                               fRel.last!, cam.dist[0], want.dist[0])
            }
            line += Self.gates("swift", d.stats, d.margin, d.confidence, D: Double(max(size.width, size.height)))
            line += Self.gatesPy(pd, D: Double(max(size.width, size.height)))
            print(line)
        }
        func med(_ v: [Double]) -> Double { v.isEmpty ? .nan : Num.median(v) }
        print(String(format: "REAL SUMMARY accept/reject agree %d/%d; both accepted %d: kp max-of-max %.3f px, median-of-max %.3f px, median-of-median %.3f px; focal rel max %.2e med %.2e; calibrate s: median %.2f max %.2f",
                     agree, clips.count, kpMax.count, kpMax.max() ?? .nan, med(kpMax), med(kpMed),
                     fRel.max() ?? .nan, med(fRel), med(times), times.max() ?? .nan))
        XCTAssertGreaterThanOrEqual(Double(agree), 0.9 * Double(clips.count), "accept/reject agreement")
        XCTAssertLessThan(med(kpMed), 1.0, "median keypoint agreement (px)")
    }

    /// Margins to the acceptance gates (positive = passes).
    static func gates(_ who: String, _ s: FitStats?, _ margin: Double?, _ conf: Double?, D: Double) -> String {
        guard let s else { return " | \(who): no candidate" }
        return String(format: " | %@: nT %d nL %d absent %d expl %+.3f margin %+.3f rmse %+.3f conf %+.3f", who,
                      s.nT, s.nL, s.nAbsent, (s.explainedFrac ?? .nan) - 0.55, (margin ?? .nan) - 0.05,
                      max(1, 0.0007 * D) - s.rmse, (conf ?? .nan) - 0.35)
    }

    static func gatesPy(_ d: JSONValue, D: Double) -> String {
        func v(_ k: String) -> Double { d[k]?.double ?? .nan }
        return String(format: " | py: nT %d nL %d absent %d expl %+.3f margin %+.3f rmse %+.3f conf %+.3f",
                      d["n_t"]?.int ?? -1, d["n_l"]?.int ?? -1, d["n_absent"]?.int ?? -1, v("explained_frac") - 0.55,
                      v("margin") - 0.05, max(1, 0.0007 * D) - v("rmse"), v("confidence") - 0.35)
    }
}
