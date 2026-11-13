import XCTest
import ShuttlCore
import simd
@testable import ShuttlReconstruct

/// fit_segment parity on the test-suite generator's flights (tests/test_reconstruct3d.py),
/// plus the Python test assertions evaluated on the Swift results.
final class FlightParityTests: XCTestCase {
    static let flights: [JSONValue] = try! fixture("flights").array!

    struct Row {
        var tag: String, shot: String, half: String
        var ok: Bool
        var land = Double.nan, speed = Double.nan, height = Double.nan, sigma = Double.nan
        var margin: Double, trueCall: String, call: String
        var ms: Double
    }

    /// Fits every exported flight once (shared by the tests below).
    static let results: [(JSONValue, ReconstructedSegment?, Double)] = flights.map { c in
        let t0 = DispatchTime.now().uptimeNanoseconds
        let seg = Reconstruct3D.fitSegment(t: c["t"]!.doubles, uv: c["uv"]!.v2, w: nil, camera: c["camera"]!.camera,
                                           prior: c["prior"]!.prior)
        return (c, seg, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }

    static func rows(_ tag: String) -> [Row] {
        results.filter { $0.0["tag"]!.str == tag }.map { c, seg, ms in
            let tr = c["truth"]!
            let land = tr["land_xy"]!.doubles
            let m = CourtModel.inOutMargin(x: land[0], y: land[1], doubles: false)
            var r = Row(tag: tag, shot: c["shot"]!.str!, half: c["half"]!.str!, ok: seg != nil, margin: m,
                        trueCall: m >= 0 ? "in" : "out", call: "unsure", ms: ms)
            if let seg {
                let vHit = SIMD3(row: tr["v_hit"]!.doubles)
                r.speed = seg.segment.speed0Kmh / (simd_length(vHit) * 3.6) - 1
                r.height = seg.segment.p0.z - tr["p_hit"]!.doubles[2]
                if let l = seg.landingXY {
                    r.land = simd_length(l - SIMD2(land[0], land[1]))
                    r.sigma = Reconstruct3D.landingSigma(seg) ?? .nan
                }
                r.call = Reconstruct3D.callInOut(seg, doubles: false).call.rawValue
            }
            return r
        }
    }

    func testFitSegmentParity() throws {
        var worst = 0.0, worstLand = 0.0, n = 0
        var landDiffs: [Double] = []
        for (i, (c, seg, _)) in Self.results.enumerated() {
            let msg = "flight \(i) \(c["tag"]!.str!) \(c["shot"]!.str!)/\(c["half"]!.str!)"
            let py = c["seg"]!
            if py.isNull { XCTAssertNil(seg, msg); continue }
            guard let seg else { XCTFail("Swift nil, Python fitted: \(msg)"); continue }
            let (e, l) = compareSegment(seg, py, tol: 1e-6, landTol: 1e-3, msg)
            worst = max(worst, e); worstLand = max(worstLand, l); landDiffs.append(l)
            n += 1
            assertClose(seg.thetaRef, py["theta_ref"]!.doubles, tol: 1e-6, "theta_ref \(msg)")
            assertClose(seg.covTheta, py["cov_theta"]!.flat, tol: 1e-4, "cov_theta \(msg)")
            // the in/out call (and its margin / σ) of the landing
            let pc = c["call"]!, call = Reconstruct3D.callInOut(seg, doubles: false)
            XCTAssertEqual(call.call.rawValue, pc["call"]!.str, "call \(msg)")
            if let m = pc["margin_m"]?.optD { assertClose(call.marginM!, m, tol: 1e-5, "margin \(msg)") }
            if let s = pc["sigma_m"]?.optD { assertClose(call.sigmaM!, s, tol: 1e-3, "sigma \(msg)") }
        }
        print("flight parity: \(n) segments, max p0/v0 rel err \(worst), max landing diff \(worstLand * 1000) mm")
    }

    /// test_accuracy_targets + test_landing_sigma_is_calibrated on the Swift fits.
    func testAccuracyTargets() {
        let rows = Self.rows("bench64")
        XCTAssertEqual(rows.count, 64)
        let land = rows.map(\.land), fails = land.filter { !$0.isFinite }.count
        XCTAssertLessThanOrEqual(fails, 1)
        XCTAssertLessThan(Num.nanmedian(land), 0.10)
        XCTAssertLessThan(Num.nanpercentile(land, 90), 0.30)
        XCTAssertLessThan(Num.nanmedian(rows.map { abs($0.speed) }), 0.05)
        XCTAssertLessThan(Num.nanmedian(rows.map { abs($0.height) }), 0.15)
        let clear = rows.filter { abs($0.margin) > 0.15 }
        let decided = clear.filter { $0.call != "unsure" }
        XCTAssertGreaterThanOrEqual(Double(decided.filter { $0.call == $0.trueCall }.count) / Double(decided.count), 0.98)
        XCTAssertLessThan(Double(clear.filter { $0.call == "unsure" }.count) / Double(clear.count), 0.15)
        let fin = rows.filter { $0.land.isFinite && $0.sigma.isFinite }
        XCTAssertGreaterThanOrEqual(Double(fin.filter { $0.land < 3 * $0.sigma }.count) / Double(fin.count), 0.75)
        XCTAssert(rows.allSatisfy { !$0.sigma.isFinite || $0.sigma > 0 })
        print(String(format: "bench64 (Swift): land median %.1f cm p90 %.1f cm, speed median %.2f %%, fails %d",
                     100 * Num.nanmedian(land), 100 * Num.nanpercentile(land, 90),
                     100 * Num.nanmedian(rows.map { abs($0.speed) }), fails))
    }

    /// test_feet_prior_helps_far_half.
    func testFeetPriorHelpsFarHalf() {
        let base = Self.rows("base24").filter { $0.half == "far" }
        let feet = Self.rows("feet24").filter { $0.half == "far" }
        let e0 = Num.nanmedian(base.map { abs($0.speed) }), e1 = Num.nanmedian(feet.map { abs($0.speed) })
        let l0 = Num.nanmedian(base.map(\.land)), l1 = Num.nanmedian(feet.map(\.land))
        XCTAssert(e1 <= e0 * 1.1 + 0.005 && l1 <= l0 * 1.2 + 0.01, "\(e0) \(e1) \(l0) \(l1)")
    }

    /// test_noise_free_flight_is_recovered.
    func testNoiseFreeFlightIsRecovered() throws {
        for (c, seg, _) in Self.results where c["tag"]!.str == "noise_free" {
            let s = try XCTUnwrap(seg), tr = c["truth"]!
            let land = try XCTUnwrap(s.landingXY)
            XCTAssertLessThan(simd_length(land - SIMD2(row: tr["land_xy"]!.doubles)), 0.05)
            XCTAssertLessThan(abs(s.segment.speed0Kmh / (simd_length(SIMD3(row: tr["v_hit"]!.doubles)) * 3.6) - 1), 0.02)
            XCTAssertLessThan(abs(s.t0 - tr["t_hit"]!.d), 1e-9)
            XCTAssertEqual(s.endReason, .floor)
            XCTAssert(s.segment.rmsePx < 0.5 && s.segment.confidence > 0 && s.segment.confidence <= 1)
        }
    }

    /// test_lens_distortion.
    func testLensDistortion() throws {
        let (c, seg, _) = try XCTUnwrap(Self.results.first { $0.0["tag"]!.str == "distortion" })
        let land = try XCTUnwrap(seg?.landingXY)
        XCTAssertLessThan(simd_length(land - SIMD2(row: c["truth"]!["land_xy"]!.doubles)), 0.15)
    }

    /// test_speed_profile_and_initial_speed (+ parity of the profile).
    func testSpeedProfile() throws {
        let (c, seg, _) = try XCTUnwrap(Self.results.first { $0.0["tag"]!.str == "speed_profile" })
        let s = try XCTUnwrap(seg)
        let (ts, kmh) = Reconstruct3D.speedProfile(s)
        XCTAssertLessThan(abs(ts[0] - s.t0), 1e-9)
        XCTAssertLessThanOrEqual(ts[ts.count - 1], s.segment.landingT! + 1e-6)
        XCTAssertLessThan(abs(kmh[0] - Reconstruct3D.initialSpeedKmh(s.segment)), 1e-6)
        XCTAssert(zip(kmh.dropFirst(), kmh).allSatisfy { $0 < $1 })
        XCTAssertLessThan(kmh[kmh.count - 1], 0.5 * kmh[0])
        XCTAssertLessThan(abs(Reconstruct3D.contactSpeedKmh(s, tContact: s.t0) - s.segment.speed0Kmh), 1e-6)
        let py = c["speed_profile"]!
        XCTAssertEqual(ts.count, py["t"]!.doubles.count)
        assertClose(kmh, py["kmh"]!.doubles, tol: 1e-6, "speed profile")
    }

    /// test_net_fault_is_detected (single segment seen until it hits the net).
    func testNetFaultSingleSegment() throws {
        for (_, seg, _) in Self.results.filter({ $0.0["tag"]!.str == "net_fault_single" }) {
            let s = try XCTUnwrap(seg)
            XCTAssertEqual(s.endReason, .net)
            XCTAssertNil(s.landingXY)
            XCTAssertLessThan(try XCTUnwrap(s.segment.netCrossingHeight), CourtModel.netHeightCentre)
        }
    }

    func testTiming() {
        let ms = Self.results.map(\.2)
        let py = Self.flights.map { $0["py_ms"]!.d }
        print(String(format: "fit_segment wall ms: Swift median %.2f p90 %.2f | Python CPU ms median %.2f p90 %.2f",
                     Num.median(ms), Num.percentile(ms, 90), Num.median(py), Num.percentile(py, 90)))
    }
}
