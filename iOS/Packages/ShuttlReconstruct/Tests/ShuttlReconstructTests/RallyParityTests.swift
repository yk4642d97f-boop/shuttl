import XCTest
import ShuttlCore
import simd
@testable import ShuttlReconstruct

/// Runs reconstructSegments on an exported rally case and compares every segment with Python.
/// The fixtures keep only player observations within ±8 frames of a hit frame: _Players.feet
/// only looks ±6 frames around hit frames, so the result is identical (the parity check
/// itself would expose any difference).
@discardableResult
func checkRally(_ c: JSONValue, _ name: String, tol: Double = 1e-6, landTol: Double = 1e-3,
                valueCheck: ((Int) -> Bool)? = nil,
                file: StaticString = #filePath, line: UInt = #line)
    -> (segs: [ReconstructedSegment], worst: Double, worstLand: Double, ms: Double, errs: [Double]) {
    let players = c["players"]!.players
    let t0 = DispatchTime.now().uptimeNanoseconds
    let segs = Reconstruct3D.reconstructSegments(track: c["track"]!.track, hits: c["hits"]!.hits,
                                                 camera: c["camera"]!.camera, meta: c["meta"]!.meta,
                                                 players: players.isEmpty ? nil : players)
    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    let py = c["segs"]!.array!
    XCTAssertEqual(segs.count, py.count, "segment count \(name)", file: file, line: line)
    var worst = 0.0, worstLand = 0.0
    var errs: [Double] = []
    for (i, (s, p)) in zip(segs, py).enumerated() {
        let msg = "\(name) seg \(i)"
        let values = valueCheck?(i) ?? true
        XCTAssertEqual(s.hitFrame, p["hit_frame"]?.optInt, "hit_frame \(msg)", file: file, line: line)
        if values, let h = p["hit_t_refined"]?.optD { assertClose(s.hitTRefined ?? .nan, h, tol: 1e-6, "hit_t_refined \(msg)", file: file, line: line) }
        XCTAssertEqual(s.segment.hitterFeetXY == nil, p["hitter_feet_xy"]!.isNull, "feet presence \(msg)", file: file, line: line)
        XCTAssertEqual(s.restXY == nil, p["rest_xy"]!.isNull, "rest presence \(msg)", file: file, line: line)
        let (e, l) = compareSegment(s, p, tol: tol, landTol: landTol, checkValues: values, msg, file: file, line: line)
        errs.append(e)
        if !values { continue }
        worst = max(worst, e); worstLand = max(worstLand, l)
        // the accessor ShuttlRally uses (rally._landing_sigma → getattr landing_sigma_m, else r3.landing_sigma)
        if let pyl = p["landing_sigma_fn"]?.optD {
            assertClose(Reconstruct3D.landingSigma(s) ?? .nan, pyl, tol: 1e-3, "landing_sigma() \(msg)", file: file, line: line)
        } else {
            XCTAssertNil(Reconstruct3D.landingSigma(s), "landing_sigma() \(msg)", file: file, line: line)
        }
    }
    return (segs, worst, worstLand, ms, errs)
}

final class RallyParityTests: XCTestCase {
    static let fx: JSONValue = try! fixture("rallies")

    func testReconstructSegmentsParity() {
        var worst = 0.0, worstLand = 0.0, n = 0, ms = 0.0, pyms = 0.0, frames = 0
        for c in Self.fx["rallies"]!.array! {
            let r = checkRally(c, c["tag"]!.str!)
            worst = max(worst, r.worst); worstLand = max(worstLand, r.worstLand); n += r.segs.count
            ms += r.ms; pyms += c["py_ms"]!.d; frames += c["track"]!["t"]!.doubles.count
        }
        print(String(format: "rally parity: %d segments, max p0/v0 rel err %.2e, max landing diff %.4f mm; " +
                     "Swift %.0f ms vs Python CPU %.0f ms for %d frames", n, worst, worstLand * 1000, ms, pyms, frames))
    }

    /// test_rally_reconstruction on the Swift results.
    func testRallyReconstruction() throws {
        var nSeg = 0, nShot = 0
        var spErr: [Double] = [], tcErr: [Double] = [], landErr: [Double] = []
        for c in Self.fx["rallies"]!.array! where c["tag"]!.str!.hasPrefix("rally_") {
            let segs = checkRally(c, c["tag"]!.str!).segs
            let truth = c["truth"]!.array!
            nShot += truth.count
            nSeg += segs.count
            let frames = truth.map { $0["frame"]!.int }
            for sg in segs {
                let k = try XCTUnwrap(frames.firstIndex(of: sg.hitFrame!))
                let s = truth[k]
                spErr.append(abs(sg.segment.speed0Kmh / (simd_length(SIMD3(row: s["v"]!.doubles)) * 3.6) - 1))
                tcErr.append(abs(sg.t0 - s["t"]!.d))
                if k < truth.count - 1 {
                    XCTAssert(sg.endReason == .hit && sg.landingXY == nil)
                } else {
                    XCTAssertEqual(sg.endReason, .floor)
                    let l = try XCTUnwrap(sg.landingXY)
                    landErr.append(simd_length(l - SIMD2(row: s["land_xy"]!.doubles)))
                }
                XCTAssert(zip(sg.segment.frames.dropFirst(), sg.segment.frames).allSatisfy { $0 > $1 })
                XCTAssertEqual(sg.segment.frames.count, sg.segment.positions.count)
            }
        }
        XCTAssertGreaterThanOrEqual(nSeg, nShot - 1)
        XCTAssertLessThan(Num.median(spErr), 0.08)
        XCTAssertLessThan(Num.median(tcErr), 0.02)
        XCTAssertLessThan(landErr.max()!, 0.4)
    }

    /// test_net_fault_is_detected: whole rally piece incl. the slide down the net.
    func testNetFaultRally() {
        for c in Self.fx["rallies"]!.array! where c["tag"]!.str!.hasPrefix("net_fault_") {
            let segs = checkRally(c, c["tag"]!.str!).segs
            XCTAssertEqual(segs.count, 1)
            XCTAssertEqual(segs.first?.endReason, .net)
            XCTAssertNil(segs.first?.landingXY)
            XCTAssertEqual(Reconstruct3D.callInOut(segs.first, doubles: false).call, .unsure)
        }
    }

    /// test_degenerate_inputs.
    func testDegenerateInputs() throws {
        let c = try XCTUnwrap(Self.fx["rallies"]!.array!.first { $0["tag"]!.str == "degenerate_no_players" })
        let track = c["track"]!.track, cam = c["camera"]!.camera, meta = c["meta"]!.meta
        XCTAssertNil(Reconstruct3D.fitSegment(t: (0..<4).map { Double($0) / 30 }, uv: Array(repeating: SIMD2(500, 500), count: 4),
                                              camera: cam))
        XCTAssertNil(Reconstruct3D.fitSegment(t: (0..<12).map { Double($0) / 30 },
                                              uv: Array(repeating: SIMD2(.nan, .nan), count: 12), camera: cam))
        XCTAssert(Reconstruct3D.reconstructSegments(track: track, hits: [], camera: cam, meta: meta).isEmpty)
        XCTAssert(Reconstruct3D.reconstructSegments(track: track, hits: c["hits"]!.hits, camera: nil, meta: meta).isEmpty)
        let segs = checkRally(c, "degenerate").segs
        XCTAssertGreaterThanOrEqual(segs.count, 1)
        XCTAssert(segs.allSatisfy { $0.v0.isFinite })
    }

    /// test_on_sibling_simulator.
    func testOnSiblingSimulator() {
        var landErr: [Double] = [], spErr: [Double] = []
        var n = 0
        for c in Self.fx["rallies"]!.array! where c["tag"]!.str!.hasPrefix("sim_") {
            let segs = checkRally(c, c["tag"]!.str!).segs
            let truth = c["truth"]!.array!
            for sg in segs {
                guard let s = truth.first(where: { $0["frame"]!.int == sg.hitFrame }) else { continue }
                n += 1
                spErr.append(abs(sg.segment.speed0Kmh / s["speed0_kmh"]!.d - 1))
                if ["in", "out"].contains(s["outcome"]!.str!), let l = sg.landingXY {
                    landErr.append(simd_length(l - SIMD2(row: s["landing_xy"]!.doubles)))
                }
            }
        }
        XCTAssertGreaterThanOrEqual(n, 6)
        XCTAssertLessThan(Num.median(spErr), 0.08)
        XCTAssert(landErr.isEmpty || Num.median(landErr) < 0.3)
    }

    /// test_refine_hit_time (+ parity).
    func testRefineHitTime() {
        var errs: [Double] = []
        for c in Self.fx["refine"]!.array! {
            let (f, tc) = Reconstruct3D.refineHitTime(track: c["track"]!.track, approxFrame: c["approx_frame"]!.int,
                                                      camera: c["camera"]!.camera)
            XCTAssertEqual(f, c["frame"]!.int)
            assertClose(tc, c["t_c"]!.d, tol: 1e-6, "refine t_c")
            errs.append(abs(tc - c["true_t"]!.d))
        }
        XCTAssert(!errs.isEmpty && Num.median(errs) < 2.0 / 120)
    }

    /// test_call_in_out_and_unsure.
    func testCallInOutAndUnsure() {
        func seg(_ land: SIMD2<Double>?, _ p0: SIMD3<Double> = SIMD3(-4, 1, 2)) -> ShotSegment3D {
            ShotSegment3D(startFrame: 0, endFrame: 10, t0: 0, p0: p0, v0: SIMD3(20, 0, 5), rmsePx: 1,
                          frames: Array(0..<10), positions: Array(repeating: .zero, count: 10), landingXY: land,
                          landingT: land == nil ? nil : 1.0)
        }
        func cov(_ s: Double) -> [Double] { [s * s, 0, 0, s * s] }
        let edge = CourtModel.halfLength + CourtModel.lineWidth / 2
        var r = Reconstruct3D.callInOut(seg(SIMD2(edge - 0.30, 0)), landingCov: cov(0.03), doubles: false)
        XCTAssert(r.call == .in && abs(r.marginM! - 0.30) < 1e-9 && r.pIn! > 0.99)
        r = Reconstruct3D.callInOut(seg(SIMD2(edge + 0.30, 0.5)), landingCov: cov(0.03), doubles: false)
        XCTAssert(r.call == .out && r.marginM! < 0)
        r = Reconstruct3D.callInOut(seg(SIMD2(edge - 0.03, 0)), landingCov: cov(0.04), doubles: false)
        XCTAssert(r.call == .unsure && r.pIn! > 0.4 && r.pIn! < 1.0)
        let y = CourtModel.singlesHalfWidth + 0.2
        XCTAssertEqual(Reconstruct3D.callInOut(seg(SIMD2(3, y)), landingCov: cov(0.02), doubles: false).call, .out)
        XCTAssertEqual(Reconstruct3D.callInOut(seg(SIMD2(3, y)), landingCov: cov(0.02), doubles: true).call, .in)
        XCTAssertGreaterThan(Reconstruct3D.callInOut(seg(SIMD2(3, CourtModel.singlesHalfWidth + 0.015)),
                                                     landingCov: cov(0.001), doubles: false).marginM!, 0)
        XCTAssertEqual(Reconstruct3D.callInOut(seg(nil), landingCov: nil, doubles: false).call, .unsure)
        XCTAssertEqual(Reconstruct3D.callInOut(nil as ReconstructedSegment?, doubles: true).call, .unsure)
    }

    /// test_serve_rules.
    func testServeRules() {
        XCTAssertEqual(Reconstruct3D.serveSideOf(SIMD2(-2.5, -1.0)), .right)
        XCTAssertEqual(Reconstruct3D.serveSideOf(SIMD2(-2.5, 1.0)), .left)
        XCTAssertEqual(Reconstruct3D.serveSideOf(SIMD2(2.5, 1.0)), .right)
        var b = Reconstruct3D.serveBox(serverHalf: .near, serveSide: .right, doubles: true)
        XCTAssert(b == (CourtModel.shortServiceX, CourtModel.doublesLongServiceX, 0.0, CourtModel.doublesHalfWidth))
        b = Reconstruct3D.serveBox(serverHalf: .far, serveSide: .right, doubles: false)
        XCTAssert(b == (-CourtModel.halfLength, -CourtModel.shortServiceX, -CourtModel.singlesHalfWidth, 0.0))
        func m(_ x: Double, _ y: Double, _ d: Bool) -> Double {
            Reconstruct3D.serveMargin(x: x, y: y, serverHalf: .near, serveSide: .right, doubles: d)
        }
        XCTAssertGreaterThan(m(4.0, 1.5, true), 0)
        XCTAssertLessThan(m(1.5, 1.5, true), 0)
        XCTAssertLessThan(m(4.0, -1.5, true), 0)
        XCTAssertLessThan(m(6.3, 1.5, true), 0)
        XCTAssertGreaterThan(m(6.3, 1.5, false), 0)
        XCTAssertLessThan(m(4.0, 2.8, false), 0)
        XCTAssertGreaterThan(m(4.0, 2.8, true), 0)
        XCTAssertGreaterThan(m(CourtModel.shortServiceX - 0.015, 1.0, true), 0)
        func seg(_ land: SIMD2<Double>, _ p0: SIMD3<Double>, feet: SIMD2<Double>? = nil) -> ShotSegment3D {
            ShotSegment3D(startFrame: 0, endFrame: 10, t0: 0, p0: p0, v0: SIMD3(20, 0, 5), rmsePx: 1,
                          frames: Array(0..<10), positions: Array(repeating: .zero, count: 10), landingXY: land,
                          landingT: 1.0, hitterFeetXY: feet)
        }
        let cv = [0.0004, 0, 0, 0.0004]
        func call(_ s: ShotSegment3D, _ d: Bool, side: Reconstruct3D.ServeSide? = nil) -> InOutCall.Call {
            Reconstruct3D.callInOut(s, landingCov: cv, doubles: d, isServe: true, serveSide: side).call
        }
        var s = seg(SIMD2(4.0, 1.5), SIMD3(-2.4, -1.0, 1.05))
        XCTAssertEqual(call(s, true), .in)
        XCTAssertEqual(call(s, true, side: .left), .out)
        s = seg(SIMD2(6.3, 1.5), SIMD3(-2.4, -1.0, 1.05))
        XCTAssertEqual(call(s, true), .out)
        XCTAssertEqual(call(s, false), .in)
        XCTAssertEqual(call(seg(SIMD2(4.0, 1.5), SIMD3(-2.4, 0.05, 1.05), feet: SIMD2(-2.7, -0.4)), true), .in)
        XCTAssertEqual(call(seg(SIMD2(4.0, 1.5), SIMD3(-2.4, 0.05, 1.05), feet: SIMD2(-2.7, 0.4)), true), .out)
        XCTAssertEqual(call(seg(SIMD2(4.0, 1.5), SIMD3(-2.4, 0.05, 1.05)), true), .unsure)
        XCTAssertEqual(call(seg(SIMD2(1.0, 1.5), SIMD3(-2.4, 0.05, 1.05)), true), .out)
    }
}
