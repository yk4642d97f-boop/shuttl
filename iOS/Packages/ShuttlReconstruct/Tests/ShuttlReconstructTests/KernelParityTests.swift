import XCTest
import ShuttlCore
import simd
@testable import ShuttlReconstruct

/// Solver-path parity: the kernels must reproduce the numba ones to round-off so the LM
/// walks the same path (same hypotheses, same iteration counts, same picks).
final class KernelParityTests: XCTestCase {
    static let k: JSONValue = try! fixture("kernels")

    func testPropagateWithSensitivities() throws {
        for (ci, c) in (Self.k["propagate"]!.array!).enumerated() {
            let p0 = SIMD3(row: c["p0"]!.doubles), v0 = SIMD3(row: c["v0"]!.doubles)
            let L = c["L"]!.d, times = c["times"]!.doubles, ncol = c["ncol"]!.int
            let n = times.count
            var P = [SIMD3<Double>](repeating: .zero, count: n), V = P
            var S = [Double](repeating: 0, count: n * 21)
            let ws = ODEWorkspace()
            times.withUnsafeBufferPointer { tb in
                P.withUnsafeMutableBufferPointer { pb in V.withUnsafeMutableBufferPointer { vb in
                    S.withUnsafeMutableBufferPointer { sb in
                        ws.propagate(p0, v0, L, tb.baseAddress!, n, ncol, pb.baseAddress!, vb.baseAddress!,
                                     ncol > 1 ? sb.baseAddress! : nil)
                    } } }
            }
            assertClose(P.flatMap(\.row), c["P"]!.flat, tol: 1e-13, "P case \(ci)")
            assertClose(V.flatMap(\.row), c["V"]!.flat, tol: 1e-13, "V case \(ci)")
            if ncol > 1 {
                let pyS = c["S"]!.array!                            // (n, 3, ncol-1)
                for i in 0..<n {
                    for r in 0..<3 {
                        let row = pyS[i].array![r].doubles
                        let mine = (0..<(ncol - 1)).map { S[i * 21 + $0 * 3 + r] }
                        assertClose(mine, row, tol: 1e-12, "S case \(ci) i \(i) r \(r)")
                    }
                }
            }
        }
    }

    func testGroundTimeAndShoot() throws {
        for c in Self.k["ground_time"]!.array! {
            let tg = groundTime(SIMD3(row: c["p0"]!.doubles), SIMD3(row: c["v0"]!.doubles), c["L"]!.d, tMax: 8, zGround: 0)
            assertClose(tg, c["tg"]!.d, tol: 1e-12, "ground time")
        }
        for c in Self.k["shoot"]!.array! {
            let v = shoot(SIMD3(row: c["p"]!.doubles), SIMD3(row: c["q"]!.doubles), c["dt"]!.d, c["L"]!.d, ODEWorkspace())
            if c["v"]!.isNull { XCTAssertNil(v) } else {
                XCTAssertNotNil(v)
                if let v { assertClose(v.row, c["v"]!.doubles, tol: 1e-9, "shoot") }
            }
        }
        let c = Self.k["v0_linear"]!
        let v = v0Linear(SIMD3(row: c["p0"]!.doubles), SIMD3(row: c["o"]!.doubles), c["d"]!.v3, c["tr"]!.doubles, c["L"]!.d)
        assertClose(v.row, c["v"]!.doubles, tol: 1e-9, "v0_linear")
    }

    func testOutlierMask() throws {
        for c in Self.k["outlier_mask"]!.array! {
            XCTAssertEqual(outlierMask(c["t"]!.doubles, c["uv"]!.v2), c["bad"]!.bools)
        }
    }

    func testResidualsHypothesesAndLMPath() throws {
        for (ci, c) in (Self.k["problems"]!.array!).enumerated() {
            let t = c["t"]!.doubles, uv = c["uv"]!.v2, w = c["w"]!.doubles
            let cam = c["camera"]!.camera, pr = c["prior"]!.prior
            let fd = c["fit_drag"]!.bool!
            let prob = Problem(t: t, uv: uv, w: w, camera: cam, prior: pr, fitDrag: fd)
            let msg = "problem \(ci) fit_drag \(fd)"
            let hyps = c["hyps"]!.rows
            let mine = hypotheses(prob, t, uv)
            XCTAssertEqual(mine.count, hyps.count, "n hyps \(msg)")
            for (a, b) in zip(mine, hyps) { assertClose(a, b, tol: 1e-8, "hyp \(msg)") }
            let (r, J) = prob.residuals(hyps[0], jac: true)
            assertClose(r, c["r"]!.doubles, tol: 1e-9, "r \(msg)")
            assertClose(J!, c["J"]!.flat, tol: 1e-8, "J \(msg)")
            XCTAssertEqual(plausible(prob, hyps[0]), c["plausible0"]!.bool!, msg)
            // coarse LM from every hypothesis: same iteration count, same end point
            for (h, py) in zip(hyps, c["coarse"]!.array!) {
                let f = lm(prob, h, maxIter: 6, tol: 1e-4)
                XCTAssertEqual(f.nIter, py["n_iter"]!.int, "coarse n_iter \(msg)")
                assertClose(f.th, py["th"]!.doubles, tol: 1e-6, "coarse th \(msg)")
                assertClose(f.cost, py["cost"]!.d, tol: 1e-6, "coarse cost \(msg)")
            }
            let f0 = lm(prob, hyps[0], maxIter: 40)
            XCTAssertEqual(f0.nIter, c["fine0"]!["n_iter"]!.int, "fine n_iter \(msg)")
            assertClose(f0.th, c["fine0"]!["th"]!.doubles, tol: 1e-6, "fine th \(msg)")
            if let py = c["solve"], !py.isNull {
                let s = try XCTUnwrap(solve(prob, t, uv, initTheta: nil), msg)
                assertClose(s.th, py["th"]!.doubles, tol: 1e-6, "solve th \(msg)")
                assertClose(s.cost, py["cost"]!.d, tol: 1e-6, "solve cost \(msg)")
            }
            if let ch = c["contact_hyps"] {
                let pc = Problem(t: t, uv: uv, w: w, camera: cam, prior: pr, fitDrag: fd, tRef: pr.startT)
                let mh = hypothesesContact(pc, t, uv)
                XCTAssertEqual(mh.count, ch.rows.count, "n contact hyps \(msg)")
                for (a, b) in zip(mh, ch.rows) { assertClose(a, b, tol: 1e-8, "contact hyp \(msg)") }
                if let first = mh.first, let pyr = c["contact_r"], !pyr.isNull {
                    let (rc, Jc) = pc.residuals(first, jac: true)
                    assertClose(rc, pyr.doubles, tol: 1e-8, "contact r \(msg)")
                    assertClose(Jc!, c["contact_J"]!.flat, tol: 1e-7, "contact J \(msg)")
                }
            }
        }
    }
}
