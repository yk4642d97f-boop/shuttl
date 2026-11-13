import simd
import XCTest
@testable import ShuttlCore

/// Swift ↔ Python reference parity for the shared contracts.
final class ParityTests: XCTestCase {

    func testCameraMatchesPython() throws {
        for (i, c) in try XCTUnwrap(fixture("camera").array).enumerated() {
            let cam = try XCTUnwrap(c["camera"]).decode(CameraModel.self)
            let uv = cam.project(try XCTUnwrap(c["world"]).v3)
            for (a, b) in zip(uv, try XCTUnwrap(c["uv"]).v2) {
                assertClose(a.x, b.x, tol: 1e-7, "cam \(i) u"); assertClose(a.y, b.y, tol: 1e-7, "cam \(i) v")
            }
            let pix = try XCTUnwrap(c["pixels"]).v2
            let dirs = try XCTUnwrap(c["ray_dirs"]).v3
            for (p, d) in zip(pix, dirs) {
                let r = cam.ray(p)
                assertClose([r.direction.x, r.direction.y, r.direction.z], d.row, tol: 1e-9, "cam \(i) ray")
            }
            assertClose(cam.center.row, try XCTUnwrap(c["center"]).doubles, tol: 1e-9)
            for (p, f) in zip(pix, try XCTUnwrap(c["floor"]).v3) { assertClose(cam.pixelToFloor(p).row, f.row, tol: 1e-7, "floor") }
            for (p, f) in zip(pix, try XCTUnwrap(c["floor_z1"]).v3) { assertClose(cam.pixelToFloor(p, z: 1).row, f.row, tol: 1e-7, "floor z1") }
            let h = cam.floorHomography().rowMajor.flatMap { $0 }
            assertClose(h, try XCTUnwrap(c["floor_homography"]).rows.flatMap { $0 }, tol: 1e-9, "H")
        }
    }

    func testCameraJSONRoundTrip() throws {
        let c = try XCTUnwrap(fixture("camera").array?.last?["camera"])
        let cam = try c.decode(CameraModel.self)
        let again = try ShuttlJSON.decoder().decode(CameraModel.self, from: ShuttlJSON.encoder().encode(cam))
        XCTAssertEqual(cam.K, again.K); XCTAssertEqual(cam.R, again.R); XCTAssertEqual(cam.dist, again.dist)
        XCTAssertTrue(cam.isDistorted)
    }

    func testPhysicsMatchesPython() throws {
        for c in try XCTUnwrap(fixture("physics").array) {
            let p0 = SIMD3(row: try XCTUnwrap(c["p0"]).doubles), v0 = SIMD3(row: try XCTUnwrap(c["v0"]).doubles)
            let s = Physics.simulateState(p0: p0, v0: v0, times: try XCTUnwrap(c["times"]).doubles)
            assertClose(s.positions.flatMap(\.row), try XCTUnwrap(c["P"]).rows.flatMap { $0 }, tol: 1e-10)
            assertClose(s.velocities.flatMap(\.row), try XCTUnwrap(c["V"]).rows.flatMap { $0 }, tol: 1e-10)
            let n = Physics.simulateState(p0: p0, v0: v0, times: try XCTUnwrap(c["neg_times"]).doubles)
            assertClose(n.positions.flatMap(\.row), try XCTUnwrap(c["P_neg"]).rows.flatMap { $0 }, tol: 1e-10)
            let tg = Physics.timeToGround(p0: p0, v0: v0)
            if let want = c["t_ground"]?.double { assertClose(try XCTUnwrap(tg), want, tol: 1e-10) } else { XCTAssertNil(tg) }
        }
    }

    func testCourtMatchesPython() throws {
        let c = try fixture("court")
        XCTAssertEqual(CourtModel.keypointNames, (c["names"]?.array ?? []).compactMap(\.string))
        assertClose(CourtModel.keypointsArray().points.flatMap(\.row), try XCTUnwrap(c["points"]).rows.flatMap { $0 }, tol: 0)
        let lines = try XCTUnwrap(c["lines"])
        for l in CourtModel.lines {
            let want = try XCTUnwrap(lines[l.name]).rows
            XCTAssertEqual([l.a.row, l.b.row], want, l.name)
        }
        for m in try XCTUnwrap(c["margins"]?.array) {
            let x = m["x"]!.double!, y = m["y"]!.double!
            assertClose(CourtModel.inOutMargin(x: x, y: y), m["singles"]!.double!, tol: 1e-12)
            assertClose(CourtModel.inOutMargin(x: x, y: y, doubles: true), m["doubles"]!.double!, tol: 1e-12)
        }
    }

    func testNumericsMatchNumpy() throws {
        let n = try fixture("numerics")
        let xs = try XCTUnwrap(n["xs"]).doubles
        assertClose(Num.median(xs), n["median"]!.double!, tol: 1e-12)
        for (q, v) in try XCTUnwrap(n["percentiles"]).objectPairs { assertClose(Num.percentile(xs, Double(q)!), v, tol: 1e-12, "p\(q)") }
        assertClose(Num.std(xs), n["std"]!.double!, tol: 1e-12)
        assertClose(Num.std(xs, ddof: 1), n["std_ddof1"]!.double!, tol: 1e-12)
        assertClose(Num.interp(n["probe"]!.doubles, n["xp"]!.doubles, n["fp"]!.doubles), n["interp"]!.doubles, tol: 1e-12)

        let a = Matrix(n["lstsq_A"]!.rows)
        assertClose(try XCTUnwrap(LinAlg.lstsq(a, n["lstsq_b"]!.doubles)), n["lstsq_x"]!.doubles, tol: 1e-9, "lstsq")
        let sym = Matrix(n["sym"]!.rows)
        assertClose(LinAlg.eigvalsh(sym), n["eigvalsh"]!.doubles, tol: 1e-9, "eigvalsh")
        assertClose(try XCTUnwrap(LinAlg.solve(sym, Array(n["lstsq_b"]!.doubles.prefix(5)))), n["solve_x"]!.doubles, tol: 1e-9, "solve")

        let dlt = Matrix(n["dlt"]!.rows)
        let (_, s, _) = LinAlg.svd(dlt)
        assertClose(Array(s.prefix(8)), n["dlt_singular"]!.doubles, tol: 1e-9, "svd")
        let null = LinAlg.nullVector(dlt), want = n["dlt_null"]!.doubles
        let sign = zip(null, want).reduce(0) { $0 + $1.0 * $1.1 } < 0 ? -1.0 : 1.0
        assertClose(null.map { $0 * sign }, want, tol: 1e-8, "null vector")
        XCTAssertLessThan((dlt * null).map { $0 * $0 }.reduce(0, +).squareRoot(), 1e-9)

        assertClose(Num.convolveSame(n["convolve_a"]!.doubles, n["convolve_v"]!.doubles), n["convolve_same"]!.doubles, tol: 1e-12)
        assertClose(Num.convolveSame(n["convolve_a"]!.doubles, n["convolve_v4"]!.doubles), n["convolve_same4"]!.doubles, tol: 1e-12)
    }

    func testSymmetricEigenVectors() {
        let a = Matrix([[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 1]])
        let (vals, vecs) = LinAlg.symmetricEigen(a)
        for k in 0..<3 {
            let v = vecs.column(k)
            let av = a * v
            for i in 0..<3 { XCTAssertEqual(av[i], vals[k] * v[i], accuracy: 1e-10) }
        }
        XCTAssertEqual(vals, vals.sorted())
    }

    func testCholeskyAndInverse() throws {
        let a = Matrix([[4, 2, 0.6], [2, 5, 1], [0.6, 1, 3]])
        let l = try XCTUnwrap(LinAlg.cholesky(a))
        let back = l * l.transposed
        for (x, y) in zip(back.data, a.data) { XCTAssertEqual(x, y, accuracy: 1e-12) }
        let inv = try XCTUnwrap(LinAlg.inverse(a))
        let id = a * inv
        for i in 0..<3 { for j in 0..<3 { XCTAssertEqual(id[i, j], i == j ? 1 : 0, accuracy: 1e-12) } }
        XCTAssertNil(LinAlg.cholesky(Matrix([[1, 2], [2, 1]])))
        XCTAssertNil(LinAlg.solve(Matrix([[1, 2], [2, 4]]), [1, 2]))
    }

    func testLenientEnumsAndNaNRoundTrip() throws {
        let json = #"{"frame":3,"t":null,"xy_px":[1.5,null],"hitter":"somebody","confidence":0.5}"#
        let hit = try ShuttlJSON.decoder().decode(HitEvent.self, from: Data(json.utf8))
        XCTAssertEqual(hit.hitter, .unknown)
        XCTAssertTrue(hit.t.isNaN); XCTAssertTrue(hit.xyPx.y.isNaN); XCTAssertEqual(hit.xyPx.x, 1.5)
        let back = String(decoding: try ShuttlJSON.encoder().encode(hit), as: UTF8.self)
        XCTAssertTrue(back.contains(#""t":null"#), back)
        XCTAssertEqual(try ShuttlJSON.decoder().decode(RallyEndReason.self, from: Data(#""in""#.utf8)), .inCourt)
    }
}

extension JSONValue {
    var objectPairs: [(String, Double)] {
        if case .object(let o) = self { return o.compactMap { k, v in v.double.map { (k, $0) } } }
        return []
    }
}
