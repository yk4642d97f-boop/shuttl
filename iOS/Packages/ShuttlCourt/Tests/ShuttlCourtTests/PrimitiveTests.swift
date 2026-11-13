import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// Bit-level checks of the cv2 / numpy building blocks against the backend's cv2 4.11.
final class PrimitiveTests: XCTestCase {
    nonisolated(unsafe) static var fx: JSONValue?
    func prim() throws -> JSONValue {
        if let f = Self.fx { return f }
        let f = try fixture("primitives"); Self.fx = f; return f
    }

    func testRodrigues() throws {
        for c in try XCTUnwrap(prim()["rodrigues"]?.array) {
            let r = c["rvec"]!.doubles
            let R = Rodrigues.matrix(SIMD3(r[0], r[1], r[2]))
            let want = c["R"]!.rows
            for i in 0..<3 { for j in 0..<3 { XCTAssertEqual(mget(R, i, j), want[i][j], accuracy: 1e-14) } }
            let back = Rodrigues.vector(simd_double3x3(rowMajor: want))
            let wb = c["back"]!.doubles
            for i in 0..<3 { XCTAssertEqual(back[i], wb[i], accuracy: 1e-9) }
        }
    }

    func testFindHomography() throws {
        for c in try XCTUnwrap(prim()["find_homography"]?.array) {
            let src = c["src"]!.rows.map { SIMD2($0[0], $0[1]) }
            let dst = c["dst"]!.rows.map { SIMD2($0[0], $0[1]) }
            let H = try XCTUnwrap(Homography.find(src, dst))
            let want = c["H"]!.mat3
            // compare as mappings (sub-mpx on the inputs)
            for p in src {
                let a = H * SIMD3(p.x, p.y, 1), b = want * SIMD3(p.x, p.y, 1)
                let d = simd_distance(SIMD2(a.x / a.z, a.y / a.z), SIMD2(b.x / b.z, b.y / b.z))
                XCTAssertLessThan(d, src.count == 4 ? 1e-6 : 2e-3, "n=\(src.count)")
            }
        }
    }

    func testGaussianBlurBitExact() throws {
        let g = try XCTUnwrap(prim()["gauss"])
        XCTAssertEqual(ImageOps.gaussianKernel(n: 7, sigma: 0.8), g["kernel"]!.floats)
        let w = g["w"]!.int!, h = g["h"]!.int!
        let out = ImageOps.gaussianBlur(PlaneF(w: w, h: h, data: g["src"]!.floats), sigma: 0.8)
        let want = g["dst"]!.floats
        var nDiff = 0, maxD: Float = 0
        for i in 0..<(w * h) where out.data[i] != want[i] { nDiff += 1; maxD = max(maxD, abs(out.data[i] - want[i])) }
        XCTAssertEqual(nDiff, 0, "blur differs in \(nDiff) px, max \(maxD)")
    }

    func testDistanceTransformBitExact() throws {
        let d = try XCTUnwrap(prim()["dist"])
        let w = d["w"]!.int!, h = d["h"]!.int!
        let m = d["mask"]!.ints
        let out = ImageOps.distanceTransform3x3({ m[$0] == 0 }, w: w, h: h)
        let inn = ImageOps.distanceTransform3x3({ m[$0] != 0 }, w: w, h: h)
        XCTAssertEqual(out.data, d["dt_out"]!.floats)
        XCTAssertEqual(inn.data, d["dt_in"]!.floats)
    }

    func testRemapBitExact() throws {
        let r = try XCTUnwrap(prim()["remap"])
        let w = r["w"]!.int!, h = r["h"]!.int!
        let src = r["src"]!.floats, X = r["X"]!.floats, Y = r["Y"]!.floats, want = r["dst"]!.floats
        var nDiff = 0, maxD: Float = 0
        src.withUnsafeBufferPointer { s in
            for i in 0..<X.count {
                let v = ImageOps.remapSample(s, w: w, h: h, X[i], Y[i])
                if v != want[i] { nDiff += 1; maxD = max(maxD, abs(v - want[i])) }
            }
        }
        XCTAssertEqual(nDiff, 0, "remap differs in \(nDiff) samples, max \(maxD)")
    }

    func testHoughLinesPExact() throws {
        let hf = try XCTUnwrap(prim()["hough"])
        let w = hf["w"]!.int!, h = hf["h"]!.int!
        let img = PlaneU8(w: w, h: h, data: hf["thin"]!.ints.map { UInt8($0 * 255) })
        let D = Double(max(w, h))
        let segs = Hough.linesP(img, rho: 1, theta: Double.pi / 720, threshold: max(12, Int(0.012 * D)),
                                minLineLength: 0.03 * D, maxLineGap: 0.006 * D)
        let want = hf["segments"]!.rows.map { SIMD4<Int32>($0.map { Int32($0) }) }
        XCTAssertEqual(segs, want)
    }

    func testEigh2Signs() throws {
        for c in try XCTUnwrap(prim()["eigh2"]?.array) {
            let a = c["a"]!.rows, ev = c["evecs"]!.rows
            let r = Lapack.eigh2(a[0][0], a[0][1], a[1][1])
            XCTAssertEqual(r.v1.x, ev[0][1], accuracy: 1e-12); XCTAssertEqual(r.v1.y, ev[1][1], accuracy: 1e-12)
            XCTAssertEqual(r.v0.x, ev[0][0], accuracy: 1e-12); XCTAssertEqual(r.v0.y, ev[1][0], accuracy: 1e-12)
        }
    }

    func testMaskFromStrength() throws {
        let m = try XCTUnwrap(prim()["mask"])
        let w = m["w"]!.int!, h = m["h"]!.int!
        let out = ImageOps.maskFromStrength(PlaneF(w: w, h: h, data: m["strength"]!.floats),
                                            minLen: m["min_len"]!.double!, minContrast: 18)
        XCTAssertEqual(out.data.map { $0 > 0 ? 1 : 0 }, m["mask"]!.ints)
    }

    func testRidgeScales() throws {
        guard case .object(let o) = try XCTUnwrap(prim()["scales"]) else { return XCTFail() }
        for (k, v) in o { XCTAssertEqual(ImageOps.ridgeScales(D: Double(k)!), v.ints, "D=\(k)") }
    }
}
