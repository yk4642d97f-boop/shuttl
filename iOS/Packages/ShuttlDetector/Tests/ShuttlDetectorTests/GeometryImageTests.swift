import CoreVideo
import XCTest
import ShuttlCore
@testable import ShuttlDetector

final class GeometryImageTests: XCTestCase {

    func testGeometryMatchesPython() throws {
        for c in try fixture("geometry").array ?? [] {
            let g = FrameGeometry(width: c["w"]!.int, height: c["h"]!.int, corner: c["corner"]!.bool!)
            let tag = "\(g.width)x\(g.height) corner=\(g.corner)"
            XCTAssertEqual(g.scale, c["scale"]!.double!, tag)
            XCTAssertEqual([g.contentSize.width, g.contentSize.height], c["content"]!.ints, tag)
            let pad = c["pad"]!.doubles, ax = c["axis"]!.doubles
            XCTAssertEqual(g.pad.x, pad[0], tag); XCTAssertEqual(g.pad.y, pad[1], tag)
            let a = g.axis
            for (u, v) in zip([a.sx, a.sy, a.ox, a.oy], ax) { assertClose(u, v, tol: 1e-12, tag) }
            let px = c["px"]!.doubles, py = c["py"]!.doubles
            for i in px.indices {
                let n = g.toNet(x: px[i], y: py[i])
                assertClose(n.x, c["net_x"]!.doubles[i], tol: 1e-12, tag); assertClose(n.y, c["net_y"]!.doubles[i], tol: 1e-12, tag)
                let f = g.toFull(x: px[i] * 0.4, y: py[i] * 0.4)
                assertClose(f.x, c["full_x"]!.doubles[i], tol: 1e-12, tag); assertClose(f.y, c["full_y"]!.doubles[i], tol: 1e-12, tag)
                let q = g.peakToFull(x: px[i] * 0.4, y: py[i] * 0.4, peakOffset: -0.2)
                assertClose(q.x, c["peak_x"]!.doubles[i], tol: 1e-12, tag); assertClose(q.y, c["peak_y"]!.doubles[i], tol: 1e-12, tag)
                let r = g.toFull(x: n.x, y: n.y)
                XCTAssertEqual(r.x, px[i], accuracy: 1e-9); XCTAssertEqual(r.y, py[i], accuracy: 1e-9)
            }
        }
    }

    /// Mismatching rows between Swift and Python per-row checksums.
    private func rowMismatch(_ canvas: NetImage, _ c: JSONValue) -> (rows: Int, maxSumDiff: Int) {
        let (s, ws) = FormulaImage.rowChecksums(canvas.pixels, width: canvas.width, height: canvas.height)
        let es = c["sum"]!.ints, ew = c["wsum"]!.ints
        var bad = 0, worst = 0
        for y in s.indices where s[y] != es[y] || ws[y] != ew[y] { bad += 1; worst = max(worst, abs(s[y] - es[y])) }
        return (bad, worst)
    }

    func testWarpAndResizeMatchOpenCV() throws {
        let fx = try fixture("images")
        let canvas = NetImage()
        var report: [String] = []
        for (mode, cases) in [("warp", fx["warp"]!), ("area", fx["area"]!), ("linear", fx["linear"]!)] {
            for c in cases.array ?? [] {
                let w = c["w"]!.int, h = c["h"]!.int
                #if DEBUG
                if w * h > 1_000_000 { continue }     // formula images are slow to build unoptimised; release checks all
                #endif
                let img = FormulaImage.make(c["kind"]!.string!, w, h, seed: c["seed"]!.int)
                let g = FrameGeometry(width: w, height: h, corner: mode == "warp")
                canvas.clear()
                PixelView.withRGB(img) { g.warp($0, into: canvas, interpolation: mode == "linear" ? .linear : .area) }
                let (bad, worst) = rowMismatch(canvas, c)
                report.append("\(mode) \(w)x\(h) \(c["kind"]!.string!): rows≠ \(bad) maxΔsum \(worst)")
                let enlarging = g.scale > 1
                switch mode {
                case "warp":
                    XCTAssertEqual(bad, 0, "warpAffine must be bit-exact (\(w)x\(h))")
                case "area" where !enlarging:
                    // cv2's float box filter: bit-exact on landscape sizes (every video size the
                    // detector sees in practice); a single ±1 row on some portrait sizes.
                    if w >= h { XCTAssertEqual(bad, 0, "area \(w)x\(h)") }
                    XCTAssertLessThanOrEqual(bad, 2, "area \(w)x\(h)")
                    XCTAssertLessThanOrEqual(worst, 2, "area \(w)x\(h)")
                case "linear" where !enlarging:
                    XCTAssertEqual(bad, 0, "linear \(w)x\(h)")
                default:
                    XCTAssertLessThanOrEqual(worst, 3 * g.contentSize.width, "\(mode) \(w)x\(h)")
                }
            }
        }
        print("[images]\n" + report.joined(separator: "\n"))
    }

    func testThumbnailAndGray() throws {
        let fx = try fixture("images")
        let src = FormulaImage.mixed(512, 288, seed: 3)
        let canvas = NetImage()
        src.data.withUnsafeBufferPointer { canvas.pixels.update(from: $0.baseAddress!, count: canvas.count) }
        XCTAssertEqual(CVExact.grayThumbnail(canvas).map(Int.init), fx["thumb"]!["gray"]!.ints)
        let g = fx["gray"]!
        let img = FormulaImage.hash(g["w"]!.int, g["h"]!.int, seed: g["seed"]!.int)
        let gray = stride(from: 0, to: img.data.count, by: 3).map {
            Int(CVExact.gray(r: img.data[$0], g: img.data[$0 + 1], b: img.data[$0 + 2]))
        }
        XCTAssertEqual(gray, g["gray"]!.ints)
    }

    /// Letterbox cost per frame from a 32BGRA pixel buffer (what VideoSource delivers).
    func testLetterboxSpeed() throws {
        for (w, h) in [(1920, 1080), (1280, 720)] {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &pb)
            let buf = try XCTUnwrap(pb)
            CVPixelBufferLockBaseAddress(buf, [])
            let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
            let rb = CVPixelBufferGetBytesPerRow(buf)
            let img = FormulaImage.mixed(w, h)
            for y in 0..<h { for x in 0..<w { for c in 0..<3 { base[y * rb + x * 4 + 2 - c] = img.data[(y * w + x) * 3 + c] }
                base[y * rb + x * 4 + 3] = 255 } }
            CVPixelBufferUnlockBaseAddress(buf, [])
            let canvas = NetImage()
            for corner in [false, true] {
                let g = FrameGeometry(width: w, height: h, corner: corner)
                let interp: ResizeInterpolation = corner ? .linear : .area
                try PixelView.withBGRA(buf) { g.warp($0, into: canvas, interpolation: interp) }
                // BGRA source must give exactly the RGB-source result.
                let ref = NetImage()
                PixelView.withRGB(img) { g.warp($0, into: ref, interpolation: interp) }
                XCTAssertTrue(memcmp(ref.pixels, canvas.pixels, ref.count) == 0, "BGRA vs RGB \(w)x\(h) corner=\(corner)")
                var ms = Double.infinity          // best of 8 blocks: the Mac is shared by many jobs
                for _ in 0..<8 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    for _ in 0..<8 { try PixelView.withBGRA(buf) { g.warp($0, into: canvas, interpolation: interp) } }
                    ms = min(ms, (CFAbsoluteTimeGetCurrent() - t0) / 8 * 1e3)
                }
                print(String(format: "[letterbox] %dx%d %@: %.2f ms/frame", w, h, corner ? "warpAffine (WASB)" : "INTER_AREA (TrackNetV3)", ms))
            }
        }
    }
}
