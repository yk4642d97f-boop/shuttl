import XCTest
import ShuttlCore
@testable import ShuttlDetector

let netW = detectorNetWidth, netH = detectorNetHeight

func gaussHM(_ cx: Double, _ cy: Double, _ sigma: Double, _ amp: Double = 1, w: Int = netW, h: Int = netH) -> [Float] {
    var hm = [Float](repeating: 0, count: w * h)
    for y in 0..<h { for x in 0..<w {
        let d = (Double(x) - cx) * (Double(x) - cx) + (Double(y) - cy) * (Double(y) - cy)
        hm[y * w + x] = Float(amp * exp(-d / (2 * sigma * sigma)))
    } }
    return hm
}

/// Deterministic generator for tests.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

final class DecodeTests: XCTestCase {

    /// Every fixture heatmap (synthetic + real torch fp32 outputs of both nets) decodes to the
    /// Python result: same count/order, positions within 1e-9 px, identical scores.
    func testDecodeMatchesPython() throws {
        var n = 0
        for c in try fixture("decode").array ?? [] {
            let w = c["w"]!.int, h = c["h"]!.int
            var hm = [Float](repeating: 0, count: w * h)
            for (i, v) in zip(c["idx"]!.ints, c["val"]!.doubles) { hm[i] = Float(v) }
            for run in c["runs"]!.array ?? [] {
                let dec = HeatmapDecoder(k: run["k"]!.int, minScore: run["min_score"]!.double!,
                                         nmsRadius: run["nms_radius"]?.int ?? 6,
                                         refine: PeakRefinement(rawValue: run["refine"]!.string!)!,
                                         refineRadius: run["refine_radius"]?.int ?? 4)
                let got = dec.decode(hm, width: w, height: h)
                let want = run["out"]!.rows
                let tag = "\(c["name"]!.string!) \(run["refine"]!.string!) k=\(dec.k) min=\(dec.minScore)"
                XCTAssertEqual(got.count, want.count, tag)
                for (g, e) in zip(got, want) {
                    XCTAssertEqual(g.x, e[0], accuracy: 1e-9, tag)
                    XCTAssertEqual(g.y, e[1], accuracy: 1e-9, tag)
                    XCTAssertEqual(g.score, e[2], tag)
                }
                n += 1
            }
        }
        XCTAssertGreaterThan(n, 150)
    }

    // Ports of test_detector.py decoding tests.

    func testDecodeSubpixelGaussian() {
        var rng = SplitMix64(seed: 42)
        for refine in [PeakRefinement.centroid, .quadratic] {
            for sigma in [1.5, 2.0, 2.5, 3.0] {
                var worst = 0.0
                for _ in 0..<8 {
                    let cx = Double.random(in: 20..<490, using: &rng), cy = Double.random(in: 20..<265, using: &rng)
                    let amp = Double.random(in: 0.4..<1.0, using: &rng)
                    let p = decodeHeatmap(gaussHM(cx, cy, sigma, amp), width: netW, height: netH, k: 1, refine: refine)
                    XCTAssertEqual(p.count, 1)
                    worst = max(worst, hypot(p[0].x - cx, p[0].y - cy))
                    // peak pixel is ≤ 0.71 px from the centre: score ≥ amp·exp(−0.5/(2σ²))
                    XCTAssert(amp * exp(-0.5 / (2 * sigma * sigma)) - 1e-6 <= p[0].score && p[0].score <= amp + 1e-6,
                              "score \(p[0].score) amp \(amp) sigma \(sigma)")
                }
                XCTAssertLessThan(worst, 0.2, "\(refine) sigma \(sigma)")
            }
        }
    }

    func testDecodeTopKSortedNMSAndThreshold() {
        let a = gaussHM(100.3, 50.7, 2.0, 0.9), b = gaussHM(300, 200, 2.0, 0.6), c = gaussHM(450, 30, 2.0, 0.05)
        let hm = zip(zip(a, b), c).map { $0.0 + $0.1 + $1 }
        let p = decodeHeatmap(hm, width: netW, height: netH, k: 3)
        XCTAssertEqual(p.count, 2)
        XCTAssertGreaterThan(p[0].score, p[1].score)
        XCTAssertLessThan(abs(p[0].x - 100.3), 0.2); XCTAssertLessThan(abs(p[1].y - 200.0), 0.2)
        XCTAssertTrue(decodeHeatmap([Float](repeating: 0, count: netW * netH), width: netW, height: netH).isEmpty)
    }

    func testPlateauGivesOneCentredPeak() {
        for refine in [PeakRefinement.centroid, .quadratic] {
            for r in [2, 3, 8] {        // binary disc (cv2.circle filled)
                var hm = [Float](repeating: 0, count: netW * netH)
                for y in 0..<netH { for x in 0..<netW where (x - 200) * (x - 200) + (y - 100) * (y - 100) <= r * r { hm[y * netW + x] = 1 } }
                let p = decodeHeatmap(hm, width: netW, height: netH, refine: refine)
                XCTAssertEqual(p.count, 1)
                XCTAssertLessThan(hypot(p[0].x - 200, p[0].y - 100), 0.1)
            }
            for amp in [3.0, 20.0] {    // sigmoid saturated at 1.0 in fp16
                let hm = gaussHM(200.3, 100.6, 2.5, amp).map { Float(Float16(min($0, 1))) }
                let p = decodeHeatmap(hm, width: netW, height: netH, refine: refine)
                XCTAssertEqual(p.count, 1)
                XCTAssertLessThan(hypot(p[0].x - 200.3, p[0].y - 100.6), 0.1, "\(refine) amp \(amp)")
            }
        }
    }

    func testDecodeSpeed() {
        var hm = gaussHM(100.3, 50.7, 2.2, 0.9)
        let b = gaussHM(300, 200, 1.8, 0.6)
        for i in hm.indices { hm[i] = max(hm[i], b[i]) }
        let dec = HeatmapDecoder()
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<200 { _ = dec.decode(hm, width: netW, height: netH) }
        let ms = (CFAbsoluteTimeGetCurrent() - t0) / 200 * 1e3
        print("[decode] \(String(format: "%.3f", ms)) ms / heatmap")
    }
}
