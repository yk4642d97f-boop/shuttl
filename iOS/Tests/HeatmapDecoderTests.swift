import XCTest
import CoreGraphics
@testable import Shuttl

final class HeatmapDecoderTests: XCTestCase {

    private struct LCG {
        var state: UInt64
        mutating func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
    }

    private func gaussianMap(width: Int, height: Int, blobs: [(x: Double, y: Double, sigma: Double, peak: Float)]) -> [Float] {
        var m = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var v: Float = 0
                for b in blobs {
                    let d2 = (Double(x) - b.x) * (Double(x) - b.x) + (Double(y) - b.y) * (Double(y) - b.y)
                    v = max(v, b.peak * Float(exp(-d2 / (2 * b.sigma * b.sigma))))
                }
                m[y * width + x] = v
            }
        }
        return m
    }

    private func decode(_ map: [Float], width: Int, height: Int, decoder: HeatmapDecoder) -> [HeatmapPeak] {
        var m = map
        return m.withUnsafeMutableBufferPointer { decoder.decode($0, width: width, height: height) }
    }

    func testQuadraticRefinementIsSubpixelOnGaussians() {
        var rng = LCG(state: 7)
        var worst = 0.0
        for sigma in [1.2, 2.0, 3.0, 4.5] {
            for _ in 0..<40 {
                let cx = 20 + 30 * rng.next(), cy = 15 + 20 * rng.next()
                let map = gaussianMap(width: 72, height: 54, blobs: [(cx, cy, sigma, 0.9)])
                let peaks = decode(map, width: 72, height: 54, decoder: HeatmapDecoder(refinement: .quadratic))
                XCTAssertEqual(peaks.count, 1)
                guard let p = peaks.first else { continue }
                worst = max(worst, hypot(p.x - cx, p.y - cy))
            }
        }
        XCTAssertLessThan(worst, 0.2, "worst sub-pixel error \(worst) px")
    }

    func testCentroidFallbackOnSaturatedPlateau() {
        // Sigmoid-saturated disk (what WASB-style heads emit): quadratic falls back to centroid.
        var rng = LCG(state: 11)
        var worst = 0.0
        for _ in 0..<40 {
            let cx = 20 + 30 * rng.next(), cy = 15 + 20 * rng.next()
            var map = [Float](repeating: 0, count: 72 * 54)
            for y in 0..<54 {
                for x in 0..<72 {
                    let d = hypot(Double(x) - cx, Double(y) - cy)
                    map[y * 72 + x] = Float(1 / (1 + exp(3 * (d - 2.5))))
                }
            }
            let p = decode(map, width: 72, height: 54, decoder: HeatmapDecoder(refinement: .quadratic)).first
            XCTAssertNotNil(p)
            if let p { worst = max(worst, hypot(p.x - cx, p.y - cy)) }
        }
        XCTAssertLessThan(worst, 0.35, "worst plateau error \(worst) px")
    }

    func testTopTwoPeaksWithNMSAndThreshold() {
        let blobs: [(x: Double, y: Double, sigma: Double, peak: Float)] = [
            (20.3, 30.6, 2.0, 0.6), (50.7, 12.2, 2.0, 0.95), (53.0, 14.0, 1.5, 0.8), (8, 8, 2.0, 0.2),
        ]
        let map = gaussianMap(width: 72, height: 54, blobs: blobs)
        let peaks = decode(map, width: 72, height: 54, decoder: HeatmapDecoder(minScore: 0.3, maxPeaks: 3, nmsRadius: 6))
        // The 0.8 blob sits inside the NMS box of the 0.95 one; the 0.2 blob is below threshold.
        XCTAssertEqual(peaks.count, 2)
        XCTAssertEqual(Double(peaks[0].score), 0.95, accuracy: 0.02)
        XCTAssertEqual(peaks[0].x, 50.7, accuracy: 1.5)
        XCTAssertEqual(peaks[1].x, 20.3, accuracy: 0.2)
        XCTAssertEqual(peaks[1].y, 30.6, accuracy: 0.2)
        XCTAssertGreaterThan(peaks[0].score, peaks[1].score)
    }

    func testEmptyMapGivesNoPeaks() {
        let map = [Float](repeating: 0.05, count: 64 * 36)
        XCTAssertTrue(decode(map, width: 64, height: 36, decoder: HeatmapDecoder()).isEmpty)
    }

    func testSigmoidAndLogitDetection() {
        var logits: [Float] = [-3, 0, 2, 8]
        XCTAssertTrue(logits.withUnsafeBufferPointer { HeatmapDecoder.looksLikeLogits($0) })
        logits.withUnsafeMutableBufferPointer { HeatmapDecoder.applySigmoid($0) }
        for (v, x) in zip(logits, [-3.0, 0, 2, 8]) {
            XCTAssertEqual(Double(v), 1 / (1 + exp(-x)), accuracy: 1e-5)
        }
        XCTAssertFalse(logits.withUnsafeBufferPointer { HeatmapDecoder.looksLikeLogits($0) })
    }

    // MARK: Frame geometry (backend detector.FrameGeometry(corner=True) + centroid_offset)

    func testLetterboxGeometryMatchesBackendFormula() {
        // 16:9 → no pad, s = 0.4 (1280×720) / 0.2667 (1920×1080).
        let g = FrameGeometry(width: 1280, height: 720, netWidth: 512, netHeight: 288, mode: .letterbox, peakOffset: 0.5)
        XCTAssertEqual(g.scaleX, 0.4, accuracy: 1e-12)
        XCTAssertEqual(g.padX, 0, accuracy: 1e-12)
        // full = (centroid + offset − pad) / s
        let peak = CGPoint(x: 200.25, y: 100.75)
        let full = g.peakToFull(x: Double(peak.x), y: Double(peak.y))
        XCTAssertEqual(Double(full.x), (200.25 + 0.5) / 0.4, accuracy: 1e-9)
        XCTAssertEqual(Double(full.y), (100.75 + 0.5) / 0.4, accuracy: 1e-9)

        // Portrait frame: pillar-boxed, same scale on both axes.
        let p = FrameGeometry(width: 1080, height: 1920, netWidth: 512, netHeight: 288, mode: .letterbox, peakOffset: 0.5)
        let s = 288.0 / 1920
        XCTAssertEqual(p.scaleX, s, accuracy: 1e-12)
        XCTAssertEqual(p.scaleY, s, accuracy: 1e-12)
        XCTAssertEqual(p.padX, (512 - 1080 * s) / 2, accuracy: 1e-12)
        XCTAssertEqual(p.padY, 0, accuracy: 1e-12)
        let q = CGPoint(x: 333.3, y: 1500.7)
        let net = p.toNet(q)
        let back = p.peakToFull(x: Double(net.x) - 0.5, y: Double(net.y) - 0.5)
        XCTAssertEqual(Double(back.x), 333.3, accuracy: 1e-9)
        XCTAssertEqual(Double(back.y), 1500.7, accuracy: 1e-9)
        XCTAssertEqual(Double(p.normalized(q).x), 333.3 / 1080, accuracy: 1e-12)
    }

    func testStretchGeometry() {
        let g = FrameGeometry(width: 1440, height: 1080, netWidth: 512, netHeight: 288, mode: .stretch)
        XCTAssertEqual(g.scaleX, 512.0 / 1440, accuracy: 1e-12)
        XCTAssertEqual(g.scaleY, 288.0 / 1080, accuracy: 1e-12)
        let p = CGPoint(x: 700, y: 300)
        let r = g.toFull(g.toNet(p))
        XCTAssertEqual(Double(r.x), 700, accuracy: 1e-9)
        XCTAssertEqual(Double(r.y), 300, accuracy: 1e-9)
    }

    // MARK: Model metadata

    func testSpecReadsExportMetadata() {
        let meta = ["model": "wasb", "seq_len": "3", "in_channels": "9", "uses_background": "0",
                    "input_order": "rgb", "normalize": "0-1", "input_width": "512", "input_height": "288",
                    "frame_order": "oldest_to_newest", "output_activation": "sigmoid", "score_threshold": "0.5",
                    "resize": "aspect_fit_center_black_pad", "centroid_offset": "0.5"]
        let s = ShuttleModelSpec(metadata: meta)
        XCTAssertEqual(s.name, "wasb")
        XCTAssertEqual(s.seqLen, 3)
        XCTAssertEqual(s.inChannels, 9)
        XCTAssertFalse(s.usesBackground)
        XCTAssertFalse(s.inputIsBGR)
        XCTAssertEqual(s.normalization, .unit)
        XCTAssertEqual(s.resize, .letterbox)
        XCTAssertEqual(s.centroidOffset, 0.5)
        XCTAssertEqual(s.scoreThreshold, 0.5)
        XCTAssertTrue(s.outputIsProbability)
        XCTAssertFalse(s.newestFirst)
        XCTAssertTrue(s.isConsistent)
    }

    func testSpecFallbacksForMissingKeys() {
        let s = ShuttleModelSpec(metadata: ["seq_len": "8", "uses_background": "1"])
        XCTAssertEqual(s.inChannels, 27)
        XCTAssertEqual(s.inputWidth, 512)
        XCTAssertEqual(s.inputHeight, 288)
        XCTAssertEqual(s.centroidOffset, 0)
        XCTAssertEqual(s.resize, .stretch)
        XCTAssertTrue(s.isConsistent)
    }
}
