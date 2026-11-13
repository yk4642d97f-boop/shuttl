import XCTest
import ShuttlCore
@testable import ShuttlDetector

// MARK: - test networks (test_detector.py doubles)

/// Heatmap s of every window = red channel (x/255) of input frame s (background slot skipped).
final class EchoNet: HeatmapNetwork {
    let spec: NetSpec
    init(_ spec: NetSpec) { self.spec = spec }
    func predict(_ windows: [NetWindow], into out: UnsafeMutablePointer<Float>) throws {
        let plane = detectorNetWidth * detectorNetHeight
        for (b, w) in windows.enumerated() {
            for s in 0..<spec.seqLen {
                let img = w.frames[s], o = out + (b * spec.seqLen + s) * plane
                for i in 0..<plane { o[i] = Float(img.pixels[3 * i]) / 255 }
            }
        }
    }
}

/// Every heatmap = constant (slot + 1) / 16 — exposes the fusion weights per slot.
final class SlotNet: HeatmapNetwork {
    let spec: NetSpec
    init(_ spec: NetSpec) { self.spec = spec }
    func predict(_ windows: [NetWindow], into out: UnsafeMutablePointer<Float>) throws {
        let plane = detectorNetWidth * detectorNetHeight
        for b in windows.indices {
            for s in 0..<spec.seqLen { (out + (b * spec.seqLen + s) * plane).update(repeating: Float(s + 1) / 16, count: plane) }
        }
    }
}

/// Every output heatmap = green channel of the background slot.
final class BgEchoNet: HeatmapNetwork {
    let spec: NetSpec
    init(_ spec: NetSpec) { self.spec = spec }
    func predict(_ windows: [NetWindow], into out: UnsafeMutablePointer<Float>) throws {
        let plane = detectorNetWidth * detectorNetHeight
        for (b, w) in windows.enumerated() {
            let bg = w.background!
            for s in 0..<spec.seqLen {
                let o = out + (b * spec.seqLen + s) * plane
                for i in 0..<plane { o[i] = Float(bg.pixels[3 * i + 1]) / 255 }
            }
        }
    }
}

func echoSpec(_ s: Int, _ bg: Bool, corner: Bool = false, offset: Double = -0.5) -> NetSpec {
    NetSpec(name: "echo", seqLen: s, usesBackground: bg, imagenetNorm: false, logits: false, cornerGeometry: corner,
            peakOffset: offset)
}

func config(stride: Int? = nil, batch: Int = 8, prefetch: Bool = false) -> DetectorConfig {
    var c = DetectorConfig()
    c.stride = stride; c.batchSize = batch; c.prefetch = prefetch
    return c
}

func meanD(_ a: [Float]) -> Double { a.reduce(0.0) { $0 + Double($1) } / Double(a.count) }

func frameValue(_ i: Int) -> UInt8 { UInt8(10 + (7 * i) % 230) }

/// 128×72 frames whose red channel carries the frame id.
func idSource(_ n: Int) -> GeneratedFrameSource {
    GeneratedFrameSource(count: n) { i in
        var img = ImageU8(width: 128, height: 72, channels: 3)
        for p in 0..<(128 * 72) { img.data[3 * p] = frameValue(i) }
        return img
    }
}

/// Static scene (green 100; 200 from frame `cut` on) + a small bright blob moving 5 px per frame.
func sceneSource(_ n: Int, cut: Int? = nil) -> GeneratedFrameSource {
    GeneratedFrameSource(count: n) { i in
        var img = ImageU8(width: 128, height: 72, channels: 3)
        let g: UInt8 = (cut.map { i >= $0 } ?? false) ? 200 : 100
        for p in 0..<(128 * 72) { img.data[3 * p + 1] = g }
        let x = (5 * i) % 120
        for y in 30..<38 { for xx in x..<(x + 8) { img.data[(y * 128 + xx) * 3 + 1] = 250 } }
        return img
    }
}

func collect(_ det: HeatmapDetector, _ src: DetectorFrameSource, _ n: Int, w: Int = 128, h: Int = 72) throws -> [(Int, [Float])] {
    var got: [(Int, [Float])] = []
    try det.iterHeatmaps(src, width: w, height: h, fps: 30, nFrames: n) { f, hm in got.append((f, Array(hm))) }
    return got
}

final class StreamingTests: XCTestCase {

    func testEnsembleWeights() throws {
        let fx = try fixture("images")["ensemble"]!
        for s in [1, 2, 3, 5, 8] {
            XCTAssertEqual(ensembleWeights(s, .weight).map(Double.init), fx["\(s)"]!["weight"]!.doubles)
            XCTAssertEqual(ensembleWeights(s, .average).map(Double.init), fx["\(s)"]!["average"]!.doubles)
        }
    }

    /// Python iter_heatmaps with the slot-constant net: identical frame order and fused values.
    func testFusionMatchesPython() throws {
        for c in try fixture("fusion").array ?? [] {
            let S = c["seq_len"]!.int, n = c["n"]!.int
            let det = HeatmapDetector(network: SlotNet(echoSpec(S, c["uses_background"]!.bool!)),
                                      config: config(stride: c["stride"]!.int, batch: c["batch_size"]!.int))
            let got = try collect(det, sceneSource(n), n)
            let tag = "S=\(S) stride=\(c["stride"]!.int) n=\(n)"
            XCTAssertEqual(got.map(\.0), c["frames"]!.ints, tag)
            for ((_, hm), v) in zip(got, c["values"]!.doubles) { XCTAssertEqual(Double(hm[0]), v, accuracy: 1e-6, tag) }
            XCTAssertEqual(det.maxBuffered, c["max_buffered"]!.int, tag)
            XCTAssertEqual(det.maxLag, c["max_lag"]!.int, tag)
        }
    }

    /// Independent re-statement of the windowing contract (test_window_fusion_weights_and_tail).
    func testWindowFusionWeightsAndTail() throws {
        func reference(_ n: Int, _ S: Int, _ stride: Int) -> [Double] {
            let w = ensembleWeights(S, stride < S ? .weight : .average).map(Double.init)
            var num = [Double](repeating: 0, count: n), den = num
            let starts = n >= S ? Array(Swift.stride(from: 0, through: n - S, by: stride)) : []
            for st in starts { for s in 0..<S { num[st + s] += w[s] * Double(s + 1) / 16; den[st + s] += w[s] } }
            if starts.isEmpty || starts.last! + S < n {
                let lo = starts.isEmpty ? 0 : starts.last! + stride, tail = max(n - S, 0)
                for s in 0..<S where lo <= tail + s && tail + s < n { num[tail + s] += w[s] * Double(s + 1) / 16; den[tail + s] += w[s] }
            }
            return zip(num, den).map { $0 / $1 }
        }
        for (S, stride, bg, n) in [(3, 3, false, 50), (3, 1, false, 31), (8, 8, true, 45), (8, 4, true, 43), (8, 3, false, 30),
                                   (8, 1, false, 20), (8, 8, false, 5), (3, 2, false, 2), (8, 8, false, 64)] {
            let det = HeatmapDetector(network: SlotNet(echoSpec(S, bg)), config: config(stride: stride, batch: 3))
            let got = try collect(det, sceneSource(n), n)
            let ref = reference(n, S, stride)
            XCTAssertEqual(got.map(\.0), Array(0..<n))
            for (f, hm) in got { XCTAssertEqual(Double(hm[0]), ref[f], accuracy: 1e-6, "S=\(S) stride=\(stride) f=\(f)") }
        }
    }

    func testStreamingAlignmentAndBoundedMemory() throws {
        for (S, stride, bg, n) in [(3, 3, false, 50), (3, 1, false, 31), (8, 8, true, 45), (8, 4, true, 43),
                                   (8, 1, false, 20), (8, 8, false, 5), (3, 2, false, 2)] {
            let det = HeatmapDetector(network: EchoNet(echoSpec(S, bg)), config: config(stride: stride, batch: 2))
            let got = try collect(det, idSource(n), n)
            XCTAssertEqual(got.map(\.0), Array(0..<n))
            for (f, hm) in got {
                XCTAssertEqual(hm.count, netW * netH)
                XCTAssertEqual(meanD(hm), Double(frameValue(f)) / 255, accuracy: 1e-5)
            }
            XCTAssertLessThanOrEqual(det.maxBuffered, S + 2 * S)
            XCTAssertLessThanOrEqual(det.maxLag, S + 2 * stride + S)
        }
    }

    func testLongStreamStaysBounded() throws {
        let n = 400, S = 3
        let det = HeatmapDetector(network: EchoNet(echoSpec(S, false)), config: config(stride: S, batch: 4))
        var count = 0
        try det.iterHeatmaps(idSource(n), width: 128, height: 72, fps: 30, nFrames: n) { _, _ in count += 1 }
        XCTAssertEqual(count, n)
        XCTAssertLessThanOrEqual(det.maxBuffered, S + 4 * S)
        XCTAssertLessThanOrEqual(det.maxLag, S + 4 * S)
    }

    func testBackgroundIsPrependedFirst() throws {
        let n = 40
        let det = HeatmapDetector(network: BgEchoNet(echoSpec(8, true)), config: config())
        let got = try collect(det, sceneSource(n), n)
        XCTAssertEqual(got.map(\.0), Array(0..<n))
        for (_, hm) in got { XCTAssertEqual(meanD(hm), 100.0 / 255, accuracy: 1e-5) }
    }

    func testBackgroundFollowsSceneCuts() throws {
        let n = 48, cut = 24
        let det = HeatmapDetector(network: BgEchoNet(echoSpec(8, true)), config: config())
        for (f, hm) in try collect(det, sceneSource(n, cut: cut), n) {
            XCTAssertEqual(meanD(hm), (f >= cut ? 200 : 100) / 255.0, accuracy: 1e-5, "\(f)")
        }
    }

    /// estimate_backgrounds on the formula scene: same segment starts and bit-identical medians.
    func testBackgroundPlanMatchesPython() throws {
        for c in try fixture("background").array ?? [] {
            let w = c["w"]!.int, h = c["h"]!.int, n = c["n"]!.int
            let cut = c["cut"]?.double.map { Int($0) }
            let kw = c["kwargs"]!
            var bc = BackgroundConfig()
            if let v = kw["max_samples"]?.double { bc.maxSamples = Int(v) }
            if let v = kw["max_segment_s"]?.double { bc.maxSegmentSeconds = v }
            if let v = kw["cut_frac"]?.double { bc.cutFraction = v }
            let src = GeneratedFrameSource(count: n) { i in
                var img = FormulaImage.mixed(w, h, seed: (cut.map { i >= $0 } ?? false) ? 1 : 0)
                let x = (9 * i) % (w - 12), y = (h / 3 + 5 * i) % (h - 12)
                for yy in y..<(y + 12) { for xx in x..<(x + 12) {
                    let p = (yy * w + xx) * 3
                    img.data[p] = 250; img.data[p + 1] = 240; img.data[p + 2] = 230
                } }
                return img
            }
            let plan = try estimateBackgrounds(source: src, geometry: FrameGeometry(width: w, height: h), fps: c["fps"]!.double!,
                                               config: bc)
            let tag = "\(w)x\(h) cut=\(String(describing: cut)) \(bc)"
            XCTAssertEqual(plan.starts, c["starts"]!.ints, tag)
            for (img, ref) in zip(plan.images, c["images"]!.array ?? []) {
                let (s, ws) = FormulaImage.rowChecksums(img.pixels, width: img.width, height: img.height)
                XCTAssertEqual(s, ref["sum"]!.ints, tag); XCTAssertEqual(ws, ref["wsum"]!.ints, tag)
            }
            XCTAssertEqual(plan.index(for: 0), 0)
            if plan.starts.count > 1 {
                XCTAssertEqual(plan.index(for: plan.starts[1]), 1); XCTAssertEqual(plan.index(for: plan.starts[1] - 1), 0)
            }
        }
    }

    /// End to end with an oracle net: one list per frame, top-k sorted, full-res subpixel
    /// coordinates (corner geometry, zero peak offset → identity).
    func testDetectFramesOutputContract() throws {
        let W = 640, H = 360, n = 12
        let pts: [(Double, Double)] = (0..<n).map { (i: Int) -> (Double, Double) in (100.25 + 30 * Double(i), 80.75 + 11 * Double(i)) }
        let sigma: Double = 2.5 * Double(W) / Double(netW)
        let src = GeneratedFrameSource(count: n) { (i: Int) -> ImageU8 in
            let g: [Float] = gaussHM(pts[i].0, pts[i].1, sigma, 230, w: W, h: H)
            var img = ImageU8(width: W, height: H, channels: 3)
            for p in 0..<(W * H) { img.data[3 * p] = UInt8(g[p]) }       // astype(uint8) truncates
            return img
        }
        let det = HeatmapDetector(network: EchoNet(echoSpec(3, false, corner: true, offset: 0)), config: config(stride: 3))
        let out = try det.detectFrames(src, width: W, height: H, fps: 30, nFrames: n)
        XCTAssertEqual(out.count, n)
        for (i, cs) in out.enumerated() {
            XCTAssert((1...3).contains(cs.count))
            XCTAssertEqual(cs.map(\.score), cs.map(\.score).sorted(by: >))
            XCTAssertLessThan(hypot(cs[0].x - pts[i].0, cs[0].y - pts[i].1), 0.35, "\(i) \(cs[0]) \(pts[i])")
        }
    }

    func testCandidatesDropLetterboxHits() {
        let det = HeatmapDetector(network: EchoNet(echoSpec(3, false)))
        let g = det.geometry(width: 640, height: 480)              // 4:3 → pillarbox
        let a = gaussHM(10, 144, 2), b = gaussHM(256, 144, 2, 0.8)
        let hm = zip(a, b).map { $0 + $1 }
        let c = hm.withUnsafeBufferPointer { det.candidates($0, geometry: g) }
        XCTAssertEqual(c.count, 1)
        XCTAssertLessThan(abs(c[0].x - 320), 1.5); XCTAssertLessThan(abs(c[0].y - 240), 1.5)
    }

    func testPeakOffsetInvertsTrainingTargets() {
        for spec in [NetSpec.wasb, .tracknetv3] {
            let g = FrameGeometry(width: 1280, height: 720, corner: spec.cornerGeometry)
            for X in [3.0, 101.7, 640.2, 1279.0] {
                let f = g.peakToFull(x: X * 0.4 - 0.5, y: X * 0.4 - 0.5, peakOffset: spec.peakOffset)
                XCTAssertEqual(f.x, X, accuracy: 1e-9); XCTAssertEqual(f.y, X, accuracy: 1e-9)
            }
        }
    }

    func testProgressIsMonotonicAndCompletes() throws {
        final class Box: @unchecked Sendable { var v: [Double] = [] }
        let seen = Box()
        let det = HeatmapDetector(network: EchoNet(echoSpec(8, true)), config: config(prefetch: true))
        let out = try det.detectFrames(sceneSource(100), width: 128, height: 72, fps: 30, nFrames: 100,
                                       progress: { seen.v.append($0) })
        XCTAssertEqual(out.count, 100)
        XCTAssertEqual(seen.v.last, 1.0)
        XCTAssert(zip(seen.v, seen.v.dropFirst()).allSatisfy { $0 <= $1 } && seen.v[0] >= 0)
    }

    /// The decode thread (prefetch) must not change results.
    func testPrefetchIsTransparent() throws {
        for (S, bg) in [(3, false), (8, true)] {
            let a = try collect(HeatmapDetector(network: EchoNet(echoSpec(S, bg)), config: config(stride: S, batch: 2)), sceneSource(37), 37)
            let b = try collect(HeatmapDetector(network: EchoNet(echoSpec(S, bg)), config: config(stride: S, batch: 2, prefetch: true)),
                                sceneSource(37), 37)
            XCTAssertEqual(a.map(\.0), b.map(\.0))
            for (x, y) in zip(a, b) { XCTAssertEqual(x.1, y.1) }
        }
    }

    /// tta_flip: a net that always outputs the x-ramp x/511 (not flip-equivariant) must average to
    /// 0.5·(x + 511 − x)/511 = 0.5; an echo net (equivariant) must be unchanged.
    func testTTAFlip() throws {
        final class RampNet: HeatmapNetwork {
            let spec = echoSpec(3, false)
            func predict(_ windows: [NetWindow], into out: UnsafeMutablePointer<Float>) throws {
                for pl in 0..<(windows.count * 3) {
                    let o = out + pl * netW * netH
                    for y in 0..<netH { for x in 0..<netW { o[y * netW + x] = Float(x) / 511 } }
                }
            }
        }
        var c = config(stride: 3); c.ttaFlip = true
        let det = HeatmapDetector(network: RampNet(), config: c)
        let got = try collect(det, idSource(7), 7)
        XCTAssertEqual(got.map(\.0), Array(0..<7))
        for (_, hm) in got { for x in [0, 1, 100, 255, 256, 511] { XCTAssertEqual(hm[50 * netW + x], 0.5, accuracy: 1e-6) } }
        let plain = try collect(HeatmapDetector(network: EchoNet(echoSpec(3, false)), config: config(stride: 3)), idSource(9), 9)
        var ct = config(stride: 3); ct.ttaFlip = true
        let tta = try collect(HeatmapDetector(network: EchoNet(echoSpec(3, false)), config: ct), idSource(9), 9)
        for (a, b) in zip(plain, tta) { XCTAssertEqual(a.1, b.1) }
    }
}
