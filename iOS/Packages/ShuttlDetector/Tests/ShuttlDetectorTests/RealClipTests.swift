import CoreML
import Darwin
import XCTest
import ShuttlCore
@testable import ShuttlDetector

// Real held-out TrackNetV2 Test clips (evaluation only): per-frame parity of the CoreML port with
// the cached Python candidates (benchmark.py --stage real, torch MPS fp16) and the benchmark's
// TrackNet-protocol F1 (top-1 ≥ 0.5, int-truncated 512×288 grid, ≤ 4 px, drop_frame range).
//
//   swift test -c release --filter RealClipTests                 # first 2 clips, both models
//   SHUTTL_DET_CLIPS=all swift test -c release --filter RealClipTests/testRealClips
//   SHUTTL_DET_MODELS=wasb SHUTTL_DET_DUMP=/tmp/swift_cands …    # dump candidates for the Python scorer
//     (backend/: .venv/bin/python -m scripts.swift_fixtures.detector --score-swift /tmp/swift_cands)

let modelsDir = packageRoot.appendingPathComponent("Models")
let compileCache = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("shuttldetector_mlcache")

/// benchmark.py real-stage matching rule (metrics.detection_metrics with tracknet_int=True, tol 4).
struct TN4: CustomStringConvertible {
    var tp = 0, fp1 = 0, fp2 = 0, fn = 0, tn = 0
    var precision: Double { tp + fp1 + fp2 > 0 ? Double(tp) / Double(tp + fp1 + fp2) : .nan }
    var recall: Double { tp + fn > 0 ? Double(tp) / Double(tp + fn) : .nan }
    var f1: Double { let p = precision, r = recall; return (p + r) > 0 && (p + r).isFinite ? 2 * p * r / (p + r) : 0 }
    static func + (a: TN4, b: TN4) -> TN4 {
        TN4(tp: a.tp + b.tp, fp1: a.fp1 + b.fp1, fp2: a.fp2 + b.fp2, fn: a.fn + b.fn, tn: a.tn + b.tn)
    }
    var description: String { String(format: "F1 %.5f P %.4f R %.4f (tp %d fp1 %d fp2 %d fn %d tn %d)", f1, precision, recall, tp, fp1, fp2, fn, tn) }
}

struct ClipTruth {
    let n: Int, width: Int, height: Int
    let vis: [Bool]
    let xy: [SIMD2<Double>]
    let mask: Range<Int>

    init(_ c: JSONValue) {
        n = c["n"]!.int; width = c["width"]!.int; height = c["height"]!.int
        var v = [Bool](repeating: false, count: n), p = [SIMD2<Double>](repeating: .zero, count: n)
        for (f, xy) in zip(c["vis_frames"]!.ints, c["vis_xy"]!.rows) { v[f] = true; p[f] = SIMD2(xy[0], xy[1]) }
        vis = v; xy = p
        let r = c["mask_range"]!.ints
        mask = r[0]..<r[1]
    }

    /// candidates_to_xy (max score, first on ties, ≥ 0.5) + detection_metrics(tracknet_int).
    func score(_ cands: [[ShuttleCandidate]]) -> TN4 {
        var r = TN4()
        let s = SIMD2(512.0 / Double(width), 288.0 / Double(height))
        for f in mask where f < n {
            let top = top1(f < cands.count ? cands[f] : [])
            switch (top, vis[f]) {
            case (nil, true): r.fn += 1
            case (nil, false): r.tn += 1
            case (_?, false): r.fp2 += 1
            case (let p?, true):
                let d = ((p * s).rounded(.towardZero) - (xy[f] * s).rounded(.towardZero))
                if (d * d).sum().squareRoot() <= 4 { r.tp += 1 } else { r.fp1 += 1 }
            }
        }
        return r
    }
}

func top1(_ cs: [ShuttleCandidate], threshold: Double = 0.5) -> SIMD2<Double>? {
    guard var best = cs.first else { return nil }
    for c in cs.dropFirst() where c.score > best.score { best = c }
    return best.score >= threshold ? SIMD2(best.x, best.y) : nil
}

func pythonCands(_ c: JSONValue, _ kind: String) -> [[ShuttleCandidate]]? {
    c["cands"]?[kind]?.array.map { frames in
        frames.map { ($0.array ?? []).map { v in let r = v.doubles; return ShuttleCandidate(x: r[0], y: r[1], score: r[2]) } }
    }
}

/// Per-frame agreement between two candidate streams.
struct Parity: CustomStringConvertible {
    var frames = 0, bothDetected = 0, detectAgree = 0, swaps = 0
    var dPos: [Double] = [], dScore: [Double] = []

    mutating func add(_ a: [[ShuttleCandidate]], _ ref: [[ShuttleCandidate]]) {
        for f in 0..<min(a.count, ref.count) {
            frames += 1
            let x = a[f].max { $0.score < $1.score }, y = ref[f].max { $0.score < $1.score }
            let dx = (x?.score ?? 0) >= 0.5, dy = (y?.score ?? 0) >= 0.5
            if dx == dy { detectAgree += 1 }
            guard let x, let y, dx || dy else { continue }
            let d = hypot(x.x - y.x, x.y - y.y)
            if d > 10 { swaps += 1; continue }       // a different peak won the frame
            if dx && dy { bothDetected += 1; dPos.append(d); dScore.append(abs(x.score - y.score)) }
        }
    }

    var agreement: Double { Double(detectAgree) / Double(max(frames, 1)) }
    var description: String {
        String(format: "frames %d agree@0.5 %.4f swaps %d | both %d: |Δxy| med %.3f p95 %.3f p99 %.3f max %.2f, ≤1.5px %.4f | |Δs| med %.4f p95 %.4f",
               frames, agreement, swaps, bothDetected, Num.median(dPos), Num.percentile(dPos, 95), Num.percentile(dPos, 99),
               dPos.max() ?? .nan, Double(dPos.filter { $0 <= 1.5 }.count) / Double(max(dPos.count, 1)),
               Num.median(dScore), Num.percentile(dScore, 95))
    }
}

/// Peak physical footprint while `body` runs (sampled every 10 ms), minus the footprint before.
final class FootprintSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var running = true
    private(set) var peak = 0
    let baseline: Int

    static func footprint() -> Int {
        var info = rusage_info_v4()
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return kr == 0 ? Int(info.ri_phys_footprint) : 0
    }

    init() {
        baseline = FootprintSampler.footprint()
        peak = baseline
        Thread.detachNewThread { [self] in
            while true {
                let f = FootprintSampler.footprint()
                lock.lock()
                peak = max(peak, f)
                let go = running
                lock.unlock()
                if !go { return }
                usleep(10_000)
            }
        }
    }

    func stop() -> (peakMB: Double, deltaMB: Double) {
        lock.lock(); running = false; let p = peak; lock.unlock()
        return (Double(p) / 1e6, Double(p - baseline) / 1e6)
    }
}

final class RealClipTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment

    func loadDetector(_ kind: String, units: MLComputeUnits = .all) async throws -> HeatmapDetector {
        let url = try XCTUnwrap(HeatmapDetector.findModel(DetectorKind(rawValue: kind)!, in: modelsDir),
                                "no \(kind) model in \(modelsDir.path) — run backend/ml/export_coreml.py")
        return try await HeatmapDetector.load(modelURL: url, computeUnits: units, cacheDirectory: compileCache)
    }

    /// The Swift scorer reproduces the benchmark's pooled F1 from the cached Python candidates.
    func testScorerReproducesBenchmark() throws {
        let real = try fixture("real")
        for kind in ["tracknetv3", "wasb"] {
            var pooled = TN4()
            for c in real["clips"]!.array! { pooled = pooled + ClipTruth(c).score(pythonCands(c, kind)!) }
            let ref = real["pooled"]![kind]!
            XCTAssertEqual(pooled.tp, ref["tp"]!.int, kind); XCTAssertEqual(pooled.fn, ref["fn"]!.int, kind)
            XCTAssertEqual(pooled.fp1, ref["fp1"]!.int, kind); XCTAssertEqual(pooled.fp2, ref["fp2"]!.int, kind)
            XCTAssertEqual(pooled.f1, ref["tn4_f1"]!.double!, accuracy: 1e-12, kind)
        }
    }

    func testRealClips() async throws {
        let real = try fixture("real")
        let clips = real["clips"]!.array!
        let first = clips[0]
        try XCTSkipUnless(FileManager.default.fileExists(atPath: repoRoot.appendingPathComponent(first["video"]!.string!).path),
                          "TrackNetV2 Test split not downloaded")
        #if DEBUG
        // Unoptimised pixel loops make this minutes long in debug builds; run it with -c release.
        try XCTSkipIf(Self.env["SHUTTL_DET_CLIPS"] == nil, "debug build: run `swift test -c release` (or set SHUTTL_DET_CLIPS)")
        #endif
        let nClips = Self.env["SHUTTL_DET_CLIPS"].map { $0 == "all" ? clips.count : Int($0) ?? 2 } ?? 2
        let kinds = (Self.env["SHUTTL_DET_MODELS"] ?? "tracknetv3,wasb").split(separator: ",").map(String.init)
        let dump = Self.env["SHUTTL_DET_DUMP"].map { URL(fileURLWithPath: $0) }
        if let dump { try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true) }
        for kind in kinds {
            let det = try await loadDetector(kind)
            let net = det.network as! CoreMLHeatmapNetwork
            var swiftTN = TN4(), pyTN = TN4(), parity = Parity()
            var frames = 0, seconds = 0.0, peakDelta = 0.0, peakAbs = 0.0
            var stats = DetectorStats()
            for c in clips.prefix(nClips) {
                let truth = ClipTruth(c)
                let video = try await VideoSource.open(repoRoot.appendingPathComponent(c["video"]!.string!))
                XCTAssertEqual(video.meta.nFrames, truth.n, "frame count vs OpenCV")
                let sampler = FootprintSampler()
                let t0 = CFAbsoluteTimeGetCurrent()
                let cands = try await det.detectVideo(video)
                seconds += CFAbsoluteTimeGetCurrent() - t0
                let mem = sampler.stop()
                peakDelta = max(peakDelta, mem.deltaMB); peakAbs = max(peakAbs, mem.peakMB)
                frames += cands.count
                let s = det.lastStats
                stats.networkSeconds += s.networkSeconds; stats.preprocessSeconds += s.preprocessSeconds
                stats.backgroundSeconds += s.backgroundSeconds; stats.decodeSeconds += s.decodeSeconds; stats.windows += s.windows
                XCTAssertEqual(cands.count, truth.n)
                for cs in cands {
                    XCTAssertLessThanOrEqual(cs.count, 3)
                    XCTAssertEqual(cs.map(\.score), cs.map(\.score).sorted(by: >))
                }
                let py = try XCTUnwrap(pythonCands(c, kind))
                let st = truth.score(cands), pt = truth.score(py)
                swiftTN = swiftTN + st; pyTN = pyTN + pt
                var p = Parity(); p.add(cands, py)
                parity.dPos += p.dPos; parity.dScore += p.dScore
                parity.frames += p.frames; parity.bothDetected += p.bothDetected; parity.detectAgree += p.detectAgree; parity.swaps += p.swaps
                print(String(format: "[real] %@ %@/%@ n=%d swift F1 %.4f py F1 %.4f | agree %.4f swaps %d | %.1f fps",
                             kind, c["match"]!.string!, c["clip"]!.string!, truth.n, st.f1, pt.f1, p.agreement, p.swaps,
                             Double(truth.n) / (CFAbsoluteTimeGetCurrent() - t0)))
                if let dump {
                    let rows = cands.map { $0.map { [$0.x, $0.y, $0.score] } }
                    let data = try JSONSerialization.data(withJSONObject: ["cands": rows])
                    try data.write(to: dump.appendingPathComponent("\(c["match"]!.string!)_\(c["clip"]!.string!)_\(kind).json"))
                }
            }
            let f = Double(max(frames, 1))
            print("[real] \(kind) clips \(nClips) frames \(frames): SWIFT \(swiftTN)")
            print("[real] \(kind) PYTHON (same clips) \(pyTN)   ΔF1 \(String(format: "%+.5f", swiftTN.f1 - pyTN.f1))")
            print("[real] \(kind) parity \(parity)")
            print(String(format: "[real] %@ timing: %.1f fps overall (%.2f ms/frame) | net %.2f ms/frame (%d windows, %.1f ms/window) | letterbox %.2f ms/frame (prefetch thread) | background %.2f ms/frame | decode %.2f ms/frame | peak footprint +%.0f MB (abs %.0f MB)",
                         kind, f / seconds, 1e3 * seconds / f, 1e3 * stats.networkSeconds / f, stats.windows,
                         1e3 * stats.networkSeconds / Double(max(stats.windows, 1)), 1e3 * stats.preprocessSeconds / f,
                         1e3 * stats.backgroundSeconds / f, 1e3 * stats.decodeSeconds / f, peakDelta, peakAbs))
            _ = net
            // Acceptance: F1 gap ≤ 0.005 vs the Python reference on the same clips.
            XCTAssertLessThanOrEqual(abs(swiftTN.f1 - pyTN.f1), nClips >= 10 ? 0.005 : 0.02, "\(kind) F1 gap")
            XCTAssertGreaterThanOrEqual(parity.agreement, 0.97, "\(kind) detection agreement")
            XCTAssertLessThanOrEqual(Num.median(parity.dPos), 0.5, "\(kind) median Δxy")
            XCTAssertLessThanOrEqual(Num.percentile(parity.dPos, 95), 1.5, "\(kind) p95 Δxy")
            XCTAssertLessThanOrEqual(Num.percentile(parity.dScore, 95), 0.05, "\(kind) p95 Δscore")
        }
    }

    /// Pure network latency per compute-unit setting (random input, after warm-up).
    func testModelLatency() async throws {
        let units: [(String, MLComputeUnits)] = [("all", .all), ("cpuAndNeuralEngine", .cpuAndNeuralEngine),
                                                 ("cpuAndGPU", .cpuAndGPU)]
        for kind in ["tracknetv3", "wasb"] {
            for (name, u) in units {
                let det = try await loadDetector(kind, units: u)
                let net = det.network
                let S = net.spec.seqLen
                let frames = (0..<S).map { i -> NetImage in
                    let img = NetImage()
                    let src = FormulaImage.mixed(512, 288, seed: i)
                    src.data.withUnsafeBufferPointer { img.pixels.update(from: $0.baseAddress!, count: img.count) }
                    return img
                }
                let w = NetWindow(start: 0, frames: frames, background: net.spec.usesBackground ? frames[0] : nil)
                let out = UnsafeMutablePointer<Float>.allocate(capacity: S * 512 * 288)
                defer { out.deallocate() }
                for _ in 0..<3 { try net.predict([w], into: out) }
                var ts: [Double] = []
                for _ in 0..<15 {
                    let t = CFAbsoluteTimeGetCurrent()
                    try net.predict([w], into: out)
                    ts.append((CFAbsoluteTimeGetCurrent() - t) * 1e3)
                }
                print(String(format: "[latency] %@ %@: %.2f ms/window (p90 %.2f) = %.2f ms/frame", kind, name, Num.median(ts),
                             Num.percentile(ts, 90), Num.median(ts) / Double(S)))
            }
        }
    }
}
