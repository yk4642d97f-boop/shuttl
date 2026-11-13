import Foundation
import ShuttlCore

// Port of backend/scripts/benchmark.py's control flow: eval_chain (e2e), eval_oracle,
// run_real (real / real_oracle), Pool.results + add_tn4, the report and the printout.
// Stage outputs come from a `BenchStages` provider; truth from bundles (synthetic) or
// the TrackNetV2 Test labels (real).

public let benchGap = 100_000                          // frame offset between pooled samples

// MARK: - Options (benchmark.py argparse namespace)

public struct BenchOptions: Sendable {
    /// `vars(args)` in argparse order (reported verbatim).
    public var args: PyDict

    public init(args: PyDict) { self.args = args }

    /// benchmark.py defaults.
    public static func defaults(backend: URL) -> BenchOptions {
        BenchOptions(args: [
            "stage": "oracle", "n": 10, "seed": 0, "rallies": 1, "fps": 60.0, "image_size": [1920, 1080],
            "style": "random", "doubles": nil, "player_step": 2, "track_noise": 1.0, "keep_videos": false,
            "sigma": 1.5, "miss_rate": 0.1, "fp_rate": 0.03, "clips": nil, "max_frames": nil, "detector": "auto",
            "device": nil, "weights_dir": nil, "score_thresh": 0.5, "cache_dir": "/tmp/shuttl_bench_cache",
            "no_cache": false, "no_real_oracle": false, "all_frames": false, "tol_px": nil, "tol_frames": 3,
            "out": .str(backend.appendingPathComponent("ml/benchmarks").path), "no_json": false,
        ])
    }

    public var stage: String { args["stage"]?.string ?? "oracle" }
    public var isReal: Bool { stage == "real" || stage == "real_oracle" }
    public var tolFrames: Int { args["tol_frames"]?.int ?? 3 }
    public var tolPxArg: Double? { args["tol_px"]?.double }
    public var scoreThresh: Double { args["score_thresh"]?.double ?? 0.5 }
    public var clips: Int? { args["clips"]?.int }
    public var maxFrames: Int? { args["max_frames"]?.int }
    public var allFrames: Bool { args["all_frames"]?.bool ?? false }
    public var noRealOracle: Bool { args["no_real_oracle"]?.bool ?? false }
    public var detector: String { args["detector"]?.string ?? "auto" }
    public var noJSON: Bool { args["no_json"]?.bool ?? false }
    public var outDir: String? { args["out"]?.string }
    public var imageSize: ImageSize {
        let l = args["image_size"]?.list?.compactMap(\.int) ?? [1920, 1080]
        return ImageSize(width: l.count > 0 ? l[0] : 1920, height: l.count > 1 ? l[1] : 1080)
    }
    /// The run's pixel tolerance: --tol-px, else 10 px (real) / 4 px @ 512×288 scaled (synthetic).
    public var tolPx: Double { tolPxArg ?? (isReal ? 10.0 : Metrics.scaledTolerance(4.0, imageSize: imageSize)) }
}

// MARK: - Samples

public struct SimSample: Sendable {
    public var seed: Int
    public var truth: SimTruth
    public var candidates: [[ShuttleCandidate]]
    public var camera: CameraModel?
    public init(seed: Int, truth: SimTruth, candidates: [[ShuttleCandidate]], camera: CameraModel? = nil) {
        self.seed = seed; self.truth = truth; self.candidates = candidates; self.camera = camera
    }
}

public struct RealSample: Sendable {
    public var clip: RealClip
    public var meta: VideoMeta
    public var truthXY: [SIMD2<Double>]
    public var truthVis: [Bool]
    public var dropMask: [Bool]?
    /// real_oracle input: `sim.corrupt_px(GT, …)` candidates (numpy RNG → recorded).
    public var oracleCandidates: [[ShuttleCandidate]]?
    public init(clip: RealClip, meta: VideoMeta, truthXY: [SIMD2<Double>], truthVis: [Bool], dropMask: [Bool]?,
                oracleCandidates: [[ShuttleCandidate]]? = nil) {
        self.clip = clip; self.meta = meta; self.truthXY = truthXY; self.truthVis = truthVis
        self.dropMask = dropMask; self.oracleCandidates = oracleCandidates
    }
}

public enum BenchSamples: Sendable {
    case synthetic([SimSample])
    case real([RealSample])
}

// MARK: - Runner

public final class BenchRunner {
    public let options: BenchOptions
    public let stages: any BenchStages
    public let pool = BenchPool()
    private var skippedMissing = Set<String>()

    public init(options: BenchOptions, stages: any BenchStages) {
        self.options = options; self.stages = stages
    }

    /// benchmark `Runner.call`: nil + SKIPPED for a missing stage, nil + error record on failure.
    func call<T>(_ stage: String, _ sample: String, _ body: () async throws -> T?) async -> T? {
        guard stages.available.contains(stage) else {
            pool.skipped[stage] = stages.missing[stage]?.string ?? "not available"
            return nil
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        defer { pool.seconds[stage, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        do {
            return try await body()
        } catch let e as RecordedStageError {
            pool.appendError(stage, e.message)
        } catch {
            pool.error(stage, sample: sample, error)
        }
        return nil
    }

    // MARK: synthetic (eval_chain / eval_oracle)

    /// `eval_chain`: linker → hits → 3-D → rallies → shots on one sample, each vs truth.
    func evalChain(_ s: SimSample, index: Int, offset: Int, prefix: String = "") async {
        let name = "seed\(s.seed)", T = s.truth
        let P = { (st: String) in "\(prefix)\(st)" }
        let ctx = context(s, index: index)
        pool.addFrames(P("detections"), pred: Metrics.candidatesToXY(s.candidates).xy, truth: T.detPx, vis: T.detVis)
        guard let track = await call("linker", name, { try await self.stages.linker(ctx, candidates: s.candidates) }) else {
            for st in ["hits", "reconstruct3d", "rally", "shots"] { pool.skip(P(st), "no track (linker missing/failed)") }
            return
        }
        pool.addFrames(P("linker"), pred: track.xy, truth: T.trkPx, vis: T.trkVis, interp: track.interpolated)
        guard let hits = await call("hits", name, { try await self.stages.hits(ctx, track: track) }) else {
            for st in ["reconstruct3d", "rally", "shots"] { pool.skip(P(st), "no hits") }
            return
        }
        pool.addHits(P("hits"), pred: hits.map(\.eval), truth: T.hits, offset: offset)
        // `if camera is not None:` — synthetic e2e always passes the true camera
        if let segs = await call("reconstruct3d", name, { try await self.stages.reconstruct(ctx, track: track) }) {
            pool.addSegments(P("reconstruct3d"), pred: segs.map(\.segment), truth: T.shots, offset: offset)
        }
        guard let rallies = await call("rally", name, { try await self.stages.rallies(ctx, track: track) }) else {
            pool.skip(P("shots"), "no rallies")
            return
        }
        pool.addRallies(P("rally"), pred: rallies, truth: T.rallies, offset: offset)
        var typed: [BenchTypedRally?] = []
        for k in rallies.indices { typed.append(await call("shots", name, { try await self.stages.shots(ctx, rally: k) })) }
        if typed.contains(where: { $0 == nil }) { return }
        let predShots = typed.compactMap { $0 }.flatMap(\.shots)
        let pairs = Metrics.matchEvents(predShots.map(\.frame), T.hits.map(\.frame), tol: 3)
        pool.addLists(P("shots")) { L in
            L.typePred += pairs.map { predShots[$0.0].shotType }
            L.typeTrue += pairs.map { T.hits[$0.1].shotType ?? "unknown" }
            L.typeCov.append(Double(pairs.count) / Double(max(T.hits.count, 1)))
        }
    }

    /// `eval_oracle`: truth fed into each stage in isolation.
    func evalOracle(_ s: SimSample, index: Int, offset: Int) async {
        let name = "seed\(s.seed)", T = s.truth
        let ctx = context(s, index: index)
        pool.addFrames("detections", pred: Metrics.candidatesToXY(s.candidates).xy, truth: T.detPx, vis: T.detVis)
        if let track = await call("linker", name, { try await self.stages.linker(ctx, candidates: s.candidates) }) {
            pool.addFrames("linker", pred: track.xy, truth: T.trkPx, vis: T.trkVis, interp: track.interpolated)
        }
        if let hits = await call("hits", name, { try await self.stages.hits(ctx, track: nil) }) {
            pool.addHits("hits", pred: hits.map(\.eval), truth: T.hits, offset: offset)
        }
        if let segs = await call("reconstruct3d", name, { try await self.stages.reconstruct(ctx, track: nil) }) {
            pool.addSegments("reconstruct3d", pred: segs.map(\.segment), truth: T.shots, offset: offset)
            var byFrame = OrderedMap<Int>()                       // {start_frame: seg} — last one wins
            for (k, sg) in segs.enumerated() { byFrame[String(sg.segment.startFrame)] = k }
            for fc in T.finalCalls ?? [] {
                var pick: Int? = nil, best = Int.max
                for (f, k) in byFrame.items {                      // min(..., key=|Δ|): first minimum
                    let d = abs(Int(f)! - fc.frame)
                    if d < best { best = d; pick = k }
                }
                if let k = pick, abs(segs[k].segment.startFrame - fc.frame) > 4 { pick = nil }
                let pred = oracleCall(pick.map { (segs[$0].segment, $0) }, ctx: ctx, doubles: T.doubles, isServe: fc.isServe)
                pool.addLists("reconstruct3d") { L in
                    L.callPred.append(pred); L.callTrue.append(fc.inOut); L.callMargin.append(fc.margin.flatMap { $0.isFinite ? $0 : nil })
                }
            }
        }
        if let rallies = await call("rally", name, { try await self.stages.rallies(ctx, track: nil) }) {
            pool.addRallies("rally", pred: rallies, truth: T.rallies, offset: offset)
        }
        for (k, trueTypes) in (T.rallyTypes ?? []).enumerated() {
            guard let typed = await call("shots", name, { try await self.stages.shots(ctx, rally: k) }) else { break }
            pool.addLists("shots") { L in
                L.typePred += typed.shots.map(\.shotType)
                L.typeTrue += trueTypes
            }
        }
    }

    /// benchmark `_call`: the in/out verdict on a rally's final shot.
    func oracleCall(_ seg: (EvalSegment, Int)?, ctx: StageContext, doubles: Bool, isServe: Bool) -> String {
        guard let (s, k) = seg else { return "unknown" }
        if let z = s.netCrossingHeight, z.isFinite, z < CourtModel.netHeightCentre { return "net" }
        switch stages.callInOut(ctx, segment: k, isServe: isServe) {
        case .call(let c): return c == "in" || c == "out" ? c : "unknown"
        case .unavailable, .raised: break
        }
        guard let l = s.landingXY else { return "unknown" }
        return CourtModel.inOutMargin(x: l.x, y: l.y, doubles: doubles) >= 0 ? "in" : "out"
    }

    func context(_ s: SimSample, index: Int) -> StageContext {
        let T = s.truth
        let meta = VideoMeta(fps: T.fps, width: T.width, height: T.height, nFrames: T.n,
                             timestamps: (0..<T.n).map { Double($0) / T.fps })
        return StageContext(sample: "seed\(s.seed)", index: index, input: .synthetic, meta: meta, camera: s.camera,
                            doubles: T.doubles)
    }

    /// `run_synthetic` (stages oracle / e2e).
    public func runSynthetic(_ samples: [SimSample]) async -> PyDict {
        var info: [PyValue] = []
        for (k, s) in samples.enumerated() {
            let T = s.truth
            info.append(.dict(["seed": .int(s.seed), "frames": .int(T.n), "shots": .int(T.nShots),
                               "rallies": .int(T.nRallies), "doubles": .bool(T.doubles)]))
            if options.stage == "oracle" {
                await evalOracle(s, index: k, offset: k * benchGap)
            } else {
                await evalChain(s, index: k, offset: k * benchGap)
            }
        }
        return ["samples": .list(info)]
    }

    // MARK: real (run_real)

    public func runReal(_ samples: [RealSample]) async throws -> PyDict {
        let detectorOn = options.stage == "real"
        let oracleOn = options.stage == "real_oracle" || !options.noRealOracle
        var clipsInfo: [PyValue] = [], perClip: [PyValue] = []
        for (ci, s) in samples.enumerated() {
            let n = s.meta.nFrames
            let mask = (s.dropMask == nil || options.allFrames) ? [Bool](repeating: true, count: n) : s.dropMask!
            var ci_ = s.clip.infoDict
            ci_["frames"] = .int(n); ci_["visible"] = .int(s.truthVis.lazy.filter { $0 }.count)
            ci_["eval_frames"] = .int(mask.lazy.filter { $0 }.count)
            clipsInfo.append(.dict(ci_))
            let size = s.meta.size
            var row: PyDict = ["clip": .str(s.clip.key)]
            let ctx = StageContext(sample: s.clip.clip, index: ci, input: .detector, meta: s.meta, video: s.clip.video)
            if detectorOn {
                if !stages.available.contains("detector") {
                    pool.skip("real_detector", stages.missing["detector"]?.string ?? "not available")
                } else {
                    var cands: [[ShuttleCandidate]]? = nil
                    do {
                        cands = try await stages.detect(ctx)
                    } catch let e as RecordedStageError {
                        pool.appendError("real_detector", e.message)
                    } catch {
                        pool.error("real_detector", sample: s.clip.clip, error)
                    }
                    for sc in stages.scalars(ctx) { pool.addScalar(sc.stage, sc.key, sc.value) }
                    if let cands {
                        guard cands.count == n else {
                            throw MetricsError.lengthMismatch("\(s.clip.key): \(cands.count) candidate frames vs \(n)")
                        }
                        let top = Metrics.candidatesToXY(cands, scoreThresh: options.scoreThresh).xy
                        pool.addFrames("real_detector", pred: top, truth: s.truthXY, vis: s.truthVis, mask: mask)
                        let r = try Metrics.detectionMetrics(top, s.truthXY, vis: s.truthVis, tolPx: 4, mask: mask,
                                                             tracknetInt: true, imageSize: size)
                        row["det_f1_tn4"] = r["f1"]
                        pool.addLists("real_detector") { $0.tn4 = ($0.tn4 ?? []) + [.init(pred: top, truth: s.truthXY, vis: s.truthVis, mask: mask, size: size)] }
                        if let track = await call("linker", s.clip.clip, { try await self.stages.linker(ctx, candidates: cands) }) {
                            pool.addFrames("real_linker", pred: track.xy, truth: s.truthXY, vis: s.truthVis,
                                           interp: track.interpolated, mask: mask)
                            pool.addLists("real_linker") { $0.tn4 = ($0.tn4 ?? []) + [.init(pred: track.xy, truth: s.truthXY, vis: s.truthVis, mask: mask, size: size)] }
                            row["link_f1_tn4"] = try Metrics.detectionMetrics(track.xy, s.truthXY, vis: s.truthVis, tolPx: 4, mask: mask,
                                                                              tracknetInt: true, imageSize: size)["f1"]
                        }
                    }
                }
            }
            if oracleOn {
                guard let cands = s.oracleCandidates else {
                    throw StageError("real_oracle needs the corrupt_px candidates of \(s.clip.key) (numpy RNG): pass a bench bundle")
                }
                var octx = ctx
                octx.input = .oracle
                pool.addFrames("real_oracle_detections", pred: Metrics.candidatesToXY(cands).xy, truth: s.truthXY,
                               vis: s.truthVis, mask: mask)
                if let track = await call("linker", s.clip.clip, { try await self.stages.linker(octx, candidates: cands) }) {
                    pool.addFrames("real_oracle_linker", pred: track.xy, truth: s.truthXY, vis: s.truthVis,
                                   interp: track.interpolated, mask: mask)
                    pool.addLists("real_oracle_linker") { $0.tn4 = ($0.tn4 ?? []) + [.init(pred: track.xy, truth: s.truthXY, vis: s.truthVis, mask: mask, size: size)] }
                }
            }
            perClip.append(.dict(row))
        }
        return ["clips": .list(clipsInfo), "per_clip": .list(perClip)]
    }

    // MARK: main

    /// benchmark `main` minus argument parsing: run, pool, report, printout.
    public func run(_ samples: BenchSamples, detectorName: String? = nil, started: Date = Date()) async throws -> BenchReport {
        let t0 = DispatchTime.now().uptimeNanoseconds
        var info: PyDict
        switch samples {
        case .synthetic(let s): info = await runSynthetic(s)
        case .real(let r): info = try await runReal(r)
        }
        let tol = options.tolPx
        if let fixed = stages.fixedSeconds {                     // recorded run: Python's timings
            pool.seconds = fixed
        }
        var results = try pool.results(tolPx: tol, tolFrames: options.tolFrames)
        if options.isReal { try pool.addTN4(to: &results) }
        if options.stage == "real" || options.stage == "video" { info["detector"] = .str(detectorName ?? options.detector) }
        let wall = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        let created = BenchReport.isoNow(started)
        var report = PyDict()
        report["stage"] = .str(options.stage)
        report["created"] = .str(created)
        report["args"] = .dict(options.args)
        report["tol_px"] = .float(tol)
        report["available"] = .strings(NP.pySorted(stages.available))
        report["missing"] = .dict(stages.missing)
        report["info"] = .dict(info)
        report["results"] = .dict(results)
        report["wall_seconds"] = .float(PyFormat.round(wall, 2))
        return BenchReport(report: report)
    }
}

// MARK: - Report

public struct BenchReport: Sendable {
    public var report: PyDict

    public var results: PyDict { report["results"]?.dict ?? PyDict() }
    public var stage: String { report["stage"]?.string ?? "" }

    /// What benchmark.py prints before "wrote …".
    public var text: String {
        var lines: [String] = []
        let tol = report["tol_px"]?.double ?? .nan
        let tolFrames = report["args"]?["tol_frames"]?.int ?? 3
        let wall = report["wall_seconds"].map(PyFormat.str) ?? "0.0"
        lines.append("== benchmark --stage \(stage)  (tol \(PyFormat.fixed(tol, 1)) px, ±\(tolFrames) frames, \(wall) s)")
        let keys = Metrics.tableKeys + ["tn4_f1", "type_coverage", "inout_accuracy", "inout_accuracy_close",
                                        "keypoint_err_px_mean", "success_mean", "frame_count_ok_mean", "fps_mean", "n_errors"]
        lines.append(Metrics.formatTable(results, keys: keys))
        for (st, r) in results {
            for e in (r["errors"]?.list ?? []).prefix(2) { lines.append("  [\(st)] \(PyFormat.str(e))") }
        }
        if stage == "real" {
            for row in report["info"]?["per_clip"]?.list ?? [] {
                guard let d = row.dict else { continue }
                lines.append("  " + d.map { k, v -> String in
                    if case .float(let f) = v { return "\(k)=\(PyFormat.fixed(f, 3))" }
                    return "\(k)=\(PyFormat.str(v))"
                }.joined(separator: "  "))
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// `datetime.now().isoformat(timespec="seconds")` (local time).
    static func isoNow(_ d: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return f.string(from: d)
    }

    /// Write `<out>/<YYYYmmdd_HHMMSS>_<stage>_swift.json` (json.dumps(indent=1)); returns the path.
    public func write(to dir: URL, date: Date = Date()) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd_HHmmss"
        let url = dir.appendingPathComponent("\(f.string(from: date))_\(stage)_swift.json")
        try PyValue.dict(report).pythonJSON(indent: 1).write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}

// MARK: - Parity check against a Python report

public struct ReportDiff: Sendable {
    public var mismatches: [String] = []
    public var maxRelDiff: Double = 0
    public var nFloats = 0
    public var nExactFloats = 0
    public var ok: Bool { mismatches.isEmpty }
}

public enum ReportCompare {
    /// Compare `actual` to `expected` (a Python report): same keys in the same order,
    /// same types (int vs float, None ≡ non-finite float), floats within `relTol`.
    /// `skip` = top-level keys that are run-specific (created, wall_seconds, …).
    public static func compare(_ actual: PyValue, _ expected: PyValue, relTol: Double = 1e-12,
                               skip: Set<String> = ["created", "wall_seconds"]) -> ReportDiff {
        var d = ReportDiff()
        walk(actual, expected, "", relTol, skip, &d, top: true)
        return d
    }

    static func walk(_ a: PyValue, _ e: PyValue, _ path: String, _ tol: Double, _ skip: Set<String>,
                     _ d: inout ReportDiff, top: Bool = false) {
        func bad(_ m: String) { if d.mismatches.count < 50 { d.mismatches.append("\(path.isEmpty ? "<root>" : path): \(m)") } }
        switch (a, e) {
        case (.dict(let x), .dict(let y)):
            let xk = x.keys.filter { !(top && skip.contains($0)) }, yk = y.keys.filter { !(top && skip.contains($0)) }
            if xk != yk { bad("keys \(xk) != expected \(yk)") }
            for k in yk where x.contains(k) { walk(x[k]!, y[k]!, path + "." + k, tol, skip, &d) }
        case (.list(let x), .list(let y)):
            if x.count != y.count { bad("length \(x.count) != \(y.count)"); return }
            for (i, (p, q)) in zip(x, y).enumerated() { walk(p, q, "\(path)[\(i)]", tol, skip, &d) }
        case (.float(let x), .float(let y)):
            d.nFloats += 1
            if !x.isFinite || !y.isFinite {
                if x.isFinite != y.isFinite || (x.isNaN != y.isNaN) { bad("\(x) != \(y)") } else { d.nExactFloats += 1 }
                return
            }
            if x == y { d.nExactFloats += 1; return }
            let r = abs(x - y) / max(abs(y), 1e-300)
            d.maxRelDiff = max(d.maxRelDiff, r)
            if r > tol { bad("\(x) != \(y) (rel \(r))") }
        case (.float(let x), .none), (.none, .float(let x)):   // Python wrote NaN as null
            d.nFloats += 1
            if x.isFinite { bad("\(x) vs None") } else { d.nExactFloats += 1 }
        default:
            if a != e { bad("\(PyFormat.str(a)) != expected \(PyFormat.str(e))") }
        }
    }
}
