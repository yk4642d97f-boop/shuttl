import Foundation
import ShuttlCore

// `shuttl-bench` — backend/scripts/benchmark.py for the Swift stack. Scores
// predictions with the Swift metrics and prints benchmark.py's table / JSON.
//
//   shuttl-bench --predictions bench_e2e.json            recorded run (bench bundle): must reproduce
//                                                        Python's report exactly (checked, exit 1 if not)
//   shuttl-bench --stage real --analysis-dir DIR         MatchAnalysis files per clip (+ candidates)
//   shuttl-bench --stage real --predictions B --analysis-dir DIR   bundle, detector/linker from DIR
//
// Sample-defining flags (--n, --seed, --clips, --max-frames, …) come from the bundle;
// scoring flags (--tol-px, --tol-frames, --score-thresh, --all-frames) may be overridden.

public enum BenchCLIError: Error, CustomStringConvertible {
    case usage(String)
    case failed(String)
    public var description: String {
        switch self { case .usage(let m): return "usage error: \(m)"; case .failed(let m): return m }
    }
}

public struct BenchCLI {
    public var backend: URL
    /// The on-device pipeline, when linked in (integration step): `shuttl-bench --stage real`
    /// without --predictions / --analysis-dir then runs it on the Test clips.
    public var liveRunner: (any AnalysisRunner)?
    public var print: (String) -> Void = { Swift.print($0, terminator: "") }

    public init(backend: URL, liveRunner: (any AnalysisRunner)? = nil) {
        self.backend = backend; self.liveRunner = liveRunner
    }

    static let valueFlags: [String: String] = [
        "--stage": "stage", "--n": "n", "--seed": "seed", "--rallies": "rallies", "--fps": "fps",
        "--image-size": "image_size", "--style": "style", "--doubles": "doubles", "--player-step": "player_step",
        "--track-noise": "track_noise", "--sigma": "sigma", "--miss-rate": "miss_rate", "--fp-rate": "fp_rate",
        "--clips": "clips", "--max-frames": "max_frames", "--detector": "detector", "--device": "device",
        "--weights-dir": "weights_dir", "--score-thresh": "score_thresh", "--cache-dir": "cache_dir",
        "--tol-px": "tol_px", "--tol-frames": "tol_frames", "--out": "out",
    ]
    static let boolFlags: [String: String] = [
        "--keep-videos": "keep_videos", "--no-cache": "no_cache", "--no-real-oracle": "no_real_oracle",
        "--all-frames": "all_frames", "--no-json": "no_json",
    ]
    static let intKeys: Set<String> = ["n", "seed", "rallies", "player_step", "clips", "max_frames", "tol_frames"]
    static let floatKeys: Set<String> = ["fps", "track_noise", "sigma", "miss_rate", "fp_rate", "score_thresh", "tol_px"]
    static let scoringKeys: Set<String> = ["tol_px", "tol_frames", "score_thresh", "all_frames", "out", "no_json"]
    static let stages = ["oracle", "e2e", "video", "real", "real_oracle"]

    public static let help = """
    usage: shuttl-bench [--stage {oracle,e2e,video,real,real_oracle}] [--predictions BUNDLE.json]
                        [--analysis-dir DIR] [--expected REPORT.json] [--no-check] [benchmark.py flags…]

    Scores recorded or Swift-produced predictions with the Swift port of metrics.py and
    prints benchmark.py's table; writes <out>/<timestamp>_<stage>_swift.json unless --no-json.
      --predictions   bench bundle from `python -m scripts.swift_fixtures.eval` (truth + every
                      stage output of one benchmark.py run); the report is checked against the
                      Python report inside it unless scoring flags were overridden or --no-check.
      --analysis-dir  MatchAnalysis JSON per clip (<match>_<clip>.json, optional
                      <match>_<clip>.candidates.json) — real stage detector/linker input.
      --expected      a benchmark.py JSON report to check against (default: the bundle's).
      --backend       backend/ directory (default: derived from the source tree).

    """

    public struct Parsed {
        public var options: BenchOptions
        public var explicit: Set<String> = []
        public var predictions: URL?
        public var analysisDir: URL?
        public var expected: URL?
        public var check = true
    }

    public func parse(_ argv: [String]) throws -> Parsed {
        var p = Parsed(options: .defaults(backend: backend))
        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < argv.count else { throw BenchCLIError.usage("\(flag) needs a value") }
            return argv[i]
        }
        while i < argv.count {
            var a = argv[i]
            var inline: String? = nil
            if a.hasPrefix("--"), let eq = a.firstIndex(of: "=") {
                inline = String(a[a.index(after: eq)...]); a = String(a[..<eq])
            }
            func val() throws -> String { if let inline { return inline }; return try value(a) }
            switch a {
            case "-h", "--help": throw BenchCLIError.usage(Self.help)
            case "--predictions": p.predictions = URL(fileURLWithPath: try val())
            case "--analysis-dir": p.analysisDir = URL(fileURLWithPath: try val())
            case "--expected": p.expected = URL(fileURLWithPath: try val())
            case "--no-check": p.check = false
            default:
                if let k = Self.boolFlags[a] {
                    p.options.args[k] = .bool(true); p.explicit.insert(k)
                } else if let k = Self.valueFlags[a] {
                    let s = try val()
                    p.options.args[k] = try Self.convert(k, s)
                    p.explicit.insert(k)
                } else {
                    throw BenchCLIError.usage("unrecognized argument \(a)\n\(Self.help)")
                }
            }
            i += 1
        }
        return p
    }

    static func convert(_ k: String, _ s: String) throws -> PyValue {
        if k == "stage" {
            guard stages.contains(s) else { throw BenchCLIError.usage("--stage must be one of \(stages)") }
            return .str(s)
        }
        if k == "doubles" { return .bool(["1", "true", "yes"].contains(s.lowercased())) }
        if k == "image_size" {
            let parts = s.lowercased().split(separator: "x").compactMap { Int($0) }
            guard parts.count == 2 else { throw BenchCLIError.usage("--image-size WxH") }
            return .list([.int(parts[0]), .int(parts[1])])
        }
        if intKeys.contains(k) {
            guard let v = Int(s) else { throw BenchCLIError.usage("--\(k) expects an int") }
            return .int(v)
        }
        if floatKeys.contains(k) {
            guard let v = Double(s) else { throw BenchCLIError.usage("--\(k) expects a float") }
            return .float(v)
        }
        return .str(s)
    }

    /// Run the CLI; returns the report (and whether it matched the Python reference).
    @discardableResult
    public func run(_ argv: [String]) async throws -> (report: BenchReport, diff: ReportDiff?) {
        let started = Date()
        let p = try parse(argv)
        var options = p.options
        if options.stage == "video" {
            throw BenchCLIError.usage("--stage video renders frames with cv2 (sim.render_video) — run it in Python (scripts/benchmark.py)")
        }
        var stages: any BenchStages
        var samples: BenchSamples
        var detectorName: String? = nil
        var expected: PyValue? = nil
        var expectedStdout: String? = nil
        var overridden = false

        if let bu = p.predictions {
            let bundle = try BenchBundle.load(bu)
            var args = bundle.args
            for k in p.explicit {
                if Self.scoringKeys.contains(k) {
                    if k != "out" && k != "no_json" && args[k] != options.args[k] { overridden = true }
                    args[k] = options.args[k]
                } else if k == "stage" {
                    if options.stage != bundle.stage {
                        throw BenchCLIError.usage("--stage \(options.stage) but the bundle recorded --stage \(bundle.stage)")
                    }
                } else if args[k] != options.args[k] {
                    throw BenchCLIError.usage("--\(k.replacingOccurrences(of: "_", with: "-")) is fixed by the bundle (recorded \(PyFormat.str(args[k] ?? .none)))")
                }
            }
            options = BenchOptions(args: args)
            let rec = RecordedBenchStages(bundle)
            if let dir = p.analysisDir {
                rec.detectorOverride = AnalysisRunnerStages(runner: MatchAnalysisFileRunner(directory: dir), stage: options.stage)
                overridden = true
            }
            stages = rec
            samples = try BenchInputs.samples(bundle: bundle, options: options, backend: backend)
            detectorName = bundle.report["info"]?["detector"]?.string
            expected = bundle.report
            expectedStdout = bundle.stdout
        } else {
            guard options.isReal else {
                throw BenchCLIError.usage("synthetic stages need a bench bundle (--predictions) until the Swift simulator is wired in")
            }
            if options.stage == "real_oracle" || !options.noRealOracle {
                if p.explicit.contains("no_real_oracle") || options.stage == "real_oracle" {
                    throw BenchCLIError.usage("real_oracle needs the numpy-RNG corrupt_px candidates: pass --predictions")
                }
                options.args["no_real_oracle"] = .bool(true)     // benchmark default runs it; impossible without a bundle
                FileHandle.standardError.write(Data("note: real_oracle skipped (needs a bench bundle)\n".utf8))
            }
            let runner: any AnalysisRunner
            if let dir = p.analysisDir { runner = MatchAnalysisFileRunner(directory: dir) }
            else if let live = liveRunner { runner = live }
            else { throw BenchCLIError.usage("no predictions: pass --predictions or --analysis-dir (no on-device runner linked)") }
            stages = AnalysisRunnerStages(runner: runner, stage: options.stage,
                                          options: AnalysisOptions(maxFrames: options.maxFrames, detector: options.detector))
            samples = .real(try await BenchInputs.discoverReal(options: options, backend: backend, analysisDir: p.analysisDir))
            detectorName = runner.name
        }

        let runner = BenchRunner(options: options, stages: stages)
        let report = try await runner.run(samples, detectorName: detectorName, started: started)
        var text = report.text
        var diff: ReportDiff? = nil
        if let e = p.expected { expected = try PyJSON.load(e); expectedStdout = nil }
        if p.check, !overridden, let expected {
            var d = ReportCompare.compare(.dict(report.report), expected)
            if let es = expectedStdout {
                let a = Self.dropHeaderTiming(text), b = Self.dropHeaderTiming(Self.stripWrote(es))
                if a != b { d.mismatches.append("printed table differs from Python's:\n--- swift\n\(a)--- python\n\(b)") }
            }
            let exact = d.nFloats == d.nExactFloats
            text += !d.ok
                ? "parity: MISMATCH (\(d.mismatches.count))\n" + d.mismatches.map { "  " + $0 }.joined(separator: "\n") + "\n"
                : exact
                ? "parity: OK — identical to the Python report (\(d.nFloats) floats, all bit-exact)\n"
                : "parity: OK within 1e-12 — \(d.nFloats - d.nExactFloats) of \(d.nFloats) floats not bit-exact (max rel diff \(String(format: "%.1e", d.maxRelDiff)))\n"
            diff = d
        }
        if !options.noJSON, let out = options.outDir {
            let url = try report.write(to: URL(fileURLWithPath: out))
            text += "wrote \(url.path)\n"
        }
        print(text)
        return (report, diff)
    }

    /// The header's wall-clock part differs run to run.
    static func dropHeaderTiming(_ s: String) -> String {
        var lines = s.components(separatedBy: "\n")
        if let first = lines.first, first.hasPrefix("== benchmark"), let r = first.range(of: " frames, ") {
            lines[0] = String(first[..<r.upperBound])
        }
        return lines.joined(separator: "\n")
    }

    static func stripWrote(_ s: String) -> String {
        s.components(separatedBy: "\n").filter { !$0.hasPrefix("wrote ") }.joined(separator: "\n")
    }
}

// MARK: - Samples from bundles / the Test split

public enum BenchInputs {
    public static func samples(bundle: BenchBundle, options: BenchOptions, backend: URL) throws -> BenchSamples {
        if options.isReal { return .real(try realSamples(bundle: bundle, options: options, backend: backend)) }
        return .synthetic(try bundle.samples.map { s in
            guard s.kind == "sim", let t = s.truth, let seed = s.seed else {
                throw BenchCLIError.failed("bundle sample is not synthetic")
            }
            return SimSample(seed: seed, truth: t, candidates: s.candidates?.frames ?? [], camera: t.camera)
        })
    }

    /// Real samples of a bundle: clip identity + frame count from the recording, truth from
    /// the Test-split labels (the Python copy when the labels are not on this machine).
    public static func realSamples(bundle: BenchBundle, options: BenchOptions, backend: URL) throws -> [RealSample] {
        let discovered = Dictionary(RealData.clips(backend: backend).map { ($0.key, $0) }, uniquingKeysWith: { a, _ in a })
        let root = RealData.realRoot(backend: backend)
        let dropFile = root.appendingPathComponent("corrected/drop_frame.json")
        return try bundle.samples.map { s in
            guard s.kind == "real", let match = s.match, let clipName = s.clip, let n = s.n else {
                throw BenchCLIError.failed("bundle sample is not a real clip")
            }
            let clip: RealClip
            if let c = discovered["\(match)/\(clipName)"] {
                clip = c
            } else {
                let vid = root.appendingPathComponent("TrackNetV2/Test/\(match)/video/\(clipName).mp4")
                let lab = s.labels.map { URL(fileURLWithPath: $0) } ?? vid
                clip = RealClip(match: match, clip: clipName, video: vid, labels: lab, corrected: lab.path.contains("/corrected/"))
            }
            let fps = s.fps ?? 30
            let meta = VideoMeta(fps: fps, width: s.width ?? 1280, height: s.height ?? 720, nFrames: n,
                                 timestamps: (0..<n).map { Double($0) / fps })
            var xy = s.truthXY ?? [], vis = s.truthVis ?? []
            if FileManager.default.fileExists(atPath: clip.labels.path) {
                (xy, vis) = try RealData.loadLabels(clip.labels, n: n)
            }
            let dm = FileManager.default.fileExists(atPath: dropFile.path)
                ? RealData.dropMask(backend: backend, match: match, clip: clipName, n: n) : s.dropMask
            return RealSample(clip: clip, meta: meta, truthXY: xy, truthVis: vis, dropMask: dm,
                              oracleCandidates: s.oracleCandidates?.frames)
        }
    }

    /// Real samples without a bundle: the Test split (frame count from the MatchAnalysis
    /// meta when available, else a video probe; capped by --max-frames).
    public static func discoverReal(options: BenchOptions, backend: URL, analysisDir: URL?) async throws -> [RealSample] {
        var out: [RealSample] = []
        let clips = RealData.clips(backend: backend, limit: options.clips)
        guard !clips.isEmpty else { throw BenchCLIError.failed("no TrackNetV2 Test clips under \(RealData.realRoot(backend: backend).path)") }
        for c in clips {
            var meta: VideoMeta
            if let dir = analysisDir, let a = try? ShuttlJSON.load(MatchAnalysis.self,
                                                                     from: dir.appendingPathComponent("\(c.match)_\(c.clip).json")) {
                meta = a.meta
            } else {
                meta = try await VideoSource.open(c.video).meta
            }
            let n = options.maxFrames.map { min(meta.nFrames, $0) } ?? meta.nFrames
            meta = VideoMeta(fps: meta.fps, width: meta.width, height: meta.height, nFrames: n,
                             timestamps: Array(meta.timestamps.prefix(n)), rotationDeg: meta.rotationDeg)
            let (xy, vis) = try RealData.loadLabels(c.labels, n: n)
            out.append(RealSample(clip: c, meta: meta, truthXY: xy, truthVis: vis,
                                  dropMask: RealData.dropMask(backend: backend, match: c.match, clip: c.clip, n: n)))
        }
        return out
    }
}
