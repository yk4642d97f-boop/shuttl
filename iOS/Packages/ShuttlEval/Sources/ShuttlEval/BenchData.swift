import Foundation
import ShuttlCore

// Inputs of the Swift benchmark (port of backend/scripts/benchmark.py's data side):
// real TrackNetV2 Test clips + labels, and "bench bundles" written by
// backend/scripts/swift_fixtures/eval.py (benchmark.py run in-process with every
// stage call recorded: truth, candidates, each stage's output or exception).

// MARK: - Candidates

/// Per-frame candidate lists; decodes `[x, y, score]` triples (bundle form) or
/// `{"x":…, "y":…, "score":…}` objects. A NaN score drops the candidate, like
/// benchmark `_arr_to_cands` (never produced by the exporter, kept for safety).
public struct CandidateFrames: Sendable, Codable, Equatable {
    public var frames: [[ShuttleCandidate]]
    public init(_ frames: [[ShuttleCandidate]]) { self.frames = frames }

    public init(from decoder: Decoder) throws {
        var outer = try decoder.unkeyedContainer()
        var out: [[ShuttleCandidate]] = []
        while !outer.isAtEnd {
            var row: [ShuttleCandidate] = []
            var inner = try outer.nestedUnkeyedContainer()
            while !inner.isAtEnd {
                if let v = try? inner.decode([Double?].self) {
                    guard v.count >= 3, let x = v[0], let y = v[1], let s = v[2], s.isFinite else { continue }
                    row.append(ShuttleCandidate(x: x, y: y, score: s))
                } else {
                    let c = try inner.decode(ShuttleCandidate.self)
                    if c.score.isFinite { row.append(c) }
                }
            }
            out.append(row)
        }
        frames = out
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        for row in frames { try c.encode(row.map { [$0.x, $0.y, $0.score] }) }
    }
}

// MARK: - Stage outputs (what the metrics consume)

/// A hit event as recorded (raw hitter string, so unexpected labels score as Python does).
public struct BenchHit: Sendable, Equatable, Decodable {
    public var frame: Int
    public var hitter: String
    enum CodingKeys: String, CodingKey { case frame, hitter }
    public init(frame: Int, hitter: String) { self.frame = frame; self.hitter = hitter }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        frame = try c.decode(Int.self, forKey: .frame)
        hitter = try c.decodeIfPresent(String.self, forKey: .hitter) ?? "unknown"
    }
    public var eval: EvalHit { EvalHit(frame: frame, hitter: hitter) }
}

/// A reconstructed flight plus (oracle bundles) the Python `call_in_out(...)['call']`
/// for the serve / rally cases; a nil call means it raised (benchmark `_call` falls back).
public struct BenchSegment: Sendable, Equatable, Decodable {
    public var segment: EvalSegment
    public var callServe: String?
    public var callRally: String?
    public var hasCalls: Bool

    public init(segment: EvalSegment, callServe: String? = nil, callRally: String? = nil, hasCalls: Bool = false) {
        self.segment = segment; self.callServe = callServe; self.callRally = callRally; self.hasCalls = hasCalls
    }

    struct Calls: Decodable { var serve: String?; var rally: String? }
    enum CodingKeys: String, CodingKey { case calls }

    public init(from decoder: Decoder) throws {
        segment = try EvalSegment(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let calls = try c.decodeIfPresent(Calls.self, forKey: .calls)
        hasCalls = calls != nil
        callServe = calls?.serve
        callRally = calls?.rally
    }
}

/// The part of a classified RallyV2 the shots scoring reads: each shot's hit frame and type.
public struct BenchTypedRally: Sendable, Equatable, Decodable {
    public struct Shot: Sendable, Equatable, Decodable {
        public var frame: Int
        public var shotType: String
        enum CodingKeys: String, CodingKey { case hit, shotType = "shot_type" }
        struct Hit: Decodable { var frame: Int }
        public init(frame: Int, shotType: String) { self.frame = frame; self.shotType = shotType }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            frame = try c.decode(Hit.self, forKey: .hit).frame
            shotType = try c.decodeIfPresent(String.self, forKey: .shotType) ?? "unknown"
        }
    }
    public var shots: [Shot]
    public init(shots: [Shot]) { self.shots = shots }
    enum CodingKeys: String, CodingKey { case shots }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        shots = try c.decodeIfPresent([Shot].self, forKey: .shots) ?? []
    }
    public init(_ r: RallyV2) { shots = r.shots.map { Shot(frame: $0.hit.frame, shotType: $0.shotType.rawValue) } }
}

/// One recorded `Runner.call`: whether the stage function existed, the exception text
/// Python's Pool recorded (if any) and the (decoded) output, nil = None.
public struct RecordedCall<T: Decodable & Sendable>: Sendable, Decodable {
    public var present: Bool
    public var error: String?
    public var out: T?
}

extension RecordedCall: Equatable where T: Equatable {}

/// A recorded linker output: eval.py `_track` (xy, visible, interpolated).
struct BenchTrack: Decodable, Sendable {
    let xy: [SIMD2<Double>], visible: [Bool], interpolated: [Bool]
    enum CodingKeys: String, CodingKey { case xy, visible, interpolated }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        xy = (try c.decodeNaNableRows(forKey: .xy) ?? []).map(SIMD2.init(row:))
        visible = try c.decode([Bool].self, forKey: .visible)
        interpolated = try c.decode([Bool].self, forKey: .interpolated)
        guard visible.count == xy.count, interpolated.count == xy.count else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription:
                "recorded track arrays differ in length (xy \(xy.count), visible \(visible.count), interpolated \(interpolated.count))"))
        }
    }
    var track: Track2D {
        Track2D(t: xy.indices.map(Double.init), xy: xy, conf: Array(repeating: 0, count: xy.count),
                visible: visible, interpolated: interpolated)
    }
}

public struct RecordedStages: Sendable, Decodable {
    public var linker: [RecordedCall<Track2D>]
    public var hits: [RecordedCall<[BenchHit]>]
    public var reconstruct3d: [RecordedCall<[BenchSegment]>]
    public var rally: [RecordedCall<[EvalRally]>]
    public var shots: [RecordedCall<BenchTypedRally>]

    enum CodingKeys: String, CodingKey { case linker, hits, reconstruct3d, rally, shots }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // eval.py `_track` records only the scored fields (xy, visible, interpolated), not a full
        // Track2D, so decode those and give the track a frame clock + zero confidence.
        linker = (try c.decodeIfPresent([RecordedCall<BenchTrack>].self, forKey: .linker) ?? []).map {
            RecordedCall(present: $0.present, error: $0.error, out: $0.out?.track)
        }
        hits = try c.decodeIfPresent([RecordedCall<[BenchHit]>].self, forKey: .hits) ?? []
        reconstruct3d = try c.decodeIfPresent([RecordedCall<[BenchSegment]>].self, forKey: .reconstruct3d) ?? []
        rally = try c.decodeIfPresent([RecordedCall<[EvalRally]>].self, forKey: .rally) ?? []
        shots = try c.decodeIfPresent([RecordedCall<BenchTypedRally>].self, forKey: .shots) ?? []
    }
    public init() { linker = []; hits = []; reconstruct3d = []; rally = []; shots = [] }
}

// MARK: - Truth

/// Synthetic ground truth of one sample (benchmark.py's view of a SimRally).
public struct SimTruth: Sendable, Decodable {
    public var seed: Int
    public var n: Int
    public var fps: Double
    public var width: Int
    public var height: Int
    public var doubles: Bool
    public var nShots: Int
    public var nRallies: Int
    /// `sim.image_truth(occluded=True)` px / mask (tracks are scored on these).
    public var trkPx: [SIMD2<Double>]
    public var trkVis: [Bool]
    /// `sim.image_truth(occluded=False)` mask; its px = trkPx where detVis (else NaN).
    public var detVis: [Bool]
    public var hits: [EvalHit]
    public var shots: [EvalSegment]                 // benchmark `_true_shot_d`
    public var rallies: [EvalRally]                 // benchmark `_rally_d`
    public var finalCalls: [FinalCall]?             // oracle: each rally's last shot
    public var rallyTypes: [[String]]?              // oracle: truth_rallies shot types
    public var camera: CameraModel?                 // the true camera

    public struct FinalCall: Sendable, Decodable, Equatable {
        public var frame: Int
        public var isServe: Bool
        public var inOut: String
        public var margin: Double?
        enum CodingKeys: String, CodingKey { case frame, isServe = "is_serve", inOut = "in_out", margin }
        public init(frame: Int, isServe: Bool, inOut: String, margin: Double?) {
            self.frame = frame; self.isServe = isServe; self.inOut = inOut; self.margin = margin
        }
    }

    public var detPx: [SIMD2<Double>] { zip(trkPx, detVis).map { $1 ? $0 : SIMD2(.nan, .nan) } }
    public var imageSize: ImageSize { ImageSize(width: width, height: height) }

    enum CodingKeys: String, CodingKey {
        case seed, n, fps, width, height, doubles, nShots = "n_shots", nRallies = "n_rallies"
        case trkPx = "trk_px", trkVis = "trk_vis", detVis = "det_vis", hits, shots, rallies
        case finalCalls = "final_calls", rallyTypes = "rally_types", camera
    }

    public init(seed: Int, n: Int, fps: Double, width: Int, height: Int, doubles: Bool, nShots: Int, nRallies: Int,
                trkPx: [SIMD2<Double>], trkVis: [Bool], detVis: [Bool], hits: [EvalHit], shots: [EvalSegment],
                rallies: [EvalRally], finalCalls: [FinalCall]? = nil, rallyTypes: [[String]]? = nil,
                camera: CameraModel? = nil) {
        self.seed = seed; self.n = n; self.fps = fps; self.width = width; self.height = height
        self.doubles = doubles; self.nShots = nShots; self.nRallies = nRallies; self.trkPx = trkPx
        self.trkVis = trkVis; self.detVis = detVis; self.hits = hits; self.shots = shots; self.rallies = rallies
        self.finalCalls = finalCalls; self.rallyTypes = rallyTypes; self.camera = camera
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        seed = try c.decode(Int.self, forKey: .seed)
        n = try c.decode(Int.self, forKey: .n)
        fps = try c.decode(Double.self, forKey: .fps)
        width = try c.decode(Int.self, forKey: .width)
        height = try c.decode(Int.self, forKey: .height)
        doubles = try c.decode(Bool.self, forKey: .doubles)
        nShots = try c.decode(Int.self, forKey: .nShots)
        nRallies = try c.decode(Int.self, forKey: .nRallies)
        trkPx = (try c.decodeNaNableRows(forKey: .trkPx) ?? []).map(SIMD2.init(row:))
        trkVis = try c.decode([Bool].self, forKey: .trkVis)
        detVis = try c.decode([Bool].self, forKey: .detVis)
        hits = try c.decode([EvalHit].self, forKey: .hits)
        shots = try c.decode([EvalSegment].self, forKey: .shots)
        rallies = try c.decode([EvalRally].self, forKey: .rallies)
        finalCalls = try c.decodeIfPresent([FinalCall].self, forKey: .finalCalls)
        rallyTypes = try c.decodeIfPresent([[String]].self, forKey: .rallyTypes)
        camera = try c.decodeIfPresent(CameraModel.self, forKey: .camera)
    }
}

// MARK: - Bundle

/// A recorded benchmark.py run (see backend/scripts/swift_fixtures/eval.py).
public struct BenchBundle: Sendable {
    public struct Sample: Sendable, Decodable {
        public var kind: String                     // "sim" | "real"
        public var seed: Int?
        public var truth: SimTruth?
        public var match: String?
        public var clip: String?
        public var n: Int?
        public var fps: Double?
        public var width: Int?
        public var height: Int?
        public var labels: String?
        public var truthXY: [SIMD2<Double>]?        // Python load_label_csv output (cross-check)
        public var truthVis: [Bool]?
        public var dropMask: [Bool]?
        public var candidates: CandidateFrames?     // detector / corrupt_detections output
        public var detectorError: String?
        public var oracleCandidates: CandidateFrames?
        public var stages: RecordedStages
        public var scalars: [(stage: String, key: String, value: Double)]

        enum CodingKeys: String, CodingKey {
            case kind, seed, truth, match, clip, n, fps, width, height, labels, truthXY = "truth_xy"
            case truthVis = "truth_vis", dropMask = "drop_mask", candidates, detectorError = "detector_error"
            case oracleCandidates = "oracle_candidates", stages, scalars
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            kind = try c.decode(String.self, forKey: .kind)
            seed = try c.decodeIfPresent(Int.self, forKey: .seed)
            truth = try c.decodeIfPresent(SimTruth.self, forKey: .truth)
            match = try c.decodeIfPresent(String.self, forKey: .match)
            clip = try c.decodeIfPresent(String.self, forKey: .clip)
            n = try c.decodeIfPresent(Int.self, forKey: .n)
            fps = try c.decodeIfPresent(Double.self, forKey: .fps)
            width = try c.decodeIfPresent(Int.self, forKey: .width)
            height = try c.decodeIfPresent(Int.self, forKey: .height)
            labels = try c.decodeIfPresent(String.self, forKey: .labels)
            truthXY = try c.decodeNaNableRows(forKey: .truthXY)?.map(SIMD2.init(row:))
            truthVis = try c.decodeIfPresent([Bool].self, forKey: .truthVis)
            dropMask = try c.decodeIfPresent([Bool].self, forKey: .dropMask)
            candidates = try c.decodeIfPresent(CandidateFrames.self, forKey: .candidates)
            detectorError = try c.decodeIfPresent(String.self, forKey: .detectorError)
            oracleCandidates = try c.decodeIfPresent(CandidateFrames.self, forKey: .oracleCandidates)
            stages = try c.decodeIfPresent(RecordedStages.self, forKey: .stages) ?? RecordedStages()
            let raw = try c.decodeIfPresent([[JSONValue]].self, forKey: .scalars) ?? []
            scalars = raw.compactMap { r in
                guard r.count == 3, let s = r[0].string, let k = r[1].string else { return nil }
                return (s, k, r[2].double ?? .nan)
            }
        }
    }

    public var argv: [String]
    public var stage: String
    public var hasCallInOut: Bool
    /// Python pool.seconds per stage (carried into the results: timings can't be replayed).
    public var poolSeconds: [String: Double]
    /// Python's report (dict order and int/float kept) and printed output.
    public var report: PyValue
    public var stdout: String
    public var samples: [Sample]

    public var args: PyDict { report["args"]?.dict ?? PyDict() }
    public var available: [String] { report["available"]?.list?.compactMap(\.string) ?? [] }
    public var missing: PyDict { report["missing"]?.dict ?? PyDict() }

    struct Raw: Decodable {
        var argv: [String]
        var stage: String
        var hasCallInOut: Bool
        var poolSeconds: [String: Double]
        var stdout: String
        var samples: [Sample]
        enum CodingKeys: String, CodingKey {
            case argv, stage, hasCallInOut = "has_call_in_out", poolSeconds = "pool_seconds", stdout, samples
        }
    }

    public static func load(_ url: URL) throws -> BenchBundle {
        let data = try Data(contentsOf: url)
        let raw = try ShuttlJSON.decoder().decode(Raw.self, from: data)
        let py = try PyJSON.parse(data)
        return BenchBundle(argv: raw.argv, stage: raw.stage, hasCallInOut: raw.hasCallInOut, poolSeconds: raw.poolSeconds,
                           report: py["report"] ?? .none, stdout: raw.stdout, samples: raw.samples)
    }
}

// MARK: - Real TrackNetV2 Test split

public struct RealClip: Sendable, Equatable {
    public var match: String
    public var clip: String
    public var video: URL
    public var labels: URL
    public var corrected: Bool

    public var key: String { "\(match)/\(clip)" }

    /// benchmark `info["clips"]` entry prefix.
    public var infoDict: PyDict {
        ["match": .str(match), "clip": .str(clip), "video": .str(video.path), "labels": .str(labels.path),
         "corrected": .bool(corrected)]
    }
}

public enum RealData {
    /// benchmark.py `REAL_ROOT` = backend/ml/data/raw/shuttlecock_trajectory.
    public static func realRoot(backend: URL) -> URL {
        backend.appendingPathComponent("ml/data/raw/shuttlecock_trajectory")
    }

    /// `real_clips(limit)`: match*/video/*.mp4 under TrackNetV2/Test sorted like pathlib
    /// (component-wise), corrected labels preferred, clips without labels skipped.
    public static func clips(backend: URL, limit: Int? = nil) -> [RealClip] {
        let root = realRoot(backend: backend)
        let test = root.appendingPathComponent("TrackNetV2/Test")
        let fm = FileManager.default
        var vids: [(String, String, URL)] = []
        let matches = (try? fm.contentsOfDirectory(atPath: test.path)) ?? []
        for m in matches where m.hasPrefix("match") && !m.hasPrefix(".") {
            let vdir = test.appendingPathComponent(m).appendingPathComponent("video")
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: vdir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            for f in (try? fm.contentsOfDirectory(atPath: vdir.path)) ?? [] where f.hasSuffix(".mp4") && !f.hasPrefix(".") {
                vids.append((m, f, vdir.appendingPathComponent(f)))
            }
        }
        vids.sort { a, b in a.0 != b.0 ? NP.pyLess(a.0, b.0) : NP.pyLess(a.1, b.1) }
        var out: [RealClip] = []
        for (m, f, url) in vids {
            let clip = String(f.dropLast(4))
            let corr = root.appendingPathComponent("corrected/\(m)/corrected_csv/\(clip)_ball.csv")
            let orig = test.appendingPathComponent("\(m)/csv/\(clip)_ball.csv")
            let hasCorr = fm.fileExists(atPath: corr.path)
            let lab = hasCorr ? corr : orig
            if fm.fileExists(atPath: lab.path) {
                out.append(RealClip(match: m, clip: clip, video: url, labels: lab, corrected: hasCorr))
            }
        }
        if let limit, limit > 0 { return Array(out.prefix(limit)) }
        return out
    }

    /// `load_label_csv`: Frame,Visibility,X,Y → (px with NaN when invisible, visible).
    /// Frames outside [0, n) are dropped; later rows overwrite earlier ones.
    public static func loadLabels(_ url: URL, n: Int) throws -> (xy: [SIMD2<Double>], vis: [Bool]) {
        let text = try String(contentsOf: url, encoding: .utf8)
        var xy = [SIMD2<Double>](repeating: SIMD2(.nan, .nan), count: n)
        var vis = [Bool](repeating: false, count: n)
        for (k, line) in text.split(whereSeparator: \.isNewline).enumerated() where k > 0 {
            let cols = line.split(separator: ",", omittingEmptySubsequences: false).map {
                Double($0.trimmingCharacters(in: .whitespaces)) ?? .nan
            }
            guard cols.count >= 4, cols[0].isFinite else { continue }
            let f = Int(cols[0].rounded(.towardZero))
            guard f >= 0 && f < n else { continue }
            if cols[1] > 0 {
                vis[f] = true
                xy[f] = SIMD2(cols[2], cols[3])
            }
        }
        return (xy, vis)
    }

    /// `drop_mask`: TrackNetV3 corrected/drop_frame.json evaluation range [start, end), or nil.
    public static func dropMask(backend: URL, match: String, clip: String, n: Int) -> [Bool]? {
        let url = realRoot(backend: backend).appendingPathComponent("corrected/drop_frame.json")
        guard let data = try? Data(contentsOf: url), let d = try? PyJSON.parse(data) else { return nil }
        let key = "\(match.replacingOccurrences(of: "match", with: ""))_\(clip)"
        guard let st = d["start"]?[key]?.double, let en = d["end"]?[key]?.double else { return nil }
        var m = [Bool](repeating: false, count: n)
        // m[int(st):int(en)] = True with Python slice semantics
        func idx(_ v: Int) -> Int { v < 0 ? max(0, n + v) : min(v, n) }
        let a = idx(Int(st.rounded(.towardZero))), b = idx(Int(en.rounded(.towardZero)))
        if a < b { for i in a..<b { m[i] = true } }
        return m
    }
}
