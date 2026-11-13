import Foundation
import ShuttlCore
import ShuttlLive
import ShuttlHits
import simd

// Hit-detector evaluation: read one match's JSON (written by dump.py from the
// extracted .npz), run the app's real `LiveStages.detectHits`, and write the
// detected contacts back out.  Scoring against the ShuttleSet labels is done in
// score.py; this binary only runs the detector so the numbers come from the
// shipping code path, not a re-implementation.
//
// Usage: hitseval <in.json> <out.json> [--threshold <logodds>] [--no-players] [--no-camera]

// MARK: - Input schema (mirrors dump.py)

struct MatchIn: Decodable {
    struct Meta: Decodable {
        var fps: Double; var width: Int; var height: Int; var nFrames: Int; var timestamps: [Double]
        enum CodingKeys: String, CodingKey { case fps, width, height, nFrames = "n_frames", timestamps }
    }
    struct Rally: Decodable {
        var set: Int, rally: Int
        var rowStart: Int, rowStop: Int
        var videoStart: Int, videoStop: Int
        var hitFrames: [Int]
        var camOk: Bool
        var camConf: Double?
        var camK: [[Double]]?
        var camR: [[Double]]?
        var camT: [Double]?
        enum CodingKeys: String, CodingKey {
            case set, rally
            case rowStart = "row_start", rowStop = "row_stop"
            case videoStart = "video_start", videoStop = "video_stop"
            case hitFrames = "hit_frames", camOk = "cam_ok", camConf = "cam_conf"
            case camK = "cam_K", camR = "cam_R", camT = "cam_t"
        }
    }
    var source: String
    var labelDir: String
    var frame: [Int]
    var meta: Meta
    var track: Track2D
    var players: [PlayerObservation]
    var rallies: [Rally]

    enum CodingKeys: String, CodingKey {
        case source, frame, meta, track, players, rallies, labelDir = "label_dir"
    }
}

// MARK: - Output schema

struct HitOut: Encodable {
    var row: Int            // row index into the match's track arrays
    var frame: Int          // video frame id (what the labels use)
    var t: Double
    var confidence: Double
    var hitter: String
    var playerTrackID: Int?
    var source: String
}

struct RallyOut: Encodable {
    var set: Int, rally: Int
    var rowStart: Int, rowStop: Int
    var videoStart: Int, videoStop: Int
    var hitFrames: [Int]
    var camUsed: Bool
    var playerCount: Int
    var hits: [HitOut]
}

struct MatchOut: Encodable {
    var source: String
    var labelDir: String
    var threshold: Double
    var usedPlayers: Bool
    var usedCamera: Bool
    var rallies: [RallyOut]
}

// MARK: - Helpers

func mat3(_ v: [[Double]]) -> simd_double3x3 {
    simd_double3x3(rows: [SIMD3(v[0][0], v[0][1], v[0][2]),
                          SIMD3(v[1][0], v[1][1], v[1][2]),
                          SIMD3(v[2][0], v[2][1], v[2][2])])
}

func argValue(_ name: String, _ args: [String]) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

// MARK: - Main

let args = Array(CommandLine.arguments.dropFirst())
let positional = args.filter { !$0.hasPrefix("--") && args.firstIndex(of: $0).map { i in i == 0 || !args[i - 1].hasPrefix("--") } ?? true }
guard positional.count >= 2 else {
    FileHandle.standardError.write(Data("usage: hitseval <in.json> <out.json> [--threshold L] [--no-players] [--no-camera]\n".utf8))
    exit(2)
}
let inURL = URL(fileURLWithPath: positional[0])
let outURL = URL(fileURLWithPath: positional[1])
let usePlayers = !args.contains("--no-players")
let useCamera = !args.contains("--no-camera")

// --threshold is the decoder's `min_hit_logodds`.  It is applied after the
// Viterbi pass, which does not depend on it, so one run at a low cut carries
// every higher cut with it exactly.
let threshold = argValue("--threshold", args).flatMap(Double.init) ?? HitConfig().minHitLogodds

// No key strategy: Track2D / PlayerObservation / VideoMeta define their own
// CodingKeys, so a global snake_case conversion would break them.
let decoder = JSONDecoder()

let data = try Data(contentsOf: inURL)
let match = try decoder.decode(MatchIn.self, from: data)

let meta = VideoMeta(fps: match.meta.fps, width: match.meta.width, height: match.meta.height,
                     nFrames: match.meta.nFrames, timestamps: match.meta.timestamps)

// frame id -> row index, so PlayerObservation.frame (a video frame) can be
// relabelled to the row the track arrays are indexed by.
var rowOfFrame: [Int: Int] = [:]
rowOfFrame.reserveCapacity(match.frame.count * 2)
for (i, f) in match.frame.enumerated() { rowOfFrame[f] = i }

var playersByRow: [Int: [PlayerObservation]] = [:]
if usePlayers {
    for p in match.players {
        guard let row = rowOfFrame[p.frame] else { continue }
        playersByRow[row, default: []].append(p)
    }
}

var cfg = HitConfig()
cfg.minHitLogodds = threshold

var out = MatchOut(source: match.source, labelDir: match.labelDir, threshold: threshold,
                   usedPlayers: usePlayers, usedCamera: useCamera, rallies: [])

for r in match.rallies {
    let lo = r.rowStart, hi = r.rowStop
    let n = match.track.n
    guard lo >= 0, lo < hi, hi <= n else { continue }

    // The detector indexes everything by one frame number, so the per-rally
    // slice is a self-consistent track of its own (the rallies were extracted
    // with the same 2 s / 6 s padding the labels use).
    let slice = Track2D(t: Array(match.track.t[lo..<hi]),
                        xy: Array(match.track.xy[lo..<hi]),
                        conf: Array(match.track.conf[lo..<hi]),
                        visible: Array(match.track.visible[lo..<hi]),
                        interpolated: Array(match.track.interpolated[lo..<hi]),
                        rawXY: match.track.rawXY.map { Array($0[lo..<hi]) })

    var camera: CameraModel? = nil
    if useCamera, r.camOk, let K = r.camK, let R = r.camR, let t = r.camT {
        camera = CameraModel(K: mat3(K), R: mat3(R), t: SIMD3(t[0], t[1], t[2]),
                             imageSize: ImageSize(width: meta.width, height: meta.height),
                             confidence: r.camConf ?? 1.0)
    }

    var players: [PlayerObservation] = []
    if usePlayers {
        for i in lo..<hi {
            if let ps = playersByRow[i] {
                for p in ps {
                    players.append(PlayerObservation(frame: i - lo, trackID: p.trackID, bbox: p.bbox,
                                                     score: p.score, side: p.side,
                                                     footWorld: p.footWorld, keypoints: p.keypoints))
                }
            }
        }
    }

// LiveStages().detectHits(track:meta:camera:players:) is a two-line wrapper —
// requireConsistent + timed() then detectHits(...) with defaults
// (LiveStages.swift:203) — and LiveStages cannot be constructed without a video
// detector.  detectHits is the same public entry point (same default rescorer:
// HitModelRegistry.installed, which is nil here), called with an explicit
// HitConfig so the threshold can be set from the command line.
let events = detectHits(track: slice, meta: meta, camera: camera,
                        players: players.isEmpty ? nil : players,
                        config: cfg)

    var hits: [HitOut] = []
    for e in events {
        let vf = (0..<match.frame.count).contains(lo + e.frame) ? match.frame[lo + e.frame] : -1
        hits.append(HitOut(row: lo + e.frame, frame: vf, t: e.t, confidence: e.confidence,
                           hitter: e.hitter.rawValue, playerTrackID: e.playerTrackID, source: e.source))
    }
    out.rallies.append(RallyOut(set: r.set, rally: r.rally, rowStart: lo, rowStop: hi,
                                videoStart: r.videoStart, videoStop: r.videoStop,
                                hitFrames: r.hitFrames, camUsed: camera != nil,
                                playerCount: players.count, hits: hits))
}

let enc = JSONEncoder()
enc.outputFormatting = [.withoutEscapingSlashes]
try enc.encode(out).write(to: outURL, options: .atomic)

let totalHits = out.rallies.reduce(0) { $0 + $1.hits.count }
let totalGT = out.rallies.reduce(0) { $0 + $1.hitFrames.count }
FileHandle.standardError.write(Data("\(match.source): \(out.rallies.count) rallies, \(totalHits) hits detected, \(totalGT) labelled\n".utf8))
