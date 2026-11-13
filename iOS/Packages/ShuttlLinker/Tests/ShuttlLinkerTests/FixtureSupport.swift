import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

// Fixture model for backend/scripts/swift_fixtures/linker.py (keys = Python names,
// NaN written as null).

struct FxMeta: Decodable {
    let fps: Double?                 // null for inf / nan (FxCase.fps_str has the value)
    let width: Int
    let height: Int
    let n_frames: Int
    let rotation_deg: Int
    let timestamps: [Double?]

    var meta: VideoMeta {
        VideoMeta(fps: fps ?? .nan, width: width, height: height, nFrames: n_frames,
                  timestamps: timestamps.map { $0 ?? .nan }, rotationDeg: rotation_deg)
    }
}

struct FxCands: Decodable {
    let n: Int
    let f: [Int]
    let x: [Double]
    let y: [Double]
    let s: [Double]

    var candidates: [[ShuttleCandidate]] {
        var out = [[ShuttleCandidate]](repeating: [], count: n)
        for i in 0..<f.count { out[f[i]].append(ShuttleCandidate(x: x[i], y: y[i], score: s[i])) }
        return out
    }
}

struct FxTrack: Decodable {
    let xy: [Double?]
    let raw_xy: [Double?]?
    let conf: [Double?]
    let visible: [Int]
    let interpolated: [Int]
    let n: Int
    let t: [Double?]?

    var xyV: [SIMD2<Double>] { pairs(xy) }
    var rawV: [SIMD2<Double>]? { raw_xy.map(pairs) }

    /// The fixture track as a Track2D (input of smooth_track / find_kinks).
    func track(t: [Double]) -> Track2D {
        var vis = [Bool](repeating: false, count: n), itp = vis
        for i in visible { vis[i] = true }
        for i in interpolated { itp[i] = true }
        return Track2D(t: t, xy: xyV, conf: conf.map { $0 ?? .nan }, visible: vis, interpolated: itp, rawXY: rawV)
    }
}

func pairs(_ a: [Double?]) -> [SIMD2<Double>] {
    stride(from: 0, to: a.count, by: 2).map { SIMD2(a[$0] ?? .nan, a[$0 + 1] ?? .nan) }
}

struct FxInter: Decodable {
    let cuts: [Int]
    let assoc_obs: [Int]
    let assoc_start: [Int]
    let assoc_xy: [Double?]
    let rep_dup: [Int]?
    let rep_clock: [Double?]?
}

struct FxKinks: Decodable {
    let frames: [Int]
    let z: [Double?]
}

struct FxSmoothKinks: Decodable {
    let kinks: [Int]
    let out: FxTrack
}

struct FxClip: Decodable {
    let match: String
    let clip: String
    let detector: String
}

struct FxCase: Decodable {
    let name: String
    let meta: FxMeta
    let cands: FxCands
    let scene_cuts: [Int]?
    let out: FxTrack
    let inter: FxInter?
    let find_kinks: FxKinks?
    let smooth: FxTrack?
    let smooth_kinks: FxSmoothKinks?
    let in_track: FxTrack?
    let cut_range: [Int]?
    let held: [Int]?
    let held_xy: [Double?]?
    let clip: FxClip?
    let py_seconds: Double
    let fps_str: String?
    /// video_io.pts_gaps(meta.timestamps) (linker_gaps.json only).
    let pts_gaps: [Int]?

    /// VideoMeta with the exact fps (inf / nan survive via fps_str).
    var videoMeta: VideoMeta {
        var m = meta.meta
        if let s = fps_str { m.fps = s == "inf" ? .infinity : s == "-inf" ? -.infinity : Double(s) ?? .nan }
        return m
    }
}

func loadCases(_ name: String) throws -> [FxCase] {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"),
                            "missing fixture \(name).json — run `.venv/bin/python -m scripts.swift_fixtures.linker` in backend/")
    return try JSONDecoder().decode([FxCase].self, from: Data(contentsOf: url))
}

func loadCases(path: String) throws -> [FxCase]? {
    guard FileManager.default.fileExists(atPath: path) else { return nil }
    return try JSONDecoder().decode([FxCase].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
}

/// Differences between a Swift track and the Python reference (empty = parity).
struct TrackDiff {
    var problems: [String] = []
    var maxXY = 0.0              // over visible (observed) frames
    var maxFill = 0.0            // over interpolated frames
    var fillsOver1e6 = 0         // fills beyond 1e-6 px (ill-conditioned pieces, see compare())
    var maxRaw = 0.0
    var maxConf = 0.0
    var maskMismatch = 0
    var ok: Bool { problems.isEmpty }
}

/// Parity policy: masks identical; raw_xy / conf exact; xy ≤ `xyTol` (1e-6 px) on visible
/// frames; fills ≤ `fillTol` (1e-3 px). Fills in a piece whose observations were mostly
/// rejected are ill-conditioned: the Python reference itself moves them by 7e-4 px when σ
/// (median of hinge-fit RSS, computed with catastrophic cancellation whose last bits depend
/// on OpenBLAS's FMA/blocking order) changes by 1e-11 relative. Such fills are counted.
func compare(_ tr: Track2D, _ ref: FxTrack, xyTol: Double = 1e-6, fillTol: Double = 1e-3,
             exactTol: Double = 1e-8) -> TrackDiff {
    var d = TrackDiff()
    guard tr.n == ref.n else { d.problems.append("n \(tr.n) vs \(ref.n)"); return d }
    let vis = flatnonzero(tr.visible), itp = flatnonzero(tr.interpolated)
    if vis != ref.visible {
        d.maskMismatch += Set(vis).symmetricDifference(ref.visible).count
        d.problems.append("visible differs at \(Array(Set(vis).symmetricDifference(ref.visible)).sorted().prefix(10))")
    }
    if itp != ref.interpolated {
        d.maskMismatch += Set(itp).symmetricDifference(ref.interpolated).count
        d.problems.append("interpolated differs at \(Array(Set(itp).symmetricDifference(ref.interpolated)).sorted().prefix(10))")
    }
    let rx = ref.xyV
    var badXY: [Int] = []
    for i in 0..<tr.n {
        let a = tr.xy[i], b = rx[i]
        if a.x.isFinite != b.x.isFinite || a.y.isFinite != b.y.isFinite { badXY.append(i); continue }
        if a.x.isFinite {
            let e = max(abs(a.x - b.x), abs(a.y - b.y))
            if tr.interpolated[i] {
                d.maxFill = max(d.maxFill, e)
                if e > xyTol { d.fillsOver1e6 += 1 }
                if e > fillTol { badXY.append(i) }
            } else {
                d.maxXY = max(d.maxXY, e)
                if e > xyTol { badXY.append(i) }
            }
        }
    }
    if !badXY.isEmpty {
        d.problems.append("xy differs at \(badXY.count) frames, first \(badXY.prefix(8)), max vis \(d.maxXY) fill \(d.maxFill)")
    }
    if let rr = ref.rawV {
        let raw = tr.rawXY ?? [SIMD2<Double>](repeating: SIMD2(.nan, .nan), count: tr.n)
        var bad = 0
        for i in 0..<tr.n {
            let a = raw[i], b = rr[i]
            if a.x.isFinite != b.x.isFinite { bad += 1; continue }
            if a.x.isFinite {
                let e = max(abs(a.x - b.x), abs(a.y - b.y))
                d.maxRaw = max(d.maxRaw, e)
                if e > exactTol { bad += 1 }
            }
        }
        if bad > 0 { d.problems.append("raw_xy differs at \(bad) frames (max \(d.maxRaw))") }
    }
    var badC = 0
    for i in 0..<tr.n {
        let b = ref.conf[i] ?? .nan
        let e = abs(tr.conf[i] - b)
        if b.isFinite { d.maxConf = max(d.maxConf, e) }
        if !(e <= exactTol) { badC += 1 }
    }
    if badC > 0 { d.problems.append("conf differs at \(badC) frames (max \(d.maxConf))") }
    return d
}

/// Track → the exporter's JSON layout (for scoring Swift outputs in Python).
func encodeTrack(_ tr: Track2D) -> [String: Any] {
    func flat(_ v: [SIMD2<Double>]) -> [Any] {
        v.flatMap { [$0.x.isFinite ? $0.x as Any : NSNull(), $0.y.isFinite ? $0.y as Any : NSNull()] }
    }
    return ["xy": flat(tr.xy), "raw_xy": flat(tr.rawXY ?? []), "conf": tr.conf,
            "visible": flatnonzero(tr.visible), "interpolated": flatnonzero(tr.interpolated), "n": tr.n]
}
