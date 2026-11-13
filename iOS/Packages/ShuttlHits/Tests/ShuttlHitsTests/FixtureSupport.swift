import Foundation
import ShuttlCore
import simd
import XCTest
@testable import ShuttlHits

// Decoding of backend/scripts/swift_fixtures/hits.py output.

struct Blob: Decodable {
    let dtype: String
    let shape: [Int]
    let data: String

    private func bytes() throws -> Data {
        guard let comp = Data(base64Encoded: data) else { throw FixtureError.bad("base64") }
        if comp.isEmpty { return Data() }
        return try (comp as NSData).decompressed(using: .zlib) as Data   // raw deflate
    }

    var doubles: [Double] {
        let d = try! bytes()
        switch dtype {
        case "f8": return d.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
        case "f4": return d.withUnsafeBytes { $0.bindMemory(to: Float.self).map(Double.init) }
        default: fatalError("not float: \(dtype)")
        }
    }

    var ints: [Int] {
        let d = try! bytes()
        switch dtype {
        case "i4": return d.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Int.init) }
        case "i8": return d.withUnsafeBytes { $0.bindMemory(to: Int64.self).map(Int.init) }
        case "u1", "b1": return d.map(Int.init)
        default: fatalError("not int: \(dtype)")
        }
    }

    var bools: [Bool] { ints.map { $0 != 0 } }
    var vec2: [SIMD2<Double>] { let v = doubles; return stride(from: 0, to: v.count, by: 2).map { SIMD2(v[$0], v[$0 + 1]) } }
}

enum FixtureError: Error { case bad(String) }

struct FTrack: Decodable {
    let t: Blob, xy: Blob, raw_xy: Blob?, visible: Blob, interpolated: Blob
    var track: Track2D {
        let vis = visible.bools
        return Track2D(t: t.doubles, xy: xy.vec2, conf: vis.map { $0 ? 1 : 0 }, visible: vis,
                       interpolated: interpolated.bools, rawXY: raw_xy?.vec2)
    }
}

struct FMeta: Decodable {
    let fps: Double, width: Int, height: Int, n_frames: Int, timestamps: Blob
    var meta: VideoMeta { VideoMeta(fps: fps, width: width, height: height, nFrames: n_frames, timestamps: timestamps.doubles) }
}

struct FCamera: Decodable {
    let K: Blob, R: Blob, t: Blob, dist: Blob, image_size: [Int], confidence: Double, reproj_rmse_px: Double?
    var camera: CameraModel {
        func m(_ b: Blob) -> simd_double3x3 { let v = b.doubles; return simd_double3x3(rowMajor: [Array(v[0..<3]), Array(v[3..<6]), Array(v[6..<9])]) }
        let tv = t.doubles
        return CameraModel(K: m(K), R: m(R), t: SIMD3(tv[0], tv[1], tv[2]),
                           imageSize: ImageSize(width: image_size[0], height: image_size[1]),
                           dist: dist.doubles, reprojRmsePx: reproj_rmse_px ?? .nan, confidence: confidence)
    }
}

struct FPlayers: Decodable {
    let frame: Blob, track_id: Blob, bbox: Blob, score: Blob, side: Blob, foot: Blob, has_kp: Blob, wrists: Blob
    var players: [PlayerObservation] {
        let fr = frame.ints, tid = track_id.ints, bb = bbox.doubles, sc = score.doubles, sd = side.ints
        let ft = foot.doubles, hk = has_kp.bools, wr = wrists.doubles
        let sides: [CourtSide] = [.unknown, .near, .far]
        return fr.indices.map { i in
            var kp: [Keypoint]? = nil
            if hk[i] {
                var k = [Keypoint](repeating: Keypoint(x: 0, y: 0, conf: 0), count: 17)
                k[HitConstants.leftWrist] = Keypoint(x: wr[6 * i], y: wr[6 * i + 1], conf: wr[6 * i + 2])
                k[HitConstants.rightWrist] = Keypoint(x: wr[6 * i + 3], y: wr[6 * i + 4], conf: wr[6 * i + 5])
                kp = k
            }
            let f = SIMD2(ft[2 * i], ft[2 * i + 1])
            return PlayerObservation(frame: fr[i], trackID: tid[i],
                                     bbox: BBox(x1: bb[4 * i], y1: bb[4 * i + 1], x2: bb[4 * i + 2], y2: bb[4 * i + 3]),
                                     score: sc[i], side: sides[sd[i]], footWorld: f.x.isNaN && f.y.isNaN ? nil : f,
                                     keypoints: kp)
        }
    }
}

struct FHit: Decodable {
    let frame: Int, t: Double, xy_px: [Double], hitter: String, confidence: Double, player_track_id: Int?
    let source: String?                    // present only when != "kink" (F4 gap contact)
}

struct FCand: Decodable {
    let tau: Double, frame: Int, lam: Double, lam_v: Double, gap_s: Double?
    let is_start: Bool, static_before: Bool, static_after: Bool, outside: Bool, settles: Bool, weak: Bool
    let hidden: Bool?
    let gap_contact: Bool?
    let kin: Double?, reach: Double, side_llr: Double, near_tid: Int?, far_tid: Int?, seg: Int
    let xy: [Double], v_in: [Double], v_out: [Double], feats: [Double?]
}

struct FScan: Decodable {
    let obs_frames: [Int], obs_t: [Double], sigma: Double, dt: Double
    let lam: [Double], lam_v: [Double], ok: [Bool], lo: [Int], hi: [Int], m: [Int]
}

struct FCase: Decodable {
    let name: String
    let track: FTrack, meta: FMeta, camera: FCamera?, players: FPlayers?
    let config: HitConfig
    let hits: [FHit]
    let cands: [FCand]?
    let scan: FScan?
    let true_frame: [Int]?
    let true_side: [String]?
    let doubles: Bool?

    struct Inputs { let track: Track2D; let meta: VideoMeta; let camera: CameraModel?; let players: [PlayerObservation]? }
    var inputs: Inputs { Inputs(track: track.track, meta: meta.meta, camera: camera?.camera, players: players?.players) }
}

struct FTiming: Decodable { let frames: Int; let seconds: Double; let ms_per_1000_frames: Double }
struct FFile: Decodable { let cases: [FCase]; let python_timing: FTiming? }

struct FLearned: Decodable {
    struct Case: Decodable { let model: String; let source: String; let name: String; let hits: [FHit]; let cands: [FCand] }
    let cases: [Case]
}

enum Fixtures {
    static func url(_ name: String) -> URL {
        Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
            ?? Bundle.module.resourceURL!.appendingPathComponent("Fixtures/\(name)")
    }

    static func load<T: Decodable>(_ name: String, as: T.Type = FFile.self) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: url(name)))
    }
}
