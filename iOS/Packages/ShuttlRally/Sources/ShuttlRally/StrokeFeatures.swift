import Foundation
import ShuttlCore

// Input features of the learned stroke classifier (BST-CG-AP, seq_len 30, 25 merged ShuttleSet
// classes; research/bst/coreml/REPORT.md). Every rule here is a transcription of the research
// pipeline that produced the measured accuracy, and is pinned by a parity test against it:
//
//   per-frame arrays  research/bst/convert.py (main loop), applied to the arrays the Swift extract
//                     tool writes (research/extract RallyExtractor.swift: player slot 0 = far,
//                     1 = near, lowest track id per slot, shuttle = track.xy where track.visible)
//   window            research/e2e/e2e_convert.py fixed_window: [c - fps//2, c + fps//2] inclusive
//   to 30 frames      repo make_seq_len_same(30, ...): truncate the tail / zero-pad the end
//
// All arithmetic is float32, like the numpy code it mirrors, so the arrays agree bit for bit.

/// One player in one frame, as the classifier sees it.
public struct StrokePlayerSample: Sendable, Equatable {
    public var box: BBox
    /// 17 COCO keypoints, frame px, with confidence.
    public var keypoints: [Keypoint]
    /// Ankle-midpoint court position (m, world frame), nil when unknown.
    public var footWorld: SIMD2<Double>?

    public init(box: BBox, keypoints: [Keypoint], footWorld: SIMD2<Double>?) {
        self.box = box; self.keypoints = keypoints; self.footWorld = footWorld
    }
}

/// The classifier's input for one hit: three arrays zero-padded at the end to 30 frames.
public struct StrokeClip: Sendable, Equatable {
    public static let seqLen = 30
    /// (30, 2, 17, 2) row-major: (kp - box centre) / box diagonal; player 0 = far/Top, 1 = near/Bottom.
    public var joints: [Float]
    /// (30, 2, 2): ankle midpoint on the court, x' = 0.5 - y/6.1, y' = (6.7 - x)/13.4; 0 = unknown.
    public var pos: [Float]
    /// (30, 2): shuttle x/W, y/H; 0 when unseen or the frame's players are not both resolved.
    public var shuttle: [Float]
    /// Real frames (the rest is padding).
    public var videoLen: Int
    /// Source frame index of each real slot.
    public var frames: [Int]
    /// Real frames with both players resolved (the gate convert.py applies to the shuttle).
    public var resolvedFrames: Int

    public init(joints: [Float], pos: [Float], shuttle: [Float], videoLen: Int, frames: [Int], resolvedFrames: Int) {
        self.joints = joints; self.pos = pos; self.shuttle = shuttle
        self.videoLen = videoLen; self.frames = frames; self.resolvedFrames = resolvedFrames
    }

    /// 1 for real frames, 0 for padding (the model's `valid` input).
    public var valid: [Float] { (0..<Self.seqLen).map { $0 < videoLen ? 1 : 0 } }
}

public enum StrokeFeatures {
    /// Our hit detector fires early: on held-out match 039 (25-30 fps) the median offset
    /// label - detected is +2 frames, and adding +2 frames to every detected hit raised end-to-end
    /// stroke accuracy 0.7944 -> 0.8178 (research/e2e, `--shift 2`). Applied as time, so it means
    /// the same at any frame rate: 2 frames at 30 fps = 1/15 s (2 at 25 fps, 4 at 60 fps).
    public static let hitLeadSeconds = 2.0 / 30.0

    /// The model was trained on 25 and 30 fps clips. Frame rates that round into this range are
    /// windowed natively (exactly as the training data); anything else is resampled to 30 fps.
    public static let nativeFPS = 24...30
    public static let resampleFPS = 30.0

    // convert.py constants: COURT_HALF_LEN, COURT_HALF_WID, as float32 like the numpy maths.
    static let courtHalfWidOver2W: Float = 0.5                // COURT_HALF_WID / (2 * COURT_HALF_WID)
    static let twoHalfWid = Float(2 * 3.05)
    static let halfLen = Float(6.7)
    static let twoHalfLen = Float(2 * 6.7)

    /// Frames a hit's clip is cut from: the detected frame plus the measured lead, then the seq30
    /// window. `window` is the source window (`fixed_window`, inclusive at 24-30 fps: fps+1
    /// frames), `used` is the first seq30 of it — the shape `make_seq_len_same(30, ...)` turns into
    /// the model's fixed 30, exactly as the measured clips are stored
    /// (`research/e2e/m039_fixed30/*_joints.npy` are 31 frames for a 31-frame source window, and
    /// the collate step keeps the first 30). nil when fewer than 2 frames are in the video, which
    /// the reference pipeline skips.
    public static func windowFrames(hitFrame: Int, fps: Double, nFrames: Int) -> (window: [Int], used: Int)? {
        let f = fps.isFinite && fps > 0 ? fps : 30
        let centre = hitFrame + Num.iround(hitLeadSeconds * f)
        var frames: [Int]
        let rf = Num.iround(f)
        if nativeFPS.contains(rf) {
            let half = rf / 2                                          // int(round(fps)) // 2
            frames = Array((centre - half)...(centre + half))          // fixed_window, inclusive
        } else {
            // one source frame per model slot, so the 30 slots hold the same 1 s as a 30 fps clip
            frames = (0..<StrokeClip.seqLen).map { k in
                centre + Num.iround(Double(k - StrokeClip.seqLen / 2) * f / resampleFPS)
            }
        }
        frames = frames.filter { $0 >= 0 && $0 < nFrames }             // convert.py: `if f in fidx`
        guard frames.count >= 2 else { return nil }
        return (frames, min(frames.count, StrokeClip.seqLen))
    }
    /// convert.py's per-frame features of one frame. `players[0]` = far, `players[1]` = near.
    /// Returns (joints 68, pos 4, shuttle 2, both players resolved).
    ///
    /// Note the exact convert.py behaviour: players are checked far first; if the far player
    /// fails, the frame's joints stay zero; if only the near player fails, the far player's joints
    /// are KEPT. The shuttle is written only when both pass. Positions ignore that gate.
    public static func frameFeatures(players: [StrokePlayerSample?], shuttleXY: SIMD2<Double>?, shuttleVisible: Bool,
                                     width: Int, height: Int) -> (joints: [Float], pos: [Float], shuttle: [Float], both: Bool) {
        precondition(players.count == 2, "players = [far, near]")
        var j = [Float](repeating: 0, count: 2 * 17 * 2)
        var pos = [Float](repeating: 0, count: 4)
        var sh: [Float] = [0, 0]
        var both = true
        for p in 0..<2 {
            guard let s = players[p], let jj = normalisedJoints(s) else { both = false; break }
            for i in 0..<34 { j[p * 34 + i] = jj[i] }
        }
        if both, shuttleVisible, let xy = shuttleXY {
            sh = [Float(xy.x) / Float(width), Float(xy.y) / Float(height)]
        }
        for p in 0..<2 {
            guard let fw = players[p]?.footWorld else { continue }
            let fx = Float(fw.x), fy = Float(fw.y)
            if fx.isFinite {
                pos[p * 2] = courtHalfWidOver2W - fy / twoHalfWid
                pos[p * 2 + 1] = (halfLen - fx) / twoHalfLen
            }
        }
        return (j, pos, sh, both)
    }

    /// (kp - box centre) / box diagonal for one player, or nil when convert.py's gate fails:
    /// box not finite, any of the 17 joints' x/y not finite, or an ankle confidence <= 0.
    static func normalisedJoints(_ s: StrokePlayerSample) -> [Float]? {
        let b = [Float(s.box.x1), Float(s.box.y1), Float(s.box.x2), Float(s.box.y2)]
        guard b.allSatisfy(\.isFinite), s.keypoints.count >= 17 else { return nil }
        let kp = s.keypoints
        for i in 0..<17 where !(Float(kp[i].x).isFinite && Float(kp[i].y).isFinite) { return nil }
        // `kx[15, 2] <= 0 or kx[16, 2] <= 0` (a NaN confidence passes, as in numpy)
        if Float(kp[15].conf) <= 0 || Float(kp[16].conf) <= 0 { return nil }
        let cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2
        let diag = max(hypotf(b[2] - b[0], b[3] - b[1]), Float(1e-6))
        var out = [Float](repeating: 0, count: 34)
        for i in 0..<17 {
            out[i * 2] = (Float(kp[i].x) - cx) / diag
            out[i * 2 + 1] = (Float(kp[i].y) - cy) / diag
        }
        return out
    }

    /// The clip for one hit from the app's own stage outputs.
    ///
    /// - players: per-frame observations (`PlayerTracker`): far/near by `side`, the lowest track
    ///   id per side and frame (the extract tool's rule).
    /// - track: the linked shuttle track (`xy` where `visible`).
    public static func clip(hitFrame: Int, meta: VideoMeta, players: PlayerFrameIndex, track: Track2D) -> StrokeClip? {
        guard let w = windowFrames(hitFrame: hitFrame, fps: meta.fps, nFrames: min(meta.nFrames, track.n)) else {
            return nil
        }
        let T = StrokeClip.seqLen
        let frames = Array(w.window.prefix(w.used))                    // make_seq_len_same's truncation
        var joints = [Float](repeating: 0, count: T * 68)
        var pos = [Float](repeating: 0, count: T * 4)
        var shuttle = [Float](repeating: 0, count: T * 2)
        var resolved = 0
        for (t, f) in frames.enumerated() {
            let pl = players.at(f)
            let r = frameFeatures(players: [pl.far, pl.near],
                                  shuttleXY: track.xy[f], shuttleVisible: track.visible[f],
                                  width: meta.width, height: meta.height)
            joints.replaceSubrange((t * 68)..<((t + 1) * 68), with: r.joints)
            pos.replaceSubrange((t * 4)..<((t + 1) * 4), with: r.pos)
            shuttle.replaceSubrange((t * 2)..<((t + 1) * 2), with: r.shuttle)
            if r.both { resolved += 1 }
        }
        return StrokeClip(joints: joints, pos: pos, shuttle: shuttle, videoLen: frames.count,
                          frames: frames, resolvedFrames: resolved)
    }
}

/// Frame -> (far, near) player sample, built once per rally from the players stage.
public struct PlayerFrameIndex: Sendable {
    var far: [Int: (tid: Int, s: StrokePlayerSample)] = [:]
    var near: [Int: (tid: Int, s: StrokePlayerSample)] = [:]
    /// More than one track on a court side: doubles (the model is singles-only).
    public private(set) var multiplePerSide = false

    public init(_ players: [PlayerObservation]?) {
        var tracks: [CourtSide: Set<Int>] = [:]
        for o in players ?? [] where o.side == .far || o.side == .near {
            tracks[o.side, default: []].insert(o.trackID)
            let s = StrokePlayerSample(box: o.bbox, keypoints: o.keypoints ?? [], footWorld: o.footWorld)
            // extract tool: `if playerTrack[slot] >= 0 && playerTrack[slot] < o.trackID { continue }`
            if o.side == .far {
                if let cur = far[o.frame], cur.tid < o.trackID { continue }
                far[o.frame] = (o.trackID, s)
            } else {
                if let cur = near[o.frame], cur.tid < o.trackID { continue }
                near[o.frame] = (o.trackID, s)
            }
        }
        multiplePerSide = tracks.values.contains { $0.count > 1 }
    }

    public func at(_ frame: Int) -> (far: StrokePlayerSample?, near: StrokePlayerSample?) {
        (far[frame]?.s, near[frame]?.s)
    }
}
