import Foundation
import ShuttlCore

/// The learned hit detector behind the app's hit stage: build the (T,16) raw matrix from the
/// tracking stages, derive the (T,51) features, run `HitNetModel`, decode (thr 0.4 / gap 0.30).
///
/// The rule detector (`ShuttlHits.detectHits`) stays reachable through `HitStage.detector`, so
/// `LiveStages` can be switched back to the ported rules without a rebuild.
public enum HitStage {

    /// Which detector the hit stage runs. `learned` is the default (the model replaced the rules);
    /// `.rules` uses the ported kinematic detector unchanged.
    public enum Detector: String, Sendable, CaseIterable {
        case learned
        case rules
    }

    /// Process-wide switch and installed model, in one lock (the hit stage is called from one
    /// pipeline at a time, but tests install and uninstall it).
    public enum Registry {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var current: HitNetModel?
        nonisolated(unsafe) private static var kind: Detector = .learned

        public static var model: HitNetModel? {
            lock.lock(); defer { lock.unlock() }
            return current
        }

        public static var detector: Detector {
            get { lock.lock(); defer { lock.unlock() }; return kind }
            set { lock.lock(); kind = newValue; lock.unlock() }
        }

        public static func install(_ model: HitNetModel?) {
            lock.lock(); current = model; lock.unlock()
        }

        /// Loads the bundled model once; returns whether a model is available afterwards.
        /// Safe to call repeatedly (a second call is a no-op when one is already installed).
        @discardableResult
        public static func loadBundled(in bundle: Bundle = .main, cacheDirectory: URL? = nil) async -> Bool {
            if model != nil { return true }
            guard let m = await HitNetModel.loadBundled(in: bundle, cacheDirectory: cacheDirectory) else { return false }
            install(m)
            return true
        }
    }

    /// The raw (T,16) matrix `HitFeatures.derive` reads, from the seam's own inputs.
    ///
    /// Column meanings and the NaN rules are README "raw columns":
    ///   0,1 shuttle x/W, y/H   — NaN where the shuttle is not visible
    ///   2   shuttle confidence — NaN where not visible
    ///   3   interpolated 0/1   — never NaN
    ///   4-7, 8-11 player boxes — NaN where that player's box is not resolved
    ///   12-15 feet world x, y  — NaN where that player's foot is not resolved
    ///
    /// Player 0 is far/Top and player 1 is near/Bottom, matching RallyExtractor and the
    /// Python training cache. Both learned outputs depend on this channel order.
    public static func rawMatrix(track: Track2D, meta: VideoMeta, players: [PlayerObservation]?) -> [Double] {
        let n = track.n
        var raw = [Double](repeating: .nan, count: n * HitFeatures.Raw.count)
        let W = Double(meta.width != 0 ? meta.width : 1280)
        let H = Double(meta.height != 0 ? meta.height : 720)
        let index = PlayerFrameIndex(players ?? [])

        for i in 0..<n {
            let base = i * HitFeatures.Raw.count
            // The shuttle columns are NaN both when there was no detection and when the linked
            // position is not finite; `visible` alone is not enough (a filled frame has a finite
            // xy but is not a real detection either — Python's mask is `shuttle_visible & finite`).
            let xy = track.xy[i]
            let visible = track.visible[i] && xy.x.isFinite && xy.y.isFinite
            if visible {
                raw[base + HitFeatures.Raw.shuttleX] = xy.x / W
                raw[base + HitFeatures.Raw.shuttleY] = xy.y / H
                raw[base + HitFeatures.Raw.shuttleConf] = track.conf[i]
            }
            raw[base + HitFeatures.Raw.shuttleInterp] = track.interpolated[i] ? 1 : 0

            for p in 0..<2 {
                let side: CourtSide = p == 0 ? .far : .near
                let o = HitFeatures.Raw.box0 + 4 * p
                guard let obs = index.observation(frame: i, side: side) else { continue }
                let b = obs.bbox
                guard b.x1.isFinite, b.y1.isFinite, b.x2.isFinite, b.y2.isFinite else { continue }
                raw[base + o + 0] = (b.x1 + b.x2) / 2 / W
                raw[base + o + 1] = (b.y1 + b.y2) / 2 / H
                raw[base + o + 2] = (b.x2 - b.x1) / W
                raw[base + o + 3] = (b.y2 - b.y1) / H
                if let f = obs.footWorld, f.x.isFinite, f.y.isFinite {
                    raw[base + HitFeatures.Raw.foot0 + 2 * p] = f.x
                    raw[base + HitFeatures.Raw.foot0 + 2 * p + 1] = f.y
                }
            }
        }
        return raw
    }

    /// Feature matrix for the seam's inputs: (T,16) raw -> (T,51).
    public static func features(track: Track2D, meta: VideoMeta, players: [PlayerObservation]?) -> [Float] {
        HitFeatures.derive(raw: rawMatrix(track: track, meta: meta, players: players),
                           nFrames: track.n, fps: meta.fps)
    }

    /// Hits for a whole track with the learned model, in `HitEvent` terms.
    ///
    /// Windowed when the track is longer than the graph's window (a whole match is), whole-track
    /// when it fits — the spelling the fixture parity test pins.
    public static func detect(track: Track2D, meta: VideoMeta, players: [PlayerObservation]?,
                              model: HitNetModel, params: HitDecoder.Params = .shipped) throws -> [HitEvent] {
        let T = track.n
        guard T > 0 else { return [] }
        let x = features(track: track, meta: meta, players: players)
        let fps = meta.fps > 0 ? meta.fps : 30
        let lg = try HitDecoder.logitsWindowed(features: x, frames: T, model: model)
        // The row's video frame number: the track is one row per decoded frame, so row i is video
        // frame i (the same convention the linker and the rule detector use).
        let dets = try HitDecoder.detections(logits: lg, fps: fps, frames: Array(0..<T), params: params)
        let index = PlayerFrameIndex(players ?? [])
        return dets.map { d in
            let xy = d.row < track.n ? track.xy[d.row] : SIMD2<Double>(.nan, .nan)
            // The rule detector attaches the hitter's track id when a player of that side was
            // resolved in the hit's frame; the learned side head gives the same clue.
            let tid = index.observation(frame: d.row, side: d.hitter)?.trackID
            return HitEvent(frame: d.frame, t: d.t, xyPx: xy, hitter: d.hitter,
                            confidence: d.confidence, playerTrackID: tid, source: "hitnet")
        }
    }
}

/// Per-frame lookup of the observations of each side (players are sampled, so most frames have no
/// observation of their own; the hit stage needs the box and foot of the hit's own frame).
struct PlayerFrameIndex {
    private var bySide: [CourtSide: [(frame: Int, obs: PlayerObservation)]]

    init(_ players: [PlayerObservation]) {
        var m: [CourtSide: [(Int, PlayerObservation)]] = [:]
        for o in players { m[o.side, default: []].append((o.frame, o)) }
        var out: [CourtSide: [(frame: Int, obs: PlayerObservation)]] = [:]
        // The training extractor keeps the lowest track id when a side has more
        // than one observation in the same frame. Keep that choice deterministic.
        for (side, v) in m {
            out[side] = v.sorted { a, b in
                a.0 == b.0 ? a.1.trackID < b.1.trackID : a.0 < b.0
            }
        }
        bySide = out
    }

    /// The observation of `side` at `frame` itself only: a box that belongs to another frame is
    /// not this frame's box, and filling it would have to invent the velocity the net reads.
    func observation(frame: Int, side: CourtSide) -> PlayerObservation? {
        guard let v = bySide[side] else { return nil }
        var lo = 0, hi = v.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if v[mid].frame < frame { lo = mid + 1 } else { hi = mid }
        }
        return lo < v.count && v[lo].frame == frame ? v[lo].obs : nil
    }
}
