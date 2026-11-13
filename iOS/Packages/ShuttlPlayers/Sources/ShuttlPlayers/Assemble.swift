import Foundation
import ShuttlCore

/// Tracking result: per-frame observations sorted by (frame, track id) plus the timing
/// movement_stats needs (players.py `PlayerTrackList`).
public struct PlayerTracks: Sendable {
    public var observations: [PlayerObservation]
    public var timestamps: [Double]?
    public var fps: Double
    public var doubles: Bool
    public var diagnostics: Diagnostics?

    public struct Diagnostics: Sendable, Equatable {
        public var doubles: Bool
        public var sampledFrames: Int
        /// track id → fraction of sampled frames with a detection (rounded to 3 d.p.)
        public var coverage: [Int: Double]
    }

    public init(observations: [PlayerObservation] = [], timestamps: [Double]? = nil, fps: Double = 30,
                doubles: Bool = false, diagnostics: Diagnostics? = nil) {
        self.observations = observations; self.timestamps = timestamps; self.fps = fps
        self.doubles = doubles; self.diagnostics = diagnostics
    }
}

public enum PlayerTracker {
    /// players._time_fn
    static func timeFn(_ meta: VideoMeta) -> (Int) -> Double {
        let ts = meta.timestamps
        let fps = meta.fps == 0 ? 30.0 : meta.fps
        return { i in
            if i < ts.count { return ts[i] }
            return ts.isEmpty ? Double(i) / fps : ts[ts.count - 1] + Double(i - ts.count + 1) / fps
        }
    }

    /// Doubles if ≥ 2 confident on-court persons per half (≥ 4 without camera) in more
    /// than 30 % of sampled frames (players.detect_doubles).
    public static func detectDoubles(_ frames: [PlayerCandidates], hasCamera: Bool) -> Bool {
        if frames.isEmpty { return false }
        var hits = 0
        for c in frames {
            if hasCamera {
                var near = 0, far = 0
                for i in 0..<c.n where c.unary[i] < 1.2 && PlayerGeometry.outsideDistance(c.pos[i]) < 0.3 {
                    if c.pos[i].x < 0 { near += 1 }
                    if c.pos[i].x > 0 { far += 1 }
                }
                hits += (near >= 2 && far >= 2) ? 1 : 0
            } else {
                hits += c.unary.filter { $0 < 1.2 }.count >= 4 ? 1 : 0
            }
        }
        return Double(hits) / Double(frames.count) > 0.3
    }

    /// Candidates that are the chosen one or a duplicate detection of it.
    static func exclusion(_ c: PlayerCandidates, _ j: Int) -> [Bool] {
        (0..<c.n).map { i in
            iou(c.boxes[j], c.boxes[i]) > 0.5 || NP.normAxis(c.pos[i] - c.pos[j]) * c.unit[i] < 0.35
        }
    }

    /// Sequentially extract up to nPaths disjoint Viterbi paths (k → j).
    static func extractPaths(_ frames: [PlayerCandidates], _ times: [Double], baseAllowed: [[Int]],
                             extra: [[Double]]?, nPaths: Int, taken: inout [[Bool]],
                             config: PlayersConfig) -> [[Int: Int]] {
        var out: [[Int: Int]] = []
        for _ in 0..<nPaths {
            let allowed = baseAllowed.enumerated().map { k, a in a.filter { !taken[k][$0] } }
            let path = PlayerViterbi.path(frames, allowed: allowed, times: times, extra: extra, config: config)
            if path.isEmpty { break }
            var d: [Int: Int] = [:]
            for (k, j) in path { d[k] = j }
            out.append(d)
            for (k, j) in d {
                let ex = exclusion(frames[k], j)
                for i in 0..<ex.count where ex[i] { taken[k][i] = true }
            }
        }
        return out
    }

    static func medianYStart(_ frames: [PlayerCandidates], _ times: [Double], _ path: [Int: Int], hasCamera: Bool) -> Double {
        let ks = path.keys.sorted()
        let t0 = times[ks[0]]
        var vals: [Double] = []
        for k in ks where times[k] - t0 <= 2.0 {
            vals.append(hasCamera ? frames[k].pos[path[k]!].y : -frames[k].footPx[path[k]!].x)
        }
        return NP.median(vals)
    }

    /// A candidate claimed by paths of both halves stays with the half its foot is in.
    static func resolveCrossSide(_ frames: [PlayerCandidates], _ slots: inout [(CourtSide, [Int: Int])]) {
        for k in 0..<frames.count {
            var claimOrder: [Int] = []
            var claims: [Int: [Int]] = [:]
            for (si, (_, p)) in slots.enumerated() {
                if let j = p[k] {
                    if claims[j] == nil { claimOrder.append(j); claims[j] = [] }
                    claims[j]!.append(si)
                }
            }
            for j in claimOrder {
                let sis = claims[j]!
                if sis.count < 2 { continue }
                let trueSide = CourtModel.sideOf(x: frames[k].pos[j].x)
                let match = sis.filter { slots[$0].0 == trueSide }
                let keep = match.isEmpty ? [sis[0]] : [match[0]]
                for si in sis where !keep.contains(si) { slots[si].1[k] = nil }
            }
        }
    }

    /// Identity assignment + densify + smoothing on already-detected candidates
    /// (players.assemble_tracks).
    public static func assembleTracks(_ frames0: [PlayerCandidates], meta: VideoMeta, camera: CameraModel?,
                                      doubles: Bool? = nil, config: PlayersConfig = .default) -> PlayerTracks {
        let tOf = timeFn(meta)
        var out = PlayerTracks(timestamps: meta.timestamps, fps: meta.fps == 0 ? 30.0 : meta.fps)
        if frames0.isEmpty { return out }
        let sorted0 = frames0.enumerated().sorted { a, b in
            a.element.frame < b.element.frame || (a.element.frame == b.element.frame && a.offset < b.offset)
        }.map(\.element)
        let times = sorted0.map { tOf($0.frame) }
        let hasCamera = camera != nil
        let frames = Candidates.addStaticPenalty(sorted0, times: times, hasCamera: hasCamera, config: config)
        let dbl = doubles ?? detectDoubles(frames, hasCamera: hasCamera)
        let perSide = dbl ? 2 : 1
        out.doubles = dbl
        let K = frames.count
        var taken = frames.map { [Bool](repeating: false, count: $0.n) }
        var slots: [(CourtSide, [Int: Int])] = []
        if hasCamera {
            for (side, sgn) in [(CourtSide.near, -1.0), (CourtSide.far, 1.0)] {
                let allowed = frames.map { c in (0..<c.n).filter { sgn * c.pos[$0].x > -config.sideTol } }
                let extra = frames.map { c in
                    c.pos.map { p -> Double in
                        3.0 * npClip((-sgn) * p.x / config.sideTol + 1.0, 0, 1) * (sgn * p.x < 0 ? 1.0 : 0.0)
                    }
                }
                var sideTaken = taken
                for p in extractPaths(frames, times, baseAllowed: allowed, extra: extra, nPaths: perSide,
                                      taken: &sideTaken, config: config) {
                    slots.append((side, p))
                }
            }
            resolveCrossSide(frames, &slots)
        } else {
            let allowed = frames.map { Array(0..<$0.n) }
            let paths = extractPaths(frames, times, baseAllowed: allowed, extra: nil, nPaths: 2 * perSide,
                                     taken: &taken, config: config)
            let vMed = paths.map { p in NP.median(p.map { frames[$0.key].footPx[$0.value].y }) }
            let order = Array(stableArgsort(vMed).reversed())            // largest image row = nearest
            for (rank, pi) in order.enumerated() {
                slots.append((rank < max(1, paths.count / 2) ? .near : .far, paths[pi]))
            }
        }
        if perSide == 2 {
            var split: [(CourtSide, [Int: Int])] = []
            for side in [CourtSide.near, .far] {
                let sp = slots.filter { $0.0 == side && !$0.1.isEmpty }.map(\.1)
                let parts = sp.count == 2 ? PartnerIdentity.splitPartners(frames, times: times, paths: sp, config: config) : sp
                split += parts.map { (side, $0) }
            }
            slots = split
        }
        slots = slots.filter { !$0.1.isEmpty }
        // ids: near 1 (/3), far 2 (/4); doubles: lower id = smaller world y at the start
        var ids: [(Int, Int)] = []
        for side in [CourtSide.near, .far] {
            var sp = slots.enumerated().filter { $0.element.0 == side }.map { ($0.offset, $0.element.1) }
            let keys = sp.map { medianYStart(frames, times, $0.1, hasCamera: hasCamera) }
            let order = stableArgsort(keys)
            sp = order.map { sp[$0] }
            let base = side == .near ? 1 : 2
            for (rank, (i, _)) in sp.enumerated() { ids.append((i, base + 2 * rank)) }
        }
        var coverage: [Int: Double] = [:]
        var obs: [PlayerObservation] = []
        for (i, tid) in ids.sorted(by: { $0.1 < $1.1 }) {
            let (side, path) = slots[i]
            obs += densify(frames, times, path, tid: tid, side: side, camera: camera, tOf: tOf, config: config)
            coverage[tid] = NP.pyRound(Double(path.count) / Double(K), 3)
        }
        obs.sort { ($0.frame, $0.trackID) < ($1.frame, $1.trackID) }
        out.observations = obs
        out.diagnostics = .init(doubles: dbl, sampledFrames: K, coverage: coverage)
        return out
    }

    /// Per-frame observations for one identity: interpolate bbox/keypoints between
    /// samples, RTS-smooth foot_world; gaps > MAX_INTERP_S split the track.
    static func densify(_ frames: [PlayerCandidates], _ times: [Double], _ path: [Int: Int], tid: Int,
                        side: CourtSide, camera: CameraModel?, tOf: (Int) -> Double,
                        config: PlayersConfig) -> [PlayerObservation] {
        let ks = path.keys.sorted()
        var segs: [[Int]] = ks.isEmpty ? [] : [[ks[0]]]
        for (a, b) in zip(ks, ks.dropFirst()) {
            if times[b] - times[a] > config.maxInterpS { segs.append([b]) } else { segs[segs.count - 1].append(b) }
        }
        var res: [PlayerObservation] = []
        for seg in segs {
            let F = seg.map { frames[$0].frame }
            let B = seg.map { frames[$0].boxes[path[$0]!] }
            let S = seg.map { frames[$0].scores[path[$0]!] }
            let KPs = seg.map { Array(frames[$0].kps.row(path[$0]!)) }
            let f0 = F[0], nD = F[F.count - 1] - f0 + 1
            var foot: [SIMD2<Double>]? = nil
            if let _ = camera {
                let td = (0..<nD).map { tOf(f0 + $0) }
                var z = [SIMD2<Double>](repeating: SIMD2(.nan, .nan), count: nD)
                var r = [Double](repeating: 0.2, count: nD)
                for (q, k) in seg.enumerated() {
                    z[F[q] - f0] = frames[k].pos[path[k]!]
                    r[F[q] - f0] = frames[k].noise[path[k]!]
                }
                foot = PlayerViterbi.rtsSmooth(t: td, z: z, r: r, accSigma: config.accSigma)
            }
            var seg0 = 0                                   // index i with F[i] <= D < F[i+1]
            for n in 0..<nD {
                let D = f0 + n
                var box: BBox, score: Double
                var kp = [Keypoint]()
                kp.reserveCapacity(17)
                if F.count > 1 {
                    while seg0 < F.count - 2 && F[seg0 + 1] <= D { seg0 += 1 }
                    let i = seg0
                    let w = Double(D - F[i]) / Double(F[i + 1] - F[i])
                    let a = B[i], b = B[i + 1]
                    box = BBox(x1: a.x1 * (1 - w) + b.x1 * w, y1: a.y1 * (1 - w) + b.y1 * w,
                               x2: a.x2 * (1 - w) + b.x2 * w, y2: a.y2 * (1 - w) + b.y2 * w)
                    score = w == 0 ? S[i] : min(S[i], S[i + 1])
                    let ka = KPs[i], kb = KPs[i + 1]
                    for j in 0..<17 {
                        let x = ka[j * 3] * (1 - w) + kb[j * 3] * w
                        let y = ka[j * 3 + 1] * (1 - w) + kb[j * 3 + 1] * w
                        let c = w == 0 ? ka[j * 3 + 2] : npMin2(ka[j * 3 + 2], kb[j * 3 + 2])
                        kp.append(Keypoint(x: f32(x), y: f32(y), conf: f32(c)))
                    }
                } else {
                    box = B[0]; score = S[0]
                    for j in 0..<17 {
                        kp.append(Keypoint(x: f32(KPs[0][j * 3]), y: f32(KPs[0][j * 3 + 1]), conf: f32(KPs[0][j * 3 + 2])))
                    }
                }
                res.append(PlayerObservation(frame: D, trackID: tid, bbox: box, score: score, side: side,
                                             footWorld: foot?[n], keypoints: kp))
            }
        }
        return res
    }
}

/// Keypoints are stored as float32 in Python (`astype(np.float32)`).
@inline(__always) func f32(_ x: Double) -> Double { Double(Float(x)) }

/// `np.minimum` (NaN propagates).
@inline(__always) func npMin2(_ a: Double, _ b: Double) -> Double {
    if a.isNaN || b.isNaN { return .nan }
    return min(a, b)
}

/// Stable argsort (numpy's small-array quicksort is insertion sort, hence stable).
func stableArgsort(_ a: [Double]) -> [Int] {
    a.indices.sorted { i, j in
        switch (a[i].isNaN, a[j].isNaN) {
        case (false, true): return true
        case (true, false): return false
        case (true, true): return i < j
        default: return a[i] < a[j] || (a[i] == a[j] && i < j)
        }
    }
}
