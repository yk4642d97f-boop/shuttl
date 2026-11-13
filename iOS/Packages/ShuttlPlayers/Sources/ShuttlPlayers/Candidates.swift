import Foundation
import ShuttlCore

/// Person candidates of one sampled frame after gating (players.py `_Cands`).
public struct PlayerCandidates: Sendable {
    public var frame: Int
    public var boxes: [BBox]
    public var scores: [Double]
    public var kps: KeypointArray
    public var footPx: [SIMD2<Double>]
    /// Floor metres (camera) or pixels (no camera).
    public var pos: [SIMD2<Double>]
    /// Player-likeness cost (lower = more player-like).
    public var unary: [Double]
    /// Metres per unit of `pos` (1 with a camera).
    public var unit: [Double]
    /// Expected foot-position noise, metres.
    public var noise: [Double]
    /// Flat (N, 38) float32 colour descriptors; nil = no image.
    public var app: [Float]?

    public var n: Int { scores.count }

    public init(frame: Int, boxes: [BBox], scores: [Double], kps: KeypointArray, footPx: [SIMD2<Double>],
                pos: [SIMD2<Double>], unary: [Double], unit: [Double], noise: [Double], app: [Float]?) {
        self.frame = frame; self.boxes = boxes; self.scores = scores; self.kps = kps; self.footPx = footPx
        self.pos = pos; self.unary = unary; self.unit = unit; self.noise = noise; self.app = app
    }

    func appRow(_ j: Int) -> ArraySlice<Float>? {
        guard let app else { return nil }
        return app[(j * Appearance.dim)..<((j + 1) * Appearance.dim)]
    }
}

public enum Candidates {
    /// Player-likeness cost in camera mode; +inf = rejected (players.unary_cost).
    public static func unaryCost(xy: SIMD2<Double>, score: Double, heightM: Double?, truncated: Bool,
                                 config: PlayersConfig = .default) -> Double {
        var c = -log(npClip(score, 0.05, 1.0))
        let d = PlayerGeometry.outsideDistance(xy)
        c = c + NP.sq(d / 0.6)
        let post = abs(xy.x) < 1.2 && abs(xy.y) > CourtModel.doublesHalfWidth + 0.25
        c = c + 6.0 * (post ? 1.0 : 0.0)
        if let heightM {
            let h = nanToNum(heightM)
            c = c + (h < 0.9 ? NP.sq((0.9 - h) / 0.2) : 0.0)
            c = c + (h > 2.9 ? NP.sq((h - 2.9) / 0.3) : 0.0)
        }
        c = c + 2.0 * (truncated ? 1.0 : 0.0)
        if !(xy.x.isFinite && xy.y.isFinite) || d > config.courtMargin { c = .infinity }
        return c
    }

    /// Gate raw detections of one frame, attach floor position + unary cost and, when a
    /// frame image is given, colour descriptors of the survivors (players.build_candidates).
    /// `appearance` injects precomputed descriptors for the survivors instead (parity tests).
    public static func build<S: RGBPixelSource>(frame: Int, detections dets: FrameDetections, camera: CameraModel?,
                                                imageSize: ImageSize, image: S?, appearance: [Float]? = nil,
                                                config: PlayersConfig = .default) -> PlayerCandidates {
        let W = Double(imageSize.width), H = Double(imageSize.height)
        let minH = pyMax(12.0, 0.025 * H)
        var keep: [Int] = []
        for i in 0..<dets.n where dets.boxes[i].y2 - dets.boxes[i].y1 >= minH && dets.scores[i] > 0.0 { keep.append(i) }
        let boxes = keep.map { dets.boxes[$0] }
        let scores = keep.map { dets.scores[$0] }
        let kps = KeypointArray(keep.map { dets.keypoints[$0] })
        let hPx = boxes.map { $0.y2 - $0.y1 }
        let (foot, fromAnkle) = PlayerGeometry.footPixels(boxes: boxes, kps: kps, kpConf: config.kpConf)
        let truncated = boxes.map { $0.y2 >= H - 2 }
        let sit = PlayerGeometry.sittingScore(kps: kps, kpConf: config.kpConf).map { config.wSit * $0 }
        let n = boxes.count
        var pos: [SIMD2<Double>], unary = [Double](repeating: 0, count: n)
        var unit: [Double], noise: [Double]
        if let camera {
            pos = PlayerGeometry.floorPositions(camera: camera, footPx: foot, fromAnkle: fromAnkle, ankleH: config.ankleH)
            for i in 0..<n {
                let height = PlayerGeometry.personHeight(camera: camera, xy: pos[i], topV: boxes[i].y1)
                unary[i] = unaryCost(xy: pos[i], score: scores[i], heightM: height, truncated: truncated[i],
                                     config: config) + sit[i]
            }
            noise = foot.map { 0.06 + 4.0 * PlayerGeometry.metresPerPx(camera: camera, foot: $0) }
            unit = [Double](repeating: 1, count: n)
        } else {
            pos = foot
            unit = hPx.map { config.personH / max($0, 1.0) }
            noise = unit.map { 0.1 + 4.0 * $0 }
            for i in 0..<n {
                let cu = abs(foot[i].x - W / 2) / (W / 2)
                var u = -log(npClip(scores[i], 0.05, 1.0)) + 2.0 * NP.sq(cu) + sit[i]
                u = u + (foot[i].y < 0.2 * H ? 3.0 : 0.0) + 2.0 * (truncated[i] ? 1.0 : 0.0)
                unary[i] = u
            }
        }
        let ok = (0..<n).filter { unary[$0].isFinite }
        let okBoxes = ok.map { boxes[$0] }, okKps = kps.subset(ok)
        var app: [Float]? = nil
        if let appearance {
            precondition(appearance.count == ok.count * Appearance.dim, "injected appearance has wrong size")
            app = appearance
        } else if let image {
            app = AppearanceDescriptor.descriptors(image, boxes: okBoxes, kps: okKps, kpConf: config.kpConf)
        }
        return PlayerCandidates(frame: frame, boxes: okBoxes, scores: ok.map { scores[$0] }, kps: okKps,
                                footPx: ok.map { foot[$0] }, pos: ok.map { pos[$0] }, unary: ok.map { unary[$0] },
                                unit: ok.map { unit[$0] }, noise: ok.map { noise[$0] }, app: app)
    }

    /// Convenience without an image.
    public static func build(frame: Int, detections: FrameDetections, camera: CameraModel?, imageSize: ImageSize,
                             appearance: [Float]? = nil, config: PlayersConfig = .default) -> PlayerCandidates {
        build(frame: frame, detections: detections, camera: camera, imageSize: imageSize,
              image: ImageU8?.none, appearance: appearance, config: config)
    }

    // MARK: static people (seated officials, spectators)

    /// Per candidate: fraction of sampled frames within ±STATIC_WIN_S with a detection
    /// on the same spot (players.static_fraction).
    public static func staticFraction(_ frames: [PlayerCandidates], times: [Double],
                                      config: PlayersConfig = .default) -> [[Double]] {
        let K = frames.count
        var out: [[Double]] = []
        out.reserveCapacity(K)
        var lo = 0, hi = 0
        for k in 0..<K {
            let c = frames[k]
            while times[k] - times[lo] > config.staticWinS { lo += 1 }
            while hi < K && times[hi] - times[k] <= config.staticWinS { hi += 1 }
            var refs = (lo..<hi).filter { $0 != k }
            if c.n == 0 || refs.count < 8 {
                out.append([Double](repeating: 0, count: c.n))
                continue
            }
            if refs.count > config.staticMaxRef {
                refs = linspaceRound(0, refs.count - 1, config.staticMaxRef).map { refs[$0] }
            }
            var hit = [Bool](repeating: false, count: c.n * refs.count)
            var any = false
            for (fi, r) in refs.enumerated() where frames[r].n > 0 {
                any = true
                for q in frames[r].pos {
                    for i in 0..<c.n {
                        let d = NP.normAxis(c.pos[i] - q) * c.unit[i]
                        if d < config.staticRM { hit[i * refs.count + fi] = true }
                    }
                }
            }
            if !any {
                out.append([Double](repeating: 0, count: c.n))
                continue
            }
            out.append((0..<c.n).map { i in
                var cnt = 0
                for fi in 0..<refs.count where hit[i * refs.count + fi] { cnt += 1 }
                return Double(cnt) / Double(refs.count)
            })
        }
        return out
    }

    /// Copies with unary += weight × ramp(static fraction, 0.7 → 0.95) (players.add_static_penalty).
    public static func addStaticPenalty(_ frames: [PlayerCandidates], times: [Double], hasCamera: Bool,
                                        config: PlayersConfig = .default) -> [PlayerCandidates] {
        let sf = staticFraction(frames, times: times, config: config)
        return zip(frames, sf).map { c, s in
            var c = c
            for i in 0..<c.n {
                let ramp = npClip((s[i] - 0.7) / 0.25, 0.0, 1.0)
                let w = hasCamera
                    ? config.wStatic * npClip(PlayerGeometry.outsideDistance(c.pos[i]) / 0.3, 0.0, 1.0)
                    : config.wStaticNoCam
                c.unary[i] = c.unary[i] + w * ramp
            }
            return c
        }
    }
}

/// `np.nan_to_num(x, nan=0.0)`.
@inline(__always) func nanToNum(_ x: Double) -> Double {
    if x.isNaN { return 0 }
    if x == .infinity { return .greatestFiniteMagnitude }
    if x == -.infinity { return -.greatestFiniteMagnitude }
    return x
}

/// `np.linspace(lo, hi, num).round().astype(int)`.
func linspaceRound(_ lo: Int, _ hi: Int, _ num: Int) -> [Int] {
    guard num > 1 else { return num == 1 ? [lo] : [] }
    let step = Double(hi - lo) / Double(num - 1)
    return (0..<num).map { i in
        let y = i == num - 1 ? Double(hi) : Double(i) * step + Double(lo)
        return Int(y.rounded(.toNearestOrEven))
    }
}
