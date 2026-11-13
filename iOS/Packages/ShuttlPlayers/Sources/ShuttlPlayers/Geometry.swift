import Foundation
import ShuttlCore

// Geometry of players.py: foot point, floor position, apparent height, floor scale,
// seated-pose score. Keypoints are a flat row-major (N, 17, 3) array like numpy.

/// Flat (N,17,3) keypoint array.
public struct KeypointArray: Sendable, Equatable {
    public var values: [Double]
    public init(values: [Double]) { precondition(values.count % 51 == 0); self.values = values }
    public init(_ rows: [[Keypoint]]) {
        var v = [Double](repeating: 0, count: rows.count * 51)
        for (i, row) in rows.enumerated() {
            for (j, k) in row.prefix(17).enumerated() {
                v[i * 51 + j * 3] = k.x; v[i * 51 + j * 3 + 1] = k.y; v[i * 51 + j * 3 + 2] = k.conf
            }
        }
        values = v
    }
    public var count: Int { values.count / 51 }
    @inline(__always) public func x(_ i: Int, _ j: Int) -> Double { values[i * 51 + j * 3] }
    @inline(__always) public func y(_ i: Int, _ j: Int) -> Double { values[i * 51 + j * 3 + 1] }
    @inline(__always) public func c(_ i: Int, _ j: Int) -> Double { values[i * 51 + j * 3 + 2] }
    public func row(_ i: Int) -> ArraySlice<Double> { values[(i * 51)..<(i * 51 + 51)] }
    public func keypoints(_ i: Int) -> [Keypoint] {
        (0..<17).map { Keypoint(x: x(i, $0), y: y(i, $0), conf: c(i, $0)) }
    }
    public func subset(_ idx: [Int]) -> KeypointArray {
        var v: [Double] = []
        v.reserveCapacity(idx.count * 51)
        for i in idx { v.append(contentsOf: row(i)) }
        return KeypointArray(values: v)
    }
}

public enum PlayerGeometry {
    /// (N,) foot pixel and "from ankles" flag per candidate (players.foot_pixels).
    /// Both ankles confident → ankle midpoint (≈ 8 cm above the floor); else bbox
    /// bottom with u = mean of the confident ankles/hips (robust to a raised arm).
    public static func footPixels(boxes: [BBox], kps: KeypointArray, kpConf: Double = PlayersConfig.default.kpConf)
        -> (foot: [SIMD2<Double>], fromAnkle: [Bool]) {
        var out: [SIMD2<Double>] = [], ank: [Bool] = []
        out.reserveCapacity(boxes.count); ank.reserveCapacity(boxes.count)
        for i in 0..<boxes.count {
            if kps.c(i, KP.lAnk) >= kpConf, kps.c(i, KP.rAnk) >= kpConf {
                out.append(SIMD2((kps.x(i, KP.lAnk) + kps.x(i, KP.rAnk)) / 2,
                                 (kps.y(i, KP.lAnk) + kps.y(i, KP.rAnk)) / 2))
                ank.append(true)
                continue
            }
            var xs: [Double] = []
            for j in [KP.lAnk, KP.rAnk, KP.lHip, KP.rHip] where kps.c(i, j) >= kpConf { xs.append(kps.x(i, j)) }
            let b = boxes[i]
            let u = xs.isEmpty ? (b.x1 + b.x2) / 2 : NP.mean(xs)
            out.append(SIMD2(pyMin(pyMax(u, b.x1), b.x2), b.y2))
            ank.append(false)
        }
        return (out, ank)
    }

    /// Foot pixels → floor metres (ankle rays cut the z = ANKLE_H plane).
    public static func floorPositions(camera: CameraModel, footPx: [SIMD2<Double>], fromAnkle: [Bool],
                                      ankleH: Double = PlayersConfig.default.ankleH) -> [SIMD2<Double>] {
        zip(footPx, fromAnkle).map { uv, a in
            let p = camera.pixelToFloor(uv, z: a ? ankleH : 0.0)
            return SIMD2(p.x, p.y)
        }
    }

    /// Distance (m) outside the doubles court rectangle, 0 inside.
    @inline(__always)
    public static func outsideDistance(_ xy: SIMD2<Double>) -> Double {
        let dx = npClipLow(abs(xy.x) - CourtModel.halfLength)
        let dy = npClipLow(abs(xy.y) - CourtModel.doublesHalfWidth)
        return hypot(dx, dy)
    }

    /// Apparent standing height (m) of a person with feet at `xy` and bbox top row `topV`.
    public static func personHeight(camera: CameraModel, xy: SIMD2<Double>, topV: Double) -> Double {
        let p0 = camera.project(SIMD3(xy.x, xy.y, 0))
        let p2 = camera.project(SIMD3(xy.x, xy.y, 2.0))
        let dv = p0.y - p2.y
        return dv > 1e-6 ? 2.0 * (p0.y - topV) / dv : .nan
    }

    /// Local floor scale (m/px) at a foot pixel: geometric mean of the du, dv steps.
    public static func metresPerPx(camera: CameraModel, foot: SIMD2<Double>) -> Double {
        let a = camera.pixelToFloor(foot)
        let bu = camera.pixelToFloor(foot + SIMD2(1.0, 0.0))
        let bv = camera.pixelToFloor(foot + SIMD2(0.0, 1.0))
        let s = (NP.normAxis(SIMD2(bu.x - a.x, bu.y - a.y)) * NP.normAxis(SIMD2(bv.x - a.x, bv.y - a.y))).squareRoot()
        return s.isFinite ? s : 0.05
    }

    /// 0 = standing/lunging … 1 = seated (thigh vs torso vertical image extent).
    public static func sittingScore(kps: KeypointArray, kpConf: Double = PlayersConfig.default.kpConf) -> [Double] {
        var out = [Double](repeating: 0, count: kps.count)
        for i in 0..<kps.count {
            let c = { (j: Int) in kps.c(i, j) >= kpConf }
            guard c(KP.lSh), c(KP.rSh), c(KP.lHip), c(KP.rHip) else { continue }
            let torso = (kps.y(i, KP.lHip) + kps.y(i, KP.rHip) - kps.y(i, KP.lSh) - kps.y(i, KP.rSh)) / 2
            var r: [Double] = []
            for (hp, kn) in [(KP.lHip, KP.lKnee), (KP.rHip, KP.rKnee)] where c(kn) {
                r.append(abs(kps.y(i, kn) - kps.y(i, hp)) / torso)
            }
            if torso > 2.0, !r.isEmpty {
                out[i] = npClip((0.5 - pyMax(r)) / 0.25, 0.0, 1.0)
            }
        }
        return out
    }
}

// MARK: - Python / numpy scalar semantics

/// Python `max(a, b)`: returns `a` unless `b > a`.
@inline(__always) func pyMax(_ a: Double, _ b: Double) -> Double { b > a ? b : a }
/// Python `min(a, b)`: returns `a` unless `b < a`.
@inline(__always) func pyMin(_ a: Double, _ b: Double) -> Double { b < a ? b : a }
/// Python `max(list)`.
func pyMax(_ xs: [Double]) -> Double { xs.dropFirst().reduce(xs[0]) { pyMax($0, $1) } }
/// `np.clip(x, lo, hi)` = minimum(maximum(x, lo), hi) (NaN propagates).
@inline(__always) func npClip(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
    if x.isNaN { return x }
    return min(max(x, lo), hi)
}
/// `np.clip(x, 0, None)`.
@inline(__always) func npClipLow(_ x: Double) -> Double { x.isNaN ? x : max(x, 0) }
