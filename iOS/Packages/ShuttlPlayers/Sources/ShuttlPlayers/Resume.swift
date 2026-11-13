import CryptoKit
import Foundation
import ShuttlCore

/// A players pass that can continue after the app was killed.
///
/// Exact, not approximate: every sampled frame's `PlayerCandidates` depends only on its own
/// pixels, the camera and the configuration, and tracks are assembled once at the end from
/// all of them. So the candidates of the frames already sampled, stored bit for bit, plus the
/// frames after `nextFrame` give exactly the uninterrupted result.
public struct PlayersResume: Sendable, Equatable {
    /// On the sampling grid (a multiple of `step`); frames `< nextFrame` are final.
    public var nextFrame: Int
    /// `PlayerTracker.resumeContext`: what the candidates depend on beyond the store's
    /// fingerprint (the camera of THIS run, step, crop rects, detector).
    public var context: String
    /// One `PlayerCandidates.encoded()` per sampled frame `< nextFrame`, in order.
    public var frames: [Data]

    public init(nextFrame: Int, context: String, frames: [Data]) {
        self.nextFrame = nextFrame; self.context = context; self.frames = frames
    }
}

/// When the players pass hands out a `PlayersResume`: at the first sampled frame after
/// `everyFrames` video frames OR `wallSeconds` of `clock` time since the last save (the pass
/// start counts as one), whichever comes first. nil disables that bound.
public struct PlayersCheckpointPolicy: Sendable {
    public var everyFrames: Int?
    public var wallSeconds: Double?
    public var clock: @Sendable () -> TimeInterval

    public init(everyFrames: Int?, wallSeconds: Double?,
                clock: @escaping @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) {
        self.everyFrames = everyFrames.flatMap { $0 > 0 ? $0 : nil }
        self.wallSeconds = wallSeconds.flatMap { $0 > 0 ? $0 : nil }
        self.clock = clock
    }
}

public extension PlayerTracker {
    static let resumeFormat = 1

    /// Identifies what a run's candidates depend on beyond the video and the app build: a
    /// resume whose context differs (another camera after a re-calibration, another step) is
    /// ignored. SHA-256 over exact bit patterns.
    static func resumeContext(camera: CameraModel?, meta: VideoMeta, step: Int, rois: [PixelRect],
                              detector: String, config: PlayersConfig = .default) -> String {
        func hex(_ x: Double) -> String { String(x.bitPattern, radix: 16) }
        var parts = ["format=\(resumeFormat)", "step=\(max(1, step))", "size=\(meta.width)x\(meta.height)",
                     "frames=\(meta.nFrames)", "detector=\(detector)", "config=\(String(describing: config))",
                     "rois=" + rois.map { "\($0.x1),\($0.y1),\($0.x2),\($0.y2)" }.joined(separator: ";")]
        if let c = camera {
            let k = [c.K.columns.0, c.K.columns.1, c.K.columns.2].flatMap { [$0.x, $0.y, $0.z] }
            let r = [c.R.columns.0, c.R.columns.1, c.R.columns.2].flatMap { [$0.x, $0.y, $0.z] }
            let nums = k + r + [c.t.x, c.t.y, c.t.z] + c.dist + [c.reprojRmsePx, c.confidence]
            parts.append("camera=\(c.imageSize.width)x\(c.imageSize.height):" + nums.map(hex).joined(separator: ","))
        } else {
            parts.append("camera=nil")
        }
        let digest = SHA256.hash(data: Data(parts.joined(separator: "\n").utf8))
        return "p\(resumeFormat)-" + digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Why a resume state can't be used (nil = it can), and its decoded candidates.
    static func validate(_ r: PlayersResume, context: String, step: Int, nFrames: Int)
        -> (problem: String?, frames: [PlayerCandidates]) {
        let step = max(1, step)
        if r.context != context { return ("context \(r.context.prefix(12)) ≠ \(context.prefix(12))", []) }
        if r.nextFrame <= 0 || r.nextFrame % step != 0 { return ("nextFrame \(r.nextFrame) not on the step-\(step) grid", []) }
        if r.nextFrame >= nFrames { return ("nextFrame \(r.nextFrame) ≥ \(nFrames) frames", []) }
        if r.frames.count != r.nextFrame / step {
            return ("\(r.frames.count) sampled frames ≠ \(r.nextFrame / step) below nextFrame \(r.nextFrame)", [])
        }
        var out: [PlayerCandidates] = []
        out.reserveCapacity(r.frames.count)
        for (k, d) in r.frames.enumerated() {
            guard let c = PlayerCandidates(encoded: d) else { return ("sampled frame \(k) unreadable", []) }
            guard c.frame == k * step else { return ("sampled frame \(k) is frame \(c.frame), expected \(k * step)", []) }
            out.append(c)
        }
        return (nil, out)
    }
}

/// Exact binary form of one frame's candidates (every Double / Float as its bit pattern, so
/// NaN, -0 and the last ulp survive): Int64 frame · per array a UInt32 count + values ·
/// UInt8 hasApp (+ count + Float32 values).
public extension PlayerCandidates {
    func encoded() -> Data {
        var b: [UInt8] = []
        b.reserveCapacity(64 + n * (4 * 8 + 8 + 51 * 8 + 4 * 8 + 3 * 8 + Appearance.dim * 4))
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { b += $0 } }
        func f64(_ v: Double) { withUnsafeBytes(of: v.bitPattern.littleEndian) { b += $0 } }
        func f64s(_ a: [Double]) { u32(a.count); for v in a { f64(v) } }
        func v2s(_ a: [SIMD2<Double>]) { u32(a.count); for v in a { f64(v.x); f64(v.y) } }
        withUnsafeBytes(of: Int64(frame).littleEndian) { b += $0 }
        u32(boxes.count); for x in boxes { f64(x.x1); f64(x.y1); f64(x.x2); f64(x.y2) }
        f64s(scores); f64s(kps.values); v2s(footPx); v2s(pos); f64s(unary); f64s(unit); f64s(noise)
        if let app {
            b.append(1); u32(app.count)
            for v in app { withUnsafeBytes(of: v.bitPattern.littleEndian) { b += $0 } }
        } else {
            b.append(0)
        }
        return Data(b)
    }

    /// nil for anything `encoded()` could not have produced.
    init?(encoded data: Data) {
        let bytes = [UInt8](data)
        var pos = 0
        func raw<T: FixedWidthInteger>(_: T.Type) -> T? {
            let w = MemoryLayout<T>.size
            guard pos + w <= bytes.count else { return nil }
            var v: T = 0
            for i in 0..<w { v |= T(truncatingIfNeeded: bytes[pos + i]) << (8 * i) }
            pos += w
            return v
        }
        func count(_ width: Int) -> Int? {
            guard let n = raw(UInt32.self), Int(n) * width <= bytes.count - pos else { return nil }
            return Int(n)
        }
        func f64() -> Double? { raw(UInt64.self).map(Double.init(bitPattern:)) }
        func f64s() -> [Double]? {
            guard let n = count(8) else { return nil }
            var a: [Double] = []; a.reserveCapacity(n)
            for _ in 0..<n { guard let v = f64() else { return nil }; a.append(v) }
            return a
        }
        func v2s() -> [SIMD2<Double>]? {
            guard let n = count(16) else { return nil }
            var a: [SIMD2<Double>] = []; a.reserveCapacity(n)
            for _ in 0..<n { guard let x = f64(), let y = f64() else { return nil }; a.append(SIMD2(x, y)) }
            return a
        }
        guard let frame = raw(Int64.self), let nb = count(32) else { return nil }
        var boxes: [BBox] = []
        boxes.reserveCapacity(nb)
        for _ in 0..<nb {
            guard let x1 = f64(), let y1 = f64(), let x2 = f64(), let y2 = f64() else { return nil }
            boxes.append(BBox(x1: x1, y1: y1, x2: x2, y2: y2))
        }
        guard let scores = f64s(), let kps = f64s(), kps.count % 51 == 0, let foot = v2s(), let pos2 = v2s(),
              let unary = f64s(), let unit = f64s(), let noise = f64s(), let has = raw(UInt8.self) else { return nil }
        var app: [Float]?
        if has == 1 {
            guard let n = count(4) else { return nil }
            var a: [Float] = []; a.reserveCapacity(n)
            for _ in 0..<n { guard let v = raw(UInt32.self) else { return nil }; a.append(Float(bitPattern: v)) }
            app = a
        } else if has != 0 {
            return nil
        }
        guard pos == bytes.count else { return nil }
        self.init(frame: Int(frame), boxes: boxes, scores: scores, kps: KeypointArray(values: kps), footPx: foot,
                  pos: pos2, unary: unary, unit: unit, noise: noise, app: app)
    }
}
