import Accelerate
import Foundation
import ShuttlCore

/// One network input window: `frames` (seq_len net canvases, oldest first) and, for
/// background nets, the segment's median background (prepended as the first RGB frame).
public struct NetWindow: Sendable {
    public let start: Int
    public let frames: [NetImage]
    public let background: NetImage?

    public init(start: Int, frames: [NetImage], background: NetImage?) {
        self.start = start; self.frames = frames; self.background = background
    }

    /// Planar float input (inChannels, H, W) = x / 255 in RGB order, background first, then
    /// oldest … newest (frame-major channels, as detector.py `_forward` reshapes them).
    public func packPlanar(into dst: UnsafeMutablePointer<Float>) {
        var c = 0
        let imgs = (background.map { [$0] } ?? []) + frames
        for img in imgs {
            let n = img.width * img.height
            for ch in 0..<3 {
                let out = dst + (c + ch) * n
                vDSP_vfltu8(img.pixels + ch, 3, out, 1, vDSP_Length(n))
                var d: Float = 255
                vDSP_vsdiv(out, 1, &d, out, 1, vDSP_Length(n))
            }
            c += 3
        }
    }
}

/// A heatmap network (the torch module of detector.py). Implementations take RGB uint8 canvases,
/// apply their own input normalisation (x/255, ImageNet mean/std) and output activation, and
/// write `windows.count × seqLen × H × W` probabilities in [0, 1] to `output`.
public protocol HeatmapNetwork: AnyObject {
    var spec: NetSpec { get }
    func predict(_ windows: [NetWindow], into output: UnsafeMutablePointer<Float>) throws
}

/// Timing / bookkeeping of the last run (detector.py `last_stats`).
public struct DetectorStats: Sendable, Equatable {
    public var frames = 0
    public var seconds = 0.0
    public var windows = 0
    /// Time inside the network (all windows).
    public var networkSeconds = 0.0
    /// Frame decode + letterbox (producer thread when prefetching).
    public var preprocessSeconds = 0.0
    public var backgroundSeconds = 0.0
    public var decodeSeconds = 0.0
    public var fps: Double { seconds > 0 ? Double(frames) / seconds : 0 }
    public init() {}
}

/// Fixed-size float buffer (one fused heatmap accumulator).
final class FloatBuffer {
    let count: Int
    let p: UnsafeMutablePointer<Float>
    var weight: Float = 0
    init(_ n: Int) { count = n; p = .allocate(capacity: n) }
    deinit { p.deallocate() }
}

/// Streaming neural detector over a heatmap net (detector.py ``HeatmapDetector``).
///
/// Memory: holds at most `seqLen + stride` preprocessed frames plus one pending batch and the
/// fusion accumulators of frames still covered by unrun windows. Not reentrant: run one
/// detection at a time per instance.
public final class HeatmapDetector: @unchecked Sendable {
    public let network: HeatmapNetwork
    public let config: DetectorConfig
    public var spec: NetSpec { network.spec }
    public let stride: Int
    public let weights: [Float]
    public let interpolation: ResizeInterpolation
    public let peakOffset: Double
    /// Instrumentation (tests): peak frames held / newest decoded − oldest unyielded.
    public private(set) var maxBuffered = 0
    public private(set) var maxLag = 0
    public private(set) var lastStats = DetectorStats()
    /// Why the last `detectFrames(resume:)` ignored its resume state (nil = used, or none given).
    public internal(set) var lastResumeProblem: String? = nil
    let decoder: HeatmapDecoder

    public init(network: HeatmapNetwork, config: DetectorConfig = DetectorConfig()) {
        self.network = network
        self.config = config
        let s = network.spec
        stride = max(1, min(config.stride ?? s.seqLen, s.seqLen))
        weights = ensembleWeights(s.seqLen, stride < s.seqLen ? config.ensemble : .average)
        interpolation = config.interpolation ?? (s.cornerGeometry ? .linear : .area)
        peakOffset = config.peakOffset ?? s.peakOffset
        decoder = HeatmapDecoder(k: config.k, minScore: config.minScore, refine: config.refine)
    }

    public func geometry(width: Int, height: Int) -> FrameGeometry {
        FrameGeometry(width: width, height: height, corner: spec.cornerGeometry)
    }

    /// Decoded top-k candidates of one heatmap in full-res px; peaks in the letterbox are dropped.
    public func candidates(_ hm: UnsafeBufferPointer<Float>, geometry g: FrameGeometry) -> [ShuttleCandidate] {
        var res: [ShuttleCandidate] = []
        for pk in decoder.decode(hm, width: g.netWidth, height: g.netHeight) {
            let f = g.peakToFull(x: pk.x, y: pk.y, peakOffset: peakOffset)
            if -1.0 <= f.x && f.x <= Double(g.width) && -1.0 <= f.y && f.y <= Double(g.height) {
                res.append(ShuttleCandidate(x: f.x, y: f.y, score: pk.score))
            }
        }
        return res
    }

    // MARK: public entry points

    /// One list per frame (count == meta.nFrames), top-k candidates sorted by score desc,
    /// full-resolution upright px, subpixel. Runs on a dedicated thread; honours task cancellation.
    public func detectVideo(_ source: VideoSource, progress: ProgressHandler? = nil) async throws -> [[ShuttleCandidate]] {
        let meta = source.meta
        let frames = VideoFrameSource(source)
        return try await runOffCooperativePool { cancelled in
            try self.detectFrames(frames, width: meta.width, height: meta.height, fps: meta.fps, nFrames: meta.nFrames,
                                  progress: progress, isCancelled: cancelled)
        }
    }

    public func detectFrames(_ source: DetectorFrameSource, width: Int, height: Int, fps: Double, nFrames: Int? = nil,
                             progress: ProgressHandler? = nil, isCancelled: @escaping () -> Bool = { false }) throws
        -> [[ShuttleCandidate]] {
        let geom = geometry(width: width, height: height)
        var out: [[ShuttleCandidate]] = []
        var tDecode = 0.0
        try iterHeatmaps(source, width: width, height: height, fps: fps, nFrames: nFrames, progress: progress,
                         isCancelled: isCancelled) { idx, hm in
            while out.count < idx { out.append([]) }
            let t = CFAbsoluteTimeGetCurrent()
            out.append(candidates(hm, geometry: geom))
            tDecode += CFAbsoluteTimeGetCurrent() - t
        }
        lastStats.decodeSeconds = tDecode
        if let n = nFrames {
            if out.count < n { out += Array(repeating: [], count: n - out.count) }
            if out.count > n { out.removeLast(out.count - n) }
        }
        return out
    }

    // MARK: streaming core

    /// Calls `body(frameIndex, fused heatmap (H·W floats in [0,1]))` in frame order.
    ///
    /// Windows start every `stride` frames. If the last full window does not reach the last
    /// frame, one tail window over the last `seqLen` real frames (clips shorter than `seqLen`:
    /// padded with the last frame, as upstream) covers the rest, contributing only to frames not
    /// yet finalised. Overlapping predictions are fused with positional weights; a frame is
    /// finalised (passed to `body` exactly once) as soon as no later window covers it.
    /// The heatmap buffer is only valid inside `body`.
    /// - resumeFrom / resumePlan: continue a run whose frames `< resumeFrom` are already final
    ///   (window-aligned, see `DetectResume`); the background pass is skipped when `resumePlan`
    ///   is given. Only exact when `stride == seqLen` (the defaults), which `detectFrames` checks.
    /// - onBoundary: after each batch, called with `nextStart` once every frame below it has
    ///   been passed to `body`, plus the background plan in use. Not called when windows
    ///   overlap (stride < seqLen), because frames past `nextStart` would then be partial.
    public func iterHeatmaps(_ source: DetectorFrameSource, width: Int, height: Int, fps: Double, nFrames: Int? = nil,
                             progress: ProgressHandler? = nil, isCancelled: @escaping () -> Bool = { false },
                             resumeFrom: Int = 0, resumePlan: BackgroundPlan? = nil,
                             onBoundary: ((Int, BackgroundPlan?) throws -> Void)? = nil,
                             body: (Int, UnsafeBufferPointer<Float>) throws -> Void) throws {
        let spec = self.spec, S = spec.seqLen, stride = self.stride
        let geom = geometry(width: width, height: height)
        let plane = geom.netWidth * geom.netHeight
        let prog = ProgressThrottle(progress)
        let t0 = CFAbsoluteTimeGetCurrent()
        var stats = DetectorStats()
        let bgShare = spec.usesBackground ? 0.15 : 0.0
        var plan: BackgroundPlan? = nil
        if spec.usesBackground {
            let p = try resumePlan ?? estimateBackgrounds(source: source, geometry: geom, fps: fps, interpolation: interpolation,
                                                          config: config.background, progress: { prog(bgShare * $0) },
                                                          nFrames: nFrames, isCancelled: isCancelled)
            plan = p.sealed()
            stats.backgroundSeconds = CFAbsoluteTimeGetCurrent() - t0
        }
        let source: DetectorFrameSource = resumeFrom > 0 ? StartingAt(base: source, start: resumeFrom) : source
        var buf: [(Int, NetImage)] = []
        buf.reserveCapacity(S + 1)
        var pending: [(start: Int, frames: [NetImage], bg: Int, lo: Int)] = []
        var acc: [Int: FloatBuffer] = [:]
        var freeBuffers: [FloatBuffer] = []
        var netOut = [Float](), netFlip = [Float]()
        let flipPool = NetImagePool(width: geom.netWidth, height: geom.netHeight)
        let wts = weights
        var nextStart = resumeFrom, total = resumeFrom, nWindows = 0, covered = resumeFrom - 1
        maxBuffered = 0
        maxLag = 0

        func run(_ limit: Int?) throws {
            guard !pending.isEmpty else { return }
            let windows = pending.map { NetWindow(start: $0.start, frames: $0.frames, background: plan?.images[$0.bg]) }
            let need = windows.count * S * plane
            if netOut.count < need { netOut = [Float](repeating: 0, count: need) }
            let t = CFAbsoluteTimeGetCurrent()
            try netOut.withUnsafeMutableBufferPointer { try network.predict(windows, into: $0.baseAddress!) }
            if config.ttaFlip {
                // y = 0.5 * (y + net(x.flip(-1)).flip(-1)) — background slot mirrored too.
                let mirrored = windows.map { w in
                    NetWindow(start: w.start, frames: w.frames.map { mirror($0, flipPool) }, background: w.background.map { mirror($0, flipPool) })
                }
                if netFlip.count < need { netFlip = [Float](repeating: 0, count: need) }
                try netFlip.withUnsafeMutableBufferPointer { try network.predict(mirrored, into: $0.baseAddress!) }
                let W = geom.netWidth, H = geom.netHeight
                netOut.withUnsafeMutableBufferPointer { y in
                    netFlip.withUnsafeBufferPointer { f in
                        for pl in 0..<(windows.count * S) {
                            let a = y.baseAddress! + pl * plane, b = f.baseAddress! + pl * plane
                            for r in 0..<H {
                                let ar = a + r * W, br = b + r * W
                                for x in 0..<W { ar[x] = 0.5 * (ar[x] + br[W - 1 - x]) }
                            }
                        }
                    }
                }
            }
            stats.networkSeconds += CFAbsoluteTimeGetCurrent() - t
            netOut.withUnsafeBufferPointer { y in
                for (b, p) in pending.enumerated() {
                    for s in 0..<S {
                        let f = p.start + s
                        if f < p.lo { continue }
                        if let limit, f >= limit { break }
                        let src = y.baseAddress! + (b * S + s) * plane
                        var w = wts[s]
                        if let a = acc[f] {
                            // a += y * w   (float32, like numpy)
                            vDSP_vsma(src, 1, &w, a.p, 1, a.p, 1, vDSP_Length(plane))
                            a.weight += w
                        } else {
                            let a = freeBuffers.popLast() ?? FloatBuffer(plane)
                            vDSP_vsmul(src, 1, &w, a.p, 1, vDSP_Length(plane))
                            a.weight = w
                            acc[f] = a
                        }
                    }
                }
            }
            pending.removeAll(keepingCapacity: true)
        }

        func finalize(upTo: Int, _ emit: (Int, UnsafeBufferPointer<Float>) throws -> Void) throws {
            for f in acc.keys.filter({ $0 < upTo }).sorted() {
                let a = acc.removeValue(forKey: f)!
                if a.weight != 1 {
                    var w = a.weight
                    vDSP_vsdiv(a.p, 1, &w, a.p, 1, vDSP_Length(plane))
                }
                try emit(f, UnsafeBufferPointer(start: a.p, count: plane))
                freeBuffers.append(a)
            }
        }

        func window(_ start: Int, _ frames: [NetImage], _ lo: Int) {
            pending.append((start, frames, plan?.index(for: start) ?? 0, lo))
            nWindows += 1
        }

        func handle(_ idx: Int, _ img: NetImage) throws {
            if idx != total { throw DetectorError.nonConsecutiveFrames(idx) }
            buf.append((idx, img))
            if buf.count > S { buf.removeFirst() }
            total = idx + 1
            if buf.count == S && buf[0].0 == nextStart {
                window(nextStart, buf.map(\.1), nextStart)
                covered = nextStart + S - 1
                nextStart += stride
                maxBuffered = max(maxBuffered, buf.count + pending.count * S)
                if pending.count >= config.batchSize {
                    try run(nil)
                    try finalize(upTo: nextStart) { f, hm in
                        maxLag = max(maxLag, total - 1 - f)
                        try body(f, hm)
                    }
                    if stride == S, let onBoundary {
                        // Frames below nextStart are final: prove they saw the background as sealed.
                        if let bad = plan?.changedSegments(), !bad.isEmpty { throw DetectorError.backgroundChanged(bad) }
                        try onBoundary(nextStart, plan)
                    }
                }
            }
            if let n = nFrames, n > 0 { prog(bgShare + (1 - bgShare) * min(Double(total) / Double(n), 1.0)) }
        }

        let interp = interpolation
        if config.prefetch {
            let pre = Prefetcher(source: source, geometry: geom, interpolation: interp, capacity: 2 * S)
            defer { pre.stop() }
            while let item = try pre.next() {
                if isCancelled() { throw CancellationError() }
                try handle(item.0, item.1)
            }
            stats.preprocessSeconds = pre.preprocessSeconds
        } else {
            let pool = NetImagePool(width: geom.netWidth, height: geom.netHeight)
            var tPre = 0.0
            var cancelled = false
            try source.forEachFrame(step: 1) { idx, view in
                if isCancelled() { cancelled = true; return false }
                let t = CFAbsoluteTimeGetCurrent()
                let img = pool.acquire()
                geom.warp(view, into: img, interpolation: interp)
                tPre += CFAbsoluteTimeGetCurrent() - t
                try handle(idx, img)
                return true
            }
            if cancelled { throw CancellationError() }
            stats.preprocessSeconds = tPre
        }
        if total > 0 && covered < total - 1 {                 // tail / clip shorter than S
            var frames = buf.map(\.1)                         // = frames max(total-S,0) ..< total
            while frames.count < S { frames.append(frames[frames.count - 1]) }   // upstream: repeat last frame
            window(max(total - S, 0), frames, min(nextStart, covered + 1))
        }
        try run(total)
        if let bad = plan?.changedSegments(), !bad.isEmpty { throw DetectorError.backgroundChanged(bad) }
        try finalize(upTo: total, body)
        stats.frames = total
        stats.windows = nWindows
        stats.seconds = CFAbsoluteTimeGetCurrent() - t0
        lastStats = stats
        progress?(1.0)
    }
}

/// Horizontal mirror of a canvas into a pooled image.
func mirror(_ img: NetImage, _ pool: NetImagePool) -> NetImage {
    let out = pool.acquire()
    let W = img.width
    for y in 0..<img.height {
        let s = img.pixels + y * W * 3, d = out.pixels + y * W * 3
        for x in 0..<W {
            let o = (W - 1 - x) * 3
            d[3 * x] = s[o]; d[3 * x + 1] = s[o + 1]; d[3 * x + 2] = s[o + 2]
        }
    }
    return out
}

// MARK: - prefetch (decode ‖ inference)

/// Runs `source` on its own thread, letterboxing every frame into a pooled canvas, with a
/// bounded hand-over queue (detector.py `_prefetch`, maxsize 2·seq_len).
final class Prefetcher: @unchecked Sendable {
    private let cond = NSCondition()
    private var queue: [(Int, NetImage)] = []
    private var finished = false
    private var failure: Error?
    private var stopped = false
    private let capacity: Int
    private(set) var preprocessSeconds = 0.0

    init(source: DetectorFrameSource, geometry geom: FrameGeometry, interpolation: ResizeInterpolation, capacity: Int) {
        self.capacity = max(1, capacity)
        let box = UncheckedBox(source)
        let thread = Thread { [self] in
            let pool = NetImagePool(width: geom.netWidth, height: geom.netHeight)
            do {
                try box.value.forEachFrame(step: 1) { idx, view in
                    let t = CFAbsoluteTimeGetCurrent()
                    let img = pool.acquire()
                    geom.warp(view, into: img, interpolation: interpolation)
                    let dt = CFAbsoluteTimeGetCurrent() - t
                    cond.lock()
                    defer { cond.unlock() }
                    preprocessSeconds += dt
                    while queue.count >= self.capacity && !stopped { cond.wait() }
                    if stopped { return false }
                    queue.append((idx, img))
                    cond.broadcast()
                    return true
                }
            } catch {
                cond.lock(); failure = error; cond.unlock()
            }
            cond.lock(); finished = true; cond.broadcast(); cond.unlock()
        }
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// Next frame, nil at the end of the stream; rethrows the producer's error.
    func next() throws -> (Int, NetImage)? {
        cond.lock()
        defer { cond.unlock() }
        while queue.isEmpty && !finished { cond.wait() }
        if !queue.isEmpty {
            let item = queue.removeFirst()
            cond.broadcast()
            return item
        }
        if let failure { throw failure }
        return nil
    }

    /// Stops the producer (it may be blocked on a full queue) and waits for it to finish.
    func stop() {
        cond.lock()
        stopped = true
        queue.removeAll()
        cond.broadcast()
        while !finished { cond.wait() }
        cond.unlock()
    }
}

struct UncheckedBox<T>: @unchecked Sendable {
    let value: T
    init(_ v: T) { value = v }
}

/// Runs blocking work on a dedicated thread (keeps the cooperative pool free) and forwards
/// task cancellation as a polled flag.
func runOffCooperativePool<T>(_ work: @escaping (@escaping () -> Bool) throws -> T) async throws -> T {
    let flag = CancelFlag()
    let job = UncheckedBox(work)
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<UncheckedBox<T>, Error>) in
            let thread = Thread {
                do { cont.resume(returning: UncheckedBox(try job.value { flag.isSet })) } catch { cont.resume(throwing: error) }
            }
            thread.qualityOfService = .userInitiated
            thread.stackSize = 8 << 20
            thread.start()
        }.value
    } onCancel: {
        flag.set()
    }
}

final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}
