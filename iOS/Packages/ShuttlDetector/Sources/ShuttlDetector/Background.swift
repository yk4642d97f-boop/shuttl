import Foundation
import ShuttlCore

/// Per-segment median backgrounds (net-canvas RGB uint8), detector.py ``BackgroundPlan``.
/// Segments split at scene cuts and every `maxSegmentSeconds`.
public final class BackgroundPlan: @unchecked Sendable {
    public internal(set) var starts: [Int] = []
    public internal(set) var images: [NetImage] = []

    public init(starts: [Int] = [], images: [NetImage] = []) { self.starts = starts; self.images = images }

    /// The same plan in read-only storage (one sealed mapping for all segments), plus a digest
    /// per segment taken at sealing. The detector runs on the sealed copy only; `verify()` then
    /// proves the background used for every window is the one that was estimated or loaded.
    public func sealed() -> BackgroundPlan {
        if images.allSatisfy(\.isSealed), digests != nil { return self }
        let sizes = images.map(\.count)
        let store = SealedPixels(count: sizes.reduce(0, +)) { dst in
            var off = 0
            for img in images { (dst + off).update(from: img.pixels, count: img.count); off += img.count }
        }
        var off = 0
        let views = images.map { img -> NetImage in
            defer { off += img.count }
            return NetImage(width: img.width, height: img.height, sealedIn: store, offset: off)
        }
        let out = BackgroundPlan(starts: starts, images: views)
        out.digests = views.map(Self.digest)
        return out
    }

    /// Per-segment digests taken when the plan was sealed (nil for an unsealed plan).
    public private(set) var digests: [UInt64]?

    /// Indices of segments whose pixels differ from their sealed digest (empty = intact).
    public func changedSegments() -> [Int] {
        guard let digests else { return [] }
        return images.indices.filter { Self.digest(images[$0]) != digests[$0] }
    }

    /// FNV-1a over 8-byte words (fast; any single-bit change alters it).
    static func digest(_ img: NetImage) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        let n = img.count, words = n / 8
        let raw = UnsafeRawPointer(img.pixels)
        for i in 0..<words { h ^= raw.loadUnaligned(fromByteOffset: 8 * i, as: UInt64.self); h &*= 0x0000_0100_0000_01b3 }
        for i in (words * 8)..<n { h ^= UInt64(img.pixels[i]); h &*= 0x0000_0100_0000_01b3 }
        return h
    }

    /// Segment index of `frame` (searchsorted(starts, frame, side="right") − 1, at least 0).
    public func index(for frame: Int) -> Int {
        var lo = 0, hi = starts.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if starts[mid] <= frame { lo = mid + 1 } else { hi = mid }
        }
        return max(lo - 1, 0)
    }
}

/// Temporal median of sampled frames per scene segment (bounded memory) — detector.py
/// ``estimate_backgrounds``. Frames are sampled every round(fps / sampleHz) frames; within a
/// segment at most 2·maxSamples are kept (reservoir: every `stride`-th sample, stride doubling).
public func estimateBackgrounds(source: DetectorFrameSource, geometry geom: FrameGeometry, fps: Double,
                                interpolation: ResizeInterpolation = .area, config: BackgroundConfig = BackgroundConfig(),
                                progress: ProgressHandler? = nil, nFrames: Int? = nil,
                                isCancelled: () -> Bool = { false }) throws -> BackgroundPlan {
    let step = max(1, Int((fps / config.sampleHz).rounded(.toNearestOrEven)))
    let segLen = max(1, Int(config.maxSegmentSeconds * fps))
    let plan = BackgroundPlan()
    var samples: [NetImage] = []
    var stride = 1, seen = 0, segStart = 0
    var prevThumb: [UInt8]? = nil
    let pool = NetImagePool(width: geom.netWidth, height: geom.netHeight)

    func close() {
        guard !samples.isEmpty else { return }
        plan.starts.append(segStart)
        plan.images.append(medianImage(samples))
    }

    var cancelled = false
    try source.forEachFrame(step: step) { idx, view in
        if isCancelled() { cancelled = true; return false }
        let img = pool.acquire()
        geom.warp(view, into: img, interpolation: interpolation)
        let thumb = CVExact.grayThumbnail(img)
        var isCut = false
        if let prev = prevThumb {
            var big = 0
            for i in thumb.indices where abs(Int(thumb[i]) - Int(prev[i])) > 40 { big += 1 }
            isCut = Double(big) / Double(thumb.count) > config.cutFraction
        }
        prevThumb = thumb
        if !samples.isEmpty && (isCut || idx - segStart >= segLen) {
            close()
            samples = []; stride = 1; seen = 0; segStart = idx
        }
        if seen % stride == 0 {
            samples.append(img)
            if samples.count >= 2 * config.maxSamples {
                samples = Swift.stride(from: 0, to: samples.count, by: 2).map { samples[$0] }
                stride *= 2
            }
        }
        seen += 1
        if let progress, let n = nFrames, n > 0 { progress(min(Double(idx) / Double(n), 1.0)) }
        return true
    }
    if cancelled { throw CancellationError() }
    close()
    if plan.images.isEmpty {
        plan.starts = [0]
        plan.images = [NetImage(width: geom.netWidth, height: geom.netHeight)]
    }
    plan.starts[0] = 0
    return plan
}

/// np.median(np.stack(samples), axis=0).astype(np.uint8): middle value, or the truncated mean
/// of the two middle values for an even count.
func medianImage(_ samples: [NetImage]) -> NetImage {
    let first = samples[0]
    let out = NetImage(width: first.width, height: first.height)
    let n = samples.count, count = first.count
    if n == 1 { out.copy(from: first); return out }
    let ptrs = samples.map(\.pixels)
    var vals = [UInt8](repeating: 0, count: n)
    let k = (n - 1) / 2
    vals.withUnsafeMutableBufferPointer { v in
        for i in 0..<count {
            for j in 0..<n { v[j] = ptrs[j][i] }
            let a = quickSelect(v, k)
            if n % 2 == 1 {
                out.pixels[i] = a
            } else {
                var b = UInt8.max
                for j in (k + 1)..<n where v[j] < b { b = v[j] }
                out.pixels[i] = UInt8((Int(a) + Int(b)) >> 1)
            }
        }
    }
    return out
}

/// k-th smallest (0-based); afterwards v[0..<k] <= v[k] <= v[k+1..<n].
@inline(__always)
private func quickSelect(_ v: UnsafeMutableBufferPointer<UInt8>, _ k: Int) -> UInt8 {
    var lo = 0, hi = v.count - 1
    while lo < hi {
        let pivot = v[(lo + hi) >> 1]
        var i = lo, j = hi
        while i <= j {
            while v[i] < pivot { i += 1 }
            while v[j] > pivot { j -= 1 }
            if i <= j { v.swapAt(i, j); i += 1; j -= 1 }
        }
        if k <= j { hi = j } else if k >= i { lo = i } else { break }
    }
    return v[k]
}
