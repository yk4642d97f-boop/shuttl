import Accelerate
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Streaming video access (port of video_io.py). Never holds a whole match in RAM.
///
/// Frames are delivered as 32BGRA `CVPixelBuffer`s, UPRIGHT (the track's rotation
/// metadata is applied, like OpenCV's auto-rotation). Frame index = position in
/// presentation order; `meta.timestamps[i]` is its time in seconds from the first frame.
public final class VideoSource: @unchecked Sendable {
    public let url: URL
    public let meta: VideoMeta
    /// Absolute presentation time of each frame, ascending.
    public let presentationTimes: [CMTime]
    let asset: AVURLAsset
    let track: AVAssetTrack
    /// Clockwise rotation (0/90/180/270) that makes the stored frame upright.
    let rotation: Int
    /// Compressed samples that shared another sample's presentation time and so never
    /// become a frame (0 for any file written by Apple's frameworks).
    public let droppedDuplicateSamples: Int

    private init(url: URL, meta: VideoMeta, presentationTimes: [CMTime], asset: AVURLAsset, track: AVAssetTrack, rotation: Int,
                 droppedDuplicateSamples: Int = 0) {
        self.url = url; self.meta = meta; self.presentationTimes = presentationTimes
        self.asset = asset; self.track = track; self.rotation = rotation
        self.droppedDuplicateSamples = droppedDuplicateSamples
    }

    /// Probe metadata and per-frame timestamps (one pass over compressed samples, no decode).
    public static func open(_ url: URL) async throws -> VideoSource {
        let asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let tracks: [AVAssetTrack]
        do { tracks = try await asset.loadTracks(withMediaType: .video) }
        catch { throw ShuttlCoreError.cannotOpenVideo(url, error.localizedDescription) }
        guard let track = tracks.first else { throw ShuttlCoreError.noVideoTrack(url) }
        let (natural, transform, nominalFPS) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate)
        let rotation = rotationDegrees(transform)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw ShuttlCoreError.readerFailed(reader.error?.localizedDescription ?? "startReading")
        }
        var pts: [CMTime] = []
        while let sb = output.copyNextSampleBuffer() {
            let n = CMSampleBufferGetNumSamples(sb)
            guard n > 0 else { continue }
            if n == 1 {
                pts.append(CMSampleBufferGetPresentationTimeStamp(sb))
            } else {
                var timing = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: n)
                var got = 0
                CMSampleBufferGetSampleTimingInfoArray(sb, entryCount: n, arrayToFill: &timing, entriesNeededOut: &got)
                pts.append(contentsOf: timing.prefix(got).map(\.presentationTimeStamp))
            }
        }
        if reader.status == .failed { throw ShuttlCoreError.readerFailed(reader.error?.localizedDescription ?? "probe") }
        pts.sort { CMTimeCompare($0, $1) < 0 }
        // Samples sharing a presentation time (a broken muxer; Apple's writers refuse them)
        // are decoded to ONE frame: AVAssetReader drops the rest. A frame is what the
        // decoder delivers, so count each PTS once — otherwise frame indices would have
        // gaps and every stage expecting consecutive frames would fail.
        let sampleCount = pts.count
        pts = dedupSorted(pts)
        let droppedDuplicates = sampleCount - pts.count

        let fps = nominalFPS > 0 ? Double(nominalFPS) : 30
        let first = pts.first.map(CMTimeGetSeconds) ?? 0
        var ts: [Double] = pts.map { (p: CMTime) -> Double in CMTimeGetSeconds(p) - first }
        // Distinct CMTimes can still collapse to equal Doubles (huge timescales): nudge by
        // one ulp so times stay strictly increasing without discarding the real clock.
        for i in ts.indices.dropFirst() where !(ts[i] > ts[i - 1]) {
            ts[i] = ts[i - 1].nextUp
        }
        let w = Int(abs(natural.width).rounded()), h = Int(abs(natural.height).rounded())
        let upright = rotation % 180 == 0 ? (w, h) : (h, w)
        let meta = VideoMeta(fps: fps, width: upright.0, height: upright.1, nFrames: ts.count,
                             timestamps: ts, rotationDeg: rotation)
        return VideoSource(url: url, meta: meta, presentationTimes: pts, asset: asset, track: track, rotation: rotation,
                           droppedDuplicateSamples: droppedDuplicates)
    }

    /// Ascending CMTimes with exact duplicates removed (first kept).
    static func dedupSorted(_ sorted: [CMTime]) -> [CMTime] {
        var out: [CMTime] = []
        out.reserveCapacity(sorted.count)
        for t in sorted where out.last.map({ CMTimeCompare($0, t) != 0 }) ?? true { out.append(t) }
        return out
    }

    /// Sequential upright frames `start..<stop` every `step` frames (all frames are
    /// decoded; skipped ones are just not converted).
    public func frames(start: Int = 0, stop: Int? = nil, step: Int = 1) throws -> FrameSequence {
        try FrameSequence(source: self, start: start, stop: min(stop ?? meta.nFrames, meta.nFrames), step: max(1, step))
    }

    /// `n` evenly spaced frames (np.linspace indices), each converted with `transform`
    /// right away so full-resolution buffers are never accumulated.
    public func sampleFrames<T>(_ n: Int, transform: (CVPixelBuffer) throws -> T) throws -> [(index: Int, value: T)] {
        let total = meta.nFrames
        guard total > 0, n > 0 else { return [] }
        let idx = VideoSource.linspaceIndices(count: min(n, total), total: total)
        let meanGap = idx.count > 1 ? Double(idx.last! - idx.first!) / Double(idx.count - 1) : 0
        var out: [(Int, T)] = []
        // A reader that failed or stalled ends the sequence early; that is an error, not fewer
        // frames, or calibration would solve (and checkpoint) a camera from whatever came out.
        if idx.count < 2 || meanGap < 120 {
            let want = Set(idx)
            let seq = try frames(stop: idx.last! + 1)
            for (i, pb) in seq where want.contains(i) { out.append((i, try transform(pb))) }
            if let e = seq.error { throw e }
        } else {
            for i in idx {
                let seq = try frames(start: i, stop: i + 1)
                for (j, pb) in seq where j == i { out.append((i, try transform(pb))) }
                if let e = seq.error { throw e }
            }
        }
        return out.map { (index: $0.0, value: $0.1) }
    }

    /// Frame index for a presentation time (nearest probed frame).
    func frameIndex(for time: CMTime) -> Int {
        var lo = 0, hi = presentationTimes.count - 1
        guard hi >= 0 else { return 0 }
        while lo < hi {
            let mid = (lo + hi) / 2
            if CMTimeCompare(presentationTimes[mid], time) < 0 { lo = mid + 1 } else { hi = mid }
        }
        if lo > 0 {
            let a = CMTimeGetSeconds(presentationTimes[lo - 1]), b = CMTimeGetSeconds(presentationTimes[lo])
            if abs(CMTimeGetSeconds(time) - a) < abs(b - CMTimeGetSeconds(time)) { return lo - 1 }
        }
        return lo
    }

    /// `sorted(set(np.linspace(0, total - 1, count).astype(int)))`.
    static func linspaceIndices(count: Int, total: Int) -> [Int] {
        guard count > 1 else { return count == 1 ? [0] : [] }
        let span = Double(total - 1), denom = Double(count - 1)
        var out = Set<Int>()
        for i in 0..<count {
            let v: Int = i == count - 1 ? total - 1 : Int(Double(i) * span / denom)
            out.insert(v)
        }
        return out.sorted()
    }

    static func rotationDegrees(_ t: CGAffineTransform) -> Int {
        let deg = Int((atan2(Double(t.b), Double(t.a)) * 180 / .pi).rounded())
        return ((deg % 360) + 360) % 360 / 90 * 90
    }
}

/// Iterator over (frameIndex, upright 32BGRA pixel buffer). Not thread-safe; use from one task.
public final class FrameSequence: Sequence, IteratorProtocol {
    private let source: VideoSource
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private let start: Int, stop: Int, step: Int
    private let tolerance: Double
    private var lastIndex = -1
    private var rotatedPool: CVPixelBufferPool?
    private var stalled: ShuttlCoreError?

    /// Longest wait for ONE decoded sample before the read is abandoned. A 1080p/4K sample takes
    /// milliseconds (a long-GOP seek well under a second); on the phone a hardware decoder held
    /// by a live camera preview kept a read blocked for 15 min before failing with -11847.
    nonisolated(unsafe) public static var stallTimeout: Double = 20
    /// Test seam (fault injection in tests of the packages above): stands in for
    /// `output.copyNextSampleBuffer()`. Never set in the app.
    nonisolated(unsafe) public static var copyNextHook: ((AVAssetReaderTrackOutput) -> CMSampleBuffer?)?

    init(source: VideoSource, start: Int, stop: Int, step: Int) throws {
        self.source = source; self.start = start; self.stop = stop; self.step = step
        reader = try AVAssetReader(asset: source.asset)
        output = AVAssetReaderTrackOutput(track: source.track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        if start > 0, start < source.presentationTimes.count {
            // Start exactly on the frame's PTS: a range starting between frames makes
            // AVAssetReader emit the PREVIOUS frame re-stamped to the range start.
            reader.timeRange = CMTimeRange(start: source.presentationTimes[start], duration: .positiveInfinity)
        }
        tolerance = 0.25 / Swift.max(source.meta.fps, 1)
        guard start < stop, reader.startReading() else {
            if start >= stop { return }
            throw ShuttlCoreError.readerFailed(reader.error?.localizedDescription ?? "startReading")
        }
    }

    deinit { if reader.status == .reading { reader.cancelReading() } }

    public func next() -> (Int, CVPixelBuffer)? {
        guard start < stop else { return nil }
        while let sb = copyNextWatched() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let i = source.frameIndex(for: pts)
            // Only frames that land on a probed PTS, strictly in order (drops re-stamped pre-roll).
            guard i > lastIndex,
                  abs(CMTimeGetSeconds(pts) - CMTimeGetSeconds(source.presentationTimes[i])) <= tolerance
            else { continue }
            lastIndex = i
            if i < start { continue }
            if i >= stop { reader.cancelReading(); return nil }
            if (i - start) % step != 0 { continue }
            guard let upright = uprightBuffer(pb) else { continue }
            return (i, upright)
        }
        return nil
    }

    /// Error that ended iteration early, if any (nil after a clean end of stream).
    public var error: Error? { stalled ?? (reader.status == .failed ? reader.error : nil) }

    /// `copyNextSampleBuffer` with a watchdog: when the decoder delivers nothing for
    /// `stallTimeout`, the timer cancels the reader (which unblocks the call) and `error`
    /// reports `decodeStalled`, so a stalled stage fails fast instead of looking like an end.
    private func copyNextWatched() -> CMSampleBuffer? {
        guard stalled == nil else { return nil }
        let fired = WatchdogFlag(), reader = self.reader, frame = lastIndex, timeout = Self.stallTimeout
        let dog = DispatchWorkItem { fired.set(); reader.cancelReading() }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + timeout, execute: dog)
        let sb = Self.copyNextHook.map { $0(output) } ?? output.copyNextSampleBuffer()
        dog.cancel()
        if fired.isSet {
            stalled = .decodeStalled(seconds: timeout, frame: frame)
            return nil
        }
        return sb
    }

    private func uprightBuffer(_ pb: CVPixelBuffer) -> CVPixelBuffer? {
        guard source.rotation != 0 else { return pb }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let (ow, oh) = source.rotation == 180 ? (w, h) : (h, w)
        if rotatedPool == nil {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: ow, kCVPixelBufferHeightKey as String: oh,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &rotatedPool)
        }
        guard let pool = rotatedPool else { return nil }
        var outPB: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outPB)
        guard let dstPB = outPB else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly); CVPixelBufferLockBaseAddress(dstPB, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly); CVPixelBufferUnlockBaseAddress(dstPB, []) }
        var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(h),
                                width: vImagePixelCount(w), rowBytes: CVPixelBufferGetBytesPerRow(pb))
        var dst = vImage_Buffer(data: CVPixelBufferGetBaseAddress(dstPB), height: vImagePixelCount(oh),
                                width: vImagePixelCount(ow), rowBytes: CVPixelBufferGetBytesPerRow(dstPB))
        // vImage rotates counter-clockwise; the stored frame needs `rotation` degrees clockwise.
        let ccwQuarterTurns: UInt8 = switch source.rotation {
        case 90: UInt8(kRotate270DegreesCounterClockwise)
        case 180: UInt8(kRotate180DegreesCounterClockwise)
        default: UInt8(kRotate90DegreesCounterClockwise)
        }
        var black: [UInt8] = [0, 0, 0, 255]
        let err = vImageRotate90_ARGB8888(&src, &dst, ccwQuarterTurns, &black, vImage_Flags(kvImageNoFlags))
        return err == kvImageNoError ? dstPB : nil
    }
}

// MARK: - Presentation-clock gaps (video_io.pts_gaps)

/// Holes in the presentation clock (paused / edited / dropped-frame files). A frame step
/// `dt[i] = t[i] − t[i−1]` is a gap when `dt[i] > max(minS, factor × median positive step)`.
/// 0.30 s = the linker's longest bridged miss burst; real footage measured ≤ 2.02 × median
/// and ≤ 34 ms. Identical rule and constants to backend video_io.py.
public enum PTSGaps {
    public static let minS = 0.30
    public static let factor = 4.0

    /// First frame index after every gap (sorted; the frame BEFORE it is index − 1) — the
    /// same convention as linker scene cuts. Needs ≥ 2 finite positive steps; non-finite
    /// times never form a gap.
    public static func frames(_ t: [Double]) -> [Int] {
        guard t.count >= 3 else { return [] }
        var pos: [Double] = []
        pos.reserveCapacity(t.count - 1)
        for i in 1..<t.count {
            let d = t[i] - t[i - 1]
            if d.isFinite && d > 0 { pos.append(d) }
        }
        guard pos.count >= 2 else { return [] }
        pos.sort()
        let m = pos.count / 2
        let med = pos.count % 2 == 1 ? pos[m] : (pos[m - 1] + pos[m]) / 2      // np.median
        let thr = Swift.max(minS, factor * med)
        var out: [Int] = []
        for i in 1..<t.count {
            let d = t[i] - t[i - 1]
            if d.isFinite && d > thr { out.append(i) }
        }
        return out
    }

    /// `diagnostics["video"]["pts_gaps"]` (same keys as pipeline_v2): frame = first frame
    /// after the gap, t_before / t_after = times (s) either side, dt_s = t_after − t_before.
    public static func info(_ t: [Double]) -> JSONValue {
        .array(frames(t).map { i in
            .object(["frame": .number(Double(i)), "t_before": .number(t[i - 1]),
                     "t_after": .number(t[i]), "dt_s": .number(t[i] - t[i - 1])])
        })
    }
}

extension VideoMeta {
    /// First frame after every presentation-clock gap (`PTSGaps.frames(timestamps)`).
    public var ptsGaps: [Int] { PTSGaps.frames(timestamps) }
}

/// Set once by the decode watchdog, read by the decoding thread.
private final class WatchdogFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}
