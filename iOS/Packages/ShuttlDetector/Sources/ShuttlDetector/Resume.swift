import Foundation
import ShuttlCore

/// A detect pass that can continue after the app was killed.
///
/// Exact, not approximate: with `stride == seqLen` (both production models) windows
/// never overlap, so every frame at or after a window-aligned `nextFrame` depends only
/// on its own frames and (TrackNetV3) the per-segment background plan, which is saved
/// here so the background pass isn't repeated either.
public struct DetectResume: Sendable, Codable, Equatable {
    /// Window-aligned; frames `< nextFrame` are final.
    public var nextFrame: Int
    /// Final candidates of frames `0..<nextFrame`.
    public var candidates: [[ShuttleCandidate]]
    /// Serialized `BackgroundPlan` (TrackNetV3), nil for models without a background.
    public var background: Data?
    /// Resume only with the same model.
    public var detectorName: String

    public init(nextFrame: Int, candidates: [[ShuttleCandidate]], background: Data?, detectorName: String) {
        self.nextFrame = nextFrame; self.candidates = candidates
        self.background = background; self.detectorName = detectorName
    }
}

extension BackgroundPlan {
    private static let magic: UInt32 = 0x5342_4731   // "SBG1"

    /// Compact binary: magic, count, then per segment start, width, height, RGB bytes.
    public func serialized() -> Data {
        var d = Data()
        func put(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        put(Self.magic)
        put(UInt32(starts.count))
        for (s, img) in zip(starts, images) {
            put(UInt32(s)); put(UInt32(img.width)); put(UInt32(img.height))
            d.append(img.pixels, count: img.count)
        }
        return d
    }

    public static func deserialize(_ data: Data) -> BackgroundPlan? {
        var off = 0
        func get() -> UInt32? {
            guard off + 4 <= data.count else { return nil }
            let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) }
            off += 4
            return UInt32(littleEndian: v)
        }
        guard get() == magic, let n = get() else { return nil }
        var starts: [Int] = [], images: [NetImage] = []
        for _ in 0..<n {
            guard let s = get(), let w = get(), let h = get() else { return nil }
            let img = NetImage(width: Int(w), height: Int(h))
            guard off + img.count <= data.count else { return nil }
            data.withUnsafeBytes { img.pixels.update(from: $0.baseAddress!.advanced(by: off).assumingMemoryBound(to: UInt8.self), count: img.count) }
            off += img.count
            starts.append(Int(s)); images.append(img)
        }
        return BackgroundPlan(starts: starts, images: images)
    }
}

public extension HeatmapDetector {
    /// Why a resume state was not used (nil = it was).
    func resumeProblem(_ r: DetectResume, nFrames: Int?) -> String? {
        let S = spec.seqLen
        if r.detectorName != spec.name { return "detector \(r.detectorName) ≠ \(spec.name)" }
        if stride != S { return "overlapping windows (stride \(stride) < \(S))" }
        if r.nextFrame <= 0 || r.nextFrame % S != 0 { return "nextFrame \(r.nextFrame) not window-aligned" }
        if r.candidates.count != r.nextFrame { return "\(r.candidates.count) candidate frames ≠ nextFrame \(r.nextFrame)" }
        if let n = nFrames, r.nextFrame > n - S { return "nextFrame \(r.nextFrame) inside the tail of \(n) frames" }
        if spec.usesBackground {
            guard let b = r.background, BackgroundPlan.deserialize(b) != nil else { return "background missing or unreadable" }
        }
        return nil
    }

    /// `detectFrames` that can start from `resume` and reports progress snapshots.
    /// `checkpoint` is called at the first window boundary after `checkpointEvery` frames
    /// (rounded to the window) OR `checkpointWallSeconds` of `clock` time since the last save
    /// (the pass start counts as one), whichever comes first. A hot phone needs 40-130 s of wall
    /// time for 120 s of video; the wall bound caps what a kill throws away. An unusable `resume`
    /// is ignored (full run), never trusted: `lastResumeProblem` says why.
    func detectFrames(_ source: DetectorFrameSource, width: Int, height: Int, fps: Double, nFrames: Int,
                      resume: DetectResume?, checkpointEvery: Int?,
                      checkpointWallSeconds: Double? = nil,
                      clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
                      checkpoint: ((DetectResume) -> Void)?,
                      progress: ProgressHandler? = nil, isCancelled: @escaping () -> Bool = { false }) throws
        -> [[ShuttleCandidate]] {
        let S = spec.seqLen
        var start = 0, plan: BackgroundPlan? = nil
        var out: [[ShuttleCandidate]] = []
        lastResumeProblem = nil
        if let r = resume {
            if let why = resumeProblem(r, nFrames: nFrames) {
                lastResumeProblem = why
            } else {
                start = r.nextFrame
                out = r.candidates
                plan = r.background.flatMap(BackgroundPlan.deserialize)
            }
        }
        let geom = geometry(width: width, height: height)
        let every = checkpointEvery.map { max(S, $0 / S * S) }
        var lastSaved = start
        var lastSavedAt = clock()
        let wall = checkpointWallSeconds.flatMap { $0 > 0 ? $0 : nil }
        // The background is serialised once: a resumed pass re-saves the bytes it loaded (never
        // a re-serialisation of memory), a fresh pass serialises its sealed plan at the first save.
        var bgData: Data? = plan != nil ? resume?.background : nil
        try iterHeatmaps(source, width: width, height: height, fps: fps, nFrames: nFrames, progress: progress,
                         isCancelled: isCancelled, resumeFrom: start, resumePlan: plan,
                         onBoundary: checkpoint == nil || (every == nil && wall == nil) ? nil : { next, usedPlan in
                             guard next > lastSaved, next <= nFrames - S else { return }
                             let byVideo = every.map { next - lastSaved >= $0 } ?? false
                             let now = clock()
                             guard byVideo || wall.map({ now - lastSavedAt >= $0 }) ?? false else { return }
                             if usedPlan != nil && bgData == nil { bgData = usedPlan?.serialized() }
                             while out.count < next { out.append([]) }
                             checkpoint?(DetectResume(nextFrame: next, candidates: Array(out.prefix(next)),
                                                      background: bgData, detectorName: self.spec.name))
                             lastSaved = next
                             // From after the save: the bound is on detect time lost, not on save cost.
                             lastSavedAt = clock()
                         }) { idx, hm in
            while out.count < idx { out.append([]) }
            out.append(candidates(hm, geometry: geom))
        }
        if out.count < nFrames { out += Array(repeating: [], count: nFrames - out.count) }
        if out.count > nFrames { out.removeLast(out.count - nFrames) }
        return out
    }

    /// `detectVideo` with resume and periodic checkpoints: every 120 s of video or 30 s of wall
    /// time since the last save, whichever comes first (nil wall = video time only).
    func detectVideo(_ source: VideoSource, resume: DetectResume?, checkpointSeconds: Double = 120,
                     checkpointWallSeconds: Double? = 30,
                     checkpoint: (@Sendable (DetectResume) -> Void)?,
                     progress: ProgressHandler? = nil) async throws -> [[ShuttleCandidate]] {
        let meta = source.meta
        let frames = VideoFrameSource(source)
        let every = Int((checkpointSeconds * max(meta.fps, 1)).rounded())
        return try await runOffCooperativePool { cancelled in
            try self.detectFrames(frames, width: meta.width, height: meta.height, fps: meta.fps, nFrames: meta.nFrames,
                                  resume: resume, checkpointEvery: every, checkpointWallSeconds: checkpointWallSeconds,
                                  checkpoint: checkpoint, progress: progress, isCancelled: cancelled)
        }
    }
}
