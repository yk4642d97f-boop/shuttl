import CoreVideo
import Foundation
import ShuttlCore

public extension PlayerTracker {
    /// Detect, filter, identify and smooth the on-court players of a video
    /// (players.track_players). One PlayerObservation per frame per player while tracked
    /// (skipped frames interpolated, gaps ≤ 0.5 s bridged), sorted by (frame, track id).
    ///
    /// - step: pose sampling step; pipeline_v2 uses `playerStep(fps:)` (15 Hz).
    /// - detector: overrides the default Apple Vision detector (tests, other models).
    /// - farCrop: nil = default: the Vision detector always adds the far-half ROI pass when
    ///   a camera is known; a custom `detector` is expected to handle it (see `defaultFarROI`,
    ///   players.py's rule). true/false forces it for the Vision detector.
    /// - batch: frames per detector call. Results do not depend on it (every frame is
    ///   processed independently); 1 keeps a single decoded buffer alive.
    /// - progress: fraction of the video processed, after every batch.
    /// - isCancelled: polled before every sampled frame; throws `CancellationError` when true.
    ///   Also honours the current task's cancellation when called from one.
    /// - resume: continue a killed pass (see `PlayersResume`); validated, never trusted. An
    ///   unusable one is ignored (full run) and `resumeReport` gets why; nil there = it was used.
    /// - checkpointPolicy / checkpoint: hand out everything needed to continue, after a batch,
    ///   when the policy says so. Never at the end of the video (the result is saved then).
    /// - onFrame: called with each sampled frame's index once its candidates are built (a
    ///   watchdog's heartbeat; a blocked detector stops calling it).
    static func trackPlayers(source: VideoSource, camera: CameraModel?, step: Int = 2, doubles: Bool? = nil,
                             detector: PoseDetector? = nil, farCrop: Bool? = nil,
                             farLayout: FarCropLayout = defaultFarLayout, batch: Int = 1,
                             config: PlayersConfig = .default,
                             resume: PlayersResume? = nil,
                             checkpointPolicy: PlayersCheckpointPolicy? = nil,
                             checkpoint: ((PlayersResume) -> Void)? = nil,
                             resumeReport: ((String?) -> Void)? = nil,
                             onFrame: ((Int) -> Void)? = nil,
                             progress: ((Double) -> Void)? = nil,
                             isCancelled: () -> Bool = { false }) throws -> PlayerTracks {
        let meta = source.meta
        let step = max(1, step)
        // Vision misses far players that YOLO still finds on the full frame (ShuttlPlayersBench:
        // far recall 0.38 vs 0.94 with a far-half crop), so its crop passes run whenever a camera
        // is known (`farCrop: false` = full frame only); custom detectors handle crops themselves.
        let rois = farCrop == false ? [] : defaultFarROIs(camera: camera, meta: meta, layout: farLayout, config: config)
        // Pose preference: the CoreML RTMPose-m port when its compiled model ships in the app
        // (the model the stroke classifier was trained on), else Apple Vision. Apple Vision stays
        // the fallback path — this is a migration, not a removal. `detector:` overrides both.
        let det = detector ?? Self.defaultPoseDetector(farROIs: rois)
        let nTotal = meta.nFrames
        var frames: [PlayerCandidates] = []
        var buf: [CVPixelBuffer] = [], bufIdx: [Int] = []

        let context = resumeContext(camera: camera, meta: meta, step: step, rois: rois,
                                    detector: String(reflecting: type(of: det)), config: config)
        var start = 0
        // Encoded form of `frames`, filled lazily at saves; a resumed pass keeps the bytes it loaded.
        var encoded: [Data] = []
        if let resume {
            let (problem, restored) = validate(resume, context: context, step: step, nFrames: nTotal)
            if problem == nil {
                frames = restored; encoded = resume.frames; start = resume.nextFrame
            }
            resumeReport?(problem)
        }
        let policy = checkpoint == nil ? nil : checkpointPolicy
        var lastSaved = start
        var lastSavedAt = policy?.clock() ?? 0

        func maybeCheckpoint() {
            guard let policy, let checkpoint, let last = frames.last?.frame else { return }
            let next = last + step
            guard next > lastSaved, next < nTotal else { return }
            let byVideo = policy.everyFrames.map { next - lastSaved >= $0 } ?? false
            guard byVideo || policy.wallSeconds.map({ policy.clock() - lastSavedAt >= $0 }) ?? false else { return }
            while encoded.count < frames.count { encoded.append(frames[encoded.count].encoded()) }
            checkpoint(PlayersResume(nextFrame: next, context: context, frames: encoded))
            lastSaved = next
            lastSavedAt = policy.clock()
        }

        func flush() throws {
            guard !buf.isEmpty else { return }
            let dets = try det.detect(frames: buf, frameIndices: bufIdx)
            precondition(dets.count == buf.count, "detector must return one result per frame")
            for ((fi, pb), d) in zip(zip(bufIdx, buf), dets) {
                let c = try BGRAPixels.with(pb) { px in
                    Candidates.build(frame: fi, detections: d, camera: camera, imageSize: meta.size,
                                     image: px, config: config)
                }
                frames.append(c)
                onFrame?(fi)
            }
            buf.removeAll(keepingCapacity: true); bufIdx.removeAll(keepingCapacity: true)
            if let progress, nTotal > 0 { progress(min(1.0, Double(frames[frames.count - 1].frame + 1) / Double(nTotal))) }
            maybeCheckpoint()
        }

        // `start` is on the step grid, so the sampled indices are those of a full pass.
        let seq = try source.frames(start: start, step: step)
        for (fi, pb) in seq {
            if isCancelled() || Task.isCancelled { throw CancellationError() }
            buf.append(pb); bufIdx.append(fi)
            if buf.count >= max(1, batch) { try flush() }
        }
        if let err = seq.error { throw err }
        try flush()
        return assembleTracks(frames, meta: meta, camera: camera, doubles: doubles, config: config)
    }

    /// Far-crop layout of the default Vision detector: `.lateral`, chosen by
    /// `ShuttlPlayersBench --layouts` (12 TrackNetV2 Test clips vs YOLO tracks, 2676 frames at
    /// 15 Hz). Phone-like frames (tripod 1.2–1.6 m, 0.5–3 m behind the baseline): far detection
    /// recall single 0.55–0.59 → lateral 0.96–0.97, track recall 0.74–0.79 → 1.00, foot p95
    /// 0.38–0.50 → 0.24–0.26 m; broadcast 0.937 → 0.962. Cost: one more Vision pass
    /// (≈ 10 ms per sampled frame on an M4: 22 → 31 ms). `.depth` is worse at the same cost.
    static let defaultFarLayout: FarCropLayout = .lateral

    /// Crop rects of the default Vision detector: `layout` over the far half when a camera
    /// is known, else none.
    static func defaultFarROIs(camera: CameraModel?, meta: VideoMeta, layout: FarCropLayout = defaultFarLayout,
                               config: PlayersConfig = .default) -> [PixelRect] {
        guard let camera else { return [] }
        return FarCrops.rects(layout, camera: camera, size: meta.size, courtMargin: config.courtMargin)
    }

    /// The far-half ROI players.track_players would crop (nil = full frame only).
    static func defaultFarROI(camera: CameraModel?, meta: VideoMeta, farCrop: Bool?,
                              config: PlayersConfig = .default) -> PixelRect? {
        guard let camera, farCrop != false,
              let fr = FarROI.compute(camera: camera, size: meta.size, courtMargin: config.courtMargin) else { return nil }
        return (farCrop == true || fr.baselineHeightPx < 0.11 * Double(meta.height)) ? fr.rect : nil
    }

    /// The pose detector a pass uses when the caller does not pass one: the CoreML RTMPose-m
    /// port when its compiled model is in the bundle, else Apple Vision.
    ///
    /// RTMPose is what the stroke classifier was trained on (0.8730 stroke accuracy vs Apple
    /// Vision's 0.5305 over 318 identical clips), so it is preferred; Vision boxes still supply
    /// the person proposals and Vision remains the fallback if the model is missing or fails to
    /// load. The model is 54 MB, so a missing/unloadable one is a real possibility (a build
    /// without the resource, or a device below the ANE floor) and must not take the pass down.
    static func defaultPoseDetector(farROIs: [PixelRect] = [], bundle: Bundle = .main) -> PoseDetector {
        if let url = RTMPoseCoreML.bundledModelURL(in: bundle), url.pathExtension == "mlmodelc",
           let model = RTMPoseCoreML.loadCompiled(url: url) {
            return RTMPoseDetector(model: model, options: .init(farROIs: farROIs))
        }
        return VisionPoseDetector(options: .init(farROIs: farROIs))
    }
}

public extension Movement {
    /// serialize.py's `movement` block: {"near": stats, "far": stats}.
    static func summary(_ tracks: PlayerTracks, config: PlayersConfig = .default) -> [CourtSide: MovementStats] {
        [.near: stats(tracks, side: .near, config: config), .far: stats(tracks, side: .far, config: config)]
    }
}
