import XCTest
import ShuttlCore
@testable import ShuttlDetector

/// Heatmap = red channel of each frame blended with the background's green channel, so
/// the output depends on both the frames and the (per-segment) background plan.
final class FrameBgNet: HeatmapNetwork {
    let spec: NetSpec
    init(_ spec: NetSpec) { self.spec = spec }
    func predict(_ windows: [NetWindow], into out: UnsafeMutablePointer<Float>) throws {
        let plane = detectorNetWidth * detectorNetHeight
        for (b, w) in windows.enumerated() {
            for s in 0..<spec.seqLen {
                let img = w.frames[s], o = out + (b * spec.seqLen + s) * plane
                for i in 0..<plane {
                    let bg = w.background.map { Float($0.pixels[3 * i + 1]) } ?? 0
                    o[i] = min(1, (Float(img.pixels[3 * i]) + 0.25 * bg) / 255)
                }
            }
        }
    }
}

/// Moving bright blob over a scene that changes at `cut`, so backgrounds differ per segment.
func blobSource(_ n: Int, cut: Int, calls: CallCounter? = nil) -> GeneratedFrameSource {
    GeneratedFrameSource(count: n) { i in
        calls?.hit()
        var img = ImageU8(width: 128, height: 72, channels: 3)
        let g: UInt8 = i >= cut ? 180 : 60
        for p in 0..<(128 * 72) { img.data[3 * p + 1] = g }
        let x = (7 * i) % 116, y = (3 * i) % 60
        for yy in y..<(y + 10) { for xx in x..<(x + 10) { img.data[(yy * 128 + xx) * 3] = 250 } }
        return img
    }
}

final class CallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var count: Int { lock.withLock { n } }
    func hit() { lock.withLock { n += 1 } }
}

final class ResumeTests: XCTestCase {
    private let fps = 30.0

    private func detector(_ spec: NetSpec, stride: Int? = nil) -> HeatmapDetector {
        var c = config(stride: stride, batch: 2)
        c.background.maxSegmentSeconds = 3      // several background segments in a short clip
        return HeatmapDetector(network: FrameBgNet(spec), config: c)
    }

    /// Stopping at every checkpoint and resuming gives exactly the uninterrupted result.
    func testResumeFromEveryCheckpointIsExact() throws {
        for spec in [NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                             cornerGeometry: false, peakOffset: -0.2),
                     NetSpec(name: "wasblike", seqLen: 3, usesBackground: false, imagenetNorm: false, logits: false,
                             cornerGeometry: true, peakOffset: -0.5)] {
            let n = 301                                            // not a window multiple: exercises the tail
            let full = try detector(spec).detectFrames(blobSource(n, cut: 140), width: 128, height: 72, fps: fps, nFrames: n)

            var saved: [DetectResume] = []
            let det = detector(spec)
            let again = try det.detectFrames(blobSource(n, cut: 140), width: 128, height: 72, fps: fps, nFrames: n,
                                             resume: nil, checkpointEvery: 40, checkpoint: { saved.append($0) })
            XCTAssertEqual(again, full, "\(spec.name): checkpointing must not change the result")
            XCTAssertGreaterThanOrEqual(saved.count, 5, spec.name)
            XCTAssertEqual(saved.map { $0.nextFrame % spec.seqLen }, Array(repeating: 0, count: saved.count))
            XCTAssertTrue(saved.allSatisfy { $0.nextFrame <= n - spec.seqLen })
            XCTAssertEqual(saved.first?.background != nil, spec.usesBackground)

            for cp in saved {
                // Through JSON, as the app would persist it.
                let restored = try JSONDecoder().decode(DetectResume.self, from: JSONEncoder().encode(cp))
                XCTAssertEqual(restored, cp)
                let calls = CallCounter()
                let d2 = detector(spec)
                let resumed = try d2.detectFrames(blobSource(n, cut: 140, calls: calls), width: 128, height: 72, fps: fps,
                                                  nFrames: n, resume: restored, checkpointEvery: nil, checkpoint: nil)
                XCTAssertNil(d2.lastResumeProblem, "\(spec.name) @\(cp.nextFrame): \(d2.lastResumeProblem ?? "")")
                XCTAssertEqual(resumed, full, "\(spec.name): resume from \(cp.nextFrame) differs")
                // A generated source can't seek (skipped frames are still made), but the
                // background pass, which reads the source a second time, must not run.
                XCTAssertEqual(calls.count, n, "\(spec.name) @\(cp.nextFrame): one pass only, no background re-estimation")
            }
        }
    }

    func testBackgroundPlanRoundTrip() throws {
        let a = NetImage(width: 4, height: 2), b = NetImage(width: 4, height: 2)
        for i in 0..<a.count { a.pixels[i] = UInt8(i); b.pixels[i] = UInt8(255 - i) }
        let plan = BackgroundPlan(starts: [0, 90], images: [a, b])
        let back = try XCTUnwrap(BackgroundPlan.deserialize(plan.serialized()))
        XCTAssertEqual(back.starts, [0, 90])
        XCTAssertEqual(back.index(for: 89), 0); XCTAssertEqual(back.index(for: 90), 1)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: back.images[1].pixels, count: b.count)),
                       Array(UnsafeBufferPointer(start: b.pixels, count: b.count)))
        XCTAssertNil(BackgroundPlan.deserialize(Data([1, 2, 3])))
        XCTAssertNil(BackgroundPlan.deserialize(plan.serialized().dropLast(3)))
    }

    /// A bad resume state is never trusted: the run starts over and says why.
    func testUnusableResumeFallsBackToFullRun() throws {
        let spec = NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                           cornerGeometry: false, peakOffset: -0.2)
        let n = 200
        let full = try detector(spec).detectFrames(blobSource(n, cut: 90), width: 128, height: 72, fps: fps, nFrames: n)
        var saved: [DetectResume] = []
        _ = try detector(spec).detectFrames(blobSource(n, cut: 90), width: 128, height: 72, fps: fps, nFrames: n,
                                            resume: nil, checkpointEvery: 48, checkpoint: { saved.append($0) })
        let good = try XCTUnwrap(saved.first)
        var wrongModel = good; wrongModel.detectorName = "wasb"
        var misaligned = good; misaligned.nextFrame += 1; misaligned.candidates.append([])
        var short = good; short.candidates.removeLast()
        var noBg = good; noBg.background = nil
        var tail = good; tail.nextFrame = 200; tail.candidates = Array(repeating: [], count: 200)   // aligned, > n − S
        for (bad, expect) in [(wrongModel, "detector"), (misaligned, "window-aligned"), (short, "candidate frames"),
                              (noBg, "background"), (tail, "tail")] {
            let d = detector(spec)
            let out = try d.detectFrames(blobSource(n, cut: 90), width: 128, height: 72, fps: fps, nFrames: n,
                                         resume: bad, checkpointEvery: nil, checkpoint: nil)
            XCTAssertEqual(out, full, expect)
            XCTAssertTrue(d.lastResumeProblem?.contains(expect) == true, "\(expect): \(d.lastResumeProblem ?? "nil")")
        }
    }

    /// The phone's path: a real video file, where resume SEEKS (VideoFrameSource start)
    /// instead of decoding from the beginning. Fake net, real AVFoundation frames.
    func testResumeOnRealVideoSeeksAndIsExact() async throws {
        let url = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test split missing")
        let video = try await VideoSource.open(url)
        let m = video.meta
        let spec = NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                           cornerGeometry: false, peakOffset: -0.2)
        let full = try detector(spec).detectFrames(VideoFrameSource(video), width: m.width, height: m.height,
                                                   fps: m.fps, nFrames: m.nFrames)
        var saved: [DetectResume] = []
        _ = try detector(spec).detectFrames(VideoFrameSource(video), width: m.width, height: m.height, fps: m.fps,
                                            nFrames: m.nFrames, resume: nil, checkpointEvery: 48,
                                            checkpoint: { saved.append($0) })
        XCTAssertGreaterThanOrEqual(saved.count, 3)
        for cp in saved {
            let d = detector(spec)
            let resumed = try d.detectFrames(VideoFrameSource(video), width: m.width, height: m.height, fps: m.fps,
                                             nFrames: m.nFrames, resume: cp, checkpointEvery: nil, checkpoint: nil)
            XCTAssertNil(d.lastResumeProblem)
            XCTAssertEqual(resumed, full, "real clip: resume from \(cp.nextFrame) differs")
        }
    }

    /// Production shape (the 45-min SIGKILL matrix bug): phone-size frames (1280×720 → the
    /// parallel area resize), prefetch on, batch 4 (TrackNetV3 default), with and without TTA
    /// flip, ≥ 5 background segments, resume
    /// in a LATE segment. The resumed pass must re-save exactly the background bytes it loaded,
    /// never change a segment in memory, and give the uninterrupted result.
    func testLateResumeKeepsEveryBackgroundSegmentBitExact() throws {
        let spec = NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                           cornerGeometry: false, peakOffset: -0.2)
        for tta in [false, true] {
        func det() -> HeatmapDetector {
            var c = config(batch: 4, prefetch: true)
            c.ttaFlip = tta
            c.background.maxSegmentSeconds = 2
            return HeatmapDetector(network: FrameBgNet(spec), config: c)
        }
        let n = 480, w = 1280, h = 720, fps = 30.0             // 2 s segments → 8 segments
        func source() -> GeneratedFrameSource {
            GeneratedFrameSource(count: n) { i in
                var img = ImageU8(width: w, height: h, channels: 3)
                let seg = i / 60
                img.data.withUnsafeMutableBufferPointer { p in
                    for y in 0..<h {
                        let blue = UInt8(truncatingIfNeeded: y + 13 * seg)
                        for x in 0..<w {
                            let o: Int = (y * w + x) * 3
                            let green: Int = (x / 40 + y / 40 + 37 * seg) % 200 + 30      // per-segment scene
                            p[o + 1] = UInt8(green)
                            p[o + 2] = blue
                        }
                    }
                    let bx = (23 * i) % (w - 40), by = (11 * i) % (h - 40)
                    for y in by..<(by + 30) { for x in bx..<(bx + 30) { p[(y * w + x) * 3] = 250 } }
                }
                return img
            }
        }
        let full = try det().detectFrames(source(), width: w, height: h, fps: fps, nFrames: n)
        var saved: [DetectResume] = []
        let first = try det().detectFrames(source(), width: w, height: h, fps: fps, nFrames: n, resume: nil,
                                           checkpointEvery: 48, checkpoint: { saved.append($0) })
        XCTAssertEqual(first, full)
        let loaded = try XCTUnwrap(saved.first?.background)
        let plan = try XCTUnwrap(BackgroundPlan.deserialize(loaded))
        XCTAssertGreaterThanOrEqual(plan.images.count, 5, "needs several segments")
        XCTAssertTrue(saved.allSatisfy { $0.background == loaded }, "a fresh pass serialises its plan once")
        for cp in saved.filter({ $0.nextFrame >= 3 * n / 5 }) {             // resume inside a late segment
            var again: [DetectResume] = []
            let d = det()
            let resumed = try d.detectFrames(source(), width: w, height: h, fps: fps, nFrames: n, resume: cp,
                                             checkpointEvery: 48, checkpoint: { again.append($0) })
            XCTAssertNil(d.lastResumeProblem)
            XCTAssertEqual(resumed, full, "resume from \(cp.nextFrame) differs from the uninterrupted pass")
            XCTAssertTrue(again.allSatisfy { $0.background == cp.background },
                          "resume from \(cp.nextFrame): the re-saved background must be the loaded bytes")
        }
        XCTAssertGreaterThanOrEqual(saved.filter { $0.nextFrame >= 3 * n / 5 }.count, 2)
        }
    }

    /// The detector runs on a sealed, digest-checked copy of the plan.
    func testSealedPlanIsReadOnlyAndDetectsChanges() throws {
        let a = NetImage(width: 8, height: 4), b = NetImage(width: 8, height: 4)
        for i in 0..<a.count { a.pixels[i] = UInt8(i % 251); b.pixels[i] = UInt8((3 * i) % 251) }
        let plan = BackgroundPlan(starts: [0, 50], images: [a, b])
        XCTAssertNil(plan.digests)
        let sealed = plan.sealed()
        XCTAssertTrue(sealed.images.allSatisfy(\.isSealed))
        XCTAssertEqual(sealed.changedSegments(), [])
        XCTAssertEqual(sealed.serialized(), plan.serialized(), "sealing keeps the bytes")
        XCTAssertTrue(sealed.sealed() === sealed)
        // A plan whose images are NOT sealed but carry digests (only via this path) would report
        // changes; the sealed pages themselves can't be written (mprotect), so change the source
        // and re-check digests of a fresh sealed copy instead.
        a.pixels[5] &+= 1
        let resealed = plan.sealed()
        XCTAssertNotEqual(resealed.digests?[0], sealed.digests?[0])
        XCTAssertEqual(resealed.digests?[1], sealed.digests?[1])
    }

    /// Overlapping windows can't be cut exactly, so no checkpoints are offered.
    func testNoCheckpointsWithOverlappingWindows() throws {
        let spec = NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                           cornerGeometry: false, peakOffset: -0.2)
        var saved = 0
        _ = try detector(spec, stride: 4).detectFrames(blobSource(160, cut: 70), width: 128, height: 72, fps: fps,
                                                       nFrames: 160, resume: nil, checkpointEvery: 16,
                                                       checkpoint: { _ in saved += 1 })
        XCTAssertEqual(saved, 0)
    }

    /// P2-1 (phone T4: killed 130 s into a hot detect, before the first 120-s-of-video save):
    /// a save also comes once `checkpointWallSeconds` of wall time passed since the last one.
    /// The clock here is driven by the frames the source produced (0.5 s each, background pass
    /// included), so the video bound alone would never fire.
    func testWallClockBoundSavesBeforeTheVideoBoundAndResumesExactly() throws {
        let spec = NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                           cornerGeometry: false, peakOffset: -0.2)
        let n = 301
        let full = try detector(spec).detectFrames(blobSource(n, cut: 140), width: 128, height: 72, fps: fps, nFrames: n)

        let calls = CallCounter()
        let clock = { Double(calls.count) * 0.5 }
        var saved: [(cp: DetectResume, at: Double)] = []
        let again = try detector(spec).detectFrames(blobSource(n, cut: 140, calls: calls), width: 128, height: 72,
                                                    fps: fps, nFrames: n, resume: nil, checkpointEvery: 100_000,
                                                    checkpointWallSeconds: 10, clock: clock,
                                                    checkpoint: { saved.append(($0, clock())) })
        XCTAssertEqual(again, full, "wall-clock saves must not change the result")
        XCTAssertGreaterThanOrEqual(saved.count, 8, "≈ every 20 frames of 0.5 s over ~290 frames")
        let frames = saved.map(\.cp.nextFrame)
        XCTAssertEqual(frames, frames.sorted())
        XCTAssertEqual(Set(frames).count, frames.count, "never two saves at one frame")
        XCTAssertTrue(frames.allSatisfy { $0 % spec.seqLen == 0 && $0 <= n - spec.seqLen })
        for (a, b) in zip(saved, saved.dropFirst()) {
            XCTAssertGreaterThanOrEqual(b.at - a.at, 10, "no save before the wall bound")
            // First boundary after the bound: at most one batch (2 windows × 8 frames) late.
            XCTAssertLessThanOrEqual(b.cp.nextFrame - a.cp.nextFrame, 20 + 16)
        }
        XCTAssertEqual(saved.first?.cp.background != nil, true)

        for (cp, _) in saved {
            let d = detector(spec)
            let resumed = try d.detectFrames(blobSource(n, cut: 140), width: 128, height: 72, fps: fps, nFrames: n,
                                             resume: cp, checkpointEvery: nil, checkpoint: nil)
            XCTAssertNil(d.lastResumeProblem, "@\(cp.nextFrame): \(d.lastResumeProblem ?? "")")
            XCTAssertEqual(resumed, full, "resume from wall-clock save @\(cp.nextFrame) differs")
        }
    }

    /// Video bound and wall bound together: whichever comes first. A wall bound that never
    /// elapses leaves the video-time saves exactly as before; no wall bound = video time only.
    func testWallClockBoundKeepsTheVideoRule() throws {
        let spec = NetSpec(name: "tnv3like", seqLen: 8, usesBackground: true, imagenetNorm: false, logits: false,
                           cornerGeometry: false, peakOffset: -0.2)
        let n = 301
        func saves(every: Int?, wall: Double?, step: Double) throws -> [Int] {
            let calls = CallCounter()
            var out: [Int] = []
            _ = try detector(spec).detectFrames(blobSource(n, cut: 140, calls: calls), width: 128, height: 72, fps: fps,
                                                nFrames: n, resume: nil, checkpointEvery: every,
                                                checkpointWallSeconds: wall, clock: { Double(calls.count) * step },
                                                checkpoint: { out.append($0.nextFrame) })
            return out
        }
        let videoOnly = try saves(every: 40, wall: nil, step: 0.5)
        XCTAssertFalse(videoOnly.isEmpty)
        XCTAssertEqual(try saves(every: 40, wall: 1e9, step: 0.5), videoOnly, "an unreached wall bound changes nothing")
        XCTAssertEqual(try saves(every: 100_000, wall: nil, step: 0.5), [], "no bound reached, no save")
        XCTAssertEqual(try saves(every: 100_000, wall: 0, step: 0.5), [], "a zero wall bound is off, not 'always'")
        // A cool phone (fast clock steps): the video bound still fires on its own schedule.
        let both = try saves(every: 40, wall: 1e9, step: 0.001)
        XCTAssertEqual(both, videoOnly)
        // A hot phone: the wall bound wins and saves more often than the video bound.
        let hot = try saves(every: 40, wall: 5, step: 0.5)
        XCTAssertGreaterThan(hot.count, videoOnly.count)
        XCTAssertLessThanOrEqual(try XCTUnwrap(hot.first), try XCTUnwrap(videoOnly.first),
                                 "the first save comes no later than with the video rule alone")
    }
}
