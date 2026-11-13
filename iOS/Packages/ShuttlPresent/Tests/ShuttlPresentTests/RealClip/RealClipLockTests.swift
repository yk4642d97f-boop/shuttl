import XCTest
import CoreVideo
import simd
import ShuttlCore
import ShuttlCourt
import ShuttlPresent

/// CourtLock on the 29 real TrackNetV2 Test clips (fixed broadcast-style cameras, players moving).
/// Frames are sampled at 2 Hz for up to 8 s and fed to `CourtLockCore` with the REAL calibrator
/// in an event-driven simulation: an attempt started at video time t with measured wall time d
/// completes at t + d·factor (factor 1 = this Mac, 2.5 = phone estimate); frames arriving while
/// it runs are only buffered. False lock = first locked camera > 8 px (max court keypoint) from a
/// full-resolution reference calibration of 7 frames at 2 Hz. Gated: SHUTTL_PRESENT_REAL=1.
final class RealClipLockTests: XCTestCase {
    struct ClipResult { var locked = false; var ttl: Double?; var falseLock = false; var kpErr: Double?; var losses = 0; var motion = 0 }

    func testLockOnRealClips() async throws {
        let clips = try realClips()
        let factors = [1.0, 2.5]
        var results: [Double: [ClipResult]] = [:]
        var times: [CourtLockPlan.Kind: [Double]] = [:]
        let cal = ShuttlCourtLockCalibrator()
        let cfg = CourtLockConfiguration()
        var refAccepted = 0
        for url in clips {
            let src = try await VideoSource.open(url)
            let step = max(1, Int((src.meta.fps / 2).rounded()))
            var work: [WorkingFrame] = [], fullFirst: [ImageU8] = []
            let ds = FrameDownsampler()
            let stop = min(src.meta.nFrames, step * 16)
            for (i, pb) in try src.frames(start: 0, stop: stop, step: step) {
                let img = try ImageU8.rgb(from: pb)
                if fullFirst.count < 7 { fullFirst.append(img) }
                work.append(ds.frame(from: img, workingSize: cfg.workingSize(for: img.size), time: src.meta.time(ofFrame: i)))
            }
            let ref = CourtCalibration.calibrateFromFrames(fullFirst, size: fullFirst[0].size)
            if ref != nil { refAccepted += 1 }
            var line = "LOCK \(clipTag(url)) \(work.count) samples, ref \(ref == nil ? "reject" : "accept")"
            for factor in factors {
                var r = ClipResult()
                var core = CourtLockCore(config: cfg)
                var inflight: (plan: CourtLockPlan, attempt: CalibrationAttempt, done: Double)?
                let t0 = work[0].time, tEnd = work.last!.time + cfg.sampleInterval
                func start(_ at: Double) {
                    guard let p = core.plan() else { return }
                    let c0 = DispatchTime.now().uptimeNanoseconds
                    let a = p.run(cal)
                    let d = Double(DispatchTime.now().uptimeNanoseconds - c0) / 1e9
                    if factor == 1 { times[p.kind, default: []].append(d) }
                    inflight = (p, a, at + d * factor)
                }
                func complete(until t: Double) {
                    while let f = inflight, f.done <= t {
                        inflight = nil
                        let was = core.state.isLocked
                        core.apply(f.plan, f.attempt)
                        if core.state.isLocked && !was && r.ttl == nil {
                            r.ttl = f.done - t0; r.locked = true
                            if let ref, let cam = core.state.camera {
                                let e = CameraAgreement.maxDisagreementPx(ref, cam)
                                r.kpErr = e; r.falseLock = e > cfg.lostPx
                            }
                        }
                        start(f.done)
                    }
                }
                for f in work {
                    complete(until: f.time)
                    core.offer(f)
                    if inflight == nil { start(f.time) }
                }
                complete(until: tEnd)
                r.losses = core.stats.losses; r.motion = core.stats.motionEvents
                results[factor, default: []].append(r)
                line += String(format: " | x%.1f %@ ttl %@ kp %@ losses %d motion %d final %@", factor,
                               r.locked ? "LOCK" : "none", r.ttl.map { String(format: "%.2fs", $0) } ?? "-",
                               r.kpErr.map { String(format: "%.2f", $0) } ?? "-", r.losses, r.motion,
                               core.state.kindName)
            }
            print(line)
        }
        func med(_ v: [Double]) -> Double { v.isEmpty ? .nan : v.sorted()[v.count / 2] }
        func pct(_ v: [Double], _ q: Double) -> Double { v.isEmpty ? .nan : v.sorted()[min(v.count - 1, Int(Double(v.count) * q))] }
        logLine("[real-lock \(Self.stamp())] 29 clips, 2 Hz samples ≤ 8 s, working 0.75x (960x540), full-res reference accepted \(refAccepted)/\(clips.count)")
        for factor in factors {
            let rs = results[factor] ?? []
            let ttl = rs.compactMap(\.ttl)
            logLine(String(format: "[real-lock] speed x%.1f (%@): lock rate %d/%d, false locks %d, time-to-lock median %.2f s (%.0f frames @30fps) p90 %.2f s max %.2f s, lock-vs-ref kp max median %.2f px worst %.2f px, losses after lock %d, motion events %d",
                           factor, factor == 1 ? "Mac M4" : "phone estimate", rs.filter(\.locked).count, rs.count,
                           rs.filter(\.falseLock).count, med(ttl), med(ttl) * 30, pct(ttl, 0.9), ttl.max() ?? .nan,
                           med(rs.compactMap(\.kpErr)), rs.compactMap(\.kpErr).max() ?? .nan,
                           rs.map(\.losses).reduce(0, +), rs.map(\.motion).reduce(0, +)))
        }
        for k in [CourtLockPlan.Kind.search, .refine] {
            let v = times[k] ?? []
            logLine(String(format: "[real-lock] per-attempt %@ at 960x540 (Mac M4 release, machine busy with release builds): n %d median %.0f ms p90 %.0f ms max %.0f ms",
                           k.rawValue, v.count, med(v) * 1000, pct(v, 0.9) * 1000, (v.max() ?? .nan) * 1000))
        }
        let mac = results[1.0] ?? []
        XCTAssertEqual(mac.filter(\.falseLock).count, 0, "false locks")
        XCTAssertGreaterThanOrEqual(mac.filter(\.locked).count, Int(0.9 * Double(clips.count)), "lock rate")
    }

    /// Cost of turning one 1080p capture frame into a working frame (the per-sample ingest cost).
    func testIngestCost() throws {
        _ = try realClips()
        let ds = FrameDownsampler()
        let cfg = CourtLockConfiguration()
        var out: [String] = []
        for (fmt, name) in [(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, "420v"), (kCVPixelFormatType_32BGRA, "BGRA")] {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, 1920, 1080, fmt, nil, &pb)
            let p = try XCTUnwrap(pb)
            let ws = cfg.workingSize(for: ImageSize(width: 1920, height: 1080))
            _ = try ds.frame(from: p, workingSize: ws, time: 0)
            var t: [Double] = []
            for _ in 0..<9 { t.append(try ms { _ = try ds.frame(from: p, workingSize: ws, time: 0) }) }
            out.append(String(format: "%@ 1920x1080→%dx%d min %.0f / median %.0f ms", name, ws.width, ws.height,
                              t.min()!, t.sorted()[4]))
        }
        let big = ImageU8(width: 1920, height: 1080, channels: 3)
        var fr: [ImageU8] = Array(repeating: ds.frame(from: big, workingSize: ImageSize(width: 1440, height: 810), time: 0).image, count: 8)
        fr[0].data[0] = 1
        let tm = (0..<5).map { _ in ms { _ = TemporalMedian.median(fr) } }.min()!
        let ta = (0..<5).map { _ in ms { _ = FrameDownsampler.areaResize(big, to: ImageSize(width: 1440, height: 810)) } }.min()!
        out.append(String(format: "median of 7 @1440x810 min %.0f ms; area resize alone min %.0f ms", tm, ta))
        logLine("[ingest-cost \(Self.stamp())] " + out.joined(separator: "; "))
    }

    static func stamp() -> String {
        let f = DateFormatter(); f.dateFormat = "HH:mm"; return f.string(from: Date())
    }
}
