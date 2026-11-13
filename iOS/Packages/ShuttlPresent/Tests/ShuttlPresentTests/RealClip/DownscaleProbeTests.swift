import XCTest
import simd
import ShuttlCore
import ShuttlCourt
import ShuttlPresent

/// Does calibrating on downscaled working frames give the same accept decision and (rescaled)
/// camera as full resolution? 7 frames at 2 Hz from the start of each of the 29 real clips —
/// what the live CourtLock buffer holds. Gated: SHUTTL_PRESENT_REAL=1.
final class DownscaleProbeTests: XCTestCase {
    func testDownscaleAgreement() async throws {
        let clips = try realClips()
        let longSides = (ProcessInfo.processInfo.environment["SHUTTL_PRESENT_SIDES"] ?? "960,640,480")
            .split(separator: ",").compactMap { Int($0) }
        var agree = [Int: Int](), within1 = [Int: Int](), kpMaxes = [Int: [Double]](), naiveMaxes = [Int: [Double]]()
        var times = [Int: [Double]](), fullTimes: [Double] = [], fullAccepted = 0
        for url in clips {
            let (meta, dec) = try await decodeFrames(url, step: max(1, Int((meta0(url) ?? 30) / 2)), count: 7)
            let full = ImageSize(width: meta.width, height: meta.height)
            let imgs = dec.map(\.1)
            var fullRes: AutoCalibrationResult!
            let tFull = ms { fullRes = CourtCalibration.calibrateFromFramesDetailed(imgs, size: full) }
            fullTimes.append(tFull)
            if fullRes.camera != nil { fullAccepted += 1 }
            var line = "PROBE \(clipTag(url)) full \(fullRes.camera == nil ? "reject" : "accept") \(Int(tFull))ms"
            for L in longSides {
                let ds = FrameDownsampler(workingLongSide: L)
                let wf = imgs.enumerated().map { ds.frame(from: $0.element, time: Double($0.offset) * 0.5) }
                let ws = wf[0].workingSize
                var r: AutoCalibrationResult!
                let t = ms { r = CourtCalibration.calibrateFromFramesDetailed(wf.map(\.image), size: ws) }
                times[L, default: []].append(t)
                let same = (r.camera == nil) == (fullRes.camera == nil)
                if same { agree[L, default: 0] += 1 }
                line += " | \(L) \(ws.width)x\(ws.height) \(r.camera == nil ? "reject" : "accept") \(Int(t))ms"
                if let c = r.camera, let f = fullRes.camera {
                    let up = CameraModel.rescale(c, to: full, pixelCentre: true)!
                    let naive = CameraModel.rescale(c, to: full, pixelCentre: false)!
                    let a = CameraAgreement.disagreementPx(f, up), b = CameraAgreement.disagreementPx(f, naive)
                    kpMaxes[L, default: []].append(a.max); naiveMaxes[L, default: []].append(b.max)
                    if a.max <= 1 { within1[L, default: 0] += 1 }
                    line += String(format: " kp max %.2f med %.2f (naive max %.2f) conf %.2f/%.2f", a.max, a.median, b.max,
                                   c.confidence, f.confidence)
                } else if !same {
                    let s = r.diagnostics.stats
                    line += " nT \(s?.nT ?? -1) nL \(s?.nL ?? -1) absent \(s?.nAbsent ?? -1)"
                        + String(format: " expl %.2f margin %.2f", s?.explainedFrac ?? .nan, r.diagnostics.margin ?? .nan)
                }
            }
            print(line)
        }
        func med(_ v: [Double]) -> Double { v.isEmpty ? .nan : v.sorted()[v.count / 2] }
        logLine(String(format: "[probe] full-res 7x2Hz frames: accepted %d/%d, calibrate median %.0f ms max %.0f ms (Mac M4 release)",
                       fullAccepted, clips.count, med(fullTimes), fullTimes.max() ?? .nan))
        for L in longSides {
            logLine(String(format: "[probe] long side %d: decision agrees %d/%d, both-accepted kp max <= 1 px %d/%d (max-of-max %.2f, median %.2f; naive-centre median %.2f), calibrate median %.0f ms max %.0f ms",
                           L, agree[L] ?? 0, clips.count, within1[L] ?? 0, kpMaxes[L]?.count ?? 0,
                           kpMaxes[L]?.max() ?? .nan, med(kpMaxes[L] ?? []), med(naiveMaxes[L] ?? []),
                           med(times[L] ?? []), times[L]?.max() ?? .nan))
        }
    }

    private func meta0(_ url: URL) -> Double? { 30 }
}
