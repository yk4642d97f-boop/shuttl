/// Automatic calibration of a (median) image and the public calibrateFromFrames.
/// Port of court_calibration._same_solution, _kp_image, _calibrate_image,
/// calibrate_from_frames.
import Foundation
import simd
import ShuttlCore

/// Why the automatic path returned no camera — Python's `diagnostics["reason"]`
/// (court_calibration.REASONS; `rawValue` is the identical string). Gates are checked in
/// declaration order and the first failing one is reported.
public enum CalibrationFailure: String, Sendable, Equatable, CaseIterable {
    case ok = "ok"
    case noFrames = "no_frames"
    /// fewer than two vanishing-point line families
    case noLineFamilies = "no_line_families"
    case noHypotheses = "no_hypotheses"
    case noCandidates = "no_candidates"
    /// no candidate passes the line-evidence gate (`_valid`)
    case evidenceGate = "evidence_gate"
    /// the winning camera's focal is more than `focalHintTolerance` off `focalHintPx`
    case focalMismatch = "focal_mismatch"
    case lowConfidence = "low_confidence"
    /// a distinct valid rival within `minMargin`
    case ambiguous = "ambiguous"
    case lowCoverage = "low_coverage"
    case implausibleFocal = "implausible_focal"
    case highRmse = "high_rmse"
    /// the median image failed; single frames gave a camera, but no two DIFFERENT frames
    /// agreed on it within `frameAgreePx` (3 px, CourtLock's definition)
    case inconsistentFrames = "inconsistent_frames"
    /// a colour model accepted, a better-fitting paint model gave no camera
    case paintMismatch = "paint_mismatch"
}

/// What the automatic path saw (Python's `diagnostics` dict).
public struct CourtCalibrationDiagnostics: Sendable, Equatable {
    /// "median" or "frame<i>": the successful attempt, else the median attempt.
    public var source = "median"
    /// `diagnostics["reason"]` of that attempt.
    public var reason: CalibrationFailure = .noFrames
    public var nHyp = 0, nLines = 0, nValid = 0
    /// Stats of the best-explained final candidate, replaced by the chosen camera's.
    public var stats: FitStats?
    public var margin: Double?, confidence: Double?, score: Double?, score2: Double?
    /// Single-frame fallback (Python `frames_agree`, `frame_agree_px`, `n_frame_accepts`,
    /// `frames_tried`, `dominated_by`); nil / empty when the median image succeeded.
    public var framesAgree: [Int] = []
    public var frameAgreePx: Double?
    public var nFrameAccepts: Int?
    public var framesTried: [Int] = []
    /// explained length of a distinct candidate that beat the winner but failed the gate
    public var dominatedBy: Double?
    /// Python `far_m_per_px` (accepted cameras): floor metres per image pixel at the
    /// far-baseline centre, worst direction — a far keypoint error of e px is up to
    /// e·farMPerPx metres on the floor. ~0.16 at 1.6 m × 5 m, ~0.09 at 2.5 m × 4.5 m,
    /// ~0.31 at 0.8 m × 5 m (70° HFOV, 1080p).
    public var farMPerPx: Double?
    /// Python `line_colour`: the paint colour model that found an accepted camera when it
    /// is not white (nil for white lines and for rejects).
    public var lineColour: LineColour?
    /// Python `paint_fit` (accepted cameras, colour images): per paint model, the median
    /// line-minus-floor score at the camera's line samples (finite values only).
    public var paintFit: [LineColour: Double] = [:]
    public init() {}
}

/// Automatic result: camera (nil = not confident) plus diagnostics.
public struct AutoCalibrationResult: Sendable {
    public var camera: CameraModel?
    public var diagnostics: CourtCalibrationDiagnostics
}

/// One refined candidate camera: (explained length, camera, stats).
struct AutoCandidate {
    var explained: Double
    var camera: CameraModel
    var stats: FitStats
}

enum AutoCalibration {
    static func m9(_ m: simd_double3x3) -> M9 {
        (mget(m, 0, 0), mget(m, 0, 1), mget(m, 0, 2), mget(m, 1, 0), mget(m, 1, 1), mget(m, 1, 2),
         mget(m, 2, 0), mget(m, 2, 1), mget(m, 2, 2))
    }

    static func simd(_ h: M9) -> simd_double3x3 {
        rowMajor3(SIMD3(h.0, h.1, h.2), SIMD3(h.3, h.4, h.5), SIMD3(h.6, h.7, h.8))
    }

    /// `cam.floor_homography()` with numpy's arithmetic (dgemm FMA chain, / H[2,2]).
    static func floorH(_ cam: CameraModel) -> M9 {
        let R = cam.R, t = cam.t
        let B: M9 = (mget(R, 0, 0), mget(R, 0, 1), t.x, mget(R, 1, 0), mget(R, 1, 1), t.y,
                     mget(R, 2, 0), mget(R, 2, 1), t.z)
        let H = HypothesisScore.matmul(m9(cam.K), B)
        let s = H.8
        return (H.0 / s, H.1 / s, H.2 / s, H.3 / s, H.4 / s, H.5 / s, H.6 / s, H.7 / s, H.8 / s)
    }

    /// `_kp_image`: the 28 court keypoints through H, NaN where unusable.
    static func kpImage(_ H: M9, _ feat: Features) -> [SIMD2<Double>] {
        let m = 0.5 * feat.D
        return courtKeypointsXY.map { p in
            let u = fma(H.2, 1, fma(H.1, p.y, H.0 * p.x))
            let v = fma(H.5, 1, fma(H.4, p.y, H.3 * p.x))
            let w = fma(H.8, 1, fma(H.7, p.y, H.6 * p.x))
            let uv = SIMD2(u / w, v / w)
            let bad = w <= 0 || !uv.x.isFinite || !uv.y.isFinite || uv.x < -m || uv.y < -m
                || uv.x > Double(feat.w) + m || uv.y > Double(feat.h) + m
            return bad ? SIMD2(Double.nan, Double.nan) : uv
        }
    }

    /// `_KP_180`: keypoint k of the 180°-rotated court ((x, y) → (−x, −y)).
    static let kp180: [Int] = courtKeypointsXY.map { p in
        var best = 0, bd = Double.infinity
        for (j, q) in courtKeypointsXY.enumerated() {
            let e = q + p, d = (e.x * e.x + e.y * e.y).squareRoot()
            if d < bd { bd = d; best = j }           // np.argmin: first minimum
        }
        return best
    }

    /// `_same_solution`: the SAME keypoints (or the 180° twin's) coincide — median distance
    /// < tol over >= 4 keypoints usable in both (a set test let a court shifted by one line
    /// spacing pass as "the same camera").
    static func sameSolution(_ ua: [SIMD2<Double>], _ ub: [SIMD2<Double>], tol: Double) -> Bool {
        var best = Double.infinity
        for vb in [ub, kp180.map { ub[$0] }] {
            var d: [Double] = []
            for (p, q) in zip(ua, vb) where p.x.isFinite && p.y.isFinite && q.x.isFinite && q.y.isFinite {
                let e = p - q
                d.append((e.x * e.x + e.y * e.y).squareRoot())
            }
            if d.count >= 4 { best = min(best, Num.median(d)) }
        }
        return best < tol
    }

    /// `_choose_hypotheses`: top 2000 by the coarse score → the first `nChosen` distinct by
    /// the `_COARSE` line score, then up to `nChosenSharp` more by the same score at ~1 px.
    static let nChosen = 16, nChosenSharp = 8
    static func choose(_ feat: Features, _ hs: HypothesisSet)
        -> (H: [M9], ref: [SIMD2<Double>], ls: [Float], chosen: [Int]) {
        let top = NumpyCompat.argsortStable(hs.score.map { -$0 }).prefix(2000)
        let H = top.map { hs.H[$0] }, ref = top.map { hs.ref[$0] }
        let ls = HypothesisScore.lineScore(feat, H, .coarse)
        let lsSharp = HypothesisScore.lineScore(feat, H, .coarse, tol: max(1.0, 0.0005 * feat.D))
        let tol = max(3.0, 0.004 * feat.D)
        var chosen: [Int] = [], kps: [[SIMD2<Double>]] = []
        func pick(_ score: [Float], _ limit: Int) {
            for i in NumpyCompat.argsortStable(score.map { -$0 }).prefix(300) {
                if score[i] <= 0 || chosen.count >= limit { break }
                let u = kpImage(H[i], feat)
                if !chosen.contains(i) && kps.allSatisfy({ !sameSolution(u, $0, tol: tol) }) {
                    chosen.append(i); kps.append(u)
                }
            }
        }
        pick(ls, nChosen)
        pick(lsSharp, nChosen + nChosenSharp)
        return (H, ref, ls, chosen)
    }
}
extension AutoCalibration {
    /// `_quick_candidates`: quick refinement of each chosen hypothesis (2 iterations, _MEDIUM,
    /// no k1) from the wide (0.5 % D) and the check (0.3 % D) window; keeps the one explaining more.
    static func candidates(_ mt: Matcher, size: ImageSize, H: [M9], ref: [SIMD2<Double>],
                           chosen: [Int]) -> [AutoCandidate] {
        let D = mt.feat.D
        let sChk = max(3.0, 0.003 * D)
        var out: [AutoCandidate] = []
        for i in chosen {
            guard let cam = Homography.camera(simd(H[i]), size: size, refXY: [ref[i]]) else { continue }
            var best: AutoCandidate?
            for search in [max(4.0, 0.005 * D), sChk] {
                guard let r = mt.refine(cam, search0: search, fitK1: .off, nIter: 2, samples: .medium)
                else { continue }
                let c = r.0
                let st = mt.fitStats(camParams(c, withK1: false), cx: mget(c.K, 0, 2), cy: mget(c.K, 1, 2),
                                     .fine, search: sChk)
                let e = st.explained ?? 0
                if best == nil || e > best!.explained { best = AutoCandidate(explained: e, camera: c, stats: st) }
            }
            if let best { out.append(best) }
        }
        return out
    }

    /// `_focal_ok`: within `focalHintTolerance` of the hint (always true without one).
    static func focalOK(_ cam: CameraModel, _ hint: Double?, _ cfg: CourtCalibrationConfig) -> Bool {
        guard let hint else { return true }
        return abs(mget(cam.K, 0, 0) / hint - 1.0) <= cfg.focalHintTolerance
    }

    /// Full refinement of the contenders, evidence gate, margin over distinct rivals.
    static func decide(_ mt: Matcher, _ cands0: [AutoCandidate], focalHint: Double? = nil,
                       vetoDominated: Bool = true, diag: inout CourtCalibrationDiagnostics) -> CameraModel? {
        let feat = mt.feat, D = feat.D, cfg = mt.config
        let sChk = max(3.0, 0.003 * D), tol = max(3.0, 0.004 * D)
        let cands = NumpyCompat.argsortStable(cands0.map { -$0.explained }).map { cands0[$0] }
        var finals: [AutoCandidate] = [], others: [AutoCandidate] = []
        for (k, c0) in cands.enumerated() {
            if k >= 3 && (c0.explained < 0.7 * cands[0].explained || finals.count >= 6) {
                others.append(c0); continue
            }
            guard let r = mt.refine(c0.camera, search0: sChk, fitK1: .auto, nIter: 4) else { continue }
            finals.append(AutoCandidate(explained: r.1.explained ?? 0, camera: r.0, stats: r.1))
        }
        // valid cameras ranked by NET evidence (`_net`: explained − unexplained visible line length)
        func net(_ st: FitStats) -> Double {
            let e = st.explained ?? 0
            return 2.0 * e - e / max(st.explainedFrac ?? 0, 1e-9)
        }
        let v0 = finals.filter { mt.valid($0.stats) }             // sorted(..., key=-net): stable
        let valid = NumpyCompat.argsortStable(v0.map { -net($0.stats) }).map { v0[$0] }
        diag.nValid = valid.count
        if let best = finals.max(by: { $0.explained < $1.explained }) { diag.stats = best.stats }
        guard let win = valid.first else { diag.reason = .evidenceGate; return nil }
        let s1 = net(win.stats), stats = win.stats
        var cam = win.camera
        let kb = kpImage(floorH(cam), feat)
        var s2 = 0.0
        for x in Array(valid.dropFirst()) + others.filter({ mt.valid($0.stats) })
        where !sameSolution(kb, kpImage(floorH(x.camera), feat), tol: tol) {
            s2 = max(s2, net(x.stats))
        }
        let margin = s1 <= 0 ? 1.0 : min(max(1 - s2 / s1, 0), 1)
        let conf = mt.confidence(stats, margin: margin)
        let f = mget(cam.K, 0, 0)
        diag.stats = stats; diag.margin = margin; diag.confidence = conf; diag.score = s1; diag.score2 = s2
        // Dominated winner → ambiguous (median and single frames): a DISTINCT final candidate
        // explaining more that failed the evidence gate (the right camera: sweep case 3 frames
        // 2-4, sweep2 low tripods #384/#421/#508/#576) — only if it has no visible model line on
        // bare floor (n_absent > 0 is evidence against it: zoom clip).
        var dominated = 0.0
        let e1 = win.explained
        if vetoDominated {
            for x in finals where x.explained > e1 && !mt.valid(x.stats) && x.stats.nAbsent == 0
                && !sameSolution(kb, kpImage(floorH(x.camera), feat), tol: tol) {
                dominated = max(dominated, x.explained)
            }
            diag.dominatedBy = dominated
        }
        // Python checks in this order and reports the first failing gate. The focal prior
        // is a veto on the winner, never a selector (a wrong hint can only reject).
        let gates: [(Bool, CalibrationFailure)] = [
            (!focalOK(cam, focalHint, cfg), .focalMismatch), (dominated > 0, .ambiguous),
            (conf < cfg.minConfidence, .lowConfidence), (margin < cfg.minMargin, .ambiguous),
            (stats.coverage < 0.4, .lowCoverage), (!(0.25 * D < f && f < 5 * D), .implausibleFocal),
            (stats.rmse > max(1.0, 0.0007 * D), .highRmse)]
        if let failed = gates.first(where: { $0.0 }) { diag.reason = failed.1; return nil }
        if cam.center.x > 0 { cam = flip180(cam) }
        cam.confidence = conf
        diag.reason = .ok
        diag.farMPerPx = AutoCalibration.farMPerPx(cam)
        return cam
    }

    /// `_calibrate_image`: automatic calibration of one (median) image; nil when not confident.
    static func calibrateImage(_ img: ImageU8, size: ImageSize, config: CourtCalibrationConfig,
                               focalHint: Double? = nil, vetoDominated: Bool = true, colour: LineColour = .white,
                               diag: inout CourtCalibrationDiagnostics) -> CameraModel? {
        let feat = Features(img, config: config, colour: colour)
        let lines = LineExtraction.merge(LineExtraction.segments(feat, maxSegments: config.maxSegments), D: feat.D)
        let fams = VanishingPoints.families(lines).filter { $0.members.count >= 2 }
        guard fams.count >= 2 else { diag.reason = .noLineFamilies; return nil }
        let (hs, _) = Hypotheses.all(feat, lines: lines, families: fams)
        guard !hs.H.isEmpty else { diag.reason = .noHypotheses; return nil }
        let (H, ref, _, chosen) = choose(feat, hs)
        let mt = Matcher(feat: feat, config: config)
        let cands = candidates(mt, size: size, H: H, ref: ref, chosen: chosen)
        guard !cands.isEmpty else { diag.reason = .noCandidates; return nil }
        diag.nHyp = hs.H.count; diag.nLines = lines.count
        return decide(mt, cands, focalHint: focalHint, vetoDominated: vetoDominated, diag: &diag)
    }
}
extension CourtCalibration {
    /// Fully automatic calibration from frames sampled across the video (`calibrate_from_frames`).
    ///
    /// Temporal median (removes moving players) → line mask → segments → vanishing-point
    /// families → line-quad × model-rectangle homographies (ordering, chirality and
    /// camera-plausibility constraints) → model/mask agreement → dense refinement.
    /// Returns nil when not confident (never a guess). A camera is accepted only with
    /// >= 3 transverse and >= 3 longitudinal lines matched, no clearly visible model line
    /// on bare floor, >= 55 % of visible line length explained, a >= 5 % margin over
    /// distinct valid rivals and rmse <= max(1, 0.0007·D). With exactly 2 transverse lines
    /// (low / close tripods) the camera also needs all 5 longitudinal lines, the two
    /// transverse lines on opposite sides of the net and coverage >= 0.70. If the median
    /// image fails, the middle frame is tried. `frames`: RGB (or BGR) ImageU8 at the upright
    /// `size`, e.g. the 24 frames video_io.sample_frames picks. `doublesHint` is accepted for
    /// API compatibility only: singles and doubles lines are always both painted.
    /// `focalHintPx`: optional focal length in pixels at `size` (device intrinsics / EXIF);
    /// the winning camera is rejected (`.focalMismatch`) when more than
    /// `config.focalHintTolerance` (15 %) off it. It never selects a different camera.
    public static func calibrateFromFrames(_ frames: [ImageU8], size: ImageSize, doublesHint: Bool? = nil,
                                           focalHintPx: Double? = nil,
                                           config: CourtCalibrationConfig = .default) -> CameraModel? {
        calibrateFromFramesDetailed(frames, size: size, doublesHint: doublesHint, focalHintPx: focalHintPx,
                                    config: config).camera
    }

    /// `calibrateFromFrames` plus the diagnostics of the attempt that succeeded (else the
    /// median's); `diagnostics.reason.rawValue` equals Python's `diagnostics["reason"]`.
    public static func calibrateFromFramesDetailed(_ frames: [ImageU8], size: ImageSize, doublesHint: Bool? = nil,
                                                   focalHintPx: Double? = nil,
                                                   config: CourtCalibrationConfig = .default) -> AutoCalibrationResult {
        guard !frames.isEmpty,
              let med = FrameOps.toImage(frames, size: size, maxFrames: config.medianMaxFrames) else {
            return AutoCalibrationResult(camera: nil, diagnostics: CourtCalibrationDiagnostics())
        }
        // line paint colour models in order (white first: white-line courts behave as
        // before); the next is tried only when every earlier one gave no camera. RGB only.
        var first: AutoCalibrationResult?
        let colours = med.channels == 3 ? LineColour.allCases : [.white]
        // Each colour attempt is a pure function of (frames, colour): when white gives no camera
        // (and no focal veto), the others are computed CONCURRENTLY and then walked in order —
        // identical decisions, reject wall time ~1/3 on a multi-core phone.
        var cache: [LineColour: AutoCalibrationResult] = [:]
        func attempt(_ c: LineColour) -> AutoCalibrationResult {
            if let r = cache[c] { return r }
            let r = calibrateColour(frames, med: med, size: size, focalHintPx: focalHintPx, config: config, colour: c)
            cache[c] = r
            return r
        }
        let w = attempt(.white)
        if w.camera == nil, w.diagnostics.reason != .focalMismatch, colours.count > 1 {
            let rest = Array(colours.dropFirst())
            let out = ConcurrentSlots<AutoCalibrationResult>(rest.count)
            DispatchQueue.concurrentPerform(iterations: rest.count) { i in
                out[i] = calibrateColour(frames, med: med, size: size, focalHintPx: focalHintPx, config: config, colour: rest[i])
            }
            for (i, c) in rest.enumerated() { cache[c] = out[i] }
        }
        for colour in colours {
            var r = attempt(colour)
            if first == nil { first = r }
            if let cam = r.camera {
                if colour != .white { r.diagnostics.lineColour = colour }
                if med.channels == 3 {
                    switch checkPaint(frames, med: med, size: size, focalHintPx: focalHintPx, config: config,
                                      colour: colour, cam: cam, r: r, cached: cache) {
                    case .keep(let k): return k
                    case .mismatch(let m): return m
                    case .whiteStands: return first!
                    }
                }
                return r
            }
            // a focal veto ends the search: no other attempt may win by agreeing with the hint
            if r.diagnostics.reason == .focalMismatch { break }
        }
        return first!                                            // rejected: the white attempt
    }

    enum PaintCheck { case keep(AutoCalibrationResult), mismatch(AutoCalibrationResult), whiteStands }

    /// `_check_paint`: the accepted camera's paint model must be the one fitting its lines;
    /// a model beating it by > minContrast is calibrated instead (white: the rejection stands).
    static func checkPaint(_ frames: [ImageU8], med: ImageU8, size: ImageSize, focalHintPx: Double?,
                           config: CourtCalibrationConfig, colour: LineColour, cam: CameraModel,
                           r: AutoCalibrationResult, cached: [LineColour: AutoCalibrationResult] = [:]) -> PaintCheck {
        let fit = AutoCalibration.paintFit(med, cam)
        var best = LineColour.allCases[0]
        for c in LineColour.allCases where fit[c]! > fit[best]! { best = c }   // first maximum
        let fitJSON = fit.filter { $0.value.isFinite }
        var r = r
        r.diagnostics.paintFit = fitJSON
        guard best != colour, fit[best]! > fit[colour]! + Double(config.minContrast) else { return .keep(r) }
        if best == .white { return .whiteStands }
        var r2 = cached[best] ?? calibrateColour(frames, med: med, size: size, focalHintPx: focalHintPx, config: config, colour: best)
        r2.diagnostics.paintFit = fitJSON
        if r2.camera == nil {
            r2.diagnostics.reason = .paintMismatch
            r2.diagnostics.lineColour = best
            return .mismatch(r2)
        }
        r2.diagnostics.lineColour = best
        return .keep(r2)
    }

    /// `_calibrate_colour`: one line colour model — the median image, then the fallback.
    static func calibrateColour(_ frames: [ImageU8], med: ImageU8, size: ImageSize, focalHintPx: Double?,
                                config: CourtCalibrationConfig, colour: LineColour) -> AutoCalibrationResult {
        var d = CourtCalibrationDiagnostics()
        let cam = AutoCalibration.calibrateImage(med, size: size, config: config, focalHint: focalHintPx,
                                                 colour: colour, diag: &d)
        // a focal veto ends the search: no single frame may win by agreeing with the hint
        guard cam == nil, frames.count > 2, d.reason != .focalMismatch else {
            return AutoCalibrationResult(camera: cam, diagnostics: d)
        }
        // the fallback frames are independent: calibrate them concurrently, then let the
        // (sequential, early-exit) fallback read the results — same decisions as trying them in turn
        let idx = AutoFallback.indices(frames.count)
        let slots = ConcurrentSlots<(CameraModel?, CourtCalibrationDiagnostics)>(idx.count)
        DispatchQueue.concurrentPerform(iterations: idx.count) { k in
            let i = idx[k]
            var fd = CourtCalibrationDiagnostics()
            fd.source = "frame\(i)"
            guard let im = FrameOps.toImage([frames[i]], size: size, maxFrames: config.medianMaxFrames) else {
                slots[k] = (nil, fd); return
            }
            let c = AutoCalibration.calibrateImage(im, size: size, config: config, focalHint: focalHintPx,
                                                   vetoDominated: true, colour: colour, diag: &fd)
            slots[k] = (c, fd)
        }
        return AutoFallback.finish(median: d, n: frames.count) { i in slots[idx.firstIndex(of: i)!] }
    }
}

/// Fixed-size result slots written by `DispatchQueue.concurrentPerform` workers (each index
/// exactly once), read after it returns.
final class ConcurrentSlots<T>: @unchecked Sendable {
    private var items: [T?]
    private let lock = NSLock()
    init(_ n: Int) { items = Array(repeating: nil, count: n) }
    subscript(i: Int) -> T {
        get { lock.lock(); defer { lock.unlock() }; return items[i]! }
        set { lock.lock(); items[i] = newValue; lock.unlock() }
    }
}

extension CourtCalibration {
}

extension AutoCalibration {
    /// `_paint_fit`: per paint model, median over visible FINE samples (projected line >= 1 px
    /// wide; sample and the floor points 0.25 m either side inside the image) of the model's
    /// score on the line minus the larger floor score (nearest pixel); -inf with < 20 samples.
    static func paintFit(_ img: ImageU8, _ cam: CameraModel) -> [LineColour: Double] {
        let S = LineSamples.fine
        let w = Double(img.width), h = Double(img.height), lw = CourtModel.lineWidth
        func inside(_ x: SIMD2<Double>) -> Bool {
            x.x.isFinite && x.y.isFinite && x.x >= 0 && x.y >= 0 && x.x <= w - 1 && x.y <= h - 1
        }
        var pix: [(Int, Int, Int)] = []                  // flat pixel index: line, side a, side b
        for (p, d) in zip(S.P, S.dir) {
            let n = SIMD3(-d.y, d.x, 0)
            let uv = cam.project(p), a = cam.project(p + 0.25 * n), b = cam.project(p - 0.25 * n)
            let w1 = cam.project(p + 0.5 * lw * n), w0 = cam.project(p - 0.5 * lw * n)
            let e = w1 - w0
            guard inside(uv), inside(a), inside(b), (e.x * e.x + e.y * e.y).squareRoot() >= 1.0 else { continue }
            func flat(_ x: SIMD2<Double>) -> Int {
                Int((x.y + 0.5).rounded(.down)) * img.width + Int((x.x + 0.5).rounded(.down))
            }
            pix.append((flat(uv), flat(a), flat(b)))
        }
        var out: [LineColour: Double] = [:]
        for c in LineColour.allCases {
            guard pix.count >= 20 else { out[c] = -.infinity; continue }
            let W = ImageOps.whiteness(img, colour: c).data
            out[c] = Num.median(pix.map { Double(min(W[$0.0] - W[$0.1], W[$0.0] - W[$0.2])) })
        }
        return out
    }
}
