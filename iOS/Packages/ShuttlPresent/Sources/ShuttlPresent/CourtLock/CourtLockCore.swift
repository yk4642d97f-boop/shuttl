import Foundation
import ShuttlCore

public struct CourtLockConfiguration: Sendable, Equatable {
    /// Minimum seconds between sampled frames (≤ 2 Hz).
    public var sampleInterval = 0.5
    /// While locked, re-check at most this often (saves battery; the phone is on a tripod).
    public var lockedAttemptInterval = 1.0
    /// Rolling buffer of downscaled frames (temporal median of up to 7 of them removes players).
    public var bufferSize = 8
    /// Working resolution = full × this. 0.75 was verified on the 29 real clips (decision 29/29,
    /// kp ≤ 0.34 px vs full res); 0.5 was not (kp up to 5 px) — see .agent-logs/ShuttlPresent.log.
    public var workingScale = 0.75
    /// Two solutions agree when their max court-keypoint disagreement is ≤ this (full-res px).
    public var agreePx = 3.0
    /// A locked camera is contradicted (phone moved) above this (full-res px).
    public var lostPx = 8.0
    /// Consecutive rejected attempts that lose a lock.
    public var lostAfterRejects = 3
    /// Consecutive rejected attempts that drop an unconfirmed candidate.
    public var candidateDropRejects = 2
    /// While locked: one full independent search every this many attempts (the rest are refines).
    public var fullSearchEvery = 4
    public var motion = MotionConfiguration()
    public init() {}

    /// Low-duty preset for watching a lock WHILE recording 1080p60: 0.5 Hz samples, a re-fit every
    /// 2 s, no full searches (≈ 10 % of one Mac M4 core measured; est. 25–35 % of one phone core).
    /// Preferred during recording: `pause()` the engine (the pipeline re-calibrates from the file).
    public static var recording: CourtLockConfiguration {
        var c = CourtLockConfiguration()
        c.sampleInterval = 2.0
        c.lockedAttemptInterval = 2.0
        c.fullSearchEvery = 1_000_000
        return c
    }

    /// Working size for a full upright frame size (exact integer divisors kept exact).
    public func workingSize(for full: ImageSize) -> ImageSize {
        guard workingScale > 0, workingScale < 1 else { return full }
        let L = max(full.width, full.height)
        return FrameDownsampler.workingSize(for: full, longSide: Int((Double(L) * workingScale).rounded()))
    }
}

/// One calibration attempt the core wants run.
public struct CourtLockPlan: Sendable {
    public enum Kind: String, Sendable { case search, refine }
    public var kind: Kind
    /// Frames to calibrate on (working resolution); `image` builds the temporal median.
    public var frames: [ImageU8]
    /// Seed camera at the working resolution (refine only).
    public var seed: CameraModel?
    public var fullSize: ImageSize
    public var generation: Int
    /// Time of the newest frame in the plan.
    public var time: Double

    /// The image the calibrator sees: temporal median of ≥ 3 frames, else the newest frame.
    public func image() -> ImageU8 {
        frames.count >= 3 ? TemporalMedian.median(frames) : frames[frames.count - 1]
    }

    public func run(_ calibrator: CourtLockCalibrator) -> CalibrationAttempt {
        let img = image()
        if kind == .refine, let seed { return calibrator.refine(seed, on: img) }
        return calibrator.search(img)
    }
}

/// Deterministic lock state machine (no threads, no clocks): `offer` sampled frames, ask for the
/// next `plan`, run it (anywhere), `apply` the result. `CourtLockEngine` drives it on queues;
/// tests drive it directly.
public struct CourtLockCore: Sendable {
    public let config: CourtLockConfiguration
    public private(set) var state: CourtLockState = .searching(.frameWholeCourt)
    /// Bumped on camera motion: plans from an older generation are discarded.
    public private(set) var generation = 0
    public private(set) var buffer: [WorkingFrame] = []
    public private(set) var stats = CourtLockStats()

    private var pending: CameraModel?            // unconfirmed solution (candidate / after a lost)
    private var locked: CameraModel?
    private var lockedConfidence = 0.0
    private var drift: CameraModel?              // agrees with itself but not with the lock
    private var driftCount = 0
    private var rejects = 0
    private var lastRejectWasSearch = false
    private var searchUsesLatest = false
    private var attemptsSinceSearch = 0
    private var streakEvidence: CalibrationEvidence?
    private var motionSuspect = false
    private var lastPlannedTime = -Double.infinity
    private var lastAttemptTime = -Double.infinity
    private var lastSampleTime = -Double.infinity

    public init(config: CourtLockConfiguration = CourtLockConfiguration()) { self.config = config }

    /// Should a frame at `time` be sampled (≤ 2 Hz)? A clock that went backwards restarts sampling.
    public func wantsSample(at time: Double) -> Bool {
        time < lastSampleTime || time - lastSampleTime >= config.sampleInterval - 1e-6
    }

    /// `wantsSample` + reserve the slot (the engine converts the frame asynchronously).
    public mutating func reserveSample(at time: Double) -> Bool {
        guard wantsSample(at: time) else { return false }
        lastSampleTime = time
        return true
    }

    public var lockedCamera: CameraModel? { locked }

    // MARK: Frames

    /// Adds a sampled frame; returns true when the published state changed (motion).
    @discardableResult
    public mutating func offer(_ f: WorkingFrame) -> Bool {
        if let prev = buffer.last, prev.fullSize != f.fullSize || prev.workingSize != f.workingSize {
            reset(keepStats: true)                              // camera switched / format changed
        }
        if let prev = buffer.last, f.time < prev.time { reset(keepStats: true) }   // new clock
        lastSampleTime = max(lastSampleTime, f.time)
        stats.samples += 1
        var changed = false
        if let prev = buffer.last {
            let m = MotionDetector.compare(prev.thumb, f.thumb, fullWidth: f.fullSize.width, config: config.motion)
            if m.moved { changed = onMotion(m) }
        }
        if motionSuspect { buffer.removeAll() }             // only post-motion frames until re-checked
        buffer.append(f)
        if buffer.count > config.bufferSize { buffer.removeFirst(buffer.count - config.bufferSize) }
        return changed
    }

    private mutating func onMotion(_ m: MotionResult) -> Bool {
        stats.motionEvents += 1
        generation += 1
        motionSuspect = true
        buffer.removeAll()
        let old = state
        switch state {
        case .locked:
            if m.shiftPx > config.lostPx {
                locked = nil; drift = nil; driftCount = 0; pending = nil
                state = .lost(.phoneMoved)
                stats.losses += 1
            }
        case .candidate:
            pending = nil
            state = .searching(.holdStill)
        case .searching:
            state = .searching(.holdStill)
        case .lost:
            pending = nil
            state = .lost(.holdStill)
        }
        return state != old
    }

    /// Back to `.searching`, buffer emptied; in-flight plans are discarded (generation bump).
    public mutating func reset(keepStats: Bool = false) {
        let s = stats, g = generation
        self = CourtLockCore(config: config)
        generation = g + 1
        if keepStats { stats = s }
    }

    // MARK: Planning

    /// The next attempt to run, or nil (nothing new to look at / waiting for the locked cadence).
    ///  • no solution yet → full search (alternating median / newest frame after rejects);
    ///  • candidate → refine it on newer frames (the confirming second solution);
    ///  • locked → refine every `lockedAttemptInterval`, a full search every `fullSearchEvery`;
    ///  • after a rejected refine → full search (fresh evidence for the hint), then alternate;
    ///  • after camera motion while locked → refine at once.
    public mutating func plan() -> CourtLockPlan? {
        guard let newest = buffer.last, newest.time > lastPlannedTime else { return nil }
        if locked != nil && !motionSuspect && rejects == 0
            && newest.time - lastAttemptTime < config.lockedAttemptInterval - 1e-6 {
            return nil
        }
        let seedFull = locked ?? pending
        var kind: CourtLockPlan.Kind
        if seedFull == nil { kind = .search }
        else if motionSuspect && locked != nil { kind = .refine }
        else if rejects > 0 { kind = lastRejectWasSearch ? .refine : .search }
        else if locked != nil && attemptsSinceSearch + 1 >= config.fullSearchEvery { kind = .search }
        else { kind = .refine }
        let seed = kind == .refine ? seedFull.flatMap { $0.rescaled(to: newest.workingSize) } : nil
        if kind == .refine && seed == nil { kind = .search }
        let frames = kind == .search && searchUsesLatest ? [newest.image] : buffer.map(\.image)
        lastPlannedTime = newest.time
        lastAttemptTime = newest.time
        return CourtLockPlan(kind: kind, frames: frames, seed: seed, fullSize: newest.fullSize,
                             generation: generation, time: newest.time)
    }

    // MARK: Results

    /// Folds an attempt result in; returns true when the published state changed.
    @discardableResult
    public mutating func apply(_ plan: CourtLockPlan, _ attempt: CalibrationAttempt) -> Bool {
        guard plan.generation == generation else { stats.discarded += 1; return false }
        stats.attempts += 1
        if plan.kind == .search { stats.searches += 1; attemptsSinceSearch = 0 } else { stats.refines += 1; attemptsSinceSearch += 1 }
        let old = state
        let cam = attempt.camera.flatMap { $0.rescaled(to: plan.fullSize) }
        let wasMotion = motionSuspect
        motionSuspect = false
        if let cam { accept(cam, plan: plan, afterMotion: wasMotion) } else { reject(plan, attempt, afterMotion: wasMotion) }
        if state.isLocked && !old.isLocked && stats.firstLockTime == nil { stats.firstLockTime = plan.time }
        return state != old
    }

    private func disagreement(_ a: CameraModel, _ b: CameraModel) -> Double { CameraAgreement.maxDisagreementPx(a, b) }

    private mutating func accept(_ cam: CameraModel, plan: CourtLockPlan, afterMotion: Bool) {
        rejects = 0; lastRejectWasSearch = false; searchUsesLatest = false; streakEvidence = nil
        if let lk = locked {
            let d = disagreement(lk, cam)
            if d <= config.agreePx {
                drift = nil; driftCount = 0
                return
            }
            if d <= config.lostPx && !afterMotion {
                // Between agree and lost: the phone settled a little. Re-lock once it is stable.
                if let dr = drift, disagreement(dr, cam) <= config.agreePx { driftCount += 1 } else { driftCount = 1 }
                drift = cam
                if driftCount >= 2 { setLocked(cam, confidence: max(cam.confidence, lockedConfidence)) }
                return
            }
            if d <= config.lostPx {
                setLocked(cam, confidence: max(cam.confidence, lockedConfidence))   // motion + a good re-fit
                return
            }
            locked = nil; drift = nil; driftCount = 0
            pending = cam
            state = .lost(.phoneMoved)
            stats.losses += 1
            return
        }
        if let p = pending, disagreement(p, cam) <= config.agreePx {
            setLocked(cam, confidence: cam.confidence > 0 ? cam.confidence : p.confidence)
            return
        }
        pending = cam
        state = .candidate(CourtProjection(camera: cam))
    }

    private mutating func setLocked(_ cam: CameraModel, confidence: Double) {
        var c = cam
        c.confidence = min(max(confidence, 0), 1)
        locked = c; lockedConfidence = c.confidence
        pending = nil; drift = nil; driftCount = 0
        state = .locked(camera: c, projection: CourtProjection(camera: c), confidence: c.confidence)
    }

    private mutating func reject(_ plan: CourtLockPlan, _ attempt: CalibrationAttempt, afterMotion: Bool) {
        rejects += 1
        stats.rejects += 1
        lastRejectWasSearch = plan.kind == .search
        if plan.kind == .search {
            streakEvidence = attempt.evidence
            searchUsesLatest.toggle()
        }
        let hint = rejectHint()
        switch state {
        case .locked:
            if afterMotion || rejects >= config.lostAfterRejects {
                locked = nil; drift = nil; driftCount = 0; pending = nil
                state = .lost(afterMotion ? .phoneMoved : hint)
                stats.losses += 1
            }
        case .candidate:
            if rejects >= config.candidateDropRejects {
                pending = nil
                state = .searching(hint)
            }
        case .searching:
            state = .searching(hint)
        case .lost(let h):
            if pending != nil && rejects >= config.candidateDropRejects { pending = nil }
            state = .lost(h == .phoneMoved && rejects < config.lostAfterRejects ? .phoneMoved : hint)
        }
    }

    private func rejectHint() -> CourtLockHint {
        guard let e = streakEvidence else { return .frameWholeCourt }
        return CourtLockHints.hint(for: e, luma: buffer.last?.luma)
    }
}

/// Counters for logs / tests.
public struct CourtLockStats: Sendable, Equatable {
    public var samples = 0
    public var attempts = 0
    public var searches = 0
    public var refines = 0
    public var rejects = 0
    public var discarded = 0
    public var losses = 0
    public var motionEvents = 0
    /// Frame time of the first lock.
    public var firstLockTime: Double?
}
