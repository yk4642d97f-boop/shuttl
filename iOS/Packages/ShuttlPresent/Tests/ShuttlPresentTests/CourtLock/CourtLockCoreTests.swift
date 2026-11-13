import XCTest
import simd
import ShuttlCore
import ShuttlPresent

/// Lock state machine with scripted calibration results (no ShuttlCourt).
final class CourtLockCoreTests: XCTestCase {
    typealias F = LockFixtures

    /// Offer a frame at t, then run one plan with `attempt` (returns the plan kind, nil = no plan).
    @discardableResult
    private func step(_ core: inout CourtLockCore, _ t: Double, _ attempt: CalibrationAttempt,
                      sx: Double = 0, brightness: Double = 1) -> CourtLockPlan.Kind? {
        core.offer(F.frame(t, sx: sx, brightness: brightness))
        guard let p = core.plan() else { return nil }
        core.apply(p, attempt)
        return p.kind
    }

    func testSearchCandidateLock() {
        var core = CourtLockCore()
        XCTAssertEqual(core.state, .searching(.frameWholeCourt))
        XCTAssertEqual(step(&core, 0, F.accept(F.camera())), .search)
        guard case .candidate(let p) = core.state else { return XCTFail("\(core.state)") }
        XCTAssertEqual(p.corners.count, 4)
        XCTAssertEqual(step(&core, 0.5, F.accept(F.camera(dx: 1.5))), .refine, "confirm with a re-fit")
        guard case .locked(let cam, _, let conf) = core.state else { return XCTFail("\(core.state)") }
        XCTAssertEqual(conf, 0.9, accuracy: 1e-9)
        XCTAssertEqual(cam.imageSize, F.size)
        XCTAssertEqual(core.stats.firstLockTime, 0.5)
        XCTAssertEqual(core.state.title, "已锁定球场")
    }

    func testDisagreeingCandidateIsReplaced() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera(dx: 10)))          // 10 px > agree 3 px
        XCTAssertEqual(core.state.kindName, "candidate")
        XCTAssertEqual(core.state.projection?.corners["nb_ld"].map { Double($0.x) } ?? 0,
                       Double(F.camera(dx: 10).project([-6.7, -3.05, 0]).x) / 640, accuracy: 1e-9)
        step(&core, 1.0, F.accept(F.camera(dx: 11)))
        XCTAssertTrue(core.state.isLocked)
    }

    func testNoNewFrameNoPlan() {
        var core = CourtLockCore()
        core.offer(F.frame(0))
        XCTAssertNotNil(core.plan())
        XCTAssertNil(core.plan(), "same newest frame must not be planned twice")
    }

    func testRejectHints() {
        func hint(_ a: CalibrationAttempt, brightness: Double = 1) -> CourtLockHint? {
            var core = CourtLockCore()
            step(&core, 0, a, brightness: brightness)
            return core.state.hint
        }
        XCTAssertEqual(hint(F.reject(nT: 2, nL: 5)), .farBaselineMissing)
        XCTAssertEqual(hint(F.reject(nT: 5, nL: 2)), .sideLineMissing)
        XCTAssertEqual(hint(F.reject(nT: 1, nL: 2)), .tooFewLines)
        XCTAssertEqual(hint(F.reject(nT: nil, nL: nil, nHyp: 0)), .tooFewLines)
        XCTAssertEqual(hint(F.reject(absent: 1)), .playerOnLine)
        XCTAssertEqual(hint(F.reject(expl: 0.4)), .linesUnclear)
        XCTAssertEqual(hint(F.reject(nValid: 2, margin: 0.02)), .ambiguousHalf)
        XCTAssertEqual(hint(F.reject(nValid: 1, margin: 0.5)), .linesUnclear)
        XCTAssertEqual(hint(F.reject(nT: 2, nL: 5), brightness: 0.2), .tooDark)
        XCTAssertEqual(CourtLockHint.farBaselineMissing.message, "底线和发球线看不全：把手机抬高一点，或往后退一步")
        XCTAssertTrue(CourtLockHint.allCases.allSatisfy { !$0.message.isEmpty })
    }

    func testLockSurvivesTwoRejectsLostOnThird() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        XCTAssertTrue(core.state.isLocked)
        XCTAssertNil(core.offer(F.frame(0.8)) ? nil : core.plan(), "locked cadence: no attempt within 1 s")
        XCTAssertEqual(step(&core, 1.5, F.reject()), .refine)
        XCTAssertTrue(core.state.isLocked, "one reject keeps the lock")
        XCTAssertEqual(step(&core, 2.0, F.reject(nT: 2)), .search, "after a failed re-fit: full search")
        XCTAssertTrue(core.state.isLocked)
        XCTAssertEqual(step(&core, 2.5, F.reject(nT: 2)), .refine)
        XCTAssertEqual(core.state, .lost(.farBaselineMissing))
        XCTAssertEqual(core.stats.losses, 1)
        // Back: a search finds it, the next re-fit confirms.
        XCTAssertEqual(step(&core, 3.0, F.accept(F.camera(dx: 0.5))), .search)
        XCTAssertEqual(core.state.kindName, "candidate")
        step(&core, 3.5, F.accept(F.camera(dx: 0.5)))
        XCTAssertTrue(core.state.isLocked)
    }

    func testAlternatingRejectsDoNotFlicker() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        var t = 1.5
        var published: [String] = []
        for i in 0..<12 {
            step(&core, t, i % 2 == 0 ? F.reject() : F.accept(F.camera(dx: 1)))
            published.append(core.state.kindName)
            t += 1
        }
        XCTAssertTrue(published.allSatisfy { $0 == "locked" }, "\(published)")
    }

    func testBigDisagreementLosesThenRelocks() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        step(&core, 1.5, F.accept(F.camera(dx: 12)))
        XCTAssertEqual(core.state, .lost(.phoneMoved))
        step(&core, 2.0, F.accept(F.camera(dx: 12.5)))
        guard case .locked(let cam, _, _) = core.state else { return XCTFail("\(core.state)") }
        XCTAssertEqual(cam.K[2, 0], 320 + 12.5, accuracy: 1e-6)
    }

    func testSmallDriftRelocksSilently() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        step(&core, 1.5, F.accept(F.camera(dx: 5)))           // 5 px: between agree and lost
        guard case .locked(let c1, _, _) = core.state else { return XCTFail() }
        XCTAssertEqual(c1.K[2, 0], 320, accuracy: 1e-6, "one drifted solution does not move the lock")
        step(&core, 2.5, F.accept(F.camera(dx: 5.5)))
        guard case .locked(let c2, _, _) = core.state else { return XCTFail() }
        XCTAssertEqual(c2.K[2, 0], 325.5, accuracy: 1e-6, "two agreeing drifted solutions re-lock")
    }

    func testPeriodicFullSearchWhileLocked() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        var kinds: [CourtLockPlan.Kind] = []
        for i in 0..<8 { kinds.append(step(&core, 1.5 + Double(i), F.accept(F.camera()))!) }
        XCTAssertEqual(kinds.filter { $0 == .search }.count, 2, "\(kinds)")
        XCTAssertTrue(core.state.isLocked)
    }

    func testPanWhileLockedLosesImmediately() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        core.offer(F.frame(1.0))
        XCTAssertTrue(core.state.isLocked)
        let changed = core.offer(F.frame(1.5, sx: 40))
        XCTAssertTrue(changed)
        XCTAssertEqual(core.state, .lost(.phoneMoved))
        XCTAssertEqual(core.buffer.count, 1, "buffer restarts with post-motion frames")
    }

    func testInFlightAttemptDiscardedAfterMotion() {
        var core = CourtLockCore()
        core.offer(F.frame(0))
        let p = core.plan()!
        core.offer(F.frame(0.5, sx: 40))                      // camera moved while calibrating
        XCTAssertEqual(core.state, .searching(.holdStill))
        XCTAssertFalse(core.apply(p, F.accept(F.camera())))
        XCTAssertEqual(core.stats.discarded, 1)
        XCTAssertEqual(core.state, .searching(.holdStill))
    }

    func testPlayerMotionIsNotCameraMotion() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        for (i, bx) in [100, 250, 400, 520].enumerated() {
            core.offer(FrameDownsampler().frame(from: F.texture(blob: bx), time: 1.0 + Double(i) * 0.5))
        }
        XCTAssertEqual(core.stats.motionEvents, 0)
        XCTAssertTrue(core.state.isLocked)
    }

    func testExposureStepIsNotMotion() {
        let a = F.frame(0), b = F.frame(0.5, brightness: 1.25)
        let m = MotionDetector.compare(a.thumb, b.thumb, fullWidth: 640)
        XCTAssertFalse(m.moved, "\(m)")
        let c = F.frame(1, sx: 30)
        let m2 = MotionDetector.compare(a.thumb, c.thumb, fullWidth: 640)
        XCTAssertTrue(m2.moved, "\(m2)")
        XCTAssertEqual(m2.shiftPx, 30, accuracy: 10.1)
    }

    func testCandidateDroppedAfterTwoRejects() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        XCTAssertEqual(step(&core, 0.5, F.reject()), .refine)
        XCTAssertEqual(core.state.kindName, "candidate")
        XCTAssertEqual(step(&core, 1.0, F.reject(nL: 1)), .search)
        XCTAssertEqual(core.state, .searching(.sideLineMissing))
    }

    func testFrameSizeChangeResets() {
        var core = CourtLockCore()
        step(&core, 0, F.accept(F.camera()))
        step(&core, 0.5, F.accept(F.camera()))
        let big = FrameDownsampler().frame(from: F.texture(size: ImageSize(width: 320, height: 180)), time: 1.0)
        core.offer(big)
        XCTAssertEqual(core.state, .searching(.frameWholeCourt))
        XCTAssertNil(core.lockedCamera)
    }
}
