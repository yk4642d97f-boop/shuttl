import XCTest
import ShuttlRuntime

final class ThermalGovernorTests: XCTestCase {
    func testPolicyWithHysteresis() {
        typealias P = ThermalPolicy
        for prev in [ThermalDecision.run, .hold, .pause] {
            XCTAssertEqual(P.decide(.nominal, previous: prev), .run)
            XCTAssertEqual(P.decide(.fair, previous: prev), .run)
            XCTAssertEqual(P.decide(.critical, previous: prev), .pause)
        }
        XCTAssertEqual(P.decide(.serious, previous: .run), .hold)
        XCTAssertEqual(P.decide(.serious, previous: .hold), .hold)
        XCTAssertEqual(P.decide(.serious, previous: .pause), .pause, "stay paused until ≤ fair")
        XCTAssertEqual(P.message(.pause, lowPower: false), "手机有点热，先歇一会儿再继续分析。")
        XCTAssertNotNil(P.message(.hold, lowPower: false))
        XCTAssertNil(P.message(.run, lowPower: false))
        XCTAssertNotNil(P.message(.run, lowPower: true))
        XCTAssertEqual(ThermalLevel(ProcessInfo.ThermalState.critical), .critical)
        XCTAssertLessThan(ThermalLevel.fair, .serious)
    }

    func testTransitionsDriveTheQueue() async {
        let q = FakeQueueControl(), c = FakeConditions()
        let changes = Recorder()
        let g = ThermalGovernor(queue: q, conditions: c) { changes.log("\($0.decision)") }
        var s = await g.refresh()
        XCTAssertEqual(s.decision, .run)
        XCTAssertNil(s.message)
        c.thermalLevel = .serious;  s = await g.refresh()
        XCTAssertEqual(s.decision, .hold)
        c.thermalLevel = .critical; s = await g.refresh()
        XCTAssertEqual(s.decision, .pause)
        XCTAssertEqual(s.message, "手机有点热，先歇一会儿再继续分析。")
        c.thermalLevel = .serious;  s = await g.refresh()
        XCTAssertEqual(s.decision, .pause, "hysteresis")
        c.thermalLevel = .fair;     s = await g.refresh()
        XCTAssertEqual(s.decision, .run)
        _ = await g.refresh()                                                   // no change → no calls
        c.thermalLevel = .critical; _ = await g.refresh()
        c.thermalLevel = .nominal;  _ = await g.refresh()
        let calls = await q.calls
        XCTAssertEqual(calls, ["hold:thermal", "pause:thermal", "release:thermal", "resume:thermal",
                               "pause:thermal", "resume:thermal"])
        // onChange fires on any status change (critical → serious changes the level only).
        XCTAssertEqual(changes.events, ["hold", "pause", "pause", "run", "pause", "run"])
    }

    func testLowPowerModeOnlyAddsAMessage() async {
        let q = FakeQueueControl(), c = FakeConditions()
        c.isLowPowerModeEnabled = true
        let g = ThermalGovernor(queue: q, conditions: c)
        let s = await g.refresh()
        XCTAssertEqual(s.decision, .run)
        XCTAssertEqual(s.message, "低电量模式已打开，分析会慢一些。")
        let calls = await q.calls
        XCTAssertEqual(calls, [])
    }

    func testCriticalPausesARealRunningJobAndFairResumesIt() async {
        let queue = AnalysisQueue(), c = FakeConditions(), rec = Recorder(), gate = Gate()
        let g = ThermalGovernor(queue: queue, conditions: c)
        let id = UUID()
        await queue.enqueue(id) { ctx in
            rec.log("start:\(ctx.attempt)")
            try await gate.wait()
        }
        await eventually("started") { rec.events.count == 1 }
        c.thermalLevel = .critical
        await g.refresh()
        await queue.waitForRunningJobToStop()
        XCTAssertEqual(queue.state(of: id), .paused(.thermal))
        gate.release()
        c.thermalLevel = .fair
        await g.refresh()
        await queue.waitUntilIdle()
        XCTAssertEqual(queue.state(of: id), .finished)
        XCTAssertEqual(rec.events, ["start:1", "start:1"])
    }

    func testNotificationTriggersRefresh() async {
        let q = FakeQueueControl(), c = FakeConditions()
        let g = ThermalGovernor(queue: q, conditions: c)
        await g.start()
        await eventually("initial refresh") { await g.status.level == .nominal }
        c.thermalLevel = .critical
        NotificationCenter.default.post(name: ProcessInfo.thermalStateDidChangeNotification, object: nil)
        await eventually("paused by notification") { await q.calls == ["pause:thermal"] }
        await g.stop()
    }
}

final class StorageGuardTests: XCTestCase {
    private let gb: Int64 = 1_000_000_000

    func testThresholds() {
        let s = StorageGuard(freeSpace: { 0 })          // defaults: 90 MB/min, stop floor 1 GB
        let plenty = s.evaluate(freeBytes: 10 * gb)
        XCTAssertEqual(plenty.level, .ok)
        XCTAssertEqual(plenty.minutesLeft, 100, accuracy: 1e-9, "(10 − 1) GB / 90 MB")
        XCTAssertNil(plenty.message)
        XCTAssertTrue(plenty.canStartRecording)

        let low = s.evaluate(freeBytes: 1_800_000_000)   // 8.9 min above the floor
        XCTAssertEqual(low.level, .low)
        XCTAssertTrue(low.canStartRecording)
        XCTAssertFalse(low.mustStopRecording)
        XCTAssertEqual(low.message, "手机空间只够再录大约 8 分钟，建议先清理一些照片或视频。")

        XCTAssertEqual(s.evaluate(freeBytes: 1_900_000_000).level, .ok, "exactly 10 min is fine")
        let refuse = s.evaluate(freeBytes: 1_400_000_000)
        XCTAssertEqual(refuse.level, .refuseStart, "< 1.5 GB")
        XCTAssertFalse(refuse.canStartRecording)
        XCTAssertFalse(refuse.mustStopRecording, "a running take continues")
        XCTAssertEqual(refuse.message, "手机空间不够了（只剩 1.4 GB），先清理一些照片或视频再录。")

        let stop = s.evaluate(freeBytes: 999_999_999)
        XCTAssertEqual(stop.level, .stopRecording)
        XCTAssertTrue(stop.mustStopRecording)
        XCTAssertEqual(stop.minutesLeft, 0)
        XCTAssertEqual(stop.message, "手机空间快满了，已自动停止录像并保存。")
        XCTAssertEqual(s.evaluate(freeBytes: -5).level, .stopRecording)
    }

    func testMinutesRuleAppliesAboveTheByteRule() {
        // A high bitrate (4K) makes "< 3 min" bite before "< 1.5 GB".
        let s = StorageGuard(bytesPerMinute: 400_000_000, freeSpace: { 0 })
        XCTAssertEqual(s.evaluate(freeBytes: 2 * gb).level, .refuseStart, "2.5 min above the floor")
        XCTAssertEqual(s.evaluate(freeBytes: 3 * gb).level, .low)
        XCTAssertEqual(s.evaluate(freeBytes: 6 * gb).level, .ok)
    }

    func testReadingFailureIsNilAndRealVolumeReads() throws {
        struct Nope: Error {}
        XCTAssertNil(StorageGuard(freeSpace: { throw Nope() }).check())
        XCTAssertEqual(StorageGuard(freeSpace: { 5_000_000_000 }).check()?.level, .ok)
        let free = try StorageGuard.freeBytes(on: FileManager.default.temporaryDirectory)
        XCTAssertGreaterThan(free, 0)
        XCTAssertNotNil(StorageGuard(volume: FileManager.default.temporaryDirectory).check())
    }
}
