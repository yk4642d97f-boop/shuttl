import XCTest
import ShuttlAnalysis
import ShuttlRuntime

final class AnalysisQueueTests: XCTestCase {
    private static func job(_ name: String, _ rec: Recorder, gate: Gate? = nil, sleep: Duration = .zero) -> AnalysisJobWork {
        { ctx in
            rec.enter(); defer { rec.leave() }
            rec.log("start:\(name):\(ctx.attempt)")
            if let gate { try await gate.wait() }
            if sleep > .zero { try await Task.sleep(for: sleep) }
            rec.log("end:\(name)")
        }
    }

    private func starts(_ rec: Recorder) -> [String] { rec.events.filter { $0.hasPrefix("start:") } }

    func testFIFOOneAtATime() async {
        let q = AnalysisQueue(), rec = Recorder()
        let ids = (0..<4).map { _ in UUID() }
        for (i, id) in ids.enumerated() { await q.enqueue(id, work: Self.job("\(i)", rec, sleep: .milliseconds(3))) }
        await q.waitUntilIdle()
        XCTAssertEqual(starts(rec), ["start:0:1", "start:1:1", "start:2:1", "start:3:1"])
        XCTAssertEqual(rec.maxConcurrent, 1)
        XCTAssertEqual(ids.map { q.state(of: $0) }, Array(repeating: .finished, count: 4))
    }

    func testConcurrentEnqueueIsSingleFlight() async {
        let q = AnalysisQueue(), rec = Recorder()
        let ids = (0..<40).map { _ in UUID() }
        await withTaskGroup(of: Void.self) { g in
            for (i, id) in ids.enumerated() {
                g.addTask { await q.enqueue(id, work: Self.job("\(i)", rec, sleep: .milliseconds(1))) }
            }
        }
        await q.waitUntilIdle()
        XCTAssertEqual(rec.maxConcurrent, 1, "never two analyses at once")
        XCTAssertEqual(starts(rec).count, 40)
        XCTAssertTrue(ids.allSatisfy { q.state(of: $0) == .finished })
    }

    func testFrontPlacementPositionsAndDuplicateEnqueue() async {
        let q = AnalysisQueue(), rec = Recorder(), gate = Gate()
        let a = UUID(), b = UUID(), c = UUID(), d = UUID()
        await q.enqueue(a, work: Self.job("a", rec, gate: gate))
        await q.enqueue(b, work: Self.job("b", rec))
        await q.enqueue(c, work: Self.job("c", rec))
        await q.enqueue(d, placement: .front, work: Self.job("d", rec))
        let again = await q.enqueue(a, work: Self.job("a2", rec))
        XCTAssertFalse(again, "a running job isn't queued twice")
        let running = await q.runningJobID
        XCTAssertEqual(running, a)
        XCTAssertEqual(q.state(of: d), .queued(position: 1))
        XCTAssertEqual(q.state(of: b), .queued(position: 2))
        XCTAssertEqual(q.state(of: c), .queued(position: 3))
        await q.enqueue(c, placement: .front, work: Self.job("c", rec))            // moves up
        XCTAssertEqual(q.state(of: c), .queued(position: 1))
        gate.release()
        await q.waitUntilIdle()
        XCTAssertEqual(starts(rec), ["start:a:1", "start:c:1", "start:d:1", "start:b:1"])
    }

    func testPauseRequeuesRunningJobAtFrontAndResumeRerunsIt() async {
        let q = AnalysisQueue(), rec = Recorder(), gate = Gate()
        let a = UUID(), b = UUID()
        await q.enqueue(a, work: Self.job("a", rec, gate: gate))
        await q.enqueue(b, work: Self.job("b", rec))
        await eventually("a started") { rec.events.contains("start:a:1") }
        await q.pause(.thermal)
        XCTAssertEqual(q.state(of: a), .paused(.thermal), "published at once")
        await q.waitForRunningJobToStop()
        let waiting = await q.waitingJobIDs
        XCTAssertEqual(waiting, [a, b], "back at the front")
        XCTAssertEqual(q.state(of: a), .paused(.thermal))
        XCTAssertEqual(q.state(of: b), .queued(position: 1))
        try? await Task.sleep(for: .milliseconds(20))
        XCTAssertFalse(rec.events.contains("start:b:1"), "nothing starts while paused")

        gate.release()
        await q.resume(.thermal)
        await q.waitUntilIdle()
        XCTAssertEqual(starts(rec), ["start:a:1", "start:a:1", "start:b:1"], "a re-runs first (attempt resets)")
        XCTAssertEqual(rec.events.filter { $0 == "end:a" }.count, 1)
        XCTAssertEqual(q.state(of: a), .finished)
        XCTAssertEqual(q.state(of: b), .finished)
        XCTAssertEqual(rec.maxConcurrent, 1)
    }

    func testPausedWhileIdleShowsReasonOnHeadAndStartsNothing() async {
        let q = AnalysisQueue(), rec = Recorder()
        let a = UUID(), b = UUID()
        await q.pause(.background)
        await q.pause(.thermal)
        await q.enqueue(a, work: Self.job("a", rec))
        await q.enqueue(b, work: Self.job("b", rec))
        XCTAssertEqual(q.state(of: a), .paused(.background))
        XCTAssertEqual(q.state(of: b), .queued(position: 1))
        await q.resume(.background)
        XCTAssertEqual(q.state(of: a), .paused(.thermal), "still paused for the other reason")
        let isPaused = await q.isPaused
        XCTAssertTrue(isPaused)
        await q.resumeAll()
        await q.waitUntilIdle()
        XCTAssertEqual(starts(rec), ["start:a:1", "start:b:1"])
    }

    func testCancelQueuedAndRunning() async {
        let q = AnalysisQueue(), rec = Recorder(), gate = Gate()
        let a = UUID(), b = UUID(), c = UUID()
        await q.enqueue(a, work: Self.job("a", rec, gate: gate))
        await q.enqueue(b, work: Self.job("b", rec))
        await q.enqueue(c, work: Self.job("c", rec))
        await q.cancel(b)
        XCTAssertEqual(q.state(of: b), .cancelled)
        XCTAssertEqual(q.state(of: c), .queued(position: 1))
        await q.cancel(UUID())                                               // unknown: no-op
        await eventually("a started") { rec.events.contains("start:a:1") }
        await q.cancel(a)
        XCTAssertEqual(q.state(of: a), .cancelled)
        await q.waitUntilIdle()
        XCTAssertEqual(starts(rec), ["start:a:1", "start:c:1"])
        XCTAssertFalse(rec.events.contains("end:a"), "the running task was cancelled")
        XCTAssertEqual(q.state(of: a), .cancelled)
        XCTAssertEqual(q.state(of: c), .finished)
    }

    func testRetryRequestReRunsUpToMaxAttempts() async {
        let q = AnalysisQueue(maxAttempts: 3, retryDelay: .milliseconds(1)), rec = Recorder()
        let ok = UUID(), bad = UUID()
        await q.enqueue(ok) { ctx in
            rec.log("ok:\(ctx.attempt)")
            if ctx.attempt < 2 { throw AnalysisRetryRequest("interrupted") }
        }
        await q.enqueue(bad) { ctx in
            rec.log("bad:\(ctx.attempt):\(ctx.isLastAttempt)")
            throw AnalysisRetryRequest("still interrupted")
        }
        await q.waitUntilIdle()
        XCTAssertEqual(rec.events, ["ok:1", "ok:2", "bad:1:false", "bad:2:false", "bad:3:true"])
        XCTAssertEqual(q.state(of: ok), .finished)
        XCTAssertEqual(q.state(of: bad), .failed("still interrupted"))
    }

    func testFailureUsesLocalizedDescription() async {
        struct Broken: LocalizedError { var errorDescription: String? { "视频打不开" } }
        let q = AnalysisQueue(), id = UUID()
        await q.enqueue(id) { _ in throw Broken() }
        await q.waitUntilIdle()
        XCTAssertEqual(q.state(of: id), .failed("视频打不开"))
    }

    func testStateStreamIsOrderedThrottledAndFinishes() async {
        let q = AnalysisQueue(), id = UUID()
        let stream = q.states(for: id)                     // before the job exists
        let collector = Task { () -> [AnalysisJobState] in
            var out: [AnalysisJobState] = []
            for await s in stream { out.append(s) }
            return out
        }
        let events = q.events()
        await q.enqueue(id) { ctx in
            ctx.progress(AnalysisProgress(fraction: 0.1, stage: "detect"))
            ctx.progress(AnalysisProgress(fraction: 0.1001, stage: "detect"))   // below the step
            ctx.progress(AnalysisProgress(fraction: 0.5, stage: "players"))
            ctx.progress(AnalysisProgress(fraction: 0.4, stage: "players"))     // backwards
        }
        let seen = await collector.value
        XCTAssertEqual(seen, [.queued(position: 0), .running(progress: 0, stage: "start"),
                              .running(progress: 0.1, stage: "detect"), .running(progress: 0.5, stage: "players"), .finished])
        var all: [AnalysisJobState] = []
        for await e in events where e.id == id {
            all.append(e.state)
            if e.state.isTerminal { break }
        }
        XCTAssertEqual(all, seen, "global events carry the same sequence")
        // Subscribing after the end yields the terminal state and finishes.
        var late: [AnalysisJobState] = []
        for await s in q.states(for: id) { late.append(s) }
        XCTAssertEqual(late, [.finished])
    }

    func testHoldLetsRunningJobFinishButStartsNothing() async {
        let q = AnalysisQueue(), rec = Recorder(), gate = Gate()
        let a = UUID(), b = UUID()
        await q.enqueue(a, work: Self.job("a", rec, gate: gate))
        await q.enqueue(b, work: Self.job("b", rec))
        await q.hold(.thermal)
        gate.release()
        await q.waitUntilIdle()
        XCTAssertEqual(q.state(of: a), .finished, "the running job was not interrupted")
        XCTAssertEqual(q.state(of: b), .paused(.thermal))
        XCTAssertFalse(rec.events.contains("start:b:1"))
        await q.resumeAll()
        XCTAssertEqual(q.state(of: b), .paused(.thermal), "resumeAll doesn't lift holds")
        await q.release(.thermal)
        await q.waitUntilIdle()
        XCTAssertEqual(q.state(of: b), .finished)
    }

    func testWaitForRunningJobToStopWhenIdleReturnsAtOnce() async {
        let q = AnalysisQueue()
        await q.waitForRunningJobToStop()
        await q.waitUntilIdle()
    }
}
