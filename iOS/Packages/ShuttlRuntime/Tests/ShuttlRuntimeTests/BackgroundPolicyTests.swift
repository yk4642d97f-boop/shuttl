import XCTest
import ShuttlRuntime

@MainActor
final class FakeBackgroundTasks: BackgroundTaskProvider {
    var begun: [String] = []
    var ended: [BackgroundTaskToken] = []
    var expirations: [@MainActor @Sendable () -> Void] = []
    private var next = 1

    func begin(name: String, expiration: @escaping @MainActor @Sendable () -> Void) -> BackgroundTaskToken? {
        begun.append(name)
        expirations.append(expiration)
        defer { next += 1 }
        return BackgroundTaskToken(rawValue: next)
    }

    func end(_ token: BackgroundTaskToken) { ended.append(token) }
}

final class BackgroundPolicyTests: XCTestCase {
    @MainActor
    func testBackgroundPausesThenEndsWindowOnceJobStoppedAndActiveResumes() async {
        let q = FakeQueueControl(), tasks = FakeBackgroundTasks()
        let p = BackgroundPolicy(queue: q, tasks: tasks, windDown: .seconds(60),
                                 sleep: { _ in try? await Task.sleep(for: .seconds(3600)) })
        await p.phaseChanged(.background).value
        var calls = await q.calls
        XCTAssertEqual(calls, ["pause:background", "wait"])
        XCTAssertEqual(tasks.begun.count, 1)
        XCTAssertEqual(tasks.ended, [BackgroundTaskToken(rawValue: 1)], "window ended as soon as the job stopped")
        XCTAssertFalse(p.holdsBackgroundTime)

        await p.phaseChanged(.background).value                  // repeated phase: nothing new
        await p.phaseChanged(.inactive).value                    // Control Center etc.: nothing
        calls = await q.calls
        XCTAssertEqual(calls, ["pause:background", "wait"])
        XCTAssertEqual(tasks.begun.count, 1)

        await p.phaseChanged(.active).value
        calls = await q.calls
        XCTAssertEqual(calls, ["pause:background", "wait", "resume:background"])
    }

    @MainActor
    func testWindDownTimeoutEndsWindowWhenJobDoesNotStop() async {
        let q = FakeQueueControl(), tasks = FakeBackgroundTasks()
        await q.setStopsImmediately(false)
        let p = BackgroundPolicy(queue: q, tasks: tasks, windDown: .milliseconds(1), sleep: { _ in })
        let t = p.phaseChanged(.background)
        await eventually("wind-down ended the window") { await MainActor.run { !p.holdsBackgroundTime } }
        XCTAssertEqual(tasks.ended.count, 1)
        await q.jobStopped()                                     // the job stops late: no double end
        await t.value
        XCTAssertEqual(tasks.ended.count, 1)
    }

    @MainActor
    func testExpirationAndReturningActiveEndTheWindow() async {
        let q = FakeQueueControl(), tasks = FakeBackgroundTasks()
        await q.setStopsImmediately(false)
        let p = BackgroundPolicy(queue: q, tasks: tasks, windDown: .seconds(60),
                                 sleep: { _ in try? await Task.sleep(for: .seconds(3600)) })
        _ = p.phaseChanged(.background)
        XCTAssertTrue(p.holdsBackgroundTime)
        tasks.expirations[0]()                                   // the OS takes the time back
        XCTAssertFalse(p.holdsBackgroundTime)
        XCTAssertEqual(tasks.ended.count, 1)

        await p.phaseChanged(.active).value
        _ = p.phaseChanged(.background)
        XCTAssertTrue(p.holdsBackgroundTime)
        await p.phaseChanged(.active).value                      // back before the job stopped
        XCTAssertFalse(p.holdsBackgroundTime)
        XCTAssertEqual(tasks.ended.count, 2)
        await q.jobStopped()
        let calls = await q.calls.filter { $0 != "wait" }
        XCTAssertEqual(calls, ["pause:background", "resume:background", "pause:background", "resume:background"],
                       "pause always lands before the following resume")

        let none = BackgroundPolicy(queue: FakeQueueControl(), tasks: nil)   // no OS time requested
        await none.phaseChanged(.background).value
        XCTAssertFalse(none.holdsBackgroundTime)
    }
}
