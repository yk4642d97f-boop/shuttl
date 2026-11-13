import Foundation
import ShuttlCore

/// Wall-clock supervision of a synchronous stage on its own thread (players: Apple Vision
/// perform(), which nothing else times out: the decode watchdog only covers the reader).
struct StageWatchConfig: Sendable {
    /// No completed frame for this long → the stage fails with a retryable `StageStalledError`. nil = off.
    var stallSeconds: Double?
    /// A `<stage>_progress` event this often, whether frames complete or not. nil = off.
    var heartbeatSeconds: Double?
    var pollSeconds: Double = 1
    /// A gap between two polls longer than this means the process was frozen (iOS suspended
    /// the app); that time counts neither as a stall nor as progress.
    var suspendGapSeconds: Double = 5
    var clock: @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
}

/// Shared by the stage thread and the watchdog. Once abandoned (stall or cancel) the stage's
/// late frames and checkpoint saves are ignored, so a pass that unblocks after the retry
/// started can't overwrite the retry's partial file.
final class StageWatch: @unchecked Sendable {
    let flag = CancelFlag()
    private let lock = NSLock()
    private let clock: @Sendable () -> TimeInterval
    private var abandoned = false
    private(set) var lastFrame = -1
    private(set) var framesDone = 0
    private(set) var lastProgressAt: TimeInterval

    init(clock: @escaping @Sendable () -> TimeInterval) {
        self.clock = clock
        lastProgressAt = clock()
    }

    func frameDone(_ index: Int) {
        let now = clock()
        lock.withLock {
            guard !abandoned else { return }
            lastFrame = index; framesDone += 1; lastProgressAt = now
        }
    }

    /// Runs `body` (a checkpoint save) unless abandoned; abandoning waits for it to finish.
    func ifLive(_ body: () -> Void) {
        lock.lock(); defer { lock.unlock() }
        if !abandoned { body() }
    }

    /// true for the call that abandoned the stage.
    func abandon() -> Bool {
        let first = lock.withLock { () -> Bool in
            defer { abandoned = true }
            return !abandoned
        }
        flag.cancel()
        return first
    }

    func snapshot() -> (lastFrame: Int, framesDone: Int, lastProgressAt: TimeInterval) {
        lock.withLock { (lastFrame, framesDone, lastProgressAt) }
    }

    func excuse(_ seconds: Double) { lock.withLock { lastProgressAt += seconds } }
}

/// Delivers the first of: the stage's result, a stall, a cancellation.
private final class FirstResult<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<T, Error>?
    private var pending: Result<T, Error>?
    private var done = false

    var isDone: Bool { lock.withLock { done } }

    func wait(_ c: CheckedContinuation<T, Error>) {
        let ready: Result<T, Error>? = lock.withLock {
            if let p = pending { pending = nil; return p }
            continuation = c
            return nil
        }
        if let ready { Self.resume(c, ready) }
    }

    func finish(_ r: Result<T, Error>) {
        let c: CheckedContinuation<T, Error>? = lock.withLock {
            guard !done else { return nil }
            done = true
            if let c = continuation { continuation = nil; return c }
            pending = r
            return nil
        }
        if let c { Self.resume(c, r) }
    }

    private static func resume(_ c: CheckedContinuation<T, Error>, _ r: Result<T, Error>) {
        switch r {
        case .success(let v): c.resume(returning: v)
        case .failure(let e): c.resume(throwing: e)
        }
    }
}

/// `detachedCancellable` with a stall watchdog and a heartbeat. `work` must call
/// `watch.frameDone` per completed unit, poll `watch.flag`, and wrap saves in `watch.ifLive`.
/// Events: `<stage>_start`, `<stage>_progress` {last_frame, frames_done, elapsed_s,
/// since_progress_s[, suspended_s]}, `<stage>_stall`. A stall or a cancellation returns at once,
/// even when the stage's thread is stuck inside a call that never returns.
func watchedDetached<T: Sendable>(stage: String, config: StageWatchConfig, fields: [String: JSONValue] = [:],
                                  onEvent: @escaping @Sendable (String, [String: JSONValue]) -> Void,
                                  _ work: @escaping @Sendable (StageWatch) throws -> T) async throws -> T {
    let watch = StageWatch(clock: config.clock)
    let result = FirstResult<T>()
    let started = config.clock()
    onEvent("\(stage)_start", fields.merging(["stall_s": config.stallSeconds.map(JSONValue.number) ?? .null]) { a, _ in a })
    let worker = Task.detached(priority: .userInitiated) {
        result.finish(Result { try work(watch) })
    }
    let monitor = Task.detached(priority: .utility) {
        var lastTick = started, lastBeat = started, suspended = 0.0
        while !Task.isCancelled, !result.isDone {
            try? await Task.sleep(nanoseconds: UInt64(max(0.001, config.pollSeconds) * 1e9))
            guard !Task.isCancelled, !result.isDone else { return }
            let now = config.clock()
            var resumed = false
            if now - lastTick > config.suspendGapSeconds {
                watch.excuse(now - lastTick)
                suspended += now - lastTick
                resumed = true                               // say so in the log right away
            }
            lastTick = now
            let s = watch.snapshot()
            var info = fields
            info["last_frame"] = .number(Double(s.lastFrame))
            info["frames_done"] = .number(Double(s.framesDone))
            info["elapsed_s"] = .number(((now - started) * 10).rounded() / 10)
            info["since_progress_s"] = .number(((now - s.lastProgressAt) * 10).rounded() / 10)
            if suspended > 0 { info["suspended_s"] = .number((suspended * 10).rounded() / 10) }
            if let hb = config.heartbeatSeconds, resumed || now - lastBeat >= hb {
                onEvent("\(stage)_progress", info)
                lastBeat = now
            }
            if let stall = config.stallSeconds, now - s.lastProgressAt >= stall, watch.abandon() {
                onEvent("\(stage)_stall", info)
                result.finish(.failure(StageStalledError(stage: stage, seconds: now - s.lastProgressAt, lastFrame: s.lastFrame)))
                return
            }
        }
    }
    defer { monitor.cancel() }
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { result.wait($0) }
    } onCancel: {
        _ = watch.abandon()
        worker.cancel()
        result.finish(.failure(CancellationError()))
    }
}
