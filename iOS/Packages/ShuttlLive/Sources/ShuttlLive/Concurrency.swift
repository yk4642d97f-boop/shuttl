import Foundation

/// A cancellation signal a synchronous, detached loop can poll.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isSet: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }
}

/// FIFO mutual exclusion for async code. HeatmapDetector keeps per-run state and
/// is not reentrant, so concurrent analyses must take turns on it.
actor AsyncGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        guard busy else { busy = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }
}

func withGate<T>(_ gate: AsyncGate, _ body: () async throws -> T) async throws -> T {
    await gate.acquire()
    do {
        try Task.checkCancellation()
        let value = try await body()
        await gate.release()
        return value
    } catch {
        await gate.release()
        throw error
    }
}

/// Runs synchronous work off the cooperative pool and cancels it (via `flag`) when
/// the calling task is cancelled; `Task.detached` alone would never see that.
func detachedCancellable<T: Sendable>(priority: TaskPriority = .userInitiated,
                                      _ work: @escaping @Sendable (CancelFlag) throws -> T) async throws -> T {
    let flag = CancelFlag()
    let task = Task.detached(priority: priority) { try work(flag) }
    return try await withTaskCancellationHandler {
        try await task.value
    } onCancel: {
        flag.cancel()
        task.cancel()
    }
}
