import Foundation
import ShuttlAnalysis

/// What ThermalGovernor / BackgroundPolicy drive (the queue, or a fake in tests).
public protocol AnalysisQueueControl: Sendable {
    /// Cancel the running job (it goes back to the front and resumes from checkpoints)
    /// and start nothing until every pause reason is resumed.
    func pause(_ reason: AnalysisPauseReason) async
    func resume(_ reason: AnalysisPauseReason) async
    /// Let the running job finish, start nothing new until released.
    func hold(_ reason: AnalysisPauseReason) async
    func release(_ reason: AnalysisPauseReason) async
    /// Returns once no job is running (immediately when idle).
    func waitForRunningJobToStop() async
}

/// FIFO analysis queue with exactly one job running at a time (the detector is not
/// reentrant and two analyses would only fight over the ANE / GPU / memory anyway).
///
/// Pausing cancels the running job's Task and puts it back at the FRONT; the pipeline's
/// checkpoints make the next run resume. Observe per job with `states(for:)` or all with
/// `events()`; `state(of:)` / `snapshot` read synchronously.
public actor AnalysisQueue: AnalysisQueueControl {
    private struct Waiting {
        let id: UUID
        var work: AnalysisJobWork
    }

    private enum StopReason: Equatable { case pause(AnalysisPauseReason), cancel }

    private struct Running {
        let id: UUID
        let token: UUID
        var work: AnalysisJobWork
        var task: Task<RunOutcome, Never>?
        var stop: StopReason?
    }

    private enum RunOutcome: Sendable { case finished, cancelled, failed(String) }

    private var waiting: [Waiting] = []
    private var current: Running?
    private var pauses: [AnalysisPauseReason] = []
    private var holds: [AnalysisPauseReason] = []
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []
    private var stopWaiters: [CheckedContinuation<Void, Never>] = []
    private let board: JobBoard

    public let maxAttempts: Int
    public let retryDelay: Duration
    public let priority: TaskPriority

    /// - maxAttempts: runs per job when it throws `AnalysisRetryRequest`.
    /// - progressStep: progress changes smaller than this (same stage) aren't published.
    public init(maxAttempts: Int = 3, retryDelay: Duration = .seconds(2), priority: TaskPriority = .userInitiated,
                progressStep: Double = 0.002) {
        self.maxAttempts = max(1, maxAttempts)
        self.retryDelay = retryDelay
        self.priority = priority
        board = JobBoard(progressStep: progressStep)
    }

    // MARK: Observation (nonisolated: safe from any thread / the main actor)

    public nonisolated func state(of id: UUID) -> AnalysisJobState? { board.state(id) }
    public nonisolated var snapshot: [UUID: AnalysisJobState] { board.snapshot }
    /// Current state first, then every change; finishes after a terminal state.
    /// For an unknown id it waits until the job is enqueued.
    public nonisolated func states(for id: UUID) -> AsyncStream<AnalysisJobState> { board.stream(id) }
    /// Every state change of every job, in order.
    public nonisolated func events() -> AsyncStream<AnalysisJobEvent> { board.events() }

    public var runningJobID: UUID? { current?.id }
    public var waitingJobIDs: [UUID] { waiting.map(\.id) }
    public var activePauseReasons: [AnalysisPauseReason] { pauses }
    public var activeHoldReasons: [AnalysisPauseReason] { holds }

    // MARK: Jobs

    /// Adds a job. Returns false when `id` is already running (nothing changes) — a
    /// waiting job with that id gets the new `work` and, for `.front`, moves up.
    @discardableResult
    public func enqueue(_ id: UUID, placement: QueuePlacement = .back, work: @escaping AnalysisJobWork) -> Bool {
        if var cur = current, cur.id == id {
            guard cur.stop == .cancel else { cur.work = work; current = cur; return false }
            // Cancelled but still winding down: queue it again; it starts once the old run ends.
        }
        if let i = waiting.firstIndex(where: { $0.id == id }) {
            var w = waiting.remove(at: i)
            w.work = work
            if placement == .front { waiting.insert(w, at: 0) } else { waiting.insert(w, at: i) }
        } else {
            let w = Waiting(id: id, work: work)
            if placement == .front { waiting.insert(w, at: 0) } else { waiting.append(w) }
        }
        publishPositions()
        startNextIfPossible()
        return true
    }

    /// Cancels a waiting job, or the running one (its Task is cancelled).
    public func cancel(_ id: UUID) {
        if let i = waiting.firstIndex(where: { $0.id == id }) {
            waiting.remove(at: i)
            board.set(id, .cancelled)
            publishPositions()
            resumeIdleWaitersIfIdle()
            return
        }
        guard var cur = current, cur.id == id, cur.stop != .cancel else { return }
        let alreadyStopping = cur.stop != nil
        cur.stop = .cancel
        current = cur
        board.endRun(id)
        board.set(id, .cancelled)
        if !alreadyStopping { cur.task?.cancel() }
    }

    /// Drops a finished / failed / cancelled job's state.
    public func forget(_ id: UUID) {
        guard current?.id != id, !waiting.contains(where: { $0.id == id }) else { return }
        board.forget(id)
    }

    // MARK: Pause / hold

    public func pause(_ reason: AnalysisPauseReason) {
        if !pauses.contains(reason) { pauses.append(reason) }
        if var cur = current, cur.stop == nil {
            cur.stop = .pause(reason)
            current = cur
            board.endRun(cur.id)
            board.set(cur.id, .paused(reason))
            cur.task?.cancel()
        }
        publishPositions()
        resumeIdleWaitersIfIdle()
    }

    public func resume(_ reason: AnalysisPauseReason) {
        pauses.removeAll { $0 == reason }
        publishPositions()
        startNextIfPossible()
    }

    /// Clears every pause reason (a "继续分析" button). Holds stay.
    public func resumeAll() {
        pauses.removeAll()
        publishPositions()
        startNextIfPossible()
    }

    public func hold(_ reason: AnalysisPauseReason) {
        if !holds.contains(reason) { holds.append(reason) }
        publishPositions()
        resumeIdleWaitersIfIdle()
    }

    public func release(_ reason: AnalysisPauseReason) {
        holds.removeAll { $0 == reason }
        publishPositions()
        startNextIfPossible()
    }

    public var isPaused: Bool { !pauses.isEmpty }

    // MARK: Waiting

    public func waitForRunningJobToStop() async {
        guard current != nil else { return }
        await withCheckedContinuation { stopWaiters.append($0) }
    }

    /// Returns when nothing runs and nothing can start (empty, paused or held).
    public func waitUntilIdle() async {
        guard !isIdle else { return }
        await withCheckedContinuation { idleWaiters.append($0) }
    }

    private var isIdle: Bool { current == nil && (waiting.isEmpty || blockingReason != nil) }

    private func resumeIdleWaitersIfIdle() {
        guard isIdle, !idleWaiters.isEmpty else { return }
        let w = idleWaiters
        idleWaiters = []
        w.forEach { $0.resume() }
    }

    // MARK: Scheduling

    private var blockingReason: AnalysisPauseReason? { pauses.first ?? holds.first }

    /// Head shows why it waits when blocked; everyone else shows their position.
    private func publishPositions() {
        let ahead = current == nil ? 0 : 1
        let blocked = blockingReason
        for (i, w) in waiting.enumerated() {
            if i == 0, let blocked, current == nil {
                board.set(w.id, .paused(blocked))
            } else {
                board.set(w.id, .queued(position: ahead + i))
            }
        }
    }

    private func startNextIfPossible() {
        guard current == nil, blockingReason == nil, !waiting.isEmpty else {
            resumeIdleWaitersIfIdle()
            return
        }
        let job = waiting.removeFirst()
        let token = UUID()
        current = Running(id: job.id, token: token, work: job.work, task: nil, stop: nil)
        board.beginRun(job.id, token: token)
        board.set(job.id, .running(progress: 0, stage: "start"))
        publishPositions()

        let board = self.board, id = job.id, work = job.work
        let maxAttempts = self.maxAttempts, retryDelay = self.retryDelay
        let task = Task.detached(priority: priority) { () -> RunOutcome in
            var attempt = 1
            while true {
                let ctx = AnalysisJobContext(id: id, attempt: attempt, maxAttempts: maxAttempts) { p in
                    board.progress(id, token: token, p)
                }
                do {
                    try await work(ctx)
                    return Task.isCancelled ? .cancelled : .finished
                } catch is CancellationError {
                    return .cancelled
                } catch let retry as AnalysisRetryRequest {
                    if Task.isCancelled { return .cancelled }
                    guard attempt < maxAttempts else { return .failed(retry.reason) }
                    attempt += 1
                    do { try await Task.sleep(for: retryDelay) } catch { return .cancelled }
                } catch {
                    if Task.isCancelled { return .cancelled }
                    return .failed((error as? LocalizedError)?.errorDescription ?? String(describing: error))
                }
            }
        }
        current?.task = task
        Task { [weak self] in
            let outcome = await task.value
            await self?.jobEnded(token: token, outcome: outcome)
        }
    }

    private func jobEnded(token: UUID, outcome: RunOutcome) {
        guard let cur = current, cur.token == token else { return }
        current = nil
        board.endRun(cur.id)
        switch (cur.stop, outcome) {
        case (.cancel?, _):
            break                                           // already published .cancelled
        case (.pause?, .finished):
            board.set(cur.id, .finished)                    // it made it before the pause landed
        case (.pause(let reason)?, _):
            waiting.insert(Waiting(id: cur.id, work: cur.work), at: 0)
            board.set(cur.id, .paused(blockingReason ?? reason))
        case (nil, .finished):
            board.set(cur.id, .finished)
        case (nil, .cancelled):
            board.set(cur.id, .cancelled)
        case (nil, .failed(let message)):
            board.set(cur.id, .failed(message))
        }
        let s = stopWaiters
        stopWaiters = []
        s.forEach { $0.resume() }
        publishPositions()
        startNextIfPossible()
        resumeIdleWaitersIfIdle()
    }
}
