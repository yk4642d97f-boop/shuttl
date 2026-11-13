import Foundation
import ShuttlAnalysis

/// Where `AnalysisQueue.enqueue` puts a job among the waiting ones.
public enum QueuePlacement: Sendable, Equatable {
    case front, back
}

/// Why the queue is not starting / running work.
public enum AnalysisPauseReason: String, Sendable, Hashable, CaseIterable {
    case background, thermal, lowStorage, user

    /// Chinese text for the match card.
    public var userMessage: String {
        switch self {
        case .background: return "App 在后台，回到 App 后会自动继续分析。"
        case .thermal: return "手机有点热，先歇一会儿再继续分析。"
        case .lowStorage: return "手机存储空间不足，清理一些空间后会继续分析。"
        case .user: return "分析已暂停。"
        }
    }
}

/// Observable state of one job.
public enum AnalysisJobState: Sendable, Equatable {
    /// Waiting; `position` = jobs ahead of it (the running one included), 0 = next.
    case queued(position: Int)
    case running(progress: Double, stage: String)
    /// Interrupted (or next in line while the queue is held); resumes from checkpoints.
    case paused(AnalysisPauseReason)
    case finished
    case failed(String)
    case cancelled

    public var isTerminal: Bool {
        switch self {
        case .finished, .failed, .cancelled: return true
        default: return false
        }
    }
}

public struct AnalysisJobEvent: Sendable, Equatable {
    public var id: UUID
    public var state: AnalysisJobState
}

/// Thrown by a job to ask for another attempt (e.g. `MatchAnalysis.outcome` is a
/// retryable `.stageFailed`). The queue re-runs it after `retryDelay`, up to `maxAttempts`.
public struct AnalysisRetryRequest: Error, Sendable, Equatable {
    public var reason: String
    public init(_ reason: String) { self.reason = reason }
}

/// Handed to each run of a job.
public struct AnalysisJobContext: Sendable {
    public let id: UUID
    /// 1-based; resets when a paused job is resumed (the pause wasn't the job's fault).
    public let attempt: Int
    public let maxAttempts: Int
    public var isLastAttempt: Bool { attempt >= maxAttempts }
    private let reporter: @Sendable (AnalysisProgress) -> Void

    init(id: UUID, attempt: Int, maxAttempts: Int, reporter: @escaping @Sendable (AnalysisProgress) -> Void) {
        self.id = id; self.attempt = attempt; self.maxAttempts = maxAttempts; self.reporter = reporter
    }

    /// Pass to `AnalysisPipeline.analyze(progress:)`. Thread-safe, cheap, throttled.
    public var progress: @Sendable (AnalysisProgress) -> Void { reporter }
}

public typealias AnalysisJobWork = @Sendable (AnalysisJobContext) async throws -> Void

/// Lock-protected state per job + stream fan-out. Separate from the actor so progress
/// (called from the pipeline's threads) is published synchronously and in order.
final class JobBoard: @unchecked Sendable {
    private let lock = NSRecursiveLock()   // finishing a stream can re-enter via onTermination
    private var states: [UUID: AnalysisJobState] = [:]
    private var runTokens: [UUID: UUID] = [:]
    private var lastPublished: [UUID: AnalysisProgress] = [:]
    private var subscribers: [UUID: [UUID: AsyncStream<AnalysisJobState>.Continuation]] = [:]
    private var eventSubscribers: [UUID: AsyncStream<AnalysisJobEvent>.Continuation] = [:]
    let progressStep: Double

    init(progressStep: Double) { self.progressStep = progressStep }

    func state(_ id: UUID) -> AnalysisJobState? { lock.withLock { states[id] } }
    var snapshot: [UUID: AnalysisJobState] { lock.withLock { states } }

    func set(_ id: UUID, _ s: AnalysisJobState) {
        lock.withLock {
            guard states[id] != s else { return }
            states[id] = s
            publish(id, s)
        }
    }

    func beginRun(_ id: UUID, token: UUID) {
        lock.withLock {
            runTokens[id] = token
            lastPublished[id] = nil
        }
    }

    /// Later progress calls of that run are ignored.
    func endRun(_ id: UUID) { lock.withLock { runTokens[id] = nil; lastPublished[id] = nil } }

    func progress(_ id: UUID, token: UUID, _ p: AnalysisProgress) {
        lock.withLock {
            guard runTokens[id] == token else { return }
            let f = p.fraction.isFinite ? min(max(p.fraction, 0), 1) : 0
            if let last = lastPublished[id] {
                if f < last.fraction { return }                              // never backwards within a run
                if last.stage == p.stage && f - last.fraction < progressStep && f < 1 { return }
            }
            lastPublished[id] = AnalysisProgress(fraction: f, stage: p.stage)
            let s = AnalysisJobState.running(progress: f, stage: p.stage)
            if states[id] != s { states[id] = s; publish(id, s) }
        }
    }

    private func publish(_ id: UUID, _ s: AnalysisJobState) {
        for c in eventSubscribers.values { c.yield(AnalysisJobEvent(id: id, state: s)) }
        guard let subs = subscribers[id] else { return }
        for c in subs.values { c.yield(s) }
        if s.isTerminal {
            subscribers[id] = nil
            for c in subs.values { c.finish() }
        }
    }

    func stream(_ id: UUID) -> AsyncStream<AnalysisJobState> {
        let (stream, c) = AsyncStream.makeStream(of: AnalysisJobState.self, bufferingPolicy: .bufferingNewest(64))
        let key = UUID()
        lock.withLock {
            if let s = states[id] {
                c.yield(s)
                if s.isTerminal { c.finish(); return }
            }
            subscribers[id, default: [:]][key] = c
        }
        c.onTermination = { [weak self] _ in self?.lock.withLock { _ = self?.subscribers[id]?.removeValue(forKey: key) } }
        return stream
    }

    func events() -> AsyncStream<AnalysisJobEvent> {
        let (stream, c) = AsyncStream.makeStream(of: AnalysisJobEvent.self, bufferingPolicy: .unbounded)
        let key = UUID()
        lock.withLock { eventSubscribers[key] = c }
        c.onTermination = { [weak self] _ in self?.lock.withLock { _ = self?.eventSubscribers.removeValue(forKey: key) } }
        return stream
    }

    func forget(_ id: UUID) {
        lock.withLock {
            states[id] = nil
            runTokens[id] = nil
            lastPublished[id] = nil
        }
    }
}
