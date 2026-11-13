import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// The app's scene phase, without importing SwiftUI (map `ScenePhase` 1:1).
public enum AppPhase: Sendable, Equatable {
    case active, inactive, background
}

public struct BackgroundTaskToken: Sendable, Hashable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
}

/// beginBackgroundTask / endBackgroundTask (UIKit on iOS, a fake in tests).
@MainActor
public protocol BackgroundTaskProvider: AnyObject {
    /// nil when the OS gives no time. `expiration` runs if the OS takes the time back.
    func begin(name: String, expiration: @escaping @MainActor @Sendable () -> Void) -> BackgroundTaskToken?
    func end(_ token: BackgroundTaskToken)
}

#if canImport(UIKit) && !os(watchOS)
@MainActor
public final class UIKitBackgroundTasks: BackgroundTaskProvider {
    public init() {}

    public func begin(name: String, expiration: @escaping @MainActor @Sendable () -> Void) -> BackgroundTaskToken? {
        let id = UIApplication.shared.beginBackgroundTask(withName: name) { MainActor.assumeIsolated { expiration() } }
        return id == .invalid ? nil : BackgroundTaskToken(rawValue: id.rawValue)
    }

    public func end(_ token: BackgroundTaskToken) {
        UIApplication.shared.endBackgroundTask(UIBackgroundTaskIdentifier(rawValue: token.rawValue))
    }
}
#endif

/// What to do on scene phase changes. The GPU / ANE may not be used in the background,
/// so analysis can't continue there. On `.background` it asks for a short background
/// window, pauses the queue (reason `.background`) and ends the window as soon as the
/// running job has stopped (its finished stages are checkpointed) or `windDown` passes.
/// On `.active` it resumes. `.inactive` (Control Center, a call banner) changes nothing.
///
///     .onChange(of: scenePhase) { _, p in policy.phaseChanged(p == .active ? .active : p == .background ? .background : .inactive) }
@MainActor
public final class BackgroundPolicy {
    private let queue: any AnalysisQueueControl
    private let tasks: BackgroundTaskProvider?
    private let windDown: Duration
    private let sleep: @Sendable (Duration) async -> Void
    private var token: BackgroundTaskToken?
    private var generation = 0
    /// Queue calls run strictly in phase order (a quick background → active must never
    /// leave the queue paused because the pause landed after the resume).
    private var lastQueueCall: Task<Void, Never>?
    public private(set) var phase: AppPhase = .active

    /// - tasks: nil = don't ask the OS for background time.
    /// - windDown: longest the background window is held while the job stops.
    public init(queue: any AnalysisQueueControl, tasks: BackgroundTaskProvider?, windDown: Duration = .seconds(5),
                sleep: @escaping @Sendable (Duration) async -> Void = { try? await Task.sleep(for: $0) }) {
        self.queue = queue
        self.tasks = tasks
        self.windDown = windDown
        self.sleep = sleep
    }

    #if canImport(UIKit) && !os(watchOS)
    /// With UIKit's beginBackgroundTask.
    public convenience init(queue: any AnalysisQueueControl, windDown: Duration = .seconds(5)) {
        self.init(queue: queue, tasks: UIKitBackgroundTasks(), windDown: windDown)
    }
    #endif

    /// True while a background window is held.
    public var holdsBackgroundTime: Bool { token != nil }

    /// Call on every scene phase change. The returned task finishes when the queue has
    /// been paused (background) or resumed (active); tests await it.
    @discardableResult
    public func phaseChanged(_ newPhase: AppPhase) -> Task<Void, Never> {
        let old = phase
        phase = newPhase
        switch newPhase {
        case .inactive:
            return Task {}
        case .background:
            guard old != .background else { return Task {} }
            generation += 1
            let gen = generation
            if token == nil {
                token = tasks?.begin(name: "shuttl.analysis.pause") { [weak self] in self?.endWindow(gen) }
            }
            let queue = self.queue, windDown = self.windDown, sleep = self.sleep
            // Whichever comes first ends the window: the job stopped, or wind-down elapsed.
            Task { [weak self] in
                await sleep(windDown)
                self?.endWindow(gen)
            }
            let paused = chain { await queue.pause(.background) }
            // Not chained: a slow job must not delay a later resume.
            return Task { [weak self] in
                await paused.value
                await queue.waitForRunningJobToStop()
                self?.endWindow(gen)
            }
        case .active:
            generation += 1
            endWindow(generation)
            let queue = self.queue
            return chain { await queue.resume(.background) }
        }
    }

    private func chain(_ op: @escaping @Sendable () async -> Void) -> Task<Void, Never> {
        let previous = lastQueueCall
        let t = Task {
            await previous?.value
            await op()
        }
        lastQueueCall = t
        return t
    }

    private func endWindow(_ gen: Int) {
        guard gen == generation, let t = token else { return }
        token = nil
        tasks?.end(t)
    }
}
