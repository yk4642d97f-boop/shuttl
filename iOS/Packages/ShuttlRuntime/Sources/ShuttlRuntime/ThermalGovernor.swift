import Foundation

/// ProcessInfo.ThermalState as a comparable value (testable without the OS).
public enum ThermalLevel: Int, Sendable, Comparable, CaseIterable {
    case nominal, fair, serious, critical

    public init(_ s: ProcessInfo.ThermalState) {
        switch s {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: self = .serious
        }
    }

    public static func < (a: ThermalLevel, b: ThermalLevel) -> Bool { a.rawValue < b.rawValue }

    public var name: String { ["nominal", "fair", "serious", "critical"][rawValue] }
}

public enum ThermalDecision: Sendable, Equatable {
    /// Start and run jobs.
    case run
    /// The running job continues; nothing new starts.
    case hold
    /// Cancel the running job (it resumes later from checkpoints); nothing starts.
    case pause
}

/// nominal / fair → run; serious → hold; critical → pause, and stay paused until the
/// phone is back to fair or better (hysteresis: serious after critical is still paused).
/// Low Power Mode doesn't stop analysis (it is only slower) — it only adds a message.
public enum ThermalPolicy {
    public static func decide(_ level: ThermalLevel, previous: ThermalDecision) -> ThermalDecision {
        switch level {
        case .critical: return .pause
        case .serious: return previous == .pause ? .pause : .hold
        case .nominal, .fair: return .run
        }
    }

    /// Chinese status line for the UI (nil = nothing to say).
    public static func message(_ decision: ThermalDecision, lowPower: Bool) -> String? {
        switch decision {
        case .pause: return "手机有点热，先歇一会儿再继续分析。"
        case .hold: return "手机有点热，这一场分析完先歇一会儿再分析下一场。"
        case .run: return lowPower ? "低电量模式已打开，分析会慢一些。" : nil
        }
    }
}

/// Where the governor reads the device state (ProcessInfo, or a fake in tests).
public protocol DeviceConditions: Sendable {
    var thermalLevel: ThermalLevel { get }
    var isLowPowerModeEnabled: Bool { get }
}

public struct SystemDeviceConditions: DeviceConditions {
    public init() {}
    public var thermalLevel: ThermalLevel { ThermalLevel(ProcessInfo.processInfo.thermalState) }
    public var isLowPowerModeEnabled: Bool { ProcessInfo.processInfo.isLowPowerModeEnabled }
}

public struct ThermalStatus: Sendable, Equatable {
    public var level: ThermalLevel
    public var lowPowerMode: Bool
    public var decision: ThermalDecision
    /// Chinese text for the UI, nil when all is well.
    public var message: String? { ThermalPolicy.message(decision, lowPower: lowPowerMode) }
}

/// Watches the thermal state and Low Power Mode and pauses / holds the analysis queue.
/// Notifications are funnelled through one loop, so queue calls happen in order.
public actor ThermalGovernor {
    private let queue: any AnalysisQueueControl
    private let conditions: any DeviceConditions
    private let onChange: (@Sendable (ThermalStatus) -> Void)?
    public private(set) var status: ThermalStatus
    private var loop: Task<Void, Never>?
    private var observers: NotificationObservers?

    public init(queue: any AnalysisQueueControl, conditions: any DeviceConditions = SystemDeviceConditions(),
                onChange: (@Sendable (ThermalStatus) -> Void)? = nil) {
        self.queue = queue
        self.conditions = conditions
        self.onChange = onChange
        status = ThermalStatus(level: .nominal, lowPowerMode: false, decision: .run)
    }

    /// Starts observing ProcessInfo notifications and applies the current state.
    public func start() {
        guard loop == nil else { return }
        let (ticks, tick) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        observers = NotificationObservers(names: [ProcessInfo.thermalStateDidChangeNotification,
                                                  .NSProcessInfoPowerStateDidChange]) { tick.yield() }
        loop = Task { [weak self] in
            for await _ in ticks { await self?.refresh() }
        }
        tick.yield()
    }

    public func stop() {
        observers = nil
        loop?.cancel()
        loop = nil
    }

    /// Reads the conditions and drives the queue (also callable directly, e.g. in tests).
    @discardableResult
    public func refresh() async -> ThermalStatus {
        let level = conditions.thermalLevel
        let low = conditions.isLowPowerModeEnabled
        let old = status
        let decision = ThermalPolicy.decide(level, previous: old.decision)
        status = ThermalStatus(level: level, lowPowerMode: low, decision: decision)
        if decision != old.decision {
            switch decision {
            case .pause:
                await queue.pause(.thermal)
                if old.decision == .hold { await queue.release(.thermal) }
            case .hold:
                await queue.hold(.thermal)
                if old.decision == .pause { await queue.resume(.thermal) }
            case .run:
                if old.decision == .pause { await queue.resume(.thermal) }
                if old.decision == .hold { await queue.release(.thermal) }
            }
        }
        if status != old { onChange?(status) }
        return status
    }
}

/// NotificationCenter block observers, removed on deinit.
final class NotificationObservers: @unchecked Sendable {
    private var tokens: [NSObjectProtocol] = []
    private let center: NotificationCenter

    init(names: [Notification.Name], center: NotificationCenter = .default, handler: @escaping @Sendable () -> Void) {
        self.center = center
        tokens = names.map { center.addObserver(forName: $0, object: nil, queue: nil) { _ in handler() } }
    }

    deinit { for t in tokens { center.removeObserver(t) } }
}
