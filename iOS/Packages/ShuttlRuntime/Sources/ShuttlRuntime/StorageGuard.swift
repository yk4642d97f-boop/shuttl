import Foundation

public struct StorageStatus: Sendable, Equatable {
    public enum Level: Int, Sendable, Comparable {
        case ok, low, refuseStart, stopRecording
        public static func < (a: Level, b: Level) -> Bool { a.rawValue < b.rawValue }
    }

    public var level: Level
    public var freeBytes: Int64
    /// Minutes of recording left at `bytesPerMinute` before the stop floor is reached.
    public var minutesLeft: Double

    /// A new recording may start.
    public var canStartRecording: Bool { level < .refuseStart }
    /// A running recording must stop now (and be saved).
    public var mustStopRecording: Bool { level == .stopRecording }

    /// Chinese text for the UI (nil when there's plenty of space).
    public var message: String? {
        let gb = String(format: "%.1f", Double(freeBytes) / 1_000_000_000)
        switch level {
        case .ok: return nil
        case .low: return "手机空间只够再录大约 \(Int(minutesLeft.rounded(.down))) 分钟，建议先清理一些照片或视频。"
        case .refuseStart: return "手机空间不够了（只剩 \(gb) GB），先清理一些照片或视频再录。"
        case .stopRecording: return "手机空间快满了，已自动停止录像并保存。"
        }
    }
}

/// Free-space policy for recording 1080p60 (and the analysis files that follow).
/// warn < 10 min · refuse to start < 3 min or < 1.5 GB · stop recording < 1 GB.
/// Sizes are decimal (1 GB = 10⁹ bytes), like the Settings app shows.
public struct StorageGuard: Sendable {
    public var bytesPerMinute: Int64
    public var warnMinutes: Double
    public var refuseMinutes: Double
    public var refuseBytes: Int64
    public var stopBytes: Int64
    private let freeSpace: @Sendable () throws -> Int64

    /// - volume: any URL on the volume to watch (default: Documents).
    /// - bytesPerMinute: default 90 MB/min ≈ HEVC 1080p60 as the camera writes it.
    public init(volume: URL? = nil, bytesPerMinute: Int64 = 90_000_000, warnMinutes: Double = 10,
                refuseMinutes: Double = 3, refuseBytes: Int64 = 1_500_000_000, stopBytes: Int64 = 1_000_000_000) {
        let url = volume ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.init(bytesPerMinute: bytesPerMinute, warnMinutes: warnMinutes, refuseMinutes: refuseMinutes,
                  refuseBytes: refuseBytes, stopBytes: stopBytes, freeSpace: { try StorageGuard.freeBytes(on: url) })
    }

    /// Injectable free-space reading (tests).
    public init(bytesPerMinute: Int64 = 90_000_000, warnMinutes: Double = 10, refuseMinutes: Double = 3,
                refuseBytes: Int64 = 1_500_000_000, stopBytes: Int64 = 1_000_000_000,
                freeSpace: @escaping @Sendable () throws -> Int64) {
        self.bytesPerMinute = max(1, bytesPerMinute)
        self.warnMinutes = warnMinutes
        self.refuseMinutes = refuseMinutes
        self.refuseBytes = refuseBytes
        self.stopBytes = stopBytes
        self.freeSpace = freeSpace
    }

    /// Reads free space now. A failed reading is treated as "plenty" (never block
    /// recording because the OS didn't answer) — `nil` tells the caller it happened.
    public func check() -> StorageStatus? {
        guard let free = try? freeSpace() else { return nil }
        return evaluate(freeBytes: free)
    }

    /// `minutesLeft` counts only space above the stop floor: minutes the user can still
    /// record before `.stopRecording` would end the take (with plain free / rate, the
    /// 10-min warning = 0.9 GB would sit below the 1 GB stop floor and never show).
    public func evaluate(freeBytes free: Int64) -> StorageStatus {
        let minutes = Double(max(free - stopBytes, 0)) / Double(bytesPerMinute)
        let level: StorageStatus.Level
        if free < stopBytes { level = .stopRecording }
        else if free < refuseBytes || minutes < refuseMinutes { level = .refuseStart }
        else if minutes < warnMinutes { level = .low }
        else { level = .ok }
        return StorageStatus(level: level, freeBytes: free, minutesLeft: minutes)
    }

    /// volumeAvailableCapacityForImportantUsage (what iOS will free up for the user's
    /// own content), falling back to the plain available capacity.
    public static func freeBytes(on url: URL) throws -> Int64 {
        let v = try url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        if let important = v.volumeAvailableCapacityForImportantUsage, important > 0 { return important }
        if let plain = v.volumeAvailableCapacity { return Int64(plain) }
        throw CocoaError(.fileReadUnknown)
    }
}
