import Foundation
import os

/// Unified-logging channels (Console.app / `log stream --predicate 'subsystem == "com.example.shuttl"'`).
/// Dynamic values are private in release logs unless marked `privacy: .public`.
public enum ShuttlLog {
    public static let subsystem = "com.example.shuttl"

    public static let pipeline = Logger(subsystem: subsystem, category: "pipeline")
    public static let analysis = Logger(subsystem: subsystem, category: "analysis")
    public static let camera = Logger(subsystem: subsystem, category: "camera")
    public static let storage = Logger(subsystem: subsystem, category: "storage")
}
