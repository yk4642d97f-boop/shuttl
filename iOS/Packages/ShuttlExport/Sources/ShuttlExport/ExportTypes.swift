import Foundation

/// Settings for a shareable analysis movie.
public struct ExportOptions: Sendable, Equatable {
    public enum Range: Sendable, Equatable {
        case wholeVideo
        case rally(index: Int)
        /// Times are seconds from the beginning of the source video.
        case timeRange(start: Double, end: Double)
    }

    public enum OutputSize: Sendable, Equatable {
        case source
        case p1080
        case p720
    }

    public var range: Range
    public var showSpeed: Bool
    public var showShotType: Bool
    public var showScore: Bool
    public var watermark: Bool
    public var outputSize: OutputSize

    public init(
        range: Range = .wholeVideo,
        showSpeed: Bool = true,
        showShotType: Bool = true,
        showScore: Bool = true,
        watermark: Bool = true,
        outputSize: OutputSize = .source
    ) {
        self.range = range
        self.showSpeed = showSpeed
        self.showShotType = showShotType
        self.showScore = showScore
        self.watermark = watermark
        self.outputSize = outputSize
    }
}

public typealias ExportRange = ExportOptions.Range
public typealias ExportOutputSize = ExportOptions.OutputSize

public enum OverlayExportError: Error, LocalizedError, Sendable {
    case invalidRange(String)
    case noVideoFrames
    case cannotCreateWriter(String)
    case readerFailed(String)
    case writerFailed(String)
    case pixelBufferFailed(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .invalidRange(let message): return "导出范围无效：\(message)"
        case .noVideoFrames: return "视频没有可导出的画面"
        case .cannotCreateWriter(let message): return "无法创建视频导出器：\(message)"
        case .readerFailed(let message): return "读取视频失败：\(message)"
        case .writerFailed(let message): return "写入视频失败：\(message)"
        case .pixelBufferFailed(let message): return "处理画面失败：\(message)"
        case .cancelled: return "导出已取消"
        }
    }
}
