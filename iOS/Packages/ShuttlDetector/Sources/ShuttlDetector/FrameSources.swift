import CoreVideo
import Foundation
import ShuttlCore

/// A re-iterable frame source (detector.py ``SteppedSource``): `forEachFrame(step:)` calls
/// `body(index, pixels)` for every `step`-th frame, indices consecutive from 0 when step == 1.
/// The pixel view is only valid inside `body`. Return false from `body` to stop early.
public protocol DetectorFrameSource {
    func forEachFrame(step: Int, _ body: (Int, PixelView) throws -> Bool) throws
}

/// Streams upright frames of a `VideoSource` (never holds the video in memory).
public struct VideoFrameSource: DetectorFrameSource {
    public let video: VideoSource
    /// Frames `start..<stop` (nil = all). Indices stay absolute.
    public let stop: Int?
    public let start: Int

    public init(_ video: VideoSource, start: Int = 0, stop: Int? = nil) {
        self.video = video; self.start = start; self.stop = stop
    }

    public func forEachFrame(step: Int, _ body: (Int, PixelView) throws -> Bool) throws {
        let seq = try video.frames(start: start, stop: stop, step: step)
        while let (i, pb) = seq.next() {
            let go = try PixelView.withBGRA(pb) { try body(i, $0) }
            if !go { return }
        }
        if let e = seq.error { throw e }
    }
}

/// Frames `start...` of another source, indices unchanged (resume after a checkpoint).
/// A `VideoFrameSource` seeks; any other source is iterated and the earlier frames skipped.
struct StartingAt: DetectorFrameSource {
    let base: DetectorFrameSource
    let start: Int

    func forEachFrame(step: Int, _ body: (Int, PixelView) throws -> Bool) throws {
        if let v = base as? VideoFrameSource, v.start == 0 {
            try VideoFrameSource(v.video, start: start, stop: v.stop).forEachFrame(step: step, body)
            return
        }
        try base.forEachFrame(step: step) { i, view in i < start ? true : try body(i, view) }
    }
}

/// Frames generated on demand (RGB `ImageU8`), e.g. synthetic test scenes. Only the requested
/// (every `step`-th) frames are generated.
public struct GeneratedFrameSource: DetectorFrameSource {
    public let count: Int
    public let make: (Int) -> ImageU8

    public init(count: Int, make: @escaping (Int) -> ImageU8) { self.count = count; self.make = make }

    public func forEachFrame(step: Int, _ body: (Int, PixelView) throws -> Bool) throws {
        var i = 0
        while i < count {
            let img = make(i)
            let go = try PixelView.withRGB(img) { try body(i, $0) }
            if !go { return }
            i += max(1, step)
        }
    }
}

/// Frames held in memory (array_source).
public struct ArrayFrameSource: DetectorFrameSource {
    public let frames: [ImageU8]
    public init(_ frames: [ImageU8]) { self.frames = frames }

    public func forEachFrame(step: Int, _ body: (Int, PixelView) throws -> Bool) throws {
        for i in stride(from: 0, to: frames.count, by: max(1, step)) {
            if try !PixelView.withRGB(frames[i], { try body(i, $0) }) { return }
        }
    }
}

/// Progress callback, fraction in 0...1.
public typealias ProgressHandler = @Sendable (Double) -> Void

/// Rate-limit a progress callback (≥0.5 % change or ≥0.25 s apart) — detector.py ``_Throttle``.
/// Used from one thread at a time.
final class ProgressThrottle: @unchecked Sendable {
    private let fn: ProgressHandler?
    private var lastV = -1.0, lastT = 0.0

    init(_ fn: ProgressHandler?) { self.fn = fn }

    func callAsFunction(_ v: Double) {
        guard let fn else { return }
        let now = CFAbsoluteTimeGetCurrent()
        if v - lastV >= 0.005 || now - lastT >= 0.25 || v >= 1.0 {
            lastV = v; lastT = now
            fn(v)
        }
    }
}
