import Foundation
import ShuttlCore

/// Progress of a resumable players pass (the stage's own resume state, opaque here): the
/// encoded candidates of every sampled frame below `nextFrame`, and the stage's context
/// (camera, step, crop rects …) that they are only valid for.
public struct PlayersCheckpoint: Sendable, Equatable {
    /// Frames `< nextFrame` are final.
    public var nextFrame: Int
    /// The players stage's resume context; a different one means "don't resume".
    public var context: String
    /// One opaque record per sampled frame, in order.
    public var frames: [Data]

    public init(nextFrame: Int, context: String, frames: [Data]) {
        self.nextFrame = nextFrame; self.context = context; self.frames = frames
    }
}

/// Load / save of players_partial.bin as the device log records it: the same verdicts
/// as `DetectPartialReport` (load: ok · missing · unreadable · fingerprint_mismatch · empty;
/// save: created · appended · unchanged · failed).
public typealias PlayersPartialReport = DetectPartialReport

/// Where a resumable players pass keeps its progress. `FileCheckpointStore` conforms.
public protocol PlayersPartialStore: Sendable {
    func loadPlayersPartialReporting() -> (PlayersCheckpoint?, PlayersPartialReport)
    /// Appends only the frames the file doesn't hold yet when it holds a prefix of `checkpoint`.
    func savePlayersPartialReporting(_ checkpoint: PlayersCheckpoint) -> PlayersPartialReport
    func clearPlayersPartial()
    /// Cheap: whether a partial file exists at all (no read).
    var hasPlayersPartial: Bool { get }
}

extension PlayersPartialStore {
    public func loadPlayersPartial() -> PlayersCheckpoint? { loadPlayersPartialReporting().0 }
    public func savePlayersPartial(_ c: PlayersCheckpoint) { _ = savePlayersPartialReporting(c) }
}

extension FileCheckpointStore: PlayersPartialStore {
    public static let playersPartialKey = "players_partial"

    public var playersPartialURL: URL { directory.appendingPathComponent("\(Self.playersPartialKey).bin") }

    private var playersPartialBytes: Int? {
        (try? FileManager.default.attributesOfItem(atPath: playersPartialURL.path)[.size] as? NSNumber)?.intValue
    }

    public var hasPlayersPartial: Bool { FileManager.default.fileExists(atPath: playersPartialURL.path) }

    public func loadPlayersPartialReporting() -> (PlayersCheckpoint?, PlayersPartialReport) {
        let url = playersPartialURL
        guard let bytes = playersPartialBytes else { return (nil, PlayersPartialReport(verdict: "missing")) }
        guard let p = PlayersPartialFile.parse(url) else {
            PlayersPartialFile.cache.forget(url)
            return (nil, PlayersPartialReport(verdict: "unreadable", bytes: bytes))
        }
        PlayersPartialFile.cache.remember(url, p.summary(fileBytes: bytes))
        guard p.fingerprint == fingerprint else {
            return (nil, PlayersPartialReport(verdict: "fingerprint_mismatch", nextFrame: p.hasDelta ? p.nextFrame : nil,
                                              bytes: bytes, error: "file \(p.fingerprint.prefix(12)) ≠ run \(fingerprint.prefix(12))"))
        }
        guard p.hasDelta else { return (nil, PlayersPartialReport(verdict: "empty", bytes: bytes)) }
        return (PlayersCheckpoint(nextFrame: p.nextFrame, context: p.context, frames: p.frames),
                PlayersPartialReport(verdict: "ok", nextFrame: p.nextFrame, bytes: bytes))
    }

    /// Frames below `nextFrame` are final, so a file holding a prefix of `checkpoint` (same
    /// fingerprint and context) only gets the new frames appended. Which prefix the file holds
    /// comes from what this process last wrote or read there (size + last record checksum
    /// re-checked on disk), so a save costs the new frames, not the whole file; without that
    /// the file is parsed once. Anything else is rewritten atomically. Never throws.
    public func savePlayersPartialReporting(_ c: PlayersCheckpoint) -> PlayersPartialReport {
        let url = playersPartialURL
        do {
            let held = PlayersPartialFile.cache.summary(url, fileBytes: playersPartialBytes)
                ?? PlayersPartialFile.parse(url).map { $0.summary(fileBytes: playersPartialBytes ?? 0) }
            if let p = held, p.fingerprint == fingerprint, p.context == c.context,
               p.frameCount <= c.frames.count, p.nextFrame <= c.nextFrame,
               p.frameCount == 0 || (PlayersPartialFile.fnv1a(c.frames[0]) == p.firstFrameSum
                                     && PlayersPartialFile.fnv1a(c.frames[p.frameCount - 1]) == p.lastFrameSum) {
                guard c.frames.count > p.frameCount || c.nextFrame != p.nextFrame else {
                    return PlayersPartialReport(verdict: "unchanged", nextFrame: c.nextFrame, bytes: p.validLength)
                }
                let tail = PlayersPartialFile.deltaRecord(start: p.frameCount, frames: c.frames[p.frameCount...],
                                                          nextFrame: c.nextFrame)
                let h = try FileHandle(forWritingTo: url)
                defer { try? h.close() }
                try h.truncate(atOffset: UInt64(p.validLength))           // drop a torn tail first
                try h.seek(toOffset: UInt64(p.validLength))
                try h.write(contentsOf: tail)
                let total = p.validLength + tail.count
                PlayersPartialFile.cache.remember(url, .init(fingerprint: fingerprint, context: c.context,
                                                             frameCount: c.frames.count, nextFrame: c.nextFrame,
                                                             firstFrameSum: c.frames.first.map(PlayersPartialFile.fnv1a) ?? 0,
                                                             lastFrameSum: c.frames.last.map(PlayersPartialFile.fnv1a) ?? 0,
                                                             validLength: total, fileBytes: total))
                return PlayersPartialReport(verdict: "appended", nextFrame: c.nextFrame, bytes: total)
            }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let all = PlayersPartialFile.header(fingerprint: fingerprint, context: c.context)
                + PlayersPartialFile.deltaRecord(start: 0, frames: c.frames[...], nextFrame: c.nextFrame)
            try Data(all).write(to: url, options: .atomic)
            PlayersPartialFile.cache.remember(url, .init(fingerprint: fingerprint, context: c.context,
                                                         frameCount: c.frames.count, nextFrame: c.nextFrame,
                                                         firstFrameSum: c.frames.first.map(PlayersPartialFile.fnv1a) ?? 0,
                                                         lastFrameSum: c.frames.last.map(PlayersPartialFile.fnv1a) ?? 0,
                                                         validLength: all.count, fileBytes: all.count))
            return PlayersPartialReport(verdict: "created", nextFrame: c.nextFrame, bytes: all.count)
        } catch {
            PlayersPartialFile.cache.forget(url)
            return PlayersPartialReport(verdict: "failed", nextFrame: c.nextFrame, bytes: playersPartialBytes,
                                        error: String(describing: error))
        }
    }

    public func clearPlayersPartial() {
        PlayersPartialFile.cache.forget(playersPartialURL)
        try? FileManager.default.removeItem(at: playersPartialURL)
    }
}

/// "SHPP" · UInt16 version · UInt32 len · fingerprint · UInt32 len · context, then delta records:
/// UInt8 kind (2) · UInt32 payload length · UInt64 FNV-1a of payload · payload =
/// Int64 start · Int64 count · Int64 nextFrame · count × (UInt32 n · n bytes).
/// Parsing stops at the first torn, corrupt or out-of-order record and keeps everything before it.
enum PlayersPartialFile {
    static let magic: [UInt8] = Array("SHPP".utf8)
    static let version: UInt16 = 1

    struct Parsed {
        var fingerprint: String
        var context: String
        var nextFrame = 0
        var frames: [Data] = []
        var hasDelta = false
        var validLength: Int

        func summary(fileBytes: Int) -> Summary {
            Summary(fingerprint: fingerprint, context: context, frameCount: frames.count, nextFrame: nextFrame,
                    firstFrameSum: frames.first.map(fnv1a) ?? 0, lastFrameSum: frames.last.map(fnv1a) ?? 0,
                    validLength: validLength, fileBytes: fileBytes)
        }
    }

    /// What a file holds, without its frames.
    struct Summary: Sendable, Equatable {
        var fingerprint: String
        var context: String
        var frameCount: Int
        var nextFrame: Int
        /// Checksums of the first and last frame held: a save only appends to a file whose ends
        /// match the checkpoint (frames below nextFrame are final, so the middle can't differ in
        /// a real run; anything else is rewritten).
        var firstFrameSum: UInt64
        var lastFrameSum: UInt64
        /// Bytes up to the end of the last accepted record.
        var validLength: Int
        /// File size when this was taken (a file changed by anyone else is re-parsed).
        var fileBytes: Int
    }

    /// The last summary per file this process wrote or read.
    final class Cache: @unchecked Sendable {
        private let lock = NSLock()
        private var byPath: [String: Summary] = [:]
        func remember(_ url: URL, _ s: Summary) { lock.withLock { byPath[url.path] = s } }
        func forget(_ url: URL) { lock.withLock { _ = byPath.removeValue(forKey: url.path) } }
        /// The cached summary if the file still has the size it had then.
        func summary(_ url: URL, fileBytes: Int?) -> Summary? {
            lock.withLock {
                guard let s = byPath[url.path], let fileBytes, s.fileBytes == fileBytes else { return nil }
                return s
            }
        }
    }

    static let cache = Cache()

    static func header(fingerprint: String, context: String) -> [UInt8] {
        var b = magic
        version.appendLittleEndian(to: &b)
        for s in [fingerprint, context] {
            let u = Array(s.utf8)
            UInt32(u.count).appendLittleEndian(to: &b)
            b += u
        }
        return b
    }

    static func deltaRecord(start: Int, frames: ArraySlice<Data>, nextFrame: Int) -> [UInt8] {
        var p: [UInt8] = []
        p.reserveCapacity(24 + frames.reduce(0) { $0 + 4 + $1.count })
        Int64(start).appendLittleEndian(to: &p)
        Int64(frames.count).appendLittleEndian(to: &p)
        Int64(nextFrame).appendLittleEndian(to: &p)
        for f in frames {
            UInt32(f.count).appendLittleEndian(to: &p)
            p += f
        }
        var b: [UInt8] = [2]
        UInt32(p.count).appendLittleEndian(to: &b)
        fnv1a(p[...]).appendLittleEndian(to: &b)
        return b + p
    }

    static func fnv1a(_ d: Data) -> UInt64 { d.withUnsafeBytes { fnv1a($0) } }
    static func fnv1a(_ bytes: ArraySlice<UInt8>) -> UInt64 { bytes.withUnsafeBytes { fnv1a($0) } }
    static func fnv1a(_ raw: UnsafeRawBufferPointer) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for x in raw { h ^= UInt64(x); h &*= 0x0000_0100_0000_01b3 }
        return h
    }

    /// nil when the file is missing or its header is unreadable.
    static func parse(_ url: URL) -> Parsed? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        var r = ByteCursor(bytes: [UInt8](data))
        guard r.take(4).map(Array.init) == magic, r.get(UInt16.self) == version,
              let fpLen = r.get(UInt32.self), let fp = r.take(Int(fpLen)),
              let cLen = r.get(UInt32.self), let ctx = r.take(Int(cLen)) else { return nil }
        var p = Parsed(fingerprint: String(decoding: fp, as: UTF8.self), context: String(decoding: ctx, as: UTF8.self),
                       validLength: r.pos)
        while let kind = r.get(UInt8.self), kind == 2, let len = r.get(UInt32.self), let sum = r.get(UInt64.self),
              let payload = r.take(Int(len)), fnv1a(payload) == sum {
            var q = ByteCursor(bytes: Array(payload))
            guard let start = q.get(Int64.self), start == p.frames.count, let count = q.get(Int64.self),
                  count >= 0, count <= Int64(payload.count / 4), let next = q.get(Int64.self),
                  next >= p.nextFrame else { break }
            var frames: [Data] = []
            frames.reserveCapacity(Int(count))
            var ok = true
            for _ in 0..<count {
                guard let n = q.get(UInt32.self), let f = q.take(Int(n)) else { ok = false; break }
                frames.append(Data(f))
            }
            guard ok, q.pos == payload.count else { break }
            p.frames += frames
            p.nextFrame = Int(next)
            p.hasDelta = true
            p.validLength = r.pos
        }
        return p
    }
}
