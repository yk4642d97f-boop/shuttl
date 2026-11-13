import Foundation
import ShuttlCore

/// Where a resumable detect pass keeps its progress. `FileCheckpointStore` conforms;
/// other stores simply don't resume detect.
public protocol DetectPartialStore: Sendable {
    /// The saved progress for this store's fingerprint, nil if none / unreadable / stale.
    func loadDetectPartial() -> DetectCheckpoint?
    /// Persists `checkpoint` (appends only what's new when the file already holds its prefix).
    func saveDetectPartial(_ checkpoint: DetectCheckpoint)
    func clearDetectPartial()
    /// `loadDetectPartial` plus why nothing was loaded (for the device log).
    func loadDetectPartialReporting() -> (DetectCheckpoint?, DetectPartialReport)
    /// `saveDetectPartial` plus what happened on disk (for the device log).
    func saveDetectPartialReporting(_ checkpoint: DetectCheckpoint) -> DetectPartialReport
}

extension DetectPartialStore {
    public func loadDetectPartialReporting() -> (DetectCheckpoint?, DetectPartialReport) {
        let c = loadDetectPartial()
        return (c, DetectPartialReport(verdict: c == nil ? "missing" : "ok", nextFrame: c?.nextFrame))
    }

    public func saveDetectPartialReporting(_ checkpoint: DetectCheckpoint) -> DetectPartialReport {
        saveDetectPartial(checkpoint)
        return DetectPartialReport(verdict: "saved", nextFrame: checkpoint.nextFrame)
    }
}

/// One load or save of detect_partial.bin, as the device log records it. A nil load has
/// four different causes, and a save is best effort; without this a phone run can't tell
/// "never saved" from "saved but not readable".
public struct DetectPartialReport: Sendable, Equatable {
    /// load: ok · missing · unreadable · fingerprint_mismatch · empty
    /// save: created · appended · unchanged · failed
    public var verdict: String
    public var nextFrame: Int?
    /// File size after the operation (nil = no file).
    public var bytes: Int?
    public var error: String?

    public init(verdict: String, nextFrame: Int? = nil, bytes: Int? = nil, error: String? = nil) {
        self.verdict = verdict; self.nextFrame = nextFrame; self.bytes = bytes; self.error = error
    }

    public var json: [String: JSONValue] {
        var o: [String: JSONValue] = ["verdict": .string(verdict)]
        if let nextFrame { o["next_frame"] = .number(Double(nextFrame)) }
        if let bytes { o["bytes"] = .number(Double(bytes)) }
        if let error { o["error"] = .string(error) }
        return o
    }
}

extension FileCheckpointStore: DetectPartialStore {
    public static let detectPartialKey = "detect_partial"

    public var detectPartialURL: URL { directory.appendingPathComponent("\(Self.detectPartialKey).bin") }

    private var detectPartialBytes: Int? {
        (try? FileManager.default.attributesOfItem(atPath: detectPartialURL.path)[.size] as? NSNumber)?.intValue
    }

    public func loadDetectPartial() -> DetectCheckpoint? { loadDetectPartialReporting().0 }

    public func loadDetectPartialReporting() -> (DetectCheckpoint?, DetectPartialReport) {
        let bytes = detectPartialBytes
        guard let bytes else { return (nil, DetectPartialReport(verdict: "missing")) }
        guard let p = DetectPartialFile.parse(detectPartialURL) else {
            return (nil, DetectPartialReport(verdict: "unreadable", bytes: bytes))
        }
        guard p.fingerprint == fingerprint else {
            return (nil, DetectPartialReport(verdict: "fingerprint_mismatch", nextFrame: p.hasDelta ? p.nextFrame : nil,
                                             bytes: bytes, error: "file \(p.fingerprint.prefix(12)) ≠ run \(fingerprint.prefix(12))"))
        }
        guard p.hasDelta else { return (nil, DetectPartialReport(verdict: "empty", bytes: bytes)) }
        let c = DetectCheckpoint(nextFrame: p.nextFrame, candidates: p.candidates, background: p.background,
                                 detectorName: p.detectorName)
        return (c, DetectPartialReport(verdict: "ok", nextFrame: p.nextFrame, bytes: bytes))
    }

    public func saveDetectPartial(_ c: DetectCheckpoint) { _ = saveDetectPartialReporting(c) }

    /// Candidates of frames before `nextFrame` are final, so an existing file whose content
    /// is a prefix of `checkpoint` only gets the delta appended (plus the background when it
    /// changed). Anything else — other fingerprint / detector, a restarted pass, a corrupt
    /// file — is rewritten atomically. Never throws: checkpoints are an optimisation.
    public func saveDetectPartialReporting(_ c: DetectCheckpoint) -> DetectPartialReport {
        let url = detectPartialURL
        do {
            if let p = DetectPartialFile.parse(url), p.fingerprint == fingerprint, p.detectorName == c.detectorName,
               p.candidates.count <= c.candidates.count, p.nextFrame <= c.nextFrame,
               c.candidates.prefix(p.candidates.count).elementsEqual(p.candidates) {
                var tail: [UInt8] = []
                let newBackground = c.background != p.background
                if newBackground { tail += DetectPartialFile.backgroundRecord(c.background) }
                if newBackground || c.candidates.count > p.candidates.count || c.nextFrame != p.nextFrame {
                    tail += DetectPartialFile.deltaRecord(start: p.candidates.count,
                                                          frames: c.candidates[p.candidates.count...], nextFrame: c.nextFrame)
                }
                guard !tail.isEmpty else {
                    return DetectPartialReport(verdict: "unchanged", nextFrame: c.nextFrame, bytes: detectPartialBytes)
                }
                let h = try FileHandle(forWritingTo: url)
                defer { try? h.close() }
                try h.truncate(atOffset: UInt64(p.validLength))       // drop a torn tail first
                try h.seek(toOffset: UInt64(p.validLength))
                try h.write(contentsOf: tail)
                return DetectPartialReport(verdict: "appended", nextFrame: c.nextFrame, bytes: p.validLength + tail.count)
            }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            var all = DetectPartialFile.header(fingerprint: fingerprint, detectorName: c.detectorName)
            if c.background != nil { all += DetectPartialFile.backgroundRecord(c.background) }
            all += DetectPartialFile.deltaRecord(start: 0, frames: c.candidates[...], nextFrame: c.nextFrame)
            try Data(all).write(to: url, options: .atomic)
            return DetectPartialReport(verdict: "created", nextFrame: c.nextFrame, bytes: all.count)
        } catch {
            return DetectPartialReport(verdict: "failed", nextFrame: c.nextFrame, bytes: detectPartialBytes,
                                       error: String(describing: error))
        }
    }

    public func clearDetectPartial() { try? FileManager.default.removeItem(at: detectPartialURL) }
}

/// "SHDP" · UInt16 version · UInt32 len · fingerprint · UInt32 len · detector name, then records:
/// UInt8 kind (1 background, 2 delta) · UInt32 payload length · UInt64 FNV-1a of payload · payload.
///   background: UInt8 has · bytes          delta: Int64 start · Int64 count · Int64 nextFrame ·
///   count × (UInt32 n · n × (x, y, score) Float64)
/// A background record takes effect with the next delta; parsing stops at the first torn,
/// corrupt or out-of-order record and keeps everything before it.
enum DetectPartialFile {
    static let magic: [UInt8] = Array("SHDP".utf8)
    static let version: UInt16 = 1

    struct Parsed {
        var fingerprint: String
        var detectorName: String
        var nextFrame = 0
        var candidates: [[ShuttleCandidate]] = []
        var background: Data?
        var hasDelta = false
        /// Bytes up to the end of the last accepted delta.
        var validLength: Int
    }

    static func header(fingerprint: String, detectorName: String) -> [UInt8] {
        var b = magic
        version.appendLittleEndian(to: &b)
        for s in [fingerprint, detectorName] {
            let u = Array(s.utf8)
            UInt32(u.count).appendLittleEndian(to: &b)
            b += u
        }
        return b
    }

    static func backgroundRecord(_ bg: Data?) -> [UInt8] {
        var p: [UInt8] = [bg == nil ? 0 : 1]
        if let bg { p += bg }
        return record(kind: 1, payload: p)
    }

    static func deltaRecord(start: Int, frames: ArraySlice<[ShuttleCandidate]>, nextFrame: Int) -> [UInt8] {
        var p: [UInt8] = []
        p.reserveCapacity(24 + frames.count * 4 + frames.reduce(0) { $0 + $1.count } * 24)
        Int64(start).appendLittleEndian(to: &p)
        Int64(frames.count).appendLittleEndian(to: &p)
        Int64(nextFrame).appendLittleEndian(to: &p)
        for f in frames {
            UInt32(f.count).appendLittleEndian(to: &p)
            for c in f { c.x.appendLittleEndian(to: &p); c.y.appendLittleEndian(to: &p); c.score.appendLittleEndian(to: &p) }
        }
        return record(kind: 2, payload: p)
    }

    private static func record(kind: UInt8, payload: [UInt8]) -> [UInt8] {
        var b: [UInt8] = [kind]
        UInt32(payload.count).appendLittleEndian(to: &b)
        fnv1a(payload[...]).appendLittleEndian(to: &b)
        return b + payload
    }

    static func fnv1a(_ bytes: ArraySlice<UInt8>) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for x in bytes { h ^= UInt64(x); h &*= 0x0000_0100_0000_01b3 }
        return h
    }

    /// nil when the file is missing or its header is unreadable.
    static func parse(_ url: URL) -> Parsed? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        var r = ByteCursor(bytes: [UInt8](data))
        guard r.take(4).map(Array.init) == magic, r.get(UInt16.self) == version,
              let fpLen = r.get(UInt32.self), let fp = r.take(Int(fpLen)),
              let nLen = r.get(UInt32.self), let name = r.take(Int(nLen)) else { return nil }
        var p = Parsed(fingerprint: String(decoding: fp, as: UTF8.self), detectorName: String(decoding: name, as: UTF8.self),
                       validLength: r.pos)
        var pending: Data?? = nil                          // background waiting for its delta
        while let kind = r.get(UInt8.self), let len = r.get(UInt32.self), let sum = r.get(UInt64.self),
              let payload = r.take(Int(len)), fnv1a(payload) == sum {
            var q = ByteCursor(bytes: Array(payload))
            if kind == 1 {
                guard let has = q.get(UInt8.self) else { break }
                pending = .some(has == 0 ? nil : Data(payload.dropFirst()))
            } else if kind == 2 {
                guard let start = q.get(Int64.self), start == p.candidates.count, let count = q.get(Int64.self),
                      count >= 0, count <= Int64(payload.count / 4), let next = q.get(Int64.self),
                      let frames = parseFrames(&q, count: Int(count)), q.pos == payload.count else { break }
                p.candidates += frames
                p.nextFrame = Int(next)
                if let bg = pending { p.background = bg; pending = nil }
                p.hasDelta = true
                p.validLength = r.pos
            } else {
                break
            }
        }
        return p
    }

    private static func parseFrames(_ q: inout ByteCursor, count: Int) -> [[ShuttleCandidate]]? {
        var out: [[ShuttleCandidate]] = []
        out.reserveCapacity(count)
        for _ in 0..<count {
            guard let n = q.get(UInt32.self), Int(n) <= (q.bytes.count - q.pos) / 24 else { return nil }
            var f: [ShuttleCandidate] = []
            f.reserveCapacity(Int(n))
            for _ in 0..<n {
                guard let x = q.get(Double.self), let y = q.get(Double.self), let s = q.get(Double.self) else { return nil }
                f.append(ShuttleCandidate(x: x, y: y, score: s))
            }
            out.append(f)
        }
        return out
    }
}

/// Bounds-checked little-endian reads over a byte array.
struct ByteCursor {
    let bytes: [UInt8]
    var pos = 0

    mutating func get<T: BinaryFixedWidth>(_: T.Type) -> T? {
        let n = T.byteWidth
        guard pos + n <= bytes.count else { return nil }
        let v = bytes.withUnsafeBytes { T(littleEndianBytes: UnsafeRawBufferPointer(rebasing: $0[pos..<(pos + n)])) }
        pos += n
        return v
    }

    mutating func take(_ n: Int) -> ArraySlice<UInt8>? {
        guard n >= 0, pos + n <= bytes.count else { return nil }
        defer { pos += n }
        return bytes[pos..<(pos + n)]
    }
}
