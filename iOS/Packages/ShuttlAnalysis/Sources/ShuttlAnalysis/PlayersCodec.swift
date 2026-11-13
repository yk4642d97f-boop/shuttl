import Foundation
import ShuttlCore

/// Compact, streaming binary form of `[PlayerObservation]`.
///
/// JSONEncoder needs ~8.8 KB of transient RAM per observation (17 keypoints), so a
/// 45-min doubles match (~650k observations) would need ~5.7 GB and be jetsam-killed.
/// This writes through a small buffer with FileHandle; peak extra memory is the buffer
/// plus the decoded array itself.
///
/// Layout, all little-endian:
///   header : "SHPO" · UInt16 version · UInt16 flags · UInt32 fpLen · fp UTF-8 · UInt64 count
///   record : Int64 frame · Int64 trackID · 4×Float64 bbox · Float64 score
///            · UInt8 side (0 unknown, 1 near, 2 far) · UInt8 has (1 foot, 2 keypoints) · UInt16 nKeypoints
///            · [2×Float64 foot] · [nKeypoints × 3×Float64 (x, y, conf)]
///   trailer: "OPHS" · UInt64 count · (version ≥ 2) 32-byte SHA-256 of every byte before it
/// Every value is stored at full precision (Float64 bit patterns, NaN payloads kept), so a
/// resumed run sees exactly the observations the killed run computed: hits / shots compare
/// keypoints and boxes against thresholds, and parity with an uninterrupted run must hold.
public enum PlayersCodec {
    public static let version: UInt16 = 2
    /// Version 1 files (no checksum) are still read by archives; checkpoints require ≥ 2.
    static let checksumVersion: UInt16 = 2
    static let magic: [UInt8] = Array("SHPO".utf8)
    static let trailerMagic: [UInt8] = Array("OPHS".utf8)
    static let flagKeypointsDropped: UInt16 = 1

    public enum CodecError: Error, CustomStringConvertible {
        case notPlayersFile, unsupportedVersion(UInt16), truncated, corrupt(String), checksumMismatch
        public var description: String {
            switch self {
            case .notPlayersFile: return "not a Shuttl players file"
            case .unsupportedVersion(let v): return "players file version \(v) is not supported"
            case .truncated: return "players file is truncated"
            case .corrupt(let why): return "players file is corrupt: \(why)"
            case .checksumMismatch: return "players file checksum does not match its content"
            }
        }
    }

    /// Streams `observations` to `url` (temp file + atomic replace). `includeKeypoints: false`
    /// writes the slim form the library needs (frame, track, side, box, score, foot).
    public static func write(_ observations: [PlayerObservation], to url: URL, fingerprint: String = "",
                             includeKeypoints: Bool = true, chunkBytes: Int = 1 << 20) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        guard fm.createFile(atPath: tmp.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        var committed = false
        defer { if !committed { try? fm.removeItem(at: tmp) } }

        let handle = try FileHandle(forWritingTo: tmp)
        var w = ChunkWriter(handle: handle, capacity: max(chunkBytes, 4096), hashing: true)
        do {
            w.bytes(magic)
            w.put(version)
            w.put(includeKeypoints ? UInt16(0) : flagKeypointsDropped)
            let fp = Array(fingerprint.utf8)
            w.put(UInt32(fp.count))
            w.bytes(fp)
            w.put(UInt64(observations.count))
            for o in observations {
                w.put(Int64(o.frame)); w.put(Int64(o.trackID))
                w.put(o.bbox.x1); w.put(o.bbox.y1); w.put(o.bbox.x2); w.put(o.bbox.y2)
                w.put(o.score)
                w.put(sideCode(o.side))
                let kps = includeKeypoints ? o.keypoints : nil
                w.put(UInt8((o.footWorld != nil ? 1 : 0) | (kps != nil ? 2 : 0)))
                w.put(UInt16(clamping: kps?.count ?? 0))
                if let f = o.footWorld { w.put(f.x); w.put(f.y) }
                if let kps {
                    for k in kps.prefix(Int(UInt16.max)) { w.put(k.x); w.put(k.y); w.put(k.conf) }
                }
                try w.flushIfNeeded()
            }
            w.bytes(trailerMagic)
            w.put(UInt64(observations.count))
            w.bytes(w.digest() ?? [])
            try w.flush()
            try handle.synchronize()
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
        if fm.fileExists(atPath: url.path) {
            _ = try fm.replaceItemAt(url, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: url)
        }
        committed = true
    }

    /// The fingerprint stored in the header (nil if the file is not a players file).
    public static func fingerprint(at url: URL) -> String? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        var r = ChunkReader(handle: h, capacity: 64 * 1024)
        return try? readHeader(&r).fingerprint
    }

    /// Reads a file written by `write`. With `expectedFingerprint`, a file for another
    /// fingerprint returns nil (a stale checkpoint) instead of observations. A checksum, when
    /// present, is always verified; `requireChecksum` also rejects version-1 files without one.
    public static func read(from url: URL, expectedFingerprint: String? = nil, requireChecksum: Bool = false,
                            chunkBytes: Int = 1 << 20) throws -> [PlayerObservation]? {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = (try? handle.seekToEnd()).map(Int.init) ?? 0
        try handle.seek(toOffset: 0)
        var r = ChunkReader(handle: handle, capacity: max(chunkBytes, 4096), hashing: true)
        let header = try readHeader(&r)
        if let expectedFingerprint, header.fingerprint != expectedFingerprint { return nil }
        // Each record is ≥ 60 bytes: never trust a count the file can't hold.
        guard header.count <= UInt64(max(fileSize, 0) / 60) else { throw CodecError.truncated }
        let n = Int(header.count)
        var out: [PlayerObservation] = []
        out.reserveCapacity(n)
        for _ in 0..<n {
            let frame = Int(try r.get(Int64.self)), track = Int(try r.get(Int64.self))
            let bbox = BBox(x1: try r.get(Double.self), y1: try r.get(Double.self),
                            x2: try r.get(Double.self), y2: try r.get(Double.self))
            let score = try r.get(Double.self)
            let side = courtSide(try r.get(UInt8.self))
            let has = try r.get(UInt8.self)
            let nk = Int(try r.get(UInt16.self))
            var foot: SIMD2<Double>?
            if has & 1 != 0 { foot = SIMD2(try r.get(Double.self), try r.get(Double.self)) }
            var kps: [Keypoint]?
            if has & 2 != 0 {
                var k: [Keypoint] = []
                k.reserveCapacity(nk)
                for _ in 0..<nk {
                    k.append(Keypoint(x: try r.get(Double.self), y: try r.get(Double.self), conf: try r.get(Double.self)))
                }
                kps = k
            } else if nk != 0 {
                throw CodecError.corrupt("keypoint count without keypoints")
            }
            out.append(PlayerObservation(frame: frame, trackID: track, bbox: bbox, score: score, side: side,
                                         footWorld: foot, keypoints: kps))
        }
        guard try r.bytes(4) == trailerMagic, try r.get(UInt64.self) == header.count else {
            throw CodecError.truncated
        }
        if header.version >= checksumVersion {
            let computed = r.consumedDigest()
            guard try r.bytes(32) == computed else { throw CodecError.checksumMismatch }
        } else if requireChecksum {
            throw CodecError.checksumMismatch
        }
        return out
    }

    private static func readHeader(_ r: inout ChunkReader) throws -> (fingerprint: String, count: UInt64, version: UInt16) {
        guard (try? r.bytes(4)) == magic else { throw CodecError.notPlayersFile }
        let v = try r.get(UInt16.self)
        guard v == 1 || v == version else { throw CodecError.unsupportedVersion(v) }
        _ = try r.get(UInt16.self)                                   // flags (informational)
        let fpLen = Int(try r.get(UInt32.self))
        guard fpLen <= 1 << 16 else { throw CodecError.corrupt("fingerprint length \(fpLen)") }
        let fp = String(decoding: try r.bytes(fpLen), as: UTF8.self)
        return (fp, try r.get(UInt64.self), v)
    }

    static func sideCode(_ s: CourtSide) -> UInt8 {
        switch s { case .near: return 1; case .far: return 2; case .unknown: return 0 }
    }

    static func courtSide(_ c: UInt8) -> CourtSide { c == 1 ? .near : c == 2 ? .far : .unknown }
}
