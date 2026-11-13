import CryptoKit
import Foundation

/// Fixed-width little-endian values the binary codecs read and write.
public protocol BinaryFixedWidth {
    static var byteWidth: Int { get }
    func appendLittleEndian(to buffer: inout [UInt8])
    init(littleEndianBytes p: UnsafeRawBufferPointer)
}

extension BinaryFixedWidth where Self: FixedWidthInteger {
    public static var byteWidth: Int { MemoryLayout<Self>.size }
    public func appendLittleEndian(to buffer: inout [UInt8]) {
        withUnsafeBytes(of: littleEndian) { buffer.append(contentsOf: $0) }
    }
    public init(littleEndianBytes p: UnsafeRawBufferPointer) {
        self.init(littleEndian: p.loadUnaligned(as: Self.self))
    }
}

extension UInt8: BinaryFixedWidth {}
extension UInt16: BinaryFixedWidth {}
extension UInt32: BinaryFixedWidth {}
extension UInt64: BinaryFixedWidth {}
extension Int32: BinaryFixedWidth {}
extension Int64: BinaryFixedWidth {}

extension Double: BinaryFixedWidth {
    public static var byteWidth: Int { 8 }
    /// Bit pattern, so NaN payloads and -0 survive.
    public func appendLittleEndian(to buffer: inout [UInt8]) { bitPattern.appendLittleEndian(to: &buffer) }
    public init(littleEndianBytes p: UnsafeRawBufferPointer) { self.init(bitPattern: UInt64(littleEndianBytes: p)) }
}

extension Float: BinaryFixedWidth {
    public static var byteWidth: Int { 4 }
    public func appendLittleEndian(to buffer: inout [UInt8]) { bitPattern.appendLittleEndian(to: &buffer) }
    public init(littleEndianBytes p: UnsafeRawBufferPointer) { self.init(bitPattern: UInt32(littleEndianBytes: p)) }
}

/// Buffered writer: values accumulate in a fixed-size buffer that is flushed to the
/// handle when full, so memory stays at `capacity` however much is written.
/// With `hashing`, `digest()` is the SHA-256 of everything put so far (flushed or not).
public struct ChunkWriter {
    private let handle: FileHandle
    private let capacity: Int
    private var buffer: [UInt8]
    private var hasher: SHA256?
    public private(set) var bytesWritten = 0

    public init(handle: FileHandle, capacity: Int = 1 << 20, hashing: Bool = false) {
        self.handle = handle
        self.capacity = capacity
        buffer = []
        buffer.reserveCapacity(capacity + 4096)
        hasher = hashing ? SHA256() : nil
    }

    public func digest() -> [UInt8]? {
        guard var h = hasher else { return nil }
        h.update(data: buffer)
        return Array(h.finalize())
    }

    public mutating func put<T: BinaryFixedWidth>(_ v: T) { v.appendLittleEndian(to: &buffer) }
    public mutating func bytes(_ b: [UInt8]) { buffer.append(contentsOf: b) }

    public mutating func flushIfNeeded() throws { if buffer.count >= capacity { try flush() } }

    public mutating func flush() throws {
        guard !buffer.isEmpty else { return }
        try handle.write(contentsOf: buffer)
        hasher?.update(data: buffer)
        bytesWritten += buffer.count
        buffer.removeAll(keepingCapacity: true)
    }
}

/// Buffered reader over a FileHandle; reads `capacity` bytes at a time.
/// With `hashing`, `consumedDigest()` is the SHA-256 of every byte returned so far.
public struct ChunkReader {
    private let handle: FileHandle
    private let capacity: Int
    private var buffer: [UInt8] = []
    private var pos = 0
    private var eof = false
    private var hasher: SHA256?
    /// buffer[..<hashedTo] is already in `hasher`.
    private var hashedTo = 0

    public init(handle: FileHandle, capacity: Int = 1 << 20, hashing: Bool = false) {
        self.handle = handle
        self.capacity = capacity
        hasher = hashing ? SHA256() : nil
    }

    public mutating func consumedDigest() -> [UInt8]? {
        hashConsumed()
        guard let h = hasher else { return nil }
        return Array(h.finalize())
    }

    private mutating func hashConsumed() {
        guard hasher != nil, pos > hashedTo else { return }
        hasher!.update(data: buffer[hashedTo..<pos])
        hashedTo = pos
    }

    /// Makes at least `n` bytes available at `pos` (fewer only at end of file).
    private mutating func fill(_ n: Int) throws {
        while buffer.count - pos < n && !eof {
            if pos > 0 { hashConsumed(); buffer.removeFirst(pos); pos = 0; hashedTo = 0 }
            // FileHandle hands back autoreleased buffers: without a pool per chunk a whole-file read
            // keeps every chunk alive until the caller's pool drains (+ file size of footprint).
            let chunk = try autoreleasepool { try handle.read(upToCount: max(capacity, n)) } ?? Data()
            if chunk.isEmpty { eof = true } else { buffer.append(contentsOf: chunk) }
        }
    }

    public mutating func get<T: BinaryFixedWidth>(_ type: T.Type) throws -> T {
        let n = T.byteWidth
        if buffer.count - pos < n { try fill(n) }
        guard buffer.count - pos >= n else { throw PlayersCodec.CodecError.truncated }
        let v = buffer.withUnsafeBytes { T(littleEndianBytes: UnsafeRawBufferPointer(rebasing: $0[pos..<(pos + n)])) }
        pos += n
        return v
    }

    public mutating func bytes(_ n: Int) throws -> [UInt8] {
        if buffer.count - pos < n { try fill(n) }
        guard buffer.count - pos >= n else { throw PlayersCodec.CodecError.truncated }
        defer { pos += n }
        return Array(buffer[pos..<(pos + n)])
    }
}
