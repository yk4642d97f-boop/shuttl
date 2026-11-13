import Foundation

/// Minimal .npz writer: an uncompressed zip of .npy (format 1.0) arrays, readable by `np.load`.
struct NPZWriter {
    private var entries: [(name: String, data: Data)] = []

    mutating func add(_ name: String, _ v: [Float], _ shape: [Int]) { add(name, "<f4", v, shape) }
    mutating func add(_ name: String, _ v: [Double], _ shape: [Int]) { add(name, "<f8", v, shape) }
    mutating func add(_ name: String, _ v: [Int32], _ shape: [Int]) { add(name, "<i4", v, shape) }
    mutating func add(_ name: String, _ v: [Int64], _ shape: [Int]) { add(name, "<i8", v, shape) }
    mutating func add(_ name: String, _ v: [Int8], _ shape: [Int]) { add(name, "|i1", v, shape) }
    mutating func add(_ name: String, _ v: [Bool], _ shape: [Int]) { add(name, "|b1", v.map { $0 ? UInt8(1) : 0 }, shape) }

    /// Fixed-width unicode array ('<U n', UTF-32LE).
    mutating func add(_ name: String, strings: [String]) {
        let width = max(1, strings.map { $0.unicodeScalars.count }.max() ?? 1)
        var codes: [UInt32] = []
        codes.reserveCapacity(width * strings.count)
        for s in strings {
            let u = s.unicodeScalars.map(\.value)
            codes += u + [UInt32](repeating: 0, count: width - u.count)
        }
        add(name, "<U\(width)", codes, [strings.count], per: width)
    }

    private mutating func add<T>(_ name: String, _ descr: String, _ values: [T], _ shape: [Int], per: Int = 1) {
        precondition(shape.reduce(1, *) * per == values.count, "\(name): shape \(shape) vs \(values.count) values")
        let dims = shape.map(String.init).joined(separator: ", ") + (shape.count == 1 ? "," : "")
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(dims)), }"
        let unpadded = 10 + header.utf8.count + 1
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var d = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        d.append(le(UInt16(header.utf8.count)))
        d.append(header.data(using: .ascii)!)
        values.withUnsafeBytes { d.append(contentsOf: $0) }
        entries.append((name + ".npy", d))
    }

    func write(to url: URL) throws {
        var out = Data(), central = Data()
        for e in entries {
            let crc = CRC32.checksum(e.data), name = Data(e.name.utf8), offset = UInt32(out.count)
            precondition(e.data.count < Int(UInt32.max) && out.count + e.data.count < Int(UInt32.max), "npz over 4 GB")
            var common = Data()
            common.le(UInt16(20), UInt16(0), UInt16(0), UInt16(0), UInt16(0x21))
            common.le(crc, UInt32(e.data.count), UInt32(e.data.count))
            common.le(UInt16(name.count), UInt16(0))
            out.le(UInt32(0x0403_4b50)); out += common; out += name; out += e.data
            central.le(UInt32(0x0201_4b50)); central.le(UInt16(20)); central += common
            central.le(UInt16(0), UInt16(0), UInt16(0)); central.le(UInt32(0), offset); central += name
        }
        let cdOffset = UInt32(out.count)
        out += central
        out.le(UInt32(0x0605_4b50)); out.le(UInt16(0), UInt16(0), UInt16(entries.count), UInt16(entries.count))
        out.le(UInt32(central.count), cdOffset); out.le(UInt16(0))
        try out.write(to: url, options: .atomic)
    }
}

private func le<T: FixedWidthInteger>(_ v: T) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }

private extension Data {
    mutating func le<T: FixedWidthInteger>(_ vs: T...) { for v in vs { Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) } } }
}

enum CRC32 {
    static let table: [UInt32] = (0..<256).map { i in
        (0..<8).reduce(UInt32(i)) { c, _ in c & 1 != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
    }

    static func checksum(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        data.withUnsafeBytes { raw in
            for b in raw.bindMemory(to: UInt8.self) { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        }
        return c ^ 0xFFFF_FFFF
    }
}
