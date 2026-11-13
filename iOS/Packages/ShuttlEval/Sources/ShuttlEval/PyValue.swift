import Foundation

/// A value as the Python reference produces it (dict / list / int / float / str / bool / None).
///
/// Two Python properties matter for parity and are kept exactly:
///  • int vs float — `metrics.format_table` prints ints as `k=3` and floats as `k=0.975`;
///  • dict insertion order — it decides table row order and JSON key order.
public indirect enum PyValue: Sendable, Equatable {
    case none
    case bool(Bool)
    case int(Int)
    case float(Double)
    case str(String)
    case list([PyValue])
    case dict(PyDict)

    /// int or float as Double.
    public var double: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .float(let d): return d
        default: return nil
        }
    }
    public var int: Int? { if case .int(let i) = self { return i }; return nil }
    public var string: String? { if case .str(let s) = self { return s }; return nil }
    public var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var list: [PyValue]? { if case .list(let l) = self { return l }; return nil }
    public var dict: PyDict? { if case .dict(let d) = self { return d }; return nil }
    public var isNone: Bool { if case .none = self { return true }; return false }

    public subscript(key: String) -> PyValue? { dict?[key] }
}

extension PyValue: ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, ExpressibleByStringLiteral,
                   ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral, ExpressibleByNilLiteral {
    public init(integerLiteral v: Int) { self = .int(v) }
    public init(floatLiteral v: Double) { self = .float(v) }
    public init(stringLiteral v: String) { self = .str(v) }
    public init(booleanLiteral v: Bool) { self = .bool(v) }
    public init(arrayLiteral elements: PyValue...) { self = .list(elements) }
    public init(nilLiteral: ()) { self = .none }
}

public extension PyValue {
    /// Optional Double → float or None (Python `None if x is None else float(x)`).
    static func optFloat(_ x: Double?) -> PyValue { x.map { .float($0) } ?? .none }
    static func strings(_ xs: [String]) -> PyValue { .list(xs.map { .str($0) }) }
    static func ints(_ xs: [Int]) -> PyValue { .list(xs.map { .int($0) }) }
}

/// An insertion-ordered `dict[str, Any]` with Python assignment semantics
/// (re-assigning an existing key keeps its position; `update` likewise).
public struct PyDict: Sendable, Equatable, Sequence, ExpressibleByDictionaryLiteral {
    public private(set) var keys: [String] = []
    private var storage: [String: PyValue] = [:]

    public init() {}
    public init(dictionaryLiteral elements: (String, PyValue)...) {
        for (k, v) in elements { self[k] = v }
    }
    public init(_ pairs: [(String, PyValue)]) {
        for (k, v) in pairs { self[k] = v }
    }

    public subscript(key: String) -> PyValue? {
        get { storage[key] }
        set {
            guard let v = newValue else { removeValue(forKey: key); return }
            if storage.updateValue(v, forKey: key) == nil { keys.append(key) }
        }
    }

    @discardableResult
    public mutating func removeValue(forKey key: String) -> PyValue? {
        guard let v = storage.removeValue(forKey: key) else { return nil }
        keys.removeAll { $0 == key }
        return v
    }

    /// `dict.update(other)`.
    public mutating func update(_ other: PyDict) { for (k, v) in other { self[k] = v } }

    public func contains(_ key: String) -> Bool { storage[key] != nil }
    public var count: Int { keys.count }
    public var isEmpty: Bool { keys.isEmpty }
    public var values: [PyValue] { keys.map { storage[$0]! } }

    public func makeIterator() -> IndexingIterator<[(key: String, value: PyValue)]> {
        keys.map { (key: $0, value: storage[$0]!) }.makeIterator()
    }

    public static func == (a: PyDict, b: PyDict) -> Bool { a.keys == b.keys && a.storage == b.storage }
}

// MARK: - Python-compatible JSON

public enum PyJSONError: Error, CustomStringConvertible {
    case syntax(String, offset: Int)
    public var description: String {
        switch self { case .syntax(let m, let o): return "JSON syntax error at byte \(o): \(m)" }
    }
}

public extension PyValue {
    /// `json.dumps(value, indent=indent)` with Python's defaults (ensure_ascii, `", "`/`": "`
    /// separators without indent, `","` with indent). Non-finite floats become `null`
    /// (benchmark.py's `_jsonable` converts them to None before dumping).
    func pythonJSON(indent: Int? = nil) -> String {
        var out = ""
        write(to: &out, indent: indent, level: 0)
        return out
    }

    private func write(to out: inout String, indent: Int?, level: Int) {
        switch self {
        case .none: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .int(let i): out += String(i)
        case .float(let d): out += d.isFinite ? PyFormat.repr(d) : "null"
        case .str(let s): PyJSON.writeString(s, to: &out)
        case .list(let items):
            guard !items.isEmpty else { out += "[]"; return }
            out += "["
            for (k, v) in items.enumerated() {
                if k > 0 { out += indent == nil ? ", " : "," }
                PyJSON.newline(&out, indent, level + 1)
                v.write(to: &out, indent: indent, level: level + 1)
            }
            PyJSON.newline(&out, indent, level)
            out += "]"
        case .dict(let d):
            guard !d.isEmpty else { out += "{}"; return }
            out += "{"
            for (k, (key, v)) in d.enumerated() {
                if k > 0 { out += indent == nil ? ", " : "," }
                PyJSON.newline(&out, indent, level + 1)
                PyJSON.writeString(key, to: &out)
                out += ": "
                v.write(to: &out, indent: indent, level: level + 1)
            }
            PyJSON.newline(&out, indent, level)
            out += "}"
        }
    }
}

public enum PyJSON {
    static func newline(_ out: inout String, _ indent: Int?, _ level: Int) {
        guard let indent else { return }
        out += "\n" + String(repeating: " ", count: indent * level)
    }

    /// Python's `ensure_ascii` string escaping.
    static func writeString(_ s: String, to out: inout String) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if u.value >= 0x20 && u.value <= 0x7E {
                    out.unicodeScalars.append(u)
                } else if u.value > 0xFFFF {
                    let v = u.value - 0x10000
                    out += String(format: "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
                } else {
                    out += String(format: "\\u%04x", u.value)
                }
            }
        }
        out += "\""
    }

    /// Parse JSON keeping dict order and the int/float distinction of each number literal
    /// (a literal with `.`, `e` or `E` is a float, like Python's `json.loads`). Accepts
    /// Python's `NaN` / `Infinity` / `-Infinity` tokens.
    public static func parse(_ data: Data) throws -> PyValue {
        var p = Parser(bytes: [UInt8](data))
        p.skipWS()
        let v = try p.value()
        p.skipWS()
        guard p.i == p.bytes.count else { throw PyJSONError.syntax("trailing data", offset: p.i) }
        return v
    }

    public static func parse(_ text: String) throws -> PyValue { try parse(Data(text.utf8)) }

    public static func load(_ url: URL) throws -> PyValue { try parse(Data(contentsOf: url)) }

    struct Parser {
        let bytes: [UInt8]
        var i = 0

        mutating func skipWS() {
            while i < bytes.count, bytes[i] == 0x20 || bytes[i] == 0x0A || bytes[i] == 0x0D || bytes[i] == 0x09 { i += 1 }
        }

        func fail(_ m: String) -> PyJSONError { .syntax(m, offset: i) }

        mutating func literal(_ s: String) -> Bool {
            let u = Array(s.utf8)
            guard i + u.count <= bytes.count, Array(bytes[i..<(i + u.count)]) == u else { return false }
            i += u.count
            return true
        }

        mutating func value() throws -> PyValue {
            guard i < bytes.count else { throw fail("unexpected end") }
            switch bytes[i] {
            case UInt8(ascii: "{"):
                i += 1
                var d = PyDict()
                skipWS()
                if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .dict(d) }
                while true {
                    skipWS()
                    guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else { throw fail("expected key") }
                    let k = try string()
                    skipWS()
                    guard i < bytes.count, bytes[i] == UInt8(ascii: ":") else { throw fail("expected ':'") }
                    i += 1
                    skipWS()
                    d[k] = try value()
                    skipWS()
                    guard i < bytes.count else { throw fail("unterminated object") }
                    if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                    if bytes[i] == UInt8(ascii: "}") { i += 1; return .dict(d) }
                    throw fail("expected ',' or '}'")
                }
            case UInt8(ascii: "["):
                i += 1
                var a: [PyValue] = []
                skipWS()
                if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .list(a) }
                while true {
                    skipWS()
                    a.append(try value())
                    skipWS()
                    guard i < bytes.count else { throw fail("unterminated array") }
                    if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                    if bytes[i] == UInt8(ascii: "]") { i += 1; return .list(a) }
                    throw fail("expected ',' or ']'")
                }
            case UInt8(ascii: "\""):
                return .str(try string())
            default:
                if literal("null") { return .none }
                if literal("true") { return .bool(true) }
                if literal("false") { return .bool(false) }
                if literal("NaN") { return .float(.nan) }
                if literal("Infinity") { return .float(.infinity) }
                if literal("-Infinity") { return .float(-.infinity) }
                return try number()
            }
        }

        mutating func number() throws -> PyValue {
            let start = i
            var isFloat = false
            while i < bytes.count {
                let c = bytes[i]
                if c == UInt8(ascii: ".") || c == UInt8(ascii: "e") || c == UInt8(ascii: "E") { isFloat = true }
                else if !(c == UInt8(ascii: "-") || c == UInt8(ascii: "+") || (c >= 0x30 && c <= 0x39)) { break }
                i += 1
            }
            guard i > start, let text = String(bytes: bytes[start..<i], encoding: .ascii) else { throw fail("bad number") }
            if !isFloat, let v = Int(text) { return .int(v) }
            guard let d = Double(text) else { throw fail("bad number \(text)") }
            return .float(d)
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= bytes.count, let s = String(bytes: bytes[i..<(i + 4)], encoding: .ascii),
                  let v = UInt32(s, radix: 16) else { throw fail("bad \\u escape") }
            i += 4
            return v
        }

        mutating func string() throws -> String {
            i += 1                                   // opening quote
            var scalars = String.UnicodeScalarView()
            var run = i
            func flush(_ upTo: Int) {
                if upTo > run, let s = String(bytes: bytes[run..<upTo], encoding: .utf8) { scalars.append(contentsOf: s.unicodeScalars) }
            }
            while i < bytes.count {
                let c = bytes[i]
                if c == UInt8(ascii: "\"") {
                    flush(i); i += 1
                    return String(scalars)
                }
                if c == UInt8(ascii: "\\") {
                    flush(i); i += 1
                    guard i < bytes.count else { break }
                    let e = bytes[i]; i += 1
                    switch e {
                    case UInt8(ascii: "n"): scalars.append("\n")
                    case UInt8(ascii: "r"): scalars.append("\r")
                    case UInt8(ascii: "t"): scalars.append("\t")
                    case UInt8(ascii: "b"): scalars.append("\u{08}")
                    case UInt8(ascii: "f"): scalars.append("\u{0C}")
                    case UInt8(ascii: "u"):
                        var v = try hex4()
                        if (0xD800..<0xDC00).contains(v), i + 6 <= bytes.count, bytes[i] == UInt8(ascii: "\\"),
                           bytes[i + 1] == UInt8(ascii: "u") {
                            i += 2
                            let lo = try hex4()
                            v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00)
                        }
                        scalars.append(Unicode.Scalar(v) ?? "\u{FFFD}")
                    default: scalars.append(Unicode.Scalar(e))
                    }
                    run = i
                    continue
                }
                i += 1
            }
            throw fail("unterminated string")
        }
    }
}

// MARK: - Python number formatting

public enum PyFormat {
    /// `repr(float)` — Swift's shortest round-trip `description` uses the same digits and the
    /// same switch to exponent notation (exp < -4 or ≥ 16), e.g. `1e-05`, `1e+16`, `0.1`.
    public static func repr(_ d: Double) -> String {
        if d.isNaN { return "nan" }
        if d.isInfinite { return d > 0 ? "inf" : "-inf" }
        return d.description
    }

    /// f"{v:.{digits}f}" (C printf; both round the exact binary value).
    public static func fixed(_ d: Double, _ digits: Int) -> String {
        if d.isNaN { return "nan" }
        if d.isInfinite { return d > 0 ? "inf" : "-inf" }
        return String(format: "%.\(digits)f", d)
    }

    /// Python `round(x, digits)` for a float: the nearest double to the correctly rounded
    /// decimal string (what CPython's dtoa-based round returns).
    public static func round(_ d: Double, _ digits: Int) -> Double {
        guard d.isFinite else { return d }
        return Double(String(format: "%.\(digits)f", d)) ?? d
    }

    /// f"{s:<width}".
    public static func ljust(_ s: String, _ width: Int) -> String {
        let n = s.unicodeScalars.count
        return n >= width ? s : s + String(repeating: " ", count: width - n)
    }

    /// `str(v)` of a Python value as f-strings print it.
    public static func str(_ v: PyValue) -> String {
        switch v {
        case .none: return "None"
        case .bool(let b): return b ? "True" : "False"
        case .int(let i): return String(i)
        case .float(let d): return repr(d)
        case .str(let s): return s
        case .list, .dict: return v.pythonJSON()        // not needed by the reports
        }
    }
}
