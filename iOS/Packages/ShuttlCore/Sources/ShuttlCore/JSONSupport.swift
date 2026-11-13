import Foundation

/// Fixture / result JSON conventions shared by every Shuttl package:
///  • keys are the Python field names (snake_case), via explicit CodingKeys;
///  • non-finite floats (NaN/±inf) are written as `null` and read back as NaN;
///  • fixtures exported from Python use `json.dump(..., allow_nan=False)` after
///    converting NaN → None, so Swift and Python agree byte-for-byte on structure.
public enum ShuttlJSON {
    public static func decoder() -> JSONDecoder { JSONDecoder() }

    public static func encoder(pretty: Bool = false) -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        return e
    }

    public static func load<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        try decoder().decode(T.self, from: Data(contentsOf: url))
    }
}

// MARK: - NaN ⇄ null

public extension KeyedDecodingContainer {
    /// Decodes a Double where JSON `null` means NaN. Returns nil only if the key is absent.
    func decodeNaNable(forKey key: Key) throws -> Double? {
        guard contains(key) else { return nil }
        if try decodeNil(forKey: key) { return .nan }
        return try decode(Double.self, forKey: key)
    }

    /// `[Double]` where `null` entries mean NaN.
    func decodeNaNableArray(forKey key: Key) throws -> [Double]? {
        guard contains(key), try !decodeNil(forKey: key) else { return nil }
        return try decode([Double?].self, forKey: key).map { $0 ?? .nan }
    }

    /// `[[Double]]` (e.g. an (N,2) array) where `null` entries mean NaN.
    func decodeNaNableRows(forKey key: Key) throws -> [[Double]]? {
        guard contains(key), try !decodeNil(forKey: key) else { return nil }
        return try decode([[Double?]].self, forKey: key).map { $0.map { $0 ?? .nan } }
    }
}

public extension KeyedEncodingContainer {
    mutating func encodeNaNable(_ x: Double, forKey key: Key) throws {
        if x.isFinite { try encode(x, forKey: key) } else { try encodeNil(forKey: key) }
    }

    mutating func encodeNaNableArray(_ xs: [Double], forKey key: Key) throws {
        try encode(xs.map { $0.isFinite ? $0 : nil }, forKey: key)
    }

    mutating func encodeNaNableRows(_ rows: [[Double]], forKey key: Key) throws {
        try encode(rows.map { $0.map { $0.isFinite ? $0 : nil } }, forKey: key)
    }
}

// MARK: - Lenient string enums

/// String enums decode unknown raw values as `.unknown` rather than failing,
/// so a new label on one side never breaks the other.
public protocol LenientStringEnum: RawRepresentable, Codable, CaseIterable, Sendable where RawValue == String {
    static var unknownCase: Self { get }
}

public extension LenientStringEnum {
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? Self.unknownCase
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(rawValue)
    }
}

// MARK: - Free-form JSON

/// Arbitrary JSON (diagnostics, score events) without a fixed schema.
public enum JSONValue: Sendable, Equatable, Codable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n): if n.isFinite { try c.encode(n) } else { try c.encodeNil() }
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    public var double: Double? { if case .number(let n) = self { return n }; return nil }
    public var string: String? { if case .string(let s) = self { return s }; return nil }
    public var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var array: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
}
