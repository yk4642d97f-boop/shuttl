import Foundation
import XCTest
import ShuttlCore
import ShuttlDetector

/// Loads a JSON fixture exported by backend/scripts/swift_fixtures/detector.py.
func fixture(_ name: String) throws -> JSONValue {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"),
                            "missing fixture \(name).json — run `.venv/bin/python -m scripts.swift_fixtures.detector` in backend/")
    return try ShuttlJSON.load(JSONValue.self, from: url)
}

/// Repository root (…/华北五省), derived from this file's location.
let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

let packageRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

extension JSONValue {
    var doubles: [Double] { (array ?? []).map { $0.double ?? .nan } }
    var ints: [Int] { (array ?? []).map { Int($0.double ?? -1) } }
    var int: Int { Int(double ?? -1) }
    var rows: [[Double]] { (array ?? []).map(\.doubles) }
}

func assertClose(_ a: Double, _ b: Double, tol: Double, _ msg: @autoclosure () -> String = "",
                 file: StaticString = #filePath, line: UInt = #line) {
    if !b.isFinite { XCTAssertFalse(a.isFinite, "\(a) vs non-finite \(msg())", file: file, line: line); return }
    XCTAssert(abs(a - b) <= tol * max(1, abs(b)), "\(a) vs \(b) (tol \(tol)) \(msg())", file: file, line: line)
}

// MARK: - Formula images (mirror of detector.py fixture exporter)

enum FormulaImage {
    static func hash(_ w: Int, _ h: Int, seed: Int = 0) -> ImageU8 {
        var img = ImageU8(width: w, height: h, channels: 3)
        let mask: UInt64 = 0xFFFF_FFFF
        img.data.withUnsafeMutableBufferPointer { p in
            var i = 0
            for y in 0..<UInt64(h) {
                for x in 0..<UInt64(w) {
                    for c in 0..<UInt64(3) {
                        var v = ((x &* 73856093) ^ (y &* 19349663) ^ ((c + UInt64(seed)) &* 83492791)) & mask
                        v = ((v ^ (v >> 13)) &* 1274126177) & mask
                        p[i] = UInt8((v >> 16) & 255)
                        i += 1
                    }
                }
            }
        }
        return img
    }

    static func smooth(_ w: Int, _ h: Int) -> ImageU8 {
        var img = ImageU8(width: w, height: h, channels: 3)
        let dw = max(w - 1, 1), dh = max(h - 1, 1), dd = max(w + h - 2, 1)
        img.data.withUnsafeMutableBufferPointer { p in
            for y in 0..<h {
                for x in 0..<w {
                    let i = (y * w + x) * 3
                    p[i] = UInt8(x * 255 / dw); p[i + 1] = UInt8(y * 255 / dh); p[i + 2] = UInt8((x + y) * 255 / dd)
                }
            }
        }
        return img
    }

    static func mixed(_ w: Int, _ h: Int, seed: Int = 0) -> ImageU8 {
        let n = hash(w, h, seed: seed)
        var s = smooth(w, h)
        for i in s.data.indices {
            let v = Int(s.data[i]) + Int(n.data[i]) % 32 - 16
            s.data[i] = UInt8(min(max(v, 0), 255))
        }
        return s
    }

    static func make(_ kind: String, _ w: Int, _ h: Int, seed: Int) -> ImageU8 {
        switch kind {
        case "hash": return hash(w, h, seed: seed)
        case "smooth": return smooth(w, h)
        default: return mixed(w, h, seed: seed)
        }
    }

    /// Per row: plain sum and Σ v·((i mod 251) + 1), i = index inside the row.
    static func rowChecksums(_ p: UnsafePointer<UInt8>, width: Int, height: Int, channels: Int = 3) -> (sum: [Int], wsum: [Int]) {
        var s = [Int](), ws = [Int]()
        let n = width * channels
        for y in 0..<height {
            var a = 0, b = 0
            for i in 0..<n { let v = Int(p[y * n + i]); a += v; b += v * (i % 251 + 1) }
            s.append(a); ws.append(b)
        }
        return (s, ws)
    }
}
