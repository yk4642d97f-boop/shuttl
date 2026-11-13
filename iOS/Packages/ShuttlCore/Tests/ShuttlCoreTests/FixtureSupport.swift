import Foundation
import XCTest
@testable import ShuttlCore

/// Loads a JSON fixture exported by backend/scripts/swift_fixtures/core.py.
func fixture(_ name: String) throws -> JSONValue {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"),
                            "missing fixture \(name).json — run `.venv/bin/python -m scripts.swift_fixtures.core` in backend/")
    return try ShuttlJSON.load(JSONValue.self, from: url)
}

/// Repository root (…/华北五省), derived from this file's location.
let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

extension JSONValue {
    var doubles: [Double] { (array ?? []).map { $0.double ?? .nan } }
    var rows: [[Double]] { (array ?? []).map(\.doubles) }
    var v2: [SIMD2<Double>] { rows.map(SIMD2.init(row:)) }
    var v3: [SIMD3<Double>] { rows.map(SIMD3.init(row:)) }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        try ShuttlJSON.decoder().decode(T.self, from: ShuttlJSON.encoder().encode(self))
    }
}

func assertClose(_ a: Double, _ b: Double, tol: Double, _ msg: @autoclosure () -> String = "",
                 file: StaticString = #filePath, line: UInt = #line) {
    // Python writes every non-finite value as null → NaN; any non-finite Swift value matches it.
    if !b.isFinite { XCTAssertFalse(a.isFinite, "\(a) vs non-finite \(msg())", file: file, line: line); return }
    XCTAssert(abs(a - b) <= tol * max(1, abs(b)), "\(a) vs \(b) (tol \(tol)) \(msg())", file: file, line: line)
}

func assertClose(_ a: [Double], _ b: [Double], tol: Double, _ msg: @autoclosure () -> String = "",
                 file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, "length \(msg())", file: file, line: line)
    for (x, y) in zip(a, b) { assertClose(x, y, tol: tol, msg(), file: file, line: line) }
}
