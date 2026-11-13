import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Deterministic generators for the degenerate-input sweep (mirror of the Python fuzz in
/// backend/scripts/swift_fixtures/linker.py `--edge`), plus the Track2D invariants.
struct SplitMix64 {
    var s: UInt64
    mutating func next() -> Double {
        s &+= 0x9E3779B97F4A7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
    }
}

enum Degenerate {
    static let sizes = [(192, 144), (1920, 1080), (0, 0), (1, 1)]
    static let kinds = ["empty", "one", "weak_line", "strong_line", "burst", "static", "noise", "dup", "huge_xy"]
    static let fpsList: [Double] = [30, 0, .nan, .infinity, 240]
    static let clocks = ["nominal", "missing", "short", "dup_run", "zeros", "nan_inside", "reversed", "jump", "all_nan"]

    static func timestamps(_ kind: String, _ n: Int, _ fps: Double) -> [Double] {
        let f = fps.isFinite && fps > 0 ? fps : 30
        var t = (0..<n).map { Double($0) / f }
        switch kind {
        case "missing": return []
        case "short": return Array(t.prefix(n / 2))
        case "dup_run": if n > 0 { for i in (n / 3)..<(2 * n / 3) { t[i] = t[n / 3] } }
        case "zeros": return [Double](repeating: 0, count: n)
        case "nan_inside": for i in (n / 2)..<Swift.min(n / 2 + 2, n) { t[i] = .nan }
        case "reversed": t.reverse()
        case "jump": for i in (n / 2)..<n { t[i] += 1e6 }
        case "all_nan": return [Double](repeating: .nan, count: n)
        default: break
        }
        return t
    }

    static func candidates(_ kind: String, _ n: Int, _ w: Int, _ h: Int, _ rng: inout SplitMix64) -> [[ShuttleCandidate]] {
        let W = Double(Swift.max(w, 1)), H = Double(Swift.max(h, 1))
        var c = [[ShuttleCandidate]](repeating: [], count: n)
        for i in 0..<n {
            let x = W * 0.1 + Double(i) * W * 0.8 / Double(Swift.max(n, 1))
            let y = H * 0.2 + Double(i) * H * 0.5 / Double(Swift.max(n, 1))
            switch kind {
            case "one": if i == n / 2 { c[i] = [ShuttleCandidate(x: W / 2, y: H / 2, score: 0.9)] }
            case "weak_line": c[i] = [ShuttleCandidate(x: x, y: y, score: 0.3)]
            case "strong_line": c[i] = [ShuttleCandidate(x: x, y: y, score: 0.9)]
            case "burst": if (i / 4) % 2 == 0 { c[i] = [ShuttleCandidate(x: x, y: y, score: i % 3 != 0 ? 0.7 : 0.2)] }
            case "static": c[i] = [ShuttleCandidate(x: W * 0.3, y: H * 0.6, score: 0.8), ShuttleCandidate(x: W * 0.7, y: H * 0.4, score: 0.3)]
            case "noise": c[i] = (0..<Int(rng.next() * 7)).map { _ in
                ShuttleCandidate(x: rng.next() * W, y: rng.next() * H, score: rng.next()) }
            case "dup": c[i] = i % 6 == 0 && i > 0 ? c[i - 1] : [ShuttleCandidate(x: x, y: y, score: 0.8)]
            case "huge_xy": c[i] = [ShuttleCandidate(x: 1e300, y: -1e300, score: .infinity),
                                    ShuttleCandidate(x: .nan, y: 5, score: 0.9), ShuttleCandidate(x: x, y: y, score: .nan)]
            default: break
            }
        }
        return c
    }

    /// Track2D invariants of test_link_accuracy_per_case / test_raw_xy_is_the_adopted_detection.
    static func check(_ tr: Track2D, n: Int, _ name: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(tr.n, n, name, file: file, line: line)
        guard tr.xy.count == n, tr.conf.count == n, tr.visible.count == n, tr.interpolated.count == n,
              (tr.rawXY?.count ?? n) == n else { return XCTFail("array lengths \(name)", file: file, line: line) }
        for f in 0..<n {
            if tr.visible[f] && tr.interpolated[f] { XCTFail("vis & interp \(name) \(f)", file: file, line: line) }
            if !tr.visible[f] && tr.conf[f] != 0 { XCTFail("conf \(name) \(f)", file: file, line: line) }
            if (tr.visible[f] || tr.interpolated[f]) && !(tr.xy[f].x.isFinite && tr.xy[f].y.isFinite) {
                XCTFail("non-finite output xy \(name) \(f)", file: file, line: line)
            }
            if let r = tr.rawXY, r[f].x.isFinite != tr.visible[f] { XCTFail("raw_xy \(name) \(f)", file: file, line: line) }
        }
    }
}
