import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Duplicate-PTS re-timing (Python test_duplicate_pts_retimed) and the per-piece solve
/// fallback (Python test_solve_pieces_per_piece_fallback).
final class RetimeTests: XCTestCase {
    private func T(_ ts: [Double], fps: Double = 30, n: Int? = nil) -> [Double] {
        let n = n ?? ts.count
        return linkerTimestamps(VideoMeta(fps: fps, width: 1, height: 1, nFrames: n, timestamps: ts), n)
    }
    private func increasing(_ t: [Double]) -> Bool { zip(t.dropFirst(), t).allSatisfy { $0 > $1 } }

    func testDuplicatePtsRetimed() {
        XCTAssertEqual(T([0, 1, 2, 2, 2, 2, 6, 7]), [0, 1, 2, 3, 4, 5, 6, 7])       // interior run
        XCTAssertEqual(T([0, 0, 0, 3, 4]), [0, 1, 2, 3, 4])                         // leading run
        XCTAssertEqual(T([0, 1, 2, 3, 3, 3]), [0, 1, 2, 3, 4, 5])                   // trailing run
        XCTAssertEqual(T([0, 1, 0.5, 2]), [0, 1, 1.5, 2])                           // reversal
        for (x, y) in zip(T([5, 5, 5, 5]), (0..<4).map { 5 + Double($0) / 30 }) { XCTAssertEqual(x, y, accuracy: 1e-12) }
        XCTAssertEqual(T([5, 5, 5, 5], fps: .infinity), T([5, 5, 5, 5]))            // fps inf → 30
        var rng = SplitMix64(s: 1)
        let ok = (0..<50).map { Double($0) / 30 + (rng.next() - 0.5) * 0.004 }
        XCTAssertEqual(T(ok), ok)                                                   // untouched
        for fps in [30.0, 60.0, 240.0] {
            for run in [1200, 5000] {                                 // frozen clock, then normal
                var frames: [Double] = (0..<40).map { Double($0) }
                frames += [Double](repeating: 40, count: run)
                frames += (0..<60).map { 41 + Double($0) }
                let ts: [Double] = frames.map { $0 / fps }
                let t = T(ts, fps: fps)
                XCTAssertTrue(increasing(t), "frozen \(run) @ \(fps)")
                XCTAssertEqual(t[40], ts[40]); XCTAssertEqual(t[40 + run], ts[40 + run])
                for k in 41...(40 + run) { XCTAssertEqual(t[k] - t[k - 1], 1 / fps / Double(run), accuracy: 1e-12) }
            }
            let lead = [Double](repeating: 0, count: 1500) + (0..<20).map { 1 + Double($0) / fps }
            let trail = (0..<20).map { Double($0) / fps } + [Double](repeating: 1, count: 1500)
            for ts in [lead, trail, [Double](repeating: 2, count: 1200)] { XCTAssertTrue(increasing(T(ts, fps: fps))) }
        }
        XCTAssertTrue(increasing(T([1e17, 1e17, 1e17, 1e17 + 64])))               // below float resolution
    }

    func testSolvePiecesPerPieceFallback() {
        let t = (0..<60).map { Double($0) / 60 }
        let fr = Array(0..<25) + Array(35..<55)                      // pieces of 25 and 20 vars
        var rng = SplitMix64(s: 2)
        let p = fr.map { SIMD2(100 + 400 * t[$0] + rng.next() - 0.5, 200 + 300 * t[$0] * t[$0] + rng.next() - 0.5) }
        let pc = Pieces(a: [0, 25], b: [25, 45], lo: [0, 35], hi: [24, 54])
        let diag = hypot(1920.0, 1080.0), cfg = LinkerConfig()
        let base = solvePieces(t, fr, p, pc, 1.5, diag, cfg)
        func failing(_ widths: Set<Int>) -> BandedSolver {
            { ab, v, rx, ry in widths.contains(v) ? false : bandedSolve(ab, v, rx, ry) }
        }
        let alone = solvePieces(t, fr, p, pc, 1.5, diag, cfg, solve: failing([45]))
        XCTAssertEqual(alone.vf, base.vf)
        for (a, b) in zip(alone.x, base.x) { XCTAssertEqual(a.x, b.x, accuracy: 1e-9); XCTAssertEqual(a.y, b.y, accuracy: 1e-9) }
        for (a, b) in zip(alone.resid, base.resid) { XCTAssertEqual(a, b, accuracy: 1e-9) }
        let one = solvePieces(t, fr, p, pc, 1.5, diag, cfg, solve: failing([45, 20]))
        for v in 0..<25 { XCTAssertEqual(one.x[v].x, base.x[v].x, accuracy: 1e-9) }
        XCTAssertTrue(one.x[25...].allSatisfy { $0.x.isNaN && $0.y.isNaN })
        XCTAssertTrue(one.resid[25...].allSatisfy(\.isNaN))
        XCTAssertTrue(one.resid[..<25].allSatisfy(\.isFinite))
    }
}
