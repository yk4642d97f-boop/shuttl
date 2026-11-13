import CoreVideo
import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// P2-2: players (~400 s on a 10-min clip on the phone) resumes after a kill / lock / -11847.
/// A resumed pass must give exactly the observations of an uninterrupted one, wherever it
/// was interrupted, and however many times.
final class PlayersResumeTests: XCTestCase {
    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        var value: Int { lock.withLock { n } }
        func hit() { lock.withLock { n += 1 } }
    }

    private func realCase() throws -> (PlayerCase, URL) {
        let pc = try PlayerCase(Fx.list(try Fx.load("real"))[0])
        let url = repoRoot.appendingPathComponent(pc.raw["relative_path"] as? String ?? "")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test clip not present: \(url.path)")
        return (pc, url)
    }

    /// Bit-exact: `String(reflecting:)` prints every Double so that it round-trips (NaN included).
    private func assertSame(_ a: PlayerTracks, _ b: PlayerTracks, _ what: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.observations.count, b.observations.count, what, file: file, line: line)
        XCTAssertEqual(String(reflecting: a.observations), String(reflecting: b.observations), what, file: file, line: line)
        XCTAssertEqual(a.doubles, b.doubles, what, file: file, line: line)
        // Diagnostics hold a dictionary, whose print order depends on the hash seed: compare by key.
        XCTAssertEqual(a.diagnostics?.doubles, b.diagnostics?.doubles, what, file: file, line: line)
        XCTAssertEqual(a.diagnostics?.sampledFrames, b.diagnostics?.sampledFrames, what, file: file, line: line)
        XCTAssertEqual(a.diagnostics?.coverage.mapValues(\.bitPattern), b.diagnostics?.coverage.mapValues(\.bitPattern),
                       what, file: file, line: line)
    }

    private struct Run {
        var tracks: PlayerTracks?
        var saves: [PlayersResume] = []
        var problems: [String?] = []
        var sampled: [Int] = []
    }

    /// One pass; cancelled before sampled frame `cancelAfter` (counted from this pass's start).
    private func run(_ src: VideoSource, _ pc: PlayerCase, detector: PoseDetector?, resume: PlayersResume? = nil,
                     every: Int? = 10, wall: Double? = nil, clock: (@Sendable () -> TimeInterval)? = nil,
                     cancelAfter: Int? = nil, camera: CameraModel?? = nil) throws -> Run {
        var r = Run()
        var polls = 0
        let policy = PlayersCheckpointPolicy(everyFrames: every, wallSeconds: wall, clock: clock ?? { 0 })
        do {
            r.tracks = try PlayerTracker.trackPlayers(
                source: src, camera: camera ?? pc.camera, step: 2, detector: detector, resume: resume,
                checkpointPolicy: policy, checkpoint: { r.saves.append($0) }, resumeReport: { r.problems.append($0) },
                isCancelled: { polls += 1; return cancelAfter.map { polls > $0 } ?? false })
        } catch is CancellationError {
            XCTAssertNotNil(cancelAfter, "cancelled without being asked to")
        }
        return r
    }

    private func replay(_ pc: PlayerCase, seen: Counter? = nil) -> ClosureDetector {
        let byFrame = Dictionary(uniqueKeysWithValues: zip(pc.frames, pc.detections))
        return ClosureDetector { frames, idx in
            for _ in idx { seen?.hit() }
            return idx.map { byFrame[$0] ?? .empty }
        }
    }

    func testCandidatesCodecIsBitExact() throws {
        let kps = KeypointArray(values: (0..<102).map { Double($0) * 0.37 - 5 })
        let odd: [Double] = [.nan, -0.0, .infinity, -.infinity, .leastNonzeroMagnitude, 1 / 3]
        let a = PlayerCandidates(frame: 4242, boxes: [BBox(x1: -0.0, y1: .nan, x2: 3, y2: 4), BBox(x1: 1, y1: 2, x2: 5, y2: 9)],
                                 scores: [0.9, .nan], kps: kps, footPx: [SIMD2(.nan, 1), SIMD2(-0.0, .infinity)],
                                 pos: [SIMD2<Double>(1.0 / 3, 2), SIMD2<Double>(3, 4)], unary: odd, unit: [1, 1], noise: [0.1, 0.2],
                                 app: (0..<(2 * Appearance.dim)).map { Float($0) / 7 })
        let empty = PlayerCandidates(frame: 0, boxes: [], scores: [], kps: KeypointArray(values: []), footPx: [], pos: [],
                                     unary: [], unit: [], noise: [], app: nil)
        var withNaNApp = a
        withNaNApp.app![3] = Float.nan
        withNaNApp.app![4] = Float(-0.0)
        for c in [a, empty, withNaNApp] {
            let d = c.encoded()
            let back = try XCTUnwrap(PlayerCandidates(encoded: d))
            XCTAssertEqual(back.encoded(), d, "re-encoding is byte-identical")
            XCTAssertEqual(String(reflecting: back), String(reflecting: c))
            XCTAssertEqual(back.app?.map(\.bitPattern), c.app?.map(\.bitPattern))
            XCTAssertEqual(back.unary.map(\.bitPattern), c.unary.map(\.bitPattern))
            XCTAssertNil(PlayerCandidates(encoded: d.dropLast()), "torn")
            XCTAssertNil(PlayerCandidates(encoded: d + [0]), "trailing byte")
        }
        XCTAssertNil(PlayerCandidates(encoded: Data([1, 2, 3])))
    }

    /// The replayed YOLO detections of the Test clip, interrupted after 1, 5, 17, 33, 50 and 66 of
    /// its 67 sampled frames, resumed from the last save each time: identical tracks.
    func testResumeAtEveryInterruptionPointIsExact() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        let full = try XCTUnwrap(try run(src, pc, detector: replay(pc), every: nil).tracks)
        XCTAssertFalse(full.observations.isEmpty)
        for k in [1, 5, 17, 33, 50, 66] {
            let cut = try run(src, pc, detector: replay(pc), every: 4, cancelAfter: k)
            XCTAssertNil(cut.tracks, "@\(k)")
            guard let last = cut.saves.last else {
                XCTAssertLessThan(k, 2, "a save every 4 frames must exist after \(k) sampled frames"); continue
            }
            XCTAssertLessThanOrEqual(last.nextFrame, 2 * k)
            XCTAssertEqual(last.frames.count, last.nextFrame / 2)
            let seen = Counter()
            let resumed = try run(src, pc, detector: replay(pc, seen: seen), resume: last)
            XCTAssertEqual(resumed.problems, [nil], "@\(k): the resume must be used")
            XCTAssertEqual(seen.value, (pc.meta.nFrames - last.nextFrame + 1) / 2,
                           "@\(k): only the frames from \(last.nextFrame) on are detected again")
            assertSame(try XCTUnwrap(resumed.tracks), full, "resumed from frame \(last.nextFrame) (cut @\(k))")
        }
    }

    /// Interrupted three times, each run resuming the previous one's last save and saving on.
    func testChainedInterruptionsAreExact() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        let full = try XCTUnwrap(try run(src, pc, detector: replay(pc), every: nil).tracks)
        var resume: PlayersResume?
        for k in [12, 9, 20] {
            let cut = try run(src, pc, detector: replay(pc), resume: resume, every: 6, cancelAfter: k)
            if resume != nil { XCTAssertEqual(cut.problems, [nil]) }
            let last = try XCTUnwrap(cut.saves.last)
            XCTAssertGreaterThan(last.nextFrame, resume?.nextFrame ?? 0, "progress made in every run")
            if let resume { XCTAssertEqual(Array(last.frames.prefix(resume.frames.count)), resume.frames, "loaded bytes re-saved as-is") }
            resume = last
        }
        let end = try run(src, pc, detector: replay(pc), resume: resume)
        XCTAssertEqual(end.problems, [nil])
        assertSame(try XCTUnwrap(end.tracks), full, "after three interruptions")
    }

    /// The phone's detector: Apple Vision with the far-court crop passes, on the Test clip.
    func testResumeWithVisionIsExactOnTheFixtureClip() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        let full = try XCTUnwrap(try run(src, pc, detector: nil, every: nil).tracks)
        let again = try XCTUnwrap(try run(src, pc, detector: nil, every: 4).tracks)
        assertSame(again, full, "Vision is deterministic and saving changes nothing")
        for k in [7, 30, 61] {
            let cut = try run(src, pc, detector: nil, every: 4, cancelAfter: k)
            let last = try XCTUnwrap(cut.saves.last, "@\(k)")
            let resumed = try run(src, pc, detector: nil, resume: last)
            XCTAssertEqual(resumed.problems, [nil], "@\(k)")
            assertSame(try XCTUnwrap(resumed.tracks), full, "Vision resumed from frame \(last.nextFrame) (cut @\(k))")
        }
    }

    func testUnusableResumeIsIgnoredAndReported() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        let full = try XCTUnwrap(try run(src, pc, detector: replay(pc), every: nil).tracks)
        let cut = try run(src, pc, detector: replay(pc), every: 10, cancelAfter: 30)
        let good = try XCTUnwrap(cut.saves.last)
        var otherCam = try XCTUnwrap(pc.camera)
        otherCam.t.x += 1e-9
        var misaligned = good; misaligned.nextFrame += 1
        var short = good; short.frames.removeLast()
        var torn = good; torn.frames[3] = torn.frames[3].dropLast()
        var swapped = good; swapped.frames.swapAt(1, 2)
        var past = good; past.nextFrame = pc.meta.nFrames + 1
        let cases: [(String, PlayersResume, CameraModel??)] = [
            ("context", good, .some(otherCam)), ("grid", misaligned, nil), ("sampled frames ≠", short, nil),
            ("unreadable", torn, nil), ("expected", swapped, nil), ("≥", past, nil),
        ]
        for (why, r, cam) in cases {
            let res = try run(src, pc, detector: replay(pc), resume: r, camera: cam)
            XCTAssertEqual(res.problems.count, 1)
            let problem = try XCTUnwrap(res.problems.first ?? nil, "\(why) must be rejected")
            XCTAssertTrue(problem.contains(why), "\(why): \(problem)")
            if cam == nil { assertSame(try XCTUnwrap(res.tracks), full, "\(why): a clean full run") }
        }
        XCTAssertEqual(try run(src, pc, detector: replay(pc), resume: good).problems, [nil])
    }

    /// The wall-clock bound (P2-1's rule for players): a save once `wallSeconds` passed since
    /// the last one, even when the video bound never fires; never two at one frame, never at
    /// the end; no bounds = no saves.
    func testWallClockPolicy() async throws {
        let (pc, url) = try realCase()
        let src = try await VideoSource.open(url)
        let seen = Counter()
        let clock: @Sendable () -> TimeInterval = { Double(seen.value) }        // 1 s per sampled frame
        let r = try run(src, pc, detector: replay(pc, seen: seen), every: nil, wall: 9.5, clock: clock)
        let frames = r.saves.map(\.nextFrame)
        XCTAssertEqual(frames, Array(stride(from: 20, to: pc.meta.nFrames, by: 20)), "one save per 10 sampled frames")
        XCTAssertTrue(frames.allSatisfy { $0 < pc.meta.nFrames })
        XCTAssertEqual(try run(src, pc, detector: replay(pc), every: nil, wall: nil).saves.count, 0)
        XCTAssertEqual(try run(src, pc, detector: replay(pc), every: 0, wall: 0).saves.count, 0, "zero = off")
        // Video bound 40 frames + a wall bound that never elapses = video saves only.
        XCTAssertEqual(try run(src, pc, detector: replay(pc), every: 40, wall: 1e9).saves.map(\.nextFrame), [40, 80, 120])
    }
}
