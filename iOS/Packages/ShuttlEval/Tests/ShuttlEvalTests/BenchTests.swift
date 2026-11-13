import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlEval

/// shuttl-bench end to end: scoring Python-recorded predictions must reproduce
/// benchmark.py's report exactly (keys, order, int/float, bit-identical floats, table text).
final class BenchTests: XCTestCase {

    func assertParity(_ argv: [String], file: StaticString = #filePath, line: UInt = #line) async throws {
        let (_, diff, text) = try await runBench(argv + ["--no-json"])
        let d = try XCTUnwrap(diff, "no parity check ran", file: file, line: line)
        XCTAssert(d.ok, "\(argv):\n" + d.mismatches.joined(separator: "\n") + "\n" + text, file: file, line: line)
        XCTAssertEqual(d.nFloats, d.nExactFloats, "not every float bit-exact (max rel \(d.maxRelDiff))", file: file, line: line)
    }

    func testCompactE2EBundle() async throws { try await assertParity(["--predictions", try fixtureURL("bench_e2e").path]) }
    func testCompactOracleBundle() async throws { try await assertParity(["--predictions", try fixtureURL("bench_oracle").path]) }
    func testCompactRealBundle() async throws { try await assertParity(["--predictions", try fixtureURL("bench_real").path]) }

    /// The full baseline runs (/tmp/shuttl_eval_bench, from `eval --full`). No accuracy
    /// constants: the numbers move as accuracy work lands and the bundles are regenerated.
    /// The contract is parity — Swift's scores == the Python scores stored in the same
    /// bundle — plus sanity bounds that hold for any sane pipeline.
    func testFullBaselineBundles() async throws {
        let names = ["e2e", "oracle", "real_tracknetv3", "real_wasb"]
        let urls = names.map { fullBenchDir.appendingPathComponent("bench_\($0).json") }
        try XCTSkipUnless(urls.allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
                          "full bundles absent — run `.venv/bin/python -m scripts.swift_fixtures.eval --only none --full`")
        let headline: [String: [(String, String)]] = [
            "e2e": [("hits", "f1"), ("reconstruct3d", "landing_err_m_median"), ("rally", "f1"), ("linker", "f1")],
            "oracle": [("hits", "f1"), ("rally", "f1"), ("linker", "f1")],
            "real_tracknetv3": [("real_detector", "f1"), ("real_linker", "f1"), ("real_linker", "tn4_f1")],
            "real_wasb": [("real_detector", "f1"), ("real_linker", "f1")],
        ]
        var summary: [String] = []
        for (n, u) in zip(names, urls) {
            try await assertParity(["--predictions", u.path])              // every metric, bit-exact
            let swift = try await runBench(["--predictions", u.path, "--no-json"]).report.results
            let python = try XCTUnwrap(BenchBundle.load(u).report["results"]?.dict, n)
            for (st, k) in headline[n]! {
                let s = try XCTUnwrap(swift[st]?[k]?.double, "\(n) \(st).\(k) missing in Swift")
                let p = try XCTUnwrap(python[st]?[k]?.double, "\(n) \(st).\(k) missing in the Python report")
                XCTAssertEqual(s.bitPattern, p.bitPattern, "\(n) \(st).\(k): Swift \(s) vs Python \(p)")
                summary.append("\(n) \(st).\(k)=\(PyFormat.fixed(s, 3))")
            }
            // sanity bounds
            for (st, res) in swift {
                for key in ["f1", "precision", "recall", "tn4_f1"] {
                    if let v = res[key]?.double, v.isFinite { XCTAssert(v >= 0 && v <= 1, "\(n) \(st).\(key)=\(v)") }
                }
                if let v = res["landing_err_m_median"]?.double, v.isFinite { XCTAssert(v >= 0 && v < 5, "\(n) \(st) landing \(v)") }
            }
            if n.hasPrefix("real") {
                XCTAssertGreaterThanOrEqual(swift["real_detector"]?["n_frames"]?.int ?? 0, 5000, "\(n): too few eval frames")
            } else {
                XCTAssertGreaterThanOrEqual(swift["rally"]?["n_true"]?.int ?? 0, 8, "\(n): too few rallies")
                XCTAssertGreaterThanOrEqual(swift["hits"]?["n_true"]?.int ?? 0, 50, "\(n): too few hits")
            }
        }
        print("full baselines: " + summary.joined(separator: "  "))
    }

    /// MatchAnalysis JSON files (the Swift-pipeline output format) → same real numbers.
    func testAnalysisDirReproducesRealStages() async throws {
        let bundle = try fixtureURL("bench_real")
        let dir = bundle.deletingLastPathComponent().appendingPathComponent("analysis_real")
        let expected = try BenchBundle.load(bundle).report
        // bundle + MatchAnalysis override of detector / detector-linker
        let (a, _, _) = try await runBench(["--predictions", bundle.path, "--analysis-dir", dir.path, "--no-json"])
        assertSamePy(a.results, expected["results"]!, "bundle + analysis-dir")
        // MatchAnalysis only: clips + labels from the Test split
        try XCTSkipUnless(!RealData.clips(backend: backendRoot).isEmpty, "TrackNetV2 Test split not downloaded")
        let (b, _, _) = try await runBench(["--stage", "real", "--clips", "3", "--max-frames", "300", "--analysis-dir", dir.path, "--no-json"])
        var want = expected["results"]!.dict!
        for k in ["real_oracle_detections", "real_oracle_linker"] { want.removeValue(forKey: k) }
        want["real_linker"] = .dict(want["real_linker"]!.dict!.filter { $0.key != "seconds" }.reduce(into: PyDict()) { $0[$1.key] = $1.value })
        var got = b.results
        if var l = got["real_linker"]?.dict { l.removeValue(forKey: "seconds"); got["real_linker"] = .dict(l) }
        assertSamePy(got, .dict(want), "analysis-dir only")
        XCTAssertEqual(b.report["info"]?["clips"], expected["info"]?["clips"])
    }

    /// Swift label CSV / drop-frame parsing = Python's (recorded in the bundle).
    func testRealLabelsAndDropMaskMatchPython() throws {
        let bundle = try BenchBundle.load(try fixtureURL("bench_real"))
        let clips = RealData.clips(backend: backendRoot)
        try XCTSkipUnless(!clips.isEmpty, "TrackNetV2 Test split not downloaded")
        XCTAssertEqual(clips.count, 29)
        for s in bundle.samples {
            let c = try XCTUnwrap(clips.first { $0.match == s.match && $0.clip == s.clip })
            XCTAssertEqual(c.labels.path, s.labels)
            let (xy, vis) = try RealData.loadLabels(c.labels, n: s.n!)
            XCTAssertEqual(vis, s.truthVis)
            assertBits(xy.flatMap(\.row), s.truthXY!.flatMap(\.row), "labels \(c.key)")
            XCTAssertEqual(RealData.dropMask(backend: backendRoot, match: c.match, clip: c.clip, n: s.n!), s.dropMask)
        }
    }

    func testReportWriteRoundTripsAndCompareDetectsChanges() async throws {
        let (r, _, _) = try await runBench(["--predictions", try fixtureURL("bench_oracle").path, "--no-json"])
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-bench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = try r.write(to: dir)
        let back = try PyJSON.load(url)
        XCTAssert(ReportCompare.compare(back, .dict(r.report), skip: []).ok)
        var tampered = r.results
        var hits = tampered["hits"]!.dict!
        hits["f1"] = .float(hits["f1"]!.double!.nextUp)
        tampered["hits"] = .dict(hits)
        XCTAssertFalse(ReportCompare.compare(.dict(tampered), .dict(r.results), relTol: 0).ok, "1 ulp must differ bit-exactly")
        XCTAssertTrue(ReportCompare.compare(.dict(tampered), .dict(r.results)).ok, "within the default 1e-12")
        hits["f1"] = .int(1)
        tampered["hits"] = .dict(hits)
        XCTAssertFalse(ReportCompare.compare(.dict(tampered), .dict(r.results), relTol: 1).ok, "int vs float must differ")
    }

    func testCLIFlags() async throws {
        let b = try fixtureURL("bench_e2e").path
        // scoring overrides re-score without the parity check
        let (r, diff, _) = try await runBench(["--predictions", b, "--tol-px", "5"])
        XCTAssertNil(diff)
        XCTAssertEqual(r.report["tol_px"]?.double, 5)
        // sample-defining flags are fixed by the bundle
        do { _ = try await runBench(["--predictions", b, "--n", "7"]); XCTFail("--n accepted") } catch BenchCLIError.usage {}
        do { _ = try await runBench(["--predictions", b, "--stage", "oracle"]); XCTFail("--stage accepted") } catch BenchCLIError.usage {}
        do { _ = try await runBench(["--stage", "e2e"]); XCTFail("synthetic without bundle accepted") } catch BenchCLIError.usage {}
        do { _ = try await runBench(["--bogus"]); XCTFail("bogus flag") } catch BenchCLIError.usage {}
    }
}
