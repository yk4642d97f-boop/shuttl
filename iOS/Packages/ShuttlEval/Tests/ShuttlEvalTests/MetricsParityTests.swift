import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlEval

/// metrics.json: every metrics.py function on hand-built + randomised inputs; the Swift
/// port must return the same dict (keys, order, int/float) with bit-identical floats.
final class MetricsParityTests: XCTestCase {

    func testEveryMetricsCase() throws {
        let fx = try pyFixture("metrics")
        let cases = try XCTUnwrap(fx["cases"]?.list)
        var seen: [String: Int] = [:]
        for (k, c) in cases.enumerated() {
            let fn = try XCTUnwrap(c["fn"]?.string), a = try XCTUnwrap(c["args"]), want = try XCTUnwrap(c["out"])
            seen[fn, default: 0] += 1
            let tag = "case #\(k) \(fn)"
            switch fn {
            case "detection":
                let got = try Metrics.detectionMetrics(a["pred"]!.v2s, a["true"]!.v2s, vis: a["vis"]!.bools,
                                                       tolPx: a["tol_px"]!.dbl, mask: a["mask"]!.bools,
                                                       tracknetInt: a["tracknet_int"]!.bool!,
                                                       imageSize: a["image_size"]!.list.map { ImageSize(width: $0[0].int!, height: $0[1].int!) })
                assertSamePy(got, want, tag)
            case "detection_cands":
                assertSamePy(try Metrics.detectionMetrics(a["cands"]!.candidates, a["true"]!.v2s), want, tag)
            case "track":
                let got = try Metrics.trackMetrics(a["pred"]!.v2s, interpolated: a["interpolated"]!.bools, a["true"]!.v2s,
                                                   vis: a["vis"]!.bools, tolPx: a["tol_px"]!.dbl, mask: a["mask"]!.bools)
                assertSamePy(got, want, tag)
            case "candidates_to_xy":
                let (xy, sc) = Metrics.candidatesToXY(a["cands"]!.candidates, scoreThresh: a["thresh"]!.dbl)
                assertBits(xy.flatMap(\.row), want["xy"]!.v2s.flatMap(\.row), tag + " xy")
                assertBits(sc, want["score"]!.dbls, tag + " score")
            case "hits":
                let got = Metrics.hitMetrics(try a["pred"]!.decode([EvalHit].self), try a["true"]!.decode([EvalHit].self),
                                             tolFrames: a["tol_frames"]!.int!)
                assertSamePy(got, want, tag)
            case "segments":
                let got = Metrics.segmentMetrics(try a["pred"]!.decode([EvalSegment].self), try a["true"]!.decode([EvalSegment].self),
                                                 tolFrames: a["tol_frames"]!.int!)
                assertSamePy(got, want, tag)
            case "landing":
                assertSamePy(Metrics.landingMetrics(a["pred"]!.optV2s, a["true"]!.optV2s), want, tag)
            case "in_out":
                let m: [Double?]? = a["margins"]!.list.map { $0.map { $0.isNone ? nil : $0.dbl } }
                let got = try Metrics.inOutMetrics(a["pred"]!.strs, a["true"]!.strs, margins: m, closeM: a["close_m"]!.dbl)
                assertSamePy(got, want, tag)
            case "shot_type":
                let labels = a["labels"]!.list.map { $0.compactMap(\.string) }
                assertSamePy(Metrics.shotTypeMetrics(a["pred"]!.strs, a["true"]!.strs, labels: labels), want, tag)
            case "confusion":
                let labels = a["labels"]!.list.map { $0.compactMap(\.string) }
                assertSamePy(Metrics.confusionMatrix(a["pred"]!.strs, a["true"]!.strs, labels: labels), want, tag)
            case "rally":
                let got = Metrics.rallyMetrics(try a["pred"]!.decode([EvalRally].self), try a["true"]!.decode([EvalRally].self),
                                               iouThresh: a["iou_thresh"]!.dbl)
                assertSamePy(got, want, tag)
            case "format_table":
                let res = try XCTUnwrap(a["results"]?.dict)
                XCTAssertEqual(Metrics.formatTable(res), want["text"]!.string!, tag)
                XCTAssertEqual(Metrics.formatTable(res, keys: ["n", "f1", "flag", "precision"]), want["text_keys"]!.string!, tag)
            case "scaled_tolerance":
                let got = a["cases"]!.list!.map { c -> Double in
                    let v = c.dbls
                    return Metrics.scaledTolerance(v[0], imageSize: ImageSize(width: Int(v[1]), height: Int(v[2])))
                }
                assertBits(got, want["values"]!.dbls, tag)
            default:
                XCTFail("unknown case \(fn)")
            }
        }
        XCTAssertEqual(Set(seen.keys), ["detection", "detection_cands", "track", "candidates_to_xy", "hits", "segments",
                                        "landing", "in_out", "shot_type", "confusion", "rally", "format_table",
                                        "scaled_tolerance"])
        XCTAssertGreaterThan(cases.count, 50)
    }

    /// The numpy reductions behind the metrics, bit for bit.
    func testNumpyProbes() throws {
        let np = try XCTUnwrap(try pyFixture("metrics")["numpy"])
        for p in np["sum"]!.list! {
            let x = p["x"]!.dbls
            assertBits([NP.sum(x), NP.mean(x), NP.median(x)], [p["sum"]!.dbl, p["mean"]!.dbl, p["median"]!.dbl], "n=\(x.count)")
        }
        for p in np["percentile"]!.list! {
            let x = p["x"]!.dbls
            assertBits(p["q"]!.dbls.map { NP.percentile(x, $0) }, p["p"]!.dbls, "percentile n=\(x.count)")
        }
        for p in np["nan"]!.list! {
            let x = p["x"]!.dbls
            assertBits([NP.nanmean(x), NP.nanmedian(x)], [p["nanmean"]!.dbl, p["nanmedian"]!.dbl], "nan n=\(x.count)")
        }
        let nm = np["norm"]!
        let V = nm["v"]!.list!.map(\.dbls)
        assertBits(V.map { NP.norm($0[0], $0[1], $0[2]) }, nm["norm3"]!.dbls, "np.linalg.norm 3-vector (FMA ddot)")
        assertBits(V.map { NP.norm($0[0], $0[1]) }, nm["norm2"]!.dbls, "np.linalg.norm 2-vector")
        assertBits(V.map { NP.rowNorm($0[0], $0[1]) }, nm["row2"]!.dbls, "norm axis=1 (N,2)")
        assertBits(V.map { NP.rowNorm($0[0], $0[1], $0[2]) }, nm["row3"]!.dbls, "norm axis=1 (N,3)")
        assertBits(V.map { hypot($0[0], $0[1]) }, nm["hypot"]!.dbls, "np.hypot")
        let ip = np["interp"]!
        let xp = ip["xp"]!.dbls, fp = ip["fp"]!.dbls
        assertBits(ip["x"]!.dbls.map { NP.interp($0, xp, fp) }, ip["y"]!.dbls, "np.interp")
        for g in np["grids"]!.list! {
            let kind = g["kind"]!.string!, a = g["a"]!.dbl, b = g["b"]!.dbl, n = g["n"]!.int ?? 0
            let got: [Double]
            switch kind {
            case "linspace_rad": got = NP.linspace(a, b, n).map(NP.radians)
            case "linspace": got = NP.linspace(a, b, n)
            case "geomspace": got = NP.geomspace(a, b, n)
            case "arange": got = NP.arange(a, b, g["step"]!.dbl)
            case "degrees": got = g["x"]!.dbls.map(NP.degrees)
            case "radians": got = g["x"]!.dbls.map(NP.radians)
            default: XCTFail(kind); continue
            }
            assertBits(got, g["v"]!.dbls, "\(kind)(\(a), \(b), \(n))")
        }
    }

    /// PyValue JSON = Python json.dumps (ensure_ascii, separators, float repr, order).
    func testPythonJSON() throws {
        let js = try XCTUnwrap(try pyFixture("metrics")["json"])
        let obj = try XCTUnwrap(js["obj"])
        XCTAssertEqual(obj.pythonJSON(), js["dumps"]!.string!)
        XCTAssertEqual(obj.pythonJSON(indent: 1), js["dumps_indent1"]!.string!)
        XCTAssertEqual(try PyJSON.parse(js["dumps_indent1"]!.string!), obj)
        XCTAssertEqual(try PyJSON.parse("[1, 1.0, 1e3, -0, NaN]").list!.map { if case .int = $0 { return "i" } else { return "f" } },
                       ["i", "f", "f", "i", "f"])
    }
}
