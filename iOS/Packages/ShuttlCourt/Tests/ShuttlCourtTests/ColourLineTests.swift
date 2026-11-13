import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// Line paint colour models (`LINE_COLOURS`): exact-input parity on full-RGB medians
/// (court.py section "colour", TestData/colour, kept out of the test bundle for size):
/// R7 yellow-line phonesim clips, sweep2 'colour' grid cases and a white control. Per model:
/// whiteness sum / min / max, mask pixel count, calibrateImage reason and camera; plus the
/// first model that accepts. Runs with SHUTTL_COURT_REAL=1.
final class ColourLineTests: XCTestCase {
    let dir = repoRoot.appendingPathComponent("iOS/Packages/ShuttlCourt/TestData/colour")

    func testWhitenessColourFormula() {
        // RGB pixels: yellow paint, green mat, white
        let img = ImageU8(width: 3, height: 1, channels: 3, data: [238, 196, 34, 60, 140, 70, 235, 235, 235])
        let w = ImageOps.whiteness(img).data, y = ImageOps.whiteness(img, colour: .yellow).data
        // min(S) − ½·(max(S) − min(S)) − ½·max(others), exact in float32
        XCTAssertEqual(w, [-68, 20, 235] as [Float])
        XCTAssertEqual(y, [158, -15, 117.5] as [Float])                 // S = R, G
        XCTAssertEqual(ImageOps.whiteness(img, colour: .red).data, [140, -10, 117.5] as [Float])
        XCTAssertEqual(ImageOps.whiteness(img, colour: .cyan).data, [-166, 5, 117.5] as [Float])
        XCTAssertEqual(LineColour.allCases.map(\.rawValue), ["white", "yellow", "cyan", "red"])
    }

    func testColourParity() throws {
        guard ProcessInfo.processInfo.environment["SHUTTL_COURT_REAL"] == "1" else {
            throw XCTSkip("set SHUTTL_COURT_REAL=1 to run the colour parity set")
        }
        let man = dir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: man.path) else {
            throw XCTSkip("run `.venv/bin/python -m scripts.swift_fixtures.court colour` first")
        }
        let cases = try XCTUnwrap(try ShuttlJSON.load(JSONValue.self, from: man)["cases"]?.array)
        XCTAssertGreaterThanOrEqual(cases.count, 10)
        var kpMax = 0.0, nAccept = 0, refMax = 0.0
        for c in cases {
            let name = c["name"]!.string!
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            let raw = try Data(contentsOf: dir.appendingPathComponent(c["planes"]!.string!)) as NSData
            let data = try raw.decompressed(using: .zlib) as Data
            XCTAssertEqual(data.count, size.width * size.height * 3, name)
            let img = ImageU8(width: size.width, height: size.height, channels: 3, data: [UInt8](data))
            var first: String?
            for colour in LineColour.allCases {
                let want = c["per_colour"]![colour.rawValue]!
                let label = "\(name) \(colour.rawValue)"
                let w = ImageOps.whiteness(img, colour: colour).data
                XCTAssertEqual(w.reduce(0.0) { $0 + Double($1) }, want["whiteness_sum"]!.double!, label)
                XCTAssertEqual(Double(w.min()!), want["whiteness_min"]!.double!, label)
                XCTAssertEqual(Double(w.max()!), want["whiteness_max"]!.double!, label)
                let feat = Features(img, config: .default, colour: colour)
                XCTAssertEqual(feat.mask.data.filter { $0 != 0 }.count, want["mask_px"]!.int!, "\(label) mask")
                var d = CourtCalibrationDiagnostics()
                let cam = AutoCalibration.calibrateImage(img, size: size, config: .default, colour: colour, diag: &d)
                XCTAssertEqual(d.reason.rawValue, want["reason"]!.string!, "\(label) reason")
                let wcam = want["camera"]?.cameraModel
                XCTAssertEqual(cam == nil, wcam == nil, "\(label) accept")
                if let cam, let wcam {
                    let a = keypointAgreement(wcam, cam)
                    kpMax = max(kpMax, a.max)
                    XCTAssertLessThan(a.max, wcam.dist[0] != 0 ? 1e-2 : 1e-3, "\(label) camera px")
                    if first == nil { first = colour.rawValue }
                    let pf = AutoCalibration.paintFit(img, cam), wpf = want["paint_fit"]!
                    for k in LineColour.allCases where pf[k]!.isFinite {
                        XCTAssertEqual(pf[k]!, wpf[k.rawValue]!.double!, accuracy: 1e-9, "\(label) paint_fit \(k)")
                    }
                }
            }
            XCTAssertEqual(first, c["first_accept"]?.string, "\(name) first accepting model")
            // manual path: Python's corner taps → calibrateFromPoints → refineCamera (paint pick)
            let rf = c["refine"]!
            var taps: [String: SIMD2<Double>] = [:]
            if case .object(let o) = rf["taps"]! {
                for (k, v) in o { let a = v.array!.map { $0.double! }; taps[k] = SIMD2(a[0], a[1]) }
            }
            let tapped = try CourtCalibration.calibrateFromPoints(taps, size: size)
            let wt = rf["tapped"]!.cameraModel!
            XCTAssertLessThan(keypointAgreement(wt, tapped).max, 1e-4, "\(name) tapped camera")
            let pf = AutoCalibration.paintFit(img, tapped)
            for k in LineColour.allCases where pf[k]!.isFinite {
                XCTAssertEqual(pf[k]!, rf["paint_fit"]![k.rawValue]!.double!, accuracy: 1e-9, "\(name) tap paint_fit \(k)")
            }
            let refined = CourtCalibration.refineCamera(tapped, frames: [img])
            let wr = rf["refined"]!.cameraModel!
            let ra = keypointAgreement(wr, refined).max
            refMax = max(refMax, ra)
            XCTAssertLessThan(ra, 1e-3, "\(name) refined camera px")
            // the whole colour-model loop (+ paint check) on the median image alone
            let loop = c["loop"]!
            let r = CourtCalibration.calibrateFromFramesDetailed([img], size: size)
            XCTAssertEqual(r.diagnostics.reason.rawValue, loop["reason"]!.string!, "\(name) loop reason")
            XCTAssertEqual(r.diagnostics.lineColour?.rawValue, loop["line_colour"]?.string, "\(name) loop line_colour")
            XCTAssertEqual(r.camera == nil, loop["camera"]?.cameraModel == nil, "\(name) loop accept")
            if let cam = r.camera, let wcam = loop["camera"]?.cameraModel {
                XCTAssertLessThan(keypointAgreement(wcam, cam).max, wcam.dist[0] != 0 ? 1e-2 : 1e-3, "\(name) loop camera")
            }
            if first != nil { nAccept += 1 }
        }
        print("PARITY colour: \(cases.count) cases, \(nAccept) accepted by some model, kp max \(kpMax) px, refine kp max \(refMax) px")
    }
}
