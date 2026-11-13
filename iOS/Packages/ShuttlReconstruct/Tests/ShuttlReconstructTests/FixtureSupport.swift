import Foundation
import XCTest
import ShuttlCore
import simd
@testable import ShuttlReconstruct

/// Loads a JSON fixture exported by backend/scripts/swift_fixtures/reconstruct.py.
/// Big fixtures are stored raw-deflate compressed as `<name>.json.deflate` (lossless).
func fixture(_ name: String) throws -> JSONValue {
    if let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures") {
        return try ShuttlJSON.load(JSONValue.self, from: url)
    }
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json.deflate", subdirectory: "Fixtures"),
                            "missing fixture \(name).json — run `.venv/bin/python -m scripts.swift_fixtures.reconstruct` in backend/")
    let raw = try (NSData(contentsOf: url).decompressed(using: .zlib)) as Data
    return try ShuttlJSON.decoder().decode(JSONValue.self, from: raw)
}

extension JSONValue {
    var d: Double { double ?? .nan }
    var optD: Double? { if case .null = self { return nil }; return double }
    var int: Int { Int(double ?? -1) }
    var optInt: Int? { if case .null = self { return nil }; return double.map { Int($0) } }
    var doubles: [Double] { (array ?? []).map { $0.double ?? .nan } }
    var rows: [[Double]] { (array ?? []).map(\.doubles) }
    var flat: [Double] { rows.flatMap { $0 } }
    var v2: [SIMD2<Double>] { rows.map(SIMD2.init(row:)) }
    var v3: [SIMD3<Double>] { rows.map(SIMD3.init(row:)) }
    var optV2: SIMD2<Double>? { if case .null = self { return nil }; return SIMD2(row: doubles) }
    var isNull: Bool { if case .null = self { return true }; return false }
    var bools: [Bool] { (array ?? []).map { $0.bool ?? false } }
    var str: String? { string }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        try ShuttlJSON.decoder().decode(T.self, from: ShuttlJSON.encoder().encode(self))
    }

    var camera: CameraModel { try! decode(CameraModel.self) }

    var prior: SegmentPrior {
        var p = SegmentPrior()
        p.startT = self["start_t"]?.optD
        p.startXYWorld = self["start_xy_world"]?.optV2
        p.startXYSigma = self["start_xy_sigma"]?.d ?? 0.5
        if let h = self["start_height"], !h.isNull { let a = h.doubles; p.startHeight = (a[0], a[1]) } else { p.startHeight = nil }
        if let s = self["expected_side"]?.str { p.expectedSide = CourtSide(rawValue: s) ?? .unknown }
        if let e = self["end"]?.str { p.end = SegmentPrior.End(rawValue: e) }
        p.endT = self["end_t"]?.optD
        p.endXYWorld = self["end_xy_world"]?.optV2
        p.endXYSigma = self["end_xy_sigma"]?.d ?? 0.5
        if let h = self["end_height"], !h.isNull { let a = h.doubles; p.endHeight = (a[0], a[1]) } else { p.endHeight = nil }
        p.restXYWorld = self["rest_xy_world"]?.optV2
        p.heightSigma = self["height_sigma"]?.d ?? 0.1
        p.dragLength = self["drag_length"]?.d ?? Physics.dragLength
        p.dragLogSigma = self["drag_log_sigma"]?.d ?? 0.08
        return p
    }

    var track: Track2D {
        Track2D(t: self["t"]!.doubles, xy: self["xy"]!.v2, conf: self["conf"]!.doubles,
                visible: self["visible"]!.bools, interpolated: self["interpolated"]!.bools)
    }

    var hits: [HitEvent] {
        (array ?? []).map { h in
            HitEvent(frame: h["frame"]!.int, t: h["t"]!.d, xyPx: SIMD2(row: h["xy_px"]!.doubles),
                     hitter: CourtSide(rawValue: h["hitter"]?.str ?? "unknown") ?? .unknown,
                     confidence: h["confidence"]?.d ?? 0, playerTrackID: h["player_track_id"]?.optInt)
        }
    }

    var players: [PlayerObservation] {
        (array ?? []).map { o in
            let b = o["bbox"]!.doubles
            return PlayerObservation(frame: o["frame"]!.int, trackID: o["track_id"]!.int,
                                     bbox: BBox(x1: b[0], y1: b[1], x2: b[2], y2: b[3]), score: o["score"]!.d,
                                     side: CourtSide(rawValue: o["side"]?.str ?? "unknown") ?? .unknown,
                                     footWorld: o["foot_world"]?.optV2)
        }
    }

    var meta: VideoMeta {
        VideoMeta(fps: self["fps"]!.d, width: self["width"]!.int, height: self["height"]!.int,
                  nFrames: self["n_frames"]!.int, timestamps: self["timestamps"]!.doubles,
                  rotationDeg: self["rotation_deg"]?.int ?? 0)
    }
}

/// |a - b| <= tol * max(1, |b|); non-finite Python values (null) must be non-finite in Swift.
func close(_ a: Double, _ b: Double, _ tol: Double) -> Bool {
    if !b.isFinite { return !a.isFinite }
    return abs(a - b) <= tol * max(1, abs(b))
}

func assertClose(_ a: Double, _ b: Double, tol: Double, _ msg: @autoclosure () -> String = "",
                 file: StaticString = #filePath, line: UInt = #line) {
    XCTAssert(close(a, b, tol), "\(a) vs \(b) (tol \(tol)) \(msg())", file: file, line: line)
}

func assertClose(_ a: [Double], _ b: [Double], tol: Double, _ msg: @autoclosure () -> String = "",
                 file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, "length \(msg())", file: file, line: line)
    var worst = 0.0, wi = -1
    for (i, (x, y)) in zip(a, b).enumerated() {
        if !close(x, y, tol) { let e = y.isFinite ? abs(x - y) / max(1, abs(y)) : .infinity; if e > worst { worst = e; wi = i } }
    }
    XCTAssert(wi < 0, "max rel err \(worst) at \(wi): \(wi >= 0 ? a[wi] : 0) vs \(wi >= 0 ? b[wi] : 0) (tol \(tol)) \(msg())",
              file: file, line: line)
}

func maxRelErr(_ a: [Double], _ b: [Double]) -> Double {
    zip(a, b).map { x, y in
        if !y.isFinite { return x.isFinite ? Double.infinity : 0 }
        return abs(x - y) / max(1, abs(y))
    }.max() ?? 0
}

/// A flat description of one Python segment for comparisons.
struct PySeg {
    let j: JSONValue
    var startFrame: Int { j["start_frame"]!.int }
    var endFrame: Int { j["end_frame"]!.int }
    var t0: Double { j["t0"]!.d }
    var p0: [Double] { j["p0"]!.doubles }
    var v0: [Double] { j["v0"]!.doubles }
    var endReason: String { j["end_reason"]?.str ?? "unknown" }
    var landingXY: SIMD2<Double>? { j["landing_xy"]?.optV2 }
    var landingT: Double? { j["landing_t"]?.optD }
    var netH: Double? { j["net_crossing_height"]?.optD }
    var netY: Double? { j["net_crossing_y"]?.optD }
    var frames: [Int] { (j["frames"]?.array ?? []).map(\.int) }
}

/// Field-by-field agreement of a Swift segment with its Python reference; returns
/// (p0/v0 max rel err, landing err m) and records failures.
@discardableResult
func compareSegment(_ s: ReconstructedSegment, _ py: JSONValue, tol: Double = 1e-6, landTol: Double = 1e-3,
                    checkValues: Bool = true,
                    _ msg: String, file: StaticString = #filePath, line: UInt = #line) -> (Double, Double) {
    let p = PySeg(j: py)
    let g = s.segment
    // discrete structure — always exact
    XCTAssertEqual(g.endReason.rawValue, p.endReason, "end_reason \(msg)", file: file, line: line)
    XCTAssertEqual(g.startFrame, p.startFrame, "start_frame \(msg)", file: file, line: line)
    XCTAssertEqual(g.endFrame, p.endFrame, "end_frame \(msg)", file: file, line: line)
    XCTAssertEqual(g.frames.count, p.frames.count, "n frames \(msg)", file: file, line: line)
    XCTAssertEqual(g.landingXY == nil, p.landingXY == nil, "landing presence \(msg)", file: file, line: line)
    XCTAssertEqual(g.netCrossingHeight == nil, p.netH == nil, "net crossing presence \(msg)", file: file, line: line)
    let e = maxRelErr(g.p0.row + g.v0.row, p.p0 + p.v0)
    if !checkValues {
        var l = 0.0
        if let a = g.landingXY, let b = p.landingXY { l = simd_length(a - b) }
        return (e, l)
    }
    XCTAssert(e <= tol, "p0/v0 rel err \(e) \(msg): \(g.p0.row + g.v0.row) vs \(p.p0 + p.v0)", file: file, line: line)
    assertClose(g.t0, p.t0, tol: 1e-9, "t0 \(msg)", file: file, line: line)
    var landErr = 0.0
    switch (g.landingXY, p.landingXY) {
    case (nil, nil): break
    case let (a?, b?):
        landErr = simd_length(a - b)
        XCTAssert(landErr <= landTol, "landing \(a) vs \(b) \(msg)", file: file, line: line)
    default: XCTFail("landing presence \(String(describing: g.landingXY)) vs \(String(describing: p.landingXY)) \(msg)", file: file, line: line)
    }
    if let a = g.netCrossingHeight, let b = p.netH { assertClose(a, b, tol: 1e-5, "net_h \(msg)", file: file, line: line) }
    if let a = g.netCrossingY, let b = p.netY { assertClose(a, b, tol: 1e-5, "net_y \(msg)", file: file, line: line) }
    assertClose(g.confidence, py["confidence"]!.d, tol: 1e-4, "confidence \(msg)", file: file, line: line)
    assertClose(g.rmsePx, py["rmse_px"]!.d, tol: 1e-5, "rmse \(msg)", file: file, line: line)
    XCTAssertEqual(s.fitDrag, py["fit_drag"]?.bool, "fit_drag \(msg)", file: file, line: line)
    XCTAssertEqual(s.nObs, py["n_obs"]?.int, "n_obs \(msg)", file: file, line: line)
    XCTAssertEqual(s.nTrimmed, py["n_trimmed"]?.int, "n_trimmed \(msg)", file: file, line: line)
    XCTAssertEqual(s.contactObserved, py["contact_observed"]?.bool, "contact_observed \(msg)", file: file, line: line)
    assertClose(s.tEnd, py["t_end"]!.d, tol: 1e-6, "t_end \(msg)", file: file, line: line)
    if let ls = py["landing_sigma_m"]?.optD {
        XCTAssertNotNil(s.landingSigmaM, "landing_sigma_m \(msg)", file: file, line: line)
        if let a = s.landingSigmaM { assertClose(a, ls, tol: 1e-3, "landing_sigma_m \(msg)", file: file, line: line) }
    } else {
        XCTAssertNil(s.landingSigmaM, "landing_sigma_m \(msg)", file: file, line: line)
    }
    if let ss = py["speed0_sigma_kmh"]?.optD, let a = s.speed0SigmaKmh {
        assertClose(a, ss, tol: 1e-3, "speed0_sigma_kmh \(msg)", file: file, line: line)
    }
    return (e, landErr)
}
