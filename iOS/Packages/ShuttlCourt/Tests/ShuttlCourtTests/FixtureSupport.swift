import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlCourt

/// Loads a JSON fixture exported by backend/scripts/swift_fixtures/court.py.
func fixture(_ name: String) throws -> JSONValue {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"),
                            "missing fixture \(name).json — run `.venv/bin/python -m scripts.swift_fixtures.court` in backend/")
    return try ShuttlJSON.load(JSONValue.self, from: url)
}

func fixtureURL(_ name: String) -> URL? {
    let ext = (name as NSString).pathExtension, base = (name as NSString).deletingPathExtension
    return Bundle.module.url(forResource: base, withExtension: ext, subdirectory: "Fixtures")
}

/// Repository root (…/华北五省), derived from this file's location.
let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

extension JSONValue {
    var doubles: [Double] { (array ?? []).map { $0.double ?? .nan } }
    var floats: [Float] { (array ?? []).map { Float($0.double ?? .nan) } }
    var ints: [Int] { (array ?? []).map { Int($0.double ?? 0) } }
    var rows: [[Double]] { (array ?? []).map(\.doubles) }
    var int: Int? { double.map { Int($0) } }
    var mat3: simd_double3x3 { simd_double3x3(rowMajor: rows) }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        try ShuttlJSON.decoder().decode(T.self, from: ShuttlJSON.encoder().encode(self))
    }
}

import simd

extension CameraModel {
    /// Projected court keypoints (all 28), NaN where behind the camera.
    var keypointPixels: [SIMD2<Double>] { project(CourtModel.keypointsArray().points) }
}

/// Max / median distance between two cameras' projections of the court keypoints that lie
/// inside the (margin-extended) frame of `a`.
func keypointAgreement(_ a: CameraModel, _ b: CameraModel, margin: Double = 0) -> (max: Double, median: Double, n: Int) {
    let pa = a.keypointPixels, pb = b.keypointPixels
    var d: [Double] = []
    let w = Double(a.imageSize.width), h = Double(a.imageSize.height)
    for (u, v) in zip(pa, pb) where u.x.isFinite && u.y.isFinite && u.x >= -margin && u.y >= -margin
        && u.x <= w + margin && u.y <= h + margin {
        d.append(v.x.isFinite ? simd_distance(u, v) : .infinity)
    }
    guard !d.isEmpty else { return (.nan, .nan, 0) }
    return (d.max()!, Num.median(d), d.count)
}

/// Port of tests/test_court_calibration.calib_errors.
func calibErrors(_ truth: CameraModel, _ est: CameraModel) -> [String: Double] {
    let P = CourtModel.keypointsArray().points
    let w = Double(truth.imageSize.width), h = Double(truth.imageSize.height)
    var rep: [Double] = [], werrNear: [Double] = [], werrFar: [Double] = [], werr: [Double] = []
    for p in P {
        let uv = truth.project(p)
        guard uv.x.isFinite, uv.y.isFinite, uv.x >= 0, uv.y >= 0, uv.x < w, uv.y < h else { continue }
        rep.append(simd_distance(est.project(p), uv))
        let g = est.pixelToFloor(uv)
        let e = simd_distance(SIMD2(g.x, g.y), SIMD2(p.x, p.y))
        werr.append(e)
        if p.x < 0 { werrNear.append(e) } else { werrFar.append(e) }
    }
    return ["n_vis": Double(rep.count),
            "reproj_med": rep.isEmpty ? .infinity : Num.nanmedian(rep),
            "reproj_max": rep.filter { !$0.isNaN }.max() ?? .infinity,
            "near_med": werrNear.isEmpty ? 0 : Num.nanmedian(werrNear),
            "far_med": werrFar.isEmpty ? 0 : Num.nanmedian(werrFar),
            "world_max": werr.filter { !$0.isNaN }.max() ?? .infinity,
            "focal_rel": abs(est.focal / truth.focal - 1)]
}
