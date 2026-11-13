import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// Fixtures exported by backend/scripts/swift_fixtures/players.py (plain JSONSerialization:
/// the files are a few MB and `Any` access keeps the test code close to the Python dicts).
enum Fx {
    static func load(_ name: String) throws -> Any {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"),
                                "missing fixture \(name).json — run `.venv/bin/python -m scripts.swift_fixtures.players` in backend/")
        return try JSONSerialization.jsonObject(with: Data(contentsOf: url))
    }

    static func dict(_ x: Any?) -> [String: Any] { x as? [String: Any] ?? [:] }
    static func list(_ x: Any?) -> [Any] { x as? [Any] ?? [] }
    /// JSON number or null (→ NaN).
    static func num(_ x: Any?) -> Double { (x as? NSNumber)?.doubleValue ?? .nan }
    static func nums(_ x: Any?) -> [Double] { list(x).map(num) }
    static func rows(_ x: Any?) -> [[Double]] { list(x).map(nums) }
    static func v2(_ x: Any?) -> [SIMD2<Double>] { rows(x).map { SIMD2($0[0], $0[1]) } }
    static func ints(_ x: Any?) -> [Int] { list(x).map { ($0 as? NSNumber)?.intValue ?? 0 } }
    static func floatBits(_ x: Any?) -> [Float] { list(x).map { Float(bitPattern: UInt32(truncating: $0 as! NSNumber)) } }

    static func meta(_ x: Any?) -> VideoMeta {
        let d = dict(x)
        return VideoMeta(fps: num(d["fps"]), width: Int(num(d["width"])), height: Int(num(d["height"])),
                         nFrames: Int(num(d["n_frames"])), timestamps: nums(d["timestamps"]),
                         rotationDeg: Int(num(d["rotation_deg"])))
    }

    static func camera(_ x: Any?) throws -> CameraModel? {
        guard let d = x as? [String: Any] else { return nil }
        let data = try JSONSerialization.data(withJSONObject: d)
        return try ShuttlJSON.decoder().decode(CameraModel.self, from: data)
    }

    static func detections(_ raw: Any?) -> FrameDetections {
        let d = dict(raw)
        let boxes = rows(d["boxes"]).map { BBox(x1: $0[0], y1: $0[1], x2: $0[2], y2: $0[3]) }
        let kps = rows(d["kps"]).map { r in (0..<17).map { Keypoint(x: r[$0 * 3], y: r[$0 * 3 + 1], conf: r[$0 * 3 + 2]) } }
        return FrameDetections(boxes: boxes, scores: nums(d["scores"]), keypoints: kps)
    }

    static func observations(_ x: Any?) -> [[String: Any]] { list(x).map(dict) }

    static func side(_ s: Any?) -> CourtSide { CourtSide(rawValue: s as? String ?? "") ?? .unknown }
}

/// A fixture case rebuilt into Swift inputs.
struct PlayerCase {
    let raw: [String: Any]
    let name: String
    let meta: VideoMeta
    let camera: CameraModel?
    let doubles: Bool?
    let frames: [Int]
    let detections: [FrameDetections]
    /// Python's descriptors of the gated survivors, nil when the case had no image.
    let appearance: [[Float]?]

    init(_ x: Any) throws {
        raw = Fx.dict(x)
        name = raw["name"] as? String ?? "?"
        meta = Fx.meta(raw["meta"])
        camera = try Fx.camera(raw["camera"])
        doubles = raw["doubles"] as? Bool
        let rawDets = Fx.list(raw["raw"])
        frames = rawDets.map { Int(Fx.num(Fx.dict($0)["frame"])) }
        detections = rawDets.map(Fx.detections)
        appearance = Fx.list(raw["candidates"]).map { c in
            let bits = Fx.dict(c)["app_bits"]
            guard let rows = bits as? [Any] else { return nil }          // (N, 38) uint32 bit patterns
            return rows.flatMap { Fx.floatBits($0) }
        }
    }

    /// Candidates exactly as Python built them (descriptors injected).
    func candidates(config: PlayersConfig = .default) -> [PlayerCandidates] {
        zip(frames, detections).enumerated().map { i, fd in
            Candidates.build(frame: fd.0, detections: fd.1, camera: camera, imageSize: meta.size,
                             appearance: i < appearance.count ? appearance[i] : nil, config: config)
        }
    }
}

func assertClose(_ a: Double, _ b: Double, tol: Double, _ msg: @autoclosure () -> String = "",
                 file: StaticString = #filePath, line: UInt = #line) {
    if b.isNaN { XCTAssertTrue(a.isNaN, "\(a) vs NaN \(msg())", file: file, line: line); return }
    if b.isInfinite { XCTAssertEqual(a, b, msg(), file: file, line: line); return }
    XCTAssert(abs(a - b) <= tol, "\(a) vs \(b) (tol \(tol)) \(msg())", file: file, line: line)
}
