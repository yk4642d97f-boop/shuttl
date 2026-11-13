import XCTest
import ShuttlCore
@testable import ShuttlCourt

/// Release timing of the automatic path (`calibrateFromFramesDetailed`) on exported frame sets
/// (white accepts, white / colour-grid / R7 rejects) — also checks the decision equals Python's.
/// Frames: SHUTTL_COURT_TIMING=<dir> written by the timing exporter (RGB raw DEFLATE per frame +
/// manifest.json). Skipped without it. Run alone on the machine for meaningful numbers.
final class TimingTests: XCTestCase {
    func testCalibrationTiming() throws {
        guard let path = ProcessInfo.processInfo.environment["SHUTTL_COURT_TIMING"] else {
            throw XCTSkip("set SHUTTL_COURT_TIMING=<dir> to time the automatic path")
        }
        let dir = URL(fileURLWithPath: path)
        let cases = try XCTUnwrap(try ShuttlJSON.load(JSONValue.self, from: dir.appendingPathComponent("manifest.json"))["cases"]?.array)
        var rows: [String] = []
        for c in cases {
            let name = c["name"]!.string!, n = c["n"]!.int!
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            var frames: [ImageU8] = []
            for k in 0..<n {
                let raw = try Data(contentsOf: dir.appendingPathComponent(name).appendingPathComponent(String(format: "f%02d.deflate", k))) as NSData
                frames.append(ImageU8(width: size.width, height: size.height, channels: 3,
                                      data: [UInt8](try raw.decompressed(using: .zlib) as Data)))
            }
            let t0 = Date()
            let r = CourtCalibration.calibrateFromFramesDetailed(frames, size: size)
            let dt = Date().timeIntervalSince(t0)
            XCTAssertEqual(r.camera != nil, c["accepted"]!.bool!, "\(name) accept")
            XCTAssertEqual(r.diagnostics.reason.rawValue, c["reason"]!.string!, "\(name) reason")
            XCTAssertEqual(r.diagnostics.lineColour?.rawValue, c["line_colour"]?.string, "\(name) line_colour")
            rows.append(String(format: "%@ %@ %.2f s", name, r.diagnostics.reason.rawValue, dt))
        }
        print("TIMING calibrateFromFramesDetailed (release):\n  " + rows.joined(separator: "\n  "))
    }
}
