import XCTest
@testable import ShuttlCore

/// A damaged camera.json checkpoint must throw (so the stage re-runs), never trap.
final class CameraDecodeTests: XCTestCase {
    static let good = """
    {"K":[[1000,0,960],[0,1000,540],[0,0,1]],"R":[[1,0,0],[0,1,0],[0,0,1]],"t":[0,0,5],
     "image_size":[1920,1080],"dist":[0,0,0,0,0],"reproj_rmse_px":0.5,"confidence":0.9}
    """

    func decode(_ s: String) throws -> CameraModel {
        try ShuttlJSON.decoder().decode(CameraModel.self, from: Data(s.utf8))
    }

    func testGoodCameraDecodes() throws {
        let cam = try decode(Self.good)
        XCTAssertEqual(cam.t, SIMD3(0, 0, 5))
        XCTAssertEqual(cam.imageSize, ImageSize(width: 1920, height: 1080))
    }

    func testStructurallyWrongCamerasThrow() {
        let broken: [(String, String)] = [
            ("short t", #""t":[0,0,5]"# ),
            ("2x3 K", #""K":[[1000,0,960],[0,1000,540],[0,0,1]]"#),
            ("ragged R", #""R":[[1,0,0],[0,1,0],[0,0,1]]"#),
            ("short image_size", #""image_size":[1920,1080]"#),
            ("K not numbers", #""K":[[1000,0,960],[0,1000,540],[0,0,1]]"#),
        ]
        let replacements = [#""t":[1,2]"#,
                            #""K":[[1000,0,960],[0,1000,540]]"#,
                            #""R":[[1,0,0],[0,1],[0,0,1]]"#,
                            #""image_size":[1920]"#,
                            #""K":"garbage""#]
        for ((name, original), bad) in zip(broken, replacements) {
            let s = Self.good.replacingOccurrences(of: original, with: bad)
            XCTAssertNotEqual(s, Self.good, "\(name): replacement did not apply")
            XCTAssertThrowsError(try decode(s), name) { error in
                XCTAssertTrue(error is DecodingError, "\(name): \(error)")
            }
        }
        XCTAssertThrowsError(try decode("{}"))
        XCTAssertThrowsError(try decode("[1,2,3]"))
        XCTAssertThrowsError(try decode(String(Self.good.prefix(40))))
    }
}
