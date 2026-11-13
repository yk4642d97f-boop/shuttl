import CoreML
import XCTest
import simd
import ShuttlCore
import ShuttlPlayers

/// Parity tests for the on-device CoreML RTMPose-m port (REPORT.md section 5).
///
/// The fixtures in `Fixtures/rtmpose/` were generated from `research/coreml/real_crops/` with the
/// **onnxruntime reference** the CoreML port was verified against (max diff 5.7e-6, 100% argmax
/// agreement): `generate_fixture.py` runs the exact rtmlib chain (pad 10% -> `bbox_xyxy2cs(1.25)`
/// -> aspect fix to 192:256 -> `get_warp_matrix` -> `warpAffine INTER_LINEAR` -> normalise ->
/// ONNX -> SimCC decode -> map to frame) and records the keypoints it produced.
///
/// These assertions are the contract. A model that returns plausible-looking but wrong
/// keypoints is worse than no model, so each of the classic mistakes — the split ratio, the
/// transposed axes, the 192w x 256h frame — shifts every keypoint past the tolerance here.
final class RTMPoseParityTests: XCTestCase {

    /// < 1 px: one model bin is 0.5 px, and the CoreML/ONNX difference is ~1 grey level, so
    /// agreement to well under a pixel is the tightest honest bar (the reference itself is
    /// 0.358 px mean from rtmlib). A split-ratio or axis error is tens of px.
    private static let maxPxError = 1.0

    private static var fixtureDir: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/rtmpose")
    }
    private static var coremlDir: URL { Self.fixtureDir.deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("research/coreml") }

    // MARK: Fixture loading

    struct Fixture: Decodable {
        struct Case: Decodable {
            var image: String
            var warp_image: String
            var image_size: [Int]
            var box: [Double]
            var center: [Double]
            var scale: [Double]
            var keypoints: [[Double]]
            var confidence: [Double]
        }
        struct DecodeCase: Decodable {
            var box: [Double]
            var image_size: [Int]
            var center: [Double]
            var scale: [Double]
            var simcc_x: String
            var simcc_y: String
            var keypoints: [[Double]]
            var confidence: [Double]
        }
        var cases: [Case]
        var decode_cases: [DecodeCase]
    }

    private func loadFixture() throws -> Fixture {
        let url = Self.fixtureDir.appendingPathComponent("parity.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("fixture missing at \(url.path)")
        }
        return try JSONDecoder().decode(Fixture.self, from: try Data(contentsOf: url))
    }

    /// Reads a binary PPM (P6) into interleaved RGB bytes. PPM is used instead of PNG because
    /// CoreGraphics colour-manages a PNG and OpenCV's read does not; the raw raster is identical
    /// on both sides, so the test measures the model, not the decoder.
    static func readPPM(_ url: URL) throws -> (rgb: [UInt8], width: Int, height: Int) {
        let bytes = [UInt8](try Data(contentsOf: url))
        var idx = 0
        func token() -> String? {
            while idx < bytes.count {
                let c = bytes[idx]
                if c == 0x23 {                                   // '#' comment to end of line
                    while idx < bytes.count && bytes[idx] != 0x0A { idx += 1 }
                    continue
                }
                if c == 0x20 || c == 0x0A || c == 0x0D || c == 0x09 { idx += 1; continue }
                break
            }
            var s = ""
            while idx < bytes.count {
                let c = bytes[idx]
                if c == 0x20 || c == 0x0A || c == 0x0D || c == 0x09 { idx += 1; break }
                s.append(Character(UnicodeScalar(c))); idx += 1
            }
            return s.isEmpty ? nil : s
        }
        guard token() == "P6", let wTok = token(), let hTok = token(), token() == "255",
              let w = Int(wTok), let h = Int(hTok) else {
            throw NSError(domain: "ppm", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "not a P6 PPM: \(url.lastPathComponent)"])
        }
        guard idx + w * h * 3 <= bytes.count else {
            throw NSError(domain: "ppm", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "truncated PPM \(url.lastPathComponent)"])
        }
        return (Array(bytes[idx..<(idx + w * h * 3)]), w, h)
    }

    private static func decodeFloat(_ b64: String) -> [Float] {
        guard let data = Data(base64Encoded: b64) else { return [] }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// The bundled compiled model, or the repo's `.mlpackage` compiled on the fly (the tests run
    /// on the simulator with the app bundle, but a plain `swift test` has neither).
    private func loadModel() async throws -> RTMPoseCoreML {
        if let url = RTMPoseCoreML.bundledModelURL(), url.pathExtension == "mlmodelc",
           let m = RTMPoseCoreML.loadCompiled(url: url) {
            return m
        }
        let pkg = Self.coremlDir.appendingPathComponent("RTMPoseM.mlpackage")
        guard FileManager.default.fileExists(atPath: pkg.path) else {
            throw XCTSkip("RTMPoseM model not available (bundle or \(pkg.path))")
        }
        return try await RTMPoseCoreML.load(modelURL: pkg)
    }

    // MARK: The parity assertion

    private func warp(_ ppm: (rgb: [UInt8], width: Int, height: Int), box: BBox) -> (nchw: [Float], center: SIMD2<Double>, scale: SIMD2<Double>, rgb: [UInt8]) {
        let (center, scale) = RTMPoseGeometry.centerScale(box: box, imageSize: ImageSize(width: ppm.width, height: ppm.height))
        let matrix = RTMPoseGeometry.warpMatrix(center: center, scale: scale)
        let nchw = ppm.rgb.withUnsafeBufferPointer { buf -> [Float] in
            let view = PosePixelView(base: buf.baseAddress!, width: ppm.width, height: ppm.height,
                                     rowBytes: ppm.width * 3, pixelStride: 3, r: 0, g: 1, b: 2)
            return RTMPoseGeometry.modelInput(src: view, matrix: matrix)
        }
        // back to interleaved bytes for comparison against cv2's warp
        let plane = 192 * 256
        var rgb = [UInt8](repeating: 0, count: plane * 3)
        for i in 0..<plane { for c in 0..<3 { rgb[i * 3 + c] = UInt8(nchw[c * plane + i]) } }
        return (nchw, center, scale, rgb)
    }

    /// Stage 1 — the Swift affine warp against cv2.warpAffine's own output for the same box.
    func testWarpMatchesOpenCV() throws {
        let fixture = try loadFixture()
        var worst = 0, sum = 0.0, n = 0
        for (ci, c) in fixture.cases.enumerated() {
            let ppm = try Self.readPPM(Self.fixtureDir.appendingPathComponent(c.image))
            let ref = try Self.readPPM(Self.fixtureDir.appendingPathComponent(c.warp_image))
            XCTAssertEqual(ref.width, 192); XCTAssertEqual(ref.height, 256)
            let w = warp(ppm, box: BBox(x1: c.box[0], y1: c.box[1], x2: c.box[2], y2: c.box[3]))
            var cw = 0
            for i in 0..<ref.rgb.count {
                let d = abs(Int(w.rgb[i]) - Int(ref.rgb[i]))
                cw = max(cw, d); sum += Double(d); n += 1
            }
            worst = max(worst, cw)
            print("[RTMPose parity] warp case \(ci): max |dRGB| = \(cw)")
        }
        print("[RTMPose parity] warp: max |dRGB| \(worst), mean \(String(format: "%.4f", sum / Double(max(n, 1))))")
        XCTAssertLessThanOrEqual(worst, 1, "Swift warp must match cv2 4.11 (the version the fixtures are built with) to 1 grey level")
    }

    /// Stage 2 — the model alone: cv2's own warped crop in, reference keypoints out. This
    /// separates model/decode error from warp error.
    func testModelOnReferenceWarpMatchesReference() async throws {
        let fixture = try loadFixture()
        let model = try await loadModel()
        var worst = 0.0, total = 0, same = 0
        for (ci, c) in fixture.cases.enumerated() {
            let ref = try Self.readPPM(Self.fixtureDir.appendingPathComponent(c.warp_image))
            let plane = 192 * 256
            var nchw = [Float](repeating: 0, count: 3 * plane)
            for i in 0..<plane { for k in 0..<3 { nchw[k * plane + i] = Float(ref.rgb[i * 3 + k]) } }
            let kp = try model.keypoints(fromNCHW: nchw, center: SIMD2(c.center[0], c.center[1]),
                                         scale: SIMD2(c.scale[0], c.scale[1]))
            for k in 0..<17 where c.confidence[k] > 0 {
                let g = c.keypoints[k]
                let e = ((kp[k].x - g[0]) * (kp[k].x - g[0]) + (kp[k].y - g[1]) * (kp[k].y - g[1])).squareRoot()
                total += 1; if e <= 0.75 { same += 1 }
                worst = max(worst, e)
                XCTAssertLessThan(e, 0.75, "case \(ci) kp \(k) on the reference warp: \(e) px")
            }
        }
        print("[RTMPose parity] model on reference warp: max px \(String(format: "%.4f", worst)), within 0.75 px \(same)/\(total)")
    }

    /// Full-path parity: frame + box -> Swift warp + CoreML + decode -> keypoints, vs the
    /// reference chain's keypoints. Asserts the same argmax per keypoint (within one bin) and
    /// max pixel error under 2 px.
    func testFullPathParityAgainstReference() async throws {
        let fixture = try loadFixture()
        let model = try await loadModel()
        var worstPx = 0.0
        var worstCase = ""
        var totalKP = 0, agreed = 0
        var maxConfDelta = 0.0

        for (ci, c) in fixture.cases.enumerated() {
            let ppm = try Self.readPPM(Self.fixtureDir.appendingPathComponent(c.image))
            let size = ImageSize(width: c.image_size[0], height: c.image_size[1])
            XCTAssertEqual(ppm.width, c.image_size[0])
            XCTAssertEqual(ppm.height, c.image_size[1])
            let box = BBox(x1: c.box[0], y1: c.box[1], x2: c.box[2], y2: c.box[3])

            // Swift's own centre/scale + warp (must equal the fixture's, as a sanity check).
            let (center, scale) = RTMPoseGeometry.centerScale(box: box, imageSize: size)
            XCTAssertEqual(center.x, c.center[0], accuracy: 1e-3, "centre.x case \(ci)")
            XCTAssertEqual(center.y, c.center[1], accuracy: 1e-3, "centre.y case \(ci)")
            XCTAssertEqual(scale.x, c.scale[0], accuracy: 1e-3, "scale.x case \(ci)")
            XCTAssertEqual(scale.y, c.scale[1], accuracy: 1e-3, "scale.y case \(ci)")

            let matrix = RTMPoseGeometry.warpMatrix(center: center, scale: scale)
            let nchw = ppm.rgb.withUnsafeBufferPointer { buf -> [Float] in
                let view = PosePixelView(base: buf.baseAddress!, width: ppm.width, height: ppm.height,
                                         rowBytes: ppm.width * 3, pixelStride: 3, r: 0, g: 1, b: 2)
                return RTMPoseGeometry.modelInput(src: view, matrix: matrix)
            }
            let kp = try model.keypoints(fromNCHW: nchw, center: center, scale: scale)

            XCTAssertEqual(kp.count, 17)
            for k in 0..<17 {
                let g = c.keypoints[k]
                let conf = c.confidence[k]
                if conf <= 0 {
                    // Reference marked it invalid; the port must too.
                    XCTAssertEqual(kp[k].x, -1, accuracy: 1e-9, "invalid kp \(k) case \(ci)")
                    XCTAssertEqual(kp[k].y, -1, accuracy: 1e-9, "invalid kp \(k) case \(ci)")
                    continue
                }
                let dx = kp[k].x - g[0], dy = kp[k].y - g[1]
                let e = (dx * dx + dy * dy).squareRoot()
                totalKP += 1
                if e <= 0.75 { agreed += 1 }
                if e > worstPx { worstPx = e; worstCase = "case \(ci) kp \(k)" }
                maxConfDelta = max(maxConfDelta, abs(kp[k].conf - conf))
                XCTAssertLessThan(e, Self.maxPxError,
                                  "\(worstCase): got (\(kp[k].x), \(kp[k].y)) want (\(g[0]), \(g[1]))")
            }
        }
        print("[RTMPose parity] full path: \(fixture.cases.count) cases, \(totalKP) keypoints, "
              + "max px error \(String(format: "%.4f", worstPx)) at \(worstCase), "
              + "within 0.75 px \(agreed)/\(totalKP), max conf delta \(String(format: "%.3e", maxConfDelta))")
        XCTAssertEqual(agreed, totalKP, "every valid keypoint must match within 0.75 px (one argmax bin is 0.5 px)")
        XCTAssertLessThan(maxConfDelta, 0.05, "confidence must agree with the reference")
    }

    /// Decode-only parity: fixed logits in, keypoints out. Isolates the SimCC maths from both the
    /// warp and CoreML, and pins the confidence rule on a synthetic all-negative case.
    func testDecodeParityIncludingConfidenceRule() throws {
        let fixture = try loadFixture()
        var totalKP = 0, agreed = 0
        var worstPx = 0.0

        for (ci, d) in fixture.decode_cases.enumerated() {
            let sx = Self.decodeFloat(d.simcc_x)
            let sy = Self.decodeFloat(d.simcc_y)
            XCTAssertEqual(sx.count, 17 * 384, "simcc_x length case \(ci)")
            XCTAssertEqual(sy.count, 17 * 512, "simcc_y length case \(ci)")
            let center = SIMD2<Double>(d.center[0], d.center[1])
            let scale = SIMD2<Double>(d.scale[0], d.scale[1])
            let kp = RTMPoseGeometry.decode(simccX: sx, simccY: sy, center: center, scale: scale)

            for k in 0..<17 {
                let g = d.keypoints[k]
                let conf = d.confidence[k]
                if conf <= 0 {
                    XCTAssertEqual(kp[k].x, -1, "confidence \(conf) <= 0 must give location -1")
                    XCTAssertEqual(kp[k].y, -1)
                    XCTAssertLessThanOrEqual(kp[k].conf, 0)
                    continue
                }
                let e = ((kp[k].x - g[0]) * (kp[k].x - g[0]) + (kp[k].y - g[1]) * (kp[k].y - g[1])).squareRoot()
                totalKP += 1
                if e <= 0.51 { agreed += 1 }              // same argmax bin
                if e > worstPx { worstPx = e }
                XCTAssertLessThan(e, 0.51 + 1e-9, "decode case \(ci) kp \(k): same bin required")
                XCTAssertEqual(kp[k].conf, conf, accuracy: 1e-5)
            }
        }
        print("[RTMPose parity] decode: \(fixture.decode_cases.count) cases, \(totalKP) keypoints, "
              + "max px error \(String(format: "%.4f", worstPx)), same-bin \(agreed)/\(totalKP)")
        XCTAssertEqual(agreed, totalKP)
    }

    /// The 192w x 256h frame and the split ratio, checked directly against a hand-built peak: a
    /// single bin lit at a known index must map to the matching frame pixel. Catches a /2 that
    /// was dropped (2x scale) or a transposed frame (192/256 swapped).
    func testSplitRatioAndFrameOrientation() {
        let binsX = 384, binsY = 512
        let center = SIMD2<Double>(100.0, 200.0)
        let scale = SIMD2<Double>(200.0, 200.0 * 256.0 / 192.0)
        // Light bin 100 in x and bin 300 in y.
        var sx = [Float](repeating: 0, count: 17 * binsX)
        var sy = [Float](repeating: 0, count: 17 * binsY)
        for k in 0..<17 { sx[k * binsX + 100] = 0.5; sy[k * binsY + 300] = 0.5 }
        let kp = RTMPoseGeometry.decode(simccX: sx, simccY: sy, center: center, scale: scale)
        // pixel_in_model = bin / 2  ->  (50, 150); then /(192,256)*scale + center - scale/2
        let expectX = 50.0 / 192.0 * scale.x + center.x - scale.x / 2
        let expectY = 150.0 / 256.0 * scale.y + center.y - scale.y / 2
        for k in 0..<17 {
            XCTAssertEqual(kp[k].x, expectX, accuracy: 1e-9)
            XCTAssertEqual(kp[k].y, expectY, accuracy: 1e-9)
            XCTAssertEqual(kp[k].conf, 0.5, accuracy: 1e-6)   // (0.5 + 0.5)/2
        }
        // Sanity: the two axes are not interchangeable (a transposed frame would swap them).
        XCTAssertNotEqual(expectX, expectY, accuracy: 1e-6)
        XCTAssertNotEqual(kp[0].x, kp[0].y, accuracy: 1e-6)
    }

    /// The model's I/O must be the verified contract, and it must load on the ANE.
    func testModelContractShapesAndComputeUnit() async throws {
        let model = try await loadModel()
        XCTAssertEqual(model.computeUnits, .cpuAndNeuralEngine)
        let desc = model.model.modelDescription
        let inShape = desc.inputDescriptionsByName.first?.value.multiArrayConstraint?.shape.map(\.intValue)
        XCTAssertEqual(inShape, [1, 3, 256, 192], "input must be 1x3x256x192")
        XCTAssertNotNil(desc.outputDescriptionsByName["simcc_x"])
        XCTAssertNotNil(desc.outputDescriptionsByName["simcc_y"])
        // Raw 0..255 contract recorded in the package metadata.
        XCTAssertTrue((model.metadata["input_contract"] ?? "").contains("raw 0..255"),
                      "shipped model must declare the raw input contract")
    }

    /// The seam: the default pose detector is RTMPose when the compiled model is in the bundle,
    /// and falls back to Apple Vision when it is not (a build without the 54 MB resource must
    /// still analyse). The test host app bundles the model, so `.main` selects RTMPose.
    func testDefaultPoseDetectorPrefersRTMPoseAndFallsBackToVision() throws {
        guard RTMPoseCoreML.bundledModelURL(in: .main) != nil else {
            throw XCTSkip("RTMPoseM is not in the test host bundle")
        }
        XCTAssertTrue(PlayerTracker.defaultPoseDetector(bundle: .main) is RTMPoseDetector,
                      "with the model bundled the default pose must be RTMPose")
        // A bundle without the model (the test bundle itself) must give Apple Vision.
        let empty = Bundle(for: RTMPoseParityTests.self)
        XCTAssertNil(RTMPoseCoreML.bundledModelURL(in: empty))
        XCTAssertTrue(PlayerTracker.defaultPoseDetector(bundle: empty) is VisionPoseDetector,
                      "without the model the default pose must stay Apple Vision")
    }
}
