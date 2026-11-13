import XCTest
import CoreML
import CoreImage
import CoreVideo
import AVFoundation
@testable import Shuttl

final class MLInferenceEngineTests: XCTestCase {

    // MARK: Helpers

    /// BGRA buffer, black, with a red Gaussian blob (σ in full-res px) centred at `c` (pixel indices).
    private func blobBuffer(width: Int, height: Int, centre c: CGPoint, sigma: Double) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        XCTAssertEqual(CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs, &pb), kCVReturnSuccess)
        let buf = try XCTUnwrap(pb)
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(buf)).assumingMemoryBound(to: UInt8.self)
        let rb = CVPixelBufferGetBytesPerRow(buf)
        for y in 0..<height {
            for x in 0..<width {
                let d2 = pow(Double(x) - Double(c.x), 2) + pow(Double(y) - Double(c.y), 2)
                let v = UInt8((255 * exp(-d2 / (2 * sigma * sigma))).rounded())
                let p = base + y * rb + 4 * x
                p[0] = 0; p[1] = 0; p[2] = v; p[3] = 255
            }
        }
        return buf
    }

    private func planeCentroid(_ p: UnsafePointer<Float>, width: Int, height: Int) -> (x: Double, y: Double, mass: Double) {
        var sw = 0.0, sx = 0.0, sy = 0.0
        for y in 0..<height {
            for x in 0..<width {
                let w = Double(p[y * width + x])
                sw += w; sx += w * Double(x); sy += w * Double(y)
            }
        }
        return (sx / max(sw, 1e-12), sy / max(sw, 1e-12), sw)
    }

    // MARK: Preprocessing

    func testPreprocessorPlacesContentLikeWarpAffine() throws {
        let pre = try XCTUnwrap(FramePreprocessor(netWidth: 512, netHeight: 288))
        let out = FloatStorage(count: 3 * pre.planeCount)
        let n = pre.planeCount
        for (w, h, c) in [(1280, 720, CGPoint(x: 401.3, y: 207.8)), (720, 1280, CGPoint(x: 300.4, y: 700.2))] {
            let g = FrameGeometry(width: w, height: h, netWidth: 512, netHeight: 288, mode: .letterbox, peakOffset: 0.5)
            let frame = CameraFrame(pixelBuffer: try blobBuffer(width: w, height: h, centre: c, sigma: 6), timestamp: .zero)
            XCTAssertTrue(pre.render(frame, geometry: g, into: out.pointer))
            let expected = g.toNet(c)
            let r = planeCentroid(out.pointer, width: 512, height: 288)
            XCTAssertEqual(r.x, Double(expected.x), accuracy: 0.1, "\(w)x\(h) x")
            XCTAssertEqual(r.y, Double(expected.y), accuracy: 0.1, "\(w)x\(h) y (vertical flip?)")
            // RGB planes: the blob is pure red.
            XCTAssertGreaterThan(r.mass, 10)
            XCTAssertLessThan(planeCentroid(out.pointer + n, width: 512, height: 288).mass, 1e-3)
            XCTAssertLessThan(planeCentroid(out.pointer + 2 * n, width: 512, height: 288).mass, 1e-3)
            XCTAssertLessThanOrEqual(out.pointer[0], 1e-6)
        }
    }

    // MARK: Model

    func testBundledModelLoadsAndInputShapeMatches() async throws {
        guard let url = MLInferenceEngine.bundledModelURL() else {
            throw XCTSkip("ShuttleTracker model not bundled")
        }
        // Fixed stride: on the simulator's CPU a prediction takes ~0.3 s, so the adaptive stride
        // would (correctly) predict only every 3rd frame.
        let engine = MLInferenceEngine(modelURL: url, options: .init(computeUnits: .cpuOnly, fixedStride: 1))
        let loaded = await engine.load()
        XCTAssertTrue(loaded)
        let loadedSpec = await engine.spec
        let spec = try XCTUnwrap(loadedSpec)
        XCTAssertEqual(spec.name, "wasb")
        XCTAssertEqual(spec.seqLen, 3)
        XCTAssertEqual(spec.inChannels, 9)
        XCTAssertEqual(spec.inputWidth, 512)
        XCTAssertEqual(spec.inputHeight, 288)
        XCTAssertEqual(spec.centroidOffset, 0.5)
        XCTAssertEqual(spec.resize, .letterbox)
        XCTAssertTrue(spec.isConsistent)

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: url, configuration: config)
        let input = try XCTUnwrap(model.modelDescription.inputDescriptionsByName[spec.inputName]?.multiArrayConstraint)
        XCTAssertEqual(input.shape.map(\.intValue), [1, spec.inChannels, spec.inputHeight, spec.inputWidth])
        let output = try XCTUnwrap(model.modelDescription.outputDescriptionsByName[spec.outputName]?.multiArrayConstraint)
        XCTAssertEqual(output.shape.map(\.intValue), [1, spec.seqLen, spec.inputHeight, spec.inputWidth])

        // Three black frames: nothing until the window is full, then one prediction decodes all
        // three heatmaps; the next frame yields just its own. No detections; real timestamps.
        var perCall: [[FrameDetections]] = []
        for i in 0..<4 {
            let pb = try blobBuffer(width: 1280, height: 720, centre: CGPoint(x: -100, y: -100), sigma: 1)
            let ts = CMTime(value: CMTimeValue(1000 + 20 * i), timescale: 600)
            perCall.append(await engine.process(CameraFrame(pixelBuffer: pb, timestamp: ts)))
        }
        XCTAssertEqual(perCall.map(\.count), [0, 0, 3, 1])
        let results = perCall.flatMap { $0 }
        XCTAssertEqual(results.map(\.frameIndex), [0, 1, 2, 3])
        XCTAssertEqual(results.map(\.timestamp), (0..<4).map { CMTime(value: CMTimeValue(1000 + 20 * $0), timescale: 600) })
        XCTAssertTrue(results.allSatisfy { $0.imageSize == CGSize(width: 1280, height: 720) && $0.detections.isEmpty })
    }

    func testMissingModelIsANoOp() async throws {
        let engine = MLInferenceEngine(modelURL: URL(fileURLWithPath: "/nonexistent/ShuttleTracker.mlpackage"))
        let pb = try blobBuffer(width: 64, height: 36, centre: .zero, sigma: 1)
        let out = await engine.process(CameraFrame(pixelBuffer: pb, timestamp: .zero))
        XCTAssertTrue(out.isEmpty)
        let available = await engine.isAvailable
        XCTAssertFalse(available)
    }

    // MARK: Real footage (held-out TrackNetV2 Test clip; skipped when the dataset is absent)

    private static let dataRoot = "/path/to/shuttl/backend/ml/data/raw/shuttlecock_trajectory"

    func testRealClipDetectionAndTracking() async throws {
        let video = URL(fileURLWithPath: Self.dataRoot + "/TrackNetV2/Test/match1/video/1_05_02.mp4")
        let labels = URL(fileURLWithPath: Self.dataRoot + "/corrected/match1/corrected_csv/1_05_02_ball.csv")
        guard let modelURL = MLInferenceEngine.bundledModelURL(),
              FileManager.default.fileExists(atPath: video.path),
              let csv = try? String(contentsOf: labels, encoding: .utf8) else {
            throw XCTSkip("model or evaluation clip not available")
        }
        var truth: [Int: CGPoint?] = [:]
        for line in csv.split(separator: "\n").dropFirst() {
            let f = line.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard f.count == 4 else { continue }
            truth[Int(f[0])] = f[1] > 0 ? CGPoint(x: f[2], y: f[3]) : nil
        }

        let asset = AVURLAsset(url: video)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let reader = try AVAssetReader(asset: asset)
        // Same pixel format as the camera (420v).
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ])
        reader.add(output)
        XCTAssertTrue(reader.startReading())

        // TrackNetV3 evaluation range of this clip starts at frame 28 (drop_frame.json).
        let evalRange = 28..<208
        let engine = MLInferenceEngine(modelURL: modelURL, options: .init(computeUnits: .cpuOnly, fixedStride: 1))
        let tracker = TrajectoryTracker()
        var errors: [Double] = []
        var visible = 0, found = 0, falsePositives = 0, invisible = 0, trackedNear = 0
        var frame = 0
        while frame < evalRange.upperBound, let sb = output.copyNextSampleBuffer() {
            defer { frame += 1 }
            guard let cf = CameraFrame(sampleBuffer: sb) else { continue }
            for r in await engine.process(cf) {
                let snap = tracker.update(r)
                guard evalRange.contains(r.frameIndex), let gt = truth[r.frameIndex] else { continue }
                let top = r.detections.first(where: { $0.confidence >= 0.5 })
                if let gt {
                    visible += 1
                    if let d = top {
                        let e = hypot(Double(d.pixel.x - gt.x), Double(d.pixel.y - gt.y))
                        errors.append(e)
                        if e <= 10 { found += 1 }
                    }
                    if snap.status == .tracking, let head = snap.trail.last,
                       hypot(Double(head.x) * 1280 - Double(gt.x), Double(head.y) * 720 - Double(gt.y)) <= 10 {
                        trackedNear += 1
                    }
                } else {
                    invisible += 1
                    if top != nil { falsePositives += 1 }
                }
            }
        }
        reader.cancelReading()
        let stats = await engine.stats
        errors.sort()
        let median = errors.isEmpty ? .infinity : errors[errors.count / 2]
        let p90 = errors.isEmpty ? .infinity : errors[Int(Double(errors.count - 1) * 0.9)]
        let recall = Double(found) / Double(max(visible, 1))
        let tracked = Double(trackedNear) / Double(max(visible, 1))
        print(String(format: "REALCLIP 1_05_02 frames %d visible %d recall@10px %.3f median %.2f px p90 %.2f px "
                     + "FP %d/%d tracked@10px %.3f predict %.1f ms preprocess %.1f ms",
                     evalRange.count, visible, recall, median, p90, falsePositives, invisible, tracked,
                     stats.predictionSeconds * 1000, stats.preprocessSeconds * 1000))
        XCTAssertEqual(stats.skipped, 0)
        XCTAssertGreaterThan(visible, 100)
        XCTAssertGreaterThan(recall, 0.85)
        XCTAssertLessThan(median, 3.0)
        XCTAssertGreaterThan(tracked, 0.7)
    }
}
