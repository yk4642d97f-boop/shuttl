import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// Stage-internal references from appearance.json / misc.json.
final class UnitParityTests: XCTestCase {
    /// Deterministic BGR pattern of the exporter, as an RGB ImageU8.
    static func pattern(_ w: Int, _ h: Int) -> ImageU8 {
        var img = ImageU8(width: w, height: h, channels: 3)
        for y in 0..<h {
            for x in 0..<w {
                img[x, y, 2] = UInt8((x * 7 + y * 13) % 256)       // B
                img[x, y, 1] = UInt8((x * 3 + y * 29) % 256)       // G
                img[x, y, 0] = UInt8((x * 11 + y * 5) % 256)       // R
            }
        }
        return img
    }

    func testOpenCVHSVBitExact() throws {
        let fx = Fx.dict(try Fx.load("appearance"))
        let bgr = Fx.rows(fx["hsv_bgr"]), hsv = Fx.rows(fx["hsv"])
        for (c, e) in zip(bgr, hsv) {
            let r = OpenCVHSV.hsv(r: Int(c[2]), g: Int(c[1]), b: Int(c[0]))
            XCTAssertEqual([r.h, r.s, r.v], e.map { Int($0) }, "BGR \(c)")
        }
        // saturation for every (v, diff): the part OpenCV computes in float
        for (v, row) in Fx.rows(fx["s_table"]).enumerated() {
            for (d, s) in row.enumerated() {
                XCTAssertEqual(OpenCVHSV.hsv(r: v - d, g: v - d, b: v).s, Int(s), "v \(v) diff \(d)")
            }
        }
    }

    func testAppearanceDescriptorsBitExact() throws {
        let fx = Fx.dict(try Fx.load("appearance"))
        let pat = Fx.dict(fx["pattern"])
        let img = Self.pattern(Int(Fx.num(pat["width"])), Int(Fx.num(pat["height"])))
        let boxes = Fx.rows(fx["boxes"]).map { BBox(x1: $0[0], y1: $0[1], x2: $0[2], y2: $0[3]) }
        let kps = KeypointArray(values: Fx.rows(fx["kps"]).flatMap { $0 })
        let desc = AppearanceDescriptor.descriptors(img, boxes: boxes, kps: kps)
        let exp = Fx.list(fx["descriptor_bits"]).map(Fx.floatBits)
        for (i, e) in exp.enumerated() {
            let got = Array(desc[(i * Appearance.dim)..<((i + 1) * Appearance.dim)])
            XCTAssertEqual(got.map(\.bitPattern), e.map(\.bitPattern), "descriptor \(i)")
        }
        for r in Fx.list(fx["regions"]).map(Fx.dict) {
            let q = Fx.nums(r["rect"])
            let h = AppearanceDescriptor.regionHist(img, q[0], q[1], q[2], q[3])
            XCTAssertEqual(h.map(\.bitPattern), Fx.floatBits(r["hist_bits"]).map(\.bitPattern), "region \(q)")
        }
        let dist = Fx.rows(fx["distances"])
        for i in 0..<exp.count {
            for j in 0..<exp.count {
                let d = AppearanceDescriptor.distance(desc[(i * 38)..<((i + 1) * 38)], desc[(j * 38)..<((j + 1) * 38)])
                XCTAssertEqual(d, dist[i][j], accuracy: 1e-15, "distance \(i) \(j)")
            }
        }
        XCTAssertEqual(AppearanceDescriptor.distance(desc[0..<38], nil), 0)
        let rows = [0, 1, 4].map { desc[($0 * 38)..<(($0 + 1) * 38)] }
        let m = try XCTUnwrap(AppearanceDescriptor.mean(rows))
        XCTAssertEqual(m.map(\.bitPattern), Fx.floatBits(fx["mean_014_bits"]).map(\.bitPattern), "np.mean axis 0")
        XCTAssertEqual(AppearanceDescriptor.distance(m[...], desc[(5 * 38)..<(6 * 38)]), Fx.num(fx["mean_distance"]),
                       accuracy: 1e-15)
    }

    func testMergeROI() throws {
        let m = Fx.dict(Fx.dict(try Fx.load("misc"))["merge"])
        let kp = [Keypoint](repeating: Keypoint(x: 0, y: 0, conf: 0), count: 17)
        func dets(_ b: Any?, _ s: Any?) -> FrameDetections {
            let bs = Fx.rows(b).map { BBox(x1: $0[0], y1: $0[1], x2: $0[2], y2: $0[3]) }
            return FrameDetections(boxes: bs, scores: Fx.nums(s), keypoints: bs.map { _ in kp })
        }
        let r = Fx.ints(m["roi"])
        let out = mergeROI(full: dets(m["full_boxes"], m["full_scores"]), crop: dets(m["crop_boxes"], m["crop_scores"]),
                           roi: PixelRect(x1: r[0], y1: r[1], x2: r[2], y2: r[3]), imageSize: ImageSize(width: 1280, height: 720))
        XCTAssertEqual(out.boxes.map { [$0.x1, $0.y1, $0.x2, $0.y2] }, Fx.rows(m["out_boxes"]))
        XCTAssertEqual(out.scores, Fx.nums(m["out_scores"]))
        // keypoints shifted to frame px
        XCTAssertEqual(out.keypoints.last?.first?.x, Double(r[0]))
    }

    func testRTSSmoother() throws {
        let m = Fx.dict(Fx.dict(try Fx.load("misc"))["rts"])
        let out = PlayerViterbi.rtsSmooth(t: Fx.nums(m["t"]), z: Fx.v2(m["z"]), r: Fx.nums(m["r"]))
        let exp = Fx.v2(m["out"])
        XCTAssertEqual(out.count, exp.count)
        var worst = 0.0
        for (a, b) in zip(out, exp) { worst = max(worst, abs(a.x - b.x), abs(a.y - b.y)) }
        XCTAssertLessThanOrEqual(worst, 1e-12, "RTS smoother")
    }

    func testMovementStatsReference() throws {
        let m = Fx.dict(Fx.dict(try Fx.load("misc"))["movement"])
        let obs = Fx.observations(m["obs"]).map { e -> PlayerObservation in
            let fw = Fx.nums(e["foot_world"])
            return PlayerObservation(frame: Int(Fx.num(e["frame"])), trackID: Int(Fx.num(e["track_id"])),
                                     bbox: BBox(x1: 0, y1: 0, x2: 0, y2: 0), score: 0.9, side: Fx.side(e["side"]),
                                     footWorld: SIMD2(fw[0], fw[1]))
        }
        let ts = Fx.nums(m["timestamps"])
        compareStats(Movement.stats(obs, side: .near, timestamps: ts, fps: 30), Fx.dict(m["near"]), "near")
        compareStats(Movement.stats(obs, side: .far, timestamps: ts, fps: 30), Fx.dict(m["far"]), "far")
        compareStats(Movement.stats(obs, side: .near, timestamps: nil, fps: 30), Fx.dict(m["near_nots"]), "near no ts")
        compareStats(Movement.stats([], side: .near, timestamps: nil, fps: nil), Fx.dict(m["empty"]), "empty")
    }

    func testFarROIAndSitting() throws {
        let misc = Fx.dict(try Fx.load("misc"))
        let f = Fx.dict(misc["far_roi"])
        let cam = try XCTUnwrap(try Fx.camera(f["camera"]))
        let roi = try XCTUnwrap(FarROI.compute(camera: cam, size: cam.imageSize))
        XCTAssertEqual([roi.rect.x1, roi.rect.y1, roi.rect.x2, roi.rect.y2], Fx.ints(f["rect"]))
        XCTAssertEqual(roi.baselineHeightPx, Fx.num(f["height_px"]), accuracy: 1e-9)
        XCTAssertEqual(FarROI.yoloImgsz(roi.rect), Int(Fx.num(f["imgsz"])))
        let s = Fx.dict(misc["sitting"])
        let got = PlayerGeometry.sittingScore(kps: KeypointArray(values: Fx.rows(s["kps"]).flatMap { $0 }))
        XCTAssertEqual(got, Fx.nums(s["out"]))
        for (fps, step) in Fx.dict(misc["player_step"]) {
            XCTAssertEqual(playerStep(fps: Double(fps)!), Int(Fx.num(step)), "fps \(fps)")
        }
    }

    func testNumpyHelpers() {
        XCTAssertEqual(NP.pyRound(2.675, 2), 2.67)          // binary 2.67499…
        XCTAssertEqual(NP.pyRound(0.125, 2), 0.12)          // exact tie → even
        XCTAssertEqual(NP.pyRound(0.375, 2), 0.38)
        XCTAssertEqual(NP.npRound(0.125, 2), 0.12)
        XCTAssertEqual(NP.percentile([1, 2, 3, 4], 95), 3.85, accuracy: 1e-15)
        XCTAssertEqual(NP.median([3, 1, 2, 4]), 2.5)
        XCTAssertEqual(NP.arange(0.0, 0.30000001, 0.1).count, 4)
        XCTAssertEqual(NP.interp(1.5, [1, 2], [10, 20]), 15)
        XCTAssertEqual(NP.interp(5, [1, 2], [10, 20]), 20)
        XCTAssertEqual(linspaceRound(0, 40, 24).last, 40)
    }
}
