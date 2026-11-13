import XCTest
import CoreVideo
import CoreMedia
import ShuttlCore
import ShuttlPresent

/// The threaded engine: one attempt in flight, drop-while-busy, stream semantics, pixel buffers.
final class CourtLockEngineTests: XCTestCase {
    typealias F = LockFixtures

    func testOneAttemptInFlightAndLocks() async throws {
        let cal = ScriptedCalibrator(delay: 0.15) { _, _ in F.accept(F.camera()) }
        let engine = CourtLockEngine(calibrator: cal)
        let collected = Collector()
        let stream = engine.states()
        let reader = Task { for await s in stream { await collected.add(s.kindName); if s.isLocked { break } } }
        let img = F.texture()
        // 3x real time: a 30 fps clock, frames every 11 ms of wall time, for 4 s of "video".
        for i in 0..<120 {
            engine.ingest(image: img, time: Double(i) / 30)
            try await Task.sleep(nanoseconds: 11_000_000)
        }
        engine.waitUntilIdle()
        // Never hang the suite: give the reader 2 s to see the lock, then cancel it.
        let deadline = Date().addingTimeInterval(2)
        while await collected.items.last != "locked" && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        reader.cancel()
        _ = await reader.value
        XCTAssertEqual(cal.peak, 1, "never more than one attempt in flight")
        XCTAssertTrue(engine.state.isLocked)
        let kinds = await collected.items
        XCTAssertEqual(Array(kinds.prefix(3)), ["searching", "candidate", "locked"], "\(kinds)")
        let st = engine.stats
        XCTAssertLessThanOrEqual(st.samples, 9, "≤ 2 Hz sampling of 4 s")
        XCTAssertLessThan(cal.callCount, st.samples + 1)
        XCTAssertNotNil(engine.lockedCameraHint())
        let hint = try XCTUnwrap(engine.lockedCameraHint(recordingSize: ImageSize(width: 1280, height: 720)))
        XCTAssertEqual(hint.imageSize, ImageSize(width: 1280, height: 720))
        XCTAssertNil(engine.lockedCameraHint(recordingSize: ImageSize(width: 1440, height: 1080)))
        XCTAssertNotNil(engine.medianAttemptSeconds(.search))
    }

    func testNewSubscriberGetsCurrentStateAndResetPublishes() async {
        let cal = ScriptedCalibrator(delay: 0) { _, _ in F.accept(F.camera()) }
        let engine = CourtLockEngine(calibrator: cal)
        engine.ingest(image: F.texture(), time: 0); engine.waitUntilIdle()
        engine.ingest(image: F.texture(), time: 0.5); engine.waitUntilIdle()
        XCTAssertTrue(engine.state.isLocked)
        let stream = engine.states()
        let reader = Task { () -> [CourtLockState] in
            var got: [CourtLockState] = []
            for await s in stream { got.append(s); if got.count == 2 { break } }
            return got
        }
        try? await Task.sleep(nanoseconds: 50_000_000)
        engine.reset()
        let timeout = Task { try? await Task.sleep(nanoseconds: 2_000_000_000); reader.cancel() }
        let got = await reader.value
        timeout.cancel()
        XCTAssertEqual(got.count, 2)
        XCTAssertEqual(got.first?.isLocked, true)
        XCTAssertEqual(got.last, .searching(.frameWholeCourt))
        XCTAssertNil(engine.lockedCameraHint())
    }

    func testPauseDropsFrames() {
        let cal = ScriptedCalibrator(delay: 0) { _, _ in F.reject(nT: 2) }
        let engine = CourtLockEngine(calibrator: cal)
        engine.pause()
        engine.ingest(image: F.texture(), time: 0); engine.waitUntilIdle()
        XCTAssertEqual(engine.stats.samples, 0)
        engine.resume()
        engine.ingest(image: F.texture(), time: 0.1); engine.waitUntilIdle()
        XCTAssertEqual(engine.stats.samples, 1)
        XCTAssertEqual(engine.state, .searching(.farBaselineMissing))
    }

    func testPixelBufferPaths() throws {
        let img = F.texture(size: ImageSize(width: 1280, height: 720))
        let bgra = try makeBGRA(img)
        let ds = FrameDownsampler()
        let a = try ds.frame(from: bgra, workingSize: ImageSize(width: 960, height: 540), time: 0)
        let b = ds.frame(from: img, workingSize: ImageSize(width: 960, height: 540), time: 0)
        XCTAssertEqual(a.image.size, ImageSize(width: 960, height: 540))
        XCTAssertEqual(maxAbsDiff(a.image, b.image), 0, "BGRA and RGB paths identical")
        let yuv = try makeYUV420v(img)
        let c = try ds.frame(from: yuv, workingSize: ImageSize(width: 640, height: 360), time: 0)
        let d = ds.frame(from: img, workingSize: ImageSize(width: 640, height: 360), time: 0)
        XCTAssertLessThanOrEqual(meanAbsDiff(c.image, d.image), 3.0, "420v → RGB within a few grey levels")
        let engine = CourtLockEngine(calibrator: ScriptedCalibrator(delay: 0) { _, _ in F.reject() })
        engine.ingest(pixelBuffer: yuv, time: CMTime(value: 0, timescale: 600)); engine.waitUntilIdle()
        XCTAssertEqual(engine.stats.samples, 1)
        XCTAssertEqual(engine.configuration.workingSize(for: ImageSize(width: 1920, height: 1080)),
                       ImageSize(width: 1440, height: 810))
        XCTAssertEqual(engine.configuration.workingSize(for: ImageSize(width: 1280, height: 720)),
                       ImageSize(width: 960, height: 540))
    }

    /// The fast area resize must be bit-identical to ShuttlCore's (the 0.75x verification used it).
    func testFastAreaResizeIsBitExact() {
        for (src, dst) in [((1920, 1080), (1440, 810)), ((1280, 720), (960, 540)), ((641, 359), (480, 270))] {
            var img = F.texture(size: ImageSize(width: src.0, height: src.1))
            for i in stride(from: 0, to: img.data.count, by: 7) { img.data[i] = UInt8((i * 31) & 255) }
            let a = FrameDownsampler.areaResize(img, to: ImageSize(width: dst.0, height: dst.1))
            let b = img.resized(width: dst.0, height: dst.1, method: .area)
            XCTAssertEqual(a.data, b.data, "\(src) → \(dst)")
        }
    }

    // MARK: helpers

    private actor Collector {
        var items: [String] = []
        func add(_ s: String) { if items.last != s { items.append(s) } }
    }

    private func maxAbsDiff(_ a: ImageU8, _ b: ImageU8) -> Int {
        zip(a.data, b.data).map { abs(Int($0) - Int($1)) }.max() ?? 0
    }

    private func meanAbsDiff(_ a: ImageU8, _ b: ImageU8) -> Double {
        Double(zip(a.data, b.data).map { abs(Int($0) - Int($1)) }.reduce(0, +)) / Double(a.data.count)
    }

    private func makeBGRA(_ img: ImageU8) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, img.width, img.height, kCVPixelFormatType_32BGRA, nil, &pb)
        let p = try XCTUnwrap(pb)
        CVPixelBufferLockBaseAddress(p, [])
        let base = CVPixelBufferGetBaseAddress(p)!.assumingMemoryBound(to: UInt8.self)
        let rb = CVPixelBufferGetBytesPerRow(p)
        for y in 0..<img.height {
            for x in 0..<img.width {
                let o = base + y * rb + x * 4
                o[0] = img[x, y, 2]; o[1] = img[x, y, 1]; o[2] = img[x, y, 0]; o[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(p, [])
        return p
    }

    /// BT.709 video-range 4:2:0 encode of an RGB image.
    private func makeYUV420v(_ img: ImageU8) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, img.width, img.height, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, nil, &pb)
        let p = try XCTUnwrap(pb)
        CVPixelBufferLockBaseAddress(p, [])
        let yb = CVPixelBufferGetBaseAddressOfPlane(p, 0)!.assumingMemoryBound(to: UInt8.self)
        let cb = CVPixelBufferGetBaseAddressOfPlane(p, 1)!.assumingMemoryBound(to: UInt8.self)
        let yrb = CVPixelBufferGetBytesPerRowOfPlane(p, 0), crb = CVPixelBufferGetBytesPerRowOfPlane(p, 1)
        func rgb(_ x: Int, _ y: Int) -> (Double, Double, Double) {
            (Double(img[x, y, 0]) / 255, Double(img[x, y, 1]) / 255, Double(img[x, y, 2]) / 255)
        }
        for y in 0..<img.height {
            for x in 0..<img.width {
                let (r, g, b) = rgb(x, y)
                let Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                yb[y * yrb + x] = UInt8(max(0, min(255, (16 + 219 * Y).rounded())))
            }
        }
        for y in stride(from: 0, to: img.height, by: 2) {
            for x in stride(from: 0, to: img.width, by: 2) {
                var sr = 0.0, sg = 0.0, sb = 0.0
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] { let c = rgb(x + dx, y + dy); sr += c.0; sg += c.1; sb += c.2 }
                let r = sr / 4, g = sg / 4, b = sb / 4
                let Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                let Cb = (b - Y) / 1.8556, Cr = (r - Y) / 1.5748
                cb[(y / 2) * crb + x] = UInt8(max(0, min(255, (128 + 224 * Cb).rounded())))
                cb[(y / 2) * crb + x + 1] = UInt8(max(0, min(255, (128 + 224 * Cr).rounded())))
            }
        }
        CVPixelBufferUnlockBaseAddress(p, [])
        return p
    }
}
