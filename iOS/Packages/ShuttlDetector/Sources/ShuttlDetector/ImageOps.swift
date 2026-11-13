import Accelerate
import CoreVideo
import Foundation
import ShuttlCore

// cv2-exact uint8 image operations used by the detector's preprocessing
// (FrameGeometry.warp, estimate_backgrounds thumbnails). Verified bit-exact against
// OpenCV 4.11 (the backend's build) through the fixtures of backend/scripts/swift_fixtures/detector.py:
//   * warpAffine INTER_LINEAR, BORDER_CONSTANT 0  (fixed point, AB_BITS 10 / INTER_BITS 5)
//   * resize INTER_AREA shrinking                   (float box filter / integer-scale fast path;
//     horizontal taps FMA-contracted, vertical pass not — as the arm64 OpenCV build does.
//     Bit-exact on landscape sizes; rare ±1 on some portrait sizes)
//   * cvtColor RGB2GRAY                             (15-bit fixed point)
// Enlarging resizes use cv2's fixed-point linear machinery (area or linear coefficients);
// those can differ from cv2 by ±1 grey level (SIMD vs scalar rounding inside OpenCV itself).

/// Read-only view of an interleaved 8-bit colour image; `r`, `g`, `b` are the byte offsets of
/// the channels inside one pixel (BGRA pixel buffers: 2, 1, 0; RGB images: 0, 1, 2).
public struct PixelView {
    public let base: UnsafePointer<UInt8>
    public let width: Int
    public let height: Int
    public let rowBytes: Int
    public let pixelStride: Int
    public let r: Int, g: Int, b: Int

    public init(base: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int, pixelStride: Int,
                r: Int, g: Int, b: Int) {
        self.base = base; self.width = width; self.height = height; self.rowBytes = rowBytes
        self.pixelStride = pixelStride; self.r = r; self.g = g; self.b = b
    }

    /// Channel offsets in output (R, G, B) order.
    @inline(__always) var offsets: (Int, Int, Int) { (r, g, b) }
}

public extension PixelView {
    /// Calls `body` with a view of a locked 32BGRA pixel buffer.
    static func withBGRA<T>(_ pb: CVPixelBuffer, _ body: (PixelView) throws -> T) throws -> T {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        guard fmt == kCVPixelFormatType_32BGRA else { throw ShuttlCoreError.unsupportedPixelFormat(fmt) }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { throw ShuttlCoreError.readerFailed("pixel buffer has no base address") }
        let v = PixelView(base: base.assumingMemoryBound(to: UInt8.self), width: CVPixelBufferGetWidth(pb),
                          height: CVPixelBufferGetHeight(pb), rowBytes: CVPixelBufferGetBytesPerRow(pb),
                          pixelStride: 4, r: 2, g: 1, b: 0)
        return try body(v)
    }

    /// Calls `body` with a view of an interleaved RGB `ImageU8` (3 channels).
    static func withRGB<T>(_ img: ImageU8, _ body: (PixelView) throws -> T) rethrows -> T {
        precondition(img.channels == 3, "RGB image expected")
        return try img.data.withUnsafeBufferPointer { buf in
            try body(PixelView(base: buf.baseAddress!, width: img.width, height: img.height, rowBytes: img.width * 3,
                               pixelStride: 3, r: 0, g: 1, b: 2))
        }
    }
}

/// Fixed-size interleaved RGB uint8 image (the network canvas). Reference type so windows can
/// share frames without copies; storage allocated once.
public final class NetImage: @unchecked Sendable {
    public let width: Int
    public let height: Int
    public let pixels: UnsafeMutablePointer<UInt8>
    public var count: Int { width * height * 3 }
    /// Set for a read-only view into `SealedPixels` (a background plan): the storage belongs to it.
    private let sealedOwner: SealedPixels?

    public init(width: Int = detectorNetWidth, height: Int = detectorNetHeight) {
        self.width = width; self.height = height
        pixels = .allocate(capacity: width * height * 3)
        pixels.initialize(repeating: 0, count: width * height * 3)
        sealedOwner = nil
    }

    init(width: Int, height: Int, sealedIn owner: SealedPixels, offset: Int) {
        self.width = width; self.height = height
        pixels = owner.base + offset
        sealedOwner = owner
    }

    /// True for background-plan images: any write to them traps (the pages are read-only).
    public var isSealed: Bool { sealedOwner != nil }

    deinit { if sealedOwner == nil { pixels.deallocate() } }

    public func clear() { precondition(!isSealed, "background plan images are read-only"); pixels.update(repeating: 0, count: count) }

    public var view: PixelView {
        PixelView(base: UnsafePointer(pixels), width: width, height: height, rowBytes: width * 3, pixelStride: 3,
                  r: 0, g: 1, b: 2)
    }

    public func copy(from other: NetImage) {
        precondition(other.count == count && !isSealed)
        pixels.update(from: other.pixels, count: count)
    }

    public func toImage() -> ImageU8 {
        ImageU8(width: width, height: height, channels: 3, data: Array(UnsafeBufferPointer(start: pixels, count: count)))
    }

    public subscript(x: Int, y: Int, c: Int) -> UInt8 { pixels[(y * width + x) * 3 + c] }
}

/// One page-aligned anonymous mapping, written once and then made read-only (mprotect).
/// Background-plan images live here, outside the malloc heap, so neither a stray write (it
/// traps at the writer) nor the allocator reusing / purging freed pages can change them.
final class SealedPixels: @unchecked Sendable {
    let base: UnsafeMutablePointer<UInt8>
    let length: Int
    private let mapped: Int

    /// Copies `fill` into a fresh mapping of `count` bytes, then seals it.
    init(count: Int, fill: (UnsafeMutablePointer<UInt8>) -> Void) {
        let page = Int(getpagesize())
        length = count
        mapped = max(page, (count + page - 1) / page * page)
        guard let raw = mmap(nil, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), raw != MAP_FAILED else {
            fatalError("mmap of \(mapped) bytes for the background plan failed (errno \(errno))")
        }
        base = raw.assumingMemoryBound(to: UInt8.self)
        fill(base)
        precondition(mprotect(raw, mapped, PROT_READ) == 0, "mprotect(PROT_READ) failed (errno \(errno))")
    }

    deinit { munmap(UnsafeMutableRawPointer(base), mapped) }
}

/// Reuses canvases whose only owner is the pool (no per-frame heap churn).
final class NetImagePool {
    private var items: [NetImage] = []
    let width: Int, height: Int
    private(set) var created = 0

    init(width: Int = detectorNetWidth, height: Int = detectorNetHeight) { self.width = width; self.height = height }

    func acquire() -> NetImage {
        for i in items.indices where isKnownUniquelyReferenced(&items[i]) { return items[i] }
        let img = NetImage(width: width, height: height)
        items.append(img)
        created += 1
        return img
    }
}

public enum CVExact {
    // MARK: warpAffine (INTER_LINEAR, BORDER_CONSTANT 0)

    /// cv2.warpAffine(src, M, (dstW, dstH), flags=INTER_LINEAR) → interleaved RGB at `dst`
    /// (row stride dstW*3). `m` = (m00, m01, m02, m10, m11, m12), full → net.
    public static func warpAffineLinear(_ src: PixelView, m: (Double, Double, Double, Double, Double, Double),
                                        dst: UnsafeMutablePointer<UInt8>, dstWidth: Int, dstHeight: Int) {
        var d = m.0 * m.4 - m.1 * m.3
        d = d != 0 ? 1.0 / d : 0
        let i0 = m.4 * d, i4 = m.0 * d, i1 = -m.1 * d, i3 = -m.3 * d
        let b1 = -i0 * m.2 - i1 * m.5, b2 = -i3 * m.2 - i4 * m.5
        let abScale = 1024.0, roundDelta = 16   // AB_SCALE / INTER_TAB_SIZE / 2
        var adelta = [Int](repeating: 0, count: dstWidth), bdelta = [Int](repeating: 0, count: dstWidth)
        for x in 0..<dstWidth {
            adelta[x] = Int((i0 * Double(x) * abScale).rounded(.toNearestOrEven))
            bdelta[x] = Int((i3 * Double(x) * abScale).rounded(.toNearestOrEven))
        }
        let w = src.width, h = src.height, rb = src.rowBytes, ps = src.pixelStride
        let (cr, cg, cb) = src.offsets
        let base = src.base
        adelta.withUnsafeBufferPointer { ad in
            bdelta.withUnsafeBufferPointer { bd in
                for y in 0..<dstHeight {
                    let x0 = Int(((i1 * Double(y) + b1) * abScale).rounded(.toNearestOrEven)) + roundDelta
                    let y0 = Int(((i4 * Double(y) + b2) * abScale).rounded(.toNearestOrEven)) + roundDelta
                    var o = dst + y * dstWidth * 3
                    for x in 0..<dstWidth {
                        let X = (x0 + ad[x]) >> 5, Y = (y0 + bd[x]) >> 5
                        let sx = X >> 5, sy = Y >> 5, fx = X & 31, fy = Y & 31
                        let w00 = (32 - fx) * (32 - fy) * 32, w01 = fx * (32 - fy) * 32
                        let w10 = (32 - fx) * fy * 32, w11 = fx * fy * 32
                        if sx >= 0 && sy >= 0 && sx + 1 < w && sy + 1 < h {
                            let p0 = base + sy * rb + sx * ps, p1 = p0 + rb
                            o[0] = UInt8(truncatingIfNeeded: (Int(p0[cr]) * w00 + Int(p0[cr + ps]) * w01 + Int(p1[cr]) * w10 + Int(p1[cr + ps]) * w11 + 16384) >> 15)
                            o[1] = UInt8(truncatingIfNeeded: (Int(p0[cg]) * w00 + Int(p0[cg + ps]) * w01 + Int(p1[cg]) * w10 + Int(p1[cg + ps]) * w11 + 16384) >> 15)
                            o[2] = UInt8(truncatingIfNeeded: (Int(p0[cb]) * w00 + Int(p0[cb + ps]) * w01 + Int(p1[cb]) * w10 + Int(p1[cb + ps]) * w11 + 16384) >> 15)
                        } else if sx >= w || sx + 1 < 0 || sy >= h || sy + 1 < 0 {
                            o[0] = 0; o[1] = 0; o[2] = 0
                        } else {
                            // Border: taps outside the image read the constant 0.
                            let in0x = sx >= 0 && sx < w, in1x = sx + 1 >= 0 && sx + 1 < w
                            let in0y = sy >= 0 && sy < h, in1y = sy + 1 >= 0 && sy + 1 < h
                            for (k, c) in [cr, cg, cb].enumerated() {
                                var acc = 0
                                if in0y && in0x { acc += Int(base[sy * rb + sx * ps + c]) * w00 }
                                if in0y && in1x { acc += Int(base[sy * rb + (sx + 1) * ps + c]) * w01 }
                                if in1y && in0x { acc += Int(base[(sy + 1) * rb + sx * ps + c]) * w10 }
                                if in1y && in1x { acc += Int(base[(sy + 1) * rb + (sx + 1) * ps + c]) * w11 }
                                o[k] = UInt8(truncatingIfNeeded: (acc + 16384) >> 15)
                            }
                        }
                        o += 3
                    }
                }
            }
        }
    }

    // MARK: resize

    /// cv2.resize(src, (dstW, dstH), interpolation) → interleaved RGB at `dst` with row stride
    /// `dstRowBytes` (so the output can be written straight into a letterbox canvas).
    public static func resize(_ src: PixelView, dst: UnsafeMutablePointer<UInt8>, dstWidth: Int, dstHeight: Int,
                              dstRowBytes: Int, interpolation: ResizeInterpolation) {
        precondition(dstWidth > 0 && dstHeight > 0)
        if dstWidth == src.width && dstHeight == src.height {
            let (cr, cg, cb) = src.offsets
            for y in 0..<dstHeight {
                let s = src.base + y * src.rowBytes, o = dst + y * dstRowBytes
                for x in 0..<dstWidth {
                    let p = s + x * src.pixelStride
                    o[3 * x] = p[cr]; o[3 * x + 1] = p[cg]; o[3 * x + 2] = p[cb]
                }
            }
            return
        }
        let invSx = Double(dstWidth) / Double(src.width), invSy = Double(dstHeight) / Double(src.height)
        let scaleX = 1.0 / invSx, scaleY = 1.0 / invSy
        if interpolation == .area && scaleX >= 1 && scaleY >= 1 {
            let isx = Int(scaleX.rounded(.toNearestOrEven)), isy = Int(scaleY.rounded(.toNearestOrEven))
            if abs(scaleX - Double(isx)) < Double.ulpOfOne && abs(scaleY - Double(isy)) < Double.ulpOfOne {
                areaFast(src, isx, isy, dst: dst, dstWidth: dstWidth, dstHeight: dstHeight, dstRowBytes: dstRowBytes)
            } else {
                areaGeneral(src, scaleX, scaleY, dst: dst, dstWidth: dstWidth, dstHeight: dstHeight, dstRowBytes: dstRowBytes)
            }
            return
        }
        linearFixed(src, scaleX: scaleX, scaleY: scaleY, invX: invSx, invY: invSy, area: interpolation == .area,
                    dst: dst, dstWidth: dstWidth, dstHeight: dstHeight, dstRowBytes: dstRowBytes)
    }

    /// resizeAreaFast (integer scale): block sum × (1.f/area), rounded half-to-even.
    private static func areaFast(_ src: PixelView, _ sx: Int, _ sy: Int, dst: UnsafeMutablePointer<UInt8>,
                                 dstWidth: Int, dstHeight: Int, dstRowBytes: Int) {
        let (cr, cg, cb) = src.offsets
        let chans = [cr, cg, cb]
        for dy in 0..<dstHeight {
            let y0 = dy * sy, y1 = Swift.min(y0 + sy, src.height)
            for dx in 0..<dstWidth {
                let x0 = dx * sx, x1 = Swift.min(x0 + sx, src.width)
                let area = (y1 - y0) * (x1 - x0)
                let scale = Float(1) / Float(area)
                for (k, c) in chans.enumerated() {
                    var sum = 0
                    for y in y0..<y1 {
                        let row = src.base + y * src.rowBytes + c
                        for x in x0..<x1 { sum += Int(row[x * src.pixelStride]) }
                    }
                    dst[dy * dstRowBytes + dx * 3 + k] = saturateU8(Float(sum) * scale)
                }
            }
        }
    }

    struct AreaTap { var di: Int; var si: Int; var alpha: Float }

    /// computeResizeAreaTab.
    static func areaTab(_ ssize: Int, _ dsize: Int, _ scale: Double) -> [AreaTap] {
        var tab: [AreaTap] = []
        tab.reserveCapacity(dsize * (Int(scale.rounded(.up)) + 1))
        for dx in 0..<dsize {
            let fsx1 = Double(dx) * scale, fsx2 = fsx1 + scale
            let cellWidth = Swift.min(scale, Double(ssize) - fsx1)
            var sx1 = Int(fsx1.rounded(.up)), sx2 = Int(fsx2.rounded(.down))
            sx2 = Swift.min(sx2, ssize - 1)
            sx1 = Swift.min(sx1, sx2)
            if Double(sx1) - fsx1 > 1e-3 {
                tab.append(AreaTap(di: dx, si: sx1 - 1, alpha: Float((Double(sx1) - fsx1) / cellWidth)))
            }
            if sx1 < sx2 {
                for sx in sx1..<sx2 { tab.append(AreaTap(di: dx, si: sx, alpha: Float(1.0 / cellWidth))) }
            }
            if fsx2 - Double(sx2) > 1e-3 {
                tab.append(AreaTap(di: dx, si: sx2, alpha: Float(Swift.min(Swift.min(fsx2 - Double(sx2), 1.0), cellWidth) / cellWidth)))
            }
        }
        return tab
    }

    /// Per-geometry tap tables of the float box filter (computed once, reused every frame).
    /// Horizontal taps are padded to a fixed count `k` per destination column with alpha 0 at
    /// the end: fma(px, 0, acc) == acc exactly, so padding never changes a result.
    final class AreaPlan: @unchecked Sendable {
        let srcW: Int, srcH: Int, dstW: Int, dstH: Int
        let k: Int
        let xSrc: [Int32], xAlpha: [Float]
        let ytab: [AreaTap]
        init(srcW: Int, srcH: Int, dstW: Int, dstH: Int) {
            self.srcW = srcW; self.srcH = srcH; self.dstW = dstW; self.dstH = dstH
            let xt = CVExact.areaTab(srcW, dstW, 1.0 / (Double(dstW) / Double(srcW)))
            var counts = [Int](repeating: 0, count: dstW)
            for t in xt { counts[t.di] += 1 }
            let k = counts.max() ?? 1
            self.k = k
            var src = [Int32](repeating: 0, count: dstW * k), alpha = [Float](repeating: 0, count: dstW * k)
            var fill = [Int](repeating: 0, count: dstW)
            for t in xt {
                src[t.di * k + fill[t.di]] = Int32(t.si); alpha[t.di * k + fill[t.di]] = t.alpha
                fill[t.di] += 1
            }
            for d in 0..<dstW where fill[d] < k { for j in fill[d]..<k { src[d * k + j] = src[d * k] } }
            xSrc = src; xAlpha = alpha
            ytab = CVExact.areaTab(srcH, dstH, 1.0 / (Double(dstH) / Double(srcH)))
        }

        /// Split points of `ytab` into ≤ `parts` ranges that never cut a destination row.
        func chunkBounds(_ parts: Int) -> [Int] {
            var b = [0]
            guard parts > 1 else { return [0, ytab.count] }
            for c in 1..<parts {
                var i = ytab.count * c / parts
                while i < ytab.count && i > 0 && ytab[i].di == ytab[i - 1].di { i += 1 }
                if i > b.last! && i < ytab.count { b.append(i) }
            }
            b.append(ytab.count)
            return b
        }
    }

    private static let planCache = PlanCache()

    final class PlanCache: @unchecked Sendable {
        private let lock = NSLock()
        private var plans: [Int: AreaPlan] = [:]
        func plan(_ sw: Int, _ sh: Int, _ dw: Int, _ dh: Int) -> AreaPlan {
            let key = ((sw * 8192 + sh) * 8192 + dw) * 8192 + dh
            lock.lock(); defer { lock.unlock() }
            if let p = plans[key] { return p }
            let p = AreaPlan(srcW: sw, srcH: sh, dstW: dw, dstH: dh)
            if plans.count > 16 { plans.removeAll() }
            plans[key] = p
            return p
        }
    }

    /// resizeArea_ (float box filter), accumulation order as in OpenCV: each source row is summed
    /// horizontally with FMA (buf = S*alpha + buf, OpenCV's contracted scalar loop), then rows are
    /// accumulated vertically without contraction (sum = sum + beta*buf). SIMD4 lanes hold the
    /// three colour channels (+1 unused), so every lane performs exactly the scalar operations.
    private static func areaGeneral(_ src: PixelView, _ scaleX: Double, _ scaleY: Double, dst: UnsafeMutablePointer<UInt8>,
                                    dstWidth: Int, dstHeight: Int, dstRowBytes: Int) {
        let plan = planCache.plan(src.width, src.height, dstWidth, dstHeight)
        // Destination rows are independent: split the vertical tap list at row boundaries and
        // run the chunks in parallel (identical results, each chunk has its own scratch).
        let chunks = dstHeight >= 64 && src.width * src.height >= 400_000 ? min(4, ProcessInfo.processInfo.activeProcessorCount) : 1
        let bounds = plan.chunkBounds(chunks)
        let job = UncheckedBox((src, dst))
        DispatchQueue.concurrentPerform(iterations: bounds.count - 1) { c in
            areaRows(job.value.0, plan, bounds[c]..<bounds[c + 1], dst: job.value.1, dstRowBytes: dstRowBytes)
        }
    }

    private static func areaRows(_ src: PixelView, _ plan: AreaPlan, _ taps: Range<Int>, dst: UnsafeMutablePointer<UInt8>,
                                 dstRowBytes: Int) {
        let n = plan.dstW, K = plan.k, W = src.width, ps = src.pixelStride
        // SIMD4 lanes = one source pixel in MEMORY order (BGRA: B,G,R,A; RGB: R,G,B,+next byte);
        // each lane performs exactly OpenCV's per-channel scalar operations.
        let rowLen = W * ps
        let rowF = UnsafeMutablePointer<Float>.allocate(capacity: rowLen + 4)
        let buf = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: n)
        let sum = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: n)
        defer { rowF.deallocate(); buf.deallocate(); sum.deallocate() }
        rowF.initialize(repeating: 0, count: rowLen + 4)
        let (cr, cg, cb) = src.offsets
        var prevDy = -1, bufRow = -1
        @inline(__always) func load(_ x: Int32) -> SIMD4<Float> {
            UnsafeRawPointer(rowF + Int(x) * ps).loadUnaligned(as: SIMD4<Float>.self)
        }
        func flush(_ dy: Int) {
            var o = dst + dy * dstRowBytes
            for i in 0..<n {
                let v = sum[i]
                o[0] = saturateU8(v[cr]); o[1] = saturateU8(v[cg]); o[2] = saturateU8(v[cb])
                o += 3
            }
        }
        plan.xSrc.withUnsafeBufferPointer { xsP in
        plan.xAlpha.withUnsafeBufferPointer { xaP in
            let xs = xsP.baseAddress!, xa = xaP.baseAddress!
            for ti in taps {
                let t = plan.ytab[ti]
                if t.si != bufRow {
                    vDSP_vfltu8(src.base + t.si * src.rowBytes, 1, rowF, 1, vDSP_Length(rowLen))   // exact
                    // buf[d] += S[x]*alpha tap by tap, one FMA each (OpenCV's contracted scalar loop);
                    // 4 columns in flight to hide the FMA latency.
                    var d = 0
                    while d + 4 <= n {
                        let b = d * K
                        var a0 = SIMD4<Float>(), a1 = SIMD4<Float>(), a2 = SIMD4<Float>(), a3 = SIMD4<Float>()
                        for j in 0..<K {
                            a0 = a0.addingProduct(load(xs[b + j]), xa[b + j])
                            a1 = a1.addingProduct(load(xs[b + K + j]), xa[b + K + j])
                            a2 = a2.addingProduct(load(xs[b + 2 * K + j]), xa[b + 2 * K + j])
                            a3 = a3.addingProduct(load(xs[b + 3 * K + j]), xa[b + 3 * K + j])
                        }
                        buf[d] = a0; buf[d + 1] = a1; buf[d + 2] = a2; buf[d + 3] = a3
                        d += 4
                    }
                    while d < n {
                        var a = SIMD4<Float>()
                        for j in 0..<K { a = a.addingProduct(load(xs[d * K + j]), xa[d * K + j]) }
                        buf[d] = a
                        d += 1
                    }
                    bufRow = t.si
                }
                let beta = SIMD4<Float>(repeating: t.alpha)
                if t.di != prevDy {
                    if prevDy >= 0 { flush(prevDy) }
                    for i in 0..<n { sum[i] = beta * buf[i] }
                    prevDy = t.di
                } else {
                    // Vertical pass: separate multiply and add (not contracted in OpenCV's SIMD loop;
                    // Swift never contracts `a + b * c`).
                    for i in 0..<n { sum[i] = sum[i] + beta * buf[i] }
                }
            }
        }}
        if prevDy >= 0 { flush(prevDy) }
    }

    /// OpenCV's fixed-point linear resize (INTER_RESIZE_COEF_BITS 11) with linear or, for
    /// enlarging INTER_AREA, area coefficients. Vertical pass uses the NEON rounding of
    /// VResizeLinearVec_32s8u.
    private static func linearFixed(_ src: PixelView, scaleX: Double, scaleY: Double, invX: Double, invY: Double, area: Bool,
                                    dst: UnsafeMutablePointer<UInt8>, dstWidth: Int, dstHeight: Int, dstRowBytes: Int) {
        func coeffs(_ dsize: Int, _ ssize: Int, _ scale: Double, _ inv: Double) -> (ofs: [Int], a0: [Int], a1: [Int]) {
            var ofs = [Int](repeating: 0, count: dsize), a0 = ofs, a1 = ofs
            for d in 0..<dsize {
                var sx: Int
                var fx: Float
                if !area {
                    fx = Float((Double(d) + 0.5) * scale - 0.5)
                    sx = Int(fx.rounded(.down))
                    fx -= Float(sx)
                } else {
                    sx = Int((Double(d) * scale).rounded(.down))
                    fx = Float(Double(d + 1) - Double(sx + 1) * inv)
                    fx = fx <= 0 ? 0 : fx - fx.rounded(.down)
                }
                if sx < 0 { fx = 0; sx = 0 }
                if sx >= ssize - 1 { fx = 0; sx = ssize - 1 }
                ofs[d] = sx
                a0[d] = Int(((1 - fx) * 2048).rounded(.toNearestOrEven))
                a1[d] = Int((fx * 2048).rounded(.toNearestOrEven))
            }
            return (ofs, a0, a1)
        }
        let (xo, xa0, xa1) = coeffs(dstWidth, src.width, scaleX, invX)
        let (yo, yb0, yb1) = coeffs(dstHeight, src.height, scaleY, invY)
        let (cr, cg, cb) = src.offsets
        let chans = [cr, cg, cb]
        let ps = src.pixelStride
        var h0 = [Int](repeating: 0, count: dstWidth * 3), h1 = h0
        func hresize(_ y: Int, into out: inout [Int]) {
            let row = src.base + y * src.rowBytes
            for dx in 0..<dstWidth {
                let s = xo[dx]
                let s1 = Swift.min(s + 1, src.width - 1)
                for (k, c) in chans.enumerated() {
                    out[dx * 3 + k] = Int(row[s * ps + c]) * xa0[dx] + Int(row[s1 * ps + c]) * xa1[dx]
                }
            }
        }
        for dy in 0..<dstHeight {
            let y0 = yo[dy], y1 = Swift.min(y0 + 1, src.height - 1)
            hresize(y0, into: &h0)
            hresize(y1, into: &h1)
            let b0 = yb0[dy], b1 = yb1[dy]
            let o = dst + dy * dstRowBytes
            for i in 0..<(dstWidth * 3) {
                let v = (((h0[i] >> 4) * b0) >> 16) + (((h1[i] >> 4) * b1) >> 16)
                o[i] = UInt8(clamping: (v + 2) >> 2)
            }
        }
    }

    @inline(__always) static func saturateU8(_ v: Float) -> UInt8 {
        let r = v.rounded(.toNearestOrEven)
        return r <= 0 ? 0 : (r >= 255 ? 255 : UInt8(r))
    }

    // MARK: colour

    /// cv2.cvtColor(RGB2GRAY) on interleaved RGB: (9798 R + 19235 G + 3735 B + 2^14) >> 15.
    @inline(__always) public static func gray(r: UInt8, g: UInt8, b: UInt8) -> UInt8 {
        UInt8(truncatingIfNeeded: (Int(r) * 9798 + Int(g) * 19235 + Int(b) * 3735 + 16384) >> 15)
    }

    /// `cv2.cvtColor(cv2.resize(img, (w, h), INTER_AREA), RGB2GRAY)` of a net canvas.
    public static func grayThumbnail(_ img: NetImage, width: Int = 64, height: Int = 36) -> [UInt8] {
        var small = [UInt8](repeating: 0, count: width * height * 3)
        small.withUnsafeMutableBufferPointer { p in
            resize(img.view, dst: p.baseAddress!, dstWidth: width, dstHeight: height, dstRowBytes: width * 3,
                   interpolation: .area)
        }
        var out = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) { out[i] = gray(r: small[3 * i], g: small[3 * i + 1], b: small[3 * i + 2]) }
        return out
    }
}

public extension FrameGeometry {
    /// Resize + letterbox (black) a frame into the net canvas (detector.py FrameGeometry.warp;
    /// the colour order of the canvas is RGB, i.e. warp + BGR2RGB of the Python code).
    func warp(_ src: PixelView, into canvas: NetImage, interpolation: ResizeInterpolation = .area) {
        precondition(src.width == width && src.height == height, "frame size does not match the geometry")
        precondition(canvas.width == netWidth && canvas.height == netHeight)
        if corner {
            CVExact.warpAffineLinear(src, m: cornerAffine, dst: canvas.pixels, dstWidth: netWidth, dstHeight: netHeight)
            return
        }
        let (cw, ch) = contentSize
        if (cw, ch) != (netWidth, netHeight) { canvas.clear() }
        let px = Int(pad.x), py = Int(pad.y)
        CVExact.resize(src, dst: canvas.pixels + (py * netWidth + px) * 3, dstWidth: cw, dstHeight: ch,
                       dstRowBytes: netWidth * 3, interpolation: interpolation)
    }
}
