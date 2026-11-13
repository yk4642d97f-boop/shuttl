import Accelerate
import CoreVideo
import Foundation

public enum ShuttlCoreError: Error, CustomStringConvertible {
    case unsupportedPixelFormat(OSType)
    case cannotOpenVideo(URL, String)
    case noVideoTrack(URL)
    case readerFailed(String)
    /// No frame arrived for `seconds` (the hardware decoder was taken by another client, e.g. a
    /// live camera preview). The reader was cancelled so the stage fails fast and can retry.
    case decodeStalled(seconds: Double, frame: Int)
    case vImage(vImage_Error)

    public var description: String {
        switch self {
        case .unsupportedPixelFormat(let f): return "unsupported pixel format \(f) (expected 32BGRA)"
        case .cannotOpenVideo(let u, let why): return "cannot open video \(u.lastPathComponent): \(why)"
        case .noVideoTrack(let u): return "no video track in \(u.lastPathComponent)"
        case .readerFailed(let why): return "video reader failed: \(why)"
        case .decodeStalled(let s, let f): return "video decode interrupted: no frame for \(Int(s)) s after frame \(f) (decoder busy)"
        case .vImage(let e): return "vImage error \(e)"
        }
    }
}

/// Row-major interleaved image, origin top-left. CPU-side stand-in for a numpy
/// (H, W[, C]) array. Channel order of colour images is **RGB** — the Python stack
/// works in OpenCV's BGR, so ported code must swap indices where it matters.
public struct ImageBuffer<Pixel: Sendable & Numeric>: Sendable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public var data: [Pixel]

    public init(width: Int, height: Int, channels: Int = 1, repeating value: Pixel = 0) {
        self.width = width; self.height = height; self.channels = channels
        data = Array(repeating: value, count: width * height * channels)
    }

    public init(width: Int, height: Int, channels: Int = 1, data: [Pixel]) {
        precondition(data.count == width * height * channels, "image data size mismatch")
        self.width = width; self.height = height; self.channels = channels; self.data = data
    }

    @inline(__always)
    public subscript(x: Int, y: Int, c: Int = 0) -> Pixel {
        get { data[(y * width + x) * channels + c] }
        set { data[(y * width + x) * channels + c] = newValue }
    }

    public var size: ImageSize { ImageSize(width: width, height: height) }
}

public typealias ImageU8 = ImageBuffer<UInt8>
public typealias ImageF32 = ImageBuffer<Float>

public enum ResizeMethod: Sendable {
    /// cv2.INTER_LINEAR: pixel-centre mapping, replicated borders.
    case linear
    /// cv2.INTER_AREA: exact box averaging when shrinking; bilinear when enlarging.
    case area
}

// MARK: - CVPixelBuffer bridges (32BGRA)

public extension ImageBuffer where Pixel == UInt8 {
    /// Luma exactly like cv2.cvtColor(BGR2GRAY): (4899 R + 9617 G + 1868 B + 8192) >> 14.
    static func gray(from pixelBuffer: CVPixelBuffer) throws -> ImageU8 {
        try withBGRA(pixelBuffer) { src in
            var out = ImageU8(width: Int(src.width), height: Int(src.height))
            try out.data.withUnsafeMutableBytes { raw in
                var dst = vImage_Buffer(data: raw.baseAddress, height: src.height, width: src.width, rowBytes: Int(src.width))
                var s = src
                let matrix: [Int16] = [1868, 9617, 4899, 0]       // memory order B, G, R, A
                let err = vImageMatrixMultiply_ARGB8888ToPlanar8(&s, &dst, matrix, 16384, nil, 8192, vImage_Flags(kvImageNoFlags))
                if err != kvImageNoError { throw ShuttlCoreError.vImage(err) }
            }
            return out
        }
    }

    /// Interleaved RGB (3 channels).
    static func rgb(from pixelBuffer: CVPixelBuffer) throws -> ImageU8 {
        try withBGRA(pixelBuffer) { src in
            var out = ImageU8(width: Int(src.width), height: Int(src.height), channels: 3)
            try out.data.withUnsafeMutableBytes { raw in
                var dst = vImage_Buffer(data: raw.baseAddress, height: src.height, width: src.width, rowBytes: Int(src.width) * 3)
                var s = src
                let err = vImageConvert_BGRA8888toRGB888(&s, &dst, vImage_Flags(kvImageNoFlags))
                if err != kvImageNoError { throw ShuttlCoreError.vImage(err) }
            }
            return out
        }
    }

    func toFloat(scale: Float = 1) -> ImageF32 {
        ImageF32(width: width, height: height, channels: channels, data: data.map { Float($0) * scale })
    }

    func resized(width w: Int, height h: Int, method: ResizeMethod = .linear) -> ImageU8 {
        toFloat().resized(width: w, height: h, method: method).toU8()
    }
}

private func withBGRA<T>(_ pb: CVPixelBuffer, _ body: (vImage_Buffer) throws -> T) throws -> T {
    let fmt = CVPixelBufferGetPixelFormatType(pb)
    guard fmt == kCVPixelFormatType_32BGRA else { throw ShuttlCoreError.unsupportedPixelFormat(fmt) }
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let buf = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb),
                            height: vImagePixelCount(CVPixelBufferGetHeight(pb)),
                            width: vImagePixelCount(CVPixelBufferGetWidth(pb)),
                            rowBytes: CVPixelBufferGetBytesPerRow(pb))
    return try body(buf)
}

// MARK: - Float images

public extension ImageBuffer where Pixel == Float {
    /// Round half away from zero and clamp to 0…255 (cv2 saturate_cast).
    func toU8() -> ImageU8 {
        ImageU8(width: width, height: height, channels: channels,
                data: data.map { UInt8(Swift.min(Swift.max($0.rounded(), 0), 255)) })
    }

    func resized(width w: Int, height h: Int, method: ResizeMethod = .linear) -> ImageF32 {
        precondition(w > 0 && h > 0)
        if w == width && h == height { return self }
        let sx = Double(width) / Double(w), sy = Double(height) / Double(h)
        if method == .area && sx >= 1 && sy >= 1 { return areaResize(w, h, sx, sy) }
        return linearResize(w, h, sx, sy)
    }

    private func linearResize(_ w: Int, _ h: Int, _ sx: Double, _ sy: Double) -> ImageF32 {
        func taps(_ n: Int, _ srcN: Int, _ scale: Double) -> [(Int, Int, Float)] {
            (0..<n).map { d in
                let s = Swift.max((Double(d) + 0.5) * scale - 0.5, 0)
                let i0 = Swift.min(Int(s), srcN - 1)
                let i1 = Swift.min(i0 + 1, srcN - 1)
                return (i0, i1, Float(s - Double(i0)))
            }
        }
        let xt = taps(w, width, sx), yt = taps(h, height, sy)
        var out = ImageF32(width: w, height: h, channels: channels)
        for (dy, (y0, y1, fy)) in yt.enumerated() {
            for (dx, (x0, x1, fx)) in xt.enumerated() {
                for c in 0..<channels {
                    let top = self[x0, y0, c] * (1 - fx) + self[x1, y0, c] * fx
                    let bot = self[x0, y1, c] * (1 - fx) + self[x1, y1, c] * fx
                    out[dx, dy, c] = top * (1 - fy) + bot * fy
                }
            }
        }
        return out
    }

    private func areaResize(_ w: Int, _ h: Int, _ sx: Double, _ sy: Double) -> ImageF32 {
        // Separable box filter: each destination pixel averages the source interval
        // [d·s, (d+1)·s) with fractional weights at the ends.
        func weights(_ n: Int, _ srcN: Int, _ scale: Double) -> [[(Int, Float)]] {
            (0..<n).map { d in
                let a = Double(d) * scale, b = Swift.min(Double(d + 1) * scale, Double(srcN))
                var taps: [(Int, Float)] = []
                var i = Int(a.rounded(.down))
                while Double(i) < b - 1e-12 {
                    let lo = Swift.max(a, Double(i)), hi = Swift.min(b, Double(i + 1))
                    if hi > lo { taps.append((Swift.min(i, srcN - 1), Float((hi - lo) / (b - a)))) }
                    i += 1
                }
                return taps
            }
        }
        let xw = weights(w, width, sx), yw = weights(h, height, sy)
        var tmp = ImageF32(width: w, height: height, channels: channels)
        for y in 0..<height {
            for (dx, taps) in xw.enumerated() {
                for c in 0..<channels {
                    var s: Float = 0
                    for (x, wt) in taps { s += self[x, y, c] * wt }
                    tmp[dx, y, c] = s
                }
            }
        }
        var out = ImageF32(width: w, height: h, channels: channels)
        for (dy, taps) in yw.enumerated() {
            for x in 0..<w {
                for c in 0..<channels {
                    var s: Float = 0
                    for (y, wt) in taps { s += tmp[x, y, c] * wt }
                    out[x, dy, c] = s
                }
            }
        }
        return out
    }
}
