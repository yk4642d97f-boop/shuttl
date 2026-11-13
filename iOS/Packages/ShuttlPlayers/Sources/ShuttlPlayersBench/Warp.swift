import Accelerate
import CoreVideo
import Foundation

/// Scales a 32BGRA frame by `s` (vImage high-quality resampling, like a lower-resolution
/// sensor) and places it at integer offset `t` on a grey canvas of `width × height`.
/// Destination pixel centre = s · source + (s − 1)/2 + t (vImageScale's pixel-centre mapping).
final class Warper {
    private var scaled: vImage_Buffer?
    private var scaledSize = (w: 0, h: 0)
    private var tmp: UnsafeMutableRawPointer?
    private var tmpSize = 0

    deinit {
        if let d = scaled?.data { free(d) }
        if let tmp { free(tmp) }
    }

    func warp(_ src: CVPixelBuffer, s: Double, t: SIMD2<Double>, width: Int, height: Int) throws -> CVPixelBuffer {
        let sw = CVPixelBufferGetWidth(src), sh = CVPixelBufferGetHeight(src)
        let w = Int((Double(sw) * s).rounded()), h = Int((Double(sh) * s).rounded())
        if scaledSize != (w, h) {
            if let d = scaled?.data { free(d) }
            var b = vImage_Buffer()
            guard vImageBuffer_Init(&b, vImagePixelCount(h), vImagePixelCount(w), 32, vImage_Flags(kvImageNoFlags)) == kvImageNoError
            else { throw NSError(domain: "warp", code: 1) }
            scaled = b; scaledSize = (w, h)
        }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }
        var sb = vImage_Buffer(data: CVPixelBufferGetBaseAddress(src), height: vImagePixelCount(sh),
                               width: vImagePixelCount(sw), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var db = scaled!
        let need = vImageScale_ARGB8888(&sb, &db, nil, vImage_Flags(kvImageGetTempBufferSize | kvImageHighQualityResampling))
        if need > tmpSize { if let tmp { free(tmp) }; tmp = malloc(need); tmpSize = need }
        guard vImageScale_ARGB8888(&sb, &db, tmp, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError
        else { throw NSError(domain: "warp", code: 2) }

        var out: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out) == kCVReturnSuccess,
              let dst = out else { throw NSError(domain: "warp", code: 3) }
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }
        let drb = CVPixelBufferGetBytesPerRow(dst)
        let dbase = CVPixelBufferGetBaseAddress(dst)!
        // grey canvas, memory order B G R A = 110 110 110 255
        let row = dbase.assumingMemoryBound(to: UInt8.self)
        for x in 0..<width { row[x * 4] = 110; row[x * 4 + 1] = 110; row[x * 4 + 2] = 110; row[x * 4 + 3] = 255 }
        for y in 1..<height { memcpy(dbase.advanced(by: y * drb), dbase, width * 4) }
        // copy the visible part of the scaled image to (tx, ty)
        let tx = Int(t.x), ty = Int(t.y)
        let x0 = max(0, tx), x1 = min(width, tx + w), y0 = max(0, ty), y1 = min(height, ty + h)
        if x1 > x0, y1 > y0 {
            let bytes = (x1 - x0) * 4
            for y in y0..<y1 {
                let srow = db.data.advanced(by: (y - ty) * db.rowBytes + (x0 - tx) * 4)
                memcpy(dbase.advanced(by: y * drb + x0 * 4), srow, bytes)
            }
        }
        return dst
    }
}
