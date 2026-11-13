import Accelerate
import CoreVideo
import Foundation
import ShuttlCore
import ShuttlCourt
import simd

/// Automatic court calibration result, flattened for storage.
struct CourtResult: Codable {
    var camera: CameraModel?
    var reason: String
    var confidence: Double?
    var rmse: Double?
    var coverage: Double?
    var source: String
    var frames: [Int]
    var seconds: Double

    var ok: Bool { camera != nil }
}

enum Court {
    /// The phone's automatic calibration (CourtCalibration.calibrateFromFramesDetailed, no focal hint).
    static func calibrate(_ frames: [ImageU8], indices: [Int], size: ImageSize) -> CourtResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let r = CourtCalibration.calibrateFromFramesDetailed(frames, size: size)
        let d = r.diagnostics
        return CourtResult(camera: r.camera, reason: d.reason.rawValue, confidence: d.confidence, rmse: d.stats?.rmse,
                           coverage: d.stats?.coverage, source: d.source, frames: indices,
                           seconds: CFAbsoluteTimeGetCurrent() - t0)
    }

    /// Frames `indices` (ascending) of the video as RGB images; one seek per frame when they are far apart.
    static func frames(_ src: VideoSource, _ indices: [Int]) throws -> [ImageU8] {
        var out: [ImageU8] = []
        let want = Set(indices)
        var i = 0
        while i < indices.count {
            var j = i
            while j + 1 < indices.count, indices[j + 1] - indices[j] < 60 { j += 1 }
            let seq = try src.frames(start: indices[i], stop: indices[j] + 1)
            for (f, pb) in seq where want.contains(f) { out.append(try rgb(pb)) }
            if let e = seq.error { throw e }
            i = j + 1
        }
        return out
    }

    static func rgb(_ pb: CVPixelBuffer) throws -> ImageU8 {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(h), width: vImagePixelCount(w),
                                rowBytes: CVPixelBufferGetBytesPerRow(pb))
        var data = [UInt8](repeating: 0, count: w * h * 3)
        let err = data.withUnsafeMutableBytes { raw -> vImage_Error in
            var dst = vImage_Buffer(data: raw.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 3)
            return vImageConvert_BGRA8888toRGB888(&src, &dst, vImage_Flags(kvImageNoFlags))
        }
        guard err == kvImageNoError else { throw ShuttlCoreError.vImage(err) }
        return ImageU8(width: w, height: h, channels: 3, data: data)
    }

    /// Median / max px distance between two cameras' projections of every court keypoint.
    static func difference(_ a: CameraModel, _ b: CameraModel) -> (median: Double, max: Double) {
        let (_, kp) = CourtModel.keypointsArray()
        let d = zip(a.project(kp), b.project(kp)).map { simd_length($0 - $1) }.filter(\.isFinite).sorted()
        guard !d.isEmpty else { return (.nan, .nan) }
        return (d[d.count / 2], d.last!)
    }
}
