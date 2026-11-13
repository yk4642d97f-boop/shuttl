import Foundation
import ShuttlCore

// sim.py — plausible capture setups (all behind / beside the NEAR half, x < 0).
//   behind        phone on a tripod 1–6 m behind the near baseline, 1.0–2.0 m high, focal
//                 0.8–1.0 × width, yaw ±5°, roll ±3°, far baseline 28–50 % down the frame;
//   elevated_side 3–6 m high at the side/corner of the near half, whole court in view;
//   broadcast     like the TrackNetV2 / BWF broadcast clips (eye 16–24 m behind, 6–10 m up,
//                 telephoto ≈ 2–2.6 × width, near baseline at 89–97 % of the height
//                 spanning 54–72 % of the width).
public enum SimCameras {
    static let corners: [SIMD3<Double>] = [
        SIMD3(-CourtModel.halfLength, -CourtModel.doublesHalfWidth, 0), SIMD3(-CourtModel.halfLength, CourtModel.doublesHalfWidth, 0),
        SIMD3(CourtModel.halfLength, -CourtModel.doublesHalfWidth, 0), SIMD3(CourtModel.halfLength, CourtModel.doublesHalfWidth, 0)]

    /// `_fit_focal`: focal so that the court's widest extent fills `fill` of the frame.
    static func fitFocal(eye: SIMD3<Double>, target: SIMD3<Double>, size: ImageSize, roll: Double, fill: Double,
                         lo: Double, hi: Double) -> CameraModel {
        let w = Double(size.width), h = Double(size.height)
        let cam = CameraModel.lookAt(eye: eye, target: target, focalPx: w, imageSize: size, rollDeg: roll)
        let uv = cam.project(corners)
        var ex = -Double.infinity, ey = -Double.infinity
        for p in uv {
            if !p.x.isNaN { ex = Swift.max(ex, abs(p.x - w / 2)) }
            if !p.y.isNaN { ey = Swift.max(ey, abs(p.y - h / 2)) }
        }
        let ext = pyMax(ex / (w / 2), ey / (h / 2))
        let f = NP.clip(w * fill / ext, lo * w, hi * w)
        return CameraModel.lookAt(eye: eye, target: target, focalPx: f, imageSize: size, rollDeg: roll)
    }

    /// `random_camera(rng, image_size, style)`.
    public static func randomCamera(_ rng: any SimRandom, imageSize size: ImageSize = ImageSize(width: 1920, height: 1080),
                                    style styleIn: String = "random") throws -> CameraModel {
        var style = styleIn
        if style == "random" { style = rng.pick(["behind", "elevated_side", "broadcast"], p: [0.5, 0.25, 0.25]) }
        let w = Double(size.width), h = Double(size.height)
        switch style {
        case "behind":
            let d = rng.uniform(1.0, 6.0), z = rng.uniform(1.0, 2.0)
            let eye = SIMD3(-CourtModel.halfLength - d, rng.uniform(-1.2, 1.2), z)
            let f = w * rng.uniform(0.8, 1.0)
            let roll = rng.uniform(-3, 3)
            let frac = rng.uniform(0.28, 0.5)                      // far baseline height in the frame
            var lo = eye.x + 0.5, hi = 12.0, mid = 0.0
            for _ in 0..<40 {                                      // bisection on the look-at x
                mid = 0.5 * (lo + hi)
                let cam = CameraModel.lookAt(eye: eye, target: SIMD3(mid, 0, 0), focalPx: f, imageSize: size, rollDeg: roll)
                let v = cam.project(SIMD3(CourtModel.halfLength, 0, 0)).y
                if !v.isFinite || v > frac * h { hi = mid } else { lo = mid }
            }
            let yaw = NP.radians(rng.uniform(-5, 5))
            let tg = SIMD3(mid, 0, 0) - eye
            let c = cos(yaw), s = sin(yaw)
            let tgt = eye + SIMD3(c * tg.x - s * tg.y, s * tg.x + c * tg.y, tg.z)
            return CameraModel.lookAt(eye: eye, target: tgt, focalPx: f, imageSize: size, rollDeg: roll)
        case "elevated_side":
            let side = rng.pick([-1.0, 1.0])
            let ex = rng.uniform(-11.0, -4.0), ey = side * rng.uniform(8.0, 13.0), ez = rng.uniform(3.5, 7.0)
            let tx = rng.uniform(-1.0, 1.0), ty = side * rng.uniform(-0.3, 0.6)
            let roll = rng.uniform(-2, 2), fill = rng.uniform(0.82, 0.94)
            return fitFocal(eye: SIMD3(ex, ey, ez), target: SIMD3(tx, ty, 0), size: size, roll: roll, fill: fill, lo: 0.4, hi: 1.6)
        case "broadcast":
            var cam: CameraModel? = nil
            for _ in 0..<20 {
                let ex = -CourtModel.halfLength - rng.uniform(16.0, 24.0)
                let ey = rng.random() < 0.8 ? rng.uniform(-1.0, 1.0) : rng.uniform(-3.0, 3.0)
                let eye = SIMD3(ex, ey, rng.uniform(6.0, 10.0))
                let vNear = rng.uniform(0.89, 0.97), widthFrac = rng.uniform(0.54, 0.72)
                let ty = rng.uniform(-0.3, 0.3), roll = rng.uniform(-0.8, 0.8)
                var f = 2.5 * w
                for _ in 0..<5 {                                   // alternate pitch (near line row) ↔ focal (width)
                    var lo = -CourtModel.halfLength, hi = 30.0
                    for _ in 0..<40 {
                        let mid = 0.5 * (lo + hi)
                        let c = CameraModel.lookAt(eye: eye, target: SIMD3(mid, ty, 0), focalPx: f, imageSize: size, rollDeg: roll)
                        cam = c
                        let v = c.project(SIMD3(-CourtModel.halfLength, 0, 0)).y
                        if v > vNear * h { hi = mid } else { lo = mid }
                    }
                    let uv = cam!.project(corners)
                    f *= widthFrac * w / abs(uv[1].x - uv[0].x)
                }
                let uv = cam!.project(corners)
                let inFrame = uv.allSatisfy { $0.x >= 0 && $0.y >= 0 && $0.x < w && $0.y < h }
                if inFrame && pyMin(uv[2].y, uv[3].y) > 0.12 * h { return cam! }
            }
            return cam!
        default:
            throw SimError.unknownCameraStyle(style)
        }
    }
}
