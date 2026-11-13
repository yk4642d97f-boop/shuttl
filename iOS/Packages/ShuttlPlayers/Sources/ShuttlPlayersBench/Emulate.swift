import Accelerate
import CoreVideo
import Foundation
import simd
import ShuttlCore
import ShuttlPlayers

/// Phone on a tripod behind the near baseline, aimed at the court centre (tools/venue/calib_sweep.py).
struct PhoneSetup {
    var name: String
    var height: Double          // m
    var distance: Double        // m behind the near baseline
    var hfovDeg: Double = 70    // iPhone main camera; ultra-wide ≈ 106
    var size = ImageSize(width: 1920, height: 1080)

    var camera: CameraModel {
        let f = Double(size.width) / (2 * tan(hfovDeg / 2 * .pi / 180))
        return CameraModel.lookAt(eye: [-CourtModel.halfLength - distance, 0, height], target: [0, 0, 0],
                                  focalPx: f, imageSize: size)
    }
}

/// Phone-like frame built from a broadcast frame: scaled by `s` and shifted by `t` so the far
/// player's pixel height equals the phone's and the far half sits where the phone's far-half
/// crop is. The crop rects keep the PHONE's size (centred on the emulated far half), so every
/// layout magnifies a far player as much as it would on the phone. Limits: perspective (the
/// phone sees the far half flatter) and the near half (smaller than on a phone) are not emulated.
struct Emulation {
    var setup: String
    var s: Double
    var t: SIMD2<Double>
    var camera: CameraModel                  // broadcast camera composed with the similarity
    var size: ImageSize
    var rects: [FarCropLayout: [PixelRect]]
    var phoneFarPx: Double                    // 1.75 m player on the far baseline, phone camera

    static func farPx(_ c: CameraModel) -> Double {
        abs(c.project([CourtModel.halfLength, 0, 0]).y - c.project([CourtModel.halfLength, 0, 1.75]).y)
    }

    static func center(_ b: (x1: Double, y1: Double, x2: Double, y2: Double)) -> SIMD2<Double> {
        [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]
    }

    /// Native broadcast frames (no warp), rects from the broadcast camera.
    static func native(_ cam: CameraModel) -> Emulation {
        var r: [FarCropLayout: [PixelRect]] = [:]
        for l in FarCropLayout.allCases { r[l] = FarCrops.rects(l, camera: cam, size: cam.imageSize) }
        return Emulation(setup: "broadcast", s: 1, t: .zero, camera: cam, size: cam.imageSize, rects: r, phoneFarPx: farPx(cam))
    }

    init(setup: String, s: Double, t: SIMD2<Double>, camera: CameraModel, size: ImageSize,
         rects: [FarCropLayout: [PixelRect]], phoneFarPx: Double) {
        self.setup = setup; self.s = s; self.t = t; self.camera = camera; self.size = size
        self.rects = rects; self.phoneFarPx = phoneFarPx
    }

    init?(phone: PhoneSetup, broadcast cam: CameraModel) {
        let pc = phone.camera, size = phone.size
        let single = FarCrops.worldBoxes(.single)[0]
        guard let pb = FarCrops.imageBounds(single, camera: pc), let bb = FarCrops.imageBounds(single, camera: cam) else { return nil }
        let sq = (Self.farPx(pc) / Self.farPx(cam) * 80).rounded() / 80        // exact binary scale
        // vImage maps destination pixel centres: dst = s·src + (s − 1)/2 + t
        let t = (Self.center(pb) - sq * Self.center(bb) - SIMD2(repeating: (sq - 1) / 2)).rounded(.toNearestOrEven)
        let a = simd_double3x3(rows: [[sq, 0, t.x + (sq - 1) / 2], [0, sq, t.y + (sq - 1) / 2], [0, 0, 1]])
        let ecam = CameraModel(K: a * cam.K, R: cam.R, t: cam.t, imageSize: size, dist: cam.dist)
        var r: [FarCropLayout: [PixelRect]] = [:]
        for l in FarCropLayout.allCases {
            r[l] = FarCrops.worldBoxes(l).compactMap { box -> PixelRect? in
                guard let p = FarCrops.rect(box, camera: pc, size: size), let e = FarCrops.imageBounds(box, camera: ecam) else { return nil }
                let w = min(Double(size.width), max(Double(p.width), e.x2 - e.x1))
                let h = min(Double(size.height), max(Double(p.height), e.y2 - e.y1))
                let c = Self.center(e)
                let x1 = min(max(c.x - w / 2, 0), Double(size.width) - w), y1 = min(max(c.y - h / 2, 0), Double(size.height) - h)
                return PixelRect(x1: Int(x1), y1: Int(y1), x2: Int(x1 + w), y2: Int(y1 + h))
            }
        }
        self.init(setup: phone.name, s: sq, t: t, camera: ecam, size: size, rects: r, phoneFarPx: Self.farPx(pc))
    }
}
