import Foundation
import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// One exact-input case of images.json: the (min, max) channel planes of a median image
/// plus Python's trace of `_calibrate_image` on it (court.py `trace_calibrate_image`).
struct ImageCase {
    let name: String
    let size: ImageSize
    let image: ImageU8          // 2 channels (min, max) — whiteness only uses min / max
    let auto: JSONValue         // {"trace": ..., "camera": ..., "diag": ...}
    var trace: JSONValue { auto["trace"]! }
}

enum ImageFixtures {
    nonisolated(unsafe) private static var cache: JSONValue?

    static func json() throws -> JSONValue {
        if let c = cache { return c }
        let j = try fixture("images"); cache = j; return j
    }

    /// Decode save_planes' raw-DEFLATE (min, max) planes.
    static func planes(_ name: String, size: ImageSize) throws -> ImageU8 {
        let url = try XCTUnwrap(fixtureURL(name), "missing \(name)")
        let raw = try Data(contentsOf: url) as NSData
        let data = try raw.decompressed(using: .zlib) as Data
        XCTAssertEqual(data.count, size.width * size.height * 2, name)
        return ImageU8(width: size.width, height: size.height, channels: 2, data: [UInt8](data))
    }

    static func cases() throws -> [ImageCase] {
        let j = try json()
        var out: [ImageCase] = []
        for grp in ["sim", "real"] {
            for c in j[grp]?.array ?? [] {
                let sz = c["size"]!.ints
                let size = ImageSize(width: sz[0], height: sz[1])
                let label = c["seed"]?.int.map { "sim\($0)" } ?? (c["clip"]?.string ?? "real")
                out.append(ImageCase(name: label, size: size,
                                     image: try planes(c["planes"]!.string!, size: size), auto: c["auto"]!))
            }
        }
        return out
    }
}

extension JSONValue {
    var vec2: SIMD2<Double> { let d = doubles; return SIMD2(d[0], d[1]) }
    var vec3: SIMD3<Double> { let d = doubles; return SIMD3(d[0], d[1], d[2]) }
    var cameraModel: CameraModel? {
        if case .null = self { return nil }
        return try? decode(CameraModel.self)
    }
}

/// Max relative difference between two homographies after normalising H[2,2] = 1.
func homographyDiff(_ a: simd_double3x3, _ b: simd_double3x3) -> Double {
    let an = a * (1 / mget(a, 2, 2)), bn = b * (1 / mget(b, 2, 2))
    var scale = 0.0, d = 0.0
    for c in 0..<3 { for r in 0..<3 { scale = max(scale, abs(bn[c, r])); d = max(d, abs(an[c, r] - bn[c, r])) } }
    return d / max(scale, 1e-300)
}
