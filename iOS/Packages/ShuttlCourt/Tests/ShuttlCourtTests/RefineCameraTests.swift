import XCTest
import simd
import ShuttlCore
@testable import ShuttlCourt

/// refine_camera parity on the exact median images (images.json): sim cases refine the
/// 4-corner-tap estimate (auto k1 and fit_k1=False), real cases refine Python's auto camera.
final class RefineCameraTests: XCTestCase {
    func testRefineCameraExactInput() throws {
        let j = try ImageFixtures.json()
        var worst = 0.0
        var jobs: [(String, ImageU8, CameraModel, CameraModel, K1Mode)] = []
        for c in j["sim"]?.array ?? [] {
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            let img = try ImageFixtures.planes(c["planes"]!.string!, size: size)
            let est = try XCTUnwrap(c["est"]!.cameraModel)
            jobs.append(("sim\(c["seed"]!.int!) auto", img, est, try XCTUnwrap(c["refined"]!.cameraModel), .auto))
            jobs.append(("sim\(c["seed"]!.int!) no_k1", img, est, try XCTUnwrap(c["refined_no_k1"]!.cameraModel), .off))
        }
        for c in j["real"]?.array ?? [] {
            guard let ref = c["refined"]?.cameraModel, let auto = c["auto"]?["camera"]?.cameraModel else { continue }
            let sz = c["size"]!.ints, size = ImageSize(width: sz[0], height: sz[1])
            jobs.append((c["clip"]!.string!, try ImageFixtures.planes(c["planes"]!.string!, size: size), auto, ref, .auto))
        }
        XCTAssertEqual(jobs.count, 9)
        for (name, img, cam0, want, k1) in jobs {
            let feat = Features(img, config: .default)
            let got = CourtCalibration.refineCamera(cam0, features: feat, fitK1: k1)
            let a = keypointAgreement(want, got)
            worst = max(worst, a.max)
            print("PARITY refine \(name): kp max \(a.max) px, k1 \(got.dist[0]) vs \(want.dist[0]), conf \(got.confidence) vs \(want.confidence)")
            XCTAssertLessThan(a.max, 1e-5, "\(name) refined keypoints px")
            XCTAssertEqual(got.dist[0], want.dist[0], accuracy: 1e-6, "\(name) k1")
            XCTAssertEqual(got.confidence, want.confidence, accuracy: 1e-6, "\(name) confidence")
        }
        print("PARITY refine worst kp px \(worst)")
    }
}
