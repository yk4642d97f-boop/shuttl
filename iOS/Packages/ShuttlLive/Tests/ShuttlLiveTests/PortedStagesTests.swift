import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlLive

/// Contract checks across package boundaries — each stage has its own parity
/// suite; these catch glue mistakes (lost side channels, argument order, shapes).
final class PortedStagesTests: XCTestCase {

    private func meta(_ n: Int, fps: Double = 60) -> VideoMeta {
        VideoMeta(fps: fps, width: 1920, height: 1080, nFrames: n, timestamps: (0..<n).map { Double($0) / fps })
    }

    /// A physics-true flight projected through a behind-baseline camera, as detector candidates.
    private func flightCandidates(camera: CameraModel, n: Int, fps: Double) -> [[ShuttleCandidate]] {
        let times = (0..<n).map { Double($0) / fps }
        let pts = Physics.simulate(p0: [-5.5, 0.4, 2.4], v0: [24, -0.8, 7], times: times)
        return pts.map { p in
            let uv = camera.project(p)
            guard uv.isFinite, p.z > 0, uv.x >= 0, uv.y >= 0, uv.x < 1920, uv.y < 1080 else { return [] }
            return [ShuttleCandidate(x: uv.x, y: uv.y, score: 0.95)]
        }
    }

    func testLinkThenReconstructThenRalliesChainRuns() throws {
        let camera = CameraModel.lookAt(eye: [-11.5, 0.3, 3.2], target: [0.5, 0, 0], focalPx: 1450,
                                        imageSize: ImageSize(width: 1920, height: 1080))
        let fps = 60.0, n = 90
        let m = meta(n, fps: fps)
        let track = try PortedStages.link(flightCandidates(camera: camera, n: n, fps: fps), meta: m)
        XCTAssertEqual(track.n, n)
        XCTAssertGreaterThan(track.visible.filter { $0 }.count, n / 2, "linker should adopt most of a clean flight")

        _ = PortedStages.hits(track: track, meta: m, camera: camera, players: nil)
        let hit = HitEvent(frame: 0, t: 0, xyPx: track.xy[0], hitter: .near, confidence: 0.9)
        let recon = PortedStages.reconstruct(track: track, hits: [hit], camera: camera, meta: m, players: nil)
        XCTAssertEqual(recon.landingSigmaM.count, recon.segments.count, "one sigma per segment")

        let rallies = PortedStages.rallies(track: track, hits: [hit], reconstruction: recon, meta: m,
                                           camera: camera, players: nil)
        let classified = rallies.map { PortedStages.shots($0, camera: camera, players: nil) }
        XCTAssertEqual(classified.map(\.startFrame), rallies.map(\.startFrame))
        _ = PortedStages.score(classified)
    }

    func testEmptyInputsDegradeWithoutCrashing() throws {
        let m = meta(30)
        let track = try PortedStages.link(Array(repeating: [], count: 30), meta: m)
        XCTAssertEqual(track.visible.filter { $0 }.count, 0)
        let camera = CameraModel.lookAt(eye: [-11, 0, 3], target: [0, 0, 0], focalPx: 1400,
                                        imageSize: ImageSize(width: 1920, height: 1080))
        let recon = PortedStages.reconstruct(track: track, hits: [], camera: camera, meta: m, players: nil)
        XCTAssertTrue(recon.segments.isEmpty)
        let rallies = PortedStages.rallies(track: track, hits: [], reconstruction: recon, meta: m, camera: nil, players: nil)
        XCTAssertTrue(PortedStages.score(rallies).count == rallies.count)
    }
}
