import XCTest
@testable import Shuttl

final class ShuttlTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testCourtMapperIdentity() throws {
        let mapper = CourtMapper.identity
        let result = mapper.imagePixelToMeters(CGPoint(x: 100, y: 200))
        XCTAssertEqual(result?.x, 100)
        XCTAssertEqual(result?.y, 200)
    }

    func testShotTypeDisplayName() throws {
        XCTAssertEqual(ShotType.smash.displayName, "扣杀")
        XCTAssertEqual(ShotType.clear.displayName, "高远")
        XCTAssertEqual(ShotType.drop.displayName, "吊球")
        XCTAssertEqual(ShotType.net.displayName, "网前")
        XCTAssertEqual(ShotType.drive.displayName, "平抽")
        XCTAssertEqual(ShotType.serve.displayName, "发球")
    }

    func testMatchStatusDisplayNames() throws {
        XCTAssertEqual(MatchStatus.recording.displayName, "录制中")
        XCTAssertEqual(MatchStatus.done.displayName, "已完成")
        XCTAssertEqual(MatchStatus.failed.displayName, "分析失败")
    }

    func testSegmentIntoRalliesSortsFramesAndSplitsOnlyPastThreshold() {
        let points = [30, 19, 9, 20, 0, 10].map { frame in
            TrajectoryPointDTO(
                frame: frame, xPx: 0, yPx: 0, xMeters: 0, yMeters: 0,
                speedKmh: 0, shotType: "unknown"
            )
        }

        let rallies = JobPollingService.segmentIntoRallies(points, fps: 30)

        XCTAssertEqual(rallies.map { $0.map(\.frame) }, [[0, 9, 10, 19, 20], [30]])
        XCTAssertTrue(JobPollingService.segmentIntoRallies([], fps: 30).isEmpty)
    }
}
