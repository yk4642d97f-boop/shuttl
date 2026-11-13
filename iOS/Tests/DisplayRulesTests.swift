import XCTest
import ShuttlRuntime
@testable import Shuttl

/// UI display rules owned by D: speeds only from calibrated matches, chip severity levels.
@MainActor
final class DisplayRulesTests: XCTestCase {
    private func match(speeds: [Float]) -> Match {
        let m = Match(createdAt: .now, durationSeconds: 60, status: .done, courtType: .singles)
        let r = Rally(startFrame: 0, endFrame: 100)
        for (i, s) in speeds.enumerated() { r.shots.append(ShotPoint(frameIndex: i, xPx: 0, yPx: 0, speedKmh: s)) }
        m.rallies = [r]
        return m
    }

    func testUncalibratedMatchHasNoSpeed() {
        let m = match(speeds: [0, 0, 0])
        XCTAssertTrue(m.lacksSpeeds)
        XCTAssertEqual(m.maxSpeedText, "--")
        XCTAssertTrue(m.measuredShots.isEmpty)
    }

    func testCalibratedMatchIgnoresShotsWithoutFlight() {
        let m = match(speeds: [0, 212, .nan, 88])
        XCTAssertFalse(m.lacksSpeeds)
        XCTAssertEqual(m.maxSpeedText, "212")
        XCTAssertEqual(m.measuredShots.count, 2)
    }

    func testChipSeverity() {
        XCTAssertEqual(SystemChipSeverity(storage: .low), .mild)
        XCTAssertEqual(SystemChipSeverity(storage: .refuseStart), .serious)
        XCTAssertEqual(SystemChipSeverity(storage: .stopRecording), .serious)
        XCTAssertEqual(SystemChipSeverity(thermal: .hold), .mild)
        XCTAssertEqual(SystemChipSeverity(thermal: .run), .mild)
        XCTAssertEqual(SystemChipSeverity(thermal: .pause), .serious)
    }

    func testChipsAreShortButSpeakTheFullSentence() {
        let guardrail = StorageGuard(freeSpace: { 0 })
        let ok = guardrail.evaluate(freeBytes: 50_000_000_000)
        XCTAssertNil(SystemChipItem(storage: ok))
        let low = guardrail.evaluate(freeBytes: 1_783_000_000)   // 8.7 min above the 1 GB floor
        XCTAssertEqual(low.level, .low)
        let chip = SystemChipItem(storage: low)
        XCTAssertEqual(chip?.text, "空间只够再录约 8 分钟")
        XCTAssertEqual(chip?.spoken, low.message)
        XCTAssertNil(SystemChipItem(thermal: .run, message: nil))
        for d in [ThermalDecision.run, .hold, .pause] {
            let c = SystemChipItem(thermal: d, message: "x")
            XCTAssertLessThanOrEqual(c?.text.count ?? 99, 14, "\(d)")
        }
    }
}
