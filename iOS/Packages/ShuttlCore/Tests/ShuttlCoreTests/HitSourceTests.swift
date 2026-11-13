import XCTest
@testable import ShuttlCore

/// HitEvent.source: default "kink" is never written, so every existing JSON stays byte-identical.
final class HitSourceTests: XCTestCase {
    func testDefaultIsOmittedAndInferredRoundTrips() throws {
        let plain = HitEvent(frame: 12, t: 0.2, xyPx: [640, 360], hitter: .near, confidence: 0.8, playerTrackID: 3)
        let pj = String(decoding: try ShuttlJSON.encoder().encode(plain), as: UTF8.self)
        XCTAssertFalse(pj.contains("source"), pj)
        XCTAssertEqual(try ShuttlJSON.decoder().decode(HitEvent.self, from: Data(pj.utf8)), plain)
        XCTAssertFalse(plain.isInferred)

        var gap = plain; gap.source = "gap_contact"
        let gj = try ShuttlJSON.encoder().encode(gap)
        XCTAssertTrue(String(decoding: gj, as: UTF8.self).contains("\"source\":\"gap_contact\""))
        let back = try ShuttlJSON.decoder().decode(HitEvent.self, from: gj)
        XCTAssertEqual(back, gap)
        XCTAssertTrue(back.isInferred)
        // Python's JSON for an old hit (no key) decodes to the default.
        let py = #"{"frame":1,"t":0.5,"xy_px":[1,2],"hitter":"far","confidence":0.4,"player_track_id":null}"#
        XCTAssertEqual(try ShuttlJSON.decoder().decode(HitEvent.self, from: Data(py.utf8)).source, "kink")
    }
}
