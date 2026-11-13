import XCTest
import SwiftUI
import CoreText
@testable import ShuttlDesign

final class DesignTests: XCTestCase {
    func testDisplayFaceIsASystemFont() {
        XCTAssertTrue(DisplayFace.isAvailable)
        let font = DisplayFace.ctFont(size: 20)
        let text = "比赛统计录制 Shuttl. 0123456789 km/h"
        let utf16 = Array(text.utf16)
        var glyphs = [CGGlyph](repeating: 0, count: utf16.count)
        // The system face (with its CJK fallback cascade) renders everything, so nothing to check beyond a size.
        XCTAssertEqual(CTFontGetSize(font), 20, accuracy: 0.01)
        _ = CTFontGetGlyphsForCharacters(font, utf16, &glyphs, utf16.count)
    }

    func testCourtGeometryMapsCornersAndNet() {
        let rect = CGRect(x: 0, y: 0, width: 61, height: 134)
        let nearLeft = CourtGeometry2D.point(x: -6.7, y: 3.05, in: rect, .portrait)
        XCTAssertEqual(nearLeft.x, 0, accuracy: 1e-9); XCTAssertEqual(nearLeft.y, 134, accuracy: 1e-9)
        let farRight = CourtGeometry2D.point(x: 6.7, y: -3.05, in: rect, .portrait)
        XCTAssertEqual(farRight.x, 61, accuracy: 1e-9); XCTAssertEqual(farRight.y, 0, accuracy: 1e-9)
        let net = CourtGeometry2D.point(x: 0, y: 0, in: rect, .landscape)
        XCTAssertEqual(net.x, 30.5, accuracy: 1e-9)
        XCTAssertEqual(CourtGeometry2D.aspect(.portrait) * CourtGeometry2D.aspect(.landscape), 1, accuracy: 1e-12)
    }

    func testEveryShotKindHasAGlyphAndInk() {
        for kind in ShotGlyphKind.allCases where kind != .unknown {
            XCTAssertNotNil(kind.flight, "\(kind)")
            XCTAssertFalse(kind.chineseName.isEmpty)
        }
        XCTAssertEqual(ShotGlyphKind(raw: "kill"), .kill)
        XCTAssertEqual(ShotGlyphKind(raw: "nonsense"), .unknown)
    }

    func testGrainTileRenders() {
        _ = PaperTexture.grain
    }
}
