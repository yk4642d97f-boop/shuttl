import XCTest
import AVFoundation
@testable import Shuttl

final class CaptureFormatSelectorTests: XCTestCase {
    private typealias C = CaptureFormatSelector.Candidate
    private let v420 = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange   // '420v'
    private let f420 = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange    // '420f'

    private func c(_ w: Int32, _ h: Int32, _ fps: Double, binned: Bool = false, pixel: FourCharCode? = nil) -> C {
        C(width: w, height: h, pixelFormat: pixel ?? v420, frameRateRanges: [1...fps], isBinned: binned)
    }

    /// The regression: the old "first ≥60 fps" rule picked the low-res format at index 0.
    func testPicksExact1080p60NotTheFirstSixtyFPSFormat() {
        let formats = [c(640, 480, 60), c(1280, 720, 60), c(1920, 1080, 30), c(1920, 1080, 60), c(3840, 2160, 60)]
        let choice = CaptureFormatSelector.select(from: formats)
        XCTAssertEqual(choice?.index, 3)
        XCTAssertEqual(choice?.fps, 60)
    }

    func testPrefersNonBinnedOverBinned() {
        let formats = [c(1920, 1080, 60, binned: true), c(1920, 1080, 60, binned: false)]
        XCTAssertEqual(CaptureFormatSelector.select(from: formats)?.index, 1)
    }

    func testBinnedIsAcceptedWhenItIsTheOnly1080p60() {
        let formats = [c(1920, 1080, 30), c(1920, 1080, 120, binned: true)]
        let choice = CaptureFormatSelector.select(from: formats)
        XCTAssertEqual(choice?.index, 1)
        XCTAssertEqual(choice?.fps, 60)
    }

    func testPrefersPlainSixtyOverSlowMotionModes() {
        let formats = [c(1920, 1080, 240), c(1920, 1080, 120), c(1920, 1080, 60)]
        XCTAssertEqual(CaptureFormatSelector.select(from: formats)?.index, 2)
    }

    func testNonBinnedBeatsLowerMaxFPS() {
        let formats = [c(1920, 1080, 60, binned: true), c(1920, 1080, 120, binned: false)]
        XCTAssertEqual(CaptureFormatSelector.select(from: formats)?.index, 1)
    }

    func testRequires420vPixelFormat() {
        let formats = [c(1920, 1080, 60, pixel: f420), c(1920, 1080, 60, pixel: v420)]
        XCTAssertEqual(CaptureFormatSelector.select(from: formats)?.index, 1)
        XCTAssertNil(CaptureFormatSelector.select(from: [c(1920, 1080, 60, pixel: f420)]))
    }

    func testFallsBackTo1080p30() {
        let formats = [c(1280, 720, 60), c(1920, 1080, 30), c(3840, 2160, 60)]
        let choice = CaptureFormatSelector.select(from: formats)
        XCTAssertEqual(choice?.index, 1)
        XCTAssertEqual(choice?.fps, 30)
    }

    func testNilWithoutAny1080pKeepsThePreset() {
        XCTAssertNil(CaptureFormatSelector.select(from: [c(1280, 720, 60), c(3840, 2160, 60)]))
        XCTAssertNil(CaptureFormatSelector.select(from: []))
    }

    func testFrameRateRangeMustContainTheTarget() {
        // A 1080p mode that only runs 100–120 fps can't be locked to 60.
        let highOnly = C(width: 1920, height: 1080, pixelFormat: v420, frameRateRanges: [100...120], isBinned: false)
        let multi = C(width: 1920, height: 1080, pixelFormat: v420, frameRateRanges: [1...30, 50...60], isBinned: false)
        let choice = CaptureFormatSelector.select(from: [highOnly, multi])
        XCTAssertEqual(choice?.index, 1)
        XCTAssertEqual(choice?.fps, 60)
    }

    func testPortraitDimensionsAreNotMistakenFor1080p() {
        // Device formats are landscape; a 1080×1920 entry would be something else entirely.
        XCTAssertNil(CaptureFormatSelector.select(from: [c(1080, 1920, 60)]))
    }
}
