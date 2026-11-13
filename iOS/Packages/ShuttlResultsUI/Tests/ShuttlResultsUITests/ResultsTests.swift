import Foundation
import XCTest
import SwiftUI
import ShuttlDesign
import simd
import ShuttlCore
@testable import ShuttlResultsUI

#if os(macOS)
import AppKit
#endif

@MainActor
final class ResultsTests: XCTestCase {
    private let realFixture = "results_match2_1_04_04_analysis"
    private let realPrefix = "results_match2_1_04_04"

    func testRealAnalysisDecodesAndSummaryMatchesPython() throws {
        let analysis = try loadAnalysisFixture(realFixture)
        let model = MatchResultViewModel(analysis: analysis)
        let expected = jsonObject(try loadJSONFixture("\(realPrefix)_summary"))

        XCTAssertEqual(model.summary.totalRallies, jsonInt(expected, "total_rallies"))
        XCTAssertEqual(model.summary.totalShots, jsonInt(expected, "total_shots"))
        assertClose(model.summary.maxSpeedKmh ?? -1, jsonDouble(expected, "max_speed_kmh") ?? -2, accuracy: 0.0001)
        assertClose(model.summary.averageSpeedKmh ?? -1, jsonDouble(expected, "avg_speed_kmh") ?? -2, accuracy: 0.0001)
        assertClose(model.summary.averageRallyShots, jsonDouble(expected, "avg_rally_shots") ?? -2, accuracy: 0.0001)
        XCTAssertEqual(model.summary.shotTypeCounts[.drive], jsonInt(jsonObject(expected["shot_type_counts"] ?? .null), "drive"))
        XCTAssertEqual(model.summary.endReasonCounts[.net], jsonInt(jsonObject(expected["end_reason_counts"] ?? .null), "net"))
        XCTAssertEqual(model.scoreProgression.count, 1)
        XCTAssertEqual(model.scoreProgression.first?.near, 1)
        XCTAssertEqual(model.scoreProgression.first?.far, 0)
        XCTAssertEqual(model.landingPoints.count, 1, "rally-level terminal landing should be plotted")
        XCTAssertFalse(model.landingPoints[0].isIn)
        XCTAssertEqual(model.rallies.first?.shots.count, 7)
    }

    func testSecondRealAnalysisAlsoDecodes() throws {
        let analysis = try loadAnalysisFixture("results_match2_1_03_03_analysis")
        let model = MatchResultViewModel(analysis: analysis)
        XCTAssertEqual(model.summary.totalRallies, 1)
        XCTAssertEqual(model.summary.totalShots, 12)
        XCTAssertTrue(model.isCalibrated)
        XCTAssertFalse(model.rallies.isEmpty)
    }

    func testMovementStatsMatchPythonReference() throws {
        let model = MatchResultViewModel(analysis: try loadAnalysisFixture(realFixture))
        for side in [CourtSide.near, .far] {
            let name = side == .near ? "near" : "far"
            let expected = jsonObject(try loadJSONFixture("\(realPrefix)_movement_\(name)"))
            let actual = try XCTUnwrap(model.movement[side])
            assertClose(actual.distanceM, jsonDouble(expected, "distance_m") ?? -1, accuracy: 0.0001)
            assertClose(actual.averageSpeedMS, jsonDouble(expected, "avg_speed_ms") ?? -1, accuracy: 0.0001)
            assertClose(actual.maxSpeedMS, jsonDouble(expected, "max_speed_ms") ?? -1, accuracy: 0.0001)
            assertClose(actual.courtCoverage, jsonDouble(expected, "court_coverage") ?? -1, accuracy: 0.0001)
            assertClose(actual.trackedSeconds, jsonDouble(expected, "tracked_s") ?? -1, accuracy: 0.0001)
            let expectedHeat = jsonRows(expected["heatmap"] ?? .null)
            XCTAssertEqual(actual.heatmap.count, expectedHeat.count)
            for (actualRow, expectedRow) in zip(actual.heatmap, expectedHeat) {
                assertClose(actualRow, expectedRow, accuracy: 0.00001)
            }
        }
    }

    func testSyntheticDoublesAllShotTypesAndEndings() throws {
        let analysis = syntheticAnalysis(camera: syntheticCamera(), includeDoubles: true)
        let model = MatchResultViewModel(analysis: analysis)

        XCTAssertEqual(model.summary.totalRallies, 3)
        XCTAssertEqual(model.summary.totalShots, ShotType.allCases.count)
        for type in ShotType.allCases {
            XCTAssertEqual(model.summary.shotTypeCounts[type], 1, type.rawValue)
        }
        XCTAssertEqual(model.summary.endReasonCounts[.inCourt], 1)
        XCTAssertEqual(model.summary.endReasonCounts[.out], 1)
        XCTAssertEqual(model.summary.endReasonCounts[.net], 1)
        XCTAssertEqual(model.summary.pointsWon[.near], 2)
        XCTAssertEqual(model.summary.pointsWon[.far], 1)
        XCTAssertEqual(model.movement[.near]?.playerStats.count, 2)
        XCTAssertEqual(model.movement[.far]?.playerStats.count, 2)
        XCTAssertEqual(model.landingPoints.count, 3)
        XCTAssertTrue(model.rallies.contains { $0.endReason == .out })
    }

    func testNoCameraNeverShowsMetricNumbersOrLandings() throws {
        let model = MatchResultViewModel(analysis: syntheticAnalysis(camera: nil, includeDoubles: false))
        XCTAssertFalse(model.isCalibrated)
        XCTAssertNil(model.summary.maxSpeedKmh)
        XCTAssertNil(model.summary.averageSpeedKmh)
        XCTAssertNil(model.summary.maxSmashKmh)
        XCTAssertTrue(model.landingPoints.isEmpty)
        XCTAssertTrue(model.rallies.flatMap(\.shots).allSatisfy { $0.speedKmh == nil })
        XCTAssertTrue(model.notices.contains { $0.kind == .calibration })
    }

    func testZeroRallyStateIsStable() throws {
        let analysis = MatchAnalysis(
            meta: VideoMeta(fps: 30, width: 1280, height: 720, nFrames: 0, timestamps: []),
            camera: nil
        )
        let model = MatchResultViewModel(analysis: analysis)
        XCTAssertEqual(model.summary.totalRallies, 0)
        XCTAssertEqual(model.summary.totalShots, 0)
        XCTAssertEqual(model.summary.averageRallyShots, 0)
        XCTAssertTrue(model.scoreProgression.isEmpty)
        XCTAssertTrue(model.landingPoints.isEmpty)
        XCTAssertTrue(model.notices.contains { $0.kind == .empty })
        XCTAssertEqual(model.filteredLandingPoints().count, 0)
    }

    func testRallyCorrectionFlags() {
        func rally(_ winner: CourtSide, _ conf: Double, _ source: String) -> RallySummary {
            RallySummary(id: 0, index: 1, startFrame: 0, endFrame: 10, shots: [], endReason: .inCourt, winner: winner,
                         server: .near, landingXY: nil, confidence: 0.9, outcomeConfidence: conf,
                         outcomeSource: source, serveMissed: false)
        }
        XCTAssertTrue(rally(.near, 0.4, "landing").isUncertain)
        XCTAssertTrue(rally(.unknown, 0.9, "landing").isUncertain)
        XCTAssertFalse(rally(.near, 0.8, "landing").isUncertain)
        XCTAssertFalse(rally(.near, 0.2, "user").isUncertain)
        XCTAssertTrue(rally(.far, 1, "user").isUserCorrected)
    }

    func testHugeFootPositionsAndNonFiniteScoresDoNotTrap() {
        let meta = VideoMeta(fps: 30, width: 1280, height: 720, nFrames: 90, timestamps: [])
        let players = (0..<60).map { f in
            PlayerObservation(frame: f, trackID: 1, bbox: BBox(x1: 0, y1: 0, x2: 1, y2: 1), score: 0.9,
                              side: .near, footWorld: SIMD2(f.isMultiple(of: 7) ? -1e20 : -2.0, f.isMultiple(of: 11) ? 1e20 : 0.5))
        }
        let score: [JSONValue] = [.object(["score": .object(["near": .number(.infinity), "far": .number(.nan)])])]
        let model = MatchResultViewModel(analysis: MatchAnalysis(meta: meta, camera: nil, players: players, score: score))
        XCTAssertNotNil(model.movement[.near])
    }

    #if os(macOS)
    func testRenderEveryScreenAtDefaultAndLargeDynamicType() throws {
        let real = MatchResultViewModel(analysis: try loadAnalysisFixture(realFixture))
        let synthetic = MatchResultViewModel(analysis: syntheticAnalysis(camera: nil, includeDoubles: true))
        let output = URL(fileURLWithPath: "/tmp/shuttl_results_ui", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

        try render(MatchResultView(viewModel: real), named: "real_overview_default", in: output, dynamicType: .large)
        try render(MatchResultView(viewModel: real), named: "real_overview_large", in: output, dynamicType: .accessibility3)
        try render(CourtDiagramView(viewModel: real), named: "real_court_default", in: output, dynamicType: .large)
        try render(RallyListView(viewModel: real), named: "real_rallies_default", in: output, dynamicType: .large)
        if let rally = real.rallies.first {
            try render(RallyDetailView(rally: rally), named: "real_rally_detail_large", in: output, dynamicType: .accessibility3)
        }
        try render(PlayerMovementView(viewModel: real), named: "real_movement_default", in: output, dynamicType: .large)
        try render(MatchResultView(viewModel: synthetic), named: "edge_no_camera_large", in: output, dynamicType: .accessibility3)
        let empty = MatchResultViewModel(analysis: MatchAnalysis(
            meta: VideoMeta(fps: 30, width: 1280, height: 720, nFrames: 0, timestamps: [])))
        try render(MatchResultView(viewModel: empty), named: "edge_empty_default", in: output, dynamicType: .large)

        let pngs = try FileManager.default.contentsOfDirectory(at: output, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "png" }
        XCTAssertGreaterThanOrEqual(pngs.count, 8)
        for png in pngs { XCTAssertGreaterThan(try png.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0, 1000, png.path) }
    }

    private func render<Content: View>(_ content: Content, named name: String, in directory: URL,
                                       dynamicType: DynamicTypeSize) throws {
        DisplayFace.register()
        let renderer = ImageRenderer(content: content
            .padding(.horizontal, content is MatchResultView ? 0 : 20)
            .frame(width: 390, height: 844, alignment: .top)
            .paperPage()
            .environment(\.dynamicTypeSize, dynamicType))
        renderer.scale = 2
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else {
            throw NSError(domain: "ShuttlResultsUITests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "ImageRenderer produced no PNG for \(name)"])
        }
        try png.write(to: directory.appendingPathComponent("\(name).png"))
    }
    #endif
}

private func syntheticCamera() -> CameraModel {
    CameraModel.lookAt(eye: SIMD3(-12, 0, 4), target: SIMD3(0, 0, 0), focalPx: 1400,
                       imageSize: ImageSize(width: 1280, height: 720))
}

private func syntheticAnalysis(camera: CameraModel?, includeDoubles: Bool) -> MatchAnalysis {
    let meta = VideoMeta(fps: 30, width: 1280, height: 720, nFrames: 360,
                         timestamps: (0..<360).map { Double($0) / 30 })
    let shotTypes = ShotType.allCases
    let endReasons: [RallyEndReason] = [.inCourt, .out, .net]
    let winners: [CourtSide] = [.near, .far, .near]
    let typeGroups: [[ShotType]] = [
        Array(shotTypes[0..<4]), Array(shotTypes[4..<8]), Array(shotTypes[8..<shotTypes.count])
    ]
    var rallies: [RallyV2] = []
    var score: [JSONValue] = []
    for rallyIndex in 0..<3 {
        let types = typeGroups[rallyIndex]
        let shots = types.enumerated().map { shotIndex, type in
            let frame = rallyIndex * 100 + shotIndex * 4
            let side: CourtSide = shotIndex.isMultiple(of: 2) ? .near : .far
            let hit = HitEvent(frame: frame, t: Double(frame) / 30,
                               xyPx: SIMD2(640, 300), hitter: side, confidence: 0.92)
            let segment = ShotSegment3D(
                startFrame: frame, endFrame: frame + 3, t0: Double(frame) / 30,
                p0: SIMD3(-1, 0, 1), v0: SIMD3(10 + Double(shotIndex), 0, 4),
                rmsePx: 1.1, frames: [], positions: [],
                landingXY: nil, confidence: 0.82, endReason: .hit
            )
            return ShotEventV2(hit: hit, segment: segment, shotType: type,
                               shotTypeConfidence: 0.9, hitterPosWorld: SIMD2(side == .near ? -2 : 2, 0),
                               hitHeightM: 1.2, strokeSide: .forehand)
        }
        let rallyShots = shots
        rallies.append(RallyV2(startFrame: rallyIndex * 100, endFrame: rallyIndex * 100 + 80,
                               shots: rallyShots, endReason: endReasons[rallyIndex],
                               landingXY: SIMD2(rallyIndex == 1 ? 6.9 : 2.0, Double(rallyIndex) - 1),
                               inOutMarginM: rallyIndex == 1 ? -0.2 : 0.3,
                               winner: winners[rallyIndex], confidence: 0.9,
                               server: .near, outcomeConfidence: 0.85))
        score.append(.object([
            "score": .object(["near": .number(Double(rallyIndex == 1 ? 1 : 2)),
                               "far": .number(Double(rallyIndex == 1 ? 1 : 0))]),
            "uncertain": .bool(false),
        ]))
    }

    var players: [PlayerObservation] = []
    if includeDoubles {
        for frame in [0, 15, 30] {
            for (trackID, side, x, y) in [(1, CourtSide.near, -2.0, 1.0), (2, CourtSide.near, -2.5, -1.0),
                                           (3, CourtSide.far, 2.0, 1.0), (4, CourtSide.far, 2.5, -1.0)] {
                players.append(PlayerObservation(frame: frame, trackID: trackID,
                                                 bbox: BBox(x1: 0, y1: 0, x2: 10, y2: 10), score: 0.9,
                                                 side: side, footWorld: SIMD2(x + Double(frame) / 100, y)))
            }
        }
    }
    return MatchAnalysis(meta: meta, camera: camera, players: players,
                          rallies: rallies, score: score)
}

private func jsonInt(_ object: [String: JSONValue], _ key: String) -> Int {
    guard let value = object[key] else { return 0 }
    if case .number(let number) = value { return Int(number) }
    return 0
}

private func jsonRows(_ value: JSONValue) -> [[Double]] {
    guard case .array(let rows) = value else { return [] }
    return rows.map { row in
        guard case .array(let values) = row else { return [] }
        return values.compactMap { $0.double }
    }
}

private func assertClose(_ actual: [Double], _ expected: [Double], accuracy: Double,
                         file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (a, e) in zip(actual, expected) {
        XCTAssert(abs(a - e) <= accuracy * max(1, abs(e)), "\(a) != \(e)", file: file, line: line)
    }
}

final class CautionNoticeTests: XCTestCase {
    func testCautionsShareOneNoteInOrderAndVanishWhenEmpty() {
        let lines = ["录像中途手机被碰过。", "有 2 个回合的落点算不出来。", "球场远端看得比较斜。"]
        let all = MatchResultViewModel(analysis: syntheticAnalysis(camera: syntheticCamera(), includeDoubles: false),
                                       cautions: lines)
        XCTAssertEqual(all.notices.first?.kind, .caution)
        XCTAssertEqual(all.cautionNotice?.lines, lines)
        XCTAssertEqual(all.notices.filter { $0.kind == .caution }.count, 1)

        let one = MatchResultViewModel(analysis: syntheticAnalysis(camera: syntheticCamera(), includeDoubles: false),
                                       cautions: [lines[2]])
        XCTAssertEqual(one.cautionNotice?.lines, [lines[2]])

        // A low-confidence rally folds into the caution card as its last line.
        var shaky = syntheticAnalysis(camera: syntheticCamera(), includeDoubles: false)
        shaky.rallies[0].confidence = 0.3
        let merged = MatchResultViewModel(analysis: shaky, cautions: [lines[2]])
        XCTAssertFalse(merged.notices.contains { $0.kind == .confidence })
        XCTAssertEqual(merged.cautionNotice?.lines.count, 2)
        XCTAssertEqual(merged.cautionNotice?.lines.first, lines[2])
        let alone = MatchResultViewModel(analysis: shaky)
        XCTAssertTrue(alone.notices.contains { $0.kind == .confidence })

        let none = MatchResultViewModel(analysis: syntheticAnalysis(camera: syntheticCamera(), includeDoubles: false),
                                        cautions: ["", ""])
        XCTAssertNil(none.cautionNotice)
    }
}
