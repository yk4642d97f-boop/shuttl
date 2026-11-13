import XCTest
import SwiftData
@testable import Shuttl

/// Opt-in end-to-end run against a real local backend (uvicorn + Celery worker):
///   TEST_RUNNER_SHUTTL_LIVE=1 TEST_RUNNER_SHUTTL_API_BASE_URL=http://127.0.0.1:8000 xcodebuild test …
/// Skipped otherwise, so the normal suite never touches the network.
@MainActor
final class LivePipelineSmokeTests: XCTestCase {
    private static let clip = "/path/to/shuttl/backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_05_02.mp4"

    func testRecordingUploadsAnalysesAndPersists() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["SHUTTL_LIVE"] == "1", "set TEST_RUNNER_SHUTTL_LIVE=1 to run against a live backend")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.clip), "sample clip not found")

        _ = try await APIClient.shared.login(email: env["SHUTTL_EMAIL"] ?? "test@shuttl.app",
                                             password: env["SHUTTL_PASSWORD"] ?? "shuttl123")

        let fileURL = VideoStore.newRecordingURL()
        try FileManager.default.copyItem(at: URL(filePath: Self.clip), to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let schema = Schema([Match.self, Rally.self, ShotPoint.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        var options = JobPollingService.Options.standard
        options.interval = .seconds(2)
        options.pendingTimeout = .seconds(120)
        options.overallTimeout = .seconds(15 * 60)
        let pipeline = MatchPipeline(context: container.mainContext, backend: LiveMatchBackend(),
                                     pollOptions: options, resumesAutomatically: false)

        let localID: UUID
        let match = try pipeline.ingestRecording(fileURL: fileURL, durationSeconds: 17.2, courtType: .singles)
        localID = match.id
        let started = Date()
        var sawUploadProgress = false, sawAnalyzing = false
        while pipeline.isRunning(match) {
            if let a = pipeline.activity(for: match) {
                if a.phase == .uploading, a.uploadFraction != nil { sawUploadProgress = true }
                if a.phase == .analyzing { sawAnalyzing = true }
            }
            try await Task.sleep(for: .milliseconds(200))
            XCTAssertLessThan(Date().timeIntervalSince(started), 16 * 60, "pipeline did not finish")
        }

        print("[LiveSmoke] status=\(match.status) id=\(match.id) job=\(match.cloudJobID ?? "-") " +
              "rallies=\(match.rallies.count) shots=\(match.allShots.count) reason=\(match.failureReason ?? "-") " +
              "elapsed=\(Int(Date().timeIntervalSince(started)))s")
        XCTAssertEqual(match.status, .done, match.failureReason ?? "")
        XCTAssertNotEqual(match.id, localID, "local id replaced by the server id")
        XCTAssertNotNil(match.cloudJobID)
        // Sampled every 200 ms: a 9 MB loopback upload can finish between samples, so only log it.
        print("[LiveSmoke] sawUploadProgress=\(sawUploadProgress) sawAnalyzing=\(sawAnalyzing)")
        for shot in match.allShots {
            XCTAssertTrue((0...1).contains(shot.normalizedX) && (0...1).contains(shot.normalizedY))
        }
        let remote = try await APIClient.shared.getMatch(matchID: match.id)
        XCTAssertEqual(remote.status, "done")
    }
}
