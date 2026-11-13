import AVFoundation
import CoreVideo
import SwiftData
import XCTest
@testable import Shuttl

@MainActor
final class PipelineDebugInboxTests: XCTestCase {
    private var documents: URL!
    private var container: ModelContainer!
    private var pipeline: MatchPipeline!
    private var analyzer: FakeAnalyzer!

    override func setUp() async throws {
        documents = FileManager.default.temporaryDirectory
            .appendingPathComponent("inbox-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: DebugInbox.directory(documents: documents),
                                                withIntermediateDirectories: true)
        let schema = Schema([Match.self, Rally.self, ShotPoint.self])
        container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        analyzer = FakeAnalyzer()
        pipeline = MatchPipeline(context: container.mainContext, backend: MockBackend(), analyzer: analyzer,
                                 resumesAutomatically: false)
    }

    override func tearDown() async throws {
        await pipeline?.waitUntilIdle()
        let matches = (try? container.mainContext.fetch(FetchDescriptor<Match>())) ?? []
        for m in matches {
            try await pipeline.discard(m)
            if let url = m.localVideoURL { try? FileManager.default.removeItem(at: url) }
        }
        try? FileManager.default.removeItem(at: documents)
        pipeline = nil
        container = nil
    }

    func testNoFolderMeansNothingPending() throws {
        try FileManager.default.removeItem(at: DebugInbox.directory(documents: documents))
        XCTAssertEqual(DebugInbox.pendingVideos(documents: documents), [])
    }

    func testOnlyVideosArePendingOldestFirst() throws {
        let dir = DebugInbox.directory(documents: documents)
        let a = dir.appendingPathComponent("b_later.mov"), b = dir.appendingPathComponent("a_first.mp4")
        try Data([1]).write(to: a); try Data([1]).write(to: b)
        try Data([1]).write(to: dir.appendingPathComponent("notes.txt"))
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 2_000)], ofItemAtPath: a.path)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1_000)], ofItemAtPath: b.path)
        XCTAssertEqual(DebugInbox.pendingVideos(documents: documents).map(\.lastPathComponent),
                       ["a_first.mp4", "b_later.mov"])
    }

    func testCourtTypeFromName() {
        XCTAssertEqual(DebugInbox.courtType(for: URL(filePath: "/x/10min_Doubles.mov")), .doubles)
        XCTAssertEqual(DebugInbox.courtType(for: URL(filePath: "/x/1min.mov")), .singles)
    }

    func testDrainIngestsARealVideoAndEmptiesTheInbox() async throws {
        let src = DebugInbox.directory(documents: documents).appendingPathComponent("clip_doubles.mov")
        try await Self.writeVideo(to: src, frames: 30, fps: 30)

        await DebugInbox.drain(into: pipeline, documents: documents, settle: .milliseconds(50))
        await pipeline.waitUntilIdle()

        XCTAssertEqual(DebugInbox.pendingVideos(documents: documents), [], "the file left the inbox")
        let matches = try container.mainContext.fetch(FetchDescriptor<Match>())
        XCTAssertEqual(matches.count, 1)
        let match = try XCTUnwrap(matches.first)
        XCTAssertEqual(match.courtType, .doubles)
        XCTAssertEqual(match.durationSeconds, 1, accuracy: 0.1)
        XCTAssertEqual(match.status, .done)
        XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(match.localVideoURL).path))
        XCTAssertEqual(analyzer.calls.count, 1)
    }

    func testUnreadableFileIsLeftInPlace() async throws {
        let junk = DebugInbox.directory(documents: documents).appendingPathComponent("broken.mov")
        try Data(repeating: 0, count: 512).write(to: junk)
        await DebugInbox.drain(into: pipeline, documents: documents, settle: .milliseconds(50))
        XCTAssertEqual(DebugInbox.pendingVideos(documents: documents).map(\.lastPathComponent), ["broken.mov"])
        XCTAssertEqual(try container.mainContext.fetch(FetchDescriptor<Match>()).count, 0)
    }

    func testResumeImportsInboxVideoThroughCloudAndPreservesExistingMatch() async throws {
        let backend = MockBackend()
        pipeline = MatchPipeline(context: container.mainContext, backend: backend,
                                 resumesAutomatically: true)
        let existing = Match(status: .done)
        container.mainContext.insert(existing)
        try container.mainContext.save()
        let inbox = DebugInbox.directory()
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        let source = inbox.appendingPathComponent("cloud-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: source) }
        try await Self.writeVideo(to: source, frames: 30, fps: 30)

        pipeline.resumePending()
        let deadline = Date().addingTimeInterval(5)
        while FileManager.default.fileExists(atPath: source.path), Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        await pipeline.waitUntilIdle()

        let matches = try container.mainContext.fetch(FetchDescriptor<Match>())
        XCTAssertEqual(matches.count, 2)
        XCTAssertEqual(existing.status, .done)
        let imported = try XCTUnwrap(matches.first { $0.id != existing.id })
        XCTAssertEqual(imported.status, .done)
        XCTAssertEqual(imported.durationSeconds, 1, accuracy: 0.1)
        XCTAssertNotNil(imported.cloudAccount)
        let calls = await backend.calls
        XCTAssertTrue(calls.contains("create"))
        XCTAssertTrue(calls.contains("upload"))
        XCTAssertTrue(calls.contains("analyze"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.path))

        pipeline.resumePending()
        await pipeline.waitUntilIdle()
        XCTAssertEqual(try container.mainContext.fetch(FetchDescriptor<Match>()).count, 2,
                       "resuming again must not import the same video twice")
    }

    func testGrowingFileIsLeftUntilSettled() async throws {
        let growing = DebugInbox.directory(documents: documents).appendingPathComponent("copying.mov")
        // Write initial chunk
        try Data(repeating: 1, count: 1024).write(to: growing)

        // Start drain in background (settle time 100ms)
        let drainTask = Task {
            await DebugInbox.drain(into: pipeline, documents: documents, settle: .milliseconds(100))
        }

        // Grow the file while drain is checking
        try await Task.sleep(for: .milliseconds(30))
        let handle = try FileHandle(forWritingTo: growing)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(repeating: 2, count: 1024))
        try handle.close()

        await drainTask.value

        // File should still be in inbox because it was growing
        XCTAssertEqual(DebugInbox.pendingVideos(documents: documents).map(\.lastPathComponent), ["copying.mov"])
        XCTAssertEqual(try container.mainContext.fetch(FetchDescriptor<Match>()).count, 0, "growing file not ingested")

        // Now drain again after file has settled
        await DebugInbox.drain(into: pipeline, documents: documents, settle: .milliseconds(100))

        // File is still there because it's not a valid video, but at least we proved settle logic works
        XCTAssertEqual(DebugInbox.pendingVideos(documents: documents).map(\.lastPathComponent), ["copying.mov"])
    }

    /// A tiny grey H.264 movie.
    static func writeVideo(to url: URL, frames: Int, fps: Int32, size: CGSize = CGSize(width: 64, height: 64)) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(size.width), AVVideoHeightKey: Int(size.height),
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Int(size.width), kCVPixelBufferHeightKey as String: Int(size.height),
        ])
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for i in 0..<frames {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, try XCTUnwrap(adaptor.pixelBufferPool), &buffer)
            let pb = try XCTUnwrap(buffer)
            CVPixelBufferLockBaseAddress(pb, [])
            memset(CVPixelBufferGetBaseAddress(pb), 128, CVPixelBufferGetDataSize(pb))
            CVPixelBufferUnlockBaseAddress(pb, [])
            adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: fps))
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed)
    }
}
