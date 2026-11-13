import XCTest
@testable import Shuttl

final class VideoStoreTests: XCTestCase {
    private let docs = URL(filePath: "/var/mobile/Containers/Data/Application/NEW-UUID/Documents", directoryHint: .isDirectory)

    func testStoresPathsInsideDocumentsRelatively() {
        let file = docs.appendingPathComponent("Matches/match_1.mov")
        XCTAssertEqual(VideoStore.storedPath(for: file, documents: docs), "Matches/match_1.mov")
    }

    func testKeepsPathsOutsideDocumentsAbsolute() {
        let file = URL(filePath: "/tmp/demo.mov")
        XCTAssertEqual(VideoStore.storedPath(for: file, documents: docs), "/tmp/demo.mov")
    }

    func testResolvesRelativePathAgainstCurrentDocuments() {
        XCTAssertEqual(VideoStore.resolve("Matches/match_1.mov", documents: docs)?.path,
                       docs.appendingPathComponent("Matches/match_1.mov").path)
    }

    func testReRootsLegacyAbsolutePathFromOldContainer() {
        // Path written by a previous install: different container UUID, file no longer there.
        let legacy = "/var/mobile/Containers/Data/Application/OLD-UUID/Documents/Matches/match_7.mov"
        XCTAssertEqual(VideoStore.resolve(legacy, documents: docs)?.path,
                       docs.appendingPathComponent("Matches/match_7.mov").path)
    }

    func testLegacyAbsolutePathThatStillExistsIsUsedAsIs() throws {
        let real = VideoStore.newRecordingURL(now: Date(timeIntervalSince1970: 42))
        try Data([1]).write(to: real)
        defer { try? FileManager.default.removeItem(at: real) }
        XCTAssertEqual(VideoStore.resolve(real.path)?.path, real.path)
    }

    func testAbsolutePathOutsideAnyDocumentsIsReturnedUnchanged() {
        XCTAssertEqual(VideoStore.resolve("/tmp/demo.mov", documents: docs)?.path, "/tmp/demo.mov")
    }

    func testNilOrEmptyResolvesToNil() {
        XCTAssertNil(VideoStore.resolve(nil, documents: docs))
        XCTAssertNil(VideoStore.resolve("", documents: docs))
    }

    func testRoundTripThroughTheRealDocumentsDirectory() throws {
        let url = VideoStore.newRecordingURL(now: Date(timeIntervalSince1970: 43))
        let stored = VideoStore.storedPath(for: url)
        XCTAssertEqual(stored, "Matches/\(url.lastPathComponent)")
        XCTAssertEqual(VideoStore.resolve(stored)?.standardizedFileURL.path, url.standardizedFileURL.path)
    }

    func testMatchLocalVideoURLUsesResolution() {
        let match = Match(localVideoPath: "Matches/match_9.mov")
        XCTAssertEqual(match.localVideoURL?.lastPathComponent, "match_9.mov")
        XCTAssertEqual(match.localVideoURL?.deletingLastPathComponent().deletingLastPathComponent().standardizedFileURL.path,
                       VideoStore.documentsDirectory.standardizedFileURL.path)
    }
}
