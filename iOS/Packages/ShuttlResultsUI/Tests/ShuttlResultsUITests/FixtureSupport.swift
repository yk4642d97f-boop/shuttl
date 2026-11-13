import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlResultsUI

func resultsFixtureURL(_ name: String) throws -> URL {
    guard let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures") else {
        throw NSError(domain: "ShuttlResultsUITests", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "missing fixture \(name).json"])
    }
    return url
}

func loadAnalysisFixture(_ name: String) throws -> MatchAnalysis {
    try ShuttlJSON.load(MatchAnalysis.self, from: resultsFixtureURL(name))
}

func loadJSONFixture(_ name: String) throws -> JSONValue {
    try ShuttlJSON.load(JSONValue.self, from: resultsFixtureURL(name))
}

func jsonObject(_ value: JSONValue) -> [String: JSONValue] {
    if case .object(let object) = value { return object }
    return [:]
}

func jsonDouble(_ object: [String: JSONValue], _ key: String) -> Double? {
    object[key]?.double
}

func assertClose(_ actual: Double, _ expected: Double, accuracy: Double = 1e-6,
                 file: StaticString = #filePath, line: UInt = #line) {
    XCTAssert(abs(actual - expected) <= accuracy * max(1, abs(expected)),
              "\(actual) != \(expected)", file: file, line: line)
}
