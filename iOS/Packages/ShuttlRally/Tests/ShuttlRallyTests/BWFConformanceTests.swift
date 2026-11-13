import CryptoKit
import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlRally

/// BWF scoring conformance: scoreMatch / impliedWinner against the DELIVERY_R2
/// reference, the same vectors as backend/tests/test_bwf_conformance.py
/// (fixture: backend/scripts/swift_fixtures/bwf.py). Hand + probe cases compare
/// every field of every rally; the 4000 random cases compare a sha256 of the
/// canonical row lines (canon() == the exporter's canon(), byte for byte).
final class BWFConformanceTests: XCTestCase {
    struct Inp: Decodable {
        var id: String
        var mode: String
        var config: [String: Int]
        var firstServer: CourtSide
        var starters: [String: [String]]?
        enum CodingKeys: String, CodingKey { case id, mode, config, firstServer = "first_server", starters }
    }
    struct Full: Decodable { var inp: Inp; var winners: [CourtSide]; var expected: [JSONValue]
        init(from d: Decoder) throws {
            inp = try Inp(from: d)
            let c = try d.container(keyedBy: K.self)
            winners = try c.decode([CourtSide].self, forKey: .winners); expected = try c.decode([JSONValue].self, forKey: .expected)
        }
        enum K: String, CodingKey { case winners, expected }
    }
    struct Digest: Decodable { var inp: Inp; var winners: [CourtSide]; var nRows: Int; var sha256: String
        init(from d: Decoder) throws {
            inp = try Inp(from: d)
            let c = try d.container(keyedBy: K.self)
            winners = try c.decode([CourtSide].self, forKey: .winners)
            nRows = try c.decode(Int.self, forKey: .nRows); sha256 = try c.decode(String.self, forKey: .sha256)
        }
        enum K: String, CodingKey { case winners, nRows = "n_rows", sha256 }
    }
    struct Infer: Decodable { var inp: Inp; var history: [CourtSide]; var next: [String: String]; var expected: CourtSide
        init(from d: Decoder) throws {
            inp = try Inp(from: d)
            let c = try d.container(keyedBy: K.self)
            history = try c.decode([CourtSide].self, forKey: .history)
            next = try c.decode([String: String].self, forKey: .next); expected = try c.decode(CourtSide.self, forKey: .expected)
        }
        enum K: String, CodingKey { case history, next, expected }
    }
    struct Doc: Decodable { var full: [Full]; var digest: [Digest]; var infer: [Infer] }

    static let doc: Doc? = try? Fixture.load(Doc.self, "bwf_conformance")

    /// The app's rows for a winners-only case, in the reference's shape ("index", no frames).
    static func rows(_ inp: Inp, _ winners: [CourtSide]) -> [[String: JSONValue]] {
        let rs = winners.enumerated().map { RallyV2(startFrame: $0.offset, endFrame: $0.offset, winner: $0.element, confidence: 1.0) }
        let c = inp.config
        let out = scoreMatch(rs, firstServer: inp.firstServer, gamesToWin: c["games_to_win"] ?? 2, points: c["points"] ?? 21,
                             cap: c["cap"] ?? 30, interval: c["interval"] ?? 11, doubles: inp.mode == "doubles",
                             starters: inp.starters)
        return out.map { v in
            guard case .object(var o) = v else { return [:] }
            o["index"] = o.removeValue(forKey: "rally")
            for k in ["start_frame", "end_frame", "observed_server"] { o.removeValue(forKey: k) }
            return o
        }
    }

    private static func s(_ v: JSONValue?) -> String {
        switch v {
        case .number(let x)?: return String(Int(x))
        case .string(let t)?: return t
        case .bool(let b)?: return b ? "1" : "0"
        case .array(let a)?: return a.map { s($0) }.joined(separator: ",")
        default: return "-"
        }
    }
    private static func pair(_ v: JSONValue?, _ a: String, _ b: String) -> String {
        guard case .object(let o)? = v else { return "-" }
        return s(o[a]) + "," + s(o[b])
    }

    /// Canonical row line (== scripts/swift_fixtures/bwf.py canon()).
    static func canon(_ r: [String: JSONValue]) -> String {
        var f = [s(r["index"]), s(r["game"]), s(r["server"]), s(r["service_court"]), s(r["winner"]),
                 pair(r["score"], "near", "far"), pair(r["games"], "near", "far"),
                 pair(r["score_by_player"], "A", "B"), pair(r["games_by_player"], "A", "B"),
                 s(r["game_over"]), s(r["game_winner"]), s(r["match_over"]), s(r["ends_changed"]),
                 s(r["next_server"]), s(r["uncertain"]), s(r["reasons"])]
        if case .object(let p)? = r["positions"] {
            f += [s(r["server_player"]), s(r["receiver_player"]), ["A1", "A2", "B1", "B2"].map { s(p[$0]) }.joined(separator: ",")]
        }
        return f.joined(separator: "|")
    }

    static func digest(_ rows: [[String: JSONValue]]) -> String {
        let text = rows.map(canon).joined(separator: "\n")
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private func loaded() throws -> Doc {
        guard let d = Self.doc else { throw XCTSkip("bwf_conformance.json.zz missing — run scripts.swift_fixtures.bwf") }
        return d
    }

    func testHandAndProbeCasesEveryField() throws {
        let d = try loaded()
        XCTAssertEqual(d.full.count, 67)                      // 40 singles + 22 doubles + 5 probes
        var bad: [String] = [], n = 0
        for c in d.full {
            let got = Self.rows(c.inp, c.winners)
            n += got.count
            var m = Mismatches()
            jsonDiff(.array(got.map { .object($0) }), .array(c.expected), path: c.inp.id, &m)
            if !m.isEmpty { bad.append(m.items.prefix(3).joined(separator: "; ")) }
        }
        print("[bwf] hand + probes: \(d.full.count - bad.count)/\(d.full.count) cases, \(n) rallies")
        XCTAssert(bad.isEmpty, bad.prefix(10).joined(separator: "\n"))
    }

    func testRandomCasesDigest() throws {
        let d = try loaded()
        XCTAssertEqual(d.digest.count, 4000)
        var bad: [String] = [], n = 0
        for c in d.digest {
            let got = Self.rows(c.inp, c.winners)
            n += got.count
            if got.count != c.nRows || Self.digest(got) != c.sha256 { bad.append(c.inp.id) }
        }
        print("[bwf] random singles + doubles: \(d.digest.count - bad.count)/\(d.digest.count) cases, \(n) rallies")
        XCTAssert(bad.isEmpty, "\(bad.count) cases differ: " + bad.prefix(10).joined(separator: ", "))
    }

    func testInferCases() throws {
        let d = try loaded()
        XCTAssertEqual(d.infer.count, 200)
        var bad: [String] = []
        for c in d.infer {
            let cf = c.inp.config
            var st = BWFScore(firstServer: c.inp.firstServer, gamesToWin: cf["games_to_win"] ?? 2, points: cf["points"] ?? 21,
                              cap: cf["cap"] ?? 30, interval: cf["interval"] ?? 11, doubles: c.inp.mode == "doubles",
                              starters: c.inp.starters)
            for w in c.history { if w != .unknown { _ = st.point(w) } else { st.begin() } }
            let obs = ServeObservation(server: c.next["server"].flatMap(CourtSide.init(rawValue:)),
                                       serviceCourt: c.next["service_court"], serverPlayer: c.next["server_player"],
                                       receiverPlayer: c.next["receiver_player"])
            let got = impliedWinner(st, obs)
            if got != c.expected { bad.append("\(c.inp.id): \(got) != \(c.expected)") }
        }
        print("[bwf] infer: \(d.infer.count - bad.count)/\(d.infer.count)")
        XCTAssert(bad.isEmpty, bad.prefix(10).joined(separator: "\n"))
    }

    /// DELIVERY_R2 D1-D4 stated directly (same cases as the Python tests).
    func testNewMatchFindings() {
        let w: [CourtSide] = Array(repeating: .near, count: 21) + Array(repeating: .far, count: 21)
        let inp = Inp(id: "d", mode: "singles", config: [:], firstServer: .near, starters: nil)
        let r = Self.rows(inp, w + [.far])
        XCTAssertEqual(Self.canon(r[42]), "42|1|far|right|far|0,1|0,0|1,0|0,0|0|-|0|0|far|0|new_match")
        let u = Self.rows(inp, w + [.unknown, .near])
        XCTAssertEqual(Self.canon(u[42]), "42|1|far|right|unknown|0,0|0,0|0,0|0,0|0|-|0|0|far|1|new_match,winner_unknown")
        XCTAssertEqual(Self.s(u[43]["reasons"]), "")
    }
}
