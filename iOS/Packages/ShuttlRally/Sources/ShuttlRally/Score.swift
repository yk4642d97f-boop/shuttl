import Foundation
import ShuttlCore

/// rally._Score: BWF rally-point state for one recording (a match, then further
/// matches if rallies continue after match over). Pairs are named by the end they
/// start game 1 at ('A' = near, 'B' = far) and keep those names for the whole
/// recording. Format: first to `nPoints`, two clear, `cap` wins; deciding-game ends
/// change when a side first reaches `interval` (0 = never). Doubles: `pos` = court
/// of A1 A2 B1 B2 (Law 11), `starters[pair][g-1]` = right-court player at game g.
struct BWFScore {
    enum Player: String { case A, B }
    static let players: [Player: (String, String)] = [.A: ("A1", "A2"), .B: ("B1", "B2")]

    struct Event {
        var gameOver = false
        var gameWinner: CourtSide? = nil
        var matchOver = false
        var endsChanged = false
        var newMatch = false
        var score: [String: Int] = [:]
        var scoreByPlayer: [String: Int] = [:]
        var games: [String: Int] = [:]
        var gamesByPlayer: [String: Int] = [:]
    }

    let gamesToWin: Int, nPoints: Int, cap: Int, interval: Int
    let doubles: Bool
    let starters: [Player: [String]]
    var points: [Player: Int] = [.A: 0, .B: 0]
    var games: [Player: Int] = [.A: 0, .B: 0]
    var game = 0
    var aEnd: CourtSide = .near
    var server: Player = .A
    var midSwitched = false
    var matchOver = false
    var pos: [String: String] = [:]

    init(firstServer: CourtSide = .near, gamesToWin: Int = 2, points: Int = 21, cap: Int = 30,
         interval: Int = 11, doubles: Bool = false, starters: [String: [String]]? = nil) {
        self.gamesToWin = gamesToWin; nPoints = points; self.cap = cap; self.interval = interval
        self.doubles = doubles
        var st: [Player: [String]] = [.A: ["A1"], .B: ["B1"]]
        for (k, v) in starters ?? [:] where !v.isEmpty { if let p = Player(rawValue: k) { st[p] = v } }
        self.starters = st
        startMatch(firstServer != .far ? .A : .B)
    }

    private mutating func startMatch(_ s: Player) {
        games = [.A: 0, .B: 0]
        game = 0
        matchOver = false
        startGame(s)
    }

    private mutating func startGame(_ s: Player) {
        game += 1
        points = [.A: 0, .B: 0]
        midSwitched = false
        server = s
        pos = [:]
        if doubles {
            for pair in [Player.A, .B] {
                let (p1, p2) = BWFScore.players[pair]!
                let lst = starters[pair]!
                let first = lst[Swift.min(game - 1, lst.count - 1)]
                pos[first] = "right"
                pos[first == p1 ? p2 : p1] = "left"
            }
        }
    }

    /// Start the next match if the last one is over (before reading the next
    /// rally's server / court, Law 10.1.1). The match winner serves; nobody changes
    /// ends after the final game (Law 8.1), so A / B keep their ends and names.
    @discardableResult
    mutating func begin() -> Bool {
        guard matchOver else { return false }
        startMatch(server)
        return true
    }

    func playerAt(_ side: CourtSide) -> Player { side == aEnd ? .A : .B }
    func sideOf(_ p: Player) -> CourtSide { p == .A ? aEnd : aEnd.opposite }
    func serverSide() -> CourtSide { sideOf(server) }
    func serviceCourt() -> String { points[server]! % 2 == 0 ? "right" : "left" }

    func playerIn(_ pair: Player, _ court: String) -> String? {
        let (p1, p2) = BWFScore.players[pair]!
        return pos[p1] == court ? p1 : (pos[p2] == court ? p2 : nil)
    }
    func serverPlayer() -> String? { doubles ? playerIn(server, serviceCourt()) : nil }
    /// Law 11.1.4: the receiver stands diagonally opposite = the same-named court.
    func receiverPlayer() -> String? { doubles ? playerIn(server == .A ? .B : .A, serviceCourt()) : nil }

    func bySide(_ d: [Player: Int]) -> [String: Int] {
        ["near": d[playerAt(.near)]!, "far": d[playerAt(.far)]!]
    }

    static func byPlayer(_ d: [Player: Int]) -> [String: Int] { ["A": d[.A]!, "B": d[.B]!] }

    func gameWon(_ pw: Int, _ pl: Int) -> Bool { pw >= cap || (pw >= nPoints && pw - pl >= 2) }

    /// Award a rally to the physical side `side`; returns the events. A caller
    /// that did not begin() the next match gets it started here.
    mutating func point(_ side: CourtSide) -> Event {
        var ev = Event()
        ev.newMatch = begin()
        let w = playerAt(side)
        let lo: Player = w == .A ? .B : .A
        points[w]! += 1
        if w == server {
            if doubles {                                 // Law 11.3.1 / 11.1.5: the serving pair swaps courts
                let (p1, p2) = BWFScore.players[w]!
                for p in [p1, p2] { pos[p] = pos[p] == "right" ? "left" : "right" }
            }
        } else {
            server = w                                   // Law 10.3.2 / 11.3.2: the receiver serves next
        }
        let pw = points[w]!, pl = points[lo]!
        ev.score = bySide(points)
        ev.scoreByPlayer = BWFScore.byPlayer(points)
        if gameWon(pw, pl) {
            games[w]! += 1
            ev.gameOver = true
            ev.gameWinner = side
            ev.games = bySide(games)
            ev.gamesByPlayer = BWFScore.byPlayer(games)
            if games[w]! >= gamesToWin {
                ev.matchOver = true
                matchOver = true
            } else {                                     // Law 7.6 / 8.1: winner serves first; change ends
                aEnd = aEnd.opposite
                startGame(w)
                ev.endsChanged = true
            }
        } else {
            ev.games = bySide(games)
            ev.gamesByPlayer = BWFScore.byPlayer(games)
            let decider = game == 2 * gamesToWin - 1
            if decider && interval > 0 && !midSwitched && pw == interval && pl < interval {
                aEnd = aEnd.opposite                     // Law 8.1.3
                midSwitched = true
                ev.endsChanged = true
            }
        }
        return ev
    }
}

/// rally._observed_server: first hitter of a rally (its receiver if the serve was missed).
func observedServer(_ r: RallyV2, minConf: Double = 0.0) -> CourtSide {
    guard var best = r.shots.first else { return .unknown }
    for s in r.shots.dropFirst() where s.hit.frame < best.hit.frame { best = s }
    let h = best.hit
    if h.hitter == .unknown || h.confidence < minConf { return .unknown }
    return r.serveMissed ? h.hitter.opposite : h.hitter
}

/// What is seen at the start of a rally (nil = not seen): rally._implied_winner_obs input.
struct ServeObservation: Equatable {
    var server: CourtSide? = nil
    var serviceCourt: String? = nil
    var serverPlayer: String? = nil
    var receiverPlayer: String? = nil
}

/// rally._implied_winner.
func impliedWinner(_ st: BWFScore, _ nextServer: CourtSide) -> CourtSide {
    guard nextServer != .unknown else { return .unknown }
    return impliedWinner(st, ServeObservation(server: nextServer))
}

/// rally._implied_winner_obs: the only rally winner consistent with what is seen
/// at the start of the next rally, else unknown (both or neither fit).
func impliedWinner(_ st: BWFScore, _ obs: ServeObservation) -> CourtSide {
    if obs.server == nil && obs.serviceCourt == nil && obs.serverPlayer == nil && obs.receiverPlayer == nil {
        return .unknown
    }
    var cands: [CourtSide] = []
    for c in [CourtSide.near, .far] {
        var s = st
        _ = s.point(c)
        s.begin()
        if let v = obs.server, s.serverSide() != v { continue }
        if let v = obs.serviceCourt, s.serviceCourt() != v { continue }
        if let v = obs.serverPlayer, s.serverPlayer() != v { continue }
        if let v = obs.receiverPlayer, s.receiverPlayer() != v { continue }
        cands.append(c)
    }
    return cands.count == 1 ? cands[0] : .unknown
}

private func dictJSON(_ d: [String: Int]) -> JSONValue { .object(d.mapValues { .int($0) }) }
private func optStr(_ s: String?) -> JSONValue { s.map { .str($0) } ?? .null }

/// rally.score_match: running BWF score after every rally. One JSON object per
/// rally with exactly Python's keys (rally, start_frame, end_frame, game, server,
/// observed_server, service_court, winner, score, games, score_by_player,
/// games_by_player, game_over, game_winner, match_over, ends_changed,
/// next_server, uncertain, reasons; doubles adds server_player, receiver_player,
/// positions). Semantics = the DELIVERY_R2 reference (BWFConformanceTests): the
/// rally after match over opens a new match (also with an unknown winner), served
/// by the last winner from the right court; uncertain = a reason other than new_match.
public func scoreMatch(_ rallies: [RallyV2], firstServer: CourtSide = .near, gamesToWin: Int = 2,
                       minConfidence: Double = 0.5, points: Int = 21, cap: Int = 30, interval: Int = 11,
                       doubles: Bool = false, starters: [String: [String]]? = nil) -> [JSONValue] {
    var st = BWFScore(firstServer: firstServer, gamesToWin: gamesToWin, points: points, cap: cap,
                      interval: interval, doubles: doubles, starters: starters)
    // sorted(..., key=start_frame) is stable
    let rs = rallies.enumerated().sorted { a, b in
        a.element.startFrame < b.element.startFrame || (a.element.startFrame == b.element.startFrame && a.offset < b.offset)
    }.map(\.element)
    var out: [JSONValue] = []
    for (i, r) in rs.enumerated() {
        var reasons: [String] = st.begin() ? ["new_match"] : []
        let expServer = st.serverSide(), court = st.serviceCourt()
        let sp = st.serverPlayer(), rp = st.receiverPlayer(), pos = st.pos
        let obs = observedServer(r)
        if obs != .unknown && obs != expServer { reasons.append("server_mismatch") }
        var w = r.winner
        if w == .unknown {
            let nxt = i + 1 < rs.count ? observedServer(rs[i + 1]) : .unknown
            w = impliedWinner(st, nxt)
            reasons.append(w != .unknown ? "winner_inferred_from_next_serve" : "winner_unknown")
        } else if r.confidence < minConfidence {
            reasons.append("low_confidence")
        }
        let game = st.game
        let ev: BWFScore.Event
        if w != .unknown {
            ev = st.point(w)
        } else {
            var e = BWFScore.Event()
            e.score = st.bySide(st.points)
            e.games = st.bySide(st.games)
            e.scoreByPlayer = BWFScore.byPlayer(st.points)
            e.gamesByPlayer = BWFScore.byPlayer(st.games)
            ev = e
        }
        var row: [String: JSONValue] = [
            "rally": .int(i), "start_frame": .int(r.startFrame), "end_frame": .int(r.endFrame),
            "game": .int(game), "server": .str(expServer.rawValue), "observed_server": .str(obs.rawValue),
            "service_court": .str(court), "winner": .str(w.rawValue),
            "score": dictJSON(ev.score), "games": dictJSON(ev.games),
            "score_by_player": dictJSON(ev.scoreByPlayer), "games_by_player": dictJSON(ev.gamesByPlayer),
            "game_over": .bool(ev.gameOver), "game_winner": ev.gameWinner.map { .str($0.rawValue) } ?? .null,
            "match_over": .bool(ev.matchOver), "ends_changed": .bool(ev.endsChanged),
            "next_server": .str(st.serverSide().rawValue),
            "uncertain": .bool(reasons.contains { $0 != "new_match" }), "reasons": .array(reasons.map { .str($0) }),
        ]
        if doubles {
            row["server_player"] = optStr(sp)
            row["receiver_player"] = optStr(rp)
            row["positions"] = .object(pos.mapValues { .str($0) })
        }
        out.append(.object(row))
    }
    return out
}
