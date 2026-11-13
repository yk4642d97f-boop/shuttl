import Foundation
import ShuttlCore

// The random-draw seam of the sim.py port. numpy's Generator (PCG64 + its ziggurat /
// distribution code) can't be reproduced bit for bit, so every draw sim.py makes goes
// through `SimRandom`, one method per numpy call it uses:
//   • `NativeRandom`  — xoshiro256** with numpy's distribution formulas (statistically
//     equivalent scenarios, deterministic per seed key);
//   • `ReplayRandom`  — replays a draw log recorded from Python
//     (backend/scripts/swift_fixtures/eval.py `--only sim`): given the same draws the
//     Swift simulator must produce the Python scenario (positions to 1e-9, same labels).
// A stream is shared mutable state like numpy's Generator, hence a class protocol.

public protocol SimRandom: AnyObject {
    /// rng.random()
    func random() -> Double
    /// rng.random(n)
    func random(_ n: Int) -> [Double]
    /// rng.uniform(lo, hi)
    func uniform(_ lo: Double, _ hi: Double) -> Double
    /// rng.uniform(lo, hi, n)
    func uniform(_ lo: Double, _ hi: Double, _ n: Int) -> [Double]
    /// rng.normal(mu, sd)
    func normal(_ mu: Double, _ sd: Double) -> Double
    /// rng.normal(mu, sd, n) (a shape (a, b) is n = a·b in C order)
    func normal(_ mu: Double, _ sd: Double, _ n: Int) -> [Double]
    /// rng.triangular(left, mode, right)
    func triangular(_ left: Double, _ mode: Double, _ right: Double) -> Double
    /// rng.geometric(p)
    func geometric(_ p: Double) -> Int
    /// rng.integers(lo, hi) — hi exclusive
    func integers(_ lo: Int, _ hi: Int) -> Int
    /// rng.choice(n, p=p) / rng.choice(list, p=p) → the chosen index
    func choice(_ n: Int, p: [Double]?) -> Int
}

public extension SimRandom {
    func integers(_ hi: Int) -> Int { integers(0, hi) }
    func choice(_ n: Int) -> Int { choice(n, p: nil) }
    /// rng.choice(list) / rng.choice(list, p=w/w.sum())
    func pick<T>(_ xs: [T], p: [Double]? = nil) -> T { xs[choice(xs.count, p: p)] }
}

/// Creates the independent streams sim.py makes with `np.random.default_rng(key)`.
/// Keys are JSON like the Python recorder writes them: `[3, 17]` for [seed, 17], `5` for an int.
public protocol SimRandomSource: AnyObject {
    func stream(_ key: String) -> any SimRandom
}

public extension SimRandomSource {
    func stream(_ parts: [Int]) -> any SimRandom { stream("[" + parts.map(String.init).joined(separator: ", ") + "]") }
    func stream(_ seed: Int) -> any SimRandom { stream(String(seed)) }
}

// MARK: - Native

/// xoshiro256** seeded by SplitMix64 from the stream key; numpy's formulas on top
/// (`random` = 53-bit double, polar-free Box–Muller normal, numpy's triangular and
/// geometric (search / inversion) rules, Lemire-free unbiased integers by rejection).
public final class NativeRandom: SimRandom {
    private var s: (UInt64, UInt64, UInt64, UInt64)
    private var spare: Double?

    public init(seed: UInt64) {
        var z = seed
        func split() -> UInt64 {
            z &+= 0x9E37_79B9_7F4A_7C15
            var x = z
            x = (x ^ (x >> 30)) &* 0xBF58_476D_1CE4_E5B9
            x = (x ^ (x >> 27)) &* 0x94D0_49BB_1331_11EB
            return x ^ (x >> 31)
        }
        s = (split(), split(), split(), split())
    }

    /// FNV-1a of the key → seed.
    public convenience init(key: String) {
        var h: UInt64 = 0xCBF2_9CE4_8422_2325
        for b in key.utf8 { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01B3 }
        self.init(seed: h)
    }

    @inline(__always) static func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 { (x << k) | (x >> (64 - k)) }

    public func next64() -> UInt64 {
        let r = Self.rotl(s.1 &* 5, 7) &* 9
        let t = s.1 << 17
        s.2 ^= s.0; s.3 ^= s.1; s.1 ^= s.2; s.0 ^= s.3
        s.2 ^= t
        s.3 = Self.rotl(s.3, 45)
        return r
    }

    public func random() -> Double { Double(next64() >> 11) * (1.0 / 9_007_199_254_740_992.0) }
    public func random(_ n: Int) -> [Double] { (0..<max(n, 0)).map { _ in random() } }
    public func uniform(_ lo: Double, _ hi: Double) -> Double { lo + (hi - lo) * random() }
    public func uniform(_ lo: Double, _ hi: Double, _ n: Int) -> [Double] { (0..<max(n, 0)).map { _ in uniform(lo, hi) } }

    func standardNormal() -> Double {
        if let v = spare { spare = nil; return v }
        var u = 0.0
        repeat { u = random() } while u <= 0
        let v = random()
        let r = (-2 * log(u)).squareRoot()
        spare = r * sin(2 * .pi * v)
        return r * cos(2 * .pi * v)
    }

    public func normal(_ mu: Double, _ sd: Double) -> Double { mu + sd * standardNormal() }
    public func normal(_ mu: Double, _ sd: Double, _ n: Int) -> [Double] { (0..<max(n, 0)).map { _ in normal(mu, sd) } }

    public func triangular(_ left: Double, _ mode: Double, _ right: Double) -> Double {
        let base = right - left, leftbase = mode - left, ratio = leftbase / base
        let leftprod = leftbase * base, rightprod = (right - mode) * base
        let u = random()
        return u <= ratio ? left + (u * leftprod).squareRoot() : right - ((1 - u) * rightprod).squareRoot()
    }

    public func geometric(_ p: Double) -> Int {
        if p >= 0.333333333333333333333333 {                  // numpy: search
            var x = 1
            let u = random()
            var sum = p, prod = p
            let q = 1 - p
            while u > sum { prod *= q; sum += prod; x += 1 }
            return x
        }
        return Int((log(1 - random()) / log1p(-p)).rounded(.up))    // inversion
    }

    public func integers(_ lo: Int, _ hi: Int) -> Int {
        precondition(hi > lo)
        let range = UInt64(hi - lo)
        let limit = UInt64.max - UInt64.max % range
        var r: UInt64
        repeat { r = next64() } while r >= limit
        return lo + Int(r % range)
    }

    public func choice(_ n: Int, p: [Double]?) -> Int {
        guard let p else { return integers(0, n) }
        var cdf = [Double](), acc = 0.0
        for v in p { acc += v; cdf.append(acc) }
        let u = random() * (cdf.last ?? 1)
        return min(Num.searchsorted(cdf, u, right: true), n - 1)
    }
}

public final class NativeRandomSource: SimRandomSource {
    public init() {}
    public func stream(_ key: String) -> any SimRandom { NativeRandom(key: key) }
}

// MARK: - Replay of recorded numpy draws

/// One recorded Generator call: method, numeric args, count (array draws), result, p (choice).
public struct RecordedDraw: Sendable, Decodable {
    public var m: String
    public var a: [Double]
    public var n: Int?
    public var r: [Double]
    public var p: [Double]?
    /// Large choice p: [Σ i·p_i, Σ p_i²] instead of the vector.
    public var ps: [Double]?

    enum CodingKeys: String, CodingKey { case m, a, n, r, p, ps }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        m = try c.decode(String.self, forKey: .m)
        a = try c.decodeNaNableArray(forKey: .a) ?? []
        n = try c.decodeIfPresent(Int.self, forKey: .n)
        if let v = try? c.decode(Double.self, forKey: .r) { r = [v] }
        else { r = try c.decodeNaNableArray(forKey: .r) ?? [] }
        p = try c.decodeNaNableArray(forKey: .p)
        ps = try c.decodeNaNableArray(forKey: .ps)
    }
}

/// Replays one recorded stream. The Swift call must match the recorded one (method,
/// count, args within 1e-9 relative, choice p within 1e-9) — the first mismatch is
/// recorded in `problems` (the draw is still served so the run continues).
public final class ReplayRandom: SimRandom {
    public let key: String
    public let draws: [RecordedDraw]
    public private(set) var pos = 0
    public private(set) var problems: [String] = []

    public init(key: String, draws: [RecordedDraw]) { self.key = key; self.draws = draws }

    public var remaining: Int { draws.count - pos }

    static func close(_ a: Double, _ b: Double) -> Bool {
        a == b || (a.isNaN && b.isNaN) || abs(a - b) <= 1e-9 * max(1, abs(a), abs(b))
    }

    func take(_ m: String, _ args: [Double], _ n: Int?, p: [Double]? = nil) -> RecordedDraw? {
        guard pos < draws.count else {
            problems.append("\(key): draw #\(pos) \(m)\(args) past the end of the log")
            return nil
        }
        let d = draws[pos]
        pos += 1
        if problems.isEmpty {
            if d.m != m || d.n != n || d.a.count != args.count || !zip(d.a, args).allSatisfy({ Self.close($0, $1) }) {
                problems.append("\(key): draw #\(pos - 1) Swift \(m)\(args) n=\(n.map(String.init) ?? "-") vs recorded \(d.m)\(d.a) n=\(d.n.map(String.init) ?? "-")")
            } else if let p, let dp = d.p, dp.count != p.count || !zip(dp, p).allSatisfy({ Self.close($0, $1) }) {
                problems.append("\(key): draw #\(pos - 1) \(m) probabilities differ: \(p.prefix(4)) vs \(dp.prefix(4)) (n=\(p.count))")
            } else if let p, let ps = d.ps, ps.count == 2 {
                let s1 = NP.sum(p.enumerated().map { Double($0.offset) * $0.element }), s2 = NP.sum(p.map { $0 * $0 })
                if !(Self.close(s1, ps[0]) && Self.close(s2, ps[1])) {
                    problems.append("\(key): draw #\(pos - 1) \(m) probability checksum differs: [\(s1), \(s2)] vs \(ps) (n=\(p.count))")
                }
            }
        }
        return d
    }

    func scalar(_ m: String, _ args: [Double], fallback: Double) -> Double {
        take(m, args, nil)?.r.first ?? fallback
    }

    func array(_ m: String, _ args: [Double], _ n: Int, fallback: Double) -> [Double] {
        let r = take(m, args, n)?.r ?? []
        return r.count == n ? r : [Double](repeating: fallback, count: n)
    }

    public func random() -> Double { scalar("random", [], fallback: 0.5) }
    public func random(_ n: Int) -> [Double] { array("random", [], n, fallback: 0.5) }
    public func uniform(_ lo: Double, _ hi: Double) -> Double { scalar("uniform", [lo, hi], fallback: (lo + hi) / 2) }
    public func uniform(_ lo: Double, _ hi: Double, _ n: Int) -> [Double] { array("uniform", [lo, hi], n, fallback: (lo + hi) / 2) }
    public func normal(_ mu: Double, _ sd: Double) -> Double { scalar("normal", [mu, sd], fallback: mu) }
    public func normal(_ mu: Double, _ sd: Double, _ n: Int) -> [Double] { array("normal", [mu, sd], n, fallback: mu) }
    public func triangular(_ l: Double, _ m: Double, _ r: Double) -> Double { scalar("triangular", [l, m, r], fallback: m) }
    public func geometric(_ p: Double) -> Int { Int(scalar("geometric", [p], fallback: 1)) }
    public func integers(_ lo: Int, _ hi: Int) -> Int { Int(scalar("integers", [Double(lo), Double(hi)], fallback: Double(lo))) }
    public func choice(_ n: Int, p: [Double]?) -> Int {
        Int(take("choice", [Double(n)], nil, p: p)?.r.first ?? 0)
    }
}

/// Serves recorded streams by key, in the order Python created them (the same key may
/// be created more than once: each request takes the next log for that key).
public final class ReplayRandomSource: SimRandomSource {
    public struct Stream: Sendable, Decodable { public var key: String; public var draws: [RecordedDraw] }
    private var queue: [String: [Stream]] = [:]
    public private(set) var served: [ReplayRandom] = []
    private var missing: [String] = []

    public init(_ streams: [Stream]) {
        for s in streams { queue[s.key, default: []].append(s) }
    }

    public func stream(_ key: String) -> any SimRandom {
        guard var q = queue[key], !q.isEmpty else {
            missing.append(key)
            let r = ReplayRandom(key: key, draws: [])
            served.append(r)
            return r
        }
        let s = q.removeFirst()
        queue[key] = q
        let r = ReplayRandom(key: key, draws: s.draws)
        served.append(r)
        return r
    }

    /// Everything that went wrong: mismatched / missing / unconsumed draws, unrequested streams.
    public var problems: [String] {
        var out = missing.map { "stream \($0) was never recorded" }
        for r in served {
            out += r.problems
            if r.remaining > 0 { out.append("\(r.key): \(r.remaining) recorded draws not consumed (Swift made \(r.pos))") }
        }
        for (k, q) in queue where !q.isEmpty { out.append("recorded stream \(k) never requested (\(q.count)×)") }
        return out
    }
}
