import Foundation
import ShuttlCore

// sim.py — shot generator: per-type specs, landing targets, shot planning, next-shot choice.
// depth = distance beyond the net on the receiving side (m); mode/value feed solveShot
// (speed values in km/h, (lo, mode, hi) = triangular); `long` = the natural error is long.

public struct ShotSpec: Sendable {
    public var depth: (Double, Double)
    public var mode: String
    public var value: [Double]
    public var long: Bool
    public var maxRise: Double? = nil
    public var maxKmh: Double? = nil
    public var maxApex: Double? = nil
    public var loose: [Double]? = nil
    public var looseApex: Double? = nil
}

public enum SimShots {
    public static let specs: [String: ShotSpec] = [
        "clear": ShotSpec(depth: (5.0, 6.6), mode: "apex", value: [4.2, 7.5], long: true),
        "lift": ShotSpec(depth: (4.6, 6.6), mode: "apex", value: [3.6, 6.5], long: true),
        "drop": ShotSpec(depth: (0.5, 2.6), mode: "clearance", value: [0.05, 0.45], long: false, maxRise: 0.3, maxKmh: 160.0),
        "net": ShotSpec(depth: (0.25, 1.8), mode: "clearance", value: [0.02, 0.2], long: false, maxKmh: 45.0, maxApex: 2.2,
                        loose: [0.3, 0.6], looseApex: 2.6),
        "push": ShotSpec(depth: (2.4, 4.8), mode: "clearance", value: [0.05, 0.35], long: true, maxKmh: 90.0, maxApex: 2.7,
                         loose: [0.35, 0.6], looseApex: 2.9),
        "block": ShotSpec(depth: (0.4, 2.4), mode: "clearance", value: [0.03, 0.3], long: false, maxKmh: 75.0, maxApex: 2.5,
                          loose: [0.3, 0.55], looseApex: 2.7),
        "smash": ShotSpec(depth: (1.8, 6.6), mode: "speed", value: [200.0, 290.0, 420.0], long: false),
        "kill": ShotSpec(depth: (1.0, 5.5), mode: "speed", value: [120.0, 170.0, 260.0], long: false),
        "drive": ShotSpec(depth: (3.2, 6.6), mode: "speed", value: [100.0, 140.0, 200.0], long: true, maxApex: 2.9),
        "serve_short": ShotSpec(depth: (2.03, 2.7), mode: "clearance", value: [0.03, 0.2], long: false),
        "serve_long": ShotSpec(depth: (5.8, 6.6), mode: "apex", value: [4.5, 7.2], long: true),
    ]
    /// Probability that a returned front-court shot is played too high (`loose`).
    public static let looseP = 0.3
    /// Preferred contact height (mean, sd) of the receiver given the incoming shot.
    public static let prefZ: [String: (Double, Double)] = [
        "clear": (2.7, 0.35), "lift": (2.6, 0.4), "serve_long": (2.6, 0.4), "smash": (0.9, 0.4), "kill": (0.9, 0.4),
        "drive": (1.3, 0.35), "push": (1.2, 0.4), "drop": (0.8, 0.4), "block": (0.7, 0.35), "net": (0.9, 0.4),
        "serve_short": (1.1, 0.35)]
    /// Final-shot outcome weights (in, out, net).
    public static let endW: [String: [Double]] = [
        "smash": [0.7, 0.12, 0.18], "kill": [0.7, 0.12, 0.18], "drop": [0.5, 0.15, 0.35], "net": [0.5, 0.15, 0.35],
        "block": [0.5, 0.15, 0.35], "clear": [0.4, 0.6, 0.0], "lift": [0.4, 0.6, 0.0], "drive": [0.45, 0.3, 0.25],
        "push": [0.45, 0.3, 0.25]]

    static func draw(_ rng: any SimRandom, _ v: [Double]) -> Double {
        v.count == 2 ? rng.uniform(v[0], v[1]) : rng.triangular(v[0], v[1], v[2])
    }

    public static func halfWidth(_ doubles: Bool) -> Double { doubles ? CourtModel.doublesHalfWidth : CourtModel.singlesHalfWidth }

    /// `_sample_target`: landing target on the receiving half; 'in'/'out' finals sit near a
    /// line (half of them within 20 cm) so close calls are well represented.
    static func sampleTarget(_ rng: any SimRandom, _ spec: ShotSpec, recvSign: Double, hw: Double, outcome: String) -> SIMD2<Double> {
        let (lo, hi) = spec.depth
        let ex = CourtModel.halfLength + CourtModel.lineWidth / 2, ey = hw + CourtModel.lineWidth / 2
        func lateral(_ limit: Double) -> Double {
            let f = rng.random() < 0.7 ? rng.uniform(0.55, 1.0) : rng.uniform(0.0, 0.55)
            return f * limit * rng.pick([-1.0, 1.0])
        }
        if outcome == "in" || outcome == "out" {
            let close = rng.random() < 0.5
            let m = close ? rng.uniform(0.005, 0.2) : rng.uniform(0.2, 1.0)
            var depth: Double, y: Double
            if spec.long {
                if outcome == "in" { depth = ex - m; y = lateral(pyMax(ey - m - 0.08, 0.3)) }
                else { depth = ex + m; y = lateral(hw - 0.1) }
            } else {
                depth = rng.uniform(lo, outcome == "in" ? pyMin(hi, ex - m - 0.08) : hi)
                y = (outcome == "in" ? ey - m : ey + m) * rng.pick([-1.0, 1.0])
            }
            return SIMD2(recvSign * depth, y)
        }
        let depth = pyMin(rng.uniform(lo, hi), 6.62)
        return SIMD2(recvSign * depth, lateral(hw - 0.1))
    }

    /// `_within_caps`: apex / rise / launch-speed realism caps.
    static func withinCaps(_ spec: ShotSpec, v0: SIMD3<Double>, stats: ShotStats, z0: Double) -> Bool {
        if let apex = stats.apex, apex.isFinite {
            if apex > (spec.maxApex ?? .infinity) || apex - z0 > (spec.maxRise ?? .infinity) { return false }
        }
        return NP.norm(v0) * 3.6 <= (spec.maxKmh ?? .infinity)
    }

    /// `_plan_shot`: sample target / parameters for a shot type and solve v0 (nil if infeasible).
    static func planShot(_ rng: any SimRandom, kind: String, p0: SIMD3<Double>, recvSign: Double, L: Double, hw: Double,
                         outcome: String, target: SIMD2<Double>? = nil, tries: Int = 6)
        -> (v0: SIMD3<Double>, stats: ShotStats, target: SIMD2<Double>)? {
        var spec = specs[kind]!
        if spec.loose != nil && outcome == "returned" && rng.random() < looseP {
            spec.value = spec.loose!; spec.maxApex = spec.looseApex
        }
        for attempt in 0..<tries {
            var val = draw(rng, spec.value)
            if spec.mode == "speed" { val = val / 3.6 * pow(0.88, Double(max(0, attempt - 2))) }
            var tgt = target ?? sampleTarget(rng, spec, recvSign: recvSign, hw: hw, outcome: outcome)
            let res: (v0: SIMD3<Double>, stats: ShotStats)?
            if outcome == "net" {
                res = SimPhysics.solveNetFault(rng, mode: spec.mode, p0: p0, target: tgt, L: L, value: val)
            } else {
                if spec.mode == "speed" {
                    let d = tgt - p0.xy
                    guard let win = SimPhysics.speedDepthWindow(p0: p0, direction: d, speed: val, L: L) else { continue }
                    let r = hypot(d.x, d.y)
                    if !(win.0 + 0.02 <= r && r <= win.1 - 0.02) {
                        if outcome != "returned" || target != nil { continue }
                        let dy = tgt.y - p0.y, ax = abs(p0.x)
                        let dlo = pyMax(win.0 * win.0 - dy * dy, 0.0).squareRoot() - ax + 0.05
                        let dhi = pyMax(win.1 * win.1 - dy * dy, 0.0).squareRoot() - ax - 0.05
                        let lo = pyMax(dlo, spec.depth.0), hi = pyMin(dhi, spec.depth.1, 6.62)
                        if lo >= hi { continue }
                        tgt = SIMD2(recvSign * rng.uniform(lo, hi), tgt.y)
                    }
                }
                res = SimPhysics.solveShot(p0: p0, target: tgt, L: L, mode: spec.mode, value: val, minClearance: 0.03)
            }
            if let res, withinCaps(spec, v0: res.v0, stats: res.stats, z0: p0.z) { return (res.v0, res.stats, tgt) }
        }
        return nil
    }

    /// `_kind_options`: shot types a player can play from contact height z and depth behind
    /// the net, in Python dict order, zero weights dropped.
    static func kindOptions(z: Double, depth: Double) -> [(String, Double)] {
        let opts: [(String, Double)]
        if z >= 2.1 {
            if depth >= 4.3 { opts = [("smash", 0.35), ("clear", 0.35), ("drop", 0.30)] }
            else if depth >= 2.5 { opts = [("smash", 0.55), ("drop", 0.30), ("clear", 0.15)] }
            else { opts = [("kill", 0.75), ("drop", 0.1), ("net", 0.15)] }
        } else if z >= 1.2 {
            if z >= 1.6 && depth < 3.2 {
                opts = depth < 2.3 ? [("kill", 0.5), ("push", 0.25), ("net", 0.25)]
                                   : [("kill", 0.4), ("drive", 0.35), ("push", 0.25)]
            } else if depth < 2.3 { opts = [("push", 0.4), ("net", 0.4), ("lift", 0.2)] }
            else if depth < 3.5 { opts = [("drive", 0.45), ("lift", z < 1.9 ? 0.3 : 0.0), ("push", 0.25)] }
            else { opts = [("drive", 0.5), ("lift", z < 1.9 ? 0.5 : 0.0), ("clear", z < 1.9 ? 0.0 : 0.5)] }
        } else if depth < 2.5 {
            opts = [("net", 0.5), ("lift", 0.4), ("push", z >= 0.7 ? 0.1 : 0.0)]
        } else {
            opts = [("lift", 0.45), ("block", 0.35), ("drive", z >= 0.8 ? 0.2 : 0.0)]
        }
        return opts.filter { $0.1 > 0 }
    }

    /// `_fallback_kinds`: every type allowed here (by weight, stable), then height-consistent defaults.
    static func fallbackKinds(z: Double, depth: Double) -> [String] {
        let opts = kindOptions(z: z, depth: depth)
        let order = opts.indices.sorted { a, b in -opts[a].1 < -opts[b].1 || (opts[a].1 == opts[b].1 && a < b) }.map { opts[$0].0 }
        let last = z >= 1.9 ? ["clear", "drop"] : z >= 1.2 ? ["drive", "lift"] : ["lift", "net", "block"]
        var seen = Set<String>(), out: [String] = []
        for k in order + last where !seen.contains(k) { seen.insert(k); out.append(k) }
        return out
    }

    /// `_choose_kind`: next shot type from contact height and distance behind the net.
    static func chooseKind(_ rng: any SimRandom, z: Double, depth: Double) -> String {
        let opts = kindOptions(z: z, depth: depth)
        let w = opts.map(\.1)
        let s = NP.sum(w)
        return opts[rng.choice(opts.count, p: w.map { $0 / s })].0
    }
}
