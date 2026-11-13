import Foundation
import ShuttlCore

// sim.py — planar shot solver and flight kernels. Drag is parallel to v and gravity is
// vertical, so a shot never leaves the vertical plane through p0 along its horizontal
// launch direction: solving in (r, z) is exact. The kernels are the numba kernels of
// sim.py (`_flight_kernel`, `_simulate_kernel`) with the same scalar arithmetic order.

public enum SimConst {
    public static let netX = 0.0
    public static let playerHeight = (1.65, 1.90)          // m, sampled per player
    public static let bodyHalfWidth = 0.33                 // m, bbox half-extent around the feet
    public static let solveDt = 1.0 / 480                  // == physics.simulate max_dt
    public static let maxAvgSpeed = 3.0                    // m/s average over a min-jerk move
    public static let reactionS = 0.12                     // s after the opponent's contact
    public static let shotTypes = ["serve", "clear", "smash", "drop", "net", "drive", "lift", "push", "block", "kill"]
}

/// Python `min(a, b, …)` / `max(a, b, …)` (first argument wins ties and NaN comparisons).
@inline(__always) func pyMin(_ a: Double, _ rest: Double...) -> Double { rest.reduce(a) { $1 < $0 ? $1 : $0 } }
@inline(__always) func pyMax(_ a: Double, _ rest: Double...) -> Double { rest.reduce(a) { $1 > $0 ? $1 : $0 } }

public struct PlanarFlightStats: Sendable {
    public var range: [Double]      // r at floor contact (NaN if it never lands within t_max)
    public var time: [Double]       // flight time
    public var apex: [Double]
    public var zNet: [Double]       // z where r == d_net (NaN if not reached)
}

public struct ShotStats: Sendable {
    public var range = Double.nan
    public var time = Double.nan
    public var apex: Double?        // net-fault solutions carry no apex
    public var zNet = Double.nan
    public var thetaDeg = Double.nan
    public var speed = Double.nan
}

public enum SimPhysics {

    /// `_flight_kernel`: RK4 of many planar flights (range with linear floor refinement,
    /// flight time, apex, z at r = d_net).
    public static func flightStats(_ speed: [Double], _ theta: [Double], z0: Double, dNet: Double, L: Double,
                                   dt: Double, tMax: Double = 5.0) -> PlanarFlightStats {
        let n = speed.count
        var out = PlanarFlightStats(range: [Double](repeating: .nan, count: n), time: [Double](repeating: .nan, count: n),
                                    apex: [Double](repeating: .nan, count: n), zNet: [Double](repeating: .nan, count: n))
        let g: Double = 9.81, inv: Double = 1.0 / L
        let nsteps = Int((tMax / dt).rounded(.up))
        let h6: Double = dt / 6, hh: Double = 0.5 * dt
        for k in 0..<n {
            var r: Double = 0, z: Double = z0
            var vr: Double = speed[k] * cos(theta[k]), vz: Double = speed[k] * sin(theta[k]), t: Double = 0
            var apex = z0, znet = Double.nan, lr = Double.nan, lt = Double.nan
            for _ in 0..<nsteps {
                var s: Double = (vr * vr + vz * vz).squareRoot() * inv
                let a1r: Double = -s * vr, a1z: Double = -g - s * vz
                let u2r: Double = vr + hh * a1r, u2z: Double = vz + hh * a1z
                s = (u2r * u2r + u2z * u2z).squareRoot() * inv
                let a2r: Double = -s * u2r, a2z: Double = -g - s * u2z
                let u3r: Double = vr + hh * a2r, u3z: Double = vz + hh * a2z
                s = (u3r * u3r + u3z * u3z).squareRoot() * inv
                let a3r: Double = -s * u3r, a3z: Double = -g - s * u3z
                let u4r: Double = vr + dt * a3r, u4z: Double = vz + dt * a3z
                s = (u4r * u4r + u4z * u4z).squareRoot() * inv
                let a4r: Double = -s * u4r, a4z: Double = -g - s * u4z
                let sr: Double = vr + 2 * u2r + 2 * u3r + u4r, sz: Double = vz + 2 * u2z + 2 * u3z + u4z
                let nr: Double = r + h6 * sr, nz: Double = z + h6 * sz
                let br: Double = a1r + 2 * a2r + 2 * a3r + a4r, bz: Double = a1z + 2 * a2z + 2 * a3z + a4z
                let nvr: Double = vr + h6 * br, nvz: Double = vz + h6 * bz
                if r < dNet && nr >= dNet {
                    let f: Double = (dNet - r) / pyMax(nr - r, 1e-12)
                    znet = z + f * (nz - z)
                }
                if nz <= 0.0 && z > 0.0 {
                    let f: Double = z / pyMax(z - nz, 1e-12)
                    lr = r + f * (nr - r); lt = t + f * dt
                    break
                }
                if nz > apex { apex = nz }
                r = nr; z = nz; vr = nvr; vz = nvz; t = t + dt
            }
            out.range[k] = lr; out.time[k] = lt; out.apex[k] = apex; out.zNet[k] = znet
        }
        return out
    }

    /// `simulate_positions` = physics.simulate(p0, v0, times, L), scalar-unrolled like
    /// `_simulate_kernel` (sub-steps of ≤ 1/480 s between consecutive times).
    public static func simulatePositions(p0: SIMD3<Double>, v0: SIMD3<Double>, times: [Double], L: Double,
                                         maxDt: Double = 1.0 / 480) -> [SIMD3<Double>] {
        let g: Double = 9.81, inv: Double = 1.0 / L
        var px = p0.x, py = p0.y, pz = p0.z, vx = v0.x, vy = v0.y, vz = v0.z
        var t: Double = 0
        var out: [SIMD3<Double>] = []
        out.reserveCapacity(times.count)
        for target in times {
            let span: Double = target - t
            let n = max(1, Int((abs(span) / maxDt).rounded(.up)))
            let h: Double = span / Double(n)
            let hh: Double = 0.5 * h, h6: Double = h / 6
            for _ in 0..<n {
                var s: Double = (vx * vx + vy * vy + vz * vz).squareRoot() * inv
                let a1x: Double = -s * vx, a1y: Double = -s * vy, a1z: Double = -g - s * vz
                let u2x: Double = vx + hh * a1x, u2y: Double = vy + hh * a1y, u2z: Double = vz + hh * a1z
                s = (u2x * u2x + u2y * u2y + u2z * u2z).squareRoot() * inv
                let a2x: Double = -s * u2x, a2y: Double = -s * u2y, a2z: Double = -g - s * u2z
                let u3x: Double = vx + hh * a2x, u3y: Double = vy + hh * a2y, u3z: Double = vz + hh * a2z
                s = (u3x * u3x + u3y * u3y + u3z * u3z).squareRoot() * inv
                let a3x: Double = -s * u3x, a3y: Double = -s * u3y, a3z: Double = -g - s * u3z
                let u4x: Double = vx + h * a3x, u4y: Double = vy + h * a3y, u4z: Double = vz + h * a3z
                s = (u4x * u4x + u4y * u4y + u4z * u4z).squareRoot() * inv
                let a4x: Double = -s * u4x, a4y: Double = -s * u4y, a4z: Double = -g - s * u4z
                let qx: Double = vx + 2 * u2x + 2 * u3x + u4x, qy: Double = vy + 2 * u2y + 2 * u3y + u4y
                let qz: Double = vz + 2 * u2z + 2 * u3z + u4z
                px += h6 * qx; py += h6 * qy; pz += h6 * qz
                let bx: Double = a1x + 2 * a2x + 2 * a3x + a4x, by: Double = a1y + 2 * a2y + 2 * a3y + a4y
                let bz: Double = a1z + 2 * a2z + 2 * a3z + a4z
                vx += h6 * bx; vy += h6 * by; vz += h6 * bz
            }
            t = target
            out.append(SIMD3(px, py, pz))
        }
        return out
    }

    /// `_crossing`: first x where the sampled curve y(x) crosses `target` (NaN if none).
    public static func crossing(_ x: [Double], _ y: [Double], _ target: Double) -> Double {
        guard x.count >= 2 else { return .nan }
        for i in 0..<(x.count - 1) where y[i].isFinite && y[i + 1].isFinite {
            if (y[i] - target) * (y[i + 1] - target) <= 0 && y[i] != y[i + 1] {
                return x[i] + (target - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])
            }
        }
        return .nan
    }

    /// `_polish_speed`: refine launch speed at fixed elevation so the fine-dt range hits D.
    static func polishSpeed(theta: Double, sGuess: Double, D: Double, z0: Double, dNet: Double, L: Double) -> Double {
        var s = sGuess
        for rel in [0.04, 0.003] {
            let ss = NP.linspace(-rel, rel, 13).map { s * (1 + $0) }
            let rr = flightStats(ss, [Double](repeating: theta, count: ss.count), z0: z0, dNet: dNet, L: L, dt: SimConst.solveDt).range
            let c = crossing(ss, rr, D)
            if !c.isFinite {
                var best = -1, bd = Double.infinity
                for k in ss.indices where rr[k].isFinite {
                    let d = abs(rr[k] - D)
                    if best < 0 || d < bd { best = k; bd = d }
                }
                if best < 0 { return .nan }
                s = ss[best]
                continue
            }
            s = c
        }
        return s
    }

    /// `speed_depth_window`: horizontal-range interval reachable at a fixed launch speed on
    /// the flat branch while clearing the net by `minClearance`.
    public static func speedDepthWindow(p0: SIMD3<Double>, direction: SIMD2<Double>, speed: Double, L: Double,
                                        minClearance: Double = 0.03) -> (Double, Double)? {
        let nrm = NP.norm(direction.x, direction.y)
        let u = SIMD2(direction.x / nrm, direction.y / nrm)
        var dNet: Double = abs(u.x) > 1e-6 ? (SimConst.netX - p0.x) / u.x : 1e9
        dNet = dNet > 0 ? dNet : 1e9
        let th = NP.linspace(-60, 45, 106).map(NP.radians)
        let st = flightStats([Double](repeating: speed, count: th.count), th, z0: p0.z, dNet: dNet, L: L, dt: 1.0 / 120)
        let rng = st.range
        guard rng.contains(where: \.isFinite) else { return nil }
        var peak = 0, pv = -Double.infinity, found = false
        for (k, v) in rng.enumerated() where !v.isNaN && (!found || v > pv) { peak = k; pv = v; found = true }
        var lo = Double.infinity, any = false
        for k in 0...peak {
            let r = rng[k], c = st.zNet[k] - CourtModel.netHeightCentre
            var ok = r.isFinite
            if dNet < 1e8 { ok = ok && c.isFinite && c >= minClearance && r > dNet + 0.3 }
            if ok { any = true; lo = Swift.min(lo, r) }
        }
        return any ? (lo, rng[peak]) : nil
    }

    /// `solve_shot`: initial velocity from p0 landing at target under drag length L.
    /// mode 'apex' (value = apex height), 'clearance' (value = z at the net − net height,
    /// may be negative for a net fault) or 'speed' (launch speed m/s, flatter solution).
    public static func solveShot(p0: SIMD3<Double>, target: SIMD2<Double>, L: Double, mode: String, value: Double,
                                 minClearance: Double? = 0.03) -> (v0: SIMD3<Double>, stats: ShotStats)? {
        let dxy = target - p0.xy
        let D = hypot(dxy.x, dxy.y)
        if D < 0.3 { return nil }
        let u = dxy / D
        var dNet: Double = abs(u.x) > 1e-6 ? (SimConst.netX - p0.x) / u.x : .infinity
        dNet = (0 < dNet && dNet < D) ? dNet : 1e9
        let z0 = p0.z, coarse = 1.0 / 120
        var theta: Double, speed: Double
        if mode == "speed" {
            let th = NP.linspace(-60, 45, 211).map(NP.radians)
            let st = flightStats([Double](repeating: value, count: th.count), th, z0: z0, dNet: dNet, L: L, dt: coarse)
            let t0 = crossing(th, st.range, D)
            if !t0.isFinite { return nil }
            var thetaS = [t0]
            var lo = t0 - NP.radians(1.0), hi = t0 + NP.radians(1.0)
            for _ in 0..<2 {
                let tt = NP.linspace(lo, hi, 11)
                let rr = flightStats([Double](repeating: value, count: tt.count), tt, z0: z0, dNet: dNet, L: L, dt: SimConst.solveDt).range
                let c = crossing(tt, rr, D)
                if !c.isFinite { break }
                let w = (hi - lo) / 10
                thetaS.append(c)
                lo = c - w; hi = c + w
            }
            theta = thetaS[thetaS.count - 1]; speed = value
        } else {
            let (loDeg, hiDeg) = mode == "apex" ? (8.0, 80.0) : (-35.0, 82.0)
            let th = NP.linspace(loDeg, hiDeg, 30).map(NP.radians)
            let sp = NP.geomspace(2.5, 120.0, 36)
            var S: [Double] = [], T: [Double] = []
            for i in th.indices { for j in sp.indices { S.append(sp[j]); T.append(th[i]) } }   // meshgrid(sp, th), C order
            let st = flightStats(S, T, z0: z0, dNet: dNet, L: L, dt: coarse)
            let m = sp.count
            let sReq = th.indices.map { i in crossing(sp, Array(st.range[(i * m)..<((i + 1) * m)]), D) }
            var q = [Double](repeating: .nan, count: th.count)
            for i in th.indices where sReq[i].isFinite {
                let row = (0..<m).map { j in mode == "apex" ? st.apex[i * m + j] : st.zNet[i * m + j] - CourtModel.netHeightCentre }
                let ok = row.indices.filter { row[$0].isFinite }
                if ok.count >= 2 { q[i] = NP.interp(sReq[i], ok.map { sp[$0] }, ok.map { row[$0] }) }
            }
            theta = crossing(th, q, value)
            if !theta.isFinite { return nil }
            let fin = th.indices.filter { sReq[$0].isFinite }
            speed = NP.interp(theta, fin.map { th[$0] }, fin.map { sReq[$0] })
        }
        if mode != "speed" {
            speed = polishSpeed(theta: theta, sGuess: speed, D: D, z0: z0, dNet: dNet, L: L)
            if !speed.isFinite { return nil }
        }
        let fine = flightStats([speed], [theta], z0: z0, dNet: dNet, L: L, dt: SimConst.solveDt)
        var stats = ShotStats(range: fine.range[0], time: fine.time[0], apex: fine.apex[0], zNet: fine.zNet[0])
        if !stats.range.isFinite || abs(stats.range - D) > 0.02 { return nil }
        if dNet < 1e8 {
            let clearance = stats.zNet - CourtModel.netHeightCentre
            if let mc = minClearance, !(clearance >= mc) { return nil }
        }
        let v0 = SIMD3(speed * cos(theta) * u.x, speed * cos(theta) * u.y, speed * sin(theta))
        stats.thetaDeg = NP.degrees(theta); stats.speed = speed
        return (v0, stats)
    }

    /// `_solve_net_fault`: a shot that meets the net 4–60 cm below the tape.
    static func solveNetFault(_ rng: any SimRandom, mode: String, p0: SIMD3<Double>, target: SIMD2<Double>, L: Double,
                              value: Double) -> (v0: SIMD3<Double>, stats: ShotStats)? {
        let zAtNet = CourtModel.netHeightCentre - rng.uniform(0.04, 0.6)
        if mode != "speed" {
            return solveShot(p0: p0, target: target, L: L, mode: "clearance", value: zAtNet - CourtModel.netHeightCentre,
                             minClearance: nil)
        }
        let d = target - p0.xy
        let nrm = NP.norm(d.x, d.y)
        let u = SIMD2(d.x / nrm, d.y / nrm)
        if abs(u.x) < 1e-6 { return nil }
        let dNet = (SimConst.netX - p0.x) / u.x
        if dNet <= 0 { return nil }
        let th = NP.linspace(-60, 40, 101).map(NP.radians)
        let st = flightStats([Double](repeating: value, count: th.count), th, z0: p0.z, dNet: dNet, L: L, dt: 1.0 / 120)
        let theta = crossing(th, st.zNet, zAtNet)
        if !theta.isFinite { return nil }
        let v0 = SIMD3(value * cos(theta) * u.x, value * cos(theta) * u.y, value * sin(theta))
        return (v0, ShotStats(apex: nil, zNet: zAtNet, thetaDeg: NP.degrees(theta), speed: value))
    }
}
