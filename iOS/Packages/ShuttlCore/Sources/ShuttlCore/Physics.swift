import Foundation
import simd

/// Shuttlecock flight dynamics. Port of backend/app/services/tracking/physics.py.
///
/// Quadratic drag dominates: dv/dt = g − (|v| / L) · v with drag length L = vt² / g,
/// so a shuttle slows from 300+ km/h to its terminal velocity (~6.8 m/s) within
/// metres — a parabola is a poor model (Cohen et al., New J. Phys. 2015).
public enum Physics {
    public static let g = 9.81
    public static let terminalVelocity = 6.8
    public static let dragLength = terminalVelocity * terminalVelocity / g
    public static let gravity = SIMD3<Double>(0, 0, -g)

    @inline(__always)
    public static func accel(_ v: SIMD3<Double>, dragLength: Double = dragLength) -> SIMD3<Double> {
        gravity - simd_length(v) * v / dragLength
    }

    /// One RK4 step of length h seconds (h may be negative).
    public static func rk4Step(p: SIMD3<Double>, v: SIMD3<Double>, h: Double,
                               dragLength: Double = dragLength) -> (p: SIMD3<Double>, v: SIMD3<Double>) {
        let half: Double = 0.5 * h
        let k1v: SIMD3<Double> = accel(v, dragLength: dragLength)
        let k1p: SIMD3<Double> = v
        let k2p: SIMD3<Double> = v + half * k1v
        let k2v: SIMD3<Double> = accel(k2p, dragLength: dragLength)
        let k3p: SIMD3<Double> = v + half * k2v
        let k3v: SIMD3<Double> = accel(k3p, dragLength: dragLength)
        let k4p: SIMD3<Double> = v + h * k3v
        let k4v: SIMD3<Double> = accel(k4p, dragLength: dragLength)
        let w: Double = h / 6
        let dp: SIMD3<Double> = k1p + 2 * k2p + 2 * k3p + k4p
        let dv: SIMD3<Double> = k1v + 2 * k2v + 2 * k3v + k4v
        return (p + w * dp, v + w * dv)
    }

    /// Integrate from (p0, v0) at t = 0 to each of `times` (seconds, monotonic, any sign).
    public static func simulateState(p0: SIMD3<Double>, v0: SIMD3<Double>, times: [Double],
                                     dragLength: Double = dragLength,
                                     maxDt: Double = 1.0 / 480) -> (positions: [SIMD3<Double>], velocities: [SIMD3<Double>]) {
        var P: [SIMD3<Double>] = [], V: [SIMD3<Double>] = []
        P.reserveCapacity(times.count); V.reserveCapacity(times.count)
        var p = p0, v = v0, t = 0.0
        for target in times {
            let span = target - t
            let n = max(1, Int((abs(span) / maxDt).rounded(.up)))
            for _ in 0..<n { (p, v) = rk4Step(p: p, v: v, h: span / Double(n), dragLength: dragLength) }
            t = target
            P.append(p); V.append(v)
        }
        return (P, V)
    }

    public static func simulate(p0: SIMD3<Double>, v0: SIMD3<Double>, times: [Double],
                                dragLength: Double = dragLength) -> [SIMD3<Double>] {
        simulateState(p0: p0, v0: v0, times: times, dragLength: dragLength).positions
    }

    /// Seconds until z crosses `zGround` (linear refinement), nil if it never does.
    public static func timeToGround(p0: SIMD3<Double>, v0: SIMD3<Double>, dragLength: Double = dragLength,
                                    zGround: Double = 0, tMax: Double = 6, dt: Double = 1.0 / 480) -> Double? {
        var p = p0, v = v0, t = 0.0
        while t < tMax {
            let next = rk4Step(p: p, v: v, h: dt, dragLength: dragLength)
            if (next.p.z <= zGround && zGround < p.z) || (p.z > zGround && zGround >= next.p.z) {
                let frac = (p.z - zGround) / max(p.z - next.p.z, 1e-12)
                return t + frac * dt
            }
            (p, v) = next
            t += dt
        }
        return nil
    }

    public static func kmh(_ speedMS: Double) -> Double { speedMS * 3.6 }
}
