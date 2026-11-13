import Foundation
import CoreGraphics
import simd

// MARK: - FlightSpeedEstimator
// Monocular launch-speed estimate for ONE completed flight (hit → hit / lost) from a calibrated
// camera: fit the 6-D launch state (p0, v0) of the backend's shuttle physics (gravity +
// quadratic drag, physics.py: vt = 6.8 m/s) to the observed pixels. Gravity and drag fix the
// metric scale a single view lacks. Returned only when the fit is tight AND the speed is well
// determined (σ/speed from the Gauss–Newton covariance) — otherwise nil; the UI must label it
// approximate. The authoritative number is the backend's multi-constraint 3-D reconstruction.

enum FlightSpeedEstimator {
    static let gravity = 9.81
    static let terminalVelocity = 6.8
    static var dragLength: Double { terminalVelocity * terminalVelocity / gravity }

    static let maxRMSEPx = 3.5
    static let maxRelativeSigma = 0.15

    static func estimateKmh(points: [TrackPoint], calibration: CourtCalibration, imageSize: CGSize) -> Double? {
        guard points.count >= 8, let cal = matched(calibration, to: imageSize) else { return nil }
        let t0 = points[0].time
        // Launch state lives at the hit (points[0]); its pixel is noisy, residuals start at 1.
        var obs = Array(points.dropFirst())
        if obs.count > 20 {
            let stride = Double(obs.count - 1) / 19
            obs = (0..<20).map { obs[Int((Double($0) * stride).rounded())] }
        }
        let times = obs.map { $0.time - t0 }
        guard let span = times.last, span > 0.1 else { return nil }

        func residuals(_ p: [Double]) -> [Double] {
            let pos = simulate(p0: SIMD3(p[0], p[1], p[2]), v0: SIMD3(p[3], p[4], p[5]), times: times)
            var r: [Double] = []
            r.reserveCapacity(obs.count * 2)
            for (w, o) in zip(pos, obs) {
                guard let px = cal.project(w) else { r += [50, 50]; continue }
                r.append(Double(px.x - o.pixel.x))
                r.append(Double(px.y - o.pixel.y))
            }
            return r
        }

        var best: (p: [Double], cost: Double)?
        for (h0, h1) in [(2.2, 0.5), (1.2, 1.2), (2.8, 2.0)] {
            guard let a = cal.pixelToFloor(points[0].pixel, height: h0),
                  let b = cal.pixelToFloor(points[points.count - 1].pixel, height: h1) else { continue }
            let T = points[points.count - 1].time - t0
            var v = (b - a) / T
            v.z += 0.5 * gravity * T
            let start = [a.x, a.y, a.z, v.x, v.y, v.z]
            let p = DenseLinearAlgebra.levenbergMarquardt(start, residuals: residuals, maxIterations: 30)
            let cost = residuals(p).reduce(0) { $0 + $1 * $1 }
            if cost < (best?.cost ?? .infinity) { best = (p, cost) }
        }
        guard let fit = best else { return nil }
        let rmse = (fit.cost / Double(obs.count)).squareRoot()
        let p0 = SIMD3(fit.p[0], fit.p[1], fit.p[2]), v0 = SIMD3(fit.p[3], fit.p[4], fit.p[5])
        let speed = simd_length(v0)
        guard rmse <= maxRMSEPx, (0...4.5).contains(p0.z), abs(p0.x) < 9, abs(p0.y) < 5,
              (1...120).contains(speed) else { return nil }
        guard let sigma = speedSigma(fit.p, residuals: residuals, noisePx: max(rmse, 1)),
              sigma / speed <= maxRelativeSigma else { return nil }
        return speed * 3.6
    }

    /// Calibration expressed in the tracker's pixel grid; nil if the orientation differs.
    static func matched(_ c: CourtCalibration, to size: CGSize) -> CourtCalibration? {
        let w = Int(size.width.rounded()), h = Int(size.height.rounded())
        guard w > 0, h > 0 else { return nil }
        if c.imageSize.width == w, c.imageSize.height == h { return c }
        let a0 = Double(c.imageSize.width) / Double(c.imageSize.height), a1 = Double(w) / Double(h)
        guard abs(a0 / a1 - 1) < 0.01 else { return nil }
        return c.scaled(to: .init(width: w, height: h))
    }

    // MARK: Physics (physics.py: dv/dt = g − |v|·v / L)

    static func accel(_ v: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3(0, 0, -gravity) - simd_length(v) * v / dragLength
    }

    /// Positions at `times` (seconds after launch, ascending), RK4 with ≤ 1/120 s steps.
    static func simulate(p0: SIMD3<Double>, v0: SIMD3<Double>, times: [Double]) -> [SIMD3<Double>] {
        var p = p0, v = v0, t = 0.0
        var out: [SIMD3<Double>] = []
        out.reserveCapacity(times.count)
        for target in times {
            let span = target - t
            if span > 0 {
                let n = max(1, Int((span * 120).rounded(.up)))
                let h = span / Double(n)
                for _ in 0..<n {
                    let k1v = accel(v), k1p = v
                    let k2v = accel(v + 0.5 * h * k1v), k2p = v + 0.5 * h * k1v
                    let k3v = accel(v + 0.5 * h * k2v), k3p = v + 0.5 * h * k2v
                    let k4v = accel(v + h * k3v), k4p = v + h * k3v
                    p += h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
                    v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
                }
                t = target
            }
            out.append(p)
        }
        return out
    }

    /// 1-σ of |v0| from the Gauss–Newton covariance σ²·(JᵀJ)⁻¹.
    private static func speedSigma(_ p: [Double], residuals: ([Double]) -> [Double], noisePx: Double) -> Double? {
        let n = p.count
        let r0 = residuals(p)
        let m = r0.count
        var jac = [Double](repeating: 0, count: m * n)
        for j in 0..<n {
            let h = 1e-6 * max(1, abs(p[j]))
            var pp = p, pm = p
            pp[j] += h; pm[j] -= h
            let rp = residuals(pp), rm = residuals(pm)
            for i in 0..<m { jac[i * n + j] = (rp[i] - rm[i]) / (2 * h) }
        }
        var jtj = [Double](repeating: 0, count: n * n)
        for i in 0..<m {
            for a in 0..<n where jac[i * n + a] != 0 {
                for b in 0..<n { jtj[a * n + b] += jac[i * n + a] * jac[i * n + b] }
            }
        }
        let v = SIMD3(p[3], p[4], p[5])
        let s = simd_length(v)
        guard s > 0 else { return nil }
        let g = [0, 0, 0, v.x / s, v.y / s, v.z / s]
        guard let x = DenseLinearAlgebra.solve(jtj, g, n: n) else { return nil }
        let variance = zip(g, x).reduce(0) { $0 + $1.0 * $1.1 } * noisePx * noisePx
        return variance > 0 ? variance.squareRoot() : nil
    }
}
