import Foundation
import CoreGraphics
import simd

// MARK: - AxisFilter
// Constant-acceleration Kalman filter on one image axis, state (p, v, a), driven by white
// jerk of spectral density q (px²/s⁵). Real dt from timestamps, so VFR is handled exactly.

struct AxisFilter: Sendable {
    var x: SIMD3<Double>
    var P: simd_double3x3

    init(position: Double, positionVar: Double, velocityVar: Double, accelVar: Double) {
        x = SIMD3(position, 0, 0)
        P = simd_double3x3(diagonal: SIMD3(positionVar, velocityVar, accelVar))
    }

    mutating func predict(dt: Double, q: Double) {
        guard dt > 0 else { return }
        let f = simd_double3x3(rows: [SIMD3(1, dt, 0.5 * dt * dt), SIMD3(0, 1, dt), SIMD3(0, 0, 1)])
        let d2 = dt * dt, d3 = d2 * dt, d4 = d3 * dt, d5 = d4 * dt
        let qm = q * simd_double3x3(rows: [
            SIMD3(d5 / 20, d4 / 8, d3 / 6),
            SIMD3(d4 / 8, d3 / 3, d2 / 2),
            SIMD3(d3 / 6, d2 / 2, dt),
        ])
        x = f * x
        P = f * P * f.transpose + qm
    }

    /// State after `dt` without committing (for gating).
    func predicted(dt: Double, q: Double) -> AxisFilter {
        var c = self
        c.predict(dt: dt, q: q)
        return c
    }

    mutating func update(_ z: Double, r: Double) {
        let s = P[0, 0] + r
        let k = P.columns.0 / s          // P · Hᵀ with H = (1, 0, 0)
        x += k * (z - x.x)
        // Joseph-free form, symmetrised: P −= K · (H P)
        let hp = SIMD3(P[0, 0], P[1, 0], P[2, 0])
        P -= simd_double3x3(columns: (k * hp.x, k * hp.y, k * hp.z))
        P = 0.5 * (P + P.transpose)
    }

    var position: Double { x.x }
    var velocity: Double { x.y }
}

// MARK: - StaticMap
// Spatial occupancy grid with exponential memory (backend linker._static_costs, online).
// A location firing in a large share of recent frames — or keeps firing while the tracked
// shuttle is far away — is a static distractor (lights, lines, a shuttle on the floor).

struct StaticMap: Sendable {
    private struct Decaying: Sendable {
        var value = 0.0
        var time = -Double.infinity
        func at(_ t: Double, tau: Double) -> Double {
            value == 0 ? 0 : value * exp(-(t - time) / tau)
        }
        mutating func add(_ amount: Double, at t: Double, tau: Double) {
            value = at(t, tau: tau) + amount
            time = t
        }
    }

    private struct Cell: Sendable {
        var all = Decaying()
        var elsewhere = Decaying()
    }

    let cellSize: Double
    let columns: Int
    let rows: Int
    private var cells: [Cell]
    private var frames = Decaying()
    private var presentFrames = Decaying()
    private var touched = Set<Int>()

    init(imageSize: CGSize, cellSize: Double) {
        self.cellSize = max(cellSize, 2)
        columns = max(1, Int((Double(imageSize.width) / self.cellSize).rounded(.up)))
        rows = max(1, Int((Double(imageSize.height) / self.cellSize).rounded(.up)))
        cells = Array(repeating: Cell(), count: columns * rows)
    }

    private func cellIndex(_ p: CGPoint) -> (Int, Int) {
        (min(max(Int(Double(p.x) / cellSize), 0), columns - 1),
         min(max(Int(Double(p.y) / cellSize), 0), rows - 1))
    }

    /// Records one frame: every candidate location, and which ones fired while the tracked
    /// shuttle (if any) was more than `farPx` away.
    mutating func observe(_ points: [CGPoint], shuttle: CGPoint?, farPx: Double, time t: Double, tau: Double) {
        frames.add(1, at: t, tau: tau)
        if shuttle != nil { presentFrames.add(1, at: t, tau: tau) }
        touched.removeAll(keepingCapacity: true)
        for p in points {
            let (cx, cy) = cellIndex(p)
            let i = cy * columns + cx
            guard touched.insert(i).inserted else { continue }   // once per cell per frame
            cells[i].all.add(1, at: t, tau: tau)
            if let s = shuttle, hypot(Double(p.x - s.x), Double(p.y - s.y)) > farPx {
                cells[i].elsewhere.add(1, at: t, tau: tau)
            }
        }
    }

    /// Additive cost (nats) for a candidate at `p`.
    func cost(at p: CGPoint, time t: Double, frameInterval: Double, config c: TrackerConfig) -> Double {
        let (cx, cy) = cellIndex(p)
        var all = 0.0, els = 0.0
        for y in max(cy - 1, 0)...min(cy + 1, rows - 1) {
            for x in max(cx - 1, 0)...min(cx + 1, columns - 1) {
                all += cells[y * columns + x].all.at(t, tau: c.staticMemory)
                els += cells[y * columns + x].elsewhere.at(t, tau: c.staticMemory)
            }
        }
        let dt = max(frameInterval, 1e-3)
        let fAll = all / max(frames.at(t, tau: c.staticMemory), c.staticMinSeconds / dt)
        let fEls = els / max(presentFrames.at(t, tau: c.staticMemory), c.staticMinPresentSeconds / dt)
        func ramp(_ f: Double, _ lo: Double, _ hi: Double) -> Double {
            c.staticCost * min(max((f - lo) / max(hi - lo, 1e-6), 0), 1)
        }
        return max(ramp(fAll, c.staticOccLo, c.staticOccHi), ramp(fEls, c.staticElsewhereLo, c.staticElsewhereHi))
    }
}

// MARK: - Line fit

/// Least-squares velocity of `pts` (px/s) and its per-axis standard error factor
/// 1/√Σ(t−t̄)² (multiply by the pixel noise σ).
func fitVelocity(_ pts: ArraySlice<TrackPoint>) -> (v: SIMD2<Double>, seScale: Double)? {
    guard pts.count >= 2 else { return nil }
    let tm = pts.reduce(0) { $0 + $1.time } / Double(pts.count)
    var stt = 0.0, sx = 0.0, sy = 0.0
    for p in pts {
        let dt = p.time - tm
        stt += dt * dt
        sx += dt * Double(p.pixel.x)
        sy += dt * Double(p.pixel.y)
    }
    guard stt > 1e-12 else { return nil }
    return (SIMD2(sx / stt, sy / stt), 1 / stt.squareRoot())
}
