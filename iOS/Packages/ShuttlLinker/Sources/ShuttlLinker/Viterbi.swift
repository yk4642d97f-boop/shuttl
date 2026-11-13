import Foundation

// Stage 2 — global association DP (`_motion`, `_viterbi` of linker.py).
// Same arithmetic order as the numba kernels (no FMA contraction in Swift).

/// `_motion`: NLL of a displacement (dx, dy) over dt given the predecessor's pair
/// velocity (vx, vy) measured over dtp; `lmin` bounds the along-track decay.
@inline(__always)
func motionNLL(_ dx: Double, _ dy: Double, _ dt: Double, _ vx: Double, _ vy: Double, _ dtp: Double,
               _ lmin: Double, _ sig2: Double, _ acc2: Double, _ up: Double) -> Double {
    let r = dt / dtp
    let q = 0.5 * dt * (dt + dtp)
    let v = sig2 * (1.0 + (1.0 + r) * (1.0 + r) + r * r) + acc2 * (q * q)
    let s = (vx * vx + vy * vy).squareRoot()
    let e2: Double
    if s < 1e-9 {
        e2 = dx * dx + dy * dy
    } else {
        let ux = vx / s, uy = vy / s
        let par = dx * ux + dy * uy
        let perp = dy * ux - dx * uy
        let sp = s / (1.0 + 0.5 * s * dtp / lmin)
        let lo = lmin * log1p(sp * dt / lmin)
        let hi = s * dt * (1.0 + up)
        let ep = par < lo ? lo - par : (par > hi ? par - hi : 0.0)
        e2 = ep * ep + perp * perp
    }
    return 0.5 * e2 / v
}

struct ViterbiResult {
    var chosen: [Int]      // candidate row per frame, −1 none
    var start: [Bool]      // chain-start flags
    var kink: [Bool]       // kink flags
}

/// `_viterbi(off, fidx, xs, ys, u, t, prm, gap, shot0)`: min-cost set of chains.
func viterbi(_ fl: Flat, u: [Double], t: [Double], prm: DPParams, gap: Int, shot0: [Int]) -> ViterbiResult {
    let n = fl.off.count - 1
    let m = fl.count
    let INF = linkerINF
    let vmax = prm.vmax, sig2 = prm.sig2, acc2 = prm.acc2, lmin = prm.lmin, up = prm.up
    let kcap = prm.kcap, cfirst = prm.cfirst, cbirth = prm.cbirth, cmiss = prm.cmiss
    let lyoung = prm.lyoung, kgap = prm.kgap
    let gatePad = 4.0 * sig2.squareRoot()

    let D = UnsafeMutablePointer<Double>.allocate(capacity: 3 * m); D.initialize(repeating: INF, count: 3 * m)
    let bpi = UnsafeMutablePointer<Int>.allocate(capacity: 3 * m); bpi.initialize(repeating: -1, count: 3 * m)
    let bps = UnsafeMutablePointer<Int>.allocate(capacity: 3 * m); bps.initialize(repeating: -1, count: 3 * m)
    let bl = UnsafeMutablePointer<Int>.allocate(capacity: m); bl.initialize(repeating: -1, count: m)
    let bls = UnsafeMutablePointer<Int>.allocate(capacity: m); bls.initialize(repeating: -1, count: m)
    let vx = UnsafeMutablePointer<Double>.allocate(capacity: 3 * m); vx.initialize(repeating: 0, count: 3 * m)
    let vy = UnsafeMutablePointer<Double>.allocate(capacity: 3 * m); vy.initialize(repeating: 0, count: 3 * m)
    let vdt = UnsafeMutablePointer<Double>.allocate(capacity: 3 * m); vdt.initialize(repeating: 1, count: 3 * m)
    let vb = UnsafeMutablePointer<Double>.allocate(capacity: m); vb.initialize(repeating: 0, count: m)
    let vbdt = UnsafeMutablePointer<Double>.allocate(capacity: m); vbdt.initialize(repeating: 1, count: m)
    var fin = [Double](repeating: 0, count: n + 1)
    var finArg = [Int](repeating: -1, count: n + 1)
    var finSt = [Int](repeating: -1, count: n + 1)
    defer {
        D.deallocate(); bpi.deallocate(); bps.deallocate(); bl.deallocate(); bls.deallocate()
        vx.deallocate(); vy.deallocate(); vdt.deallocate(); vb.deallocate(); vbdt.deallocate()
    }

    fl.off.withUnsafeBufferPointer { off in
    fl.xs.withUnsafeBufferPointer { xs in
    fl.ys.withUnsafeBufferPointer { ys in
    u.withUnsafeBufferPointer { u in
    t.withUnsafeBufferPointer { t in
        for f in 0..<n {
            for j in off[f]..<off[f + 1] {
                var b1 = fin[f] + cbirth + u[j]            // fresh via birth (or a kink below)
                var b1i = -1, b1s = -1
                var b1v = 0.0, b1dt = 1.0
                var b0 = INF, b0i = -1, b0s = -1           // tracking
                var b2 = INF, b2i = -1, b2s = -1           // young
                var v0x = 0.0, v0y = 0.0, v0t = 1.0, v2x = 0.0, v2y = 0.0, v2t = 1.0
                let g0 = Swift.max(f - gap, shot0[f])
                let xj = xs[j], yj = ys[j], uj = u[j], tf = t[f]
                var g = f - 1
                while g >= g0 {
                    let dtr = tf - t[g]
                    let dt = dtr >= 1e-6 ? dtr : 1e-6        // max(dt, 1e-6)
                    let skip = cmiss * Double(f - g - 1) + uj
                    let lim = vmax * dt + gatePad
                    for i in off[g]..<off[g + 1] {
                        let d0 = D[3 * i], d1 = D[3 * i + 1], d2 = D[3 * i + 2]
                        let dmin = Swift.min(Swift.min(d0, d1), d2)
                        if dmin + skip >= Swift.max(b0, b2) && dmin + skip + kcap >= b1 { continue }
                        let dx = xj - xs[i], dy = yj - ys[i]
                        if dx * dx + dy * dy > lim * lim { continue }
                        if f - g <= kgap && dmin + skip + kcap < b1 {   // kink
                            b1 = dmin + skip + kcap
                            b1i = i
                            b1s = d0 == dmin ? 0 : (d1 == dmin ? 1 : 2)
                            let sIn: Double
                            if b1s == 1 { sIn = 0.0 } else {
                                let a = vx[3 * i + b1s], b = vy[3 * i + b1s]
                                sIn = (a * a + b * b).squareRoot()
                            }
                            let bv = (dx * dx + dy * dy).squareRoot() / dt - sIn * (1.0 + up)
                            b1v = bv > 0.0 ? bv : 0.0
                            b1dt = dt
                        }
                        var c = d1 + skip + cfirst                   // fresh → young
                        let vbi = vb[i]
                        if vbi > 0.0 {                               // drag can slow it, not stop it
                            let sp = vbi / (1.0 + 0.5 * vbi * vbdt[i] / lyoung)
                            let lo = lyoung * log1p(sp * dt / lyoung)
                            let d = (dx * dx + dy * dy).squareRoot()
                            if d < lo {
                                let r = dt / vbdt[i]
                                let va = sig2 * (1.0 + (1.0 + r) * (1.0 + r) + r * r)
                                c += 0.5 * (lo - d) * (lo - d) / va
                            }
                        }
                        if c < b2 {
                            b2 = c; b2i = i; b2s = 1
                            v2x = dx / dt; v2y = dy / dt; v2t = dt
                        }
                        // → tracking, from states 0 and 2
                        var s = 0
                        while s <= 2 {
                            let base = D[3 * i + s] + skip
                            if base < b0 {
                                let cm = motionNLL(dx, dy, dt, vx[3 * i + s], vy[3 * i + s], vdt[3 * i + s],
                                                   s == 0 ? lmin : lyoung, sig2, acc2, up)
                                if cm < kcap && base + cm < b0 {
                                    b0 = base + cm; b0i = i; b0s = s
                                    v0x = dx / dt; v0y = dy / dt; v0t = dt
                                }
                            }
                            s += 2
                        }
                    }
                    g -= 1
                }
                D[3 * j] = b0; bpi[3 * j] = b0i; bps[3 * j] = b0s
                D[3 * j + 1] = b1; bpi[3 * j + 1] = b1i; bps[3 * j + 1] = b1s
                D[3 * j + 2] = b2; bpi[3 * j + 2] = b2i; bps[3 * j + 2] = b2s
                vb[j] = b1v; vbdt[j] = b1dt
                if b1i < 0 { bl[j] = finArg[f]; bls[j] = finSt[f] }
                vx[3 * j] = v0x; vy[3 * j] = v0y; vdt[3 * j] = v0t
                vx[3 * j + 2] = v2x; vy[3 * j + 2] = v2y; vdt[3 * j + 2] = v2t
            }
            fin[f + 1] = fin[f]; finArg[f + 1] = finArg[f]; finSt[f + 1] = finSt[f]
            for j in off[f]..<off[f + 1] {
                for s in 0..<3 where D[3 * j + s] < fin[f + 1] {
                    fin[f + 1] = D[3 * j + s]; finArg[f + 1] = j; finSt[f + 1] = s
                }
            }
        }
    }}}}}

    var chosen = [Int](repeating: -1, count: n)
    var start = [Bool](repeating: false, count: n)
    var kink = [Bool](repeating: false, count: n)
    var j = finArg[n], s = finSt[n]
    while j >= 0 {
        let f = fl.fidx[j]
        chosen[f] = j
        if s == 1 && bpi[3 * j + 1] < 0 {                 // birth
            start[f] = true
            let nj = bl[j], ns = bls[j]
            j = nj; s = ns
        } else {
            kink[f] = s == 1
            let nj = bpi[3 * j + s], ns = bps[3 * j + s]
            j = nj; s = ns
        }
    }
    return ViterbiResult(chosen: chosen, start: start, kink: kink)
}
