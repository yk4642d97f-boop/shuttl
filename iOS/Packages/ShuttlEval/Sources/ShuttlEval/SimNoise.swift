import Foundation
import ShuttlCore

// sim.py — detector noise model (corrupt_px / corrupt_detections) and the truth
// converters (truth_track / hits / players / segments / rallies) + benchmark truth.

public enum SimNoise {
    /// `_markov_on`: bursty on/off sequence with stationary on-probability p_on.
    static func markovOn(_ rng: any SimRandom, _ n: Int, _ pOnIn: Double, _ meanOn: Double) -> [Bool] {
        let pOn = NP.clip(pOnIn, 0.0, 0.95)
        if pOn <= 0 { return [Bool](repeating: false, count: n) }
        let qOff = 1.0 / pyMax(meanOn, 1.0)
        let qOn = qOff * pOn / (1 - pOn)
        var out = [Bool](repeating: false, count: n)
        var state = rng.random() < pOn
        let u = rng.random(n)
        for i in 0..<n {
            out[i] = state
            state = state ? (u[i] >= qOff) : (u[i] < qOn)
        }
        return out
    }

    /// `corrupt_px`: true pixel track + visible mask → per-frame detector candidates
    /// (Gaussian jitter inflated with motion blur and 3 % outliers, blur-dependent score,
    /// bursty misses, static / moving / uniform false positives, dead-shuttle hits).
    public static func corruptPx(_ px: [SIMD2<Double>], visible: [Bool], imageSize: ImageSize, rng: any SimRandom,
                                 sigmaPx: Double = 1.5, missRate: Double = 0.1, burstMiss: Bool = true,
                                 falsePositiveRate: Double = 0.03, staticUV: [SIMD2<Double>]? = nil,
                                 movingUV: [[SIMD2<Double>]]? = nil, dead: [(Int, Int, SIMD2<Double>)]? = nil,
                                 deadShuttleRate: Double = 0.5, maxCandidates: Int = 3) -> [[ShuttleCandidate]] {
        let n = px.count, w = Double(imageSize.width), h = Double(imageSize.height)
        let vis = (0..<n).map { visible[$0] && px[$0].isFinite }
        let scale = hypot(w, h) / hypot(1280, 720)
        var step = [Double](repeating: 0, count: n)
        for i in 1..<max(n, 1) {
            let d = NP.rowNorm(px[i].x - px[i - 1].x, px[i].y - px[i - 1].y)
            step[i] = d.isNaN ? 0 : d
        }
        let blur = step.map { NP.clip($0 / (40.0 * scale), 0.0, 1.5) }
        var wgt = blur.map { 0.5 + $0 }
        if vis.contains(true) {
            let m = NP.mean((0..<n).filter { vis[$0] }.map { wgt[$0] })
            wgt = wgt.map { $0 / m }
        }
        let pMiss = wgt.map { NP.clip(missRate * $0, 0.0, 0.95) }
        var miss = [Bool](repeating: false, count: n)
        if burstMiss && missRate > 0 {
            let u1 = rng.random(n)
            for i in 0..<n where u1[i] < 0.5 * pMiss[i] { miss[i] = true }
            let u2 = rng.random(n)
            for s in (0..<n) where vis[s] && u2[s] < 0.5 * pMiss[s] / 9.0 {
                let len = rng.integers(3, 16)
                for k in s..<min(s + len, n) { miss[k] = true }
            }
        } else {
            let u = rng.random(n)
            for i in 0..<n where u[i] < pMiss[i] { miss[i] = true }
        }
        var cands = [[ShuttleCandidate]](repeating: [], count: n)
        for i in 0..<n where vis[i] && !miss[i] {
            let sig = sigmaPx * pow(scale, 0.5) * (1 + 0.3 * blur[i]) * (rng.random() < 0.03 ? 4.0 : 1.0)
            let e = rng.normal(0, sig, 2)
            let sc = NP.clip(rng.normal(0.9 - 0.22 * blur[i], 0.08), 0.12, 0.999)
            cands[i].append(ShuttleCandidate(x: px[i].x + e[0], y: px[i].y + e[1], score: sc))
        }
        func emit(_ mask: [Bool], _ uv: (Int) -> SIMD2<Double>, _ lo: Double, _ hi: Double, _ jitter: Double) {
            for i in 0..<n where mask[i] {
                let q = uv(i)
                if q.isFinite && 0 <= q.x && q.x < w && 0 <= q.y && q.y < h {
                    let e = rng.normal(0, jitter, 2)
                    cands[i].append(ShuttleCandidate(x: q.x + e[0], y: q.y + e[1], score: rng.uniform(lo, hi)))
                }
            }
        }
        let fp = pyMax(falsePositiveRate, 0.0)
        if fp > 0 {
            var st = staticUV ?? []
            if staticUV == nil {
                for _ in 0..<rng.integers(2, 7) {
                    if rng.random() < 0.7 {
                        let x = rng.uniform(0.1, 0.9) * w
                        st.append(SIMD2(x, rng.uniform(0.3, 0.95) * h))
                    } else {
                        let x = rng.uniform(0, w)
                        st.append(SIMD2(x, rng.uniform(0, 0.3 * h)))
                    }
                }
            }
            for uv in st { emit(markovOn(rng, n, 0.5 * fp / Double(st.count), 12.0), { _ in uv }, 0.25, 0.8, 0.6) }
            var mv = movingUV ?? []
            if movingUV == nil {                                  // two random walkers (players' rackets)
                for k in 0..<2 {
                    let v0 = SIMD2(rng.uniform(0.2, 0.8) * w, (0.35 + 0.45 * Double(k)) * h)
                    let steps = rng.normal(0, 0.004 * w, n * 2)
                    var walk: [SIMD2<Double>] = [], acc = SIMD2<Double>(0, 0)
                    for i in 0..<n { acc += SIMD2(steps[2 * i], steps[2 * i + 1]); walk.append(acc) }
                    var sum = SIMD2<Double>(0, 0)
                    for p in walk { sum += p }
                    let mean = sum / Double(n)
                    mv.append(walk.map { p in
                        let q = v0 + p - mean
                        return SIMD2(NP.clip(q.x, 0, w - 1), NP.clip(q.y, 0, h - 1))
                    })
                }
            }
            for uv in mv { emit(markovOn(rng, n, 0.35 * fp / Double(mv.count), 4.0), { uv[$0] }, 0.15, 0.7, 2.0) }
            let u = rng.random(n)
            let cx = rng.uniform(0, w, n), cy = rng.uniform(0, h, n)
            emit(u.map { $0 < 0.15 * fp }, { SIMD2(cx[$0], cy[$0]) }, 0.1, 0.5, 0.0)
        }
        if deadShuttleRate > 0 {
            for (a, b, uv) in dead ?? [] where b >= a {
                var m = [Bool](repeating: false, count: n)
                let u = rng.random(b + 1 - a)
                for (k, i) in (a...b).enumerated() where i < n && u[k] < deadShuttleRate { m[i] = true }
                emit(m, { _ in uv }, 0.35, 0.9, 0.8)
            }
        }
        for i in 0..<n {
            if cands[i].count > 1 {
                let c = cands[i]
                cands[i] = c.indices.sorted { a, b in c[a].score > c[b].score || (c[a].score == c[b].score && a < b) }.map { c[$0] }
            }
            if cands[i].count > maxCandidates { cands[i].removeSubrange(maxCandidates...) }
        }
        return cands
    }

    /// `corrupt_detections`: per-frame detector output like a heatmap detector's top-k peaks,
    /// with distractors on court lines / lights, rackets and shoes, and the dead shuttle.
    public static func corruptDetections(_ sim: SimScenario, rng: any SimRandom, sigmaPx: Double = 1.5, missRate: Double = 0.1,
                                         burstMiss: Bool = true, falsePositiveRate: Double = 0.03,
                                         deadShuttleRate: Double = 0.5, maxCandidates: Int = 3) -> [[ShuttleCandidate]] {
        let w = Double(sim.imageSize.width), h = Double(sim.imageSize.height)
        var st: [SIMD2<Double>] = []
        for _ in 0..<rng.integers(2, 7) {
            if rng.random() < 0.7 {                              // somewhere on a painted line
                let line = CourtModel.lines[rng.integers(CourtModel.lines.count)]
                let f = rng.uniform(0.0, 1.0)                    // rng.uniform()
                let a = line.a, b = line.b
                st.append(sim.camera.project(SIMD3(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), 0)))
            } else {                                              // light / wall fixture
                let x = rng.uniform(0, w)
                st.append(SIMD2(x, rng.uniform(0, 0.3 * h)))
            }
        }
        var moving: [[SIMD2<Double>]] = []
        for p in sim.players {
            moving.append(p.racket.map { sim.camera.project($0) })
            moving.append(p.foot.map { sim.camera.project(SIMD3($0.x, $0.y, 0.05)) })
        }
        let dead = sim.resting.map { ($0.from, $0.to, sim.camera.project($0.p)) }
        return corruptPx(sim.shuttlePx, visible: sim.visible, imageSize: sim.imageSize, rng: rng, sigmaPx: sigmaPx,
                         missRate: missRate, burstMiss: burstMiss, falsePositiveRate: falsePositiveRate, staticUV: st,
                         movingUV: moving, dead: dead, deadShuttleRate: deadShuttleRate, maxCandidates: maxCandidates)
    }
}
