#if DEBUG
import Foundation
import simd
import ShuttlCore

/// A synthetic but physically plausible on-device analysis for demo mode, so the
/// results screens (landings, score, movement heatmap) can be reviewed without a
/// real recording. Deterministic for a given seed.
enum DemoAnalysis {
    static func make(rallies rallyCount: Int = 14, seed: UInt64 = 20260927) -> MatchAnalysis {
        var rng = LCG(state: seed)
        let fps = 60.0
        let size = ImageSize(width: 1920, height: 1080)
        let camera = CameraModel.lookAt(eye: SIMD3(-11.5, 0, 4.2), target: SIMD3(0.5, 0, 0.4), focalPx: 1500, imageSize: size)

        var rallies: [RallyV2] = []
        var players: [PlayerObservation] = []
        var score: [JSONValue] = []
        var nearPoints = 0, farPoints = 0
        var frame = 90

        for r in 0..<rallyCount {
            let server: CourtSide = r.isMultiple(of: 3) ? .far : .near
            let shotCount = Int.random(in: 3...13, using: &rng)
            var shots: [ShotEventV2] = []
            var hitter = server
            var pos: [CourtSide: SIMD2<Double>] = [.near: SIMD2(-3.4, 0.4), .far: SIMD2(3.6, -0.5)]
            let rallyStart = frame
            var hitFrame = frame + 30
            var landing = SIMD2<Double>(0, 0)
            let endRoll = Double.random(in: 0..<1, using: &rng)
            let end: RallyEndReason = endRoll < 0.62 ? .inCourt : endRoll < 0.86 ? .out : .net

            for s in 0..<shotCount {
                let sign: Double = hitter == .near ? 1 : -1      // direction toward the opponent
                let from = pos[hitter]!
                let type = s == 0 ? ShuttlCore.ShotType.serve : pick(from: from, sign: sign, rng: &rng)
                let (speed, depth, height) = profile(type, rng: &rng)
                let isLast = s == shotCount - 1
                var target = SIMD2(sign * depth, Double.random(in: -2.6...2.6, using: &rng))
                if isLast && end == .out {
                    target = Bool.random(using: &rng) ? SIMD2(sign * Double.random(in: 6.75...7.3, using: &rng), target.y)
                                                      : SIMD2(target.x, (target.y >= 0 ? 1 : -1) * Double.random(in: 3.1...3.5, using: &rng))
                } else if isLast && end == .net {
                    target = SIMD2(-sign * Double.random(in: 0.1...0.5, using: &rng), target.y * 0.6)
                }
                let dir = simd_normalize(SIMD2(target.x - from.x, target.y - from.y))
                let elev = elevation(type)
                let v = speed / 3.6
                let v0 = SIMD3(dir.x * v * cos(elev), dir.y * v * cos(elev), v * sin(elev))
                let flightFrames = Int(max(18, min(95, simd_length(target - from) / max(v * 0.45, 3) * fps)))
                let p0 = SIMD3(from.x, from.y, height)
                let px = camera.project(p0)
                let hit = ShuttlCore.HitEvent(frame: hitFrame, t: Double(hitFrame) / fps, xyPx: px, hitter: hitter,
                                   confidence: Double.random(in: 0.78...0.97, using: &rng))
                let segment = ShotSegment3D(startFrame: hitFrame, endFrame: hitFrame + flightFrames, t0: hit.t,
                                            p0: p0, v0: v0, rmsePx: Double.random(in: 0.8...2.2, using: &rng),
                                            frames: [], positions: [],
                                            landingXY: isLast ? target : nil,
                                            landingT: isLast ? Double(hitFrame + flightFrames) / fps : nil,
                                            confidence: Double.random(in: 0.7...0.95, using: &rng),
                                            endReason: isLast ? (end == .net ? .net : .floor) : .hit)
                shots.append(ShotEventV2(hit: hit, segment: segment, shotType: type,
                                         shotTypeConfidence: Double.random(in: 0.6...0.95, using: &rng),
                                         hitterPosWorld: from, hitHeightM: height,
                                         strokeSide: type == .smash || type == .clear ? .overhead : .forehand))

                // The receiver runs toward the landing spot; the hitter recovers to base.
                let receiver: CourtSide = hitter == .near ? .far : .near
                let catchAt = SIMD2(target.x - sign * 0.5, target.y)
                players += track(side: receiver, from: pos[receiver]!, to: catchAt, frames: hitFrame..<(hitFrame + flightFrames), rng: &rng)
                let base = SIMD2(-sign * 3.3, 0)
                players += track(side: hitter, from: from, to: base, frames: hitFrame..<(hitFrame + flightFrames), rng: &rng)
                pos[receiver] = catchAt
                pos[hitter] = base
                landing = target
                hitFrame += flightFrames
                if !isLast { hitter = receiver }
            }

            let lastHitter = hitter
            let winner: CourtSide = end == .inCourt ? lastHitter : (lastHitter == .near ? .far : .near)
            if winner == .near { nearPoints += 1 } else { farPoints += 1 }
            let margin = CourtModel.inOutMargin(x: landing.x, y: landing.y, doubles: false)
            rallies.append(RallyV2(startFrame: rallyStart, endFrame: hitFrame + 20, shots: shots, endReason: end,
                                   landingXY: end == .net ? nil : landing, inOutMarginM: margin, winner: winner,
                                   confidence: Double.random(in: 0.7...0.95, using: &rng), server: server,
                                   segmentationConfidence: 0.9, outcomeConfidence: r % 5 == 1 ? 0.42 : Double.random(in: 0.66...0.95, using: &rng),
                                   outcomeSource: "demo"))
            score.append(.object([
                "score": .object(["near": .number(Double(nearPoints)), "far": .number(Double(farPoints))]),
                "uncertain": .bool(false),
            ]))
            frame = hitFrame + Int.random(in: 240...600, using: &rng)
        }

        let meta = VideoMeta(fps: fps, width: size.width, height: size.height, nFrames: frame, timestamps: [])
        return MatchAnalysis(meta: meta, camera: camera, players: players, rallies: rallies, score: score,
                             processingTimeSeconds: 41)
    }

    // MARK: - Shot model

    private static func pick(from: SIMD2<Double>, sign: Double, rng: inout LCG) -> ShuttlCore.ShotType {
        let depthFromNet = abs(from.x)
        let options: [ShuttlCore.ShotType] = depthFromNet < 2.4
            ? [.net, .net, .push, .lift, .lift, .kill, .block]
            : depthFromNet < 4.6 ? [.drive, .drive, .push, .smash, .block, .drop, .lift]
            : [.clear, .clear, .smash, .smash, .drop, .drop]
        return options.randomElement(using: &rng)!
    }

    /// (launch speed km/h, target distance from the net m, contact height m)
    private static func profile(_ t: ShuttlCore.ShotType, rng: inout LCG) -> (Double, Double, Double) {
        func r(_ a: ClosedRange<Double>) -> Double { Double.random(in: a, using: &rng) }
        switch t {
        case .smash: return (r(220...338), r(2.5...5.5), r(2.5...2.9))
        case .kill:  return (r(150...240), r(1.2...3.2), r(2.2...2.6))
        case .clear: return (r(150...210), r(5.4...6.5), r(2.4...2.8))
        case .lift:  return (r(110...170), r(5.0...6.4), r(0.4...0.9))
        case .drop:  return (r(60...110), r(0.8...2.2), r(2.3...2.7))
        case .net:   return (r(20...45), r(0.3...1.4), r(0.6...1.2))
        case .block: return (r(40...80), r(0.6...2.0), r(0.9...1.3))
        case .drive: return (r(110...175), r(3.0...5.6), r(1.2...1.6))
        case .push:  return (r(70...120), r(2.2...4.2), r(1.0...1.4))
        case .serve: return (r(40...80), r(2.2...3.8), r(1.0...1.1))
        // The learned classifier's extra serve kinds (short / long) fly like the plain serve.
        case .shortServe, .longServe: return (r(40...80), r(2.2...3.8), r(1.0...1.1))
        // 勾球: a cross-court net shot, so the net-shot profile.
        case .crossNet: return (r(20...45), r(0.3...1.4), r(0.6...1.2))
        case .unknown: return (r(60...120), r(2...5), r(1...2))
        }
    }

    private static func elevation(_ t: ShuttlCore.ShotType) -> Double {
        switch t {
        case .smash: return -0.28
        case .kill:  return -0.5
        case .clear, .lift: return 0.72
        case .drop:  return 0.05
        case .net:   return 0.55
        case .serve: return 0.45
        case .shortServe, .longServe: return 0.45
        case .crossNet: return 0.55
        default:     return 0.12
        }
    }

    /// Foot positions every 4 frames along an eased path with a little sway.
    private static func track(side: CourtSide, from: SIMD2<Double>, to: SIMD2<Double>,
                              frames: Range<Int>, rng: inout LCG) -> [PlayerObservation] {
        let n = max(1, frames.count)
        let clampX: (Double) -> Double = { side == .near ? min(-0.3, max(-6.6, $0)) : max(0.3, min(6.6, $0)) }
        return stride(from: frames.lowerBound, to: frames.upperBound, by: 4).map { f in
            let u = Double(f - frames.lowerBound) / Double(n)
            let e = u * u * (3 - 2 * u)
            let p = from + (to - from) * e
            let sway = SIMD2(Double.random(in: -0.08...0.08, using: &rng), Double.random(in: -0.08...0.08, using: &rng))
            let foot = SIMD2(clampX(p.x + sway.x), min(3.0, max(-3.0, p.y + sway.y)))
            return PlayerObservation(frame: f, trackID: side == .near ? 1 : 2, bbox: BBox(x1: 0, y1: 0, x2: 1, y2: 1),
                                     score: 0.9, side: side, footWorld: foot)
        }
    }

    private struct LCG: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state ^ (state >> 29)
        }
    }
}
#endif
