import Foundation
import simd
import ShuttlCore
import ShuttlPresent

/// Synthetic frames and cameras for the lock state machine.
enum LockFixtures {
    static let size = ImageSize(width: 640, height: 360)

    /// A camera 5.8 m behind the near baseline, 3.5 m up, seeing the whole court.
    static func camera(dx: Double = 0, dy: Double = 0, roll: Double = 0, conf: Double = 0.9,
                       size: ImageSize = size, eye: SIMD3<Double> = [-12.5, 0, 3.5], focalScale: Double = 0.78) -> CameraModel {
        var c = CameraModel.lookAt(eye: eye, target: [0.5, 0, 0], focalPx: focalScale * Double(size.width),
                                   imageSize: size, rollDeg: roll)
        c.K[2, 0] += dx; c.K[2, 1] += dy          // shifts every projected point by (dx, dy) px
        c.confidence = conf
        return c
    }

    /// Smooth texture (structure at thumbnail scale) shifted by (sx, sy) px; `blob` paints a
    /// 60x110 px "player" at the given x.
    static func texture(sx: Double = 0, sy: Double = 0, brightness: Double = 1, blob: Int? = nil,
                        size: ImageSize = size) -> ImageU8 {
        var img = ImageU8(width: size.width, height: size.height, channels: 3)
        for y in 0..<size.height {
            for x in 0..<size.width {
                let u = Double(x) + sx, v = Double(y) + sy
                var g = 128 + 55 * sin(u / 23 + 0.7) * cos(v / 17) + 35 * sin((u + 2 * v) / 41) + 20 * cos(u / 9 - v / 31)
                if let bx = blob, x >= bx, x < bx + 60, y >= 150, y < 260 { g = 30 }
                let val = UInt8(max(0, min(255, g * brightness)))
                img[x, y, 0] = val; img[x, y, 1] = val; img[x, y, 2] = UInt8(max(0, min(255, g * brightness * 0.9)))
            }
        }
        return img
    }

    static func frame(_ t: Double, sx: Double = 0, brightness: Double = 1, blob: Int? = nil) -> WorkingFrame {
        FrameDownsampler().frame(from: texture(sx: sx, brightness: brightness, blob: blob), time: t)
    }

    static func accept(_ c: CameraModel) -> CalibrationAttempt {
        CalibrationAttempt(camera: c, evidence: CalibrationEvidence(nLines: 20, nHypotheses: 1000, nValid: 1,
                                                                    transverseLines: 6, longitudinalLines: 5, absentLines: 0,
                                                                    explainedFraction: 0.8, margin: 0.9, confidence: c.confidence))
    }

    static func reject(nT: Int? = 6, nL: Int? = 5, absent: Int = 0, expl: Double = 0.8, nValid: Int = 0,
                       margin: Double? = nil, nHyp: Int = 1000) -> CalibrationAttempt {
        CalibrationAttempt(camera: nil, evidence: CalibrationEvidence(nLines: 20, nHypotheses: nHyp, nValid: nValid,
                                                                      transverseLines: nT, longitudinalLines: nL,
                                                                      absentLines: absent, explainedFraction: expl,
                                                                      margin: margin))
    }
}

/// Scripted calibrator for engine tests; counts concurrency.
final class ScriptedCalibrator: CourtLockCalibrator, @unchecked Sendable {
    private let lock = NSLock()
    private var inFlight = 0
    private(set) var maxInFlight = 0
    private(set) var calls: [CourtLockPlan.Kind] = []
    let delay: Double
    let respond: @Sendable (CourtLockPlan.Kind, Int) -> CalibrationAttempt

    init(delay: Double, respond: @escaping @Sendable (CourtLockPlan.Kind, Int) -> CalibrationAttempt) {
        self.delay = delay; self.respond = respond
    }

    private func run(_ kind: CourtLockPlan.Kind) -> CalibrationAttempt {
        let n: Int = lock.withLock {
            inFlight += 1; maxInFlight = max(maxInFlight, inFlight); calls.append(kind); return calls.count - 1
        }
        Thread.sleep(forTimeInterval: delay)
        let r = respond(kind, n)
        lock.withLock { inFlight -= 1 }
        return r
    }

    func search(_ image: ImageU8) -> CalibrationAttempt { run(.search) }
    func refine(_ seed: CameraModel, on image: ImageU8) -> CalibrationAttempt { run(.refine) }
    var callCount: Int { lock.withLock { calls.count } }
    var peak: Int { lock.withLock { maxInFlight } }
}

extension CourtLockState {
    var kindName: String {
        switch self {
        case .searching: return "searching"
        case .candidate: return "candidate"
        case .locked: return "locked"
        case .lost: return "lost"
        }
    }
}
