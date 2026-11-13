import Foundation
import ShuttlCore

/// The hit decoder: `common.detections_from_logits` with `lam = 0`, plus the streaming
/// spelling (`infer.predict_windowed`, 384/192) the on-device pipeline uses for a whole match.
///
/// The shipped decoder is `thr 0.4`, `gap_s 0.30`, `lam 0` (README "Decoding"). `lam > 0` selects
/// the Python-side alternation DP, which is deliberately NOT ported: it is not the shipped
/// default, and a DP spelled differently would change detections without any model to blame.
public enum HitDecoder {

    /// Decoder settings. `lam` exists only to reject a non-zero value loudly (see `alternation`).
    public struct Params: Sendable, Equatable {
        public var thr: Double
        public var gapS: Double
        public var lam: Double

        public init(thr: Double = 0.4, gapS: Double = 0.3, lam: Double = 0) {
            self.thr = thr; self.gapS = gapS; self.lam = lam
        }

        /// The params the shipped model was measured with.
        public static let shipped = Params()
    }

    /// One detected hit, in the model's row terms.
    public struct Detection: Sendable, Equatable {
        /// Row index into the rally's frames (the model's own timeline).
        public var row: Int
        /// Video frame number of that row (`frames[row]`).
        public var frame: Int
        /// `row / fps`.
        public var t: Double
        public var confidence: Double
        /// `"far"` when the side head says the hitter is the far/Top player, else `"near"`.
        public var hitter: CourtSide
    }

    /// Raised when a caller asks for the alternation decoder this port does not implement.
    public struct AlternationUnsupported: Error, CustomStringConvertible {
        public let lam: Double
        public var description: String {
            "the alternation decoder (lam = \(lam)) is not ported; use lam = 0 (the shipped default)"
        }
    }

    /// `np.exp` in float32, then `1 / (1 + exp(-x))` — `common.sigmoid` applied elementwise.
    /// NaN in the logits propagates (Python's numpy does the same); callers see a NaN probability
    /// and the peak test below rejects it, which is the honest behaviour for a broken model output.
    static func sigmoid(_ x: [Float]) -> [Float] { x.map { 1 / (1 + expf(-$0)) } }

    /// `common.local_peaks(p, floor)`: a strict rise into the frame and a non-strict fall out of it,
    /// above the floor. Missing neighbours count as -1, so the first and last frame can be peaks.
    static func localPeaks(_ p: [Float], floor: Float) -> [Int] {
        var out: [Int] = []
        for i in 0..<p.count {
            let left: Float = i > 0 ? p[i - 1] : -1
            let right: Float = i + 1 < p.count ? p[i + 1] : -1
            if p[i] > left && p[i] >= right && p[i] >= floor { out.append(i) }
        }
        return out
    }

    /// `common.decode_nms`: sort the peaks by descending probability (ties keep the earlier frame,
    /// which is numpy's `kind="stable"`), then greedily keep a peak when it is at least
    /// `max(1, round(gap_s * fps))` frames from every already-kept peak. Output is ascending frame.
    static func decodeNMS(_ p: [Float], fps: Double, thr: Double, gapS: Double) -> [Int] {
        let gap = max(1, Int((gapS * fps).rounded(.toNearestOrEven)))
        let idx = localPeaks(p, floor: Float(thr))
        let order = idx.sorted { a, b in p[a] > p[b] || (p[a] == p[b] && a < b) }
        var kept: [Int] = []
        for i in order where kept.allSatisfy({ abs(i - $0) >= gap }) { kept.append(i) }
        return kept.sorted()
    }

    /// `common.detections_from_logits(logits, fps, frames, {thr, gap_s, lam})` for `lam = 0`.
    /// `logits` is (T,2) row-major; `frames` is the video frame of each row.
    public static func detections(logits: [Float], fps: Double, frames: [Int],
                                  params: Params = .shipped) throws -> [Detection] {
        guard params.lam <= 0 else { throw AlternationUnsupported(lam: params.lam) }
        let T = frames.count
        guard T > 0, logits.count == 2 * T else { return [] }
        let hitLogits = stride(from: 0, to: 2 * T, by: 2).map { logits[$0] }
        let sideLogits = stride(from: 1, to: 2 * T, by: 2).map { logits[$0] }
        let p = sigmoid(hitLogits)
        let st = sigmoid(sideLogits)
        return decodeNMS(p, fps: fps, thr: params.thr, gapS: params.gapS).map { i in
            Detection(row: i, frame: frames[i], t: Double(i) / fps, confidence: Double(p[i]),
                      hitter: st[i] > 0.5 ? .far : .near)
        }
    }

    /// Detections from a (T,51) feature matrix, deriving nothing: the model call and the decode.
    public static func detect(features: [Float], frames T: Int, fps: Double,
                              frames frameNumbers: [Int], model: HitLogitsProvider,
                              params: Params = .shipped) throws -> [Detection] {
        let lg = try model.logits(features: features, frames: T)
        return try detections(logits: lg, fps: fps, frames: frameNumbers, params: params)
    }

    // MARK: - Streaming (whole-match) spelling

    /// `infer.predict_windowed`: overlapping windows of `win` frames at `stride`, keeping only the
    /// centre `stride` frames of each (and the true edge at the two ends).
    ///
    /// The graph is non-causal, so a whole match (tens of thousands of frames) cannot be one call;
    /// this is the path the 0.9212 headline was measured through. It differs from a whole-rally
    /// call by up to ~0.1 logit and moves 1 of 3327 detections in the held-out matches
    /// (README "Parity"), which is a documented property of windowing, not an export error.
    public static let windowSize = 384
    public static let windowStride = 192

    /// Logits for one feature matrix, whole-rally when it fits in a window and windowed otherwise.
    /// `features` is (T,51) row-major. Returns (T,2) row-major.
    public static func logitsWindowed(features: [Float], frames T: Int, model: HitLogitsProvider,
                                      win: Int = windowSize, step: Int = windowStride) throws -> [Float] {
        guard T > 0 else { return [] }
        func slice(_ a: Int, _ b: Int) -> [Float] {
            Array(features[(a * HitFeatures.count)..<(b * HitFeatures.count)])
        }
        guard T > win, step > 0, win > step, win <= HitNetModel.maxFrames else {
            return try model.logits(features: features, frames: T)
        }
        var out = [Float](repeating: 0, count: 2 * T)
        var starts = Array(Swift.stride(from: 0, through: T - win, by: step))
        if starts[starts.count - 1] != T - win { starts.append(T - win) }
        let margin = (win - step) / 2
        for (k, a) in starts.enumerated() {
            let lg = try model.logits(features: slice(a, a + win), frames: win)
            let lo = k == 0 ? 0 : margin
            let hi = k == starts.count - 1 ? win : win - margin
            for r in lo..<hi {
                out[2 * (a + r)] = lg[2 * r]
                out[2 * (a + r) + 1] = lg[2 * r + 1]
            }
        }
        return out
    }
}
