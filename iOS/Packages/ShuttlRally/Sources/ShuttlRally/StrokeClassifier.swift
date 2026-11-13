import Foundation
import ShuttlCore

/// The learned stroke classifier, and the switch that puts it in front of the rule-based one.
///
/// `ShotClassifier.classifyShots` (the port of shots.py, still the default and the fallback)
/// calls `StrokeClassifier.refine` right after its own pass. When the model is available, the
/// rules' verdict is replaced by the model's, for hits whose clip is usable:
///
///   - the 3-D fit, the wrist-overhead pose test, the hitter lookup and the stroke side stay the
///     port's (they feed the ruling too; only the *type* changes);
///   - the type and confidence become the model's, and the strike side moves with it;
///   - the first hit of a rally stays a serve (its type is only refined between short and long).
///
/// Anything unusable — no hit, no players, a clip without real frames, a prediction that fails —
/// leaves the rule-based verdict exactly as it was.
public enum StrokeClassifier {
    private static let state = State()

    /// The model a caller installed (and pinned for the process), or the bundled one.
    public static var model: BSTStrokeModel? { state.model }

    /// Left behind by the last `refine` (the pipeline logs it as a stage event). Not part of the
    /// rally: the analysis JSON shape is pinned to the Python one.
    public static var lastDiagnostics: StrokeDiagnostics? { state.diagnostics }

    /// Where the bundled model is looked up; tests point it at a directory.
    public static func install(_ model: BSTStrokeModel?, pinned: Bool = false) {
        state.set(model, pinned: pinned)
    }

    /// Loads the bundled `BSTStroke.mlpackage` once and pins it. Returns whether a model is
    /// available afterwards. Safe to call repeatedly.
    @discardableResult
    public static func loadBundled(in bundle: Bundle = .main, cacheDirectory: URL? = nil) async -> Bool {
        if state.model != nil { return true }
        let m = await BSTStrokeModel.loadBundled(in: bundle, cacheDirectory: cacheDirectory)
        if let m { state.set(m, pinned: true) }
        return m != nil
    }

    /// A rally with the model's types instead of the rules' where it could rule.
    public static func refine(_ rally: RallyV2, meta: VideoMeta?, track: Track2D?,
                              players: [PlayerObservation]?) -> RallyV2 {
        guard let model = state.model, let meta, let track, let players, !players.isEmpty else { return rally }
        let index = PlayerFrameIndex(players)
        var out = rally
        var refined = 0, unsure = 0, unusable = 0, kept = 0
        for i in out.shots.indices {
            let shot = out.shots[i]
            // No usable clip: the window left the video, or not one frame of it has both players
            // resolved (an all-zero input would be a verdict about nothing).
            guard let clip = StrokeFeatures.clip(hitFrame: shot.hit.frame, meta: meta, players: index, track: track),
                  clip.resolvedFrames > 0 else {
                unusable += 1
                kept += 1
                continue
            }
            guard let logits = try? model.logits(clip), let best = argmax(logits) else {
                unusable += 1
                kept += 1
                continue
            }
            guard let decoded = StrokeClasses.decode(best), let prob = softmaxProbability(logits, best) else {
                unsure += 1
                kept += 1
                continue
            }
            // The serve is the rally's first hit by definition; the model only says which kind.
            if i == 0 && shot.shotType == .serve {
                if decoded.type == .shortServe || decoded.type == .longServe {
                    out.shots[i].shotType = decoded.type
                    out.shots[i].shotTypeConfidence = prob
                    refined += 1
                } else {
                    unsure += 1
                    kept += 1
                }
                continue
            }
            out.shots[i].shotType = decoded.type
            out.shots[i].shotTypeConfidence = prob
            // The model names the striker's side for every class; the 3-D fit keeps the court
            // side of the shot, so it follows the type rather than being overwritten.
            if out.shots[i].hit.hitter != decoded.striker { out.shots[i].hit.hitter = decoded.striker }
            refined += 1
        }
        state.record(StrokeDiagnostics(model: BSTStrokeModel.resourceName, refined: refined,
                                       unsure: unsure, unusable: unusable, kept: kept,
                                       shots: out.shots.count, multiplePerSide: index.multiplePerSide))
        return out
    }

    /// The winning class and its softmax probability (the ensemble's logits are raw, so this is a
    /// temperature-1 softmax; ties go to the lower index, as numpy's argmax does).
    public static func argmax(_ logits: [Float]) -> Int? {
        var best: Int? = nil
        for (i, v) in logits.enumerated() where best == nil || v > logits[best!] { best = i }
        return best
    }

    static func softmaxProbability(_ logits: [Float], _ index: Int) -> Double? {
        guard logits.indices.contains(index) else { return nil }
        let m = logits.max() ?? 0
        var sum = 0.0
        for v in logits { sum += exp(Double(v - m)) }
        guard sum > 0 else { return nil }
        return exp(Double(logits[index] - m)) / sum
    }

    /// Which classifier produced this rally's shot types (kept in the analysis diagnostics).
    public struct StrokeDiagnostics: Sendable, Equatable, Codable {
        public var model: String
        /// Hits whose type the model set.
        public var refined: Int
        /// Hits the model resolved to class 0 "未知球種" (or to no serve kind): rules kept.
        public var unsure: Int
        /// Hits with no usable clip (no player pose, or the window left the video).
        public var unusable: Int
        /// Hits that kept the rule-based verdict (unsure + unusable).
        public var kept: Int
        public var shots: Int
        /// More than one tracked player per court side: the model is singles-only.
        public var multiplePerSide: Bool

        public var json: JSONValue {
            .object(Dictionary(uniqueKeysWithValues: fields.map { ($0.key, $0.value) }))
        }

        /// The same facts flat, for stage metrics and logs (every value a scalar).
        public var fields: [String: JSONValue] {
            ["model": .string(model), "refined": .number(Double(refined)),
             "unsure": .number(Double(unsure)), "unusable": .number(Double(unusable)),
             "kept": .number(Double(kept)), "shots": .number(Double(shots)),
             "multiple_per_side": .bool(multiplePerSide)]
        }
    }

    private final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: BSTStrokeModel?
        private var isPinned = false
        private var last: StrokeDiagnostics?

        var model: BSTStrokeModel? { lock.withLock { stored } }
        var diagnostics: StrokeDiagnostics? { lock.withLock { last } }

        func set(_ m: BSTStrokeModel?, pinned: Bool) {
            lock.withLock {
                if isPinned && !pinned { return }        // a pinned model stays until a pin replaces it
                stored = m
                isPinned = pinned
                last = nil
            }
        }

        func record(_ d: StrokeDiagnostics?) { lock.withLock { last = d } }
    }
}
