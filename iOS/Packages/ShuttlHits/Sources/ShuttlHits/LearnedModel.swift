import Foundation
#if canImport(CoreML)
import CoreML
#endif

/// A learned candidate re-scorer: (N, F) features in `HIT_FEATURES` order → (N,) logits.
/// Its logit is blended into each candidate's kinematic log-odds with `HitConfig.learnedBlend`
/// (hits.py `_LEARNED`). Throwing = "model failed; use the model-based scores".
public protocol HitRescorer: Sendable {
    func logits(_ features: [[Double]]) throws -> [Double]
}

/// Logistic model `((X - mu) / sd) @ w + b` — the `.npz` model of `load_learned_hit_model`,
/// stored as JSON {"w": [F], "b": x, "mu": [F] | x, "sd": [F] | x}
/// (convert with backend/scripts/swift_fixtures/hits_model_to_json.py).
public struct LinearHitModel: HitRescorer, Codable, Equatable {
    public var w: [Double]
    public var b: Double
    public var mu: [Double]          // per feature (scalar broadcast on load); default 0
    public var sd: [Double]          // default 1
    public var features: [String]?   // optional feature names recorded by the converter

    public init(w: [Double], b: Double, mu: [Double]? = nil, sd: [Double]? = nil) {
        self.w = w; self.b = b
        self.mu = mu ?? [Double](repeating: 0, count: w.count)
        self.sd = sd ?? [Double](repeating: 1, count: w.count)
    }

    enum CodingKeys: String, CodingKey { case w, b, mu, sd, features }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let wv = try c.decode([Double].self, forKey: .w)
        w = wv
        b = try LinearHitModel.scalar(c, .b) ?? 0
        func vec(_ k: CodingKeys, _ def: Double) throws -> [Double] {
            if let v = try? c.decode([Double].self, forKey: k) { return v }
            if let s = try LinearHitModel.scalar(c, k) { return [Double](repeating: s, count: wv.count) }
            return [Double](repeating: def, count: wv.count)
        }
        mu = try vec(.mu, 0)
        sd = try vec(.sd, 1)
        features = try c.decodeIfPresent([String].self, forKey: .features)
        guard mu.count == w.count, sd.count == w.count else {
            throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "w/mu/sd length mismatch"))
        }
    }

    private static func scalar(_ c: KeyedDecodingContainer<CodingKeys>, _ k: CodingKeys) throws -> Double? {
        if let v = try? c.decode(Double.self, forKey: k) { return v }
        if let a = try? c.decode([Double].self, forKey: k), a.count == 1 { return a[0] }
        return nil
    }

    public static func load(contentsOf url: URL) throws -> LinearHitModel {
        try JSONDecoder().decode(LinearHitModel.self, from: Data(contentsOf: url))
    }

    public func logits(_ X: [[Double]]) throws -> [Double] {
        try X.map { row in
            guard row.count == w.count else { throw HitModelError.featureCount(row.count, w.count) }
            var z = 0.0
            for f in 0..<w.count { z += ((row[f] - mu[f]) / sd[f]) * w[f] }
            return z + b
        }
    }
}

public enum HitModelError: Error, Equatable {
    case featureCount(Int, Int)
    case badOutput
}

#if canImport(CoreML)
/// Hook for a future CoreML re-scorer: a model with one MLMultiArray input of shape
/// (N, F) float32/double and one output with N logits. Untested (no model exists yet).
public final class CoreMLHitRescorer: HitRescorer, @unchecked Sendable {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let lock = NSLock()

    public init(model: MLModel, inputName: String = "features", outputName: String = "logits") {
        self.model = model; self.inputName = inputName; self.outputName = outputName
    }

    public convenience init(compiledModelAt url: URL, inputName: String = "features",
                            outputName: String = "logits") throws {
        try self.init(model: MLModel(contentsOf: url), inputName: inputName, outputName: outputName)
    }

    public func logits(_ X: [[Double]]) throws -> [Double] {
        guard let f = X.first?.count, !X.isEmpty else { return [] }
        let arr = try MLMultiArray(shape: [NSNumber(value: X.count), NSNumber(value: f)], dataType: .float32)
        for (i, row) in X.enumerated() { for (j, v) in row.enumerated() { arr[i * f + j] = NSNumber(value: Float(v)) } }
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
        lock.lock(); defer { lock.unlock() }
        let out = try model.prediction(from: input)
        guard let y = out.featureValue(for: outputName)?.multiArrayValue, y.count == X.count else {
            throw HitModelError.badOutput
        }
        return (0..<y.count).map { y[$0].doubleValue }
    }
}
#endif

/// Process-wide installed re-scorer (Python's module-global `_LEARNED`).
public enum HitModelRegistry {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var current: (any HitRescorer)?

    public static var installed: (any HitRescorer)? {
        lock.lock(); defer { lock.unlock() }
        return current
    }

    /// Install (or with nil uninstall) a re-scorer.
    public static func install(_ model: (any HitRescorer)?) {
        lock.lock(); current = model; lock.unlock()
    }

    /// Mirror of `load_learned_hit_model(path)`: a JSON linear model; nil uninstalls.
    /// Returns success; on failure nothing is installed (like Python).
    @discardableResult
    public static func load(path: String?) -> Bool {
        guard let path else { install(nil); return true }
        do {
            install(try LinearHitModel.load(contentsOf: URL(fileURLWithPath: path)))
            return true
        } catch {
            install(nil)
            return false
        }
    }
}
