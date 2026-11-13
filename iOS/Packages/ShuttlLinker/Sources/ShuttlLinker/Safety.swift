import Foundation

// Degenerate-input safety. On the phone an out-of-range index, an Int(NaN) or an Int
// overflow is a crash, not an exception, so every such site either mirrors numpy exactly
// (saturating casts, wrap-free arithmetic) or throws where Python would raise.

/// Errors `linkCandidates` throws instead of crashing (Python raises at the same places).
public enum LinkerError: Error, Equatable, Sendable {
    /// A config value Python cannot run with (division by zero, NaN → int, negative counts).
    case invalidConfig(String)
    /// Timestamps no clock can be built from (Python: int(nan), np.bincount of a negative block,
    /// or the trailing-weak search running off a non-monotone clock).
    case invalidTimestamps(String)
    /// Index arithmetic would overflow Int64 (Python: silent wrap or MemoryError).
    case inputTooLarge(String)
}

/// numpy's float64 → int64 `astype` on arm64 (the phone and the reference Mac):
/// truncates toward zero and saturates; NaN → 0.
@inline(__always) func npInt64(_ x: Double) -> Int {
    if x.isNaN { return 0 }
    if x >= 9.2233720368547758e18 { return Int.max }
    if x <= -9.2233720368547758e18 { return Int.min }
    return Int(x)
}

/// Python `int(x)` of a float that must be finite: throws where Python raises.
@inline(__always) func pyInt(_ x: Double, _ what: String) throws -> Int {
    guard x.isFinite else { throw LinkerError.invalidTimestamps("\(what) is \(x)") }
    guard abs(x) < 9.2e18 else { throw LinkerError.inputTooLarge("\(what) = \(x)") }
    return Int(x)
}

/// a · b + c with an Int64 overflow check (numpy would wrap silently).
@inline(__always) func mulAdd(_ a: Int, _ b: Int, _ c: Int) throws -> Int {
    let (p, o1) = a.multipliedReportingOverflow(by: b)
    let (s, o2) = p.addingReportingOverflow(c)
    if o1 || o2 { throw LinkerError.inputTooLarge("index key \(a)·\(b)+\(c) overflows Int64") }
    return s
}

extension LinkerConfig {
    /// Rejects values the Python reference cannot run with and Swift would crash on.
    public func validate() throws {
        let doubles: [(String, Double)] = [
            ("minScore", minScore), ("maxSpeed", maxSpeed), ("maxLinkGapS", maxLinkGapS),
            ("sigmaPx", sigmaPx), ("accelSigma", accelSigma), ("dragLengthMin", dragLengthMin),
            ("dragLengthYoung", dragLengthYoung), ("speedUpMax", speedUpMax), ("kinkCost", kinkCost),
            ("kinkMaxGapS", kinkMaxGapS), ("firstCost", firstCost), ("birthCost", birthCost),
            ("missCost", missCost), ("detReward", detReward), ("scoreWeight", scoreWeight),
            ("strongScore", strongScore), ("weakReachS", weakReachS), ("weakFlySpeed", weakFlySpeed),
            ("weakFlyMinS", weakFlyMinS), ("readoptGate", readoptGate), ("staticCell", staticCell),
            ("staticBlockS", staticBlockS), ("staticMinS", staticMinS), ("staticOccLo", staticOccLo),
            ("staticOccHi", staticOccHi), ("staticFar", staticFar), ("staticMinPresentS", staticMinPresentS),
            ("staticLo", staticLo), ("staticHi", staticHi), ("staticCost", staticCost),
            ("kinkWindowS", kinkWindowS), ("kinkZ", kinkZ), ("kinkZRefine", kinkZRefine),
            ("kinkMinDv", kinkMinDv), ("kinkNmsS", kinkNmsS), ("kinkLocS", kinkLocS), ("maxGapS", maxGapS),
            ("jerkFloor", jerkFloor), ("jerkDragLen", jerkDragLen), ("jerkCorrS", jerkCorrS),
            ("huberK", huberK), ("outlierK", outlierK), ("repeatMinMove", repeatMinMove),
            ("kinkResidK", kinkResidK)]
        for (k, v) in doubles where !v.isFinite { throw LinkerError.invalidConfig("\(k) = \(v)") }
        if staticBlockS == 0 { throw LinkerError.invalidConfig("staticBlockS = 0 (Python: ZeroDivisionError)") }
        let ints: [(String, Int, Int)] = [
            ("maxCandidates", maxCandidates, 0), ("kinkMinPts", kinkMinPts, 1), ("kinkMaxPts", kinkMaxPts, 1),
            ("kinkLocPts", kinkLocPts, 1), ("kinkDeg", kinkDeg, 0), ("kinkLocDeg", kinkLocDeg, 0),
            ("refineRounds", refineRounds, 0), ("cutMinCells", cutMinCells, 1)]
        for (k, v, lo) in ints where v < lo { throw LinkerError.invalidConfig("\(k) = \(v) < \(lo)") }
    }
}
