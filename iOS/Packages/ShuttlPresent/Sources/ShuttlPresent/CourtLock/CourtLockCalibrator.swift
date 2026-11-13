import Foundation
import ShuttlCore
import ShuttlCourt

/// What one calibration attempt saw, independent of ShuttlCourt's internal types (so fakes can
/// build it). Mirrors `CourtCalibrationDiagnostics` + the best candidate's `FitStats`.
public struct CalibrationEvidence: Sendable, Equatable {
    public var nLines: Int
    public var nHypotheses: Int
    public var nValid: Int
    /// Best candidate (the chosen camera's when accepted); nil = no candidate at all.
    public var transverseLines: Int?
    public var longitudinalLines: Int?
    public var absentLines: Int?
    public var explainedFraction: Double?
    public var coverage: Double?
    public var rmsePx: Double?
    public var margin: Double?
    public var confidence: Double?

    public init(nLines: Int = 0, nHypotheses: Int = 0, nValid: Int = 0, transverseLines: Int? = nil,
                longitudinalLines: Int? = nil, absentLines: Int? = nil, explainedFraction: Double? = nil,
                coverage: Double? = nil, rmsePx: Double? = nil, margin: Double? = nil, confidence: Double? = nil) {
        self.nLines = nLines; self.nHypotheses = nHypotheses; self.nValid = nValid
        self.transverseLines = transverseLines; self.longitudinalLines = longitudinalLines
        self.absentLines = absentLines; self.explainedFraction = explainedFraction; self.coverage = coverage
        self.rmsePx = rmsePx; self.margin = margin; self.confidence = confidence
    }

    public init(_ d: CourtCalibrationDiagnostics) {
        self.init(nLines: d.nLines, nHypotheses: d.nHyp, nValid: d.nValid,
                  transverseLines: d.stats?.nT, longitudinalLines: d.stats?.nL, absentLines: d.stats?.nAbsent,
                  explainedFraction: d.stats?.explainedFrac, coverage: d.stats?.coverage,
                  rmsePx: d.stats.map { $0.rmse }, margin: d.margin, confidence: d.confidence)
    }
}

/// Result of one attempt at the working resolution.
public struct CalibrationAttempt: Sendable {
    /// Accepted camera at the size of the image it was given; nil = rejected.
    public var camera: CameraModel?
    public var evidence: CalibrationEvidence
    public init(camera: CameraModel?, evidence: CalibrationEvidence) { self.camera = camera; self.evidence = evidence }
}

/// Seam between the lock state machine and ShuttlCourt (tests inject fakes).
public protocol CourtLockCalibrator: Sendable {
    /// Full automatic search on one (median) RGB image.
    func search(_ image: ImageU8) -> CalibrationAttempt
    /// Local re-fit of `seed` (at `image`'s size) on `image`; camera nil when the lines no longer fit.
    func refine(_ seed: CameraModel, on image: ImageU8) -> CalibrationAttempt
}

/// The real calibrator: ShuttlCourt's automatic path and dense refinement.
public struct ShuttlCourtLockCalibrator: CourtLockCalibrator {
    public var config: CourtCalibrationConfig
    public init(config: CourtCalibrationConfig = .default) { self.config = config }

    public func search(_ image: ImageU8) -> CalibrationAttempt {
        let r = CourtCalibration.calibrateFromFramesDetailed([image], size: image.size, config: config)
        return CalibrationAttempt(camera: r.camera, evidence: CalibrationEvidence(r.diagnostics))
    }

    /// `refineCamera` never returns a worse camera: when the fit degrades it hands back its input
    /// unchanged. That fallback is reported as a reject ("the lines no longer fit this camera").
    public func refine(_ seed: CameraModel, on image: ImageU8) -> CalibrationAttempt {
        let out = CourtCalibration.refineCamera(seed, frames: [image], config: config)
        let unchanged = out.K == seed.K && out.R == seed.R && out.t == seed.t && out.dist == seed.dist
        return CalibrationAttempt(camera: unchanged ? nil : out,
                                  evidence: CalibrationEvidence(confidence: unchanged ? nil : out.confidence))
    }
}

/// Diagnostics → hint (see the table on `CourtLockHint`).
public enum CourtLockHints {
    /// Mean luma below this (0…255) on a rejected attempt → too dark.
    public static let darkLuma = 40.0

    public static func hint(for e: CalibrationEvidence, luma: LumaStats?, minExplained: Double = 0.55,
                            minMargin: Double = 0.05) -> CourtLockHint {
        if let luma, luma.mean < darkLuma { return .tooDark }
        guard e.nHypotheses > 0, let nT = e.transverseLines, let nL = e.longitudinalLines else { return .tooFewLines }
        if nT < 3 && nL < 3 { return .tooFewLines }
        if nT < 3 { return .farBaselineMissing }
        if nL < 3 { return .sideLineMissing }
        if (e.absentLines ?? 0) > 0 { return .playerOnLine }
        if let x = e.explainedFraction, x < minExplained { return .linesUnclear }
        if e.nValid > 0, let m = e.margin, m < minMargin { return .ambiguousHalf }
        if e.nValid > 0 { return .linesUnclear }            // confidence / coverage / rmse / focal gate
        return .frameWholeCourt
    }
}
