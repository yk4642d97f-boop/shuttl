import CoreGraphics
import Foundation
import ShuttlCore

/// What the record screen shows while the user frames the tripod shot.
///
///  searching(hint) → candidate(projection) → locked(camera, projection, confidence)
///                         ↑                              ↓ phone moved / lines stop fitting
///                         └──────────── lost(hint) ←─────┘
public enum CourtLockState: Sendable, Equatable {
    /// No acceptable court yet; `hint` says what to change.
    case searching(CourtLockHint)
    /// First acceptable solution, waiting for a second agreeing one (draw the lines, but not "locked").
    case candidate(CourtProjection)
    /// Two consecutive solutions agreed within `CourtLockConfiguration.agreePx`. `camera` is at the
    /// full upright frame resolution of the ingested frames; `confidence` is the calibrator's 0…1.
    case locked(camera: CameraModel, projection: CourtProjection, confidence: Double)
    /// Was locked; the phone moved or the lines stopped fitting.
    case lost(CourtLockHint)

    public var isLocked: Bool { if case .locked = self { return true }; return false }

    /// Court overlay to draw (candidate / locked), nil otherwise.
    public var projection: CourtProjection? {
        switch self {
        case .candidate(let p): return p
        case .locked(_, let p, _): return p
        default: return nil
        }
    }

    public var camera: CameraModel? { if case .locked(let c, _, _) = self { return c }; return nil }

    /// The one thing to tell the user now: the search/lost hint, or framing advice for a found court
    /// (nil = nothing to fix).
    public var hint: CourtLockHint? {
        switch self {
        case .searching(let h), .lost(let h): return h
        case .candidate(let p): return p.framing.advice
        case .locked(_, let p, _): return p.framing.advice
        }
    }

    /// Short status line (Chinese).
    public var title: String {
        switch self {
        case .searching: return "正在找球场线…"
        case .candidate: return "找到球场了，先别动"
        case .locked: return "已锁定球场"
        case .lost: return "球场丢了，正在重新找"
        }
    }

    /// Hint copy to show under the title ("" when there is nothing to say).
    public var message: String { hint?.message ?? "" }
}

/// Setup hints. Search / lost hints are derived from the calibrator's diagnostics (and the frame's
/// brightness); framing hints come from the solved camera's geometry. Mapping (see `CourtLockHints`):
///
/// | hint | evidence |
/// |---|---|
/// | tooDark | mean luma < 40/255 on a rejected attempt |
/// | tooFewLines | < 2 line families / no hypothesis, or best candidate matched < 3 transverse AND < 3 longitudinal lines |
/// | farBaselineMissing | best candidate: < 3 transverse (baseline / service) lines matched, ≥ 3 longitudinal |
/// | sideLineMissing | best candidate: < 3 longitudinal (side / centre) lines matched, ≥ 3 transverse |
/// | playerOnLine | best candidate passes the line counts but a clearly visible model line lies on bare floor (n_absent > 0) — what a person / bag standing on a line through the whole median causes |
/// | linesUnclear | line counts OK but < 55 % of the visible model line length is explained, or confidence / coverage / rmse gates fail |
/// | ambiguousHalf | a valid solution exists but a distinct rival explains the lines almost as well (margin < 5 %) — typically only part of the court is visible |
/// | frameWholeCourt | generic fallback (no diagnostics) |
/// | holdStill | global image shift between samples (camera moving) |
/// | phoneMoved | locked camera contradicted (> 8 px) or motion followed by a failed re-fit |
/// | tilted, farBaselineOutOfFrame, nearBaselineOutOfFrame, leftSideOutOfFrame, rightSideOutOfFrame, bothSidesOutOfFrame, courtTooSmall | geometry of the solved camera (candidate / locked only) |
public enum CourtLockHint: String, Sendable, Equatable, CaseIterable, Codable {
    case frameWholeCourt
    case tooDark
    case tooFewLines
    case farBaselineMissing
    case sideLineMissing
    case playerOnLine
    case linesUnclear
    case ambiguousHalf
    case holdStill
    case phoneMoved
    case tilted
    case farBaselineOutOfFrame
    case nearBaselineOutOfFrame
    case leftSideOutOfFrame
    case rightSideOutOfFrame
    case bothSidesOutOfFrame
    case courtTooSmall

    public var message: String {
        switch self {
        case .frameWholeCourt: return "把整片场地框进画面"
        case .tooDark: return "画面太暗：打开场馆灯，或换个亮一点的位置"
        case .tooFewLines: return "场地线太少：把整片场地都框进画面"
        case .farBaselineMissing: return "底线和发球线看不全：把手机抬高一点，或往后退一步"
        case .sideLineMissing: return "有一侧边线看不全：往球场中间挪一点"
        case .playerOnLine: return "有条线被挡住了：线上有人的话请让一下，或稍等片刻"
        case .linesUnclear: return "场地线看不清：擦一下镜头，别用变焦，扶稳手机"
        case .ambiguousHalf: return "只看到一部分场地：把远端底线也框进画面"
        case .holdStill: return "手机在动：先把三脚架放稳"
        case .phoneMoved: return "手机好像动了：放稳后会自动重新锁定"
        case .tilted: return "手机有点歪，扶正一下"
        case .farBaselineOutOfFrame: return "远端底线出画了：镜头往上抬一点"
        case .nearBaselineOutOfFrame: return "近端底线出画了：镜头往下压一点，或往后退一步"
        case .leftSideOutOfFrame: return "左边线出画了：镜头往左转一点"
        case .rightSideOutOfFrame: return "右边线出画了：镜头往右转一点"
        case .bothSidesOutOfFrame: return "两条边线都出画了：往后退一步"
        case .courtTooSmall: return "球场在画面里太小：往前走几步"
        }
    }

    /// Framing advice (the court is found, recording works, but the shot could be better).
    public var isFramingAdvice: Bool {
        switch self {
        case .tilted, .farBaselineOutOfFrame, .nearBaselineOutOfFrame, .leftSideOutOfFrame,
             .rightSideOutOfFrame, .bothSidesOutOfFrame, .courtTooSmall: return true
        default: return false
        }
    }
}
