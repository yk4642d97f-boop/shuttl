import Foundation
import ShuttlCore

/// Why automatic court calibration gave up, as one short sentence the user can act on.
/// Built from `diagnostics["calibration_auto"]` (keys written by LiveStages: `reason` from
/// court_calibration.REASONS plus line counts). Every message ends where the card's button
/// takes over, so none of them has to mention manual corners itself.
enum CalibrationAdvice {
    /// The setup the calibration sweep found reliable, and that keeps the whole court in frame.
    static let setup = "三脚架约 1.6 米高、底线后约 5 米、正中间"

    /// nil when the court was found (or there's nothing to go on, e.g. a manual calibration).
    static func message(for analysis: MatchAnalysis) -> String? {
        guard analysis.camera == nil else { return nil }
        guard let auto = analysis.diagnostics["calibration_auto"] else { return nil }
        return message(reason: auto["reason"]?.string ?? "",
                       transverse: auto["n_t"]?.double.map(Int.init),
                       longitudinal: auto["n_l"]?.double.map(Int.init),
                       absent: auto["n_absent"]?.double.map(Int.init))
    }

    /// Floor metres per image pixel at the far baseline above which far landings can be off by
    /// ~0.1 m or more (ShuttlCourt sweep2: catches 13/13 such cameras, never trips at 1.6 m × 5 m).
    static let farPrecisionThreshold = 0.22
    static let farPrecisionNote = "球场远端看得比较斜，落点可能差十几厘米。下次把三脚架架高一点（约 1.6 米）会更准。"

    /// For a court that WAS found but is seen too flat at the far end (low / far tripod):
    /// results are shown, with this sentence. nil when precise enough or unknown.
    static func precisionNote(for analysis: MatchAnalysis) -> String? {
        guard analysis.camera != nil,
              let far = analysis.diagnostics["calibration_auto"]?["far_m_per_px"]?.double else { return nil }
        return precisionNote(farMPerPx: far)
    }

    static func precisionNote(farMPerPx: Double) -> String? {
        farMPerPx.isFinite && farMPerPx > farPrecisionThreshold ? farPrecisionNote : nil
    }

    static func message(reason: String, transverse: Int?, longitudinal: Int?, absent: Int?) -> String {
        switch reason {
        case "no_frames":
            return "这段视频读不出画面，没法认球场。"
        case "no_line_families", "no_hypotheses", "no_candidates":
            return "画面里没找到清楚的场地线。拍的时候把整片球场框进画面，按\(setup)来架。"
        case "evidence_gate":
            if let a = absent, a > 0 {
                return "有场地线一直被挡着（人站在线上或放着包），没法确认是哪块场。"
            }
            if let t = transverse, t < 3 {
                return "底线和发球线看得不够全。下次把手机架高一点、往后退，\(setup)。"
            }
            if let l = longitudinal, l < 5 {
                return "两边的边线没拍全。下次站在球场正中间的延长线上，往后退一些。"
            }
            return "场地线看得不够全，没法确认是哪块场。下次按\(setup)来架。"
        case "focal_mismatch", "implausible_focal":
            return "认出来的球场和这台手机的镜头对不上，为了不给错误的球速和落点，先不用它。"
        case "inconsistent_frames":
            return "几张画面认出来的球场位置不一致，可能拍的时候手机动过。架好以后别再碰手机。"
        case "ambiguous":
            return "画面里像球场的线不止一套（旁边的场或地上的彩色胶带），分不清是哪一块。"
        case "low_coverage", "low_confidence", "high_rmse":
            return "场地线太淡或太模糊，认得不够准。光线亮一点、镜头擦干净会好很多。"
        default:
            return "这一场没认出球场。下次按\(setup)来架。"
        }
    }
}
