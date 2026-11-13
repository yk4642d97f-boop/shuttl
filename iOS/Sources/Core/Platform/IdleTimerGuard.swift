import UIKit

/// Keeps the screen awake while any holder needs it (recording a match, analysing
/// on this phone). Auto-lock would background the app, which stops a recording
/// and pauses an analysis.
@MainActor
enum IdleTimerGuard {
    private static var holders: [String: Int] = [:]

    static func acquire(_ reason: String) {
        holders[reason, default: 0] += 1
        apply()
    }

    static func release(_ reason: String) {
        guard let n = holders[reason] else { return }
        holders[reason] = n > 1 ? n - 1 : nil
        apply()
    }

    static var isHeld: Bool { !holders.isEmpty }

    private static func apply() {
        UIApplication.shared.isIdleTimerDisabled = !holders.isEmpty
    }
}
