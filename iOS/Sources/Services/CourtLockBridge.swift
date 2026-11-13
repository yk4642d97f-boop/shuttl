import CoreGraphics
import ShuttlPresent

/// CourtLockEngine state → what the record screen's framing overlay draws.
extension CourtLockDisplay {
    init(_ state: CourtLockState, mirrored: Bool = false) {
        let phase: Phase
        switch state {
        case .searching: phase = .searching
        case .candidate: phase = .candidate
        case .locked: phase = .locked
        case .lost: phase = .lost
        }
        var outline = state.projection?.outline
        if mirrored { outline = outline?.map { CGPoint(x: 1 - $0.x, y: $0.y) } }
        let size = state.projection.map { CGSize(width: $0.frameSize.width, height: $0.frameSize.height) } ?? .zero
        self.init(phase: phase, title: state.title, message: state.message, outline: outline, imageSize: size)
    }
}
