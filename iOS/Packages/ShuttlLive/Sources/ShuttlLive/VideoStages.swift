import Foundation
import ShuttlCore
import ShuttlPlayers

/// Stages that read the video themselves.
public enum VideoStages {
    /// Apple Vision pose + the players.py tracker. The far-court crop pass is on
    /// whenever a camera is known (Vision misses small far players without it).
    public static func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                                    resume: PlayersResume? = nil, checkpointPolicy: PlayersCheckpointPolicy? = nil,
                                    checkpoint: ((PlayersResume) -> Void)? = nil,
                                    resumeReport: ((String?) -> Void)? = nil,
                                    detector: PoseDetector? = nil, onFrame: ((Int) -> Void)? = nil,
                                    progress: @escaping @Sendable (Double) -> Void,
                                    isCancelled: () -> Bool = { false }) throws -> [PlayerObservation] {
        try PlayerTracker.trackPlayers(source: source, camera: camera, step: step, doubles: doubles, detector: detector,
                                       resume: resume, checkpointPolicy: checkpointPolicy, checkpoint: checkpoint,
                                       resumeReport: resumeReport, onFrame: onFrame,
                                       progress: { progress($0) }, isCancelled: isCancelled).observations
    }
}
