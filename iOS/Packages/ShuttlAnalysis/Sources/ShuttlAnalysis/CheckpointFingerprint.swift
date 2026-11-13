import CryptoKit
import Foundation
import ShuttlCore

/// Bump when a checkpoint's on-disk form changes, so old files are ignored.
public let checkpointFormatVersion = 2

/// Identifies one analysis input for `FileCheckpointStore`: the video file (size +
/// modification date), the app build and every option that changes a stage's result
/// (manual corners, singles / doubles, camera hint, player step, calibration frames).
/// Re-running with corrected corner taps must not reuse the old camera / players.
///
/// Stable across launches and devices: SHA-256 over a canonical text in which every
/// Double is written as its exact bit pattern and dictionary keys are sorted.
public func checkpointFingerprint(videoSize: Int, modified: Date?, build: String, options: PipelineOptions) -> String {
    var parts: [String] = [
        "format=\(checkpointFormatVersion)",
        "size=\(videoSize)",
        "modified=\(modified.map { canonical(($0.timeIntervalSince1970 * 1000).rounded()) } ?? "nil")",
        "build=\(build)",
    ]
    parts.append(contentsOf: canonicalOptions(options))
    let digest = SHA256.hash(data: Data(parts.joined(separator: "\n").utf8))
    return "v\(checkpointFormatVersion)-" + digest.map { String(format: "%02x", $0) }.joined()
}

/// The canonical lines hashed by `checkpointFingerprint` (exposed for tests / debugging).
public func canonicalOptions(_ o: PipelineOptions) -> [String] {
    var lines = [
        "doubles=\(o.doubles.map { $0 ? "true" : "false" } ?? "nil")",
        "cornersNormalized=\(o.cornersNormalized)",
        "playerStep=\(o.playerStep.map(String.init) ?? "nil")",
        "calibrationFrames=\(o.calibrationFrames)",
    ]
    // Only when set, so fingerprints of runs without a hint stay what they were.
    if let f = o.focalHintPx { lines.append("focalHintPx=\(canonical(f))") }
    if let corners = o.manualCorners {
        let body = corners.keys.sorted().map { k in "\(k):\(canonical(corners[k]!.x)),\(canonical(corners[k]!.y))" }
        lines.append("corners=" + body.joined(separator: ";"))
    } else {
        lines.append("corners=nil")
    }
    if let c = o.cameraHint {
        let k = [c.K.columns.0, c.K.columns.1, c.K.columns.2].flatMap { [$0.x, $0.y, $0.z] }
        let r = [c.R.columns.0, c.R.columns.1, c.R.columns.2].flatMap { [$0.x, $0.y, $0.z] }
        let nums = k + r + [c.t.x, c.t.y, c.t.z] + c.dist + [c.reprojRmsePx, c.confidence]
        lines.append("camera=\(c.imageSize.width)x\(c.imageSize.height):" + nums.map(canonical).joined(separator: ","))
    } else {
        lines.append("camera=nil")
    }
    return lines
}

/// Exact, locale-free text for a Double (NaN / -0 included).
private func canonical(_ x: Double) -> String { String(x.bitPattern, radix: 16) }
