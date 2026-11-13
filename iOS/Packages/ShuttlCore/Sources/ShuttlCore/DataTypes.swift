import Foundation
import simd

// Tracking v2 contracts — port of the v2 section of backend/app/services/tracking/data_types.py.
// Pixel coords are FULL-RESOLUTION pixels of the upright frame; world coords follow
// `CourtModel` (origin = court centre, +x → far baseline, +z up). Time is seconds from
// video start — use timestamps, not frame / fps (phone video is often variable frame rate).
// JSON keys are the Python field names; NaN ⇄ null (see `ShuttlJSON`).

public enum CourtSide: String, LenientStringEnum {
    case near, far, unknown
    public static var unknownCase: CourtSide { .unknown }
    public var opposite: CourtSide { self == .near ? .far : self == .far ? .near : .unknown }
}

// MARK: - Video

public struct VideoMeta: Sendable, Equatable, Codable {
    public var fps: Double                  // nominal
    public var width: Int                   // upright frame size
    public var height: Int
    public var nFrames: Int
    public var timestamps: [Double]         // seconds, one per frame, presentation order
    public var rotationDeg: Int

    public init(fps: Double, width: Int, height: Int, nFrames: Int, timestamps: [Double], rotationDeg: Int = 0) {
        self.fps = fps; self.width = width; self.height = height
        self.nFrames = nFrames; self.timestamps = timestamps; self.rotationDeg = rotationDeg
    }

    public var size: ImageSize { ImageSize(width: width, height: height) }

    /// Timestamp of a frame index, falling back to nominal fps outside the table.
    public func time(ofFrame f: Int) -> Double {
        f >= 0 && f < timestamps.count ? timestamps[f] : Double(f) / max(fps, 1e-9)
    }

    enum CodingKeys: String, CodingKey {
        case fps, width, height, nFrames = "n_frames", timestamps, rotationDeg = "rotation_deg"
    }
}

// MARK: - Shuttle

public struct ShuttleCandidate: Sendable, Equatable, Codable {
    public var x: Double                    // px
    public var y: Double                    // px
    public var score: Double                // detector confidence 0…1
    public init(x: Double, y: Double, score: Double) { self.x = x; self.y = y; self.score = score }
}

/// One shuttle position per frame after linking / cleaning.
public struct Track2D: Sendable, Codable {
    public var t: [Double]                  // (N,) seconds
    public var xy: [SIMD2<Double>]          // (N,) px; NaN where not visible and not filled
    public var conf: [Double]               // 0…1 (0 for filled frames)
    public var visible: [Bool]              // a real detection was used
    public var interpolated: [Bool]         // gap-filled
    /// The adopted detection itself (unsmoothed), NaN where none was adopted.
    /// Kink / hit analysis should prefer it: smoothing can bend a path where the data doesn't.
    public var rawXY: [SIMD2<Double>]?

    public init(t: [Double], xy: [SIMD2<Double>], conf: [Double], visible: [Bool],
                interpolated: [Bool], rawXY: [SIMD2<Double>]? = nil) {
        self.t = t; self.xy = xy; self.conf = conf; self.visible = visible
        self.interpolated = interpolated; self.rawXY = rawXY
    }

    public var n: Int { t.count }

    enum CodingKeys: String, CodingKey { case t, xy, conf, visible, interpolated, rawXY = "raw_xy" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        t = try c.decodeNaNableArray(forKey: .t) ?? []
        xy = (try c.decodeNaNableRows(forKey: .xy) ?? []).map(SIMD2.init(row:))
        conf = try c.decodeNaNableArray(forKey: .conf) ?? []
        visible = try c.decode([Bool].self, forKey: .visible)
        interpolated = try c.decode([Bool].self, forKey: .interpolated)
        rawXY = try c.decodeNaNableRows(forKey: .rawXY)?.map(SIMD2.init(row:))
        if let why = inconsistency {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: why))
        }
    }

    /// nil when every per-frame array has one entry per timestamp; otherwise what differs.
    /// Stages index these arrays with the same frame number, so a mismatch would trap.
    public var inconsistency: String? {
        let n = t.count
        var bad: [String] = []
        if xy.count != n { bad.append("xy \(xy.count)") }
        if conf.count != n { bad.append("conf \(conf.count)") }
        if visible.count != n { bad.append("visible \(visible.count)") }
        if interpolated.count != n { bad.append("interpolated \(interpolated.count)") }
        if let rawXY, rawXY.count != n { bad.append("raw_xy \(rawXY.count)") }
        return bad.isEmpty ? nil : "Track2D arrays differ in length (t \(n), " + bad.joined(separator: ", ") + ")"
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeNaNableArray(t, forKey: .t)
        try c.encodeNaNableRows(xy.map(\.row), forKey: .xy)
        try c.encodeNaNableArray(conf, forKey: .conf)
        try c.encode(visible, forKey: .visible)
        try c.encode(interpolated, forKey: .interpolated)
        if let rawXY { try c.encodeNaNableRows(rawXY.map(\.row), forKey: .rawXY) }
    }
}

// MARK: - Players

/// COCO-17 keypoint order (what the Python stack and its fixtures use).
/// Vision's body-pose joints are mapped onto this order by the players stage.
public enum COCOKeypoint: Int, CaseIterable, Sendable {
    case nose, leftEye, rightEye, leftEar, rightEar
    case leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist
    case leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle
}

public struct Keypoint: Sendable, Equatable, Codable {
    public var x: Double                    // px
    public var y: Double                    // px
    public var conf: Double
    public init(x: Double, y: Double, conf: Double) { self.x = x; self.y = y; self.conf = conf }

    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        x = try c.decodeNil() ? .nan : try c.decode(Double.self)
        y = try c.decodeNil() ? .nan : try c.decode(Double.self)
        conf = try c.decodeNil() ? 0 : try c.decode(Double.self)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(x.isFinite ? x : nil); try c.encode(y.isFinite ? y : nil); try c.encode(conf)
    }
}

/// [x1, y1, x2, y2] px.
public struct BBox: Sendable, Equatable, Codable {
    public var x1, y1, x2, y2: Double
    public init(x1: Double, y1: Double, x2: Double, y2: Double) { self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2 }
    public var width: Double { x2 - x1 }
    public var height: Double { y2 - y1 }
    public var center: SIMD2<Double> { [(x1 + x2) / 2, (y1 + y2) / 2] }

    public init(from decoder: Decoder) throws {
        let v = try decoder.singleValueContainer().decode([Double].self)
        guard v.count == 4 else { throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "bbox needs 4 values")) }
        (x1, y1, x2, y2) = (v[0], v[1], v[2], v[3])
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode([x1, y1, x2, y2])
    }
}

public struct PlayerObservation: Sendable, Equatable, Codable {
    public var frame: Int
    public var trackID: Int
    public var bbox: BBox
    public var score: Double
    public var side: CourtSide
    public var footWorld: SIMD2<Double>?    // (x, y) m on the floor
    public var keypoints: [Keypoint]?       // 17 COCO keypoints, px

    public init(frame: Int, trackID: Int, bbox: BBox, score: Double, side: CourtSide = .unknown,
                footWorld: SIMD2<Double>? = nil, keypoints: [Keypoint]? = nil) {
        self.frame = frame; self.trackID = trackID; self.bbox = bbox; self.score = score
        self.side = side; self.footWorld = footWorld; self.keypoints = keypoints
    }

    public func keypoint(_ k: COCOKeypoint) -> Keypoint? {
        guard let keypoints, keypoints.count > k.rawValue else { return nil }
        return keypoints[k.rawValue]
    }

    enum CodingKeys: String, CodingKey {
        case frame, trackID = "track_id", bbox, score, side, footWorld = "foot_world", keypoints
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        frame = try c.decode(Int.self, forKey: .frame)
        trackID = try c.decode(Int.self, forKey: .trackID)
        bbox = try c.decode(BBox.self, forKey: .bbox)
        score = try c.decode(Double.self, forKey: .score)
        side = try c.decodeIfPresent(CourtSide.self, forKey: .side) ?? .unknown
        footWorld = try c.decodeNaNableArray(forKey: .footWorld).map(SIMD2.init(row:))
        keypoints = try c.decodeIfPresent([Keypoint].self, forKey: .keypoints)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(frame, forKey: .frame)
        try c.encode(trackID, forKey: .trackID)
        try c.encode(bbox, forKey: .bbox)
        try c.encode(score, forKey: .score)
        try c.encode(side, forKey: .side)
        if let footWorld { try c.encodeNaNableArray(footWorld.row, forKey: .footWorld) }
        try c.encodeIfPresent(keypoints, forKey: .keypoints)
    }
}

// MARK: - Hits, flights, shots, rallies

public struct HitEvent: Sendable, Equatable, Codable {
    public var frame: Int
    public var t: Double
    public var xyPx: SIMD2<Double>
    public var hitter: CourtSide
    public var confidence: Double
    public var playerTrackID: Int?
    /// How the contact was found: "kink" (a visible direction change) or an inferred kind such
    /// as "gap_contact". Encoded only when it isn't "kink" (Python does the same).
    public var source: String

    public static let defaultSource = "kink"

    public init(frame: Int, t: Double, xyPx: SIMD2<Double>, hitter: CourtSide = .unknown,
                confidence: Double = 0, playerTrackID: Int? = nil, source: String = HitEvent.defaultSource) {
        self.frame = frame; self.t = t; self.xyPx = xyPx; self.hitter = hitter
        self.confidence = confidence; self.playerTrackID = playerTrackID; self.source = source
    }

    public var isInferred: Bool { source != Self.defaultSource }

    enum CodingKeys: String, CodingKey {
        case frame, t, xyPx = "xy_px", hitter, confidence, playerTrackID = "player_track_id", source
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        frame = try c.decode(Int.self, forKey: .frame)
        t = try c.decodeNaNable(forKey: .t) ?? .nan
        xyPx = SIMD2(row: try c.decodeNaNableArray(forKey: .xyPx) ?? [.nan, .nan])
        hitter = try c.decodeIfPresent(CourtSide.self, forKey: .hitter) ?? .unknown
        confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 0
        playerTrackID = try c.decodeIfPresent(Int.self, forKey: .playerTrackID)
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? Self.defaultSource
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(frame, forKey: .frame)
        try c.encodeNaNable(t, forKey: .t)
        try c.encodeNaNableArray(xyPx.row, forKey: .xyPx)
        try c.encode(hitter, forKey: .hitter)
        try c.encode(confidence, forKey: .confidence)
        try c.encodeIfPresent(playerTrackID, forKey: .playerTrackID)
        if isInferred { try c.encode(source, forKey: .source) }
    }
}

public enum SegmentEndReason: String, LenientStringEnum {
    case hit, floor, net, unknown
    public static var unknownCase: SegmentEndReason { .unknown }
}

/// Flight between two consecutive hits (or hit → floor / net / out of view).
public struct ShotSegment3D: Sendable, Codable {
    public var startFrame: Int
    public var endFrame: Int
    public var t0: Double                   // time of p0 / v0
    public var p0: SIMD3<Double>            // m
    public var v0: SIMD3<Double>            // m/s
    public var dragLength: Double
    public var rmsePx: Double               // reprojection error of the fit
    public var frames: [Int]                // frame indices covered
    public var positions: [SIMD3<Double>]   // fitted positions at those frames
    public var landingXY: SIMD2<Double>?    // floor contact, if reached
    public var landingT: Double?
    public var netCrossingHeight: Double?   // z at x = 0, if crossed
    public var confidence: Double
    public var netCrossingY: Double?        // y at x = 0, if crossed
    public var endReason: SegmentEndReason
    public var hitterFeetXY: SIMD2<Double>? // hitter's feet on the floor at contact

    public init(startFrame: Int, endFrame: Int, t0: Double, p0: SIMD3<Double>, v0: SIMD3<Double>,
                dragLength: Double = Physics.dragLength, rmsePx: Double, frames: [Int], positions: [SIMD3<Double>],
                landingXY: SIMD2<Double>? = nil, landingT: Double? = nil, netCrossingHeight: Double? = nil,
                confidence: Double = 0, netCrossingY: Double? = nil, endReason: SegmentEndReason = .unknown,
                hitterFeetXY: SIMD2<Double>? = nil) {
        self.startFrame = startFrame; self.endFrame = endFrame; self.t0 = t0; self.p0 = p0; self.v0 = v0
        self.dragLength = dragLength; self.rmsePx = rmsePx; self.frames = frames; self.positions = positions
        self.landingXY = landingXY; self.landingT = landingT; self.netCrossingHeight = netCrossingHeight
        self.confidence = confidence; self.netCrossingY = netCrossingY; self.endReason = endReason
        self.hitterFeetXY = hitterFeetXY
    }

    public var speed0Kmh: Double { simd_length(v0) * 3.6 }

    enum CodingKeys: String, CodingKey {
        case startFrame = "start_frame", endFrame = "end_frame", t0, p0, v0, dragLength = "drag_length"
        case rmsePx = "rmse_px", frames, positions, landingXY = "landing_xy", landingT = "landing_t"
        case netCrossingHeight = "net_crossing_height", confidence, netCrossingY = "net_crossing_y"
        case endReason = "end_reason", hitterFeetXY = "hitter_feet_xy"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        startFrame = try c.decode(Int.self, forKey: .startFrame)
        endFrame = try c.decode(Int.self, forKey: .endFrame)
        t0 = try c.decodeNaNable(forKey: .t0) ?? .nan
        p0 = SIMD3(row: try c.decodeNaNableArray(forKey: .p0) ?? [.nan, .nan, .nan])
        v0 = SIMD3(row: try c.decodeNaNableArray(forKey: .v0) ?? [.nan, .nan, .nan])
        dragLength = try c.decodeNaNable(forKey: .dragLength) ?? Physics.dragLength
        rmsePx = try c.decodeNaNable(forKey: .rmsePx) ?? .nan
        frames = try c.decodeIfPresent([Int].self, forKey: .frames) ?? []
        positions = (try c.decodeNaNableRows(forKey: .positions) ?? []).map(SIMD3.init(row:))
        landingXY = try c.decodeNaNableArray(forKey: .landingXY).map(SIMD2.init(row:))
        landingT = try c.decodeIfPresent(Double.self, forKey: .landingT)
        netCrossingHeight = try c.decodeIfPresent(Double.self, forKey: .netCrossingHeight)
        confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 0
        netCrossingY = try c.decodeIfPresent(Double.self, forKey: .netCrossingY)
        endReason = try c.decodeIfPresent(SegmentEndReason.self, forKey: .endReason) ?? .unknown
        hitterFeetXY = try c.decodeNaNableArray(forKey: .hitterFeetXY).map(SIMD2.init(row:))
        guard positions.count == frames.count else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription:
                "ShotSegment3D frames (\(frames.count)) and positions (\(positions.count)) differ in length"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(startFrame, forKey: .startFrame)
        try c.encode(endFrame, forKey: .endFrame)
        try c.encodeNaNable(t0, forKey: .t0)
        try c.encodeNaNableArray(p0.row, forKey: .p0)
        try c.encodeNaNableArray(v0.row, forKey: .v0)
        try c.encodeNaNable(dragLength, forKey: .dragLength)
        try c.encodeNaNable(rmsePx, forKey: .rmsePx)
        try c.encode(frames, forKey: .frames)
        try c.encodeNaNableRows(positions.map(\.row), forKey: .positions)
        if let landingXY { try c.encodeNaNableArray(landingXY.row, forKey: .landingXY) }
        try c.encodeIfPresent(landingT, forKey: .landingT)
        try c.encodeIfPresent(netCrossingHeight, forKey: .netCrossingHeight)
        try c.encode(confidence, forKey: .confidence)
        try c.encodeIfPresent(netCrossingY, forKey: .netCrossingY)
        try c.encode(endReason, forKey: .endReason)
        if let hitterFeetXY { try c.encodeNaNableArray(hitterFeetXY.row, forKey: .hitterFeetXY) }
    }
}

public enum ShotType: String, LenientStringEnum {
    case smash, clear, drop, net, drive, serve, lift, push, kill, block, unknown
    /// Types only the learned stroke classifier (BST, ShuttleSet classes) tells apart:
    /// 勾球 cross-court net shot, 發短球 short service, 發長球 long service.
    case crossNet = "cross_net", shortServe = "short_serve", longServe = "long_serve"
    public static var unknownCase: ShotType { .unknown }

    /// Fine-grained types collapse onto the closest legacy type (serialize.LEGACY_TYPE).
    public var legacy: ShotType {
        switch self {
        case .lift: return .clear
        case .push: return .drive
        case .kill: return .smash
        case .block: return .net
        case .crossNet: return .net
        case .shortServe, .longServe: return .serve
        default: return self
        }
    }
}

public enum StrokeSide: String, LenientStringEnum {
    case forehand, backhand, overhead, unknown
    public static var unknownCase: StrokeSide { .unknown }
}

public struct ShotEventV2: Sendable, Codable {
    public var hit: HitEvent
    public var segment: ShotSegment3D?
    public var shotType: ShotType
    public var shotTypeConfidence: Double
    public var hitterPosWorld: SIMD2<Double>?
    public var hitHeightM: Double?
    public var strokeSide: StrokeSide

    public init(hit: HitEvent, segment: ShotSegment3D?, shotType: ShotType = .unknown, shotTypeConfidence: Double = 0,
                hitterPosWorld: SIMD2<Double>? = nil, hitHeightM: Double? = nil, strokeSide: StrokeSide = .unknown) {
        self.hit = hit; self.segment = segment; self.shotType = shotType; self.shotTypeConfidence = shotTypeConfidence
        self.hitterPosWorld = hitterPosWorld; self.hitHeightM = hitHeightM; self.strokeSide = strokeSide
    }

    enum CodingKeys: String, CodingKey {
        case hit, segment, shotType = "shot_type", shotTypeConfidence = "shot_type_confidence"
        case hitterPosWorld = "hitter_pos_world", hitHeightM = "hit_height_m", strokeSide = "stroke_side"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hit = try c.decode(HitEvent.self, forKey: .hit)
        segment = try c.decodeIfPresent(ShotSegment3D.self, forKey: .segment)
        shotType = try c.decodeIfPresent(ShotType.self, forKey: .shotType) ?? .unknown
        shotTypeConfidence = try c.decodeIfPresent(Double.self, forKey: .shotTypeConfidence) ?? 0
        hitterPosWorld = try c.decodeNaNableArray(forKey: .hitterPosWorld).map(SIMD2.init(row:))
        hitHeightM = try c.decodeIfPresent(Double.self, forKey: .hitHeightM)
        strokeSide = try c.decodeIfPresent(StrokeSide.self, forKey: .strokeSide) ?? .unknown
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hit, forKey: .hit)
        try c.encodeIfPresent(segment, forKey: .segment)
        try c.encode(shotType, forKey: .shotType)
        try c.encode(shotTypeConfidence, forKey: .shotTypeConfidence)
        if let hitterPosWorld { try c.encodeNaNableArray(hitterPosWorld.row, forKey: .hitterPosWorld) }
        try c.encodeIfPresent(hitHeightM, forKey: .hitHeightM)
        try c.encode(strokeSide, forKey: .strokeSide)
    }
}

public enum RallyEndReason: String, LenientStringEnum {
    case inCourt = "in", out, net, unknown
    public static var unknownCase: RallyEndReason { .unknown }
}

public struct RallyV2: Sendable, Codable {
    public var startFrame: Int
    public var endFrame: Int
    public var shots: [ShotEventV2]
    public var endReason: RallyEndReason
    public var landingXY: SIMD2<Double>?
    public var inOutMarginM: Double?
    public var winner: CourtSide
    public var confidence: Double
    public var server: CourtSide
    public var serveMissed: Bool             // shuttle already in flight before the first detected hit
    public var segmentationConfidence: Double
    public var outcomeConfidence: Double
    public var outcomeSource: String         // how end_reason / winner were decided

    public init(startFrame: Int, endFrame: Int, shots: [ShotEventV2] = [], endReason: RallyEndReason = .unknown,
                landingXY: SIMD2<Double>? = nil, inOutMarginM: Double? = nil, winner: CourtSide = .unknown,
                confidence: Double = 0, server: CourtSide = .unknown, serveMissed: Bool = false,
                segmentationConfidence: Double = 0, outcomeConfidence: Double = 0, outcomeSource: String = "unknown") {
        self.startFrame = startFrame; self.endFrame = endFrame; self.shots = shots; self.endReason = endReason
        self.landingXY = landingXY; self.inOutMarginM = inOutMarginM; self.winner = winner
        self.confidence = confidence; self.server = server; self.serveMissed = serveMissed
        self.segmentationConfidence = segmentationConfidence; self.outcomeConfidence = outcomeConfidence
        self.outcomeSource = outcomeSource
    }

    enum CodingKeys: String, CodingKey {
        case startFrame = "start_frame", endFrame = "end_frame", shots, endReason = "end_reason"
        case landingXY = "landing_xy", inOutMarginM = "in_out_margin_m", winner, confidence, server
        case serveMissed = "serve_missed", segmentationConfidence = "segmentation_confidence"
        case outcomeConfidence = "outcome_confidence", outcomeSource = "outcome_source"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        startFrame = try c.decode(Int.self, forKey: .startFrame)
        endFrame = try c.decode(Int.self, forKey: .endFrame)
        shots = try c.decodeIfPresent([ShotEventV2].self, forKey: .shots) ?? []
        endReason = try c.decodeIfPresent(RallyEndReason.self, forKey: .endReason) ?? .unknown
        landingXY = try c.decodeNaNableArray(forKey: .landingXY).map(SIMD2.init(row:))
        inOutMarginM = try c.decodeIfPresent(Double.self, forKey: .inOutMarginM)
        winner = try c.decodeIfPresent(CourtSide.self, forKey: .winner) ?? .unknown
        confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 0
        server = try c.decodeIfPresent(CourtSide.self, forKey: .server) ?? .unknown
        serveMissed = try c.decodeIfPresent(Bool.self, forKey: .serveMissed) ?? false
        segmentationConfidence = try c.decodeIfPresent(Double.self, forKey: .segmentationConfidence) ?? 0
        outcomeConfidence = try c.decodeIfPresent(Double.self, forKey: .outcomeConfidence) ?? 0
        outcomeSource = try c.decodeIfPresent(String.self, forKey: .outcomeSource) ?? "unknown"
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(startFrame, forKey: .startFrame)
        try c.encode(endFrame, forKey: .endFrame)
        try c.encode(shots, forKey: .shots)
        try c.encode(endReason, forKey: .endReason)
        if let landingXY { try c.encodeNaNableArray(landingXY.row, forKey: .landingXY) }
        try c.encodeIfPresent(inOutMarginM, forKey: .inOutMarginM)
        try c.encode(winner, forKey: .winner)
        try c.encode(confidence, forKey: .confidence)
        try c.encode(server, forKey: .server)
        try c.encode(serveMissed, forKey: .serveMissed)
        try c.encode(segmentationConfidence, forKey: .segmentationConfidence)
        try c.encode(outcomeConfidence, forKey: .outcomeConfidence)
        try c.encode(outcomeSource, forKey: .outcomeSource)
    }
}

/// Everything one analysis run produces (pipeline_v2.MatchAnalysis).
public struct MatchAnalysis: Sendable, Codable {
    public var meta: VideoMeta
    public var track: Track2D?
    public var camera: CameraModel?
    public var players: [PlayerObservation]
    public var hits: [HitEvent]
    public var segments: [ShotSegment3D]
    public var rallies: [RallyV2]
    public var score: [JSONValue]
    public var diagnostics: [String: JSONValue]
    public var processingTimeSeconds: Double

    public init(meta: VideoMeta, track: Track2D? = nil, camera: CameraModel? = nil, players: [PlayerObservation] = [],
                hits: [HitEvent] = [], segments: [ShotSegment3D] = [], rallies: [RallyV2] = [], score: [JSONValue] = [],
                diagnostics: [String: JSONValue] = [:], processingTimeSeconds: Double = 0) {
        self.meta = meta; self.track = track; self.camera = camera; self.players = players; self.hits = hits
        self.segments = segments; self.rallies = rallies; self.score = score; self.diagnostics = diagnostics
        self.processingTimeSeconds = processingTimeSeconds
    }

    enum CodingKeys: String, CodingKey {
        case meta, track, camera, players, hits, segments, rallies, score, diagnostics
        case processingTimeSeconds = "processing_time_seconds"
    }
}

// MARK: - Row helpers

public extension SIMD2 where Scalar == Double {
    /// From a JSON row / numpy row; missing entries become NaN.
    init(row: [Double]) { self.init(row.count > 0 ? row[0] : .nan, row.count > 1 ? row[1] : .nan) }
    var row: [Double] { [x, y] }
    var isFinite: Bool { x.isFinite && y.isFinite }
}

public extension SIMD3 where Scalar == Double {
    init(row: [Double]) {
        self.init(row.count > 0 ? row[0] : .nan, row.count > 1 ? row[1] : .nan, row.count > 2 ? row[2] : .nan)
    }
    var row: [Double] { [x, y, z] }
    var isFinite: Bool { x.isFinite && y.isFinite && z.isFinite }
    var xy: SIMD2<Double> { SIMD2(x, y) }
}
