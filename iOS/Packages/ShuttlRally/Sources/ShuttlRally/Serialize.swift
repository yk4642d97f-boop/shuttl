import Foundation
import ShuttlCore
import simd

/// Port of backend/app/services/tracking/serialize.py: MatchAnalysis → the
/// JSON-safe dict (schema version 2) the results screen reads. Field names,
/// rounding (Python `round`, numpy `np.round` for the track) and null
/// handling match Python field-for-field. Pixel positions are normalised to
/// 0–1 of the upright frame; world positions are metres in the court frame.
public enum RallySerialization {
    public static let schemaVersion = 2

    /// serialize.legacy_shot_type.
    public static func legacyShotType(_ t: ShotType) -> ShotType { t.legacy }

    // MARK: summary

    /// serialize.summarize(rallies).
    public static func summarize(_ rallies: [RallyV2]) -> JSONValue {
        let shots = rallies.flatMap(\.shots)
        let speeds = shots.compactMap { $0.segment.map { Py.norm3($0.v0) * 3.6 } }
        let smash = shots.compactMap { s -> Double? in
            guard let seg = s.segment, s.shotType == .smash || s.shotType == .kill else { return nil }
            return Py.norm3(seg.v0) * 3.6
        }
        var counts: [String: JSONValue] = [:]
        for s in shots { counts[s.shotType.rawValue] = .number((counts[s.shotType.rawValue]?.double ?? 0) + 1) }
        var ends: [String: JSONValue] = [:]
        for r in rallies { ends[r.endReason.rawValue] = .number((ends[r.endReason.rawValue]?.double ?? 0) + 1) }
        return .object([
            "total_rallies": .int(rallies.count), "total_shots": .int(shots.count),
            "max_speed_kmh": Py.f(Py.max(speeds, default: 0.0), 1),
            "avg_speed_kmh": Py.f(speeds.isEmpty ? 0.0 : Py.npMean(speeds), 1),
            "max_smash_kmh": Py.f(Py.max(smash, default: 0.0), 1),
            "avg_smash_kmh": Py.f(smash.isEmpty ? 0.0 : Py.npMean(smash), 1),
            "shot_type_counts": .object(counts), "end_reason_counts": .object(ends),
            "avg_rally_shots": rallies.isEmpty ? .number(0.0)
                : Py.f(Double(shots.count) / Double(rallies.count), 1),
        ])
    }

    // MARK: per rally / shot / flight

    private static func xy(_ p: SIMD2<Double>?, _ nd: Int = 3) -> JSONValue {
        guard let p else { return .null }
        return .array([Py.f(p.x, nd), Py.f(p.y, nd)])
    }

    private static func xyz(_ p: SIMD3<Double>, _ nd: Int = 3) -> JSONValue {
        .array([Py.f(p.x, nd), Py.f(p.y, nd), Py.f(p.z, nd)])
    }

    /// serialize._segment (flight dict); `camera` nil → path_norm null.
    public static func segmentDict(_ seg: ShotSegment3D?, width: Int, height: Int,
                                   camera: CameraModel?) -> JSONValue {
        guard let seg else { return .null }
        let w = Double(width), h = Double(height)
        var apex: Double? = nil
        if !seg.positions.isEmpty {                     // np.nanmax (all NaN → NaN)
            let zs = seg.positions.map(\.z).filter { !$0.isNaN }
            apex = zs.max() ?? .nan
        }
        var path: JSONValue = .null
        if let camera, !seg.positions.isEmpty {
            path = .array(seg.positions.map { p in
                let uv = camera.project(p)
                return .array([Py.f(uv.x / w, 4), Py.f(uv.y / h, 4)])
            })
        }
        return .object([
            "start_frame": .int(seg.startFrame), "end_frame": .int(seg.endFrame),
            "speed0_kmh": Py.f(Py.norm3(seg.v0) * 3.6, 1),
            "p0": xyz(seg.p0), "v0": xyz(seg.v0),
            "apex_m": Py.f(apex, 2),
            "net_clearance_m": Py.f(seg.netCrossingHeight.map { $0 - CourtModel.netHeightCentre }, 2),
            "net_crossing_y": Py.f(seg.netCrossingY, 2), "end_reason": .str(seg.endReason.rawValue),
            "landing_xy": xy(seg.landingXY), "landing_t": Py.f(seg.landingT),
            "rmse_px": Py.f(seg.rmsePx, 2), "confidence": Py.f(seg.confidence, 2),
            "frames": .array(seg.frames.map { .int($0) }),
            "world": .array(seg.positions.map { xyz($0) }),
            "path_norm": path,
        ])
    }

    /// serialize._shot.
    public static func shotDict(_ s: ShotEventV2, width: Int, height: Int, camera: CameraModel?) -> JSONValue {
        let w = Double(width), h = Double(height)
        var d: [String: JSONValue] = [
            "frame": .int(s.hit.frame), "t": Py.f(s.hit.t),
            "hitter": .str(s.hit.hitter.rawValue), "hit_confidence": Py.f(s.hit.confidence, 2),
            "xy_norm": .array([Py.f(s.hit.xyPx.x / w, 4), Py.f(s.hit.xyPx.y / h, 4)]),
            "shot_type": .str(s.shotType.rawValue), "legacy_shot_type": .str(s.shotType.legacy.rawValue),
            "shot_type_confidence": Py.f(s.shotTypeConfidence, 2),
            "hitter_pos": xy(s.hitterPosWorld), "hit_height_m": Py.f(s.hitHeightM, 2),
            "stroke_side": .str(s.strokeSide.rawValue),
            "flight": segmentDict(s.segment, width: width, height: height, camera: camera),
        ]
        // serialize._shot: "hit_source" only for an inferred hit (e.g. F4 "gap_contact")
        if s.hit.isInferred { d["hit_source"] = .str(s.hit.source) }
        return .object(d)
    }

    /// serialize._rally.
    public static func rallyDict(_ r: RallyV2, width: Int, height: Int, camera: CameraModel?) -> JSONValue {
        .object([
            "start_frame": .int(r.startFrame), "end_frame": .int(r.endFrame),
            "end_reason": .str(r.endReason.rawValue), "winner": .str(r.winner.rawValue),
            "server": .str(r.server.rawValue),
            "landing_xy": xy(r.landingXY), "in_out_margin_m": Py.f(r.inOutMarginM),
            "confidence": Py.f(r.confidence, 2),
            "outcome_confidence": Py.f(r.outcomeConfidence, 2), "outcome_source": .str(r.outcomeSource),
            "shots": .array(r.shots.map { shotDict($0, width: width, height: height, camera: camera) }),
        ])
    }

    /// serialize._track_runs: contiguous runs of known positions.
    public static func trackRuns(_ track: Track2D?, width: Int, height: Int) -> JSONValue {
        guard let track else { return .array([]) }
        let w = Double(width), h = Double(height)
        let ok = track.xy.map(\.isFinite)
        var runs: [JSONValue] = []
        var i = 0
        let n = track.n
        while i < n {
            if !ok[i] { i += 1; continue }
            var j = i
            while j < n && ok[j] { j += 1 }
            // np.round(seg / [w, h], 4): multiply, round half to even, divide
            let xy: [JSONValue] = (i..<j).map { k in
                let p = track.xy[k]
                return .array([.number(Num.roundHalfEven(p.x / w, digits: 4)),
                               .number(Num.roundHalfEven(p.y / h, digits: 4))])
            }
            let interp: [JSONValue] = (i..<j).compactMap { k in
                k < track.interpolated.count && track.interpolated[k] ? .int(k - i) : nil
            }
            runs.append(.object(["start_frame": .int(i), "t0": Py.f(track.t[i]),
                                 "xy": .array(xy), "interp": .array(interp)]))
            i = j
        }
        return .array(runs)
    }

    /// CameraModel.to_dict() as serialized (floats rounded to 6 digits by _jsonable).
    public static func cameraDict(_ c: CameraModel) -> JSONValue {
        let rows: (simd_double3x3) -> JSONValue = { m in
            .array(m.rowMajor.map { .array($0.map { Py.f($0, 6) }) })
        }
        return .object([
            "K": rows(c.K), "R": rows(c.R), "t": .array([c.t.x, c.t.y, c.t.z].map { Py.f($0, 6) }),
            "dist": .array(c.dist.map { Py.f($0, 6) }),
            "image_size": .array([.int(c.imageSize.width), .int(c.imageSize.height)]),
            "reproj_rmse_px": Py.f(c.reprojRmsePx, 6), "confidence": Py.f(c.confidence, 6),
        ])
    }

    /// serialize._jsonable on free-form JSON: every number rounded to 6 digits.
    static func jsonable(_ v: JSONValue) -> JSONValue {
        switch v {
        case .number(let x): return Py.f(x, 6)
        case .array(let a): return .array(a.map(jsonable))
        case .object(let o): return .object(o.mapValues(jsonable))
        default: return v
        }
    }

    /// serialize.to_dict(MatchAnalysis). `movement` (players.movement_stats) is
    /// owned by the players stage: pass its already-built dict, or nil to omit
    /// the key (Python omits it when there are no players).
    public static func toDict(_ res: MatchAnalysis, movement: JSONValue? = nil) -> JSONValue {
        let w = res.meta.width != 0 ? res.meta.width : 1
        let h = res.meta.height != 0 ? res.meta.height : 1
        var out: [String: JSONValue] = [
            "schema_version": .int(schemaVersion),
            "video": .object(["fps": Py.f(res.meta.fps, 3), "width": .int(res.meta.width),
                              "height": .int(res.meta.height), "frames": .int(res.meta.nFrames)]),
            "camera": res.camera.map(cameraDict) ?? .null,
            "calibrated": .bool(res.camera != nil),
            "summary": summarize(res.rallies),
            "rallies": .array(res.rallies.map { rallyDict($0, width: w, height: h, camera: res.camera) }),
            "score": .array(res.score),
            "track": trackRuns(res.track, width: w, height: h),
            "diagnostics": .object(res.diagnostics),
            "processing_time_seconds": Py.f(res.processingTimeSeconds, 2),
        ]
        if let movement { out["movement"] = movement }
        return jsonable(.object(out))
    }
}

/// serialize.summarize — free-function spelling of the public API.
public func summarize(_ rallies: [RallyV2]) -> JSONValue { RallySerialization.summarize(rallies) }
