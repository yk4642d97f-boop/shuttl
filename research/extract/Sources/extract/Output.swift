import Foundation
import ShuttlCore
import simd

enum Output {
    static func write(_ rallies: [RallyOut], matchCourt: CourtResult, meta: VideoMeta, args: Args, detector: String,
                      to url: URL) throws {
        var z = NPZWriter()
        let N = rallies.reduce(0) { $0 + $1.n }, R = rallies.count
        var frame: [Int64] = [], t: [Double] = [], rallyIndex: [Int32] = []
        var cand: [Float] = [], xy: [Float] = [], vis: [Bool] = [], interp: [Bool] = [], conf: [Float] = []
        var pStart: [Int64] = [0], pKp: [Float] = [], pBox: [Float] = [], pScore: [Float] = [], pSrc: [Int8] = []
        var pRow: [Int64] = []
        var plPerson: [Int64] = [], plTrack: [Int32] = [], plBox: [Float] = [], plKp: [Float] = [], plFoot: [Float] = []
        var rSet: [Int32] = [], rId: [Int32] = [], rSpan: [Int64] = [], rRow: [Int64] = [0], rHits: [Int64] = [], rHitStart: [Int64] = [0]
        var rErr: [String] = [], rCamSrc: [Int8] = [], rCamOk: [Bool] = [], rCamDiff: [Double] = [], rCamReason: [String] = []
        var rCamConf: [Double] = [], rK: [Double] = [], rR: [Double] = [], rT: [Double] = [], rH: [Double] = [], rTiming: [Double] = []
        let timingKeys = ["calibrate", "detect", "link", "pose", "track", "total"]

        for (ri, r) in rallies.enumerated() {
            let personBase = Int64(pScore.count), rowBase = Int64(frame.count)
            frame += (0..<r.n).map { Int64(r.spec.start + $0) }
            t += r.t
            rallyIndex += [Int32](repeating: Int32(ri), count: r.n)
            cand += r.candidates; xy += r.shuttle; vis += r.visible; interp += r.interpolated; conf += r.shuttleConf
            pKp += r.personKp; pBox += r.personBox; pScore += r.personScore; pSrc += r.personSource
            if r.n > 0 {
                pStart += r.personStart.dropFirst().map { personBase + Int64($0) }
                for k in 0..<r.n {
                    pRow += [Int64](repeating: rowBase + Int64(k), count: Int(r.personStart[k + 1] - r.personStart[k]))
                }
            }
            plPerson += r.playerPerson.map { $0 < 0 ? -1 : personBase + Int64($0) }
            plTrack += r.playerTrack; plBox += r.playerBox; plKp += r.playerKp; plFoot += r.playerFoot
            rSet.append(Int32(r.spec.set)); rId.append(Int32(r.spec.rally))
            rSpan += [Int64(r.spec.start), Int64(r.spec.stop)]
            rRow.append(Int64(frame.count))
            rHits += r.spec.hitFrames.map(Int64.init); rHitStart.append(Int64(rHits.count))
            rErr.append(r.error ?? "")
            rCamSrc.append(r.cameraSource)
            rCamOk.append(r.court?.ok ?? false)
            rCamDiff.append(r.cameraDiffPx)
            rCamReason.append(r.court?.reason ?? "")
            rCamConf.append(r.court?.confidence ?? .nan)
            let cam = r.court?.camera
            rK += matrix(cam?.K); rR += matrix(cam?.R); rH += matrix(cam?.floorHomography())
            rT += cam.map { [$0.t.x, $0.t.y, $0.t.z] } ?? [.nan, .nan, .nan]
            rTiming += timingKeys.map { r.timing[$0] ?? .nan }
        }

        z.add("frame", frame, [N]); z.add("t", t, [N]); z.add("rally_index", rallyIndex, [N])
        z.add("shuttle_cand", cand, [N, 3, 3]); z.add("shuttle_xy", xy, [N, 2]); z.add("shuttle_visible", vis, [N])
        z.add("shuttle_interp", interp, [N]); z.add("shuttle_conf", conf, [N])
        let P = pScore.count
        z.add("person_start", pStart, [N + 1]); z.add("person_row", pRow, [P]); z.add("person_kp", pKp, [P, 19, 3])
        z.add("person_box", pBox, [P, 4]); z.add("person_score", pScore, [P]); z.add("person_source", pSrc, [P])
        z.add("player_person", plPerson, [N, 2]); z.add("player_track", plTrack, [N, 2]); z.add("player_box", plBox, [N, 2, 4])
        z.add("player_kp", plKp, [N, 2, 17, 3]); z.add("player_foot_world", plFoot, [N, 2, 2])
        z.add("rally_set", rSet, [R]); z.add("rally_id", rId, [R]); z.add("rally_span", rSpan, [R, 2])
        z.add("rally_row_start", rRow, [R + 1]); z.add("rally_hit_frames", rHits, [rHits.count]); z.add("rally_hit_start", rHitStart, [R + 1])
        z.add("rally_error", strings: rErr); z.add("rally_cam_source", rCamSrc, [R]); z.add("rally_cam_ok", rCamOk, [R])
        z.add("rally_cam_diff_px", rCamDiff, [R]); z.add("rally_cam_reason", strings: rCamReason); z.add("rally_cam_conf", rCamConf, [R])
        z.add("rally_cam_K", rK, [R, 3, 3]); z.add("rally_cam_R", rR, [R, 3, 3]); z.add("rally_cam_t", rT, [R, 3])
        z.add("rally_cam_H", rH, [R, 3, 3]); z.add("rally_timing", rTiming, [R, timingKeys.count])
        z.add("rally_timing_keys", strings: timingKeys)

        let mc = matchCourt.camera
        z.add("cam_ok", [mc != nil], []); z.add("cam_K", matrix(mc?.K), [3, 3]); z.add("cam_R", matrix(mc?.R), [3, 3])
        z.add("cam_t", mc.map { [$0.t.x, $0.t.y, $0.t.z] } ?? [.nan, .nan, .nan], [3])
        z.add("cam_dist", mc.map { $0.dist + [Double](repeating: 0, count: max(0, 5 - $0.dist.count)) }.map { Array($0.prefix(5)) } ?? [Double](repeating: .nan, count: 5), [5])
        z.add("cam_H", matrix(mc?.floorHomography()), [3, 3])
        z.add("cam_frames", matchCourt.frames.map(Int64.init), [matchCourt.frames.count])
        z.add("joint_names", strings: Joints.names)
        z.add("fps", [meta.fps], []); z.add("width", [Int32(meta.width)], []); z.add("height", [Int32(meta.height)], [])
        z.add("n_video_frames", [Int64(meta.nFrames)], [])
        let info: [String: Any] = [
            "video": args.video, "labels": args.labels, "detector": detector, "pose_step": args.poseStep,
            "match_cam_reason": matchCourt.reason, "match_cam_confidence": matchCourt.confidence ?? NSNull(),
            "match_cam_rmse": matchCourt.rmse ?? NSNull(), "match_cam_coverage": matchCourt.coverage ?? NSNull(),
            "created": ISO8601DateFormatter().string(from: Date()), "format": "shuttl-extract v1",
        ]
        let json = String(data: try JSONSerialization.data(withJSONObject: info, options: [.sortedKeys]), encoding: .utf8)!
        z.add("info_json", strings: [json])
        try z.write(to: url)
    }

    static func matrix(_ m: simd_double3x3?) -> [Double] {
        guard let m else { return [Double](repeating: .nan, count: 9) }
        return (0..<3).flatMap { r in (0..<3).map { c in m[c, r] } }
    }
}
