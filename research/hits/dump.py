#!/usr/bin/env python
"""Dump one extracted match (.npz) to the JSON the Swift `hitseval` CLI reads.

Swift cannot read .npz, so this is the bridge.  It writes exactly the shapes
ShuttlCore's Codable types expect:

  track   -> Track2D   (t, xy[N][2], conf, visible, interpolated; raw_xy omitted:
                        the extractor never stored separate raw detections, so
                        track.xy IS the unsmoothed detection and preferRaw keeps it)
  meta    -> VideoMeta (fps, width, height, n_frames, timestamps)
  rallies -> per-rally dicts: cam_K/cam_R/cam_t (3x3/3x3/3), cam_ok, cam_conf,
             row_start, row_stop (indices into the track arrays), hit_frames
             (ground-truth video frame numbers, from the ShuttleSet labels)

Usage: dump.py <in.npz> <out.json>
"""
import json
import os
import sys

import numpy as np


def rows(a):
    """(N,2) float array -> [[x,y], ...] with NaN -> None (Swift's NaN convention)."""
    out = []
    for r in a:
        out.append([None if not np.isfinite(v) else float(v) for v in r])
    return out


def _plain(o):
    """numpy scalars (bool_, int64, float32) that json refuses."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError("not JSON serialisable: %r" % (o,))


def mat3(a):
    return [[float(v) for v in r] for r in np.asarray(a, dtype=float)]


# The detector reads player bboxes (for reach) and the wrist keypoints (COCO 9/10).
# player_* slots are [far, near] and PlayerObservation.side is taken from that
# index; keypoints are only emitted when all 17 are present and finite (a partial
# slot would just be noise for the reach test).
def players_of(d, rows_lo, rows_hi):
    box = d["player_box"][rows_lo:rows_hi]      # (n, 2, 4)
    kp = d["player_kp"][rows_lo:rows_hi]        # (n, 2, 17, 3)
    trk = d["player_track"][rows_lo:rows_hi]    # (n, 2)
    out = []
    sides = ("far", "near")
    for i in range(box.shape[0]):
        for s in (0, 1):
            b = box[i, s]
            if not np.isfinite(b).all():
                continue
            o = {
                "frame": int(rows_lo + i),
                "track_id": int(trk[i, s]),
                "bbox": [float(v) for v in b],
                "score": 1.0,
                "side": sides[s],
            }
            k = kp[i, s]
            if np.isfinite(k).all():
                o["keypoints"] = [[float(a), float(b2), float(c)] for a, b2, c in k]
            out.append(o)
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = np.load(src, allow_pickle=True)

    frame = d["frame"].astype(np.int64)
    t = d["t"].astype(float)
    xy = d["shuttle_xy"].astype(float)
    conf = d["shuttle_conf"].astype(float)
    visible = d["shuttle_visible"].astype(bool)
    interp = d["shuttle_interp"].astype(bool)

    n = frame.size
    # VideoMeta.timestamps is indexed by video frame number and the extractor's
    # t[i] is exactly frame[i]/fps, so the whole table is f/fps.
    n_frames = int(d["n_video_frames"])
    meta = {
        "fps": float(d["fps"]),
        "width": int(d["width"]),
        "height": int(d["height"]),
        "n_frames": n_frames,
        "timestamps": [i / float(d["fps"]) for i in range(n_frames)],
    }

    track = {
        "t": [float(v) for v in t],
        "xy": rows(xy),
        "conf": [0.0 if not np.isfinite(v) else float(v) for v in conf],
        "visible": [bool(v) for v in visible],
        "interpolated": [bool(v) for v in interp],
    }

    r_start = d["rally_row_start"].astype(np.int64)
    r_span = d["rally_span"].astype(np.int64)
    sset = d["rally_set"].astype(np.int64)
    sid = d["rally_id"].astype(np.int64)
    hit_start = d["rally_hit_start"].astype(np.int64)
    hit_frames = d["rally_hit_frames"].astype(np.int64)
    cam_ok = d["rally_cam_ok"].astype(bool)
    cam_conf = d["rally_cam_conf"].astype(float)
    cK, cR, cT = d["rally_cam_K"], d["rally_cam_R"], d["rally_cam_t"]

    rallies = []
    for i in range(sset.size):
        a, b = int(r_start[i]), int(r_start[i + 1])
        if b <= a:
            continue
        ok = bool(cam_ok[i]) and np.isfinite(cK[i]).all() and np.isfinite(cT[i]).all()
        rallies.append({
            "set": int(sset[i]),
            "rally": int(sid[i]),
            "row_start": a,
            "row_stop": b,
            "video_start": int(r_span[i][0]),
            "video_stop": int(r_span[i][1]),
            "hit_frames": [int(v) for v in hit_frames[hit_start[i]:hit_start[i + 1]]],
            "cam_ok": ok,
            "cam_conf": None if not np.isfinite(cam_conf[i]) else float(cam_conf[i]),
            "cam_K": mat3(cK[i]) if ok else None,
            "cam_R": mat3(cR[i]) if ok else None,
            "cam_t": [float(v) for v in cT[i]] if ok else None,
        })

    info = json.loads(str(d["info_json"][0]))
    players = players_of(d, 0, n)
    out = {
        "source": os.path.basename(src),
        "label_dir": os.path.basename(str(info.get("labels", "")).rstrip("/")),
        "frame": [int(v) for v in frame],
        "meta": meta,
        "track": track,
        "players": players,
        "rallies": rallies,
    }
    with open(dst, "w") as f:
        json.dump(out, f, default=_plain)
    print("%s -> %s  (rows %d, rallies %d, gt hits %d, players %d)"
          % (os.path.basename(src), os.path.basename(dst), n, len(rallies),
             sum(len(r["hit_frames"]) for r in rallies), len(players)))


if __name__ == "__main__":
    main()
