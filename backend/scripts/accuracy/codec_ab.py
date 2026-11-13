#!/usr/bin/env python
"""Codec sensitivity of rally outcomes (E, coordinator.log 2026-09-28 20:30): the same match
encoded twice (H.264 / HEVC), the app's semantic-result.json of each (read-only copies).
Re-runs rally.segment_rallies (Python, current tree or --rally FILE) on each run's own track /
hits / segments / camera, pairs rallies across the two runs (start within 1 s) and reports
winner / end-reason agreement, how many are uncertain (< 0.6), and where the pairs differ.

  .venv/bin/python -m scripts.accuracy.codec_ab --a /tmp/accC/codec/h264.json --b /tmp/accC/codec/hevc.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import HitEvent, ShotSegment3D, Track2D, VideoMeta


def _arr(v, shape_cols=None):
    a = np.array([np.nan if x is None else x for x in v] if shape_cols is None else
                 [[np.nan if y is None else y for y in (row or [None] * shape_cols)] for row in v], float)
    return a


def load(path: str) -> dict:
    d = json.load(open(path))
    m = d["meta"]
    tr = d["track"]
    track = Track2D(t=_arr(tr["t"]), xy=_arr(tr["xy"], 2), conf=_arr(tr["conf"]),
                    visible=np.array(tr["visible"], bool), interpolated=np.array(tr["interpolated"], bool))
    if tr.get("raw_xy") is not None:
        track.raw_xy = _arr(tr["raw_xy"], 2)
    meta = VideoMeta(fps=float(m["fps"]), width=int(m["width"]), height=int(m["height"]),
                     n_frames=int(m["n_frames"]), timestamps=_arr(m["timestamps"]))
    c = d["camera"]
    cam = CameraModel(K=np.array(c["K"], float), R=np.array(c["R"], float), t=np.array(c["t"], float),
                      image_size=tuple(c["image_size"]), dist=np.array(c.get("dist") or [0] * 5, float),
                      reproj_rmse_px=float(c.get("reproj_rmse_px") or np.nan), confidence=float(c.get("confidence") or 0))
    hits = [HitEvent(frame=int(h["frame"]), t=float(h["t"]), xy_px=tuple(h["xy_px"]), hitter=h["hitter"],
                     confidence=float(h["confidence"]), player_track_id=h.get("player_track_id")) for h in d["hits"]]
    segs = []
    for s in d["segments"]:
        g = ShotSegment3D(start_frame=int(s["start_frame"]), end_frame=int(s["end_frame"]), t0=float(s["t0"]),
                          p0=np.array(s["p0"], float), v0=np.array(s["v0"], float),
                          drag_length=float(s["drag_length"]), rmse_px=float(s["rmse_px"] or np.nan),
                          frames=np.array(s["frames"], int), positions=_arr(s["positions"], 3),
                          net_crossing_height=s.get("net_crossing_height"), confidence=float(s["confidence"]),
                          net_crossing_y=s.get("net_crossing_y"), end_reason=s.get("end_reason") or "unknown",
                          hitter_feet_xy=tuple(s["hitter_feet_xy"]) if s.get("hitter_feet_xy") else None)
        for k in ("landing_xy", "landing_t", "hit_frame", "landing_sigma_m"):
            if s.get(k) is not None:
                setattr(g, k, tuple(s[k]) if k == "landing_xy" else s[k])
        segs.append(g)
    return {"track": track, "meta": meta, "camera": cam, "hits": hits, "segs": segs, "app_rallies": d["rallies"]}


def pair(ra: list, rb: list, fps: float, tol_s: float = 1.0) -> list[tuple[int, int]]:
    """Greedy one-to-one pairs of rallies whose start frames are within tol_s."""
    c = sorted((abs(a.start_frame - b.start_frame), i, j) for i, a in enumerate(ra) for j, b in enumerate(rb)
               if abs(a.start_frame - b.start_frame) <= tol_s * fps)
    ua, ub, out = set(), set(), []
    for _, i, j in c:
        if i not in ua and j not in ub:
            ua.add(i), ub.add(j), out.append((i, j))
    return sorted(out)


def rallies(d: dict, mod=R, cfg=None) -> list:
    return mod.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], config=cfg)


def report(name: str, ra: list, rb: list, fps: float, thr: float = 0.6) -> dict:
    P = pair(ra, rb, fps)
    agree = sum(ra[i].winner == rb[j].winner for i, j in P)
    both_known = [(i, j) for i, j in P if ra[i].winner in R.SIDES and rb[j].winner in R.SIDES]
    flips = sum(ra[i].winner != rb[j].winner for i, j in both_known)
    unc_a = sum(getattr(r, "outcome_confidence", 0.0) < thr for r in ra)
    unc_b = sum(getattr(r, "outcome_confidence", 0.0) < thr for r in rb)
    cert = [(i, j) for i, j in P if min(ra[i].outcome_confidence, rb[j].outcome_confidence) >= thr]
    cert_ok = sum(ra[i].winner == rb[j].winner for i, j in cert)
    endd = Counter((ra[i].end_reason, rb[j].end_reason) for i, j in P if ra[i].end_reason != rb[j].end_reason)
    sa = R.score_match(ra)
    sb = R.score_match(rb)
    fin = lambda s: (s[-1]["games"], s[-1]["score"]) if s else None      # noqa: E731
    print(f"[{name}] rallies {len(ra)} / {len(rb)}, pairs {len(P)}: winner agree {agree}/{len(P)} "
          f"(near<->far flips {flips}), uncertain (<{thr}) {unc_a}/{len(ra)} and {unc_b}/{len(rb)}; "
          f"both certain {len(cert)}: agree {cert_ok}")
    print(f"      end_reason diffs {dict(endd.most_common(6))}")
    print(f"      final games/score A {fin(sa)}  B {fin(sb)}")
    return {"pairs": P, "agree": agree, "cert": (len(cert), cert_ok)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/tmp/accC/codec/h264.json")
    ap.add_argument("--b", default="/tmp/accC/codec/hevc.json")
    ap.add_argument("--rally", default=None, help="alternative rally.py file (before/after)")
    ap.add_argument("--ab", nargs="*", default=[])
    a = ap.parse_args(argv)
    A, B = load(a.a), load(a.b)
    fps = A["meta"].fps
    mod = R
    if a.rally:
        spec = importlib.util.spec_from_file_location("rally_alt", a.rally)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rally_alt"] = mod
        spec.loader.exec_module(mod)
    cfg = None
    if a.ab:
        from scripts.accuracy.cfg_diff import parse_cfg
        cfg = parse_cfg(a.ab)
    ra, rb = rallies(A, mod, cfg), rallies(B, mod, cfg)
    # the app's own (Swift) rallies, for reference
    same = [(x.start_frame, x.end_frame, x.winner) == (y["start_frame"], y["end_frame"], y["winner"])
            for x, y in zip(ra, A["app_rallies"])]
    print(f"python rally.py vs app rallies (A): {sum(same)}/{len(A['app_rallies'])} identical (start, end, winner)"
          f" [{len(ra)} vs {len(A['app_rallies'])}]")
    report(a.rally or "current", ra, rb, fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def load_players(path: str) -> list:
    """ShuttlAnalysis PlayersCodec (players-comparable.bin, v1/v2, little-endian) -> PlayerObservation."""
    import struct
    from app.services.tracking.data_types import PlayerObservation
    b = open(path, "rb").read()
    assert b[:4] == b"SHPO", "not a players file"
    o = 8
    fp_len = struct.unpack_from("<I", b, o)[0]; o += 4 + fp_len
    n = struct.unpack_from("<Q", b, o)[0]; o += 8
    side = {0: "unknown", 1: "near", 2: "far"}
    out = []
    for _ in range(n):
        frame, tid = struct.unpack_from("<qq", b, o); o += 16
        bbox = np.array(struct.unpack_from("<4d", b, o)); o += 32
        score = struct.unpack_from("<d", b, o)[0]; o += 8
        sc, has, nk = struct.unpack_from("<BBH", b, o); o += 4
        foot = kps = None
        if has & 1:
            foot = np.array(struct.unpack_from("<2d", b, o)); o += 16
        if has & 2:
            kps = np.array(struct.unpack_from(f"<{3 * nk}d", b, o)).reshape(nk, 3); o += 24 * nk
        out.append(PlayerObservation(frame=int(frame), track_id=int(tid), bbox=bbox, score=float(score),
                                     side=side[sc], foot_world=foot, keypoints=kps))
    assert b[o:o + 4] == b"OPHS" and struct.unpack_from("<Q", b, o + 4)[0] == n, "bad trailer"
    return out
