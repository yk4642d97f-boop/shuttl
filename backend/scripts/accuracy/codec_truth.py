#!/usr/bin/env python
"""Real-footage winner TRUTH for the 45-min endurance clip (E's H.264 / HEVC runs). The clip is
TrackNetV2 Test/match1 (11 rallies, 1280x720 30 fps) repeated in shuffled order, upscaled x1.5
to 1920x1080 @60, each rally followed by 8-20 s of its frozen last frame (tools/endurance/
driver ClipMaker.swift). The labels (corrected csv) end with the shuttle at rest on the floor.

Truth per source rally, from the LABELS + the run's calibrated camera (not from the app):
  last contact  = last direction reversal of the labelled track (hit_rescorer_data.reversal_labels)
  hitter side   = floor side of that contact pixel back-projected at z = 0.3 .. 2.8 m (must agree)
  landing       = the resting pixel (last labelled frame) on the floor: side + in/out margin (singles)
  winner        = hitter if it lands in on the other side, else the opponent; 'close' if |margin| < 0.15 m
Each app rally is assigned to the source whose frozen hold follows it (resting position after
the rally's end, nearest source end pixel; rally length breaks ties).

  .venv/bin/python -m scripts.accuracy.codec_truth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from app.services.tracking import court_model as cm  # noqa: E402

SCALE = 1.5                                 # 1280x720 -> 1920x1080 (AVMutableVideoComposition, uniform)
SRC_FPS, OUT_FPS = 30.0, 60.0


def sources() -> list[dict]:
    import hit_rescorer_data as D
    out = []
    for c in D.real_test_clips():
        if c["match"] != "match1":
            continue
        rows = np.loadtxt(str(c["labels"]), delimiter=",", skiprows=1, ndmin=2)
        n = int(rows[:, 0].max()) + 1
        xy, vis = D.load_tracknet_csv(c["labels"], n)
        out.append({"clip": c["clip"], "n": n, "xy": xy, "vis": vis, "corrected": c["corrected"],
                    "rev": D.reversal_labels(xy, vis, SRC_FPS)})
    return out


def truth(src: dict, cam) -> dict:
    """Winner from where the shuttle came to rest (labels + camera). A shuttle resting clearly
    away from the net (|x| >= 2 m) was last hit from the OTHER side (a net fault drops at the
    net); in / out from the singles lines. Not at rest, near the net or |margin| < 0.2 m: unclear."""
    xy, vis = src["xy"], src["vis"]
    idx = np.flatnonzero(vis)
    tail = xy[idx[-3:]]
    at_rest = len(idx) >= 3 and float(np.max(np.linalg.norm(tail - tail[-1], axis=1))) <= 3.0
    rest_px = xy[idx[-1]] * SCALE
    land = cam.pixel_to_floor(rest_px[None])[0]
    t = {"clip": src["clip"], "n": src["n"], "rest_px": rest_px, "land": land[:2], "at_rest": at_rest,
         "winner": "unknown", "end": "unknown", "margin": None, "close": True}
    if not at_rest or not np.isfinite(land).all() or abs(float(land[0])) < 2.0:
        return t
    lx, ly = float(land[0]), float(land[1])
    hitter = "far" if cm.side_of(lx) == "near" else "near"
    margin = float(cm.in_out_margin(lx, ly, doubles=False))
    t.update(hitter=hitter, margin=margin, end="in" if margin >= 0 else "out", close=abs(margin) < 0.2,
             winner=hitter if margin >= 0 else cm.side_of(lx))
    return t


def assign(rallies: list, track_xy: np.ndarray, truths: list[dict]) -> list[dict | None]:
    """Source of each app rally: resting pixel 0.5-1.5 s after its end vs each source's end pixel."""
    out = []
    n = len(track_xy)
    for r in rallies:
        a, b = min(n - 1, r.end_frame + int(0.5 * OUT_FPS)), min(n, r.end_frame + int(1.5 * OUT_FPS))
        seg = track_xy[a:b]
        seg = seg[np.isfinite(seg).all(1)]
        if len(seg) < 10:
            out.append(None)
            continue
        p = np.median(seg, 0)
        d = [(float(np.hypot(*(p - t["rest_px"]))), t) for t in truths]
        near = [x for x in d if x[0] < 25.0]
        if not near:
            out.append(None)
            continue
        dur = (r.end_frame - r.start_frame) / OUT_FPS
        best = min(near, key=lambda x: (abs(x[1]["n"] / SRC_FPS - dur), x[0])) if len(near) > 1 else near[0]
        out.append(best[1])
    return out


def accuracy(rallies: list, truths_of: list, thr: float = 0.6) -> dict:
    """Winner accuracy vs truth on rallies with a clear truth; coverage of known winners."""
    ok = n = known = conf_n = conf_ok = 0
    for r, t in zip(rallies, truths_of):
        if t is None or t["winner"] == "unknown" or t["close"]:
            continue
        n += 1
        known += r.winner in ("near", "far")
        ok += r.winner == t["winner"]
        if getattr(r, "outcome_confidence", 0.0) >= thr:
            conf_n += 1
            conf_ok += r.winner == t["winner"]
    return {"n": n, "acc": ok / max(n, 1), "known": known, "acc_known": ok / max(known, 1),
            "certain": conf_n, "acc_certain": conf_ok / max(conf_n, 1)}


def main(argv=None):
    from scripts.accuracy.codec_ab import load, rallies
    A = load("/tmp/accC/codec/h264.json")
    T = [truth(s, A["camera"]) for s in sources()]
    for t in T:
        print(f"  {t['clip']}: {t['n']:3d} f  at rest {t['at_rest']} land ({t['land'][0]:+.2f}, {t['land'][1]:+.2f}) "
              f"{t['end']:7s} margin {t['margin'] if t['margin'] is None else round(t['margin'], 2)}"
              f" -> winner {t['winner']}{' (unclear)' if t['close'] else ''}")
    for name, path in (("h264", "/tmp/accC/codec/h264.json"), ("hevc", "/tmp/accC/codec/hevc.json")):
        D = A if name == "h264" else load(path)
        rs = rallies(D)
        ts = assign(rs, D["track"].xy, T)
        a = accuracy(rs, ts)
        print(f"[{name}] rallies {len(rs)}, assigned {sum(t is not None for t in ts)}; truth-clear {a['n']}: winner acc "
              f"{a['acc']:.3f} (known {a['known']}, acc among known {a['acc_known']:.3f}; certain >= 0.6 {a['certain']}: {a['acc_certain']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
