#!/usr/bin/env python
"""Features of the trailing-trim candidates (last contact of a multi-contact group
after the leading / no-flight trims), for real last contacts vs spurious trailing
kinks. Physical cues: does the incoming fit, propagated past the contact, still
explain the 2-D track after it (a kink on the same flight) or does the track leave
it (a new flight)? Velocity reversal / |dv| between the incoming fit at the
contact and the contact's own fit; the incoming fit's floor time vs the contact.

  sim dev:  SHUTTL_ACC_HASH=789165f7ee .venv/bin/python -m scripts.accuracy.trail_features --src sim
  test gen: .venv/bin/python -m scripts.accuracy.trail_features --src test --fps 60
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from app.services.tracking import rally as R


def features(track, hits, segs, meta, camera, cfg=None, horizon_s=0.5):
    cfg = cfg or R.RallyConfig()
    mo = R._Motion(track, float(meta.height), cfg)
    hs = R._clean_hits(hits, mo.n)
    hs = [h if h.t is not None and math.isfinite(h.t) else R.replace(h, t=float(mo.t[h.frame])) for h in hs]
    sg = R._match_segments(hs, list(segs or []), mo.t, cfg)
    hs, sg, conts = R._merge_same_flight(hs, sg, mo.t, cfg)
    groups, _ = R._group_hits(hs, sg, mo, cfg, conts)
    hs, _ = R._decode_sides(hs, sg, conts, groups, mo.t, cfg)
    out = []
    for g in groups:
        while g and R._leading_nonhit(g, hs, sg, mo, True, cfg):
            g = g[1:]
        while len(g) > 1 and sg[g[-1]] is None and not R._is_flight(hs[g[-1]], None, mo, cfg):
            g = g[:-1]
        if len(g) < 2:
            continue
        h, hp = hs[g[-1]], hs[g[-2]]
        inc = conts[g[-2]] if conts[g[-2]] is not None else sg[g[-2]]
        own = sg[g[-1]]
        f = dict(frame=h.frame, conf=float(h.confidence or 0.0), moving=R._moving_after(mo, h),
                 gap=float(h.t - hp.t), own=own is not None, inc=inc is not None)
        if inc is not None:
            st = R._state_at(inc, float(h.t))
            f["z_in"], f["vx_in"], f["x_in"] = st[2], st[3], st[0]
            nh = inc.net_crossing_height
            f["inc_net"] = bool(getattr(inc, "end_reason", None) == "net" or (nh is not None and nh < R.cm.NET_HEIGHT_CENTRE))
            f["own_w"] = R._fit_weight(own, R.RallyConfig()) if own is not None else None
            f["own_x0"] = float(own.p0[0]) if own is not None else None
            f["d3"] = float(np.linalg.norm(np.asarray(own.p0, float) - np.asarray(st[:3]))) if own is not None else None
            ft = R._floor_time(inc, float(h.t) + 2.0)
            f["floor_dt"] = (ft - float(h.t)) if ft is not None else None
            f["t_cov"] = float(mo.t[min(max(int(inc.end_frame), 0), mo.n - 1)]) - float(h.t)
            if own is not None:
                d = [st[3 + k] - float(own.v0[k]) for k in range(3)]
                f["dv"] = math.sqrt(sum(x * x for x in d))
                f["rev"] = st[3] * float(own.v0[0]) < 0
            # 2-D residual of the track after the contact against both fits
            f1 = min(mo.n, mo.frame_at(float(h.t) + horizon_s) + 1)
            fr = [k for k in range(h.frame + 1, f1) if mo.fin[k]]
            if camera is not None and len(fr) >= 3:
                res = {}
                for name, s in (("inc", inc), ("own", own)):
                    if s is None:
                        continue
                    P = np.array([R._state_at(s, float(mo.t[k]))[:3] for k in fr])
                    uv = camera.project(P)
                    res[name] = float(np.median(np.linalg.norm(uv - mo.xy[fr], axis=1))) / mo.h
                f["res_inc"], f["res_own"] = res.get("inc"), res.get("own")
                f["n_after"] = len(fr)
        f["cos2d"], f["cosfit"] = _dir_cues(mo, h, inc, camera)
        f["launch"] = R._launches(h, own, inc, cfg)
        for lag in (0.1, 0.2):
            f[f"dep{int(lag*1000)}"] = _departure(mo, h, inc, camera, lag)
        out.append(f)
    return out


def _dir_cues(mo, h, inc, camera, win=0.15):
    """cos2d: image velocity over `win` s before vs after the contact (a racket
    contact turns the image motion around, < 0). cosfit: observed displacement
    after the contact vs the incoming fit's projected displacement (same
    flight ~ +1)."""
    f = h.frame
    b = [k for k in range(mo.frame_at(h.t - win), f + 1) if mo.fin[k]]
    a = [k for k in range(f, min(mo.n, mo.frame_at(h.t + win) + 1)) if mo.fin[k] and not mo.rest[k]]
    c2 = cf = None
    if len(b) >= 2 and len(a) >= 2:
        vb, va = mo.xy[b[-1]] - mo.xy[b[0]], mo.xy[a[-1]] - mo.xy[a[0]]
        nb, na = np.linalg.norm(vb), np.linalg.norm(va)
        if nb > 0.005 * mo.h and na > 0.005 * mo.h:
            c2 = float(vb @ va / (nb * na))
    if inc is not None and camera is not None and len(a) >= 2:
        p = camera.project(np.array([R._state_at(inc, float(mo.t[k]))[:3] for k in (a[0], a[-1])]))
        dp, do = p[1] - p[0], mo.xy[a[-1]] - mo.xy[a[0]]
        npd, nd = np.linalg.norm(dp), np.linalg.norm(do)
        if npd > 0.005 * mo.h and nd > 0.005 * mo.h:
            cf = float(dp @ do / (npd * nd))
    return c2, cf


def _fmt(f):
    keys = ("conf", "moving", "gap", "z_in", "x_in", "inc_net", "own_w", "own_x0", "floor_dt", "t_cov", "dv", "rev",
            "res_inc", "res_own", "n_after", "cos2d", "cosfit", "launch", "d3", "dep100", "dep200")
    return " ".join(f"{k} {f[k]:.3f}" if isinstance(f.get(k), float) else f"{k} {f.get(k)}" for k in keys)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=("sim", "test"), default="sim")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dump", default="")
    a = ap.parse_args(argv)
    rows = []
    if a.src == "test":
        from scripts.accuracy.seg_quality_ab import inputs
        for s in range(min(a.n, 20)):
            c, hits, src, segs = inputs(s, a.fps)
            for f in features(c.track, hits, segs, c.meta, c.camera):
                f["real"], f["style"] = True, "test"
                rows.append(f)
    else:
        from scripts.accuracy.common import load_samples, sim_args
        for style in ("behind", "random"):
            for d in load_samples(sim_args(style, "1920x1080", a.fps, 5), list(range(a.seed, a.seed + a.n)), 4):
                tf = np.array([h["frame"] for h in d["true_hits"]])
                for f in features(d["track"], d["hits"], d["segs"], d["meta"], d["camera"]):
                    fr = f["frame"]
                    real = bool(len(tf) and np.min(np.abs(tf - fr)) <= 3)
                    if not real:
                        ins = [t for t in d["true_rallies"] if t.start_frame - 3 <= fr <= t.end_frame + 3]
                        if not ins or fr <= d["true_shots"][ins[0].first_shot + ins[0].n_shots - 1].frame + 3:
                            continue
                        f["after_last"] = fr - d["true_shots"][ins[0].first_shot + ins[0].n_shots - 1].frame
                        f["tend"] = ins[0].end_reason
                    f["real"], f["style"], f["seed"] = real, style, d["seed"]
                    rows.append(f)
    if a.dump:
        import json
        json.dump(rows, open(a.dump, "w"), default=float)
    for f in rows:
        if a.src == "test" or not f["real"] or f["conf"] < 0.8:
            print(("REAL " if f["real"] else "SPUR ") + f.get("style", "") + " " + _fmt(f))
    print(f"rows {len(rows)} real {sum(r['real'] for r in rows)} spurious {sum(not r['real'] for r in rows)}")



def _departure(mo, h, inc, camera, lag):
    """|projected incoming fit - observed track| / H at the moving sample closest to
    contact + lag (within the observed flight: before the first rest)."""
    if inc is None or camera is None:
        return None
    f_hi = min(mo.n, mo.frame_at(h.t + lag + 0.1) + 1)
    fr = mo.first_rest_after(h.frame, f_hi)
    ks = [k for k in range(h.frame + 1, fr if fr is not None else f_hi) if mo.fin[k]]
    if not ks:
        return None
    k = min(ks, key=lambda k: abs(float(mo.t[k]) - (h.t + lag)))
    if abs(float(mo.t[k]) - (h.t + lag)) > 0.06:
        return None
    uv = camera.project(np.array([R._state_at(inc, float(mo.t[k]))[:3]]))[0]
    return float(np.linalg.norm(uv - mo.xy[k])) / mo.h


if __name__ == "__main__":
    main()
