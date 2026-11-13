#!/usr/bin/env python
"""
Rally error analysis on the sim e2e chain (cached upstream, see common.py).

For every wrong rally (unmatched prediction, unmatched truth, wrong end_reason,
wrong winner, landing error > 1 m) it dumps true vs predicted start/end,
end_reason, winner, landing and the evidence rally.py used (outcome_source,
confidences, last-hit match), assigns a failure bucket and prints the counts.
Also prints the outcome_confidence reliability table.

  .venv/bin/python -m scripts.accuracy.rally_errors --style behind --seed 0 --n 40
  .venv/bin/python -m scripts.accuracy.rally_errors --style random --dump 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def _iou(a, b):
    return M._interval_iou(a, b)


def match(pred, true, thr=0.5):
    P = [(r.start_frame, r.end_frame) for r in pred]
    T = [(r.start_frame, r.end_frame) for r in true]
    cand = sorted(((_iou(p, t), i, j) for i, p in enumerate(P) for j, t in enumerate(T)), reverse=True)
    up, ut, pairs = set(), set(), []
    for iou, i, j in cand:
        if iou < thr:
            break
        if i in up or j in ut:
            continue
        up.add(i), ut.add(j), pairs.append((i, j))
    return pairs


def _near(frames, f, tol=3):
    return any(abs(int(g) - int(f)) <= tol for g in frames)


def analyse(d: dict, cfg: R.RallyConfig | None = None) -> tuple[list[dict], dict]:
    """Rallies of one sample -> (error records, per-sample counts)."""
    pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                             players=d["players"], config=cfg)
    true = d["true_rallies"]
    shots = d["true_shots"]
    pairs = match(pred, true)
    mp = {i: j for i, j in pairs}
    mt = {j: i for i, j in pairs}
    true_hit_frames = [h["frame"] for h in d["true_hits"]]
    recs = []
    rel = []
    for i, r in enumerate(pred):
        pl = [s.hit.frame for s in r.shots]
        rec = {"seed": d["seed"], "pred": i, "p_start": r.start_frame, "p_end": r.end_frame,
               "p_end_reason": r.end_reason, "p_winner": r.winner,
               "p_landing": None if r.landing_xy is None else [round(v, 2) for v in r.landing_xy],
               "p_margin": None if r.in_out_margin_m is None else round(r.in_out_margin_m, 3),
               "source": getattr(r, "outcome_source", "?"),
               "oconf": round(getattr(r, "outcome_confidence", 0.0), 3),
               "sconf": round(getattr(r, "segmentation_confidence", 0.0), 3),
               "n_pred_shots": len(pl), "serve_missed": bool(getattr(r, "serve_missed", False))}
        ov = [j for j, t in enumerate(true) if min(r.end_frame, t.end_frame) >= max(r.start_frame, t.start_frame)]
        if i not in mp:
            if not ov:
                b = "fp_between_rallies"          # feed / knock-up / spurious hits in dead time
            elif len(ov) >= 2:
                b = "merged_rallies"
            elif any(k != i and min(p.end_frame, true[ov[0]].end_frame) >= max(p.start_frame, true[ov[0]].start_frame)
                     for k, p in enumerate(pred)):
                b = "split_rally"
            else:
                b = "boundary_off"
            rec.update(bucket=b, true=ov)
            recs.append(rec)
            continue
        j = mp[i]
        t = true[j]
        tshots = shots[t.first_shot:t.first_shot + t.n_shots]
        t_last = tshots[-1].frame
        last_ok = bool(pl) and abs(pl[-1] - t_last) <= 3
        last_found = _near(pl, t_last)
        rec.update(true=j, t_start=t.start_frame, t_end=t.end_frame, t_end_reason=t.end_reason,
                   t_winner=t.winner, t_server=t.server,
                   t_landing=None if t.landing_xy is None else [round(v, 2) for v in t.landing_xy],
                   t_margin=None if t.margin is None else round(t.margin, 3),
                   n_true_shots=t.n_shots, last_hit_ok=last_ok, last_hit_found=last_found,
                   last_true_hit_detected=_near([h.frame for h in d["hits"]], t_last),
                   extra_after=sum(1 for f in pl if f > t_last + 3))
        err = None
        if r.landing_xy is not None and t.landing_xy is not None:
            err = float(np.hypot(r.landing_xy[0] - t.landing_xy[0], r.landing_xy[1] - t.landing_xy[1]))
            rec["land_err"] = round(err, 2)
        rel.append((rec["oconf"], r.winner == t.winner, r.end_reason == t.end_reason, r.winner in R.SIDES))
        bad_end = r.end_reason != t.end_reason
        bad_win = r.winner != t.winner
        bad_land = err is not None and err > 1.0
        if not (bad_end or bad_win or bad_land):
            continue
        # bucket the outcome error
        if not last_ok:
            if not rec["last_true_hit_detected"]:
                b = "last_hit_missed"               # final contact never detected
            elif rec["extra_after"]:
                b = "extra_hit_after_end"           # spurious hit after the true last one
            else:
                b = "last_hit_dropped_by_rally"
        elif r.end_reason == "unknown":
            b = f"unknown_{rec['source']}"
        elif {r.end_reason, t.end_reason} == {"in", "out"}:
            b = "in_out_confused_close" if (t.margin is not None and abs(t.margin) < 0.2) else "in_out_confused"
        elif "net" in (r.end_reason, t.end_reason):
            b = f"net_confused_{t.end_reason}_as_{r.end_reason}"
        elif bad_land:
            b = "landing_far"
        else:
            b = f"winner_only_{rec['source']}"
        rec.update(bucket=b, bad_end=bad_end, bad_win=bad_win, bad_land=bad_land)
        recs.append(rec)
    for j, t in enumerate(true):
        if j in mt:
            continue
        ov = [i for i, r in enumerate(pred) if min(r.end_frame, t.end_frame) >= max(r.start_frame, t.start_frame)]
        tshots = shots[t.first_shot:t.first_shot + t.n_shots]
        serve_found = _near([h.frame for h in d["hits"]], tshots[0].frame)
        if not ov:
            b = "fn_missed_entirely"
        elif len(ov) >= 2:
            b = "fn_split"
        elif any(k != j and min(pred[ov[0]].end_frame, u.end_frame) >= max(pred[ov[0]].start_frame, u.start_frame)
                 for k, u in enumerate(true)):
            b = "fn_merged"
        else:
            b = "fn_boundary_off"
        recs.append({"seed": d["seed"], "true": j, "t_start": t.start_frame, "t_end": t.end_frame,
                     "n_true_shots": t.n_shots, "serve_detected": serve_found, "pred": ov,
                     "pred_spans": [(pred[i].start_frame, pred[i].end_frame) for i in ov],
                     "bucket": b})
    return recs, {"pred": len(pred), "true": len(true), "tp": len(pairs), "rel": rel}


def reliability(rel: list[tuple], bins=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [x for x in rel if lo <= x[0] < hi]
        out.append({"bin": f"[{lo:.1f},{min(hi, 1.0):.1f}{')' if hi < 1 else ']'}", "n": len(sel),
                    "winner_acc": round(float(np.mean([x[1] for x in sel])), 3) if sel else None,
                    "end_acc": round(float(np.mean([x[2] for x in sel])), 3) if sel else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=0, help="print this many error records per bucket")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    allrec, rel = [], []
    P = T = TP = 0
    for d in samples:
        recs, c = analyse(d)
        allrec += recs
        rel += c["rel"]
        P, T, TP = P + c["pred"], T + c["true"], TP + c["tp"]
    prec, rec_ = TP / max(P, 1), TP / max(T, 1)
    n_m = len(rel)
    print(f"{args.style} seeds {args.seed}..{args.seed + args.n - 1}: pred {P} true {T} tp {TP} "
          f"P {prec:.3f} R {rec_:.3f} F1 {2 * prec * rec_ / max(prec + rec_, 1e-9):.3f} | "
          f"end {np.mean([x[2] for x in rel]):.3f} win {np.mean([x[1] for x in rel]):.3f} (n={n_m})")
    cnt = Counter(r["bucket"] for r in allrec)
    for b, k in cnt.most_common():
        print(f"  {k:4d}  {b}")
    print("reliability (outcome_confidence -> winner acc):")
    for row in reliability(rel):
        print("  ", row)
    if args.dump:
        by = {}
        for r in allrec:
            by.setdefault(r["bucket"], []).append(r)
        for b, lst in by.items():
            print(f"--- {b}")
            for r in lst[:args.dump]:
                print("   ", json.dumps(r, default=str))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"records": allrec, "buckets": cnt, "reliability": reliability(rel)}, f,
                      default=str, indent=1)


if __name__ == "__main__":
    main()
