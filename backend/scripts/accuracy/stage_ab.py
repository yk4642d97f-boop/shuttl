#!/usr/bin/env python
"""Rally + shots stage evaluation on the cached upstream (common.py), pooled
like scripts/benchmark.py, plus the numbers benchmark.py does not report:
  side_shots   hitter-side accuracy of the FINAL rally shots (what the app shows),
               matched to true hits (+-3 frames, one-to-one per sample)
  side_last    same, only the last shot of each matched rally (decides the winner)
  hits_side    hits.py label accuracy on the same matching (for reference)
  reliability  outcome_confidence bins -> winner accuracy on matched rallies

  .venv/bin/python -m scripts.accuracy.stage_ab --style behind --seed 1000 --n 40
Library use: evaluate(samples, rally_module=R, shots_module=SH) -> dict.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from app.services.tracking import shots as SH
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match, reliability

GAP = 100_000


def _rd(r, off):
    return {"start_frame": r.start_frame + off, "end_frame": r.end_frame + off,
            "end_reason": r.end_reason, "winner": r.winner, "landing_xy": r.landing_xy,
            "margin": getattr(r, "in_out_margin_m", getattr(r, "margin", None))}


def evaluate(samples: list[dict], rally_module=R, shots_module=SH, with_shots: bool = True,
             config=None) -> dict:
    P, T, rel = [], [], []
    side = [0, 0]
    side_last = [0, 0]
    hits_side = [0, 0]
    typ_p, typ_t = [], []
    secs = {"rally": 0.0, "shots": 0.0}
    for k, d in enumerate(samples):
        t0 = time.process_time()
        pred = rally_module.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                            players=d["players"], config=config)
        secs["rally"] += time.process_time() - t0
        off = k * GAP
        P += [_rd(r, off) for r in pred]
        T += [{"start_frame": t.start_frame + off, "end_frame": t.end_frame + off,
               "end_reason": t.end_reason, "winner": t.winner, "landing_xy": t.landing_xy,
               "margin": t.margin} for t in d["true_rallies"]]
        th = d["true_hits"]
        tf = np.array([h["frame"] for h in th], int)
        shots = [s for r in pred for s in r.shots]
        pf = np.array([s.hit.frame for s in shots], int)
        for i, j in M._match_events(pf, tf, 3):
            side[0] += 1
            side[1] += shots[i].hit.hitter == th[j]["hitter"]
        hf = np.array([h.frame for h in d["hits"]], int)
        for i, j in M._match_events(hf, tf, 3):
            hits_side[0] += 1
            hits_side[1] += d["hits"][i].hitter == th[j]["hitter"]
        for i, j in match(pred, d["true_rallies"]):
            r, t = pred[i], d["true_rallies"][j]
            rel.append((getattr(r, "outcome_confidence", 0.0), r.winner == t.winner,
                        r.end_reason == t.end_reason, r.winner in R.SIDES))
            if r.shots:
                tl = d["true_shots"][t.first_shot + t.n_shots - 1]
                if abs(r.shots[-1].hit.frame - tl.frame) <= 3:
                    side_last[0] += 1
                    side_last[1] += r.shots[-1].hit.hitter == th[t.first_shot + t.n_shots - 1]["hitter"]
        if with_shots:
            t0 = time.process_time()
            typed = [shots_module.classify_shots(r, d["camera"], d["players"]) for r in pred]
            secs["shots"] += time.process_time() - t0
            ts = [s for r in typed for s in r.shots]
            pf2 = np.array([s.hit.frame for s in ts], int)
            for i, j in M._match_events(pf2, tf, 3):
                typ_p.append(ts[i].shot_type)
                typ_t.append(th[j]["shot_type"])
    m = M.rally_metrics(P, T)
    out = {k: m[k] for k in ("n_pred", "n_true", "tp", "precision", "recall", "f1", "end_reason_acc",
                             "winner_acc", "landing_err_m_median", "landing_err_m_p95")}
    out["side_shots"] = side[1] / max(side[0], 1)
    out["side_shots_n"] = side[0]
    out["side_last"] = side_last[1] / max(side_last[0], 1)
    out["side_last_n"] = side_last[0]
    out["hits_side"] = hits_side[1] / max(hits_side[0], 1)
    if typ_p:
        out["shots_acc"] = float(np.mean([a == b for a, b in zip(typ_p, typ_t)]))
    out["reliability"] = reliability(rel)
    hi = [x for x in rel if x[0] >= 0.8]
    out["conf08_n"] = len(hi)
    out["conf08_win"] = float(np.mean([x[1] for x in hi])) if hi else float("nan")
    mid = [x for x in rel if x[0] >= 0.6]
    out["conf06_n"] = len(mid)
    out["conf06_win"] = float(np.mean([x[1] for x in mid])) if mid else float("nan")
    out["secs"] = secs
    return out


def fmt(r: dict) -> str:
    s = (f"F1 {r['f1']:.3f} (P {r['precision']:.3f} R {r['recall']:.3f} {r['n_pred']}/{r['n_true']}) "
         f"end {r['end_reason_acc']:.3f} win {r['winner_acc']:.3f} "
         f"land {r['landing_err_m_median']:.3f}/{r['landing_err_m_p95']:.2f} | "
         f"side shots {r['side_shots']:.3f} (n {r['side_shots_n']}) last {r['side_last']:.3f} "
         f"hits {r['hits_side']:.3f}")
    if "shots_acc" in r:
        s += f" | shots {r['shots_acc']:.3f}"
    s += (f" | c>=.8 {r['conf08_n']}:{r['conf08_win']:.3f} c>=.6 {r['conf06_n']}:{r['conf06_win']:.3f}"
          f" | {r['secs']['rally']:.2f}s/{r['secs']['shots']:.2f}s")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--ab", nargs="*", default=None,
                    help="also run with these RallyConfig overrides (key=val ...) and print both")
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    runs = [("current", None)]
    if args.ab is not None:
        kw = {}
        for kv in args.ab:
            k, v = kv.split("=")
            kw[k] = type(getattr(R.RallyConfig(), k))(v) if v not in ("True", "False") else v == "True"
        runs.insert(0, (" ".join(args.ab), R.RallyConfig(**kw)))
    for name, cfg in runs:
        r = evaluate(samples, with_shots=not args.no_shots, config=cfg)
        print(f"{args.style} s{args.seed}..{args.seed + args.n - 1} [{name}]: {fmt(r)}")
        for row in r["reliability"]:
            print("   ", row)


if __name__ == "__main__":
    main()
