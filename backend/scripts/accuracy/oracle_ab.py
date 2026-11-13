#!/usr/bin/env python
"""Oracle-stage rally A/B (benchmark.py --stage oracle inputs: noisy truth track,
truth hits, truth segments, truth players; same RNG order) for RallyConfig
overrides. Prints rally F1 / end / winner and the rallies that differ.

  .venv/bin/python -m scripts.accuracy.oracle_ab --style behind --seed 1000 --n 20 --ab unseen_arrival_ok=False
"""
from __future__ import annotations

import argparse

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from app.services.tracking import sim as S
from scripts.accuracy.cfg_diff import parse_cfg
from scripts.accuracy.common import add_common_args, make_sim, sim_args

GAP = 100_000


def oracle_inputs(a, seed):
    sim = make_sim(a, seed)
    rng = np.random.default_rng([seed, 11])
    S.corrupt_detections(sim, rng, a.sigma, a.miss_rate, True, a.fp_rate)   # same RNG order as benchmark
    tr = S.truth_track(sim)
    tr.xy = tr.xy + rng.normal(0, 1.0, tr.xy.shape)
    return sim, tr, S.truth_hits(sim), S.truth_segments(sim), S.truth_players(sim, every=a.player_step)


def run(inputs, cfg):
    P, T, per = [], [], []
    for k, (sim, tr, th, ts, pl) in enumerate(inputs):
        pred = R.segment_rallies(tr, th, ts, sim.meta, sim.camera, players=pl, config=cfg)
        per.append(pred)
        P += [{"start_frame": r.start_frame + k * GAP, "end_frame": r.end_frame + k * GAP,
               "end_reason": r.end_reason, "winner": r.winner, "landing_xy": r.landing_xy,
               "margin": r.in_out_margin_m} for r in pred]
        T += [{"start_frame": t.start_frame + k * GAP, "end_frame": t.end_frame + k * GAP,
               "end_reason": t.end_reason, "winner": t.winner, "landing_xy": t.landing_xy,
               "margin": t.margin} for t in sim.rallies]
    return M.rally_metrics(P, T), per


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ab", nargs="*", default=[])
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    inputs = [oracle_inputs(a, s) for s in range(args.seed, args.seed + args.n)]
    ma, pa = run(inputs, parse_cfg(args.ab))
    mb, pb = run(inputs, R.RallyConfig())
    for name, m in ((" ".join(args.ab) or "default", ma), ("current", mb)):
        print(f"{args.style} [{name}] F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"end {m['end_reason_acc']:.3f} win {m['winner_acc']:.3f} n_pred {m['n_pred']}")
    for k, (x, y) in enumerate(zip(pa, pb)):
        sx = [(r.start_frame, r.end_frame, r.end_reason, r.winner, r.outcome_source) for r in x]
        sy = [(r.start_frame, r.end_frame, r.end_reason, r.winner, r.outcome_source) for r in y]
        if sx != sy:
            tr = [(t.start_frame, t.end_frame, t.end_reason, t.winner) for t in inputs[k][0].rallies]
            print(f"seed {args.seed + k}\n  true {tr}\n  A    {sx}\n  B    {sy}")


if __name__ == "__main__":
    main()
