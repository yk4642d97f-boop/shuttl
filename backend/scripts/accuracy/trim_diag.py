#!/usr/bin/env python
"""Which contacts the trailing trim (trail_max_conf / trail_min_flight_s) drops on
the tests/test_rally.py generator: world kind (rally last / mid, feed, spurious),
confidence, observed moving time after it, true flight duration + end, and the
kink's own fit / continuation. Replicates segment_rallies up to the trim.

  .venv/bin/python -m scripts.accuracy.trim_diag --fps 60
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.seg_quality_ab import inputs
from tests import test_rally as T


def trimmed(c, hits, segs, cfg):
    mo = R._Motion(c.track, float(c.meta.height), cfg)
    hs = R._clean_hits(hits, mo.n)
    sg = R._match_segments(hs, list(segs), mo.t, cfg)
    hs, sg, conts = R._merge_same_flight(hs, sg, mo.t, cfg)
    groups, _ = R._group_hits(hs, sg, mo, cfg, conts)
    hs, _ = R._decode_sides(hs, sg, conts, groups, mo.t, cfg)
    out = []
    for g in groups:
        while g and R._leading_nonhit(g, hs, sg, mo, True, cfg):
            g = g[1:]
        while len(g) > 1 and sg[g[-1]] is None and not R._is_flight(hs[g[-1]], None, mo, cfg):
            g = g[:-1]
        while len(g) > 1 and (hs[g[-1]].confidence or 0.0) < cfg.trail_max_conf \
                and R._moving_after(mo, hs[g[-1]]) < cfg.trail_min_flight_s \
                and not (cfg.trail_launch_guard and R._launches(
                    hs[g[-1]], sg[g[-1]], conts[g[-2]] if conts[g[-2]] is not None else sg[g[-2]], cfg)):
            out.append((hs[g[-1]], sg[g[-1]], conts[g[-1]], R._moving_after(mo, hs[g[-1]]), len(g)))
            g = g[:-1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", nargs="*", type=float, default=[30.0, 60.0, 120.0])
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args(argv)
    cfg = R.RallyConfig()
    for fps in a.fps:
        tally = Counter()
        for s in range(a.seeds):
            c, hits, src, segs = inputs(s, fps)
            w = c.world
            by_frame = {h.frame: (k, src[k]) for k, h in enumerate(hits)}
            for h, seg, cont, mv, n in trimmed(c, hits, segs, cfg):
                wi = by_frame.get(h.frame, (None, None))[1]
                wh = w.hits[wi] if wi is not None else None
                kind = wh["kind"] if wh else "?"
                fl = w.flights[wh["flight"]] if wh and wh["flight"] is not None else None
                if kind == "rally":
                    r = w.rallies[wh["rally"]]
                    last = wi == max(i for i, x in enumerate(w.hits) if x["rally"] == wh["rally"])
                    kind = f"rally-{'last' if last else 'mid'}-{r['end']}"
                dur = (len(fl["P"]) - 1) * T.H if fl else None
                tally[kind] += 1
                print(f"fps {fps:g} seed {s} f {h.frame} {kind:16s} conf {h.confidence:.2f} moving {mv:.2f}s "
                      f"true_dur {dur if dur is None else round(dur, 2)} fit {'y' if seg is not None else 'n'} "
                      f"cont {'y' if cont is not None else 'n'} group_len {n}")
        print(f"fps {fps:g} trimmed: {dict(tally)}")


if __name__ == "__main__":
    main()
