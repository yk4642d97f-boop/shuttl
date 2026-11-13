#!/usr/bin/env python
"""RallyConfig A/B on the inputs of tests/test_rally.py::test_segmentation_quality
(same worlds, render, emulated hits and 3-D segments, same RNG order). Prints
F1 / end / winner per fps for every variant, plus the truth rallies a variant
matches differently from the default (seed, truth index, pred span, why).

  .venv/bin/python -m scripts.accuracy.seg_quality_ab --ab trail_max_conf=0.0 --ab onehit_min_flight_s=0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.tracking import rally as R  # noqa: E402
from scripts.accuracy.cfg_diff import parse_cfg  # noqa: E402
from tests import test_rally as T  # noqa: E402


def inputs(seed: int, fps: float, style: str = "behind"):
    w = T._world(seed, 10)
    c = T._render(w, fps, seed, style)
    rng = np.random.default_rng(seed + 12345)
    hits, src = T._emulate_hits(c, rng)
    segs = T._emulate_segments(c, hits, src, rng)
    return c, hits, src, segs


def run(data, cfg):
    st, per = T.Stats(), []
    for c, hits, src, segs in data:
        pred = R.segment_rallies(c.track, hits, segs, c.meta, c.camera, config=cfg)
        st.add(pred, c)
        per.append((pred, dict((j, i) for i, j in T._match(pred, c.truth, c.fps))))
    return st, per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="append", nargs="*", default=[])
    ap.add_argument("--fps", nargs="*", type=float, default=[30.0, 60.0, 120.0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    variants = [("default", R.RallyConfig())] + [(" ".join(v), parse_cfg(v)) for v in a.ab]
    for fps in a.fps:
        data = [inputs(s, fps) for s in range(a.seeds)]
        res = [(name, *run(data, cfg)) for name, cfg in variants]
        for name, st, _ in res:
            print(f"fps {fps:g} [{name}] {st}  pred {st.n_pred}")
        if a.quiet:
            continue
        base = res[0][2]
        for name, _, per in res[1:]:
            for k, ((pb, mb), (pv, mv)) in enumerate(zip(base, per)):
                c = data[k][0]
                for j, tr in enumerate(c.truth):
                    ib, iv = mb.get(j), mv.get(j)
                    if (ib is None) == (iv is None):
                        continue
                    wr = c.world.rallies[j]
                    def span(p, i):
                        return None if i is None else (p[i].start_frame, p[i].end_frame, len(p[i].shots))
                    print(f"  [{name}] seed {k} truth {j} {tr} end {wr['end']} "
                          f"default {span(pb, ib)} variant {span(pv, iv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
