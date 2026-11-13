"""
Shared helpers for the accuracy analysis scripts: build the same seeded sims as
scripts/benchmark.py (identical RNG streams), run the upstream chain
(linker -> hits -> reconstruct3d) and cache it per seed so rally.py / shots.py
can be iterated in seconds.

Cache key = style, size, fps, rallies, seed + a hash of the hit stage sources
(hits.py, the installed hit model and reconstruct3d.py), so changing hits.py
recomputes hits and segments automatically.

Run from backend/.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.tracking import sim as S  # noqa: E402

CACHE = Path(os.environ.get("SHUTTL_ACC_CACHE", "/tmp/astra7/cache"))


def sim_args(style: str, size: str, fps: float, rallies: int) -> argparse.Namespace:
    w, h = (int(v) for v in size.lower().split("x"))
    return argparse.Namespace(image_size=(w, h), style=style, fps=fps, rallies=rallies,
                              doubles=None, sigma=1.5, miss_rate=0.1, fp_rate=0.03,
                              player_step=2)


def make_sim(a: argparse.Namespace, seed: int) -> S.SimRally:
    """Same as scripts/benchmark.make_sim."""
    rng = np.random.default_rng([seed, 7])
    cam = S.random_camera(rng, a.image_size, a.style)
    doubles = a.doubles if a.doubles is not None else bool(rng.random() < 0.3)
    if a.rallies > 1:
        return S.simulate_match(seed, a.rallies, camera=cam, fps=a.fps, doubles=doubles)
    return S.simulate_rally(seed, camera=cam, fps=a.fps, doubles=doubles)


def stage_hash() -> str:
    # pin a cached upstream (e.g. while another session edits linker.py) so that
    # rally-stage A/Bs keep comparing on identical inputs
    if os.environ.get("SHUTTL_ACC_HASH"):
        return os.environ["SHUTTL_ACC_HASH"]
    h = hashlib.sha1()
    for p in ("app/services/tracking/hits.py", "app/services/tracking/reconstruct3d.py",
              "app/services/tracking/linker.py", "ml/weights/hit_rescorer.npz"):
        f = ROOT / p
        if f.exists():
            h.update(f.read_bytes())
    h.update(os.environ.get("SHUTTL_HIT_MODEL", "default").encode())
    h.update(os.environ.get("SHUTTL_HITS_OVERRIDE", "").encode())
    return h.hexdigest()[:10]


def hits_config():
    """HitConfig with SHUTTL_HITS_OVERRIDE="key=val key=val" applied (A/B of hits.py options
    through the whole cached upstream chain)."""
    from app.services.tracking import hits as H
    kw = {}
    for kv in os.environ.get("SHUTTL_HITS_OVERRIDE", "").split():
        k, v = kv.split("=")
        cur = getattr(H.HitConfig(), k)
        kw[k] = (v == "True") if isinstance(cur, bool) else type(cur)(v)
    return H.HitConfig(**kw)


def _key(a: argparse.Namespace, seed: int) -> Path:
    w, h = a.image_size
    return CACHE / f"{a.style}_{w}x{h}_{int(a.fps)}_r{a.rallies}_s{seed}_{stage_hash()}.pkl"


def _compute(job: tuple[argparse.Namespace, int]) -> dict:
    a, seed = job
    from app.services.tracking import hits as H
    from app.services.tracking import linker, reconstruct3d
    sim = make_sim(a, seed)
    rng = np.random.default_rng([seed, 11])
    cands = S.corrupt_detections(sim, rng, a.sigma, a.miss_rate, True, a.fp_rate)
    players = S.truth_players(sim, every=a.player_step)
    track = linker.link_candidates(cands, sim.meta)
    hits = H.detect_hits(track, sim.meta, sim.camera, players, hits_config())
    segs = reconstruct3d.reconstruct_segments(track, hits, sim.camera, sim.meta, players) or []
    return {"seed": seed, "track": track, "hits": hits, "segs": segs, "players": players,
            "meta": sim.meta, "camera": sim.camera, "true_hits": sim.hits,
            "true_rallies": sim.rallies, "true_shots": sim.shots, "doubles": sim.doubles,
            "n": sim.n}


def load_samples(a: argparse.Namespace, seeds: list[int], workers: int = 4) -> list[dict]:
    """Cached upstream outputs per seed (computed with <= `workers` processes)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[int, dict] = {}
    todo = []
    for s in seeds:
        k = _key(a, s)
        if k.exists():
            with open(k, "rb") as f:
                out[s] = pickle.load(f)
        else:
            todo.append(s)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for d in ex.map(_compute, [(a, s) for s in todo]):
                with open(_key(a, d["seed"]), "wb") as f:
                    pickle.dump(d, f)
                out[d["seed"]] = d
    return [out[s] for s in seeds]


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--style", default="behind")
    ap.add_argument("--image-size", default="1920x1080")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--rallies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
