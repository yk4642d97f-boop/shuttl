"""Reference fixtures for ShuttlLinker (Swift port of tracking/linker.py).

Dumps linker inputs (per-frame candidates + VideoMeta) and Python outputs
(Track2D incl. raw_xy / visible / interpolated / conf, plus a few
intermediates) for
  (a) synthetic scenarios with ground truth — tests/test_linker.py generators
      (pinhole camera + drag physics, σ 1.5 px, misses + bursts, uniform FPs,
      static distractors, ±2 ms clock jitter) incl. walking weak distractors,
      a held shuttle, scene cuts, 25p-in-30p repeats, VFR drops, 120 fps
      (coarse kink localisation) and degenerate inputs; plus one sim.py match;
  (b) real held-out clips (TrackNetV2 Test split) from the benchmark's cached
      detector candidates /tmp/shuttl_bench_cache/<match>_<clip>_<det>.npz.

Bundle (Tests/ShuttlLinkerTests/Fixtures): linker_sim.json, linker_real.json
(6 clips × 2 detectors). Full 29 × 2 real set → /tmp/shuttl_linker_parity/
real_full.json for the gated full-parity test; `--score DIR` scores Swift
outputs written there with the benchmark's metrics.

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.linker [--score]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.services.tracking import linker as L
from app.services.tracking import sim as S
from app.services.tracking import video_io
from app.services.tracking.data_types import ShuttleCandidate, Track2D, VideoMeta
from scripts.swift_fixtures.common import SWIFT_PACKAGES, TEST_SPLIT, dump
from tests.test_linker import H, W, _cut_case, _dead_window, _gap_case, make_case

OUT = SWIFT_PACKAGES / "ShuttlLinker/Tests/ShuttlLinkerTests/Fixtures"
FULL = Path("/tmp/shuttl_linker_parity")
CACHE = Path("/tmp/shuttl_bench_cache")
BUNDLE_CLIPS = ["match1_1_05_02", "match1_2_03_10", "match2_1_19_15", "match2_2_02_05",
                "match3_1_05_02", "match3_1_09_12"]


# ── encoding ──────────────────────────────────────────────────────────────────

def _r(v: np.ndarray, nd: int = 9) -> list:
    return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(v, float).ravel()]


def enc_cands(cands: list[list[ShuttleCandidate]]) -> dict:
    f, x, y, s = [], [], [], []
    for i, cs in enumerate(cands):
        for c in cs:
            f.append(i); x.append(float(c.x)); y.append(float(c.y)); s.append(float(c.score))
    return {"n": len(cands), "f": f, "x": x, "y": y, "s": s}


def enc_meta(meta: VideoMeta) -> dict:
    return {"fps": float(meta.fps), "width": int(meta.width), "height": int(meta.height),
            "n_frames": int(meta.n_frames), "rotation_deg": int(meta.rotation_deg),
            "timestamps": [float(v) for v in np.asarray(meta.timestamps, float)]}


def enc_track(tr: Track2D, with_t: bool = False) -> dict:
    d = {"xy": _r(tr.xy), "raw_xy": _r(tr.raw_xy) if tr.raw_xy is not None else None,
         "conf": _r(tr.conf), "visible": np.flatnonzero(tr.visible).tolist(),
         "interpolated": np.flatnonzero(tr.interpolated).tolist(), "n": int(len(tr.t))}
    if with_t:
        d["t"] = [float(v) for v in tr.t]
    return d


def round_cands(cands: list[list[ShuttleCandidate]]) -> list[list[ShuttleCandidate]]:
    """Short decimals for the synthetic inputs (1e-3 px, 1e-4 score): same meaning,
    compact JSON, bit-identical in both languages."""
    return [[ShuttleCandidate(round(c.x, 3), round(c.y, 3), round(c.score, 4)) for c in cs] for cs in cands]


def intermediates(cands: list[list[ShuttleCandidate]], meta: VideoMeta, cuts_in) -> dict:
    """First-pass internals to localise a mismatch (cuts, association, repeats)."""
    n = len(cands)
    t = L._timestamps(meta, n)
    cfg = L.LinkerConfig()
    wd, ht = float(meta.width or 1920), float(meta.height or 1080)
    cuts = L._cut_list(n, cuts_in, L._detect_cuts(cands, t, wd, ht, cfg))
    xy, sc, start = L._associate(cands, t, wd, ht, cfg, L._shot0(n, cuts))
    rep = L._find_repeats(t, xy, L._probe_candidates(cands, xy, cfg, wd, ht), float(np.hypot(wd, ht)), cfg)
    return {"cuts": cuts.tolist(), "assoc_obs": np.flatnonzero(np.isfinite(xy[:, 0])).tolist(),
            "assoc_start": np.flatnonzero(start).tolist(), "assoc_xy": _r(xy),
            "rep_dup": None if rep is None else np.flatnonzero(rep[0]).tolist(),
            "rep_clock": None if rep is None else [float(v) for v in rep[1]]}


def case(name: str, cands, meta: VideoMeta, scene_cuts=None, inter: bool = True,
         with_t: bool | None = None, **extra) -> dict:
    t0 = time.perf_counter()
    tr = L.link_candidates(cands, meta, scene_cuts=scene_cuts)
    el = time.perf_counter() - t0
    d = {"name": name, "meta": enc_meta(meta), "cands": enc_cands(cands),
         "scene_cuts": None if scene_cuts is None else [int(c) for c in scene_cuts],
         "out": enc_track(tr, with_t=name.startswith("deg") if with_t is None else with_t), "py_seconds": el}
    if inter and len(cands):
        d["inter"] = intermediates(cands, meta, scene_cuts)
    d.update(extra)
    if not name.startswith("edge"):
        print(f"  {name}: {len(cands)} frames, vis {int(tr.visible.sum())}, interp "
              f"{int(tr.interpolated.sum())}, {el * 1e3:.0f} ms")
    return d


# ── synthetic scenarios ───────────────────────────────────────────────────────

def sim_cases() -> list[dict]:
    out = []
    for seed, fps, head in ((0, 30.0, None), (1, 60.0, None), (2, 120.0, 1800)):
        c = make_case(seed, fps)
        cands, meta = round_cands(c.cands), c.meta
        if head:
            cands = cands[:head]
            meta = VideoMeta(fps=fps, width=W, height=H, n_frames=head, timestamps=meta.timestamps[:head])
        d = case(f"sim_s{seed}_{int(fps)}", cands, meta)
        if seed == 1:                                   # smooth_track / find_kinks API parity
            tr = L.link_candidates(cands, meta)
            kf, z = L.find_kinks(tr, image_size=(W, H), return_scores=True)
            d["find_kinks"] = {"frames": kf.tolist(), "z": _r(z)}
            d["smooth"] = enc_track(L.smooth_track(tr, image_size=(W, H)))
            hf = c.hit_frames.tolist()
            d["smooth_kinks"] = {"kinks": hf, "out": enc_track(L.smooth_track(tr, kinks=hf, image_size=(W, H)))}
            d["in_track"] = enc_track(tr)
            d["in_track"]["t"] = [float(v) for v in tr.t]
        out.append(d)

    # weak walking distractor + held shuttle (strong / weak) — test_weak_walking_distractors_not_adopted
    seed = 3
    c = make_case(seed, 60.0)
    rng = np.random.default_rng(seed)
    n = c.meta.n_frames
    walk = np.cumsum(rng.normal(0, 1.0, (n, 2)) + [1.2, 0.3], 0) + [300.0, 700.0]
    a, b = _dead_window(c)
    mid = (a + b) // 2
    held = np.arange(mid - 24, mid + 24)
    hp = np.array([900.0, 500.0]) + np.cumsum(rng.normal(0, 0.4, (len(held), 2)), 0)
    for strong in (True, False):
        cands = [list(q) for q in c.cands]
        for i in range(n):
            if rng.random() < 0.8:
                cands[i].append(ShuttleCandidate(float(walk[i, 0]), float(walk[i, 1]),
                                                 float(rng.uniform(0.2, 0.45))))
        for k, i in enumerate(held):
            cands[i].append(ShuttleCandidate(float(hp[k, 0]), float(hp[k, 1]),
                                             float(rng.uniform(0.7, 0.9) if strong else rng.uniform(0.2, 0.4))))
        out.append(case(f"sim_weak_walk_{'strong' if strong else 'weak'}_held", round_cands(cands), c.meta,
                        held=[int(held[0]), int(held[-1]) + 1], held_xy=_r(hp)))

    # scene cut: passed in, and inferred from the candidate stream
    c, cands, cut, cut2 = _cut_case(4)
    cands = round_cands(cands)
    out.append(case("sim_cut_explicit", cands, c.meta, scene_cuts=[cut], cut_range=[cut, cut2]))
    out.append(case("sim_cut_detected", cands, c.meta, cut_range=[cut, cut2]))

    # 25p content in a 30p container with a pulldown phase jump
    src = make_case(5, 25.0)
    k = np.arange(int(src.meta.n_frames * 30 / 25))
    s_idx = np.minimum(k * 25 // 30 + 2 * (k >= len(k) // 2), src.meta.n_frames - 1)
    cands = round_cands([src.cands[i] for i in s_idx])
    out.append(case("sim_repeat_25p_in_30p", cands,
                    VideoMeta(fps=30.0, width=W, height=H, n_frames=len(k), timestamps=k / 30.0)))

    # variable frame rate (15 % of frames dropped)
    c = make_case(3, 60.0)
    rng = np.random.default_rng(5)
    keep = np.sort(np.flatnonzero(rng.random(c.meta.n_frames) > 0.15))
    out.append(case("sim_vfr", round_cands([c.cands[i] for i in keep]),
                    VideoMeta(fps=60.0, width=W, height=H, n_frames=len(keep),
                              timestamps=c.meta.timestamps[keep])))

    # 720p (resolution-relative config)
    c = make_case(6, 60.0)
    sc = 720 / 1080
    cands = round_cands([[ShuttleCandidate(q.x * sc, q.y * sc, q.score) for q in fr] for fr in c.cands])
    out.append(case("sim_720p", cands, VideoMeta(fps=60.0, width=1280, height=720, n_frames=c.meta.n_frames,
                                                 timestamps=c.meta.timestamps)))

    # sim.py match (2 rallies, random camera, benchmark corruption model)
    rng = np.random.default_rng([11, 7])
    cam = S.random_camera(rng, (1920, 1080), "random")
    sm = S.simulate_match(11, 2, camera=cam, fps=60.0)
    cands = S.corrupt_px(sm.shuttle_px, sm.visible, (1920, 1080), np.random.default_rng(12), 1.5, 0.1, True, 0.03)
    out.append(case("simpy_match_60", round_cands(cands),
                    VideoMeta(fps=60.0, width=1920, height=1080, n_frames=sm.n, timestamps=sm.timestamps)))

    # degenerate inputs — test_degenerate_inputs
    n = 50
    meta = VideoMeta(fps=30.0, width=W, height=H, n_frames=n, timestamps=np.arange(n) / 30)
    out.append(case("deg_zero_frames", [], VideoMeta(fps=30.0, width=W, height=H, n_frames=0,
                                                     timestamps=np.zeros(0)), inter=False))
    out.append(case("deg_all_empty", [[] for _ in range(n)], meta))
    one = [[] for _ in range(n)]
    one[10] = [ShuttleCandidate(100.0, 100.0, 0.9)]
    out.append(case("deg_lone_blip", one, meta))
    line = [[ShuttleCandidate(100.0 + 5 * i, 200.0 + 2 * i, 0.9)] for i in range(n)]
    bad_t = VideoMeta(fps=30.0, width=W, height=H, n_frames=n,
                      timestamps=np.r_[np.arange(n - 5) / 30, np.full(5, (n - 6) / 30)])
    out.append(case("deg_bad_timestamps", line, bad_t))
    short_t = VideoMeta(fps=0.0, width=0, height=0, n_frames=n, timestamps=np.arange(20) / 25)
    out.append(case("deg_short_timestamps_nofps", line, short_t))
    return out


# ── real clips ────────────────────────────────────────────────────────────────

def real_cases(clips: list[str] | None) -> list[dict]:
    out = []
    metas: dict[str, VideoMeta] = {}
    for vid in sorted(TEST_SPLIT.glob("match*/video/*.mp4")):
        key = f"{vid.parent.parent.name}_{vid.stem}"
        if clips is not None and key not in clips:
            continue
        m = video_io.probe(str(vid))
        for det in ("tracknetv3", "wasb"):
            p = CACHE / f"{key}_{det}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            n = min(int(z["n_done"]), m.n_frames)
            arr = z["cands"][:n]
            cands = [[ShuttleCandidate(float(x), float(y), float(s)) for x, y, s in row if np.isfinite(s)]
                     for row in arr]
            meta = VideoMeta(fps=m.fps, width=m.width, height=m.height, n_frames=n,
                             timestamps=m.timestamps[:n], rotation_deg=m.rotation_deg)
            out.append(case(f"{key}_{det}", cands, meta, inter=False,
                            clip={"match": vid.parent.parent.name, "clip": vid.stem, "detector": det}))
        metas[key] = m
    return out


# ── scoring Swift outputs with the benchmark metrics ──────────────────────────

def score(swift_dir: Path) -> None:
    from scripts import benchmark as B
    ref = json.loads((FULL / "real_full.json").read_text())
    for det in ("tracknetv3", "wasb"):
        pools = {"python": B.Pool(), "swift": B.Pool()}
        n_eq = n_cases = 0
        for d in ref:
            if d["clip"]["detector"] != det:
                continue
            sw_p = swift_dir / f"{d['name']}.json"
            if not sw_p.exists():
                continue
            sw = json.loads(sw_p.read_text())
            n = d["out"]["n"]
            clip = {"match": d["clip"]["match"], "clip": d["clip"]["clip"]}
            lab = B.REAL_ROOT / "corrected" / clip["match"] / "corrected_csv" / f"{clip['clip']}_ball.csv"
            if not lab.exists():
                lab = TEST_SPLIT / clip["match"] / "csv" / f"{clip['clip']}_ball.csv"
            true_xy, vis = B.load_label_csv(str(lab), n)
            dm = B.drop_mask(clip["match"], clip["clip"], n)
            mask = np.ones(n, bool) if dm is None else dm
            size = (d["meta"]["width"], d["meta"]["height"])
            for who, o in (("python", d["out"]), ("swift", sw)):
                xy = np.array([np.nan if v is None else v for v in o["xy"]], float).reshape(-1, 2)
                interp = np.zeros(n, bool)
                interp[o["interpolated"]] = True
                pools[who].add_frames("real_linker", xy, true_xy, vis, interp, mask)
                pools[who].add_list("real_linker", "_tn4", [(xy, true_xy, vis, mask, size)])
            n_cases += 1
            n_eq += (sw["visible"] == d["out"]["visible"] and sw["interpolated"] == d["out"]["interpolated"])
        for who, pool in pools.items():
            res = pool.results(10.0, 3)
            B.add_tn4(pool, res)
            r = res.get("real_linker", {})
            print(f"{det:10s} {who:6s} clips {n_cases}: f1 {r.get('f1', float('nan')):.4f}  tn4_f1 "
                  f"{r.get('tn4_f1', float('nan')):.4f}  P {r.get('precision', float('nan')):.4f}  "
                  f"R {r.get('recall', float('nan')):.4f}")
        print(f"{det:10s} identical masks on {n_eq}/{n_cases} clips")


# ── degenerate inputs (venue 192×144 clips, empty / 1-frame / weak-only, broken clocks) ──

def _edge_ts(kind: str, n: int, fps: float) -> np.ndarray:
    f = fps if np.isfinite(fps) and fps > 0 else 30.0
    t = np.arange(n) / f
    if kind == "missing": return np.zeros(0)
    if kind == "short": return t[: n // 2]
    if kind == "dup_run" and n: t[n // 3: 2 * n // 3] = t[n // 3]
    if kind == "zeros": return np.zeros(n)
    if kind == "nan_inside": t[n // 2:n // 2 + 2] = np.nan
    if kind == "reversed": return t[::-1].copy()
    return t


def _edge_cands(kind: str, n: int, w: int, h: int, rng: np.random.Generator) -> list:
    W, H = max(w, 1), max(h, 1)
    out = [[] for _ in range(n)]
    for i in range(n):
        x, y = W * 0.1 + i * W * 0.8 / max(n, 1), H * 0.2 + i * H * 0.5 / max(n, 1)
        if kind == "one" and i == n // 2: out[i] = [ShuttleCandidate(W / 2, H / 2, 0.9)]
        if kind in ("weak_line", "strong_line"): out[i] = [ShuttleCandidate(x, y, 0.3 if kind[0] == "w" else 0.9)]
        if kind == "burst" and (i // 4) % 2 == 0: out[i] = [ShuttleCandidate(x, y, 0.7 if i % 3 else 0.2)]
        if kind == "noise":
            out[i] = [ShuttleCandidate(round(float(rng.uniform(0, W)), 3), round(float(rng.uniform(0, H)), 3),
                                       round(float(rng.uniform(0, 1)), 4)) for _ in range(int(rng.integers(0, 4)))]
    return out


def edge_cases() -> list[dict]:
    """All inputs Python links after the venue fixes; out = Python's Track2D (with t),
    plus smooth_track / find_kinks of it. `fps_str` keeps inf/nan (JSON has only null)."""
    out, rng = [], np.random.default_rng(3)

    def add(name, cands, meta, **kw):
        d = case("edge_" + name, cands, meta, inter=False, with_t=True, **kw)
        d["fps_str"] = repr(float(meta.fps))
        if d["cands"]["n"]:
            tr = L.link_candidates(cands, meta)
            size = (meta.width, meta.height) if meta.width and meta.height else None
            d["smooth"] = enc_track(L.smooth_track(tr, image_size=size), with_t=True)
            kf, z = L.find_kinks(tr, image_size=size, return_scores=True)
            d["find_kinks"] = {"frames": kf.tolist(), "z": _r(z)}
        out.append(d)

    for n in (0, 1, 2, 3, 5, 13, 40):
        for (w, h) in ((192, 144), (1920, 1080)):
            for kind in ("empty", "one", "weak_line", "strong_line", "burst", "noise"):
                if kind == "empty" and n not in (0, 1, 13):
                    continue
                for fps in (30.0, float("inf")):          # fps 0 / nan fall back to 30 (== the 30 cases)
                    for tk in ("nominal", "missing", "short", "dup_run", "zeros", "nan_inside", "reversed"):
                        add(f"n{n}_{w}x{h}_{kind}_fps{fps:g}_{tk}", _edge_cands(kind, n, w, h, rng),
                            VideoMeta(fps=fps, width=w, height=h, n_frames=n, timestamps=_edge_ts(tk, n, fps)))
    for w, h in ((192, 144), (64, 48), (1920, 1080), (0, 0)):         # venue: weak-only paths
        n, v = 12, 0.5 * float(np.hypot(w or 1920, h or 1080)) / 30
        m = VideoMeta(fps=30.0, width=w, height=h, n_frames=n, timestamps=np.arange(n) / 30)
        W, Hh = w or 1920, h or 1080
        add(f"weakfly_{w}x{h}", [[ShuttleCandidate(0.1 * W + v * i * 0.9, 0.2 * Hh + v * i * 0.4, 0.3)]
                                 for i in range(n)], m)
        add(f"weakstill_{w}x{h}", [[ShuttleCandidate(0.3 * W + 0.01 * i, 0.6 * Hh, 0.3)] for i in range(n)], m)
    for fps in (30.0, 60.0):                                            # frozen clock → repaired reversal
        ts = np.r_[np.arange(40), np.full(1200, 40.0), 41 + np.arange(60)] / fps
        add(f"frozen_clock_{fps:g}", [[ShuttleCandidate(100 + 9 * (i % 50), 200 + 4 * (i % 50),
                                                        0.9 if i % 7 else 0.3)] for i in range(len(ts))],
            VideoMeta(fps=fps, width=1920, height=1080, n_frames=len(ts), timestamps=ts))
    for where in ("lead", "trail", "all"):                              # 1200-frame runs at the ends
        ts = {"lead": np.r_[np.zeros(1200), 1 + np.arange(60)], "trail": np.r_[np.arange(60), np.full(1200, 59.0)],
              "all": np.full(1200, 3.0)}[where] / 30.0
        add(f"dup_run_{where}_1200", [[ShuttleCandidate(200 + 7 * (i % 90), 300 + 3 * (i % 90), 0.9)]
                                      if i % 5 == 0 else [] for i in range(len(ts))],
            VideoMeta(fps=30.0, width=1920, height=1080, n_frames=len(ts), timestamps=ts))
    for tv in (np.nan, np.inf):                                        # 1 frame, broken timestamp
        add(f"one_frame_t{tv}", [[ShuttleCandidate(10.0, 10.0, 0.9)]],
            VideoMeta(fps=30.0, width=192, height=144, n_frames=1, timestamps=np.array([tv])))
    ts = np.r_[np.arange(20), 3e9 + np.arange(20)] / 30.0               # 3-year clock jump (5e6 blocks)
    add("clock_jump_1e8s", _edge_cands("strong_line", 40, 1920, 1080, rng),
        VideoMeta(fps=30.0, width=1920, height=1080, n_frames=40, timestamps=ts))
    add("nan_fps_dup_run", _edge_cands("burst", 13, 1920, 1080, rng),
        VideoMeta(fps=float("nan"), width=1920, height=1080, n_frames=13, timestamps=_edge_ts("dup_run", 13, 30.0)))
    venue = Path("/tmp/sc_linker_venue/match_1790485029.pkl")        # real 192×144 venue clip
    if venue.exists():
        import pickle
        cands, meta = pickle.load(open(venue, "rb"))
        add("venue_match_1790485029", round_cands(cands), meta)
    print(f"  {len(out)} edge cases")
    return out


def gap_cases() -> list[dict]:
    """Presentation-clock gaps (video_io.pts_gaps = cuts): synthetic 0.43 s drops and 5 s
    jumps (tests.test_linker._gap_case), the real F3 / F4 robustness repros (detector
    candidates cached by the B agent, skipped if absent) and gap-rule edge clocks. Each case
    carries `pts_gaps` and the linked track's smooth_track / find_kinks outputs."""
    import pickle
    out = []

    def add(name, cands, meta):
        d = case(name, cands, meta, with_t=True)
        tr = L.link_candidates(cands, meta)
        kf, z = L.find_kinks(tr, image_size=(meta.width, meta.height), return_scores=True)
        d["find_kinks"] = {"frames": kf.tolist(), "z": _r(z)}
        d["smooth"] = enc_track(L.smooth_track(tr, image_size=(meta.width, meta.height)))
        d["in_track"] = enc_track(tr, with_t=True)
        d["pts_gaps"] = video_io.pts_gaps(meta.timestamps).tolist()
        out.append(d)

    for seed in (1, 4, 8):
        for kind in ("drop", "jump"):
            _, cands, meta, _ = _gap_case(seed, kind)
            add(f"gap_{kind}_s{seed}", round_cands(cands), meta)
    corpus = Path(__file__).resolve().parents[3] / "tools/robustness/corpus"
    for clip in ("real_jump.mov", "real_drop.mov"):
        pk = Path("/tmp/B_cache") / f"{clip}.pkl"
        if pk.exists() and (corpus / clip).exists():
            cands = pickle.load(open(pk, "rb"))["cands"]
            add(f"gap_real_{clip.split('.')[0]}", cands, video_io.probe(str(corpus / clip)))
    # the rule itself on edge clocks (tiny candidate sets: link must just not crash)
    rng = np.random.default_rng(11)
    clocks = {"at_min_below": np.r_[np.arange(40) / 60, 39 / 60 + 0.2999 + np.arange(40) / 60],
              "at_min_above": np.r_[np.arange(40) / 60, 39 / 60 + 0.3001 + np.arange(40) / 60],
              "slow_10fps_035": np.r_[np.arange(20) * 0.1, 1.9 + 0.35 + np.arange(20) * 0.1],
              "slow_10fps_045": np.r_[np.arange(20) * 0.1, 1.9 + 0.45 + np.arange(20) * 0.1],
              "two_gaps": np.r_[np.arange(30) / 30, 2 + np.arange(30) / 30, 5 + np.arange(30) / 30],
              "gap_at_1": np.r_[0.0, 3.0 + np.arange(30) / 30],
              "gap_at_end": np.r_[np.arange(30) / 30, 9.0],
              "vfr_60_30": np.r_[0, np.cumsum(np.where(rng.random(300) < 0.3, 1 / 30, 1 / 60))]}
    for name, ts in clocks.items():
        n = len(ts)
        cands = [[ShuttleCandidate(round(100 + 4 * i + float(rng.normal(0, 1)), 3), 300.0, 0.9)]
                 if rng.random() < 0.8 else [] for i in range(n)]
        add(f"gap_edge_{name}", cands, VideoMeta(fps=60.0, width=640, height=360, n_frames=n, timestamps=ts))
    print(f"  {len(out)} gap cases")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="score Swift outputs in /tmp/shuttl_linker_parity/swift")
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--edge", action="store_true", help="only (re)write linker_edge.json")
    ap.add_argument("--gaps", action="store_true", help="only (re)write linker_gaps.json")
    args = ap.parse_args()
    if args.gaps:
        dump(gap_cases(), OUT / "linker_gaps.json")
        print(f"wrote linker_gaps.json: {(OUT / 'linker_gaps.json').stat().st_size / 1e6:.2f} MB")
        return
    if args.edge:
        dump(edge_cases(), OUT / "linker_edge.json")
        print(f"wrote linker_edge.json: {(OUT / 'linker_edge.json').stat().st_size / 1e6:.2f} MB")
        return
    if args.score:
        score(FULL / "swift")
        return
    print("synthetic:")
    dump(sim_cases(), OUT / "linker_sim.json")
    print("real (bundle subset):")
    real = real_cases(None if not args.no_full else BUNDLE_CLIPS)
    dump([d for d in real if d["clip"]["match"] + "_" + d["clip"]["clip"] in BUNDLE_CLIPS],
         OUT / "linker_real.json")
    if not args.no_full:
        dump(real, FULL / "real_full.json")
    for p in sorted(OUT.iterdir()):
        print(f"wrote {p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
