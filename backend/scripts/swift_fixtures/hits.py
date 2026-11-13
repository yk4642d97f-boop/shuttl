"""Reference fixtures for the Swift ShuttlHits package (port of tracking/hits.py).

Inputs are stored as zlib/raw-deflate + base64 binary blobs (exact float64, or
float32-quantised where noted: Python outputs are then computed on the quantised
inputs), outputs as plain JSON. Files (Tests/ShuttlHitsTests/Fixtures/):
  hits_unit.json     analytic cases ported from tests/test_hits.py (+ scan lam_v)
  hits_sim_e2e.json  benchmark.py --stage e2e --n 12 (seeds 0-11, 60 fps, 1920x1080,
                     30 % doubles): corrupted detections -> Python linker -> hits,
                     true camera + truth players every 2 frames, plus truth hits
  hits_scen.json     tests/test_hits.py build_scenario subset (float32 inputs)
  hits_real.json     29 TrackNetV2 Test clips: cached WASB candidates -> linker -> hits
  hits_behind.json   phone-like 'behind' camera, 5-rally matches (seeds 0-5, 1920x1080 @60): the
                     case where hidden contacts above the frame occur (hidden candidates asserted)
  hits_learned.json  the same detect_hits with a synthetic linear model installed
  hits_model.json    that model, converted by hits_model_to_json.py

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.hits
"""

from __future__ import annotations

import base64
import dataclasses
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

from app.services.tracking import hits as H
from app.services.tracking import sim as S
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import PlayerObservation, Track2D, VideoMeta
from app.services.tracking.linker import link_candidates
from scripts.swift_fixtures.common import REPO, SWIFT_PACKAGES, TEST_SPLIT, dump

BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from scripts import benchmark as B  # noqa: E402
from tests import test_hits as TH  # noqa: E402

OUT = SWIFT_PACKAGES / "ShuttlHits/Tests/ShuttlHitsTests/Fixtures"
CACHE = Path("/tmp/shuttl_bench_cache")
SIDE_CODE = {"unknown": 0, "near": 1, "far": 2}


# ── binary blobs ─────────────────────────────────────────────────────────────

def blob(a, dtype: str) -> dict:
    a = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = c.compress(a.astype(a.dtype.newbyteorder("<")).tobytes()) + c.flush()
    return {"dtype": np.dtype(dtype).str.lstrip("<|"), "shape": list(a.shape),
            "data": base64.b64encode(raw).decode()}


def q32(a):
    """float32-quantise (keeps NaN), back to float64."""
    return None if a is None else np.asarray(np.asarray(a, float).astype(np.float32), float)


def enc_track(tr: Track2D) -> dict:
    return {"t": blob(tr.t, "f8"), "xy": blob(tr.xy, "f8"),
            "raw_xy": None if tr.raw_xy is None else blob(tr.raw_xy, "f8"),
            "visible": blob(np.asarray(tr.visible, bool), "u1"),
            "interpolated": blob(np.asarray(tr.interpolated, bool), "u1")}


def enc_meta(m: VideoMeta) -> dict:
    return {"fps": float(m.fps), "width": int(m.width), "height": int(m.height), "n_frames": int(m.n_frames),
            "timestamps": blob(m.timestamps, "f8")}


def enc_camera(c: CameraModel | None) -> dict | None:
    if c is None:
        return None
    return {"K": blob(c.K, "f8"), "R": blob(c.R, "f8"), "t": blob(c.t, "f8"), "dist": blob(c.dist, "f8"),
            "image_size": list(c.image_size), "confidence": float(c.confidence),
            "reproj_rmse_px": float(c.reproj_rmse_px)}


def enc_players(pl: list[PlayerObservation] | None) -> dict | None:
    if pl is None:
        return None
    n = len(pl)
    foot = np.full((n, 2), np.nan)
    has_kp = np.zeros(n, bool)
    wr = np.zeros((n, 2, 3))
    for i, o in enumerate(pl):
        if o.foot_world is not None:
            foot[i] = np.asarray(o.foot_world, float)[:2]
        if o.keypoints is not None:
            kp = np.asarray(o.keypoints, float)
            assert kp.shape == (17, 3), kp.shape
            # hits reads only the wrists; the rest must be ignorable (conf <= 0.3)
            other = np.delete(kp, [H.LEFT_WRIST, H.RIGHT_WRIST], axis=0)
            assert np.all(~(other[:, 2] > 0.3)), "non-wrist keypoints would be dropped"
            has_kp[i] = True
            wr[i] = kp[[H.LEFT_WRIST, H.RIGHT_WRIST]]
    return {"frame": blob([o.frame for o in pl], "i4"), "track_id": blob([o.track_id for o in pl], "i4"),
            "bbox": blob(np.array([np.asarray(o.bbox, float)[:4] for o in pl]).reshape(n, 4), "f8"),
            "score": blob([o.score for o in pl], "f8"),
            "side": blob([SIDE_CODE.get(o.side, 0) for o in pl], "u1"),
            "foot": blob(foot, "f8"), "has_kp": blob(has_kp, "u1"), "wrists": blob(wr, "f8")}


# ── outputs ──────────────────────────────────────────────────────────────────

def out_hits(hs) -> list[dict]:
    return [{"frame": int(h.frame), "t": float(h.t), "xy_px": [float(h.xy_px[0]), float(h.xy_px[1])],
             "hitter": h.hitter, "confidence": float(h.confidence),
             "player_track_id": None if h.player_track_id is None else int(h.player_track_id),
             **({"source": h.source} if getattr(h, "source", "kink") != "kink" else {})} for h in hs]


def out_cands(cs) -> list[dict]:
    keys = ("tau", "frame", "lam", "lam_v", "gap_s", "is_start", "static_before", "static_after", "outside",
            "settles", "kin", "reach", "side_llr", "near_tid", "far_tid", "seg", "weak", "hidden")
    rows = []
    for c in cs:
        d = {k: getattr(c, k) for k in keys}
        if c.gap_contact:
            d["gap_contact"] = True                # only when set: older fixtures stay byte-identical
        d.update(xy=c.xy, v_in=c.v_in, v_out=c.v_out, feats=c.feats)
        rows.append(d)
    return rows


def case(name: str, tr: Track2D, meta: VideoMeta, cam=None, players=None, config: dict | None = None,
         with_cands: bool = True, with_scan: bool = False, **extra) -> dict:
    cfg = H.HitConfig(**(config or {}))
    hs = H.detect_hits(tr, meta, cam, players, cfg)
    d = {"name": name, "track": enc_track(tr), "meta": enc_meta(meta), "camera": enc_camera(cam),
         "players": enc_players(players), "config": config or {}, "hits": out_hits(hs)}
    if with_cands:
        ctr = H._prefer_raw(tr) if cfg.use_raw_xy else tr
        d["cands"] = out_cands(H.detect_candidates(ctr, meta, cam, players, cfg))
    if with_scan:
        d["scan"] = scan_dump(tr, meta, cam, cfg)
    d.update(extra)
    d["_src"] = {"args": (tr, meta, cam, players), "config": config or {}}   # not written
    return d


def scan_dump(tr: Track2D, meta: VideoMeta, cam, cfg: H.HitConfig) -> dict:
    """Intermediates of detect_candidates: cleaned obs, sigma, dt, grid lam/lam_v."""
    tr = H._prefer_raw(tr) if cfg.use_raw_xy else tr
    W = meta.width or (cam.image_size[0] if cam is not None else 1280)
    Hh = meta.height or (cam.image_size[1] if cam is not None else 720)
    diag = float(np.hypot(W, Hh))
    oi = H._observed(tr)
    track_t = np.asarray(tr.t, float)
    xy = np.asarray(tr.xy, float)[oi]
    keep, t = H._content_clock(oi, xy, track_t)
    oi, t, xy = oi[keep], t[keep], xy[keep]
    sigma = cfg.sigma_px or H._estimate_sigma(t, xy)
    keep = H._reject_outliers(t, xy, sigma, diag)
    oi, t, xy = oi[keep], t[keep], xy[keep]
    dt = float(np.median(np.diff(track_t)))
    grid = track_t[(track_t >= t[0]) & (track_t <= t[-1])]
    sc = H._scan(t, xy, grid, cfg, sigma)
    return {"obs_frames": oi, "obs_t": t, "sigma": sigma, "dt": dt, "lam": sc["lam"], "lam_v": sc["lam_v"],
            "ok": sc["ok"], "lo": sc["lo"], "hi": sc["hi"], "m": sc["m"]}


def q_track(tr: Track2D) -> Track2D:
    return dataclasses.replace(tr, xy=q32(tr.xy), raw_xy=q32(tr.raw_xy))


def q_players(pl):
    if pl is None:
        return None
    return [dataclasses.replace(o, bbox=q32(o.bbox), foot_world=q32(o.foot_world),
                                keypoints=q32(o.keypoints)) for o in pl]


# ── case generators ──────────────────────────────────────────────────────────

def unit_cases() -> list[dict]:
    """tests/test_hits.py unit tests (same generators, same seeds)."""
    out = []
    tr, meta = TH._arc_track(noise=1.0)
    out.append(case("contract_arc", tr, meta, with_scan=True))
    empty = Track2D(t=np.arange(5) / 30.0, xy=np.full((5, 2), np.nan), conf=np.zeros(5),
                    visible=np.zeros(5, bool), interpolated=np.zeros(5, bool))
    out.append(case("empty", empty, VideoMeta(30.0, 1280, 720, 5, empty.t)))
    for fps in (30.0, 60.0, 120.0):
        tr, meta = TH._arc_track(fps=fps, noise=1.5, seed=int(fps))
        out.append(case(f"single_kink_{int(fps)}", tr, meta, with_scan=fps == 60.0))
    tr, meta = TH._arc_track(fps=60.0, noise=1.5, seed=3)
    out.append(case("occlusion_gap", TH._mask(tr, (tr.t > 0.95) & (tr.t < 1.1)), meta, with_scan=True))
    fps = 60.0
    t = np.arange(int(2.0 * fps)) / fps
    xy = np.zeros((len(t), 2))
    a = t < 1.0
    xy[a] = np.column_stack([500 + 150 * t[a], 200 + 300 * t[a] + 60 * t[a] ** 2])
    xy[~a] = xy[a][-1] + [150 / fps, 300 / fps]
    xy += np.random.default_rng(1).normal(0, 1.5, xy.shape)
    tr = Track2D(t=t, xy=xy, conf=np.ones(len(t)), visible=np.ones(len(t), bool),
                 interpolated=np.zeros(len(t), bool))
    out.append(case("landing_rest", tr, VideoMeta(fps, 1280, 720, len(t), t)))
    tr, meta = TH._arc_track(fps=60.0, noise=1.5, seed=5)
    tr = TH._mask(tr, tr.t < 0.5)
    rng = np.random.default_rng(0)
    xy = tr.xy.copy()
    idx = rng.choice(np.flatnonzero(tr.visible), 4, replace=False)
    xy[idx] = rng.uniform([0, 0], [1280, 720], (4, 2))
    tr = Track2D(t=tr.t, xy=xy, conf=tr.conf, visible=tr.visible, interpolated=tr.interpolated)
    out.append(case("serve_dead_time_outliers", tr, meta, with_scan=True))
    tr25, _ = TH._arc_track(fps=25.0, dur=2.0, noise=1.0, seed=7)
    n30 = 60
    src = np.minimum((np.arange(n30) * 25) // 30, tr25.n - 1)
    t30 = np.arange(n30) / 30.0
    tr = Track2D(t=t30, xy=tr25.xy[src], conf=np.ones(n30), visible=np.ones(n30, bool),
                 interpolated=np.zeros(n30, bool))
    out.append(case("repeated_frames_25p_in_30p", tr, VideoMeta(30.0, 1280, 720, n30, t30), with_scan=True))
    sc = TH.build_scenario(4, 60.0, players=False)
    rng = np.random.default_rng(4)
    t = np.maximum.accumulate(sc.track.t + rng.uniform(-0.3, 0.3, sc.track.n) / 60.0)
    tr = Track2D(t=t, xy=sc.track.xy, conf=sc.track.conf, visible=sc.track.visible,
                 interpolated=sc.track.interpolated)
    out.append(case("variable_frame_rate", tr, VideoMeta(60.0, sc.meta.width, sc.meta.height, tr.n, t.copy()),
                    sc.camera, None, true_frame=sc.true_frame, true_side=sc.true_side))
    tr, meta = TH._arc_track(fps=60.0)
    raw = tr.xy.copy()
    raw[::7] = np.nan
    sm = dataclasses.replace(tr, xy=tr.xy + 3.0, raw_xy=raw)
    out.append(case("prefers_raw_xy", sm, meta))
    out.append(case("raw_xy_disabled", sm, meta, config={"use_raw_xy": False}))
    tr, meta = TH._arc_track(fps=60.0, noise=1.5, seed=9)
    out.append(case("refine_vc_sigma2", tr, meta, config={"refine_model": "vc", "sigma_px": 2.0}))
    return out


def _wrists_only(kp):
    """hits.py reads only the wrist keypoints (_reach_distance): the rest set to conf 0 so the
    fixture encoder (wrists only) is lossless for hits; asserted equal where it is used."""
    if kp is None:
        return None
    kp = np.array(kp, float)
    m = np.ones(len(kp), bool)
    m[[H.LEFT_WRIST, H.RIGHT_WRIST]] = False
    kp[m, 2] = 0.0
    return kp


def gap_cases() -> list[dict]:
    """F4 gap contact: test_hits.test_gap_contact_after_a_pts_gap (a PTS hole ending AT the contact,
    one inside the outgoing / incoming flight, gap contacts off) + the robustness F4 clip pair
    (tools/robustness/repro/F4-dropped-frames-speed: real_drop.mov and its gap-free source) through
    pipeline_v2's own detect -> link -> calibrate -> players stages (camera + players)."""
    out = []
    for fps in (30.0, 60.0):
        tr, meta = TH._arc_track(fps=fps, noise=1.5, seed=1, dur=2.5)
        for nm, (a, b) in (("hidden", (0.55, 1.0 - 0.5 / fps)), ("mid_out", (1.35, 1.8)), ("mid_in", (0.2, 0.7))):
            g, gm = TH._pts_drop(tr, meta, a, b)
            out.append(case(f"gap_{nm}_{int(fps)}", g, gm))
        g, gm = TH._pts_drop(tr, meta, 0.55, 1.0 - 0.5 / fps)
        out.append(case(f"gap_hidden_off_{int(fps)}", g, gm, config={"gap_contacts": False}))
    from app.services.tracking import pipeline_v2
    d = REPO / "tools/robustness/repro/F4-dropped-frames-speed"
    for nm in ("real_drop.mov", "real_reencode.mp4"):
        a = pipeline_v2.analyze_video(str(d / nm), pipeline_v2.PipelineOptions())
        pl = [dataclasses.replace(o, keypoints=_wrists_only(o.keypoints)) for o in a.players or []]
        ref = [(h.frame, h.hitter, h.confidence) for h in H.detect_hits(a.track, a.meta, a.camera, a.players)]
        c = case(f"f4_{nm.split('.')[0]}", a.track, a.meta, a.camera, pl)
        assert [(h["frame"], h["hitter"], h["confidence"]) for h in c["hits"]] == ref, "wrist-only players changed hits"
        print(f"  {nm}: hits {[(h['frame'], h['hitter'], h.get('source', 'kink')) for h in c['hits']]}", flush=True)
        out.append(c)
    return out


def sim_e2e_cases() -> list[dict]:
    """benchmark.py --stage e2e --n 12 (the 0.857 F1 run), exact inputs."""
    args = B.parse_args(["--stage", "e2e", "--n", "12", "--no-json"])
    out = []
    for k in range(args.n):
        seed = args.seed + k
        sim = B.make_sim(args, seed)
        rng = np.random.default_rng([seed, 11])
        cands = S.corrupt_detections(sim, rng, args.sigma, args.miss_rate, True, args.fp_rate)
        tr = link_candidates(cands, sim.meta)
        pl = S.truth_players(sim, every=args.player_step)
        out.append(case(f"e2e_seed{seed}", tr, sim.meta, sim.camera, pl, doubles=bool(sim.doubles),
                        true_frame=[int(h["frame"]) for h in sim.hits], true_side=[h["hitter"] for h in sim.hits]))
    return out


def behind_cases() -> list[dict]:
    """5-rally matches from a low camera behind the near baseline (benchmark
    --style behind --rallies 5), corrupted detections -> linker -> hits."""
    args = B.parse_args(["--stage", "e2e", "--n", "6", "--rallies", "5", "--style", "behind", "--no-json"])
    out, n_hidden = [], 0
    for k in range(args.n):
        seed = args.seed + k
        sim = B.make_sim(args, seed)
        rng = np.random.default_rng([seed, 11])
        cands = S.corrupt_detections(sim, rng, args.sigma, args.miss_rate, True, args.fp_rate)
        tr = link_candidates(cands, sim.meta)
        pl = S.truth_players(sim, every=args.player_step)
        c = case(f"behind_seed{seed}", tr, sim.meta, sim.camera, pl, doubles=bool(sim.doubles),
                 true_frame=[int(h["frame"]) for h in sim.hits], true_side=[h["hitter"] for h in sim.hits])
        n_hidden += sum(1 for x in c["cands"] if x["hidden"])
        out.append(c)
    assert n_hidden > 0, "behind fixture must exercise hidden contacts"
    return out


def scen_cases() -> list[dict]:
    """tests/test_hits.py build_scenario subset; float32-quantised inputs."""
    specs = [(s, 60.0, True, True, {}) for s in range(6)]                       # camera + players
    specs += [(s, 60.0, False, True, {"sides_known": False}) for s in (2, 3)]   # image-order sides
    specs += [(s, 30.0, True, True, {}) for s in (6, 7)] + [(8, 120.0, True, True, {})]
    specs += [(s, 60.0, False, False, {}) for s in (9, 10)]                     # 2-D only
    specs += [(100, 60.0, True, True, {"n_rallies": 3})]                        # multi-rally
    out = []
    for seed, fps, cam, pl, kw in specs:
        sc = TH.build_scenario(seed, fps, players=pl, **kw)
        tr = q_track(sc.track)
        players = q_players(sc.players) if pl else None
        tag = "cam" if cam else "nocam"
        tag += "_pl" if pl else ""
        tag += "_unknownside" if kw.get("sides_known") is False else ""
        tag += f"_{kw['n_rallies']}rallies" if "n_rallies" in kw else ""
        out.append(case(f"scen_s{seed}_{int(fps)}_{tag}", tr, sc.meta, sc.camera if cam else None, players,
                        true_frame=sc.true_frame, true_side=sc.true_side))
    return out


def real_cases() -> list[dict]:
    """29 TrackNetV2 Test clips: cached WASB candidates -> Python linker -> hits (2-D only)."""
    from app.services.tracking import video_io
    out = []
    for clip in B.real_clips():
        meta = video_io.probe(clip["video"])
        cache = CACHE / f"{clip['match']}_{clip['clip']}_wasb.npz"
        if not cache.exists():
            print("  missing cache", cache)
            continue
        z = np.load(cache)
        n = meta.n_frames
        if int(z["n_done"]) < n:
            print("  incomplete cache", cache)
            continue
        cands = B._arr_to_cands(z["cands"][:n])
        tr = link_candidates(cands, meta)
        out.append(case(f"real_{clip['match']}_{clip['clip']}", tr, meta))
    return out


def learned_model() -> dict:
    rng = np.random.default_rng(2026)
    F = len(H.HIT_FEATURES)
    w = rng.normal(0, 0.8, F)
    w[H.HIT_FEATURES.index("gap_s")] = -0.7        # no zero weight (inf * 0) anywhere
    return {"w": w, "b": 0.3, "mu": rng.normal(0, 0.5, F), "sd": rng.uniform(0.5, 2.0, F)}


def learned_cases(sources: dict[str, dict], model_json: Path) -> list[dict]:
    from scripts.swift_fixtures.hits_model_to_json import convert
    out = []
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "m.npz")
        np.savez(p, **learned_model())
        convert(p, str(model_json))
        pz = str(Path(td) / "zero.npz")
        np.savez(pz, w=np.zeros(len(H.HIT_FEATURES)), b=0.0)
        refs = [("hits_sim_e2e.json", "e2e_seed3"), ("hits_sim_e2e.json", "e2e_seed8"),
                ("hits_sim_e2e.json", "e2e_seed10"), ("hits_scen.json", "scen_s1_60_cam_pl"),
                ("hits_scen.json", "scen_s2_60_nocam_pl_unknownside"), ("hits_unit.json", "contract_arc"),
                ("hits_unit.json", "serve_dead_time_outliers"), ("hits_real.json", "real_match1_1_05_02")]
        try:
            for model, mp in (("synthetic", p), ("zero", pz)):
                assert H.load_learned_hit_model(mp)
                for fname, name in refs:
                    if model == "zero" and name != "contract_arc":
                        continue
                    src = sources[fname][name]
                    hs = H.detect_hits(*src["args"], H.HitConfig(**src["config"]))
                    cfg = H.HitConfig(**src["config"])
                    tr = src["args"][0]
                    ctr = H._prefer_raw(tr) if cfg.use_raw_xy else tr
                    cs = H.detect_candidates(ctr, *src["args"][1:], cfg)
                    out.append({"model": model, "source": fname, "name": name, "hits": out_hits(hs),
                                "cands": out_cands(cs)})
        finally:
            H.load_learned_hit_model(None)
    return out


def timing(cases: list[dict]) -> dict:
    import time
    frames, sec = 0, 0.0
    for c in cases:
        src = c["_src"]
        t0 = time.perf_counter()
        H.detect_hits(*src["args"], H.HitConfig(**src["config"]))
        sec += time.perf_counter() - t0
        frames += src["args"][0].n
    return {"frames": frames, "seconds": sec, "ms_per_1000_frames": 1000 * sec / max(frames, 1) * 1000}


def main() -> None:
    import time
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}
    info = {}
    for fname, gen in (("hits_unit.json", unit_cases), ("hits_sim_e2e.json", sim_e2e_cases),
                       ("hits_scen.json", scen_cases), ("hits_real.json", real_cases),
                       ("hits_behind.json", behind_cases), ("hits_gap.json", gap_cases)):
        t0 = time.time()
        cases = gen()
        sources[fname] = {c["name"]: c.pop("_src") | {} for c in cases}
        for c in cases:
            c["_src"] = sources[fname][c["name"]]
        tm = timing(cases)
        for c in cases:
            c.pop("_src")
        p = dump({"cases": cases, "python_timing": tm}, OUT / fname)
        info[fname] = tm
        print(f"{fname}: {len(cases)} cases, {p.stat().st_size / 1e3:.0f} kB, {time.time() - t0:.1f}s, "
              f"python {tm['ms_per_1000_frames']:.0f} ms/1000 frames")
    lc = learned_cases(sources, OUT / "hits_model.json")
    p = dump({"cases": lc}, OUT / "hits_learned.json")
    print(f"hits_learned.json: {len(lc)} cases, {p.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
