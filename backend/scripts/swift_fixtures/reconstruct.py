"""Reference fixtures for the Swift ShuttlReconstruct package (reconstruct3d.py).

Writes iOS/Packages/ShuttlReconstruct/Tests/ShuttlReconstructTests/Fixtures/:
  kernels.json   ODE propagation (+ sensitivities), ground time, residual vector +
                 Jacobian, outlier mask, linear v0, shooting, hypotheses, coarse/fine
                 LM paths on a few problems — debugging-grade parity of the solver path.
  flights.json   single fit_segment flights from the test-suite generator
                 (tests/test_reconstruct3d.py): run_benchmark(64, seed=11) + the seed-5
                 base / feet-prior sets + noise-free, lens-distortion, speed-profile and
                 net-fault cases; inputs, truth and the Python outputs.
  rallies.json   reconstruct_segments on test-suite rallies (test_rally_reconstruction,
                 net fault, degenerate, refine_hit_time) and on sim.py rallies.
  e2e.json.deflate  (raw deflate) benchmark --stage e2e --n 12 (60 fps): sim → corrupt_detections → linker
                 → hits → reconstruct_segments, exactly the benchmark chain; + sim truth.
  real.json.deflate a few real TrackNetV2 Test clips: cached detector candidates → linker,
                 court_calibration camera, hits → reconstruct_segments.
Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.reconstruct
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from app.services.tracking import court_model as cm
from app.services.tracking import reconstruct3d as r3
from app.services.tracking.data_types import Track2D
from scripts.swift_fixtures.common import SWIFT_PACKAGES, TEST_SPLIT, dump, to_jsonable

OUT = SWIFT_PACKAGES / "ShuttlReconstruct/Tests/ShuttlReconstructTests/Fixtures"
RND = 4                                    # px decimals kept for generated pixel inputs


# ── converters ───────────────────────────────────────────────────────────────

def _opt(x):
    return None if x is None else x


def seg_d(s, full_positions: bool = True) -> dict:
    """ShotSegment3D + every non-contract extra reconstruct3d attaches."""
    g = lambda k, d=None: getattr(s, k, d)          # noqa: E731
    pos = np.asarray(s.positions, float)
    d = {
        "start_frame": int(s.start_frame), "end_frame": int(s.end_frame), "t0": s.t0,
        "p0": s.p0, "v0": s.v0, "drag_length": s.drag_length, "rmse_px": s.rmse_px,
        "frames": np.asarray(s.frames, int), "n_positions": int(len(pos)),
        "landing_xy": s.landing_xy, "landing_t": s.landing_t,
        "net_crossing_height": s.net_crossing_height, "confidence": s.confidence,
        "net_crossing_y": g("net_crossing_y"), "end_reason": g("end_reason", "unknown"),
        "hitter_feet_xy": g("hitter_feet_xy"),
        "theta_ref": g("theta_ref"), "t_ref": g("t_ref"), "cov_theta": g("cov_theta"),
        "landing_cov": g("landing_cov"), "landing_sigma_m": g("landing_sigma_m"),
        "speed0_sigma_kmh": g("speed0_sigma_kmh"), "n_obs": g("n_obs"),
        "inlier_frac": g("inlier_frac"), "landing_extrapolated_xy": g("landing_extrapolated_xy"),
        "fit_drag": g("fit_drag"), "t_end": g("t_end"), "contact_observed": g("contact_observed"),
        "contact_gap_s": g("contact_gap_s"), "n_trimmed": g("n_trimmed"),
        "hit_frame": g("hit_frame"), "hit_t_refined": g("hit_t_refined"), "rest_xy": g("rest_xy"),
        "ends_at_pts_gap": g("ends_at_pts_gap", False), "starts_after_pts_gap": g("starts_after_pts_gap", False),
        "landing_sigma_fn": r3.landing_sigma(s),
    }
    if full_positions:
        d["positions"] = pos
    elif len(pos):
        d["positions"] = pos[[0, len(pos) // 2, -1]]
    return d


def track_d(tr: Track2D) -> dict:
    return {"t": tr.t, "xy": tr.xy, "conf": tr.conf, "visible": np.asarray(tr.visible, bool),
            "interpolated": np.asarray(tr.interpolated, bool)}


def rounded_track(tr: Track2D) -> Track2D:
    return Track2D(t=np.asarray(tr.t, float), xy=np.round(np.asarray(tr.xy, float), RND),
                   conf=np.round(np.asarray(tr.conf, float), 4), visible=np.asarray(tr.visible, bool),
                   interpolated=np.asarray(tr.interpolated, bool))


def hits_d(hits) -> list[dict]:
    return [{"frame": int(h.frame), "t": h.t, "xy_px": list(h.xy_px), "hitter": h.hitter,
             "confidence": h.confidence, "player_track_id": h.player_track_id} for h in hits]


def players_near(players, hit_frames, radius: int = 8) -> list:
    """Only observations that _Players.feet can ever pick (within ±radius of a hit frame
    — feet() looks ±6): exact, see the Swift test docs."""
    if not players:
        return []
    hf = np.asarray(sorted(set(int(f) for f in hit_frames)), int)
    keep = []
    for o in players:
        if len(hf) and np.min(np.abs(hf - int(o.frame))) <= radius:
            keep.append(o)
    return keep


def players_d(players) -> list[dict]:
    return [{"frame": int(o.frame), "track_id": int(o.track_id), "bbox": np.asarray(o.bbox, float),
             "score": float(o.score), "side": o.side,
             "foot_world": None if o.foot_world is None else np.asarray(o.foot_world, float)[:2]}
            for o in players]


def meta_d(meta) -> dict:
    return {"fps": meta.fps, "width": int(meta.width), "height": int(meta.height),
            "n_frames": int(meta.n_frames), "timestamps": np.asarray(meta.timestamps, float),
            "rotation_deg": int(getattr(meta, "rotation_deg", 0) or 0)}


def dump_deflate(obj, path: Path) -> Path:
    """Same JSON as common.dump, raw-deflate compressed (RFC 1951) — lossless; Swift reads it
    with NSData.decompressed(using: .zlib). Used for the big e2e / real fixtures."""
    import json
    import zlib
    raw = json.dumps(to_jsonable(obj), allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(co.compress(raw) + co.flush())
    return path


def cpu_ms(fn, *a, **kw):
    t0 = time.process_time()
    out = fn(*a, **kw)
    return out, (time.process_time() - t0) * 1e3


# ── kernels ──────────────────────────────────────────────────────────────────

def kernel_cases(T) -> dict:
    rng = np.random.default_rng(101)
    out: dict = {"propagate": [], "ground_time": [], "problems": [], "outlier_mask": [], "shoot": []}
    for p0, v0, L, times in [
        ([-5.0, 0.5, 2.6], [30.0, -1.0, 8.0], 4.71, [-0.2, -0.05, 0.0, 0.01, 0.1, 0.35, 0.8]),
        ([-1.0, 0.0, 2.8], [80.0, 2.0, -12.0], 3.9, [0.004, 0.02, 0.3, 0.9]),
        ([4.5, -1.5, 0.4], [-12.0, 1.5, 9.0], 5.5, [-0.3, -0.1, 0.05, 1.2]),
    ]:
        th7 = np.r_[p0, v0, np.log(L)]
        ts = np.sort(np.asarray(times, float))
        for ncol in (1, 7, 8):
            P, V, S = np.empty((len(ts), 3)), np.empty((len(ts), 3)), np.zeros((len(ts), 3, 7))
            r3._propagate(np.array(p0, float), np.array(v0, float), L, ts, ncol, P, V, S)
            out["propagate"].append({"p0": p0, "v0": v0, "L": L, "times": ts, "ncol": ncol,
                                     "P": P, "V": V, "S": S[:, :, : max(ncol - 1, 0)]})
        out["ground_time"].append({"p0": p0, "v0": v0, "L": L,
                                   "tg": r3._ground_time(np.array(p0, float), np.array(v0, float), L, 8.0, 0.0)})
    for k in range(3):
        p = np.array([rng.uniform(-6, -1), rng.uniform(-2, 2), rng.uniform(0.5, 2.8)])
        q = np.array([rng.uniform(0, 6), rng.uniform(-2, 2), rng.uniform(0.5, 3)])
        dt = float(rng.uniform(0.15, 0.6))
        v = r3._shoot(p, q, dt, r3.DRAG_LENGTH)
        out["shoot"].append({"p": p, "q": q, "dt": dt, "L": r3.DRAG_LENGTH, "v": v})

    # full solver-path dumps on a few generated flights
    specs = [("smash", "near", "low", 60.0, None), ("clear", "far", "high", 60.0, None),
             ("drop", "near", "high", 30.0, 0.3), ("net", "far", "low", 120.0, None)]
    for shot, half, kind, fps, feet in specs:
        cam = T.make_camera(kind, rng)
        fl = T.sample_flight(rng, shot, half, cam, fps)
        uv = np.round(fl.uv, RND)
        pr = r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half, end="floor", end_t=fl.t_rest)
        if feet is not None:
            pr.start_xy_world = fl.p_hit[:2] + rng.normal(0, feet, 2)
            pr.start_xy_sigma = max(feet, 0.1)
        ok = np.isfinite(uv).all(1)
        to, uvo, wo = fl.t[ok], uv[ok], np.ones(int(ok.sum()))
        out["outlier_mask"].append({"t": to, "uv": uvo, "bad": r3._outlier_mask(to, uvo)})
        for fit_drag in (True, False):
            prob = r3._Problem(to, uvo, wo, cam, pr, fit_drag)
            hyps = r3._hypotheses(prob, to, uvo)
            coarse = [r3._lm(prob, h, max_iter=6, tol=1e-4) for h in hyps]
            th = hyps[0]
            r, J = prob.residuals(th, jac=True)
            fine = r3._lm(prob, hyps[0], max_iter=40)
            best = r3._solve(prob, to, uvo)
            case = {"t": to, "uv": uvo, "w": wo, "camera": cam.to_dict(), "prior": pr,
                    "fit_drag": fit_drag, "t_ref": prob.t_ref, "hyps": hyps,
                    "coarse": [{"th": f.th, "cost": f.cost, "n_iter": f.n_iter} for f in coarse],
                    "fine0": {"th": fine.th, "cost": fine.cost, "n_iter": fine.n_iter},
                    "solve": None if best is None else {"th": best.th, "cost": best.cost, "n_iter": best.n_iter},
                    "r": r, "J": J, "plausible0": bool(r3._plausible(prob, th))}
            # contact-epoch problem (t_ref = contact time, forward integration)
            if fit_drag:
                pc = r3._Problem(to, uvo, wo, cam, pr, fit_drag, t_ref=pr.start_t)
                hc = r3._hypotheses_contact(pc, to, uvo)
                case["contact_hyps"] = hc
                rc, Jc = pc.residuals(hc[0], jac=True) if hc else (None, None)
                case["contact_r"], case["contact_J"] = rc, Jc
            out["problems"].append(case)
    # v0_linear on the first problem's early rays
    pb = out["problems"][0]
    cam = r3.CameraModel.from_dict(pb["camera"])
    o, d = cam.ray(pb["uv"][:8])
    tr = pb["t"][:8] - pb["t"][0]
    p0 = o + 12.0 * d[0]
    out["v0_linear"] = {"p0": p0, "o": o, "d": d, "tr": tr, "L": r3.DRAG_LENGTH,
                        "v": r3._v0_linear(p0, o, np.ascontiguousarray(d), np.ascontiguousarray(tr),
                                           r3.DRAG_LENGTH)}
    return out


# ── single flights ──────────────────────────────────────────────────────────

def flight_case(T, fl, cam, prior, tag: str, **kw) -> dict:
    uv = np.round(fl.uv, RND)
    seg, ms = cpu_ms(r3.fit_segment, fl.t, uv, None, cam, prior)
    return {"tag": tag, "shot": fl.shot, "half": fl.half, "t": fl.t, "uv": uv,
            "camera": cam.to_dict(), "prior": prior, "truth": {
                "p_hit": fl.p_hit, "v_hit": fl.v_hit, "t_hit": fl.t_hit, "land_xy": fl.land_xy,
                "t_land": fl.t_land, "t_rest": fl.t_rest},
            "py_ms": ms, "seg": None if seg is None else seg_d(seg, full_positions=False),
            "call": None if seg is None else r3.call_in_out(seg, doubles=False), **kw}


def bench_flights(T, n: int, seed: int, feet_sigma: float | None, tag: str) -> list[dict]:
    """== tests.test_reconstruct3d.run_benchmark (same RNG stream), exporting the inputs."""
    rng = np.random.default_rng(seed)
    out = []
    kinds = ("low", "high")
    for i in range(n):
        shot = list(T.SHOTS)[i % len(T.SHOTS)]
        half = "near" if (i // len(T.SHOTS)) % 2 == 0 else "far"
        kind = kinds[int(rng.integers(len(kinds)))]
        cam = T.make_camera(kind, rng)
        fps = float(rng.choice([30, 60, 120]))
        fl = T.sample_flight(rng, shot, half, cam, fps)
        pr = r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half, end="floor", end_t=fl.t_rest)
        if feet_sigma is not None:
            pr.start_xy_world = fl.p_hit[:2] + rng.normal(0, feet_sigma, 2)
            pr.start_xy_sigma = max(feet_sigma, 0.1)
        out.append(flight_case(T, fl, cam, pr, tag, cam_kind=kind, fps=fps))
    return out


def special_flights(T) -> list[dict]:
    out = []
    rng = np.random.default_rng(2)                      # test_noise_free_flight_is_recovered
    for shot, half, kind in [("smash", "near", "low"), ("clear", "far", "high"),
                             ("drop", "far", "low"), ("drive", "near", "high")]:
        cam = T.make_camera(kind, rng)
        fl = T.sample_flight(rng, shot, half, cam, 60.0, noise=0.0, miss=0.0, n_out=0)
        pr = r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half, end="floor", end_t=fl.t_rest)
        out.append(flight_case(T, fl, cam, pr, "noise_free"))
    rng = np.random.default_rng(7)                      # test_lens_distortion
    cam = T.make_camera("low", rng)
    cam.dist = np.array([-0.12, 0.03, 0.0, 0.0, 0.0])
    fl = T.sample_flight(rng, "clear", "near", cam, 60.0)
    out.append(flight_case(T, fl, cam, r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half,
                                                       end="floor", end_t=fl.t_rest), "distortion"))
    rng = np.random.default_rng(4)                      # test_speed_profile_and_initial_speed
    cam = T.make_camera("low", rng)
    fl = T.sample_flight(rng, "smash", "near", cam, 120.0)
    c = flight_case(T, fl, cam, r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half, end="floor",
                                                end_t=fl.t_rest), "speed_profile")
    seg = r3.fit_segment(fl.t, np.round(fl.uv, RND), None, cam,
                         r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half, end="floor", end_t=fl.t_rest))
    ts, kmh = r3.speed_profile(seg)
    c["speed_profile"] = {"t": ts, "kmh": kmh, "contact_kmh_t0": r3.contact_speed_kmh(seg, seg.t0)}
    out.append(c)
    rng = np.random.default_rng(3)                      # test_net_fault_is_detected (single fit)
    for kind in ("high", "low"):
        cam = T.make_camera(kind, rng)
        track, hits, meta, TT, P = T._net_fault_track(rng, cam, 60.0)
        pre = (P[:, 0] < -0.05) & track.visible
        pr = r3.SegmentPrior(start_t=0.0, expected_side="near")
        uv = np.round(track.xy[pre], RND)
        seg, ms = cpu_ms(r3.fit_segment, TT[pre], uv, None, cam, pr)
        out.append({"tag": "net_fault_single", "shot": "drop", "half": "near", "t": TT[pre], "uv": uv,
                    "camera": cam.to_dict(), "prior": pr, "truth": None, "py_ms": ms,
                    "seg": None if seg is None else seg_d(seg, full_positions=False),
                    "call": None if seg is None else r3.call_in_out(seg, doubles=False)})
    return out


# ── rallies ─────────────────────────────────────────────────────────────────

def rally_case(tag: str, track, hits, cam, meta, players, truth=None) -> dict:
    track = rounded_track(track)
    segs, ms = cpu_ms(r3.reconstruct_segments, track, hits, cam, meta, players)
    hf = [h.frame for h in hits]
    return {"tag": tag, "track": track_d(track), "hits": hits_d(hits), "camera": cam.to_dict(),
            "meta": meta_d(meta), "players": players_d(players_near(players, hf)),
            "py_ms": ms, "segs": [seg_d(s, full_positions=False) for s in segs], "truth": truth}


def rally_cases(T) -> list[dict]:
    out = []
    rng = np.random.default_rng(21)                     # test_rally_reconstruction
    for kind, fps in [("low", 60.0), ("high", 30.0), ("broadcast", 60.0)]:
        cam = T.make_camera(kind, rng)
        R = T.make_rally(rng, cam, fps, n_shots=4)
        truth = [{"frame": h.frame, "t": s.t, "v": s.v, "p": s.p,
                  "land_xy": None if s.land_xy is None else s.land_xy} for h, s in zip(R.hits, R.shots)]
        out.append(rally_case(f"rally_{kind}", R.track, R.hits, cam, R.meta, R.players, truth))
    rng = np.random.default_rng(3)                      # test_net_fault_is_detected (rally)
    for kind in ("high", "low"):
        cam = T.make_camera(kind, rng)
        track, hits, meta, TT, P = T._net_fault_track(rng, cam, 60.0)
        out.append(rally_case(f"net_fault_{kind}", track, hits, cam, meta, None))
    rng = np.random.default_rng(0)                      # test_degenerate_inputs
    cam = T.make_camera("low", rng)
    R = T.make_rally(rng, cam, 30.0, n_shots=2)
    out.append(rally_case("degenerate_no_players", R.track, R.hits, cam, R.meta, None))
    # sim.py rallies (test_on_sibling_simulator)
    from app.services.tracking import sim
    for seed, style in [(1, "behind"), (2, "elevated_side"), (3, "broadcast")]:
        rng = np.random.default_rng(seed)
        cam = sim.random_camera(rng, style=style)
        simr = sim.simulate_rally(seed, camera=cam, fps=60, n_shots=4)
        truth = [{"frame": s.frame, "speed0_kmh": s.speed0_kmh, "outcome": s.outcome,
                  "landing_xy": s.landing_xy} for s in simr.shots]
        out.append(rally_case(f"sim_{style}", T._sim_track(simr, rng), T._sim_hits(simr), cam, simr.meta,
                              sim.truth_players(simr, every=2), truth))
    return out


def refine_cases(T) -> list[dict]:
    out = []
    rng = np.random.default_rng(8)                      # test_refine_hit_time
    for _ in range(3):
        cam = T.make_camera("low", rng)
        R = T.make_rally(rng, cam, 120.0, n_shots=2)
        if len(R.shots) < 2:
            continue
        h = R.hits[1]
        track = rounded_track(R.track)
        (f, tc), ms = cpu_ms(r3.refine_hit_time, track, h.frame + 3, cam)
        out.append({"track": track_d(track), "camera": cam.to_dict(), "approx_frame": h.frame + 3,
                    "frame": f, "t_c": tc, "true_t": R.shots[1].t, "py_ms": ms})
    return out


def gap_cases(T) -> dict:
    """Presentation-clock gaps: test-suite rallies with a 0.43 s drop / 5 s jump in the 2nd
    shot (tests.test_reconstruct3d._with_gap), with the hit set of the gap clip and with an
    extra hit inside the gap (the contact a gap-aware hit detector would report); the real
    F3 / F4 robustness repros (B-agent stage cache, skipped if absent); refine_hit_time next
    to a gap."""
    import dataclasses
    import pickle
    from app.services.tracking import hits as H
    from app.services.tracking import linker, video_io
    rallies, refine = [], []
    rng = np.random.default_rng(33)                      # test_no_flight_fitted_across_pts_gap
    for kind in ("drop", "jump"):
        for camkind, fps in [("low", 60.0), ("high", 30.0), ("broadcast", 60.0)]:
            cam = T.make_camera(camkind, rng)
            R0 = T.make_rally(rng, cam, fps, n_shots=4)
            R, g = T._with_gap(R0, kind)
            rallies.append(rally_case(f"gap_{kind}_{camkind}", R.track, R.hits, cam, R.meta, R.players))
            t = R.track.t
            h1 = R.hits[1]
            ins = dataclasses.replace(h1, frame=g, t=float(0.5 * (t[g - 1] + t[g])),
                                      hitter="far" if h1.hitter == "near" else "near")
            hs = sorted([h for h in R.hits if h.frame != g] + [ins], key=lambda h: h.t)
            rallies.append(rally_case(f"gap_{kind}_{camkind}_hit_in_gap", R.track, hs, cam, R.meta, R.players))
    rng = np.random.default_rng(34)                      # test_refine_hit_time_stays_within_gap_span
    cam = T.make_camera("low", rng)
    R0 = T.make_rally(rng, cam, 60.0, n_shots=4)
    R, g = T._with_gap(R0, "jump", frac=0.02)
    track = rounded_track(R.track)
    for h in R.hits:
        (f, tc), ms = cpu_ms(r3.refine_hit_time, track, h.frame, cam)
        refine.append({"track": track_d(track), "camera": cam.to_dict(), "approx_frame": h.frame,
                       "frame": f, "t_c": tc, "true_t": float("nan"), "py_ms": ms})
    corpus = Path(__file__).resolve().parents[3] / "tools/robustness/corpus"
    for clip in ("real_jump.mov", "real_drop.mov"):
        pk = Path("/tmp/B_cache") / f"{clip}.pkl"
        if not (pk.exists() and (corpus / clip).exists()):
            continue
        c = pickle.load(open(pk, "rb"))
        meta = video_io.probe(str(corpus / clip))
        tr = linker.link_candidates(c["cands"], meta)
        hs = H.detect_hits(tr, meta, c["camera"], c["players"] or None)
        rallies.append(rally_case(f"gap_real_{clip.split('.')[0]}", tr, hs, c["camera"], meta, c["players"]))
    print(f"  gaps: {len(rallies)} rallies, {len(refine)} refine")
    return {"rallies": rallies, "refine": refine}


# ── benchmark e2e ───────────────────────────────────────────────────────────

def e2e_cases() -> dict:
    from app.services.tracking import hits as H
    from app.services.tracking import linker
    from app.services.tracking import metrics as M
    from app.services.tracking import sim as S
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.benchmark import _seg_d, _true_shot_d, make_sim, parse_args
    args = parse_args(["--stage", "e2e", "--n", "12", "--no-json"])
    cases, P_all, T_all = [], [], []
    for k in range(args.n):
        seed = args.seed + k
        simr = make_sim(args, seed)
        rng = np.random.default_rng([seed, 11])
        cands = S.corrupt_detections(simr, rng, args.sigma, args.miss_rate, True, args.fp_rate)
        players = S.truth_players(simr, every=args.player_step)
        track = linker.link_candidates(cands, simr.meta)
        hits = H.detect_hits(track, simr.meta, simr.camera, players)
        segs, ms = cpu_ms(r3.reconstruct_segments, track, hits, simr.camera, simr.meta, players)
        P = [_seg_d(s) for s in segs]
        Tt = [_true_shot_d(s) for s in simr.shots]
        P_all.append((P, Tt, k))
        cases.append({"seed": seed, "track": track_d(track), "hits": hits_d(hits),
                      "camera": simr.camera.to_dict(), "meta": meta_d(simr.meta),
                      "players": players_d(players_near(players, [h.frame for h in hits])),
                      "py_ms": ms, "segs": [seg_d(s, full_positions=False) for s in segs],
                      "truth": [{"start_frame": d["start_frame"], "p0": d["p0"], "v0": d["v0"],
                                 "landing_xy": d["landing_xy"],
                                 "net_crossing_height": d["net_crossing_height"]} for d in Tt]})
    off = 100_000
    Pp = [dict(d, start_frame=d["start_frame"] + k * off) for P, _, k in P_all for d in P]
    Tp = [dict(d, start_frame=d["start_frame"] + k * off) for _, Tt, k in P_all for d in Tt]
    met = M.segment_metrics(Pp, Tp, args.tol_frames + 1)
    print("e2e reconstruct3d:", {k: met[k] for k in ("n_pred", "matched", "landing_err_m_median",
                                                     "landing_err_m_n", "speed_err_rel_median")})
    return {"cases": cases, "tol_frames": args.tol_frames + 1, "py_metrics": met}


# ── real clips ──────────────────────────────────────────────────────────────

REAL_CLIPS = [("match1", "1_05_02"), ("match2", "1_04_04"), ("match3", "1_05_02"), ("match1", "1_07_03"),
              ("match3", "1_09_12")]


def real_cases() -> list[dict]:
    from app.services.tracking import court_calibration, linker, video_io
    from app.services.tracking import hits as H
    from app.services.tracking.data_types import ShuttleCandidate
    out = []
    for match, clip in REAL_CLIPS:
        video = TEST_SPLIT / match / "video" / f"{clip}.mp4"
        cache = Path("/tmp/shuttl_bench_cache") / f"{match}_{clip}_tracknetv3.npz"
        if not video.exists() or not cache.exists():
            print("skip (missing)", match, clip)
            continue
        meta = video_io.probe(str(video))
        z = np.load(cache)
        arr = z["cands"][: meta.n_frames]
        cands = [[ShuttleCandidate(float(x), float(y), float(s)) for x, y, s in row if np.isfinite(s)]
                 for row in arr]
        track = linker.link_candidates(cands, meta)
        frames = [f for _, f in video_io.sample_frames(str(video), 24, meta.n_frames)]
        cam = court_calibration.calibrate_from_frames(frames, (meta.width, meta.height))
        if cam is None:
            print("skip (no camera)", match, clip)
            continue
        hits = H.detect_hits(track, meta, cam, None)
        segs, ms = cpu_ms(r3.reconstruct_segments, track, hits, cam, meta, None)
        spread, same_structure = self_spread(track, hits, cam, meta, segs)
        print(f"real {match}/{clip}: {meta.n_frames} frames, {len(hits)} hits, {len(segs)} segs, {ms:.0f} ms; "
              f"self-spread > 1e-6 on {int(np.sum(spread > 1e-6))} segs (max {spread.max(initial=0):.1e}), "
              f"structure stable: {same_structure}")
        out.append({"clip": f"{match}/{clip}", "track": track_d(track), "hits": hits_d(hits),
                    "camera": cam.to_dict(), "meta": meta_d(meta), "players": [], "py_ms": ms,
                    "segs": [seg_d(s, full_positions=False) for s in segs],
                    "self_spread": spread, "self_structure_stable": same_structure})
    return out


def _structure(segs) -> list:
    return [(int(s.start_frame), int(s.end_frame), getattr(s, "end_reason", "unknown"),
             getattr(s, "hit_frame", None), s.landing_xy is None, s.net_crossing_height is None) for s in segs]


def _cont(s) -> np.ndarray:
    """Continuous outputs whose reproducibility is measured: p0, v0, t_end, hit_t_refined,
    landing xy (0 if none)."""
    lx = (0.0, 0.0) if s.landing_xy is None else s.landing_xy
    return np.r_[s.p0, s.v0, getattr(s, "t_end", s.t0), getattr(s, "hit_t_refined", s.t0), lx]


def _ulp_camera(cam, rng):
    """The camera with every ray direction and projected pixel off by ±1 ulp — the size of
    the Swift-simd vs numpy-BLAS/LAPACK difference of ShuttlCore.CameraModel.ray/project."""
    import copy
    import types
    c = copy.deepcopy(cam)
    base_ray, base_proj = type(cam).ray, type(cam).project
    eps = np.finfo(float).eps

    def ray(self, uv):
        o, d = base_ray(self, uv)
        return o, d * (1 + rng.choice([-1, 1], d.shape) * eps)

    def project(self, pts):
        uv = base_proj(self, pts)
        return uv * (1 + rng.choice([-1, 1], uv.shape) * eps)

    c.ray, c.project = types.MethodType(ray, c), types.MethodType(project, c)
    return c


def self_spread(track, hits, cam, meta, base) -> tuple[np.ndarray, bool]:
    """How reproducible the reference itself is: per base segment, the max relative change
    of p0, v0, t_end, hit_t_refined and landing over reruns with perturbed inputs — track
    xy and camera R/t/K each scaled by (1 ± 1 ulp) and by (1 ± 1e-12), and camera rays /
    projections off by ±1 ulp (the Swift-vs-numpy camera difference), 3 draws each — and
    whether the discrete structure (count, frames, end_reason, landing/net presence) ever
    moved. Real broadcast clips have ill-conditioned flights on which the LM amplifies
    round-off (the Swift-simd vs numpy-BLAS ulp differences act the same way); the Swift
    test holds the values of self-stable segments (spread <= 1e-6) to 1e-6 and the
    structure of all of them."""
    import copy
    rng = np.random.default_rng(0)
    ref = np.array([_cont(s) for s in base]).reshape(-1, 10)
    spread = np.zeros(len(base))
    stable = True
    eps = np.finfo(float).eps
    for rel in (eps, 1e-12, None):
        for k in range(6):
            tr, cm2 = track, cam
            if rel is None:
                if k >= 3:
                    continue
                cm2 = _ulp_camera(cam, rng)
            elif k < 3:
                tr = Track2D(t=track.t, xy=track.xy * (1 + rng.choice([-1, 1], track.xy.shape) * rel),
                             conf=track.conf, visible=track.visible, interpolated=track.interpolated)
            else:
                cm2 = copy.deepcopy(cam)
                cm2.R = cm2.R * (1 + rng.choice([-1, 1], cm2.R.shape) * rel)
                cm2.K = cm2.K * (1 + rng.choice([-1, 1], cm2.K.shape) * rel)
                cm2.t = cm2.t * (1 + rng.choice([-1, 1], cm2.t.shape) * rel)
            segs = r3.reconstruct_segments(tr, hits, cm2, meta, None)
            if _structure(segs) != _structure(base):
                stable = False
                spread[:] = np.inf
                continue
            cur = np.array([_cont(s) for s in segs]).reshape(-1, 10)
            spread = np.maximum(spread, np.max(np.abs(cur - ref) / np.maximum(1, np.abs(ref)), axis=1))
    return spread, stable


def main() -> None:
    import tests.test_reconstruct3d as T
    T.r3.fit_segment(np.arange(10) / 30, np.random.default_rng(0).random((10, 2)) * 100 + 500, None,
                     T.make_camera("low", np.random.default_rng(0)))          # numba warm-up
    which = set(sys.argv[1:]) or {"kernels", "flights", "rallies", "e2e", "real", "gaps"}
    if "kernels" in which:
        print("wrote", dump(kernel_cases(T), OUT / "kernels.json"))
    if "flights" in which:
        fl = (bench_flights(T, 64, 11, None, "bench64") + bench_flights(T, 24, 5, None, "base24")
              + bench_flights(T, 24, 5, 0.3, "feet24") + special_flights(T))
        ms = np.array([c["py_ms"] for c in fl])
        print(f"flights: {len(fl)}  python fit_segment CPU ms median {np.median(ms):.1f} p90 {np.percentile(ms, 90):.1f}")
        print("wrote", dump(fl, OUT / "flights.json"))
    if "rallies" in which:
        print("wrote", dump({"rallies": rally_cases(T), "refine": refine_cases(T)}, OUT / "rallies.json"))
    if "gaps" in which:
        print("wrote", dump(gap_cases(T), OUT / "gaps.json"))
    if "e2e" in which:
        (OUT / "e2e.json").unlink(missing_ok=True)
        print("wrote", dump_deflate(e2e_cases(), OUT / "e2e.json.deflate"))
    if "real" in which:
        print("wrote", dump_deflate(real_cases(), OUT / "real.json.deflate"))


if __name__ == "__main__":
    main()
