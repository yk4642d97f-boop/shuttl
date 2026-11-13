"""Reference fixtures for the Swift ShuttlRally package (rally.py, shots.py,
serialize.summarize / per-rally dict shape).

Cases
  sim_e2e      benchmark.py --stage e2e --n 12 (seeds 0-11, one rally each, 60 fps,
               random camera style, 30 % doubles): corrupted detections -> linker ->
               hits -> reconstruct3d, then rally + shots. Truth intervals included so
               the Swift tests can reproduce the benchmark rally F1.
  sim_match    same chain on sim.simulate_match (multi-rally, singles + doubles):
               feeds / grouping / next-serve inference / score_match.
  sim_corrupt  test_rally.test_validate_with_sim_simulate_match inputs (truth hits +
               segments, corrupted).
  world        test_rally's own generator (feeds, carries, spurious hits), 3-D and 2-D only.
  real         TrackNetV2 Test clips (corrected labels): label track -> hits.detect_hits
               with the automatically calibrated camera -> reconstruct3d -> rally + shots
               (clips where calibration fails run 2-D only).

Players are reduced losslessly: `players_rally` (subsampled, same detect_doubles
verdict, asserted to give identical rallies) and `players_shots` (every observation
within +-8 frames of a hit: exactly what shots._PlayerIndex.at can see, asserted).
`landing_sigma_m` is the effective per-segment sigma rally._landing_sigma would use
before the extrapolation term (None -> config default).

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.rally [--only sim_e2e,real]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.services.tracking import rally as R
from app.services.tracking import serialize as SER
from app.services.tracking import shots as SH
from app.services.tracking import sim as S
from app.services.tracking.data_types import HitEvent, PlayerObservation, RallyV2, ShotEventV2, Track2D
from scripts.swift_fixtures.common import SWIFT_PACKAGES, TEST_SPLIT, to_jsonable

OUT = SWIFT_PACKAGES / "ShuttlRally/Tests/ShuttlRallyTests/Fixtures"
CACHE = Path("/tmp/shuttl_rally_fixture_cache")


def _sigma(seg) -> float | None:
    s = getattr(seg, "landing_sigma_m", None)
    if s is None or not math.isfinite(s):
        try:
            from app.services.tracking import reconstruct3d as r3
            s = r3.landing_sigma(seg) if hasattr(r3, "landing_sigma") else None
        except Exception:
            s = None
    return None if s is None or not math.isfinite(s) else float(s)


def _same(a, b) -> bool:
    return json.dumps(to_jsonable(a), sort_keys=True) == json.dumps(to_jsonable(b), sort_keys=True)


def _players_rally(players, track, hits, segs, meta, camera):
    full = R.segment_rallies(track, hits, segs, meta, camera, players=players)
    if not players:
        return players, full
    for step in (30, 20, 10, 4, 2):
        sub = [p for p in players if p.frame % step == 0]
        if R.detect_doubles(sub) != R.detect_doubles(players):
            continue
        out = R.segment_rallies(track, hits, segs, meta, camera, players=sub)
        if _same(out, full):
            return sub, full
    return players, full


def _players_shots(players, hits):
    if not players:
        return players
    frames = np.array(sorted({int(h.frame) for h in hits}), int)
    if not len(frames):
        return []
    out = []
    for p in players:
        k = np.searchsorted(frames, p.frame)
        near = any(0 <= j < len(frames) and abs(int(frames[j]) - p.frame) <= 8 for j in (k - 1, k))
        if near:
            out.append(p)
    return out


def _strip_seg(d: dict | None) -> dict | None:
    if d is not None:
        d["frames"], d["positions"] = [], []
    return d


def _compact(case: dict, keep_dicts: bool) -> dict:
    """JSON-ready, size-reduced case: meta.timestamps == track.t (dropped); expected
    rallies carry segments without frames/positions (the Swift comparison checks
    segment identity by start_frame); input segments keep frames/positions only
    when the serialized rally dicts (which read them) are part of the case."""
    out = to_jsonable(case)
    out["meta"]["timestamps"] = None
    for key in ("rallies", "shots"):
        for r in out[key]:
            for sh in r["shots"]:
                _strip_seg(sh["segment"])
    if not keep_dicts:
        out.pop("rally_dicts")
        for sg in out["segments"]:
            _strip_seg(sg)
    return out


def dump_z(obj, path: Path) -> Path:
    """Raw-DEFLATE compressed compact JSON (Swift: NSData.decompressed(using: .zlib))."""
    raw = json.dumps(to_jsonable(obj), allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    path.write_bytes(co.compress(raw) + co.flush())
    return path


def make_case(name, track, hits, segs, meta, camera, players, truth=None, extra=None,
              keep_dicts: bool = False) -> dict:
    t0 = time.perf_counter()
    p_rally, rallies = _players_rally(players, track, hits, segs, meta, camera)
    t_rally = time.perf_counter() - t0
    p_shots = _players_shots(players, hits)
    t0 = time.perf_counter()
    typed = [SH.classify_shots(r, camera, p_shots) for r in rallies]
    t_shots = time.perf_counter() - t0
    if players:
        typed_full = [SH.classify_shots(r, camera, players) for r in rallies]
        assert _same(typed, typed_full), f"{name}: players_shots reduction changed classify_shots"
    w, h = meta.width or 1, meta.height or 1
    case = {
        "name": name, "meta": meta, "track": track, "hits": hits, "segments": segs,
        "landing_sigma_m": [_sigma(s) for s in segs], "camera": camera,
        "players_rally": p_rally, "players_shots": p_shots,
        "doubles": R.detect_doubles(players),
        "rallies": rallies, "shots": typed,
        "score": R.score_match(typed), "summary": SER.summarize(typed),
        "rally_dicts": [SER._jsonable(SER._rally(r, w, h, camera)) for r in typed],
        "py_seconds": {"segment_rallies": t_rally, "classify_shots": t_shots},
        # rally.refine_hitters on the input hits (joint side decode, for a reconstruct3d re-run)
        "refined_sides": [h.hitter for h in R.refine_hitters(track, hits, segs, meta, camera, players)[0]],
    }
    if truth is not None:
        case["truth"] = truth
    if extra:
        case.update(extra)
    return _compact(case, keep_dicts)


# ── sim ──────────────────────────────────────────────────────────────────────

def _chain(sim, rng, player_step=2):
    from app.services.tracking import hits as HM, linker, reconstruct3d
    cands = S.corrupt_detections(sim, rng, 1.5, 0.1, True, 0.03)
    players = S.truth_players(sim, every=player_step)
    track = linker.link_candidates(cands, sim.meta)
    hits = HM.detect_hits(track, sim.meta, sim.camera, players)
    segs = reconstruct3d.reconstruct_segments(track, hits, sim.camera, sim.meta, players)
    return track, hits, segs, players


def sim_e2e() -> list[dict]:
    from scripts import benchmark
    args = benchmark.parse_args(["--stage", "e2e", "--n", "12", "--no-json"])
    cases = []
    for seed in range(12):
        sim = benchmark.make_sim(args, seed)
        rng = np.random.default_rng([seed, 11])
        track, hits, segs, players = _chain(sim, rng, args.player_step)
        truth = [[int(r.start_frame), int(r.end_frame)] for r in sim.rallies]
        cases.append(make_case(f"sim_e2e_{seed}", track, hits, segs, sim.meta, sim.camera, players, truth,
                               {"sim_doubles": bool(sim.doubles)}, keep_dicts=True))
        print(f"  sim_e2e seed {seed}: frames={sim.n} hits={len(hits)} segs={len(segs)} "
              f"rallies={len(cases[-1]['rallies'])}", flush=True)
    return cases


def sim_match() -> list[dict]:
    cases = []
    for seed, n_r, doubles in ((101, 4, False), (102, 4, True), (104, 4, True)):
        rng = np.random.default_rng([seed, 7])
        cam = S.random_camera(rng, (1920, 1080), "random")
        sim = S.simulate_match(seed, n_r, camera=cam, fps=60, doubles=doubles)
        track, hits, segs, players = _chain(sim, np.random.default_rng([seed, 11]))
        truth = [[int(r.start_frame), int(r.end_frame)] for r in sim.rallies]
        cases.append(make_case(f"sim_match_{seed}", track, hits, segs, sim.meta, cam, players, truth,
                               {"sim_doubles": doubles}))
        print(f"  sim_match seed {seed}: frames={sim.n} rallies={len(cases[-1]['rallies'])}/{n_r}", flush=True)
    return cases


def sim_corrupt() -> list[dict]:
    """test_rally.test_validate_with_sim_simulate_match (corrupted branch)."""
    cases = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        cam = S.random_camera(rng, (1920, 1080), style="behind" if seed % 2 == 0 else "broadcast")
        sr = S.simulate_match(seed, 6, camera=cam, fps=60)
        px, mask = sr.image_truth(occluded=False, resting=True)
        vis = mask & (rng.random(sr.n) > 0.1)
        xy = np.where(vis[:, None], px + rng.normal(0, 1.5, px.shape), np.nan)
        track = Track2D(t=sr.timestamps.copy(), xy=xy, conf=vis.astype(float), visible=vis,
                        interpolated=np.zeros(sr.n, bool))
        hits, segs = S.truth_hits(sr), S.truth_segments(sr)
        truth = [(r.start_frame, r.end_frame) for r in sr.rallies]
        keep = rng.random(len(hits)) > 0.05
        c_hits = [replace(h, hitter=("far" if h.hitter == "near" else "near") if rng.random() < 0.02 else h.hitter,
                          confidence=float(rng.uniform(0.55, 0.98)))
                  for h, k in zip(hits, keep) if k]
        dead = [(truth[i][1] + 60, truth[i + 1][0] - 60) for i in range(len(truth) - 1)]
        for a, b in [d for d in dead if d[1] > d[0]][:2]:
            f = int(rng.integers(a, b))
            c_hits.append(HitEvent(frame=f, t=float(sr.timestamps[f]), xy_px=(0.0, 0.0),
                                   hitter=str(rng.choice(("near", "far"))), confidence=0.3))
        c_segs = [s for s, k in zip(segs, keep) if k and rng.random() > 0.05]
        cases.append(make_case(f"sim_corrupt_{seed}", track, c_hits, c_segs, sr.meta, cam, None,
                               [list(map(int, t)) for t in truth]))
        print(f"  sim_corrupt seed {seed}: rallies={len(cases[-1]['rallies'])}/{len(truth)}", flush=True)
        if seed == 0:
            # robustness F3: every final flight that reports a floor landing comes down at (-11.1, -0.3) m
            # (fitted across a timestamp jump) -> the implausible-landing gate of the 3-D outcome
            f3 = [replace(s, landing_xy=(-11.1, -0.3)) if s.landing_xy is not None else s for s in c_segs]
            for a, b in zip(f3, c_segs):
                for k in ("hit_frame", "end_reason", "landing_sigma_m", "net_crossing_y"):
                    if hasattr(b, k):
                        setattr(a, k, getattr(b, k))
            cases.append(make_case("sim_corrupt_0_f3_landing", track, c_hits, f3, sr.meta, cam, None,
                                   [list(map(int, t)) for t in truth]))
            src = [r.outcome_source for r in R.segment_rallies(track, c_hits, f3, sr.meta, cam)]
            assert "implausible_landing" in src, src
            print(f"  sim_corrupt seed 0 F3 landings: outcome sources {src}", flush=True)
    return cases


def world() -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests import test_rally as TR
    cases = []
    for seed, fps, with_3d in ((0, 60.0, True), (1, 30.0, True), (2, 60.0, False)):
        w = TR._world(seed, 10)
        c = TR._render(w, fps, seed)
        rng = np.random.default_rng(seed + 12345)
        hits, src = TR._emulate_hits(c, rng)
        segs = TR._emulate_segments(c, hits, src, rng) if with_3d else []
        players = TR._players_doubles(c, seed == 1) if seed == 1 else None
        cam = c.camera if with_3d else None
        cases.append(make_case(f"world_{seed}_{int(fps)}{'' if with_3d else '_2d'}", c.track, hits, segs,
                               c.meta, cam, players, [list(map(int, t)) for t in c.truth]))
        print(f"  world seed {seed}: frames={c.meta.n_frames} rallies={len(cases[-1]['rallies'])}/{len(c.truth)}",
              flush=True)
    return cases



CODEC_SRC = Path("/tmp/accC/codec2")          # read-only copies of E's 45-min H.264 / HEVC runs (v2)


def codec_cases() -> list[dict] | None:
    """Addendum 3: frames 76700-81000 of both 45-min runs (rally 77495 + the next one, whose missed serve
    is the next-serve evidence). H.264: a 3-D call (far, 0.67) contradicted by the next serve -> capped;
    HEVC: overruled from an inferred serve -> capped. None when the sources are gone (keep the fixture)."""
    from scripts.accuracy.codec_ab import load, load_players
    if not all((CODEC_SRC / f).exists() for f in ("h264.json", "hevc.json", "h264_players.bin", "hevc_players.bin")):
        return None
    a, b = 76700, 81000
    cases = []
    for nm in ("h264", "hevc"):
        d = load(str(CODEC_SRC / f"{nm}.json"))
        pl = load_players(str(CODEC_SRC / f"{nm}_players.bin"))
        tr = d["track"]
        sl = slice(a, b)
        track = Track2D(t=np.asarray(tr.t)[sl], xy=np.asarray(tr.xy)[sl], conf=np.asarray(tr.conf)[sl],
                        visible=np.asarray(tr.visible)[sl], interpolated=np.asarray(tr.interpolated)[sl])
        if getattr(tr, "raw_xy", None) is not None:
            track.raw_xy = np.asarray(tr.raw_xy)[sl]
        meta = replace(d["meta"], n_frames=b - a, timestamps=np.asarray(d["meta"].timestamps)[sl])
        hits = [replace(h, frame=h.frame - a) for h in d["hits"] if a <= h.frame < b]
        segs = []
        for s0 in d["segs"]:
            if a <= s0.start_frame and s0.end_frame < b:
                g = replace(s0, start_frame=s0.start_frame - a, end_frame=s0.end_frame - a,
                            frames=np.asarray(s0.frames) - a)
                for k in ("landing_sigma_m",):
                    if hasattr(s0, k):
                        setattr(g, k, getattr(s0, k))
                if hasattr(s0, "hit_frame"):
                    g.hit_frame = int(s0.hit_frame) - a
                segs.append(g)
        players = [replace(o, frame=o.frame - a) for o in pl if a <= o.frame < b]
        c = make_case(f"codec_{nm}_77495", track, hits, segs, meta, d["camera"], players)
        r0 = c["rallies"][0]
        print(f"  codec {nm}: rallies {[(r['start_frame'] + a, r['winner'], round(r['outcome_confidence'], 3), r['outcome_source']) for r in c['rallies']]}",
              flush=True)
        assert r0["outcome_confidence"] == 0.5, r0
        cases.append(c)
    return cases

# ── real clips ───────────────────────────────────────────────────────────────

def _label_track(path: Path, n: int, t: np.ndarray) -> Track2D:
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    xy, vis = np.full((n, 2), np.nan), np.zeros(n, bool)
    f = rows[:, 0].astype(int)
    ok = (f >= 0) & (f < n)
    f, r = f[ok], rows[ok]
    v = r[:, 1] > 0
    vis[f[v]] = True
    xy[f[v]] = r[v, 2:4]
    return Track2D(t=t.copy(), xy=xy, conf=vis.astype(float), visible=vis, interpolated=np.zeros(n, bool))


def real(limit: int | None = None) -> list[dict]:
    from app.services.tracking import court_calibration as CC, hits as HM, reconstruct3d, video_io
    from app.services.tracking.data_types import VideoMeta
    CACHE.mkdir(exist_ok=True)
    root = TEST_SPLIT.parent.parent
    cases = []
    vids = sorted(TEST_SPLIT.glob("match*/video/*.mp4"))[:limit]
    for vid in vids:
        match, clip = vid.parent.parent.name, vid.stem
        corr = root / "corrected" / match / "corrected_csv" / f"{clip}_ball.csv"
        lab = corr if corr.exists() else vid.parent.parent / "csv" / f"{clip}_ball.csv"
        cpath = CACHE / f"{match}_{clip}.pkl"
        if cpath.exists():
            meta, cam = pickle.loads(cpath.read_bytes())
        else:
            meta = video_io.probe(str(vid))
            frames = [f for _, f in video_io.sample_frames(str(vid), 24, meta.n_frames)]
            try:
                cam = CC.calibrate_from_frames(frames, (meta.width, meta.height))
            except Exception as e:                    # pragma: no cover
                print("   calibration error", e)
                cam = None
            cpath.write_bytes(pickle.dumps((meta, cam)))
        meta = VideoMeta(fps=meta.fps, width=meta.width, height=meta.height, n_frames=meta.n_frames,
                         timestamps=np.asarray(meta.timestamps, float))
        track = _label_track(lab, meta.n_frames, meta.timestamps)
        hits = HM.detect_hits(track, meta, cam, None)
        segs = reconstruct3d.reconstruct_segments(track, hits, cam, meta, None) if cam is not None else []
        case = make_case(f"real_{match}_{clip}", track, hits, segs, meta, cam, None, keep_dicts=True)
        print(f"  real {match}/{clip}: frames={meta.n_frames} cam={cam is not None} hits={len(hits)} "
              f"segs={len(segs)} rallies={len(case['rallies'])}", flush=True)
        if case["rallies"]:
            cases.append(case)
    return cases


# ── shots unit cases (tests/test_shots.py) ───────────────────────────────────

def _seg_min(seg) -> dict:
    d = to_jsonable(seg)
    d["frames"], d["positions"] = [], []          # classifier reads p0/v0/t0/drag/confidence only
    return d


def shots_cases() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests import test_shots as TS
    flights = []
    for kind, seg in TS._flights():
        first = kind.startswith("serve")
        lab, conf, f = SH.classify_segment(seg, None, is_first=first)
        flights.append({"kind": kind, "segment": _seg_min(seg), "is_first": first, "label": lab, "conf": conf,
                        "features": None if f is None else to_jsonable(f.__dict__)})
    rng = np.random.default_rng(7)
    noisy = []
    for kind, base in TS._flights():
        p0 = base.p0 + rng.normal(0, [0.1, 0.1, 0.08])
        sp = np.linalg.norm(base.v0)
        d = base.v0 / sp + rng.normal(0, 0.04, 3)
        v0 = d / np.linalg.norm(d) * sp * (1 + rng.normal(0, 0.07))
        seg = replace(base, p0=p0, v0=v0)
        first = kind.startswith("serve")
        lab, conf, _ = SH.classify_segment(seg, None, is_first=first)
        noisy.append({"kind": kind, "segment": _seg_min(seg), "is_first": first, "label": lab, "conf": conf})
    # pose / handedness rally, 2-D fallback, late back-propagation
    rally, players = TS._rally()
    fb_rally = replace(rally, shots=[replace(s, segment=None) for s in rally.shots])
    fr, fplayers = TS._fallback_rally()
    late = TS._late([-5.5, 0.3, 2.8], TS._v(300, -10), 2 / 30)
    late_rally = RallyV2(0, 60, shots=[
        ShotEventV2(HitEvent(-60, -1.0, (0.0, 0.0), "far"), None),
        ShotEventV2(HitEvent(frame=0, t=0.0, xy_px=(0.0, 0.0), hitter="near", confidence=0.9), late)])
    blow = TS._seg([-5.0, 0.0, 2.0], TS._v(200, -5))
    blow.t0 = 0.25
    rallies = {
        "pose": (rally, players, None), "pose_flip": (rally, players, {2: True}),
        "fallback_3d_removed": (fb_rally, players, None), "fallback": (fr, fplayers, None),
        "late": (late_rally, None, None),
    }
    rally_cases = []
    for name, (r, pl, hand) in rallies.items():
        out = SH.classify_shots(r, None, pl, handedness=hand)
        rally_cases.append({"name": name, "rally": r, "players": pl,
                            "handedness": None if hand is None else [[k, v] for k, v in hand.items()],
                            "expected": out})
    # stroke side on projected skeletons
    from app.services.tracking.camera import CameraModel
    cam = CameraModel.look_at(np.array([-12.0, 0, 6.0]), np.array([0, 0, 0.5]), 1400.0, (1920, 1080))
    strokes = []
    for foot, facing in [((-4.0, 0.5), a) for a in (0, 30, -30, 150, 180)] + \
            [((4.0, -0.5), a) for a in (180, 150, 210, 0, 30)] + [((5.0, 2.0), 180)] + \
            [((-4.0, 0.0), 90), ((-4.0, 0.0), 270), ((4.0, 0.0), 90), ((4.0, 0.0), 270)]:
        for hand in "RL":
            for st in ("FH", "BH"):
                k = TS._skeleton_px(cam, foot, facing, hand, st)
                strokes.append({"kps": k, "right": hand == "R",
                                "side": SH.stroke_side(k, 0.0, hand == "R"),
                                "above": list(SH.wrist_above_head(k))})
    contact = []
    for t_hit in (0.0, 1.0, -5.0):
        p, v = SH.contact_state(late, t_hit)
        contact.append({"segment": _seg_min(late), "t_hit": t_hit, "p": p, "v": v})
    p, v = SH.contact_state(blow, 0.0)
    contact.append({"segment": _seg_min(blow), "t_hit": 0.0, "p": p, "v": v})
    oh = [{"h": h, "po": po, "p": SH.overhead_prob(h, po)}
          for h in (None, 1.0, 2.1, 2.6, float("nan")) for po in (None, 0.5, 0.1, 0.95)]
    return {"flights": flights, "noisy": noisy, "rallies": rally_cases, "strokes": strokes,
            "contact": contact, "overhead": oh}


def score_cases() -> list[dict]:
    """tests/test_rally.py BWF sequences (+ random long sequences with unknowns)."""
    def R_(winner, i, first_hitter=None, conf=0.9, serve_missed=False):
        shots = []
        if first_hitter is not None:
            shots = [ShotEventV2(hit=HitEvent(frame=i * 100, t=i * 100 / 30, xy_px=(0.0, 0.0),
                                              hitter=first_hitter, confidence=0.9), segment=None)]
        r = RallyV2(start_frame=i * 100, end_frame=i * 100 + 50, shots=shots, winner=winner, confidence=conf)
        r.serve_missed = serve_missed
        return r
    seqs = [
        (["near"] * 21 + ["far"], {}),
        (["near", "far"] * 20 + ["near", "far", "near", "far", "near", "near"], {}),
        (["near", "far"] * 19 + ["near", "near"], {}),
        (["near", "far"] * 29 + ["far"], {}),
        (["near", "far"] * 28 + ["near", "near"], {}),
        (["near", "far", "far", "near", "near", "near", "far"], {}),
        (["far"], {"first_server": "far"}),
        (["near"] * 42 + ["far"] + ["near"] * 11 + ["far"] * 10, {}),
        (["near"] * 42 + ["near"] * 22, {}),
        (["near"] * 21, {"games_to_win": 1}),
    ]
    cases = []
    for seq, kw in seqs:
        rs = [R_(w, i) for i, w in enumerate(seq)]
        cases.append({"rallies": rs, "kwargs": kw, "expected": R.score_match(rs, **kw)})
    rs = [R_("near", 0, "near"), R_("unknown", 1, "near"), R_("near", 2, "far")]
    cases.append({"rallies": rs, "kwargs": {}, "expected": R.score_match(rs)})
    rs = [R_("near", 0, "near"), R_("near", 1, "far")]
    cases.append({"rallies": rs, "kwargs": {}, "expected": R.score_match(rs)})
    rng = np.random.default_rng(5)
    for k in range(6):
        n = int(rng.integers(30, 140))
        rs = []
        for i in range(n):
            w = str(rng.choice(["near", "far", "unknown"], p=[0.45, 0.45, 0.1]))
            fh = str(rng.choice(["near", "far", "unknown"])) if rng.random() < 0.8 else None
            rs.append(R_(w, i, fh, conf=float(rng.uniform(0.2, 1.0)), serve_missed=bool(rng.random() < 0.1)))
        order = rng.permutation(n) if k % 2 else np.arange(n)
        rs = [rs[j] for j in order]
        kw = {"games_to_win": int(rng.integers(1, 3)), "first_server": str(rng.choice(["near", "far"]))}
        cases.append({"rallies": rs, "kwargs": kw, "expected": R.score_match(rs, **kw)})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--real-limit", type=int, default=None)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    want = lambda k: not only or k in only          # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    if want("score"):
        dump_z(score_cases(), OUT / "score.json.zz")
    if want("shots"):
        dump_z(shots_cases(), OUT / "shots_cases.json.zz")
    for key, fn in (("sim_e2e", sim_e2e), ("sim_match", sim_match), ("sim_corrupt", sim_corrupt),
                    ("world", world)):
        if want(key):
            print(key, flush=True)
            dump_z(fn(), OUT / f"{key}.json.zz")
    if want("codec"):
        cc = codec_cases()
        if cc is None:
            print("codec: sources missing, codec.json.zz kept as is", flush=True)
        else:
            dump_z(cc, OUT / "codec.json.zz")
    if want("real"):
        print("real", flush=True)
        dump_z(real(a.real_limit), OUT / "real.json.zz")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
