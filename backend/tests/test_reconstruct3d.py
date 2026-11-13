"""Tests for monocular 3-D shuttle reconstruction (reconstruct3d.py).

Synthetic flights are generated with physics.simulate_state and projected with
CameraModel.look_at cameras ('low' = tripod behind the near baseline, 'high' =
elevated), then corrupted (1.5 px noise, 10 % dropouts, outliers). Run this file
directly for the full accuracy table: `python -m tests.test_reconstruct3d 400`.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import pytest
from numba import njit

from app.services.tracking import court_model as cm
from app.services.tracking import reconstruct3d as r3
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, ShotSegment3D, Track2D, VideoMeta,
)
from app.services.tracking.physics import DRAG_LENGTH, G, simulate_state, time_to_ground

SIZE = (1920, 1080)
# eye / target (m), focal range (px @1920x1080), eye jitter sigma (m)
CAMS = {
    "low": dict(eye=(-10.5, 0.3, 1.5), target=(0.0, 0.0, 1.0), f=(1400, 1700), j=(0.2, 0.3, 0.05)),
    "high": dict(eye=(-14.0, 0.0, 5.5), target=(0.0, 0.0, 0.5), f=(1400, 1700), j=(0.3, 0.3, 0.2)),
    "broadcast": dict(eye=(-14.5, 0.0, 7.5), target=(0.3, 0.0, 0.3), f=(1150, 1350),
                      j=(0.4, 0.4, 0.3)),
    "side": dict(eye=(0.5, -9.5, 4.0), target=(0.0, 0.0, 1.0), f=(900, 1100), j=(0.5, 0.3, 0.3)),
}

# name: (hitter |x| range, contact height, speed km/h, target |x| range (other half),
#        target |y| max, arc) — near-half geometry; far-half flights are mirrored.
SHOTS = {
    "smash": ((2.5, 5.5), (2.6, 3.0), (250, 400), (1.0, 6.0), 2.4, "low"),
    "clear": ((4.0, 6.2), (1.8, 2.8), (130, 220), (4.8, 7.1), 2.8, "high"),
    "drop": ((3.5, 6.0), (2.2, 2.9), (50, 110), (0.4, 2.8), 2.6, "low"),
    "net": ((0.6, 2.2), (0.3, 1.2), (12, 35), (0.3, 1.6), 2.6, "high"),
    "drive": ((1.5, 4.0), (0.9, 1.6), (110, 200), (2.5, 6.8), 2.8, "low"),
    "lift": ((0.8, 2.6), (0.3, 1.0), (90, 170), (4.5, 7.1), 2.8, "high"),
    "serve_short": ((2.0, 2.6), (0.95, 1.12), (18, 40), (2.0, 2.9), 2.4, "high"),
    "serve_long": ((2.0, 3.0), (0.95, 1.12), (100, 170), (5.9, 7.0), 2.6, "high"),
}

# elevation range (deg) per shot type
ELEV = {"smash": (-32, -6), "clear": (18, 48), "drop": (-18, 12), "net": (5, 55),
        "drive": (-4, 10), "lift": (30, 62), "serve_short": (5, 30), "serve_long": (32, 58)}


@dataclass
class Flight:
    shot: str
    half: str                  # hitter's half
    p_hit: np.ndarray
    v_hit: np.ndarray
    t_hit: float
    land_xy: np.ndarray
    t_land: float
    t: np.ndarray              # frame times (all frames of the segment)
    uv: np.ndarray             # noisy observations (NaN = missing)
    uv_true: np.ndarray
    t_rest: float              # first frame time with the shuttle on the floor


def make_camera(kind: str, rng: np.random.Generator) -> CameraModel:
    c = CAMS[kind]
    eye = np.array(c["eye"]) + rng.normal(0, c["j"])
    return CameraModel.look_at(eye, np.array(c["target"]), float(rng.uniform(*c["f"])), SIZE)


# ── generator physics: numba mirror of physics.py (same RK4, same step rule) ──
@njit(cache=True)
def _acc(v: np.ndarray, L: float) -> np.ndarray:
    a = -math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * v / L
    a[2] -= G
    return a


@njit(cache=True)
def _step(p: np.ndarray, v: np.ndarray, h: float, L: float):
    k1v = _acc(v, L)
    k2p = v + 0.5 * h * k1v
    k2v = _acc(k2p, L)
    k3p = v + 0.5 * h * k2v
    k3v = _acc(k3p, L)
    k4p = v + h * k3v
    k4v = _acc(k4p, L)
    return (p + h / 6 * (v + 2 * k2p + 2 * k3p + k4p),
            v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


@njit(cache=True)
def sim_state(p0: np.ndarray, v0: np.ndarray, times: np.ndarray, L: float = DRAG_LENGTH,
              max_dt: float = 1 / 480):
    """== physics.simulate_state (monotonic times)."""
    n = times.shape[0]
    P, V = np.empty((n, 3)), np.empty((n, 3))
    p, v, t = p0.astype(np.float64), v0.astype(np.float64), 0.0
    for i in range(n):
        span = times[i] - t
        m = max(1, int(math.ceil(abs(span) / max_dt)))
        for _ in range(m):
            p, v = _step(p, v, span / m, L)
        t = times[i]
        P[i], V[i] = p, v
    return P, V


@njit(cache=True)
def ground_time(p0: np.ndarray, v0: np.ndarray, L: float = DRAG_LENGTH, t_max: float = 6.0,
                dt: float = 1 / 480) -> float:
    """== physics.time_to_ground (z_ground = 0); -1 if never."""
    p, v, t = p0.astype(np.float64), v0.astype(np.float64), 0.0
    while t < t_max:
        pn, vn = _step(p, v, dt, L)
        if pn[2] <= 0.0 < p[2] or (p[2] > 0.0 >= pn[2]):
            return t + (p[2] / max(p[2] - pn[2], 1e-12)) * dt
        p, v, t = pn, vn, t + dt
    return -1.0


@njit(cache=True)
def _screen(pp: np.ndarray, vv: np.ndarray):
    """Per candidate launch: (time to floor, landing xy, z at the x = 0 crossing)."""
    K = pp.shape[0]
    out = np.full((K, 4), np.nan)
    for k in range(K):
        p, v, t = pp[k].copy(), vv[k].copy(), 0.0
        while t < 6.0:
            pn, vn = _step(p, v, 1 / 480, DRAG_LENGTH)
            if out[k, 3] != out[k, 3] and (p[0] > 0.0) != (pn[0] > 0.0):
                out[k, 3] = p[2] + p[0] / (p[0] - pn[0]) * (pn[2] - p[2])
            if pn[2] <= 0.0:
                f = p[2] / (p[2] - pn[2])
                out[k, 0] = t + f / 480
                out[k, 1] = p[0] + f * (pn[0] - p[0])
                out[k, 2] = p[1] + f * (pn[1] - p[1])
                break
            p, v, t = pn, vn, t + 1 / 480
    return out[:, 0], out[:, 1:3], out[:, 3]


def sample_launch(rng: np.random.Generator, shot: str, half: str, p: np.ndarray | None = None,
                  ty_sign: float | None = None, net_fault: bool = False, batch: int = 96,
                  ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """(contact point, launch velocity, flight time to the floor, landing xy) of a
    shot that clears the net and lands (roughly) in the other half. Candidates are
    screened in a vectorised batch; the returned truth comes from physics.py.
    net_fault=True instead returns a flight that crosses x = 0 below the net."""
    hx, hz, spd, tx, ty, _ = SHOTS[shot]
    sgn = -1.0 if half == "near" else 1.0          # hitter x sign
    for _ in range(40):
        if p is None:
            pp = np.column_stack([sgn * rng.uniform(*hx, batch), rng.uniform(-2.3, 2.3, batch),
                                  rng.uniform(*hz, batch)])
            if shot.startswith("serve"):
                pp[:, 1] = rng.uniform(0.3, 1.8, batch) * rng.choice([-1, 1], batch)
        else:
            pp = np.repeat(np.asarray(p, float)[None], batch, 0)
        tgt = np.column_stack([-sgn * rng.uniform(*tx, batch), rng.uniform(-ty, ty, batch)])
        if ty_sign is not None:
            tgt[:, 1] = ty_sign * rng.uniform(0.2, ty, batch)
        az = np.arctan2(tgt[:, 1] - pp[:, 1], tgt[:, 0] - pp[:, 0])
        el = np.radians(rng.uniform(*ELEV[shot], batch))
        sp = rng.uniform(*spd, batch) / 3.6
        v = sp[:, None] * np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                                            np.sin(el)])
        tg, land, znet = _screen(np.ascontiguousarray(pp), np.ascontiguousarray(v))
        with np.errstate(invalid="ignore"):
            ok = (tg >= 0.15) & (np.sign(land[:, 0]) != sgn) & (np.abs(land[:, 0]) <= 7.6) \
                & (np.abs(land[:, 1]) <= 3.6)
            ok &= (znet < cm.NET_HEIGHT_CENTRE - 0.1) & (znet > 0.5) if net_fault \
                else (znet >= cm.NET_HEIGHT_POST + 0.02)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        k = int(idx[0])
        tg_true = ground_time(pp[k], v[k])
        if tg_true < 0:
            continue
        land_true = sim_state(pp[k], v[k], np.array([tg_true]))[0][0, :2]
        return pp[k].copy(), v[k].copy(), float(tg_true), land_true
    raise RuntimeError("could not sample flight")


def corrupt(rng: np.random.Generator, uv_true: np.ndarray, noise: float, miss: float,
            n_out: int, keep_first: bool = True) -> np.ndarray:
    uv = uv_true + rng.normal(0, noise, uv_true.shape)
    inside = ((uv_true[:, 0] >= 0) & (uv_true[:, 0] < SIZE[0]) & (uv_true[:, 1] >= 0)
              & (uv_true[:, 1] < SIZE[1]))
    drop = rng.random(len(uv)) < miss
    if keep_first:
        drop[0] = False
    uv[drop | ~inside] = np.nan
    good = np.nonzero(np.isfinite(uv[:, 0]))[0]
    if n_out and len(good) > 6:
        for i in rng.choice(good[1:-1], size=min(n_out, len(good) - 2), replace=False):
            uv[i] += rng.uniform(20, 80, 2) * rng.choice([-1, 1], 2)
    return uv


def sample_flight(rng: np.random.Generator, shot: str, half: str, camera: CameraModel,
                  fps: float, noise: float = 1.5, miss: float = 0.1, n_out: int = 2) -> Flight:
    """One hit → floor flight observed from contact until just before landing."""
    p, v, tg, land = sample_launch(rng, shot, half)
    t_hit = float(rng.uniform(1.0, 1.0 + 1 / fps))
    t0 = math.floor(t_hit * fps + 1) / fps           # first frame after contact
    t = np.arange(t0, t_hit + tg, 1 / fps)
    t_rest = float(t[-1] + 1 / fps)
    P, _ = sim_state(p, v, t - t_hit)
    uv_true = camera.project(P)
    uv = corrupt(rng, uv_true, noise, miss, n_out)
    return Flight(shot, half, p, v, t_hit, land, t_hit + tg, t, uv, uv_true, t_rest)


@dataclass
class Shot:
    shot: str
    half: str
    p: np.ndarray
    v: np.ndarray
    t: float                   # true contact time
    t_end: float               # next contact or landing
    land_xy: np.ndarray | None # only for the last shot


@dataclass
class Rally:
    track: Track2D
    hits: list[HitEvent]
    players: list[PlayerObservation]
    meta: VideoMeta
    shots: list[Shot]
    camera: CameraModel


def _next_shot_type(rng: np.random.Generator, x_abs: float, z: float) -> list[str]:
    if z > 2.2 and x_abs > 3.0:
        opts = ["smash", "clear", "drop"]
    elif z < 1.3 and x_abs < 2.6:
        opts = ["net", "lift"]
    elif z < 1.3:
        opts = ["lift", "drive"]
    else:
        opts = ["drive", "drop", "lift", "clear"]
    rng.shuffle(opts)
    return opts


def make_rally(rng: np.random.Generator, camera: CameraModel, fps: float, n_shots: int = 4,
               noise: float = 1.5, miss: float = 0.1, n_out: int = 2) -> Rally:
    """Serve + chained shots; the last one lands and the shuttle skids to rest.
    Hit events are frame-quantised (as a detector would report them)."""
    half = "near" if rng.random() < 0.5 else "far"
    sgn = -1.0 if half == "near" else 1.0
    p = np.array([sgn * rng.uniform(2.1, 2.8), rng.uniform(0.3, 1.6) * rng.choice([-1, 1]),
                  rng.uniform(0.95, 1.12)])
    serve = "serve_short" if rng.random() < 0.5 else "serve_long"
    p, v, tg, land = sample_launch(rng, serve, half, p, ty_sign=-np.sign(p[1]) * 1.0)
    t_c = 0.5 + float(rng.uniform(0, 1 / fps))
    shots = [Shot(serve, half, p, v, t_c, t_c + tg, land)]
    for _ in range(n_shots - 1):
        cur = shots[-1]
        grid = np.linspace(0, cur.t_end - cur.t, 400)
        Pg, _ = sim_state(cur.p, cur.v, grid)
        rsgn = -np.sign(cur.p[0])
        okm = (rsgn * Pg[:, 0] > 0.6) & (Pg[:, 2] > 0.35) & (Pg[:, 2] < 3.0)
        cand = np.nonzero(okm)[0]
        if len(cand) < 5:
            break
        k = int(cand[int(rng.uniform(0.35, 0.95) * (len(cand) - 1))])
        pc = Pg[k]
        rhalf = "near" if rsgn < 0 else "far"
        for st in _next_shot_type(rng, abs(pc[0]), pc[2]):
            try:
                _, v2, tg2, land2 = sample_launch(rng, st, rhalf, pc)
                break
            except RuntimeError:
                continue
        else:
            break
        t2 = cur.t + float(grid[k])
        cur.t_end, cur.land_xy = t2, None
        shots.append(Shot(st, rhalf, pc.copy(), v2, t2, t2 + tg2, land2))

    last = shots[-1]
    t_rest_end = last.t_end + 0.6
    T = np.arange(0.0, t_rest_end, 1 / fps)
    P = np.full((len(T), 3), np.nan)
    for s in shots:
        m = (T > s.t) & (T <= s.t_end)
        if m.any():
            P[m] = sim_state(s.p, s.v, T[m] - s.t)[0]
    # after landing: short skid along the flight direction, then rest
    vl = sim_state(last.p, last.v, np.array([last.t_end - last.t]))[1][0]
    dirn = vl[:2] / max(np.linalg.norm(vl[:2]), 1e-6)
    skid = float(rng.uniform(0.0, 0.25))
    m = T > last.t_end
    frac = np.clip((T[m] - last.t_end) / 0.1, 0, 1)
    P[m] = np.column_stack([last.land_xy[0] + dirn[0] * skid * frac,
                            last.land_xy[1] + dirn[1] * skid * frac, np.full(m.sum(), 0.02)])
    uv_true = np.full((len(T), 2), np.nan)
    okp = np.isfinite(P).all(1)
    uv_true[okp] = camera.project(P[okp])
    uv = corrupt(rng, np.nan_to_num(uv_true, nan=-1e4), noise, miss, n_out, keep_first=False)
    uv[~okp] = np.nan
    vis = np.isfinite(uv).all(1)
    track = Track2D(t=T, xy=uv, conf=np.where(vis, 0.9, 0.0), visible=vis,
                    interpolated=np.zeros(len(T), bool))
    hits, players = [], []
    for s in shots:
        f = int(round(s.t * fps))
        uvh = camera.project(s.p[None])[0]
        tid = 1 if s.half == "near" else 2
        hits.append(HitEvent(frame=f, t=float(T[f]), xy_px=(float(uvh[0]), float(uvh[1])),
                             hitter=s.half, confidence=0.9, player_track_id=tid))
        ang = rng.uniform(0, 2 * np.pi)
        foot = s.p[:2] + rng.uniform(0.2, 0.8) * np.array([np.cos(ang), np.sin(ang)])
        for df in (-2, 0, 2):
            players.append(PlayerObservation(frame=f + df, track_id=tid,
                                             bbox=np.array([0, 0, 1, 1.0]), score=0.9,
                                             side=s.half, foot_world=foot))
    meta = VideoMeta(fps=fps, width=SIZE[0], height=SIZE[1], n_frames=len(T), timestamps=T)
    return Rally(track, hits, players, meta, shots, camera)


# ── benchmark helpers (also used by `python -m tests.test_reconstruct3d N`) ─────

def fit_flight(fl: Flight, cam: CameraModel, rng: np.random.Generator | None = None,
               feet_sigma: float | None = None) -> tuple[ShotSegment3D | None, float]:
    """Fit a sampled flight with the priors reconstruct_segments would use
    (contact time, hitter side, floor end at the first rest frame); optional
    hitter-feet prior (true contact xy + N(0, feet_sigma))."""
    pr = r3.SegmentPrior(start_t=fl.t_hit, expected_side=fl.half, end="floor", end_t=fl.t_rest)
    if feet_sigma is not None:
        pr.start_xy_world = fl.p_hit[:2] + rng.normal(0, feet_sigma, 2)
        pr.start_xy_sigma = max(feet_sigma, 0.1)
    t0 = time.process_time()          # CPU time: robust to other load on the machine
    seg = r3.fit_segment(fl.t, fl.uv, None, cam, pr)
    return seg, (time.process_time() - t0) * 1e3


def run_benchmark(n: int, seed: int = 0, kinds: tuple[str, ...] = ("low", "high"),
                  feet_sigma: float | None = None) -> list[dict]:
    """n random floor-ending flights: every shot type, both halves, 30/60/120 fps."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        shot = list(SHOTS)[i % len(SHOTS)]
        half = "near" if (i // len(SHOTS)) % 2 == 0 else "far"
        kind = kinds[int(rng.integers(len(kinds)))]
        cam = make_camera(kind, rng)
        fps = float(rng.choice([30, 60, 120]))
        fl = sample_flight(rng, shot, half, cam, fps)
        seg, ms = fit_flight(fl, cam, rng, feet_sigma)
        true_m = cm.in_out_margin(float(fl.land_xy[0]), float(fl.land_xy[1]), False)
        row = dict(shot=shot, half=half, cam=kind, fps=fps, ms=ms, margin=true_m,
                   true_call="in" if true_m >= 0 else "out", land=np.nan, speed=np.nan,
                   height=np.nan, sigma=np.nan, call="unsure", ok=seg is not None)
        if seg is not None:
            row["speed"] = seg.speed0_kmh / (np.linalg.norm(fl.v_hit) * 3.6) - 1
            row["height"] = float(seg.p0[2] - fl.p_hit[2])
            if seg.landing_xy is not None:
                row["land"] = float(np.linalg.norm(np.subtract(seg.landing_xy, fl.land_xy)))
                row["sigma"] = r3.landing_sigma(seg) or np.nan
            row["call"] = r3.call_in_out(seg, doubles=False)["call"]
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    a = {k: np.array([r[k] for r in rows], float) for k in ("land", "speed", "height", "ms",
                                                              "margin", "sigma")}
    call = np.array([r["call"] for r in rows])
    truth = np.array([r["true_call"] for r in rows])
    clear = np.abs(a["margin"]) > 0.15
    decided = clear & (call != "unsure")
    close = np.abs(a["margin"]) < 0.2
    fin = np.isfinite(a["land"]) & np.isfinite(a["sigma"])
    return dict(
        n=len(rows), fails=int(np.sum(~np.isfinite(a["land"]))),
        land_med=float(np.nanmedian(a["land"])), land_p90=float(np.nanpercentile(a["land"], 90)),
        speed_med=float(np.nanmedian(np.abs(a["speed"]))),
        speed_p90=float(np.nanpercentile(np.abs(a["speed"]), 90)),
        height_med=float(np.nanmedian(np.abs(a["height"]))),
        ms_med=float(np.median(a["ms"])), ms_p90=float(np.percentile(a["ms"], 90)),
        acc_clear=float(np.mean(call[clear] == truth[clear])) if clear.any() else np.nan,
        acc_clear_decided=float(np.mean(call[decided] == truth[decided])) if decided.any() else np.nan,
        unsure_clear=float(np.mean(call[clear] == "unsure")) if clear.any() else np.nan,
        acc_close=float(np.mean(call[close] == truth[close])) if close.any() else np.nan,
        n_close=int(close.sum()),
        within_3sigma=float(np.mean(a["land"][fin] < 3 * a["sigma"][fin])) if fin.any() else np.nan,
    )


# ── tests ──────────────────────────────────────────────────────────────────────

def test_generator_matches_physics():
    rng = np.random.default_rng(1)
    for _ in range(3):
        p, v, tg, land = sample_launch(rng, "smash", "near")
        ts = np.sort(rng.uniform(0, tg, 25))
        P0, V0 = simulate_state(p, v, ts)
        P1, V1 = sim_state(p, v, ts)
        assert np.abs(P0 - P1).max() < 1e-9 and np.abs(V0 - V1).max() < 1e-9
        assert abs(time_to_ground(p, v) - tg) < 1e-9


def test_noise_free_flight_is_recovered():
    rng = np.random.default_rng(2)
    for shot, half, kind in [("smash", "near", "low"), ("clear", "far", "high"),
                             ("drop", "far", "low"), ("drive", "near", "high")]:
        cam = make_camera(kind, rng)
        fl = sample_flight(rng, shot, half, cam, 60.0, noise=0.0, miss=0.0, n_out=0)
        seg, _ = fit_flight(fl, cam)
        assert seg is not None and seg.landing_xy is not None, shot
        assert np.linalg.norm(np.subtract(seg.landing_xy, fl.land_xy)) < 0.05, shot
        assert abs(seg.speed0_kmh / (np.linalg.norm(fl.v_hit) * 3.6) - 1) < 0.02, shot
        assert abs(seg.t0 - fl.t_hit) < 1e-9 and seg.end_reason == "floor"
        assert seg.rmse_px < 0.5 and 0 < seg.confidence <= 1


@pytest.fixture(scope="module")
def bench_rows() -> list[dict]:
    return run_benchmark(64, seed=11)


def test_accuracy_targets(bench_rows):
    s = summarize(bench_rows)
    assert s["fails"] <= 1, s
    assert s["land_med"] < 0.10, s
    assert s["land_p90"] < 0.30, s
    assert s["speed_med"] < 0.05, s
    assert s["acc_clear_decided"] >= 0.98, s
    assert s["unsure_clear"] < 0.15, s
    assert s["height_med"] < 0.15, s


def test_landing_sigma_is_calibrated(bench_rows):
    s = summarize(bench_rows)
    assert s["within_3sigma"] >= 0.75, s
    sig = np.array([r["sigma"] for r in bench_rows], float)
    assert np.all(sig[np.isfinite(sig)] > 0)


def test_solve_time(bench_rows):
    ms = np.array([r["ms"] for r in bench_rows])
    assert np.median(ms) < 30.0, np.median(ms)    # CPU ms; target 15 ms, slack for load


def test_feet_prior_helps_far_half():
    rng_seed = 5
    base = [r for r in run_benchmark(24, rng_seed) if r["half"] == "far"]
    feet = [r for r in run_benchmark(24, rng_seed, feet_sigma=0.3) if r["half"] == "far"]
    e0 = np.nanmedian([abs(r["speed"]) for r in base])
    e1 = np.nanmedian([abs(r["speed"]) for r in feet])
    l0 = np.nanmedian([r["land"] for r in base])
    l1 = np.nanmedian([r["land"] for r in feet])
    assert e1 <= e0 * 1.1 + 0.005 and l1 <= l0 * 1.2 + 0.01, (e0, e1, l0, l1)


def test_lens_distortion():
    rng = np.random.default_rng(7)
    cam = make_camera("low", rng)
    cam.dist = np.array([-0.12, 0.03, 0.0, 0.0, 0.0])
    fl = sample_flight(rng, "clear", "near", cam, 60.0)
    seg, _ = fit_flight(fl, cam)
    assert seg is not None and seg.landing_xy is not None
    assert np.linalg.norm(np.subtract(seg.landing_xy, fl.land_xy)) < 0.15


def _net_fault_track(rng: np.random.Generator, cam: CameraModel, fps: float
                     ) -> tuple[Track2D, list[HitEvent], VideoMeta, np.ndarray, np.ndarray]:
    """A near-half drop into the net: free flight until x = 0, then the shuttle
    slides down the net face (x ~ 0) and lies at its foot."""
    p, v, tg, _ = sample_launch(rng, "drop", "near", net_fault=True)
    T = np.arange(0.0, tg + 1.5, 1 / fps)
    P, _ = sim_state(p, v, T)
    k = int(np.argmax(P[:, 0] >= 0.0))                 # first sample past the net plane
    t_n, z_n, y_n = T[k], P[k - 1, 2], P[k - 1, 1]
    after = T >= t_n
    z_after = np.maximum(z_n - 2.5 * (T[after] - t_n), 0.02)
    P[after] = np.column_stack([np.full(after.sum(), 0.08), np.full(after.sum(), y_n), z_after])
    uv = cam.project(P) + rng.normal(0, 1.5, (len(T), 2))
    uv[0] = np.nan                                     # contact frame itself: occluded
    vis = np.isfinite(uv).all(1)
    track = Track2D(t=T, xy=uv, conf=np.where(vis, 0.9, 0.0), visible=vis,
                    interpolated=np.zeros(len(T), bool))
    uvh = cam.project(p[None])[0]
    hits = [HitEvent(frame=0, t=0.0, xy_px=(float(uvh[0]), float(uvh[1])), hitter="near",
                     confidence=0.9)]
    meta = VideoMeta(fps=fps, width=SIZE[0], height=SIZE[1], n_frames=len(T), timestamps=T)
    return track, hits, meta, T, P


def test_net_fault_is_detected():
    rng = np.random.default_rng(3)
    for kind in ("high", "low"):
        cam = make_camera(kind, rng)
        track, hits, meta, T, P = _net_fault_track(rng, cam, 60.0)
        pre = (P[:, 0] < -0.05) & track.visible
        # single segment seen until it hits the net: extrapolated crossing is below the tape
        pr = r3.SegmentPrior(start_t=0.0, expected_side="near")
        seg = r3.fit_segment(T[pre], track.xy[pre], None, cam, pr)
        assert seg is not None and seg.end_reason == "net" and seg.landing_xy is None, kind
        assert seg.net_crossing_height < cm.NET_HEIGHT_CENTRE
        # whole rally piece incl. the slide down the net and the rest at its foot
        segs = r3.reconstruct_segments(track, hits, cam, meta, None)
        assert len(segs) == 1 and segs[0].end_reason == "net", (kind, segs[0].end_reason)
        assert segs[0].landing_xy is None
        assert r3.call_in_out(segs[0], doubles=False)["call"] == "unsure"


def test_rally_reconstruction():
    rng = np.random.default_rng(21)
    n_seg = n_shot = 0
    sp_err, tc_err, land_err = [], [], []
    for kind, fps in [("low", 60.0), ("high", 30.0), ("broadcast", 60.0)]:
        cam = make_camera(kind, rng)
        R = make_rally(rng, cam, fps, n_shots=4)
        segs = r3.reconstruct_segments(R.track, R.hits, cam, R.meta, R.players)
        n_shot += len(R.shots)
        n_seg += len(segs)
        frames = [h.frame for h in R.hits]
        for sg in segs:
            k = frames.index(sg.hit_frame)
            s = R.shots[k]
            sp_err.append(abs(sg.speed0_kmh / (np.linalg.norm(s.v) * 3.6) - 1))
            tc_err.append(abs(sg.t0 - s.t))
            if k < len(R.shots) - 1:                 # ended by the next contact: no landing
                assert sg.end_reason == "hit" and sg.landing_xy is None
            else:
                assert sg.end_reason == "floor" and sg.landing_xy is not None
                land_err.append(np.linalg.norm(np.subtract(sg.landing_xy, s.land_xy)))
            assert np.all(np.diff(sg.frames) > 0) and len(sg.frames) == len(sg.positions)
    assert n_seg >= n_shot - 1, (n_seg, n_shot)
    assert np.median(sp_err) < 0.08, sp_err
    assert np.median(tc_err) < 0.02, tc_err
    assert max(land_err) < 0.4, land_err


def test_refine_hit_time():
    rng = np.random.default_rng(8)
    errs = []
    for _ in range(3):
        cam = make_camera("low", rng)
        fps = 120.0
        R = make_rally(rng, cam, fps, n_shots=2)
        if len(R.shots) < 2:
            continue
        h = R.hits[1]
        f, tc = r3.refine_hit_time(R.track, h.frame + 3, cam)
        errs.append(abs(tc - R.shots[1].t))
    assert errs and np.median(errs) < 2.0 / 120, errs


def test_speed_profile_and_initial_speed():
    rng = np.random.default_rng(4)
    cam = make_camera("low", rng)
    fl = sample_flight(rng, "smash", "near", cam, 120.0)
    seg, _ = fit_flight(fl, cam)
    ts, kmh = r3.speed_profile(seg)
    assert abs(ts[0] - seg.t0) < 1e-9 and ts[-1] <= seg.landing_t + 1e-6
    assert abs(kmh[0] - r3.initial_speed_kmh(seg)) < 1e-6
    assert np.all(np.diff(kmh) < 0)                 # a smash only slows down
    assert kmh[-1] < 0.5 * kmh[0]
    assert abs(r3.contact_speed_kmh(seg, seg.t0) - seg.speed0_kmh) < 1e-6


def _seg_at(land: tuple[float, float] | None, sigma: float, p0=(-4.0, 1.0, 2.0)) -> ShotSegment3D:
    seg = ShotSegment3D(start_frame=0, end_frame=10, t0=0.0, p0=np.array(p0, float),
                        v0=np.array([20.0, 0.0, 5.0]), drag_length=DRAG_LENGTH, rmse_px=1.0,
                        frames=np.arange(10), positions=np.zeros((10, 3)), landing_xy=land,
                        landing_t=None if land is None else 1.0)
    seg.landing_cov = None if land is None else np.eye(2) * sigma ** 2
    return seg


def test_call_in_out_and_unsure():
    edge = cm.HALF_LENGTH + cm.LINE_WIDTH / 2
    r = r3.call_in_out(_seg_at((edge - 0.30, 0.0), 0.03), doubles=False)
    assert r["call"] == "in" and abs(r["margin_m"] - 0.30) < 1e-9 and r["p_in"] > 0.99
    r = r3.call_in_out(_seg_at((edge + 0.30, 0.5), 0.03), doubles=False)
    assert r["call"] == "out" and r["margin_m"] < 0
    r = r3.call_in_out(_seg_at((edge - 0.03, 0.0), 0.04), doubles=False)
    assert r["call"] == "unsure" and 0.4 < r["p_in"] < 1.0
    # side line: singles vs doubles
    y = cm.SINGLES_HALF_WIDTH + 0.2
    assert r3.call_in_out(_seg_at((3.0, y), 0.02), doubles=False)["call"] == "out"
    assert r3.call_in_out(_seg_at((3.0, y), 0.02), doubles=True)["call"] == "in"
    # line is in
    assert r3.call_in_out(_seg_at((3.0, cm.SINGLES_HALF_WIDTH + 0.015), 0.001),
                          doubles=False)["margin_m"] > 0
    assert r3.call_in_out(_seg_at(None, 0.0), doubles=False)["call"] == "unsure"
    assert r3.call_in_out(None, doubles=True)["call"] == "unsure"


def test_serve_rules():
    # near server on the right (server's right = -y) → far half, receiver's right (+y)
    assert r3.serve_side_of(np.array([-2.5, -1.0])) == "right"
    assert r3.serve_side_of(np.array([-2.5, 1.0])) == "left"
    assert r3.serve_side_of(np.array([2.5, 1.0])) == "right"       # far server faces -x
    x0, x1, y0, y1 = r3.serve_box("near", "right", doubles=True)
    assert (x0, x1, y0, y1) == (cm.SHORT_SERVICE_X, cm.DOUBLES_LONG_SERVICE_X, 0.0,
                                cm.DOUBLES_HALF_WIDTH)
    x0, x1, y0, y1 = r3.serve_box("far", "right", doubles=False)
    assert (x0, x1, y0, y1) == (-cm.HALF_LENGTH, -cm.SHORT_SERVICE_X, -cm.SINGLES_HALF_WIDTH, 0.0)
    m = r3.serve_margin
    assert m(4.0, 1.5, "near", "right", True) > 0
    assert m(1.5, 1.5, "near", "right", True) < 0                  # short of the short line
    assert m(4.0, -1.5, "near", "right", True) < 0                 # wrong (non-diagonal) court
    assert m(6.3, 1.5, "near", "right", True) < 0                  # past doubles long line
    assert m(6.3, 1.5, "near", "right", False) > 0                 # singles: baseline is long
    assert m(4.0, 2.8, "near", "right", False) < 0                 # singles side line
    assert m(4.0, 2.8, "near", "right", True) > 0
    assert m(cm.SHORT_SERVICE_X - 0.015, 1.0, "near", "right", True) > 0   # line is in
    # call_in_out with is_serve: server half/side from p0
    seg = _seg_at((4.0, 1.5), 0.02, p0=(-2.4, -1.0, 1.05))
    assert r3.call_in_out(seg, doubles=True, is_serve=True)["call"] == "in"
    assert r3.call_in_out(seg, doubles=True, is_serve=True, serve_side="left")["call"] == "out"
    seg = _seg_at((6.3, 1.5), 0.02, p0=(-2.4, -1.0, 1.05))
    assert r3.call_in_out(seg, doubles=True, is_serve=True)["call"] == "out"
    assert r3.call_in_out(seg, doubles=False, is_serve=True)["call"] == "in"
    # contact across the body near the centre line: the server's feet decide the
    # court; with neither feet nor a clear contact side, disagreeing courts → unsure
    seg = _seg_at((4.0, 1.5), 0.02, p0=(-2.4, 0.05, 1.05))
    seg.hitter_feet_xy = (-2.7, -0.4)
    assert r3.call_in_out(seg, doubles=True, is_serve=True)["call"] == "in"
    seg.hitter_feet_xy = (-2.7, 0.4)
    assert r3.call_in_out(seg, doubles=True, is_serve=True)["call"] == "out"
    seg.hitter_feet_xy = None
    assert r3.call_in_out(seg, doubles=True, is_serve=True)["call"] == "unsure"
    seg = _seg_at((1.0, 1.5), 0.02, p0=(-2.4, 0.05, 1.05))     # short in both courts
    assert r3.call_in_out(seg, doubles=True, is_serve=True)["call"] == "out"


def test_degenerate_inputs():
    rng = np.random.default_rng(0)
    cam = make_camera("low", rng)
    assert r3.fit_segment(np.arange(4) / 30, np.full((4, 2), 500.0), None, cam) is None
    uv = np.full((12, 2), np.nan)
    assert r3.fit_segment(np.arange(12) / 30, uv, None, cam) is None
    R = make_rally(rng, cam, 30.0, n_shots=2)
    assert r3.reconstruct_segments(R.track, [], cam, R.meta, None) == []
    assert r3.reconstruct_segments(R.track, R.hits, None, R.meta, None) == []
    segs = r3.reconstruct_segments(R.track, R.hits, cam, R.meta, None)   # no players
    assert len(segs) >= 1 and all(np.isfinite(s.v0).all() for s in segs)


def _sim_track(simr, rng: np.random.Generator, noise: float = 1.5, miss: float = 0.1) -> Track2D:
    xy = np.where(simr.visible[:, None], simr.shuttle_px, np.nan) + rng.normal(0, noise,
                                                                              simr.shuttle_px.shape)
    xy[rng.random(len(xy)) < miss] = np.nan
    vis = np.isfinite(xy).all(1)
    return Track2D(t=simr.timestamps.copy(), xy=xy, conf=np.where(vis, 0.9, 0.0), visible=vis,
                   interpolated=np.zeros(len(vis), bool))


def _sim_hits(simr) -> list[HitEvent]:
    """sim truth hits, frame-quantised like a detector's."""
    out = []
    for h in sim.truth_hits(simr):
        out.append(HitEvent(frame=h.frame, t=float(simr.timestamps[h.frame]), xy_px=h.xy_px,
                            hitter=h.hitter, confidence=h.confidence,
                            player_track_id=h.player_track_id))
    return out


try:
    from app.services.tracking import sim
    _HAVE_SIM = all(hasattr(sim, a) for a in ("simulate_rally", "random_camera", "truth_hits",
                                               "truth_players"))
except Exception:            # sibling module missing or mid-edit
    sim = None
    _HAVE_SIM = False


@pytest.mark.skipif(not _HAVE_SIM, reason="sim.py not available")
def test_on_sibling_simulator():
    land_err, sp_err, n = [], [], 0
    for seed, style in [(1, "behind"), (2, "elevated_side"), (3, "broadcast")]:
        rng = np.random.default_rng(seed)
        cam = sim.random_camera(rng, style=style)
        simr = sim.simulate_rally(seed, camera=cam, fps=60, n_shots=4)
        segs = r3.reconstruct_segments(_sim_track(simr, rng), _sim_hits(simr), cam, simr.meta,
                                       sim.truth_players(simr, every=2))
        by = {s.frame: s for s in simr.shots}
        for sg in segs:
            s = by.get(sg.hit_frame)
            if s is None:
                continue
            n += 1
            sp_err.append(abs(sg.speed0_kmh / s.speed0_kmh - 1))
            if s.outcome in ("in", "out") and sg.landing_xy is not None:
                land_err.append(np.linalg.norm(np.subtract(sg.landing_xy, s.landing_xy)))
    assert n >= 6
    assert np.median(sp_err) < 0.08, sp_err
    assert not land_err or np.median(land_err) < 0.3, land_err


if __name__ == "__main__":           # full accuracy table: python -m tests.test_reconstruct3d 400
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    kinds = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("low", "high")
    feet = float(sys.argv[3]) if len(sys.argv) > 3 else None
    r3.fit_segment(np.arange(10) / 30, np.random.rand(10, 2) * 100 + 500, None,
                   make_camera("low", np.random.default_rng(0)))        # JIT warm-up
    rows = run_benchmark(N, seed=0, kinds=kinds, feet_sigma=feet)
    print(f"{'shot':12s} {'half':4s}  n land_med land_p90 spd_med height_med  ms_med")
    for shot in SHOTS:
        for half in ("near", "far"):
            g = [r for r in rows if r["shot"] == shot and r["half"] == half]
            if g:
                s = summarize(g)
                print(f"{shot:12s} {half:4s} {s['n']:2d} {100 * s['land_med']:6.1f}cm "
                      f"{100 * s['land_p90']:6.1f}cm {100 * s['speed_med']:5.1f}% "
                      f"{100 * s['height_med']:6.1f}cm {s['ms_med']:6.1f}")
    for k in kinds + ("ALL",):
        g = rows if k == "ALL" else [r for r in rows if r["cam"] == k]
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in summarize(g).items()})


# ── presentation-clock gaps (video_io.pts_gaps) ──────────────────────────────

def _with_gap(R: Rally, kind: str, frac: float = 0.5) -> tuple[Rally, int]:
    """Rally R with a clock hole in the middle of its 2nd shot: 'drop' removes ~0.43 s of
    frames (F4 repro), 'jump' adds 5 s to every later time (F3 repro). Returns the rally
    re-indexed on the new frame axis and the first frame after the gap."""
    import dataclasses
    T = R.track.t
    s = R.shots[1]
    g0 = int(np.searchsorted(T, s.t + frac * (s.t_end - s.t)))
    fps = R.meta.fps
    if kind == "drop":
        keep = np.r_[np.arange(g0), np.arange(g0 + int(round(0.43 * fps)), len(T))]
        t2 = T[keep]
    else:
        keep = np.arange(len(T))
        t2 = T.copy()
        t2[g0:] += 5.0
    tr = Track2D(t=t2, xy=R.track.xy[keep], conf=R.track.conf[keep], visible=R.track.visible[keep],
                 interpolated=R.track.interpolated[keep])
    fmap = np.full(len(T), -1)
    fmap[keep] = np.arange(len(keep))
    hits = [dataclasses.replace(h, frame=int(fmap[h.frame]), t=float(t2[fmap[h.frame]]))
            for h in R.hits if fmap[h.frame] >= 0]
    pl = [dataclasses.replace(p, frame=int(fmap[p.frame])) for p in R.players
          if 0 <= p.frame < len(T) and fmap[p.frame] >= 0]
    meta = VideoMeta(fps=fps, width=R.meta.width, height=R.meta.height, n_frames=len(keep), timestamps=t2)
    return Rally(tr, hits, pl, meta, R.shots, R.camera), g0


@pytest.mark.parametrize("kind", ["drop", "jump"])
def test_no_flight_fitted_across_pts_gap(kind):
    """A flight is never fitted across a clock hole: every segment lies on one side of it,
    the flight cut by it reports no landing / net call (its end was not seen), and the
    other flights are unchanged by the hole."""
    from app.services.tracking.video_io import pts_gaps
    rng = np.random.default_rng(33)
    n_cut = 0
    for camkind, fps in [("low", 60.0), ("high", 30.0), ("broadcast", 60.0)]:
        cam = make_camera(camkind, rng)
        R0 = make_rally(rng, cam, fps, n_shots=4)
        R, g = _with_gap(R0, kind)
        assert pts_gaps(R.track.t).tolist() == [g]
        segs = r3.reconstruct_segments(R.track, R.hits, cam, R.meta, R.players)
        ref = {s.hit_frame: s for s in r3.reconstruct_segments(R0.track, R0.hits, cam, R0.meta, R0.players)}
        for sg in segs:
            fr = np.asarray(sg.frames)
            assert np.all(fr < g) or np.all(fr >= g), (camkind, fr.min(), fr.max(), g)
            if getattr(sg, "ends_at_pts_gap", False):
                n_cut += 1
                assert sg.end_reason == "unknown" and sg.landing_xy is None and fr.max() < g
                assert sg.speed0_sigma_kmh <= r3.GAP_SPEED_REL_SIGMA * sg.speed0_kmh
            if sg.landing_xy is not None:
                assert abs(sg.landing_xy[0]) < cm.HALF_LENGTH + 1 and abs(sg.landing_xy[1]) < cm.DOUBLES_HALF_WIDTH + 1
        # the first shot ends before the gap: only its end time (refined where it meets the
        # 2nd flight, which the gap now cuts) couples it to the gap -> within 0.5 %
        first = [s for s in segs if s.hit_frame == R.hits[0].frame]
        if first and R0.hits[0].frame in ref:
            a, b = first[0], ref[R0.hits[0].frame]
            if np.asarray(a.frames).max() < g and b.end_reason == "hit" and a.end_reason == "hit":
                assert abs(a.speed0_kmh - b.speed0_kmh) < 5e-3 * b.speed0_kmh
    assert n_cut >= 1


def test_refine_hit_time_stays_within_gap_span():
    """refine_hit_time never pairs a split's two sides across a clock hole."""
    rng = np.random.default_rng(34)
    cam = make_camera("low", rng)
    R0 = make_rally(rng, cam, 60.0, n_shots=4)
    R, g = _with_gap(R0, "jump", frac=0.02)           # hole right after the 2nd contact
    for h in R.hits:
        f, tc = r3.refine_hit_time(R.track, h.frame, cam)
        assert 0 <= f < len(R.track.t) and np.isfinite(tc)
        if h.frame < g:
            assert tc <= R.track.t[g - 1] + 1e-9 or f >= g
