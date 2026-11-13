"""Linker tests on synthetic rallies (pinhole camera + quadratic-drag physics).

Standalone: builds its own data with CameraModel.look_at + physics.simulate_state
(no dependency on sim.py / metrics.py). Corruption per the linker brief:
σ=1.5 px noise, 10 % random misses + bursts of 5–15 frames, 5 % uniform false
positives, 2 static distractors firing in 60 % of frames, ±2 ms timestamp
jitter, 30/60/120 fps.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import numpy as np
import pytest

from app.services.tracking import physics
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import ShuttleCandidate, Track2D, VideoMeta
from app.services.tracking.linker import (
    LinkerConfig, find_kinks, link_candidates, smooth_track,
)

W, H = 1920, 1080
CAM = CameraModel.look_at(np.array([-10.5, 0.0, 1.6]), np.array([0.0, 0.0, 1.0]),
                          1500.0, (W, H))
SIGMA = 1.5
OBS_DV = 0.1 * np.hypot(W, H)      # px/s: smallest image-space Δv counted as a findable hit
SEEDS = list(range(10))


# ── synthetic rallies ─────────────────────────────────────────────────────────

def _integrate(p0: np.ndarray, V0: np.ndarray, dt: float, n: int) -> np.ndarray:
    """Batched RK4 of the physics.py ODE (for shot planning only)."""
    L = physics.DRAG_LENGTH

    def acc(v: np.ndarray) -> np.ndarray:
        return physics.GRAVITY - np.linalg.norm(v, axis=-1, keepdims=True) * v / L

    p = np.broadcast_to(p0, V0.shape).copy()
    v = V0.copy()
    P = np.empty((n + 1,) + V0.shape)
    P[0] = p
    for i in range(n):
        k1v, k1p = acc(v), v
        k2v, k2p = acc(v + 0.5 * dt * k1v), v + 0.5 * dt * k1v
        k3v, k3p = acc(v + 0.5 * dt * k2v), v + 0.5 * dt * k2v
        k4v, k4p = acc(v + dt * k3v), v + dt * k3v
        p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        P[i + 1] = p
    return P


@dataclass
class Shot:
    t0: float
    p0: np.ndarray
    v0: np.ndarray
    dur: float
    is_hit: bool          # a real hit ends the previous shot (not a serve)


def _plan_shot(rng: np.random.Generator, p0: np.ndarray, side: int) -> tuple:
    """Random physically valid shot from p0 (side −1 near, +1 far) that crosses
    the net above 1.55 m and is intercepted by the opponent 0.3–1.3 s later."""
    dt = 1 / 240
    times = np.arange(int(1.35 / dt) + 1) * dt
    el = np.radians(np.arange(-35.0, 76.0, 3.0))
    for _ in range(400):
        kmh = rng.uniform(50, 350)
        tx, ty = -side * rng.uniform(1.5, 6.2), rng.uniform(-2.6, 2.6)
        az = np.arctan2(ty - p0[1], tx - p0[0])
        V0 = kmh / 3.6 * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                                   np.sin(el)], 1)
        P = _integrate(p0, V0, dt, len(times) - 1)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        alive = np.cumprod(z > 0.05, axis=0).astype(bool)
        crossed = np.cumsum(np.sign(x) == -side, axis=0) > 0
        first = np.argmax(np.sign(x) == -side, axis=0)
        net_ok = z[first, np.arange(len(el))] > 1.55
        reach = (alive & crossed & net_ok[None, :] & (-side * x > 1.0) & (-side * x < 6.5)
                 & (z > 0.25) & (z < 2.8) & (np.abs(y) < 3.2)
                 & ((times > 0.3) & (times < 1.3))[:, None])
        ii, ee = np.nonzero(reach)
        if len(ii):
            k = rng.integers(len(ii))
            return V0[ee[k]], float(times[ii[k]]), P[ii[k], ee[k]].copy()
    raise RuntimeError("no valid shot")


@functools.lru_cache(maxsize=None)
def rally_plan(seed: int, n_rallies: int = 2) -> tuple[list[Shot], float]:
    """Shots (fps-independent) and total duration."""
    rng = np.random.default_rng(1000 + seed)
    shots: list[Shot] = []
    t = 0.8
    for _ in range(n_rallies):
        side = -1 if rng.random() < 0.5 else 1
        p = np.array([side * rng.uniform(2.0, 4.0), rng.uniform(-1.5, 1.5), 1.0])
        for k in range(int(rng.integers(9, 14))):
            v0, dur, p_end = _plan_shot(rng, p, side)
            shots.append(Shot(t, p, v0, dur, is_hit=k > 0))
            t, p, side = t + dur, p_end, -side
        # last shot falls to the floor
        tg = physics.time_to_ground(shots[-1].p0, shots[-1].v0) or shots[-1].dur
        shots[-1] = Shot(shots[-1].t0, shots[-1].p0, shots[-1].v0, float(tg), shots[-1].is_hit)
        t = shots[-1].t0 + shots[-1].dur + rng.uniform(2.0, 3.0)   # dead time
    return shots, t + 0.5


def _image_dv(prev: Shot, nxt: Shot, eps: float = 1e-3) -> float:
    """|v_img(after) − v_img(before)| at the contact (px/s)."""
    pb = physics.simulate_state(prev.p0, prev.v0, np.array([nxt.t0 - prev.t0 - eps,
                                                            nxt.t0 - prev.t0]))[0]
    pa = physics.simulate_state(nxt.p0, nxt.v0, np.array([0.0, eps]))[0]
    vb = np.diff(CAM.project(pb), axis=0)[0] / eps
    va = np.diff(CAM.project(pa), axis=0)[0] / eps
    return float(np.hypot(*(va - vb)))


@dataclass
class Case:
    cands: list[list[ShuttleCandidate]]
    meta: VideoMeta
    truth: np.ndarray          # (N,2) px, NaN when no shuttle in view
    truth_all: np.ndarray      # (N,2) px incl. off-image, NaN only in dead time
    detected: np.ndarray       # (N,) true detection delivered
    hit_frames: np.ndarray     # frames nearest to true hit times
    hit_t: np.ndarray
    hit_dv: np.ndarray         # true image-space velocity change at each hit (px/s)


def make_case(seed: int, fps: float, sigma: float = SIGMA, fp_rate: float = 0.05,
              n_static: int = 2, static_rate: float = 0.6) -> Case:
    shots, dur = rally_plan(seed)
    rng = np.random.default_rng(seed * 7919 + int(fps))
    n = int(dur * fps)
    t = np.arange(n) / fps + rng.uniform(-0.002, 0.002, n)
    t[0] = 0.0
    t = np.maximum.accumulate(t)
    world = np.full((n, 3), np.nan)
    for s in shots:
        sel = np.flatnonzero((t >= s.t0) & (t < s.t0 + s.dur))
        if len(sel):
            world[sel] = physics.simulate_state(s.p0, s.v0, t[sel] - s.t0)[0]
    truth = np.full((n, 2), np.nan)
    ok = np.isfinite(world[:, 0])
    truth[ok] = CAM.project(world[ok])
    truth_all = truth.copy()
    inside = (np.isfinite(truth[:, 0]) & (truth[:, 0] >= 0) & (truth[:, 0] < W)
              & (truth[:, 1] >= 0) & (truth[:, 1] < H))
    truth[~inside] = np.nan
    miss = rng.random(n) < 0.10
    for _ in range(max(1, int(dur / 3))):
        a = int(rng.integers(0, n))
        miss[a:a + int(rng.integers(5, 16))] = True
    detected = inside & ~miss
    statics = np.column_stack([rng.uniform(100, W - 100, n_static),
                               rng.uniform(H * 0.35, H - 60, n_static)])
    cands: list[list[ShuttleCandidate]] = []
    for i in range(n):
        c: list[ShuttleCandidate] = []
        if detected[i]:
            x, y = truth[i] + rng.normal(0, sigma, 2)
            c.append(ShuttleCandidate(float(x), float(y), float(rng.uniform(0.55, 1.0))))
        if rng.random() < fp_rate:
            c.append(ShuttleCandidate(float(rng.uniform(0, W)), float(rng.uniform(0, H)),
                                      float(rng.uniform(0.1, 0.9))))
        for sx, sy in statics:
            if rng.random() < static_rate:
                c.append(ShuttleCandidate(float(sx + rng.normal(0, 0.7)),
                                          float(sy + rng.normal(0, 0.7)),
                                          float(rng.uniform(0.4, 0.95))))
        rng.shuffle(c)
        cands.append(c)
    hit_t = np.array([s.t0 for s in shots if s.is_hit])
    hit_dv = np.array([_image_dv(shots[k - 1], shots[k]) for k in range(len(shots))
                       if shots[k].is_hit])
    hf = np.clip(np.searchsorted(t, hit_t), 1, n - 1)
    hf = np.where(np.abs(t[hf - 1] - hit_t) <= np.abs(t[hf] - hit_t), hf - 1, hf)
    meta = VideoMeta(fps=float(fps), width=W, height=H, n_frames=n, timestamps=t)
    return Case(cands, meta, truth, truth_all, detected, hf, hit_t, hit_dv)


# ── metrics ───────────────────────────────────────────────────────────────────

def observable_hits(case: Case, win_s: float = 0.2, min_pts: int = 3,
                    min_dv: float = OBS_DV) -> np.ndarray:
    """Hits with ≥ min_pts true detections within win_s on each side and an
    image-space velocity change ≥ min_dv px/s (a hit inside a burst of misses,
    off-image, or seen along the line of sight cannot be found in 2-D)."""
    t = case.meta.timestamps
    ok = case.hit_dv >= min_dv
    for k, ht in enumerate(case.hit_t):
        before = case.detected & (t >= ht - win_s) & (t < ht)
        after = case.detected & (t > ht) & (t <= ht + win_s)
        ok[k] &= before.sum() >= min_pts and after.sum() >= min_pts
    return ok


def evaluate(case: Case, tr: Track2D) -> dict:
    err = np.hypot(*(tr.xy - case.truth).T)
    det = case.detected
    have = np.isfinite(tr.xy[:, 0])
    no_truth = ~np.isfinite(case.truth[:, 0])
    in_view = ~no_truth
    err_all = np.hypot(*(tr.xy - case.truth_all).T)
    fp = tr.visible & ~(err_all <= 10.0)           # adopted detection that is no shuttle
    kf = find_kinks(tr, image_size=(W, H))
    hits_in = case.hit_frames[observable_hits(case)]
    found = (np.array([np.min(np.abs(kf - h)) <= 2 for h in hits_in])
             if len(kf) else np.zeros(len(hits_in), bool))
    near_hit = np.zeros(len(tr.t), bool)
    for h in hits_in:
        near_hit[max(h - 1, 0):h + 2] = True
    e_det = err[det]
    return {
        "within4": float(np.mean(e_det < 4.0)),
        "median": float(np.nanmedian(e_det)),
        "p95": float(np.nanpercentile(np.where(np.isfinite(e_det), e_det, 1e3), 95)),
        "fp_adopt": float(fp.sum() / max(tr.visible.sum(), 1)),
        "out_bad": float(np.mean(~(err_all[have] <= 10.0))) if np.any(have) else 0.0,
        "fill_med": float(np.nanmedian(err[tr.interpolated & in_view]))
        if np.any(tr.interpolated & in_view) else 0.0,
        "hit_recall": float(found.mean()) if len(found) else 1.0,
        "n_kinks": len(kf), "n_hits": len(hits_in),
        "false_kinks": int(np.sum([np.min(np.abs(case.hit_frames - k)) > 2 for k in kf]))
        if len(case.hit_frames) else len(kf),
        "hit_err": float(np.nanmax(np.where(det & near_hit, err, np.nan)))
        if np.any(det & near_hit) else 0.0,
        "hit_within4": float(np.mean(err[det & near_hit] < 4.0)) if np.any(det & near_hit) else 1.0,
    }


# ── helpers for the tests ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _run(seed: int, fps: float) -> tuple[Case, Track2D, dict, float]:
    case = make_case(seed, fps)
    t0 = time.perf_counter()
    tr = link_candidates(case.cands, case.meta)
    el = time.perf_counter() - t0
    return case, tr, evaluate(case, tr), el


def _hit_frame_errors(case: Case, tr: Track2D) -> np.ndarray:
    """Errors at detected frames adjacent to observable hits (h−1, h, h+1)."""
    err = np.hypot(*(tr.xy - case.truth).T)
    out = []
    for h in case.hit_frames[observable_hits(case)]:
        for f in range(max(h - 1, 0), min(h + 2, len(err))):
            if case.detected[f]:
                out.append(err[f])
    return np.asarray(out)


FPS = [30.0, 60.0, 120.0]


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _warm_jit() -> None:
    c = make_case(0, 30.0)
    link_candidates(c.cands[:200], VideoMeta(fps=30.0, width=W, height=H, n_frames=200,
                                             timestamps=c.meta.timestamps[:200]))


@pytest.mark.parametrize("fps", FPS)
@pytest.mark.parametrize("seed", SEEDS)
def test_link_accuracy_per_case(seed: int, fps: float) -> None:
    case, tr, m, _ = _run(seed, fps)
    assert tr.xy.shape == (case.meta.n_frames, 2) and tr.t.shape == (case.meta.n_frames,)
    assert m["within4"] >= 0.95, m                 # truly detected frames within 4 px
    assert m["median"] < SIGMA, m                  # smoothing beats the detector noise
    assert m["fp_adopt"] < 0.01, m                 # adopted false positives
    assert m["out_bad"] < 0.05, m                  # any output (incl. fills) > 10 px off
    assert not np.any(tr.visible & tr.interpolated)
    assert np.all(tr.conf[~tr.visible] == 0.0)
    assert np.all(np.isfinite(tr.xy[tr.visible | tr.interpolated]))


@pytest.mark.parametrize("fps", FPS)
def test_link_accuracy_pooled(fps: float) -> None:
    ms = [_run(s, fps)[2] for s in SEEDS]
    within4 = np.mean([m["within4"] for m in ms])
    fp = max(m["fp_adopt"] for m in ms)
    assert within4 >= 0.97, within4
    assert fp < 0.01, fp


@pytest.mark.parametrize("fps", FPS)
def test_kinks_found_at_hits(fps: float) -> None:
    ms = [_run(s, fps)[2] for s in SEEDS]
    n = np.array([m["n_hits"] for m in ms])
    recall = float(np.sum([m["hit_recall"] * m["n_hits"] for m in ms]) / n.sum())
    false = sum(m["false_kinks"] for m in ms)
    assert recall >= 0.85 if fps < 45 else recall >= 0.9, recall
    assert false <= 0.2 * n.sum(), (false, n.sum())


@pytest.mark.parametrize("fps", FPS)
def test_no_smoothing_across_hits(fps: float) -> None:
    e = np.concatenate([_hit_frame_errors(*_run(s, fps)[:2]) for s in SEEDS])
    assert np.median(e) < 2.0, np.median(e)
    assert np.mean(e < 4.0) >= 0.9, np.mean(e < 4.0)


def test_static_distractors_not_adopted() -> None:
    for seed in SEEDS[:4]:
        case, tr, _, _ = _run(seed, 60.0)
        # visible output never sits on a static distractor while the shuttle is elsewhere
        err_all = np.hypot(*(tr.xy - case.truth_all).T)
        bad = tr.visible & ~(err_all <= 10.0)
        assert bad.sum() <= 0.002 * tr.visible.sum(), bad.sum()


def test_gap_filling_and_dead_time() -> None:
    case, tr, m, _ = _run(1, 60.0)
    t = tr.t
    in_play = np.isfinite(case.truth_all[:, 0])
    # dead time (no shuttle for > max_gap_s) stays NaN
    dead = ~in_play
    assert np.mean(np.isfinite(tr.xy[dead, 0])) < 0.02
    # fills are close to the truth where the truth is known
    assert m["fill_med"] < 3.0, m
    # never fill a gap longer than max_gap_s between two real observations
    obs = np.flatnonzero(tr.visible)
    cfg = LinkerConfig()
    for a, b in zip(obs[:-1], obs[1:]):
        if t[b] - t[a] > cfg.max_gap_s + 1e-9:
            inner = np.arange(a + 1, b)
            far = inner[(t[inner] - t[a] > 0.05) & (t[b] - t[inner] > 0.05)]
            assert not np.any(tr.interpolated[far]), (a, b)


@pytest.mark.parametrize("scale", [720 / 1080, 2160 / 1080])
def test_resolution_invariance(scale: float) -> None:
    """Same rally at 720p / 4K (pixels scaled, noise scaled): same quality."""
    case = make_case(2, 60.0)
    cands = [[ShuttleCandidate(c.x * scale, c.y * scale, c.score) for c in fr]
             for fr in case.cands]
    meta = VideoMeta(fps=60.0, width=round(W * scale), height=round(H * scale),
                     n_frames=case.meta.n_frames, timestamps=case.meta.timestamps)
    tr = link_candidates(cands, meta)
    err = np.hypot(*(tr.xy / scale - case.truth).T)[case.detected]
    assert np.mean(err < 4.0) >= 0.96
    assert np.nanmedian(err) < SIGMA


def test_variable_frame_rate() -> None:
    """Drop 15 % of frames at random (VFR / decoder drops): timestamps rule."""
    case = make_case(3, 60.0)
    rng = np.random.default_rng(5)
    keep = np.sort(np.flatnonzero(rng.random(case.meta.n_frames) > 0.15))
    cands = [case.cands[i] for i in keep]
    meta = VideoMeta(fps=60.0, width=W, height=H, n_frames=len(keep),
                     timestamps=case.meta.timestamps[keep])
    tr = link_candidates(cands, meta)
    err = np.hypot(*(tr.xy - case.truth[keep]).T)[case.detected[keep]]
    assert np.mean(err < 4.0) >= 0.96
    assert np.nanmedian(err) < SIGMA


def test_smooth_track_and_find_kinks_api() -> None:
    case, tr, _, _ = _run(4, 60.0)
    kf, z = find_kinks(tr, image_size=(W, H), return_scores=True)
    assert kf.dtype.kind == "i" and np.all(np.diff(kf) >= 0) and len(z) == len(kf)
    assert np.all(z > 0)
    # re-smoothing is idempotent-ish and keeps the hits sharp
    tr2 = smooth_track(tr, image_size=(W, H))
    e1 = np.hypot(*(tr.xy - case.truth).T)[case.detected]
    e2 = np.hypot(*(tr2.xy - case.truth).T)[case.detected]
    assert np.nanmedian(e2) < np.nanmedian(e1) + 0.3
    # forcing the true hit frames as kinks: hit frames stay sharp
    raw = Track2D(t=case.meta.timestamps.copy(), xy=np.full((case.meta.n_frames, 2), np.nan),
                  conf=np.zeros(case.meta.n_frames), visible=case.detected.copy(),
                  interpolated=np.zeros(case.meta.n_frames, bool))
    rng = np.random.default_rng(0)
    raw.xy[case.detected] = case.truth[case.detected] + rng.normal(0, SIGMA, (case.detected.sum(), 2))
    raw.conf[case.detected] = 0.9
    tr3 = smooth_track(raw, kinks=case.hit_frames, image_size=(W, H))
    e3 = _hit_frame_errors(case, tr3)
    assert np.median(e3) < 2.0 and np.mean(e3 < 4.0) >= 0.9
    assert np.nanmedian(np.hypot(*(tr3.xy - case.truth).T)[case.detected]) < SIGMA


def test_degenerate_inputs() -> None:
    meta = VideoMeta(fps=30.0, width=W, height=H, n_frames=0, timestamps=np.zeros(0))
    tr = link_candidates([], meta)
    assert tr.xy.shape == (0, 2)
    n = 50
    meta = VideoMeta(fps=30.0, width=W, height=H, n_frames=n, timestamps=np.arange(n) / 30)
    tr = link_candidates([[] for _ in range(n)], meta)
    assert np.all(np.isnan(tr.xy)) and not tr.visible.any()
    one = [[] for _ in range(n)]
    one[10] = [ShuttleCandidate(100.0, 100.0, 0.9)]
    tr = link_candidates(one, meta)                 # a lone blip is not a track
    assert not tr.visible.any()
    # missing/duplicate timestamps are repaired, not fatal
    meta_bad = VideoMeta(fps=30.0, width=W, height=H, n_frames=n,
                         timestamps=np.r_[np.arange(n - 5) / 30, np.full(5, (n - 6) / 30)])
    line = [[ShuttleCandidate(100.0 + 5 * i, 200.0 + 2 * i, 0.9)] for i in range(n)]
    tr = link_candidates(line, meta_bad)
    assert tr.visible.sum() >= n - 6
    assert find_kinks(tr).dtype.kind == "i"


@pytest.mark.parametrize("size", [(192, 144), (64, 48), (1920, 1080)])
def test_weak_only_candidates(size: tuple[int, int]) -> None:
    """Regression (venue 192×144 phone clips): a path with no strong (≥ strong_score)
    observation used to raise IndexError in _weak_unsupported. With no strong obs,
    weak obs have no support: only a flying weak piece survives (weak_fly_speed)."""
    w, h = size
    diag = float(np.hypot(w, h))
    fps, n = 30.0, 12
    meta = VideoMeta(fps=fps, width=w, height=h, n_frames=n, timestamps=np.arange(n) / fps)
    v = 0.5 * diag / fps                                  # 0.5 D/s > weak_fly_speed (0.3 D/s)
    fly = [[ShuttleCandidate(0.1 * w + v * i * 0.9, 0.2 * h + v * i * 0.4, 0.3)] for i in range(n)]
    tr = link_candidates(fly, meta)
    assert tr.visible.sum() == n and not tr.interpolated.any()
    still = [[ShuttleCandidate(0.3 * w + 0.01 * i, 0.6 * h, 0.3)] for i in range(n)]
    tr = link_candidates(still, meta)                     # weak and not flying: dropped
    assert not tr.visible.any() and np.all(np.isnan(tr.xy))
    # one strong blip elsewhere does not support the weak run either
    mixed = [list(c) for c in still]
    mixed[3].append(ShuttleCandidate(0.8 * w, 0.2 * h, 0.51))
    tr = link_candidates(mixed, meta)
    assert not tr.visible.any()
    for sz, cands in (((w, h), [[] for _ in range(n)]), ((w, h), [[ShuttleCandidate(1.0, 1.0, 0.2)]])):
        m = VideoMeta(fps=fps, width=sz[0], height=sz[1], n_frames=len(cands),
                      timestamps=np.arange(len(cands)) / fps)
        assert not link_candidates(cands, m).visible.any()


def test_duplicate_pts_retimed() -> None:
    """Duplicate PTS are repeated frames: each run keeps its first time and is spread
    evenly to the next distinct time; a trailing / whole-clip run advances by the median
    step (else 1/fps); reversals repeat the previous time first. Always strictly increasing,
    and a strictly increasing clock is returned bit-identical."""
    from app.services.tracking.linker import _timestamps

    def T(ts, fps=30.0, n=None):
        n = len(ts) if n is None else n
        return _timestamps(VideoMeta(fps=fps, width=1, height=1, n_frames=n, timestamps=np.asarray(ts, float)), n)

    assert T([0, 1, 2, 2, 2, 2, 6, 7]).tolist() == [0, 1, 2, 3, 4, 5, 6, 7]       # interior run
    assert T([0, 0, 0, 3, 4]).tolist() == [0, 1, 2, 3, 4]                         # leading run
    assert T([0, 1, 2, 3, 3, 3]).tolist() == [0, 1, 2, 3, 4, 5]                   # trailing run
    assert T([0, 1, 0.5, 2]).tolist() == [0, 1, 1.5, 2]                           # reversal
    assert np.allclose(T([5, 5, 5, 5]), 5 + np.arange(4) / 30)                     # whole clip
    assert np.allclose(T([5, 5, 5, 5], fps=float("inf")), 5 + np.arange(4) / 30)  # fps inf → 30
    ok = np.arange(50) / 30 + np.random.default_rng(0).uniform(-0.002, 0.002, 50)
    assert np.array_equal(T(ok), ok)                                               # untouched
    for fps in (30.0, 60.0, 240.0):
        for run in (1200, 5000):                                   # frozen clock, then normal
            ts = np.r_[np.arange(40), np.full(run, 40.0), 41 + np.arange(60)] / fps
            t = T(ts, fps)
            assert np.all(np.diff(t) > 0) and t[40] == ts[40] and t[40 + run] == ts[40 + run]
            assert np.allclose(np.diff(t[40:41 + run]), 1.0 / fps / run)
        for ts in (np.r_[np.zeros(1500), 1 + np.arange(20) / fps],        # long leading run
                   np.r_[np.arange(20) / fps, np.full(1500, 1.0)],        # long trailing run
                   np.full(1200, 2.0)):                                   # whole clip
            assert np.all(np.diff(T(ts, fps)) > 0)
    assert np.all(np.diff(T([1e17, 1e17, 1e17, 1e17 + 64.0])) > 0)               # below float resolution


def test_solve_pieces_per_piece_fallback(monkeypatch) -> None:
    """If the whole-video banded solve raises LinAlgError, each piece is solved alone (the
    system is block diagonal): healthy pieces equal the whole solve, a failing piece stays
    unsmoothed (NaN → _finish passes its observations through)."""
    from app.services.tracking import linker as lk
    t = np.arange(60) / 60.0
    fr = np.r_[np.arange(0, 25), np.arange(35, 55)]                   # pieces of 25 and 20 vars
    p = np.column_stack([100 + 400 * t[fr], 200 + 300 * t[fr] ** 2])
    p += np.random.default_rng(0).normal(0, 1.0, p.shape)
    a, b = np.array([0, 25]), np.array([25, 45])
    lo, hi = fr[a], fr[b - 1]
    args = (t, fr, p, a, b, lo, hi, 1.5, float(np.hypot(1920, 1080)), lk.LinkerConfig())
    vf, x, r = lk._solve_pieces(*args)
    real = lk.solveh_banded

    def failing(widths):
        def solve(ab, rhs, **kw):
            if ab.shape[1] in widths:
                raise np.linalg.LinAlgError("forced")
            return real(ab, rhs, **kw)
        return solve

    monkeypatch.setattr(lk, "solveh_banded", failing({45}))
    vf2, x2, r2 = lk._solve_pieces(*args)
    assert np.array_equal(vf, vf2) and np.allclose(x, x2, rtol=0, atol=1e-9) and np.allclose(r, r2, atol=1e-9)
    monkeypatch.setattr(lk, "solveh_banded", failing({45, 20}))
    _, x3, r3 = lk._solve_pieces(*args)
    assert np.allclose(x3[:25], x[:25], rtol=0, atol=1e-9) and np.all(np.isnan(x3[25:]))
    assert np.allclose(r3[:25], r[:25], atol=1e-9) and np.all(np.isnan(r3[25:]))


def test_duplicate_timestamp_run_does_not_raise() -> None:
    """Regression: a run of duplicate timestamps inside an observed piece used to be nudged
    1e-3/fps apart, which made the jerk prior not positive definite → LinAlgError from the
    whole-video banded solve (and chaotic tracks where it did not raise). The run is now
    re-timed onto the frame grid, so the piece is well conditioned."""
    n, fps = 13, 30.0
    t = np.arange(n) / fps
    t[4:8] = t[4]
    cands = [[ShuttleCandidate(192.0 + i * 1536.0 / n, 216.0 + i * 540.0 / n, 0.7 if i % 3 else 0.2)]
             if (i // 4) % 2 == 0 else [] for i in range(n)]
    for fps_meta in (fps, 0.0, float("nan")):
        tr = link_candidates(cands, VideoMeta(fps=fps_meta, width=1920, height=1080, n_frames=n, timestamps=t))
        assert tr.xy.shape == (n, 2) and np.all(np.diff(tr.t) > 0)
        assert not np.any(tr.visible & tr.interpolated)
        assert np.all(tr.conf[~tr.visible] == 0.0)
        assert np.all(np.isfinite(tr.xy[tr.visible | tr.interpolated]))
        smooth_track(tr, image_size=(1920, 1080))
        find_kinks(tr, image_size=(1920, 1080))


def test_one_hour_60fps_runtime() -> None:
    """216 000 frames (tiled synthetic rallies incl. distractors) in < 20 s."""
    cases = [make_case(s, 60.0) for s in range(3)]
    cands: list[list[ShuttleCandidate]] = []
    ts, t0 = [], 0.0
    while len(cands) < 216_000:
        for c in cases:
            cands += c.cands
            ts.append(c.meta.timestamps + t0)
            t0 = ts[-1][-1] + 1 / 60
    cands = cands[:216_000]
    t = np.concatenate(ts)[:216_000]
    meta = VideoMeta(fps=60.0, width=W, height=H, n_frames=len(t), timestamps=t)
    t1 = time.perf_counter()
    tr = link_candidates(cands, meta)
    el = time.perf_counter() - t1
    assert tr.xy.shape == (216_000, 2)
    assert el < 20.0, el


def test_repeated_frames_25p_in_30p() -> None:
    """25 fps content in a 30 fps container (every 6th frame repeats the
    previous image, identical detections), with a pulldown phase jump in the
    middle: about as good as the native 25 fps clip, no false repeats."""
    from app.services.tracking.linker import _associate, _find_repeats, _probe_candidates
    rows = []
    for seed in (5, 6, 7, 8):
        src = make_case(seed, 25.0)
        k = np.arange(int(src.meta.n_frames * 30 / 25))
        s_idx = np.minimum(k * 25 // 30 + 2 * (k >= len(k) // 2), src.meta.n_frames - 1)
        cands = [src.cands[i] for i in s_idx]
        meta = VideoMeta(fps=30.0, width=W, height=H, n_frames=len(k), timestamps=k / 30.0)
        tr = link_candidates(cands, meta)
        det = src.detected[s_idx]
        err = np.hypot(*(tr.xy - src.truth[s_idx]).T)[det]
        kf = find_kinks(tr, image_size=(W, H))
        hits = np.searchsorted(s_idx, src.hit_frames[observable_hits(src)])
        found = [np.min(np.abs(kf - h)) <= 2 for h in hits] if len(kf) else [False] * len(hits)
        xy, _, _ = _associate(cands, meta.timestamps, W, H, LinkerConfig())
        rep = _find_repeats(meta.timestamps, xy, _probe_candidates(cands, xy, LinkerConfig(), W, H),
                            float(np.hypot(W, H)),
                            LinkerConfig())
        true_dup = np.r_[False, s_idx[1:] == s_idx[:-1]]
        f = rep[0] if rep is not None else np.zeros(len(k), bool)
        tr0 = link_candidates(src.cands, src.meta)
        e0 = np.hypot(*(tr0.xy - src.truth).T)[src.detected]
        rows.append((np.mean(err < 4.0), np.nanmedian(err), found, (f & true_dup).sum() / true_dup.sum(),
                     (f & ~true_dup).sum(), np.mean(e0 < 4.0)))
    within4 = np.mean([r[0] for r in rows])
    assert within4 >= 0.95 and within4 >= np.mean([r[5] for r in rows]) - 0.02, rows
    assert np.mean([r[1] for r in rows]) < SIGMA and max(r[1] for r in rows) < 1.15 * SIGMA
    assert np.mean(np.concatenate([r[2] for r in rows])) >= 0.8
    assert np.mean([r[3] for r in rows]) >= 0.6          # repeats recovered …
    assert sum(r[4] for r in rows) == 0                  # … and never invented


# ── regressions from the real-clip review ─────────────────────────────────────

def _dead_window(case: Case, min_s: float = 1.6) -> tuple[int, int]:
    """(a, b) frames of the longest stretch without a shuttle in play."""
    dead = ~np.isfinite(case.truth_all[:, 0])
    best, cur = (0, 0), None
    for i, d in enumerate(np.r_[dead, False]):
        if d and cur is None:
            cur = i
        elif not d and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    assert case.meta.timestamps[best[1] - 1] - case.meta.timestamps[best[0]] >= min_s
    return best


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_weak_walking_distractors_not_adopted(seed: int) -> None:
    """(a) A shoe / head that walks (1–3 px/frame at 60 fps) and fires weak
    candidates (0.2–0.45) in 80 % of frames — including right next to the
    flight — is never tracked; a held shuttle in dead time is kept iff it is
    seen strongly (TrackNetV2 match1 convention), not when it is only weak."""
    case = make_case(seed, 60.0)
    rng = np.random.default_rng(seed)
    n = case.meta.n_frames
    walk = np.cumsum(rng.normal(0, 1.0, (n, 2)) + [1.2, 0.3], 0) + [300.0, 700.0]
    a, b = _dead_window(case)
    mid = (a + b) // 2
    held = np.arange(mid - 24, mid + 24)                      # 0.8 s in the hand
    hp = np.array([900.0, 500.0]) + np.cumsum(rng.normal(0, 0.4, (len(held), 2)), 0)
    for strong in (True, False):
        cands = [list(c) for c in case.cands]
        for i in range(n):
            if rng.random() < 0.8:
                cands[i].append(ShuttleCandidate(float(walk[i, 0]), float(walk[i, 1]),
                                                 float(rng.uniform(0.2, 0.45))))
        for k, i in enumerate(held):
            cands[i].append(ShuttleCandidate(float(hp[k, 0]), float(hp[k, 1]),
                                             float(rng.uniform(0.7, 0.9) if strong
                                                   else rng.uniform(0.2, 0.4))))
        tr = link_candidates(cands, case.meta)
        on_walk = np.hypot(*(tr.xy - walk).T) < 10
        assert np.sum(on_walk & np.isfinite(tr.xy[:, 0])) <= 3, np.flatnonzero(on_walk)
        kept = np.mean(np.hypot(*(tr.xy[held] - hp).T) < 10)
        assert (kept >= 0.8) if strong else (kept <= 0.05), (strong, kept)
        err = np.hypot(*(tr.xy - case.truth).T)[case.detected]
        assert np.mean(err < 4.0) >= 0.95


def _cut_case(seed: int) -> tuple[Case, list[list[ShuttleCandidate]], int, int]:
    """Mid-flight camera cut at frame c: the new shot shows the court from
    elsewhere (everything shifted, other static spots) and the shuttle is
    missed for 6 frames — a jump a linker could otherwise bridge and fill."""
    case = make_case(seed, 60.0)
    t = case.meta.timestamps
    ok = case.detected & (t > 4.0) & (t < t[-1] - 4.0)
    c = int(np.flatnonzero(ok & np.r_[ok[1:], False] & np.r_[False, ok[:-1]])[len(ok) // 7])
    rng = np.random.default_rng(seed + 99)
    new_static = np.column_stack([rng.uniform(100, W - 100, 3), rng.uniform(400, H - 60, 3)])
    cands = []
    for i, cs in enumerate(case.cands):
        if i < c:
            cands.append(list(cs))
            continue
        keep = [] if i < c + 6 else [ShuttleCandidate(q.x + 45.0, q.y + 25.0, q.score)
                                     for q in cs if q.score >= 0.55 or rng.random() < 0.2]
        for sx, sy in new_static:
            if rng.random() < 0.6:
                keep.append(ShuttleCandidate(float(sx + rng.normal(0, 0.7)),
                                             float(sy + rng.normal(0, 0.7)), 0.6))
        cands.append(keep)
    return case, cands, c, c + 6


@pytest.mark.parametrize("seed", [1, 4, 8])
def test_never_links_across_scene_cut(seed: int) -> None:
    """(b) Nothing is linked, filled or extrapolated across a cut, whether it
    is passed in or inferred from the candidate stream (static spots end and
    others start at once); normal clips never show a cut."""
    case, cands, c, c2 = _cut_case(seed)
    for cuts in ([c], None):
        tr = link_candidates(cands, case.meta, scene_cuts=cuts)
        assert np.all(np.isnan(tr.xy[c:c2, 0])), (cuts, np.flatnonzero(np.isfinite(tr.xy[c:c2, 0])))
    from app.services.tracking.linker import _detect_cuts
    for s in SEEDS:
        for fps in FPS:
            k = make_case(s, fps)
            assert len(_detect_cuts(k.cands, k.meta.timestamps, W, H, LinkerConfig())) == 0


def test_raw_xy_is_the_adopted_detection() -> None:
    case, tr, _, _ = _run(2, 60.0)
    assert tr.raw_xy is not None and tr.raw_xy.shape == tr.xy.shape
    fin = np.isfinite(tr.raw_xy[:, 0])
    assert np.array_equal(fin, tr.visible)
    for f in np.flatnonzero(fin)[::7]:
        d = min(np.hypot(q.x - tr.raw_xy[f, 0], q.y - tr.raw_xy[f, 1]) for q in case.cands[f])
        assert d < 1e-6
    tr2 = smooth_track(tr, image_size=(W, H))
    assert tr2.raw_xy is not None and np.array_equal(np.isfinite(tr2.raw_xy[:, 0]), tr2.visible)


# ── presentation-clock gaps (video_io.pts_gaps) ──────────────────────────────

def _gap_case(seed: int, kind: str) -> tuple[Case, list[list[ShuttleCandidate]], VideoMeta, int]:
    """A mid-flight hole in the clock at frame g: 'drop' removes 26 frames (0.43 s at
    60 fps, the F4 repro), 'jump' shifts every later time by 5 s (the F3 repro)."""
    case = make_case(seed, 60.0)
    t = case.meta.timestamps
    ok = case.detected & (t > 4.0) & (t < t[-1] - 4.0)
    g = int(np.flatnonzero(ok & np.r_[ok[1:], False] & np.r_[False, ok[:-1]])[len(ok) // 9])
    if kind == "drop":
        keep = np.r_[np.arange(g), np.arange(g + 26, len(t))]
        ts = t[keep]
    else:
        keep = np.arange(len(t))
        ts = t.copy()
        ts[g:] += 5.0
    cands = [case.cands[i] for i in keep]
    meta = VideoMeta(fps=60.0, width=W, height=H, n_frames=len(keep), timestamps=ts)
    return case, cands, meta, g


@pytest.mark.parametrize("kind", ["drop", "jump"])
@pytest.mark.parametrize("seed", [1, 4, 8])
def test_pts_gap_is_a_cut(seed: int, kind: str) -> None:
    """A clock hole acts exactly like a scene cut at its first frame: same track as with
    the cut passed explicitly, the first observation after it starts a new chain, and
    smoothing / kinks never reach across it."""
    from app.services.tracking.video_io import pts_gaps
    case, cands, meta, g = _gap_case(seed, kind)
    assert pts_gaps(meta.timestamps).tolist() == [g]
    tr = link_candidates(cands, meta)
    ref = link_candidates(cands, meta, scene_cuts=[g])
    for a, b in [(tr.xy, ref.xy), (tr.visible, ref.visible), (tr.interpolated, ref.interpolated),
                 (tr.conf, ref.conf), (tr.t, ref.t)]:
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    # no frame on either side is a fill between the two sides
    last, first = np.flatnonzero(tr.visible[:g]), np.flatnonzero(tr.visible[g:]) + g
    if len(last) and len(first):
        assert not tr.interpolated[last[-1]:first[0] + 1].any()
    # re-smoothing / kinks: identical to the explicit cut
    sm, sm_ref = smooth_track(tr, image_size=(W, H)), smooth_track(tr, image_size=(W, H), scene_cuts=[g])
    assert np.array_equal(sm.xy, sm_ref.xy, equal_nan=True)
    k, k_ref = find_kinks(tr, image_size=(W, H)), find_kinks(tr, image_size=(W, H), scene_cuts=[g])
    assert np.array_equal(k, k_ref)


def test_pts_gap_changes_nothing_without_gap() -> None:
    """Normal clocks (iid jitter, VFR drops of 15 %) never produce a gap."""
    from app.services.tracking.video_io import pts_gaps
    for s in SEEDS:
        for fps in (30.0, 60.0, 120.0):
            assert pts_gaps(make_case(s, fps).meta.timestamps).size == 0
    case = make_case(3, 60.0)
    rng = np.random.default_rng(5)
    keep = np.sort(np.flatnonzero(rng.random(case.meta.n_frames) > 0.15))
    assert pts_gaps(case.meta.timestamps[keep]).size == 0
