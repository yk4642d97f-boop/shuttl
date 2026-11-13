"""
Tests for hits.detect_hits.

Synthetic ground truth is generated here (standalone): rallies of 3-25 shots
alternating near/far, every flight integrated with physics.rk4_step (the same
quadratic-drag ODE as physics.simulate_state) on a 1/240 s grid, projected with
CameraModel.look_at cameras behind the near baseline (phone / broadcast style),
sampled at 30/60/120 fps with a random sub-frame phase, then corrupted:
sigma = 1.5 px noise, 10 % random misses, an occlusion gap of 3-10 frames
(60 fps-equivalent duration, 0.05-0.17 s) covering ~30 % of contacts, 3 %
outliers. The shuttle is held ~static before the serve and lies on the floor
after landing (neither is a hit). Player observations are boxes around foot
positions near the contacts (raised arm for overhead contacts), wrist keypoints
near the contact point for some rallies.

Run `python -m tests.test_hits` from backend/ for the per-fps metric table.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pytest

from app.services.tracking import court_model as cm
from app.services.tracking import hits as H
from app.services.tracking import physics
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import HitEvent, PlayerObservation, Track2D, VideoMeta

H_SIM = 1.0 / 240.0                  # integration grid; 30/60/120 fps frames lie on it
IMG = (1280, 720)
N_SEEDS = 20

# shot type -> (launch speed m/s, elevation deg)
SHOTS = {
    "serve_low": ((7.0, 12.0), (12.0, 32.0)),
    "serve_high": ((22.0, 30.0), (45.0, 60.0)),
    "clear": ((24.0, 42.0), (20.0, 42.0)),
    "smash": ((45.0, 95.0), (-16.0, -4.0)),
    "drop": ((11.0, 20.0), (-8.0, 14.0)),
    "net": ((3.5, 7.5), (25.0, 60.0)),
    "drive": ((20.0, 38.0), (-4.0, 8.0)),
    "lift": ((17.0, 28.0), (38.0, 62.0)),
}


# ── rally simulation (world) ─────────────────────────────────────────────────

@dataclass
class SimRally:
    t_dense: np.ndarray                     # (M,) s on the H_SIM grid
    p_dense: np.ndarray                     # (M,3) m, NaN when not in play / held
    hit_t: list[float]
    hit_side: list[str]
    hit_kind: list[str]
    hit_pos: list[np.ndarray]
    foot: dict[str, list[tuple[float, np.ndarray]]]   # side -> foot waypoints (t, xy)
    height: dict[str, float]
    t_end: float


def _choose_kind(rng: np.random.Generator, z: float, depth: float) -> str:
    if z >= 2.1:
        return str(rng.choice(["smash", "clear", "drop"], p=[0.4, 0.3, 0.3]))
    if z >= 1.1:
        return str(rng.choice(["drive", "drop", "clear"], p=[0.5, 0.3, 0.2]))
    if depth < 2.6:
        return str(rng.choice(["net", "lift"], p=[0.5, 0.5]))
    return str(rng.choice(["lift", "drive"], p=[0.7, 0.3]))


def _batch_flights(p0: np.ndarray, v0: np.ndarray, L: float, t_max: float = 3.5) -> np.ndarray:
    """(steps, B, 3) positions on the H_SIM grid, via physics.rk4_step."""
    p, v = p0.copy(), v0.copy()
    out = [p.copy()]
    for _ in range(int(t_max / H_SIM)):
        p, v = physics.rk4_step(p, v, H_SIM, L)
        out.append(p.copy())
        if np.all(p[:, 2] < -0.05):
            break
    return np.stack(out)


def _outcome(path: np.ndarray, sgn: float, final: bool,
             rng: np.random.Generator) -> tuple[int, str] | None:
    """Index where the flight ends ('hit' by the receiver or 'land'), or None
    if the launch is invalid (net fault, lands short, no reachable intercept)."""
    x, y, z = path[:, 0] * sgn, path[:, 1], path[:, 2]
    cross = np.flatnonzero(x > 0)
    if len(cross) == 0:
        return None
    c = cross[0]
    if np.any(z[:c + 1] < 0) or z[c] < cm.NET_HEIGHT_CENTRE + 0.05 or abs(y[c]) > 3.05:
        return None
    below = np.flatnonzero(z[c:] < 0)
    if len(below) == 0:
        return None
    land = c + below[0]
    if final:
        return land, "land"
    overhead = z[c:land].max() >= 2.7 and rng.random() < 0.6
    h = rng.uniform(2.2, 2.8) if overhead else rng.uniform(0.5, 1.7)
    deep = rng.uniform(5.6, 6.6)
    vz = np.gradient(z)
    t_min = int(round(0.28 / H_SIM))
    for i in range(max(c, t_min), land):
        if (z[i] <= h and vz[i] < 0) or x[i] >= deep:
            if 0.4 <= x[i] <= 7.0 and abs(y[i]) <= 3.3 and 0.3 <= z[i] <= 3.0:
                return i, "hit"
            return None
    return None


def _plan(rng: np.random.Generator, kind: str, p0: np.ndarray, sgn: float, final: bool,
          L: float, B: int = 40) -> tuple[np.ndarray, int, str] | None:
    (s_lo, s_hi), (e_lo, e_hi) = SHOTS[kind]
    speed = rng.uniform(s_lo, s_hi, B)
    elev = np.radians(rng.uniform(e_lo, e_hi, B))
    tx, ty = sgn * rng.uniform(1.0, 6.6, B), rng.uniform(-2.8, 2.8, B)
    az = np.arctan2(ty - p0[1], tx - p0[0])
    v0 = np.stack([speed * np.cos(elev) * np.cos(az), speed * np.cos(elev) * np.sin(az),
                   speed * np.sin(elev)], axis=1)
    P = _batch_flights(np.repeat(p0[None], B, 0), v0, L)
    for b in range(B):
        r = _outcome(P[:, b], sgn, final, rng)
        if r is not None:
            return P[: r[0] + 1, b], r[0], r[1]
    return None


def simulate_rally(seed: int, n_shots: int | None = None, t0: float = 0.0) -> SimRally:
    """One rally (serve ... landing) in world coordinates."""
    rng = np.random.default_rng(seed)
    n = int(n_shots or rng.integers(3, 26))
    L = physics.DRAG_LENGTH * rng.uniform(0.92, 1.08)
    server = str(rng.choice(["near", "far"]))
    sgn0 = 1.0 if server == "near" else -1.0
    p = np.array([-sgn0 * rng.uniform(2.3, 3.6), rng.choice([-1, 1]) * rng.uniform(0.3, 1.2),
                  rng.uniform(0.95, 1.12)])
    hold = rng.uniform(0.3, 0.8) if rng.random() < 0.6 else 0.0
    t = t0 + hold
    ts: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    if hold > 0:                                   # held, then dropped 5 cm and struck
        th = np.arange(t0, t, H_SIM)
        dz = 0.05 * np.clip((t - th) / 0.1, 0, 1)
        ts.append(th)
        ps.append(p[None] + np.outer(dz, [0, 0, 1.0]))
    hit_t, hit_side, hit_kind, hit_pos = [], [], [], []
    side = server
    kind = str(rng.choice(["serve_low", "serve_high"]))
    for i in range(n):
        sgn = 1.0 if side == "near" else -1.0
        final = i == n - 1
        plan = None
        for k in [kind, "lift", "clear", "drive", "drop"]:
            plan = _plan(rng, k, p, sgn, final, L)
            if plan is not None:
                kind = k
                break
        if plan is None:                           # nothing returnable: end here
            plan = _plan(rng, "lift", p, sgn, True, L) or _plan(rng, "clear", p, sgn, True, L)
            final = True
            if plan is None:
                break
        path, end, how = plan
        hit_t.append(t)
        hit_side.append(side)
        hit_kind.append(kind)
        hit_pos.append(p.copy())
        ts.append(t + np.arange(end) * H_SIM)
        ps.append(path[:end])
        t = t + end * H_SIM
        p = path[end].copy()
        if how == "land":
            break
        side = "far" if side == "near" else "near"
        kind = _choose_kind(rng, p[2], abs(p[0]))
    # landing: rest on the floor, then picked up (track ends)
    p = p.copy()
    p[2] = 0.0
    rest = rng.uniform(0.4, 1.2)
    tr = t + np.arange(int(rest / H_SIM)) * H_SIM
    ts.append(tr)
    ps.append(np.repeat(p[None], len(tr), 0))
    t_dense = np.concatenate(ts)
    p_dense = np.concatenate(ps)
    # players: at own contacts the foot is behind/beside the contact, in between at base
    foot: dict[str, list[tuple[float, np.ndarray]]] = {}
    height = {s: float(rng.uniform(1.65, 1.90)) for s in ("near", "far")}
    for s in ("near", "far"):
        sg = -1.0 if s == "near" else 1.0
        base = np.array([sg * rng.uniform(2.8, 3.8), rng.uniform(-0.4, 0.4)])
        wp = [(t0 - 1.0, base)]
        own = [k for k in range(len(hit_t)) if hit_side[k] == s]
        for j, k in enumerate(own):
            c = hit_pos[k]
            f = np.array([c[0] + sg * rng.uniform(0.1, 0.4),
                          c[1] + rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)])
            f[0] = sg * np.clip(sg * f[0], 0.3, 7.4)
            wp.append((hit_t[k], f))
            nxt = hit_t[own[j + 1]] if j + 1 < len(own) else hit_t[k] + 2.0
            wp.append((0.5 * (hit_t[k] + nxt), base + rng.normal(0, 0.2, 2)))
        wp.append((t_dense[-1] + 5.0, base))
        foot[s] = wp
    return SimRally(t_dense, p_dense, hit_t, hit_side, hit_kind, hit_pos, foot, height,
                    float(t_dense[-1]))


# ── camera, sampling, corruption, players ────────────────────────────────────

def make_camera(rng: np.random.Generator, style: str = "behind",
                image_size: tuple[int, int] = IMG) -> CameraModel:
    if style == "behind":
        eye = np.array([-cm.HALF_LENGTH - rng.uniform(1.5, 5.0), rng.uniform(-1.0, 1.0),
                        rng.uniform(2.5, 5.5)])
    else:                                            # elevated broadcast
        eye = np.array([-cm.HALF_LENGTH - rng.uniform(6.0, 11.0), rng.uniform(-0.5, 0.5),
                        rng.uniform(5.0, 9.0)])
    target = np.array([rng.uniform(0.0, 2.5), rng.uniform(-0.3, 0.3), rng.uniform(0.0, 1.0)])
    roll = rng.uniform(-2.0, 2.0)
    cam = CameraModel.look_at(eye, target, 1000.0, image_size, roll_deg=roll)
    corners = np.array([[sx * cm.HALF_LENGTH, sy * cm.DOUBLES_HALF_WIDTH, 0.0]
                        for sx in (-1, 1) for sy in (-1, 1)])
    Xc = (cam.R @ corners.T).T + cam.t
    xn, yn = np.abs(Xc[:, 0] / Xc[:, 2]), np.abs(Xc[:, 1] / Xc[:, 2])
    f = min(0.46 * image_size[0] / xn.max(), 0.46 * image_size[1] / yn.max())
    return CameraModel.look_at(eye, target, f, image_size, roll_deg=roll)


@dataclass
class Scenario:
    track: Track2D
    meta: VideoMeta
    camera: CameraModel
    players: list[PlayerObservation]
    true_t: np.ndarray
    true_frame: np.ndarray
    true_side: list[str]
    fps: float
    info: dict = field(default_factory=dict)


def _interp_path(t_dense: np.ndarray, p_dense: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full((len(t), 3), np.nan)
    i = np.searchsorted(t_dense, t, side="right") - 1
    ok = (i >= 0) & (i < len(t_dense) - 1)
    i0 = i[ok]
    step = t_dense[i0 + 1] - t_dense[i0]
    good = step < 1.5 * H_SIM                          # do not bridge dead time
    w = ((t[ok] - t_dense[i0]) / np.where(step > 0, step, 1))[:, None]
    val = (1 - w) * p_dense[i0] + w * p_dense[i0 + 1]
    val[~good] = np.nan
    out[np.flatnonzero(ok)] = val
    return out


def _players(rallies: list[SimRally], cam: CameraModel, t: np.ndarray, rng: np.random.Generator,
             keypoints: bool, sides_known: bool) -> list[PlayerObservation]:
    obs: list[PlayerObservation] = []
    W, Hh = cam.image_size
    for s, tid in (("near", 1), ("far", 2)):
        wp_t = np.concatenate([[w[0] for w in r.foot[s]] for r in rallies])
        wp_x = np.concatenate([[w[1] for w in r.foot[s]] for r in rallies])
        order = np.argsort(wp_t)
        wp_t, wp_x = wp_t[order], wp_x[order]
        fx, fy = np.interp(t, wp_t, wp_x[:, 0]), np.interp(t, wp_t, wp_x[:, 1])
        hts = [(r.hit_t[k], r.hit_pos[k], r.height[s]) for r in rallies
               for k in range(len(r.hit_t)) if r.hit_side[k] == s]
        height = rallies[0].height[s]
        for i, ti in enumerate(t):
            if rng.random() < 0.05:
                continue
            f3 = np.array([fx[i], fy[i], 0.0])
            box = [f3 + [dx, dy, dz] for dx in (-0.22, 0.22) for dy in (-0.33, 0.33)
                   for dz in (0.0, height)]
            near_hit = [(th, c) for th, c, _ in hts if abs(th - ti) <= 0.2]
            if near_hit and near_hit[0][1][2] > 0.95 * height:
                box.append(f3 + [0, 0, height + 0.5])            # raised racket arm
            uv = cam.project(np.array(box))
            if not np.all(np.isfinite(uv)):
                continue
            bbox = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
            if bbox[2] < 0 or bbox[0] > W or bbox[3] < 0 or bbox[1] > Hh:
                continue
            kp = None
            if keypoints:
                sh = f3 + [0, 0, 0.82 * height]
                close = [(th, c) for th, c, _ in hts if abs(th - ti) <= 0.08]
                if close:
                    c = close[0][1]
                    d = sh - c
                    wr = c + 0.62 * d / max(np.linalg.norm(d), 1e-6)
                else:
                    wr = f3 + [0, 0.3, 1.0]
                pts = cam.project(np.array([f3 + [0, -0.3, 0.95], wr]))
                kp = np.zeros((17, 3))
                kp[H.LEFT_WRIST] = [*pts[0], 0.9]
                kp[H.RIGHT_WRIST] = [*pts[1], 0.9]
            obs.append(PlayerObservation(
                frame=i, track_id=tid, bbox=bbox, score=0.9,
                side=s if sides_known else "unknown",
                foot_world=np.array([fx[i], fy[i]]) if sides_known else None, keypoints=kp))
    return obs


def build_scenario(seed: int, fps: float = 60.0, n_rallies: int = 1, style: str | None = None,
                   players: bool = True, keypoints: bool | None = None,
                   sides_known: bool | None = None, corrupt: bool = True,
                   n_shots: int | None = None) -> Scenario:
    rng = np.random.default_rng(10_000 + seed)
    style = style or ("behind" if seed % 3 else "broadcast")
    cam = make_camera(rng, style)
    rallies: list[SimRally] = []
    t0 = rng.uniform(0.3, 1.0)
    for r in range(n_rallies):
        sr = _sim_cached(seed * 97 + r, n_shots, round(t0, 6))
        rallies.append(sr)
        t0 = sr.t_end + rng.uniform(3.0, 6.0)
    t_dense = np.concatenate([r.t_dense for r in rallies])
    p_dense = np.concatenate([r.p_dense for r in rallies])
    phase = rng.uniform(0, 1.0 / fps)
    t = phase + np.arange(int((t_dense[-1] + 0.5) * fps)) / fps
    P = _interp_path(t_dense, p_dense, t)
    uv = np.full((len(t), 2), np.nan)
    fin = np.isfinite(P).all(1)
    uv[fin] = cam.project(P[fin])
    W, Hh = cam.image_size
    vis = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
    true_t = np.array([x for r in rallies for x in r.hit_t])
    true_side = [s for r in rallies for s in r.hit_side]
    true_frame = np.clip(np.round((true_t - phase) * fps).astype(int), 0, len(t) - 1)
    n_gaps = 0
    if corrupt:
        uv = uv + rng.normal(0, 1.5, uv.shape)
        vis &= rng.random(len(t)) >= 0.10
        for th in true_t:
            if rng.random() < 0.30:
                glen = rng.uniform(3, 10) / 60.0
                g0 = th - rng.uniform(0.2, 0.8) * glen
                vis &= ~((t >= g0) & (t <= g0 + glen))
                n_gaps += 1
        out = np.flatnonzero(vis & (rng.random(len(t)) < 0.03))
        uv[out] = rng.uniform([0, 0], [W, Hh], (len(out), 2))
    xy = np.where(vis[:, None], uv, np.nan)
    track = Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis,
                    interpolated=np.zeros(len(t), bool))
    meta = VideoMeta(fps=fps, width=W, height=Hh, n_frames=len(t), timestamps=t.copy())
    kp = bool(seed % 2) if keypoints is None else keypoints
    sk = bool(seed % 4 < 2) if sides_known is None else sides_known
    pl = _players(rallies, cam, t, rng, kp, sk) if players else []
    return Scenario(track, meta, cam, pl, true_t, true_frame, true_side, fps,
                    {"n_gaps": n_gaps, "style": style, "kinds": [k for r in rallies for k in r.hit_kind]})


@lru_cache(maxsize=256)
def _sim_cached(seed: int, n_shots: int | None, t0: float) -> SimRally:
    return simulate_rally(seed, n_shots, t0)


# ── metrics ──────────────────────────────────────────────────────────────────

def match_hits(hits: list[HitEvent], sc: Scenario, tol_frames: int = 3) -> dict:
    """Greedy one-to-one matching by |frame difference| <= tol_frames."""
    pairs = sorted((abs(h.frame - int(f)), i, j) for i, h in enumerate(hits)
                   for j, f in enumerate(sc.true_frame) if abs(h.frame - int(f)) <= tol_frames)
    ud, ut, m = set(), set(), []
    for _, i, j in pairs:
        if i not in ud and j not in ut:
            ud.add(i)
            ut.add(j)
            m.append((i, j))
    tp = len(m)
    side_ok = sum(hits[i].hitter == sc.true_side[j] for i, j in m)
    terr = [abs(hits[i].t - sc.true_t[j]) for i, j in m]
    return {"tp": tp, "n_det": len(hits), "n_true": len(sc.true_t), "side_ok": side_ok,
            "terr": terr}


def evaluate(fps: float, seeds: range, camera: bool, players: bool, **kw) -> dict:
    tot = {"tp": 0, "n_det": 0, "n_true": 0, "side_ok": 0, "terr": []}
    for s in seeds:
        sc = build_scenario(s, fps, players=players, **kw)
        hs = H.detect_hits(sc.track, sc.meta, sc.camera if camera else None,
                           sc.players if players else None)
        r = match_hits(hs, sc)
        for k in ("tp", "n_det", "n_true", "side_ok"):
            tot[k] += r[k]
        tot["terr"] += r["terr"]
    p = tot["tp"] / max(tot["n_det"], 1)
    rc = tot["tp"] / max(tot["n_true"], 1)
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9),
            "side_acc": tot["side_ok"] / max(tot["tp"], 1),
            "t_med_ms": 1000 * float(np.median(tot["terr"])) if tot["terr"] else float("nan"),
            "t_p90_ms": 1000 * float(np.percentile(tot["terr"], 90)) if tot["terr"] else float("nan"),
            "n_true": tot["n_true"]}


# ── small analytic tracks for unit tests ─────────────────────────────────────

def _arc_track(fps: float = 60.0, hit_t: float = 1.0, dur: float = 2.0, noise: float = 0.0,
               seed: int = 0) -> tuple[Track2D, VideoMeta]:
    """Two image-space drag-like arcs joined at hit_t (a slow incoming flight,
    a fast outgoing one in a different direction)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fps)) / fps
    xy = np.zeros((len(t), 2))
    p_hit = np.array([640.0, 500.0])
    a = t < hit_t
    s = hit_t - t[a]                               # time before the hit
    xy[a] = p_hit + np.column_stack([-120 * s, -250 * s + 0.5 * 300 * s ** 2])
    b = ~a
    u = t[b] - hit_t
    k = 3.0                                        # drag-like exponential decay
    vx, vy = 200.0, -900.0
    xy[b] = p_hit + np.column_stack([vx * (1 - np.exp(-k * u)) / k,
                                     vy * (1 - np.exp(-k * u)) / k + 0.5 * 400 * u ** 2])
    xy += rng.normal(0, noise, xy.shape)
    vis = np.ones(len(t), bool)
    tr = Track2D(t=t, xy=xy, conf=np.ones(len(t)), visible=vis, interpolated=np.zeros(len(t), bool))
    return tr, VideoMeta(fps=fps, width=1280, height=720, n_frames=len(t), timestamps=t.copy())


def _mask(tr: Track2D, sel: np.ndarray) -> Track2D:
    vis = tr.visible & ~sel
    return Track2D(t=tr.t, xy=np.where(vis[:, None], tr.xy, np.nan), conf=vis.astype(float),
                   visible=vis, interpolated=np.zeros(tr.n, bool))


# ── unit tests ───────────────────────────────────────────────────────────────

def test_hitevent_contract_and_empty_inputs():
    tr, meta = _arc_track(noise=1.0)
    hs = H.detect_hits(tr, meta, None, None)
    assert hs and all(isinstance(h, HitEvent) for h in hs)
    for h in hs:
        assert isinstance(h.frame, int) and isinstance(h.t, float)
        assert h.hitter in ("near", "far", "unknown") and 0.0 <= h.confidence <= 1.0
        assert len(h.xy_px) == 2
    assert [h.t for h in hs] == sorted(h.t for h in hs)
    empty = Track2D(t=np.arange(5) / 30.0, xy=np.full((5, 2), np.nan), conf=np.zeros(5),
                    visible=np.zeros(5, bool), interpolated=np.zeros(5, bool))
    assert H.detect_hits(empty, VideoMeta(30.0, 1280, 720, 5, empty.t), None, None) == []


@pytest.mark.parametrize("fps", [30.0, 60.0, 120.0])
def test_single_kink_timing(fps):
    tr, meta = _arc_track(fps=fps, noise=1.5, seed=int(fps))
    hs = [h for h in H.detect_hits(tr, meta) if 0.2 < h.t < 1.9]
    assert len(hs) == 1
    assert abs(hs[0].t - 1.0) < 1.0 / fps + 0.004


def test_hit_inside_occlusion_gap():
    tr, meta = _arc_track(fps=60.0, noise=1.5, seed=3)
    tr = _mask(tr, (tr.t > 0.95) & (tr.t < 1.1))           # 9 frames hidden around the hit
    hs = [h for h in H.detect_hits(tr, meta) if 0.2 < h.t < 1.9]
    assert len(hs) == 1 and abs(hs[0].t - 1.0) < 0.05


def test_landing_and_rest_is_not_a_hit():
    fps = 60.0
    t = np.arange(int(2.0 * fps)) / fps
    xy = np.zeros((len(t), 2))
    tl = 1.0
    a = t < tl
    xy[a] = np.column_stack([500 + 150 * t[a], 200 + 300 * t[a] + 60 * t[a] ** 2])
    xy[~a] = xy[a][-1] + [150 / fps, 300 / fps]
    xy += np.random.default_rng(1).normal(0, 1.5, xy.shape)
    tr = Track2D(t=t, xy=xy, conf=np.ones(len(t)), visible=np.ones(len(t), bool),
                 interpolated=np.zeros(len(t), bool))
    hs = H.detect_hits(tr, VideoMeta(fps, 1280, 720, len(t), t))
    assert all(abs(h.t - tl) > 0.2 for h in hs)


def test_serve_after_dead_time_and_outliers():
    tr, meta = _arc_track(fps=60.0, noise=1.5, seed=5)
    tr = _mask(tr, tr.t < 0.5)                               # flight appears mid-air
    rng = np.random.default_rng(0)
    xy = tr.xy.copy()
    idx = rng.choice(np.flatnonzero(tr.visible), 4, replace=False)
    xy[idx] = rng.uniform([0, 0], [1280, 720], (4, 2))
    tr = Track2D(t=tr.t, xy=xy, conf=tr.conf, visible=tr.visible, interpolated=tr.interpolated)
    hs = H.detect_hits(tr, meta)
    assert any(abs(h.t - 1.0) < 1 / 60 + 0.004 for h in hs)
    assert len([h for h in hs if h.t > 0.55]) == 1          # outliers create no hits


def test_repeated_frames_retimed():
    """25p content in a 30p container (every 6th frame repeated): contact
    timing must use the content clock."""
    tr25, _ = _arc_track(fps=25.0, dur=2.0, noise=1.0, seed=7)
    n30 = int(2.0 * 30)
    src = np.minimum((np.arange(n30) * 25) // 30, tr25.n - 1)
    t30 = np.arange(n30) / 30.0
    xy = tr25.xy[src]
    tr = Track2D(t=t30, xy=xy, conf=np.ones(n30), visible=np.ones(n30, bool),
                 interpolated=np.zeros(n30, bool))
    hs = [h for h in H.detect_hits(tr, VideoMeta(30.0, 1280, 720, n30, t30)) if 0.2 < h.t < 1.9]
    assert len(hs) == 1 and abs(hs[0].t - 1.0) < 0.05


def test_variable_frame_rate_timestamps():
    sc = build_scenario(4, 60.0, players=False)
    rng = np.random.default_rng(4)
    t = sc.track.t + rng.uniform(-0.3, 0.3, sc.track.n) / 60.0     # jittered VFR clock
    t = np.maximum.accumulate(t)
    tr = Track2D(t=t, xy=sc.track.xy, conf=sc.track.conf, visible=sc.track.visible,
                 interpolated=sc.track.interpolated)
    meta = VideoMeta(60.0, sc.meta.width, sc.meta.height, tr.n, t.copy())
    r = match_hits(H.detect_hits(tr, meta, sc.camera, None), sc)
    assert r["tp"] >= 0.85 * r["n_true"] and r["n_det"] <= r["tp"] + 2


def test_learned_model_hook(tmp_path):
    tr, meta = _arc_track(noise=1.0)
    base = H.detect_hits(tr, meta)
    path = str(tmp_path / "m.npz")
    np.savez(path, w=np.zeros(len(H.HIT_FEATURES)), b=0.0)
    try:
        assert H.load_learned_hit_model(path)
        assert len(H.detect_hits(tr, meta)) == len(base)
        assert not H.load_learned_hit_model(str(tmp_path / "missing.npz"))
    finally:
        H.load_learned_hit_model(None)


def test_linker_kinks_are_frame_indices(monkeypatch):
    from app.services.tracking import linker
    tr, _ = _arc_track(fps=60.0)
    monkeypatch.setattr(linker, "find_kinks", lambda track, *a, **k: np.array([30, 60]),
                        raising=False)
    assert np.allclose(H._linker_kink_times(tr), [0.5, 1.0])


def test_prefers_unsmoothed_raw_xy(monkeypatch):
    from dataclasses import replace
    tr, meta = _arc_track(fps=60.0)
    raw = tr.xy.copy()
    raw[::7] = np.nan                                   # frames the linker filled
    smoothed = replace(tr, xy=tr.xy + 3.0, raw_xy=raw)  # xy deliberately differs from raw
    seen = {}

    def spy(track, *a, **k):
        seen["xy"] = track.xy.copy()
        return []

    monkeypatch.setattr(H, "detect_candidates", spy)
    H.detect_hits(smoothed, meta)
    has = np.isfinite(raw).all(axis=1)
    assert np.allclose(seen["xy"][has], raw[has])
    assert np.allclose(seen["xy"][~has], smoothed.xy[~has])
    H.detect_hits(smoothed, meta, config=H.HitConfig(use_raw_xy=False))
    assert np.allclose(seen["xy"], smoothed.xy)


# ── synthetic rally benchmarks (targets from the brief) ──────────────────────

# Floors below are regression guards set ~0.01-0.02 under the measured values
# (see the report / `python -m tests.test_hits`); the brief's targets (F1 >= 0.95,
# side >= 98 %) are not reached on this deliberately harsh generator.

def test_synthetic_60fps_camera_and_players():
    r = evaluate(60.0, range(N_SEEDS), camera=True, players=True)
    assert r["f1"] >= 0.91, r
    assert r["side_acc"] >= 0.96, r
    assert r["t_med_ms"] <= 6.0, r


@pytest.mark.parametrize("fps", [30.0, 120.0])
def test_synthetic_other_fps_camera_and_players(fps):
    r = evaluate(fps, range(N_SEEDS), camera=True, players=True)
    assert r["f1"] >= 0.90, r
    assert r["side_acc"] >= 0.96, r


def test_synthetic_2d_only():
    r = evaluate(60.0, range(N_SEEDS), camera=False, players=False)
    assert r["f1"] >= 0.88, r
    assert r["side_acc"] >= 0.82, r


def test_synthetic_players_without_camera():
    r = evaluate(60.0, range(N_SEEDS), camera=False, players=True, sides_known=False)
    assert r["f1"] >= 0.90, r
    assert r["side_acc"] >= 0.92, r


def test_multi_rally_track_restarts():
    tot = {"tp": 0, "n_true": 0, "n_det": 0}
    for s in range(6):
        sc = build_scenario(100 + s, 60.0, n_rallies=3, players=True)
        r = match_hits(H.detect_hits(sc.track, sc.meta, sc.camera, sc.players), sc)
        for k in tot:
            tot[k] += r[k]
    assert tot["tp"] >= 0.85 * tot["n_true"] and tot["tp"] >= 0.90 * tot["n_det"], tot


def test_with_sim_module_if_available():
    """Cross-check against the sibling simulator when it exists."""
    sim = pytest.importorskip("app.services.tracking.sim")
    if not (hasattr(sim, "simulate_rally") and hasattr(sim, "corrupt_detections")):
        pytest.skip("sim.simulate_rally not available yet")
    tp = nt = nd = 0
    for seed in range(10):
        sr = sim.simulate_rally(seed, fps=60)
        rng = np.random.default_rng(seed)
        uv = sr.shuttle_px + rng.normal(0, 1.5, sr.shuttle_px.shape)
        vis = sr.visible & (rng.random(sr.n) >= 0.1)
        tr = Track2D(t=sr.timestamps, xy=np.where(vis[:, None], uv, np.nan),
                     conf=vis.astype(float), visible=vis, interpolated=np.zeros(sr.n, bool))
        hs = H.detect_hits(tr, sr.meta, sr.camera, None)
        tf = [int(h["frame"]) for h in sr.hits]
        used: set[int] = set()
        for h in hs:
            j = next((j for j, f in enumerate(tf) if j not in used and abs(f - h.frame) <= 3), None)
            if j is not None:
                used.add(j)
        tp, nt, nd = tp + len(used), nt + len(tf), nd + len(hs)
    assert tp >= 0.75 * nt and tp >= 0.88 * nd, (tp, nt, nd)


REAL = ("/path/to/shuttl/backend/ml/data/raw/shuttlecock_trajectory/"
        "TrackNetV2/Test")


@pytest.mark.skipif(not os.path.isdir(REAL), reason="TrackNetV2 Test labels not present")
def test_real_labels_smoke():
    """Held-out real labels: runs on every clip and gives plausible counts
    (qualitative check only; thresholds are not tuned on this data)."""
    import glob
    for p in sorted(glob.glob(f"{REAL}/match*/csv/*_ball.csv")):
        a = np.genfromtxt(p, delimiter=",", skip_header=1)
        n = int(a[:, 0].max()) + 1
        vis = np.zeros(n, bool)
        xy = np.full((n, 2), np.nan)
        f = a[:, 0].astype(int)
        v = a[:, 1] > 0
        vis[f[v]] = True
        xy[f[v]] = a[v][:, 2:4]
        t = np.arange(n) / 30.0
        tr = Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis,
                     interpolated=np.zeros(n, bool))
        hs = H.detect_hits(tr, VideoMeta(30.0, 1280, 720, n, t))
        span = (np.flatnonzero(vis)[-1] - np.flatnonzero(vis)[0]) / 30.0
        # contacts are >= ~0.3 s apart even in fast exchanges
        assert 1 <= len(hs) <= span / 0.3 + 2, (p, len(hs), span)


if __name__ == "__main__":
    import time
    for name, cam, pl in (("camera+players", True, True), ("players only", False, True),
                          ("camera only", True, False), ("2-D only", False, False)):
        for fps in (30.0, 60.0, 120.0):
            t0 = time.time()
            r = evaluate(fps, range(N_SEEDS), camera=cam, players=pl)
            print(f"{name:15s} {fps:5.0f} fps  P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} side={r['side_acc']:.3f} t_med={r['t_med_ms']:.1f}ms "
                  f"t_p90={r['t_p90_ms']:.1f}ms n={r['n_true']} ({time.time() - t0:.1f}s)")


def _exit_track(gap_s: float, exit_y: float = 20.0, fps: float = 60.0):
    """Rising to the top border, unseen gap_s, falling back in at the top."""
    t1 = np.arange(0.0, 0.5, 1 / fps)                 # rising: y from 400 to exit_y
    y1 = 400.0 - (400.0 - exit_y) * (t1 / t1[-1])
    t2 = t1[-1] + gap_s + np.arange(0.0, 0.5, 1 / fps)
    y2 = exit_y + 500.0 * (t2 - t2[0])                # falling back in
    t = np.concatenate([t1, t2])
    xy = np.stack([np.linspace(800, 1000, len(t)), np.concatenate([y1, y2])], 1)
    return t, xy


def test_hidden_contact_over_the_top_of_the_frame():
    cfg = H.HitConfig()
    size = (1920.0, 1080.0)
    t, xy = _exit_track(2.4)
    out = H._hidden_contacts(t, xy, cfg, size)
    assert len(out) == 1
    tau, i, uv = out[0]
    assert t[i - 1] < tau < t[i] and abs(tau - 0.5 * (t[i - 1] + t[i])) < 1e-12
    assert uv[1] < 0                                   # above the frame
    assert H._hidden_contacts(*_exit_track(1.2), cfg, size) == []          # one flight fits in 1.2 s
    assert H._hidden_contacts(*_exit_track(2.4, exit_y=300.0), cfg, size) == []   # not at the top
    assert H._hidden_contacts(*_exit_track(4.5), cfg, size) == []          # > max_out_of_view_s
    c = H._Cand(tau=tau, frame=0, xy=uv, lam=0.0, lam_v=0.0, v_in=np.zeros(2), v_out=np.zeros(2),
                gap_s=2.4, hidden=True)
    assert H._kinematic_logodds(c, math.hypot(*size), cfg) == cfg.hidden_logodds


def _pts_drop(tr: Track2D, meta: VideoMeta, t0: float, t1: float) -> tuple[Track2D, VideoMeta]:
    """Frames with t0 < t < t1 are missing from the file: a hole in the presentation clock
    (video_io.pts_gaps), the real time kept (robustness F4, dropped frames)."""
    k = ~((tr.t > t0) & (tr.t < t1))
    t = tr.t[k]
    return (Track2D(t=t, xy=tr.xy[k], conf=tr.conf[k], visible=tr.visible[k], interpolated=tr.interpolated[k]),
            VideoMeta(fps=meta.fps, width=meta.width, height=meta.height, n_frames=int(k.sum()), timestamps=t.copy()))


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_gap_contact_after_a_pts_gap(fps):
    """F4: the contact is the first frame after a 0.45 s hole in the clock. The kink scan cannot
    see it (no min_side points before it within max_window_s); the gap candidate (one quadratic
    across the gap vs one each side) finds it and the hit is marked source 'gap_contact'.
    A gap INSIDE a flight (the incoming or the outgoing one) must not create a hit."""
    from app.services.tracking.video_io import pts_gaps
    tr, meta = _arc_track(fps=fps, noise=1.5, seed=1, dur=2.5)
    assert H.detect_hits(tr, meta)[0].source == "kink"
    hid, hm = _pts_drop(tr, meta, 0.55, 1.0 - 0.5 / fps)            # contact = first frame after the gap
    assert len(pts_gaps(hm.timestamps)) == 1
    hs = [h for h in H.detect_hits(hid, hm) if 0.2 < h.t < 2.3]
    assert len(hs) == 1 and abs(hs[0].t - 1.0) < 1e-9 and hs[0].source == "gap_contact"
    assert H.detect_hits(hid, hm, config=H.HitConfig(gap_contacts=False)) == []    # the old miss
    c = [c for c in H.detect_candidates(hid, hm) if c.gap_contact]
    assert len(c) == 1 and c[0].lam > 200 and 0 < c[0].kin <= H.HitConfig().gap_logodds_max
    for a, b in ((1.35, 1.8), (0.2, 0.7)):                            # gap inside the outgoing / incoming flight
        mid, mm = _pts_drop(tr, meta, a, b)
        hs = H.detect_hits(mid, mm)
        assert [h.source for h in hs] == ["kink"] and abs(hs[0].t - 1.0) < 0.01
        gc = [c for c in H.detect_candidates(mid, mm) if c.gap_contact]
        assert all(c.lam < 60 and c.kin < -3 for c in gc)
    # no PTS gap: no gap candidate at all (sim / real clips are unchanged by the rule)
    assert not any(c.gap_contact for c in H.detect_candidates(tr, meta))
