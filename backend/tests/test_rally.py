"""Tests for rally segmentation, point outcome and BWF scoring (rally.py).

Synthetic matches are generated here (standalone; no dependency on hits.py /
sim.py): flights integrate the same quadratic-drag ODE as physics.py (pure-
Python RK4 for speed), hitters alternate near/far, rallies end in / out / net,
dead time between rallies = shuttle resting on the floor, carried, sometimes
FED back by the loser (slow lob caught by the server), then held still before
the serve. The 2-D track is the camera projection with sigma = 1.5 px noise,
10 % misses, 3-10 frame occlusion gaps around 30 % of hits, 3 % outliers.
Hit events emulate a good detector (3 % misses, +-1 frame jitter, 2 % wrong
hitter, spurious low-confidence hits in dead time); 3-D segments emulate
reconstruct3d (landing sigma 7 cm/axis, 5 % fit failures, no fit across a
missed hit).

Measured with the full generator: see `test_segmentation_quality`,
`test_end_reason_given_correct_segments` (numbers printed with -s).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from app.services.tracking import court_model as cm
from app.services.tracking import rally
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, RallyV2, ShotEventV2, ShotSegment3D, Track2D, VideoMeta,
)
from app.services.tracking.physics import DRAG_LENGTH, G

# ═══ synthetic world (dense 240 Hz, frame-rate independent) ══════════════════

H = 1.0 / 240.0
NET_H = cm.NET_HEIGHT_CENTRE


def _rk4(s: list[float], h: float, L: float) -> list[float]:
    def f(q):
        k = math.sqrt(q[3] * q[3] + q[4] * q[4] + q[5] * q[5]) / L
        return [q[3], q[4], q[5], -k * q[3], -k * q[4], -G - k * q[5]]
    k1 = f(s)
    k2 = f([a + 0.5 * h * b for a, b in zip(s, k1)])
    k3 = f([a + 0.5 * h * b for a, b in zip(s, k2)])
    k4 = f([a + h * b for a, b in zip(s, k3)])
    return [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4)]


@dataclass
class Flight:
    P: np.ndarray                  # (M,3) dense states from contact
    V: np.ndarray
    landing: tuple[float, float] | None
    net_h: float | None            # z at x = 0 (first crossing)
    net_hit: bool                  # struck the net and fell back


def _fly(p0, v0, L, t_max=4.0) -> Flight:
    s = [*map(float, p0), *map(float, v0)]
    P, V = [s[:3]], [s[3:]]
    net_h, net_hit, landing = None, False, None
    t = 0.0
    while t < t_max:
        n = _rk4(s, H, L)
        if net_h is None and s[0] * n[0] < 0 and not net_hit:
            a = s[0] / (s[0] - n[0])
            zc = s[2] + a * (n[2] - s[2])
            net_h = zc
            if zc < NET_H and abs(s[1]) < cm.DOUBLES_HALF_WIDTH:
                # hits the net: drops back on the hitter's side
                net_hit = True
                sgn = 1.0 if s[0] < 0 else -1.0
                n = [-0.04 * sgn, n[1], zc, -0.3 * sgn, 0.0, -0.5]
        if n[2] <= 0.0:
            a = s[2] / (s[2] - n[2])
            landing = (s[0] + a * (n[0] - s[0]), s[1] + a * (n[1] - s[1]))
            P.append([landing[0], landing[1], 0.0]); V.append(n[3:])
            break
        s, t = n, t + H
        P.append(s[:3]); V.append(s[3:])
    return Flight(np.array(P), np.array(V), landing, net_h, net_hit)


# shot families: speed km/h, elevation deg, target depth from net (m)
SHOTS = {
    "smash": ((200, 380), (-18, -6), (2.5, 6.0)),
    "clear": ((70, 140), (25, 50), (5.0, 6.5)),
    "drop": ((40, 75), (-5, 18), (0.6, 2.8)),
    "drive": ((90, 170), (-4, 8), (3.0, 6.3)),
    "net": ((12, 28), (15, 60), (0.3, 1.8)),
    "lift": ((70, 130), (35, 60), (4.8, 6.5)),
    "push": ((50, 90), (-2, 12), (2.5, 5.0)),
}


def _next_type(rng, z: float, depth: float) -> str:
    if z > 2.1:
        return rng.choice(["smash", "clear", "drop"] if depth > 2.0 else ["smash", "drop"])
    if z > 1.1:
        return rng.choice(["drive", "push", "drop", "clear"])
    return rng.choice(["lift", "net", "lift"] if depth < 3.0 else ["lift", "lift", "drive"])


def _launch(rng, pos, side: str, kind: str, spread: float = 0.0):
    """Initial velocity towards a random target in the opponent half."""
    (s0, s1), (e0, e1), (d0, d1) = SHOTS[kind]
    sgn = 1.0 if side == "near" else -1.0
    depth = rng.uniform(d0, d1 + spread)
    ty = rng.uniform(-2.4 - spread, 2.4 + spread)
    dx, dy = sgn * depth - pos[0], ty - pos[1]
    az = math.atan2(dy, dx)
    sp, el = rng.uniform(s0, s1) / 3.6, math.radians(rng.uniform(e0, e1))
    return np.array([sp * math.cos(el) * math.cos(az), sp * math.cos(el) * math.sin(az),
                     sp * math.sin(el)])


def _margin(xy, doubles=False) -> float:
    return cm.in_out_margin(xy[0], xy[1], doubles)


def _serve_box_margin(xy, server: str, server_y: float) -> float:
    """Independent singles serve-box check (truth for the generator)."""
    x, y = xy
    xr = x if server == "near" else -x
    ty = -1.0 if server_y > 0 else 1.0
    lw = cm.LINE_WIDTH / 2
    return min(xr - cm.SHORT_SERVICE_X + lw, cm.HALF_LENGTH + lw - xr,
               cm.SINGLES_HALF_WIDTH + lw - abs(y), ty * y + lw)


FLIGHT, REST, CARRY, HOLD, HIDDEN = 0, 1, 2, 3, 4
SERVES = {"short_serve": ((25, 42), (8, 35), (2.1, 3.2)),
          "long_serve": ((80, 140), (40, 62), (5.5, 6.6))}
SHOTS.update(SERVES)


@dataclass
class World:
    seed: int
    P: np.ndarray                    # (M,3) dense 240 Hz shuttle positions (NaN = hidden)
    kind: np.ndarray                 # (M,) FLIGHT / REST / CARRY / HOLD / HIDDEN
    hits: list[dict] = field(default_factory=list)      # i, hitter, kind('rally'|'feed'|'spurious'), flight
    flights: list[dict] = field(default_factory=list)
    rallies: list[dict] = field(default_factory=list)
    L: float = DRAG_LENGTH


def _gen_world(seed: int, n_rallies: int = 10) -> World:
    rng = np.random.default_rng(seed)
    L = DRAG_LENGTH * rng.uniform(0.92, 1.08)
    Ps: list[np.ndarray] = []
    Ks: list[np.ndarray] = []
    hits: list[dict] = []
    flights: list[dict] = []
    rallies: list[dict] = []
    cur = [0]

    def add(P, k):
        P = np.asarray(P, float).reshape(-1, 3)
        Ps.append(P)
        Ks.append(np.full(len(P), k))
        cur[0] += len(P)

    def still(pos, dur, k, sway=0.0):
        m = max(2, int(round(dur / H)))
        P = np.repeat(np.asarray(pos, float)[None], m, 0)
        if sway:
            tt = np.arange(m) * H
            for ax in range(3):
                P[:, ax] += sway * np.sin(2 * np.pi * rng.uniform(0.3, 0.9) * tt + rng.uniform(0, 6))
        add(P, k)

    def carry(a, b, dur):
        m = max(2, int(round(dur / H)))
        s = np.linspace(0, 1, m)[:, None]
        P = np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * s
        P[:, 2] += 0.04 * np.sin(np.linspace(0, 5 * np.pi, m))
        add(P, CARRY)

    def serve_pos(side):
        sgn = -1.0 if side == "near" else 1.0
        return np.array([sgn * rng.uniform(2.3, 3.2), rng.choice([-1, 1]) * rng.uniform(0.3, 1.4),
                         rng.uniform(1.0, 1.15)])

    def judge(fl: Flight, hitter: str, serve: bool, sy: float) -> tuple[str, float | None]:
        if fl.net_hit:
            return "net", None
        if fl.landing is None:
            return "none", None
        if cm.side_of(fl.landing[0]) == hitter:
            return "net", None
        m = _serve_box_margin(fl.landing, hitter, sy) if serve else _margin(fl.landing)
        return ("in" if m >= 0 else "out"), m

    def shot(pos, hitter, kind, want, serve=False):
        for it in range(60):
            if it == 30 and want == "in":
                kind = "lift"                # low / awkward contact: a lift always clears

            if want == "out":
                v0 = _launch(rng, pos, hitter, kind, spread=1.2) * rng.uniform(1.0, 1.3)
            elif want == "net":
                v0 = _launch(rng, pos, hitter, kind)
                el = rng.uniform(-22, -6)
                sp, az = np.linalg.norm(v0), math.atan2(v0[1], v0[0])
                el0 = math.asin(v0[2] / sp) + math.radians(el)
                v0 = sp * rng.uniform(0.5, 1.0) * np.array(
                    [math.cos(el0) * math.cos(az), math.cos(el0) * math.sin(az), math.sin(el0)])
            else:
                v0 = _launch(rng, pos, hitter, kind)
            fl = _fly(pos, v0, L)
            o, m = judge(fl, hitter, serve, pos[1])
            if o == want and (m is None or abs(m) > 0.01):
                return fl, o, m
        return fl, o, m

    def intercept(fl: Flight, recv: str) -> int | None:
        x, z, vz = fl.P[:, 0], fl.P[:, 2], fl.V[:, 2]
        on = (x < -0.5) if recv == "near" else (x > 0.5)
        ok = on & (z > 0.25) & (np.arange(len(z)) >= int(0.3 / H))
        c = np.flatnonzero(ok & (vz < 0) & (z <= rng.uniform(0.3, 3.0)))
        if len(c):
            return int(c[0])
        c = np.flatnonzero(ok)
        return int(c[len(c) // 2]) if len(c) else None

    def aim(pos, target, kmh, el_deg):
        d = np.asarray(target, float)[:2] - np.asarray(pos, float)[:2]
        az, sp, el = math.atan2(d[1], d[0]), kmh / 3.6, math.radians(el_deg)
        return np.array([sp * math.cos(el) * math.cos(az), sp * math.cos(el) * math.sin(az),
                         sp * math.sin(el)])

    server = "near" if rng.random() < 0.5 else "far"
    pos = serve_pos(server)
    still([np.nan] * 3, rng.uniform(0.4, 1.0), HIDDEN)
    for r in range(n_rallies):
        n_shots = int(np.clip(rng.geometric(1 / 8.0), 1, 25))
        end = str(rng.choice(["in", "out", "net"], p=[0.5, 0.3, 0.2]))
        still(pos, rng.uniform(1.0, 2.5), HOLD, sway=0.004)
        hitter, serve_y, r_first = server, float(pos[1]), len(hits)
        for k in range(n_shots):
            last = k == n_shots - 1
            want = end if last else "in"
            if k == 0:
                kind = "short_serve" if rng.random() < 0.7 else "long_serve"
                (s0, s1), (e0, e1), (d0, d1) = SHOTS[kind]
                sgn = 1.0 if hitter == "near" else -1.0
                fl = o = m = None
                for _ in range(80):
                    tgt = (sgn * rng.uniform(d0, d1 + (1.5 if want == "out" else 0.0)),
                           -math.copysign(1.0, serve_y) * rng.uniform(0.2, 2.4 + (0.8 if want == "out" else 0)))
                    el = rng.uniform(e0, e1) - (rng.uniform(8, 25) if want == "net" else 0.0)
                    fl = _fly(pos, aim(pos, tgt, rng.uniform(s0, s1), el), L)
                    o, m = judge(fl, hitter, True, serve_y)
                    if o == want and (m is None or abs(m) > 0.01):
                        break
            else:
                kind = _next_type(rng, pos[2], abs(pos[0]))
                fl, o, m = shot(pos, hitter, kind, want)
            if o == "none":
                o = "out"
            j = None if last or o != "in" else intercept(fl, _opp(hitter))
            last = last or j is None
            hits.append(dict(i=cur[0], hitter=hitter, kind="rally", rally=r, flight=len(flights)))
            Pf = fl.P if last else fl.P[: j + 1]
            flights.append(dict(i0=cur[0], P=Pf, V=fl.V[: len(Pf)], full=fl, hitter=hitter,
                                end=("net" if o == "net" else "floor") if last else "hit",
                                outcome=o, margin=m, kind=kind))
            add(Pf[:-1], FLIGHT)
            pos = Pf[-1].copy()
            if last:
                break
            hitter = _opp(hitter)
        winner = hitter if o == "in" else _opp(hitter)
        rallies.append(dict(i0=hits[r_first]["i"], i1=cur[0], end=o, winner=winner,
                            server=server, margin=m, landing=(float(pos[0]), float(pos[1])),
                            n_shots=len(hits) - r_first, last_hitter=hitter))
        # ── dead time: rest, then carry or feed back to the next server ──
        land = np.array([pos[0], pos[1], 0.0])
        still(land, rng.uniform(0.8, 2.0), REST)
        loser, server = _opp(winner), winner
        nxt = serve_pos(server)
        spurious_range = None
        if cm.side_of(land[0]) == loser and rng.random() < 0.5:
            hand = np.array([land[0] * 0.7, land[1] * 0.5, 1.0])
            i_c = cur[0]
            carry(land, hand, rng.uniform(0.6, 1.8))
            spurious_range = (i_c, cur[0])
            ff = None
            for _ in range(30):
                ff = _fly(hand, aim(hand, nxt, rng.uniform(18, 38), rng.uniform(35, 62)), L)
                if not ff.net_hit and ff.net_h is not None and ff.net_h > NET_H + 0.2:
                    break
            x, z, vz = ff.P[:, 0], ff.P[:, 2], ff.V[:, 2]
            on = (x < -0.3) if server == "near" else (x > 0.3)
            c = np.flatnonzero(on & (vz < 0) & (z < rng.uniform(0.9, 1.5)) & (z > 0.3))
            hits.append(dict(i=cur[0], hitter=loser, kind="feed", rally=None, flight=len(flights)))
            Pf = ff.P[: c[0] + 1] if len(c) else ff.P
            flights.append(dict(i0=cur[0], P=Pf, V=ff.V[: len(Pf)], full=ff, hitter=loser,
                                end="catch" if len(c) else "floor", outcome="feed", margin=None,
                                kind="feed"))
            add(Pf[:-1], FLIGHT)
            carry(Pf[-1], nxt, rng.uniform(0.8, 2.5))
        else:
            i_c = cur[0]
            carry(land, nxt, rng.uniform(1.5, 4.0))
            spurious_range = (i_c, cur[0])
        if rng.random() < 0.15:
            a, b = spurious_range
            hits.append(dict(i=int(rng.integers(a + 10, max(a + 11, b - 10))), hitter=str(rng.choice(SIDES)),
                             kind="spurious", rally=None, flight=None))
        if rng.random() < 0.15:
            still([np.nan] * 3, rng.uniform(0.5, 2.0), HIDDEN)
        pos = nxt
    still(pos, 1.0, CARRY)
    hits.sort(key=lambda h: h["i"])
    return World(seed, np.vstack(Ps), np.concatenate(Ks), hits, flights, rallies, L)


SIDES = ("near", "far")


def _opp(s: str) -> str:
    return "far" if s == "near" else "near"


def _camera(rng, style: str = "behind") -> CameraModel:
    """Random phone camera behind the near baseline that sees the whole court
    (users frame the full court) — resampled until all 4 corners are inside."""
    corners = np.array([[sx * cm.HALF_LENGTH, sy * cm.DOUBLES_HALF_WIDTH, 0.0]
                        for sx in (-1, 1) for sy in (-1, 1)])
    for _ in range(200):
        cam = _camera_try(rng, style)
        uv = cam.project(corners)
        W, Hh = cam.image_size
        if np.all(np.isfinite(uv)) and np.all((uv >= 0.03 * np.array([W, Hh])) &
                                              (uv <= 0.97 * np.array([W, Hh]))):
            return cam
    return cam


def _camera_try(rng, style: str) -> CameraModel:
    if style == "elevated":
        size = (1280, 720)
        eye = [-cm.HALF_LENGTH - rng.uniform(4, 7), rng.uniform(-0.5, 0.5), rng.uniform(4.5, 6.5)]
        return CameraModel.look_at(eye, [rng.uniform(-1, 1), 0, 0], rng.uniform(0.7, 1.0) * size[0], size)
    size = (1920, 1080)
    eye = [-cm.HALF_LENGTH - rng.uniform(2.0, 6.0), rng.uniform(-0.6, 0.6), rng.uniform(1.4, 2.6)]
    return CameraModel.look_at(eye, [rng.uniform(-2.5, 0.0), 0.0, rng.uniform(0.0, 1.0)],
                               rng.uniform(0.5, 0.8) * size[0], size, roll_deg=rng.uniform(-2, 2))


@dataclass
class Clip:
    world: World
    fps: float
    camera: CameraModel
    meta: VideoMeta
    track: Track2D
    uv_true: np.ndarray
    t: np.ndarray
    hit_frames: list[int]           # frame of every world hit (same order as world.hits)
    truth: list[tuple[int, int]]    # (start_frame, end_frame) per true rally


def _render(w: World, fps: float, seed: int, style: str = "behind") -> Clip:
    rng = np.random.default_rng(seed * 7919 + int(fps))
    cam = _camera(rng, style)
    T = len(w.P) * H
    n = int(T * fps)
    t = np.arange(n) / fps
    if rng.random() < 0.3:              # variable frame rate
        t = np.cumsum(np.r_[0.0, (1 / fps) * rng.uniform(0.85, 1.15, n - 1)])
        t = t[t < (len(w.P) - 1) * H]
        n = len(t)
    s = t / H
    i0 = np.clip(np.floor(s).astype(int), 0, len(w.P) - 2)
    a = (s - i0)[:, None]
    P = w.P[i0] * (1 - a) + w.P[i0 + 1] * a
    kind = w.kind[np.clip(np.round(s).astype(int), 0, len(w.P) - 1)]
    uv = np.full((n, 2), np.nan)
    ok = np.isfinite(P).all(1)
    uv[ok] = cam.project(P[ok])
    Wd, Ht = cam.image_size
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < Ht)
    uv[~inside] = np.nan
    uv_true = uv.copy()
    xy = uv + rng.normal(0, 1.5, uv.shape)
    vis = inside & (rng.random(n) > 0.10)                             # 10 % misses
    hit_frames = [int(np.clip(np.searchsorted(t, h["i"] * H), 0, n - 1)) for h in w.hits]
    for h, f in zip(w.hits, hit_frames):                              # occlusion near hits
        if h["kind"] != "spurious" and rng.random() < 0.3:
            g = int(rng.integers(3, 11))
            s0 = f - int(rng.integers(0, g))
            vis[max(s0, 0): s0 + g] = False
    rest = kind == REST
    vis &= ~(rest & (rng.random() < 0.3))                            # linker may drop static shuttles
    out_idx = np.flatnonzero(vis & (rng.random(n) < 0.03))            # 3 % outliers
    xy[out_idx] += rng.normal(0, 60, (len(out_idx), 2))
    xy[~vis] = np.nan
    track = Track2D(t=t, xy=xy, conf=np.where(vis, 0.9, 0.0), visible=vis,
                    interpolated=np.zeros(n, bool))
    meta = VideoMeta(fps=fps, width=Wd, height=Ht, n_frames=n, timestamps=t)
    truth = [(int(np.searchsorted(t, r["i0"] * H)), int(min(np.searchsorted(t, r["i1"] * H), n - 1)))
             for r in w.rallies]
    return Clip(w, fps, cam, meta, track, uv_true, t, hit_frames, truth)


def _emulate_hits(c: Clip, rng, miss=0.03, flip=0.02) -> tuple[list[HitEvent], list[int]]:
    """Detector-like hits; returns (hits, index of the world hit or -1)."""
    out, src = [], []
    jit = max(1, int(round(c.fps / 60)))
    for k, (h, f) in enumerate(zip(c.world.hits, c.hit_frames)):
        if h["kind"] != "spurious" and rng.random() < miss:
            continue
        f2 = int(np.clip(f + rng.integers(-jit, jit + 1), 0, c.meta.n_frames - 1))
        side = h["hitter"] if rng.random() > flip else _opp(h["hitter"])
        conf = rng.uniform(0.15, 0.45) if h["kind"] == "spurious" else rng.uniform(0.55, 0.98)
        uv = c.uv_true[f2]
        out.append(HitEvent(frame=f2, t=float(c.t[f2]),
                            xy_px=(float(uv[0]), float(uv[1])) if np.isfinite(uv).all() else (0.0, 0.0),
                            hitter=side, confidence=float(conf)))
        src.append(k)
    return out, src


def _state_at(fl: dict, t_rel: float) -> tuple[np.ndarray, np.ndarray]:
    """True (p, v) of a world flight `t_rel` s after its contact (clamped)."""
    P, V = fl["P"], fl["V"]
    s = min(max(t_rel / H, 0.0), len(P) - 1.001)
    i = int(s)
    a = s - i
    return P[i] * (1 - a) + P[i + 1] * a, V[i] * (1 - a) + V[i + 1] * a


def _emulate_segments(c: Clip, hits: list[HitEvent], src: list[int], rng,
                      fail: float = 0.05, land_sigma: float = 0.07) -> list[ShotSegment3D]:
    """reconstruct3d-like flights, one per detected hit, with its semantics:
    a flight is chained to the next detected hit if that comes within 2.8 s
    (then no landing is reported, even if the shuttle actually fell and the
    next "hit" is a feed / spurious one); a chain across a missed contact is a
    failed fit; unchained flights report the floor contact (sigma 7 cm/axis),
    a net crossing below the tape, or nothing (caught)."""
    w, t, n = c.world, c.t, c.meta.n_frames
    order = np.argsort([h.frame for h in hits])
    out: list[ShotSegment3D] = []
    for pos, k in enumerate(order):
        h, wi = hits[k], src[k]
        wh = w.hits[wi]
        if wh["flight"] is None or rng.random() < fail:
            continue
        fl = w.flights[wh["flight"]]
        nxt = hits[order[pos + 1]] if pos + 1 < len(order) else None
        chained = nxt is not None and nxt.t - h.t <= 2.8
        if fl["end"] == "hit":
            true_next = chained and src[order[pos + 1]] == wi + 1
            if not true_next:
                continue                              # fit across a missed contact
        t_c = fl["i0"] * H
        dur = (len(fl["P"]) - 1) * H
        t_end = min(nxt.t, t_c + dur) if chained else t_c + dur
        f0 = min(int(np.searchsorted(t, h.t + 1e-6)), n - 1)
        f1 = max(min(int(np.searchsorted(t, t_end)) - 1, n - 1), f0)
        frames = np.arange(f0, f1 + 1)
        pos_true = np.array([_state_at(fl, t[f] - t_c)[0] for f in frames])
        p0, v0 = _state_at(fl, h.t - t_c)
        p0 = p0 + rng.normal(0, 0.03, 3)
        v0 = v0 * (1 + rng.normal(0, 0.015)) + rng.normal(0, 0.15, 3)
        full = fl["full"]
        landing = landing_t = None
        end_reason = "hit" if chained else "unknown"
        if not chained and fl["end"] == "floor" and full.landing is not None:
            landing = (full.landing[0] + rng.normal(0, land_sigma),
                       full.landing[1] + rng.normal(0, land_sigma))
            landing_t = t_c + dur + rng.normal(0, 0.01)
            end_reason = "floor"
        elif not chained and fl["end"] == "net":
            end_reason = "net"
        crosses = fl["end"] == "net" or bool(np.any(fl["P"][:, 0] * fl["P"][0, 0] < 0))
        net_h = (float(full.net_h + rng.normal(0, 0.03))
                 if crosses and full.net_h is not None else None)
        seg = ShotSegment3D(start_frame=int(f0), end_frame=int(f1), t0=float(h.t), p0=p0, v0=v0,
                            drag_length=float(w.L * (1 + rng.normal(0, 0.03))),
                            rmse_px=float(rng.uniform(0.8, 2.0)), frames=frames,
                            positions=pos_true + rng.normal(0, 0.03, pos_true.shape),
                            landing_xy=landing, landing_t=landing_t, net_crossing_height=net_h,
                            confidence=float(rng.uniform(0.5, 0.95)))
        seg.hit_frame = int(h.frame)
        seg.end_reason = end_reason
        seg.landing_sigma_m = land_sigma * 1.2 if landing is not None else None
        if net_h is not None:                         # reconstruct3d extra: y at the crossing
            xs = full.P[:, 0]
            k = np.flatnonzero(xs[:-1] * xs[1:] <= 0)
            seg.net_crossing_y = float(full.P[k[0], 1]) if len(k) else None
        out.append(seg)
    return out


def _players_doubles(c: Clip, doubles: bool) -> list:
    """Minimal PlayerObservations (side only) every 10th frame."""
    from app.services.tracking.data_types import PlayerObservation
    per = 2 if doubles else 1
    obs = []
    for f in range(0, c.meta.n_frames, 10):
        for side in SIDES:
            for k in range(per):
                obs.append(PlayerObservation(frame=f, track_id=(0 if side == "near" else 2) + k,
                                             bbox=np.array([0, 0, 10, 10.0]), score=0.9, side=side))
    return obs


# ═══ evaluation ══════════════════════════════════════════════════════════════

def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def _match(pred: list[RallyV2], truth: list[tuple[int, int]], fps: float,
           thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching -> (pred, truth) pairs. A pair matches when
    the frame intervals have IoU >= thr, or when the serve is found (start
    within 0.5 s) and the end within 2.5 s (a final flight that leaves the
    view makes the end of short rallies uncertain without a 3-D fit)."""
    def score(p: RallyV2, tr: tuple[int, int]) -> float:
        v = _iou((p.start_frame, p.end_frame), tr)
        if abs(p.start_frame - tr[0]) <= 0.5 * fps and abs(p.end_frame - tr[1]) <= 2.5 * fps:
            v = max(v, thr)
        return v
    pairs = sorted(((score(p, tr), i, j)
                    for i, p in enumerate(pred) for j, tr in enumerate(truth)), reverse=True)
    used_p, used_t, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); out.append((i, j))
    return out


@dataclass
class Stats:
    tp: int = 0
    n_pred: int = 0
    n_true: int = 0
    end_ok: int = 0
    end_n: int = 0
    win_ok: int = 0
    win_n: int = 0
    start_err: list = field(default_factory=list)

    def add(self, pred: list[RallyV2], clip: Clip) -> None:
        m = _match(pred, clip.truth, clip.fps)
        self.tp += len(m); self.n_pred += len(pred); self.n_true += len(clip.truth)
        for i, j in m:
            tr, p = clip.world.rallies[j], pred[i]
            self.start_err.append(abs(p.start_frame - clip.truth[j][0]) / clip.fps)
            self.end_n += 1
            self.end_ok += p.end_reason == tr["end"]
            self.win_n += 1
            self.win_ok += p.winner == tr["winner"]

    @property
    def f1(self) -> float:
        return 2 * self.tp / max(self.n_pred + self.n_true, 1)

    def __str__(self) -> str:
        return (f"F1={self.f1:.3f} (tp={self.tp} pred={self.n_pred} true={self.n_true}) "
                f"end_acc={self.end_ok / max(self.end_n, 1):.3f} "
                f"winner_acc={self.win_ok / max(self.win_n, 1):.3f} "
                f"start_err_med={np.median(self.start_err) if self.start_err else float('nan'):.3f}s")


def _run(seed: int, fps: float, with_3d: bool = True, n_rallies: int = 10,
         style: str = "behind") -> tuple[list[RallyV2], Clip]:
    w = _world(seed, n_rallies)
    c = _render(w, fps, seed, style)
    rng = np.random.default_rng(seed + 12345)
    hits, src = _emulate_hits(c, rng)
    segs = _emulate_segments(c, hits, src, rng) if with_3d else []
    pred = rally.segment_rallies(c.track, hits, segs, c.meta, c.camera if with_3d else None)
    return pred, c


@functools.lru_cache(maxsize=64)
def _world(seed: int, n_rallies: int = 10) -> World:
    """Worlds are frame-rate independent: generate once, render at every fps."""
    return _gen_world(seed, n_rallies)


# ═══ BWF scoring ═════════════════════════════════════════════════════════════

def _R(winner: str, i: int, first_hitter: str | None = None) -> RallyV2:
    shots = []
    if first_hitter is not None:
        shots = [ShotEventV2(hit=HitEvent(frame=i * 100, t=i * 100 / 30, xy_px=(0.0, 0.0),
                                          hitter=first_hitter, confidence=0.9), segment=None)]
    return RallyV2(start_frame=i * 100, end_frame=i * 100 + 50, shots=shots,
                   winner=winner, confidence=0.9)


def _play(seq: list[str], **kw) -> list[dict]:
    return rally.score_match([_R(w, i) for i, w in enumerate(seq)], **kw)


def test_score_straight_game_then_ends_change():
    sc = _play(["near"] * 21 + ["far"])
    assert [e["game_over"] for e in sc[:21]] == [False] * 20 + [True]
    assert sc[20]["score"] == {"near": 21, "far": 0}
    assert sc[20]["games"] == {"near": 1, "far": 0}
    assert sc[20]["ends_changed"] and not sc[20]["match_over"]
    # A (started near) won game 1: serves first in game 2, now from the far end
    assert sc[20]["next_server"] == "far"
    assert sc[21]["game"] == 2 and sc[21]["server"] == "far" and sc[21]["service_court"] == "right"
    assert sc[21]["score_by_player"] == {"A": 1, "B": 0}
    assert sc[21]["score"] == {"near": 0, "far": 1}


def test_score_deuce_needs_two_point_lead():
    seq = ["near", "far"] * 20 + ["near", "far", "near", "far", "near", "near"]
    sc = _play(seq)
    assert sc[39]["score"] == {"near": 20, "far": 20} and not sc[39]["game_over"]
    assert [e["game_over"] for e in sc[40:]] == [False] * 5 + [True]
    assert sc[-1]["score"] == {"near": 24, "far": 22}
    assert sc[-1]["game_winner"] == "near"
    # 21-19 ends the game immediately
    sc = _play(["near", "far"] * 19 + ["near", "near"])
    assert sc[-1]["game_over"] and sc[-1]["score"] == {"near": 21, "far": 19}


def test_score_cap_at_30():
    seq = ["near", "far"] * 29                    # 29-29 (no 2-point lead ever)
    sc = _play(seq + ["far"])
    assert not any(e["game_over"] for e in sc[:-1])
    assert sc[57]["score"] == {"near": 29, "far": 29}
    assert sc[-1]["game_over"] and sc[-1]["score"] == {"near": 29, "far": 30}
    assert sc[-1]["game_winner"] == "far"
    # 30-28 also ends it (2-point lead reached at 30)
    sc = _play(["near", "far"] * 28 + ["near", "near"])
    assert sc[-1]["game_over"] and sc[-1]["score"] == {"near": 30, "far": 28}


def test_service_court_and_server_follow_rally_winner():
    seq = ["near", "far", "far", "near", "near", "near", "far"]
    sc = _play(seq)
    exp, pts, server = [], {"near": 0, "far": 0}, "near"
    for w in seq:                                 # BWF: even own score -> right court
        exp.append((server, "right" if pts[server] % 2 == 0 else "left"))
        pts[w] += 1
        server = w
    assert [(e["server"], e["service_court"]) for e in sc] == exp
    assert exp[:4] == [("near", "right"), ("near", "left"), ("far", "left"), ("far", "right")]
    assert all(e["next_server"] == e["winner"] for e in sc)
    sc = rally.score_match([_R("far", 0)], first_server="far")
    assert sc[0]["server"] == "far" and sc[0]["service_court"] == "right"
    assert sc[0]["score"] == {"near": 0, "far": 1}


def test_score_game_three_mid_game_end_change_and_match_over():
    seq = ["near"] * 21                           # game 1: A (near) 21-0; ends change
    seq += ["near"] * 21                          # game 2: B (now near) 21-0; ends change
    sc = _play(seq)
    assert sc[41]["games_by_player"] == {"A": 1, "B": 1} and sc[41]["ends_changed"]
    assert sc[41]["next_server"] == "far"         # B won game 2 at near, serves game 3 from far
    # game 3: A back at the near end; A leads 11-0 -> ends change mid-game
    seq3 = seq + ["far"] + ["near"] * 11
    sc = _play(seq3)
    g3 = sc[42:]
    assert all(e["game"] == 3 for e in g3)
    assert g3[0]["score_by_player"] == {"A": 0, "B": 1}
    assert [e["ends_changed"] for e in g3] == [False] * 11 + [True]
    assert g3[-1]["score_by_player"] == {"A": 11, "B": 1}
    # after the switch A plays from the far end
    sc = _play(seq3 + ["far"] * 10)
    assert sc[-1]["score_by_player"] == {"A": 21, "B": 1}
    assert sc[-1]["match_over"] and sc[-1]["games_by_player"] == {"A": 2, "B": 1}
    assert not any(e["ends_changed"] for e in sc[54:])


def test_score_decider_switch_only_once():
    seq = ["near"] * 21 + ["near"] * 21           # 1-1 in games
    seq += ["near"] * 11 + ["near"] * 11          # A reaches 11 (switch), then B reaches 11
    sc = _play(seq)
    assert sum(e["ends_changed"] for e in sc[42:]) == 1
    assert sc[-1]["score_by_player"] == {"A": 11, "B": 11}


def test_unknown_winner_inferred_from_next_serve_or_left_uncertain():
    rs = [_R("near", 0, "near"), _R("unknown", 1, "near"), _R("near", 2, "far")]
    sc = rally.score_match(rs)
    assert sc[1]["winner"] == "far" and "winner_inferred_from_next_serve" in sc[1]["reasons"]
    assert sc[1]["uncertain"] and sc[1]["score"] == {"near": 1, "far": 1}
    assert "server_mismatch" not in sc[2]["reasons"]
    # a rally served by the side that should not serve is flagged
    sc = rally.score_match([_R("near", 0, "near"), _R("near", 1, "far")])
    assert "server_mismatch" in sc[1]["reasons"] and sc[1]["uncertain"]
    rs = [_R("near", 0, "near"), _R("unknown", 1, "near")]
    sc = rally.score_match(rs)
    assert sc[1]["winner"] == "unknown" and "winner_unknown" in sc[1]["reasons"]
    assert sc[1]["score"] == {"near": 1, "far": 0}
    # 20-0 game point: either outcome makes the player at the far end serve next
    # (near wins -> game over, A serves game 2 from the far end), so ambiguous
    st = rally._Score()
    for _ in range(20):
        st.point("near")
    assert rally._implied_winner(st, "far") == "unknown"
    assert rally._implied_winner(st, "near") == "unknown"


def test_service_court_margin_and_doubles_detection():
    # near server on the right (y < 0 is right as seen from behind near baseline)
    m_in = rally.service_court_margin(3.0, 1.0, "near", -0.8, doubles=False)
    m_wrong = rally.service_court_margin(3.0, -1.0, "near", -0.8, doubles=False)
    m_short = rally.service_court_margin(1.5, 1.0, "near", -0.8, doubles=False)
    assert m_in > 0 > m_wrong and m_short < 0
    # doubles long service line is 0.76 m inside the baseline
    assert rally.service_court_margin(6.3, 1.0, "near", -0.8, doubles=True) < 0
    assert rally.service_court_margin(6.3, 1.0, "near", -0.8, doubles=False) > 0
    assert rally.service_court_margin(-3.0, -1.0, "far", 0.8, doubles=False) > 0
    from app.services.tracking.data_types import PlayerObservation
    obs = [PlayerObservation(frame=f, track_id=k, bbox=np.zeros(4), score=0.9, side=s)
           for f in range(10) for k, s in enumerate(["near", "near", "far", "far"])]
    assert rally.detect_doubles(obs) is True
    assert rally.detect_doubles(obs[::2]) is False
    assert rally.detect_doubles(None) is None


# ═══ segmentation / outcome on synthetic matches ═════════════════════════════

SEEDS = list(range(20))


@pytest.mark.parametrize("fps", [30.0, 60.0, 120.0])
def test_segmentation_quality(fps):
    st = Stats()
    for s in SEEDS:
        pred, c = _run(s, fps)
        st.add(pred, c)
        for p in pred:
            assert p.start_frame <= p.end_frame
            assert p.winner in ("near", "far", "unknown")
            assert p.end_reason in ("in", "out", "net", "unknown")
            assert 0.0 <= p.confidence <= 1.0
            assert all(a.hit.frame < b.hit.frame for a, b in zip(p.shots, p.shots[1:]))
    print(f"\n[rally] fps={fps:g} 3-D, emulated detector: {st}")
    assert st.f1 >= SEG_F1_MIN
    assert st.win_ok / st.win_n >= 0.90


SEG_F1_MIN = 0.975


def test_no_camera_no_segments_path():
    st = Stats()
    for s in SEEDS[:10]:
        pred, c = _run(s, 60.0, with_3d=False)
        st.add(pred, c)
        assert all(p.end_reason == "unknown" and p.landing_xy is None and p.in_out_margin_m is None
                   for p in pred)
        assert all(s_.segment is None for p in pred for s_ in p.shots)
    print(f"\n[rally] fps=60 2-D only (segments=[], camera=None): {st}")
    assert st.f1 >= 0.95


def test_end_reason_given_correct_segments():
    """Exact hits (all contacts, right hitter) + near-exact 3-D flights."""
    st = Stats()
    for s in SEEDS:
        w = _world(s, 10)
        c = _render(w, 60.0, s)
        rng = np.random.default_rng(s + 999)
        hits, src = [], []
        for k, (h, f) in enumerate(zip(w.hits, c.hit_frames)):
            if h["kind"] == "spurious":
                continue
            uv = c.uv_true[f]
            hits.append(HitEvent(frame=f, t=float(c.t[f]),
                                 xy_px=(float(uv[0]), float(uv[1])) if np.isfinite(uv).all() else (0.0, 0.0),
                                 hitter=h["hitter"], confidence=0.9))
            src.append(k)
        segs = _emulate_segments(c, hits, src, rng, fail=0.0, land_sigma=0.02)
        pred = rally.segment_rallies(c.track, hits, segs, c.meta, c.camera)
        st.add(pred, c)
    print(f"\n[rally] correct hits + segments: {st}")
    assert st.f1 >= 0.98
    assert st.end_ok / st.end_n >= 0.95
    assert st.win_ok / st.win_n >= 0.95


def test_feeds_are_not_rallies():
    n_feed = n_fp = 0
    for s in SEEDS:
        pred, c = _run(s, 60.0)
        feeds = [f for h, f in zip(c.world.hits, c.hit_frames) if h["kind"] == "feed"]
        n_feed += len(feeds)
        n_fp += sum(any(p.start_frame - 3 <= f <= p.start_frame + 3 for p in pred) for f in feeds)
    print(f"\n[rally] feeds kept as rallies: {n_fp}/{n_feed}")
    assert n_feed > 20 and n_fp / n_feed <= 0.10


def test_shots_carry_their_segments():
    pred, c = _run(3, 60.0)
    n_with = 0
    for p in pred:
        for s in p.shots:
            if s.segment is not None:
                n_with += 1
                assert s.hit.frame - 3 <= s.segment.start_frame <= s.hit.frame + 30
                assert s.hitter_pos_world is not None and s.hit_height_m is not None
        assert p.shots and (p.shots[0].shot_type == "serve") != getattr(p, "serve_missed", False)
    assert n_with >= 0.8 * sum(len(p.shots) for p in pred)


def test_degenerate_inputs():
    t = np.arange(300) / 30.0
    xy = np.full((300, 2), np.nan)
    empty = Track2D(t=t[:0], xy=xy[:0], conf=t[:0], visible=np.zeros(0, bool),
                    interpolated=np.zeros(0, bool))
    meta = VideoMeta(fps=30, width=1280, height=720, n_frames=300, timestamps=t)
    assert rally.segment_rallies(empty, [], [], meta, None) == []
    nan_track = Track2D(t=t, xy=xy, conf=np.zeros(300), visible=np.zeros(300, bool),
                        interpolated=np.zeros(300, bool))
    assert rally.segment_rallies(nan_track, [], [], meta, None) == []
    # hits outside the track are ignored; one far-out hit without motion -> nothing
    assert rally.segment_rallies(nan_track, [HitEvent(frame=900, t=30.0, xy_px=(0, 0))], [],
                                 meta, None) == []
    assert rally.score_match([]) == []


def test_hitless_activity_fallback():
    pred, c = _run(2, 60.0, with_3d=False)
    pred0 = rally.segment_rallies(c.track, [], [], c.meta, None)
    assert pred0 and all(not p.shots and p.winner == "unknown" for p in pred0)
    # every true rally with an exchange (>= 3 shots, i.e. >= ~1 s of flight)
    # overlaps some moving stretch
    long_ = [tr for tr, r in zip(c.truth, c.world.rallies) if r["n_shots"] >= 3]
    assert long_ and all(any(_iou((p.start_frame, p.end_frame), tr) > 0 for p in pred0)
                         for tr in long_)


# ═══ real clips (TrackNetV2 Test split, labelled tracks) ═════════════════════

DATA = Path(__file__).resolve().parents[1] / "ml/data/raw/shuttlecock_trajectory"


def _real_clips() -> list[tuple[str, Path]]:
    out = []
    for p in sorted((DATA / "TrackNetV2/Test").glob("match*/csv/*_ball.csv")):
        corr = DATA / "corrected" / p.parts[-3] / "corrected_csv" / p.name
        out.append((f"{p.parts[-3]}/{p.stem}", corr if corr.exists() else p))
    return out


def _label_track(path: Path, fps: float = 30.0) -> tuple[Track2D, VideoMeta]:
    a = np.genfromtxt(path, delimiter=",", names=True)
    n = int(a["Frame"].max()) + 1
    xy = np.full((n, 2), np.nan)
    vis = np.zeros(n, bool)
    f = a["Frame"].astype(int)
    v = a["Visibility"] > 0
    xy[f[v]] = np.column_stack([a["X"][v], a["Y"][v]])
    vis[f[v]] = True
    t = np.arange(n) / fps
    return (Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis, interpolated=np.zeros(n, bool)),
            VideoMeta(fps=fps, width=1280, height=720, n_frames=n, timestamps=t))


@pytest.mark.skipif(not (DATA / "TrackNetV2/Test").exists(), reason="real Test split not present")
def test_real_clips_are_one_rally_each():
    clips = _real_clips()
    assert len(clips) >= 20
    try:
        from app.services.tracking import hits as hits_mod
        detect = hits_mod.detect_hits
    except Exception:                              # sibling module broken / missing
        detect = None
    n_one_act = n_one_hits = n_hits_ok = 0
    lines = []
    for name, path in clips:
        track, meta = _label_track(path)
        act = rally.segment_rallies(track, [], [], meta, None)
        n_one_act += len(act) == 1
        res = f"{name}: activity={len(act)}"
        if detect is not None:
            try:
                hs = detect(track, meta)
            except Exception as e:                  # pragma: no cover - sibling failure
                hs = None
                res += f" hits: error {type(e).__name__}"
            if hs is not None:
                n_hits_ok += 1
                rs = rally.segment_rallies(track, hs, [], meta, None)
                n_one_hits += len(rs) == 1
                res += f" hits={len(hs)} rallies={len(rs)} shots={[len(r.shots) for r in rs]}"
        lines.append(res)
    print("\n" + "\n".join(lines))
    print(f"[rally] real clips: activity path one-rally {n_one_act}/{len(clips)}; "
          f"with hits.detect_hits one-rally {n_one_hits}/{n_hits_ok}")
    assert n_one_act >= 0.9 * len(clips)
    if n_hits_ok:
        assert n_one_hits >= 0.85 * n_hits_ok


# ═══ cross-check with the sibling simulator (sim.py), if available ═══════════

def _sim():
    try:
        from app.services.tracking import sim
        return sim if hasattr(sim, "simulate_match") and hasattr(sim, "truth_segments") else None
    except Exception:
        return None


@pytest.mark.skipif(_sim() is None, reason="sim.simulate_match not available")
def test_validate_with_sim_simulate_match():
    """Independent generator (sim.py: solved shots, players, 3-8 s dead time).
    Clean: truth hits + truth segments on a noisy track. Corrupted on top:
    5 % hits dropped, 2 % hitters flipped, 5 % segments lost, 2 extra weak
    contacts per match in dead time."""
    sim = _sim()
    acc = {k: [0, 0, 0, 0, 0, 0] for k in ("clean", "corrupt")}   # tp pred true end win matched
    for seed in range(6):
        rng = np.random.default_rng(seed)
        cam = sim.random_camera(rng, (1920, 1080), style="behind" if seed % 2 == 0 else "broadcast")
        sr = sim.simulate_match(seed, 6, camera=cam, fps=60)
        px, mask = sr.image_truth(occluded=False, resting=True)
        vis = mask & (rng.random(sr.n) > 0.1)
        xy = np.where(vis[:, None], px + rng.normal(0, 1.5, px.shape), np.nan)
        track = Track2D(t=sr.timestamps.copy(), xy=xy, conf=vis.astype(float), visible=vis,
                        interpolated=np.zeros(sr.n, bool))
        hits, segs = sim.truth_hits(sr), sim.truth_segments(sr)
        truth = [(r.start_frame, r.end_frame) for r in sr.rallies]
        keep = rng.random(len(hits)) > 0.05
        c_hits = [replace(h, hitter=_opp(h.hitter) if rng.random() < 0.02 else h.hitter,
                          confidence=float(rng.uniform(0.55, 0.98)))
                  for h, k in zip(hits, keep) if k]
        dead = [(truth[i][1] + 60, truth[i + 1][0] - 60) for i in range(len(truth) - 1)]
        for a, b in [d for d in dead if d[1] > d[0]][:2]:
            f = int(rng.integers(a, b))
            c_hits.append(HitEvent(frame=f, t=float(sr.timestamps[f]), xy_px=(0.0, 0.0),
                                   hitter=str(rng.choice(SIDES)), confidence=0.3))
        c_segs = [s for s, k in zip(segs, keep) if k and rng.random() > 0.05]
        for name, hs, ss in (("clean", hits, segs), ("corrupt", c_hits, c_segs)):
            pred = rally.segment_rallies(track, hs, ss, sr.meta, cam)
            m = _match(pred, truth, sr.fps)
            a = acc[name]
            a[0] += len(m); a[1] += len(pred); a[2] += len(truth); a[5] += len(m)
            for i, j in m:
                a[3] += pred[i].end_reason == sr.rallies[j].end_reason
                a[4] += pred[i].winner == sr.rallies[j].winner
    res = {}
    for name, (tp, npred, ntrue, eo, wo, nm) in acc.items():
        res[name] = (2 * tp / max(npred + ntrue, 1), eo / max(nm, 1), wo / max(nm, 1))
        print(f"\n[rally] sim.simulate_match {name}: F1={res[name][0]:.3f} ({tp}/{npred}/{ntrue}) "
              f"end_acc={res[name][1]:.3f} winner_acc={res[name][2]:.3f}")
    assert res["clean"][0] >= 0.97 and res["clean"][1] >= 0.95 and res["clean"][2] >= 0.95
    # corrupted: a missed final contact gives end_reason 'unknown' by design
    # (winner then comes from the next serve), so only F1 / winner are gated
    assert res["corrupt"][0] >= 0.95 and res["corrupt"][2] >= 0.9


# ═══ 2026-09-28 accuracy handoff: same-flight merge, out-of-view gaps, serve gate ══

def _flight_seg(t0, p0, v0, start, end, **kw):
    return ShotSegment3D(start_frame=start, end_frame=end, t0=t0, p0=np.asarray(p0, float),
                         v0=np.asarray(v0, float), drag_length=DRAG_LENGTH, rmse_px=1.0,
                         frames=np.arange(start, end + 1), positions=np.zeros((end - start + 1, 3)),
                         confidence=0.6, **kw)


def test_merge_same_flight_drops_contact_without_impulse():
    t = np.arange(400) / 60.0
    a = _flight_seg(1.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 8.0], 60, 110, end_reason="hit")
    # the same flight propagated to t = 1.5 s: a "contact" there without impulse
    st = rally._state_at(a, 1.5)
    b = _flight_seg(1.5, st[:3], [st[3] + 1.0, st[4], st[5]], 90, 140, end_reason="net")
    # a real contact: the velocity reverses
    c = _flight_seg(2.5, [5.0, 0.0, 1.0], [-20.0, 0.0, 8.0], 150, 200)
    hs = [HitEvent(frame=60, t=1.0, xy_px=(0, 0), hitter="near", confidence=0.9),
          HitEvent(frame=90, t=1.5, xy_px=(0, 0), hitter="far", confidence=0.6),
          HitEvent(frame=150, t=2.5, xy_px=(0, 0), hitter="far", confidence=0.9)]
    kh, ks, kc = rally._merge_same_flight(hs, [a, b, c], t, rally.RallyConfig())
    assert [h.frame for h in kh] == [60, 150]
    assert ks == [a, c] and kc == [b, None]
    # outside the earlier fit's coverage it is judged a contact
    far = _flight_seg(3.0, st[:3], st[3:], 180, 220)
    kh, _, _ = rally._merge_same_flight(hs[:1] + [replace(hs[1], frame=180, t=3.0)], [a, far], t,
                                        rally.RallyConfig())
    assert len(kh) == 2


def test_merged_flight_net_test_uses_launch_fit():
    """The continuation fit's own 'net' label was decided for the wrong hitter:
    a flight that crossed well above the tape and lands in court is 'in'."""
    hit = HitEvent(frame=60, t=1.0, xy_px=(0, 0), hitter="near", confidence=0.9)
    launch = _flight_seg(1.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 8.0], 60, 110,
                         net_crossing_height=2.0, net_crossing_y=0.0, end_reason="hit")
    cont = _flight_seg(1.6, [3.0, 0.5, 1.2], [4.0, 0.0, -2.0], 96, 140, end_reason="net",
                       landing_xy=(4.0, 0.5), landing_t=2.1)
    cfg = rally.RallyConfig()
    o = rally._outcome_3d(hit, cont, 2.1, False, False, None, cfg, launch)
    assert o.end_reason == "in" and o.winner == "near"
    o_old = rally._outcome_3d(hit, cont, 2.1, False, False, None, cfg)
    assert o_old.end_reason == "net"


def test_long_gap_out_of_view_is_not_a_dead_ball():
    """A 3.4 s hit gap with the shuttle out of view 2.8 s and flying fast on
    both sides of the hole is one rally; the same gap with the shuttle lying
    still in between is two."""
    fps, n = 60.0, 600
    t = np.arange(n) / fps
    xy = np.stack([200 + 400 * t, 500 - 300 * np.sin(3 * t)], 1)   # always moving fast
    vis = np.ones(n, bool)
    vis[(t > 1.3) & (t < 4.1)] = False
    xy[~vis] = np.nan
    tr = Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis, interpolated=np.zeros(n, bool))
    cfg = rally.RallyConfig()
    mo = rally._Motion(tr, 1080.0, cfg)
    a = HitEvent(frame=60, t=1.0, xy_px=(0, 0), hitter="near", confidence=0.9)
    b = HitEvent(frame=264, t=4.4, xy_px=(0, 0), hitter="far", confidence=0.9)
    assert rally._unseen_flight(mo, a, b, cfg)
    groups, cues = rally._group_hits([a, b], [None, None], mo, cfg)
    assert groups == [[0, 1]]
    # shuttle at rest where it reappears -> dead ball
    xy2 = xy.copy()
    rest = (t > 4.1) & (t < 4.39)
    xy2[rest] = xy2[np.flatnonzero(rest)[0]]
    xy2[(t >= 4.1) & (t < 4.4) & ~rest] = xy2[np.flatnonzero(rest)[0]]
    tr2 = replace(tr, xy=xy2)
    mo2 = rally._Motion(tr2, 1080.0, replace(cfg, static_s=0.25))
    assert not rally._unseen_flight(mo2, a, b, replace(cfg, static_s=0.25))


def _hit(frame, t, side, conf=0.9):
    return HitEvent(frame=frame, t=t, xy_px=(0, 0), hitter=side, confidence=conf)


def test_joint_side_decode_fixes_a_label_the_fits_contradict():
    """near, far, (far: wrong), far rally: the third contact's own launch fit
    flies towards the far baseline from the near half and the incoming fit is
    on the near half at its time -> near; the rest stay; posteriors honest."""
    t = np.arange(600) / 60.0
    cfg = rally.RallyConfig()
    s0 = _flight_seg(1.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 6.0], 60, 110)      # near -> far
    s1 = _flight_seg(2.0, [5.0, 0.0, 1.0], [-20.0, 0.0, 6.0], 120, 170)     # far -> near
    s2 = _flight_seg(3.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 6.0], 180, 230)     # near -> far
    s3 = _flight_seg(4.0, [5.0, 0.0, 1.0], [-20.0, 0.0, 6.0], 240, 290)
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "far"), _hit(180, 3.0, "far"), _hit(240, 4.0, "far")]
    out, p = rally._decode_sides(hs, [s0, s1, s2, s3], [None] * 4, [[0, 1, 2, 3]], t, cfg)
    assert [h.hitter for h in out] == ["near", "far", "near", "far"]
    assert all(0.9 < q <= 1.0 for q in p)
    assert rally._own_side_reading(s2, cfg) == "near"
    assert rally._incoming_side_reading(s1, hs[2], t, cfg) == "near"
    # labels are kept where no fit disagrees
    out2, _ = rally._decode_sides(hs[:2], [s0, s1], [None] * 2, [[0, 1]], t, cfg)
    assert [h.hitter for h in out2] == ["near", "far"]


def test_joint_side_decode_alternation_and_missed_contacts():
    t = np.arange(900) / 60.0
    cfg = rally.RallyConfig()
    # no fits: an unknown label between two known ones follows the alternation
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "unknown"), _hit(180, 3.0, "near")]
    out, p = rally._decode_sides(hs, [None] * 3, [None] * 3, [[0, 1, 2]], t, cfg)
    assert [h.hitter for h in out] == ["near", "far", "near"]
    assert p[1] > 0.9
    # the same side twice across 2.6 s (room for a missed contact) is kept
    hs = [_hit(60, 1.0, "near"), _hit(216, 3.6, "near")]
    out, _ = rally._decode_sides(hs, [None] * 2, [None] * 2, [[0, 1]], t, cfg)
    assert [h.hitter for h in out] == ["near", "near"]
    # ... but not 0.8 s apart: one of the two labels must be wrong; with equal
    # evidence the per-hit (marginal) posteriors say so honestly (~0.5)
    hs = [_hit(60, 1.0, "near"), _hit(108, 1.8, "near")]
    out, p = rally._decode_sides(hs, [None] * 2, [None] * 2, [[0, 1]], t, cfg)
    assert max(p) < 0.6
    # a far-side reading of the second contact's own fit decides it
    s1 = _flight_seg(1.8, [5.0, 0.0, 1.0], [-20.0, 0.0, 6.0], 108, 150)
    out, p = rally._decode_sides(hs, [None, s1], [None] * 2, [[0, 1]], t, cfg)
    assert [h.hitter for h in out] == ["near", "far"] and min(p) > 0.8
    # no evidence at all: label kept, P = 0.5
    hs = [_hit(60, 1.0, "unknown")]
    out, p = rally._decode_sides(hs, [None], [None], [[0]], t, cfg)
    assert out[0].hitter == "unknown" and p == [0.5]
    # the old path is still available
    groups = [[0, 1, 2]]
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "unknown"), _hit(180, 3.0, "near")]
    assert [h.hitter for h in rally._fill_hitters(hs, [None] * 3, groups)] == ["near", "far", "near"]


def test_outcome_confidence_carries_the_side_posterior():
    hit = _hit(60, 1.0, "near")
    seg = _flight_seg(1.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 6.0], 60, 110,
                      landing_xy=(4.0, 0.5), landing_t=2.0, net_crossing_height=2.0)
    cfg = rally.RallyConfig()
    o1 = rally._outcome_3d(hit, seg, 2.0, False, False, None, cfg, side_p=1.0)
    o2 = rally._outcome_3d(hit, seg, 2.0, False, False, None, cfg, side_p=0.5)
    assert o1.winner == o2.winner == "near" and o1.end_reason == "in"
    assert o2.confidence == pytest.approx(0.5 * o1.confidence)


def _gap_track(fps=60.0, n=600, hold_until=None, unseen=(1.3, 4.1), y0=500.0, ya=300.0):
    t = np.arange(n) / fps
    xy = np.stack([200 + 400 * t, y0 - ya * np.sin(3 * t)], 1)     # always moving fast
    if hold_until is not None:                                      # served out of a still hold
        xy[t <= hold_until] = xy[np.searchsorted(t, hold_until)]
    vis = np.ones(n, bool)
    vis[(t > unseen[0]) & (t < unseen[1])] = False
    xy[~vis] = np.nan
    return Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis, interpolated=np.zeros(n, bool))


def test_serve_hold_touching_the_contact_is_not_dead_time():
    """A serve at t = 1 s out of a 1 s hold, then out of view 2.8 s, then fast
    flight into the next contact: one rally. The hold cluster reaches the serve's
    contact frame, which used to count as 'rest between the hits'."""
    cfg = rally.RallyConfig()
    tr = _gap_track(hold_until=1.0)
    mo = rally._Motion(tr, 1080.0, cfg)
    a = HitEvent(frame=60, t=1.0, xy_px=(0, 0), hitter="near", confidence=0.9)
    b = HitEvent(frame=264, t=4.4, xy_px=(0, 0), hitter="far", confidence=0.9)
    assert mo.rest[60] and not mo.rest[62]
    assert rally._past_hold(mo, 60, 264) == 61
    groups, _ = rally._group_hits([a, b], [None, None], mo, cfg)
    assert groups == [[0, 1]]
    groups, _ = rally._group_hits([a, b], [None, None], mo, replace(cfg, skip_serve_hold=False))
    assert groups == [[0], [1]]


def test_unseen_arrival_judged_by_last_sighting():
    """Out of view until just before the next contact: a shuttle last seen in
    fast flight high in the frame (over the top) is a flight (one rally); one
    last seen low (a dead shuttle falling at the net) or resting is a dead ball."""
    cfg = rally.RallyConfig()
    tr = _gap_track(unseen=(1.3, 4.39), y0=150.0, ya=100.0)        # last seen at y ~ 219 px of 1080
    mo = rally._Motion(tr, 1080.0, cfg)
    a = HitEvent(frame=60, t=1.0, xy_px=(0, 0), hitter="near", confidence=0.9)
    b = HitEvent(frame=264, t=4.4, xy_px=(0, 0), hitter="far", confidence=0.9)
    assert rally._median_speed(mo, b.t - cfg.launch_window_s, b.t - 1e-3) is None
    assert rally._unseen_flight(mo, a, b, cfg)
    assert not rally._unseen_flight(mo, a, b, replace(cfg, unseen_arrival_ok=False))
    # same flight vanishing low in the frame (y ~ 706 px): not trusted as unseen flight
    low = _gap_track(unseen=(1.3, 4.39))
    assert not rally._unseen_flight(rally._Motion(low, 1080.0, cfg), a, b, cfg)
    # last sighting at rest (lying still for 0.6 s before it went out of view): dead ball
    xy = tr.xy.copy()
    still = (tr.t > 0.6) & (tr.t <= 1.3)
    xy[still] = xy[np.flatnonzero(still)[0]]
    tr2 = replace(tr, xy=xy)
    a2 = HitEvent(frame=30, t=0.5, xy_px=(0, 0), hitter="near", confidence=0.9)
    mo2 = rally._Motion(tr2, 1080.0, cfg)
    assert not rally._unseen_flight(mo2, a2, b, cfg)


def test_next_serve_needs_a_held_first_contact():
    """Who serves next names the previous winner only when the next group's
    first contact is struck out of a seen hold (a real serve); a first contact
    of a falsely split rally (shuttle in flight / unseen before it) is ignored.
    The verdict's confidence carries P(next server side)."""
    seg = _flight_seg(0.0, [-3.0, 1.0, 1.2], [15.0, 0.0, 6.0], 0, 30)
    cfg = rally.RallyConfig()

    def cand(hitter, held, side_p=1.0):
        h = HitEvent(frame=0, t=0.0, xy_px=(0, 0), hitter=hitter, confidence=0.9)
        return rally._Cand(idx=[0], hits=[h], segs=[seg if hitter == "near" else None], start_frame=0,
                           end_frame=30, end_t=0.5, outcome=rally._Outcome(), seg_conf=0.9,
                           cue_before=None, held_before=held, first_side_p=side_p)
    # rally 0 unknown; rally 1 served by near out of a hold -> near won rally 0 (near served first)
    cs = [cand("near", True), cand("near", True, side_p=0.7)]
    rally._infer_from_next_serve(cs, cfg)
    assert cs[0].outcome.winner == "near" and cs[0].outcome.source == "next_serve"
    assert cs[0].outcome.confidence == pytest.approx(0.8 * 0.9 * 0.7)
    cs = [cand("near", True), cand("near", False)]
    rally._infer_from_next_serve(cs, cfg)
    assert cs[0].outcome.winner == "unknown"
    cs = [cand("near", True), cand("near", False)]
    rally._infer_from_next_serve(cs, replace(cfg, next_serve_needs_hold=False))
    assert cs[0].outcome.winner == "near"


def test_next_serve_ignores_first_contact_outside_service_area():
    seg_ok = _flight_seg(0.0, [-3.0, 1.0, 1.2], [15.0, 0.0, 6.0], 0, 30)
    seg_net = _flight_seg(0.0, [-0.5, 1.0, 1.2], [15.0, 0.0, 6.0], 0, 30)
    h = HitEvent(frame=0, t=0.0, xy_px=(0, 0), hitter="near", confidence=0.9)
    mk = lambda s: rally._Cand(idx=[0], hits=[h], segs=[s], start_frame=0, end_frame=30,  # noqa: E731
                               end_t=0.5, outcome=rally._Outcome(), seg_conf=0.9, cue_before=None)
    assert rally._serve_launch_ok(mk(seg_ok))
    assert not rally._serve_launch_ok(mk(seg_net))       # launched at the net: not a serve
    assert rally._serve_launch_ok(mk(None))


def test_x_bracket_bounds_the_drag_ode():
    """_x_bracket bounds x(t) of the exact drag flight; the RK4 (h = 1/60 s)
    solution the reading would otherwise use stays within it + the 0.05 m
    margin (measured worst 5.6 mm up to 504 km/h), so the incoming-side
    shortcut never disagrees with integrating."""
    rng = np.random.default_rng(3)
    for _ in range(400):
        sp = rng.uniform(1.0, 110.0)
        el, az = rng.uniform(-0.6, 1.2), rng.uniform(-0.5, 0.5) + (0 if rng.random() < 0.5 else math.pi)
        v0 = sp * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        seg = _flight_seg(1.0, [rng.uniform(-7, 7), rng.uniform(-3, 3), rng.uniform(0.2, 3.0)], v0, 60, 200)
        for tq in (1.0, 1.01, 1.2, 1.7, 2.5, 3.4):
            lo, hi = rally._x_bracket(seg, tq)
            x = rally._state_at(seg, tq)[0]
            assert lo - 0.01 <= x <= hi + 0.01, (sp, el, tq, lo, x, hi)


def test_spurious_trailing_contact_is_dropped_and_its_fit_kept():
    """A weak last 'contact' followed by almost no motion (a kink as the shuttle
    comes down) is dropped; its fit becomes the end of the real last flight, so
    the landing is judged from it for the real last hitter."""
    fps, n = 60.0, 600
    t = np.arange(n) / fps
    xy = np.stack([300 + 300 * t, 300 + 200 * np.sin(2 * t)], 1)
    rest = t >= 3.05                                     # lies still from 3.05 s on
    xy[rest] = xy[np.flatnonzero(rest)[0]]
    vis = np.ones(n, bool)
    tr = Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis, interpolated=np.zeros(n, bool))
    meta = VideoMeta(fps=fps, width=1920, height=1080, n_frames=n, timestamps=t)
    cfg = rally.RallyConfig()
    mo = rally._Motion(tr, 1080.0, cfg)
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "far"), _hit(180, 3.0, "near", conf=0.6)]
    assert rally._moving_after(mo, hs[2]) < cfg.trail_min_flight_s
    assert rally._moving_after(mo, hs[1]) > cfg.trail_min_flight_s
    s0 = _flight_seg(1.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 6.0], 60, 118, end_reason="hit")
    s1 = _flight_seg(2.0, [5.0, 0.0, 1.0], [-18.0, 0.0, 5.0], 120, 178, end_reason="hit")
    s2 = _flight_seg(3.0, [-4.0, 0.5, 0.4], [-3.0, 0.0, -2.0], 180, 186, end_reason="floor",
                     landing_xy=(-4.2, 0.5), landing_t=3.05)
    cam = _camera(np.random.default_rng(1))
    out = rally.segment_rallies(tr, hs, [s0, s1, s2], meta, cam)
    assert len(out) == 1 and [s.hit.frame for s in out[0].shots] == [60, 120]
    assert out[0].landing_xy == (-4.2, 0.5) and out[0].winner == "far" and out[0].end_reason == "in"
    # a confident last contact is never dropped by this rule
    hs_c = hs[:2] + [replace(hs[2], confidence=0.95)]
    s2f = _flight_seg(3.0, [-4.0, 0.5, 1.4], [18.0, 0.0, 4.0], 180, 240)   # a real launch
    keep = rally.segment_rallies(tr, hs_c, [s0, s1, s2f], meta, cam)
    assert [s.hit.frame for s in keep[0].shots] == [60, 120, 180]


def test_refine_hitters_returns_decoded_sides_on_the_input_list():
    t = np.arange(600) / 60.0
    tr = Track2D(t=t, xy=np.stack([300 + 100 * t, 400 + 50 * np.sin(5 * t)], 1), conf=np.ones(600),
                 visible=np.ones(600, bool), interpolated=np.zeros(600, bool))
    meta = VideoMeta(fps=60.0, width=1920, height=1080, n_frames=600, timestamps=t)
    s0 = _flight_seg(1.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 6.0], 60, 110)
    s1 = _flight_seg(2.0, [5.0, 0.0, 1.0], [-20.0, 0.0, 6.0], 120, 170)
    s2 = _flight_seg(3.0, [-5.0, 0.0, 1.0], [20.0, 0.0, 6.0], 180, 230)
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "far"), _hit(180, 3.0, "far")]
    out, changed = rally.refine_hitters(tr, hs, [s0, s1, s2], meta)
    assert [h.hitter for h in out] == ["near", "far", "near"] and changed == [False, False, True]
    assert [h.frame for h in out] == [60, 120, 180] and hs[2].hitter == "far"     # input untouched


# ═══ 2026-09-29 phase 3 (C): trailing-trim launch guard, serve hold in the one-hit trim ══

def _project_track(cam, pieces, n=600, fps=60.0, hold=None):
    """2-D track = camera projection of fitted flights: pieces [(seg, t_from, t_to)];
    after the last piece the shuttle lies still; hold (t_from, t_to, xyz) = a still hold."""
    t = np.arange(n) / fps
    xy = np.full((n, 2), np.nan)
    for seg, a, b in pieces:
        k = (t >= a - 1e-9) & (t < b - 1e-9)
        xy[k] = cam.project(np.array([rally._state_at(seg, float(tt))[:3] for tt in t[k]]))
    seg, _, b = pieces[-1]
    xy[t >= b - 1e-9] = cam.project(np.array([rally._state_at(seg, b)[:3]]))[0]
    if hold is not None:
        xy[(t >= hold[0]) & (t < hold[1])] = cam.project(np.array([hold[2]]))[0]
    vis = np.isfinite(xy).all(1)
    return Track2D(t=t, xy=xy, conf=vis.astype(float), visible=vis, interpolated=np.zeros(n, bool)), \
        VideoMeta(fps=fps, width=1920, height=1080, n_frames=n, timestamps=t)


def _three_shots():
    s0 = _flight_seg(1.0, [-5.0, 0.0, 1.0], [18.0, 0.0, 10.0], 60, 119, end_reason="hit")
    s1 = _flight_seg(2.0, rally._state_at(s0, 2.0)[:3], [-16.0, 0.0, 9.0], 120, 167, end_reason="hit")
    return s0, s1


def test_trailing_trim_keeps_a_weak_real_kill():
    """A real final kill (0.55 s to the floor) with a weak contact confidence: its
    own fit reverses the incoming flight with a racket-sized |dv| and the 2-D track
    leaves the incoming fit's projection, so the trailing trim keeps it (the
    confidence rule alone dropped it and flipped the winner)."""
    cam = _camera(np.random.default_rng(1))
    s0, s1 = _three_shots()
    s2 = _flight_seg(2.8, rally._state_at(s1, 2.8)[:3], [25.0, 0.0, -8.0], 168, 200, end_reason="floor")
    t_land = rally._floor_time(s2, 5.0)
    s2.landing_xy, s2.landing_t = tuple(float(v) for v in rally._state_at(s2, t_land)[:2]), t_land
    tr, meta = _project_track(cam, [(s0, 1.0, 2.0), (s1, 2.0, 2.8), (s2, 2.8, t_land)])
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "far"), _hit(168, 2.8, "near", conf=0.6)]
    cfg = rally.RallyConfig()
    mo = rally._Motion(tr, 1080.0, cfg)
    assert rally._moving_after(mo, hs[2], cfg) < cfg.trail_min_flight_s <= rally._moving_after(mo, hs[1], cfg)
    assert rally._departure(mo, hs[2], s1, cam, cfg.trail_depart_lag_s) >= cfg.trail_depart_min
    assert rally._launches(hs[2], s2, s1, cfg, mo, cam)
    out = rally.segment_rallies(tr, hs, [s0, s1, s2], meta, cam)
    assert len(out) == 1 and [s.hit.frame for s in out[0].shots] == [60, 120, 168]
    assert out[0].winner == "near" and out[0].end_reason == "in"
    off = rally.segment_rallies(tr, hs, [s0, s1, s2], meta, cam, config=replace(cfg, trail_launch_guard=False))
    assert [s.hit.frame for s in off[0].shots] == [60, 120]
    # no launch evidence: an undetermined own fit, an own fit in the incoming direction
    assert not rally._launches(hs[2], replace(s2, confidence=0.15), s1, cfg, mo, cam)
    assert not rally._launches(hs[2], replace(s2, v0=np.array([-25.0, 0.0, -8.0])), s1, cfg, mo, cam)
    assert not rally._launches(hs[2], None, s1, cfg, mo, cam)


def test_trailing_trim_vetoes_a_misplaced_launch_fit():
    """A weak last 'contact' whose own 3-D fit is placed metres away and reverses
    x, while the 2-D track goes on exactly where the incoming flight predicts:
    one flight, no contact. 3-D alone reads a launch; the 2-D departure test
    (track vs the incoming fit's projection, 0.1 s later) vetoes it."""
    cam = _camera(np.random.default_rng(1))
    s0, s1 = _three_shots()
    t_land = rally._floor_time(s1, 6.0)
    tr, meta = _project_track(cam, [(s0, 1.0, 2.0), (s1, 2.0, t_land)])
    hs = [_hit(60, 1.0, "near"), _hit(120, 2.0, "far"), _hit(180, 3.0, "near", conf=0.6)]
    bad = _flight_seg(3.0, [3.0, 0.0, 1.0], [8.0, 0.0, 2.0], 180, 205, end_reason="floor")
    cfg = rally.RallyConfig()
    mo = rally._Motion(tr, 1080.0, cfg)
    assert rally._moving_after(mo, hs[2], cfg) < cfg.trail_min_flight_s
    assert rally._launches(hs[2], bad, s1, cfg)                       # 3-D evidence only
    assert rally._departure(mo, hs[2], s1, cam, cfg.trail_depart_lag_s) < 1e-6
    assert not rally._launches(hs[2], bad, s1, cfg, mo, cam)
    out = rally.segment_rallies(tr, hs, [s0, s1, bad], meta, cam)
    assert len(out) == 1 and [s.hit.frame for s in out[0].shots] == [60, 120]


def test_one_hit_trim_skips_the_serves_own_hold():
    """A one-shot rally (ace) served out of a 0.8 s still hold that reaches just
    past the detected contact frame (the detector fires 1 frame early; tests
    generator: 3 real serves at 60/120 fps): the hold is not the next rest, so
    the serve is seen moving (not dropped as a dead-time contact) and the rally
    ends where it lands."""
    cam = _camera(np.random.default_rng(1))
    s0 = _flight_seg(1.0, [-3.0, 1.0, 1.1], [14.0, -2.0, 7.0], 60, 110, end_reason="floor")
    t_land = rally._floor_time(s0, 5.0)
    tr, meta = _project_track(cam, [(s0, 1.0, t_land)], hold=(0.2, 1.02, [-3.0, 1.0, 1.1]))
    cfg = rally.RallyConfig()
    mo = rally._Motion(tr, 1080.0, cfg)
    h = _hit(60, 1.0, "near")
    assert mo.rest[60] and mo.rest[61] and not mo.rest[62]
    assert rally._launch_frame(mo, 60, mo.n, cfg) == 61
    assert rally._moving_after(mo, h, cfg) >= cfg.onehit_min_flight_s
    assert rally._moving_after(mo, h, replace(cfg, skip_serve_hold=False)) == 0.0
    # the rally end is looked for past the hold too (2-D path: it ended at the contact)
    out = rally.segment_rallies(tr, [h], [], meta, None)
    assert len(out) == 1 and out[0].end_frame >= int(t_land * 60) - 1
    old = rally.segment_rallies(tr, [h], [], meta, None, config=replace(cfg, skip_serve_hold=False))
    assert old[0].end_frame <= 62


def test_landing_outside_the_hall_makes_no_call():
    """Robustness F3: a final flight fitted across a timestamp jump comes down at (-11.1, -0.3) m,
    4.4 m behind the near baseline. A floor point more than landing_outside_max_m (3 m) beyond the
    doubles court on either axis is no landing: the 3-D outcome makes no call from it
    (source 'implausible_landing', = ShuttlAnalysis LandingGuard), with the same bound as the 2-D
    outcome. The rally keeps its span; the next-serve rule may still name the winner."""
    cfg = rally.RallyConfig()
    hit = _hit(60, 1.0, "far")
    far = _flight_seg(1.0, [5.0, 0.0, 1.0], [-20.0, 0.0, 3.0], 60, 110, landing_xy=(-11.1, -0.3), landing_t=2.0,
                      net_crossing_height=2.0, net_crossing_y=0.0, end_reason="floor")
    o = rally._outcome_3d(hit, far, 2.0, False, False, None, cfg, side_p=1.0)
    assert (o.winner, o.end_reason, o.landing_xy, o.source, o.confidence) == ("unknown", "unknown", None,
                                                                              "implausible_landing", 0.0)
    assert o.end_t == 2.0
    # just inside the bound (out by 2.9 m) is still a normal 'out' call; the y axis gates too
    ok = replace(far, landing_xy=(-(cm.HALF_LENGTH + 2.9), -0.3))
    o2 = rally._outcome_3d(hit, ok, 2.0, False, False, None, cfg, side_p=1.0)
    assert (o2.end_reason, o2.winner, o2.source) == ("out", "near", "3d")
    wide = replace(far, landing_xy=(-4.0, cm.DOUBLES_HALF_WIDTH + 3.2))
    assert rally._outcome_3d(hit, wide, 2.0, False, False, None, cfg, side_p=1.0).source == "implausible_landing"
    loose = replace(cfg, landing_outside_max_m=5.0)
    assert rally._outcome_3d(hit, far, 2.0, False, False, None, loose, side_p=1.0).end_reason == "out"
    assert rally._landing_plausible(cm.HALF_LENGTH + 3.0, 0.0, cfg) and not rally._landing_plausible(math.nan, 0.0, cfg)


def test_next_serve_conflict_and_inferred_serve_are_not_certain():
    """Addendum 3 (codec pair, rally 77495): the next serve CONTRADICTS a geometric call below
    ns_conflict_max_q -> the call is kept but capped at ns_conflict_conf (< the app's 0.6 'uncertain');
    a verdict that rests on a serve nobody saw (inferred from a missed serve) is capped too. A seen next
    serve overruling a weak call or deciding an unknown keeps its confidence; agreement changes nothing;
    a call >= ns_conflict_max_q contradicted by the next serve stays confident."""
    seg = _flight_seg(0.0, [-3.0, 1.0, 1.2], [15.0, 0.0, 6.0], 0, 30)
    cfg = rally.RallyConfig()

    def cand(hitter, held=True, missed=False, w="unknown", q=0.0):
        h = HitEvent(frame=0, t=0.0, xy_px=(0, 0), hitter=hitter, confidence=0.9)
        return rally._Cand(idx=[0], hits=[h], segs=[seg if hitter == "near" else None], start_frame=0, end_frame=30,
                           end_t=0.5, outcome=rally._Outcome(winner=w, end_reason="out" if w != "unknown" else "unknown",
                                                             confidence=q, source="3d"),
                           seg_conf=0.9, cue_before=None, held_before=held, serve_missed=missed, first_side_p=1.0)

    def run(first, nxt):
        cs = [first, nxt]
        rally._infer_from_next_serve(cs, cfg)
        o = cs[0].outcome
        return o.winner, o.end_reason, round(o.confidence, 6), o.source

    # next rally served by near out of a seen hold -> near won rally 0 (near served first)
    assert run(cand("near", w="far", q=0.67), cand("near")) == ("far", "out", 0.5, "3d")          # kept, capped
    assert run(cand("near", w="far", q=0.85), cand("near")) == ("far", "out", 0.85, "3d")        # strong call kept
    assert run(cand("near", w="near", q=0.67), cand("near")) == ("near", "out", 0.67, "3d")      # agreement
    assert run(cand("near", w="far", q=0.4), cand("near")) == ("near", "unknown", 0.72, "next_serve")  # seen: overrule
    assert run(cand("near"), cand("near")) == ("near", "unknown", 0.72, "next_serve")            # seen: decide
    # the next serve was missed (receiver seen first, serve_missed): its server is inferred, not seen
    inferred = cand("far", held=False, missed=True)
    assert inferred.server == "near"
    assert run(cand("near", w="far", q=0.53), inferred) == ("near", "unknown", 0.5, "next_serve")
    assert run(cand("near"), cand("far", held=False, missed=True)) == ("near", "unknown", 0.5, "next_serve")
    off = replace(cfg, ns_conflict_conf=1.0)
    cs = [cand("near", w="far", q=0.67), cand("near")]
    rally._infer_from_next_serve(cs, off)
    assert cs[0].outcome.confidence == 0.67
