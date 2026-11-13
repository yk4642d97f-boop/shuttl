"""
Synthetic ground truth for the v2 tracking stack: rallies, detector noise,
cameras and rendered video. Every stage is measured against this.

  simulate_rally(seed, camera=None, fps=60, n_shots=None, doubles=False) -> SimRally
  simulate_match(seed, n_rallies, camera=None, fps=60, doubles=False) -> SimRally
  corrupt_detections(sim, rng, sigma_px=1.5, miss_rate=0.1, burst_miss=True,
                     false_positive_rate=0.03) -> list[list[ShuttleCandidate]]
  corrupt_px(px, visible, image_size, rng, ...)   same noise model on any pixel track
  truth_track / truth_hits / truth_players / truth_segments / truth_rallies (sim)
                                                  truth as pipeline data types
  random_camera(rng, image_size=(1920,1080), style='behind'|'elevated_side'|'broadcast'|'random')
  render_video(sim, path, image_size=None, draw_players=True, motion_blur=True) -> str

Physics is the *same* drag ODE the reconstruction uses (physics.py): shots
are solved (grid search + bracketed polish on a vectorised RK4, dt = 1/480 s
like physics.simulate) so the free flight lands on the sampled target to
< 1 cm. Shot parameter ranges follow the badminton literature: Cohen et al.
2015 (drag length 4.7 m ±8 %), Phomsoupha & Laffaye 2015 (smash 200–420 km/h
initial, contact 2.4–3.1 m; clear launch 60–150 km/h), BWF Laws §9 (serve
contact < 1.15 m). The receiver intercepts the flight at a reachable point
(min-jerk footwork, ≤ ~3 m/s average) with a height preference that depends
on the incoming shot; unreachable shots are winners.

Ground truth conventions: frame i ↔ timestamps[i]; shuttle_world is NaN
when the shuttle is not flying (before the serve, after it reaches the floor;
a net fault's fall off the net is included, RallyTruth.end_frame = net contact);
`visible` = in play ∧ inside the image ∧ not hidden behind a player's torso.
Detector misses/false positives are NOT in `visible` — corrupt_detections adds
them. Player dict keys: 'near', 'far' (+ 'near2', 'far2' in doubles).
Everything is deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.tracking import court_model as cm
from app.services.tracking import physics
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, ShotSegment3D, ShuttleCandidate, Track2D, VideoMeta,
)

NET_X = 0.0
PLAYER_HEIGHT = (1.65, 1.90)          # m, sampled per player
BODY_HALF_WIDTH = 0.33                # m, bbox half-extent around the feet
SOLVE_DT = 1 / 480                    # == physics.simulate max_dt
MAX_AVG_SPEED = 3.0                   # m/s average over a min-jerk move (peak ≈ 5.6)
REACTION_S = 0.12                     # s after the opponent's contact
SHOT_TYPES = ("serve", "clear", "smash", "drop", "net", "drive", "lift",
              "push", "block", "kill")


# ── ground-truth containers ──────────────────────────────────────────────────

@dataclass
class SimShot:
    """One stroke: true free-flight parameters and where it would land."""
    index: int
    hitter: str                        # 'near' | 'far'
    player: str                        # player key ('near', 'far', 'near2', 'far2')
    shot_type: str
    frame: int                         # contact frame (nearest timestamp)
    t0: float                          # contact time, s
    p0: np.ndarray                     # (3,) m
    v0: np.ndarray                     # (3,) m/s
    drag_length: float
    target_xy: tuple[float, float]     # intended landing point
    landing_xy: tuple[float, float]    # free-flight floor contact (x, y)
    landing_t: float                   # absolute time of that contact
    net_crossing_height: float         # z at x = 0
    in_out: str                        # 'in' | 'out' | 'net' — for the free flight
    margin: float                      # in_out_margin (serve: serve_margin) at landing_xy, NaN for 'net'
    end_t: float                       # flight ends here (next contact / floor / net)
    end_frame: int
    outcome: str                       # 'returned' | 'in' | 'out' | 'net'

    @property
    def speed0_kmh(self) -> float:
        return float(np.linalg.norm(self.v0) * 3.6)


@dataclass
class RallyTruth:
    """Rally-level ground truth (one entry per rally in a match)."""
    start_frame: int                   # serve contact frame
    end_frame: int                     # floor / net contact frame
    end_reason: str                    # 'in' | 'out' | 'net'
    winner: str                        # 'near' | 'far'
    server: str
    landing_xy: tuple[float, float] | None
    margin: float | None               # in_out_margin (None for net)
    first_shot: int                    # index into SimRally.shots
    n_shots: int


@dataclass
class SimRally:
    camera: CameraModel
    fps: float
    timestamps: np.ndarray             # (N,) s
    shuttle_world: np.ndarray          # (N,3) m, NaN when not in play
    shuttle_px: np.ndarray             # (N,2) px, NaN when not in play / behind camera
    visible: np.ndarray                # (N,) bool
    in_play: np.ndarray                # (N,) bool
    occluded: np.ndarray               # (N,) bool (in play, in frame, behind a torso)
    hits: list[dict]                   # {frame, t, hitter, player, pos, shot_type, shot}
    shots: list[SimShot]
    rallies: list[RallyTruth]
    players: dict[str, dict]           # key → {'side', 'foot' (N,2), 'bbox' (N,4), 'height', 'racket' (N,3)}
    doubles: bool
    drag_length: float                 # of the (last) rally
    seed: int
    resting: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    # (from_frame, to_frame, world xyz) where a dead shuttle lies/is held — for render & FP model

    # convenience views of the (single) rally
    @property
    def n(self) -> int:
        return len(self.timestamps)

    @property
    def end_reason(self) -> str:
        return self.rallies[-1].end_reason

    @property
    def landing_xy(self) -> tuple[float, float] | None:
        return self.rallies[-1].landing_xy

    @property
    def winner(self) -> str:
        return self.rallies[-1].winner

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self.camera.image_size)

    @property
    def meta(self) -> VideoMeta:
        w, h = self.camera.image_size
        return VideoMeta(fps=self.fps, width=int(w), height=int(h), n_frames=self.n,
                         timestamps=self.timestamps.copy())

    def image_truth(self, occluded: bool = False, resting: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Where the shuttle is in the image, like real TrackNet labels: (N,2) px
        and mask. In-play frames inside the frame (occluded ones only if
        occluded=True — use that for gap-filled tracks) plus, if resting, the
        dead shuttle held before a serve / lying after a rally."""
        px = self.shuttle_px.copy()
        mask = self.visible | (self.occluded if occluded else False)
        if resting and self.resting:
            w, h = self.camera.image_size
            for a, b, p in self.resting:
                uv = self.camera.project(p)[0]
                if np.all(np.isfinite(uv)) and 0 <= uv[0] < w and 0 <= uv[1] < h:
                    px[a:b + 1] = uv
                    mask[a:b + 1] = True
        px[~mask] = np.nan
        return px, mask

    @property
    def player_foot(self) -> dict[str, np.ndarray]:
        return {k: p["foot"] for k, p in self.players.items()}

    @property
    def player_bbox(self) -> dict[str, np.ndarray]:
        return {k: p["bbox"] for k, p in self.players.items()}


# ── planar shot solver ───────────────────────────────────────────────────────
# Drag is parallel to v and gravity is vertical, so a shot never leaves the
# vertical plane through p0 along its horizontal launch direction: solving in
# (r, z) is exact, not an approximation.

def _flight_kernel(vr0, vz0, z0, d_net, L, dt, t_max, out):  # pragma: no cover - jitted
    """Scalar RK4 per planar flight; identical arithmetic to _flight_stats_np."""
    g, inv = 9.81, 1.0 / L
    nsteps = int(np.ceil(t_max / dt))
    for k in range(vr0.shape[0]):
        r, z, vr, vz, t = 0.0, z0, vr0[k], vz0[k], 0.0
        apex, znet, lr, lt = z0, np.nan, np.nan, np.nan
        for _ in range(nsteps):
            s = np.sqrt(vr * vr + vz * vz) * inv
            a1r, a1z = -s * vr, -g - s * vz
            u2r, u2z = vr + 0.5 * dt * a1r, vz + 0.5 * dt * a1z
            s = np.sqrt(u2r * u2r + u2z * u2z) * inv
            a2r, a2z = -s * u2r, -g - s * u2z
            u3r, u3z = vr + 0.5 * dt * a2r, vz + 0.5 * dt * a2z
            s = np.sqrt(u3r * u3r + u3z * u3z) * inv
            a3r, a3z = -s * u3r, -g - s * u3z
            u4r, u4z = vr + dt * a3r, vz + dt * a3z
            s = np.sqrt(u4r * u4r + u4z * u4z) * inv
            a4r, a4z = -s * u4r, -g - s * u4z
            nr = r + dt / 6 * (vr + 2 * u2r + 2 * u3r + u4r)
            nz = z + dt / 6 * (vz + 2 * u2z + 2 * u3z + u4z)
            nvr = vr + dt / 6 * (a1r + 2 * a2r + 2 * a3r + a4r)
            nvz = vz + dt / 6 * (a1z + 2 * a2z + 2 * a3z + a4z)
            if r < d_net and nr >= d_net:
                f = (d_net - r) / max(nr - r, 1e-12)
                znet = z + f * (nz - z)
            if nz <= 0.0 and z > 0.0:
                f = z / max(z - nz, 1e-12)
                lr, lt = r + f * (nr - r), t + f * dt
                break
            if nz > apex:
                apex = nz
            r, z, vr, vz, t = nr, nz, nvr, nvz, t + dt
        out[k, 0], out[k, 1], out[k, 2], out[k, 3] = lr, lt, apex, znet


def _simulate_kernel(p0, v0, times, L, max_dt, out):  # pragma: no cover - jitted
    """physics.simulate_state positions, scalar-unrolled (same sub-stepping)."""
    g, inv = 9.81, 1.0 / L
    px, py, pz = p0[0], p0[1], p0[2]
    vx, vy, vz = v0[0], v0[1], v0[2]
    t = 0.0
    for i in range(times.shape[0]):
        span = times[i] - t
        n = max(1, int(np.ceil(abs(span) / max_dt)))
        h = span / n
        for _ in range(n):
            s = np.sqrt(vx * vx + vy * vy + vz * vz) * inv
            a1x, a1y, a1z = -s * vx, -s * vy, -g - s * vz
            u2x, u2y, u2z = vx + 0.5 * h * a1x, vy + 0.5 * h * a1y, vz + 0.5 * h * a1z
            s = np.sqrt(u2x * u2x + u2y * u2y + u2z * u2z) * inv
            a2x, a2y, a2z = -s * u2x, -s * u2y, -g - s * u2z
            u3x, u3y, u3z = vx + 0.5 * h * a2x, vy + 0.5 * h * a2y, vz + 0.5 * h * a2z
            s = np.sqrt(u3x * u3x + u3y * u3y + u3z * u3z) * inv
            a3x, a3y, a3z = -s * u3x, -s * u3y, -g - s * u3z
            u4x, u4y, u4z = vx + h * a3x, vy + h * a3y, vz + h * a3z
            s = np.sqrt(u4x * u4x + u4y * u4y + u4z * u4z) * inv
            a4x, a4y, a4z = -s * u4x, -s * u4y, -g - s * u4z
            px += h / 6 * (vx + 2 * u2x + 2 * u3x + u4x)
            py += h / 6 * (vy + 2 * u2y + 2 * u3y + u4y)
            pz += h / 6 * (vz + 2 * u2z + 2 * u3z + u4z)
            vx += h / 6 * (a1x + 2 * a2x + 2 * a3x + a4x)
            vy += h / 6 * (a1y + 2 * a2y + 2 * a3y + a4y)
            vz += h / 6 * (a1z + 2 * a2z + 2 * a3z + a4z)
        t = times[i]
        out[i, 0], out[i, 1], out[i, 2] = px, py, pz


try:  # numba is optional: identical results, ~100× faster
    from numba import njit as _njit
    _flight_jit = _njit(cache=True)(_flight_kernel)
    _simulate_jit = _njit(cache=True)(_simulate_kernel)
except Exception:  # pragma: no cover
    _flight_jit = _simulate_jit = None


def simulate_positions(p0: np.ndarray, v0: np.ndarray, times: np.ndarray,
                       L: float) -> np.ndarray:
    """== physics.simulate(p0, v0, times, L) (times relative to the contact)."""
    times = np.ascontiguousarray(times, dtype=np.float64)
    if _simulate_jit is None:
        return physics.simulate(p0, v0, times, L)
    out = np.empty((len(times), 3))
    _simulate_jit(np.asarray(p0, float), np.asarray(v0, float), times, float(L),
                  1 / 480, out)
    return out


def _flight_stats(speed: np.ndarray, theta: np.ndarray, z0: float, d_net: float,
                  L: float, dt: float, t_max: float = 5.0) -> dict[str, np.ndarray]:
    """RK4 of many planar flights (same scheme as physics.rk4_step).

    Returns range r at floor contact (linear refinement like
    physics.time_to_ground), flight time, apex z and z where r == d_net
    (NaN if never reached before landing)."""
    speed, theta = np.broadcast_arrays(np.asarray(speed, float), np.asarray(theta, float))
    shape = speed.shape
    if _flight_jit is not None:
        out = np.empty((speed.size, 4))
        _flight_jit(np.ascontiguousarray((speed * np.cos(theta)).ravel()),
                    np.ascontiguousarray((speed * np.sin(theta)).ravel()),
                    float(z0), float(d_net), float(L), float(dt), float(t_max), out)
        return {k: out[:, i].reshape(shape) for i, k in
                enumerate(("range", "time", "apex", "z_net"))}
    return _flight_stats_np(speed, theta, z0, d_net, L, dt, t_max)


def _flight_stats_np(speed: np.ndarray, theta: np.ndarray, z0: float, d_net: float,
                     L: float, dt: float, t_max: float = 5.0) -> dict[str, np.ndarray]:
    """Vectorised numpy fallback of _flight_kernel."""
    shape = speed.shape
    vr = (speed * np.cos(theta)).ravel()
    vz = (speed * np.sin(theta)).ravel()
    n = vr.size
    r, z = np.zeros(n), np.full(n, float(z0))
    land_r, land_t = np.full(n, np.nan), np.full(n, np.nan)
    apex, z_net = np.full(n, float(z0)), np.full(n, np.nan)
    alive = np.ones(n, bool)
    g, inv_L, t = physics.G, 1.0 / L, 0.0

    def acc(ur, uz):
        s = np.sqrt(ur * ur + uz * uz) * inv_L
        return -s * ur, -g - s * uz

    for _ in range(int(np.ceil(t_max / dt))):
        a1r, a1z = acc(vr, vz)
        u2r, u2z = vr + 0.5 * dt * a1r, vz + 0.5 * dt * a1z
        a2r, a2z = acc(u2r, u2z)
        u3r, u3z = vr + 0.5 * dt * a2r, vz + 0.5 * dt * a2z
        a3r, a3z = acc(u3r, u3z)
        u4r, u4z = vr + dt * a3r, vz + dt * a3z
        a4r, a4z = acc(u4r, u4z)
        nr = r + dt / 6 * (vr + 2 * u2r + 2 * u3r + u4r)
        nz = z + dt / 6 * (vz + 2 * u2z + 2 * u3z + u4z)
        vr = vr + dt / 6 * (a1r + 2 * a2r + 2 * a3r + a4r)
        vz = vz + dt / 6 * (a1z + 2 * a2z + 2 * a3z + a4z)
        cross_net = alive & (r < d_net) & (nr >= d_net)
        if cross_net.any():
            f = (d_net - r[cross_net]) / np.maximum(nr[cross_net] - r[cross_net], 1e-12)
            z_net[cross_net] = z[cross_net] + f * (nz[cross_net] - z[cross_net])
        land = alive & (nz <= 0.0) & (z > 0.0)
        if land.any():
            f = z[land] / np.maximum(z[land] - nz[land], 1e-12)
            land_r[land] = r[land] + f * (nr[land] - r[land])
            land_t[land] = t + f * dt
            alive &= ~land
        apex = np.where(alive, np.maximum(apex, nz), apex)
        r, z, t = np.where(alive, nr, r), np.where(alive, nz, z), t + dt
        if not alive.any():
            break
    return {k: v.reshape(shape) for k, v in
            (("range", land_r), ("time", land_t), ("apex", apex), ("z_net", z_net))}


def _crossing(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """First x where the sampled curve y(x) crosses `target` upward (NaN if none)."""
    ok = np.isfinite(y)
    for i in range(len(x) - 1):
        if ok[i] and ok[i + 1] and (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (target - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i]))
    return float("nan")


def _polish_speed(theta: float, s_guess: float, D: float, z0: float, d_net: float,
                  L: float) -> float:
    """Refine launch speed at fixed elevation so the fine-dt range hits D."""
    s = s_guess
    for rel in (0.04, 0.003):
        ss = s * (1 + np.linspace(-rel, rel, 13))
        rr = _flight_stats(ss, np.full_like(ss, theta), z0, d_net, L, SOLVE_DT)["range"]
        c = _crossing(ss, rr, D)
        if not np.isfinite(c):
            ok = np.isfinite(rr)
            if not ok.any():
                return float("nan")
            s = float(ss[ok][np.nanargmin(np.abs(rr[ok] - D))])
            continue
        s = c
    return s


def speed_depth_window(p0: np.ndarray, direction: np.ndarray, speed: float, L: float,
                       min_clearance: float = 0.03) -> tuple[float, float] | None:
    """Horizontal-range interval reachable at a fixed launch speed on the flat
    branch while clearing the net by `min_clearance` (fast shots from deep
    cannot land short: they would hit the net)."""
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    d_net = (NET_X - p0[0]) / u[0] if abs(u[0]) > 1e-6 else 1e9
    d_net = d_net if d_net > 0 else 1e9
    th = np.radians(np.linspace(-60, 45, 106))
    st = _flight_stats(np.full_like(th, speed), th, float(p0[2]), d_net, L, 1 / 120)
    rng_, zc = st["range"], st["z_net"] - cm.NET_HEIGHT_CENTRE
    if not np.isfinite(rng_).any():
        return None
    peak = int(np.nanargmax(rng_))
    r, c = rng_[:peak + 1], zc[:peak + 1]
    ok = np.isfinite(r)
    if d_net < 1e8:
        with np.errstate(invalid="ignore"):
            ok &= np.isfinite(c) & (c >= min_clearance) & (r > d_net + 0.3)
    if not ok.any():
        return None
    return float(r[ok].min()), float(rng_[peak])


def solve_shot(p0: np.ndarray, target_xy: tuple[float, float], L: float, mode: str,
               value: float, min_clearance: float | None = 0.03) -> tuple[np.ndarray, dict] | None:
    """Initial velocity from p0 landing at target_xy under drag length L.

    mode: 'apex' (value = apex height, m), 'clearance' (value = z at the net −
    NET_HEIGHT_CENTRE, may be negative for a net fault) or 'speed' (value =
    launch speed m/s; the flatter solution is used). Returns (v0, stats) or
    None if infeasible (caller resamples)."""
    p0 = np.asarray(p0, float)
    dxy = np.asarray(target_xy, float) - p0[:2]
    D = float(np.hypot(*dxy))
    if D < 0.3:
        return None
    u = dxy / D
    d_net = (NET_X - p0[0]) / u[0] if abs(u[0]) > 1e-6 else np.inf
    d_net = d_net if 0 < d_net < D else 1e9
    z0 = float(p0[2])
    coarse = 1 / 120
    if mode == "speed":
        th = np.radians(np.linspace(-60, 45, 211))
        st = _flight_stats(np.full_like(th, value), th, z0, d_net, L, coarse)
        theta = _crossing(th, st["range"], D)
        if not np.isfinite(theta):
            return None
        theta_s = [theta]
        # fine: bisect theta at the fixed speed on the ascending branch
        lo, hi = theta - np.radians(1.0), theta + np.radians(1.0)
        for _ in range(2):
            tt = np.linspace(lo, hi, 11)
            rr = _flight_stats(np.full_like(tt, value), tt, z0, d_net, L, SOLVE_DT)["range"]
            c = _crossing(tt, rr, D)
            if not np.isfinite(c):
                break
            w = (hi - lo) / 10
            theta_s.append(c)
            lo, hi = c - w, c + w
        theta, speed = theta_s[-1], float(value)
    else:
        lo_deg, hi_deg = (8.0, 80.0) if mode == "apex" else (-35.0, 82.0)
        th = np.radians(np.linspace(lo_deg, hi_deg, 30))
        sp = np.geomspace(2.5, 120.0, 36)
        S, T = np.meshgrid(sp, th)
        st = _flight_stats(S, T, z0, d_net, L, coarse)
        s_req = np.array([_crossing(sp, st["range"][i], D) for i in range(len(th))])
        stat = st["apex"] if mode == "apex" else st["z_net"] - cm.NET_HEIGHT_CENTRE
        q = np.full(len(th), np.nan)
        for i in range(len(th)):
            if np.isfinite(s_req[i]):
                row = stat[i]
                ok = np.isfinite(row)
                if ok.sum() >= 2:
                    q[i] = np.interp(s_req[i], sp[ok], row[ok])
        theta = _crossing(th, q, value)
        if not np.isfinite(theta):
            return None
        speed = float(np.interp(theta, th[np.isfinite(s_req)], s_req[np.isfinite(s_req)]))
    if mode != "speed":
        speed = _polish_speed(theta, speed, D, z0, d_net, L)
        if not np.isfinite(speed):
            return None
    fine = _flight_stats(np.array([speed]), np.array([theta]), z0, d_net, L, SOLVE_DT)
    stats = {k: float(v[0]) for k, v in fine.items()}
    if not np.isfinite(stats["range"]) or abs(stats["range"] - D) > 0.02:
        return None
    if d_net < 1e8:
        clearance = stats["z_net"] - cm.NET_HEIGHT_CENTRE
        if min_clearance is not None and not clearance >= min_clearance:
            return None
    v0 = np.array([speed * np.cos(theta) * u[0], speed * np.cos(theta) * u[1],
                   speed * np.sin(theta)])
    stats.update(theta_deg=float(np.degrees(theta)), speed=speed)
    return v0, stats


def _solve_net_fault(rng: np.random.Generator, mode: str, p0: np.ndarray,
                     target_xy: tuple[float, float], L: float, value: float
                     ) -> tuple[np.ndarray, dict] | None:
    """A shot that meets the net 4–60 cm below the tape (fast shots keep their speed)."""
    z_at_net = cm.NET_HEIGHT_CENTRE - float(rng.uniform(0.04, 0.6))
    if mode != "speed":
        return solve_shot(p0, target_xy, L, "clearance", z_at_net - cm.NET_HEIGHT_CENTRE,
                          min_clearance=None)
    d = np.asarray(target_xy, float) - p0[:2]
    u = d / np.linalg.norm(d)
    if abs(u[0]) < 1e-6:
        return None
    d_net = (NET_X - p0[0]) / u[0]
    if d_net <= 0:
        return None
    th = np.radians(np.linspace(-60, 40, 101))
    st = _flight_stats(np.full_like(th, value), th, float(p0[2]), d_net, L, 1 / 120)
    theta = _crossing(th, st["z_net"], z_at_net)
    if not np.isfinite(theta):
        return None
    v0 = np.array([value * np.cos(theta) * u[0], value * np.cos(theta) * u[1],
                   value * np.sin(theta)])
    return v0, {"speed": value, "theta_deg": float(np.degrees(theta)), "z_net": z_at_net}


# ── shot generator ───────────────────────────────────────────────────────────
# depth = distance beyond the net on the receiving side (m); mode/value feed
# solve_shot (speed values in km/h, (lo, mode, hi) = triangular); `long` =
# the shot's natural error is long (baseline) rather than wide (side line).
SHOT_SPECS: dict[str, dict] = {
    "clear":       dict(depth=(5.0, 6.6), mode="apex", value=(4.2, 7.5), long=True),
    "lift":        dict(depth=(4.6, 6.6), mode="apex", value=(3.6, 6.5), long=True),
    "drop":        dict(depth=(0.5, 2.6), mode="clearance", value=(0.05, 0.45), long=False,
                        max_rise=0.3, max_kmh=160.0),
    "net":         dict(depth=(0.25, 1.8), mode="clearance", value=(0.02, 0.2), long=False,
                        max_apex=2.2, max_kmh=45.0, loose=(0.3, 0.6), loose_apex=2.6),
    "push":        dict(depth=(2.4, 4.8), mode="clearance", value=(0.05, 0.35), long=True,
                        max_apex=2.7, max_kmh=90.0, loose=(0.35, 0.6), loose_apex=2.9),
    "block":       dict(depth=(0.4, 2.4), mode="clearance", value=(0.03, 0.3), long=False,
                        max_apex=2.5, max_kmh=75.0, loose=(0.3, 0.55), loose_apex=2.7),
    "smash":       dict(depth=(1.8, 6.6), mode="speed", value=(200.0, 290.0, 420.0), long=False),
    "kill":        dict(depth=(1.0, 5.5), mode="speed", value=(120.0, 170.0, 260.0), long=False),
    "drive":       dict(depth=(3.2, 6.6), mode="speed", value=(100.0, 140.0, 200.0), long=True,
                        max_apex=2.9),
    "serve_short": dict(depth=(2.03, 2.7), mode="clearance", value=(0.03, 0.2), long=False),
    "serve_long":  dict(depth=(5.8, 6.6), mode="apex", value=(4.5, 7.2), long=True),
}
# optional per-type realism caps: max_apex (m), max_rise (apex − contact height, m),
# max_kmh (launch speed) — violating solutions are resampled. `loose`: with
# probability LOOSE_P a returned front-court shot is played too high (clearance
# range `loose`, apex cap `loose_apex`) — the opponent's kill opportunity.
LOOSE_P = 0.3
# preferred contact height (mean, sd) of the receiver given the incoming shot
_PREF_Z = {"clear": (2.7, 0.35), "lift": (2.6, 0.4), "serve_long": (2.6, 0.4),
           "smash": (0.9, 0.4), "kill": (0.9, 0.4), "drive": (1.3, 0.35),
           "push": (1.2, 0.4), "drop": (0.8, 0.4), "block": (0.7, 0.35),
           "net": (0.9, 0.4), "serve_short": (1.1, 0.35)}
# final-shot outcome weights (in, out, net)
_END_W = {"smash": (0.7, 0.12, 0.18), "kill": (0.7, 0.12, 0.18), "drop": (0.5, 0.15, 0.35),
          "net": (0.5, 0.15, 0.35), "block": (0.5, 0.15, 0.35), "clear": (0.4, 0.6, 0.0),
          "lift": (0.4, 0.6, 0.0), "drive": (0.45, 0.3, 0.25), "push": (0.45, 0.3, 0.25)}


def _draw(rng: np.random.Generator, v: tuple) -> float:
    return float(rng.uniform(v[0], v[1]) if len(v) == 2 else rng.triangular(*v))


def _half_width(doubles: bool) -> float:
    return cm.DOUBLES_HALF_WIDTH if doubles else cm.SINGLES_HALF_WIDTH


def _sample_target(rng: np.random.Generator, spec: dict, recv_sign: float, hw: float,
                   outcome: str) -> tuple[float, float]:
    """Landing target on the receiving half; 'in'/'out' finals sit near a line
    (half of them within 20 cm) so close calls are well represented."""
    lo, hi = spec["depth"]
    ex, ey = cm.HALF_LENGTH + cm.LINE_WIDTH / 2, hw + cm.LINE_WIDTH / 2

    def lateral(limit: float) -> float:
        f = rng.uniform(0.55, 1.0) if rng.random() < 0.7 else rng.uniform(0.0, 0.55)
        return float(f * limit * rng.choice([-1.0, 1.0]))

    if outcome in ("in", "out"):
        close = rng.random() < 0.5
        m = rng.uniform(0.005, 0.2) if close else rng.uniform(0.2, 1.0)
        if spec["long"]:
            if outcome == "in":
                depth, y = ex - m, lateral(max(ey - m - 0.08, 0.3))
            else:
                depth, y = ex + m, lateral(hw - 0.1)
        else:
            depth = rng.uniform(lo, min(hi, ex - m - 0.08) if outcome == "in" else hi)
            y = (ey - m if outcome == "in" else ey + m) * rng.choice([-1.0, 1.0])
        return recv_sign * float(depth), float(y)
    depth = min(rng.uniform(lo, hi), 6.62)
    return recv_sign * float(depth), lateral(hw - 0.1)


def _plan_shot(rng: np.random.Generator, kind: str, p0: np.ndarray, recv_sign: float,
               L: float, hw: float, outcome: str, target: tuple[float, float] | None = None,
               tries: int = 6) -> tuple[np.ndarray, dict, tuple[float, float]] | None:
    """Sample target/parameters for a shot type and solve v0 (None if infeasible)."""
    spec = SHOT_SPECS[kind]
    if "loose" in spec and outcome == "returned" and rng.random() < LOOSE_P:
        spec = dict(spec, value=spec["loose"], max_apex=spec["loose_apex"])
    for attempt in range(tries):
        val = _draw(rng, spec["value"])
        if spec["mode"] == "speed":
            val = val / 3.6 * (0.88 ** max(0, attempt - 2))
        tgt = target if target is not None else _sample_target(rng, spec, recv_sign, hw, outcome)
        if outcome == "net":
            res = _solve_net_fault(rng, spec["mode"], p0, tgt, L, val)
        else:
            if spec["mode"] == "speed":
                d = np.subtract(tgt, p0[:2])
                win = speed_depth_window(p0, d, val, L)
                if win is None:
                    continue
                r = float(np.hypot(*d))
                if not (win[0] + 0.02 <= r <= win[1] - 0.02):
                    if outcome != "returned" or target is not None:
                        continue
                    dy, ax = tgt[1] - p0[1], abs(p0[0])
                    dlo = np.sqrt(max(win[0] ** 2 - dy ** 2, 0.0)) - ax + 0.05
                    dhi = np.sqrt(max(win[1] ** 2 - dy ** 2, 0.0)) - ax - 0.05
                    lo, hi = max(dlo, spec["depth"][0]), min(dhi, spec["depth"][1], 6.62)
                    if lo >= hi:
                        continue
                    tgt = (recv_sign * float(rng.uniform(lo, hi)), tgt[1])
            res = solve_shot(p0, tgt, L, spec["mode"], val, min_clearance=0.03)
        if res is not None and _within_caps(spec, res[0], res[1], float(p0[2])):
            return res[0], res[1], tgt
    return None


def _within_caps(spec: dict, v0: np.ndarray, stats: dict, z0: float) -> bool:
    apex = stats.get("apex")
    if apex is not None and np.isfinite(apex):
        if apex > spec.get("max_apex", np.inf) or apex - z0 > spec.get("max_rise", np.inf):
            return False
    return float(np.linalg.norm(v0)) * 3.6 <= spec.get("max_kmh", np.inf)


def _kind_options(z: float, depth: float) -> dict[str, float]:
    """Shot types a player can play from contact height z and depth behind the net."""
    if z >= 2.1:
        if depth >= 4.3:
            opts = {"smash": 0.35, "clear": 0.35, "drop": 0.30}
        elif depth >= 2.5:
            opts = {"smash": 0.55, "drop": 0.30, "clear": 0.15}
        else:
            opts = {"kill": 0.75, "drop": 0.1, "net": 0.15}
    elif z >= 1.2:
        if z >= 1.6 and depth < 3.2:          # above the tape, mid/front court: attack
            opts = ({"kill": 0.5, "push": 0.25, "net": 0.25} if depth < 2.3
                    else {"kill": 0.4, "drive": 0.35, "push": 0.25})
        elif depth < 2.3:
            opts = {"push": 0.4, "net": 0.4, "lift": 0.2}
        elif depth < 3.5:
            opts = {"drive": 0.45, "lift": 0.3 if z < 1.9 else 0.0, "push": 0.25}
        else:
            opts = {"drive": 0.5, "lift": 0.5 if z < 1.9 else 0.0, "clear": 0.0 if z < 1.9 else 0.5}
    elif depth < 2.5:
        opts = {"net": 0.5, "lift": 0.4, "push": 0.1 if z >= 0.7 else 0.0}
    else:
        opts = {"lift": 0.45, "block": 0.35, "drive": 0.2 if z >= 0.8 else 0.0}
    return {k: v for k, v in opts.items() if v > 0}


def _fallback_kinds(z: float, depth: float) -> list[str]:
    """Plan order when the chosen type is infeasible: every type allowed at this
    contact (by weight), then height-consistent defaults — labels stay physical."""
    opts = _kind_options(z, depth)
    order = sorted(opts, key=lambda k: -opts[k])
    last = (["clear", "drop"] if z >= 1.9 else ["drive", "lift"] if z >= 1.2
            else ["lift", "net", "block"])
    return list(dict.fromkeys(order + last))


def _choose_kind(rng: np.random.Generator, z: float, depth: float) -> str:
    """Next shot type from contact height and distance behind the net."""
    opts = _kind_options(z, depth)
    k = list(opts)
    p = np.array([opts[x] for x in k])
    return str(rng.choice(k, p=p / p.sum()))


# ── rally engine ─────────────────────────────────────────────────────────────

def _minjerk(s: np.ndarray | float) -> np.ndarray | float:
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class _Mover:
    """Piecewise min-jerk footwork; a new move starts from wherever the player is
    and interrupts later moves (moves are appended chronologically)."""

    def __init__(self, xy):
        self.p0 = np.asarray(xy, float)
        self.moves: list[tuple[float, float, np.ndarray, np.ndarray]] = []

    def at(self, t: float) -> np.ndarray:
        for t0, t1, a, b in reversed(self.moves):
            if t >= t0:
                s = 1.0 if t >= t1 else _minjerk((t - t0) / (t1 - t0))
                return a + (b - a) * s
        return self.p0.copy()

    def move(self, t0: float, t1: float, goal) -> None:
        self.moves = [m for m in self.moves if m[0] <= t0]
        a = self.at(t0)
        self.moves.append((t0, max(t1, t0 + 0.05), a, np.asarray(goal, float)))

    def recover(self, t0: float, goal, speed: float = 2.2) -> None:
        d = float(np.linalg.norm(np.asarray(goal) - self.at(t0)))
        self.move(t0, t0 + 0.3 + d / speed, goal)

    def positions(self, ts: np.ndarray) -> np.ndarray:
        out = np.tile(self.p0, (len(ts), 1))
        for k, (t0, t1, a, b) in enumerate(self.moves):
            nxt = self.moves[k + 1][0] if k + 1 < len(self.moves) else np.inf
            m = (ts >= t0) & (ts < nxt)
            if m.any():
                s = _minjerk(np.clip((ts[m] - t0) / (t1 - t0), 0.0, 1.0))
                out[m] = a + (b - a) * s[:, None]
        return out


@dataclass
class _Flight:
    """Free flight from (p0, v0) at absolute time t0; dense samples for contact search."""
    t0: float
    p0: np.ndarray
    v0: np.ndarray
    L: float
    ts: np.ndarray                    # relative sample times (1/120 s) before landing
    P: np.ndarray                     # positions at ts
    t_land: float                     # relative; inf if never lands within t_max
    p_land: np.ndarray
    t_net: float | None = None        # relative time of the x = 0 crossing
    p_net: np.ndarray | None = None
    t_end: float = np.inf             # absolute end of this segment (set by the rally)


def _refine_crossing(p0, v0, L, ta: float, tb: float, f) -> float:
    tf = np.linspace(ta, tb, 33)
    val = f(simulate_positions(p0, v0, tf, L))
    j = int(np.nonzero(val <= 0)[0][0]) if (val <= 0).any() else len(tf) - 1
    if j == 0:
        return float(tf[0])
    return float(tf[j - 1] + val[j - 1] / (val[j - 1] - val[j]) * (tf[j] - tf[j - 1]))


def _free_flight(t0: float, p0: np.ndarray, v0: np.ndarray, L: float,
                 t_max: float = 6.0) -> _Flight:
    ts = np.arange(0.0, t_max, 1 / 120)
    P = simulate_positions(p0, v0, ts, L)
    below = np.nonzero(P[:, 2] <= 0.0)[0]
    if len(below) and below[0] > 0:
        i = int(below[0])
        t_land = _refine_crossing(p0, v0, L, ts[i - 1], ts[i], lambda Q: Q[:, 2])
        p_land = simulate_positions(p0, v0, np.array([t_land]), L)[0]
        keep = ts < t_land
        ts, P = ts[keep], P[keep]
    else:
        t_land, p_land = np.inf, P[-1]
    fl = _Flight(t0, np.asarray(p0, float), np.asarray(v0, float), L, ts, P, t_land, p_land)
    side0 = np.sign(p0[0] - NET_X)
    cross = np.nonzero(np.sign(P[:, 0] - NET_X) != side0)[0]
    if side0 != 0 and len(cross):
        i = int(cross[0])
        fl.t_net = _refine_crossing(p0, v0, L, ts[i - 1], ts[i],
                                    lambda Q: side0 * (Q[:, 0] - NET_X))
        fl.p_net = simulate_positions(p0, v0, np.array([fl.t_net]), L)[0]
    return fl


def _reach(z: np.ndarray) -> np.ndarray:
    """Horizontal foot→contact distance a player covers with arm + racket (+ lunge)."""
    return np.where(z >= 1.9, 0.35, np.where(z >= 0.9, 0.8, 1.15))


def _find_contact(rng: np.random.Generator, fl: _Flight, keys: list[str], movers: dict,
                  players: dict, recv_sign: float, incoming: str, hw: float
                  ) -> tuple[str, int, np.ndarray] | None:
    """Pick the receiver's contact sample (player key, sample index, foot xy)."""
    ts, P = fl.ts, fl.P
    depth = recv_sign * (P[:, 0] - NET_X)
    base_ok = (depth > 0.3) & (depth < cm.HALF_LENGTH + 1.3) & (np.abs(P[:, 1]) < hw + 1.0)
    base_ok &= (P[:, 2] > 0.12) & (ts > REACTION_S + 0.12)
    mu, sd = _PREF_Z.get(incoming, (1.5, 0.6))
    if incoming in ("net", "push", "block", "drop", "serve_short") and rng.random() < 0.45:
        mu, sd = 1.9, 0.3                    # attacking player takes it early and high
    best = []
    for key in keys:
        pl, mv = players[key], movers[key]
        zmax = 1.33 * pl["height"] + 0.55
        ok = base_ok & (P[:, 2] < zmax)
        if not ok.any():
            continue
        start = mv.at(fl.t0 + REACTION_S)
        dist = np.maximum(np.linalg.norm(P[:, :2] - start, axis=1) - _reach(P[:, 2]), 0.0)
        ok &= dist <= MAX_AVG_SPEED * (ts - REACTION_S) + 0.05
        if not ok.any():
            continue
        w = (np.exp(-0.5 * ((P[:, 2] - mu) / sd) ** 2) + 0.03) * np.exp(-dist / 2.0)
        w[~ok] = 0.0
        best.append((key, w, start))
    if not best:
        return None
    W = np.concatenate([b[1] for b in best])
    pick = int(rng.choice(len(W), p=W / W.sum()))
    key, _, start = best[pick // len(ts)]
    j = pick % len(ts)
    c = P[j]
    if c[2] >= 1.9:            # overhead: feet behind and beside the contact
        foot = c[:2] + np.array([recv_sign * rng.uniform(0.1, 0.35),
                                 -players[key]["racket_side"] * 0.2])
    else:
        d = c[:2] - start
        n = float(np.linalg.norm(d))
        u = d / n if n > 1e-6 else np.array([-recv_sign, 0.0])
        foot = c[:2] - u * min(_reach(c[2:3])[0] * rng.uniform(0.7, 1.0), max(n, 0.3))
    foot[0] = recv_sign * max(recv_sign * foot[0], 0.35)
    return key, j, foot


def _bases(side_sign: float, doubles: bool, rng: np.random.Generator) -> list[np.ndarray]:
    if not doubles:
        return [np.array([side_sign * rng.uniform(3.4, 4.1), rng.uniform(-0.3, 0.3)])]
    x = side_sign * rng.uniform(3.4, 4.2)
    return [np.array([x, -1.3]), np.array([x, 1.3])]


# ── rally loop ───────────────────────────────────────────────────────────────

def _new_players(rng: np.random.Generator, doubles: bool) -> dict[str, dict]:
    keys = [("near", -1.0), ("far", 1.0)] + ([("near2", -1.0), ("far2", 1.0)] if doubles else [])
    out = {}
    for k, s in keys:
        right_handed = rng.random() < 0.87
        out[k] = {"side": "near" if s < 0 else "far", "sign": s,
                  "height": float(rng.uniform(*PLAYER_HEIGHT)),
                  # world-y direction of the racket hand (a right-hander facing +x has it at -y)
                  "racket_side": s * (1.0 if right_handed else -1.0)}
    return out


def _side_keys(players: dict, side: str) -> list[str]:
    return [k for k, p in players.items() if p["side"] == side]


def _serve_positions(rng: np.random.Generator, players: dict, server_side: str,
                     doubles: bool) -> tuple[str, dict[str, np.ndarray], float]:
    """Serve formation: (server key, foot xy per player, y-sign of the server's court)."""
    s = -1.0 if server_side == "near" else 1.0
    court = float(rng.choice([-1.0, 1.0]))
    srv_keys, rcv_keys = _side_keys(players, server_side), _side_keys(
        players, "far" if server_side == "near" else "near")
    server = str(rng.choice(srv_keys))
    receiver = str(rng.choice(rcv_keys))
    pos = {}
    if doubles:
        pos[server] = np.array([s * rng.uniform(2.1, 2.6), court * rng.uniform(0.2, 0.6)])
        pos[receiver] = np.array([-s * rng.uniform(2.3, 2.8), -court * rng.uniform(0.3, 0.8)])
    else:
        pos[server] = np.array([s * rng.uniform(2.4, 3.4), court * rng.uniform(0.25, 0.9)])
        pos[receiver] = np.array([-s * rng.uniform(2.9, 3.8), -court * rng.uniform(0.3, 1.0)])
    for k in srv_keys:
        if k != server:
            pos[k] = np.array([s * rng.uniform(3.8, 4.4), -court * rng.uniform(0.2, 0.8)])
    for k in rcv_keys:
        if k != receiver:
            pos[k] = np.array([-s * rng.uniform(3.8, 4.4), court * rng.uniform(0.3, 0.9)])
    return server, pos, court


def serve_margin(x: float, y: float, recv_sign: float, court: float, doubles: bool) -> float:
    """Signed distance (m) of a serve landing from the receiver's service-court
    edge (lines are in): short service line, long service line (baseline in
    singles), centre line and side line of the court diagonal to the server
    (the server stands at y-sign `court`, the receiver's box has y-sign −court)."""
    lw = cm.LINE_WIDTH / 2
    long_x = cm.DOUBLES_LONG_SERVICE_X if doubles else cm.HALF_LENGTH
    d, yy = recv_sign * x, -court * y
    return float(min(d - (cm.SHORT_SERVICE_X - lw), long_x + lw - d, yy + lw,
                     _half_width(doubles) + lw - yy))


def _serve_target(rng: np.random.Generator, kind: str, recv_sign: float, court: float,
                  doubles: bool, hw: float, outcome: str) -> tuple[float, float]:
    """Target in the diagonally opposite service court; 'in'/'out' finals sit
    within 0.5–20 cm of a service-court line half of the time."""
    lw = cm.LINE_WIDTH / 2
    long_x = cm.DOUBLES_LONG_SERVICE_X if doubles else cm.HALF_LENGTH
    x_lo, x_hi, y_lo, y_hi = cm.SHORT_SERVICE_X - lw, long_x + lw, -lw, hw + lw
    short = kind == "serve_short"
    if outcome in ("in", "out"):
        m = rng.uniform(0.005, 0.2) if rng.random() < 0.5 else rng.uniform(0.2, 0.7)
        sg = 1.0 if outcome == "in" else -1.0
        edges, w = (("short", "side", "centre"), (0.6, 0.25, 0.15)) if short else \
            (("long", "side", "centre"), (0.6, 0.25, 0.15))
        edge = str(rng.choice(edges, p=w))
        depth = rng.uniform(2.1, 2.8) if short else rng.uniform(long_x - 0.8, long_x - 0.15)
        yy = rng.uniform(0.35, hw - 0.35)
        if edge == "short":
            depth = x_lo + sg * m
        elif edge == "long":
            depth = x_hi - sg * m
        elif edge == "side":
            yy = y_hi - sg * m
        else:
            yy = y_lo + sg * m
        return recv_sign * float(depth), -court * float(yy)
    if short:
        depth, y = rng.uniform(2.05, 2.8), rng.uniform(0.1, 1.4)
    else:
        depth = rng.uniform(5.2, 5.9) if doubles else rng.uniform(5.6, 6.6)
        y = rng.uniform(0.2, hw - 0.15)
    return recv_sign * float(depth), -court * float(y)


def _flight_px_ok(fl: _Flight) -> bool:
    return len(fl.ts) > 3


def _play_rally(rng: np.random.Generator, players: dict, movers: dict, bases: dict,
                t_serve: float, server_side: str, n_shots: int | None, doubles: bool,
                L: float, first: bool) -> dict:
    """Play one rally from the serve at t_serve. Returns shots/segments/truth."""
    hw = _half_width(doubles)
    server, pos, court = _serve_positions(rng, players, server_side, doubles)
    for k, xy in pos.items():                      # walk into the serve formation
        if first:
            movers[k].p0 = xy.copy()
        else:
            movers[k].move(t_serve - 2.2, t_serve - 0.6, xy)
    target_n = int(n_shots) if n_shots is not None else int(np.clip(rng.geometric(1 / 8.5), 1, 40))
    pl = players[server]
    s = pl["sign"]
    p0 = np.array([pos[server][0] - s * 0.35, pos[server][1] + pl["racket_side"] * 0.3,
                   rng.uniform(0.88, 1.12)])
    kind = "serve_short" if rng.random() < (0.88 if doubles else 0.45) else "serve_long"
    hitter, t0 = server, float(t_serve)
    shots, segments, hits = [], [], []
    end: dict = {}
    for i in range(10_000):
        hs = players[hitter]["sign"]
        recv_sign, recv_side = -hs, ("far" if hs < 0 else "near")
        final = i >= target_n - 1
        outcome = "returned"
        if final:
            base_kind = "serve" if kind.startswith("serve") else kind
            w = (0.45, 0.25, 0.30) if base_kind == "serve" else _END_W.get(base_kind, (0.5, 0.25, 0.25))
            outcome = str(rng.choice(["in", "out", "net"], p=np.array(w) / sum(w)))
        chosen = None
        kinds = [kind] + [k for k in _fallback_kinds(float(p0[2]), abs(float(p0[0])))
                          if k != kind]
        if kind.startswith("serve"):
            kinds = [kind, "serve_long" if kind == "serve_short" else "serve_short"]
        tries = 10 if n_shots is not None else 3
        for kk in kinds:
            for attempt in range(tries):
                tgt = (_serve_target(rng, kk, recv_sign, court, doubles, hw, outcome)
                       if kk.startswith("serve") else None)
                planned = _plan_shot(rng, kk, p0, recv_sign, L, hw, outcome, target=tgt, tries=3)
                if planned is None:
                    continue
                v0, st, tgt = planned
                fl = _free_flight(t0, p0, v0, L)
                if not _flight_px_ok(fl):
                    continue
                if outcome != "returned":
                    chosen = (kk, v0, tgt, fl, None, outcome)
                    break
                contact = _find_contact(rng, fl, _side_keys(players, recv_side), movers,
                                        players, recv_sign, kk, hw)
                if contact is not None:
                    chosen = (kk, v0, tgt, fl, contact, outcome)
                    break
                if attempt == tries - 1:          # unreachable → a clean winner
                    chosen = (kk, v0, tgt, fl, None, "winner")
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("shot planning failed")          # pragma: no cover
        kind, v0, tgt, fl, contact, outcome = chosen
        # ---- free-flight truth of this stroke
        land_xy = (float(fl.p_land[0]), float(fl.p_land[1])) if np.isfinite(fl.t_land) else (np.nan, np.nan)
        z_net = float(fl.p_net[2]) if fl.p_net is not None else float("nan")
        crosses = fl.t_net is not None and fl.t_net < fl.t_land
        if crosses and z_net < cm.NET_HEIGHT_CENTRE:
            ff_io, ff_margin = "net", float("nan")
        else:
            ff_margin = (serve_margin(*land_xy, recv_sign, court, doubles) if i == 0
                         else cm.in_out_margin(*land_xy, doubles=doubles))
            ff_io = "in" if ff_margin >= 0 else "out"
        hits.append({"t": t0, "hitter": players[hitter]["side"], "player": hitter,
                     "pos": p0.copy(), "shot_type": "serve" if kind.startswith("serve") else kind,
                     "shot": len(shots)})
        home = bases[hitter]
        if kind in ("net", "push", "block", "drop", "kill", "smash") and rng.random() < 0.5:
            home = np.array([hs * rng.uniform(2.0, 2.7), home[1] * 0.5])   # follow in to the net
        movers[hitter].recover(t0 + 0.12, home)
        if contact is not None:                               # returned by the receiver
            key, j, foot = contact
            t_c = t0 + float(fl.ts[j])
            movers[key].move(t0 + REACTION_S, t_c - 0.03, foot)
            for k in _side_keys(players, recv_side):
                if k != key:
                    movers[k].recover(t0 + REACTION_S + 0.1, bases[k])
            seg_end, shot_outcome = t_c, "returned"
            segments.append((t0, t_c, p0.copy(), v0.copy(), L))
        elif ff_io == "net":
            t_n = t0 + fl.t_net
            segments.append((t0, t_n, p0.copy(), v0.copy(), L))
            # dead shuttle drops off the net tape/mesh on the hitter's side
            pd = fl.p_net + np.array([hs * 0.05, 0.0, 0.0])
            vd = np.array([hs * rng.uniform(0.2, 1.0), rng.normal(0, 0.2), rng.uniform(-0.5, 0.3)])
            drop = _free_flight(t_n, pd, vd, L, t_max=3.0)
            t_floor = t_n + drop.t_land
            segments.append((t_n, t_floor, pd, vd, L))
            seg_end, shot_outcome = t_n, "net"
            end = dict(reason="net", t_end=t_n, t_floor=t_floor, rest=drop.p_land,
                       landing=None, margin=None, winner=players[hitter]["side"] == "near" and "far" or "near")
        else:                                                   # lands: in / out
            t_l = t0 + fl.t_land
            segments.append((t0, t_l, p0.copy(), v0.copy(), L))
            seg_end, shot_outcome = t_l, ff_io
            hitter_side = players[hitter]["side"]
            other = "far" if hitter_side == "near" else "near"
            end = dict(reason=ff_io, t_end=t_l, t_floor=t_l, rest=fl.p_land, landing=land_xy,
                       margin=ff_margin, winner=hitter_side if ff_io == "in" else other)
            for k in _side_keys(players, recv_side):   # receiver gives up / watches it land
                movers[k].recover(t0 + REACTION_S + 0.15, bases[k], speed=1.2)
        shots.append(SimShot(
            index=len(shots), hitter=players[hitter]["side"], player=hitter,
            shot_type="serve" if kind.startswith("serve") else kind, frame=-1, t0=t0,
            p0=p0.copy(), v0=v0.copy(), drag_length=L, target_xy=(float(tgt[0]), float(tgt[1])),
            landing_xy=land_xy, landing_t=t0 + fl.t_land, net_crossing_height=z_net,
            in_out=ff_io, margin=ff_margin, end_t=seg_end, end_frame=-1, outcome=shot_outcome))
        if contact is None:
            break
        hitter, t0, p0 = key, t_c, fl.P[j].copy()
        kind = _choose_kind(rng, float(p0[2]), abs(float(p0[0])))
    for k in players:                                  # everyone relaxes after the rally
        movers[k].recover(end["t_floor"] + 0.4, bases[k], speed=1.0)
    return dict(shots=shots, segments=segments, hits=hits, end=end, server=server_side,
                t_serve=t_serve, p_serve=hits[0]["pos"])


# ── assembly: continuous truth → per-frame arrays ────────────────────────────

def _simulate(seed: int, camera: CameraModel | None, fps: float, n_rallies: int,
              n_shots: int | None, doubles: bool, dead_time: tuple[float, float]) -> SimRally:
    rng = np.random.default_rng([int(seed), 17])
    if camera is None:
        camera = random_camera(np.random.default_rng([int(seed), 99]), style="random")
    players = _new_players(rng, doubles)
    base_list = {"near": _bases(-1.0, doubles, rng), "far": _bases(1.0, doubles, rng)}
    bases = {}
    for side in ("near", "far"):
        for k, b in zip(_side_keys(players, side), base_list[side]):
            bases[k] = b
    movers = {k: _Mover(bases[k]) for k in players}
    t = float(rng.uniform(0.5, 1.2))
    server = str(rng.choice(["near", "far"]))
    rallies_raw, L = [], physics.DRAG_LENGTH
    for r in range(n_rallies):
        L = physics.DRAG_LENGTH * float(rng.uniform(0.92, 1.08))
        ral = _play_rally(rng, players, movers, bases, t, server, n_shots, doubles, L, r == 0)
        rallies_raw.append(ral)
        server = ral["end"]["winner"]
        if r < n_rallies - 1:
            t = ral["end"]["t_floor"] + float(rng.uniform(*dead_time))
    t_total = rallies_raw[-1]["end"]["t_floor"] + float(rng.uniform(0.6, 1.2))
    n = int(np.ceil(t_total * fps))
    ts = np.arange(n) / float(fps)

    def frame_of(tt: float) -> int:
        return int(np.clip(np.round(tt * fps), 0, n - 1))

    world = np.full((n, 3), np.nan)
    shots: list[SimShot] = []
    hits: list[dict] = []
    rallies: list[RallyTruth] = []
    resting: list[tuple[int, int, np.ndarray]] = []
    prev_rest_end = 0
    for r, ral in enumerate(rallies_raw):
        for (a, b, p0, v0, Ls) in ral["segments"]:
            m = (ts >= a) & (ts < b)
            if m.any():
                world[m] = simulate_positions(p0, v0, ts[m] - a, Ls)
        base = len(shots)
        for sh in ral["shots"]:
            sh.index = len(shots)
            sh.frame, sh.end_frame = frame_of(sh.t0), frame_of(sh.end_t)
            shots.append(sh)
        for h in ral["hits"]:
            h = dict(h, frame=frame_of(h["t"]), shot=h["shot"] + base)
            hits.append(h)
        e = ral["end"]
        f_serve, f_end, f_floor = frame_of(ral["t_serve"]), frame_of(e["t_end"]), frame_of(e["t_floor"])
        rallies.append(RallyTruth(
            start_frame=f_serve, end_frame=f_end, end_reason=e["reason"], winner=e["winner"],
            server=ral["server"], landing_xy=e["landing"], margin=e["margin"],
            first_shot=base, n_shots=len(ral["shots"])))
        # dead shuttle: held by the server before the serve, lying on the floor afterwards
        hold_from = max(prev_rest_end, f_serve - int(fps * rng.uniform(1.0, 2.0))) if r else 0
        hold_to = int(np.searchsorted(ts, ral["t_serve"])) - 1       # last frame before contact
        if hold_to >= hold_from:
            resting.append((hold_from, hold_to, ral["p_serve"].copy()))
        rest = np.array([e["rest"][0], e["rest"][1], 0.03])
        lie_to = n - 1 if r == n_rallies - 1 else min(n - 1, f_floor + int(fps * rng.uniform(1.0, 2.5)))
        if lie_to > f_floor:
            resting.append((f_floor + 1, lie_to, rest))
        prev_rest_end = lie_to + 1
    in_play = np.isfinite(world[:, 0])
    px = camera.project(np.nan_to_num(world))
    px[~in_play] = np.nan
    feet = {k: movers[k].positions(ts) for k in players}
    pdict = {}
    for k, p in players.items():
        bbox, depth = _player_boxes(camera, feet[k], p["height"])
        racket = _racket_track(ts, feet[k], p, [h for h in hits if h["player"] == k])
        pdict[k] = {"side": p["side"], "foot": feet[k], "bbox": bbox, "height": p["height"],
                    "racket": racket, "depth": depth, "racket_side": p["racket_side"]}
    w, h = camera.image_size
    with np.errstate(invalid="ignore"):
        inside = in_play & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    occluded = _occlusion(camera, world, px, inside, pdict)
    return SimRally(camera=camera, fps=float(fps), timestamps=ts, shuttle_world=world,
                    shuttle_px=px, visible=inside & ~occluded, in_play=in_play,
                    occluded=occluded, hits=hits, shots=shots, rallies=rallies,
                    players=pdict, doubles=doubles, drag_length=L, seed=int(seed),
                    resting=resting)


def _player_boxes(camera: CameraModel, foot: np.ndarray, height: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pixel bbox [x1,y1,x2,y2] of a (2·BODY_HALF_WIDTH)² × height box on the feet,
    and the camera depth of the body centre (for occlusion)."""
    n = len(foot)
    offs = np.array([[dx, dy, dz] for dx in (-1, 1) for dy in (-1, 1) for dz in (0, 1)], float)
    offs[:, :2] *= BODY_HALF_WIDTH
    offs[:, 2] *= height
    P = np.concatenate([foot, np.zeros((n, 1))], axis=1)[:, None, :] + offs[None]
    uv = camera.project(P.reshape(-1, 3)).reshape(n, 8, 2)
    bbox = np.concatenate([np.nanmin(uv, axis=1), np.nanmax(uv, axis=1)], axis=1)
    centre = np.concatenate([foot, np.full((n, 1), height * 0.55)], axis=1)
    depth = (centre @ camera.R.T + camera.t)[:, 2]
    return bbox, depth


def _racket_track(ts: np.ndarray, foot: np.ndarray, pl: dict, own_hits: list[dict]) -> np.ndarray:
    """Racket-head world position: carried beside the body, swung onto each contact."""
    s, rs = pl["sign"], pl["racket_side"]
    rest = np.column_stack([foot[:, 0] - s * 0.25, foot[:, 1] + rs * 0.45,
                            np.full(len(ts), 0.62 * pl["height"])])
    out = rest.copy()
    for h in own_hits:
        w = np.exp(-0.5 * ((ts - h["t"]) / 0.07) ** 2)[:, None]
        out = out * (1 - w) + np.asarray(h["pos"])[None] * w
    return out


def _occlusion(camera: CameraModel, world: np.ndarray, px: np.ndarray, inside: np.ndarray,
               players: dict) -> np.ndarray:
    """Shuttle behind a player's torso (central 55 % width, head→knee band)."""
    occ = np.zeros(len(world), bool)
    idx = np.nonzero(inside)[0]
    if not len(idx):
        return occ
    zc = (world[idx] @ camera.R.T + camera.t)[:, 2]
    for p in players.values():
        b = p["bbox"][idx]
        cx, bw, bh = (b[:, 0] + b[:, 2]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        u, v = px[idx, 0], px[idx, 1]
        hit = ((np.abs(u - cx) < 0.275 * bw) & (v > b[:, 1] + 0.08 * bh) & (v < b[:, 1] + 0.75 * bh)
               & (zc > p["depth"][idx] + 0.25))
        occ[idx[hit]] = True
    return occ


def simulate_rally(seed: int, camera: CameraModel | None = None, fps: float = 60,
                   n_shots: int | None = None, doubles: bool = False) -> SimRally:
    """One rally (serve → floor/net), with a short lead-in and tail.

    n_shots forces the stroke count (a shot the receiver cannot reach after
    retries still ends the rally early as a winner); None samples a realistic
    length (geometric, mean ≈ 8.5). camera=None → random_camera(style='random')
    from a seed-derived stream (the rally itself does not depend on the camera)."""
    return _simulate(seed, camera, fps, 1, n_shots, doubles, (3.0, 8.0))


def simulate_match(seed: int, n_rallies: int, camera: CameraModel | None = None,
                   fps: float = 60, doubles: bool = False, n_shots: int | None = None,
                   dead_time: tuple[float, float] = (3.0, 8.0)) -> SimRally:
    """n_rallies rallies separated by 3–8 s of dead time (shuttle lying / held /
    out of play, players walking back). Winner serves next. `rallies` holds
    per-rally truth; `shots`/`hits` are global and chronological."""
    return _simulate(seed, camera, fps, n_rallies, n_shots, doubles, dead_time)


# ── cameras ──────────────────────────────────────────────────────────────────

_CORNERS = np.array([[-cm.HALF_LENGTH, -cm.DOUBLES_HALF_WIDTH, 0], [-cm.HALF_LENGTH, cm.DOUBLES_HALF_WIDTH, 0],
                     [cm.HALF_LENGTH, -cm.DOUBLES_HALF_WIDTH, 0], [cm.HALF_LENGTH, cm.DOUBLES_HALF_WIDTH, 0]])


def _fit_focal(eye, target, image_size, roll, fill: float, lo: float, hi: float) -> CameraModel:
    """Focal so that the court's widest extent fills `fill` of the frame."""
    w, h = image_size
    cam = CameraModel.look_at(eye, target, float(w), image_size, roll)
    uv = cam.project(_CORNERS)
    ext = max(np.nanmax(np.abs(uv[:, 0] - w / 2)) / (w / 2), np.nanmax(np.abs(uv[:, 1] - h / 2)) / (h / 2))
    f = float(np.clip(w * fill / ext, lo * w, hi * w))
    return CameraModel.look_at(eye, target, f, image_size, roll)


def random_camera(rng: np.random.Generator | int | None = None, image_size: tuple[int, int] = (1920, 1080),
                  style: str = "random") -> CameraModel:
    """Plausible capture setups (all behind/beside the NEAR half, x < 0).

    behind        phone on a tripod 1–6 m behind the near baseline, 1.0–2.0 m high,
                  26 mm-equivalent lens (focal 0.8–1.0 × width), yaw ±5°, roll ±3°;
                  pitched so the far baseline sits 28–50 % down the frame (near
                  corners and high near-court flights may be cut off, as with real phones).
    elevated_side 3–6 m high at the side/corner of the near half, whole court in view.
    broadcast     like the TrackNetV2 / BWF broadcast clips (measured by calibrating the
                  3 real Test matches: eye 16–24 m behind the near baseline, 6–10 m up,
                  telephoto focal ≈ 2–2.6 × width, pitch ≈ 14–16°; near baseline at
                  89–97 % of the height spanning 54–72 % of the width, far baseline
                  ≈ 35–42 % down, whole court in frame).
    """
    rng = np.random.default_rng(rng)
    if style == "random":
        style = str(rng.choice(["behind", "elevated_side", "broadcast"], p=[0.5, 0.25, 0.25]))
    w, h = image_size
    if style == "behind":
        d, z = rng.uniform(1.0, 6.0), rng.uniform(1.0, 2.0)
        eye = np.array([-cm.HALF_LENGTH - d, rng.uniform(-1.2, 1.2), z])
        f = w * rng.uniform(0.8, 1.0)
        roll = rng.uniform(-3, 3)
        frac = rng.uniform(0.28, 0.5)              # far baseline height in the frame
        lo, hi = eye[0] + 0.5, 12.0
        for _ in range(40):                        # bisection on the look-at x
            mid = 0.5 * (lo + hi)
            cam = CameraModel.look_at(eye, [mid, 0.0, 0.0], f, image_size, roll)
            v = cam.project([[cm.HALF_LENGTH, 0.0, 0.0]])[0, 1]
            if not np.isfinite(v) or v > frac * h:
                hi = mid                           # far line too low → look further up
            else:
                lo = mid
        yaw = np.radians(rng.uniform(-5, 5))
        tgt = np.array([mid, 0.0, 0.0]) - eye
        c, s = np.cos(yaw), np.sin(yaw)
        tgt = eye + np.array([c * tgt[0] - s * tgt[1], s * tgt[0] + c * tgt[1], tgt[2]])
        return CameraModel.look_at(eye, tgt, f, image_size, roll)
    if style == "elevated_side":
        side = rng.choice([-1.0, 1.0])
        eye = np.array([rng.uniform(-11.0, -4.0), side * rng.uniform(8.0, 13.0), rng.uniform(3.5, 7.0)])
        tgt = np.array([rng.uniform(-1.0, 1.0), side * rng.uniform(-0.3, 0.6), 0.0])
        return _fit_focal(eye, tgt, image_size, rng.uniform(-2, 2), rng.uniform(0.82, 0.94), 0.4, 1.6)
    if style == "broadcast":
        for _ in range(20):
            eye = np.array([-cm.HALF_LENGTH - rng.uniform(16.0, 24.0),
                            rng.uniform(-1.0, 1.0) if rng.random() < 0.8 else rng.uniform(-3.0, 3.0),
                            rng.uniform(6.0, 10.0)])
            v_near, width_frac = rng.uniform(0.89, 0.97), rng.uniform(0.54, 0.72)
            ty, roll = rng.uniform(-0.3, 0.3), rng.uniform(-0.8, 0.8)
            f = 2.5 * w
            for _ in range(5):                     # alternate pitch (near line row) ↔ focal (width)
                lo, hi = -cm.HALF_LENGTH, 30.0
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    cam = CameraModel.look_at(eye, [mid, ty, 0.0], f, image_size, roll)
                    v = cam.project([[-cm.HALF_LENGTH, 0.0, 0.0]])[0, 1]
                    lo, hi = (lo, mid) if v > v_near * h else (mid, hi)   # look nearer ↔ line rises
                uv = cam.project(_CORNERS)
                f *= width_frac * w / abs(uv[1, 0] - uv[0, 0])
            uv = cam.project(_CORNERS)
            if np.all((uv >= [0, 0]) & (uv < [w, h])) and uv[2:, 1].min() > 0.12 * h:
                return cam
        return cam
    raise ValueError(f"unknown camera style {style!r}")


# ── converters: truth → pipeline data types ─────────────────────────────────

def truth_track(sim: SimRally) -> Track2D:
    """Perfect 2-D track (visible frames only)."""
    xy = np.where(sim.visible[:, None], sim.shuttle_px, np.nan)
    return Track2D(t=sim.timestamps.copy(), xy=xy, conf=sim.visible.astype(float),
                   visible=sim.visible.copy(), interpolated=np.zeros(sim.n, bool))


def truth_hits(sim: SimRally) -> list[HitEvent]:
    out = []
    for h in sim.hits:
        uv = sim.camera.project(h["pos"])[0]
        out.append(HitEvent(frame=int(h["frame"]), t=float(h["t"]), xy_px=(float(uv[0]), float(uv[1])),
                            hitter=h["hitter"], confidence=1.0,
                            player_track_id=_PLAYER_IDS.get(h["player"])))
    return out


_PLAYER_IDS = {"near": 1, "far": 2, "near2": 3, "far2": 4}


def truth_players(sim: SimRally, every: int = 1) -> list[PlayerObservation]:
    """SimRally → per-frame PlayerObservation (bbox, side, foot_world); frames with
    the player completely outside the image are skipped."""
    w, h = sim.image_size
    out = []
    for i in range(0, sim.n, every):
        for k, p in sim.players.items():
            b = p["bbox"][i]
            if not np.all(np.isfinite(b)) or b[2] < 0 or b[0] >= w or b[3] < 0 or b[1] >= h:
                continue
            out.append(PlayerObservation(frame=i, track_id=_PLAYER_IDS[k], bbox=b.astype(float).copy(),
                                         score=1.0, side=p["side"], foot_world=p["foot"][i].copy()))
    return out


def truth_segments(sim: SimRally) -> list[ShotSegment3D]:
    """True flights (hit → next contact / floor / net) as ShotSegment3D."""
    out = []
    for sh in sim.shots:
        fr = np.arange(sh.frame, sh.end_frame + 1)
        fr = fr[(sim.timestamps[fr] >= sh.t0) & (sim.timestamps[fr] <= sh.end_t)]
        pos = simulate_positions(sh.p0, sh.v0, sim.timestamps[fr] - sh.t0, sh.drag_length)
        lands = sh.outcome in ("in", "out")
        out.append(ShotSegment3D(
            start_frame=sh.frame, end_frame=sh.end_frame, t0=sh.t0, p0=sh.p0.copy(), v0=sh.v0.copy(),
            drag_length=sh.drag_length, rmse_px=0.0, frames=fr, positions=pos,
            landing_xy=tuple(sh.landing_xy) if lands else None,
            landing_t=sh.landing_t if lands else None,
            net_crossing_height=sh.net_crossing_height if np.isfinite(sh.net_crossing_height) else None,
            confidence=1.0))
    return out


def truth_rallies(sim: SimRally, with_shots: bool = True) -> list:
    """RallyV2 list mirroring the ground truth (shot types filled)."""
    from app.services.tracking.data_types import RallyV2, ShotEventV2
    hits, segs = truth_hits(sim), truth_segments(sim)
    out = []
    for r in sim.rallies:
        shots = []
        if with_shots:
            for k in range(r.first_shot, r.first_shot + r.n_shots):
                sh = sim.shots[k]
                shots.append(ShotEventV2(hit=hits[k], segment=segs[k], shot_type=sh.shot_type,
                                         shot_type_confidence=1.0,
                                         hitter_pos_world=tuple(sim.players[sh.player]["foot"][sh.frame]),
                                         hit_height_m=float(sh.p0[2])))
        out.append(RallyV2(start_frame=r.start_frame, end_frame=r.end_frame, shots=shots,
                           end_reason=r.end_reason, landing_xy=r.landing_xy,
                           in_out_margin_m=r.margin, winner=r.winner, confidence=1.0))
    return out


# ── detector noise model ─────────────────────────────────────────────────────

def _markov_on(rng: np.random.Generator, n: int, p_on: float, mean_on: float) -> np.ndarray:
    """Bursty on/off sequence with stationary on-probability p_on."""
    p_on = float(np.clip(p_on, 0.0, 0.95))
    if p_on <= 0:
        return np.zeros(n, bool)
    q_off = 1.0 / max(mean_on, 1.0)                 # on → off
    q_on = q_off * p_on / (1 - p_on)                # off → on
    out = np.zeros(n, bool)
    state = rng.random() < p_on
    u = rng.random(n)
    for i in range(n):
        out[i] = state
        state = (u[i] >= q_off) if state else (u[i] < q_on)
    return out


def corrupt_px(px: np.ndarray, visible: np.ndarray, image_size: tuple[int, int],
               rng: np.random.Generator | int | None = None, sigma_px: float = 1.5,
               miss_rate: float = 0.1, burst_miss: bool = True,
               false_positive_rate: float = 0.03, static_uv: list[np.ndarray] | None = None,
               moving_uv: list[np.ndarray] | None = None,
               dead: list[tuple[int, int, np.ndarray]] | None = None,
               dead_shuttle_rate: float = 0.5, max_candidates: int = 3
               ) -> list[list[ShuttleCandidate]]:
    """Camera-free core of corrupt_detections: true pixel track (N,2) + visible
    mask → per-frame candidates. static_uv: fixed distractor pixels (random
    court-ish spots if None); moving_uv: (N,2) tracks of moving distractors
    (rackets/shoes; synthetic random walks if None); dead: (from, to, uv) spans
    where a dead shuttle lies. Also used on REAL labelled tracks (benchmark
    'real_oracle')."""
    rng = np.random.default_rng(rng)
    px = np.asarray(px, float)
    vis = np.asarray(visible, bool) & np.all(np.isfinite(px), axis=1)
    n, (w, h) = len(px), image_size
    scale = float(np.hypot(w, h) / np.hypot(1280, 720))
    step = np.zeros(n)
    d = np.linalg.norm(np.diff(px, axis=0), axis=1)
    step[1:] = np.nan_to_num(d, nan=0.0)
    blur = np.clip(step / (40.0 * scale), 0.0, 1.5)          # ~ streak length / blob size
    wgt = 0.5 + blur
    if vis.any():
        wgt = wgt / wgt[vis].mean()
    p_miss = np.clip(miss_rate * wgt, 0.0, 0.95)
    miss = np.zeros(n, bool)
    if burst_miss and miss_rate > 0:
        miss |= rng.random(n) < 0.5 * p_miss
        for s in np.nonzero(vis & (rng.random(n) < 0.5 * p_miss / 9.0))[0]:
            miss[s:s + int(rng.integers(3, 16))] = True
    else:
        miss |= rng.random(n) < p_miss
    cands: list[list[ShuttleCandidate]] = [[] for _ in range(n)]
    for i in np.nonzero(vis & ~miss)[0]:
        sig = sigma_px * scale ** 0.5 * (1 + 0.3 * blur[i]) * (4.0 if rng.random() < 0.03 else 1.0)
        x, y = px[i] + rng.normal(0, sig, 2)
        sc = float(np.clip(rng.normal(0.9 - 0.22 * blur[i], 0.08), 0.12, 0.999))
        cands[i].append(ShuttleCandidate(float(x), float(y), sc))

    def emit(mask: np.ndarray, uv: np.ndarray, lo: float, hi: float, jitter: float) -> None:
        for i in np.nonzero(mask)[0]:
            if np.all(np.isfinite(uv[i])) and 0 <= uv[i, 0] < w and 0 <= uv[i, 1] < h:
                x, y = uv[i] + rng.normal(0, jitter, 2)
                cands[i].append(ShuttleCandidate(float(x), float(y), float(rng.uniform(lo, hi))))

    fp = max(float(false_positive_rate), 0.0)
    if fp > 0:
        if static_uv is None:
            static_uv = [np.array([rng.uniform(0.1, 0.9) * w, rng.uniform(0.3, 0.95) * h])
                         if rng.random() < 0.7 else np.array([rng.uniform(0, w), rng.uniform(0, 0.3 * h)])
                         for _ in range(int(rng.integers(2, 7)))]
        for uv in static_uv:
            emit(_markov_on(rng, n, 0.5 * fp / len(static_uv), 12.0), np.tile(uv, (n, 1)),
                 0.25, 0.8, 0.6)
        if moving_uv is None:                         # two random walkers (players' rackets)
            moving_uv = []
            for k in range(2):
                v0 = np.array([rng.uniform(0.2, 0.8) * w, (0.35 + 0.45 * k) * h])
                walk = np.cumsum(rng.normal(0, 0.004 * w, (n, 2)), axis=0)
                moving_uv.append(np.clip(v0 + walk - walk.mean(0), 0, [w - 1, h - 1]))
        for uv in moving_uv:
            emit(_markov_on(rng, n, 0.35 * fp / len(moving_uv), 4.0), uv, 0.15, 0.7, 2.0)
        emit(rng.random(n) < 0.15 * fp, np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)]),
             0.1, 0.5, 0.0)
    if dead_shuttle_rate > 0:
        for a, b, uv in dead or []:
            m = np.zeros(n, bool)
            m[a:b + 1] = rng.random(b + 1 - a) < dead_shuttle_rate
            emit(m, np.tile(np.asarray(uv, float), (n, 1)), 0.35, 0.9, 0.8)
    for i in range(n):
        if len(cands[i]) > 1:
            cands[i].sort(key=lambda c: -c.score)
        del cands[i][max_candidates:]
    return cands


def corrupt_detections(sim: SimRally, rng: np.random.Generator | int | None = None,
                       sigma_px: float = 1.5, miss_rate: float = 0.1, burst_miss: bool = True,
                       false_positive_rate: float = 0.03, dead_shuttle_rate: float = 0.5,
                       max_candidates: int = 3) -> list[list[ShuttleCandidate]]:
    """Per-frame detector output like a heatmap detector's top-k peaks.

    True shuttle (visible frames): Gaussian noise sigma_px (×√(diag/1280×720
    diag), inflated with motion blur, 3 % heavy-tailed outliers), score ≈ 0.9
    falling with blur (some true peaks drop below the 0.5 threshold). Misses ≈
    miss_rate of visible frames, more likely when fast; with burst_miss half
    of them come as 3–15 frame bursts. False positives, false_positive_rate
    expected per frame: 50 % static bursty spots (line points, lights), 35 %
    moving spots (rackets, shoes), 15 % uniform clutter. The dead shuttle (held
    before a serve, lying after a rally) is detected with probability
    dead_shuttle_rate. Sorted by score, ≤ max_candidates per frame."""
    rng = np.random.default_rng(rng)
    n, (w, h) = sim.n, sim.image_size
    static = []
    for _ in range(int(rng.integers(2, 7))):
        if rng.random() < 0.7:                      # somewhere on a painted line
            a, b = list(cm.COURT_LINES.values())[int(rng.integers(len(cm.COURT_LINES)))]
            f = rng.uniform()
            static.append(sim.camera.project([[a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), 0.0]])[0])
        else:                                       # light / wall fixture
            static.append(np.array([rng.uniform(0, w), rng.uniform(0, 0.3 * h)]))
    moving = []
    for p in sim.players.values():
        moving.append(sim.camera.project(p["racket"]))
        moving.append(sim.camera.project(np.column_stack([p["foot"], np.full(n, 0.05)])))
    dead = [(a, b, sim.camera.project(p)[0]) for a, b, p in sim.resting]
    return corrupt_px(sim.shuttle_px, sim.visible, (w, h), rng, sigma_px, miss_rate, burst_miss,
                      false_positive_rate, static, moving, dead, dead_shuttle_rate, max_candidates)


# ── video renderer ───────────────────────────────────────────────────────────

_SUB = 16                                   # cv2 fixed-point subpixel shift (4 bits)


def _to_cam(camera: CameraModel, P: np.ndarray) -> np.ndarray:
    return np.asarray(P, float) @ camera.R.T + camera.t


def _cam_to_px(camera: CameraModel, C: np.ndarray) -> np.ndarray:
    xy = C[:, :2] / C[:, 2:3]
    if np.any(camera.dist):
        from app.services.tracking.camera import _distort
        xy = _distort(xy, camera.dist)
    return xy @ camera.K[:2, :2].T + camera.K[:2, 2]


def _clip_poly_px(camera: CameraModel, P: np.ndarray, near: float = 0.3) -> np.ndarray | None:
    """World polygon → pixel polygon, clipped against the camera's near plane."""
    C = _to_cam(camera, P)
    out = []
    for i in range(len(C)):
        a, b = C[i], C[(i + 1) % len(C)]
        if a[2] > near:
            out.append(a)
        if (a[2] > near) != (b[2] > near):
            out.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
    if len(out) < 3:
        return None
    return _cam_to_px(camera, np.array(out))


def _fill(img: np.ndarray, polys: list[np.ndarray], color) -> None:
    pts = [np.round(np.clip(p, -1e5, 1e5) * _SUB).astype(np.int32) for p in polys if p is not None]
    if pts:
        cv2.fillPoly(img, pts, color, lineType=cv2.LINE_AA, shift=4)


def _hall(rng: np.random.Generator, camera: CameraModel) -> dict:
    """Random sports-hall palette and geometry (box around the court and camera)."""
    c = camera.center
    court = [(62, 118, 52), (58, 104, 44), (128, 86, 40), (96, 70, 44)][int(rng.integers(4))]
    return {
        "court": np.array(court, float) * rng.uniform(0.85, 1.15),
        "surround": np.array([(46, 70, 44), (80, 64, 50), (58, 58, 62)][int(rng.integers(3))], float)
        * rng.uniform(0.8, 1.2),
        "wall": np.array([(70, 60, 55), (95, 80, 70), (48, 46, 52)][int(rng.integers(3))], float),
        "ceiling": np.array([35, 35, 38], float) * rng.uniform(0.7, 1.3),
        "x_far": cm.HALF_LENGTH + rng.uniform(3.5, 8.0),
        "x_near": min(c[0] - 2.0, -cm.HALF_LENGTH - 4.0),
        "y_half": max(abs(c[1]) + 2.0, cm.DOUBLES_HALF_WIDTH + rng.uniform(3.0, 7.0)),
        "z_ceil": max(c[2] + 2.5, rng.uniform(8.0, 12.0)),
        "surround_m": rng.uniform(1.5, 2.5),
    }


def _render_static(camera: CameraModel, hall: dict, rng: np.random.Generator,
                   size: tuple[int, int]) -> np.ndarray:
    """Per-pixel ray cast of floor / walls / ceiling (half resolution, upscaled —
    all low-frequency) + full-resolution anti-aliased court lines."""
    W, Hf = size
    w, h = (W + 1) // 2, (Hf + 1) // 2
    lines_cam = camera
    camera = CameraModel(K=np.diag([w / W, h / Hf, 1.0]) @ camera.K, R=camera.R, t=camera.t,
                         image_size=(w, h), dist=camera.dist)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    if np.any(camera.dist):
        o, d = camera.ray(np.column_stack([u.ravel(), v.ravel()]))
        d = d.reshape(h, w, 3).astype(np.float32)
    else:
        Ki = np.linalg.inv(camera.K).astype(np.float32)
        xy = np.stack([u * Ki[0, 0] + Ki[0, 2], v * Ki[1, 1] + Ki[1, 2], np.ones_like(u)], axis=-1)
        d = xy @ camera.R.astype(np.float32)             # rows of R = camera axes in world
        o = camera.center
    big = 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = np.where(d[..., 2] < -1e-9, -o[2] / d[..., 2], big)
        s_ceil = np.where(d[..., 2] > 1e-9, (hall["z_ceil"] - o[2]) / d[..., 2], big)
        s_xf = np.where(d[..., 0] > 1e-9, (hall["x_far"] - o[0]) / d[..., 0], big)
        s_xn = np.where(d[..., 0] < -1e-9, (hall["x_near"] - o[0]) / d[..., 0], big)
        s_yl = np.where(d[..., 1] > 1e-9, (hall["y_half"] - o[1]) / d[..., 1], big)
        s_yr = np.where(d[..., 1] < -1e-9, (-hall["y_half"] - o[1]) / d[..., 1], big)
    S_ = np.stack([s_floor, s_ceil, s_xf, s_xn, s_yl, s_yr])
    which = np.argmin(S_, axis=0)
    s = np.min(S_, axis=0)
    P = o + d * s[..., None]
    img = np.zeros((h, w, 3), np.float32)
    # floor: court mat + surround with perspective-correct low-frequency texture
    tex = cv2.resize(rng.normal(0, 1, (64, 64)).astype(np.float32), (512, 512),
                     interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (512, 512)).astype(np.float32)
    mx = ((P[..., 0] + 20.0) / 40.0 * 511).astype(np.float32)
    my = ((P[..., 1] + 20.0) / 40.0 * 511).astype(np.float32)
    t1 = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    t2 = cv2.remap(fine, mx * 8 % 511, my * 8 % 511, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fl = which == 0
    m = hall["surround_m"]
    on_court = (np.abs(P[..., 0]) < cm.HALF_LENGTH + m) & (np.abs(P[..., 1]) < cm.DOUBLES_HALF_WIDTH + m)
    col = np.where(on_court[..., None], hall["court"], hall["surround"])
    shade = 1.0 + 0.05 * t1 + 0.025 * t2
    img[fl] = (col * shade[..., None])[fl]
    # walls: dark with a band of advertising boards (bright rectangles = realistic FPs)
    wl = (which >= 2)
    along = np.where((which == 2) | (which == 3), P[..., 1], P[..., 0])
    board = wl & (P[..., 2] < 1.0) & (P[..., 2] > 0.05)
    stripes = (np.floor(along / rng.uniform(2.5, 5.0)) % 2).astype(np.float32)
    img[wl] = (hall["wall"] * (0.85 + 0.15 * (P[..., 2] / hall["z_ceil"]))[..., None])[wl]
    bc = np.array([rng.uniform(80, 220), rng.uniform(80, 220), rng.uniform(80, 220)], np.float32)
    img[board] = (bc * (0.6 + 0.4 * stripes[..., None]))[board]
    # ceiling with rows of lights
    ce = which == 1
    lights = ce & (np.abs(((P[..., 0] + 100) % 6.0) - 3.0) < 0.35) & (np.abs(((P[..., 1] + 100) % 5.0) - 2.5) < 0.6)
    img[ce] = hall["ceiling"]
    img[lights] = (250, 250, 245)
    img = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, Hf), interpolation=cv2.INTER_LINEAR)
    camera = lines_cam
    # court lines: 40 mm polygons around each centre-line, projected (true perspective width)
    lw = cm.LINE_WIDTH / 2
    polys = []
    for (x1, y1), (x2, y2) in cm.COURT_LINES.values():
        a, b = np.array([x1, y1]), np.array([x2, y2])
        t = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-t[1], t[0]])
        a, b = a - t * lw, b + t * lw
        q = np.array([a + nrm * lw, b + nrm * lw, b - nrm * lw, a - nrm * lw])
        polys.append(_clip_poly_px(camera, np.column_stack([q, np.zeros(4)])))
    _fill(img, polys, (235, 235, 232))
    return img


def _net_layer(camera: CameraModel, size: tuple[int, int]) -> tuple[tuple[slice, slice], np.ndarray, np.ndarray] | None:
    """Net overlay: ROI, alpha (h,w,1) float32 and colour image (mesh, tape, posts)."""
    w, h = size
    W, ys = cm.DOUBLES_HALF_WIDTH, np.linspace(-cm.DOUBLES_HALF_WIDTH, cm.DOUBLES_HALF_WIDTH, 25)
    top = cm.NET_HEIGHT_CENTRE + (cm.NET_HEIGHT_POST - cm.NET_HEIGHT_CENTRE) * (ys / W) ** 2
    mesh = np.concatenate([np.column_stack([np.zeros_like(ys), ys, top - 0.075]),
                           np.column_stack([np.zeros_like(ys), ys[::-1], np.full_like(ys, 0.79)])])
    tape = np.concatenate([np.column_stack([np.zeros_like(ys), ys, top]),
                           np.column_stack([np.zeros_like(ys), ys[::-1], (top - 0.075)[::-1]])])
    pm, pt = _clip_poly_px(camera, mesh), _clip_poly_px(camera, tape)
    if pm is None:
        return None
    alpha = np.zeros((h, w), np.float32)
    color = np.zeros((h, w, 3), np.uint8)
    _fill(alpha, [pm], 0.33)
    _fill(color, [pm], (30, 30, 34))
    _fill(alpha, [pt], 1.0)
    _fill(color, [pt], (240, 240, 240))
    for py in (-W, W):                         # posts
        uv = camera.project(np.array([[0.0, py, 0.0], [0.0, py, cm.NET_HEIGHT_POST]]))
        if np.all(np.isfinite(uv)):
            depth = _to_cam(camera, [[0.0, py, 0.8]])[0, 2]
            th = max(2, int(round(0.04 * camera.K[0, 0] / max(depth, 0.5))))
            p1, p2 = tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int))
            cv2.line(alpha, p1, p2, 1.0, th, cv2.LINE_AA)
            cv2.line(color, p1, p2, (40, 40, 200), th, cv2.LINE_AA)
    ys_, xs_ = np.nonzero(alpha > 0.01)
    if not len(xs_):
        return None
    roi = (slice(max(ys_.min() - 1, 0), min(ys_.max() + 2, h)), slice(max(xs_.min() - 1, 0), min(xs_.max() + 2, w)))
    return roi, alpha[roi][..., None].copy(), color[roi].astype(np.float32)


def _skeletons(sim: SimRally, key: str, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World joints (N, 14, 3) for a player: ankles, knees, hips, shoulders, head,
    racket-arm elbow/hand, racket head, other elbow/hand; and head radius (m)."""
    p = sim.players[key]
    foot, H, n = p["foot"], p["height"], len(ts)
    s = -1.0 if p["side"] == "near" else 1.0
    fwd = np.array([-s, 0.0])                                  # faces the net
    lat = np.array([0.0, 1.0])
    rs = p.get("racket_side", s)
    step = np.r_[0.0, np.linalg.norm(np.diff(foot, axis=0), axis=1)]
    phase = np.cumsum(step) / 0.8 * 2 * np.pi
    speed = np.r_[0.0, step[1:] * sim.fps]
    amp = 0.22 * np.clip(speed / 2.0, 0, 1)
    J = np.zeros((n, 14, 3))

    def put(k, xy, z):
        J[:, k, :2], J[:, k, 2] = xy, z

    sw = np.sin(phase)[:, None]
    put(0, foot + lat * 0.13 + fwd * amp[:, None] * sw, 0.05)
    put(1, foot - lat * 0.13 - fwd * amp[:, None] * sw, 0.05)
    put(2, foot + lat * 0.12 + fwd * 0.5 * amp[:, None] * sw + fwd * 0.05, 0.28 * H)
    put(3, foot - lat * 0.12 - fwd * 0.5 * amp[:, None] * sw + fwd * 0.05, 0.28 * H)
    put(4, foot + lat * 0.10, 0.53 * H)
    put(5, foot - lat * 0.10, 0.53 * H)
    put(6, foot + lat * 0.19, 0.82 * H)
    put(7, foot - lat * 0.19, 0.82 * H)
    put(8, foot, 0.93 * H)
    sh = J[:, 6] if rs > 0 else J[:, 7]
    other = J[:, 7] if rs > 0 else J[:, 6]
    head_r = np.asarray(p["racket"], float)
    vec = head_r - sh
    dist = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)
    u = vec / dist
    reach = np.clip(dist - 0.62, 0.25, 0.62)
    hand = sh + u * reach
    J[:, 10] = hand
    J[:, 9] = 0.5 * (sh + hand) + np.array([0, 0, -0.06])
    J[:, 11] = np.where(dist > 1.3, hand + u * 0.66, head_r)
    J[:, 13] = other + np.array([0, 0, -0.55])
    J[:, 13, :2] += fwd * 0.12
    J[:, 12] = 0.5 * (other + J[:, 13]) + np.array([0, 0, 0.02])
    J[:, 12, :2] += fwd * 0.05
    return J, 0.068 * H


_BONES = [(0, 2, "leg"), (1, 3, "leg"), (2, 4, "leg"), (3, 5, "leg"), (6, 9, "arm"),
          (9, 10, "arm"), (7, 12, "arm"), (12, 13, "arm")]
_WIDTH_M = {"leg": 0.12, "arm": 0.075}


def _draw_player(img: np.ndarray, uv: np.ndarray, depth: np.ndarray, f: float, head_r: float,
                 colors: dict) -> None:
    if not np.all(np.isfinite(uv)) or np.any(depth < 0.3):
        return
    px = lambda k: tuple(int(v) for v in np.round(uv[k] * _SUB))      # noqa: E731
    sc = f / depth
    for a, b, kind in _BONES:
        th = max(1, int(round(_WIDTH_M[kind] * 0.5 * (sc[a] + sc[b]))))
        c = colors["shorts"] if kind == "leg" else colors["skin"]
        cv2.line(img, px(a), px(b), c, th, cv2.LINE_AA, 4)
    torso = np.round(uv[[6, 7, 5, 4]] * _SUB).astype(np.int32)
    cv2.fillConvexPoly(img, torso, colors["shirt"], cv2.LINE_AA, 4)
    th = max(1, int(round(0.2 * sc[6])))
    cv2.line(img, px(6), px(7), colors["shirt"], th, cv2.LINE_AA, 4)
    cv2.circle(img, px(8), max(1, int(round(head_r * sc[8] * _SUB))), colors["skin"], -1, cv2.LINE_AA, 4)
    cv2.line(img, px(10), px(11), (30, 30, 30), max(1, int(round(0.015 * sc[11]))), cv2.LINE_AA, 4)
    r = max(1, int(round(0.11 * sc[11] * _SUB)))
    cv2.circle(img, px(11), r, (200, 200, 200), max(1, int(round(0.012 * sc[11]))), cv2.LINE_AA, 4)


def _draw_shuttle(img: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> None:
    """White blob of radius r px, smeared from a to b (motion blur ∝ displacement)."""
    L = float(np.hypot(*(b - a)))
    h, w = img.shape[:2]
    pad = int(np.ceil(r)) + 3
    x0, y0 = int(np.floor(min(a[0], b[0]))) - pad, int(np.floor(min(a[1], b[1]))) - pad
    x1, y1 = int(np.ceil(max(a[0], b[0]))) + pad, int(np.ceil(max(a[1], b[1]))) + pad
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    pa = (int(round((a[0] - x0) * _SUB)), int(round((a[1] - y0) * _SUB)))
    pb = (int(round((b[0] - x0) * _SUB)), int(round((b[1] - y0) * _SUB)))
    th = max(1, int(round(2 * r)))
    if L < 0.5:
        cv2.circle(m, pb, max(int(round(r * _SUB)), 8), 1.0, -1, cv2.LINE_AA, 4)
    else:
        cv2.line(m, pa, pb, 1.0, th, cv2.LINE_AA, 4)
    gain = float(np.clip((2 * r + 1.0) / (2 * r + 1.0 + 0.5 * L), 0.3, 1.0))
    m = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None] * gain
    roi = img[Y0:Y1, X0:X1].astype(np.float32)
    img[Y0:Y1, X0:X1] = (roi * (1 - m) + 248.0 * m).astype(np.uint8)


def render_video(sim: SimRally, path: str, image_size: tuple[int, int] | None = None,
                 draw_players: bool = True, motion_blur: bool = True, seed: int | None = None,
                 exposure: float = 0.5, noise_sigma: float = 3.0) -> str:
    """Render the rally/match to an .mp4 (OpenCV 'mp4v', sim.fps, sim.n frames).

    Static hall ray-cast once per video (court mat + surround with perspective
    texture, walls with ad boards, ceiling lights, 40 mm lines as projected
    anti-aliased polygons, net mesh + 75 mm tape + posts); per frame, in depth
    order: players (articulated silhouettes: legs, torso, head, racket arm,
    racket) and the shuttle (≈6.5 cm → radius 0.033·f/depth px, smeared over
    `exposure` of the frame interval when motion_blur), far-side objects drawn
    behind the net; then sensor noise and ±2 % exposure flicker. image_size ≠
    camera size rescales K (truth px then scale by the same factor)."""
    rng = np.random.default_rng(sim.seed if seed is None else seed)
    cam = sim.camera
    size = tuple(int(v) for v in (image_size or cam.image_size))
    if size != tuple(cam.image_size):
        sx, sy = size[0] / cam.image_size[0], size[1] / cam.image_size[1]
        K = np.diag([sx, sy, 1.0]) @ cam.K
        cam = CameraModel(K=K, R=cam.R, t=cam.t, image_size=size, dist=cam.dist)
    w, h = size
    hall = _hall(rng, cam)
    static = _render_static(cam, hall, rng, size)
    net = _net_layer(cam, size)
    f = float(cam.K[0, 0])
    ts = sim.timestamps
    players = []
    if draw_players:
        for k in sim.players:
            J, head_r = _skeletons(sim, k, ts)
            C = _to_cam(cam, J.reshape(-1, 3)).reshape(J.shape)
            uv = _cam_to_px(cam, C.reshape(-1, 3)).reshape(len(ts), J.shape[1], 2)
            dark = rng.uniform(25, 90, 3)
            colors = {"shirt": tuple(float(v) for v in dark * rng.uniform(0.6, 1.4, 3)),
                      "shorts": tuple(float(v) for v in rng.uniform(15, 45, 3)),
                      "skin": tuple(float(v) for v in np.array([95, 125, 170]) * rng.uniform(0.7, 1.1))}
            players.append((k, uv, C[..., 2], head_r, colors, sim.players[k]["foot"][:, 0]))
    world = sim.shuttle_world
    rest = np.full((len(ts), 3), np.nan)
    for a, b, p in sim.resting:
        rest[a:b + 1] = p
    bases, bases_net = [], []                     # static + sensor noise (± net)
    for _ in range(3):
        g = rng.standard_normal((h, w, 3), dtype=np.float32) * noise_sigma
        b = np.clip(static.astype(np.float32) + g, 0, 255).astype(np.uint8)
        bases.append(b)
        bn = b.copy()
        if net is not None:
            roi, al, col = net
            bn[roi] = (b[roi].astype(np.float32) * (1 - al) + col * al).astype(np.uint8)
        bases_net.append(bn)
    ramp = np.arange(256, dtype=np.float32)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(sim.fps), size)
    if not vw.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    flicker = 1.0 + 0.02 * np.sin(2 * np.pi * ts * rng.uniform(0.3, 1.2) + rng.uniform(0, 6.3)) \
        + rng.normal(0, 0.004, len(ts))
    try:
        for i in range(len(ts)):
            j = int(rng.integers(3))
            img = bases_net[j].copy()
            items = []                             # (depth, far_side, draw fn, bbox)
            for k, uv, depth, head_r, colors, fx in players:
                q, dm = uv[i], float(np.nanmean(depth[i]))
                pad = 0.35 * f / max(dm, 0.3) + 4
                items.append((dm, fx[i] > 0,
                              lambda uv=uv, d=depth, hr=head_r, c=colors: _draw_player(img, uv[i], d[i], f, hr, c),
                              (np.nanmin(q[:, 0]) - pad, np.nanmin(q[:, 1]) - pad,
                               np.nanmax(q[:, 0]) + pad, np.nanmax(q[:, 1]) + pad)))
            p = world[i] if np.isfinite(world[i, 0]) else rest[i]
            if np.isfinite(p[0]):
                c1 = _to_cam(cam, p[None])[0]
                if c1[2] > 0.3:
                    prev = world[i - 1] if (i > 0 and np.isfinite(world[i - 1, 0]) and np.isfinite(world[i, 0])) else p
                    p0 = p + (prev - p) * exposure if motion_blur else p
                    c0 = _to_cam(cam, p0[None])[0]
                    b_ = _cam_to_px(cam, c1[None])[0]
                    a_ = _cam_to_px(cam, c0[None])[0] if c0[2] > 0.3 else b_
                    r = max(0.033 * f / c1[2], 0.7)
                    pad = r + 4
                    items.append((float(c1[2]), p[0] > 0,
                                  lambda a_=a_, b_=b_, r=r: _draw_shuttle(img, a_, b_, r),
                                  (min(a_[0], b_[0]) - pad, min(a_[1], b_[1]) - pad,
                                   max(a_[0], b_[0]) + pad, max(a_[1], b_[1]) + pad)))
            items.sort(key=lambda it: -it[0])
            far_items = [it for it in items if it[1]]
            box = None
            if far_items and net is not None:
                (ry, rx), al, col = net
                bb = np.array([it[3] for it in far_items], float)
                bb = bb[np.all(np.isfinite(bb), axis=1)]
                if len(bb):
                    X0, Y0 = int(max(bb[:, 0].min(), rx.start)), int(max(bb[:, 1].min(), ry.start))
                    X1, Y1 = int(min(bb[:, 2].max() + 1, rx.stop)), int(min(bb[:, 3].max() + 1, ry.stop))
                    if X1 > X0 and Y1 > Y0:
                        box = (X0, Y0, X1, Y1)
                        img[Y0:Y1, X0:X1] = bases[j][Y0:Y1, X0:X1]   # un-netted background
            for it in far_items:
                it[2]()
            if box is not None:                   # far-side objects sit behind the net
                X0, Y0, X1, Y1 = box
                a = al[Y0 - ry.start:Y1 - ry.start, X0 - rx.start:X1 - rx.start]
                c = col[Y0 - ry.start:Y1 - ry.start, X0 - rx.start:X1 - rx.start]
                img[Y0:Y1, X0:X1] = (img[Y0:Y1, X0:X1].astype(np.float32) * (1 - a) + c * a
                                     ).astype(np.uint8)
            for it in items:
                if not it[1]:
                    it[2]()
            if abs(flicker[i] - 1.0) > 1e-3:
                img = cv2.LUT(img, np.clip(ramp * flicker[i], 0, 255).astype(np.uint8))
            vw.write(img)
    finally:
        vw.release()
    return str(path)
