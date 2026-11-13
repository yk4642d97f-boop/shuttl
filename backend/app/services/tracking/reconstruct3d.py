"""
Monocular 3-D reconstruction of shuttle flights (MonoTrack-style physics fit).

Every flight between two contacts is a deterministic quadratic-drag ODE
trajectory (physics.py) with 6 unknowns (p0, v0 at the first observed time)
plus optionally the drag length L. Each frame gives 2 constraints (u, v), so
with >= ~8 frames the robust least-squares fit of reprojection error is
over-determined.

Identifiability (camera behind the near baseline):
  * Along each pixel ray the depth is unobservable from one frame. Across frames
    the *known* gravity g fixes metric scale: the image-space curvature of a
    flight at depth D is ~ f g / D px/s^2, so depth ~ f g / curvature.
  * Drag adds a second scale: deceleration is |v|^2 / L, i.e. it depends on the
    absolute (metric) speed, not just the angular one. With L poorly observed on
    short/slow segments we keep a soft prior on log L (and freeze it when the
    segment is too short to inform it).
  * Residual ambiguity is a (near-)1-D valley "closer + slower vs farther +
    faster" along the first ray, worst for far-half flights seen from a low
    camera (rays graze the floor: ~0.1 m of depth per px at 17 m). Hence the
    multi-hypothesis init over start depth, the hitter-position prior (feet xy
    + contact height, the strongest cue), and the floor-contact end constraint
    (z = 0 between the last airborne frame and the next one).
  * Plausibility (start inside the court volume +-2 m, speed < 500 km/h,
    z >= 0 in flight, in front of the camera) prunes mirror solutions.

Uncertainty: the Gauss-Newton covariance (J^T W J)^-1 scaled by the robust
reduced chi^2 is propagated to the landing point numerically; call_in_out uses
it to return 'unsure' when |margin| < 2 sigma.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from app.services.tracking import court_model as cm
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, ShotSegment3D, Track2D, VideoMeta,
)
from app.services.tracking.physics import DRAG_LENGTH, G
from app.services.tracking.video_io import pts_gaps

STEP = 1.0 / 480.0               # fixed RK4 step (s)
SIGMA_PX = 1.5                   # nominal detection noise (px)
F_SCALE = 1.6                    # soft-L1 scale in sigma units (~2.4 px)
MAX_SPEED = 500.0 / 3.6          # m/s, plausibility bound
LOG_L0 = math.log(DRAG_LENGTH)
H_MAX = 1.0 / 120.0              # adaptive RK4: h = min(H_MAX, C_STEP * L / |v|)
C_STEP = 0.08                    # drag-timescale fraction; RK4 error ~ C_STEP^5
NCOL = 8                         # state column 0 + 7 sensitivity columns

# ─── numba kernels ──────────────────────────────────────────────────────────
# The augmented state is two (3, 8) blocks: Y = [p | dp/dθ], X = [v | dv/dθ],
# θ = (p0, v0, log L). dY/dt = X, dX/dt = F(X) (drag ODE + variational eq.).


@njit(cache=True)
def _rhs(X, L, ncol, out):
    v0, v1, v2 = X[0, 0], X[1, 0], X[2, 0]
    sp = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    k = sp / L
    out[0, 0] = -k * v0
    out[1, 0] = -k * v1
    out[2, 0] = -G - k * v2
    inv = 1.0 / sp if sp > 1e-9 else 0.0
    for c in range(1, ncol):
        vs = (v0 * X[0, c] + v1 * X[1, c] + v2 * X[2, c]) * inv
        for r in range(3):
            out[r, c] = -(sp * X[r, c] + X[r, 0] * vs) / L
    if ncol == NCOL:
        for r in range(3):
            out[r, 7] += k * X[r, 0]


@njit(cache=True)
def _rk4(Y, X, h, L, ncol, W):
    K1, K2, K3, K4, Xt, SY = W[0], W[1], W[2], W[3], W[4], W[5]
    _rhs(X, L, ncol, K1)
    for r in range(3):
        for c in range(ncol):
            SY[r, c] = X[r, c]
            Xt[r, c] = X[r, c] + 0.5 * h * K1[r, c]
            SY[r, c] += 2.0 * Xt[r, c]
    _rhs(Xt, L, ncol, K2)
    for r in range(3):
        for c in range(ncol):
            Xt[r, c] = X[r, c] + 0.5 * h * K2[r, c]
            SY[r, c] += 2.0 * Xt[r, c]
    _rhs(Xt, L, ncol, K3)
    for r in range(3):
        for c in range(ncol):
            Xt[r, c] = X[r, c] + h * K3[r, c]
            SY[r, c] += Xt[r, c]
    _rhs(Xt, L, ncol, K4)
    for r in range(3):
        for c in range(ncol):
            Y[r, c] += h / 6.0 * SY[r, c]
            X[r, c] += h / 6.0 * (K1[r, c] + 2.0 * K2[r, c] + 2.0 * K3[r, c] + K4[r, c])


@njit(cache=True)
def _hlim(X, L):
    sp = math.sqrt(X[0, 0] ** 2 + X[1, 0] ** 2 + X[2, 0] ** 2)
    return min(H_MAX, C_STEP * L / max(sp, 1e-6))


@njit(cache=True)
def _init_state(p0, v0, ncol):
    Y = np.zeros((3, NCOL))
    X = np.zeros((3, NCOL))
    for r in range(3):
        Y[r, 0] = p0[r]
        X[r, 0] = v0[r]
        if ncol > 1:
            Y[r, 1 + r] = 1.0
            X[r, 4 + r] = 1.0
    return Y, X


@njit(cache=True)
def _propagate(p0, v0, L, times, ncol, P, V, S):
    """Integrate to sorted `times` (any sign, relative to the p0/v0 epoch).
    P, V: (N,3) positions/velocities; S: (N,3,7) dp/dθ, θ = (p0, v0, log L)."""
    W = np.empty((6, 3, NCOL))
    n = times.shape[0]
    first_pos = 0
    while first_pos < n and times[first_pos] < 0.0:
        first_pos += 1
    for direction in range(2):
        Y, X = _init_state(p0, v0, ncol)
        t = 0.0
        if direction == 0:
            idx_range = range(first_pos, n)
        else:
            idx_range = range(first_pos - 1, -1, -1)
        for i in idx_range:
            rem = times[i] - t
            sgn = 1.0 if rem >= 0 else -1.0
            rem = abs(rem)
            while rem > 1e-12:
                h = min(rem, _hlim(X, L))
                _rk4(Y, X, sgn * h, L, ncol, W)
                rem -= h
            t = times[i]
            for r in range(3):
                P[i, r] = Y[r, 0]
                V[i, r] = X[r, 0]
                for c in range(ncol - 1):
                    S[i, r, c] = Y[r, c + 1]


@njit(cache=True)
def _ground_time(p0, v0, L, t_max, z_ground):
    """Seconds (>= 0) until z first descends through z_ground; -1 if never."""
    W = np.empty((6, 3, NCOL))
    Y, X = _init_state(p0, v0, 1)
    if Y[2, 0] <= z_ground:
        return 0.0
    t = 0.0
    while t < t_max:
        h = _hlim(X, L)
        Yn, Xn = Y.copy(), X.copy()
        _rk4(Yn, Xn, h, L, 1, W)
        if Yn[2, 0] <= z_ground:
            a, b = 0.0, h            # bracket then secant/bisection on sub-step
            za, zb = Y[2, 0] - z_ground, Yn[2, 0] - z_ground
            for _ in range(30):
                m = a + (b - a) * za / (za - zb)
                if not (a < m < b):
                    m = 0.5 * (a + b)
                Ym, Xm = Y.copy(), X.copy()
                _rk4(Ym, Xm, m, L, 1, W)
                zm = Ym[2, 0] - z_ground
                if abs(zm) < 1e-7:
                    return t + m
                if zm > 0:
                    a, za = m, zm
                else:
                    b, zb = m, zm
            return t + 0.5 * (a + b)
        Y, X = Yn, Xn
        t += h
    return -1.0


# ─── priors ─────────────────────────────────────────────────────────────────

@dataclass
class SegmentPrior:
    """Soft priors for one flight. Times are absolute seconds (same clock as t).

    start_*: at the contact time `start_t` (default: first observed time); the
    reported p0/v0 are at start_t when it is given.
    end='floor': the shuttle was airborne at the last observation and down by
    `end_t` (e.g. first frame it rests on the floor); `end_xy_world` = rest
    point on the floor if seen. end='hit': contact at `end_t`, near
    `end_xy_world` (receiver's feet) with height in `end_height`.
    """
    start_t: float | None = None
    start_xy_world: np.ndarray | None = None
    start_xy_sigma: float = 0.5
    start_height: tuple[float, float] | None = (0.05, 3.5)
    expected_side: str | None = None               # hitter's half: 'near' | 'far'
    end: str | None = None                         # 'floor' | 'hit' | None
    end_t: float | None = None
    end_xy_world: np.ndarray | None = None
    end_xy_sigma: float = 0.5
    end_height: tuple[float, float] | None = (0.05, 3.5)
    rest_xy_world: np.ndarray | None = None         # where it lay afterwards (net test only)
    end_at_gap: bool = False                       # observations stop at a hole in the video
                                                   # clock: how it ended was never seen
    height_sigma: float = 0.1
    drag_length: float = DRAG_LENGTH
    drag_log_sigma: float = 0.08


# ─── residual model (numba) ─────────────────────────────────────────────────
# Residual vector, all in noise-sigma units, fixed layout (inactive hinge rows
# are exactly 0 with a zero gradient):
#   [0, 2n)          reprojection (u, v) per observation     — soft-L1
#   [2n, 3n)         z >= -2 cm at each observation           — L2 below
#   [3n, 4n)         camera depth >= 1 m at each observation
#   next 3           loose box on p0, 1 speed cap, 1 log L prior
#   start: 2 xy + 1 height hinge + 1 side hinge; end: 2 xy + 1 height/floor hinge
N_FIXED = 3 + 1 + 1 + 4 + 3
# prior float vector layout (NaN = inactive)
PS_X, PS_Y, PS_SIG, PS_HLO, PS_HHI, PS_SIDE, PS_HSIG = range(7)
PE_X, PE_Y, PE_SIG, PE_HLO, PE_HHI, PE_HSIG = range(7, 13)
P_LOGL, P_LOGSIG, P_NPRI = 13, 14, 15
BOX_LO = np.array([-9.5, -5.5, 0.0])
BOX_HI = np.array([9.5, 5.5, 8.0])


@njit(cache=True)
def _viol(x, lo, hi):
    """Signed violation of [lo, hi]; NaN bound = open."""
    if lo == lo and x < lo:
        return x - lo
    if hi == hi and x > hi:
        return x - hi
    return 0.0


@njit(cache=True)
def _resid_jac(th, fit_drag, L_fixed, times_s, inv, n, i_s, i_e, uv, sw, R, cam_t,
               fx, fy, skew, cx, cy, pri, r, J, want_jac, P, V, S):
    """Fill r (and J if want_jac) for θ = (p0, v0[, log L]); see layout above.
    uv are *undistorted* (pinhole) pixels, R/cam_t world→camera; `inv` maps the
    combined time index (obs..., start, end) to the sorted propagation order."""
    npar = 7 if fit_drag else 6
    L = math.exp(th[6]) if fit_drag else L_fixed
    ncol = (NCOL if fit_drag else NCOL - 1) if want_jac else 1
    _propagate(th[:3].copy(), th[3:6].copy(), L, times_s, ncol, P, V, S)
    m = r.shape[0]
    for i in range(m):
        r[i] = 0.0
    if want_jac:
        for i in range(m):
            for c in range(npar):
                J[i, c] = 0.0
    s_obs = 1.0 / SIGMA_PX
    for i in range(n):
        k = inv[i]
        px, py, pz = P[k, 0], P[k, 1], P[k, 2]
        X = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + cam_t[0]
        Y = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + cam_t[1]
        Z = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + cam_t[2]
        Zc = max(Z, 0.05)
        xn, yn = X / Zc, Y / Zc
        wi = sw[i] * s_obs
        r[2 * i] = (fx * xn + skew * yn + cx - uv[i, 0]) * wi
        r[2 * i + 1] = (fy * yn + cy - uv[i, 1]) * wi
        zv = _viol(pz, -0.02, np.nan)
        r[2 * n + i] = zv / 0.05
        dv = _viol(Z, 1.0, np.nan)
        r[3 * n + i] = dv / 0.1
        if want_jac:
            iz = 1.0 / Zc
            for c in range(npar):
                dX = R[0, 0] * S[k, 0, c] + R[0, 1] * S[k, 1, c] + R[0, 2] * S[k, 2, c]
                dY = R[1, 0] * S[k, 0, c] + R[1, 1] * S[k, 1, c] + R[1, 2] * S[k, 2, c]
                dZ = R[2, 0] * S[k, 0, c] + R[2, 1] * S[k, 1, c] + R[2, 2] * S[k, 2, c]
                dxn = (dX - xn * dZ) * iz
                dyn = (dY - yn * dZ) * iz
                J[2 * i, c] = (fx * dxn + skew * dyn) * wi
                J[2 * i + 1, c] = fy * dyn * wi
                if zv != 0.0:
                    J[2 * n + i, c] = S[k, 2, c] / 0.05
                if dv != 0.0:
                    J[3 * n + i, c] = dZ / 0.1
    row = 4 * n
    # loose box on p0
    for a in range(3):
        bv = _viol(th[a], BOX_LO[a], BOX_HI[a])
        r[row + a] = bv / 0.2
        if want_jac and bv != 0.0:
            J[row + a, a] = 1.0 / 0.2
    row += 3
    sp0 = math.sqrt(th[3] ** 2 + th[4] ** 2 + th[5] ** 2)
    if sp0 > MAX_SPEED:
        r[row] = (sp0 - MAX_SPEED) / 2.0
        if want_jac:
            for a in range(3):
                J[row, 3 + a] = th[3 + a] / sp0 / 2.0
    row += 1
    if fit_drag:
        r[row] = (th[6] - pri[P_LOGL]) / pri[P_LOGSIG]
        if want_jac:
            J[row, 6] = 1.0 / pri[P_LOGSIG]
    row += 1
    # start constraints (at combined-time index i_s)
    if i_s >= 0:
        k = inv[i_s]
        sig = pri[PS_SIG]
        if sig == sig:
            r[row] = (P[k, 0] - pri[PS_X]) / sig
            r[row + 1] = (P[k, 1] - pri[PS_Y]) / sig
            if want_jac:
                for c in range(npar):
                    J[row, c] = S[k, 0, c] / sig
                    J[row + 1, c] = S[k, 1, c] / sig
        hz = _viol(P[k, 2], pri[PS_HLO], pri[PS_HHI])
        r[row + 2] = hz / pri[PS_HSIG]
        side = pri[PS_SIDE]
        hx = 0.0
        if side == side:
            hx = _viol(P[k, 0], np.nan, 0.3) if side < 0 else _viol(P[k, 0], -0.3, np.nan)
        r[row + 3] = hx / 0.1
        if want_jac:
            for c in range(npar):
                if hz != 0.0:
                    J[row + 2, c] = S[k, 2, c] / pri[PS_HSIG]
                if hx != 0.0:
                    J[row + 3, c] = S[k, 0, c] / 0.1
    row += 4
    if i_e >= 0:
        k = inv[i_e]
        sig = pri[PE_SIG]
        if sig == sig:
            r[row] = (P[k, 0] - pri[PE_X]) / sig
            r[row + 1] = (P[k, 1] - pri[PE_Y]) / sig
            if want_jac:
                for c in range(npar):
                    J[row, c] = S[k, 0, c] / sig
                    J[row + 1, c] = S[k, 1, c] / sig
        hz = _viol(P[k, 2], pri[PE_HLO], pri[PE_HHI])
        r[row + 2] = hz / pri[PE_HSIG]
        if want_jac and hz != 0.0:
            for c in range(npar):
                J[row + 2, c] = S[k, 2, c] / pri[PE_HSIG]


@njit(cache=True)
def _robust_cost(r, n, wts):
    """Soft-L1 per (u, v) pair on the first 2n rows, L2 elsewhere; IRLS weights."""
    fs2 = F_SCALE * F_SCALE
    cost = 0.0
    for i in range(n):
        z = (r[2 * i] ** 2 + r[2 * i + 1] ** 2) / fs2
        root = math.sqrt(1.0 + z)
        cost += 2.0 * fs2 * (root - 1.0)
        wts[2 * i] = 1.0 / root
        wts[2 * i + 1] = 1.0 / root
    for i in range(2 * n, r.shape[0]):
        cost += r[i] * r[i]
        wts[i] = 1.0
    return cost


@njit(cache=True)
def _chol_solve(A, b, out):
    """Solve SPD A x = b (small, in place on a copy). Returns False if not SPD."""
    m = A.shape[0]
    Lc = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= Lc[i, k] * Lc[j, k]
            if i == j:
                if s <= 0.0:
                    return False
                Lc[i, i] = math.sqrt(s)
            else:
                Lc[i, j] = s / Lc[j, j]
    y = np.zeros(m)
    for i in range(m):
        s = b[i]
        for k in range(i):
            s -= Lc[i, k] * y[k]
        y[i] = s / Lc[i, i]
    for i in range(m - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, m):
            s -= Lc[k, i] * out[k]
        out[i] = s / Lc[i, i]
    return True


@njit(cache=True)
def _lm_kernel(th0, max_iter, tol, fit_drag, L_fixed, times_s, inv, n, i_s, i_e, uv, sw,
               R, cam_t, fx, fy, skew, cx, cy, pri):
    """Robust (IRLS soft-L1) Levenberg–Marquardt. Returns (θ, cost, iterations)."""
    npar = 7 if fit_drag else 6
    m = 4 * n + N_FIXED
    nt = times_s.shape[0]
    P, V, S = np.empty((nt, 3)), np.empty((nt, 3)), np.zeros((nt, 3, 7))
    r, J = np.zeros(m), np.zeros((m, npar))
    r2, J2 = np.zeros(m), np.zeros((1, npar))
    wts, wts2 = np.ones(m), np.ones(m)
    th = th0.copy()
    _resid_jac(th, fit_drag, L_fixed, times_s, inv, n, i_s, i_e, uv, sw, R, cam_t,
               fx, fy, skew, cx, cy, pri, r, J, True, P, V, S)
    cost = _robust_cost(r, n, wts)
    lam = 1e-3
    A = np.zeros((npar, npar))
    g = np.zeros(npar)
    Ad = np.zeros((npar, npar))
    step = np.zeros(npar)
    it = 0
    for it in range(1, max_iter + 1):
        for a in range(npar):
            g[a] = 0.0
            for b in range(npar):
                A[a, b] = 0.0
        for i in range(m):
            w = wts[i]
            ri = r[i] * w
            for a in range(npar):
                ja = J[i, a]
                if ja == 0.0:
                    continue
                g[a] += ja * ri
                jw = ja * w
                for b in range(a + 1):
                    A[a, b] += jw * J[i, b]
        dmax = 0.0
        for a in range(npar):
            for b in range(a):
                A[b, a] = A[a, b]
            dmax = max(dmax, A[a, a])
        improved = False
        c_new = cost
        th_new = th.copy()
        for _ in range(8):
            for a in range(npar):
                for b in range(npar):
                    Ad[a, b] = A[a, b]
                Ad[a, a] += lam * max(A[a, a], 1e-9 * max(dmax, 1e-12))
            ok = _chol_solve(Ad, -g, step)
            if not ok:
                lam *= 10.0
                continue
            for a in range(npar):
                th_new[a] = th[a] + step[a]
            if fit_drag:
                th_new[6] = min(max(th_new[6], LOG_L0 - 1.0), LOG_L0 + 1.0)
            v = math.sqrt(th_new[3] ** 2 + th_new[4] ** 2 + th_new[5] ** 2)
            if v > 1.5 * MAX_SPEED:                      # keep the ODE sane
                for a in range(3, 6):
                    th_new[a] *= 1.5 * MAX_SPEED / v
            _resid_jac(th_new, fit_drag, L_fixed, times_s, inv, n, i_s, i_e, uv, sw, R, cam_t,
                       fx, fy, skew, cx, cy, pri, r2, J2, False, P, V, S)
            c_new = _robust_cost(r2, n, wts2)
            if c_new == c_new and c_new < cost:
                improved = True
                break
            lam *= 8.0
        if not improved:
            break
        rel = (cost - c_new) / max(cost, 1e-12)
        smax = 0.0
        for a in range(npar):
            smax = max(smax, abs(th_new[a] - th[a]))
            th[a] = th_new[a]
        _resid_jac(th, fit_drag, L_fixed, times_s, inv, n, i_s, i_e, uv, sw, R, cam_t,
                   fx, fy, skew, cx, cy, pri, r, J, True, P, V, S)
        cost = _robust_cost(r, n, wts)
        lam = max(lam / 5.0, 1e-7)
        if rel < tol or smax < 1e-7:
            break
    return th, cost, it


def _prior_vec(pr: SegmentPrior, end_is_floor: bool) -> np.ndarray:
    v = np.full(P_NPRI, np.nan)
    if pr.start_xy_world is not None:
        v[PS_X], v[PS_Y] = np.asarray(pr.start_xy_world, float)[:2]
        v[PS_SIG] = pr.start_xy_sigma
    if pr.start_height is not None:
        v[PS_HLO], v[PS_HHI] = pr.start_height
    v[PS_HSIG] = pr.height_sigma
    if pr.expected_side in ("near", "far"):
        v[PS_SIDE] = -1.0 if pr.expected_side == "near" else 1.0
    if pr.end_xy_world is not None:
        v[PE_X], v[PE_Y] = np.asarray(pr.end_xy_world, float)[:2]
        v[PE_SIG] = pr.end_xy_sigma
    if end_is_floor:
        v[PE_HHI], v[PE_HSIG] = 0.0, 0.03          # down by end_t
    else:
        if pr.end_height is not None:
            v[PE_HLO], v[PE_HHI] = pr.end_height
        v[PE_HSIG] = pr.height_sigma
    v[P_LOGL], v[P_LOGSIG] = math.log(pr.drag_length), pr.drag_log_sigma
    return v


class _Problem:
    """Reprojection + prior residuals for θ = (p0, v0[, log L]) at t_ref."""

    def __init__(self, t: np.ndarray, uv: np.ndarray, w: np.ndarray, camera: CameraModel,
                 prior: SegmentPrior | None, fit_drag: bool, t_ref: float | None = None):
        self.cam = camera
        self.prior = prior or SegmentPrior()
        self.t_ref = float(t[0]) if t_ref is None else float(t_ref)   # θ epoch
        self.tr = t - self.t_ref
        self.uv = uv                                   # as observed (distorted) pixels
        self.sw = np.sqrt(np.clip(w, 0.0, None))
        self.n_obs = len(t)
        self.fit_drag = fit_drag
        self.npar = 7 if fit_drag else 6
        self.L_fixed = float(self.prior.drag_length)
        pr = self.prior
        K = camera.K
        if np.any(camera.dist):                       # residuals are pinhole: undistort once
            _, d = camera.ray(uv)
            dc = d @ camera.R.T
            xy = dc[:, :2] / dc[:, 2:3]
            self.uv_ideal = np.ascontiguousarray(xy @ K[:2, :2].T + K[:2, 2])
        else:
            self.uv_ideal = np.ascontiguousarray(uv, dtype=float)
        extra = []
        self.i_start = self.i_end = -1
        if pr.start_t is not None:
            self.i_start = self.n_obs + len(extra)
            extra.append(pr.start_t - self.t_ref)
        self.end_is_floor = pr.end == "floor"
        if pr.end in ("floor", "hit") and pr.end_t is not None:
            self.i_end = self.n_obs + len(extra)
            extra.append(pr.end_t - self.t_ref)
        times = np.concatenate([self.tr, np.asarray(extra, float)])
        self.order = np.argsort(times, kind="stable")
        self.inv = np.empty_like(self.order)
        self.inv[self.order] = np.arange(len(times))
        self.times_sorted = np.ascontiguousarray(times[self.order])
        self.n_t = len(times)
        self.pri = _prior_vec(pr, self.end_is_floor)
        self._cam_args = (np.ascontiguousarray(camera.R, dtype=float),
                          np.ascontiguousarray(camera.t, dtype=float),
                          float(K[0, 0]), float(K[1, 1]), float(K[0, 1]),
                          float(K[0, 2]), float(K[1, 2]))

    def _args(self, fit_drag: bool | None = None):
        fd = self.fit_drag if fit_drag is None else fit_drag
        return (fd, self.L_fixed, self.times_sorted, self.inv, self.n_obs, self.i_start,
                self.i_end, self.uv_ideal, self.sw, *self._cam_args, self.pri)

    def L_of(self, th: np.ndarray) -> float:
        return float(np.exp(th[6])) if len(th) > 6 else self.L_fixed

    def residuals(self, th: np.ndarray, jac: bool = True, fit_drag: bool | None = None):
        """Residuals (noise-sigma units) and analytic Jacobian (or None)."""
        fd = self.fit_drag if fit_drag is None else fit_drag
        npar = 7 if fd else 6
        m = 4 * self.n_obs + N_FIXED
        r, J = np.zeros(m), np.zeros((m, npar) if jac else (1, npar))
        nt = self.n_t
        P, V, S = np.empty((nt, 3)), np.empty((nt, 3)), np.zeros((nt, 3, 7))
        args = self._args(fd)
        _resid_jac(np.ascontiguousarray(th, dtype=float), args[0], args[1], args[2], args[3],
                   args[4], args[5], args[6], args[7], args[8], *args[9:], r, J, jac, P, V, S)
        return r, (J if jac else None)

    def simulate(self, th: np.ndarray, sens: bool = True):
        """Positions/velocities/sensitivities at [obs times..., extra times...]."""
        n = self.n_t
        Ps, Vs, Ss = np.empty((n, 3)), np.empty((n, 3)), np.zeros((n, 3, 7))
        fd = len(th) > 6
        ncol = (NCOL if fd else NCOL - 1) if sens else 1
        _propagate(np.ascontiguousarray(th[:3]), np.ascontiguousarray(th[3:6]), self.L_of(th),
                   self.times_sorted, ncol, Ps, Vs, Ss)
        return Ps[self.inv], Vs[self.inv], Ss[self.inv][:, :, : len(th)]


# ─── robust Levenberg–Marquardt driver ──────────────────────────────────────

@dataclass
class _Fit:
    th: np.ndarray
    cost: float
    n_iter: int


def _robust(r: np.ndarray, n_obs: int) -> tuple[float, np.ndarray]:
    wts = np.ones_like(r)
    return float(_robust_cost(r, n_obs, wts)), wts


def _lm(prob: _Problem, th: np.ndarray, max_iter: int = 40, tol: float = 1e-7) -> _Fit:
    th_out, cost, it = _lm_kernel(np.ascontiguousarray(th, dtype=float), max_iter, tol,
                                  *prob._args())
    return _Fit(th_out, float(cost), int(it))


# ─── initialisation ─────────────────────────────────────────────────────────

VOLUME_LO = np.array([-cm.HALF_LENGTH - 2.0, -cm.DOUBLES_HALF_WIDTH - 2.0, 0.0])
VOLUME_HI = np.array([cm.HALF_LENGTH + 2.0, cm.DOUBLES_HALF_WIDTH + 2.0, 6.0])
N_DEPTHS = 9


def _ray_box(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Parameter interval [s0, s1] (s > 0) of the ray o + s d inside a box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (lo - o) / d, (hi - o) / d
    s0 = np.nanmax(np.minimum(a, b))
    s1 = np.nanmin(np.maximum(a, b))
    return max(s0, 0.3), s1


@njit(cache=True)
def _v0_linear(p0, o, d, tr, L):
    """v0 so that p0 + v0 τ + ½ g t² lies on the (unit) rays d_i: linear LS in v0
    (row block [d]× τ, normal matrix τ² (I − d dᵀ)). Drag enters through the
    time warp τ = L ln(1 + |v| t / L) / |v|, iterated from |v| = 0."""
    v = np.zeros(3)
    A = np.empty((3, 3))
    b = np.empty(3)
    out = np.zeros(3)
    for _ in range(4):
        sp = max(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2), 1e-3)
        A[:, :] = 0.0
        b[:] = 0.0
        for i in range(tr.shape[0]):
            tau = L * math.log1p(sp * tr[i] / L) / sp
            q0, q1, q2 = p0[0] - o[0], p0[1] - o[1], p0[2] - o[2] - 0.5 * G * tr[i] ** 2
            dq = d[i, 0] * q0 + d[i, 1] * q1 + d[i, 2] * q2
            q = (q0 - dq * d[i, 0], q1 - dq * d[i, 1], q2 - dq * d[i, 2])   # (I − d dᵀ) q
            for a in range(3):
                b[a] -= tau * q[a]
                for c in range(3):
                    A[a, c] += tau * tau * ((1.0 if a == c else 0.0) - d[i, a] * d[i, c])
        for a in range(3):
            A[a, a] += 1e-9
        if not _chol_solve(A, b, out):
            break
        v[:] = out
    sp = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if sp > MAX_SPEED:
        v *= MAX_SPEED / sp
    return v


@njit(cache=True)
def _outlier_mask(t, uv, half=4, thr=10.0, passes=3):
    """True = likely outlier: > thr px from a quadratic fit of its (unflagged)
    neighbours. Iterated so an outlier does not mask itself via its neighbours."""
    n = t.shape[0]
    bad = np.zeros(n, np.bool_)
    if n < 6:
        return bad
    A = np.empty((2 * half + 1, 3))
    B = np.empty((2 * half + 1, 2))
    for _ in range(passes):
        new = np.zeros(n, np.bool_)
        for i in range(n):
            m = 0
            for j in range(max(0, i - 2 * half), min(n, i + 2 * half + 1)):
                if j == i or bad[j] or m >= 2 * half:
                    continue
                if abs(j - i) > half and m >= 4:
                    continue
                tt = t[j] - t[i]
                A[m, 0], A[m, 1], A[m, 2] = 1.0, tt, tt * tt
                B[m, 0], B[m, 1] = uv[j, 0], uv[j, 1]
                m += 1
            if m < 5:
                continue
            c = np.linalg.lstsq(A[:m], B[:m])[0]
            res = np.empty(m)
            for k in range(m):
                du = A[k, 0] * c[0, 0] + A[k, 1] * c[1, 0] + A[k, 2] * c[2, 0] - B[k, 0]
                dv = A[k, 0] * c[0, 1] + A[k, 1] * c[1, 1] + A[k, 2] * c[2, 1] - B[k, 1]
                res[k] = math.sqrt(du * du + dv * dv)
            e = math.sqrt((c[0, 0] - uv[i, 0]) ** 2 + (c[0, 1] - uv[i, 1]) ** 2)
            new[i] = e > max(thr, 6.0 * np.median(res))
        if np.array_equal(new, bad):
            break
        bad = new
    return bad


def _hypotheses(prob: _Problem, t: np.ndarray, uv: np.ndarray,
                feet_only: bool = False) -> list[np.ndarray]:
    """Initial θ candidates: a sweep of start depths along the first ray (uniform in
    inverse depth over the court volume) plus depths at the hitter's feet prior;
    v0 for each from the early rays by linear least squares."""
    cam, pr = prob.cam, prob.prior
    if feet_only and pr.start_xy_world is None:
        return []
    good = ~_outlier_mask(np.ascontiguousarray(t), np.ascontiguousarray(uv))
    if good.sum() < 5:
        good[:] = True
    good[0] = True                     # the first ray anchors the depth sweep
    t, uv = t[good], uv[good]
    o, d = cam.ray(uv)
    tr = t - t[0]
    # short-time window only (the linear v0 model breaks over a detection gap, e.g.
    # a clear leaving the top of the frame); >= 2 rays (4 constraints on v0),
    # at most 10 spread over the window
    early = np.nonzero(tr <= 0.25)[0]
    if len(early) < 2:
        early = np.arange(min(2, len(tr)))
    if len(early) > 10:
        early = early[np.round(np.linspace(0, len(early) - 1, 10)).astype(int)]
    s0, s1 = _ray_box(o, d[0], VOLUME_LO, VOLUME_HI)
    depths: list[float] = []
    if s1 > s0 and not feet_only:
        # uniform in inverse depth → denser sampling far away where rays graze
        inv = np.linspace(1 / s0, 1 / s1, N_DEPTHS)
        depths += list(1 / inv)
    if pr.start_xy_world is not None:
        # point on the first ray horizontally closest to the hitter's feet
        dh = d[0, :2]
        s = float(np.dot(np.asarray(pr.start_xy_world[:2]) - o[:2], dh) / max(dh @ dh, 1e-9))
        if s > 0.3:
            depths.append(s)
            depths += [s * 0.93, s * 1.07]
    L = prob.L_fixed
    hyps = []
    for s in depths:
        p0 = o + s * d[0]
        v0 = _v0_linear(p0, o, np.ascontiguousarray(d[early]), np.ascontiguousarray(tr[early]), L)
        hyps.append(np.r_[p0, v0, math.log(L)] if prob.fit_drag else np.r_[p0, v0])
    # a detection gap right after the start leaves the linear v0 ill-posed (a few
    # rays within < 0.15 s): also shoot each start depth to the first ray after the
    # gap, at the paired depth of that ray's sweep and its two neighbours
    span = float(tr[early[-1]])
    later = np.nonzero(tr > span + 0.05)[0]
    if span < 0.15 and len(later) and len(depths) >= N_DEPTHS:
        j = int(later[0])
        q0, q1 = _ray_box(o, d[j], VOLUME_LO, VOLUME_HI)
        if q1 > q0:
            qd = 1 / np.linspace(1 / q0, 1 / q1, N_DEPTHS)
            for i in range(N_DEPTHS):
                p0 = o + depths[i] * d[0]
                for k in (i - 1, i, i + 1):
                    if 0 <= k < N_DEPTHS:
                        v0 = _shoot(p0, o + qd[k] * d[j], float(tr[j]), L)
                        if v0 is not None:
                            hyps.append(np.r_[p0, v0, math.log(L)] if prob.fit_drag
                                        else np.r_[p0, v0])
    return hyps


def _shoot(p_c: np.ndarray, q: np.ndarray, dt: float, L: float) -> np.ndarray | None:
    """Launch velocity taking the drag flight from p_c to q in dt seconds (Newton on
    v with the ODE sensitivities), or None if unreachable."""
    v = (q - p_c) / dt
    v[2] += 0.5 * G * dt
    P, V, S = np.empty((1, 3)), np.empty((1, 3)), np.zeros((1, 3, 7))
    tt = np.array([dt])
    err = np.inf
    for _ in range(20):
        _propagate(p_c, v, L, tt, NCOL - 1, P, V, S)
        r = P[0] - q
        err = float(np.linalg.norm(r))
        if err < 2e-3:
            break
        try:
            step = np.linalg.solve(S[0, :, 3:6], r)
        except np.linalg.LinAlgError:
            return None
        n = float(np.linalg.norm(step))
        if n > 30.0:
            step *= 30.0 / n
        v = v - step
        sp = float(np.linalg.norm(v))
        if sp > MAX_SPEED:
            v *= MAX_SPEED / sp
    return v if err < 0.05 else None


def _hypotheses_contact(prob: _Problem, t: np.ndarray, uv: np.ndarray) -> list[np.ndarray]:
    """Hypotheses at the contact epoch (prob.t_ref < first observation): contact
    points (hitter's feet, else a line down the hitter's half) x contact heights,
    each shot through a depth sweep on the first observed ray."""
    pr = prob.prior
    good = ~_outlier_mask(np.ascontiguousarray(t), np.ascontiguousarray(uv))
    i0 = int(np.argmax(good)) if good.any() else 0
    o, d = prob.cam.ray(uv[i0:i0 + 1])
    dt = float(t[i0] - prob.t_ref)
    if pr.start_xy_world is not None:
        xys = [np.asarray(pr.start_xy_world, float)[:2]]
    else:
        sgn = -1.0 if pr.expected_side == "near" else 1.0
        xys = [np.array([sgn * a, 0.0]) for a in (2.0, 4.0, 6.0)]
    lo, hi = pr.start_height if pr.start_height is not None else (0.05, 3.5)
    heights = sorted({float(np.clip(h, lo, hi)) for h in (1.0, 2.0, 2.7)})
    s0, s1 = _ray_box(o, d[0], VOLUME_LO, VOLUME_HI)
    if not s1 > s0:
        return []
    depths = 1 / np.linspace(1 / s0, 1 / s1, N_DEPTHS)
    L = prob.L_fixed
    hyps = []
    for xy in xys:
        for h in heights:
            p_c = np.array([xy[0], xy[1], h])
            for sd in depths:
                v = _shoot(p_c, o + sd * d[0], dt, L)
                if v is not None:
                    hyps.append(np.r_[p_c, v, math.log(L)] if prob.fit_drag else np.r_[p_c, v])
    return hyps


# ─── segment fit ────────────────────────────────────────────────────────────

def _plausible(prob: _Problem, th: np.ndarray) -> bool:
    Pa, Va, _ = prob.simulate(th, sens=False)
    P = Pa[: prob.n_obs]
    Pc = P @ prob.cam.R.T + prob.cam.t
    ok = bool(np.all(th[:3] >= VOLUME_LO - [0, 0, 0.1]) and np.all(th[:3] <= VOLUME_HI)
              and np.linalg.norm(th[3:6]) < MAX_SPEED
              and P[:, 2].min() > -0.15 and Pc[:, 2].min() > 0.5)
    if ok and prob.i_start >= 0:                     # also at the (extrapolated) contact
        ok = bool(np.linalg.norm(Va[prob.i_start]) < MAX_SPEED and Pa[prob.i_start, 2] > -0.15)
    return ok


def _pick(prob: _Problem, fits: list[_Fit]) -> _Fit | None:
    best, best_key = None, None
    for f in fits:
        key = (_plausible(prob, f.th), -f.cost)
        if best is None or key > best_key:
            best, best_key = f, key
    return best


def _solve(prob: _Problem, t: np.ndarray, uv: np.ndarray,
           init: np.ndarray | None = None) -> _Fit | None:
    contact_epoch = prob.t_ref < t[0] - 1e-6
    if init is not None and not contact_epoch:            # warm start (+ feet depths)
        hyps = [init] + _hypotheses(prob, t, uv, feet_only=True)
        return _pick(prob, [_lm(prob, h, max_iter=40) for h in hyps])
    hyps = _hypotheses_contact(prob, t, uv) if contact_epoch else _hypotheses(prob, t, uv)
    if init is not None:
        hyps = [init] + hyps
    if not hyps:
        return None
    coarse = [_lm(prob, h, max_iter=6, tol=1e-4) for h in hyps]
    coarse.sort(key=lambda f: f.cost)
    best: _Fit | None = None
    kept: list[np.ndarray] = []
    for f in coarse:
        if len(kept) >= 3:
            break
        if any(np.linalg.norm(f.th[:3] - k[:3]) < 0.15 for k in kept):
            continue
        kept.append(f.th)
        ff = _lm(prob, f.th, max_iter=40)
        ok = _plausible(prob, ff.th)
        if best is None or (ok, -ff.cost) > (best_ok, -best.cost):
            best, best_ok = ff, ok
    return best


def fit_segment(t: np.ndarray, uv: np.ndarray, w: np.ndarray | None, camera: CameraModel,
                prior: SegmentPrior | None = None, fit_drag: bool = True,
                frames: np.ndarray | None = None,
                init: tuple[np.ndarray, float] | None = None) -> ShotSegment3D | None:
    """Fit one ODE flight to pixel observations (NaN / zero-weight rows skipped).

    Parametrised internally at the first observed time; the returned p0/v0/t0
    are at the contact time prior.start_t when given (else first observation).
    `frames` labels the rows of t (default 0..N-1); `init` = (θ7, epoch) warm
    start, e.g. (seg.theta_ref, seg.t_ref) of an earlier fit. Returns None with
    < 5 valid points or no plausible solution. Non-contract extras on the result:
    theta_ref/t_ref/cov_theta (7x7 over p0, v0, log L at t_ref), landing_cov,
    landing_sigma_m, speed0_sigma_kmh, end_reason ('hit'|'floor'|'net'|'unknown'),
    landing_extrapolated_xy, n_obs, inlier_frac, t_end.
    """
    with _blas_single_thread():
        return _fit_segment(t, uv, w, camera, prior, fit_drag, frames, init)


def _fit_segment(t, uv, w, camera, prior, fit_drag, frames, init) -> ShotSegment3D | None:
    t = np.asarray(t, float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    frames = np.arange(len(t)) if frames is None else np.asarray(frames)
    ok = np.isfinite(uv).all(1) & np.isfinite(t) & (w > 0)
    if ok.sum() < 5:
        return None
    ok_idx = np.nonzero(ok)[0]
    ok_idx = ok_idx[np.argsort(t[ok_idx], kind="stable")]
    to, uvo, wo = t[ok_idx], uv[ok_idx], w[ok_idx]
    prior = prior or SegmentPrior()
    gap = float(to[0] - prior.start_t) if prior.start_t is not None else 0.0
    res = _fit_core(to, uvo, wo, camera, prior, fit_drag, init)
    if res is None:
        # the whole arc is no plausible single flight (it includes post-flight
        # points: slide down the net, skid / rest on the floor) → longest prefix
        k = _consistent_prefix(to, uvo, wo, camera, prior, fit_drag, None, None)
        pr = dataclasses.replace(prior, end=None, end_t=None, end_xy_world=None)
        res = None if k is None else _fit_core(to[:k], uvo[:k], wo[:k], camera, pr,
                                               fit_drag, None)
        if res is None:
            return None
        prob, fit, contact_seen = res
        seg = _finalize(prob, fit, t, frames[ok_idx[:k]], frames)
        seg.contact_observed, seg.contact_gap_s = contact_seen, gap
        seg.n_trimmed = int(len(to) - k)
        return seg
    prob, fit, contact_seen = res
    n_keep = len(to)
    # a trailing run of outliers = the flight ended (net, floor skid, mislink)
    cut = _trailing_outliers(prob, fit.th)
    # a tail the whole-arc fit cannot follow (the fit got dragged by post-flight
    # points: slide down the net, carried shuttle, ...) → longest consistent prefix
    k = None if cut else _consistent_prefix(to, uvo, wo, camera, prior, fit_drag, prob, fit)
    if cut or k:
        keep = len(to) - cut if cut else k
        pr = prior if cut else dataclasses.replace(prior, end=None, end_t=None,
                                                   end_xy_world=None)
        res2 = _fit_core(to[:keep], uvo[:keep], wo[:keep], camera, pr, fit_drag,
                         (_th7(fit.th, prob.L_fixed), prob.t_ref))
        if res2 is not None:
            prob, fit, contact_seen = res2
            n_keep = keep
    seg = _finalize(prob, fit, t, frames[ok_idx[:n_keep]], frames)
    seg.contact_observed, seg.contact_gap_s = contact_seen, gap   # False: p0/v0 at 1st sighting
    seg.n_trimmed = int(len(to) - n_keep)
    return seg


def _fit_core(to: np.ndarray, uvo: np.ndarray, wo: np.ndarray, camera: CameraModel,
              prior: SegmentPrior, fit_drag: bool, init: tuple[np.ndarray, float] | None
              ) -> tuple[_Problem, _Fit, bool] | None:
    """Fit sorted valid observations → (problem, fit, contact_observed) or None."""
    contact_seen = prior.start_t is None or to[0] - prior.start_t <= MAX_BACK_EXTRAP
    # drag is weakly observable on short/slow flights → keep it at the prior
    fit_drag_c = fit_drag and len(to) >= 10
    fd = fit_drag_c and (to[-1] - to[0] > 0.3)
    if not contact_seen and (prior.start_xy_world is not None or prior.expected_side):
        # contact long before the first sighting (the flight left the frame): fit at
        # the contact epoch, integrating forward (the start priors then pin the
        # depth; backward extrapolation of the drag ODE would be ill-conditioned)
        prob = _Problem(to, uvo, wo, camera, prior, fit_drag_c, t_ref=prior.start_t)
        fit = _solve(prob, to, uvo, _init_theta(init, prob))
        if fit is not None and _plausible(prob, fit.th):
            return prob, fit, True
    pr = prior if contact_seen else dataclasses.replace(
        prior, start_t=None, start_xy_world=None, expected_side=None)
    prob = _Problem(to, uvo, wo, camera, pr, fd)
    fit = _solve(prob, to, uvo, _init_theta(init, prob))
    if fit is None or not _plausible(prob, fit.th):
        return None
    return prob, fit, contact_seen


def _pred_err(camera: CameraModel, prob: _Problem, th: np.ndarray, to: np.ndarray,
              uvo: np.ndarray) -> np.ndarray:
    """Pixel distance between observations and the fitted flight at times `to`."""
    P, _ = _state_at(_th7(th, prob.L_fixed), to - prob.t_ref)
    return np.linalg.norm(camera.project(P) - uvo, axis=1)


TAIL_BAD_FRAC = 0.25            # > this share of the last 40 % off by > thr → inconsistent
PREFIX_THR_MULT = 3.0           # growth stops at 2 points off by > this x the clean-seed sigma
PREFIX_MIN_THR = 4.0            # px
PREFIX_GAIN = 1.8               # whole-arc / trimmed median error on the kept points


def _consistent_prefix(to: np.ndarray, uvo: np.ndarray, wo: np.ndarray, camera: CameraModel,
                       prior: SegmentPrior, fit_drag: bool, prob: _Problem | None,
                       fit: _Fit | None) -> int | None:
    """Longest prefix one free flight explains when the whole-arc fit (prob/fit;
    None = no plausible whole-arc fit) looks contaminated by post-flight points
    (slide down the net, skid / rest, carried shuttle); else None.

    The noise level is fixed once from the shortest seed prefix (30 %; the
    post-flight part is at the end), so it cannot inflate while the prefix grows
    into contaminated points. Seeds at 30/50/70 % are grown to the first two
    consecutive points off by > PREFIX_THR_MULT x that noise and refitted until
    stable; a trim is accepted only if its fit follows its own tail and beats the
    whole-arc fit clearly on the kept points."""
    n = len(to)
    if n < 12:
        return None
    pr = dataclasses.replace(prior, end=None, end_t=None, end_xy_world=None)
    k0 = max(6, int(0.3 * n))
    r0 = _fit_core(to[:k0], uvo[:k0], wo[:k0], camera, pr, fit_drag, None)
    if r0 is None:
        return None
    e0 = _pred_err(camera, r0[0], r0[1].th, to[:k0], uvo[:k0])
    noise = max(float(np.median(e0)), 0.5)
    thr = max(PREFIX_MIN_THR, PREFIX_THR_MULT * noise / 1.177)   # Rayleigh median → sigma
    e_all = None
    if prob is not None and fit is not None:
        e_all = _pred_err(camera, prob, fit.th, to, uvo)
        if not np.mean(e_all[int(0.6 * n):] > thr) > TAIL_BAD_FRAC:
            return None                             # the whole arc is consistent
    best = None
    for frac in (0.3, 0.5, 0.7):
        k, good = max(6, int(frac * n)), None
        for it in range(8):                         # grow → refit until stable
            res = r0 if (frac == 0.3 and it == 0) else _fit_core(
                to[:k], uvo[:k], wo[:k], camera, pr, fit_drag, None)
            if res is None:
                break
            ek = _pred_err(camera, res[0], res[1].th, to, uvo)
            if np.mean(ek[int(0.6 * k):k] > thr) > TAIL_BAD_FRAC:
                break                               # its own tail is inconsistent
            good = k
            j = k
            while j < n and not (ek[j] > thr and (j + 1 >= n or ek[j + 1] > thr)):
                j += 1
            if j >= n and e_all is not None:        # explains everything: no trim
                good = None
            if j <= k or j >= n:
                break
            k = j
        if good is not None and 5 <= good < n - 1 and (best is None or good > best):
            best = good
    if best is not None and e_all is not None:
        rb = _fit_core(to[:best], uvo[:best], wo[:best], camera, pr, fit_drag, None)
        if rb is None:
            return None
        eb = _pred_err(camera, rb[0], rb[1].th, to[:best], uvo[:best])
        # accept if the whole arc really misses the dropped points (not one outlier)
        # and the trimmed flight follows the kept ones better — or if the whole arc
        # is a compromise that misses the kept points by far (> PREFIX_GAIN x)
        e_kept, e_trim = float(np.median(e_all[:best])), float(np.median(eb))
        if not ((np.median(e_all[best:]) > thr and e_trim < 0.8 * e_kept)
                or (best >= 10 and e_kept > PREFIX_GAIN * e_trim)):
            return None
    return best


def _init_theta(init: tuple[np.ndarray, float] | None, prob: _Problem) -> np.ndarray | None:
    """Warm start (θ7, epoch) moved to prob's epoch (None if it blows up)."""
    if init is None:
        return None
    th7i, ti = np.asarray(init[0], float), float(init[1])
    if abs(ti - prob.t_ref) > 1e-9:
        if abs(ti - prob.t_ref) > 1.0:
            return None
        Pi, Vi = _state_at(th7i, np.array([prob.t_ref - ti]))
        if not (np.all(np.isfinite(Vi)) and np.linalg.norm(Vi[0]) < 1.5 * MAX_SPEED):
            return None
        th7i = np.r_[Pi[0], Vi[0], th7i[6]]
    return th7i if prob.fit_drag else th7i[:6]


def _trailing_outliers(prob: _Problem, th: np.ndarray) -> int:
    """Length of the run of gross outliers (> max(8 px, 4x median)) at the end of
    the observations, if >= 2 and >= MIN_KEEP points remain; else 0."""
    r, _ = prob.residuals(th, jac=False)
    n = prob.n_obs
    e = np.linalg.norm(r[: 2 * n].reshape(n, 2), axis=1) * SIGMA_PX / np.maximum(prob.sw, 1e-9)
    thr = max(8.0, 4.0 * float(np.median(e)))
    k = 0
    while k < n and e[n - 1 - k] > thr:
        k += 1
    return k if k >= 2 and n - k >= 5 else 0


def _th7(th: np.ndarray, L: float) -> np.ndarray:
    return np.asarray(th, float) if len(th) == 7 else np.r_[th, math.log(L)]


def _state_at(th7: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions/velocities at τ = `times` (any order/sign) relative to the θ epoch."""
    ts = np.asarray(times, float).ravel()
    order = np.argsort(ts, kind="stable")
    n = len(ts)
    Ps, Vs, S = np.empty((n, 3)), np.empty((n, 3)), np.zeros((n, 3, 7))
    _propagate(np.ascontiguousarray(th7[:3], dtype=float),
               np.ascontiguousarray(th7[3:6], dtype=float),
               float(np.exp(th7[6])), np.ascontiguousarray(ts[order]), 1, Ps, Vs, S)
    P, V = np.empty_like(Ps), np.empty_like(Vs)
    P[order], V[order] = Ps, Vs
    return P, V


def _landing(th7: np.ndarray, t_from: float = 0.0) -> tuple[np.ndarray, float] | None:
    """First floor contact (xy, τ) after τ = t_from, relative to the θ epoch."""
    p, v = _state_at(th7, np.array([t_from]))
    tg = _ground_time(p[0], v[0], float(np.exp(th7[6])), 8.0, 0.0)
    if tg < 0:
        return None
    P, _ = _state_at(th7, np.array([t_from + tg]))
    return P[0, :2].copy(), t_from + tg


def _landing_cov(th7: np.ndarray, cov7: np.ndarray, t_from: float) -> np.ndarray | None:
    """Landing-point covariance by central differences of _landing through cov7."""
    Jl = np.zeros((2, 7))
    for k in range(7):
        e = 1e-4
        hp, hm = th7.copy(), th7.copy()
        hp[k] += e
        hm[k] -= e
        lp, lm = _landing(hp, t_from), _landing(hm, t_from)
        if lp is None or lm is None:
            return None
        Jl[:, k] = (lp[0] - lm[0]) / (2 * e)
    return Jl @ cov7 @ Jl.T


def _speed_sigma(th7: np.ndarray, cov7: np.ndarray, tau: float) -> float:
    """1-σ of |v(τ)| (m/s) by central differences."""
    g = np.zeros(7)
    for k in range(7):
        e = 1e-4
        hp, hm = th7.copy(), th7.copy()
        hp[k] += e
        hm[k] -= e
        vp = np.linalg.norm(_state_at(hp, np.array([tau]))[1][0])
        vm = np.linalg.norm(_state_at(hm, np.array([tau]))[1][0])
        g[k] = (vp - vm) / (2 * e)
    return float(np.sqrt(max(g @ cov7 @ g, 0.0)))


def net_height_at(y: float) -> float:
    """Net top height (m) at lateral position y (sags from posts to centre)."""
    a = min(abs(y) / cm.DOUBLES_HALF_WIDTH, 1.0)
    return cm.NET_HEIGHT_CENTRE + (cm.NET_HEIGHT_POST - cm.NET_HEIGHT_CENTRE) * a * a


def _net_crossing(th7: np.ndarray, tau0: float, tau1: float) -> tuple[float, float, float] | None:
    """First crossing of x = 0 in [tau0, tau1]: (z, y, τ)."""
    if tau1 <= tau0:
        return None
    grid = np.arange(tau0, tau1 + 1e-9, 1 / 480)
    if len(grid) < 2:
        return None
    P, _ = _state_at(th7, grid)
    s = np.sign(P[:, 0])
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    a = P[i, 0] / (P[i, 0] - P[i + 1, 0])
    z = P[i, 2] + a * (P[i + 1, 2] - P[i, 2])
    y = P[i, 1] + a * (P[i + 1, 1] - P[i, 1])
    return float(z), float(y), float(grid[i] + a * (grid[i + 1] - grid[i]))


def _calib_floor_cov(cam: CameraModel, xy: tuple[float, float]) -> np.ndarray:
    """Floor-point covariance (m²) from the camera's calibration reprojection RMSE."""
    rms = cam.reproj_rmse_px
    if not (rms == rms) or rms <= 0:
        return np.zeros((2, 2))
    uv = cam.project(np.array([[xy[0], xy[1], 0.0]]))
    if not np.all(np.isfinite(uv)):
        return np.zeros((2, 2))
    Jf = np.zeros((2, 2))
    for k in range(2):
        d = np.zeros((1, 2))
        d[0, k] = 1.0
        Jf[:, k] = (cam.pixel_to_floor(uv + d)[0, :2] - cam.pixel_to_floor(uv - d)[0, :2]) / 2
    if not np.all(np.isfinite(Jf)):
        return np.zeros((2, 2))
    return (rms ** 2) * (Jf @ Jf.T)


MAX_BACK_EXTRAP = 0.35          # s; contact state is extrapolated at most this far back
NET_SEEN_S = 0.1                # s; tracked this long past the net plane → it got over
OPEN_LAND_S = 0.6               # s; open-ended flight: extrapolate a landing at most this far
OWN_SIDE_LAND_M = 0.3           # fitted landing this far on the hitter's half → never got over
OWN_SIDE_REST_M = 0.15          # at rest this far on the hitter's half → never got over


def _finalize(prob: _Problem, fit: _Fit, t_all: np.ndarray, frames_obs: np.ndarray,
              frames_all: np.ndarray) -> ShotSegment3D:
    th7 = _th7(fit.th, prob.L_fixed)
    n, pr, t_ref = prob.n_obs, prob.prior, prob.t_ref
    # covariance with log L always included (its prior row keeps it well-posed, and
    # marginalises drag when it was held fixed), scaled by the robust reduced chi²
    r7, J7 = prob.residuals(th7, jac=True, fit_drag=True)
    ruv = r7[: 2 * n].reshape(n, 2)
    e = np.linalg.norm(ruv * SIGMA_PX / np.maximum(prob.sw, 1e-9)[:, None], axis=1)
    inl = e < max(3 * np.median(e), 4.0)
    if not inl.any():
        inl[:] = True
    rmse = float(np.sqrt(np.mean(e[inl] ** 2)))
    _, w7 = _robust(r7, n)
    A = J7.T @ (J7 * w7[:, None])
    dof = max(2 * int(inl.sum()) - prob.npar, 1)
    s2 = max(1.0, float((ruv[inl] ** 2).sum()) / dof)
    try:
        cov7 = np.linalg.inv(A) * s2
    except np.linalg.LinAlgError:
        cov7 = np.full((7, 7), np.nan)
    cov_ok = bool(np.all(np.isfinite(cov7)))

    # contact epoch: report p0/v0 at the hit time when it is known and close
    tau_c = 0.0
    if pr.start_t is not None:
        tc = float(pr.start_t - t_ref)
        if -MAX_BACK_EXTRAP <= tc <= 0.05:
            tau_c = tc
    Pc, Vc = _state_at(th7, np.array([tau_c]))
    tau_last = float(prob.tr[-1])
    P_last, V_last = _state_at(th7, np.array([tau_last]))

    # how the flight ended
    landing_xy = landing_t = landing_cov = extrap_xy = None
    end_reason = "unknown"
    lnd = None
    side = pr.expected_side
    hitter_sgn = ((-1.0 if side == "near" else 1.0) if side in ("near", "far")
                  else math.copysign(1.0, Pc[0, 0]))
    # where the shuttle came to rest afterwards, signed towards the hitter's half
    rest_x = (hitter_sgn * float(pr.rest_xy_world[0])
              if pr.rest_xy_world is not None and pr.end != "hit" else None)
    if pr.end == "hit":
        end_reason = "hit"
        tau_end = float(pr.end_t - t_ref) if pr.end_t is not None else tau_last
    elif pr.end_at_gap:
        tau_end = tau_last                           # unseen end: no landing, no net call
    else:
        t_from = tau_last
        if P_last[0, 2] <= 0.0:                      # fitted path already down: back up
            for back in (0.01, 0.03, 0.06, 0.12, 0.25):
                if _state_at(th7, np.array([tau_last - back]))[0][0, 2] > 0:
                    t_from = tau_last - back
                    break
        lnd = _landing(th7, t_from)
        tau_end = lnd[1] if lnd is not None else tau_last
    nc = _net_crossing(th7, tau_c, tau_end)
    net_h = nc[0] if nc is not None else None
    if lnd is not None:
        # open end: descending and reaching the floor soon after the last sighting
        floorish = pr.end == "floor" or (pr.end is None and V_last[0, 2] < 0
                                          and lnd[1] - tau_last < OPEN_LAND_S)
        # a net fault needs the (fitted) crossing below the tape AND no evidence that
        # the shuttle got over: followed by the observations > NET_SEEN_S past the
        # crossing (depth is too noisy to test the side of the last point), or lying
        # on the far side afterwards
        over = nc is not None and tau_last > nc[2] + NET_SEEN_S
        # never got across: the fitted flight lands on the hitter's own half
        own = nc is None and hitter_sgn * float(lnd[0][0]) > OWN_SIDE_LAND_M
        if rest_x is not None:
            over = over or rest_x < -0.8
            own = own or rest_x > OWN_SIDE_REST_M
        if not over and (own or (nc is not None and nc[0] < net_height_at(nc[1]))):
            end_reason = "net"
        elif floorish:
            end_reason = "floor"
            landing_xy = (float(lnd[0][0]), float(lnd[0][1]))
            landing_t = float(t_ref + lnd[1])
            if cov_ok:
                landing_cov = _landing_cov(th7, cov7, t_from)
                if landing_cov is not None:
                    landing_cov = landing_cov + _calib_floor_cov(prob.cam, landing_xy)
        else:
            extrap_xy = (float(lnd[0][0]), float(lnd[0][1]))
    elif pr.end != "hit" and not pr.end_at_gap and rest_x is not None and rest_x > OWN_SIDE_REST_M:
        end_reason = "net"                           # no landing fitted, but at rest on its own half

    # frames covered: contact (or first obs) → landing / next contact / last obs
    tau_hi = tau_end if end_reason in ("floor", "hit") else tau_last
    tau_all = np.asarray(t_all, float) - t_ref
    cover = np.isfinite(tau_all) & (tau_all >= min(tau_c, 0.0) - 1e-9) & (tau_all <= tau_hi + 1e-9)
    P_cov, _ = _state_at(th7, tau_all[cover])
    fr_cov = np.asarray(frames_all)[cover]

    lsig = (float(np.sqrt(max(np.linalg.eigvalsh(landing_cov).max(), 0.0)))
            if landing_cov is not None and np.all(np.isfinite(landing_cov)) else None)
    sp_sig = _speed_sigma(th7, cov7, tau_c) * 3.6 if cov_ok else None
    sp0 = float(np.linalg.norm(Vc[0])) * 3.6
    conf = min(1.0, n / 15.0) / (1.0 + (rmse / 3.0) ** 2)
    if lsig is not None:
        conf /= 1.0 + lsig / 0.3
    if sp_sig is not None and sp0 > 1:
        conf /= 1.0 + (sp_sig / sp0) / 0.1
    seg = ShotSegment3D(
        start_frame=int(fr_cov[0]) if len(fr_cov) else int(frames_obs[0]),
        end_frame=int(fr_cov[-1]) if len(fr_cov) else int(frames_obs[-1]),
        t0=float(t_ref + tau_c), p0=Pc[0].copy(), v0=Vc[0].copy(),
        drag_length=float(np.exp(th7[6])), rmse_px=rmse, frames=fr_cov, positions=P_cov,
        landing_xy=landing_xy, landing_t=landing_t, net_crossing_height=net_h,
        confidence=float(conf))
    # non-contract extras (read via getattr; see landing_covariance / speed_profile)
    seg.theta_ref = th7.copy()              # (p0, v0, log L) at t_ref
    seg.t_ref = float(t_ref)                # first observed time
    seg.cov_theta = cov7
    seg.landing_cov = landing_cov
    seg.landing_sigma_m = lsig
    seg.speed0_sigma_kmh = sp_sig
    seg.n_obs = int(n)
    seg.inlier_frac = float(inl.mean())
    seg.end_reason = end_reason             # 'hit' | 'floor' | 'net' | 'unknown'
    seg.landing_extrapolated_xy = extrap_xy
    seg.net_crossing_y = nc[1] if nc is not None else None
    seg.fit_drag = bool(prob.fit_drag)
    seg.t_end = float(t_ref + tau_hi)       # landing / next contact / last observation
    return seg


def landing_covariance(segment: ShotSegment3D) -> np.ndarray | None:
    """2x2 landing-point covariance (m²): fit covariance propagated through the
    ODE plus calibration error; None if the flight has no landing point."""
    cov = getattr(segment, "landing_cov", None)
    if cov is not None or segment.landing_t is None:
        return cov
    cov7, th7 = getattr(segment, "cov_theta", None), getattr(segment, "theta_ref", None)
    if cov7 is None or th7 is None:
        return None
    t_ref = getattr(segment, "t_ref", segment.t0)
    return _landing_cov(th7, cov7, max(0.0, segment.landing_t - t_ref - 0.05))


def landing_sigma(segment: ShotSegment3D) -> float | None:
    """Largest 1-σ landing uncertainty (m), from landing_covariance."""
    cov = landing_covariance(segment)
    if cov is None or not np.all(np.isfinite(cov)):
        return None
    return float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))


def _seg_theta(segment: ShotSegment3D) -> tuple[np.ndarray, float]:
    th7 = getattr(segment, "theta_ref", None)
    if th7 is not None:
        return th7, float(getattr(segment, "t_ref", segment.t0))
    return np.r_[segment.p0, segment.v0, math.log(segment.drag_length)], float(segment.t0)


def speed_profile(segment: ShotSegment3D, dt: float = 1 / 120) -> tuple[np.ndarray, np.ndarray]:
    """(t seconds, speed km/h) from contact (t0) to landing / segment end."""
    th7, t_ref = _seg_theta(segment)
    t_end = segment.landing_t
    if t_end is None:
        t_end = getattr(segment, "t_end", None)
    if t_end is None:
        t_end = segment.t0 + 0.5
    ts = np.arange(segment.t0, max(t_end, segment.t0 + dt) + 1e-9, dt)
    _, V = _state_at(th7, ts - t_ref)
    return ts, np.linalg.norm(V, axis=1) * 3.6


def initial_speed_kmh(segment: ShotSegment3D) -> float:
    """Speed right after contact (the 'smash speed'), km/h."""
    return float(np.linalg.norm(segment.v0) * 3.6)


def contact_speed_kmh(segment: ShotSegment3D, t_contact: float) -> float:
    """Speed (km/h) of the fitted flight at an arbitrary contact time."""
    th7, t_ref = _seg_theta(segment)
    _, V = _state_at(th7, np.array([t_contact - t_ref]))
    return float(np.linalg.norm(V[0]) * 3.6)


# ─── in / out calls ─────────────────────────────────────────────────────────

SIGMA_FLOOR_M = 0.01            # model-error floor on the landing σ (spin, wind, L variation)
SIGMA_DEFAULT_M = 0.15          # when no covariance is available
# Calibration of the covariance σ along the boundary normal, measured on 347 synthetic
# + sim final shots (1.5 px noise, 10 % dropouts): z = error / σ has std 1.03 but
# MAD 0.74 (heavy tails from rare wrong-depth fits) → scale 0.85 keeps the tails
# covered while the old 2 cm floor alone made the calls ~2x too conservative
# (coverage 0.82 → 0.87 at 100 % accuracy of decided calls).
SIGMA_CAL = 0.85
SERVE_SIDE_MIN_Y = 0.05         # |feet y| (m) that decides the server's service court
SERVE_SIDE_P0_Y = 0.25          # |contact y| (m) that decides it when no feet are known


def _rally_margin(x: float, y: float, doubles: bool) -> tuple[float, np.ndarray]:
    """court_model.in_out_margin and the outward normal of the binding boundary."""
    half_w = cm.DOUBLES_HALF_WIDTH if doubles else cm.SINGLES_HALF_WIDTH
    mx = cm.HALF_LENGTH + cm.LINE_WIDTH / 2 - abs(x)
    my = half_w + cm.LINE_WIDTH / 2 - abs(y)
    if mx <= my:
        return float(mx), np.array([math.copysign(1.0, x), 0.0])
    return float(my), np.array([0.0, math.copysign(1.0, y)])


def serve_side_of(server_xy: np.ndarray, server_half: str | None = None) -> str:
    """Service court ('right' | 'left', from the server's own view facing the net)
    of a server standing at server_xy. Near server faces +x, so its right is −y."""
    x, y = float(server_xy[0]), float(server_xy[1])
    half = server_half or cm.side_of(x)
    return "right" if (y < 0) == (half == "near") else "left"


def serve_box(server_half: str, serve_side: str, doubles: bool) -> tuple[float, float, float, float]:
    """Legal receiving service court (x_min, x_max, y_min, y_max) on line centre-lines:
    diagonally opposite the server, between the short service line and the doubles
    long service line (doubles) or the baseline (singles), from the centre line to
    the doubles / singles side line."""
    tx = 1.0 if server_half == "near" else -1.0                   # target half
    server_ys = (-1.0 if serve_side == "right" else 1.0) * (1.0 if server_half == "near" else -1.0)
    ty = -server_ys                                                # diagonal
    back = cm.DOUBLES_LONG_SERVICE_X if doubles else cm.HALF_LENGTH
    half_w = cm.DOUBLES_HALF_WIDTH if doubles else cm.SINGLES_HALF_WIDTH
    x0, x1 = sorted((tx * cm.SHORT_SERVICE_X, tx * back))
    y0, y1 = sorted((0.0, ty * half_w))
    return x0, x1, y0, y1


def _serve_margin(x: float, y: float, server_half: str, serve_side: str,
                  doubles: bool) -> tuple[float, np.ndarray]:
    x0, x1, y0, y1 = serve_box(server_half, serve_side, doubles)
    h = cm.LINE_WIDTH / 2                      # every bounding line counts as in
    cands = [(x - (x0 - h), (-1.0, 0.0)), ((x1 + h) - x, (1.0, 0.0)),
             (y - (y0 - h), (0.0, -1.0)), ((y1 + h) - y, (0.0, 1.0))]
    m, nrm = min(cands, key=lambda c: c[0])
    return float(m), np.array(nrm)


def serve_margin(x: float, y: float, server_half: str, serve_side: str, doubles: bool) -> float:
    """Signed distance (m) of a serve landing from its service-court boundary
    (positive = in). Landing on the wrong half/court gives a large negative value."""
    return _serve_margin(x, y, server_half, serve_side, doubles)[0]


def call_in_out(segment: ShotSegment3D | None, doubles: bool, is_serve: bool = False,
                serve_side: str | None = None) -> dict:
    """IN/OUT call from the landing point and its covariance.

    Returns dict(call='in'|'out'|'unsure', margin_m, sigma_m, p_in); 'unsure' when
    |margin| < 2σ (σ along the normal of the binding boundary) or no landing.
    For serves the server's half comes from the contact point (p0); `serve_side`
    ('right'|'left') defaults to the server's feet (`hitter_feet_xy` extra set by
    reconstruct_segments), else to p0 when it is clearly off the centre line (the
    contact is often across the body, near y = 0); if neither decides, the call is
    made in both service courts and is 'unsure' where they disagree."""
    if segment is None or segment.landing_xy is None:
        return {"call": "unsure", "margin_m": None, "sigma_m": None, "p_in": None}
    x, y = float(segment.landing_xy[0]), float(segment.landing_xy[1])
    ambiguous = False
    if is_serve:
        half = cm.side_of(float(segment.p0[0]))
        side = serve_side
        feet = getattr(segment, "hitter_feet_xy", None)
        if side is None and feet is not None and abs(float(feet[1])) > SERVE_SIDE_MIN_Y:
            side = serve_side_of(np.asarray(feet, float), half)
        if side is None and abs(float(segment.p0[1])) > SERVE_SIDE_P0_Y:
            side = serve_side_of(segment.p0, half)
        if side is None:                           # try both courts
            (mr, nr), (ml, nl) = (_serve_margin(x, y, half, sd, doubles) for sd in ("right", "left"))
            ambiguous = (mr >= 0) != (ml >= 0)
            margin, nrm = (mr, nr) if mr >= ml else (ml, nl)
        else:
            margin, nrm = _serve_margin(x, y, half, side, doubles)
    else:
        margin, nrm = _rally_margin(x, y, doubles)
    cov = landing_covariance(segment)
    if cov is not None and np.all(np.isfinite(cov)):
        sigma = SIGMA_CAL * float(np.sqrt(max(nrm @ cov @ nrm, 0.0)))
    else:
        sigma = SIGMA_DEFAULT_M
    sigma = math.hypot(sigma, SIGMA_FLOOR_M)
    call = "unsure" if abs(margin) < 2 * sigma or ambiguous else ("in" if margin > 0 else "out")
    p_in = 0.5 * (1.0 + math.erf(margin / (sigma * math.sqrt(2))))
    return {"call": call, "margin_m": margin, "sigma_m": sigma, "p_in": p_in}


# ─── rally-level reconstruction ─────────────────────────────────────────────

MAX_FLIGHT_S = 2.8              # longest plausible flight (high serve / defensive lift)
MAX_GAP_S = 1.2                 # a longer detection gap ends a flight with no next hit
REST_S = 0.25                   # stationary this long → on the floor
MIN_OBS = 6
# A flight whose observations stop at a presentation-clock gap has neither a seen end nor
# an end constraint; its speed is kept only when the fit pins it down (σ ≤ 15 % of it).
# Measured: normal chained / open fits 5–10 %; the gap-cut arcs of the robustness repros
# 23 % (81 km/h vs 54 true) and 37 % (213 km/h vs 54 true).
GAP_SPEED_REL_SIGMA = 0.15
FEET_SIGMA_M = 0.7              # feet → contact point (arm + racket reach)
FEET_SIGMA_RECV_M = 0.9


def _track_weights(track: Track2D) -> np.ndarray:
    """Per-frame fit weight: real detections by confidence, filled frames 0."""
    xy = np.asarray(track.xy, float)
    ok = np.isfinite(xy).all(1)
    vis = np.asarray(track.visible, bool) if track.visible is not None else ok
    interp = (np.asarray(track.interpolated, bool) if track.interpolated is not None
              else np.zeros(len(ok), bool))
    conf = np.asarray(track.conf, float) if track.conf is not None else np.ones(len(ok))
    w = np.where(vis & ~interp & ok, np.clip(np.nan_to_num(conf, nan=1.0), 0.3, 1.0), 0.0)
    if not w.any():                                # no flags: trust every finite point
        w = ok.astype(float)
    return w


def _rest_start(t: np.ndarray, xy: np.ndarray, w: np.ndarray, i0: int, i1: int,
                rest_px: float) -> int | None:
    """First index in [i0, i1) from which every detection for REST_S stays within
    rest_px (the shuttle lies on the floor), or None."""
    idx = np.nonzero(w[i0:i1] > 0)[0] + i0
    for a, i in enumerate(idx):
        j = a + 1
        while j < len(idx) and t[idx[j]] - t[i] <= REST_S:
            if np.hypot(*(xy[idx[j]] - xy[i])) > rest_px:
                break
            j += 1
        else:
            if j - a >= 4 and t[idx[j - 1]] - t[i] >= 0.6 * REST_S:
                return int(i)
    return None


def _near_border(uv: np.ndarray, size: tuple[int, int], frac: float = 0.1) -> bool:
    """Pixel within frac of the image size from the top / left / right edge (the
    shuttle can leave the view there; the bottom edge is the near floor)."""
    w, h = size
    return bool(uv[1] < frac * h or uv[0] < frac * w or uv[0] > (1 - frac) * w)


class _Players:
    """Feet positions by frame (nearest observation within max_df frames)."""

    def __init__(self, players: list[PlayerObservation] | None):
        self.by_frame: dict[int, list[PlayerObservation]] = {}
        for o in players or []:
            if o.foot_world is not None and np.all(np.isfinite(o.foot_world[:2])):
                self.by_frame.setdefault(int(o.frame), []).append(o)
        self.frames = np.array(sorted(self.by_frame), int)

    def feet(self, frame: int, side: str, track_id: int | None, uv: tuple[float, float],
             max_df: int = 6) -> np.ndarray | None:
        if not len(self.frames):
            return None
        k = int(np.clip(np.searchsorted(self.frames, frame), 0, len(self.frames) - 1))
        cands = [f for f in (self.frames[max(k - 1, 0)], self.frames[k]) if abs(f - frame) <= max_df]
        if not cands:
            return None
        obs = self.by_frame[min(cands, key=lambda f: abs(f - frame))]
        if track_id is not None:
            for o in obs:
                if o.track_id == track_id:
                    return np.asarray(o.foot_world[:2], float)
        pool = [o for o in obs if o.side == side] if side in ("near", "far") else []
        if not pool:
            return None
        if len(pool) > 1:                          # doubles: player whose box is nearest the shuttle
            def dpx(o: PlayerObservation) -> float:
                x1, y1, x2, y2 = o.bbox
                return math.hypot(uv[0] - min(max(uv[0], x1), x2), uv[1] - min(max(uv[1], y1), y2))
            pool.sort(key=dpx)
        return np.asarray(pool[0].foot_world[:2], float)


def _closest_approach(seg_in: ShotSegment3D, seg_out: ShotSegment3D, t_lo: float,
                      t_hi: float) -> tuple[float, float] | None:
    """Time in [t_lo, t_hi] where the incoming and outgoing fitted flights are
    closest in 3-D, and that distance (m)."""
    if t_hi <= t_lo:
        return None
    ts = np.linspace(t_lo, t_hi, max(int((t_hi - t_lo) * 2000), 8))
    th_a, ta = _seg_theta(seg_in)
    th_b, tb = _seg_theta(seg_out)
    Pa, _ = _state_at(th_a, ts - ta)
    Pb, _ = _state_at(th_b, ts - tb)
    d = np.linalg.norm(Pa - Pb, axis=1)
    k = int(np.argmin(d))
    return float(ts[k]), float(d[k])


def _reepoch(seg: ShotSegment3D, t_c: float) -> None:
    """Move the reported contact state (t0, p0, v0) of a fitted segment to t_c."""
    th7, t_ref = _seg_theta(seg)
    tau = t_c - t_ref
    if not (-MAX_BACK_EXTRAP <= tau <= 0.05):
        return
    P, V = _state_at(th7, np.array([tau]))
    seg.t0, seg.p0, seg.v0 = float(t_c), P[0].copy(), V[0].copy()
    cov7 = getattr(seg, "cov_theta", None)
    if cov7 is not None and np.all(np.isfinite(cov7)):
        seg.speed0_sigma_kmh = _speed_sigma(th7, cov7, tau) * 3.6


def refine_hit_time(track: Track2D, approx_frame: int, camera: CameraModel,
                    window: int = 6) -> tuple[int, float]:
    """Refine a contact time: choose the split within ±window frames where
    separate smooth fits before/after explain the 2-D track best, then fit the
    drag ODE to each side and take the time where the two 3-D flights meet
    (closest approach). Returns (frame, t); falls back to the input frame."""
    t = np.asarray(track.t, float)
    xy = np.asarray(track.xy, float)
    w = _track_weights(track)
    n = len(t)
    f0 = int(np.clip(approx_frame, 0, n - 1))
    ok = w > 0
    span = 0.25
    best = None
    # presentation-clock gaps split the track: neither side of a split may reach
    # across one (span id per frame; all zeros on a gap-free clip)
    sid = np.zeros(n, np.int64)
    gaps = pts_gaps(t)
    if len(gaps):
        sid[gaps] = 1
        sid = np.cumsum(sid)
    for f in range(max(1, f0 - window), min(n - 1, f0 + window) + 1):
        if sid[f] != sid[f - 1]:
            continue                                  # a split AT a gap pairs across it
        a = np.nonzero(ok & (t < t[f]) & (t >= t[f] - span) & (sid == sid[f]))[0]
        b = np.nonzero(ok & (t >= t[f]) & (t <= t[f] + span) & (sid == sid[f]))[0]
        if len(a) < 4 or len(b) < 4:
            continue
        err = 0.0
        for idx in (a, b):
            tt = t[idx] - t[f]
            A = np.column_stack([np.ones_like(tt), tt, tt * tt])
            c, *_ = np.linalg.lstsq(A, xy[idx], rcond=None)
            e = np.linalg.norm(A @ c - xy[idx], axis=1)
            err += float(np.minimum(e, 15.0).sum())
        if best is None or err < best[0]:
            best = (err, f)
    if best is None:
        return f0, float(t[f0])
    f = best[1]
    a = np.nonzero(ok & (t < t[f]) & (t >= t[f] - 0.6) & (sid == sid[f]))[0]
    b = np.nonzero(ok & (t >= t[f]) & (t <= t[f] + 0.6) & (sid == sid[f]))[0]
    t_c = float(0.5 * (t[a[-1]] + t[b[0]])) if len(a) and len(b) else float(t[f])
    if len(a) >= MIN_OBS and len(b) >= MIN_OBS:
        s_in = fit_segment(t[a], xy[a], w[a], camera, SegmentPrior(), frames=a)
        s_out = fit_segment(t[b], xy[b], w[b], camera, SegmentPrior(), frames=b)
        if s_in is not None and s_out is not None:
            ca = _closest_approach(s_in, s_out, float(t[a[-1]]) - 0.01, float(t[b[0]]) + 0.01)
            if ca is not None and ca[1] < 0.5:
                t_c = ca[0]
    frame = int(np.argmin(np.abs(t - t_c)))
    return frame, float(t_c)


@dataclass
class _Piece:
    hit: HitEvent
    i0: int                          # track index range [i0, i1)
    i1: int
    end: str                         # 'hit' | 'floor' | 'open'
    end_t: float | None
    next_hit: HitEvent | None
    rest_xy: np.ndarray | None = None
    seg: ShotSegment3D | None = None
    t_contact: float | None = None
    at_gap: bool = False             # cut at a presentation-clock gap (end unseen)
    gap_start: bool = False          # hit before a gap, first frame after it (contact unseen)


def _fit_piece(pc: _Piece, t: np.ndarray, xy: np.ndarray, w: np.ndarray, frames: np.ndarray,
               camera: CameraModel, prior: SegmentPrior,
               init: tuple[np.ndarray, float] | None = None) -> ShotSegment3D | None:
    sl = slice(pc.i0, pc.i1)
    ts, xs, ws, fs = t[sl], xy[sl], w[sl].copy(), frames[sl]
    seg = fit_segment(ts, xs, ws, camera, prior, frames=fs, init=init)
    if seg is None or prior.end != "floor" or seg.landing_t is None:
        return seg
    # drop post-contact (skid/bounce) detections the robust loss had to absorb
    obs_t = ts[ws > 0]
    dt = float(np.median(np.diff(obs_t))) if len(obs_t) > 1 else 1 / 30
    late = (ws > 0) & (ts > seg.landing_t + dt)
    if late.any() and (ws > 0).sum() - late.sum() >= MIN_OBS:
        ws[late] = 0.0
        seg2 = fit_segment(ts, xs, ws, camera, prior, frames=fs,
                           init=(seg.theta_ref, seg.t_ref))
        seg = seg2 if seg2 is not None else seg
    return seg


_TP_CTRL = None


def _blas_single_thread():
    """Tiny matrices: OpenBLAS thread spin-up costs more than it saves (measured:
    4.7x the CPU time and 1.4x the wall time per fit with threads on)."""
    global _TP_CTRL
    try:
        if _TP_CTRL is None:
            from threadpoolctl import ThreadpoolController
            _TP_CTRL = ThreadpoolController()
        return _TP_CTRL.limit(limits=1, user_api="blas")
    except Exception:               # threadpoolctl missing or unsupported
        return contextlib.nullcontext()


def reconstruct_segments(track: Track2D, hits: list[HitEvent], camera: CameraModel,
                         meta: VideoMeta, players: list[PlayerObservation] | None = None
                         ) -> list[ShotSegment3D]:
    """Fit a 3-D flight to every hit → (next hit | floor | out of view) piece.

    Pass 1 fits each piece with side/height priors; contact times are then refined
    where consecutive flights meet in 3-D; pass 2 refits with the hitter's and
    receiver's feet (if players are given) at the refined contact times. Pieces
    ending without a next hit look for the shuttle at rest on the floor (landing
    bracket) and extrapolate the ODE to z = 0. Segments carry `hit_frame`,
    `end_reason` and `rest_xy` (floor point where the shuttle came to rest) extras.
    """
    with _blas_single_thread():
        return _reconstruct(track, hits, camera, meta, players)


def _reconstruct(track: Track2D, hits: list[HitEvent], camera: CameraModel, meta: VideoMeta,
                 players: list[PlayerObservation] | None) -> list[ShotSegment3D]:
    t = np.asarray(track.t, float)
    xy = np.asarray(track.xy, float).reshape(-1, 2)
    n = len(t)
    if n < MIN_OBS or not hits or camera is None:
        return []
    w = _track_weights(track)
    ts_meta = np.asarray(getattr(meta, "timestamps", []), float) if meta is not None else np.array([])
    if len(ts_meta) == n and np.allclose(ts_meta, t, atol=1e-4, equal_nan=True):
        frames = np.arange(n)
    elif len(ts_meta):
        frames = np.clip(np.searchsorted(ts_meta, t - 1e-6), 0, len(ts_meta) - 1)
    else:
        frames = np.arange(n)
    width = float(getattr(meta, "width", 0) or camera.image_size[0])
    rest_px = max(4.0, 6.0 * width / 1920.0)
    hits = sorted([h for h in hits if np.isfinite(h.t)], key=lambda h: h.t)
    feet = _Players(players)

    # presentation-clock gaps (video_io.pts_gaps): a flight is never fitted across one.
    # A piece keeps only the frames of its hit's gap-free span; a hit inside a gap
    # starts its piece after the gap (contact unobserved, as when first seen late).
    gaps = pts_gaps(t)
    pieces: list[_Piece] = []
    for k, h in enumerate(hits):
        nxt = hits[k + 1] if k + 1 < len(hits) else None
        i0 = int(np.searchsorted(t, h.t + 1e-6))
        g = int(np.searchsorted(gaps, i0, side="right"))
        i_gap = int(gaps[g]) if g < len(gaps) else n    # first frame after the next gap
        # contact before a gap, first frame after it: the contact was never seen
        gap_start = g > 0 and int(gaps[g - 1]) == i0
        t_lim = h.t + MAX_FLIGHT_S
        chained = nxt is not None and nxt.t <= t_lim
        if chained and i_gap < n and nxt.t >= t[i_gap - 1] + 1e-6:
            chained = False                          # the next hit is beyond the gap
        i1 = int(np.searchsorted(t, nxt.t - 1e-6)) if chained else int(np.searchsorted(t, t_lim))
        at_gap = not chained and i_gap < n and i1 >= i_gap   # flight window reaches the gap
        i1 = min(i1, i_gap)
        valid = np.nonzero(w[i0:i1] > 0)[0] + i0
        if not chained and len(valid) > 1:        # a long gap ends the flight ...
            for g in np.nonzero(np.diff(t[valid]) > MAX_GAP_S)[0]:
                a, b = valid[g], valid[g + 1]     # ... unless it left and re-entered the frame
                if not (_near_border(xy[a], camera.image_size)
                        and _near_border(xy[b], camera.image_size)):
                    i1 = int(a + 1)
                    at_gap = False                # a detection gap ended it first
                    break
        end, end_t, rest_xy = ("hit", nxt.t, None) if chained else ("open", None, None)
        if not chained:
            r = _rest_start(t, xy, w, i0, i1, rest_px)
            if r is not None:
                at_gap = False                    # it came to rest before the gap
            if r is not None and np.sum(w[i0:r] > 0) >= MIN_OBS:
                end, end_t = "floor", float(t[r])
                p_rest = camera.pixel_to_floor(xy[r:r + 1])[0]
                rest_xy = p_rest[:2] if np.all(np.isfinite(p_rest)) else None
                i1 = r
        if np.sum(w[i0:i1] > 0) < MIN_OBS:
            continue
        pieces.append(_Piece(h, i0, i1, end, end_t, nxt if chained else None, rest_xy,
                             at_gap=at_gap and end == "open", gap_start=gap_start))

    def prior_for(pc: _Piece, t_c: float, with_feet: bool) -> SegmentPrior:
        side = pc.hit.hitter if pc.hit.hitter in ("near", "far") else None
        # contact inside unseen time: no contact epoch (p0/v0 at the first sighting,
        # never back-extrapolated across the gap)
        pr = SegmentPrior(start_t=None if pc.gap_start else t_c, expected_side=side)
        if pc.end == "hit":
            pr.end, pr.end_t = "hit", pc.end_t
        elif pc.end == "floor":
            pr.end, pr.end_t = "floor", pc.end_t
            pr.rest_xy_world = pc.rest_xy
        pr.end_at_gap = pc.at_gap
        if with_feet:
            f = feet.feet(pc.hit.frame, pc.hit.hitter, pc.hit.player_track_id, pc.hit.xy_px)
            if f is not None and not pc.gap_start:     # feet = contact point, unseen here
                pr.start_xy_world, pr.start_xy_sigma = f, FEET_SIGMA_M
            if pc.next_hit is not None:
                nh = pc.next_hit
                g = feet.feet(nh.frame, nh.hitter, nh.player_track_id, nh.xy_px)
                if g is not None:
                    pr.end_xy_world, pr.end_xy_sigma = g, FEET_SIGMA_RECV_M
        return pr

    # pass 1: no feet priors; contact time = detected hit time
    for pc in pieces:
        pc.t_contact = float(pc.hit.t)
        pc.seg = _fit_piece(pc, t, xy, w, frames, camera, prior_for(pc, pc.t_contact, False))

    # contact-time refinement where an incoming and an outgoing flight meet
    for a, b in zip(pieces[:-1], pieces[1:]):
        if a.next_hit is not b.hit or a.seg is None or b.seg is None or b.gap_start:
            continue
        ia = np.nonzero(w[a.i0:a.i1] > 0)[0]
        ib = np.nonzero(w[b.i0:b.i1] > 0)[0]
        if not len(ia) or not len(ib):
            continue
        t_last, t_first = float(t[a.i0 + ia[-1]]), float(t[b.i0 + ib[0]])
        ca = _closest_approach(a.seg, b.seg, t_last - 0.005, t_first + 0.005)
        if ca is not None and ca[1] < 0.6:
            b.t_contact = ca[0]
            a.end_t = ca[0]

    # pass 2: refined contact times (+ feet priors) warm-started from pass 1
    use_feet = len(feet.frames) > 0
    out: list[ShotSegment3D] = []
    for pc in pieces:
        moved = abs(pc.t_contact - pc.hit.t) > 1e-4 or (pc.end == "hit" and pc.end_t != pc.next_hit.t)
        if pc.seg is not None and (use_feet or moved):
            pr = prior_for(pc, pc.t_contact, use_feet)
            seg2 = _fit_piece(pc, t, xy, w, frames, camera, pr, init=(pc.seg.theta_ref, pc.seg.t_ref))
            if seg2 is not None:
                pc.seg = seg2
            elif not pc.gap_start:                    # never re-epoch across a gap
                _reepoch(pc.seg, pc.t_contact)
        elif pc.seg is None and use_feet:
            pc.seg = _fit_piece(pc, t, xy, w, frames, camera, prior_for(pc, pc.t_contact, True))
        if pc.seg is None:
            continue
        seg = pc.seg
        if pc.at_gap:                              # end unseen: only a well-posed speed survives
            sp_sig = getattr(seg, "speed0_sigma_kmh", None)
            if sp_sig is None or not np.isfinite(sp_sig) or sp_sig > GAP_SPEED_REL_SIGMA * seg.speed0_kmh:
                continue
        seg.hit_frame = int(pc.hit.frame)
        seg.hit_t_refined = float(pc.t_contact)
        f = feet.feet(pc.hit.frame, pc.hit.hitter, pc.hit.player_track_id, pc.hit.xy_px)
        seg.hitter_feet_xy = None if f is None else (float(f[0]), float(f[1]))
        seg.rest_xy = None if pc.rest_xy is None else (float(pc.rest_xy[0]), float(pc.rest_xy[1]))
        seg.ends_at_pts_gap = bool(pc.at_gap)       # end unseen: a clock hole follows
        seg.starts_after_pts_gap = bool(pc.gap_start)   # contact unseen: p0/v0 at first sighting
        if pc.end == "hit":
            seg.end_reason = "hit"
        out.append(seg)
    return out
