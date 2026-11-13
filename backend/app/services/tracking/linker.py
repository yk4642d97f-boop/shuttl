"""
Shuttle linker: noisy per-frame detector candidates → one clean 2-D track.

All LinkerConfig lengths are fractions of the image diagonal D and all times
are seconds (VideoMeta.timestamps), so one config serves 720p/1080p/4K, any
fps and variable frame rate.

1. Static false-positive map. Candidates are binned per ~20 s block (3×3 cell
   neighbourhood, one count per frame). Pass 1 penalises cells busy in
   30–60 % of frames; pass 2 (after a first association) counts only frames
   where the chosen shuttle is present *and far away* — a light / line
   fragment / floor shuttle keeps firing while the shuttle flies elsewhere; a
   shuttle held before the serve is never "elsewhere" and is not penalised.
2. Global association: Viterbi over candidates, O(N·G·k²), numba-JIT (pure
   Python fallback). Predecessors are candidates in the previous G frames
   (G = max_link_gap_s, so miss bursts are bridged by one link, miss_cost per
   skipped frame). Three states per candidate: tracking (2nd-order motion NLL:
   prediction from the pair velocity with a drag-aware along-track interval,
   variance growing with dt), fresh (after a birth or a kink; a kink costs
   exactly kink_cost wherever it falls between frames and may skip ≤
   kink_max_gap_s, else it is a birth) and young (first velocity check after a hit, where drag
   decelerates hardest). Unary = −(det_reward + logit(score)) + static cost.
   A path may end and restart anywhere via a running best-finished path
   (dead time is free, a birth costs birth_cost). Weak candidates (below
   the detector's decision threshold) may only continue shuttle flight: a
   piece between births/kinks needs ≥ min_strong strong detections, weak
   ones must lie ≤ weak_reach_s from one, a leading weak run must fly, a
   trailing one must keep the flight direction, and a weak pick never beats
   a strong non-static candidate of the same frame (detector ghost of the
   previous position). Rejected weak runs are banned and the DP re-run, so
   shoes, heads and rackets in dead time are not tracked. No link crosses a
   scene cut (passed in, or inferred when ≥ cut_min_cells persistent static
   spots end/start at once). Chains shorter than min_chain_len are dropped.
   Fills that hold a strong on-trajectory detection ≥ 6σ away take it (real
   broadcast clocks are irregular: every other frame can lag half a step).
   Track2D.raw_xy is the adopted detection itself. This is a tracklet graph solved exactly: short consistent chains
   are rewarded, isolated FPs never pay for their birth.
3. Repeated frames (25p broadcast in a 30p file, phones under load): a frame
   whose detection sits on the previous frame's point while the path moves
   fast and straight through it shows the previous image. Such frames are
   removed, the rest re-timed on a content clock (period-exact moving mean
   when the repeats are periodic), re-associated, and the repeat slots copy
   the previous slot's state.
4. Kinks (hits): at every pair of consecutive observations a GLRT tests a
   velocity step — a cubic through a ±kink_window_s window (absorbs drag,
   gravity, perspective) plus a hinge at the pair midpoint with a free
   post-hit curvature; z = |Δv|/(σ·sd) is Rayleigh under "no hit" (σ from the
   fit residuals). Accepted at z > kink_z, then NMS, then localised by min RSS
   of the hinge position: a coarse pass over the scan window (high fps) and a
   fine pass (±kink_loc_s, short line+curvature windows) — the GLRT argmax is
   often a pair off, min-RSS lands on the right frame pair ~75 % and within one
   frame ~95 % of the time.
5. Piecewise smoothing: flight pieces split at kinks and at gaps > max_gap_s;
   each is smoothed by a robust (IRLS Huber + hard gate) penalised LS fit
   with an integrated-white-jerk prior whose density follows the speed (drag
   jerk ∝ v³/L²) — the batch MAP equivalent of a constant-acceleration RTS
   smoother, solved for the whole video as one banded Cholesky system. Where
   the fit leaves a run of ≥ 2 large residuals next to a pair with z >
   kink_z_refine, a kink is inserted and the fit repeated (≤ refine_rounds).
   Nothing is ever smoothed across a kink.
6. Gap filling: unobserved frames inside a piece are free variables of the
   same fit (the smoother's prediction), flagged `interpolated`, conf 0. At a
   kink each side is extrapolated ≤ max_extrap_frames toward the hit time.
   Longer gaps and dead time stay NaN.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solveh_banded

from app.services.tracking.data_types import ShuttleCandidate, Track2D, VideoMeta
from app.services.tracking.video_io import pts_gaps

try:                                    # optional JIT for the association DP
    import numba as _numba
    _HAVE_NUMBA = True
except Exception:                       # pragma: no cover - fallback path
    _numba = None
    _HAVE_NUMBA = False


def _jit(f):
    """numba.njit with on-disk cache when possible; identity without numba."""
    if not _HAVE_NUMBA:
        return f
    try:
        return _numba.njit(cache=True, nogil=True)(f)
    except Exception:                   # no cache locator (e.g. zipped install)
        return _numba.njit(nogil=True)(f)


_INF = 1e30


@dataclass
class LinkerConfig:
    """All lengths are fractions of the image diagonal D (1080p: D = 2203 px),
    all times in seconds, costs in nats (−log-likelihood units)."""

    # ── candidates ─────────────────────────────────────────────────────────
    max_candidates: int = 5          # top-k per frame kept (by score)
    min_score: float = 0.05          # ignore weaker candidates
    # ── association DP ─────────────────────────────────────────────────────
    max_speed: float = 8.0           # D/s hard gate (≈ 400 km/h near the camera)
    max_link_gap_s: float = 0.30     # longest burst of misses bridged by one link
    sigma_px: float = 1.2e-3         # detector noise (1080p: 2.6 px)
    accel_sigma: float = 1.5         # D/s² unmodelled accel. (gravity/perspective)
    drag_length_min: float = 0.12    # D, smallest image-space drag length
    drag_length_young: float = 0.015 # D, same for the 1st velocity check after a hit
    speed_up_max: float = 0.25       # along-track overshoot allowed (perspective)
    kink_cost: float = 5.0           # cost of a kink (hit): velocity is re-learned
    kink_max_gap_s: float = 0.10     # a kink link may skip at most this long (else: birth)
    first_cost: float = 1.0          # motion cost of a link with no velocity yet
    birth_cost: float = 10.0         # starting a new chain
    miss_cost: float = 0.35          # per skipped frame inside a chain
    det_reward: float = 2.0          # base reward for adopting a candidate
    score_weight: float = 1.0        # × logit(score)
    min_chain_len: int = 4           # shorter chains are discarded
    # ── weak candidates (below the detector's decision threshold) ─────────
    strong_score: float = 0.5        # detector decision threshold (TrackNetV3 / WASB)
    min_strong: int = 2              # strong obs a flight piece needs (piece = between
                                     # births / kinks) …
    weak_reach_s: float = 0.3        # … and weak obs must be this close to one of them
    weak_fly_speed: float = 0.3      # D/s: a weak-only piece is kept only if it flies
    weak_fly_min_s: float = 0.25     # this fast (median) for this long
    weak_ban_rounds: int = 3         # re-associate after banning rejected weak runs
    readopt_gate: float = 0.014      # D: a fill takes a strong detection this close
    # ── scene cuts (never linked or filled across) ────────────────────────
    cut_detect: bool = True          # infer cuts from the static-candidate fingerprint
    cut_min_cells: int = 3           # persistent static spots that change at once
    # ── static distractors ─────────────────────────────────────────────────
    static_cell: float = 6e-3        # D, histogram cell (1080p: 13 px)
    static_block_s: float = 20.0     # temporal block of the occupancy histogram
    static_min_s: float = 10.0       # occupancy denominator floor (short clips)
    static_occ_lo: float = 0.30      # plain occupancy: penalty starts
    static_occ_hi: float = 0.60      # plain occupancy: full penalty
    static_far: float = 0.03         # D, "shuttle is elsewhere" distance
    static_min_present_s: float = 1.0  # denominator floor of the 2nd pass
    static_lo: float = 0.10          # occupancy-while-elsewhere: penalty starts
    static_hi: float = 0.30          # occupancy-while-elsewhere: full penalty
    static_cost: float = 6.0         # full penalty (nats)
    # ── kinks (hits) ───────────────────────────────────────────────────────
    kink_window_s: float = 0.12      # GLRT window on each side of a pair
    kink_min_pts: int = 3            # points needed per side
    kink_max_pts: int = 12           # cap on window points (high fps)
    kink_deg: int = 3                # polynomial degree of the no-hit flight model
    kink_z: float = 4.5              # GLRT threshold (Rayleigh z; misfit tail ≈ 5–6)
    kink_z_refine: float = 3.5       # lower threshold where residuals demand a kink
    kink_min_dv: float = 0.05        # D/s absolute floor on the velocity step
    kink_nms_s: float = 0.15         # min separation of hits
    kink_loc_s: float = 0.04         # min-RSS localisation search radius
    kink_loc_pts: int = 5            # points per side for localisation (short: post-hit
                                     # drag deceleration is not polynomial for long)
    kink_loc_deg: int = 1            # localisation model: line + free curvature after τ
    kink_loc_coarse: bool = True     # first search the whole scan window when it
                                     # is ≥ 2× the localisation window (high fps)
    # ── smoothing / gap filling ────────────────────────────────────────────
    max_gap_s: float = 0.25          # fill gaps up to this inside a piece
    max_extrap_frames: int = 3       # extrapolation toward a hit time
    jerk_floor: float = 60.0         # D/s³ baseline jerk (perspective, gravity)
    jerk_drag_len: float = 0.25      # D, drag length used in the v³/L² jerk prior
    jerk_corr_s: float = 0.03        # correlation time of the jerk process
    huber_k: float = 2.0             # sigmas, IRLS Huber knee
    outlier_k: float = 5.0           # sigmas, hard rejection on residuals
    irls_iters: int = 4
    refine_kinks: bool = True        # insert missed kinks from residual runs
    refine_rounds: int = 3           # max rounds of residual-driven kink insertion
    # ── repeated video frames (25p in a 30p container, 24p in 30p, …) ─────
    repeat_detect: bool = True       # detect periodic repeats and use a content clock
    repeat_min_move: float = 2e-3    # D/frame, neighbours must move this much
    repeat_min_count: int = 4        # seed repeats needed before inferring their period
    kink_resid_k: float = 3.0        # sigmas, "large" residual for a run


# ══════════════════════════════════════════════════════════════════════════════
# Association DP (numba kernels; plain Python if numba is missing)
# prm = [vmax, sig2, acc2, lmin, up, K, c_first, c_birth, c_miss, lmin_young, kink_gap]
# ══════════════════════════════════════════════════════════════════════════════

@_jit
def _motion(dx, dy, dt, vx, vy, dtp, lmin, prm):
    """NLL of a displacement (dx, dy) over dt given the predecessor's pair
    velocity (vx, vy) measured over dtp; `lmin` = image-space drag length
    bounding how much the along-track speed may decay (uncapped; gating is
    done by the caller)."""
    sig2, acc2, up = prm[1], prm[2], prm[4]
    r = dt / dtp
    var = (sig2 * (1.0 + (1.0 + r) * (1.0 + r) + r * r)
           + acc2 * (0.5 * dt * (dt + dtp)) ** 2)
    s = np.sqrt(vx * vx + vy * vy)
    if s < 1e-9:
        e2 = dx * dx + dy * dy
    else:
        ux, uy = vx / s, vy / s
        par = dx * ux + dy * uy
        perp = dy * ux - dx * uy
        # speed at the predecessor may be lower than the pair speed (drag),
        # then decays further: along-track travel ∈ [lo, hi]
        sp = s / (1.0 + 0.5 * s * dtp / lmin)
        lo = lmin * np.log1p(sp * dt / lmin)
        hi = s * dt * (1.0 + up)
        ep = lo - par if par < lo else (par - hi if par > hi else 0.0)
        e2 = ep * ep + perp * perp
    return 0.5 * e2 / var


@_jit
def _viterbi(off, fidx, xs, ys, u, t, prm, gap, shot0):
    """Min-cost set of chains through the candidates (see module docstring).

    Three states per candidate: 1 = fresh (entered by a birth or a kink; the
    next link pays only `first_cost` and has no velocity to check), 2 = young
    (one link after fresh: velocity known but measured right after a hit,
    where quadratic drag decelerates hardest → loose along-track bound
    drag_length_young), 0 = tracking (full motion NLL). A kink costs exactly
    `kink_cost` wherever it falls between two frames, so the first points
    after a hit are never dropped to dodge a second motion penalty. shot0[f]
    = first frame of f's shot: no link crosses a scene cut. Returns chosen
    candidate per frame (−1 none), chain-start flags, kink flags."""
    n = off.shape[0] - 1
    m = xs.shape[0]
    vmax, sig2, lmin, kcap, cfirst = prm[0], prm[1], prm[3], prm[5], prm[6]
    cbirth, cmiss, lyoung, kgap = prm[7], prm[8], prm[9], int(prm[10])
    gate_pad = 4.0 * np.sqrt(sig2)
    D = np.full((m, 3), _INF)
    bpi = np.full((m, 3), -1, np.int64)
    bps = np.full((m, 3), -1, np.int64)
    bl = np.full(m, -1, np.int64)           # birth: end of the previous finished path
    bls = np.full(m, -1, np.int64)
    vx = np.zeros((m, 3))
    vy = np.zeros((m, 3))
    vdt = np.ones((m, 3))
    vb = np.zeros(m)                        # fresh state: post-kink speed lower bound
    vbdt = np.ones(m)
    fin = np.zeros(n + 1)                   # best finished path over frames < f
    fin_arg = np.full(n + 1, -1, np.int64)
    fin_st = np.full(n + 1, -1, np.int64)
    for f in range(n):
        for j in range(off[f], off[f + 1]):
            b1 = fin[f] + cbirth + u[j]     # fresh via birth (or a kink below)
            b1i, b1s = -1, -1
            b1v, b1dt = 0.0, 1.0
            b0, b0i, b0s = _INF, -1, -1     # tracking
            b2, b2i, b2s = _INF, -1, -1     # young
            v0x, v0y, v0t, v2x, v2y, v2t = 0.0, 0.0, 1.0, 0.0, 0.0, 1.0
            g0 = max(f - gap, shot0[f])
            for g in range(f - 1, g0 - 1, -1):
                dt = max(t[f] - t[g], 1e-6)
                skip = cmiss * (f - g - 1) + u[j]
                lim = vmax * dt + gate_pad
                for i in range(off[g], off[g + 1]):
                    dmin = min(D[i, 0], D[i, 1], D[i, 2])
                    if dmin + skip >= max(b0, b2) and dmin + skip + kcap >= b1:
                        continue
                    dx, dy = xs[j] - xs[i], ys[j] - ys[i]
                    if dx * dx + dy * dy > lim * lim:
                        continue
                    if f - g <= kgap and dmin + skip + kcap < b1:   # kink
                        b1 = dmin + skip + kcap
                        b1i = i
                        b1s = 0 if D[i, 0] == dmin else (1 if D[i, 1] == dmin else 2)
                        # the hit is somewhere in (t_g, t_f): whatever the incoming
                        # speed cannot explain was covered at the outgoing speed
                        sin = 0.0 if b1s == 1 else np.sqrt(vx[i, b1s] ** 2 + vy[i, b1s] ** 2)
                        b1v = max(np.sqrt(dx * dx + dy * dy) / dt - sin * (1.0 + prm[4]), 0.0)
                        b1dt = dt
                    c = D[i, 1] + skip + cfirst                  # fresh → young
                    if vb[i] > 0.0:              # drag can slow it, not stop it
                        sp = vb[i] / (1.0 + 0.5 * vb[i] * vbdt[i] / lyoung)
                        lo = lyoung * np.log1p(sp * dt / lyoung)
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < lo:
                            r = dt / vbdt[i]
                            var = sig2 * (1.0 + (1.0 + r) * (1.0 + r) + r * r)
                            c += 0.5 * (lo - d) * (lo - d) / var
                    if c < b2:
                        b2, b2i, b2s = c, i, 1
                        v2x, v2y, v2t = dx / dt, dy / dt, dt
                    for s in (0, 2):                             # → tracking
                        base = D[i, s] + skip
                        if base >= b0:
                            continue
                        c = _motion(dx, dy, dt, vx[i, s], vy[i, s], vdt[i, s],
                                    lmin if s == 0 else lyoung, prm)
                        if c < kcap and base + c < b0:
                            b0, b0i, b0s = base + c, i, s
                            v0x, v0y, v0t = dx / dt, dy / dt, dt
            D[j, 0], bpi[j, 0], bps[j, 0] = b0, b0i, b0s
            D[j, 1], bpi[j, 1], bps[j, 1] = b1, b1i, b1s
            D[j, 2], bpi[j, 2], bps[j, 2] = b2, b2i, b2s
            vb[j], vbdt[j] = b1v, b1dt
            if b1i < 0:
                bl[j], bls[j] = fin_arg[f], fin_st[f]
            vx[j, 0], vy[j, 0], vdt[j, 0] = v0x, v0y, v0t
            vx[j, 2], vy[j, 2], vdt[j, 2] = v2x, v2y, v2t
        fin[f + 1], fin_arg[f + 1], fin_st[f + 1] = fin[f], fin_arg[f], fin_st[f]
        for j in range(off[f], off[f + 1]):
            for s in range(3):
                if D[j, s] < fin[f + 1]:
                    fin[f + 1], fin_arg[f + 1], fin_st[f + 1] = D[j, s], j, s
    chosen = np.full(n, -1, np.int64)
    start = np.zeros(n, np.bool_)
    kink = np.zeros(n, np.bool_)
    j, s = fin_arg[n], fin_st[n]
    while j >= 0:
        f = fidx[j]
        chosen[f] = j
        if s == 1 and bpi[j, 1] < 0:        # birth
            start[f] = True
            j, s = bl[j], bls[j]
        else:
            kink[f] = s == 1
            j, s = bpi[j, s], bps[j, s]
    return chosen, start, kink


# ══════════════════════════════════════════════════════════════════════════════
# Candidate preparation and static-distractor map
# ══════════════════════════════════════════════════════════════════════════════

def _timestamps(meta: VideoMeta, n: int) -> np.ndarray:
    """meta.timestamps (length-fixed, strictly increasing) or nominal fps. Duplicate /
    reversed PTS are re-timed (_retime_duplicates); a non-finite fps counts as unknown (30)."""
    fps = float(meta.fps) if meta.fps and np.isfinite(meta.fps) and meta.fps > 0 else 30.0
    ts = np.asarray(meta.timestamps if meta.timestamps is not None else [], np.float64)
    if len(ts) < n:
        start = ts[-1] + 1.0 / fps if len(ts) else 0.0
        ts = np.concatenate([ts, start + np.arange(n - len(ts)) / fps])
    ts = ts[:n].copy()
    if n > 1 and not np.all(np.isfinite(ts)):
        ts = np.arange(n) / fps
    if n > 1 and np.any(np.diff(ts) <= 0):     # repair duplicates / reversals
        ts = _retime_duplicates(ts, 1.0 / fps)
    return ts


def _retime_duplicates(ts: np.ndarray, step0: float) -> np.ndarray:
    """Duplicate PTS = repeated frames. A reversed or stalled clock repeats the previous
    time (running max); each run of equal times keeps its first frame's time and spreads
    its repeats evenly up to the next distinct time. A trailing run (or a whole-clip run)
    has no next time and advances by the clip's median positive step (else `step0`).
    Strictly increasing afterwards (a step below the float resolution becomes one ulp)."""
    t = np.maximum.accumulate(ts)
    n = len(t)
    d = np.diff(t)
    pos = d[d > 0]
    step_end = float(np.median(pos)) if len(pos) else step0
    out = t.copy()
    starts = np.flatnonzero(np.r_[True, d > 0])            # first frame of each run
    ends = np.r_[starts[1:], n]                           # one past its last frame
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        step = (t[b] - t[a]) / (b - a) if b < n else step_end
        out[a + 1:b] = t[a] + np.arange(1, b - a) * step
    bad = np.flatnonzero(np.diff(out) <= 0)
    if len(bad):
        for i in range(int(bad[0]) + 1, n):
            if not out[i] > out[i - 1]:
                out[i] = np.nextafter(out[i - 1], np.inf)
    return out


def _flatten(candidates: list[list[ShuttleCandidate]], cfg: LinkerConfig, width: float,
             height: float) -> tuple[np.ndarray, ...]:
    """CSR layout: frame f owns rows off[f]:off[f+1] (top-k by score)."""
    n = len(candidates)
    counts = np.zeros(n, np.int64)
    rows: list[tuple[float, float, float]] = []
    for f, cands in enumerate(candidates):
        if not cands:
            continue
        cs = [c for c in cands if c is not None and c.score >= cfg.min_score
              and np.isfinite(c.x) and np.isfinite(c.y)]
        if len(cs) > cfg.max_candidates:
            cs = sorted(cs, key=lambda c: -c.score)[:cfg.max_candidates]
        counts[f] = len(cs)
        rows.extend((float(c.x), float(c.y), float(c.score)) for c in cs)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    arr = np.asarray(rows, np.float64).reshape(-1, 3)
    fidx = np.repeat(np.arange(n, dtype=np.int64), counts)
    xs = np.clip(arr[:, 0], -0.05 * width, 1.05 * width)
    ys = np.clip(arr[:, 1], -0.05 * height, 1.05 * height)
    return off, fidx, xs, ys, np.clip(arr[:, 2], 0.0, 1.0)


def _neighbour_counts(key_cell: np.ndarray, fidx: np.ndarray, mask: np.ndarray,
                      cx: np.ndarray, cy: np.ndarray, blk: np.ndarray, ncx: int,
                      ncy: int, n_frames: int) -> np.ndarray:
    """For every candidate: distinct frames in its block with a masked
    candidate in the 3×3 cell neighbourhood."""
    m = len(fidx)
    out = np.zeros(m)
    sel = np.flatnonzero(mask)
    if len(sel) == 0:
        return out
    fk = np.unique(key_cell[sel] * (n_frames + 1) + fidx[sel])
    ukey, cnt = np.unique(fk // (n_frames + 1), return_counts=True)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            q = (blk * ncy + (cy + oy)) * ncx + (cx + ox)
            pos = np.minimum(np.searchsorted(ukey, q), len(ukey) - 1)
            out += np.where(ukey[pos] == q, cnt[pos], 0)
    return out


def _static_costs(fidx: np.ndarray, xs: np.ndarray, ys: np.ndarray, t: np.ndarray,
                  diag: float, cfg: LinkerConfig,
                  path_xy: np.ndarray | None = None) -> np.ndarray:
    """Additive cost per candidate for persistent static distractors.

    Without a path: plain occupancy (a cell busy in ≥ 30–60 % of a block's
    frames). With a path from a first pass: occupancy counted only in frames
    where the chosen shuttle is present *and far away* — a location that keeps
    firing while the shuttle flies elsewhere is a distractor, whereas a shuttle
    held still before a serve is never "elsewhere" and is not penalised."""
    m, n = len(fidx), len(t)
    if m == 0:
        return np.zeros(0)
    fps = 1.0 / max(float(np.median(np.diff(t))) if n > 1 else 1 / 30, 1e-4)
    cell = max(cfg.static_cell * diag, 2.0)
    cx = np.floor(xs / cell).astype(np.int64)
    cy = np.floor(ys / cell).astype(np.int64)
    cx, cy = cx - cx.min() + 2, cy - cy.min() + 2
    ncx, ncy = int(cx.max()) + 3, int(cy.max()) + 3
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    nblk = max(1, int(dur // cfg.static_block_s))     # equal blocks, all ≥ block_s
    fblk = np.minimum(np.floor((t - t[0]) / max(dur, 1e-9) * nblk), nblk - 1).astype(np.int64)
    blk = fblk[fidx]
    key = (blk * ncy + cy) * ncx + cx
    floor_frames = cfg.static_min_s * fps

    def ramp(frac: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return cfg.static_cost * np.clip((frac - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    frames_b = np.bincount(fblk, minlength=nblk).astype(np.float64)
    occ = _neighbour_counts(key, fidx, np.ones(m, bool), cx, cy, blk, ncx, ncy, n)
    cost = ramp(occ / np.maximum(frames_b, floor_frames)[blk], cfg.static_occ_lo,
                cfg.static_occ_hi)
    if path_xy is not None:
        px = path_xy[fidx]
        present = np.isfinite(px[:, 0])
        far = present & (np.hypot(xs - px[:, 0], ys - px[:, 1]) > cfg.static_far * diag)
        pres_b = np.bincount(fblk[np.isfinite(path_xy[:, 0])], minlength=nblk)
        denom = np.maximum(pres_b.astype(np.float64), cfg.static_min_present_s * fps)
        els = _neighbour_counts(key, fidx, far, cx, cy, blk, ncx, ncy, n)
        cost = np.maximum(cost, ramp(els / denom[blk], cfg.static_lo, cfg.static_hi))
    return cost


def _dp_params(cfg: LinkerConfig, diag: float, t: np.ndarray) -> tuple[np.ndarray, int]:
    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    prm = np.array([cfg.max_speed * diag, (cfg.sigma_px * diag) ** 2,
                    (cfg.accel_sigma * diag) ** 2, cfg.drag_length_min * diag,
                    cfg.speed_up_max, cfg.kink_cost, cfg.first_cost, cfg.birth_cost,
                    cfg.miss_cost, cfg.drag_length_young * diag, 0.0], np.float64)
    gap = max(1, int(np.ceil(cfg.max_link_gap_s / max(dt_med, 1e-4) - 1e-6)))
    prm[10] = max(1, int(np.ceil(cfg.kink_max_gap_s / max(dt_med, 1e-4) - 1e-6)))
    return prm, gap


def _associate(candidates: list[list[ShuttleCandidate]], t: np.ndarray, width: float,
               height: float, cfg: LinkerConfig, shot0: np.ndarray | None = None
               ) -> tuple[np.ndarray, ...]:
    """Global association: 2 passes (static map: occupancy, then "while
    elsewhere"; pass 2 also uses the noise measured on pass 1), then ≤
    weak_ban_rounds re-associations that ban weak candidates the support rule
    rejected (so the DP re-plans around them). Returns per-frame (xy, score,
    start), xy NaN where nothing was adopted; short chains removed."""
    n = len(t)
    diag = float(np.hypot(width, height))
    xy = np.full((n, 2), np.nan)
    sc = np.zeros(n)
    start = np.zeros(n, bool)
    off, fidx, xs, ys, score = _flatten(candidates, cfg, width, height)
    if len(xs) == 0:
        return xy, sc, start
    s = np.clip(score, 1e-3, 1 - 1e-3)
    u0 = -(cfg.det_reward + cfg.score_weight * np.log(s / (1 - s)))
    prm, gap = _dp_params(cfg, diag, t)
    sh = np.zeros(n, np.int64) if shot0 is None else np.asarray(shot0, np.int64)
    ban = np.zeros(len(xs))
    path, static = None, None
    for rnd in range(2 + max(cfg.weak_ban_rounds, 0)):
        if rnd < 2:
            static = _static_costs(fidx, xs, ys, t, diag, cfg, path)
        chosen, start, kink = _viterbi(off, fidx, xs, ys, u0 + static + ban, t, prm, gap, sh)
        ok = chosen >= 0
        path = np.full((n, 2), np.nan)
        path[ok, 0], path[ok, 1] = xs[chosen[ok]], ys[chosen[ok]]
        if rnd == 0 and ok.sum() >= 8 * cfg.kink_min_pts:
            fo = np.flatnonzero(ok)          # detector noise differs per detector /
            seg0 = _segments(t[fo], start[fo], cfg.max_gap_s)          # resolution
            s_est = _estimate_sigma(t[fo], path[fo], seg0, cfg.sigma_px * diag, cfg)
            prm[1] = max(prm[1], s_est ** 2)
        strong_elsewhere = np.zeros(n, bool)        # a real strong candidate in the frame
        se = (score >= cfg.strong_score) & (static < 1.0)
        strong_elsewhere[fidx[se]] = True
        bad = _weak_unsupported(t, path, np.where(ok, score[np.maximum(chosen, 0)], 0.0),
                                start & ok, kink & ok, diag, cfg, strong_elsewhere)
        if rnd >= 1 and (not bad.any() or rnd == 1 + max(cfg.weak_ban_rounds, 0)):
            break
        if rnd >= 1:
            ban[chosen[bad]] = _INF / 1e6     # never adopt these again
    xy, sc[ok] = path, score[chosen[ok]]
    adopted = np.flatnonzero(ok)
    xy[bad], sc[bad] = np.nan, 0.0        # whatever is still unsupported is dropped
    ok = np.isfinite(xy[:, 0])
    # restart the chain after a dropped run (we were on something else there)
    cid0 = np.cumsum(start) - 1
    prev_bad = np.zeros(n, bool)
    if len(adopted) > 1:
        prev_bad[adopted[1:]] = bad[adopted[:-1]]
    obs = np.flatnonzero(ok)
    start = np.zeros(n, bool)
    if len(obs):
        start[obs[np.r_[True, (cid0[obs][1:] != cid0[obs][:-1]) | prev_bad[obs][1:]]]] = True
        cid = np.cumsum(start) - 1
        lens = np.bincount(cid[obs], minlength=int(cid[obs].max()) + 1)
        short = obs[lens[cid[obs]] < cfg.min_chain_len]
        xy[short], sc[short], start[short] = np.nan, 0.0, False
    return xy, sc, start


def _weak_unsupported(t: np.ndarray, path: np.ndarray, sc: np.ndarray, start: np.ndarray,
                      kink: np.ndarray, diag: float, cfg: LinkerConfig,
                      strong_elsewhere: np.ndarray | None = None) -> np.ndarray:
    """Adopted weak (< strong_score) observations without shuttle support.
    Pieces = runs between births and kinks. A weak obs is kept only if its
    piece has ≥ min_strong strong obs and one of them is ≤ weak_reach_s away,
    or if the whole piece flies (median speed ≥ weak_fly_speed for ≥
    weak_fly_min_s — a far, blurred shuttle). Shoes, heads and rackets give
    weak, slow or erratic runs; a flying shuttle is seen strongly within a few
    frames. A held shuttle is kept iff strong detections support it. Weak
    obs may only *continue* a track: a leading weak run (before the piece's
    first strong obs) must itself fly. A weak obs is never taken when its
    frame has a strong non-static candidate (the weak one is then typically
    the detector's ghost of the previous position)."""
    n = len(t)
    out = np.zeros(n, bool)
    fr = np.flatnonzero(np.isfinite(path[:, 0]))
    if len(fr) == 0:
        return out
    pid = np.cumsum(start[fr] | kink[fr]) - 1
    pid -= pid.min()
    strong = sc[fr] >= cfg.strong_score
    n_strong = np.bincount(pid, weights=strong, minlength=int(pid.max()) + 1)
    to = t[fr]
    # time to the nearest strong obs of the same piece (pieces are contiguous)
    big = 1e9
    key = pid * big + to
    sk = key[strong]
    j = np.searchsorted(sk, key)
    d = np.full(len(fr), np.inf)       # no strong obs at all: no weak obs is near one
    if len(sk):                        # (indexing the empty sk would raise IndexError)
        for jj in (j - 1, j):
            v = (jj >= 0) & (jj < len(sk))
            jc = np.clip(jj, 0, len(sk) - 1)
            same = v & (np.floor(sk[jc] / big) == pid)
            d = np.where(same, np.minimum(d, np.abs(sk[jc] - key)), d)
    weak = ~strong
    bad = weak & ((n_strong[pid] < cfg.min_strong) | (d > cfg.weak_reach_s))
    # leading weak runs (before the first strong obs of their piece) must fly
    first_strong = np.full(int(pid.max()) + 1, np.inf)
    np.minimum.at(first_strong, pid[strong], to[strong])
    lead = weak & (to < first_strong[pid])
    for p in np.unique(pid[lead & ~bad]):
        sel = np.flatnonzero(lead & (pid == p))
        v = np.hypot(*np.diff(path[fr[sel]], axis=0).T) / np.maximum(np.diff(to[sel]), 1e-6) \
            if len(sel) > 1 else np.zeros(1)
        if np.median(v) < cfg.weak_fly_speed * diag:
            bad[sel] = True
    # trailing weak runs (after the last strong obs) must keep the direction the
    # strong obs were flying in (drag slows a shuttle, it doesn't turn it)
    last_strong = np.full(int(pid.max()) + 1, -np.inf)
    np.maximum.at(last_strong, pid[strong], to[strong])
    trail = weak & ~bad & (to > last_strong[pid]) & np.isfinite(last_strong[pid])
    tol = 4.0 * cfg.sigma_px * diag
    for p in np.unique(pid[trail]):
        ps = np.flatnonzero(pid == p)
        k = ps[np.searchsorted(to[ps], last_strong[p])]  # last strong obs of the piece
        if k == ps[0]:
            continue
        v = (path[fr[k]] - path[fr[k - 1]]) / max(to[k] - to[k - 1], 1e-6)
        sp = float(np.hypot(*v))
        if sp < cfg.weak_fly_speed * diag:                # it was not flying: nothing to keep
            continue
        u = v / sp
        sel = np.flatnonzero(trail & (pid == p))
        dd = path[fr[sel]] - path[fr[k]]
        along = dd @ u
        perp = np.abs(dd[:, 0] * u[1] - dd[:, 1] * u[0])
        off_dir = (along < -tol) | (perp > tol + 0.5 * np.maximum(along, 0.0))
        bad[sel[np.maximum.accumulate(off_dir)]] = True   # from the first departure on
    if strong_elsewhere is not None:
        bad |= weak & strong_elsewhere[fr]
    # flying weak-only pieces survive
    for p in np.unique(pid[bad]):
        sel = np.flatnonzero(pid == p)
        if len(sel) < 3 or to[sel[-1]] - to[sel[0]] < cfg.weak_fly_min_s:
            continue
        v = np.hypot(*np.diff(path[fr[sel]], axis=0).T) / np.maximum(np.diff(to[sel]), 1e-6)
        if np.median(v) >= cfg.weak_fly_speed * diag:
            bad[sel] = False
    out[fr[bad]] = True
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Kinks (hits): unexplained velocity change between two local line fits
# ══════════════════════════════════════════════════════════════════════════════

def _segments(t_obs: np.ndarray, start_obs: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Segment label per observation: new segment at chain starts / long gaps."""
    brk = np.zeros(len(t_obs), bool)
    if len(t_obs):
        brk[0] = True
        brk[1:] = np.diff(t_obs) > max_gap_s
        brk |= start_obs
    return np.cumsum(brk) - 1


def _pair_windows(t_obs: np.ndarray, seg: np.ndarray, cfg: LinkerConfig,
                  max_pts: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every consecutive same-segment observation pair (m, m+1) with its
    two-sided window (≤ kink_window_s and ≤ max_pts points per side):
    returns (m (M,), idx (M,2W), valid (M,2W)); columns 0..W-1 are the left
    side (m-W+1 … m), W..2W-1 the right (m+1 … m+W)."""
    n = len(t_obs)
    dt_med = float(np.median(np.diff(t_obs))) if n > 1 else 1 / 30
    W = int(np.clip(round(cfg.kink_window_s / max(dt_med, 1e-4)) + 1,
                    cfg.kink_min_pts, max_pts or cfg.kink_max_pts))
    m = np.arange(max(n - 1, 0))
    m = m[seg[m] == seg[m + 1]]
    k = np.arange(W)
    li = m[:, None] - k[None, :]                       # left: m, m-1, …
    ri = m[:, None] + 1 + k[None, :]                   # right: m+1, m+2, …
    lc, rc = np.clip(li, 0, max(n - 1, 0)), np.clip(ri, 0, max(n - 1, 0))
    tw = max(cfg.kink_window_s, 1e-3)
    lv = ((li >= 0) & (seg[lc] == seg[m][:, None])
          & ((t_obs[m][:, None] - t_obs[lc] <= tw) | (k[None, :] < cfg.kink_min_pts)))
    rv = ((ri < n) & (seg[rc] == seg[m][:, None])
          & ((t_obs[rc] - t_obs[m + 1][:, None] <= tw) | (k[None, :] < cfg.kink_min_pts)))
    ok = (lv.sum(1) >= cfg.kink_min_pts) & (rv.sum(1) >= cfg.kink_min_pts)
    idx = np.concatenate([lc[:, ::-1], rc], 1)[ok]
    val = np.concatenate([lv[:, ::-1], rv], 1)[ok]
    return m[ok], idx, val


def _hinge_fit(t_obs: np.ndarray, p: np.ndarray, idx: np.ndarray, val: np.ndarray,
               tau: np.ndarray, tw: float, deg: int, free_curv: bool = True,
               t_eval: np.ndarray | None = None, chunk: int = 32768) -> tuple[np.ndarray, ...]:
    """Batched LS fit over each window, s = (t − τ)/tw:
        p(t) = Σ_d c_d s^d  [+ c₊ s₊²]  + b s₊,     s₊ = max(s, 0)
    i.e. position continuous at τ, velocity step b (the hit), and with
    `free_curv` an independent curvature after τ (drag deceleration right
    after a hit is far stronger than before it). Returns (b (M,2) px/s,
    Var(b)/σ² (M,), RSS (M,), dof (M,)) and, with t_eval, the fitted value
    (M,2) and its Var/σ² (M,) at t_eval."""
    M, K = idx.shape
    P = deg + 2 + int(free_curv)
    b = np.zeros((M, 2))
    cbb = np.full(M, np.inf)
    rss = np.zeros(M)
    dof = 2.0 * (val.sum(1) - P)
    pred = np.zeros((M, 2))
    vpred = np.full(M, np.inf)

    def design(sv: np.ndarray) -> np.ndarray:
        sp = np.maximum(sv, 0.0)
        cols = [np.ones_like(sv)]
        for _ in range(deg):
            cols.append(cols[-1] * sv)
        if free_curv:
            cols.append(sp * sp)
        cols.append(sp)
        return np.stack(cols, -1)

    for c0 in range(0, M, chunk):
        sl = slice(c0, min(c0 + chunk, M))
        v = val[sl].astype(np.float64)
        X = design((t_obs[idx[sl]] - tau[sl, None]) / tw) * v[..., None]
        Y = p[idx[sl]] * v[..., None]
        Xt = X.transpose(0, 2, 1)
        A = Xt @ X
        A[:, np.arange(P), np.arange(P)] += 1e-9
        cols = [Xt @ Y, np.broadcast_to(np.eye(P)[:, -1:], (len(v), P, 1))]
        if t_eval is not None:
            xe = design((t_eval[sl] - tau[sl]) / tw)
            cols.append(xe[..., None])
        rhs = np.concatenate(cols, 2)
        sol = np.linalg.solve(A, rhs)
        b[sl] = sol[:, -1, :2] / tw
        cbb[sl] = np.maximum(sol[:, -1, 2], 1e-12) / tw ** 2
        rss[sl] = np.maximum((Y * Y).sum((1, 2)) - (rhs[..., :2] * sol[..., :2]).sum((1, 2)), 0.0)
        if t_eval is not None:
            pred[sl] = (xe[..., None] * sol[..., :2]).sum(1)
            vpred[sl] = np.maximum((xe * sol[..., 3]).sum(1), 0.0)
    if t_eval is not None:
        return b, cbb, rss, dof, pred, vpred
    return b, cbb, rss, dof


def _sigma_from_rss(rss: np.ndarray, dof: np.ndarray) -> float:
    """Robust per-axis noise σ from window residuals: median of RSS/dof with
    the χ² median correction (Wilson–Hilferty); NaN if too few windows."""
    ok = dof >= 2
    if ok.sum() < 20:
        return float("nan")
    k = dof[ok]
    r = rss[ok] / k / (1.0 - 2.0 / (9.0 * k)) ** 3
    return float(np.sqrt(np.median(r)))


def _kink_scan(t_obs: np.ndarray, p: np.ndarray, seg: np.ndarray, sigma: float | None,
               cfg: LinkerConfig) -> tuple[np.ndarray, ...]:
    """GLRT for a velocity step between every consecutive observation pair
    (hinge at the pair midpoint; cubic + free post-τ curvature absorb drag,
    gravity and perspective). z = |b| / (σ·sd(b)) is Rayleigh(1) under "no
    hit". σ = None → estimated from the window residuals. Returns (m, z,
    |Δv| px/s, σ used)."""
    m, idx, val = _pair_windows(t_obs, seg, cfg)
    if len(m) == 0:
        return m, np.zeros(0), np.zeros(0), float(sigma or np.nan)
    tm = 0.5 * (t_obs[m] + t_obs[m + 1])
    bb, cbb, rss, dof = _hinge_fit(t_obs, p, idx, val, tm, max(cfg.kink_window_s, 1e-3),
                                   cfg.kink_deg, True)
    if sigma is None:
        sigma = _sigma_from_rss(rss, dof)
    dv = np.hypot(bb[:, 0], bb[:, 1])
    sg = sigma if np.isfinite(sigma) else 1.0
    return m, dv / (max(sg, 1e-6) * np.sqrt(cbb)), dv, float(sigma)


def _locate_stage(t_obs: np.ndarray, p: np.ndarray, seg: np.ndarray, m: np.ndarray,
                  tau: np.ndarray, cfg: LinkerConfig, max_pts: int | None, deg: int,
                  reach: float, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """One min-RSS localisation pass: for each kink (pair m, time τ) take the
    window of pair m and put the hinge at n_grid positions inside every pair
    of that window within `reach` of τ; keep the least-RSS position."""
    mm, idx, val = _pair_windows(t_obs, seg, cfg, max_pts)
    if len(mm) == 0 or len(m) == 0:
        return m, tau
    pos = np.minimum(np.searchsorted(mm, m), len(mm) - 1)
    ok = mm[pos] == m
    tm = 0.5 * (t_obs[mm] + t_obs[mm + 1])
    g = np.linspace(0.0, 1.0, n_grid)
    rows_i, rows_t, rows_m, owner = [], [], [], []
    for j in np.flatnonzero(ok):
        k = pos[j]
        vt = t_obs[idx[k][val[k]]]
        lo = int(np.searchsorted(tm, max(tau[j] - reach, vt.min()) - 1e-9))
        hi = int(np.searchsorted(tm, min(tau[j] + reach, vt.max()) + 1e-9, side="right"))
        ks = np.arange(min(lo, k), max(hi, k + 1))
        ks = ks[(seg[mm[ks]] == seg[mm[k]]) & (t_obs[mm[ks]] >= vt.min())
                & (t_obs[mm[ks] + 1] <= vt.max())]
        if len(ks) == 0:
            continue
        ta, tb = t_obs[mm[ks]], t_obs[mm[ks] + 1]
        rows_t.append((ta[:, None] + g[None, :] * (tb - ta)[:, None]).ravel())
        rows_m.append(np.repeat(mm[ks], n_grid))
        rows_i.append(np.full(len(ks) * n_grid, k))
        owner.append(np.full(len(ks) * n_grid, j))
    if not owner:
        return m, tau
    taus, mc, own, ki = (np.concatenate(rows_t), np.concatenate(rows_m),
                         np.concatenate(owner), np.concatenate(rows_i))
    rss = _hinge_fit(t_obs, p, idx[ki], val[ki], taus, max(cfg.kink_window_s, 1e-3),
                     deg, True)[2]
    order = np.lexsort((rss, own))
    best = order[np.r_[True, own[order][1:] != own[order][:-1]]]
    m, tau = m.copy(), tau.copy()
    m[own[best]] = mc[best]
    tau[own[best]] = taus[best]
    return m, tau


def _kink_locate(t_obs: np.ndarray, p: np.ndarray, seg: np.ndarray, m: np.ndarray,
                 cfg: LinkerConfig) -> tuple[np.ndarray, np.ndarray]:
    """Hit time τ and pair for detected kinks, by min RSS of the hinge model
    (position-continuous, velocity step + free curvature after τ):
      1. coarse: anywhere in the long scan window of the detected pair — a
         window that contains a hit gives a large GLRT response for hinges a
         few frames off, so the argmax pair is often wrong at high fps;
      2. fine: ±kink_loc_s with a short window (kink_loc_pts per side,
         quadratic) where post-hit drag deceleration is still well modelled.
    On synthetic rallies this puts ~80 % of hits on the exact frame pair and
    ~97 % within one frame."""
    if len(m) == 0:
        return m, np.zeros(0)
    tau = 0.5 * (t_obs[m] + t_obs[np.minimum(m + 1, len(t_obs) - 1)])
    dt = float(np.median(np.diff(t_obs))) if len(t_obs) > 1 else 1 / 30
    w_scan = min(round(cfg.kink_window_s / max(dt, 1e-4)) + 1, cfg.kink_max_pts)
    if cfg.kink_loc_coarse and w_scan >= 2 * cfg.kink_loc_pts:
        m, tau = _locate_stage(t_obs, p, seg, m, tau, cfg, None, cfg.kink_deg, np.inf, 3)
    return _locate_stage(t_obs, p, seg, m, tau, cfg, min(cfg.kink_loc_pts, cfg.kink_max_pts),
                         cfg.kink_loc_deg, max(cfg.kink_loc_s, 2.5 * dt), 9)


def _nms_1d(tau: np.ndarray, z: np.ndarray, sep: float) -> np.ndarray:
    """Greedy NMS in time (highest z first), O(n log n). Sorted indices."""
    kept_t: list[float] = []
    keep: list[int] = []
    for i in np.argsort(-z, kind="stable"):
        ti = float(tau[i])
        pos = bisect.bisect_left(kept_t, ti)
        if pos > 0 and ti - kept_t[pos - 1] < sep:
            continue
        if pos < len(kept_t) and kept_t[pos] - ti < sep:
            continue
        kept_t.insert(pos, ti)
        keep.append(int(i))
    return np.sort(np.asarray(keep, np.int64))


# ══════════════════════════════════════════════════════════════════════════════
# Piecewise robust smoothing (banded MAP with an integrated-white-jerk prior)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_sigma(t_obs: np.ndarray, p: np.ndarray, seg: np.ndarray, prior: float,
                    cfg: LinkerConfig) -> float:
    """Per-axis detector noise from the residuals of the kink-scan windows
    (cubic + hinge fits, so flight curvature and hits are not counted as
    noise); falls back to `prior` on short tracks. Clipped to [0.4, 2.5]·prior."""
    if len(t_obs) < 4 * cfg.kink_min_pts:
        return prior
    s = _kink_scan(t_obs, p, seg, None, cfg)[3]
    if not np.isfinite(s):
        return prior
    return float(np.clip(s, 0.4 * prior, 2.5 * prior))


def _pieces(fr: np.ndarray, t: np.ndarray, seg: np.ndarray, kink_m: np.ndarray,
            kink_tau: np.ndarray, cfg: LinkerConfig) -> tuple[np.ndarray, ...]:
    """Observation ranges [a, b) and frame ranges [lo, hi] of flight pieces.
    Pieces break at segment changes and after each kink observation m; at a
    kink both sides are extrapolated ≤ max_extrap_frames toward τ."""
    n_obs = len(fr)
    cut = np.zeros(n_obs + 1, bool)
    cut[0] = cut[n_obs] = True
    cut[1:n_obs] |= seg[1:] != seg[:-1]
    cut[kink_m + 1] = True
    edges = np.flatnonzero(cut)
    a, b = edges[:-1], edges[1:]
    lo, hi = fr[a].copy(), fr[b - 1].copy()
    tau_at = np.full(n_obs, np.nan)
    tau_at[kink_m] = kink_tau
    ex = cfg.max_extrap_frames
    for p in range(len(a)):
        if ex <= 0:
            break
        m = b[p] - 1                                      # extend forward to τ
        if m < n_obs - 1 and np.isfinite(tau_at[m]):
            f = fr[m]
            while hi[p] - f < ex and hi[p] + 1 < fr[m + 1] and t[hi[p] + 1] <= tau_at[m]:
                hi[p] += 1
        m = a[p] - 1                                      # extend backward to τ
        if m >= 0 and np.isfinite(tau_at[m]):
            f = fr[a[p]]
            while f - lo[p] < ex and lo[p] - 1 > fr[m] and t[lo[p] - 1] > tau_at[m]:
                lo[p] -= 1
    return a, b, lo, hi


def _solve_pieces(t: np.ndarray, fr: np.ndarray, p: np.ndarray, a: np.ndarray,
                  b: np.ndarray, lo: np.ndarray, hi: np.ndarray, sigma: float,
                  diag: float, cfg: LinkerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust smoothing of all pieces at once. Returns (var frames (V,),
    var xy (V,2), residual/σ per observation (n_obs,), NaN for unsmoothed)."""
    n_obs = len(fr)
    resid = np.full(n_obs, np.nan)
    good = (b - a) >= 3                                   # need ≥ 3 obs per piece
    a, b, lo, hi = a[good], b[good], lo[good], hi[good]
    if len(a) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 2)), resid
    L = hi - lo + 1
    poff = np.zeros(len(L) + 1, np.int64)
    np.cumsum(L, out=poff[1:])
    V = int(poff[-1])
    vp = np.repeat(np.arange(len(L)), L)
    vf = lo[vp] + (np.arange(V) - poff[vp])
    h0 = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    tn = (t[vf] - t[vf[0]]) / h0                          # normalised time
    oi = np.concatenate([np.arange(a[k], b[k]) for k in range(len(a))])
    ov = poff[np.repeat(np.arange(len(a)), b - a)] + (fr[oi] - lo[np.repeat(np.arange(len(a)), b - a)])
    po = p[oi]
    # reference for the tiny ridge: piecewise-linear interpolation of the data
    xref = np.empty((V, 2))
    for k in range(len(a)):
        s = slice(poff[k], poff[k + 1])
        for d in range(2):
            xref[s, d] = np.interp(vf[s], fr[a[k]:b[k]], p[a[k]:b[k], d])
    # third-divided-difference rows over 4 consecutive vars of one piece
    r0 = np.flatnonzero(vp[:-3] == vp[3:]) if V >= 4 else np.zeros(0, np.int64)
    T = tn[r0[:, None] + np.arange(4)[None, :]]
    C = np.empty_like(T)
    for k in range(4):
        prod = np.ones(len(r0))
        for j in range(4):
            if j != k:
                prod *= T[:, k] - T[:, j]
        C[:, k] = 6.0 / prod
    hbar = (T[:, 3] - T[:, 0]) / 3.0
    Lj = cfg.jerk_drag_len * diag
    jfloor = cfg.jerk_floor * diag
    inv_s2 = 1.0 / max(sigma, 1e-6) ** 2
    ridge = 1e-8 * inv_s2

    def irls(v0: int, v1: int, ra: int, rb: int, oa: int, ob: int) -> tuple[np.ndarray, np.ndarray]:
        """IRLS over vars v0:v1 (whole pieces), jerk rows ra:rb, observations oa:ob."""
        rr, ovs, pos = r0[ra:rb] - v0, ov[oa:ob] - v0, po[oa:ob]
        Cs, hb, xr, tv = C[ra:rb], hbar[ra:rb], xref[v0:v1], t[vf[v0:v1]]
        w_obs = np.ones(ob - oa)
        x = xr.copy()
        for it in range(max(1, cfg.irls_iters)):
            # speed-adaptive jerk prior (quadratic drag: |j| ~ v³/L²)
            seg_v = np.hypot(*(np.diff(x, axis=0)).T) / np.maximum(np.diff(tv), 1e-6)
            vr = np.maximum(np.maximum(seg_v[rr], seg_v[rr + 1]), seg_v[rr + 2]) if len(rr) else seg_v[:0]
            J = jfloor + vr ** 3 / (Lj * Lj)
            qn = J * J * cfg.jerk_corr_s * h0 ** 5
            wr = hb / (0.55 * qn)
            ab = np.zeros((4, v1 - v0))
            ab[3] += ridge
            ab[3, ovs] += w_obs * inv_s2
            for i in range(4):
                for j in range(i, 4):
                    ab[3 - (j - i), rr + j] += wr * Cs[:, i] * Cs[:, j]
            rhs = ridge * xr
            np.add.at(rhs, ovs, (w_obs * inv_s2)[:, None] * pos)
            x = solveh_banded(ab, rhs, check_finite=False)
            r = np.hypot(*(x[ovs] - pos).T) / max(sigma, 1e-6)
            w_obs = np.where(r <= cfg.huber_k, 1.0, cfg.huber_k / np.maximum(r, 1e-9))
            if it >= 1:
                w_obs[r > cfg.outlier_k] = 0.0
        return x, r

    try:
        x, r = irls(0, V, 0, len(r0), 0, len(oi))
    except np.linalg.LinAlgError:
        # a piece whose jerk prior is not positive definite (near-duplicate timestamps,
        # repaired 1e-3/fps apart, blow up the divided differences). The system is block
        # diagonal per piece: solve the pieces one by one and leave only the failing ones
        # unsmoothed (NaN → _finish passes their observations through).
        x, r = np.full((V, 2), np.nan), np.full(len(oi), np.nan)
        rb = np.searchsorted(vp[r0], np.arange(len(a) + 1))
        ob = np.r_[0, np.cumsum(b - a)]
        for k in range(len(a)):
            try:
                x[poff[k]:poff[k + 1]], r[ob[k]:ob[k + 1]] = irls(
                    poff[k], poff[k + 1], rb[k], rb[k + 1], ob[k], ob[k + 1])
            except np.linalg.LinAlgError:
                pass
    resid[oi] = r
    return vf, x, resid


# ══════════════════════════════════════════════════════════════════════════════
# Finishing stage shared by link_candidates / smooth_track
# ══════════════════════════════════════════════════════════════════════════════

def _geometry(width: float | None, height: float | None, xy: np.ndarray) -> float:
    if width and height:
        return float(np.hypot(width, height))
    fin = xy[np.isfinite(xy[:, 0])]
    if len(fin) == 0:
        return float(np.hypot(1920, 1080))
    span = np.ptp(fin, axis=0)                  # unknown frame size: guess ≥ 720p
    return float(max(np.hypot(*span) * 1.2, np.hypot(1280, 720)))


def _obs_pairs_for_frames(t: np.ndarray, fr: np.ndarray, kf: np.ndarray, m_scan: np.ndarray,
                          z_scan: np.ndarray) -> np.ndarray:
    """Map hit frames (nearest frame to the hit) to observation pairs (m, m+1):
    among the pairs whose interval touches [t_f − dt/2, t_f + dt/2], the one
    with the strongest GLRT response (unscanned pairs count as z = 0)."""
    if len(fr) < 2 or len(kf) == 0:
        return np.zeros(0, np.int64)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    to = t[fr]
    zmap = np.zeros(len(fr) - 1)
    zmap[m_scan] = z_scan
    out = []
    for f in kf:
        lo = int(np.searchsorted(to, t[f] - 0.5 * dt, side="right")) - 1
        hi = int(np.searchsorted(to, t[f] + 0.5 * dt, side="left"))
        lo, hi = max(lo, 0), min(hi, len(fr) - 1)
        if hi <= lo:
            continue
        cand = np.arange(lo, hi)
        out.append(int(cand[np.argmax(zmap[cand])]))
    return np.unique(np.asarray(out, np.int64))


def _finish(t: np.ndarray, xy: np.ndarray, sc: np.ndarray, start: np.ndarray, diag: float,
            cfg: LinkerConfig, kink_frames: np.ndarray | None = None
            ) -> tuple[Track2D, np.ndarray, np.ndarray, np.ndarray]:
    """Spikes → kinks → pieces → robust smoothing → gap filling.

    Kinks: GLRT pairs above kink_z, plus (refinement rounds) pairs above
    kink_z_refine where the piecewise fit leaves a run of ≥ 2 large residuals
    — the signature of a hit the smoother is rounding off. `kink_frames`
    forces the split points instead (no detection). Returns the track plus
    the kink frames, times and scores used."""
    n = len(t)
    out_xy = np.full((n, 2), np.nan)
    conf, vis, interp = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    no = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
    fr_all = np.flatnonzero(np.isfinite(xy[:, 0]))
    if len(fr_all) == 0:
        return (Track2D(t=t, xy=out_xy, conf=conf, visible=vis, interpolated=interp,
                        raw_xy=np.full((n, 2), np.nan)), *no)
    p_all, st = xy[fr_all], start[fr_all].copy()
    st[0] = True
    seg_all = _segments(t[fr_all], st, cfg.max_gap_s)
    sigma = _estimate_sigma(t[fr_all], p_all, seg_all, cfg.sigma_px * diag, cfg)
    fr, p, seg = fr_all, p_all, seg_all
    to = t[fr]
    ms, zs, dvs, _ = _kink_scan(to, p, seg, sigma, cfg)
    big_dv = dvs > cfg.kink_min_dv * diag
    if kink_frames is not None:
        km = _obs_pairs_for_frames(t, fr, np.asarray(kink_frames, np.int64), ms, zs)
        kz = np.full(len(km), np.inf)
        rounds = 0
    else:
        forced = np.zeros(len(ms), bool)
        rounds = cfg.refine_rounds if cfg.refine_kinks else 0
    for rnd in range(rounds + 1):
        if kink_frames is None:
            acc = ((zs > cfg.kink_z) | forced) & big_dv
            km, kz = ms[acc], zs[acc]
            keep = _nms_1d(0.5 * (to[km] + to[km + 1]), kz, cfg.kink_nms_s) if len(km) else km
            km, kz = km[keep], kz[keep]
        km, ktau = _kink_locate(to, p, seg, km, cfg)
        keep = _nms_1d(ktau, kz, cfg.kink_nms_s) if len(km) else km
        km, ktau, kz = km[keep], ktau[keep], kz[keep]
        a, b, lo, hi = _pieces(fr, t, seg, km, ktau, cfg)
        vf, vx, resid = _solve_pieces(t, fr, p, a, b, lo, hi, sigma, diag, cfg)
        if rnd == rounds or len(ms) == 0:
            break
        # a run of ≥ 2 large residuals among obs m−1 … m+2 around a pair
        big = np.nan_to_num(resid, nan=0.0) > cfg.kink_resid_k
        cb = np.concatenate([[0], np.cumsum(big)])
        lo_i, hi_i = np.maximum(ms - 1, 0), np.minimum(ms + 3, len(fr))
        run = (cb[hi_i] - cb[lo_i]) >= 2
        new = run & (zs > cfg.kink_z_refine) & big_dv & ~forced & ~(zs > cfg.kink_z)
        if not np.any(new):
            break
        forced |= new
    out_xy[vf] = vx
    smoothed = np.isfinite(resid)
    out_xy[fr[~smoothed]] = p[~smoothed]              # tiny pieces: pass through
    obs_ok = np.zeros(n, bool)
    obs_ok[fr] = ~(resid > cfg.outlier_k)
    vis = obs_ok & np.isfinite(out_xy[:, 0])
    interp = np.isfinite(out_xy[:, 0]) & ~vis
    conf = np.where(vis, np.clip(sc, 0.0, 1.0), 0.0)
    kf = np.searchsorted(t, ktau) if len(ktau) else np.zeros(0, np.int64)
    if len(kf):                                       # frame nearest τ
        kf = np.clip(kf, 1, n - 1)
        kf = np.where(np.abs(t[kf - 1] - ktau) <= np.abs(t[kf] - ktau), kf - 1, kf)
    raw = np.full((n, 2), np.nan)                     # the adopted detection itself
    raw[vis] = xy[vis]
    return (Track2D(t=t, xy=out_xy, conf=conf, visible=vis, interpolated=interp, raw_xy=raw),
            kf.astype(np.int64), np.asarray(ktau, float), np.asarray(kz, float))


def _find_repeats(t: np.ndarray, path: np.ndarray, probe: np.ndarray, diag: float,
                  cfg: LinkerConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Repeated video frames (the container shows the previous image again:
    25p broadcast in a 30p file repeats every 6th frame, with a phase that
    can drift after edits; phones duplicate frames under load).

    Slot f is a repeat if its probe point (nearest candidate / track point)
    sits on path[f−1] (≤ max(2σ, 0.15·step)) while the path moves fast and
    straight through it: steps f−2→f−1 and f−1→f+1 both > 6σ, same direction
    (cos > 0.8) and similar length (a repeat hides no time: f+1 is one content
    step after f−1). A real flight never stops for one frame and continues
    unchanged; a hit that returns the shuttle through the same pixel reverses
    direction. Returns (is_dup (N,), content clock (N,)) or None. With a
    dominant repeat period P (≥ repeat_min_count seeds; smallest P whose
    grid slots look like repeats, probe ≤ 0.35·step — a network detector
    re-localises a repeated image within ~1–4 px) the P-grid between seeds
    and ±3 periods around each seed (with evidence) is marked too, and unique frames are
    re-timed by a period-exact moving mean of their container times (the
    25p-in-30p clock is uniform at 40 ms, not 33/33/33/33/67); otherwise
    container times are kept."""
    n = len(t)
    if not cfg.repeat_detect or n < 8:
        return None
    ok = np.isfinite(path[:, 0])
    fr = np.flatnonzero(ok)
    if len(fr) < 4 * cfg.repeat_min_count:
        return None
    sig = _estimate_sigma(t[fr], path[fr], np.zeros(len(fr), np.int64), cfg.sigma_px * diag, cfg)
    f = np.arange(2, n - 1)
    f = f[ok[f - 1] & ok[f - 2] & ok[f + 1] & np.isfinite(probe[f, 0])]
    if len(f) == 0:
        return None
    a = path[f - 1] - path[f - 2]
    b = path[f + 1] - path[f - 1]
    la, lb = np.hypot(*a.T), np.hypot(*b.T)
    cos = (a * b).sum(1) / np.maximum(la * lb, 1e-9)
    fast = np.minimum(la, lb) > max(6.0 * sig, cfg.repeat_min_move * diag)
    d = np.hypot(*(probe[f] - path[f - 1]).T)
    dup_f = f[fast & (cos > 0.8) & (lb < 1.6 * la) & (lb > 0.6 * la)
              & (d <= np.maximum(2.0 * sig, 0.15 * la))]
    if len(dup_f) < 2:
        return None
    is_dup = np.zeros(n, bool)
    is_dup[dup_f] = True
    # contradiction: the path clearly moves between slot f−1 and f
    contra = np.zeros(n, bool)
    g = np.arange(2, n)
    g = g[ok[g] & ok[g - 1] & ok[g - 2]]
    stp = np.hypot(*(path[g] - path[g - 1]).T)
    prev = np.hypot(*(path[g - 1] - path[g - 2]).T)
    contra[g] = stp > np.maximum(3.0 * sig, 0.5 * prev)
    pr_ok = np.isfinite(probe[:, 0])

    def grid_slots(q: int) -> np.ndarray:
        """Phase-local P-grid between consecutive seeds whose gap is a multiple
        of q (≤ 100 periods)."""
        out: list[int] = []
        for a_, b_ in zip(dup_f[:-1], dup_f[1:]):
            if (b_ - a_) % q == 0 and b_ - a_ <= 100 * q:
                out.extend(range(a_ + q, b_, q))
        return np.asarray([f_ for f_ in out if 1 < f_ < n], np.int64)

    def local_ext(q: int) -> np.ndarray:
        """≤ 3 periods either side of every seed (covers phase jumps and clip
        ends), kept only with evidence: the probe sits on the previous point, or
        the slot has no candidate at all."""
        e = (dup_f[:, None] + q * np.r_[-3, -2, -1, 1, 2, 3][None, :]).ravel()
        e = np.unique(e[(e > 1) & (e < n)])
        la_e = np.where(ok[e - 1] & ok[e - 2], np.hypot(*(path[e - 1] - path[e - 2]).T), 0.0)
        de = np.where(pr_ok[e] & ok[e - 1], np.hypot(*(probe[e] - path[e - 1]).T), np.inf)
        ev = de <= np.maximum(2.0 * sig, 0.35 * la_e)
        return e[ev | ~pr_ok[e]]

    P, fill = 0, np.zeros(0, np.int64)
    if len(dup_f) >= cfg.repeat_min_count:
        dd = np.diff(dup_f)
        dd = dd[dd <= 300]
        for q in range(3, 31):          # smallest period whose grid looks like repeats
            if not len(dd) or np.mean(dd % q == 0) < 0.6:
                continue
            gq = grid_slots(q)
            ev = gq[ok[gq - 1] & ok[gq - 2] & pr_ok[gq]] if len(gq) else gq
            la_q = np.hypot(*(path[ev - 1] - path[ev - 2]).T) if len(ev) else np.zeros(0)
            ev, la_q = ev[la_q > 4.0 * sig], la_q[la_q > 4.0 * sig]
            if len(ev) >= 3:
                dq = np.hypot(*(probe[ev] - path[ev - 1]).T)
                if np.mean(dq <= np.maximum(2.0 * sig, 0.35 * la_q)) < 0.7:
                    continue
            P, fill = q, np.union1d(gq, local_ext(q))
            break
    if P and len(fill):
        is_dup[fill[~contra[fill]]] = True
    keep = ~is_dup
    tc = t.copy()
    if P:
        tu = t[keep]
        w = np.ones(P - 1) if (P - 1) % 2 else np.r_[0.5, np.ones(P - 2), 0.5]
        h = len(w) // 2
        if h and len(tu) > 2 * h + 1:
            pad = np.r_[2 * tu[0] - tu[h:0:-1], tu, 2 * tu[-1] - tu[-2:-h - 2:-1]]
            sm = np.convolve(pad, w / w.sum(), mode="valid")
            if len(sm) == len(tu) and np.all(np.diff(sm) > 0):
                tc[keep] = sm
    # repeated slots share the previous slot's (content) time
    idx = np.maximum.accumulate(np.where(keep, np.arange(n), 0))
    return is_dup, tc[idx]


def _probe_candidates(candidates: list[list[ShuttleCandidate]], path: np.ndarray,
                      cfg: LinkerConfig, width: float, height: float) -> np.ndarray:
    """Per frame: the candidate nearest to the previous path point, for the
    repeated-frame test — NaN (no evidence) when a stronger candidate sits
    where the shuttle should be now (previous point + last step, ≤ half a
    step off): multi-frame detectors (TrackNetV3, WASB) leave a weak ghost
    peak at the previous position, which must not make a normal frame look
    repeated."""
    n = len(candidates)
    q = np.full((n, 2), np.nan)
    off, fidx, xs, ys, score = _flatten(candidates, cfg, width, height)
    if len(xs) == 0 or n < 3:
        return q
    prev = np.full((n, 2), np.nan)
    prev[1:] = path[:-1]
    fwd = np.full((n, 2), np.nan)
    fwd[2:] = 2 * path[1:-1] - path[:-2]
    step = np.hypot(*(fwd - prev).T)
    d = np.hypot(xs - prev[fidx, 0], ys - prev[fidx, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    order = np.lexsort((d, fidx))                     # per frame, nearest first
    first = order[np.r_[True, fidx[order][1:] != fidx[order][:-1]]]
    f = fidx[first][np.isfinite(d[first])]
    near = first[np.isfinite(d[first])]
    q[f, 0], q[f, 1] = xs[near], ys[near]
    best_near = np.zeros(n)
    best_near[f] = score[near]
    df = np.hypot(xs - fwd[fidx, 0], ys - fwd[fidx, 1])
    moved = np.isfinite(df) & (df <= 0.5 * step[fidx]) & (score > best_near[fidx])
    q[fidx[moved]] = np.nan
    return q


def _expand(tr_u: Track2D, keep: np.ndarray, t: np.ndarray) -> Track2D:
    """Unique-slot track → container frames: a repeated slot shows the same
    image as the previous slot, so it gets that slot's state."""
    src = np.cumsum(keep) - 1                         # unique index per slot
    raw = None if tr_u.raw_xy is None else tr_u.raw_xy[src].copy()
    return Track2D(t=t, xy=tr_u.xy[src].copy(), conf=tr_u.conf[src].copy(),
                   visible=tr_u.visible[src].copy(),
                   interpolated=tr_u.interpolated[src].copy(), raw_xy=raw)


def _kinks_to_frames(kf_u: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.flatnonzero(keep)[kf_u] if len(kf_u) else kf_u


def _cut_list(n: int, explicit, detected: np.ndarray | None = None) -> np.ndarray:
    """Sorted unique first-frames of new shots (0 < f < n)."""
    parts = [np.asarray(explicit if explicit is not None else [], np.int64).ravel()]
    if detected is not None:
        parts.append(np.asarray(detected, np.int64))
    c = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
    return c[(c > 0) & (c < n)]


def _shot0(n: int, cuts: np.ndarray) -> np.ndarray:
    """First frame of each frame's shot."""
    if len(cuts) == 0:
        return np.zeros(n, np.int64)
    b = np.r_[0, cuts]
    return b[np.searchsorted(b, np.arange(n), side="right") - 1].astype(np.int64)


def _cut_starts(xy: np.ndarray, start: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """start flags + the first observed frame at/after every cut."""
    st = start.copy()
    fr = np.flatnonzero(np.isfinite(xy[:, 0]))
    if len(cuts) and len(fr):
        j = np.searchsorted(fr, cuts)
        st[fr[j[j < len(fr)]]] = True
    return st


def _detect_cuts(candidates: list[list[ShuttleCandidate]], t: np.ndarray, width: float,
                 height: float, cfg: LinkerConfig) -> np.ndarray:
    """Scene cuts from the candidate stream alone (conservative): every video
    has persistent static detector responses (line fragments, lights, logos);
    a cut ends several of them and/or starts several new ones at the same
    moment, while the shuttle, players or an occlusion change one at a time.
    A cut = ≥ cut_min_cells distinct persistent spots (≥ 3 s, ≥ 30 % of
    frames, local maxima of occupancy) ending/starting within ±0.1 s."""
    n = len(t)
    if not cfg.cut_detect or n < 30:
        return np.zeros(0, np.int64)
    off, fidx, xs, ys, _ = _flatten(candidates, cfg, width, height)
    if len(xs) < 30:
        return np.zeros(0, np.int64)
    diag = float(np.hypot(width, height))
    cell = max(cfg.static_cell * diag, 2.0)
    cx = np.floor(xs / cell).astype(np.int64) + 2
    cy = np.floor(ys / cell).astype(np.int64) + 2
    ncx = int(cx.max()) + 3
    ck = cy * ncx + cx
    pair = np.unique(ck * (n + 1) + fidx)
    pc, pf = pair // (n + 1), pair % (n + 1)
    ukey, occ = np.unique(pc, return_counts=True)
    nb = np.zeros(len(ukey))                          # max occupancy of the 8 neighbours
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            if ox or oy:
                q = ukey + oy * ncx + ox
                pos = np.minimum(np.searchsorted(ukey, q), len(ukey) - 1)
                nb = np.maximum(nb, np.where(ukey[pos] == q, occ[pos], 0))
    peak = set(ukey[occ > nb].tolist())
    dt = float(np.median(np.diff(t)))
    brk = np.r_[True, (pc[1:] != pc[:-1]) | (t[pf[1:]] - t[pf[:-1]] > 1.0)]
    r0 = np.flatnonzero(brk)
    r1 = np.r_[r0[1:], len(pc)] - 1
    span = t[pf[r1]] - t[pf[r0]]
    dens = (r1 - r0 + 1) * dt / np.maximum(span, dt)
    good = (span >= 3.0) & (dens >= 0.3) & np.isin(pc[r0], list(peak))
    ev_t, ev_c = [], []
    for a, b in zip(r0[good], r1[good]):
        if pf[a] > 0 and t[pf[a]] - t[0] > 0.5:
            ev_t.append(t[pf[a]]); ev_c.append(pc[a])             # appears
        if pf[b] < n - 1 and t[-1] - t[pf[b]] > 0.5:
            ev_t.append(t[min(pf[b] + 1, n - 1)]); ev_c.append(pc[a])   # vanishes
    if len(ev_t) < cfg.cut_min_cells:
        return np.zeros(0, np.int64)
    o = np.argsort(ev_t)
    et, ec = np.asarray(ev_t)[o], np.asarray(ev_c)[o]
    cuts, i = [], 0
    while i < len(et):
        j = np.searchsorted(et, et[i] + 0.2, side="right")
        if len(set(ec[i:j].tolist())) >= cfg.cut_min_cells:
            cuts.append(int(np.searchsorted(t, np.median(et[i:j]) - 1e-9)))
            i = j
        else:
            i += 1
    return np.asarray(cuts, np.int64)


def _readopt(tr: Track2D, candidates: list[list[ShuttleCandidate]], width: float,
             height: float, cfg: LinkerConfig) -> Track2D:
    """Interpolated frames that hold a strong, non-static candidate within
    readopt_gate of the fill take that detection. Real video is not always
    sampled on a clean clock (field-order swaps, blended pulldown: every other
    frame lags half a step); the DP rightly refuses such a frame as motion,
    but the detection is still the best estimate of where the shuttle is in
    that image, and it is what the labels (and players) see."""
    off, fidx, xs, ys, score = _flatten(candidates, cfg, width, height)
    if len(xs) == 0:
        return tr
    t = np.asarray(tr.t, float)
    diag = float(np.hypot(width, height))
    static = _static_costs(fidx, xs, ys, t, diag, cfg, None)
    fill = tr.xy[fidx]
    d = np.hypot(xs - fill[:, 0], ys - fill[:, 1])
    ok = (tr.interpolated[fidx] & (score >= cfg.strong_score) & (static < 1.0)
          & (d <= cfg.readopt_gate * diag))
    if not ok.any():
        return tr
    # … and lies ON the local trajectory (only early/late along it): a sample
    # off the clock does, a detector outlier generally does not
    n = len(t)
    i = np.flatnonzero(ok)
    pt = np.column_stack([xs[i], ys[i]])
    dmin = np.full(len(i), np.inf)
    for g in (-2, -1, 0, 1):
        a_i, b_i = np.clip(fidx[i] + g, 0, n - 1), np.clip(fidx[i] + g + 1, 0, n - 1)
        a, b = tr.xy[a_i], tr.xy[b_i]
        ab = b - a
        lam = np.clip(((pt - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
        dd = np.hypot(*(pt - a - lam[:, None] * ab).T)
        dmin = np.where(np.isfinite(dd), np.minimum(dmin, dd), dmin)
    # the fill must be clearly wrong for that image (≥ 6σ and ≥ 0.35 of the local
    # step); a fill within noise of the detection is the better estimate
    a_i, b_i = np.clip(fidx[i] - 1, 0, n - 1), np.clip(fidx[i] + 1, 0, n - 1)
    loc_step = 0.5 * np.hypot(*(tr.xy[b_i] - tr.xy[a_i]).T)
    far = d[i] >= np.maximum(6.0 * cfg.sigma_px * diag, 0.35 * np.nan_to_num(loc_step))
    ok[i[~((dmin <= 4.0 * cfg.sigma_px * diag) & far)]] = False
    if not ok.any():
        return tr
    i = np.flatnonzero(ok)
    i = i[np.lexsort((-score[i], fidx[i]))]
    i = i[np.r_[True, fidx[i][1:] != fidx[i][:-1]]]       # strongest per frame
    f = fidx[i]
    xy, vis, itp, conf = tr.xy.copy(), tr.visible.copy(), tr.interpolated.copy(), tr.conf.copy()
    raw = np.full_like(xy, np.nan) if tr.raw_xy is None else tr.raw_xy.copy()
    xy[f, 0], xy[f, 1] = xs[i], ys[i]
    raw[f] = xy[f]
    vis[f], itp[f], conf[f] = True, False, score[i]
    return Track2D(t=tr.t, xy=xy, conf=conf, visible=vis, interpolated=itp, raw_xy=raw)


def link_candidates(candidates: list[list[ShuttleCandidate]], meta: VideoMeta,
                    config: LinkerConfig | None = None,
                    scene_cuts: np.ndarray | list[int] | None = None) -> Track2D:
    """Per-frame detector candidates → one clean, gap-filled 2-D shuttle track.
    `scene_cuts` = first frames of new shots (else `meta.scene_cuts` if the
    contract ever gets it); cuts detected from the candidate stream are
    added. Nothing is linked, filled or extrapolated across a cut.
    `raw_xy` = the adopted detection (unsmoothed), NaN elsewhere."""
    cfg = config or LinkerConfig()
    n = len(candidates)
    t = _timestamps(meta, n)
    if n == 0:
        z = np.zeros(0)
        return Track2D(t=t, xy=np.zeros((0, 2)), conf=z, visible=z.astype(bool),
                       interpolated=z.astype(bool), raw_xy=np.zeros((0, 2)))
    width = float(meta.width or 1920)
    height = float(meta.height or 1080)
    diag = float(np.hypot(width, height))
    explicit = scene_cuts if scene_cuts is not None else getattr(meta, "scene_cuts", None)
    # a hole in the presentation clock (video_io.pts_gaps) is a cut: nothing is linked,
    # filled, smoothed or kink-fitted across time the camera never delivered
    cuts = _cut_list(n, explicit, np.r_[_detect_cuts(candidates, t, width, height, cfg),
                                        pts_gaps(t)])
    xy, sc, start = _associate(candidates, t, width, height, cfg, _shot0(n, cuts))
    rep = _find_repeats(t, xy, _probe_candidates(candidates, xy, cfg, width, height), diag, cfg) \
        if cfg.repeat_detect else None
    if rep is not None and len(cuts):
        rep[0][cuts] = False                        # a new shot is never a repeat
    if rep is None:
        tr = _finish(t, xy, sc, _cut_starts(xy, start, cuts), diag, cfg)[0]
        return _readopt(tr, candidates, width, height, cfg)
    keep = ~rep[0]                  # re-link the unique frames on the content clock
    ku = np.flatnonzero(keep)
    cu = np.unique(np.searchsorted(ku, cuts))       # cuts in unique-slot indices
    cu = cu[(cu > 0) & (cu < len(ku))]
    xy, sc, start = _associate([candidates[i] for i in ku], rep[1][ku], width, height, cfg,
                               _shot0(len(ku), cu))
    cand_u = [candidates[i] for i in ku]
    tr_u = _finish(rep[1][ku], xy, sc, _cut_starts(xy, start, cu), diag, cfg)[0]
    return _expand(_readopt(tr_u, cand_u, width, height, cfg), keep, t)


def _track_obs(track: Track2D, prefer_raw: bool = True
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visible observations of a track: the raw adopted detection when the
    track carries it and prefer_raw (re-smoothing from the data, not from a
    previous smoothing), else xy."""
    xy = np.asarray(track.xy, float).copy()
    raw = getattr(track, "raw_xy", None)
    if prefer_raw and raw is not None and np.shape(raw) == xy.shape:
        raw = np.asarray(raw, float)
        xy = np.where(np.isfinite(raw), raw, xy)
    vis = np.asarray(track.visible, bool) & np.isfinite(xy[:, 0])
    xy[~vis] = np.nan
    sc = np.where(vis, np.asarray(track.conf, float), 0.0)
    return xy, sc, np.zeros(len(xy), bool)


def _prepare(track: Track2D, scene_cuts, image_size, prefer_raw: bool = True) -> tuple:
    t = np.asarray(track.t, float)
    xy, sc, start = _track_obs(track, prefer_raw)
    diag = _geometry(*(image_size or (None, None)), xy)
    cuts = _cut_list(len(t), scene_cuts, pts_gaps(t))    # clock holes are cuts too
    return t, xy, sc, _cut_starts(xy, start, cuts), diag, cuts


def smooth_track(track: Track2D, kinks: np.ndarray | None = None,
                 config: LinkerConfig | None = None,
                 image_size: tuple[int, int] | None = None,
                 scene_cuts: np.ndarray | list[int] | None = None) -> Track2D:
    """Re-smooth an existing track from its visible frames (raw_xy when
    present). `kinks` = frame indices of hits (frame nearest each hit; the
    split goes between the adjacent visible frames where the velocity step is
    strongest); None → detect them. Gaps ≤ max_gap_s are filled, flagged
    `interpolated`, never across `scene_cuts`."""
    cfg = config or LinkerConfig()
    t, xy, sc, start, diag, cuts = _prepare(track, scene_cuts, image_size)
    kf = None
    if kinks is not None:
        kf = np.asarray(kinks, np.int64).ravel()
        kf = kf[(kf >= 0) & (kf < len(t))]
    rep = _find_repeats(t, xy, xy, diag, cfg)
    if rep is not None and len(cuts):
        rep[0][cuts] = False
    if rep is None:
        return _finish(t, xy, sc, start, diag, cfg, kf)[0]
    keep = ~rep[0]
    ku = np.flatnonzero(keep)
    if kf is not None:                                # frame → unique-slot index
        kf = np.unique(np.cumsum(keep)[kf] - 1)
    tr_u = _finish(rep[1][ku], xy[ku], sc[ku], start[ku], diag, cfg, kf)[0]
    return _expand(tr_u, keep, t)


def find_kinks(track: Track2D, config: LinkerConfig | None = None,
               image_size: tuple[int, int] | None = None,
               return_scores: bool = False,
               scene_cuts: np.ndarray | list[int] | None = None
               ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Frame indices (sorted, frame nearest the estimated hit time) of sharp
    velocity changes in a track's visible frames — the split points
    smooth_track would use. Score = GLRT z (Rayleigh under
    no-hit; > kink_z, or > kink_z_refine where the residuals demanded a
    split). Use `track.t[k]` for times."""
    cfg = config or LinkerConfig()
    # on xy: a linked track carries the linker's own hit splits as cusps (the
    # residual-driven ones cannot be re-derived from raw points without its
    # chain structure); measured on synthetic rallies smoothing adds no false
    # kinks (30/60/120 fps: 9/10/15 vs 5/11/15 on raw_xy) and +1–3 % recall
    t, xy, sc, start, diag, cuts = _prepare(track, scene_cuts, image_size, prefer_raw=False)
    rep = _find_repeats(t, xy, xy, diag, cfg)
    if rep is not None and len(cuts):
        rep[0][cuts] = False
    if rep is None:
        _tr, kf, _tau, z = _finish(t, xy, sc, start, diag, cfg)
    else:
        keep = ~rep[0]
        ku = np.flatnonzero(keep)
        _tr, kf, _tau, z = _finish(rep[1][ku], xy[ku], sc[ku], start[ku], diag, cfg)
        kf = _kinks_to_frames(kf, keep)
    order = np.argsort(kf, kind="stable")
    kf, z = kf[order], z[order]
    return (kf, z) if return_scores else kf
