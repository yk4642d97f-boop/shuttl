"""
Racket-contact (hit) detection from a 2-D shuttle track, optionally helped by a
calibrated camera and player observations.

Why not "speed maxima": the shuttle is fastest *just after* a contact and drag
decelerates it hard, so a hit is a kinematic change point, not a speed peak.

1. Observations. Only real detections (visible & finite) are used; frames the
   linker gap-filled carry no independent evidence. Isolated outliers are
   removed by a two-sided prediction test (a point must be explained by the
   motion on at least one side of it).
2. Change-point score. For every frame time tau, observations in a window on
   each side (point-count based, >= min_side per side, growing up to
   max_window_s so occlusion gaps around hits are bridged; dt from timestamps)
   are fitted by  H0: one quadratic through tau  and  H1: two quadratics joined
   continuously at tau (velocity and acceleration may jump = racket impulse).
   Lambda(tau) = (RSS0 - RSS1) / s2_eff is the Gaussian likelihood ratio;
   s2_eff = max(sigma^2, H1 misfit per dof) so drag curvature a quadratic can
   not follow is not mistaken for a hit. Inside a gap, the ML tau is where the
   two extrapolated arcs meet, i.e. the hidden contact.
3. Candidates = local maxima of Lambda, NMS (nms_s), tau refined to the ML
   change time on a sub-frame grid with the window fixed. Kinematic log-odds
   from Lambda, speed after the change (slow = carrying / picking up), static
   after (= landing on the floor, not a hit; the rally module uses it), meeting
   point outside the image (= apex of a clear out of view / exit, not a hit).
   Flight starts after dead time (serve contact just before the first
   detection) are extra candidates. So are hidden contacts: a gap of >= 2 s
   that the shuttle left upwards over the top of the frame and re-entered
   downwards there (a low phone behind the baseline does not see the near
   player's overhead contact; one flight cannot stay above the frame that
   long). Its time is the gap midpoint (reconstruct3d refines it where the two
   fitted flights meet), its pixel above the frame.
4. Hitter evidence per candidate: player reach likelihood (wrist + 0.65 m
   racket, else bbox expanded for overhead reach), player side from `side` /
   foot_world / camera floor projection / image order; camera ray test (does
   the pixel ray pass through the near or far hitting volume); a pairwise
   image-y cue between consecutive hits (near contacts appear lower).
5. Sequence decoding (Viterbi over candidates x {near, far}): hits alternate
   near/far; same-side successors only across gaps long enough to hide a
   missed hit; min separation; free restart after dead time; any candidate may
   be rejected. Forward-backward on the accepted chain gives hitter posteriors.

A learned re-scorer can be plugged in with load_learned_hit_model(); it sees
the per-candidate feature vector HIT_FEATURES and its logit is blended in.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import HitEvent, PlayerObservation, Track2D, VideoMeta

logger = logging.getLogger(__name__)

RACKET_REACH_M = 0.65          # racket head beyond the wrist
PLAYER_HEIGHT_M = 1.75         # bbox height ~ standing height (px -> m scale)
LEFT_WRIST, RIGHT_WRIST = 9, 10  # COCO-17

@dataclass
class HitConfig:
    # change-point scan
    window_s: float = 0.15          # nominal half-window
    max_window_s: float = 0.40      # half-window may grow to this to get min_side points
    min_side: int = 3               # observations required on each side of tau
    want_side: int = 5              # windows grow (<= max_window_s) to hold this many
    max_side: int = 24              # cap (keeps 240 fps windows small)
    sigma_px: float | None = None   # detection noise; None = estimate from track
    nms_s: float = 0.20             # candidate NMS radius (decoder enforces min_sep_s)
    refine_steps: int = 9           # sub-frame grid over +-1 frame for tau
    refine_model: str = "1"         # contact time model: '1' two quadratics | 'vc' cubic+kink
    lam_min: float = 12.0           # minimum Lambda to become a candidate
    lam_min_weak: float = 4.0       # weaker kinks qualify only inside a player's reach
    weak_reach_min: float = 0.9     # reach term required for such weak candidates
    lam_ref: float = 15.0           # Lambda at which kinematic log-odds = 0
    # plausibility (speeds normalised by image diagonal per second)
    min_speed_out: float = 0.10     # slower after the change = carried / rolling
    static_px: float = 6.0          # spread below which the shuttle is at rest (per 720p)
    static_s: float = 0.35          # look-ahead / -behind for rest tests
    settle_s: float = 0.8           # comes to rest within this after a change ...
    settle_frac: float = 0.05       # ... never leaving this radius (x diag) = floor bounce
    settle_speed: float = 0.25      # ... and leaves slower than this (diag/s)
    dead_time_s: float = 1.0        # gap before a flight start = new serve
    max_out_of_view_s: float = 4.0  # longer gaps are dead time even at the border
    border_band: float = 0.06       # fraction of W/H: shuttle leaving/entering view
    border_px: float = 8.0          # meeting point this far outside image = not a hit
    # sequence decoder
    min_sep_s: float = 0.22         # two contacts closer than this are impossible
    short_flight_s: float = 0.28    # flight-duration prior plateau [short, long]
    long_flight_s: float = 1.7
    long_flight_sd: float = 0.45
    missed_hit_s: float = 0.6       # same-side successor needs room for two flights
    log_eps_same: float = -3.0      # log prob per missed contact between two accepted
    log_eps_fast_same: float = -9.0 # same-side within missed_hit_s (almost impossible)
    pair_y_weight: float = 1.0      # weight of the pairwise image-y side cue
    player_weight: float = 1.0
    camera_weight: float = 1.0
    min_hit_logodds: float = -4.0   # final gate on a decoded hit's own log-odds
    use_linker_kinks: bool = False  # linker.find_kinks as extra candidates (measured: no gain)
    # Score contacts on the unsmoothed detections (Track2D.raw_xy) where the linker kept them:
    # sim e2e 60 fps hit F1 0.798 -> 0.857, all of it precision (0.755 -> 0.868).
    use_raw_xy: bool = True
    # hidden contact: the shuttle leaves over the top of the frame going up and comes back
    # over the top coming down after >= hidden_gap_s. One flight cannot stay above a low
    # phone's frame that long (dev: 0/0 without a contact at >= 2 s, 102/102 with one; a
    # near player's overhead contact is itself above the frame); below 1.5 s mostly none.
    hidden_contacts: bool = True
    hidden_gap_s: float = 2.0
    hidden_logodds: float = 2.0      # kinematic log-odds of such a candidate (no 2-D kink to score)
    # contact right after a presentation-clock hole (video_io.pts_gaps, robustness F4: the smash contact
    # was the first frame after a 13-frame drop): the kink scan cannot see it (no min_side points before
    # it inside max_window_s). A candidate at the first observation after the gap, scored by one
    # quadratic across the gap vs one on each side (a continuation of the pre-gap flight fits one);
    # the sequence decoder (alternation, flight durations) decides like for any candidate.
    gap_contacts: bool = True
    gap_first_obs_s: float = 0.1     # the first observation after the gap must be this close to it
    gap_side: int = 6                # observations used on each side of the gap
    gap_lam_ref: float = 150.0       # Lambda (quadratic across vs split) at which its log-odds = 0
    gap_logodds_max: float = 2.0     # cap: an inferred contact never outscores a seen kink much
    learned_blend: float = 0.5      # weight of a learned model's logit


HIT_FEATURES = ("log_lam", "log_lam_v", "speed_in", "speed_out", "log_speed_ratio",
                "cos_turn", "vy_flip", "static_before", "static_after", "outside",
                "gap_s", "player_reach", "is_flight_start")

_LEARNED: Callable[[np.ndarray], np.ndarray] | None = None


def load_learned_hit_model(path: str | None) -> bool:
    """Install a learned candidate re-scorer. Supports a numpy logistic model
    (.npz with w, b[, mu, sd] over HIT_FEATURES) or TorchScript (.pt) mapping
    (N, F) features -> (N,) logits. Pass None to uninstall. Returns success."""
    global _LEARNED
    if path is None:
        _LEARNED = None
        return True
    try:
        if path.endswith(".npz"):
            d = np.load(path)
            w, b = np.asarray(d["w"], float), float(d["b"])
            mu = np.asarray(d["mu"], float) if "mu" in d else 0.0
            sd = np.asarray(d["sd"], float) if "sd" in d else 1.0
            _LEARNED = lambda X: ((X - mu) / sd) @ w + b          # noqa: E731
        else:
            import torch
            mod = torch.jit.load(path, map_location="cpu").eval()

            def _run(X: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return mod(torch.as_tensor(X, dtype=torch.float32)).reshape(-1).numpy()
            _LEARNED = _run
        return True
    except Exception:
        logger.exception("could not load learned hit model %s", path)
        _LEARNED = None
        return False


# ── observations ─────────────────────────────────────────────────────────────

def _observed(track: Track2D) -> np.ndarray:
    """Indices of frames with a real (non gap-filled) finite detection."""
    xy = np.asarray(track.xy, float)
    ok = np.isfinite(xy).all(axis=1)
    vis = getattr(track, "visible", None)
    if vis is not None and len(vis) == len(ok) and np.any(np.asarray(vis, bool) & ok):
        ok &= np.asarray(vis, bool)
    else:
        interp = getattr(track, "interpolated", None)
        if interp is not None and len(interp) == len(ok):
            ok &= ~np.asarray(interp, bool)
    return np.flatnonzero(ok)


def _content_clock(frames: np.ndarray, xy: np.ndarray, track_t: np.ndarray,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Handle repeated video frames (e.g. 25p broadcast in a 30p container
    repeats every 6th frame; phones duplicate frames under load). A repeated
    frame shows the previous image, so the detection is identical while the
    shuttle moves. Returns (keep mask, content times). If repeats are periodic
    the whole clock is re-timed (unique frames are uniformly spaced in real
    time); otherwise isolated repeats are just dropped."""
    n = len(frames)
    keep = np.ones(n, bool)
    tc = track_t[frames].astype(float)
    if n < 4:
        return keep, tc
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    consec = np.diff(frames) == 1
    rep = np.zeros(n, bool)
    # obs i repeats obs i-1: tiny step between them, normal motion on both sides
    s_prev, s_cur, s_next = step[:-2], step[1:-1], step[2:]
    around = np.minimum(s_prev, s_next)
    rep[2:-1] = (consec[1:-1] & consec[:-2] & consec[2:] & (around > 3.0)
                 & (s_cur < np.maximum(0.25 * around, 0.35)))
    if not rep.any():
        return keep, tc
    keep &= ~rep
    fr = frames[rep]
    if len(fr) >= 4:
        d = np.diff(fr)
        vals, cnt = np.unique(d, return_counts=True)
        P = int(vals[np.argmax(cnt)])
        ok_mult = np.mean((d % P) == 0) if P >= 3 else 0.0
        if P >= 3 and ok_mult >= 0.8:
            ph_vals, ph_cnt = np.unique(fr % P, return_counts=True)
            phase = int(ph_vals[np.argmax(ph_cnt)])
            all_f = np.arange(len(track_t))
            is_dup = (all_f - phase) % P == 0
            c = np.cumsum(~is_dup) - 1                    # content index per slot
            dt = float(np.median(np.diff(track_t))) if len(track_t) > 1 else 1 / 30
            t_content = track_t[0] + c * dt * P / (P - 1)
            keep &= ~is_dup[frames]
            return keep, t_content[frames]
    return keep, tc


def _estimate_sigma(t: np.ndarray, xy: np.ndarray, default: float = 1.5) -> float:
    """Robust per-axis noise from 3rd differences of consecutive, moving samples
    (var(d3) = 20 sigma^2 for white noise; jerk of real flight is small)."""
    if len(t) < 8:
        return default
    dt = np.diff(t)
    med = np.median(dt)
    uni = np.abs(dt - med) < 0.25 * med
    ok = uni[:-2] & uni[1:-1] & uni[2:]
    d3 = xy[3:] - 3 * xy[2:-1] + 3 * xy[1:-2] - xy[:-3]
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    moving = step[:-2] > 0.5                   # static stretches say nothing about noise
    sel = ok & moving
    if sel.sum() < 6:
        return default
    v = d3[sel].ravel()
    s = 1.4826 * np.median(np.abs(v - np.median(v))) / math.sqrt(20.0)
    return float(np.clip(s, 0.5, 8.0))


def _reject_outliers(t: np.ndarray, xy: np.ndarray, sigma: float, diag: float) -> np.ndarray:
    """Keep-mask. A point is an outlier if constant-velocity extrapolations
    from BOTH sides (two neighbours each) miss it by far more than noise plus a
    motion-proportional allowance (hard drag deceleration)."""
    n = len(t)
    keep = np.ones(n, bool)
    if n < 5:
        return keep
    for _ in range(2):
        idx = np.flatnonzero(keep)
        tt, pp = t[idx], xy[idx]
        m = len(idx)
        if m < 5:
            break
        err_l = np.full(m, np.inf)
        err_r = np.full(m, np.inf)
        a, b = pp[:-2], pp[1:-1]
        ta, tb = tt[:-2], tt[1:-1]
        v = (b - a) / (tb - ta)[:, None]
        pred = b + v * (tt[2:] - tb)[:, None]
        err_l[2:] = np.linalg.norm(pp[2:] - pred, axis=1)
        step_l = np.full(m, np.inf)
        step_l[2:] = np.linalg.norm(v, axis=1) * (tt[2:] - tb)
        c, d = pp[2:], pp[1:-1]
        tc, td = tt[2:], tt[1:-1]
        v2 = (d - c) / (td - tc)[:, None]
        pred2 = d + v2 * (tt[:-2] - td)[:, None]
        err_r[:-2] = np.linalg.norm(pp[:-2] - pred2, axis=1)
        step_r = np.full(m, np.inf)
        step_r[:-2] = np.linalg.norm(v2, axis=1) * (td - tt[:-2])
        tol_l = 6 * sigma + 0.02 * diag + 1.0 * np.where(np.isfinite(step_l), step_l, 0)
        tol_r = 6 * sigma + 0.02 * diag + 1.0 * np.where(np.isfinite(step_r), step_r, 0)
        bad = (err_l > tol_l) & (err_r > tol_r)
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep

# ── change-point scan ────────────────────────────────────────────────────────

def _windows(t: np.ndarray, taus: np.ndarray, cfg: HitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per tau: observation index range [lo, hi) and the split index m
    (points < m are left of tau). Windows hold >= min_side points per side if
    reachable within max_window_s, and at most max_side."""
    n = len(t)
    m = np.searchsorted(t, taus, side="left")          # t[m-1] < tau <= t[m]
    # nominal windows span window_s of *observations* beyond the gap edges, so
    # a contact hidden in an occlusion gap still gets well-measured arcs
    t_l = np.where(m > 0, t[np.clip(m - 1, 0, n - 1)], taus)
    t_r = np.where(m < n, t[np.clip(m, 0, n - 1)], taus)
    lo_w = np.searchsorted(t, np.minimum(t_l, taus) - cfg.window_s, side="left")
    hi_w = np.searchsorted(t, np.maximum(t_r, taus) + cfg.window_s, side="right")
    lo_x = np.searchsorted(t, taus - cfg.max_window_s, side="left")
    hi_x = np.searchsorted(t, taus + cfg.max_window_s, side="right")
    want = max(cfg.want_side, cfg.min_side)
    lo = np.maximum(np.minimum(lo_w, np.maximum(m - want, lo_x)), lo_x)
    hi = np.minimum(np.maximum(hi_w, np.minimum(m + want, hi_x)), hi_x)
    lo = np.maximum(lo, m - cfg.max_side)
    hi = np.minimum(hi, m + cfg.max_side)
    return np.clip(lo, 0, n), np.clip(hi, 0, n), m


def _fit_batch(t: np.ndarray, xy: np.ndarray, taus: np.ndarray, lo: np.ndarray,
               hi: np.ndarray, m: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """Least-squares fits, all continuous at tau, for many taus at once:
    H0 one quadratic, Hv quadratic + velocity jump, H1 velocity + acceleration
    jump, and the drag-tolerant pair H0c (one cubic: smoothly varying drag
    deceleration) / Hvc (cubic + velocity jump). Returns RSS per model, point
    counts and the Hvc kinematics at tau (contact pixel, velocity in / out)."""
    B = len(taus)
    width = int(np.max(hi - lo)) if B else 0
    width = max(width, 1)
    j = lo[:, None] + np.arange(width)[None, :]
    valid = j < hi[:, None]
    jc = np.minimum(j, len(t) - 1)
    s = (t[jc] - taus[:, None]) / scale                # normalised time
    W = valid.astype(float)
    right = (jc >= m[:, None]) & valid
    left = valid & ~right
    sp = np.where(right, s, 0.0)
    ones = np.ones_like(s)
    s2, s3 = s * s, s * s * s
    designs = {"0": [ones, s, s2], "v": [ones, s, s2, sp], "1": [ones, s, s2, sp, sp * sp],
               "0c": [ones, s, s2, s3], "vc": [ones, s, s2, s3, sp]}
    Y = xy[jc]                                          # (B, w, 2)
    out: dict[str, np.ndarray] = {"n_left": left.sum(1), "n_right": right.sum(1)}
    for name, cols in designs.items():
        X = np.stack(cols, axis=-1)
        Xw = X * W[..., None]
        A = np.einsum("bwi,bwj->bij", Xw, X)
        A += 1e-9 * np.eye(X.shape[-1])[None]
        rhs = np.einsum("bwi,bwk->bik", Xw, Y)
        beta = np.linalg.solve(A, rhs)                   # (B, p, 2)
        res = (Y - np.einsum("bwi,bik->bwk", X, beta)) * W[..., None]
        out["rss" + name] = np.einsum("bwk,bwk->b", res, res)
        if name == "vc":
            out["p"] = beta[:, 0, :]
            out["v_in"] = beta[:, 1, :] / scale
            out["v_out"] = (beta[:, 1, :] + beta[:, 4, :]) / scale
    return out


def _lambda(fit: dict[str, np.ndarray], sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Likelihood-ratio statistics with a misfit-aware noise variance:
    lam (quadratic vs two quadratics) and lam_v (cubic vs cubic + velocity
    kink = the contact statistic; the cubic absorbs drag, so the strong
    deceleration just after a contact is not mistaken for a second one)."""
    n_obs = 2 * (fit["n_left"] + fit["n_right"])       # x and y
    dof1 = np.maximum(n_obs - 10, 1)
    s2 = np.maximum(sigma ** 2, np.minimum(fit["rss1"], fit["rssvc"]) / dof1)
    lam = (fit["rss0"] - fit["rss1"]) / s2
    lam_v = (fit["rss0c"] - fit["rssvc"]) / s2
    return lam, lam_v, s2


def _scan(t: np.ndarray, xy: np.ndarray, taus: np.ndarray, cfg: HitConfig,
          sigma: float, chunk: int = 2048) -> dict[str, np.ndarray]:
    """Lambda(tau) over a grid of taus (chunked to bound memory)."""
    T = len(taus)
    lam = np.zeros(T)
    lam_v = np.zeros(T)
    ok = np.zeros(T, bool)
    lo, hi, m = _windows(t, taus, cfg)
    nl, nr = m - lo, hi - m
    good = (nl >= cfg.min_side) & (nr >= cfg.min_side)
    gi = np.flatnonzero(good)
    for c0 in range(0, len(gi), chunk):
        sel = gi[c0:c0 + chunk]
        f = _fit_batch(t, xy, taus[sel], lo[sel], hi[sel], m[sel], cfg.window_s)
        l1, lv, _ = _lambda(f, sigma)
        lam[sel], lam_v[sel], ok[sel] = l1, lv, True
    return {"lam": lam, "lam_v": lam_v, "ok": ok, "lo": lo, "hi": hi, "m": m}

# ── candidates ───────────────────────────────────────────────────────────────

@dataclass
class _Cand:
    tau: float
    frame: int
    xy: np.ndarray
    lam: float
    lam_v: float
    v_in: np.ndarray
    v_out: np.ndarray
    gap_s: float
    is_start: bool = False
    static_before: bool = False
    static_after: bool = False
    outside: bool = False
    settles: bool = False        # small bounce then rest: landing, not a contact
    kin: float = 0.0             # kinematic log-odds of being a contact
    reach: float = 0.0           # player-proximity log-odds term
    side_llr: float = 0.0        # log p(near)/p(far) from unary cues
    near_tid: int | None = None
    far_tid: int | None = None
    seg: int = 0                 # rally segment (between dead times / rest)
    weak: bool = False           # below lam_min: kept only if a player is in reach
    hidden: bool = False         # contact above the frame inside a top-exit/top-entry gap
    gap_contact: bool = False    # inferred at the first observation after a PTS gap (F4)
    feats: np.ndarray | None = None


def _local_maxima(lam: np.ndarray, ok: np.ndarray, thr: float) -> np.ndarray:
    x = np.where(ok, lam, -np.inf)
    left = np.concatenate([[-np.inf], x[:-1]])
    right = np.concatenate([x[1:], [-np.inf]])
    return np.flatnonzero((x >= left) & (x > right) & (x >= thr))


def _nms(times: np.ndarray, scores: np.ndarray, radius: float) -> list[int]:
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(abs(times[i] - times[k]) >= radius for k in keep):
            keep.append(int(i))
    return sorted(keep, key=lambda k: times[k])


def _is_static(t: np.ndarray, xy: np.ndarray, t0: float, t1: float, tol: float) -> bool:
    sel = (t > t0) & (t < t1)
    if sel.sum() < 3:
        return False
    p = xy[sel]
    return bool(np.max(np.linalg.norm(p - np.median(p, axis=0), axis=1)) < tol)


def _dead_gaps(t: np.ndarray, xy: np.ndarray, cfg: HitConfig, size: tuple[float, float]) -> np.ndarray:
    """Indices i where the gap t[i] -> t[i+1] is dead time (play interrupted).
    A gap where the shuttle left AND re-entered through the image border (a
    high clear above a low camera) is out-of-view flight, not dead time."""
    W, H = size
    bx, by = cfg.border_band * W, cfg.border_band * H

    def at_border(p: np.ndarray) -> bool:
        return bool(p[0] < bx or p[0] > W - bx or p[1] < by or p[1] > H - by)

    out = []
    for i in np.flatnonzero(np.diff(t) > cfg.dead_time_s):
        if (t[i + 1] - t[i] <= cfg.max_out_of_view_s and at_border(xy[i])
                and at_border(xy[i + 1])):
            continue
        out.append(int(i))
    return np.array(out, int)


def _settles(t: np.ndarray, xy: np.ndarray, tau: float, cfg: HitConfig, tol: float,
             radius: float) -> bool:
    """After tau the shuttle stays within `radius` of where it is and ends at
    rest before settle_s: a floor landing with a small bounce/roll."""
    sel = (t > tau) & (t < tau + cfg.settle_s)
    if sel.sum() < 4:
        return False
    p = xy[sel]
    if float(np.max(np.linalg.norm(p - p[0], axis=1))) >= radius:
        return False
    tt = t[sel]
    # labels / detections of a shuttle lying on the floor jitter by a few px
    return _is_static(t, xy, float(tt[-1]) - cfg.static_s, float(tt[-1]) + 1e-9, 2.0 * tol)


def _rally_breaks(t: np.ndarray, xy: np.ndarray, cfg: HitConfig, tol: float,
                  dead: np.ndarray) -> np.ndarray:
    """Times where play is certainly interrupted: observation gaps longer than
    dead_time_s, and the start of every stretch where the shuttle rests
    (static for >= static_s: held before a serve, lying on the floor)."""
    br = [0.5 * (t[i] + t[i + 1]) for i in dead]
    j = 0
    was_static = False
    for i in range(len(t)):
        while t[j] < t[i] - cfg.static_s:
            j += 1
        span = xy[j:i + 1]
        st = (i - j >= 2 and t[i] - t[j] >= 0.8 * cfg.static_s
              and float(np.max(np.linalg.norm(span - np.median(span, axis=0), axis=1))) < tol)
        if st and not was_static:
            br.append(float(t[j]))
        was_static = st
    return np.array(sorted(br))


def _refine(t: np.ndarray, xy: np.ndarray, tau: float, lo: int, hi: int, dt: float,
            cfg: HitConfig, sigma: float) -> dict[str, np.ndarray] | None:
    """ML change time with the point set fixed, on a sub-frame grid over
    +-1 frame around tau widened to the whole observation gap bracketing it
    (inside an occlusion gap the contact is where the two arcs meet, which the
    coarse scan can miss by several frames)."""
    n = len(t)
    m0 = int(np.searchsorted(t, tau, side="left"))
    a = min(tau - dt, t[m0 - 2] if m0 >= 2 else tau - dt)
    b = max(tau + dt, t[m0 + 1] if m0 + 1 < n else tau + dt)
    steps = max(cfg.refine_steps, int(math.ceil((b - a) / dt * 4)) + 1)
    taus = np.linspace(a, b, steps)
    m = np.searchsorted(t, taus, side="left")
    ok = (m - lo >= cfg.min_side) & (hi - m >= cfg.min_side)
    if not ok.any():
        return None
    taus, m = taus[ok], m[ok]
    k = len(taus)
    f = _fit_batch(t, xy, taus, np.full(k, lo), np.full(k, hi), m, cfg.window_s)
    key = {"vc": "rssvc", "1": "rss1", "v": "rssv"}.get(cfg.refine_model, "rssvc")
    b = int(np.argmin(f[key]))
    lam, lam_v, _ = _lambda(f, sigma)
    return {"tau": taus[b], "m": m[b], "lam": lam[b], "lam_v": lam_v[b],
            **{kk: f[kk][b] for kk in ("p", "v_in", "v_out")}}


def _kinematic_logodds(c: _Cand, diag: float, cfg: HitConfig) -> float:
    if c.hidden:
        return cfg.hidden_logodds
    if c.gap_contact:
        lo = min(1.5 * math.log(max(c.lam, 1e-6) / cfg.gap_lam_ref), cfg.gap_logodds_max)
        s_out = float(np.linalg.norm(c.v_out)) / diag
        lo -= 4.0 * float(np.clip((cfg.min_speed_out - s_out) / cfg.min_speed_out, 0, 1))
        if c.static_after or (c.settles and s_out < cfg.settle_speed):
            lo -= 6.0                     # at rest after the gap: nothing was launched
        if c.outside:
            lo -= 5.0
        return float(np.clip(lo, -8.0, 8.0))
    s_in = float(np.linalg.norm(c.v_in)) / diag
    s_out = float(np.linalg.norm(c.v_out)) / diag
    if c.is_start:
        lo = 1.0
    else:
        lo = 1.5 * math.log(max(c.lam_v, 1e-6) / cfg.lam_ref)
        cos = float(np.dot(c.v_in, c.v_out) / (np.linalg.norm(c.v_in) * np.linalg.norm(c.v_out) + 1e-12))
        if cos > 0.95 and s_out < 1.2 * s_in:
            lo -= 2.0                     # decelerating continuation: drag misfit
    lo -= 4.0 * float(np.clip((cfg.min_speed_out - s_out) / cfg.min_speed_out, 0, 1))
    if c.static_after or (c.settles and s_out < cfg.settle_speed):
        lo -= 6.0                         # came to rest (maybe after a bounce): landing
    if c.outside:
        lo -= 5.0                         # arcs meet outside the image: apex/exit
    return float(np.clip(lo, -8.0, 8.0))


def _hidden_contacts(t: np.ndarray, xy: np.ndarray, cfg: HitConfig,
                     size: tuple[float, float]) -> list[tuple[float, int, np.ndarray]]:
    """(tau, obs index after the gap, contact pixel) for out-of-view gaps of
    hidden_gap_s .. max_out_of_view_s that the shuttle left upwards through the
    top border band and re-entered downwards through it. tau = gap midpoint
    (reconstruct3d refines it where the two fitted flights meet); the pixel is
    above the frame, between exit and entry columns (the camera-ray side test
    reads it as a high contact near the camera)."""
    W, H = size
    by = cfg.border_band * H
    out = []
    n = len(t)
    for i in np.flatnonzero(np.diff(t) >= cfg.hidden_gap_s):
        i = int(i)
        if i < 2 or i + 3 > n or t[i + 1] - t[i] > cfg.max_out_of_view_s:
            continue
        if not (xy[i, 1] < by and xy[i + 1, 1] < by):
            continue
        if not (xy[i, 1] < xy[i - 2, 1] and xy[i + 2, 1] > xy[i + 1, 1]):   # leaves up, comes down
            continue
        uv = np.array([0.5 * (xy[i, 0] + xy[i + 1, 0]), -0.1 * H])
        out.append((float(0.5 * (t[i] + t[i + 1])), i + 1, uv))
    return out


def _gap_contacts(t: np.ndarray, xy: np.ndarray, oi: np.ndarray, track_t: np.ndarray, cfg: HitConfig,
                  sigma: float) -> list[tuple[float, int, float, np.ndarray, np.ndarray]]:
    """(tau, obs index after the gap, Lambda, v_in, v_out) per presentation-clock gap
    (video_io.pts_gaps of the track clock) whose first observation after it is within
    gap_first_obs_s and that has >= min_side observations on each side (up to gap_side).
    tau = that observation's time (the contact is at the first frame after the gap or
    unseen inside it). Lambda = quadratic across the gap vs a quadratic on each side
    (_lambda of _fit_batch 'lam'): a flight that simply continues through the hole fits one."""
    from app.services.tracking.video_io import pts_gaps
    out = []
    for g in pts_gaps(track_t):
        m = int(np.searchsorted(oi, g))
        if m < 1 or m >= len(t) or t[m] - track_t[g] > cfg.gap_first_obs_s:
            continue
        nl, nr = min(cfg.gap_side, m), min(cfg.gap_side, len(t) - m)
        if nl < cfg.min_side or nr < cfg.min_side:
            continue
        tau = np.array([0.5 * (t[m - 1] + t[m])])
        f = _fit_batch(t, xy, tau, np.array([m - nl]), np.array([m + nr]), np.array([m]), cfg.window_s)
        lam, _, _ = _lambda(f, sigma)
        out.append((float(t[m]), m, float(lam[0]), np.asarray(f["v_in"][0], float), np.asarray(f["v_out"][0], float)))
    return out


def _flight_starts(t: np.ndarray, xy: np.ndarray, cfg: HitConfig, diag: float,
                   dt: float, dead: np.ndarray, t_first: float) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    """(tau, obs index, xy, v_out) for flights that begin after dead time."""
    out = []
    starts = np.concatenate([[0], dead + 1]).astype(int)
    for i in starts:
        if i == 0 and t[0] - t_first < 2.0 * dt:
            continue                     # track opens mid-flight: contact is before the video
        j = np.searchsorted(t, t[i] + cfg.window_s, side="right")
        j = max(j, i + cfg.min_side)
        if j > len(t) or t[j - 1] - t[i] > cfg.max_window_s:
            continue
        tt, pp = t[i:j] - t[i], xy[i:j]
        A = np.column_stack([np.ones_like(tt), tt])
        coef, *_ = np.linalg.lstsq(A, pp, rcond=None)
        v = coef[1]
        if np.linalg.norm(v) / diag >= cfg.min_speed_out:
            out.append((float(t[i] - 0.5 * dt), int(i), pp[0].copy(), v))
    return out

# ── hitter evidence ──────────────────────────────────────────────────────────

class _PlayerIndex:
    """Player observations by frame with a resolved court side per observation."""

    def __init__(self, players: list[PlayerObservation], camera: CameraModel | None):
        self.by_frame: dict[int, list[tuple[PlayerObservation, str]]] = {}
        unknown: list[PlayerObservation] = []
        for o in players:
            s = self._side(o, camera)
            if s == "unknown":
                unknown.append(o)
            self.by_frame.setdefault(int(o.frame), []).append((o, s))
        if unknown:
            self._resolve_by_image_order(unknown)

    @staticmethod
    def _side(o: PlayerObservation, camera: CameraModel | None) -> str:
        from app.services.tracking.court_model import side_of
        if o.side in ("near", "far"):
            return o.side
        fw = o.foot_world
        if fw is not None and np.all(np.isfinite(np.asarray(fw, float)[:2])):
            return side_of(float(fw[0]))
        if camera is not None:
            b = np.asarray(o.bbox, float)
            p = camera.pixel_to_floor(np.array([[(b[0] + b[2]) / 2, b[3]]]))[0]
            if np.all(np.isfinite(p)):
                return side_of(float(p[0]))
        return "unknown"

    def _resolve_by_image_order(self, unknown: list[PlayerObservation]) -> None:
        """No geometry: lower bbox bottom (larger y) = near. Threshold from a
        1-D 2-means over all bbox bottoms (robust when one player is missing)."""
        ys = np.array([float(np.asarray(o.bbox)[3]) for o in unknown])
        c = np.percentile(ys, [20, 80]).astype(float)
        for _ in range(10):
            a = np.abs(ys[:, None] - c[None]).argmin(1)
            c = np.array([ys[a == k].mean() if np.any(a == k) else c[k] for k in (0, 1)])
        thr = c.mean() if abs(c[1] - c[0]) > 1 else np.inf
        ids = {id(o) for o in unknown}
        for f, lst in self.by_frame.items():
            self.by_frame[f] = [(o, ("near" if float(np.asarray(o.bbox)[3]) > thr else "far"))
                                if id(o) in ids else (o, s) for o, s in lst]

    def around(self, frame: int, radius: int) -> list[tuple[PlayerObservation, str]]:
        """Nearest observation per track within +-radius frames."""
        best: dict[int, tuple[int, PlayerObservation, str]] = {}
        for d in range(-radius, radius + 1):
            for o, s in self.by_frame.get(frame + d, ()):
                if o.track_id not in best or abs(d) < best[o.track_id][0]:
                    best[o.track_id] = (abs(d), o, s)
        return [(o, s) for _, o, s in best.values()]


def _reach_distance(o: PlayerObservation, uv: np.ndarray) -> float:
    """Distance (in units of the player's bbox height) from pixel uv to the
    region a racket held by this player can reach: a disc of racket (+ slack)
    radius around a confident wrist keypoint if there is one (precise, and it
    separates the players where their reach boxes overlap in the image),
    otherwise the bbox expanded (more upward for overhead contacts)."""
    x1, y1, x2, y2 = (float(v) for v in np.asarray(o.bbox, float)[:4])
    h = max(y2 - y1, 1.0)
    kp = o.keypoints
    if kp is not None:
        kp = np.asarray(kp, float)
        r = (RACKET_REACH_M / PLAYER_HEIGHT_M) * h * 1.25
        dw = [max(0.0, float(np.hypot(*(uv - kp[k, :2]))) - r) / h
              for k in (LEFT_WRIST, RIGHT_WRIST)
              if kp.shape[0] > k and kp[k, 2] > 0.3 and np.all(np.isfinite(kp[k, :2]))]
        if dw:
            return min(dw)
    dx = max(x1 - 0.45 * h - uv[0], 0.0, uv[0] - (x2 + 0.45 * h))
    dy = max(y1 - 0.65 * h - uv[1], 0.0, uv[1] - (y2 + 0.05 * h))
    return math.hypot(dx, dy) / h


def _player_evidence(c: _Cand, idx: _PlayerIndex, radius: int, cfg: HitConfig) -> None:
    """Fill reach (hit log-odds term), side_llr (player part) and track ids."""
    obs = idx.around(c.frame, radius)
    if not obs:
        return
    best = {"near": (-np.inf, None), "far": (-np.inf, None)}
    for o, s in obs:
        if s not in best:
            continue
        lp = -0.5 * (_reach_distance(o, c.xy) / 0.15) ** 2
        if lp > best[s][0]:
            best[s] = (lp, o.track_id)
    ln, lf = best["near"][0], best["far"][0]
    top = max(ln, lf)
    if np.isfinite(top):
        c.reach = cfg.player_weight * float(np.clip(1.0 + top, -3.0, 1.0))
    c.near_tid = best["near"][1] if ln > -4.5 else None
    c.far_tid = best["far"][1] if lf > -4.5 else None
    ln = ln if np.isfinite(ln) else -3.0      # side not seen: weak default
    lf = lf if np.isfinite(lf) else -3.0
    c.side_llr += cfg.player_weight * float(np.clip(ln - lf, -8.0, 8.0))


def _height_prior(z: np.ndarray) -> np.ndarray:
    """Unnormalised density of contact heights (m)."""
    return np.clip((z - 0.05) / 0.25, 0, 1) * np.clip((3.3 - z) / 0.5, 0, 1)


def _ray_side_llr(camera: CameraModel, uv: np.ndarray, doubles_margin: float = 1.2) -> float:
    """log P(near)/P(far) that a contact seen at pixel uv lies in the near or
    far hitting volume (court + margins, contact height prior, volume-uniform
    prior along the ray)."""
    from app.services.tracking import court_model as cm
    o, d = camera.ray(np.asarray(uv, float)[None])
    lam = np.linspace(0.5, 45.0, 400)
    P = o[None] + lam[:, None] * d[0][None]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    w = lam ** 2 * _height_prior(z)
    w *= (np.abs(y) <= cm.DOUBLES_HALF_WIDTH + doubles_margin)
    w *= (np.abs(x) <= cm.HALF_LENGTH + 1.8)
    near = float(w[x < -0.05].sum())
    far = float(w[x > 0.05].sum())
    if near + far <= 0:
        return 0.0
    eps = 1e-3 * (near + far)
    return float(np.clip(math.log((near + eps) / (far + eps)), -6.0, 6.0))

# ── sequence decoding ────────────────────────────────────────────────────────

_SIDES = ("near", "far")


def _flight_logprior(d: float, cfg: HitConfig) -> float:
    """Log prior of one flight lasting d seconds, 0 on the plateau of normal
    rally tempo (drives/kills ~0.3 s ... high serves/clears ~1.8 s)."""
    if d > cfg.long_flight_s:
        return -((d - cfg.long_flight_s) / cfg.long_flight_sd) ** 2
    if d < cfg.short_flight_s:
        return -((cfg.short_flight_s - d) / 0.08) ** 2
    return 0.0


def _log_sigmoid(x: float) -> float:
    return -math.log1p(math.exp(-x)) if x > -30 else x


class _Decoder:
    """Viterbi over candidates x {near, far} with alternation/timing priors,
    then forward-backward on the accepted chain for hitter posteriors."""

    def __init__(self, cands: list[_Cand], cfg: HitConfig, diag: float, pair_w: float):
        self.c, self.cfg, self.diag, self.pair_w = cands, cfg, diag, pair_w
        self.hit = np.array([c.kin + c.reach for c in cands])
        # side log-likelihoods normalised so uninformative evidence scores 0
        self.side = np.array([[_log_sigmoid(c.side_llr) + math.log(2),
                               _log_sigmoid(-c.side_llr) + math.log(2)] for c in cands])

    def _pair(self, j: int, k: int, sj: int, sk: int) -> float:
        """Image-y ordering cue: of two consecutive contacts on opposite sides
        the near one appears lower (larger y). Penalty only (never rewards)."""
        if sj == sk or self.pair_w <= 0:
            return 0.0
        dy = (self.c[k].xy[1] - self.c[j].xy[1]) / (0.05 * self.diag)
        sign = 1.0 if sk == 0 else -1.0             # k near => expect dy > 0
        return -self.pair_w * float(np.clip(-sign * dy, 0.0, 3.0))

    def trans(self, j: int, k: int, sj: int, sk: int) -> float:
        """Transition log-prob between consecutive accepted contacts of one
        rally segment. The contacts in between that were missed (m >= 0, parity
        fixed by the sides) each cost log_eps_same, and every one of the m+1
        flights must have a plausible duration (flight_logprior of the mean)."""
        gap = self.c[k].tau - self.c[j].tau
        cfg = self.cfg
        if gap < cfg.min_sep_s:
            return -np.inf
        same = sj == sk
        if same and gap < cfg.missed_hit_s:
            return cfg.log_eps_fast_same
        best = -np.inf
        for m in range(1 if same else 0, 8, 2):
            v = m * cfg.log_eps_same + (m + 1) * _flight_logprior(gap / (m + 1), cfg)
            best = max(best, v)
            if gap / (m + 1) < cfg.long_flight_s:
                break
        pair = self._pair(j, k, sj, sk) if gap < 2 * cfg.long_flight_s else 0.0
        return best + pair

    def viterbi(self) -> list[tuple[int, int]]:
        """Best chain. Crossing a rally break (dead time / shuttle at rest) is
        free; within a segment consecutive contacts pay trans()."""
        n = len(self.c)
        if n == 0:
            return []
        V = np.full((n, 2), -np.inf)
        bp: list[list[tuple[int, int] | None]] = [[None, None] for _ in range(n)]
        seg = [c.seg for c in self.c]
        prev_best, prev_arg, s0 = 0.0, None, 0
        for k in range(n):
            if k > 0 and seg[k] != seg[k - 1]:
                for j in range(s0, k):
                    for s in range(2):
                        if V[j, s] > prev_best:
                            prev_best, prev_arg = V[j, s], (j, s)
                s0 = k
            for sk in range(2):
                best, arg = prev_best, prev_arg
                for j in range(s0, k):
                    for sj in range(2):
                        if V[j, sj] == -np.inf:
                            continue
                        v = V[j, sj] + self.trans(j, k, sj, sk)
                        if v > best:
                            best, arg = v, (j, sj)
                V[k, sk], bp[k][sk] = self.hit[k] + self.side[k, sk] + best, arg
        k, s = np.unravel_index(int(np.argmax(V)), V.shape)
        if V[k, s] <= 0:
            return []
        chain: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = (int(k), int(s))
        while cur is not None:
            chain.append(cur)
            cur = bp[cur[0]][cur[1]]
        return chain[::-1]

    def posteriors(self, idx: list[int]) -> np.ndarray:
        """P(near) for each accepted candidate (forward-backward, 2 states;
        no coupling across rally breaks)."""
        m = len(idx)
        if m == 0:
            return np.zeros(0)
        E = self.side[idx]                                     # (m, 2) log
        T = np.zeros((max(m - 1, 0), 2, 2))
        for i in range(m - 1):
            if self.c[idx[i]].seg != self.c[idx[i + 1]].seg:
                continue
            for a in range(2):
                for b in range(2):
                    v = self.trans(idx[i], idx[i + 1], a, b)
                    T[i, a, b] = v if np.isfinite(v) else -50.0
        fw = np.zeros((m, 2))
        fw[0] = E[0]
        for i in range(1, m):
            fw[i] = E[i] + np.logaddexp.reduce(fw[i - 1][:, None] + T[i - 1], axis=0)
        bw = np.zeros((m, 2))
        for i in range(m - 2, -1, -1):
            bw[i] = np.logaddexp.reduce(T[i] + (E[i + 1] + bw[i + 1])[None, :], axis=1)
        post = fw + bw
        post -= np.logaddexp.reduce(post, axis=1, keepdims=True)
        return np.exp(post[:, 0])

# ── entry point ──────────────────────────────────────────────────────────────

def _linker_kink_times(track: Track2D) -> list[float]:
    """Optional extra candidate times (s) from linker.find_kinks, which returns
    frame indices (integers; floats are taken as times, dicts/objects by 't')."""
    try:
        from app.services.tracking import linker
        fk = getattr(linker, "find_kinks", None)
        if fk is None:
            return []
        res = fk(track)
        if isinstance(res, tuple):
            res = res[0]
        t = np.asarray(track.t, float)
        out = []
        for k in (res if res is not None else []):
            if isinstance(k, dict):
                v = k.get("t", k.get("time"))
            elif isinstance(k, (int, np.integer)):
                v = float(t[int(k)]) if 0 <= int(k) < len(t) else None
            elif isinstance(k, (float, np.floating)):
                v = float(k)
            else:
                v = getattr(k, "t", None)
            if v is not None and np.isfinite(v):
                out.append(float(v))
        return out
    except Exception:
        logger.debug("linker.find_kinks unavailable", exc_info=True)
        return []


def _features(c: _Cand, diag: float) -> np.ndarray:
    s_in = float(np.linalg.norm(c.v_in)) / diag
    s_out = float(np.linalg.norm(c.v_out)) / diag
    cos = float(np.dot(c.v_in, c.v_out) / (np.linalg.norm(c.v_in) * np.linalg.norm(c.v_out) + 1e-12))
    return np.array([math.log(max(c.lam, 1e-6)), math.log(max(c.lam_v, 1e-6)), s_in, s_out,
                     math.log((s_out + 1e-3) / (s_in + 1e-3)), cos,
                     float(np.sign(c.v_in[1]) != np.sign(c.v_out[1])), float(c.static_before),
                     float(c.static_after), float(c.outside), c.gap_s, c.reach, float(c.is_start)])


def detect_candidates(track: Track2D, meta: VideoMeta, camera: CameraModel | None = None,
                      players: list[PlayerObservation] | None = None,
                      config: HitConfig | None = None) -> list[_Cand]:
    """All scored hit candidates (before sequence decoding). Useful for debugging
    and as training rows for a learned re-scorer."""
    cfg = config or HitConfig()
    W = meta.width or (camera.image_size[0] if camera is not None else 1280)
    H = meta.height or (camera.image_size[1] if camera is not None else 720)
    diag = math.hypot(W, H)
    oi = _observed(track)
    if len(oi) < 2 * cfg.min_side:
        return []
    track_t = np.asarray(track.t, float)
    xy = np.asarray(track.xy, float)[oi]
    keep, t = _content_clock(oi, xy, track_t)
    oi, t, xy = oi[keep], t[keep], xy[keep]
    sigma = cfg.sigma_px or _estimate_sigma(t, xy)
    keep = _reject_outliers(t, xy, sigma, diag)
    oi, t, xy = oi[keep], t[keep], xy[keep]
    if len(t) < 2 * cfg.min_side:
        return []
    track_t = np.asarray(track.t, float)
    dt = float(np.median(np.diff(track_t))) if len(track_t) > 1 else 1.0 / max(meta.fps, 1.0)
    grid = track_t[(track_t >= t[0]) & (track_t <= t[-1])]
    sc = _scan(t, xy, grid, cfg, sigma)
    score = sc["lam_v"]                     # velocity-kink statistic peaks at the contact
    lam_thr = min(cfg.lam_min, cfg.lam_min_weak) if players else cfg.lam_min
    peaks = list(_local_maxima(score, sc["ok"], lam_thr))
    if cfg.use_linker_kinks:
        for tk in _linker_kink_times(track):
            g = int(np.argmin(np.abs(grid - tk)))
            if sc["ok"][g] and all(abs(grid[g] - grid[p]) > 2 * dt for p in peaks):
                peaks.append(g)
    peaks = [peaks[i] for i in _nms(grid[peaks], score[peaks], cfg.nms_s)] if peaks else []
    cands: list[_Cand] = []
    static_tol = cfg.static_px * diag / math.hypot(1280, 720)
    for g in peaks:
        r = _refine(t, xy, float(grid[g]), int(sc["lo"][g]), int(sc["hi"][g]), dt, cfg, sigma)
        if r is None:
            continue
        m = int(r["m"])
        gap = float(t[m] - t[m - 1]) if 0 < m < len(t) else 0.0
        cands.append(_Cand(tau=float(r["tau"]), frame=0, xy=np.asarray(r["p"], float),
                           lam=float(r["lam"]), lam_v=float(r["lam_v"]),
                           v_in=np.asarray(r["v_in"], float), v_out=np.asarray(r["v_out"], float),
                           gap_s=gap, weak=bool(score[g] < cfg.lam_min)))
    dead = _dead_gaps(t, xy, cfg, (W, H))
    for tau, i, p0, v in _flight_starts(t, xy, cfg, diag, dt, dead, float(track_t[0])):
        if all(abs(c.tau - tau) > cfg.nms_s for c in cands):
            cands.append(_Cand(tau=tau, frame=0, xy=p0, lam=0.0, lam_v=0.0, v_in=np.zeros(2),
                               v_out=np.asarray(v, float),
                               gap_s=float(t[i] - t[i - 1]) if i > 0 else np.inf, is_start=True))
    if cfg.hidden_contacts:
        for tau, i, uv in _hidden_contacts(t, xy, cfg, (W, H)):
            if all(abs(c.tau - tau) > cfg.nms_s for c in cands):
                cands.append(_Cand(tau=tau, frame=0, xy=uv, lam=0.0, lam_v=0.0, v_in=np.zeros(2),
                                   v_out=np.zeros(2), gap_s=float(t[i] - t[i - 1]), hidden=True))
    if cfg.gap_contacts:
        for tau, i, lam, v_in, v_out in _gap_contacts(t, xy, oi, track_t, cfg, sigma):
            if all(abs(c.tau - tau) > cfg.nms_s for c in cands):
                cands.append(_Cand(tau=tau, frame=0, xy=xy[i].copy(), lam=lam, lam_v=0.0, v_in=v_in,
                                   v_out=v_out, gap_s=float(t[i] - t[i - 1]), gap_contact=True))
    cands.sort(key=lambda c: c.tau)
    breaks = _rally_breaks(t, xy, cfg, static_tol, dead)
    for c in cands:
        # a start candidate sits just before its break-ending first detection
        c.seg = int(np.searchsorted(breaks, c.tau + (dt if c.is_start else 0.0)))
    pidx = _PlayerIndex(players, camera) if players else None
    radius = max(1, int(round(0.05 * (meta.fps or 30.0))))
    cam_w = 0.0
    if camera is not None:
        cam_w = cfg.camera_weight * (float(np.clip(camera.confidence, 0.3, 1.0))
                                     if camera.confidence > 0 else 1.0)
    for c in cands:
        c.frame = int(np.clip(np.searchsorted(track_t, c.tau), 0, len(track_t) - 1))
        if c.frame > 0 and abs(track_t[c.frame - 1] - c.tau) < abs(track_t[c.frame] - c.tau):
            c.frame -= 1
        if not c.hidden:                 # a hidden contact has no pixel to test these on
            b = cfg.border_px * diag / math.hypot(1280, 720)
            c.outside = bool(c.xy[0] < -b or c.xy[0] > W + b or c.xy[1] < -b or c.xy[1] > H + b)
            c.static_after = _is_static(t, xy, c.tau, c.tau + cfg.static_s, static_tol)
            c.static_before = _is_static(t, xy, c.tau - cfg.static_s, c.tau, static_tol)
            c.settles = _settles(t, xy, c.tau, cfg, static_tol, cfg.settle_frac * diag)
        c.kin = _kinematic_logodds(c, diag, cfg)
        if pidx is not None and not c.hidden:
            _player_evidence(c, pidx, radius, cfg)
        if cam_w > 0 and np.all(np.isfinite(c.xy)):
            c.side_llr += cam_w * _ray_side_llr(camera, c.xy)
        c.feats = _features(c, diag)
    cands = [c for c in cands if not c.weak or c.reach >= cfg.weak_reach_min * cfg.player_weight]
    if _LEARNED is not None and cands:
        try:
            z = np.asarray(_LEARNED(np.stack([c.feats for c in cands])), float).reshape(-1)
            for c, zi in zip(cands, z):
                c.kin = (1 - cfg.learned_blend) * c.kin + cfg.learned_blend * float(zi)
        except Exception:
            logger.exception("learned hit model failed; using model-based scores")
    return cands


def _prefer_raw(track: Track2D) -> Track2D:
    raw = getattr(track, "raw_xy", None)
    if raw is None or len(raw) != track.n:
        return track
    raw = np.asarray(raw, float)
    has = np.isfinite(raw).all(axis=1)
    return replace(track, xy=np.where(has[:, None], raw, track.xy))


def detect_hits(track: Track2D, meta: VideoMeta, camera: CameraModel | None = None,
                players: list[PlayerObservation] | None = None,
                config: HitConfig | None = None) -> list[HitEvent]:
    """Racket contacts in time order. HitEvent.t is the sub-frame ML contact
    time (s), frame the nearest frame, xy_px the fitted contact pixel,
    hitter 'near'/'far' ('unknown' if the posterior < 0.6), confidence =
    P(contact), player_track_id = the hitter's track if a player was in reach."""
    cfg = config or HitConfig()
    track = _prefer_raw(track) if cfg.use_raw_xy else track
    cands = detect_candidates(track, meta, camera, players, cfg)
    if not cands:
        return []
    diag = math.hypot(meta.width or 1280, meta.height or 720)
    pair_w = cfg.pair_y_weight * (0.5 if (camera is not None or players) else 1.0)
    dec = _Decoder(cands, cfg, diag, pair_w)
    chain = [(k, s) for k, s in dec.viterbi() if dec.hit[k] >= cfg.min_hit_logodds]
    idx = [k for k, _ in chain]
    p_near = dec.posteriors(idx)
    out: list[HitEvent] = []
    for k, pn in zip(idx, p_near):
        c = cands[k]
        side = "near" if pn >= 0.6 else ("far" if pn <= 0.4 else "unknown")
        tid = c.near_tid if side == "near" else (c.far_tid if side == "far" else None)
        out.append(HitEvent(frame=c.frame, t=float(c.tau),
                            xy_px=(float(c.xy[0]), float(c.xy[1])), hitter=side,
                            confidence=float(1.0 / (1.0 + math.exp(-dec.hit[k]))),
                            player_track_id=tid, source="gap_contact" if c.gap_contact else "kink"))
    return out
