"""
Rally segmentation, point outcome (in / out / net -> winner) and BWF scoring.

Segmentation (hits are the backbone):
  1. Hits are grouped into rallies. Consecutive hits a -> b stay in one rally
     unless a "dead ball" cue lies between them:
       * b.t - a.t > max_hit_gap_s (2.6 s; the longest flights, high serves and
         defensive lifts, stay up ~1.5-2.2 s), unless the track shows one
         continuous flight in between (no rest, no unseen stretch > 2.2 s,
         mostly fast: a contact the detector missed);
       * the fitted 3-D flight after a reaches the floor (its landing_t, or
         the drag ODE integrated from p0/v0 when the fit was chained to b)
         more than landing_slack_s before b: nobody hits a shuttle lying on
         the floor;
       * the 2-D track rests (>= static_s within a few px; up to 2 outlier
         samples tolerated) strictly between them: shuttle on the floor or
         held in a hand.
  2. Rally start = serve contact (first hit). If the shuttle was already in
     fast flight between the last rest and the first detected hit of a
     multi-hit group, the serve was missed: the start moves back to where
     that flight began and the receiver is the first listed hitter
     (RallyV2.serve_missed extra). Rally end = fitted floor contact of the
     last flight if reconstructed, else the first 2-D rest after the last
     hit (also after a continuous flight = missed trailing contacts), else
     the last observed frame of that flight.
  3. Non-rally hits and groups are dropped:
       * a contact is a real hit only if a flight follows: a 3-D segment, or
         image speed right after it (median over 0.3 s) >= 0.3 image heights
         / s, or >= 0.08 and 2.5x the speed just before it. The relative test
         matters: perspective scales image speed ~5x between near and far
         court, so a far short serve is slower than a near hand carry, but it
         always jumps out of a still hold. Leading non-flights (pick-ups),
         a leading weak/duplicate contact < 0.6 s before a stronger one, and
         trailing non-flights (catches) are removed;
       * feeds. Returning the shuttle to the server between points (very
         common in amateur play) looks like a one-hit rally. A one-hit group
         gets feed log-odds (prior -1: aces, serve faults and missed returns
         are real one-shot points) from: hitter = loser of the previous rally
         (only the winner may serve; +3, hitter = previous winner: -3, scaled
         down to 0 as the previous call's confidence drops from 0.8 to 0.45);
         the next group is served by the other side (+0.8; the same side
         serving again means it won this point: -1.5); hit soon (< 4 s)
         after the previous rally ended (+0.8; > 8 s: -0.8); held still just
         before (-1.5, a serve stance) vs moving (+1.0, launched out of a
         carry); 3-D only: flight caught in the air instead of landing
         (+1.0), hit from outside every service court (+0.7), launch
         < 45 km/h (+0.3) or > 90 km/h (-1.0, flick / high serve). A second
         "spurious contact" log-odds (weak detector confidence, no 3-D fit
         where others exist, slow) is also computed; drop if either > 0.
         Multi-hit knock-ups between points are NOT detected (failure mode).
       * spurious contacts: a rally's last contact with confidence < 0.8 that is
         followed by < 0.6 s of observed motion (a kink as the flight comes down,
         not a launch) is dropped, its fit kept as the end of the real last flight;
         a lone contact without a 3-D fit (others have one) followed by < 0.3 s of
         motion is not a rally.
  Hitter sides (joint decode, _decode_sides): per rally group a 2-state HMM
     over the contacts. Evidence per contact: the hits.py label, the side its
     own 3-D launch fit reads (launch point / direction) and the half the
     incoming fit is on at the contact time, scored as agreement / conflict
     with the label those fits were made with; transitions alternate unless
     the gap leaves room for a missed contact. Forward-backward gives each hit
     P(side), which multiplies the outcome confidence. (joint_sides=False: the
     old rule, label replaced by the launch side when the fit starts > 1.5 m
     inside the other half.)
  A long hit gap is not dead time when a hold touching the first contact is
     that serve's own hold (_past_hold), or when the arrival at the next
     contact is unseen but the shuttle was last seen in fast flight.
  4. Without hits, moving stretches of the track (>= 1 s, bridging unseen
     gaps up to 2.2 s) become low-confidence rallies without shots.

Outcome (needs a 3-D flight for the last shot or a camera; else 'unknown'):
  * the last flight lands back on the hitter's side, the fit reports
    end_reason 'net', or it crosses x = 0 between the posts below net height
    -> 'net' (hitter loses). A crossing < 12 cm under the tape whose fit saw
    the shuttle come down beyond the net counts as over (net cord);
  * else margin = court_model.in_out_margin(landing, doubles) -> 'in' (hitter
    wins) / 'out' (hitter loses). A one-shot rally is judged against the
    service court diagonal to the server (short / long service lines, centre
    line). Confidence = Phi(|margin| / sigma_landing) x fit quality.
  * a missing landing point is extrapolated with the drag ODE (the rally is
    over, so the shuttle must come down); sigma grows with extrapolation time.
  * if the 2-D track keeps flying > 0.35 s past the end of the last fit (which
    neither landed nor hit the net), the final contact was missed: outcome
    'unknown' (source 'missed_final'), winner left to the next-serve rule.
  * no segment but a camera: a 2-D rest point right after the last hit is
    back-projected to the floor (low confidence, <= 0.4).
  * unknown / low-confidence winners are resolved from who serves the next
    point (BWF: the rally winner serves), replaying the score so that end
    changes are handled; ambiguous cases stay 'unknown'.

Scoring (score_match): BWF rally-point scoring, best of 3 games to 21, a game
at 20-20 continues until a 2-point lead, 29-29 -> next point wins (cap 30);
the rally winner serves next, from the right service court on an even own
score and the left on odd; the game winner serves first in the next game;
ends change after every game and in game 3 when a side first reaches 11.
Rally winners are physical camera sides ('near'/'far'); the scorer tracks
which player stands at which end. RallyV2.confidence = segmentation
confidence x outcome confidence (segmentation only when winner is unknown).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from app.services.tracking import court_model as cm
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, RallyV2, ShotEventV2, ShotSegment3D, Track2D, VideoMeta,
)
from app.services.tracking.physics import DRAG_LENGTH, G

SIDES = ("near", "far")

@dataclass
class RallyConfig:
    max_hit_gap_s: float = 2.6        # longer without a hit -> dead ball ...
    max_hit_gap_missed_s: float = 4.5 # ... unless the track keeps flying (missed hit)
    max_unseen_s: float = 2.2         # ... with no longer unobserved stretch
    max_hit_gap_unseen_s: float = 6.0 # a gap this long may still be <= 3 flights of ~2 s whose
                                      # contacts all went unseen (high lifts leave a low camera's view)
    max_unseen_flight_s: float = 4.5  # ... out of view at most two of the longest (~2.2 s) flights
    min_contact_dv: float = 5.0       # m/s: a racket contact changes the shuttle velocity by more
                                      # (measured >= 10 m/s for 90% of real contacts); less = the same
                                      # flight seen twice (a spurious kink, e.g. just above the floor)
    max_unseen_land_s: float = 1.5    # last seen in flight -> seen at rest within this
    missed_final_s: float = 0.35      # track flies on this long past the last fit's end
    landing_slack_s: float = 0.35     # fitted floor contact this long before the next hit
    min_landing_conf: float = 0.25    # trust landing_t for splitting above this fit confidence
    static_s: float = 0.6             # 2-D stillness this long = shuttle at rest
    static_radius_frac: float = 0.01  # of image height (7 px at 720p; gravity alone
                                      # moves a flying shuttle >100 px in 0.6 s)
    hold_window_s: float = 0.7        # pre-serve hold must end this close to the serve
    min_flight_speed_hps: float = 0.3 # image heights / s right after a real hit
    min_launch_speed_hps: float = 0.08  # ... or this and 2.5x the speed before it
    launch_window_s: float = 0.3
    flight_window_s: float = 0.5
    feed_threshold: float = 0.0       # feed log-odds above this -> dropped
    net_margin_m: float = 0.0         # crossing below net height - margin -> 'net' ...
    net_cord_tol_m: float = 0.12      # ... unless within this and seen landing beyond the net
    default_landing_sigma_m: float = 0.12
    landing_outside_max_m: float = 3.0  # a floor point this far beyond the doubles court (either axis)
                                        # is no landing (flight fitted across a timestamp jump / mis-linked
                                        # track, robustness F3: -11.1 m): no call from it. = ShuttlAnalysis
                                        # LandingGuard.marginM and the old fixed 3 m of _outcome_2d
    override_conf: float = 0.55       # next-serve evidence overrides outcomes below this
    # the next serve CONTRADICTS a geometric call below ns_conflict_max_q, or the verdict rests on a serve
    # that was not seen (inferred from a missed serve): the rally's confidence is capped at
    # ns_conflict_conf (below the app's 0.6 "uncertain" line). Sim s0/s1000/s2000 (ns_conflict.py): a kept
    # call contradicted by a seen next serve was right 17/47 (36 %) at 0.67 stated confidence; verdicts
    # from an inferred serve against / without a call 5/12. 45-min codec pair, rally 77495: H.264 kept far
    # (0.67), HEVC took near from an inferred serve (0.73), both unflagged.
    ns_conflict_conf: float = 0.5
    ns_conflict_max_q: float = 0.8    # a call at or above this is kept confident (>= 0.8: 26/26 right)
    net_call_q: float = 0.75          # net faults are decided near the net, where the hitter
                                      # side (which names the winner) is least certain
    side_conflict_q: float = 0.6      # launch direction contradicts the hitter label (only
                                      # without the joint side decode, joint_sides=False)
    # joint hitter-side decode (see _decode_sides); LLRs are log P(obs | label right) /
    # P(obs | label wrong), measured on sim dev seeds 0-39 (behind camera), rounded
    joint_sides: bool = True
    side_label_llr: float = 2.0       # hits.py label alone: ~88 % right from behind
    side_own_agree: float = 1.6       # own launch fit agrees with the label (it was fitted WITH
                                      # the label as a prior, so agreement is weak evidence)
    side_own_conflict: float = -4.5   # ... contradicts it although pulled towards it: 66/71 wrong
    side_own_mixed: float = -2.2      # launch x and launch vx point at different sides
    side_in_agree: float = 1.5        # incoming fit, propagated to the contact, is on the label's half
    side_in_conflict: float = -3.8    # ... on the other half: 73/85 wrong
    side_unlabelled: float = 2.5      # |LLR| of one fit reading for a hit without a label
    side_fit_q_lo: float = 0.1        # fit readings count x ramp(fit confidence, lo -> hi): a fit
    side_fit_q_hi: float = 0.3        # below lo is not determined (real broadcast clips: start at the
                                      # net, rmse 3-24 px); sim: conflict LLR -4.6 at conf >= 0.3, ~-2 below 0.2
    side_vx_dead: float = 0.5         # m/s: |launch vx| below this reads no direction (net dribble)
    side_x_dead: float = 0.1          # m: a fit position this close to the net reads no side
    side_same_mid_s: float = 1.9      # consecutive contacts on the SAME side need a missed contact
    side_same_width_s: float = 0.15   # between them, i.e. two flights (>= ~0.9 s each): logistic in
                                      # the gap (measured P(same) 0 below 1.2 s, 0.37 at 2 s, 0.89 > 2.1 s)
    side_same_eps: float = 0.02       # floor/ceiling: spurious contacts break the alternation
    next_serve_side_q: bool = True    # next-serve verdict confidence x P(next server's side)
    # spurious contacts after the final one / in dead time (dev: scripts/accuracy/trailing_cue.py,
    # onehit_cue.py). A racket contact launches a flight that is seen moving for most of its
    # duration; a spurious kink (floor approach, hand) is followed by < ~0.5 s of motion.
    trail_max_conf: float = 0.8       # a rally's last contact below this confidence ...
    trail_min_flight_s: float = 0.6   # ... followed by less observed motion is dropped
                                      # (dev: 25/58 trailing spurious, 1/288 real last contacts)
    trail_launch_guard: bool = True   # ... unless its own fit shows a racket launch (_launches): real
                                      # short final flights (kills, net shots < 0.6 s) have low
                                      # confidence too when the detector is less sure than hits.py on sim
                                      # (tests/test_rally generator: 17/17 trim-eligible were real)
    trail_launch_w: float = 0.5       # fit weight (_fit_weight) needed to read a launch
    trail_depart_lag_s: float = 0.1   # ... and the 2-D track this long after the contact must have LEFT
    trail_depart_min: float = 0.01    # the incoming fit's projection by >= this x image height (a new
                                      # flight; real launches >= 0.018 H, sim fit noise ~0.001-0.004 H;
                                      # 3 of 4 mis-read launches on sim stayed within 0.004 H)
    trail_keep_fit: bool = True       # the dropped kink's fit describes the end of the real last flight
    onehit_min_flight_s: float = 0.3  # a lone contact without a 3-D fit (others have one) followed by
                                      # less motion is no rally (dev: 15/29 spurious, 3/44 real)
    next_serve_needs_hold: bool = True  # a serve is struck from a still hold (Laws 9.1): only a first
                                        # contact with the shuttle seen resting just before names the
                                        # server (dev: unseen 0/25, moving 3/10 were real serves)
    skip_serve_hold: bool = True      # long-gap test starts after a hold touching the first contact
                                      # (7/100 in-rally long gaps were split only by the serve's hold)
    unseen_arrival_ok: bool = True    # out-of-view gap whose arrival at the next contact is unseen:
                                      # judged by the last sighting (fast flight = not a served hold)
    unseen_exit_max_y: float = 0.35   # ... which must be in the top 35 % of the frame (in-rally cases
                                      # left at <= 0.30 H over the top; net faults vanished at 0.40-0.54 H)
    side_call_mode: str = "fit"       # side the last flight's landing is judged for: 'fit' = its own
                                      # launch reading if clear, else the label it was fitted with;
                                      # 'label' = always the fitted-with label
    doubles: bool | None = None       # None = auto from players (4 per frame)
    activity_min_s: float = 1.0       # hit-less fallback: min moving stretch
    activity_gap_s: float = 2.2       # (high clears leave broadcast frames for ~1.8 s)
    seg_match_before: int = 3         # segment may start this many frames before its hit
    seg_match_after_s: float = 0.4    # ... or this long after it


def _opp(side: str) -> str:
    return {"near": "far", "far": "near"}.get(side, "unknown")


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ─── 2-D track kinematics ────────────────────────────────────────────────────

class _Motion:
    """Per-frame image kinematics of a Track2D: real (detected) samples, speed
    in image heights per second and a mask of frames where the shuttle rests."""

    def __init__(self, track: Track2D, height: float, cfg: RallyConfig):
        self.t = np.asarray(track.t, float)
        self.n = len(self.t)
        xy = np.asarray(track.xy, float).reshape(-1, 2)
        fin = np.isfinite(xy).all(1) & np.isfinite(self.t)
        vis = np.asarray(track.visible, bool) if track.visible is not None else fin
        interp = (np.asarray(track.interpolated, bool) if track.interpolated is not None
                  else np.zeros(self.n, bool))
        real = fin & vis & ~interp
        if not real.any():
            real = fin
        self.xy, self.fin, self.real = xy, fin, real
        self.dt = float(np.median(np.diff(self.t))) if self.n > 1 else 1.0 / 30.0
        self.h = max(float(height), 1.0)
        self.speed = self._speed(fin)                       # NaN where unknown
        self.rest = self._rest(cfg.static_s, cfg.static_radius_frac * self.h)

    def _speed(self, ok: np.ndarray, base: float = 0.05) -> np.ndarray:
        """Image speed over a >= `base` s baseline (per-frame differences are
        noise-dominated at high fps: 1.5 px at 120 fps is already 250 px/s)."""
        sp = np.full(self.n, np.nan)
        idx = np.flatnonzero(ok)
        if len(idx) < 2:
            return sp
        tt = self.t[idx]
        j = np.searchsorted(tt, tt - base + 1e-9, side="right") - 1
        good = j >= 0
        jj = np.where(good, j, 0)
        dt = tt - tt[jj]
        good &= (dt > 0) & (dt <= 4 * base)
        d = np.linalg.norm(self.xy[idx] - self.xy[idx[jj]], axis=1)
        sp[idx[good]] = d[good] / dt[good] / self.h
        return sp

    def _rest(self, static_s: float, radius: float, max_bad: int = 2) -> np.ndarray:
        """Greedy scan of real samples: a cluster staying within `radius` px of
        its first point for >= static_s seconds is marked as rest. Up to
        `max_bad` consecutive samples outside the radius (detector outliers)
        do not break a cluster."""
        rest = np.zeros(self.n, bool)
        idx = np.flatnonzero(self.real)
        if len(idx) < 3:
            return rest
        t = self.t[idx].tolist()
        x = self.xy[idx, 0].tolist()
        y = self.xy[idx, 1].tolist()
        r2 = radius * radius
        max_skip = max(0.25, static_s / 2)
        i, m = 0, len(idx)
        while i < m:
            last, bad, k = i, 0, i + 1
            xi, yi = x[i], y[i]
            while k < m and t[k] - t[last] <= max_skip:
                if (x[k] - xi) ** 2 + (y[k] - yi) ** 2 <= r2:
                    last, bad = k, 0
                else:
                    bad += 1
                    if bad > max_bad:
                        break
                k += 1
            if last > i and t[last] - t[i] >= static_s:
                rest[idx[i]: idx[last] + 1] = True
                i = last + 1
            else:
                i += 1
        return rest

    def frame_at(self, t: float) -> int:
        k = int(np.searchsorted(self.t, t))
        if k <= 0:
            return 0
        if k >= self.n:
            return self.n - 1
        return k if abs(self.t[k] - t) < abs(self.t[k - 1] - t) else k - 1

    def peak_speed(self, f0: int, window_s: float, q: float = 60.0) -> float | None:
        """Robust image speed (q-th percentile, immune to a few outliers) in
        (f0, f0 + window]; None with < 3 samples (not observed)."""
        if f0 < 0 or f0 >= self.n or window_s <= 0:
            return None
        f1 = int(np.searchsorted(self.t, self.t[f0] + window_s, side="right"))
        v = self.speed[f0 + 1: f1]
        v = v[np.isfinite(v)]
        if len(v) < 3:
            return None
        return float(np.percentile(v, q))

    def first_rest_after(self, f0: int, f1: int) -> int | None:
        """First rest frame in (f0, f1)."""
        f0, f1 = max(f0 + 1, 0), min(f1, self.n)
        if f1 <= f0:
            return None
        k = np.flatnonzero(self.rest[f0:f1])
        return int(f0 + k[0]) if len(k) else None

    def flight_end(self, f0: int, f1: int, max_skip_s: float = 0.5) -> int:
        """Last finite frame of the flight starting at f0 (stops at a rest, a
        gap > max_skip_s, or f1)."""
        f1 = min(f1, self.n)
        last, last_t = f0, self.t[f0] if 0 <= f0 < self.n else 0.0
        for f in range(f0 + 1, f1):
            if self.rest[f]:
                return f
            if self.fin[f]:
                if self.t[f] - last_t > max_skip_s:
                    break
                last, last_t = f, self.t[f]
            elif self.t[f] - last_t > max_skip_s:
                break
        return last

    def static_between(self, f0: int, f1: int, margin_s: float = 0.15) -> bool:
        """Rest strictly inside two hits (a hold cluster may touch a contact
        frame, so `margin_s` next to each hit is ignored)."""
        a = int(np.searchsorted(self.t, self.t[f0] + margin_s, side="right"))
        b = int(np.searchsorted(self.t, self.t[f1] - margin_s, side="left"))
        return b > a and bool(self.rest[a:b].any())


# ─── hits ↔ segments ─────────────────────────────────────────────────────────

def _clean_hits(hits: list[HitEvent], n: int) -> list[HitEvent]:
    """Sorted, one hit per frame (highest confidence), frames inside the track."""
    best: dict[int, HitEvent] = {}
    for h in hits or []:
        f = int(h.frame)
        if 0 <= f < n and (f not in best or h.confidence > best[f].confidence):
            best[f] = h
    return [best[f] for f in sorted(best)]


def _match_segments(hits: list[HitEvent], segments: list[ShotSegment3D],
                    t: np.ndarray, cfg: RallyConfig) -> list[ShotSegment3D | None]:
    """The flight starting at each hit: the segment whose start_frame lies in
    [hit - seg_match_before, min(next hit, hit + seg_match_after_s)]."""
    out: list[ShotSegment3D | None] = [None] * len(hits)
    if not hits or not segments:
        return out
    frames = np.array([h.frame for h in hits])
    best_d: list[float] = [np.inf] * len(hits)
    n = len(t)
    for seg in segments:
        s = int(seg.start_frame)
        k = int(np.searchsorted(frames, s + cfg.seg_match_before, side="right")) - 1
        if k < 0:
            continue
        f = int(frames[k])
        nxt = int(frames[k + 1]) if k + 1 < len(frames) else n
        t_s = t[min(max(s, 0), n - 1)]
        if s >= nxt or t_s - t[f] > cfg.seg_match_after_s:
            continue
        d = abs(s - f)
        if d < best_d[k]:
            out[k], best_d[k] = seg, d
    return out


def _fill_hitters(hits: list[HitEvent], segs: list[ShotSegment3D | None],
                  groups: list[list[int]], sure_x: float = 1.5) -> list[HitEvent]:
    """Copy hits; set the hitter from the 3-D flight start when it is unknown,
    or when the fit starts well inside the other half (|x| > sure_x: the fit
    is more reliable than a 2-D hitter call); then resolve remaining unknowns
    by alternation with the nearest known hit of the rally."""
    out = [replace(h) for h in hits]
    for i, h in enumerate(out):
        if segs[i] is None:
            continue
        x = float(segs[i].p0[0])
        if h.hitter not in SIDES or (abs(x) > sure_x and cm.side_of(x) != h.hitter):
            out[i] = replace(h, hitter=cm.side_of(x))
    for g in groups:
        known = [i for i in g if out[i].hitter in SIDES]
        if not known:
            continue
        for pos, i in enumerate(g):
            if out[i].hitter in SIDES:
                continue
            j = min(known, key=lambda k: abs(g.index(k) - pos))
            dist = abs(g.index(j) - pos)
            side = out[j].hitter if dist % 2 == 0 else _opp(out[j].hitter)
            out[i] = replace(out[i], hitter=side)
    return out


def _logaddexp(a: float, b: float) -> float:
    """log(e^a + e^b) on plain floats (the Swift port uses the same formula)."""
    if a < b:
        a, b = b, a
    if b == -math.inf:
        return a
    return a + math.log1p(math.exp(b - a))


def _own_side_reading(seg: ShotSegment3D | None, cfg: RallyConfig) -> str | None:
    """Hitter side read from the hit's own launch fit: a racket sends the shuttle
    towards the opponent (launch vx > 0 = a near hitter) from the hitter's half.
    'mixed' when launch direction and launch point name different sides, None
    without a fit or when both are too close to zero to read."""
    if seg is None:
        return None
    vx, x = float(seg.v0[0]), float(seg.p0[0])
    rv = "near" if vx > cfg.side_vx_dead else ("far" if vx < -cfg.side_vx_dead else None)
    rx = "near" if x < -cfg.side_x_dead else ("far" if x > cfg.side_x_dead else None)
    if rv is not None and rx is not None and rv != rx:
        return "mixed"
    return rv if rv is not None else rx


def _incoming_side_reading(prev: ShotSegment3D | None, h: HitEvent, t: np.ndarray,
                           cfg: RallyConfig) -> str | None:
    """Half of the court the incoming flight (the previous contact's fit, drag
    ODE propagated to this contact's time) is on when this contact happens =
    the side of its hitter. Only while that fit covers the contact (its last
    fitted frame + 0.25 s), not on a long extrapolation."""
    if prev is None or not float(prev.t0) < h.t:
        return None
    n = len(t)
    t_cov = float(t[min(max(int(prev.end_frame), 0), n - 1)]) + 0.25
    if h.t > t_cov:
        return None
    lo, hi = _x_bracket(prev, h.t)
    m = 0.05                                    # m: RK4 (h = 1/60 s) exceeds the exact bracket by <= 5.6 mm
    if lo - m > cfg.side_x_dead:
        return "far"
    if hi + m < -cfg.side_x_dead:
        return "near"
    if lo - m >= -cfg.side_x_dead and hi + m <= cfg.side_x_dead:
        return None
    x = _state_at(prev, h.t)[0]
    return "near" if x < -cfg.side_x_dead else ("far" if x > cfg.side_x_dead else None)


def _x_bracket(seg: ShotSegment3D, t: float) -> tuple[float, float]:
    """Exact bounds of the fitted flight's x at time t without integrating.
    Under the drag ODE vx never changes sign and d|vx|/dt = -|v||vx|/L, with
    |vx| <= |v| <= u(s), u the 1-D Riccati speed u' = g - u^2/L from |v0|
    (terminal vt = sqrt(gL)). So the x-travel D over tau = t - t0 obeys
    L ln(1 + |vx0| tau / L) >= D >= |vx0| int_0^tau exp(-int_0^s u / L) ds
    (closed forms below) for the exact flight. Decides ~92 % of incoming
    readings on sim, never disagreeing with RK4 (4445 checks)."""
    x0, vx0 = float(seg.p0[0]), float(seg.v0[0])
    tau = t - float(seg.t0)
    L = float(seg.drag_length or DRAG_LENGTH)
    ax = abs(vx0)
    if ax == 0.0 or tau <= 0.0:
        return x0, x0
    d_hi = L * math.log1p(ax * tau / L)
    sp = math.sqrt(float(seg.v0[0]) ** 2 + float(seg.v0[1]) ** 2 + float(seg.v0[2]) ** 2)
    vt = math.sqrt(G * L)
    k = vt / L
    if sp > vt:
        c = math.atanh(vt / sp)                 # u = vt coth(k s + c): exp(-int u/L) = sinh c / sinh(k s + c)
        d_lo = ax * math.sinh(c) / k * math.log(math.tanh((k * tau + c) / 2) / math.tanh(c / 2))
    else:
        d_lo = ax * (1.0 - math.exp(-k * tau)) / k   # u <= vt
    sgn = 1.0 if vx0 > 0 else -1.0
    a, b = x0 + sgn * d_lo, x0 + sgn * d_hi
    return (a, b) if a <= b else (b, a)


def _fit_weight(seg: ShotSegment3D | None, cfg: RallyConfig) -> float:
    """How much a fit's side reading counts: ramp of its confidence (which
    folds in observation count, reprojection rmse and landing/speed sigma)."""
    if seg is None:
        return 0.0
    q = float(seg.confidence or 0.0)
    return min(max((q - cfg.side_fit_q_lo) / (cfg.side_fit_q_hi - cfg.side_fit_q_lo), 0.0), 1.0)


def _side_llr(label: str, own: str | None, inc: str | None, cfg: RallyConfig,
              w_own: float = 1.0, w_inc: float = 1.0) -> float:
    """log P(evidence | near hitter) - log P(evidence | far hitter). The 3-D fits
    were made with the hits.py label as a prior, so their readings are scored
    as agreement / conflict with that label (a fit that contradicts the prior
    it was pulled towards is strong evidence against it); each fit's terms are
    weighted by its _fit_weight."""
    if label in SIDES:
        e = cfg.side_label_llr
        if own in SIDES:
            e += w_own * (cfg.side_own_agree if own == label else cfg.side_own_conflict)
        elif own == "mixed":
            e += w_own * cfg.side_own_mixed
        if inc in SIDES:
            e += w_inc * (cfg.side_in_agree if inc == label else cfg.side_in_conflict)
        return e if label == "near" else -e
    e = 0.0
    for r, w in ((own, w_own), (inc, w_inc)):
        if r == "near":
            e += w * cfg.side_unlabelled
        elif r == "far":
            e -= w * cfg.side_unlabelled
    return e


def _decode_sides(hits: list[HitEvent], segs: list[ShotSegment3D | None],
                  conts: list[ShotSegment3D | None], groups: list[list[int]], t: np.ndarray,
                  cfg: RallyConfig) -> tuple[list[HitEvent], list[float]]:
    """Joint hitter-side decode per rally group (2-state HMM, forward-backward):
    emissions = hits.py label + own launch fit + incoming fit (_side_llr),
    transitions = sides alternate unless the gap leaves room for a missed
    contact (P(same side) logistic in the gap, floored at side_same_eps for
    spurious contacts). Each hit gets the side of higher posterior; returns
    the relabelled hits and per hit P(that side) (0.5 = no evidence at all,
    the label is kept)."""
    out = [replace(h) for h in hits]
    prob = [0.5] * len(hits)
    for g in groups:
        m = len(g)
        e = []
        for i in g:
            h = hits[i]
            prev = None
            if i > 0:
                prev = conts[i - 1] if conts[i - 1] is not None else segs[i - 1]
            e.append(_side_llr(h.hitter, _own_side_reading(segs[i], cfg),
                               _incoming_side_reading(prev, h, t, cfg), cfg,
                               _fit_weight(segs[i], cfg), _fit_weight(prev, cfg)))
        ls, ld = [0.0] * m, [0.0] * m               # log P(same) / P(other) into contact k
        for k in range(1, m):
            z = (hits[g[k]].t - hits[g[k - 1]].t - cfg.side_same_mid_s) / cfg.side_same_width_s
            z = max(z, -30.0)                        # exp overflow guard (below eps anyway)
            ps = min(max(1.0 / (1.0 + math.exp(-z)), cfg.side_same_eps), 1.0 - cfg.side_same_eps)
            ls[k], ld[k] = math.log(ps), math.log(1.0 - ps)
        fn, ff = [0.0] * m, [0.0] * m
        fn[0], ff[0] = 0.5 * e[0], -0.5 * e[0]
        for k in range(1, m):
            fn[k] = 0.5 * e[k] + _logaddexp(fn[k - 1] + ls[k], ff[k - 1] + ld[k])
            ff[k] = -0.5 * e[k] + _logaddexp(fn[k - 1] + ld[k], ff[k - 1] + ls[k])
        bn, bf = [0.0] * m, [0.0] * m
        for k in range(m - 2, -1, -1):
            en, ef = 0.5 * e[k + 1] + bn[k + 1], -0.5 * e[k + 1] + bf[k + 1]
            bn[k] = _logaddexp(ls[k + 1] + en, ld[k + 1] + ef)
            bf[k] = _logaddexp(ld[k + 1] + en, ls[k + 1] + ef)
        for k, i in enumerate(g):
            d = (fn[k] + bn[k]) - (ff[k] + bf[k])        # log P(near) / P(far)
            if d != 0.0:
                out[i] = replace(out[i], hitter="near" if d > 0 else "far")
                prob[i] = 1.0 / (1.0 + math.exp(-abs(d)))
    return out, prob


def _alternation(hits: list[HitEvent]) -> float:
    """Fraction of consecutive known-hitter pairs that alternate (1 if < 2)."""
    s = [h.hitter for h in hits if h.hitter in SIDES]
    if len(s) < 2:
        return 1.0
    return float(np.mean([a != b for a, b in zip(s[:-1], s[1:])]))


# ─── grouping ────────────────────────────────────────────────────────────────

def _state_at(seg: ShotSegment3D, t: float, h: float = 1 / 60) -> list[float]:
    """[x, y, z, vx, vy, vz] of the fitted flight at absolute time t (drag ODE
    from p0/v0 in steps of h, the last one shortened to end exactly at t).
    h = 1/60 s: RK4 velocity error << 0.1 m/s here, far below min_contact_dv
    (decisions identical to h = 1/240 on 160 sim matches)."""
    s = [float(v) for v in (*seg.p0, *seg.v0)]
    L = float(seg.drag_length or DRAG_LENGTH)
    tt, stop = 0.0, t - float(seg.t0)
    while stop - tt > 1e-9:
        step = min(h, stop - tt)
        s = _rk4(s, step, L)
        tt += step
    return s


def _merge_same_flight(hits: list[HitEvent], segs: list[ShotSegment3D | None], t: np.ndarray,
                       cfg: RallyConfig) -> tuple[list[HitEvent], list[ShotSegment3D | None],
                                                  list[ShotSegment3D | None]]:
    """Drop contacts where the shuttle's 3-D velocity barely changes: the fit
    before the contact, propagated to the contact time, already moves like the
    fit after it (|dv| < min_contact_dv), so both fits describe one flight and
    nobody hit it. Only judged while the earlier fit actually covers the
    contact (its last fitted frame + 0.25 s), not on a long extrapolation.
    Returns the kept hits, their own segments and per hit the continuation
    segment that describes the end of its flight (None = its own segment)."""
    out_h: list[HitEvent] = []
    out_s: list[ShotSegment3D | None] = []
    out_c: list[ShotSegment3D | None] = []
    n = len(t)
    for h, s in zip(hits, segs):
        if out_h and s is not None:
            prev = out_c[-1] if out_c[-1] is not None else out_s[-1]
            if prev is not None:
                t_cov = float(t[min(max(int(prev.end_frame), 0), n - 1)]) + 0.25
                t0 = float(s.t0)
                vxn = float(s.v0[0])
                # exact shortcut: drag only shrinks vx, never flips its sign, so a
                # reversed x-direction with |vx| >= min_contact_dv already means |dv| >= it
                reversed_x = float(prev.v0[0]) * vxn < 0 and abs(vxn) >= cfg.min_contact_dv
                # exact bound: under drag |v| grows at most by g*dt, so a much faster new
                # flight (0.5 m/s numerical margin) is always a contact
                dt0 = t0 - float(prev.t0)
                sp_n = math.sqrt(float(s.v0[0]) ** 2 + float(s.v0[1]) ** 2 + float(s.v0[2]) ** 2)
                sp_p = math.sqrt(float(prev.v0[0]) ** 2 + float(prev.v0[1]) ** 2 + float(prev.v0[2]) ** 2)
                faster = sp_n >= sp_p + G * dt0 + cfg.min_contact_dv + 0.5
                if not reversed_x and not faster and float(prev.t0) < t0 <= t_cov:
                    st = _state_at(prev, t0)
                    dx = st[3] - float(s.v0[0])
                    dy = st[4] - float(s.v0[1])
                    dz = st[5] - float(s.v0[2])
                    if math.sqrt(dx * dx + dy * dy + dz * dz) < cfg.min_contact_dv:
                        out_c[-1] = s
                        continue
        out_h.append(h)
        out_s.append(s)
        out_c.append(None)
    return out_h, out_s, out_c


def _group_hits(hits: list[HitEvent], segs: list[ShotSegment3D | None], mo: _Motion,
                cfg: RallyConfig, conts: list[ShotSegment3D | None] | None = None,
                ) -> tuple[list[list[int]], list[str]]:
    """Split the hit sequence at dead-ball cues. Returns index groups and, per
    boundary, the cue that caused it ('gap' | 'landing' | 'static')."""
    if not hits:
        return [], []
    groups, cues, cur = [], [], [0]
    for i in range(1, len(hits)):
        a, b = hits[i - 1], hits[i]
        cue = None
        gap = b.t - a.t
        # a serve's pre-contact hold touching a's contact frame is not dead time after a
        fa = _past_hold(mo, a.frame, b.frame) if cfg.skip_serve_hold else a.frame
        if gap > cfg.max_hit_gap_s and not (gap < cfg.max_hit_gap_missed_s and
                                            _continuous_flight(mo, fa, b.frame, cfg)) \
                and not (gap < cfg.max_hit_gap_unseen_s and _unseen_flight(mo, a, b, cfg, fa)):
            cue = "gap"
        else:
            seg = conts[i - 1] if conts is not None and conts[i - 1] is not None else segs[i - 1]
            ft = None
            if seg is not None and (seg.confidence or 0.0) >= cfg.min_landing_conf:
                # only within the time span the fit covers (+0.25 s): beyond
                # it the ODE may be flying past a contact nobody detected
                t_cov = float(mo.t[min(max(int(seg.end_frame), 0), mo.n - 1)]) + 0.25
                ft = _floor_time(seg, min(b.t - cfg.landing_slack_s, t_cov))
            # the fitted/extrapolated floor contact must agree with the 2-D
            # track: a shuttle still in continuous fast flight afterwards
            # means a contact was missed (the ODE flew past it), not a landing
            if ft is not None and ft < b.t - cfg.landing_slack_s and \
                    not _continuous_flight(mo, mo.frame_at(ft), b.frame, cfg):
                cue = "landing"
            elif mo.static_between(a.frame, b.frame):
                cue = "static"
        if cue is None:
            cur.append(i)
        else:
            groups.append(cur)
            cues.append(cue)
            cur = [i]
    groups.append(cur)
    return groups, cues


# ─── court rules ─────────────────────────────────────────────────────────────

def service_court_margin(x: float, y: float, server: str, server_y: float | None,
                         doubles: bool) -> float:
    """Signed distance (m, + = inside, lines are in) of a serve landing from
    the service court diagonal to the server: between short service line and
    long service line (doubles) / baseline (singles), on the side of the centre
    line opposite to the server's y. `server_y` None or ~0 -> either half."""
    s = 1.0 if server == "near" else -1.0          # target half: +x for a near server
    lw = cm.LINE_WIDTH / 2
    xr = s * x                                     # distance from net into target half
    back = (cm.DOUBLES_LONG_SERVICE_X if doubles else cm.HALF_LENGTH) + lw
    half_w = (cm.DOUBLES_HALF_WIDTH if doubles else cm.SINGLES_HALF_WIDTH) + lw
    m = min(xr - (cm.SHORT_SERVICE_X - lw), back - xr, half_w - abs(y))
    if server_y is not None and abs(server_y) > 0.15:
        ty = -math.copysign(1.0, server_y)         # diagonal: opposite sign of y
        m = min(m, ty * y + lw)
    return float(m)


def detect_doubles(players: list[PlayerObservation] | None, min_frac: float = 0.3) -> bool | None:
    """True if >= 2 players per side are seen together in >= min_frac of the
    frames that have players at all; None without player observations."""
    if not players:
        return None
    per: dict[int, dict[str, int]] = {}
    for p in players:
        if p.score < 0.3 or p.side not in SIDES:
            continue
        d = per.setdefault(int(p.frame), {"near": 0, "far": 0})
        d[p.side] += 1
    if not per:
        return None
    four = sum(1 for d in per.values() if d["near"] >= 2 and d["far"] >= 2)
    return four >= min_frac * len(per)


# ─── outcome ─────────────────────────────────────────────────────────────────

@dataclass
class _Outcome:
    end_reason: str = "unknown"
    winner: str = "unknown"
    landing_xy: tuple[float, float] | None = None
    margin: float | None = None
    confidence: float = 0.0
    end_t: float | None = None
    source: str = "none"


def _rk4(s: list[float], h: float, L: float) -> list[float]:
    """One RK4 step of the physics.py drag ODE on a scalar state
    [x, y, z, vx, vy, vz] (pure floats: ~10x faster than numpy on 3-vectors)."""
    def f(q: list[float]) -> list[float]:
        k = math.sqrt(q[3] * q[3] + q[4] * q[4] + q[5] * q[5]) / L
        return [q[3], q[4], q[5], -k * q[3], -k * q[4], -G - k * q[5]]
    k1 = f(s)
    k2 = f([a + 0.5 * h * b for a, b in zip(s, k1)])
    k3 = f([a + 0.5 * h * b for a, b in zip(s, k2)])
    k4 = f([a + h * b for a, b in zip(s, k3)])
    return [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4)]


def _floor_time(seg: ShotSegment3D, t_until: float, h: float = 1 / 120) -> float | None:
    """Absolute time the fitted flight reaches the floor, if before t_until
    (the fit's own landing_t, else the drag ODE integrated from p0/v0; a
    flight chained to a next "hit" has no reported landing)."""
    if seg.landing_t is not None:
        return float(seg.landing_t)
    s = [float(v) for v in (*seg.p0, *seg.v0)]
    if not all(math.isfinite(v) for v in s) or s[2] <= 0.0:
        return None
    L = float(seg.drag_length or DRAG_LENGTH)
    t, stop = 0.0, t_until - float(seg.t0)
    while t < stop:
        n = _rk4(s, h, L)
        if n[2] <= 0.0:
            return float(seg.t0) + t + h * s[2] / (s[2] - n[2])
        s, t = n, t + h
    return None


def _flight_end(seg: ShotSegment3D, t_last: float, h: float = 1 / 240,
                t_max: float = 6.0) -> tuple[tuple[float, float] | None, float | None,
                                             float, float | None, float | None]:
    """(landing_xy, landing_t, extrapolated seconds, net_crossing_height,
    net_crossing_y) of a final flight. Integrates the fitted drag ODE from t0;
    if the fit has no floor contact the shuttle is followed past `t_last` down
    to z = 0 (the rally is over, so it must come down)."""
    L = float(seg.drag_length or DRAG_LENGTH)
    land_xy, land_t, net_h = seg.landing_xy, seg.landing_t, seg.net_crossing_height
    net_y = getattr(seg, "net_crossing_y", None)
    if land_xy is not None and (net_h is None or net_y is not None):
        return land_xy, land_t, 0.0, net_h, net_y
    s = [float(v) for v in (*seg.p0, *seg.v0)]
    t, t_stop = 0.0, (land_t - seg.t0) if land_t is not None else t_max
    extra = 0.0
    while t < t_stop:
        n = _rk4(s, h, L)
        if s[0] * n[0] < 0 and net_y is None:
            a = s[0] / (s[0] - n[0])
            net_y = s[1] + a * (n[1] - s[1])
            if net_h is None:
                net_h = s[2] + a * (n[2] - s[2])
            if land_xy is not None:
                break
        if n[2] <= 0.0 and s[2] > 0.0:
            if land_xy is None:
                a = s[2] / (s[2] - n[2])
                land_xy = (s[0] + a * (n[0] - s[0]), s[1] + a * (n[1] - s[1]))
                land_t = seg.t0 + t + a * h
                extra = max(0.0, land_t - t_last)
            break
        s, t = n, t + h
    return land_xy, land_t, extra, net_h, net_y


def _landing_sigma(seg: ShotSegment3D, extra_s: float, cfg: RallyConfig) -> float:
    s = getattr(seg, "landing_sigma_m", None)
    if s is None or not math.isfinite(s):
        try:
            from app.services.tracking import reconstruct3d as r3
            s = r3.landing_sigma(seg) if hasattr(r3, "landing_sigma") else None
        except Exception:
            s = None
    s = cfg.default_landing_sigma_m if s is None or not math.isfinite(s) else float(s)
    return max(s, 0.03) + 0.6 * extra_s


def _landing_plausible(x: float, y: float, cfg: RallyConfig) -> bool:
    """Floor point inside the hall: within landing_outside_max_m of the doubles court."""
    return (math.isfinite(x) and math.isfinite(y) and abs(x) <= cm.HALF_LENGTH + cfg.landing_outside_max_m
            and abs(y) <= cm.DOUBLES_HALF_WIDTH + cfg.landing_outside_max_m)


def _outcome_3d(hit: HitEvent, seg: ShotSegment3D, t_last: float, is_serve: bool,
                doubles: bool | None, server_y: float | None, cfg: RallyConfig,
                launch: ShotSegment3D | None = None, side_p: float | None = None) -> _Outcome:
    """`seg` = the fit describing the end of the final flight; `launch` = the
    fit starting at the contact when that is a different one (a spurious
    contact was merged: the continuation fit's own net/own-half label was
    decided for the wrong hitter, so the net test uses the launch fit).
    `side_p` = P(hitter side) from the joint decode; it replaces the
    side_conflict_q heuristic (the decode already weighs the launch fit)."""
    hitter = hit.hitter if hit.hitter in SIDES else cm.side_of(float((launch or seg).p0[0]))
    land_xy, land_t, extra, net_h, net_y = _flight_end(seg, t_last)
    if land_xy is not None and not _landing_plausible(float(land_xy[0]), float(land_xy[1]), cfg):
        # the fitted flight comes down outside the hall: its geometry (landing AND net crossing) is
        # not this rally's flight. No call (the next-serve rule may still name the winner); the rally
        # keeps its span (end_t), exactly what ShuttlAnalysis LandingGuard does to such a call
        return _Outcome(end_t=land_t, source="implausible_landing")
    seg_net = getattr(seg, "end_reason", None) == "net"
    if launch is not None:
        seg_net = False
        if launch.net_crossing_height is not None:
            net_h = float(launch.net_crossing_height)
            net_y = getattr(launch, "net_crossing_y", None)
    if net_y is not None and abs(net_y) > cm.DOUBLES_HALF_WIDTH + 0.05:
        net_h = None                              # passed outside the posts: judged by landing
    fit_q = 0.6 + 0.4 * min(max(seg.confidence or 0.0, 0.0), 1.0)
    # every call names the winner through the hitter label: a launch fit
    # flying towards the hitter's own baseline (> 2 m/s: not a net-cord
    # dribble) contradicts that label, so the call is much less certain
    vx0 = float((launch or seg).v0[0])
    if side_p is not None:
        fit_q *= side_p
    elif hit.hitter in SIDES and abs(vx0) > 2.0 and (vx0 > 0) != (hitter == "near"):
        fit_q *= cfg.side_conflict_q
    out = _Outcome(end_t=land_t, source="3d")
    net_line = cm.NET_HEIGHT_CENTRE - cfg.net_margin_m
    fit_land = seg.landing_xy                     # observed descent to the floor (fit's own)
    went_over = fit_land is not None and fit_land[0] != 0 and cm.side_of(fit_land[0]) != hitter
    if seg_net or (
            net_h is not None and net_h < net_line
            and not (went_over and net_h > net_line - cfg.net_cord_tol_m)):
        out.end_reason, out.winner = "net", _opp(hitter)
        out.landing_xy = land_xy
        out.confidence = fit_q * cfg.net_call_q * (
            _phi((net_line - net_h) / 0.08) if net_h is not None else 0.8)
        return out
    if land_xy is None:
        return out
    out.landing_xy = land_xy
    if land_xy[0] != 0 and cm.side_of(land_xy[0]) == hitter:
        # came down on the hitter's own side: hit the net (or a frame shot backwards)
        out.end_reason, out.winner = "net", _opp(hitter)
        out.confidence = fit_q * cfg.net_call_q * _phi(abs(land_xy[0]) / 0.3)
        return out
    sig = _landing_sigma(seg, extra, cfg)
    x, y = land_xy
    if is_serve:
        dbl = bool(doubles)
        margin = service_court_margin(x, y, hitter, server_y, dbl)
        ambiguous = doubles is None and cm.DOUBLES_LONG_SERVICE_X < abs(x) < cm.HALF_LENGTH
    else:
        margin = cm.in_out_margin(x, y, doubles=bool(doubles))
        ambiguous = doubles is None and cm.SINGLES_HALF_WIDTH < abs(y) < cm.DOUBLES_HALF_WIDTH
    out.margin = float(margin)
    out.end_reason = "in" if margin >= 0 else "out"
    out.winner = hitter if margin >= 0 else _opp(hitter)
    out.confidence = fit_q * _phi(abs(margin) / sig) * (0.5 if ambiguous else 1.0)
    return out


def _outcome_2d(hit: HitEvent, f_rest: int | None, mo: _Motion,
                camera: CameraModel | None, doubles: bool | None, cfg: RallyConfig) -> _Outcome:
    """Camera but no 3-D flight: back-project the first 2-D rest after the last
    hit (shuttle on the floor). Low confidence: it may be a catch."""
    out = _Outcome(source="2d")
    if f_rest is not None:
        out.end_t = float(mo.t[f_rest])
    if camera is None or f_rest is None or hit.hitter not in SIDES:
        return out
    uv = mo.xy[f_rest]
    try:
        p = camera.pixel_to_floor(uv[None, :])[0]
    except Exception:
        return out
    if not np.all(np.isfinite(p[:2])):
        return out
    x, y = float(p[0]), float(p[1])
    if not _landing_plausible(x, y, cfg):
        return out                                   # not on the floor near the court
    out.landing_xy = (x, y)
    if cm.side_of(x) == hit.hitter:
        out.end_reason, out.winner, out.confidence = "net", _opp(hit.hitter), 0.3
        return out
    margin = cm.in_out_margin(x, y, doubles=bool(doubles))
    out.margin = float(margin)
    out.end_reason = "in" if margin >= 0 else "out"
    out.winner = hit.hitter if margin >= 0 else _opp(hit.hitter)
    out.confidence = 0.4 * _phi(abs(margin) / 0.3)
    return out


# ─── BWF score state ─────────────────────────────────────────────────────────

PLAYERS = {"A": ("A1", "A2"), "B": ("B1", "B2")}


class _Score:
    """BWF rally-point state for one recording (a match, then further matches if
    rallies continue after match over). Players / pairs are named by the end they
    start game 1 at: 'A' = near end, 'B' = far end, and keep those names for the
    whole recording (DELIVERY_R2 D2). `a_end` = where A stands now. Format:
    first to `points`, two clear, `cap` wins; deciding-game ends change when a
    side first reaches `interval` (0 = never). Doubles: `pos` = court of each of
    A1 A2 B1 B2 (Law 11), `starters[pair][g-1]` = the pair's player in the right
    court at the start of game g (Law 11.6; default A1 / B1)."""

    def __init__(self, first_server: str = "near", games_to_win: int = 2, points: int = 21,
                 cap: int = 30, interval: int = 11, doubles: bool = False,
                 starters: dict | None = None):
        self.games_to_win, self.n_points, self.cap, self.interval = games_to_win, points, cap, interval
        self.doubles = bool(doubles)
        st = {"A": ["A1"], "B": ["B1"]}
        for k, v in (starters or {}).items():
            v = [v] if isinstance(v, str) else list(v)
            if k in st and v:                    # an empty list keeps the default (Swift: same)
                st[k] = v
        self.starters = st
        self.a_end = "near"
        self._start_match("A" if first_server != "far" else "B")

    def _start_match(self, server: str) -> None:
        self.games = {"A": 0, "B": 0}
        self.game = 0
        self.match_over = False
        self._start_game(server)

    def _start_game(self, server: str) -> None:
        self.game += 1
        self.points = {"A": 0, "B": 0}
        self.mid_switched = False
        self.server = server
        self.pos: dict[str, str] = {}
        if self.doubles:
            for pair, (p1, p2) in PLAYERS.items():
                lst = self.starters[pair]
                s = lst[min(self.game - 1, len(lst) - 1)]
                self.pos[s], self.pos[p2 if s == p1 else p1] = "right", "left"

    def begin(self) -> bool:
        """Start the next match if the last one is over (before reading the
        server / court of the next rally: Law 10.1.1, a new match starts at 0-0
        in the right court, D1). The match winner serves; nobody changes ends
        after the final game (Law 8.1), so A / B keep their ends and names."""
        if not self.match_over:
            return False
        self._start_match(self.server)
        return True

    def copy(self) -> "_Score":
        s = _Score.__new__(_Score)
        s.__dict__.update(self.__dict__)
        s.points, s.games, s.pos = dict(self.points), dict(self.games), dict(self.pos)
        return s

    def player_at(self, side: str) -> str:
        return "A" if side == self.a_end else "B"

    def side_of(self, player: str) -> str:
        return self.a_end if player == "A" else _opp(self.a_end)

    def server_side(self) -> str:
        return self.side_of(self.server)

    def service_court(self) -> str:
        return "right" if self.points[self.server] % 2 == 0 else "left"

    def player_in(self, pair: str, court: str) -> str | None:
        return next((p for p in PLAYERS[pair] if self.pos.get(p) == court), None)

    def server_player(self) -> str | None:
        return self.player_in(self.server, self.service_court()) if self.doubles else None

    def receiver_player(self) -> str | None:
        """Law 11.1.4: the receiver stands diagonally opposite = the same-named court."""
        o = "B" if self.server == "A" else "A"
        return self.player_in(o, self.service_court()) if self.doubles else None

    def by_side(self, d: dict[str, int]) -> dict[str, int]:
        return {s: d[self.player_at(s)] for s in SIDES}

    def game_won(self, pw: int, pl: int) -> bool:
        return pw >= self.cap or (pw >= self.n_points and pw - pl >= 2)

    def point(self, side: str) -> dict:
        """Award a rally to the physical side `side`; returns the events. A
        caller that did not begin() the next match gets it started here."""
        ev = {"game_over": False, "game_winner": None, "match_over": False,
              "ends_changed": False, "new_match": self.begin()}
        w = self.player_at(side)
        lo = "B" if w == "A" else "A"
        self.points[w] += 1
        if w == self.server:
            if self.doubles:                     # Law 11.3.1 / 11.1.5: the serving pair swaps courts
                for p in PLAYERS[w]:
                    self.pos[p] = "left" if self.pos[p] == "right" else "right"
        else:
            self.server = w                      # Law 10.3.2 / 11.3.2: the receiver serves next
        pw, pl = self.points[w], self.points[lo]
        ev["score"] = self.by_side(self.points)  # sides as played in this rally
        ev["score_by_player"] = dict(self.points)
        if self.game_won(pw, pl):
            self.games[w] += 1
            ev["game_over"], ev["game_winner"] = True, side
            ev["games"] = self.by_side(self.games)
            ev["games_by_player"] = dict(self.games)
            if self.games[w] >= self.games_to_win:
                ev["match_over"] = self.match_over = True
            else:                                # Law 7.6 / 8.1: game winner serves first; change ends
                self.a_end = _opp(self.a_end)
                self._start_game(w)
                ev["ends_changed"] = True
        else:
            ev["games"] = self.by_side(self.games)
            ev["games_by_player"] = dict(self.games)
            decider = self.game == 2 * self.games_to_win - 1
            if decider and self.interval > 0 and not self.mid_switched \
                    and pw == self.interval and pl < self.interval:
                self.a_end = _opp(self.a_end)    # Law 8.1.3
                self.mid_switched = True
                ev["ends_changed"] = True
        return ev


def _observed_server(r: RallyV2, min_conf: float = 0.0) -> str:
    """First hitter of a rally (its receiver if segment_rallies flagged the
    serve itself as missed)."""
    if not r.shots:
        return "unknown"
    h = min(r.shots, key=lambda s: s.hit.frame).hit
    if h.hitter not in SIDES or (h.confidence or 0.0) < min_conf:
        return "unknown"
    return _opp(h.hitter) if getattr(r, "serve_missed", False) else h.hitter


def _implied_winner(st: _Score, next_server: str) -> str:
    """Winner of the current rally implied by who serves next ('unknown' when
    both outcomes give the same next server, e.g. game point + end change)."""
    if next_server not in SIDES:
        return "unknown"
    return _implied_winner_obs(st, {"server": next_server})


_OBS_KEYS = ("server", "service_court", "server_player", "receiver_player")


def _implied_winner_obs(st: _Score, obs: dict) -> str:
    """Winner of the current rally implied by what is seen at the start of the
    next one: any of server (side), service_court, server_player /
    receiver_player (doubles); None / missing = not seen. 'unknown' when both
    outcomes, or neither, fit (DELIVERY_R2 infer semantics; a new match after
    match over starts 0-0 in the right court with the match winner serving)."""
    seen = {k: v for k, v in (obs or {}).items() if k in _OBS_KEYS and v is not None}
    if not seen:
        return "unknown"
    cands = []
    for c in SIDES:
        s = st.copy()
        s.point(c)
        s.begin()
        view = {"server": s.server_side(), "service_court": s.service_court(),
                "server_player": s.server_player(), "receiver_player": s.receiver_player()}
        if all(view[k] == v for k, v in seen.items()):
            cands.append(c)
    return cands[0] if len(cands) == 1 else "unknown"


# ─── candidates, feeds, next-serve inference ─────────────────────────────────

@dataclass
class _Cand:
    idx: list[int]                  # indices into the cleaned hit list
    hits: list[HitEvent]
    segs: list[ShotSegment3D | None]
    start_frame: int
    end_frame: int
    end_t: float
    outcome: _Outcome
    seg_conf: float
    cue_before: str | None
    serve_missed: bool = False      # shuttle already flying before the first hit
    first_side_p: float = 1.0       # P(first hitter side) from the joint decode (names the server)
    held_before: bool = True        # shuttle seen resting (held) right before the first hit

    @property
    def server(self) -> str:
        if not self.hits or self.hits[0].hitter not in SIDES:
            return "unknown"
        return _opp(self.hits[0].hitter) if self.serve_missed else self.hits[0].hitter


def _median_speed(mo: _Motion, t0: float, t1: float) -> float | None:
    """Median image speed (image heights / s) of samples in (t0, t1]; robust
    to detector outliers. None with < 3 samples."""
    a = int(np.searchsorted(mo.t, t0, side="right"))
    b = int(np.searchsorted(mo.t, t1, side="right"))
    v = mo.speed[a:b]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= 3 else None


def _is_flight(h: HitEvent, seg: ShotSegment3D | None, mo: _Motion, cfg: RallyConfig,
               t_next: float | None = None) -> bool:
    """A real hit launches a flight: a 3-D segment, or image motion right after
    the contact that is fast in absolute terms or clearly faster than just
    before it (image speed scales ~5x between near and far court, so a far
    short serve can be slower than a near hand carry; the jump from the
    still hold / the slow carry is scale-free). Unobserved counts as flight."""
    if seg is not None and np.linalg.norm(seg.v0) > 4.0:
        return True
    t_end = h.t + cfg.launch_window_s
    if t_next is not None:
        t_end = min(t_end, t_next - 1e-3)
    after = _median_speed(mo, h.t, t_end)
    if after is None:
        # unobserved right after: occlusion by the racket/body is normal, but a
        # shuttle seen before the contact and then gone for 0.8 s was picked up
        seen_before = bool(mo.fin[mo.frame_at(h.t - 0.5):h.frame].any())
        seen_after = _median_speed(mo, h.t, h.t + 0.8 if t_next is None else
                                   min(h.t + 0.8, t_next - 1e-3)) is not None
        return seen_after or not seen_before
    if after >= cfg.min_flight_speed_hps:
        return True
    before = _median_speed(mo, h.t - 0.4, h.t - 0.05)
    return after >= cfg.min_launch_speed_hps and (before is None or after >= 2.5 * before)


def _continuous_flight(mo: _Motion, f0: int, f1: int, cfg: RallyConfig) -> bool:
    """Between two hits the shuttle never rested and every observed stretch
    moved fast, with no unobserved stretch longer than max_unseen_s: a missed
    hit inside a long exchange (e.g. a lift that left the frame), not dead time
    (which shows the shuttle lying still, carried slowly, or long unseen)."""
    if f1 <= f0 + 1:
        return True
    if mo.rest[f0:f1 + 1].any():
        return False
    ok = mo.fin[f0:f1 + 1]
    tt = np.r_[mo.t[f0], mo.t[f0:f1 + 1][ok], mo.t[f1]]
    if np.max(np.diff(tt)) > cfg.max_unseen_s:
        return False
    v = mo.speed[f0:f1 + 1]
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return False
    # the apex of a high lift seen from behind is slow in the image, so only
    # require the flight to be fast most of the time (a carry is slow throughout)
    return float(np.mean(v < cfg.min_flight_speed_hps)) < 0.5


def _past_hold(mo: _Motion, f0: int, f1: int, max_s: float = 0.1) -> int:
    """First frame after the rest run that touches contact frame f0: a serve is
    played out of a hold and that hold cluster reaches the contact frame itself
    (the shuttle has barely moved yet), which is not dead time AFTER the
    contact. Only a run ending within max_s of the contact is skipped (a launched
    shuttle leaves the rest radius within a few frames); f0 if it does not rest."""
    f = f0
    while f < f1 and mo.rest[f] and mo.t[f] - mo.t[f0] <= max_s:
        f += 1
    return f


def _unseen_flight(mo: _Motion, a: HitEvent, b: HitEvent, cfg: RallyConfig,
                   f_start: int | None = None) -> bool:
    """A long hit gap that is flight out of view, not dead time: the shuttle
    never rests (a dead shuttle lies on the floor or is held before the serve),
    what is observed moves fast most of the time, at most max_unseen_flight_s
    is unobserved at a stretch, and it is seen arriving at b in fast flight
    (a serve starts from a still hold, never from a flying shuttle).
    `f_start` replaces a.frame as the first frame looked at (see _past_hold)."""
    f0, f1 = (a.frame if f_start is None else f_start), b.frame
    if f1 <= f0 + 1 or mo.rest[f0:f1 + 1].any():
        return False
    ok = mo.fin[f0:f1 + 1]
    tt = np.r_[mo.t[f0], mo.t[f0:f1 + 1][ok], mo.t[f1]]
    # only the out-of-view case: in-view flight is judged by _continuous_flight,
    # and a gap this long seen in view would need several missed contacts in view
    unseen = float(np.max(np.diff(tt)))
    if unseen > cfg.max_unseen_flight_s or unseen <= cfg.max_unseen_s:
        return False
    v = mo.speed[f0:f1 + 1]
    v = v[np.isfinite(v)]
    if len(v) < 3 or float(np.mean(v < cfg.min_flight_speed_hps)) >= 0.5:
        return False
    arr = _median_speed(mo, b.t - cfg.launch_window_s, b.t - 1e-3)
    if arr is None and cfg.unseen_arrival_ok:
        # the arrival itself is unobserved (it came down out of view onto b): a
        # shuttle last seen in fast flight that never rested was not served from
        # a hold (13/100 in-rally long gaps, 0/156 between rallies on dev)
        idx = np.flatnonzero(mo.fin[f0:f1]) + f0
        if len(idx) < 3:
            return False
        # ... and it left the view high up (flying above the frame), not low in
        # the image where a dead shuttle falls at the net / onto the floor
        if float(mo.xy[idx[-1], 1]) > cfg.unseen_exit_max_y * mo.h:
            return False
        sp = mo.speed[idx[-3:]]
        sp = sp[np.isfinite(sp)]
        return len(sp) >= 2 and float(np.median(sp)) >= cfg.min_flight_speed_hps
    return arr is not None and arr >= cfg.min_flight_speed_hps


def _launch_frame(mo: _Motion, f0: int, f1: int, cfg: RallyConfig) -> int:
    """Frame the flight after contact f0 is looked at from: the last frame of a
    rest run touching the contact (a serve's own pre-contact hold, _past_hold),
    else f0. Such a hold is not the NEXT rest: read as one, a real serve had
    0 s of motion and its rally ended at the contact (tests/test_rally
    generator: 3 real serves at 60/120 fps, 0.76-1.94 s flights, dropped as
    one-hit dead-time contacts). skip_serve_hold=False: always f0."""
    if not cfg.skip_serve_hold:
        return f0
    return max(f0, _past_hold(mo, f0, f1) - 1)


def _moving_after(mo: _Motion, h: HitEvent, cfg: RallyConfig | None = None,
                  horizon_s: float = 4.0) -> float:
    """Observed moving time (s; real samples not at rest, x median frame step)
    after contact h (past its own hold, _launch_frame) until the shuttle's
    next rest or horizon_s."""
    f_hi = min(mo.n, mo.frame_at(h.t + horizon_s) + 1)
    f = _launch_frame(mo, h.frame, f_hi, cfg or RallyConfig())
    fr = mo.first_rest_after(f, f_hi)
    end = fr if fr is not None else f_hi
    if end <= f + 1:
        return 0.0
    k = int(np.sum(mo.fin[f + 1:end] & ~mo.rest[f + 1:end]))
    return k * mo.dt


def _departure(mo: _Motion, h: HitEvent, prev: ShotSegment3D, camera: CameraModel,
               lag: float) -> float | None:
    """Image distance (x image height) between the observed track and the
    incoming fit's projection at the moving sample closest to h.t + lag (before
    the next rest, within 0.06 s of it); None when not observed."""
    f_hi = min(mo.n, mo.frame_at(h.t + lag + 0.1) + 1)
    fr = mo.first_rest_after(h.frame, f_hi)
    end = fr if fr is not None else f_hi
    best, bd = -1, 0.0
    for k in range(h.frame + 1, end):
        if mo.fin[k]:
            d = abs(float(mo.t[k]) - (h.t + lag))
            if best < 0 or d < bd:
                best, bd = k, d
    if best < 0 or bd > 0.06:
        return None
    st = _state_at(prev, float(mo.t[best]))
    uv = camera.project(np.array([[st[0], st[1], st[2]]]))[0]
    if not (math.isfinite(float(uv[0])) and math.isfinite(float(uv[1]))):
        return None
    return math.hypot(float(uv[0]) - float(mo.xy[best, 0]), float(uv[1]) - float(mo.xy[best, 1])) / mo.h


def _launches(h: HitEvent, seg: ShotSegment3D | None, prev: ShotSegment3D | None,
              cfg: RallyConfig, mo: _Motion | None = None, camera: CameraModel | None = None) -> bool:
    """Contact h has 3-D evidence of a racket launch: its own fit is determined
    (_fit_weight >= trail_launch_w), the incoming fit still has the shuttle in
    the air at the contact, and the new flight REVERSES the incoming x-velocity
    (every rally shot sends the shuttle back over the net) with a racket-sized
    |dv| (>= min_contact_dv), and (with a camera) the 2-D track trail_depart_lag_s
    after the contact has left the incoming fit's projection by >= trail_depart_min
    image heights. Spurious trailing kinks have no own fit, an undetermined one
    (a net bounce: fit weight 0-0.46), no x reversal (a kink on the descending
    flight), an incoming fit already below the floor, or (4/58 sim, all 3-D
    fits placed 2.8-5.3 m from the incoming shuttle) a track that stays on
    the incoming flight (0.001-0.004 H; 0.033 H once). Real short final
    flights the confidence rule dropped (tests/test_rally generator, 46 at
    30/60/120 fps) pass 43 times (departure >= 0.018 H; the rest have no own fit)."""
    if seg is None or prev is None or _fit_weight(seg, cfg) < cfg.trail_launch_w:
        return False
    st = _state_at(prev, float(h.t))
    if not st[2] > 0.0:
        return False                              # incoming flight already on the floor: no reading
    vx = float(seg.v0[0])
    if st[3] * vx >= 0.0:
        return False
    dx, dy, dz = st[3] - vx, st[4] - float(seg.v0[1]), st[5] - float(seg.v0[2])
    if math.sqrt(dx * dx + dy * dy + dz * dz) < cfg.min_contact_dv:
        return False
    if mo is not None and camera is not None:
        # a mis-placed own fit can "reverse" in 3-D while the 2-D track goes on
        # exactly where the incoming flight predicts (one flight, no contact)
        dep = _departure(mo, h, prev, camera, cfg.trail_depart_lag_s)
        if dep is not None and dep < cfg.trail_depart_min:
            return False
    return True


def _hold_before(mo: _Motion, f: int, t: float, cfg: RallyConfig) -> float:
    """+1: the shuttle rested (held still) right before this hit, as before
    every serve; -1: visible and moving throughout; 0: not observed."""
    f0 = mo.frame_at(t - cfg.hold_window_s)
    if f <= f0:
        return 0.0
    if mo.rest[f0:f].any():
        return 1.0
    return -1.0 if mo.fin[f0:f].mean() >= 0.5 else 0.0


def _nonrally_log_odds(c: _Cand, prev: _Cand | None, nxt: _Cand | None, mo: _Motion,
                       doubles: bool | None, have_3d: bool, cfg: RallyConfig) -> float:
    """max(feed-vs-rally, spurious-vs-rally) log-odds of a one-hit group."""
    h, seg = c.hits[0], c.segs[0]
    hold = _hold_before(mo, h.frame, h.t, cfg)
    conf = h.confidence or 0.0
    prev_w = prev.outcome.winner if prev is not None else "unknown"
    rule = 0 if h.hitter not in SIDES or prev_w not in SIDES else (1 if h.hitter != prev_w else -1)
    # the serving rule is only as reliable as the previous call
    w_rule = 3.0 * float(np.clip((prev.outcome.confidence - 0.45) / 0.35, 0.0, 1.0)) \
        if prev is not None else 0.0

    feed = -1.0 + w_rule * rule
    if nxt is not None and h.hitter in SIDES and nxt.hits[0].hitter in SIDES \
            and nxt.hits[0].t - h.t < 15.0:
        feed += 0.8 if nxt.hits[0].hitter != h.hitter else -1.5
    if prev is not None:
        dtp = h.t - prev.end_t
        feed += 0.8 if dtp < 4.0 else (-0.8 if dtp > 8.0 else 0.0)
    feed += {1.0: -1.5, -1.0: 1.0}.get(hold, 0.0)
    if seg is not None:
        sp = float(np.linalg.norm(seg.v0)) * 3.6
        feed += 0.3 if sp < 45 else (-1.0 if sp > 90 else 0.0)
        caught = seg.landing_xy is None and c.outcome.end_reason != "net"
        feed += 1.0 if caught else -0.3
        xr, y = abs(float(seg.p0[0])), float(seg.p0[1])
        back = cm.DOUBLES_LONG_SERVICE_X if doubles else cm.HALF_LENGTH
        if xr < cm.SHORT_SERVICE_X - 0.4 or xr > back + 0.6 or abs(y) > cm.DOUBLES_HALF_WIDTH + 0.6:
            feed += 0.7

    spur = -1.5 + 0.6 * rule
    spur += 1.2 if conf < 0.4 else (-1.0 if conf > 0.7 else 0.0)
    spur += {1.0: -1.5, -1.0: 1.0}.get(hold, 0.0)
    if have_3d:
        spur += 1.2 if seg is None else -1.5
    pk = mo.peak_speed(h.frame, cfg.flight_window_s)
    if pk is None:                  # no flight observed after the contact at all
        spur += 1.0
    elif pk < 2 * cfg.min_flight_speed_hps:
        spur += 0.5
    return max(feed, spur)


def _serve_launch_ok(c: _Cand) -> bool:
    """The first contact of a rally can be its serve: without a 3-D fit, or
    launched on the server's half between the short service line and just
    behind the baseline (the server stands in a service court; a launch
    outside it means the first contact is not the serve, e.g. a missed serve
    and a return, so it says nothing about who serves)."""
    s = c.segs[0] if c.segs else None
    if c.serve_missed or s is None:
        return True
    x = float(s.p0[0])
    # 0.3 m in front of the short service line (reach over it), 0.8 m behind the baseline
    return (cm.side_of(x) == c.server
            and cm.SHORT_SERVICE_X - 0.3 < abs(x) < cm.HALF_LENGTH + 0.8)


def _infer_from_next_serve(cands: list[_Cand], cfg: RallyConfig,
                           first_server: str = "near") -> None:
    """Resolve unknown / low-confidence winners from who serves next (in place)."""
    st = _Score(first_server)
    for i, c in enumerate(cands):
        w = c.outcome.winner
        nxt = cands[i + 1] if i + 1 < len(cands) else None
        ns = "unknown"
        seen = False
        if nxt is not None:
            h0 = nxt.hits[0] if nxt.hits else None
            if h0 is not None and nxt.server in SIDES and (h0.confidence or 0.0) >= 0.3 \
                    and nxt.seg_conf >= 0.5 and _serve_launch_ok(nxt) \
                    and (not cfg.next_serve_needs_hold or nxt.serve_missed or nxt.held_before):
                ns = nxt.server
                seen = bool(nxt.held_before and not nxt.serve_missed)    # the serve itself was observed
        implied = _implied_winner(st, ns)
        q0 = c.outcome.confidence
        if implied in SIDES and (w not in SIDES or
                                 (w != implied and c.outcome.confidence < cfg.override_conf)):
            if w in SIDES and w != implied:
                # measured call contradicted: keep the geometry, drop the verdict
                c.outcome.end_reason = "unknown"
            c.outcome.winner = implied
            # the implied winner is only right if the next server's side is
            c.outcome.confidence = 0.8 * c.seg_conf * (nxt.first_side_p if cfg.next_serve_side_q else 1.0)
            c.outcome.source = "next_serve"
            if not seen:
                # the verdict rests on a serve nobody saw (inferred from a missed serve): not certain.
                # A SEEN serve is independent evidence (Laws 10.3: the winner serves): kept at its
                # confidence (sim: overruling a call < 0.55 right 100/114, deciding an unknown 26/26)
                c.outcome.confidence = min(c.outcome.confidence, cfg.ns_conflict_conf)
        elif implied in SIDES and w in SIDES and w != implied and q0 < cfg.ns_conflict_max_q:
            # the call is kept, but the next serve says the other side won it: two independent
            # readings disagree, so the call is not certain either
            c.outcome.confidence = min(c.outcome.confidence, cfg.ns_conflict_conf)
        if c.outcome.winner in SIDES:
            st.point(c.outcome.winner)


def _make_shot(h: HitEvent, seg: ShotSegment3D | None, first: bool) -> ShotEventV2:
    pos = height = None
    if seg is not None:
        pos = (float(seg.p0[0]), float(seg.p0[1]))
        height = float(seg.p0[2])
    return ShotEventV2(hit=h, segment=seg,
                       shot_type="serve" if first else "unknown",
                       shot_type_confidence=0.3 if first else 0.0,
                       hitter_pos_world=pos, hit_height_m=height)


def _to_rally(c: _Cand) -> RallyV2:
    o = c.outcome
    known = o.winner in SIDES
    conf = c.seg_conf * (o.confidence if known else 1.0)
    r = RallyV2(start_frame=int(c.start_frame), end_frame=int(c.end_frame),
                shots=[_make_shot(h, s, k == 0 and not c.serve_missed)
                       for k, (h, s) in enumerate(zip(c.hits, c.segs))],
                end_reason=o.end_reason, landing_xy=o.landing_xy, in_out_margin_m=o.margin,
                winner=o.winner, confidence=float(np.clip(conf, 0.0, 1.0)))
    # extra (non-contract) attributes; lost if a later stage rebuilds the rally
    r.server = c.server
    r.serve_missed = bool(c.serve_missed)
    r.segmentation_confidence = float(c.seg_conf)
    r.outcome_confidence = float(o.confidence)
    r.outcome_source = o.source
    return r


# ─── public API ──────────────────────────────────────────────────────────────

_CUE_SCORE = {"landing": 1.0, "static": 1.0, "gap": 0.8, None: 0.7}


def _activity_rallies(mo: _Motion, cfg: RallyConfig) -> list[RallyV2]:
    """No hits at all: contiguous moving stretches of the track."""
    moving = mo.fin & ~mo.rest & (np.nan_to_num(mo.speed) >= cfg.min_flight_speed_hps)
    idx = np.flatnonzero(moving)
    if not len(idx):
        return []
    runs, s, p = [], int(idx[0]), int(idx[0])
    for f in idx[1:]:
        if mo.t[f] - mo.t[p] > cfg.activity_gap_s:
            runs.append((s, p))
            s = int(f)
        p = int(f)
    runs.append((s, p))
    return [RallyV2(start_frame=s, end_frame=e, confidence=0.3)
            for s, e in runs if mo.t[e] - mo.t[s] >= cfg.activity_min_s]


def _missed_serve_start(mo: _Motion, f: int, t: float, cfg: RallyConfig,
                        f_min: int = 0) -> int | None:
    """If the shuttle was already in fast flight before the first detected hit
    of a group (the serve itself was missed), the frame where that flight
    began: the end of the last rest (hold) / unseen stretch before it."""
    f_lo = max(f_min, mo.frame_at(t - 3.0))
    ks = np.flatnonzero(mo.rest[f_lo:f])
    a = int(f_lo + ks[-1] + 1) if len(ks) else f_lo
    ok = np.flatnonzero(mo.fin[a:f]) + a
    if len(ok) < 3:
        return None
    # start after the last long unseen stretch
    gaps = np.flatnonzero(np.diff(mo.t[ok]) > 0.4)
    if len(gaps):
        ok = ok[gaps[-1] + 1:]
    if len(ok) < 3 or mo.t[f] - mo.t[ok[0]] < 0.3:
        return None
    v = mo.speed[ok[0] + 1: f]
    v = v[np.isfinite(v)]
    if len(v) < 3 or float(np.mean(v < cfg.min_flight_speed_hps)) >= 0.3:
        return None
    return int(ok[0])


def _extend_flight(mo: _Motion, f0: int, f_end: int, f_limit: int, cfg: RallyConfig) -> int:
    """Last observed frame of continuous flight after the last detected hit
    (stops at a rest, an unobserved stretch > max_unseen_s, or 4 s); `f_end`
    if that stretch does not look like flight."""
    f1 = min(f_limit, mo.frame_at(mo.t[f0] + 4.0))
    obs = np.flatnonzero(mo.fin[f0 + 1:f1 + 1] & ~mo.rest[f0 + 1:f1 + 1]) + f0 + 1
    if not len(obs) or obs[-1] <= f_end:
        return f_end
    rs = np.flatnonzero(mo.rest[f0 + 1:f1 + 1])
    if len(rs):
        obs = obs[obs < f0 + 1 + rs[0]]
    tt = np.r_[mo.t[f0], mo.t[obs]]
    gaps = np.flatnonzero(np.diff(tt) > cfg.max_unseen_s)
    if len(gaps):
        obs = obs[:gaps[0]]
    if not len(obs) or obs[-1] <= f_end:
        return f_end
    return int(obs[-1]) if _continuous_flight(mo, f0, int(obs[-1]), cfg) else f_end


def _leading_nonhit(g: list[int], hs: list[HitEvent], segs: list[ShotSegment3D | None],
                    mo: _Motion, have_3d: bool, cfg: RallyConfig) -> bool:
    """Is the first hit of a group a pick-up / hand motion rather than a serve?
    Yes if no flight follows it before the next hit; or, when another hit
    follows within 0.6 s, if that one comes from the same side with at least
    the same confidence (a serve is never followed that fast by the server's
    own contact) or the first is weak and has no 3-D flight although other
    flights were reconstructed."""
    h0 = hs[g[0]]
    h1 = hs[g[1]] if len(g) > 1 else None
    if not _is_flight(h0, segs[g[0]], mo, cfg, h1.t if h1 is not None else None):
        return True
    if h1 is None or h1.t - h0.t > 0.6:
        return False
    c0, c1 = h0.confidence or 0.0, h1.confidence or 0.0
    if h0.hitter in SIDES and h0.hitter == h1.hitter and c0 <= c1:
        return True
    return have_3d and segs[g[0]] is None and c0 < 0.4 <= c1


def segment_rallies(track: Track2D, hits: list[HitEvent], segments: list[ShotSegment3D],
                    meta: VideoMeta, camera: CameraModel | None = None,
                    players: list[PlayerObservation] | None = None,
                    config: RallyConfig | None = None) -> list[RallyV2]:
    """Split a match into rallies with shots, end reason, landing and winner.
    See the module docstring. `segments` may be empty (no camera)."""
    cfg = config or RallyConfig()
    if track is None or track.n == 0:
        return []
    height = float(meta.height) if meta is not None and meta.height else (
        float(camera.image_size[1]) if camera is not None else 720.0)
    mo = _Motion(track, height, cfg)
    doubles = cfg.doubles if cfg.doubles is not None else detect_doubles(players)
    hs = _clean_hits(hits, mo.n)
    hs = [h if h.t is not None and math.isfinite(h.t) else replace(h, t=float(mo.t[h.frame]))
          for h in hs]
    if not hs:
        return _activity_rallies(mo, cfg)
    segs = _match_segments(hs, list(segments or []), mo.t, cfg)
    hs, segs, conts = _merge_same_flight(hs, segs, mo.t, cfg)
    groups, cues = _group_hits(hs, segs, mo, cfg, conts)
    labels0 = [h.hitter for h in hs]                  # the side prior the 3-D fits were made with
    if cfg.joint_sides:
        hs, side_p = _decode_sides(hs, segs, conts, groups, mo.t, cfg)
    else:
        hs, side_p = _fill_hitters(hs, segs, groups), None
    have_3d = camera is not None and bool(segments)

    cands: list[_Cand] = []
    for gi, g in enumerate(groups):
        # a serve launches a real flight: drop leading pick-ups / carries,
        # and a trailing "hit" without any flight (a catch)
        while g and _leading_nonhit(g, hs, segs, mo, have_3d, cfg):
            g = g[1:]
        while len(g) > 1 and segs[g[-1]] is None and not _is_flight(hs[g[-1]], None, mo, cfg):
            g = g[:-1]
        cont_last = None                  # fit of a dropped spurious trailing kink: the real end
        while len(g) > 1 and (hs[g[-1]].confidence or 0.0) < cfg.trail_max_conf \
                and _moving_after(mo, hs[g[-1]], cfg) < cfg.trail_min_flight_s \
                and not (cfg.trail_launch_guard and _launches(
                    hs[g[-1]], segs[g[-1]], conts[g[-2]] if conts[g[-2]] is not None else segs[g[-2]], cfg,
                    mo, camera)):
            dropped = conts[g[-1]] if conts[g[-1]] is not None else segs[g[-1]]
            if cfg.trail_keep_fit and cont_last is None and dropped is not None:
                cont_last = dropped
            g = g[:-1]
        if len(g) == 1 and have_3d and segs[g[0]] is None \
                and _moving_after(mo, hs[g[0]], cfg) < cfg.onehit_min_flight_s:
            continue
        if not g:
            continue
        g_hits, g_segs = [hs[i] for i in g], [segs[i] for i in g]
        last = g_hits[-1]
        limit = hs[groups[gi + 1][0]].frame - 1 if gi + 1 < len(groups) else mo.n - 1
        # the flight is looked at from past a hold touching the last contact (a
        # one-shot rally's serve): that hold is not where the rally ended
        f_la = _launch_frame(mo, last.frame, limit + 1, cfg)
        f_end_2d = mo.flight_end(f_la, limit + 1)
        # 2-D landing: the shuttle comes to rest shortly after it was last seen,
        # or after a continuous flight (trailing contacts the detector missed)
        f_rest = mo.first_rest_after(f_la, min(limit + 1, mo.frame_at(last.t + 4.0) + 1))
        if f_rest is not None and mo.t[f_rest] - mo.t[f_end_2d] > cfg.max_unseen_land_s \
                and not _continuous_flight(mo, f_la, f_rest - 1, cfg):
            f_rest = None
        if f_rest is None:
            f_end_2d = _extend_flight(mo, f_la, f_end_2d, limit, cfg)
        c_last = conts[g[-1]] if conts[g[-1]] is not None else cont_last
        if c_last is not None and g_segs[-1] is None:
            c_last = None                 # no launch fit to pair the continuation with
        seg_last = c_last if c_last is not None else g_segs[-1]
        launch = g_segs[-1] if c_last is not None else None
        server_y = float(g_segs[0].p0[1]) if g_segs[0] is not None else None
        f_prev = hs[groups[gi - 1][-1]].frame + 1 if gi > 0 else 0
        # missed serve: only trusted when a real exchange follows (a one-hit
        # "rally" preceded by motion is usually a feed launched from a carry)
        ms = _missed_serve_start(mo, g_hits[0].frame, g_hits[0].t, cfg, f_prev) \
            if len(g) >= 2 else None
        if ms is not None:
            server_y = None
        # observed flight (not rest) going on well past the end of the last
        # fit that neither landed nor ended at the net = missed final contact
        missed_final = False
        if seg_last is not None and seg_last.landing_xy is None and last.hitter in SIDES:
            nh = seg_last.net_crossing_height
            net_bound = getattr(seg_last, "end_reason", None) == "net" or (
                nh is not None and nh < cm.NET_HEIGHT_CENTRE)
            if launch is not None:
                lh = launch.net_crossing_height
                net_bound = lh is not None and lh < cm.NET_HEIGHT_CENTRE
            f_se = min(max(int(seg_last.end_frame), 0), mo.n - 1)
            f_fly = f_end_2d if f_rest is None else min(f_end_2d, f_rest - 1)
            missed_final = (not net_bound and f_fly > f_se
                            and mo.t[f_fly] - mo.t[f_se] > cfg.missed_final_s
                            and _continuous_flight(mo, f_se, f_fly, cfg))
        # every call names the winner through the last hitter's side, so its
        # confidence includes P(that side) from the joint decode. The 3-D fit of
        # the last flight was made with the hits.py label as its side prior, so
        # its landing / net geometry is judged for the side the fit is
        # consistent with: its own launch reading when clear, else the label it
        # was fitted with. When that differs from the decoded side the call's
        # confidence is P(decoded side is wrong), and the next-serve rule can
        # overrule it
        last_p = side_p[g[-1]] if side_p is not None else None
        call_hit = last
        if last_p is not None and seg_last is not None:
            own = _own_side_reading(g_segs[-1], cfg)
            if _fit_weight(g_segs[-1], cfg) < 0.5:
                own = None                               # an undetermined fit reads no side
            fit_side = labels0[g[-1]] if (cfg.side_call_mode == "label" or own not in SIDES) else own
            if fit_side in SIDES and fit_side != last.hitter:
                call_hit, last_p = replace(last, hitter=fit_side), 1.0 - last_p
        if missed_final:
            # the fitted flight ends at a contact nobody detected and the track
            # flies on: the final shot is unobserved in 3-D. Judging it from a
            # 2-D rest point proved unreliable -> unknown, resolved from the
            # next serve
            o = _Outcome(source="missed_final")
        elif seg_last is not None:
            fe = int(seg_last.end_frame)
            if f_end_2d > last.frame:
                fe = min(fe, f_end_2d)
            fe = min(max(fe, 0), mo.n - 1)
            o = _outcome_3d(call_hit, seg_last, float(mo.t[fe]), len(g) == 1 and ms is None,
                            doubles, server_y, cfg, launch, side_p=last_p)
        elif camera is not None:
            o = _outcome_2d(last, f_rest, mo, camera, doubles, cfg)
            if last_p is not None:
                o.confidence *= last_p
        else:
            o = _Outcome()
        if o.end_t is not None and o.source in ("3d", "implausible_landing"):
            f_end = int(np.clip(mo.frame_at(o.end_t), last.frame, max(limit, last.frame)))
        elif f_rest is not None:
            f_end = f_rest
        else:
            f_end = max(f_end_2d, last.frame)
        cb = cues[gi - 1] if gi > 0 else None
        ca = cues[gi] if gi < len(cues) else None
        boundary = 0.5 * (_CUE_SCORE.get(cb, 0.7) + _CUE_SCORE.get(ca, 0.7))
        hconf = float(np.mean([min(max(h.confidence or 0.0, 0.0), 1.0) for h in g_hits]))
        seg_conf = 0.3 + 0.35 * boundary + 0.2 * _alternation(g_hits) + 0.15 * hconf
        cands.append(_Cand(idx=g, hits=g_hits, segs=g_segs,
                           start_frame=ms if ms is not None else g_hits[0].frame,
                           end_frame=f_end, end_t=float(mo.t[f_end]), outcome=o,
                           seg_conf=float(min(seg_conf, 1.0)), cue_before=cb,
                           serve_missed=ms is not None,
                           first_side_p=side_p[g[0]] if side_p is not None else 1.0,
                           held_before=_hold_before(mo, g_hits[0].frame, g_hits[0].t, cfg) == 1.0))

    # one-hit groups: feeds (shuttle returned to the server) / spurious hits
    kept: list[_Cand] = []
    for i, c in enumerate(cands):
        if len(c.hits) == 1:
            prev = kept[-1] if kept else None
            nxt = cands[i + 1] if i + 1 < len(cands) else None
            if _nonrally_log_odds(c, prev, nxt, mo, doubles, have_3d, cfg) > cfg.feed_threshold:
                continue
        kept.append(c)

    _infer_from_next_serve(kept, cfg)
    return [_to_rally(c) for c in kept]


def refine_hitters(track: Track2D, hits: list[HitEvent], segments: list[ShotSegment3D],
                   meta: VideoMeta, camera: CameraModel | None = None,
                   players: list[PlayerObservation] | None = None,
                   config: RallyConfig | None = None) -> tuple[list[HitEvent], list[bool]]:
    """Hitter sides from the joint decode (the same one segment_rallies runs),
    returned on the INPUT hit list (same order and length) plus per hit whether
    its side changed. For an orchestrator that re-runs reconstruct3d with the
    corrected sides (every fit uses the hitter side as its prior) and then
    calls segment_rallies again. Hits merged into a flight or outside the track
    keep their label."""
    cfg = config or RallyConfig()
    if track is None or track.n == 0 or not hits:
        return [replace(h) for h in hits or []], [False] * len(hits or [])
    height = float(meta.height) if meta is not None and meta.height else (
        float(camera.image_size[1]) if camera is not None else 720.0)
    mo = _Motion(track, height, cfg)
    hs = _clean_hits(hits, mo.n)
    hs = [h if h.t is not None and math.isfinite(h.t) else replace(h, t=float(mo.t[h.frame]))
          for h in hs]
    segs = _match_segments(hs, list(segments or []), mo.t, cfg)
    hs, segs, conts = _merge_same_flight(hs, segs, mo.t, cfg)
    groups, _ = _group_hits(hs, segs, mo, cfg, conts)
    dec, _ = _decode_sides(hs, segs, conts, groups, mo.t, cfg)
    side = {h.frame: d.hitter for h, d in zip(hs, dec)}
    out, changed = [], []
    for h in hits:
        s = side.get(int(h.frame), h.hitter)
        out.append(replace(h, hitter=s))
        changed.append(s != h.hitter)
    return out, changed


def score_match(rallies: list[RallyV2], first_server: str = "near", games_to_win: int = 2,
                min_confidence: float = 0.5, points: int = 21, cap: int = 30, interval: int = 11,
                doubles: bool = False, starters: dict | None = None) -> list[dict]:
    """Running BWF score after every rally (see module docstring). Conformance:
    tests/test_bwf_conformance.py runs every DELIVERY_R2 score vector.

    Per rally: rally (index), start_frame, end_frame, game, server (expected
    physical side from the rules), observed_server (first hitter),
    service_court ('right'|'left' of the server), winner, score / games
    ({'near','far'} after this rally, sides as played in it), score_by_player
    / games_by_player ({'A','B'}: A started game 1 at the near end, the same
    people for the whole recording), game_over, game_winner, match_over,
    ends_changed (ends swap before the next rally), next_server, uncertain +
    reasons. Doubles adds server_player, receiver_player ('A1'..'B2') and
    positions (court of each player BEFORE the rally; Law 11).
    The rally after match over starts a new match (reason 'new_match'), also
    when its winner is unknown; the new match is served by the last match's
    winner from the right court (0-0, Law 10.1.1). Rallies with an unknown
    winner leave the score unchanged. uncertain = a reason other than
    'new_match' (winner unknown / inferred, low confidence, server mismatch)."""
    st = _Score(first_server, games_to_win, points, cap, interval, doubles, starters)
    rs = sorted(rallies, key=lambda r: r.start_frame)
    out: list[dict] = []
    for i, r in enumerate(rs):
        reasons: list[str] = ["new_match"] if st.begin() else []
        exp_server, court = st.server_side(), st.service_court()
        who = (st.server_player(), st.receiver_player(), dict(sorted(st.pos.items())))
        obs = _observed_server(r)
        if obs in SIDES and obs != exp_server:
            reasons.append("server_mismatch")
        w = r.winner if r.winner in SIDES else "unknown"
        if w == "unknown":
            nxt = _observed_server(rs[i + 1]) if i + 1 < len(rs) else "unknown"
            w = _implied_winner(st, nxt)
            reasons.append("winner_inferred_from_next_serve" if w in SIDES else "winner_unknown")
        elif (r.confidence or 0.0) < min_confidence:
            reasons.append("low_confidence")
        game = st.game
        if w in SIDES:
            ev = st.point(w)
        else:
            ev = {"score": st.by_side(st.points), "games": st.by_side(st.games),
                  "score_by_player": dict(st.points), "games_by_player": dict(st.games),
                  "game_over": False, "game_winner": None, "match_over": False,
                  "ends_changed": False}
        row = {
            "rally": i, "start_frame": int(r.start_frame), "end_frame": int(r.end_frame),
            "game": game, "server": exp_server, "observed_server": obs,
            "service_court": court, "winner": w,
            "score": ev["score"], "games": ev["games"],
            "score_by_player": ev["score_by_player"], "games_by_player": ev["games_by_player"],
            "game_over": ev["game_over"], "game_winner": ev["game_winner"],
            "match_over": ev["match_over"], "ends_changed": ev["ends_changed"],
            "next_server": st.server_side(),
            "uncertain": any(x != "new_match" for x in reasons), "reasons": reasons,
        }
        if doubles:
            row["server_player"], row["receiver_player"], row["positions"] = who
        out.append(row)
    return out
