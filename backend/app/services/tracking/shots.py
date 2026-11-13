"""
Shot-type classification for a reconstructed rally.

Taxonomy: a reduced version of ShuttleSet's 18 stroke types (Wang et al.,
"ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton
Tactical Analysis", KDD 2023). Mapping ShuttleSet → ours:

  short / long service                        → serve
  smash, wrist smash                          → smash
  rush                                        → kill
  clear                                       → clear
  drop, passive drop, cut                     → drop
  net shot, return net                        → net
  lob, defensive return lob                   → lift
  drive, driven flight, back-court drive      → drive
  push                                        → push
  defensive return drive / soft smash return  → block

Values beyond the persisted ShotType enum: 'lift', 'push', 'kill', 'block'
(see EXTRA_SHOT_TYPES / PERSISTED_FALLBACK).

Method: a transparent fuzzy decision tree. Each class is a product of soft
conditions sigmoid(margin / softness) over physical features of the 3-D flight
(contact height, distance behind the net, launch speed & elevation, apex, net
clearance, landing depth, descent angle) plus hitter pose (overhead contact).
Scores are normalised into a confidence. Everything is deterministic.

Features are taken at the *contact*: segments are fitted from their first
observed frame, so the state is back-propagated through the drag ODE to the
hit time (contact_state) — without this a 300 km/h smash seen 2 frames late
(30 fps) reads as a 140 km/h front-court kill.

Stroke side: wrist offset along the image shoulder line, oriented by the
anatomical (COCO) left/right labels, so near (back to camera) and far
(facing it, mirrored) players need no special case. Side-on views and
near-midline wrists return 'unknown' rather than a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from app.services.tracking import court_model as cm
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, RallyV2, ShotEventV2, ShotSegment3D,
)
from app.services.tracking.physics import DRAG_LENGTH, G

SHOT_TYPES = ("serve", "smash", "kill", "clear", "drop", "net", "lift",
              "drive", "push", "block", "unknown")
EXTRA_SHOT_TYPES = ("lift", "push", "kill", "block")
# Closest persisted ShotType for DBs that only know the original enum.
PERSISTED_FALLBACK = {"lift": "clear", "push": "drive", "kill": "smash", "block": "net"}

# COCO keypoint indices
_NOSE, _LSH, _RSH, _LWR, _RWR, _LHIP, _RHIP = 0, 5, 6, 9, 10, 11, 12
_KP_CONF = 0.3


def _sig(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _gt(a: float, b: float, s: float) -> float:
    """Soft 'a > b' with softness s (same units as a)."""
    return _sig((a - b) / s)


def _lt(a: float, b: float, s: float) -> float:
    return _sig((b - a) / s)


# ── fast scalar shuttle integrator (pure floats: ~10x faster than numpy for 3-vectors)

def _acc(vx: float, vy: float, vz: float, k: float) -> tuple[float, float, float]:
    sp = math.sqrt(vx * vx + vy * vy + vz * vz) * k
    return -sp * vx, -sp * vy, -G - sp * vz


def _rk4(p: tuple, v: tuple, h: float, k: float) -> tuple[tuple, tuple]:
    (x, y, z), (vx, vy, vz) = p, v
    a1 = _acc(vx, vy, vz, k)
    b = (vx + .5 * h * a1[0], vy + .5 * h * a1[1], vz + .5 * h * a1[2])
    a2 = _acc(*b, k)
    c = (vx + .5 * h * a2[0], vy + .5 * h * a2[1], vz + .5 * h * a2[2])
    a3 = _acc(*c, k)
    d = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2])
    a4 = _acc(*d, k)
    p2 = (x + h / 6 * (vx + 2 * b[0] + 2 * c[0] + d[0]),
          y + h / 6 * (vy + 2 * b[1] + 2 * c[1] + d[1]),
          z + h / 6 * (vz + 2 * b[2] + 2 * c[2] + d[2]))
    v2 = (vx + h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
          vy + h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]),
          vz + h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2]))
    return p2, v2


def _drag_k(drag_length: float | None) -> float:
    L = drag_length if drag_length and math.isfinite(drag_length) and drag_length > 0.5 else DRAG_LENGTH
    return 1.0 / L


MAX_BACKPROP_S = 0.25            # never extrapolate a fit further back than this
_MAX_SPEED = 500.0 / 3.6         # m/s, plausibility bound for the backward solution


def contact_state(seg: ShotSegment3D, t_hit: float | None) -> tuple[np.ndarray, np.ndarray]:
    """(p, v) of the shuttle at the contact time `t_hit`.

    Segments are parametrised at their first *observed* frame, which usually
    trails the contact by 1–3 frames (motion blur / racket occlusion). Drag
    decelerates fast shots violently (a 300 km/h smash is at ~140 km/h two
    frames later at 30 fps), so integrate the drag ODE backwards to the hit.
    Bounded to MAX_BACKPROP_S and stopped before the backward solution becomes
    implausible (it blows up in finite time). Only corrects hit-before-fit."""
    p = np.asarray(seg.p0, float)
    v = np.asarray(seg.v0, float)
    t0 = getattr(seg, "t0", None)
    if t_hit is None or t0 is None or not (math.isfinite(t_hit) and math.isfinite(t0)):
        return p, v
    dt = float(t0) - float(t_hit)
    if not 1e-4 < dt <= MAX_BACKPROP_S:
        return p, v
    k = _drag_k(seg.drag_length)
    n = max(1, math.ceil(dt * 480))
    h = -dt / n
    pp, vv = tuple(map(float, p)), tuple(map(float, v))
    for _ in range(n):
        p2, v2 = _rk4(pp, vv, h, k)
        sp = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
        if not (sp <= _MAX_SPEED and p2[2] >= 0.0 and all(map(math.isfinite, p2))):
            break
        pp, vv = p2, v2
    return np.array(pp), np.array(vv)


@dataclass(frozen=True)
class FlightFeatures:
    """Physical descriptors of one stroke (metres, km/h, degrees)."""
    contact_height: float        # p0.z
    contact_depth: float         # |p0.x|, distance of contact behind the net
    speed_kmh: float             # |v0|
    elevation_deg: float         # launch angle above horizontal (negative = downward)
    apex: float                  # max z of the free flight
    net_clearance: float         # z at net crossing − tape height (NaN if not crossed)
    landing_depth: float         # landing x on opponent side, from net (negative = own side)
    descent_deg: float           # angle below horizontal at landing
    flight_time: float           # s to floor


def flight_features(p0: np.ndarray, v0: np.ndarray, drag_length: float | None = None,
                    dt: float = 1 / 240, t_max: float = 5.0) -> FlightFeatures:
    """Integrate the free flight from contact to the floor and summarise it."""
    k = _drag_k(drag_length)
    p = (float(p0[0]), float(p0[1]), float(p0[2]))
    v = (float(v0[0]), float(v0[1]), float(v0[2]))
    # direction towards the opponent: from the hitter's half, or (contact right
    # at the net, where reconstruction noise can flip the sign) from v0
    if abs(p[0]) > 0.3 or v[0] == 0.0:
        toward = -1.0 if p[0] > 0 else 1.0
    else:
        toward = 1.0 if v[0] > 0 else -1.0
    apex, net_z, t = p[2], math.nan, 0.0
    while t < t_max:
        p2, v2 = _rk4(p, v, dt, k)
        if math.isnan(net_z) and (p[0] - 0.0) * (p2[0] - 0.0) <= 0 and p[0] != p2[0]:
            f = p[0] / (p[0] - p2[0])
            net_z = p[2] + f * (p2[2] - p[2])
        if p2[2] <= 0.0:
            f = p[2] / max(p[2] - p2[2], 1e-12)
            p = tuple(a + f * (b - a) for a, b in zip(p, p2))
            v = tuple(a + f * (b - a) for a, b in zip(v, v2))
            t += f * dt
            break
        p, v, t = p2, v2, t + dt
        apex = max(apex, p[2])
    hv0 = math.hypot(float(v0[0]), float(v0[1]))
    hv = math.hypot(v[0], v[1])
    return FlightFeatures(
        contact_height=float(p0[2]), contact_depth=abs(float(p0[0])),
        speed_kmh=float(np.linalg.norm(v0) * 3.6),
        elevation_deg=math.degrees(math.atan2(float(v0[2]), hv0)),
        apex=apex,
        net_clearance=net_z - cm.NET_HEIGHT_CENTRE if not math.isnan(net_z) else math.nan,
        landing_depth=p[0] * toward,
        descent_deg=math.degrees(math.atan2(-v[2], hv)),
        flight_time=t)


# ── soft conditions (NaN input → 0.5, i.e. uninformative) ─────────────────────

def _g(a: float, b: float, s: float) -> float:
    return 0.5 if a is None or math.isnan(a) else _gt(a, b, s)


def _l(a: float, b: float, s: float) -> float:
    return 0.5 if a is None or math.isnan(a) else _lt(a, b, s)


def _between(a: float, lo: float, hi: float, s: float) -> float:
    return _g(a, lo, s) * _l(a, hi, s)


def _scores_3d(f: FlightFeatures, overhead: float) -> dict[str, float]:
    """Soft decision tree over flight features. `overhead` ∈ [0,1] is the
    probability of an above-head contact (3-D height fused with pose)."""
    h, d, s, e = f.contact_height, f.contact_depth, f.speed_kmh, f.elevation_deg
    rise, L, nc = f.apex - f.contact_height, f.landing_depth, f.net_clearance
    # Height bands: underarm (< 1.35 m) | mid (chest/shoulder) | overhead (> 2.1 m).
    # Every class is defined over the mid band too, so no contact height is a
    # dead zone: lift/drive/net = not overhead; drop = not underarm (ShuttleSet's
    # passive drop); block = underarm only.
    low = 1.0 - overhead
    under = _l(h, 1.35, 0.12)                    # clearly underarm contact
    down = _l(e, -4.0, 2.0)                      # launched downward
    # > 220 km/h is only reachable overhead: a flat one is still a (flat) smash
    flat_fast = _l(e, 2.0, 2.0) * _g(s, 220.0, 15.0)
    lofted = _g(f.apex, 3.2, 0.25)               # high arc (lift/clear) vs flat (drive/push)
    front, rear = _l(d, 2.5, 0.25), _g(d, 2.5, 0.25)   # one front-court boundary
    return {
        "smash": overhead * max(down, flat_fast) * _g(s, 140.0, 12.0) * rear,
        "kill":  _g(h, 1.7, 0.12) * down * _g(s, 60.0, 10.0) * front,
        "clear": overhead * _g(e, 5.0, 3.0) * _g(L, 4.4, 0.35) * _g(rise, 0.6, 0.15),
        "drop":  (1.0 - under) * _g(d, 2.5, 0.3) * _l(L, 3.0, 0.3) * _l(s, 130.0, 12.0)
                 * _l(e, 40.0, 5.0),
        # net speed bound 70 km/h (was 50): ShuttleSet cross-court net shot launch p90 66 km/h
        # (DELIVERY_R3 APP_THRESHOLDS_REVIEW.md); with push 6.2 m: shot type +0.40 pt, paired
        # bootstrap CI > 0 on sim dev s0-39, held-out s1000-1039 and confirm s2000-2039 (accuracy2.log)
        "net":   low * front * _l(f.apex, 3.0, 0.2) * _l(L, 2.3, 0.3)
                 * _l(s, 70.0, 8.0) * _l(nc, 0.9, 0.15),
        "block": under * rear * _l(L, 2.7, 0.25) * _l(s, 75.0, 10.0),
        "lift":  low * lofted * _g(e, 8.0, 3.0) * _g(L, 4.2, 0.35),
        # fast and not lofted; from a low contact a drive must climb to clear the tape
        "drive": low * _g(h, 0.5, 0.15) * (1.0 - lofted) * _between(e, -12.0, 25.0, 2.5)
                 * _g(s, 85.0, 10.0) * _g(L, 2.5, 0.3),
        # flat over the tape (low apex, whatever the contact height) into mid-court:
        # beyond net shots / blocks (< ~2.5 m), short of lifts
        "push":  low * _l(h, 1.95, 0.12) * _l(f.apex, 2.7, 0.15)
                 * _between(s, 22.0, 90.0, 8.0) * _between(L, 2.3, 6.2, 0.25)   # 6.2 m (was 5.0): ShuttleSet push p90 6.12 m
                 * _l(d, 4.0, 0.3),                    # front/mid court, not the rear
    }


def _serve_score(f: FlightFeatures) -> float:
    """How serve-like a first stroke is (legal serves are struck below 1.15 m
    from inside the service court, i.e. behind the short service line)."""
    return (_l(f.contact_height, 1.35, 0.1) * _g(f.contact_depth, 1.7, 0.2)
            * _g(f.elevation_deg, -2.0, 4.0))


# ── pose helpers ──────────────────────────────────────────────────────────────

def _kp(k: np.ndarray, i: int) -> np.ndarray | None:
    return k[i, :2] if k[i, 2] >= _KP_CONF else None


def wrist_above_head(kps: np.ndarray | None) -> tuple[float, int | None]:
    """(P(overhead contact), index of the raised wrist or None) from COCO pose.
    Margin is the wrist's height above the nose/shoulders in torso lengths."""
    if kps is None or len(kps) < 17:
        return 0.5, None
    k = np.asarray(kps, float)
    sh = [p for p in (_kp(k, _LSH), _kp(k, _RSH)) if p is not None]
    hp = [p for p in (_kp(k, _LHIP), _kp(k, _RHIP)) if p is not None]
    if not sh:
        return 0.5, None
    sh_y = float(np.mean([p[1] for p in sh]))
    torso = abs(float(np.mean([p[1] for p in hp])) - sh_y) if hp else 0.0
    ys = k[k[:, 2] >= _KP_CONF, 1]          # undetected keypoints may sit at (0, 0)
    torso = max(torso, 1e-3 + 0.25 * float(ys.max() - ys.min()))
    nose = _kp(k, _NOSE)
    head_y = float(nose[1]) if nose is not None else sh_y - 0.35 * torso
    best, idx = None, None
    for w in (_LWR, _RWR):
        p = _kp(k, w)
        if p is not None and (best is None or p[1] < best):
            best, idx = float(p[1]), w
    if best is None:
        return 0.5, None
    return _gt((head_y - best) / torso, 0.0, 0.15), idx


# Calibrated on projected 3-D skeletons over 360° of body yaw at 5 court spots
# (broadcast camera): every wrong sign had across-body width / torso ≤ 0.35.
_MIN_WIDTH_RATIO = 0.4   # across-body shoulder (hip) width / torso length: side-on below
_DEAD_ZONE = 0.25        # |wrist lateral offset| / shoulder width below this = ambiguous


def _body_right(k: np.ndarray) -> tuple[np.ndarray, float] | None:
    """(image-plane unit vector towards the player's anatomical right, shoulder
    or hip width in px). COCO labels are anatomical, so this is correct for a
    player seen from behind (near side) and mirrored facing the camera (far
    side) alike. None when the view is side-on (the lateral axis points along
    the viewing ray and forehand/backhand is not observable in 2-D)."""
    sh = (_kp(k, _LSH), _kp(k, _RSH))
    hp = (_kp(k, _LHIP), _kp(k, _RHIP))
    mid = lambda a, b: (a + b) / 2 if a is not None and b is not None else None  # noqa: E731
    ms, mh = mid(*sh), mid(*hp)
    torso = float(np.linalg.norm(ms - mh)) if ms is not None and mh is not None else 0.0
    # Side-on guard: from an elevated camera a side-on player's shoulder line is
    # drawn along the image torso axis (depth and height confounded), so require
    # enough across-body extent (component perpendicular to the torso axis).
    perp = (np.array([-(ms - mh)[1], (ms - mh)[0]]) / torso) if torso > 1e-6 else None
    for l, r in (sh, hp):
        if l is None or r is None:
            continue
        v = r - l
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        if perp is not None and abs(float(v @ perp)) < _MIN_WIDTH_RATIO * torso:
            continue
        return v / n, n
    return None


def stroke_side(kps: np.ndarray | None, overhead: float, right_handed: bool | None) -> str:
    """'overhead' | 'forehand' | 'backhand' | 'unknown'. Racket wrist = the
    dominant hand; forehand when it is on the racket-arm side of the torso
    (lateral offset along the anatomical right axis, sign flipped for
    left-handers). Near-midline or side-on views → 'unknown'."""
    if overhead >= 0.6:
        return "overhead"
    if kps is None or right_handed is None:
        return "unknown"
    k = np.asarray(kps, float)
    if k.shape[0] < 17:
        return "unknown"
    br = _body_right(k)
    wr = _kp(k, _RWR if right_handed else _LWR)
    pts = [p for p in (_kp(k, i) for i in (_LSH, _RSH, _LHIP, _RHIP)) if p is not None]
    if br is None or wr is None or not pts:
        return "unknown"
    rv, width = br
    lateral = float((wr - np.mean(pts, axis=0)) @ rv)
    if not right_handed:
        lateral = -lateral
    if abs(lateral) < _DEAD_ZONE * width:
        return "unknown"
    return "forehand" if lateral > 0 else "backhand"


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def overhead_prob(contact_height: float | None, pose_overhead: float | None) -> float:
    """Fuse 3-D contact height (overhead ≈ above 2.1 m) with pose evidence in log-odds."""
    z = 0.0
    if contact_height is not None and math.isfinite(contact_height):
        z += _logit(_gt(contact_height, 2.1, 0.12))
    if pose_overhead is not None and pose_overhead != 0.5:
        z += 0.7 * _logit(pose_overhead)
    return _sig(z)


def _normalise(scores: dict[str, float], floor: float = 0.02) -> tuple[str, float]:
    best = max(scores, key=lambda k: (scores[k], k))
    total = sum(scores.values())
    if scores[best] < floor:
        return "unknown", 0.0
    # share of evidence × absolute support
    conf = (scores[best] / (total + 0.05)) * min(1.0, 0.5 + scores[best])
    return best, float(min(max(conf, 0.0), 1.0))


def _scores_2d(depth: float | None, next_depth: float | None, dt: float | None,
               overhead: float, dist_m: float | None) -> dict[str, float]:
    """Coarse fallback without a 3-D flight: hitter position, where the
    opponent intercepts it (next hitter depth), and inter-hit time."""
    d = math.nan if depth is None else depth
    L = math.nan if next_depth is None else next_depth
    T = math.nan if dt is None else dt
    v = dist_m / dt if (dist_m is not None and dt) else math.nan   # mean ground speed m/s
    low = 1 - overhead
    return {
        "smash": overhead * _g(d, 2.5, 0.3) * _l(T, 0.75, 0.1),
        "kill":  overhead * _l(d, 2.5, 0.3) * _l(T, 0.6, 0.1),
        "clear": overhead * _g(L, 4.0, 0.4) * _g(T, 1.1, 0.15),
        "drop":  overhead * _g(d, 3.0, 0.3) * _l(L, 3.0, 0.4) * _g(T, 0.8, 0.15),
        "net":   low * _l(d, 2.3, 0.3) * _l(L, 2.5, 0.4),
        "lift":  low * _g(L, 4.0, 0.4) * _g(T, 1.1, 0.15),
        "drive": low * _l(T, 0.8, 0.1) * _g(v, 8.0, 1.5),
        "push":  low * _between(L, 2.0, 4.5, 0.4) * _l(v, 8.0, 1.5),
        "block": low * _g(d, 2.5, 0.3) * _l(L, 2.5, 0.4),
    }


# ── hitter lookup ─────────────────────────────────────────────────────────────

class _PlayerIndex:
    """Per-track sorted frames for nearest-in-time lookups."""

    def __init__(self, players: list[PlayerObservation] | None):
        self.by_track: dict[int, tuple[np.ndarray, list[PlayerObservation]]] = {}
        groups: dict[int, list[PlayerObservation]] = {}
        for o in players or []:
            groups.setdefault(o.track_id, []).append(o)
        for tid, obs in groups.items():
            obs.sort(key=lambda o: o.frame)
            self.by_track[tid] = (np.array([o.frame for o in obs]), obs)

    def at(self, frame: int, max_gap: int = 8) -> list[PlayerObservation]:
        out = []
        for frames, obs in self.by_track.values():
            i = int(np.searchsorted(frames, frame))
            cand = [j for j in (i - 1, i) if 0 <= j < len(obs)]
            j = min(cand, key=lambda j: abs(frames[j] - frame))
            if abs(frames[j] - frame) <= max_gap:
                out.append(obs[j])
        return out


def _pick_hitter(hit: HitEvent, contact: np.ndarray | None,
                 cands: list[PlayerObservation]) -> PlayerObservation | None:
    """Hitter = explicit track id, else the player on the hitting side whose
    feet are closest to the 3-D contact point, else closest bbox in pixels."""
    if not cands:
        return None
    if hit.player_track_id is not None:
        for o in cands:
            if o.track_id == hit.player_track_id:
                return o
    side = hit.hitter
    if side not in ("near", "far") and contact is not None:
        side = cm.side_of(float(contact[0]))
    pool = [o for o in cands if o.side == side] or cands
    if contact is not None and all(o.foot_world is not None for o in pool):
        p = np.asarray(contact[:2], float)
        return min(pool, key=lambda o: float(np.linalg.norm(np.asarray(o.foot_world[:2]) - p)))
    u, v = hit.xy_px

    def dpx(o: PlayerObservation) -> float:
        x1, y1, x2, y2 = o.bbox
        cx, cy = min(max(u, x1), x2), min(max(v, y1), y2)
        return math.hypot(u - cx, v - cy)
    return min(pool, key=dpx)


def _infer_handedness(raised: dict[int, list[int]]) -> dict[int, bool]:
    """Per track: right-handed iff the right wrist is raised in the majority of
    overhead contacts (ties / no evidence → right-handed, the population prior)."""
    return {tid: sum(w == _RWR for w in ws) >= sum(w == _LWR for w in ws)
            for tid, ws in raised.items()}


# ── public API ────────────────────────────────────────────────────────────────

def _valid(seg: ShotSegment3D | None) -> bool:
    return (seg is not None and seg.p0 is not None and seg.v0 is not None
            and bool(np.all(np.isfinite(seg.p0))) and bool(np.all(np.isfinite(seg.v0))))


def classify_segment(seg: ShotSegment3D | None, pose_overhead: float | None = None,
                     is_first: bool = False, t_hit: float | None = None,
                     ) -> tuple[str, float, FlightFeatures | None]:
    """Classify one 3-D flight. Returns (type, confidence, features).
    `t_hit` (contact time) back-propagates the fitted state to the contact."""
    if not _valid(seg):
        return "unknown", 0.0, None
    p0, v0 = contact_state(seg, t_hit)
    f = flight_features(p0, v0, seg.drag_length)
    oh = overhead_prob(f.contact_height, pose_overhead)
    if is_first:
        sv = _serve_score(f)
        if sv >= 0.1:
            return "serve", float(min(1.0, 0.55 + 0.45 * sv)), f
    label, conf = _normalise(_scores_3d(f, oh))
    fit_q = seg.confidence if seg.confidence and seg.confidence > 0 else 1.0
    return label, conf * (0.5 + 0.5 * min(fit_q, 1.0)), f


def classify_shots(rally: RallyV2, camera: CameraModel | None,
                   players: list[PlayerObservation] | None = None,
                   handedness: dict[int, bool] | None = None) -> RallyV2:
    """Fill shot_type, shot_type_confidence, hitter_pos_world, hit_height_m and
    stroke_side for every ShotEventV2 of `rally` (returns a new RallyV2).
    `handedness` (track_id → right-handed) overrides per-rally inference."""
    del camera  # features are world-frame; the camera is only needed upstream
    shots = sorted(rally.shots, key=lambda s: s.hit.frame)
    # contact point = fitted state back-propagated to the hit time
    contacts = [contact_state(s.segment, s.hit.t)[0] if _valid(s.segment) else None
                for s in shots]
    idx = _PlayerIndex(players)
    hitters = [_pick_hitter(s.hit, c, idx.at(s.hit.frame)) for s, c in zip(shots, contacts)]
    poses = [wrist_above_head(h.keypoints if h is not None else None) for h in hitters]

    raised: dict[int, list[int]] = {}
    for c, h, (po, w) in zip(contacts, hitters, poses):
        hz = float(c[2]) if c is not None else None
        if h is not None and w is not None and overhead_prob(hz, po) > 0.7:
            raised.setdefault(h.track_id, []).append(w)
    hand = _infer_handedness(raised)
    hand.update(handedness or {})

    out: list[ShotEventV2] = []
    for i, (s, c, h, (po, _)) in enumerate(zip(shots, contacts, hitters, poses)):
        seg = s.segment if c is not None else None
        pos = None
        if h is not None and h.foot_world is not None:
            pos = (float(h.foot_world[0]), float(h.foot_world[1]))
        elif c is not None:
            pos = (float(c[0]), float(c[1]))
        height = float(c[2]) if c is not None else None
        if seg is not None:
            label, conf, _ = classify_segment(seg, po, is_first=(i == 0), t_hit=s.hit.t)
        else:
            label, conf = _classify_2d(i, shots, pos, hitters, po)
        oh = overhead_prob(height, po)
        right = hand.get(h.track_id, True) if h is not None else None
        side = stroke_side(h.keypoints if h is not None else None, oh, right)
        out.append(replace(s, shot_type=label, shot_type_confidence=round(conf, 4),
                           hitter_pos_world=pos, hit_height_m=height, stroke_side=side))
    return replace(rally, shots=out)


def _classify_2d(i: int, shots: list[ShotEventV2], pos: tuple[float, float] | None,
                 hitters: list[PlayerObservation | None], pose_overhead: float) -> tuple[str, float]:
    """Fallback without a 3-D segment: hitter court position (from players),
    next hitter position and inter-hit time. Confidence capped at 0.45."""
    if i == 0:
        return "serve", 0.4
    nxt = shots[i + 1] if i + 1 < len(shots) else None
    nh = hitters[i + 1] if i + 1 < len(hitters) else None
    npos = (tuple(nh.foot_world[:2]) if nh is not None and nh.foot_world is not None
            else (tuple(nxt.segment.p0[:2]) if nxt is not None and _valid(nxt.segment)
                  else None))
    depth = abs(pos[0]) if pos is not None else None
    ndepth = abs(npos[0]) if npos is not None else None
    dt = (nxt.hit.t - shots[i].hit.t) if nxt is not None else None
    dist = (math.dist(pos, npos) if pos is not None and npos is not None else None)
    oh = 0.5 if pose_overhead is None else pose_overhead
    scores = _scores_2d(depth, ndepth, dt, oh, dist)
    if all(v is None for v in (depth, ndepth, dt)):
        return "unknown", 0.0
    label, conf = _normalise(scores, floor=0.05)
    return label, min(conf * 0.6, 0.45)
