"""Shot classification tests on physically simulated 3-D flights."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from app.services.tracking import court_model as cm
from app.services.tracking import shots as S
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, RallyV2, ShotEventV2, ShotSegment3D,
)
from app.services.tracking.physics import DRAG_LENGTH, simulate_state

# Realistic launch parameters per stroke (hitter on the near side, x < 0).
# contact height z (m), depth behind net |x| (m), speed (km/h), elevation (deg),
# plus the outcome constraint the stroke must satisfy (landing depth from net).
PARAMS: dict[str, dict] = {
    "serve_short": dict(z=(0.85, 1.12), d=(2.0, 2.8), v=(22, 40), e=(5, 25), land=(1.9, 3.2)),
    "serve_long":  dict(z=(0.85, 1.12), d=(2.0, 3.0), v=(90, 150), e=(40, 65), land=(5.2, 7.0)),
    "smash": dict(z=(2.35, 3.05), d=(3.2, 6.3), v=(170, 380), e=(-22, -5), land=(1.0, 6.7)),
    "kill":  dict(z=(1.85, 2.7), d=(0.7, 2.3), v=(70, 200), e=(-40, -8), land=(0.3, 6.7)),
    "clear": dict(z=(2.2, 2.95), d=(4.0, 6.5), v=(110, 210), e=(12, 50), land=(4.7, 7.0)),
    "drop":  dict(z=(2.15, 2.95), d=(3.8, 6.4), v=(40, 120), e=(-22, 12), land=(0.4, 2.7)),
    "net":   dict(z=(0.3, 1.35), d=(0.3, 2.0), v=(10, 38), e=(15, 70), land=(0.1, 2.0),
                  apex=2.4),   # tumbles just over the tape
    "lift":  dict(z=(0.2, 1.1), d=(0.4, 3.5), v=(80, 170), e=(28, 65), land=(4.7, 7.0)),
    "drive": dict(z=(0.95, 1.8), d=(1.8, 5.0), v=(95, 200), e=(-6, 7), land=(3.0, 9.0)),
    "push":  dict(z=(0.85, 1.6), d=(0.9, 2.8), v=(35, 75), e=(-8, 12), land=(2.0, 4.4)),
    "block": dict(z=(0.3, 1.25), d=(3.0, 6.0), v=(25, 60), e=(5, 38), land=(0.4, 2.6)),
}
LABEL = {k: k.split("_")[0] for k in PARAMS}


def make_segment(kind: str, rng: np.random.Generator, far: bool = False,
                 fps: float = 60.0, max_tries: int = 400) -> ShotSegment3D:
    """Rejection-sample a legal flight of `kind` (crosses the net, satisfies the
    landing constraint) and return it as a ShotSegment3D."""
    p = PARAMS[kind]
    for _ in range(max_tries):
        z, d = rng.uniform(*p["z"]), rng.uniform(*p["d"])
        spd, el = rng.uniform(*p["v"]) / 3.6, math.radians(rng.uniform(*p["e"]))
        y0 = rng.uniform(-2.3, 2.3)
        y_t = rng.uniform(-2.4, 2.4)
        az = math.atan2(y_t - y0, d + 3.5)
        v0 = spd * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                             math.sin(el)])
        p0 = np.array([-d, y0, z])
        feat = S.flight_features(p0, v0, DRAG_LENGTH)
        if math.isnan(feat.net_clearance) or feat.net_clearance < 0.02:
            continue
        if not (p["land"][0] <= feat.landing_depth <= p["land"][1]):
            continue
        if feat.apex > p.get("apex", 99.0):
            continue
        break
    else:
        raise RuntimeError(f"cannot sample {kind}")
    if far:                                   # mirror to the far half
        p0, v0 = p0 * [-1, -1, 1], v0 * [-1, -1, 1]
    n = max(3, int(feat.flight_time * fps))
    ts = np.arange(n) / fps
    # fixture-only trajectory (classifier reads p0/v0): a 1/120 s RK4 step is
    # within 0.1 mm of the 1/480 s default for a 380 km/h smash, and 4x cheaper
    P, _ = simulate_state(p0, v0, ts, max_dt=1 / 120)
    return ShotSegment3D(start_frame=0, end_frame=n - 1, t0=0.0, p0=p0, v0=v0,
                         drag_length=DRAG_LENGTH, rmse_px=1.0, frames=np.arange(n),
                         positions=P, landing_xy=None,
                         net_crossing_height=cm.NET_HEIGHT_CENTRE + feat.net_clearance,
                         confidence=0.9)


def _print_confusion(conf: dict[str, Counter]) -> None:
    labels = [k for k in S.SHOT_TYPES if k in conf or any(k in c for c in conf.values())]
    print("\ntrue\\pred " + " ".join(f"{p[:5]:>5}" for p in labels))
    for t in labels:
        if t in conf:
            print(f"{t:>9} " + " ".join(f"{conf[t].get(p, 0):5d}" for p in labels))


@lru_cache(maxsize=None)
def _flights(seed: int = 42, per_kind: int = 40) -> tuple[tuple[str, ShotSegment3D], ...]:
    """Shared, generated-once evaluation set (half near, half far side).
    Treat as read-only: copy with dataclasses.replace before modifying."""
    rng = np.random.default_rng(seed)
    return tuple((kind, make_segment(kind, rng, far=bool(i % 2)))
                 for kind in PARAMS for i in range(per_kind))


def test_accuracy_clean_3d():
    conf: dict[str, Counter] = {}
    ok = n = 0
    for kind, seg in _flights():
        label, c, _ = S.classify_segment(seg, None, is_first=kind.startswith("serve"))
        conf.setdefault(LABEL[kind], Counter())[label] += 1
        ok += label == LABEL[kind]
        n += 1
        assert 0.0 <= c <= 1.0
    _print_confusion(conf)
    acc = ok / n
    print(f"accuracy {acc:.3f} on {n} clean flights")
    assert acc >= 0.90
    for t, c in conf.items():          # no class may collapse
        assert c[t] / sum(c.values()) >= 0.8, (t, dict(c))


def test_accuracy_noisy_3d():
    """Reconstruction-like noise: 10 cm position, 7 % speed, ~2.3° direction."""
    rng = np.random.default_rng(7)
    ok = n = 0
    for kind, base in _flights():
        p0 = base.p0 + rng.normal(0, [0.1, 0.1, 0.08])
        sp = np.linalg.norm(base.v0)
        d = base.v0 / sp + rng.normal(0, 0.04, 3)
        v0 = d / np.linalg.norm(d) * sp * (1 + rng.normal(0, 0.07))
        seg = replace(base, p0=p0, v0=v0)
        label, _, _ = S.classify_segment(seg, None, is_first=kind.startswith("serve"))
        ok += label == LABEL[kind]
        n += 1
    print(f"accuracy {ok / n:.3f} on {n} noisy flights")
    assert ok / n >= 0.85


def _seg(p0, v0, conf=0.9) -> ShotSegment3D:
    return ShotSegment3D(0, 10, 0.0, np.asarray(p0, float), np.asarray(v0, float),
                         DRAG_LENGTH, 1.0, np.arange(3), np.zeros((3, 3)), confidence=conf)


def _v(speed_kmh, elev_deg, toward=+1.0):
    s, e = speed_kmh / 3.6, math.radians(elev_deg)
    return [toward * s * math.cos(e), 0.0, s * math.sin(e)]


def test_borderline_confidence_is_lower():
    lab_hi, c_hi, _ = S.classify_segment(_seg([-5.0, 0, 2.8], _v(300, -12)))
    lab_lo, c_lo, _ = S.classify_segment(_seg([-5.0, 0, 2.15], _v(145, -5)))
    assert lab_hi == "smash" and c_hi > 0.6
    assert c_lo < c_hi


def test_speed_boundary_smash_vs_drop():
    """Same steep overhead contact: fast → smash, slow → drop."""
    fast = S.classify_segment(_seg([-5.5, 0, 2.7], _v(260, -10)))[0]
    slow = S.classify_segment(_seg([-5.5, 0, 2.7], _v(75, -8)))[0]
    assert (fast, slow) == ("smash", "drop")


def test_kill_needs_front_court():
    assert S.classify_segment(_seg([-1.5, 0, 2.3], _v(160, -20)))[0] == "kill"
    assert S.classify_segment(_seg([-5.0, 0, 2.8], _v(260, -12)))[0] == "smash"


def test_mid_height_contacts_have_no_dead_zone():
    """Chest/shoulder-height contacts (1.35–2.05 m: neither underarm nor
    overhead) still get their natural class with real confidence; the same
    rising launch is a lift below head height and a clear overhead; a fast
    shot that must climb from a low contact to clear the tape is a drive, and a
    slow one that climbs just over the tape into mid-court is a push (not a lift:
    the arc stays flat, apex ≈ 2.3 m)."""
    cases = [([-2.0, 0, 1.8], _v(80, 38), "lift"), ([-2.0, 0, 2.5], _v(85, 38), "clear"),
             ([-4.0, 0, 1.6], _v(45, 15), "drop"), ([-4.0, 0, 0.8], _v(50, 30), "block"),
             ([-4.0, 0, 0.9], _v(130, 14), "drive"), ([-1.2, 0, 1.6], _v(22, 15), "net"),
             ([-1.5, 0, 0.5], _v(42, 45), "push")]
    for p0, v0, want in cases:
        lab, conf, _ = S.classify_segment(_seg(p0, v0))
        assert (lab, conf > 0.5) == (want, True), (p0, want, lab, conf)


def _pose(cx: float, cy: float, wrist: str = "right", up: bool = True,
          facing_camera: bool = False, out: bool = True) -> np.ndarray:
    """Synthetic COCO pose, 100 px tall. Viewed from behind (near player) the
    anatomical right is image-right; facing the camera it is image-left."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    r = -1 if facing_camera else 1
    k[0, :2] = (cx, cy - 45)
    k[5, :2], k[6, :2] = (cx - 12 * r, cy - 35), (cx + 12 * r, cy - 35)   # shoulders
    k[11, :2], k[12, :2] = (cx - 8 * r, cy), (cx + 8 * r, cy)             # hips
    k[15, :2], k[16, :2] = (cx - 8, cy + 50), (cx + 8, cy + 50)
    k[9, :2], k[10, :2] = (cx - 14 * r, cy - 10), (cx + 14 * r, cy - 10)  # wrists low
    idx = 10 if wrist == "right" else 9
    sgn = r if wrist == "right" else -r
    if up:
        k[idx, :2] = (cx + 10 * sgn, cy - 70)
    else:
        k[idx, :2] = (cx + (30 if out else -25) * sgn, cy - 15)
    return k


def _obs(frame, tid, side, fw, kps) -> PlayerObservation:
    cx, cy = kps[0, 0], kps[0, 1] + 45
    return PlayerObservation(frame, tid, np.array([cx - 20, cy - 60, cx + 20, cy + 55]),
                             0.9, side, np.asarray(fw, float), kps)


def _rally() -> tuple[RallyV2, list[PlayerObservation]]:
    rng = np.random.default_rng(5)
    kinds = [("serve_long", False), ("clear", True), ("smash", False), ("block", True),
             ("net", False), ("lift", True), ("drop", False)]
    shots, players = [], []
    for i, (kind, far) in enumerate(kinds):
        seg = make_segment(kind, rng, far=far)
        f = 30 * i
        seg.t0 = f / 60                       # fit epoch == contact time
        side = "far" if far else "near"
        tid = 2 if far else 1
        hit = HitEvent(frame=f, t=f / 60, xy_px=(0.0, 0.0), hitter=side, confidence=0.9)
        shots.append(ShotEventV2(hit=hit, segment=seg))
        overhead = kind in ("clear", "smash", "drop")
        # near player (id 1) is right-handed, far player (id 2) left-handed
        k = _pose(500 + 300 * far, 400, wrist="left" if far else "right", up=overhead,
                  facing_camera=far, out=(i % 2 == 0))
        players.append(_obs(f, tid, side, seg.p0[:2] + [0.3, 0.1], k))
    return RallyV2(0, 300, shots=shots), players


def test_classify_rally_end_to_end():
    rally, players = _rally()
    out = S.classify_shots(rally, None, players)
    got = [s.shot_type for s in out.shots]
    assert got == ["serve", "clear", "smash", "block", "net", "lift", "drop"]
    assert out.shots[1].stroke_side == out.shots[2].stroke_side == "overhead"
    for s in out.shots:
        assert s.hit_height_m == pytest.approx(float(s.segment.p0[2]))
        assert s.hitter_pos_world is not None and s.shot_type_confidence > 0.3
    # the input rally is not mutated
    assert all(s.shot_type == "unknown" for s in rally.shots)


def test_handedness_and_stroke_side():
    """Near player (id 1) right-handed, far player (id 2) left-handed; handedness
    must be inferred from the raised wrist on overhead contacts. Underarm shots:
    3 (far, crossed) → backhand, 4 (near, outside) → forehand, 5 (far, crossed)."""
    rally, players = _rally()
    out = S.classify_shots(rally, None, players)
    sides = [s.stroke_side for s in out.shots]
    assert sides == ["forehand", "overhead", "overhead", "backhand", "forehand",
                     "backhand", "overhead"]
    # overriding the far player as right-handed reads its (outside) right wrist;
    # the near player is unaffected
    flip = S.classify_shots(rally, None, players, handedness={2: True})
    assert [s.stroke_side for s in flip.shots] == [
        "forehand", "overhead", "overhead", "forehand", "forehand", "forehand", "overhead"]


def _skeleton_px(cam, foot_xy, facing_deg: float, hand: str, stroke: str) -> np.ndarray:
    """COCO keypoints of a 3-D skeleton projected by a real camera. `facing_deg`
    is the chest direction (0 = +x); anatomical right = facing × up."""
    f = np.array([math.cos(math.radians(facing_deg)), math.sin(math.radians(facing_deg)), 0.0])
    rt = np.cross(f, [0.0, 0.0, 1.0])
    c = np.array([foot_xy[0], foot_xy[1], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    P = np.zeros((17, 3))
    P[0] = c + 1.62 * up + 0.08 * f
    P[1:5] = P[0]
    for i, s in ((5, -1), (6, 1)):
        P[i] = c + 1.42 * up + s * 0.2 * rt          # shoulders
        P[i + 2] = c + 1.15 * up + s * 0.25 * rt     # elbows
        P[i + 4] = c + 0.95 * up + s * 0.25 * rt     # wrists
        P[i + 6] = c + 0.95 * up + s * 0.12 * rt     # hips
        P[i + 8] = c + 0.5 * up + s * 0.14 * rt      # knees
        P[i + 10] = c + s * 0.15 * rt                # ankles
    arm = rt if hand == "R" else -rt
    P[10 if hand == "R" else 9] = (c + 1.0 * up + 0.55 * arm + 0.3 * f if stroke == "FH"
                                   else c + 1.1 * up - 0.45 * arm + 0.3 * f)
    k = np.full((17, 3), 0.9)
    k[:, :2] = cam.project(P)
    return k


def test_stroke_side_projected_skeletons():
    """First-principles geometry: near player (back to camera) and far player
    (facing it, image mirrored), both hands, both sides, turned bodies."""
    from app.services.tracking.camera import CameraModel
    cam = CameraModel.look_at(np.array([-12.0, 0, 6.0]), np.array([0, 0, 0.5]), 1400.0,
                              (1920, 1080))
    cases = [((-4.0, 0.5), a) for a in (0, 30, -30, 150, 180)] + \
            [((4.0, -0.5), a) for a in (180, 150, 210, 0, 30)] + [((5.0, 2.0), 180)]
    for foot, facing in cases:
        for hand in "RL":
            for st, exp in (("FH", "forehand"), ("BH", "backhand")):
                k = _skeleton_px(cam, foot, facing, hand, st)
                assert S.stroke_side(k, 0.0, hand == "R") == exp, (foot, facing, hand, st)
    # side-on (lateral axis along the viewing ray) is not observable → no guess
    for foot in ((-4.0, 0.0), (4.0, 0.0)):
        for facing in (90, 270):
            for hand in "RL":
                k = _skeleton_px(cam, foot, facing, hand, "FH")
                assert S.stroke_side(k, 0.0, hand == "R") == "unknown"


def test_stroke_side_ambiguous_or_missing():
    k = _pose(100, 100, up=False)
    assert S.stroke_side(k, 0.9, True) == "overhead"
    assert S.stroke_side(None, 0.1, True) == "unknown"
    assert S.stroke_side(k, 0.1, None) == "unknown"
    mid = k.copy()
    mid[10, :2] = mid[[5, 6, 11, 12], :2].mean(0)          # wrist on the midline
    assert S.stroke_side(mid, 0.1, True) == "unknown"
    low = k.copy()
    low[10, 2] = 0.1                                        # racket wrist not detected
    assert S.stroke_side(low, 0.1, True) == "unknown"


def test_pose_overhead_detection():
    assert S.wrist_above_head(_pose(100, 100, up=True))[0] > 0.9
    assert S.wrist_above_head(_pose(100, 100, up=False))[0] < 0.1
    assert S.wrist_above_head(None)[0] == 0.5


def test_fallback_2d_low_confidence_and_deterministic():
    rally, players = _rally()
    for s in rally.shots:
        s.segment = None
    a = S.classify_shots(rally, None, players)
    b = S.classify_shots(rally, None, players)
    assert [s.shot_type for s in a.shots] == [s.shot_type for s in b.shots]
    assert a.shots[0].shot_type == "serve"
    assert all(s.shot_type_confidence <= 0.45 for s in a.shots)
    assert all(s.hit_height_m is None for s in a.shots)


def _late(p0, v0, lag_s: float) -> ShotSegment3D:
    """Segment whose fit epoch is `lag_s` after the contact at t=0."""
    P, V = simulate_state(np.asarray(p0, float), np.asarray(v0, float), np.array([0.0, lag_s]))
    s = _seg(P[-1], V[-1])
    s.t0 = lag_s
    return s


def test_late_first_observation_backpropagated():
    """Fits start at the first observed frame, typically 1–3 frames after the
    contact. A 300 km/h smash is at ~140 km/h and 3 m further on two frames
    (30 fps) later; the classifier must evaluate the state at the hit time."""
    late = _late([-5.5, 0.3, 2.8], _v(300, -10), 2 / 30)
    assert late.speed0_kmh < 160 and abs(late.p0[0]) < 3.0       # looks like a kill
    assert S.classify_segment(late, t_hit=0.0)[0] == "smash"
    p, v = S.contact_state(late, 0.0)
    assert p == pytest.approx([-5.5, 0.3, 2.8], abs=0.01)
    assert np.linalg.norm(v) * 3.6 == pytest.approx(300, rel=0.01)
    # contact after the fit epoch, or implausibly far back: left unchanged
    assert np.array_equal(S.contact_state(late, 1.0)[0], late.p0)
    assert np.array_equal(S.contact_state(late, -5.0)[0], late.p0)
    # end to end: hit height is the contact height, not the first-observed one
    hit = HitEvent(frame=0, t=0.0, xy_px=(0.0, 0.0), hitter="near", confidence=0.9)
    out = S.classify_shots(RallyV2(0, 60, shots=[
        ShotEventV2(HitEvent(-60, -1.0, (0.0, 0.0), "far"), None),
        ShotEventV2(hit, late)]), None, None)
    assert out.shots[1].shot_type == "smash"
    assert out.shots[1].hit_height_m == pytest.approx(2.8, abs=0.01)
    assert out.shots[1].hitter_pos_world == pytest.approx((-5.5, 0.3), abs=0.01)


def test_backprop_stops_before_blow_up():
    """Backward drag integration diverges in finite time; must stay plausible."""
    s = _seg([-5.0, 0.0, 2.0], _v(200, -5))
    s.t0 = 0.25
    p, v = S.contact_state(s, 0.0)
    assert np.linalg.norm(v) <= 500 / 3.6 + 1e-6 and np.all(np.isfinite(p))


def test_serve_only_when_serve_like():
    """First detected hit that is clearly not a serve (serve missed) keeps its
    flight label; later shots are never serves; hits are sorted by frame."""
    rng = np.random.default_rng(9)
    smash = make_segment("smash", rng)
    lab = S.classify_segment(smash, None, is_first=True)[0]
    assert lab == "smash"
    low = make_segment("serve_short", rng)
    assert S.classify_segment(low, None, is_first=False)[0] != "serve"
    a = ShotEventV2(HitEvent(100, 100 / 60, (0.0, 0.0), "near"),
                    replace(make_segment("clear", rng), t0=100 / 60))
    b = ShotEventV2(HitEvent(10, 10 / 60, (0.0, 0.0), "near"),
                    replace(make_segment("serve_long", rng), t0=10 / 60))
    out = S.classify_shots(RallyV2(0, 200, shots=[a, b]), None, None)
    assert [s.hit.frame for s in out.shots] == [10, 100]
    assert [s.shot_type for s in out.shots] == ["serve", "clear"]


def _fallback_rally() -> tuple[RallyV2, list[PlayerObservation]]:
    """No 3-D segments; hitter feet, pose and timing only: serve → far clear from
    the baseline (overhead) → near drop from the baseline → far net shot → near."""
    spec = [  # (t, side, foot, overhead)
        (0.0, "near", (-3.0, 0.5), False),
        (1.2, "far", (5.8, -0.5), True),
        (2.7, "near", (-5.9, 1.0), True),
        (3.9, "far", (1.6, -0.3), False),
        (4.8, "near", (-1.5, 0.4), False),
    ]
    shots, players = [], []
    for t, side, foot, oh in spec:
        f = int(t * 60)
        tid = 2 if side == "far" else 1
        k = _pose(500, 400, wrist="right", up=oh, facing_camera=(side == "far"))
        players.append(_obs(f, tid, side, foot, k))
        shots.append(ShotEventV2(HitEvent(f, t, (500.0, 400.0), side, 0.9), None))
    return RallyV2(0, 300, shots=shots), players


def test_fallback_2d_plausible_labels():
    rally, players = _fallback_rally()
    out = S.classify_shots(rally, None, players)
    assert [s.shot_type for s in out.shots][:4] == ["serve", "clear", "drop", "net"]
    assert all(s.shot_type_confidence <= 0.45 for s in out.shots)
    assert out.shots[1].hitter_pos_world == pytest.approx((5.8, -0.5))
    assert out.shots[1].stroke_side == "overhead"


def test_deterministic_and_fast():
    rally, players = _rally()
    a = S.classify_shots(rally, None, players)
    b = S.classify_shots(rally, None, players)
    assert [(s.shot_type, s.shot_type_confidence, s.stroke_side) for s in a.shots] == \
           [(s.shot_type, s.shot_type_confidence, s.stroke_side) for s in b.shots]
    big = RallyV2(0, 10_000, shots=[_shift(s, 210 * j) for j in range(5) for s in rally.shots])
    t = time.perf_counter()
    S.classify_shots(big, None, players)
    per_shot = (time.perf_counter() - t) / len(big.shots)
    print(f"classify_shots: {per_shot * 1e3:.2f} ms/shot")
    assert per_shot < 0.05                  # generous: shared, loaded CI machine


def _shift(s: ShotEventV2, frames: int) -> ShotEventV2:
    f = s.hit.frame + frames
    seg = replace(s.segment, t0=s.segment.t0 + frames / 60) if s.segment is not None else None
    return replace(s, hit=replace(s.hit, frame=f, t=f / 60), segment=seg)


def test_empty_and_missing():
    assert S.classify_shots(RallyV2(0, 0), None, None).shots == []
    hit = HitEvent(frame=0, t=0.0, xy_px=(1.0, 1.0))
    out = S.classify_shots(RallyV2(0, 5, shots=[ShotEventV2(hit, None)]), None, None)
    assert out.shots[0].shot_type == "serve"


def test_r3_thresholds_long_push_and_fast_net():
    """DELIVERY_R3 proposal, adopted after a paired-bootstrap check (accuracy2.log, task 4):
    push lands up to 6.2 m from the net (ShuttleSet push p90 6.12 m; 5.0 m rejected most real
    pushes), net shots up to 70 km/h (cross-court net p90 66 km/h). R3 club push flights #0 / #1
    (landing 5.44 / 5.83 m) are pushes now; an otherwise identical slow net shot stays 'net'
    when launched at 60 km/h."""
    for p0, v0 in (([-1.713, -1.76, 1.516], [16.983, -1.835, 4.316]),
                   ([1.79, 0.763, 1.569], [-17.237, 3.958, 6.364])):
        f = S.flight_features(np.asarray(p0), np.asarray(v0), DRAG_LENGTH)
        assert 5.0 < f.landing_depth < 6.2
        assert S.classify_segment(_seg(p0, v0))[0] == "push"
    # a front-court net shot (low contact, apex 1.7 m, lands 1.6 m past the net) at 58 / 64 km/h:
    # still 'net' (the old 50 km/h bound scored it 0.5x / 0.2x of its slow twin)
    base = S.FlightFeatures(contact_height=0.6, contact_depth=1.2, speed_kmh=45.0, elevation_deg=20.0, apex=1.7,
                            net_clearance=0.2, landing_depth=1.6, descent_deg=40.0, flight_time=0.8)
    slow = S._scores_3d(base, 0.0)["net"]
    for kmh in (58.0, 64.0):
        sc = S._scores_3d(replace(base, speed_kmh=kmh), 0.0)
        assert S._normalise(sc)[0] == "net" and sc["net"] > 0.6 * slow
