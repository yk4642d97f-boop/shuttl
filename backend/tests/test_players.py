"""Player tracking tests: synthetic YOLO-like detections from a known camera.

No weights needed except the optional integration test (skipped if absent)."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from app.services.tracking import court_model as cm
from app.services.tracking import players as P
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import VideoMeta

FPS = 30.0
W, H = 1280, 720


def _camera() -> CameraModel:
    # elevated broadcast-style view from behind the near baseline
    return CameraModel.look_at(eye=np.array([-24.0, 0.0, 8.0]), target=np.array([0.5, 0.0, 0.0]),
                               focal_px=1900.0, image_size=(W, H))


# ── synthetic people ─────────────────────────────────────────────────────────

def _person(cam: CameraModel, x: float, y: float, z0: float = 0.0, sitting: bool = False,
            height: float = 1.75, rng: np.random.Generator | None = None,
            kp_noise: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """bbox (4,) and COCO keypoints (17,3) of a person with feet at (x, y, z0)."""
    s = height / 1.75
    if sitting:   # seated: hips 0.5 m high, knees forward, head at 1.25 m
        pts = {0: (0.25, 0, 1.25), 5: (0.2, 0.2, 1.05), 6: (0.2, -0.2, 1.05), 11: (0.1, 0.15, 0.5),
               12: (0.1, -0.15, 0.5), 13: (-0.35, 0.15, 0.5), 14: (-0.35, -0.15, 0.5),
               15: (-0.4, 0.15, 0.08), 16: (-0.4, -0.15, 0.08)}
    else:
        pts = {0: (0, 0, 1.62 * s), 5: (0, 0.2, 1.45 * s), 6: (0, -0.2, 1.45 * s),
               11: (0, 0.13, 0.95 * s), 12: (0, -0.13, 0.95 * s), 13: (0, 0.14, 0.5 * s),
               14: (0, -0.14, 0.5 * s), 15: (0, 0.14, 0.08), 16: (0, -0.14, 0.08)}
    pts.update({1: pts[0], 2: pts[0], 3: pts[0], 4: pts[0], 7: pts[5], 8: pts[6],
                9: pts[11], 10: pts[12]})
    X = np.array([[x + pts[i][0], y + pts[i][1], z0 + pts[i][2]] for i in range(17)])
    uv = cam.project(X)
    if rng is not None:
        uv = uv + rng.normal(0, kp_noise, uv.shape)
    kps = np.column_stack([uv, np.full(17, 0.9)])
    top = cam.project(np.array([[x, y, z0 + (1.3 if sitting else 1.78 * s)]]))[0]
    bot = cam.project(np.array([[x, y, z0]]))[0]
    hpx = bot[1] - top[1]
    box = np.array([bot[0] - 0.22 * hpx, top[1], bot[0] + 0.22 * hpx, bot[1] + 0.01 * hpx])
    return box, kps


def _officials(cam: CameraModel) -> list[tuple[np.ndarray, np.ndarray]]:
    """Umpire on a raised chair, service judge, 4 line judges, 3 spectators."""
    out = [_person(cam, 0.0, -cm.DOUBLES_HALF_WIDTH - 1.0, z0=1.3, sitting=True),     # umpire
           _person(cam, 0.6, cm.DOUBLES_HALF_WIDTH + 0.9, sitting=True)]              # service judge
    for x, y in ((-cm.HALF_LENGTH - 2.2, 2.2), (cm.HALF_LENGTH + 2.2, -2.0),
                 (-3.5, cm.DOUBLES_HALF_WIDTH + 1.3), (3.5, -cm.DOUBLES_HALF_WIDTH - 1.3)):
        out.append(_person(cam, x, y, sitting=True))                                   # line judges
    for x, y in ((cm.HALF_LENGTH + 5.0, 1.0), (cm.HALF_LENGTH + 6.0, -3.0), (-9.5, -4.5)):
        out.append(_person(cam, x, y))                                                 # spectators
    return out


def _trajectory(rng: np.random.Generator, n: int, x_lo: float, x_hi: float,
                y_lo: float, y_hi: float) -> np.ndarray:
    """Smooth random footwork (≤ ~5 m/s) inside a box, (n,2)."""
    knots = max(3, n // 20)
    kx = rng.uniform(x_lo, x_hi, knots)
    ky = rng.uniform(y_lo, y_hi, knots)
    t = np.linspace(0, knots - 1, n)
    xy = np.column_stack([np.interp(t, np.arange(knots), kx), np.interp(t, np.arange(knots), ky)])
    k = np.ones(9) / 9
    return np.column_stack([np.convolve(np.pad(xy[:, i], 4, mode="edge"), k, "valid") for i in range(2)])


def _scene(n: int = 240, doubles: bool = False, seed: int = 0, drop: float = 0.08,
           officials: bool = True, cam: CameraModel | None = None):
    """GT trajectories {pid: (n,2)} and per-frame raw detections."""
    rng = np.random.default_rng(seed)
    cam = cam or _camera()
    if doubles:
        gt = {"n_left": _trajectory(rng, n, -6.0, -1.0, 0.3, 2.8),
              "n_right": _trajectory(rng, n, -6.0, -1.0, -2.8, -0.3),
              "f_left": _trajectory(rng, n, 1.0, 6.0, 0.3, 2.8),
              "f_right": _trajectory(rng, n, 1.0, 6.0, -2.8, -0.3)}
    else:
        gt = {"near": _trajectory(rng, n, -6.2, -1.2, -2.3, 2.3),
              "far": _trajectory(rng, n, 1.2, 6.2, -2.3, 2.3)}
    static = _officials(cam) if officials else []
    dets = []
    for f in range(n):
        boxes, scores, kps = [], [], []
        for traj in gt.values():
            if rng.random() < drop:
                continue
            b, k = _person(cam, *traj[f], rng=rng)
            boxes.append(b), scores.append(rng.uniform(0.6, 0.95)), kps.append(k)
        for b, k in static:
            boxes.append(b + rng.normal(0, 0.5, 4)), scores.append(0.8), kps.append(k)
        dets.append((np.array(boxes).reshape(-1, 4), np.array(scores), np.array(kps).reshape(-1, 17, 3)))
    return cam, gt, dets


def _meta(n: int) -> VideoMeta:
    return VideoMeta(fps=FPS, width=W, height=H, n_frames=n, timestamps=np.arange(n) / FPS)


def _assemble(cam, dets, n, step=2, doubles=None):
    frames = [P.build_candidates(f, dets[f], cam, (W, H)) for f in range(0, n, step)]
    return P.assemble_tracks(frames, _meta(n), cam, doubles)


def _by_track(obs):
    out: dict[int, list] = {}
    for o in obs:
        out.setdefault(o.track_id, []).append(o)
    return out


# ── foot point / gating ──────────────────────────────────────────────────────

def test_foot_point_prefers_ankles_then_bbox_bottom():
    cam = _camera()
    box, kps = _person(cam, -3.0, 1.0)
    foot, ank = P.foot_pixels(box[None], kps[None])
    assert ank[0] and np.allclose(foot[0], (kps[15, :2] + kps[16, :2]) / 2)
    kps2 = kps.copy()
    kps2[15, 2] = 0.1                              # one ankle occluded → bbox bottom
    foot2, ank2 = P.foot_pixels(box[None], kps2[None])
    assert not ank2[0] and foot2[0, 1] == pytest.approx(box[3])
    pos = P.floor_positions(cam, foot, ank)
    assert np.allclose(pos[0], (-3.0, 1.0), atol=0.02)


def test_gating_rejects_officials_and_spectators():
    cam = _camera()
    people = _officials(cam)
    boxes = np.array([b for b, _ in people])
    kps = np.array([k for _, k in people])
    c = P.build_candidates(0, (boxes, np.full(len(people), 0.9), kps), cam, (W, H))
    # every non-player is either gated out or costs more than a missed frame
    assert np.all(c.unary > P.C_MISS)
    b, k = _person(cam, -6.5, 2.9)                # player at the back corner, on the line
    b2, k2 = _person(cam, -7.4, 0.0)              # retrieving a shuttle behind the baseline
    c = P.build_candidates(0, (np.array([b, b2]), np.array([0.9, 0.9]), np.array([k, k2])), cam, (W, H))
    assert c.n == 2 and np.all(c.unary < 1.5)


def test_umpire_on_chair_is_not_a_player():
    cam = _camera()
    b, k = _person(cam, 0.0, -cm.DOUBLES_HALF_WIDTH - 1.0, z0=1.3, sitting=True)
    c = P.build_candidates(0, (b[None], np.array([0.95]), k[None]), cam, (W, H))
    assert c.n == 0 or c.unary[0] > P.C_MISS


# ── identity / sides ─────────────────────────────────────────────────────────

def _check_identity(obs, gt, id_of: dict[str, int], tol: float = 0.35, min_cov: float = 0.9):
    tracks = _by_track(obs)
    assert set(tracks) == set(id_of.values())
    n = len(next(iter(gt.values())))
    for name, tid in id_of.items():
        o = tracks[tid]
        frames = np.array([x.frame for x in o])
        assert len(set(frames)) == len(frames)
        assert len(o) >= min_cov * n
        err = np.array([np.linalg.norm(x.foot_world - gt[name][x.frame]) for x in o])
        assert np.median(err) < 0.12, (name, np.median(err))
        assert np.percentile(err, 99) < tol, (name, np.percentile(err, 99))
        side = {x.side for x in o}
        assert side == {"near" if name.startswith("n") else "far"}


def test_singles_exactly_two_consistent_ids():
    cam, gt, dets = _scene(n=240, seed=1)
    obs = _assemble(cam, dets, 240)
    assert obs.doubles is False
    _check_identity(obs, gt, {"near": 1, "far": 2})
    # keypoints kept (17,3) and every frame (skipped ones interpolated)
    assert all(o.keypoints.shape == (17, 3) for o in obs)
    assert {o.frame for o in obs if o.track_id == 1} >= set(range(10, 230))


def test_doubles_four_ids_left_right_order():
    cam, gt, dets = _scene(n=240, doubles=True, seed=2, drop=0.05)
    obs = _assemble(cam, dets, 240)
    assert obs.doubles is True                      # auto-detected
    # ids: near 1 = smaller world y (right side of court seen from the camera), 3; far 2, 4
    _check_identity(obs, gt, {"n_right": 1, "n_left": 3, "f_right": 2, "f_left": 4})


def test_player_near_net_keeps_side():
    """Near player lunges to the net (foot noise pushes x above 0); far player close by."""
    cam = _camera()
    rng = np.random.default_rng(5)
    n = 120
    near = np.column_stack([np.linspace(-4.0, -0.1, n), np.linspace(1.0, 0.5, n)])
    far = np.column_stack([np.linspace(3.0, 0.6, n), np.linspace(-1.0, -0.3, n)])
    dets = []
    for f in range(n):
        bs, ss, ks = [], [], []
        for traj in (near, far):
            b, k = _person(cam, *traj[f], rng=rng, kp_noise=3.0)
            bs.append(b), ss.append(0.9), ks.append(k)
        dets.append((np.array(bs), np.array(ss), np.array(ks)))
    obs = _assemble(cam, dets, n, step=1, doubles=False)
    tr = _by_track(obs)
    assert set(tr) == {1, 2}
    assert all(o.side == "near" for o in tr[1]) and all(o.side == "far" for o in tr[2])
    e1 = [np.linalg.norm(o.foot_world - near[o.frame]) for o in tr[1]]
    assert max(e1) < 0.4


def test_missing_player_is_not_replaced_by_line_judge():
    """Far player undetected for 0.8 s: no id jumps to the line judge / umpire."""
    cam, gt, dets = _scene(n=150, seed=3, drop=0.0)
    far_box = [P.foot_pixels(d[0][1:2], d[2][1:2])[0][0] for d in dets]
    for f in range(60, 84):                          # remove the far player's detection
        b, s, k = dets[f]
        dets[f] = (np.delete(b, 1, 0), np.delete(s, 1), np.delete(k, 1, 0))
    obs = _assemble(cam, dets, 150)
    far = [o for o in obs if o.track_id == 2]
    for o in far:
        assert np.linalg.norm(o.foot_world - gt["far"][o.frame]) < 0.6
    assert far_box                                    # sanity: scene built


def test_no_camera_keeps_two_central_people():
    cam, gt, dets = _scene(n=120, seed=4)
    frames = [P.build_candidates(f, dets[f], None, (W, H)) for f in range(0, 120, 2)]
    obs = P.assemble_tracks(frames, _meta(120), None, doubles=False)
    tr = _by_track(obs)
    assert set(tr) == {1, 2}
    near_px = np.array([cam.project(np.array([[*gt["near"][o.frame], 0.0]]))[0] for o in tr[1]])
    foot = np.array([(o.bbox[0] + o.bbox[2]) / 2 for o in tr[1]])
    assert np.median(np.abs(foot - near_px[:, 0])) < 10
    assert all(o.foot_world is None and o.side == "near" for o in tr[1])
    assert all(o.side == "far" for o in tr[2])


def _doubles_frames(cam, trajs: dict, n: int, rng, colours: dict | None = None,
                    hidden: dict | None = None, step: int = 2):
    """Candidates for 4 players; overlapping partners merge into one box (the
    detector reports one person). `colours` {name: (shirt BGR, shorts BGR)} renders
    an image so appearance descriptors are computed; `hidden` {name: frames}."""
    frames = []
    for f in range(0, n, step):
        people = []
        for name, tr in trajs.items():
            if hidden and f in hidden.get(name, ()):
                continue
            b, k = _person(cam, *tr[f], rng=rng)
            people.append((name, b, k))
        keep = []
        for i, (name, b, k) in enumerate(people):
            if any(P._iou_1n(b, np.array([people[j][1]]))[0] > 0.45 for j in keep):
                continue                                  # merged with an earlier box
            keep.append(i)
        img = None
        if colours is not None:
            img = np.full((H, W, 3), 90, np.uint8)
            for i in keep:
                name, b, k = people[i]
                shirt, shorts = colours[name]
                xs = k[[5, 6, 11, 12], 0]
                x1, x2 = int(xs.min()) - 6, int(xs.max()) + 6
                ys, yh, yk = int(k[5:7, 1].mean()), int(k[11:13, 1].mean()), int(k[13:15, 1].mean())
                cv2.rectangle(img, (x1, ys), (x2, yh), shirt, -1)
                cv2.rectangle(img, (x1, yh), (x2, yk), shorts, -1)
        dets = (np.array([people[i][1] for i in keep]).reshape(-1, 4), np.full(len(keep), 0.85),
                np.array([people[i][2] for i in keep]).reshape(-1, 17, 3))
        frames.append(P.build_candidates(f, dets, cam, (W, H), image=img))
    return frames


def test_doubles_partners_crossing_keep_identity():
    """Near partners run through each other (boxes merge at the crossing): motion
    continuity, not left/right order, must keep the ids."""
    cam, n = _camera(), 180
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, n)
    trajs = {"n_a": np.column_stack([-3.5 - 0.3 * t, 2.2 - 4.4 * t]),     # left → right
             "n_b": np.column_stack([-3.2 + 0.2 * t, -2.2 + 4.4 * t]),    # right → left
             "f_c": np.column_stack([3.0 + 0 * t, 1.5 + 0 * t]),
             "f_d": np.column_stack([3.5 + 0 * t, -1.5 + 0 * t])}
    frames = _doubles_frames(cam, trajs, n, rng)
    obs = P.assemble_tracks(frames, _meta(n), cam, doubles=True)
    _check_identity(obs, trajs, {"n_b": 1, "n_a": 3, "f_d": 2, "f_c": 4}, tol=0.45)


def test_doubles_side_switch_resolved_by_appearance():
    """Partners swap left/right while both are undetected for 1 s (e.g. rotating
    between points). Motion alone prefers 'nobody moved'; shirt colours say swap."""
    cam, n = _camera(), 180
    rng = np.random.default_rng(12)
    a = np.array([[-3.5, 0.9]] * 60 + [[-3.5, -0.9]] * 120)
    b = np.array([[-3.5, -0.9]] * 60 + [[-3.5, 0.9]] * 120)
    trajs = {"a": a, "b": b, "c": np.array([[3.0, 1.5]] * n), "d": np.array([[3.5, -1.5]] * n)}
    colours = {"a": ((30, 30, 220), (40, 40, 40)), "b": ((220, 120, 20), (235, 235, 235)),
               "c": ((40, 200, 40), (40, 40, 40)), "d": ((40, 200, 230), (40, 40, 40))}
    gap = set(range(60, 90))
    frames = _doubles_frames(cam, trajs, n, rng, colours, hidden={"a": gap, "b": gap})
    obs = P.assemble_tracks(frames, _meta(n), cam, doubles=True)
    tr = _by_track(obs)
    assert {1, 3} <= set(tr)
    for tid, name in ((1, "b"), (3, "a")):            # b starts at y<0 → id 1
        err = [np.linalg.norm(o.foot_world - trajs[name][o.frame]) for o in tr[tid]]
        assert np.max(err) < 0.3, (tid, name, np.max(err))
        assert min(o.frame for o in tr[tid]) <= 2 and max(o.frame for o in tr[tid]) >= n - 3
    # without images the same scene has no evidence for the swap
    plain = _doubles_frames(cam, trajs, n, np.random.default_rng(12), None,
                            hidden={"a": gap, "b": gap})
    tr0 = _by_track(P.assemble_tracks(plain, _meta(n), cam, doubles=True))
    late = [o for o in tr0[1] if o.frame >= 100]
    assert np.median([o.foot_world[1] for o in late]) < 0      # stays on its own side


def test_sitting_score_standing_seated_lunge():
    cam = _camera()
    _, stand = _person(cam, -3.0, 0.0)
    _, sit = _person(cam, 0.6, cm.DOUBLES_HALF_WIDTH + 0.9, sitting=True)
    lunge = stand.copy()
    lunge[13, :2] = lunge[11, :2] + (lunge[13, :2] - lunge[11, :2]) * np.array([1.0, 0.35])
    s = P.sitting_score(np.array([stand, sit, lunge]))
    assert s[0] == 0.0 and s[1] > 0.9 and s[2] == 0.0


def test_appearance_descriptor_separates_colours():
    cam = _camera()
    b, k = _person(cam, -3.0, 0.0)
    imgs = []
    for shirt in ((30, 30, 220), (32, 34, 215), (220, 120, 20)):
        img = np.full((H, W, 3), 90, np.uint8)
        xs = k[[5, 6, 11, 12], 0]
        cv2.rectangle(img, (int(xs.min()) - 6, int(k[5, 1])), (int(xs.max()) + 6, int(k[11, 1])),
                      shirt, -1)
        imgs.append(P.appearance_descriptors(img, b[None], k[None])[0])
    assert P.appearance_distance(imgs[0], imgs[1]) < 0.1
    assert P.appearance_distance(imgs[0], imgs[2]) > 0.4
    assert P.appearance_distance(imgs[0], None) == 0.0


# ── smoothing / stats ────────────────────────────────────────────────────────

def test_rts_smoother_reduces_noise_and_fills_gaps():
    rng = np.random.default_rng(0)
    t = np.arange(90) / 30.0
    truth = np.column_stack([1.5 * np.sin(2 * t), 2.0 * t - 1.0])
    z = truth + rng.normal(0, 0.1, truth.shape)
    z[40:50] = np.nan                                 # measurement gap
    sm = P.rts_smooth(t, z, np.full(len(t), 0.1))
    ok = np.isfinite(z[:, 0])
    raw = np.sqrt(np.nanmean(np.sum((z[ok] - truth[ok]) ** 2, 1)))
    rms = np.sqrt(np.mean(np.sum((sm - truth) ** 2, 1)))
    assert rms < 0.6 * raw
    assert np.all(np.linalg.norm(sm[40:50] - truth[40:50], axis=1) < 0.2)


def test_movement_stats_known_path():
    from app.services.tracking.data_types import PlayerObservation
    n = 150                                           # 5 s at 30 fps, 2 m/s along y
    obs = [PlayerObservation(frame=f, track_id=1, bbox=np.zeros(4), score=0.9, side="near",
                             foot_world=np.array([-3.0, -2.5 + 2.0 * f / FPS]))
           for f in range(n)]
    obs += [PlayerObservation(frame=f, track_id=2, bbox=np.zeros(4), score=0.9, side="far",
                              foot_world=np.array([3.0, 0.0])) for f in range(n)]
    st = P.movement_stats(obs, "near", fps=FPS)
    assert st["distance_m"] == pytest.approx(2.0 * (n - 1) / FPS, rel=0.02)
    assert st["avg_speed_ms"] == pytest.approx(2.0, rel=0.03)
    assert st["max_speed_ms"] == pytest.approx(2.0, rel=0.05)
    heat = np.array(st["heatmap"])
    assert heat.shape == (P.HEAT_ROWS, P.HEAT_COLS) and heat.sum() == pytest.approx(1.0, abs=1e-3)
    row = int(3.0 / cm.HALF_LENGTH * P.HEAT_ROWS)
    assert heat[row].sum() == pytest.approx(1.0, abs=1e-3)   # all time at 3 m from the net
    assert 0 < st["court_coverage"] <= 1
    far = P.movement_stats(obs, "far", fps=FPS)
    assert far["distance_m"] == 0.0 and far["court_coverage"] == pytest.approx(1 / 120, abs=1e-3)
    empty = P.movement_stats([], "near")
    assert empty["distance_m"] == 0.0 and len(empty["heatmap"]) == P.HEAT_ROWS


def test_movement_stats_uses_list_timestamps():
    cam, gt, dets = _scene(n=120, seed=6)
    obs = _assemble(cam, dets, 120)
    st = P.movement_stats(obs, "near")
    assert 0.5 < st["tracked_s"] <= 4.0 and st["distance_m"] > 0
    assert st["max_speed_ms"] < P.SPEED_CAP


def test_far_crop_merge_drops_people_cut_by_the_crop():
    """Crop detections replace full-frame ones, except persons truncated by an
    inner crop border (a near player at the net reaching into the far ROI)."""
    kp = np.zeros((1, 17, 3))
    roi = (100, 50, 900, 300)
    full = (np.array([[400.0, 120, 440, 220], [500.0, 200, 580, 420]]), np.array([0.6, 0.9]),
            np.concatenate([kp, kp]))
    far_c = np.array([[300.0, 69, 341, 171]])             # crop px → frame (400,119,441,221)
    cut_c = np.array([[398.0, 150, 481, 250]])            # bottom = crop bottom edge
    crop = (np.concatenate([far_c, cut_c]), np.array([0.8, 0.7]), np.concatenate([kp, kp]))
    b, s, k = P._merge_roi(full, crop, roi, (1280, 720))
    assert len(b) == 2 and k.shape == (2, 17, 3)
    assert np.allclose(sorted(s), [0.8, 0.9])             # far box from the crop, near kept
    near = b[np.argmax(b[:, 3])]
    assert np.allclose(near, [500, 200, 580, 420])


# ── end to end with a mocked detector ────────────────────────────────────────

def test_track_players_streams_video_with_mock_detector(tmp_path):
    n = 40
    cam, gt, dets = _scene(n=n, seed=7)
    path = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(n):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(img, str(f), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        vw.write(img)
    vw.release()
    seen: list[int] = []

    def fake_detector(frames, idx):
        assert all(f.shape == (H, W, 3) for f in frames)
        seen.extend(idx)
        return [dets[i] for i in idx]

    prog: list[float] = []
    obs = P.track_players(path, _meta(n), cam, step=2, doubles=False, progress=prog.append,
                          detector=fake_detector, batch=4)
    assert seen == list(range(0, n, 2))
    assert prog and prog[-1] == pytest.approx(1.0, abs=0.05)
    assert {o.track_id for o in obs} == {1, 2}
    assert obs == sorted(obs, key=lambda o: (o.frame, o.track_id))


# ── optional: real YOLO pose model ───────────────────────────────────────────

_W8 = os.path.join(P.WEIGHTS_DIR, P.DEFAULT_MODEL)


@pytest.mark.skipif(not os.path.exists(_W8), reason="pose weights not present")
def test_real_pose_model_on_bundled_image():
    ultra = pytest.importorskip("ultralytics")
    img_path = os.path.join(os.path.dirname(ultra.__file__), "assets", "bus.jpg")
    if not os.path.exists(img_path):
        pytest.skip("ultralytics sample image missing")
    img = cv2.imread(img_path)
    det = P.make_yolo_detector(device="cpu", imgsz=640)
    boxes, scores, kps = det([img], [0])[0]
    assert len(boxes) >= 3 and kps.shape[1:] == (17, 3)
    foot, ank = P.foot_pixels(boxes, kps)
    assert ank.sum() >= 2                              # standing people: ankles found
    assert np.all(foot[:, 1] > boxes[:, 1] + 0.7 * (boxes[:, 3] - boxes[:, 1]))
