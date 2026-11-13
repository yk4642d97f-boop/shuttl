"""Tests for court_calibration: synthetic renders with known cameras.

The renderer draws every 40 mm court line with exact perspective width
(analytic pixel-footprint coverage), textured floors, net (mesh, tape,
posts), a wall with a white skirting line and vertical white edges, lights,
moving white-shirt players, blur, noise and JPEG artefacts.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from app.services.tracking import court_calibration as cc
from app.services.tracking import court_model as cm
from app.services.tracking.camera import CameraModel

SIZE = (1920, 1080)
_L, _W, _HW = cm.HALF_LENGTH, cm.DOUBLES_HALF_WIDTH, cm.LINE_WIDTH / 2


# ----------------------------------------------------------------------------- renderer

def _band_cov(lo: np.ndarray, hi: np.ndarray, a: float, b: float) -> np.ndarray:
    """Fraction of the pixel footprint [lo, hi] covered by the band [a, b]."""
    ov = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
    return ov / np.maximum(hi - lo, 1e-6)


def _floor_coords(cam: CameraModel) -> tuple[np.ndarray, ...]:
    """World floor (X, Y) at pixel corners → per-pixel centre, footprint min/max, valid."""
    w, h = cam.image_size
    Hm = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])
    Hi = np.linalg.inv(Hm)
    U, V = np.meshgrid(np.arange(w + 1, dtype=np.float64) - 0.5,
                       np.arange(h + 1, dtype=np.float64) - 0.5)
    p = Hi[:, 0, None, None] * U + Hi[:, 1, None, None] * V + Hi[:, 2, None, None]
    ok = p[2] > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(ok, p[0] / p[2], np.nan)
        Y = np.where(ok, p[1] / p[2], np.nan)
    corners_x = np.stack([X[:-1, :-1], X[1:, :-1], X[:-1, 1:], X[1:, 1:]])
    corners_y = np.stack([Y[:-1, :-1], Y[1:, :-1], Y[:-1, 1:], Y[1:, 1:]])
    valid = np.all(np.isfinite(corners_x), 0)
    with np.errstate(invalid="ignore"):
        xc, yc = np.nanmean(corners_x, 0), np.nanmean(corners_y, 0)
        return (xc.astype(np.float32), yc.astype(np.float32),
                np.nanmin(corners_x, 0).astype(np.float32), np.nanmax(corners_x, 0).astype(np.float32),
                np.nanmin(corners_y, 0).astype(np.float32), np.nanmax(corners_y, 0).astype(np.float32),
                valid & (np.nanmax(np.abs(corners_x), 0) < 200))


def _poly(img: np.ndarray, cam: CameraModel, P: np.ndarray, color, alpha: float = 1.0) -> None:
    uv = cam.project(P)
    if not np.all(np.isfinite(uv)):
        return
    pts = np.round(uv * 16).astype(np.int32)
    if alpha >= 1:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA, shift=4)
    else:
        over = img.copy()
        cv2.fillPoly(over, [pts], color, cv2.LINE_AA, shift=4)
        cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)


def render_background(cam: CameraModel, rng: np.random.Generator, style: str = "green",
                      distractors: bool = True, lines: dict | None = None,
                      line_colour: tuple | None = None) -> np.ndarray:
    """Static scene (court, floor, net, wall) as float32 BGR. `line_colour` (BGR): paint the
    court lines in that colour instead of white (same random draws either way)."""
    w, h = cam.image_size
    xc, yc, x0, x1, y0, y1, valid = _floor_coords(cam)
    img = np.empty((h, w, 3), np.float32)
    wall = np.array(rng.choice([[150, 160, 170], [90, 80, 70], [60, 60, 120]]), np.float32)
    img[:] = wall * (0.85 + 0.3 * np.linspace(0, 1, h, dtype=np.float32))[:, None, None]
    hall_x, hall_y = _L + rng.uniform(3, 9), _W + rng.uniform(2, 7)
    floor = valid & (np.abs(xc) < hall_x) & (np.abs(yc) < hall_y)
    if style == "wood":
        plank = np.floor(np.nan_to_num(yc) / 0.12).astype(np.int64)
        tone = (np.sin(plank * 12.9898) * 43758.5453) % 1.0
        grain = np.sin(np.nan_to_num(xc) * 23.0 + plank) * 0.5 + 0.5
        base = np.array([60, 120, 175], np.float32)
        tex = base[None, None] * (0.8 + 0.25 * tone[..., None] + 0.08 * grain[..., None])
        surround = tex
    else:
        base = np.array([70, 140, 60] if style == "green" else [165, 105, 40], np.float32)
        tex = base[None, None] * (0.93 + 0.1 * rng.random((h, w, 1), dtype=np.float32))
        surround = np.array(rng.choice([[170, 110, 40], [120, 120, 120], [60, 60, 170]]),
                            np.float32)[None, None] * np.ones((h, w, 1), np.float32)
    mat = floor & (np.abs(xc) < _L + 2.0) & (np.abs(yc) < _W + 1.5)
    img[floor] = surround[floor]
    img[mat] = tex[mat]
    # court lines (exact 40 mm width in perspective)
    cov = np.zeros((h, w), np.float32)
    for (a, b) in (lines or cm.COURT_LINES).values():
        if abs(a[0] - b[0]) < 1e-9:     # transverse: X = const
            c = _band_cov(x0, x1, a[0] - _HW, a[0] + _HW) * \
                _band_cov(y0, y1, min(a[1], b[1]) - _HW, max(a[1], b[1]) + _HW)
        else:
            c = _band_cov(y0, y1, a[1] - _HW, a[1] + _HW) * \
                _band_cov(x0, x1, min(a[0], b[0]) - _HW, max(a[0], b[0]) + _HW)
        np.maximum(cov, np.where(valid, c, 0), out=cov)
    white = np.float32(rng.uniform(215, 250))
    paint = white if line_colour is None else np.array(line_colour, np.float32) * (white / 235)
    img = img * (1 - cov[..., None]) + paint * cov[..., None]
    if distractors:   # white skirting line where floor meets the far wall
        sk = _band_cov(x0, x1, hall_x - 0.12, hall_x) * floor
        img = img * (1 - sk[..., None]) + 235 * sk[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    # net: mesh, tape (top sags from 1.55 at posts to 1.524 at centre), posts
    ys = np.linspace(-_W, _W, 25)
    top = cm.NET_HEIGHT_CENTRE + (cm.NET_HEIGHT_POST - cm.NET_HEIGHT_CENTRE) * (ys / _W) ** 2
    zeros = np.zeros_like(ys)
    mesh = np.vstack([np.column_stack([zeros, ys, top]),
                      np.column_stack([zeros, ys[::-1], np.full_like(ys, 0.79)])])
    _poly(img, cam, mesh, (30, 30, 30), alpha=0.35)
    tape = np.vstack([np.column_stack([zeros, ys, top]),
                      np.column_stack([zeros, ys[::-1], top[::-1] - 0.038])])
    _poly(img, cam, tape, (240, 240, 240))
    for s in (-1, 1):
        post = np.array([[0.02, s * _W - 0.02, 0], [0.02, s * _W + 0.02, 0],
                         [0.02, s * _W + 0.02, 1.55], [0.02, s * _W - 0.02, 1.55]])
        _poly(img, cam, post, (40, 60, 40))
    if distractors:
        sky = ~floor
        rows = np.flatnonzero(sky.any(1))
        if len(rows) > 10:
            for _ in range(rng.integers(2, 5)):     # ceiling / wall lights
                y = int(rng.uniform(0, rows.max()))
                x = int(rng.uniform(0, w))
                if sky[min(y, h - 1), x]:
                    cv2.ellipse(img, (x, y), (int(rng.uniform(15, 50)), int(rng.uniform(8, 25))),
                                0, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
            for _ in range(2):                        # white vertical wall edges
                x = int(rng.uniform(0, w))
                y_bot = np.flatnonzero(sky[:, x])
                if len(y_bot):
                    cv2.line(img, (x, 0), (x, int(y_bot.max())), (235, 235, 235), 3, cv2.LINE_AA)
    return img


def _player(img: np.ndarray, cam: CameraModel, x: float, y: float, shirt) -> None:
    """Crude player: dark shorts/legs + shirt box, 0.5 m wide."""
    right = np.array([0.0, 0.25, 0.0])
    base = np.array([x, y, 0.0])
    _poly(img, cam, np.array([base - right, base + right, base + right + [0, 0, 0.9],
                              base - right + [0, 0, 0.9]]), (40, 40, 40))
    _poly(img, cam, np.array([base - right + [0, 0, 0.9], base + right + [0, 0, 0.9],
                              base + right + [0, 0, 1.6], base - right + [0, 0, 1.6]]), shirt)


def render_frames(cam: CameraModel, rng: np.random.Generator, n: int = 5, style: str = "green",
                  distractors: bool = True, lines: dict | None = None) -> list[np.ndarray]:
    """n frames of the static scene with moving players + per-frame degradations."""
    bg = render_background(cam, rng, style, distractors, lines)
    blur = rng.uniform(0.5, 1.2)
    sigma = rng.uniform(2, 5)
    q = int(rng.integers(60, 91))
    grad = (0.85 + 0.25 * np.linspace(0, 1, bg.shape[1], dtype=np.float32))[None, :, None]
    frames = []
    for _ in range(n):
        f = bg.copy()
        if distractors:
            for half in (-1, 1):
                _player(f, cam, half * rng.uniform(1.0, 5.5), rng.uniform(-2.5, 2.5),
                        (245, 245, 245) if half < 0 else (50, 50, 200))
        f = cv2.GaussianBlur(f, (0, 0), blur).astype(np.float32) * grad
        f += rng.normal(0, sigma, f.shape).astype(np.float32)
        f = np.clip(f, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        frames.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    return frames


# ----------------------------------------------------------------------------- cameras

def behind_camera(rng: np.random.Generator, size=SIZE) -> CameraModel:
    """Phone behind the near baseline: 0.9–2.5 m high, 1–8 m back, yaw ±15°, roll ±5°."""
    w, _ = size
    f = rng.uniform(0.6, 1.2) * w
    eye = np.array([-_L - rng.uniform(1, 8), rng.uniform(-1.5, 1.5), rng.uniform(0.9, 2.5)])
    tgt = np.array([rng.uniform(-1.0, 3.0), 0.0, 0.0])
    yaw = np.radians(rng.uniform(-15, 15))
    d = tgt - eye
    c, s = np.cos(yaw), np.sin(yaw)
    d[:2] = [c * d[0] - s * d[1], s * d[0] + c * d[1]]
    return CameraModel.look_at(eye, eye + d, f, size, roll_deg=rng.uniform(-5, 5))


def broadcast_camera(rng: np.random.Generator, size=SIZE) -> CameraModel:
    """Elevated broadcast view ~(-14, 0, 6) looking at the court centre."""
    eye = np.array([-14 + rng.uniform(-3, 3), rng.uniform(-1, 1), 6 + rng.uniform(-1.5, 2)])
    return CameraModel.look_at(eye, np.array([rng.uniform(-1, 1), 0, 0]),
                               rng.uniform(0.8, 1.2) * size[0], size, rng.uniform(-2, 2))


def side_camera(rng: np.random.Generator, size=SIZE) -> CameraModel:
    """Side-ish view ~45° off the court axis."""
    a = np.radians(rng.uniform(35, 55)) * rng.choice([-1, 1])
    r = rng.uniform(11, 16)
    eye = np.array([-r * np.cos(a), -r * np.sin(a), rng.uniform(3, 7)])
    return CameraModel.look_at(eye, np.zeros(3), rng.uniform(0.6, 0.9) * size[0], size,
                               rng.uniform(-3, 3))


# ----------------------------------------------------------------------------- metrics

def calib_errors(true: CameraModel, est: CameraModel) -> dict:
    """Keypoint reprojection (px), floor back-projection errors (m) per half, focal error."""
    _, P = cm.keypoints_array()
    w, h = true.image_size
    uv = true.project(P)
    vis = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) \
        & (uv[:, 0] < w) & (uv[:, 1] < h)
    rep = np.linalg.norm(est.project(P[vis]) - uv[vis], axis=1)
    g = est.pixel_to_floor(uv[vis])
    werr = np.linalg.norm(g[:, :2] - P[vis, :2], axis=1)
    near = P[vis, 0] < 0
    return {"n_vis": int(vis.sum()),
            "reproj_med": float(np.nanmedian(rep)) if len(rep) else np.inf,
            "reproj_max": float(np.nanmax(rep)) if len(rep) else np.inf,
            "near_med": float(np.nanmedian(werr[near])) if near.any() else 0.0,
            "far_med": float(np.nanmedian(werr[~near])) if (~near).any() else 0.0,
            "world_max": float(np.nanmax(werr)) if len(werr) else np.inf,
            "focal_rel": abs(est.K[0, 0] / true.K[0, 0] - 1)}


def is_accurate(e: dict) -> bool:
    return e["reproj_med"] < 3 and e["near_med"] < 0.02 and e["far_med"] < 0.03


def is_garbage(e: dict) -> bool:
    """A returned camera that is materially wrong (what must never happen)."""
    return not (e["reproj_med"] < 8 and e["near_med"] < 0.10 and e["far_med"] < 0.25)


# ----------------------------------------------------------------------------- manual path

def _pts(cam: CameraModel, names: list[str], rng=None, noise: float = 0.0) -> dict:
    _, P = cm.keypoints_array(names)
    uv = cam.project(P)
    if rng is not None and noise:
        uv = uv + rng.uniform(-noise, noise, uv.shape)
    return {n: (float(u), float(v)) for n, (u, v) in zip(names, uv)}


CORNERS = ["nb_ld", "nb_rd", "fb_ld", "fb_rd"]


@pytest.mark.parametrize("seed", range(10))
def test_points_exact(seed: int) -> None:
    rng = np.random.default_rng(seed)
    true = [behind_camera, broadcast_camera, side_camera][seed % 3](rng)
    est = cc.calibrate_from_points(_pts(true, CORNERS), SIZE)
    e = calib_errors(true, est)
    assert e["world_max"] < 1e-3 and e["focal_rel"] < 1e-3
    assert est.confidence > 0.7 and est.reproj_rmse_px < 0.01


@pytest.mark.parametrize("seed", range(30))
def test_points_click_noise(seed: int) -> None:
    """±2 px taps on the 4 outer corners.

    Four taps determine the homography exactly, so the solver cannot beat
    the taps themselves: from a phone 1 m high and 12 m back, 1 px at the far
    baseline is 10–35 cm of floor, and a ±2 px tap alone moves the far corner
    up to 60 cm (the 5 cm target of the brief is geometrically impossible
    there; it is met by taps + refine_camera, tested below). So assert
    (a) sub-tap-noise reprojection and (b) the solver does not amplify the
    noise: floor error <= 1.5x that of the plain 4-point homography through
    the same taps (+5 mm). Measured over the 30 seeds: ratio median ~0.9.
    """
    rng = np.random.default_rng(100 + seed)
    true = [behind_camera, broadcast_camera][seed % 2](rng)
    pts = _pts(true, CORNERS, rng, 2.0)
    est = cc.calibrate_from_points(pts, SIZE)
    e = calib_errors(true, est)
    # reference: the plain 4-point homography through the same noisy taps
    _, Pc = cm.keypoints_array(CORNERS)
    _, P = cm.keypoints_array()
    Hn = cv2.getPerspectiveTransform(Pc[:, :2].astype(np.float32),
                                     np.array([pts[n] for n in CORNERS], np.float32))
    uv = true.project(P)
    vis = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) \
        & (uv[:, 0] < SIZE[0]) & (uv[:, 1] < SIZE[1])
    q = np.column_stack([uv[vis], np.ones(vis.sum())]) @ np.linalg.inv(Hn).T
    herr = np.linalg.norm(q[:, :2] / q[:, 2:] - P[vis, :2], axis=1)
    near = P[vis, 0] < 0
    assert e["reproj_med"] < 2.5, e
    assert e["near_med"] <= 1.5 * np.median(herr[near]) + 0.005, e
    assert e["far_med"] <= 1.5 * np.median(herr[~near]) + 0.005, e
    assert e["focal_rel"] < 0.03, e
    if seed % 2:                        # elevated broadcast view: ~3–6 cm per px at the far end
        assert e["near_med"] < 0.03 and e["far_med"] < 0.08, e


@pytest.mark.parametrize("seed", range(8))
def test_points_then_refine(seed: int) -> None:
    """The app path: ±2 px corner taps → refine_camera on the lines → < 2 cm / 3 cm."""
    rng = np.random.default_rng(300 + seed)
    true = [behind_camera, broadcast_camera][seed % 2](rng)
    est = cc.calibrate_from_points(_pts(true, CORNERS, rng, 2.0), SIZE)
    frames = render_frames(true, rng, style=["green", "wood", "blue"][seed % 3])
    ref = cc.refine_camera(est, frames)
    e = calib_errors(true, ref)
    assert is_accurate(e) and e["focal_rel"] < 0.03, e
    assert ref.reproj_rmse_px < 1.0 and 0 < ref.confidence <= 1


def test_points_mirrored_and_far_side_labels() -> None:
    """Left/right swapped taps and a camera behind the 'far' baseline are normalised."""
    rng = np.random.default_rng(7)
    true = broadcast_camera(rng)
    swapped = {"nb_ld": "nb_rd", "nb_rd": "nb_ld", "fb_ld": "fb_rd", "fb_rd": "fb_ld"}
    pts = _pts(true, CORNERS)
    est = cc.calibrate_from_points({swapped[k]: v for k, v in pts.items()}, SIZE)
    assert calib_errors(true, est)["world_max"] < 1e-3
    near_far = {"nb_ld": "fb_rd", "nb_rd": "fb_ld", "fb_ld": "nb_rd", "fb_rd": "nb_ld"}
    est = cc.calibrate_from_points({near_far[k]: v for k, v in pts.items()}, SIZE)
    assert est.center[0] < 0 and calib_errors(true, est)["world_max"] < 1e-3


def test_points_extra_keypoints_and_errors() -> None:
    rng = np.random.default_rng(3)
    true = behind_camera(rng)
    names = CORNERS + ["ns_ld", "ns_rd", "fs_c", "nl_ls"]
    est = cc.calibrate_from_points({**_pts(true, names), "bogus": (1.0, 2.0)}, SIZE)
    assert calib_errors(true, est)["world_max"] < 1e-3
    with pytest.raises(ValueError):
        cc.calibrate_from_points(_pts(true, CORNERS[:3]), SIZE)
    with pytest.raises(ValueError):   # collinear taps
        cc.calibrate_from_points({n: (float(i), float(i)) for i, n in enumerate(CORNERS)}, SIZE)


def test_helpers_roundtrip() -> None:
    true = CameraModel.look_at(np.array([-14.0, 0, 6]), np.zeros(3), 1500, SIZE)
    _, P = cm.keypoints_array()
    H = cc.homography_court_to_image(true)
    q = np.column_stack([P[:, :2], np.ones(len(P))]) @ H.T
    assert np.allclose(q[:, :2] / q[:, 2:], true.project(P), atol=1e-6)
    back = cc.image_to_court(true, true.project(P))
    assert np.allclose(back, P, atol=1e-6)


# ----------------------------------------------------------------------------- automatic path

_STYLES = ["green", "wood", "blue"]
_AUTO: dict[tuple[str, int], tuple[CameraModel, CameraModel | None, dict]] = {}
_KINDS = {"behind": behind_camera, "broadcast": broadcast_camera, "side": side_camera}


def _auto(kind: str, seed: int) -> tuple[CameraModel, CameraModel | None, dict]:
    """(true camera, automatic estimate or None, errors) — cached across tests."""
    if (kind, seed) not in _AUTO:
        rng = np.random.default_rng(1000 + seed)
        true = _KINDS[kind](rng)
        est = cc.calibrate_from_frames(render_frames(true, rng, style=_STYLES[seed % 3]), SIZE)
        _AUTO[kind, seed] = (true, est, {} if est is None else calib_errors(true, est))
    return _AUTO[kind, seed]


N_BEHIND = 30


@pytest.mark.parametrize("seed", range(N_BEHIND))
def test_auto_behind_never_garbage(seed: int) -> None:
    """Randomised phone-behind-baseline views (incl. cut-off near baselines, wood
    floors, lights, wall edges, white-shirt players): a returned camera is right."""
    true, est, e = _auto("behind", seed)
    if est is not None:
        assert not is_garbage(e), e
        assert 0.35 <= est.confidence <= 1 and est.center[0] < 0 and est.reproj_rmse_px < 1.5


def test_auto_behind_success_rate() -> None:
    """≥ 90 % of the behind cases calibrate to < 3 px / < 2 cm near / < 3 cm far / focal < 3 %."""
    good = []
    for s in range(N_BEHIND):
        _, est, e = _auto("behind", s)
        good.append(est is not None and is_accurate(e) and e["focal_rel"] < 0.03)
    assert np.mean(good) >= 0.9, [s for s, g in enumerate(good) if not g]


@pytest.mark.parametrize("kind,seed", [(k, s) for k in ("broadcast", "side") for s in range(5)])
def test_auto_other_views(kind: str, seed: int) -> None:
    """Elevated broadcast view (~(-14, 0, 6)) and 45° side views."""
    _, est, e = _auto(kind, seed)
    assert est is not None
    assert is_accurate(e) and e["focal_rel"] < 0.03, e


_TENNIS = {  # tennis court lines in the same world convention (net at x = 0)
    "bl0": ((-11.885, -5.485), (-11.885, 5.485)), "bl1": ((11.885, -5.485), (11.885, 5.485)),
    "sv0": ((-6.40, -4.115), (-6.40, 4.115)), "sv1": ((6.40, -4.115), (6.40, 4.115)),
    "d0": ((-11.885, -5.485), (11.885, -5.485)), "d1": ((-11.885, 5.485), (11.885, 5.485)),
    "s0": ((-11.885, -4.115), (11.885, -4.115)), "s1": ((-11.885, 4.115), (11.885, 4.115)),
    "c": ((-6.40, 0.0), (6.40, 0.0))}


def _grid_lines(rng: np.random.Generator) -> dict:
    """Random rectangular grid of painted lines (multi-sport hall floor markings)."""
    xs = np.cumsum(rng.uniform(0.8, 5.0, rng.integers(4, 8)))
    ys = np.cumsum(rng.uniform(0.5, 3.5, rng.integers(3, 6)))
    xs, ys = xs - xs.mean(), ys - ys.mean()
    d = {f"t{i}": ((x, ys.min()), (x, ys.max())) for i, x in enumerate(xs)}
    d.update({f"l{i}": ((xs.min(), y), (xs.max(), y)) for i, y in enumerate(ys)})
    return d


@pytest.mark.parametrize("case", ["empty0", "empty1", "tennis0", "tennis1", "grid0", "grid1",
                                  "grid11", "grid15", "noise"])
def test_auto_returns_none_without_badminton_court(case: str) -> None:
    """No court / other sports' lines / random line grids / pure noise → None."""
    seed = int(case.lstrip("emptyngrisdoe") or 0)
    rng = np.random.default_rng(1000 + seed)
    if case == "noise":
        frames = [rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8) for _ in range(3)]
        assert cc.calibrate_from_frames(frames, SIZE) is None
        return
    if case.startswith("tennis"):
        true, lines = broadcast_camera(rng), _TENNIS
    else:
        true = behind_camera(rng)
        lines = _grid_lines(rng) if case.startswith("grid") \
            else {"none": ((900.0, 900.0), (900.0, 900.001))}      # nothing in view
    frames = render_frames(true, rng, style=_STYLES[seed % 3], lines=lines)
    assert cc.calibrate_from_frames(frames, SIZE) is None
    assert cc.calibrate_from_frames([], SIZE) is None


def test_evidence_gate_rejects_shifted_court() -> None:
    """Regression for 'behind s=1 wood' (a garbage camera, 8 px / 4 m off, passed
    with confidence 0.53): the gate accepts the true camera and rejects the same
    camera with the court slid by one service-line spacing (0.76 m)."""
    rng = np.random.default_rng(1001)
    true = behind_camera(rng)
    feat = cc._Features(cc._to_image(render_frames(true, rng, style="wood"), SIZE))
    cx, cy = SIZE[0] / 2, SIZE[1] / 2
    search = max(3.0, 0.003 * feat.D)
    shifted = CameraModel(K=true.K, R=true.R, t=true.t + true.R @ np.array([0.76, 0, 0]),
                          image_size=SIZE)
    ok = cc._fit_stats(feat, cc._cam_params(true, False), cx, cy, cc._FINE, search)
    bad = cc._fit_stats(feat, cc._cam_params(shifted, False), cx, cy, cc._FINE, search)
    assert cc._valid(ok) and ok["explained_frac"] > 0.8, ok
    assert not cc._valid(bad) and cc._confidence(bad, 1.0, feat.D) == 0.0, bad


def test_auto_on_sim_renderer(tmp_path) -> None:
    """If the sibling renderer (sim.py) is available: calibrate its rendered video."""
    sim = pytest.importorskip("app.services.tracking.sim")
    if not all(hasattr(sim, n) for n in ("random_camera", "simulate_rally", "render_video")):
        pytest.skip("sim renderer not available")
    from app.services.tracking import video_io
    cam = sim.random_camera(np.random.default_rng(500), style="broadcast")
    path = sim.render_video(sim.simulate_rally(0, camera=cam, fps=30, n_shots=2),
                            str(tmp_path / "rally.mp4"), seed=0)
    frames = [f for _, f in video_io.sample_frames(path, 24)]
    est = cc.calibrate_from_frames(frames, cam.image_size)
    assert est is not None
    e = calib_errors(cam, est)
    assert is_accurate(e) and e["focal_rel"] < 0.03, e


# ----------------------------------------------------------------------------- n_t == 2 gate (venue sweep)

def _low_tripod(dist: float, seed: int, style: str) -> tuple[CameraModel, list]:
    """0.8 m high tripod `dist` behind the near baseline, 70° HFOV (venue calib_sweep): the
    near baseline / long service line are out of frame → only 2 transverse lines match."""
    eye = np.array([-_L - dist, 0.0, 0.8])
    cam = CameraModel.look_at(eye, np.zeros(3), SIZE[0] / (2 * np.tan(np.radians(35))), SIZE)
    return cam, render_frames(cam, np.random.default_rng(seed), style=style)


@pytest.mark.parametrize("seed,style", [(10000000, "green"), (10000002, "green"), (10000001, "wood")])
def test_auto_low_close_tripod_two_transverse_lines(seed: int, style: str) -> None:
    """Formerly rejected (n_t == 2) — and seed 10000002 accepted a camera 3.4 m off. Now the
    right camera passes the n_t == 2 exception (n_l == 5, net straddled, coverage >= 0.70)."""
    true, frames = _low_tripod(0.5, seed, style)
    d: dict = {}
    est = cc.calibrate_from_frames(frames, SIZE, diagnostics=d)
    assert est is not None and d["reason"] == "ok" and d["n_t"] == 2 and d["t_straddle"], d
    e = calib_errors(true, est)
    assert e["world_max"] < 0.15 and e["focal_rel"] < 0.03, e


def test_valid_n_t2_needs_all_conditions() -> None:
    ok = {"n_t": 2, "n_l": 5, "n_absent": 0, "explained_frac": 0.78, "rmse": 0.3, "coverage": 0.8,
          "t_straddle": True}
    assert cc._valid(ok)
    for k, v in (("n_l", 4), ("t_straddle", False), ("coverage", 0.69), ("n_absent", 1),
                 ("explained_frac", 0.54), ("n_t", 1)):
        assert not cc._valid({**ok, k: v}), k


def test_focal_hint_and_reasons() -> None:
    true, frames = _low_tripod(0.5, 10000000, "green")
    f = float(true.K[0, 0])
    d: dict = {}
    assert cc.calibrate_from_frames(frames, SIZE, diagnostics=d, focal_hint_px=f) is not None
    assert d["reason"] == "ok"
    d = {}
    assert cc.calibrate_from_frames(frames, SIZE, diagnostics=d, focal_hint_px=0.6 * f) is None
    assert d["reason"] == "focal_mismatch"
    # a WRONG hint is only a veto: on the scene where a 0.6 f camera exists (3.4 m off,
    # the old false accept) it must reject, not select that camera
    true3, frames3 = _low_tripod(0.5, 10000002, "green")
    d = {}
    assert cc.calibrate_from_frames(frames3, SIZE, diagnostics=d,
                                    focal_hint_px=0.6 * float(true3.K[0, 0])) is None
    assert d["reason"] == "focal_mismatch"
    d = {}
    assert cc.calibrate_from_frames([], SIZE, diagnostics=d) is None and d["reason"] == "no_frames"
    d = {}
    blank = [np.full((SIZE[1], SIZE[0], 3), 120, np.uint8)] * 3
    assert cc.calibrate_from_frames(blank, SIZE, diagnostics=d) is None
    assert d["reason"] == "no_line_families" and d["reason"] in cc.REASONS


# ----------------------------------------------------------------------------- single-frame fallback

def _behind(h: float, dist: float) -> CameraModel:
    """venue calib_sweep camera: `dist` behind the near baseline, `h` high, 70° HFOV."""
    return CameraModel.look_at(np.array([-_L - dist, 0.0, h]), np.zeros(3),
                               SIZE[0] / (2 * np.tan(np.radians(35))), SIZE)


def still_player_frames(cam: CameraModel, rng: np.random.Generator, style: str, still: range,
                        at=(-_L, 0.0), n: int = 24) -> list[np.ndarray]:
    """n frames with the usual moving players plus one player standing STILL at `at` (on the
    near baseline by default) in the frames of `still` — through the median, that player
    hides the line he stands on."""
    bg = render_background(cam, rng, style, True, None)
    frames = []
    for k in range(n):
        f = bg.copy()
        for half in (-1, 1):
            _player(f, cam, half * rng.uniform(1.0, 5.5), rng.uniform(-2.5, 2.5),
                    (245, 245, 245) if half < 0 else (50, 50, 200))
        if k in still:
            _player(f, cam, at[0], at[1], (230, 60, 60))
        f = cv2.GaussianBlur(f, (0, 0), 0.8)
        f = f.astype(np.float32) + rng.normal(0, 3, f.shape).astype(np.float32)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


def test_fallback_rejects_case3_frames_that_agree_on_a_wrong_camera(monkeypatch) -> None:
    """Sweep case 3 with the median forced to fail: frames 2, 3, 4 each ALONE pass the gate
    with the same camera 3.35 m wrong (they agree within 0.4 px, so '>= 2 frames agree' is
    not enough); in each, the right camera explains more but misses the gate (coverage
    0.62–0.69). The dominated-winner veto rejects them; only frame 1 is right → no pair."""
    _, frames = _low_tripod(0.5, 10000002, "green")
    orig = cc._calibrate_image

    def median_fails(img, size, diag=None, focal_hint=None, veto_dominated=False, colour="white"):
        if diag is not None and diag.get("source") == "median":
            diag["reason"] = "evidence_gate"
            return None
        return orig(img, size, diag, focal_hint=focal_hint, veto_dominated=veto_dominated, colour=colour)

    monkeypatch.setattr(cc, "_calibrate_image", median_fails)
    d: dict = {}
    assert cc.calibrate_from_frames(frames, SIZE, diagnostics=d) is None
    assert d["reason"] == "inconsistent_frames" and d["n_frame_accepts"] == 1, d
    assert d["frames_tried"] == [2, 1, 3, 0, 4]
    # without the veto frames 2 and 3 would have agreed on the wrong camera
    d2: dict = {}
    assert orig(cc._to_image([frames[2]], SIZE), SIZE, d2, veto_dominated=True) is None
    assert d2["reason"] == "ambiguous" and d2["dominated_by"] > d2["score"]


def test_fallback_player_still_on_a_line() -> None:
    """A player standing still on the near baseline in frames 4–20 of 24: the median hides
    that line (n_absent → evidence_gate, the known false reject); frames 3 and 21 (outside
    the still window) agree within 0.1 px on the right camera → accepted. Still for the
    WHOLE clip every frame is occluded the same way → an honest reject."""
    true = _behind(1.6, 3.0)
    frames = still_player_frames(true, np.random.default_rng(4), "blue", range(4, 21))
    dm: dict = {}
    assert cc._calibrate_image(cc._to_image(frames, SIZE), SIZE, dm) is None
    assert dm["reason"] == "evidence_gate" and dm["n_absent"] >= 1
    d: dict = {}
    est = cc.calibrate_from_frames(frames, SIZE, diagnostics=d)
    assert est is not None and d["reason"] == "ok" and d["frames_agree"] == [3, 21], d
    assert d["frame_agree_px"] <= cc.FRAME_AGREE_PX and calib_errors(true, est)["world_max"] < 0.05
    always = still_player_frames(true, np.random.default_rng(7), "green", range(24))
    d = {}
    assert cc.calibrate_from_frames(always, SIZE, diagnostics=d) is None
    assert d["reason"] == "evidence_gate" and d["n_frame_accepts"] == 0, d


def test_camera_agreement_px() -> None:
    a = _behind(1.6, 3.0)
    assert cc._camera_agreement_px(a, a) == 0.0
    b = CameraModel(K=a.K.copy(), R=a.R, t=a.t + a.R @ np.array([0.02, 0, 0]), image_size=SIZE)
    assert 0 < cc._camera_agreement_px(a, b) == cc._camera_agreement_px(b, a)
    assert cc._fallback_indices(24) == [12, 6, 18, 3, 21] and cc._fallback_indices(3) == [1, 0, 2]
    assert "inconsistent_frames" in cc.REASONS


def test_candidate_with_no_matched_lines_does_not_raise(monkeypatch) -> None:
    """_fit_stats returns an early dict WITHOUT 'explained' when no model line matches; the
    candidate loops read it as 0 (Swift: `explained ?? 0`). Before: KeyError on sweep2 case
    2050 (motion blur, h2.0 d6.0) — the whole calibration raised instead of choosing."""
    true = _behind(1.6, 3.0)
    frames = still_player_frames(true, np.random.default_rng(4), "blue", range(0))
    orig = cc._fit_stats
    calls = {"n": 0}

    def sometimes_empty(feat, p, cx, cy, samples, search):
        calls["n"] += 1
        if calls["n"] % 3 == 1:
            return {"coverage": 0.0, "rmse": np.inf, "n_t": 0, "n_l": 0, "n_vis": 0, "n_inl": 0,
                    "n_absent": 0, "absent_frac": 0.0, "t_straddle": False}
        return orig(feat, p, cx, cy, samples, search)

    monkeypatch.setattr(cc, "_fit_stats", sometimes_empty)
    d: dict = {}
    cc.calibrate_from_frames(frames, SIZE, diagnostics=d)       # must not raise
    assert calls["n"] > 0 and d["reason"] in cc.REASONS, d


def _sweep2_camera(h: float, dist: float, lat: float, pitch: float) -> CameraModel:
    """tools/venue/calib_sweep2 cases.make_camera: eye `dist` behind the near baseline, `h`
    high, `lat` sideways, aimed at the court centre with a `pitch` degree aim error."""
    eye = np.array([-_L - dist, lat, h])
    d = -eye
    horiz = np.hypot(d[0], d[1])
    ang = np.arctan2(d[2], horiz) + np.radians(pitch)
    tgt = eye + np.array([d[0], d[1], np.tan(ang) * horiz])
    return CameraModel.look_at(eye, tgt, SIZE[0] / (2 * np.tan(np.radians(35))), SIZE)


def test_same_solution_is_identity_aware() -> None:
    """A court shifted by one line spacing (3.96 m: long ↔ short service lines) puts most
    keypoints onto OTHER keypoints; the old set test called it 'the same camera'. The 180°
    twin (same image lines, court labels rotated) must still be the same solution."""
    true = _behind(1.6, 5.0)
    feat = cc._Features(np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    tol = max(3.0, 0.004 * feat.D)
    shift = np.array([[1, 0, 2 * cm.SHORT_SERVICE_X], [0, 1, 0], [0, 0, 1]], float)   # 3.96 m
    Ht = true.floor_homography()
    u = cc._kp_image(Ht, feat)
    assert not cc._same_solution(u, cc._kp_image(Ht @ shift, feat), tol)
    assert cc._same_solution(u, cc._kp_image(cc._flip_180(true).floor_homography(), feat), tol)
    assert cc._same_solution(u, u, tol) and sorted(cc._KP_180) == list(range(len(cc._KP)))


@pytest.mark.parametrize("h,dist,lat,pitch,seed,style,want", [
    (0.7, 1.25, 1.0, 0.0, 72254, "wood", "ambiguous"),   # sweep2 #508: accepted 3.96 m off before
    (0.6, 0.5, 0.0, 0.0, 71205, "blue", None),           # #384: accepted 2.03 m off before
    (1.8, 7.0, 0.0, 3.0, 86112, "green", "ok"),          # #1426: 3.97 m off before; net ranking → right
    (1.2, 7.0, 1.0, -3.0, 83794, "blue", "ok"),          # #1192: accepted 1.76 m off, now right
])
def test_sweep2_former_false_accepts(h, dist, lat, pitch, seed, style, want) -> None:
    """The dominated-winner veto on the median (the right camera explains more but misses the
    gate: n_t == 2 without straddle), the identity-aware distinctness test, ranking valid
    cameras by net evidence (a shifted court on a colour-blend ridge at the mat edge explains
    more but leaves more model line unexplained) and the two-window quick refinement turn
    these former wrong cameras into rejects or the right camera."""
    true = _sweep2_camera(h, dist, lat, pitch)
    frames = render_frames(true, np.random.default_rng(seed), n=5, style=style)
    d: dict = {}
    est = cc.calibrate_from_frames(frames, SIZE, diagnostics=d)
    if est is not None:
        assert calib_errors(true, est)["world_max"] < 0.10, (d.get("reason"), calib_errors(true, est))
    if want is not None:
        assert d["reason"] == want, d.get("reason")


def test_whiteness_colour_models() -> None:
    """Paint scores (BGR): white keeps the old formula; yellow paint (low B) is ~as dark as a
    green mat on the white model but far above it on the yellow model; each model scores
    its own paint above the floors of the sweep renderer."""
    px = {"white": (235, 235, 235), "yellow": (34, 196, 238), "light_blue": (235, 200, 120),
          "red": (40, 40, 196), "green_mat": (70, 140, 60), "blue_mat": (165, 105, 40),
          "wood": (60, 120, 175)}
    img = np.array([list(px.values())], np.uint8)
    w = {c: dict(zip(px, cc._whiteness(img, c)[0])) for c in cc.LINE_COLOURS}
    f = img.astype(np.float32)[0]
    assert np.array_equal(cc._whiteness(img)[0], f.min(1) - 0.5 * (f.max(1) - f.min(1)))
    assert w["white"]["yellow"] < w["white"]["green_mat"] + cc._MIN_CONTRAST
    for colour, paint in (("white", "white"), ("yellow", "yellow"), ("cyan", "light_blue"), ("red", "red")):
        for floor in ("green_mat", "blue_mat", "wood"):
            if (paint, floor) in (("red", "wood"),):
                continue                         # red on wood: 20 over the floor, marginal
            assert w[colour][paint] > w[colour][floor] + 2 * cc._MIN_CONTRAST, (colour, paint, floor)
    assert cc.LINE_COLOURS[0] == "white"


@pytest.mark.parametrize("colour,style,seed", [((34, 196, 238), "green", 5), ((34, 196, 238), "blue", 6),
                                               ((235, 200, 120), "green", 7), ((40, 40, 196), "blue", 8)])
def test_auto_coloured_court_lines(colour, style, seed) -> None:
    """Yellow / light blue / red court lines → the right camera. Yellow paint scores below a
    green mat on the white model, so a colour model must find it (diagnostics['line_colour']);
    light blue / red lines bright enough on the white model are found by it (no key)."""
    true = _behind(1.6, 5.0)
    rng = np.random.default_rng(seed)
    bg = render_background(true, rng, style, True, line_colour=colour)
    frames = []
    for _ in range(5):
        fr = bg.copy()
        for half in (-1, 1):
            _player(fr, true, half * rng.uniform(1.0, 5.5), rng.uniform(-2.5, 2.5), (60, 60, 200))
        fr = cv2.GaussianBlur(fr, (0, 0), 0.8).astype(np.float32) + rng.normal(0, 3, fr.shape).astype(np.float32)
        frames.append(np.clip(fr, 0, 255).astype(np.uint8))
    d: dict = {}
    est = cc.calibrate_from_frames(frames, SIZE, diagnostics=d)
    assert est is not None, d.get("reason")
    if colour == (34, 196, 238):
        assert d.get("line_colour") == "yellow", d.get("line_colour")
    assert d.get("line_colour") in (None, "yellow", "cyan", "red")
    assert calib_errors(true, est)["world_max"] < 0.10


@pytest.mark.parametrize("colour,style,seed", [((34, 196, 238), "green", 15), ((40, 40, 196), "blue", 16),
                                               ((30, 120, 232), "green", 17)])
def test_points_then_refine_coloured_lines(colour, style, seed) -> None:
    """Manual path on yellow / red / orange lines: ±2 px corner taps → refine_camera picks the
    paint model by _paint_fit under the tapped camera and refines like on white lines; the
    white-only refinement (the old function) does not converge there."""
    rng = np.random.default_rng(seed)
    true = _behind(1.6, 5.0)
    bg = render_background(true, rng, style, True, line_colour=colour)
    med = np.clip(cv2.GaussianBlur(bg, (0, 0), 0.8), 0, 255).astype(np.uint8)
    est = cc.calibrate_from_points(_pts(true, CORNERS, rng, 2.0), SIZE)
    ref = cc.refine_camera(est, [med])
    e = calib_errors(true, ref)
    assert ref is not est and is_accurate(e) and e["focal_rel"] < 0.03, e
    assert max(cc._paint_fit(med, est), key=cc._paint_fit(med, est).get) != "white"


def test_refine_white_lines_unchanged() -> None:
    """White lines: refine_camera == the white-model refinement (the old function), exactly."""
    rng = np.random.default_rng(21)
    true = _behind(1.6, 5.0)
    frames = render_frames(true, rng, style="wood")
    est = cc.calibrate_from_points(_pts(true, CORNERS, rng, 2.0), SIZE)
    med = cc._to_image(frames, SIZE)
    a, b = cc.refine_camera(est, frames), cc._refine_colour(est, med, "auto", None, "white")
    assert np.array_equal(a.K, b.K) and np.array_equal(a.R, b.R) and np.array_equal(a.t, b.t)
