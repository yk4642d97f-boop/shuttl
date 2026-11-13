"""
Camera calibration (intrinsics + pose) from badminton court lines.

Public API
  calibrate_from_points(image_pts, image_size)          -> CameraModel   (user-tapped keypoints)
  calibrate_from_frames(frames, image_size, doubles_hint) -> CameraModel | None  (automatic)
  refine_camera(camera, frames_or_median)                -> CameraModel   (dense line refinement)
  detect_line_mask(img)                                  -> uint8 mask (255 = court-line pixel)
  homography_court_to_image(camera), image_to_court(camera, uv, z=0)

Camera model: square pixels, principal point at the image centre, optional k1.
World frame / names: court_model.py. The near half (x < 0) is always the camera side.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.optimize import least_squares

from app.services.tracking import court_model as cm
from app.services.tracking.camera import CameraModel

logger = logging.getLogger(__name__)

_ROT_Z_180 = np.diag([-1.0, -1.0, 1.0])


# ----------------------------------------------------------------------------- helpers

def homography_court_to_image(camera: CameraModel) -> np.ndarray:
    """3x3 H mapping floor (X, Y, 1) in metres → pixels (ignores distortion)."""
    return camera.floor_homography()


def image_to_court(camera: CameraModel, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Pixels (N,2) → world points (N,3) on the plane Z=z (NaN where the ray misses)."""
    return camera.pixel_to_floor(np.atleast_2d(np.asarray(uv, float)), z=z)


def _proper_orientation(H: np.ndarray, xy: np.ndarray) -> bool:
    """True if H maps the floor as seen from ABOVE (camera over the floor).

    The Jacobian of a floor→image homography has det = det(H)/w^3; seen from
    above, the counter-clockwise world (x, y) maps to a negative-determinant
    frame in (u right, v down) image coordinates.
    """
    w = H[2] @ np.array([xy[0], xy[1], 1.0])
    return bool(np.linalg.det(H) * w ** 3 < 0)


def _focal_from_homography(H: np.ndarray, cx: float, cy: float, scale: float
                           ) -> tuple[float, float]:
    """Zhang's constraints with known principal point + square pixels.

    Returns (focal_px or nan, conditioning weight). Works on (3,3) or (N,3,3).
    """
    H = np.asarray(H, float)
    T = np.array([[1 / scale, 0, -cx / scale], [0, 1 / scale, -cy / scale], [0, 0, 1.0]])
    Hc = T @ H
    Hc = Hc / np.linalg.norm(Hc.reshape(*Hc.shape[:-2], 9), axis=-1)[..., None, None]
    h1, h2 = Hc[..., :, 0], Hc[..., :, 1]
    a1 = h1[..., 0] * h2[..., 0] + h1[..., 1] * h2[..., 1]
    b1 = h1[..., 2] * h2[..., 2]
    a2 = h1[..., 0] ** 2 + h1[..., 1] ** 2 - h2[..., 0] ** 2 - h2[..., 1] ** 2
    b2 = h1[..., 2] ** 2 - h2[..., 2] ** 2
    den = a1 * a1 + a2 * a2
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -(a1 * b1 + a2 * b2) / den          # w = scale^2 / f^2
        f = np.where(w > 0, scale / np.sqrt(np.abs(w)), np.nan)
    return f, np.sqrt(den)


def _pose_from_homography(H: np.ndarray, K: np.ndarray, ref_xy: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Decompose floor homography with known K → (R, t), ref point in front."""
    M = np.linalg.solve(K, H)
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if (M @ np.array([ref_xy[0], ref_xy[1], 1.0]))[2] * lam < 0:
        lam = -lam
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


def _make_camera(f: float, rvec: np.ndarray, t: np.ndarray, image_size: tuple[int, int],
                 k1: float = 0.0, cx: float | None = None, cy: float | None = None
                 ) -> CameraModel:
    w, h = image_size
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    dist = np.zeros(5)
    dist[0] = k1
    return CameraModel(K=K, R=R, t=np.asarray(t, float).copy(), image_size=tuple(image_size),
                       dist=dist)


def _project_params(p: np.ndarray, P: np.ndarray, cx: float, cy: float,
                    return_z: bool = False):
    """Fast projection for optimisation. p = [log f, rvec(3), t(3), (k1)]."""
    f = np.exp(p[0])
    R = cv2.Rodrigues(np.asarray(p[1:4], float).reshape(3, 1))[0]
    c = P @ R.T + p[4:7]
    z = np.maximum(c[:, 2], 1e-6)
    x, y = c[:, 0] / z, c[:, 1] / z
    if len(p) > 7:
        rad = 1 + p[7] * (x * x + y * y)
        x, y = x * rad, y * rad
    uv = np.column_stack([f * x + cx, f * y + cy])
    return (uv, c[:, 2]) if return_z else uv


def _cam_params(cam: CameraModel, with_k1: bool) -> np.ndarray:
    f = 0.5 * (cam.K[0, 0] + cam.K[1, 1])
    rvec = cv2.Rodrigues(cam.R)[0].ravel()
    p = [np.log(f), *rvec, *cam.t]
    if with_k1:
        p.append(float(cam.dist[0]) if len(cam.dist) else 0.0)
    return np.array(p, float)


def _flip_180(cam: CameraModel) -> CameraModel:
    """Re-express the camera in the court frame rotated 180° about z (near↔far)."""
    return CameraModel(K=cam.K.copy(), R=cam.R @ _ROT_Z_180, t=cam.t.copy(),
                       image_size=cam.image_size, dist=cam.dist.copy(),
                       reproj_rmse_px=cam.reproj_rmse_px, confidence=cam.confidence)


def _default_focal(image_size: tuple[int, int]) -> float:
    return 0.9 * max(image_size)


# ----------------------------------------------------------------------------- manual path

def calibrate_from_points(image_pts: dict[str, tuple[float, float]],
                          image_size: tuple[int, int]) -> CameraModel:
    """Camera from ≥4 named floor keypoints (court_model.COURT_KEYPOINTS names).

    Homography DLT → focal from Zhang's constraints (principal point at centre)
    → pose → Levenberg-Marquardt on reprojection error (weak focal prior).
    Mirrored left/right labels are detected (camera would be under the floor)
    and corrected; if the result puts the camera behind the far baseline the
    frame is rotated 180° so the camera side is always x < 0.
    Raises ValueError on insufficient / degenerate input.
    """
    names = [n for n in image_pts if n in cm.COURT_KEYPOINTS]
    unknown = [n for n in image_pts if n not in cm.COURT_KEYPOINTS]
    if unknown:
        logger.warning("ignoring unknown keypoint names: %s", unknown)
    if len(names) < 4:
        raise ValueError(f"need >= 4 known court keypoints, got {len(names)}")
    uv = np.array([image_pts[n] for n in names], float).reshape(-1, 2)
    xy = np.array([cm.COURT_KEYPOINTS[n] for n in names], float)
    if not np.all(np.isfinite(uv)):
        raise ValueError("non-finite keypoint coordinates")
    H, _ = cv2.findHomography(xy, uv, 0)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("degenerate keypoint configuration")
    if not _proper_orientation(H, xy.mean(0)):
        logger.warning("keypoints are left/right mirrored w.r.t. court_model; swapping labels")
        xy[:, 1] *= -1
        H, _ = cv2.findHomography(xy, uv, 0)
    cam = _camera_from_homography(H, image_size, xy)
    if cam is None:
        raise ValueError("keypoints do not correspond to a physical camera")
    cam = _refine_on_points(cam, np.column_stack([xy, np.zeros(len(xy))]), uv)
    if cam.center[0] > 0.5:
        logger.warning("camera is on the x>0 side of the given labels; rotating court frame")
        cam = _flip_180(cam)
    n = len(names)
    cam.confidence = float(np.clip(np.exp(-cam.reproj_rmse_px / 4.0) * min(1.0, 0.75 + 0.03 * n),
                                   0.0, 1.0))
    return cam


def _camera_from_homography(H: np.ndarray, image_size: tuple[int, int],
                            ref_xy: np.ndarray, f: float | None = None) -> CameraModel | None:
    """Floor homography → CameraModel (focal via Zhang unless given)."""
    w, h = image_size
    cx, cy = w / 2, h / 2
    if f is None:
        f_est, _ = _focal_from_homography(H, cx, cy, float(max(w, h)))
        f = float(f_est) if np.isfinite(f_est) and 0.15 * w < f_est < 8 * w else _default_focal(image_size)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R, t = _pose_from_homography(H, K, np.asarray(ref_xy, float).reshape(-1, 2).mean(0))
    cam = CameraModel(K=K, R=R, t=t, image_size=tuple(image_size))
    if cam.center[2] <= 0:
        return None
    return cam


def _refine_on_points(cam: CameraModel, P: np.ndarray, uv: np.ndarray,
                      sigma_px: float = 1.0, logf_sigma: float = 1.0) -> CameraModel:
    """LM over (log f, rvec, t) on point reprojection with a weak focal prior."""
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    p0 = _cam_params(cam, with_k1=False)
    logf_prior = np.log(_default_focal(cam.image_size))

    def res(p: np.ndarray) -> np.ndarray:
        r = (_project_params(p, P, cx, cy) - uv).ravel() / sigma_px
        return np.append(r, (p[0] - logf_prior) / logf_sigma)

    sol = least_squares(res, p0, method="lm", max_nfev=400)
    out = _make_camera(np.exp(sol.x[0]), sol.x[1:4], sol.x[4:7], cam.image_size, cx=cx, cy=cy)
    if out.center[2] <= 0:
        out = cam
    err = np.linalg.norm(out.project(P) - uv, axis=1)
    out.reproj_rmse_px = float(np.sqrt(np.mean(err ** 2)))
    return out


# ----------------------------------------------------------------------------- automatic path

_MIN_CONTRAST = 18.0       # whiteness units (0..255) a line must exceed its neighbours by


# Line paint colour models, tried in this order by calibrate_from_frames (first accept wins):
# the channels the paint is bright in (BGR indices here; Swift: RGB, by name). BWF allows
# white or yellow lines; multi-sport halls also paint light blue / red courts.
LINE_COLOURS = ("white", "yellow", "cyan", "red")
_COLOUR_CHANNELS = {"white": (0, 1, 2), "yellow": (1, 2), "cyan": (0, 1), "red": (2,)}


def _whiteness(img: np.ndarray, colour: str = "white") -> np.ndarray:
    """Per-pixel 'line paint' score for a paint colour model (float32).

    white: min channel minus half the channel spread — white/grey lines keep all channels
    high; green/blue/red floors and coloured shirts or ads have a low minimum channel.
    Other colours, with S = the channels the paint is bright in and O = the others:
    min(S) − ½·(max(S) − min(S)) − ½·max(O). Yellow paint (B low) scores ~22 on the white
    model — no more than a green mat — but ~150 on the yellow one, 60-90 above the floor.
    """
    if img.ndim == 2:
        return img.astype(np.float32)
    f = img.astype(np.float32)
    if colour == "white":
        mn, mx = f.min(axis=2), f.max(axis=2)
        return mn - 0.5 * (mx - mn)
    s = _COLOUR_CHANNELS[colour]
    o = [k for k in range(3) if k not in s]
    fs = f[..., list(s)]
    mn, mx = fs.min(axis=2), fs.max(axis=2)
    return (mn - 0.5 * (mx - mn)) - 0.5 * f[..., o].max(axis=2)


def _ridge_strength(img: np.ndarray, colour: str = "white") -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale 'line pixel' test (Farin et al.): a pixel is on a thin bright
    line if it is brighter than BOTH neighbours at ±s px horizontally (for
    vertical-ish lines) or vertically (horizontal-ish lines), on the `colour` model's
    paint score. Returns (strength_h_test, strength_v_test), float32 >= 0.
    """
    W = cv2.GaussianBlur(_whiteness(img, colour), (0, 0), 0.8)
    h, w = W.shape
    smax = max(4, int(round(0.012 * max(h, w))))
    scales = sorted({int(s) for s in np.round(np.geomspace(2, smax, 6))})
    best_h = np.zeros_like(W)
    best_v = np.zeros_like(W)
    for s in scales:
        P = cv2.copyMakeBorder(W, s, s, s, s, cv2.BORDER_REFLECT)
        rh = np.minimum(W - P[s:s + h, 0:w], W - P[s:s + h, 2 * s:2 * s + w])
        rv = np.minimum(W - P[0:h, s:s + w], W - P[2 * s:2 * s + h, s:s + w])
        np.maximum(best_h, rh, out=best_h)
        np.maximum(best_v, rv, out=best_v)
    return best_h, best_v


def _mask_from_strength(strength: np.ndarray, min_len: float) -> np.ndarray:
    """Threshold + drop small connected components (specks, text dots, lights)."""
    m = (strength > _MIN_CONTRAST).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ext = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = ext >= min_len
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def detect_line_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of thin bright court-line pixels.

    Ridge test at several widths on a whiteness image (colour-gated), so large
    bright blobs (lights, windows, shirts wider than ~2.4 % of the frame) and
    coloured paint are rejected; small isolated components are removed.
    """
    sh, sv = _ridge_strength(img)
    return _mask_from_strength(np.maximum(sh, sv), 0.012 * max(img.shape[:2]))


def _median_image(frames: list[np.ndarray] | np.ndarray, max_frames: int = 7) -> np.ndarray:
    """Temporal median of up to `max_frames` evenly spaced frames (removes players)."""
    if isinstance(frames, np.ndarray) and frames.ndim in (2, 3) and frames.dtype == np.uint8 \
            and (frames.ndim == 2 or frames.shape[2] in (3, 4)):
        return frames
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError("no frames")
    if len(frames) <= 2:
        return frames[0]
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)
    stack = np.stack([frames[i] for i in idx])
    k = len(idx) // 2
    return np.partition(stack, k, axis=0)[k]


class _Features:
    """Per-image line evidence shared by initialisation, scoring and refinement."""

    def __init__(self, img: np.ndarray, colour: str = "white"):
        self.h, self.w = img.shape[:2]
        self.D = float(max(self.h, self.w))
        self.colour = colour
        sh, sv = _ridge_strength(img, colour)
        self.strength = np.maximum(sh, sv)
        self.mask = _mask_from_strength(self.strength, 0.012 * self.D)
        self.dt = cv2.distanceTransform(255 - self.mask, cv2.DIST_L2, 3)
        # centre-line pixels: local maxima of the in-mask distance transform
        # across the line (x-axis test for vertical-ish, y-axis for horizontal-ish)
        dti = cv2.distanceTransform(self.mask, cv2.DIST_L2, 3)
        P = cv2.copyMakeBorder(dti, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cx = (dti >= P[1:-1, :-2]) & (dti >= P[1:-1, 2:])
        cy = (dti >= P[:-2, 1:-1]) & (dti >= P[2:, 1:-1])
        thin = (self.mask > 0) & np.where(sh >= sv, cx, cy)
        self.thin = thin.astype(np.uint8) * 255


_MAX_SEGMENTS = 600      # real courts give ~120–180; texture/noise gives thousands


def _segments(feat: _Features) -> np.ndarray:
    """HoughLinesP on centre-line pixels → (N,4) segments x1,y1,x2,y2 (longest kept)."""
    D = feat.D
    segs = cv2.HoughLinesP(feat.thin, 1, np.pi / 720, threshold=max(12, int(0.012 * D)),
                           minLineLength=0.03 * D, maxLineGap=0.006 * D)
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)
    if len(segs) > _MAX_SEGMENTS:     # bounds the O(N·lines) merge (pure noise: 13 s → <1 s)
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs = segs[np.argsort(-L, kind="stable")[:_MAX_SEGMENTS]]
    return segs


class _Line:
    __slots__ = ("l", "p0", "p1", "weight")

    def __init__(self, pts: np.ndarray, wts: np.ndarray, weight: float):
        c = (pts * wts[:, None]).sum(0) / wts.sum()
        d = pts - c
        cov = (d * wts[:, None]).T @ d
        evals, evecs = np.linalg.eigh(cov)
        u = evecs[:, 1]                                  # direction
        n = np.array([-u[1], u[0]])
        self.l = np.array([n[0], n[1], -n @ c])          # a²+b²=1
        s = d @ u
        self.p0, self.p1 = c + s.min() * u, c + s.max() * u
        self.weight = weight

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    @property
    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / (np.linalg.norm(d) + 1e-12)


def _merge_segments(segs: np.ndarray, D: float, max_lines: int = 40) -> list[_Line]:
    """Greedy collinear merge (longest first) → infinite lines with support weight."""
    if len(segs) == 0:
        return []
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    tol_d = max(1.5, 0.0015 * D)
    tol_a = np.sin(np.radians(1.0))
    groups: list[list[int]] = []
    glines: list[_Line] = []
    for i in np.argsort(-L):
        p, q = segs[i, :2], segs[i, 2:]
        u = (q - p) / (L[i] + 1e-12)
        for g, gl in enumerate(glines):
            n = gl.l[:2]
            if abs(n @ u) > tol_a:
                continue
            if abs(gl.l @ [*p, 1]) > tol_d or abs(gl.l @ [*q, 1]) > tol_d:
                continue
            groups[g].append(i)
            idx = groups[g]
            pts = np.vstack([segs[idx, :2], segs[idx, 2:]])
            glines[g] = _Line(pts, np.tile(L[idx], 2), float(L[idx].sum()))
            break
        else:
            groups.append([i])
            glines.append(_Line(np.vstack([p, q]), np.array([L[i], L[i]]), float(L[i])))
    glines.sort(key=lambda g: -g.weight)
    return glines[:max_lines]


def _vp_inliers(v: np.ndarray, mids: np.ndarray, dirs: np.ndarray, lens: np.ndarray
                ) -> np.ndarray:
    """Lines (mid, dir) consistent with homogeneous vanishing point(s) v (K,3) → (K,N) bool."""
    ux = v[:, None, 0] - v[:, None, 2] * mids[None, :, 0]
    uy = v[:, None, 1] - v[:, None, 2] * mids[None, :, 1]
    nu = np.hypot(ux, uy) + 1e-12
    sin = np.abs(dirs[None, :, 0] * uy - dirs[None, :, 1] * ux) / nu
    tol = np.clip(2.5 / lens, 0.003, 0.03)
    return sin < tol[None, :]


def _vp_families(lines: list[_Line], max_fam: int = 3) -> list[tuple[np.ndarray, list[_Line]]]:
    """Group lines by vanishing point (exhaustive pair hypotheses, greedy)."""
    fams = []
    remaining = list(lines)
    while len(remaining) >= 2 and len(fams) < max_fam:
        Lh = np.array([ln.l for ln in remaining])
        mids = np.array([ln.mid for ln in remaining])
        dirs = np.array([ln.direction for ln in remaining])
        lens = np.array([np.linalg.norm(ln.p1 - ln.p0) for ln in remaining])
        wts = np.array([ln.weight for ln in remaining])
        i, j = np.triu_indices(len(remaining), 1)
        V = np.cross(Lh[i], Lh[j])
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-18
        inl = _vp_inliers(V, mids, dirs, lens)
        score = inl.astype(float) @ wts
        b = int(np.argmax(score))
        members = np.flatnonzero(inl[b])
        if len(members) < 2:
            break
        # least-squares VP over members (smallest eigvec of sum w l l^T)
        A = Lh[members] * np.sqrt(wts[members])[:, None]
        v = np.linalg.svd(A)[2][-1]
        inl2 = _vp_inliers(v[None], mids, dirs, lens)[0]
        members = np.flatnonzero(inl2 | inl[b])
        fams.append((v, [remaining[k] for k in members]))
        remaining = [remaining[k] for k in range(len(remaining)) if k not in set(members)]
    return fams


_XT = np.array([-cm.HALF_LENGTH, -cm.DOUBLES_LONG_SERVICE_X, -cm.SHORT_SERVICE_X,
                cm.SHORT_SERVICE_X, cm.DOUBLES_LONG_SERVICE_X, cm.HALF_LENGTH])
_YL = np.array([-cm.DOUBLES_HALF_WIDTH, -cm.SINGLES_HALF_WIDTH, 0.0,
                cm.SINGLES_HALF_WIDTH, cm.DOUBLES_HALF_WIDTH])


def _model_combos() -> tuple[np.ndarray, np.ndarray]:
    """All (X_a→u=0, X_b→u=1, Y_c→v=0, Y_d→v=1) assignments: affine maps model→unit square.

    X pairs are ordered (both orientations), Y pairs increasing only: together
    with the chirality test this covers every proper camera up to the 180°
    court symmetry, which is resolved afterwards (camera side = x < 0).
    """
    A, ref = [], []
    for a in range(6):
        for b in range(6):
            if a == b:
                continue
            for c in range(5):
                for d in range(c + 1, 5):
                    xa, xb, yc, yd = _XT[a], _XT[b], _YL[c], _YL[d]
                    A.append([[1 / (xb - xa), 0, -xa / (xb - xa)],
                              [0, 1 / (yd - yc), -yc / (yd - yc)], [0, 0, 1]])
                    ref.append([(xa + xb) / 2, (yc + yd) / 2])
    return np.array(A), np.array(ref)


_COMBO_A, _COMBO_REF = _model_combos()


def _line_samples(step: float, inset: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (M,3), unit directions (M,3) and line ids (M,) along COURT_LINES."""
    pts, dirs, ids = [], [], []
    for k, (a, b) in enumerate(cm.COURT_LINES.values()):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        u = (b - a) / L
        n = max(2, int(round((L - 2 * inset) / step)) + 1)
        s = np.linspace(inset, L - inset, n)
        xy = a + s[:, None] * u
        pts.append(np.column_stack([xy, np.zeros(n)]))
        dirs.append(np.tile([u[0], u[1], 0.0], (n, 1)))
        ids.append(np.full(n, k))
    return np.vstack(pts), np.vstack(dirs), np.concatenate(ids)


_LINE_NAMES = list(cm.COURT_LINES)
_IS_TRANSVERSE = np.array([abs(a[0] - b[0]) < 1e-9 for a, b in cm.COURT_LINES.values()])
_VCOARSE = _line_samples(1.0)
_COARSE = _line_samples(0.3)
_MEDIUM = _line_samples(0.2)
_FINE = _line_samples(0.1)
_KP = np.array(list(cm.COURT_KEYPOINTS.values()), float)


def _order_family(v: np.ndarray, lines: list[_Line], c: np.ndarray) -> list[_Line]:
    """Sort a family's lines by where they cross a transversal through c."""
    d = v[:2] - v[2] * c
    d = d / (np.linalg.norm(d) + 1e-18)
    lt = np.array([d[0], d[1], -d @ c])
    n = np.array([-d[1], d[0]])
    s = []
    for ln in lines:
        X = np.cross(ln.l, lt)
        s.append(n @ (X[:2] / X[2] - c) if abs(X[2]) > 1e-12 else 0.0)
    return [lines[i] for i in np.argsort(s)]


def _square_to_quad(q: np.ndarray) -> np.ndarray:
    """(N,4,2) corners for unit-square (0,0),(1,0),(1,1),(0,1) → (N,3,3) (Heckbert)."""
    x0, x1, x2, x3 = q[:, 0, 0], q[:, 1, 0], q[:, 2, 0], q[:, 3, 0]
    y0, y1, y2, y3 = q[:, 0, 1], q[:, 1, 1], q[:, 2, 1], q[:, 3, 1]
    dx1, dx2, dx3 = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, dy3 = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    den = dx1 * dy2 - dx2 * dy1
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (dx3 * dy2 - dx2 * dy3) / den
        h = (dx1 * dy3 - dx3 * dy1) / den
    H = np.empty((len(q), 3, 3))
    H[:, 0] = np.column_stack([x1 - x0 + g * x1, x3 - x0 + h * x3, x0])
    H[:, 1] = np.column_stack([y1 - y0 + g * y1, y3 - y0 + h * y3, y0])
    H[:, 2] = np.column_stack([g, h, np.ones(len(q))])
    return H


def _plausible(H: np.ndarray, ref: np.ndarray, cx: float, cy: float, D: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised 'is this floor homography a real camera' test → (ok mask, focal)."""
    f, _ = _focal_from_homography(H, cx, cy, D)
    ok = np.isfinite(f) & (f > 0.2 * D) & (f < 6 * D)
    f = np.where(ok, f, D)
    m1 = np.stack([(H[:, 0, 0] - cx * H[:, 2, 0]) / f, (H[:, 1, 0] - cy * H[:, 2, 0]) / f,
                   H[:, 2, 0]], 1)
    m2 = np.stack([(H[:, 0, 1] - cx * H[:, 2, 1]) / f, (H[:, 1, 1] - cy * H[:, 2, 1]) / f,
                   H[:, 2, 1]], 1)
    m3 = np.stack([(H[:, 0, 2] - cx * H[:, 2, 2]) / f, (H[:, 1, 2] - cy * H[:, 2, 2]) / f,
                   H[:, 2, 2]], 1)
    n1, n2 = np.linalg.norm(m1, axis=1), np.linalg.norm(m2, axis=1)
    ortho = np.abs(np.sum(m1 * m2, 1)) / (n1 * n2 + 1e-18)
    ratio = np.abs(n1 / (n2 + 1e-18) - 1)
    lam = 2 / (n1 + n2 + 1e-18)
    depth = m1[:, 2] * ref[:, 0] + m2[:, 2] * ref[:, 1] + m3[:, 2]
    lam = lam * np.sign(depth)
    r1, r2, t = m1 * lam[:, None], m2 * lam[:, None], m3 * lam[:, None]
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3, axis=1, keepdims=True) + 1e-18
    height = -np.sum(r3 * t, 1)
    dist = np.sqrt(np.sum(r1 * t, 1) ** 2 + np.sum(r2 * t, 1) ** 2 + height ** 2)
    ok &= (ortho < 0.15) & (ratio < 0.15) & (height > 0.3) & (height < 80) & (dist < 150)
    return ok, f


def _project_H(H: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,3,3) × (M,2) → pixels (N,M,2), w (N,M). One GEMM (einsum was 50x slower)."""
    X = np.column_stack([xy, np.ones(len(xy))])
    p = (H.reshape(-1, 3) @ X.T).reshape(len(H), 3, len(xy))
    w = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([p[:, 0] / w, p[:, 1] / w], -1)
    return uv, w


def _dt_lookup(feat: _Features, uv: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance-to-line lookup; returns (dt, in_image) with dt=inf outside."""
    u, v = uv[..., 0], uv[..., 1]
    inside = valid & (u >= 0) & (v >= 0) & (u <= feat.w - 1) & (v <= feat.h - 1)
    ui = np.where(inside, u, 0).astype(np.int64)
    vi = np.where(inside, v, 0).astype(np.int64)
    dt = np.where(inside, feat.dt[vi, ui], np.inf)
    return dt, inside


def _line_score(feat: _Features, H: np.ndarray, samples=_COARSE, tol: float | None = None) -> np.ndarray:
    """Projected-length-weighted agreement of the court model with the line mask
    (samples within `tol` px of a line pixel count +, beyond 3·tol −½; default tol max(2, 0.25 % D))."""
    P, _, ids = samples
    X = np.column_stack([P[:, :2], np.ones(len(P))]).astype(np.float32)
    Hn = H / np.abs(H).reshape(len(H), -1).max(1)[:, None, None]     # float32-safe scale
    p = (Hn.astype(np.float32).reshape(-1, 3) @ X.T).reshape(len(H), 3, len(P))
    w = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = p[:, 0] / w, p[:, 1] / w
        # weight = projected spacing to the neighbouring samples on the same line
        d = np.hypot(np.diff(u, axis=1), np.diff(v, axis=1))
    d = np.where((ids[1:] == ids[:-1])[None] & np.isfinite(d), d, 0.0)
    step = np.zeros_like(u)
    step[:, :-1] += 0.5 * d
    step[:, 1:] += 0.5 * d
    np.clip(step, 0, 0.05 * feat.D, out=step)
    inside = (w > 0) & (u >= 0) & (v >= 0) & (u <= feat.w - 1) & (v <= feat.h - 1)
    flat = np.where(inside, v, 0).astype(np.int64) * feat.w + np.where(inside, u, 0).astype(np.int64)
    dt = feat.dt.ravel()[flat]
    tol = max(2.0, 0.0025 * feat.D) if tol is None else tol
    good = (dt <= tol) & inside
    bad = (dt > 3 * tol) & inside
    return np.sum(step * good, 1) - 0.5 * np.sum(step * bad, 1)


def _hypotheses(feat: _Features, fam_t: tuple, fam_l: tuple, max_lines: int = 8,
                chunk: int = 60000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate line-quad × model-rectangle homographies → (H (N,3,3), score, ref (N,2)).

    Score = coarse length-weighted model/mask agreement (_line_score every
    0.5 m). A keypoint-count score was used before and is biased: it favours
    hypotheses that squeeze more of the court into the frame, and on striped
    floors (wood planks) nearly every intersection lands on some ridge — the
    correct camera for a cut-off court then fell out of the candidate pool.
    """
    c = np.array([feat.w / 2, feat.h / 2])
    vt, lt = fam_t
    vl, ll = fam_l
    T = _order_family(vt, sorted(lt, key=lambda g: -g.weight)[:max_lines], c)
    L = _order_family(vl, sorted(ll, key=lambda g: -g.weight)[:max_lines], c)
    empty = (np.zeros((0, 3, 3)), np.zeros(0), np.zeros((0, 2)))
    if len(T) < 2 or len(L) < 2:
        return empty
    X = np.cross(np.array([t.l for t in T])[:, None, :], np.array([m.l for m in L])[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        Pts = X[..., :2] / X[..., 2:3]
    far = ~np.all(np.isfinite(Pts), -1) | (np.linalg.norm(Pts - c, axis=-1) > 5 * feat.D)
    quads = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            for k in range(len(L)):
                for m in range(k + 1, len(L)):
                    if far[i, k] or far[j, k] or far[j, m] or far[i, m]:
                        continue
                    quads.append([Pts[i, k], Pts[j, k], Pts[j, m], Pts[i, m]])
    if not quads:
        return empty
    Hq = _square_to_quad(np.array(quads))
    Hq = Hq[np.all(np.isfinite(Hq.reshape(len(Hq), -1)), 1)]
    nm = len(_COMBO_A)
    out_H, out_s, out_r = [], [], []
    qstep = max(1, chunk // nm)
    for s in range(0, len(Hq), qstep):
        Hb = np.einsum("qij,mjk->qmik", Hq[s:s + qstep], _COMBO_A).reshape(-1, 3, 3)
        ref = np.tile(_COMBO_REF, (min(qstep, len(Hq) - s), 1))
        ok, _ = _plausible(Hb, ref, c[0], c[1], feat.D)
        if not ok.any():
            continue
        Hb, ref = Hb[ok], ref[ok]
        out_H.append(Hb)
        out_s.append(np.concatenate([_line_score(feat, Hb[i:i + 4000], _VCOARSE)
                                     for i in range(0, len(Hb), 4000)]))
        out_r.append(ref)
    if not out_H:
        return empty
    return np.concatenate(out_H), np.concatenate(out_s), np.concatenate(out_r)


def _match_lines(feat: _Features, p: np.ndarray, cx: float, cy: float, samples,
                 search: float):
    """For each visible model sample, search along the image normal for the
    nearest line ridge (strength-weighted centroid → sub-pixel).

    Returns (idx visible samples, found mask, target px (n,2), normals (n,2)).
    Runs touching the search window or much wider than the projected 40 mm
    line (line crossings, blobs) are rejected.
    """
    P, Dir, _ = samples
    uv, z = _project_params(p, P, cx, cy, return_z=True)
    uv2 = _project_params(p, P + 0.05 * Dir, cx, cy)
    Nw = np.column_stack([-Dir[:, 1], Dir[:, 0], np.zeros(len(Dir))])
    uw1 = _project_params(p, P + 0.5 * cm.LINE_WIDTH * Nw, cx, cy)
    uw0 = _project_params(p, P - 0.5 * cm.LINE_WIDTH * Nw, cx, cy)
    tan = uv2 - uv
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    wexp = np.abs(np.sum((uw1 - uw0) * nrm, 1))
    m = 2.0
    vis = (z > 0.1) & np.all(np.isfinite(uv), 1) & (uv[:, 0] >= m) & (uv[:, 1] >= m) \
        & (uv[:, 0] <= feat.w - 1 - m) & (uv[:, 1] <= feat.h - 1 - m)
    idx = np.flatnonzero(vis)
    n = len(idx)
    if n == 0:
        return idx, np.zeros(0, bool), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    # window = centroid search range + room for the (blurred) line itself
    S = int(np.ceil(search + min(float(np.max(wexp[idx])), 0.02 * feat.D) + 4))
    offs = np.arange(-S, S + 1, dtype=np.float32)
    X = (uv[idx, 0, None] + offs[None] * nrm[idx, 0, None]).astype(np.float32)
    Y = (uv[idx, 1, None] + offs[None] * nrm[idx, 1, None]).astype(np.float32)
    prof = cv2.remap(feat.strength, X, Y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    thr = 0.5 * _MIN_CONTRAST
    above = prof > thr
    start = above.copy()
    start[:, 1:] &= ~above[:, :-1]
    rid = np.cumsum(start, 1) * above
    Rn = int(rid.max()) + 1
    key = (np.arange(n)[:, None] * Rn + rid).ravel()


    def agg(v: np.ndarray) -> np.ndarray:
        return np.bincount(key, v, n * Rn).reshape(n, Rn)

    strong = agg((prof >= _MIN_CONTRAST).ravel().astype(float))
    # run width at half of the run's own peak (FWHM): robust to blur tails and
    # the multi-scale ridge filter's skirts; blobs and line crossings stay wide.
    # The sub-pixel centre is the centroid of the FWHM core only (skirts are
    # asymmetric next to other lines and would bias it).
    pk = np.zeros(n * Rn)
    np.maximum.at(pk, key, np.where(above, prof, 0.0).ravel())
    half = 0.5 * pk[key]
    core = above.ravel() & (prof.ravel() >= half)
    fw = agg(core.astype(float))
    wts = np.where(core, prof.ravel() - half + 1e-3, 0.0)
    ws = agg(wts)
    os_ = agg(wts * np.tile(offs, n))
    edge = np.zeros_like(above)
    edge[:, 0], edge[:, -1] = above[:, 0], above[:, -1]
    touch = agg(edge.ravel().astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        cen = os_ / ws
    ok = (fw > 0) & (strong > 0) & (touch == 0) & (fw <= 2 * wexp[idx, None] + 4) \
        & (np.abs(cen) <= search)
    ok[:, 0] = False
    d = np.where(ok, np.abs(cen), np.inf)
    j = np.argmin(d, 1)
    r = np.arange(n)
    found = np.isfinite(d[r, j])
    off = np.where(found, cen[r, j], 0.0)
    return idx, found, uv[idx] + off[:, None] * nrm[idx], nrm[idx], wexp[idx]


_ABSENT_FOUND = 0.15     # a model line with < 15 % of its samples on a ridge is "absent"
_ABSENT_MIN_W = 1.0      # ...counted only if its projected width is >= 1 px (clearly visible)
_ABSENT_MIN_LEN = 0.03   # ...and it spans >= 3 % of the image size inside the frame
_EXPLAIN_SIGMA = 1.0     # px, residual scale of the 'explained length' score


def _fit_stats(feat: _Features, p: np.ndarray, cx: float, cy: float, samples,
               search: float) -> dict:
    """Coverage / RMSE / matched-line statistics for camera params p.

    Besides what the camera explains (coverage, rmse, n_t/n_l = distinct
    transverse / longitudinal model lines matched), it measures the evidence
    AGAINST it: 'absent' model lines that project clearly into the frame
    (>= 1 px wide, long) but have no paint under them. A wrong assignment
    (court shifted by one line spacing, wrong focal) always predicts lines on
    bare floor; lines hidden by the net tape or players are rare in a
    temporal median.
    """
    P, _, ids = samples
    idx, found, target, nrm, wexp = _match_lines(feat, p, cx, cy, samples, search)
    if len(idx) == 0:
        return {"coverage": 0.0, "rmse": np.inf, "n_t": 0, "n_l": 0, "n_vis": 0, "n_inl": 0,
                "n_absent": 0, "absent_frac": 0.0, "t_straddle": False}
    uv = _project_params(p, P[idx], cx, cy)
    r = np.sum((uv - target) * nrm, 1)
    inl = found & (np.abs(r) < max(1.5, 0.0015 * feat.D))
    occ = _behind_net(p, P[idx], uv, cx, cy)
    xs, ys = set(), set()
    lines = list(cm.COURT_LINES.values())
    n_absent, absent_len, vis_len, explained = 0, 0.0, 0.0, 0.0
    lid = ids[idx]
    for k in range(len(lines)):
        sel = lid == k
        nv = int(sel.sum())
        if nv < 2:
            continue
        frac = float(inl[sel].mean())
        if nv >= 5 and frac >= 0.35:
            (a, b) = lines[k]
            (xs if _IS_TRANSVERSE[k] else ys).add(round(a[0] if _IS_TRANSVERSE[k] else a[1], 2))
        seg = np.linalg.norm(np.diff(uv[sel], axis=0), axis=1)
        seg = np.minimum(seg, 0.05 * feat.D)
        Lk = float(seg.sum())
        vis_len += Lk
        # explained length: projected line length whose ridge centre lies within
        # ~1 px of the model (Gaussian weight) — sharp where the DT score is blunt
        stp = np.zeros(nv)
        stp[:-1] += 0.5 * seg
        stp[1:] += 0.5 * seg
        rk = np.where(found[sel], r[sel], np.inf)
        explained += float(np.sum(stp * np.exp(-0.5 * (rk / _EXPLAIN_SIGMA) ** 2)))
        clear = sel & ~occ            # samples not seen through / behind the net band
        if clear.sum() < 2:
            continue
        seg = np.linalg.norm(np.diff(uv[clear], axis=0), axis=1)
        Lc = float(np.minimum(seg, 0.05 * feat.D).sum())
        if Lc >= _ABSENT_MIN_LEN * feat.D and np.median(wexp[clear]) >= _ABSENT_MIN_W \
                and float(found[clear].mean()) < _ABSENT_FOUND:
            n_absent += 1
            absent_len += Lc
    return {"coverage": float(inl.mean()),
            "rmse": float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf,
            "n_t": len(xs), "n_l": len(ys), "n_vis": int(len(idx)), "n_inl": int(inl.sum()),
            "n_absent": n_absent, "absent_frac": absent_len / max(vis_len, 1e-9),
            "explained": explained, "explained_frac": explained / max(vis_len, 1e-9),
            # matched transverse lines on BOTH sides of the net (x < 0 and x > 0)
            "t_straddle": any(x < 0 for x in xs) and any(x > 0 for x in xs)}


def _behind_net(p: np.ndarray, Pw: np.ndarray, uv: np.ndarray, cx: float, cy: float
                ) -> np.ndarray:
    """Floor samples on the far side of the net whose image falls inside the
    projected net band (mesh + tape, 0.65–1.7 m): their paint is hidden or
    confused with the white tape, so they are no evidence against a camera."""
    R = cv2.Rodrigues(np.asarray(p[1:4], float).reshape(3, 1))[0]
    C = -R.T @ np.asarray(p[4:7], float)
    W = cm.DOUBLES_HALF_WIDTH + 0.15
    q = _project_params(p, np.array([[0, -W, 0.65], [0, W, 0.65], [0, W, 1.7], [0, -W, 1.7]]),
                        cx, cy)
    far = np.sign(Pw[:, 0]) * np.sign(C[0]) < 0
    cr = np.stack([(q[(i + 1) % 4, 0] - q[i, 0]) * (uv[:, 1] - q[i, 1])
                   - (q[(i + 1) % 4, 1] - q[i, 1]) * (uv[:, 0] - q[i, 0]) for i in range(4)], 1)
    inside = np.all(cr > 0, 1) | np.all(cr < 0, 1)
    return far & inside


def _refine(feat: _Features, cam: CameraModel, search0: float, fit_k1: bool | str = "auto",
            n_iter: int = 4, samples=_FINE) -> tuple[CameraModel, dict] | None:
    """ICP-style refinement: normal search → robust LM (soft-L1) → shrink window."""
    P = samples[0]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    logf0 = np.log(_default_focal(cam.image_size))

    def solve(p: np.ndarray, search: float, iters: int) -> tuple[np.ndarray, float]:
        rmse = search
        for _ in range(iters):
            idx, found, tg, nn, _ = _match_lines(feat, p, cx, cy, samples, search)
            sel = idx[found]
            if len(sel) < 20:
                return p, np.inf
            Ps, tg, nn = P[sel], tg[found], nn[found]

            def res(q: np.ndarray) -> np.ndarray:
                d = np.sum((_project_params(q, Ps, cx, cy) - tg) * nn, 1)
                return np.append(d, 0.05 * (q[0] - logf0))

            sol = least_squares(res, p, loss="soft_l1", f_scale=1.0, max_nfev=50,
                                x_scale="jac")
            p = sol.x
            d = res(p)[:-1]
            inl = np.abs(d) < max(2.0, 0.5 * search)
            rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf
            search = float(np.clip(4 * rmse + 2.0, 2.5, search))
        return p, search

    p, search = solve(_cam_params(cam, with_k1=False), search0, n_iter)
    if not np.isfinite(search):
        return None
    stats = _fit_stats(feat, p, cx, cy, samples, search)
    if fit_k1:
        pk, sk = solve(np.append(p, 0.0), search, 2)
        if np.isfinite(sk) and abs(pk[7]) < 0.6:
            sk_stats = _fit_stats(feat, pk, cx, cy, samples, sk)
            # k1 must clearly pay for itself (a 0.9 ratio flip-flopped between
            # clips of the same fixed broadcast camera: 0.88 vs 0.91)
            better = sk_stats["rmse"] < 0.95 * stats["rmse"] and \
                sk_stats["coverage"] >= stats["coverage"] - 0.01 and \
                sk_stats.get("explained", 0.0) >= stats.get("explained", 0.0)
            if fit_k1 is True or better:
                p, search, stats = pk, sk, sk_stats
    out = _make_camera(np.exp(p[0]), p[1:4], p[4:7], cam.image_size,
                       k1=float(p[7]) if len(p) > 7 else 0.0, cx=cx, cy=cy)
    if out.center[2] <= 0.2:
        return None
    out.reproj_rmse_px = stats["rmse"]
    return out, stats


def _net(stats: dict) -> float:
    """Net line evidence: explained − unexplained visible model line length (px)."""
    e = stats.get("explained", 0.0)
    return 2.0 * e - e / max(stats.get("explained_frac", 0.0), 1e-9)


def _valid(stats: dict) -> bool:
    """Hard evidence gate for an automatic camera.

    >= 3 distinct transverse AND >= 3 longitudinal model lines matched (with
    only 2 lines of a family, the model spacing along that axis is never
    checked: any pair of the 6 transverse lines fits two image lines and the
    focal length absorbs the error), and no clearly visible model line lies on
    bare floor ('absent'). Measured on 30 synthetic behind views: every wrong
    candidate violated one of these, no correct one did.
    Plus: >= MIN_EXPLAINED of the visible model line length must have a ridge
    within ~1 px (correct synthetic cameras 0.74–0.98; the best camera found
    on random line grids / tennis courts 0.21–0.50).

    Exception n_t == 2 (low / close tripods: the near baseline and near long
    service line leave the frame, the far baseline and far long service line
    merge): accepted only with all 5 longitudinal lines matched (lateral scale
    and vanishing point fully checked), the two transverse lines on OPPOSITE
    sides of the net (a pair on one half lets the whole court slide along x) and
    coverage >= _N_T2_MIN_COVERAGE. On the venue sweep's 156 zero-absent n_t == 2
    candidates: every right one (floor error <= 0.15 m, 51/51) had n_l == 5,
    straddled the net and coverage 0.752–0.863; of the 24 wrong ones with n_l == 5
    that straddle, 4 have coverage >= 0.70 (none of which wins its case).
    """
    if stats.get("n_absent", 0) != 0 or stats.get("explained_frac", 1.0) < MIN_EXPLAINED \
            or not bool(np.isfinite(stats["rmse"])):
        return False
    if stats["n_t"] >= 3 and stats["n_l"] >= 3:
        return True
    return (stats["n_t"] == 2 and stats["n_l"] == 5 and bool(stats.get("t_straddle", False))
            and stats["coverage"] >= _N_T2_MIN_COVERAGE)


_N_T2_MIN_COVERAGE = 0.70   # right n_t == 2 cameras: 0.752–0.863 (sweep); 0.70 leaves headroom


def _confidence(stats: dict, margin: float, D: float) -> float:
    if not _valid(stats):
        return 0.0
    c = min(1.0, stats["explained_frac"] / 0.75) * np.exp(-stats["rmse"] / (0.004 * D))
    c *= min(1.0, margin / 0.3) ** 0.5
    c *= min(1.0, (stats["n_t"] + stats["n_l"]) / 6.0) ** 0.5
    return float(np.clip(c, 0.0, 1.0))


def _to_image(img_or_frames, image_size: tuple[int, int]) -> np.ndarray:
    img = _median_image(img_or_frames)
    w, h = image_size
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    return img


def refine_camera(camera: CameraModel, frames_or_median, fit_k1: bool | str = "auto",
                  search_px: float | None = None, **_: object) -> CameraModel:
    """Dense refinement of `camera` against detected court-line pixels.

    Minimises point-to-line distances between model line samples (every 10 cm)
    and sub-pixel ridge centres found along the image normal, over focal,
    rotation, translation (+ k1 if it helps), with a soft-L1 loss. Never
    returns a worse camera: falls back to the input if the fit degrades
    (coverage drops, fewer lines, or a model line lands on bare floor).
    Line paint model (colour images): the one that fits the lines best under `camera`
    (_paint_fit, first maximum in LINE_COLOURS order). A non-white model's refinement is kept
    only if its paint fit did not drop, else the white refinement is used (sweep2 colour grid
    #2186, light blue on a blue mat: cyan refine 2.2 → 24 px with fit 53.5 → 50.0; white
    refine 2.0 px). White-line courts: white wins (by 26-68) → exactly the old refinement.
    """
    img = _to_image(frames_or_median, camera.image_size)
    if img.ndim == 3:
        fit = _paint_fit(img, camera)
        best = max(LINE_COLOURS, key=lambda c: fit[c])
        if best != "white":
            cam = _refine_colour(camera, img, fit_k1, search_px, best)
            if cam is not camera and _paint_fit(img, cam)[best] >= fit[best]:
                return cam
    return _refine_colour(camera, img, fit_k1, search_px, "white")


def _refine_colour(camera: CameraModel, img: np.ndarray, fit_k1, search_px, colour: str) -> CameraModel:
    """refine_camera on one paint model's line evidence (the input camera if it degrades)."""
    feat = _Features(img, colour)
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    search = search_px or max(8.0, 0.012 * feat.D)
    before = _fit_stats(feat, _cam_params(camera, with_k1=bool(np.any(camera.dist[:1]))),
                        cx, cy, _FINE, max(3.0, 0.003 * feat.D))
    out = _refine(feat, camera, search, fit_k1=fit_k1)
    if out is None:
        return camera
    cam, stats = out
    if stats["coverage"] < before["coverage"] - 0.02 or stats["n_t"] < 2 or stats["n_l"] < 2 \
            or stats["n_absent"] > before["n_absent"]:
        return camera
    cam.confidence = max(camera.confidence, _confidence(stats, 1.0, feat.D))
    return cam


MIN_CONFIDENCE = 0.35
MIN_MARGIN = 0.05        # a distinct, equally valid camera within 5 % score → ambiguous → None
MIN_EXPLAINED = 0.55     # fraction of visible model line length explained within ~1 px
# Optional focal prior (EXIF / device intrinsics, `focal_hint_px`): a winning camera whose
# focal differs by more than this is rejected ("focal_mismatch"); it never changes WHICH
# camera wins, so a wrong hint cannot produce a wrong camera.
# Venue sweep: right cameras are <= 3.7 % off the true focal, the accepted wrong camera
# (case 3) 39 %, wrong n_t == 2 rivals 10–58 %. 0.15 leaves ~4x headroom for the hint's
# own error (video crop / stabilisation vs the lens EXIF; unmeasured — revisit on device).
FOCAL_HINT_TOL = 0.15

# diagnostics["reason"]: why _calibrate_image returned None ("ok" when it did not).
# Same strings in the Swift port (CourtCalibrationDiagnostics.reason).
REASONS = ("ok", "no_frames", "no_line_families", "no_hypotheses", "no_candidates",
           "evidence_gate", "focal_mismatch", "low_confidence", "ambiguous", "low_coverage",
           "implausible_focal", "high_rmse", "inconsistent_frames", "paint_mismatch")

# Single-frame fallback (the median image failed): a camera is accepted only when >= 2
# DIFFERENT sampled frames independently give it, agreeing within FRAME_AGREE_PX — the
# ShuttlPresent CourtLock definition: symmetric max projection distance of the 28 court
# keypoints + 2 net-post tops visible in either camera (CameraAgreement.maxDisagreementPx,
# agreePx = 3.0). On venue sweep case 3, frame 2 ALONE accepts a camera 3.35 m wrong.
FRAME_AGREE_PX = 3.0
FALLBACK_FRAMES = 5      # frames tried (at 1/2, 1/4, 3/4, 1/8, 7/8 of the list), early exit


def _focal_ok(cam: CameraModel, focal_hint: float | None) -> bool:
    return focal_hint is None or abs(cam.K[0, 0] / focal_hint - 1.0) <= FOCAL_HINT_TOL


# keypoint k of the 180°-rotated court ((x, y) → (−x, −y)) = keypoint _KP_180[k]
_KP_180 = np.array([int(np.argmin(np.linalg.norm(_KP + p, axis=1))) for p in _KP])


def _same_solution(ua: np.ndarray, ub: np.ndarray, tol: float) -> bool:
    """Two cameras' projected keypoints (_kp_image, same keypoint order) coincide: median
    distance between the SAME keypoints (or the 180° twin's) < tol, over >= 4 keypoints
    usable in both. A set comparison (nearest keypoint of any name) was used before: a court
    shifted by one line spacing (the long / short service lines are 3.96 m apart on both
    sides of the net) maps most keypoints onto other keypoints and passed as 'the same'
    (sweep2 #508: the true camera could then not veto a 3.96 m-shifted winner)."""
    best = np.inf
    for vb in (ub, ub[_KP_180]):
        ok = np.all(np.isfinite(ua), 1) & np.all(np.isfinite(vb), 1)
        if ok.sum() >= 4:
            best = min(best, float(np.median(np.linalg.norm(ua[ok] - vb[ok], axis=1))))
    return bool(best < tol)


N_CHOSEN = 16            # distinct hypotheses refined, in COARSE line-score order
N_CHOSEN_SHARP = 8       # ... plus up to this many more in 1 px-tolerance line-score order


def _choose_hypotheses(feat: _Features, H: np.ndarray) -> list[int]:
    """Distinct hypotheses to refine: the first N_CHOSEN distinct ones by the COARSE line
    score (DT tolerance max(2, 0.25 % D)), then up to N_CHOSEN_SHARP more by the same score
    at ~1 px tolerance. Dense ridge bands (far lines of adjacent courts, net tapes, boards
    near the horizon; wood planks) give the coarse score to edge-on courts fitted into them:
    R7 doubles — right hypothesis coarse rank #130 (distinct #50), 1 px rank #3. Sweep2 #56 /
    #1937 / #2034: right one distinct #14 / #15 / #9 (8 were refined before)."""
    ls = _line_score(feat, H)
    ls_sharp = _line_score(feat, H, tol=max(1.0, 0.0005 * feat.D))
    chosen: list[int] = []
    kps: list[np.ndarray] = []
    tol = max(3.0, 0.004 * feat.D)

    def pick(score: np.ndarray, limit: int) -> None:
        for i in np.argsort(-score, kind="stable")[:300]:
            if score[i] <= 0 or len(chosen) >= limit:
                break
            u = _kp_image(H[i], feat)
            if int(i) not in chosen and all(not _same_solution(u, k, tol) for k in kps):
                chosen.append(int(i))
                kps.append(u)
    pick(ls, N_CHOSEN)
    pick(ls_sharp, N_CHOSEN + N_CHOSEN_SHARP)
    return chosen


def _quick_candidates(feat: _Features, H: np.ndarray, REF: np.ndarray, chosen: list[int],
                      image_size: tuple[int, int]) -> list[tuple[float, CameraModel, dict]]:
    """Quick refinement of each chosen hypothesis → (explained, camera, stats), ranked later by
    EXPLAINED line length (ridge within ~1 px of the model; the DT-tolerance score that picks
    hypotheses is too blunt to rank refined cameras: a wrong telephoto camera fitting 3 lines
    loosely out-scored the right one, rmse 1.4 vs 0.4). Refined from two search windows —
    the wide one (0.5 % D) and the check window (0.3 % D) — keeping the one explaining more:
    far lines a few px apart let the wide window pull a near-true start into the wrong basin
    (sweep2 #1192: 0.33 m start → 1.98 m wide, 0.025 m narrow)."""
    s_chk = max(3.0, 0.003 * feat.D)
    cands: list[tuple[float, CameraModel, dict]] = []
    for i in chosen:
        cam = _camera_from_homography(H[i], image_size, REF[i])
        if cam is None:
            continue
        best = None
        for search in (max(4.0, 0.005 * feat.D), s_chk):
            out = _refine(feat, cam, search, fit_k1=False, n_iter=2, samples=_MEDIUM)
            if out is None:
                continue
            c = out[0]
            st = _fit_stats(feat, _cam_params(c, False), c.K[0, 2], c.K[1, 2], _FINE, s_chk)
            if best is None or st.get("explained", 0.0) > best[0]:
                best = (st.get("explained", 0.0), c, st)
        if best is not None:
            cands.append(best)
    return cands


def _kp_image(H: np.ndarray, feat: _Features) -> np.ndarray:
    uv, w = _project_H(H[None], _KP)
    uv, w = uv[0], w[0]
    m = 0.5 * feat.D
    bad = (w <= 0) | ~np.all(np.isfinite(uv), 1) | (uv[:, 0] < -m) | (uv[:, 1] < -m) \
        | (uv[:, 0] > feat.w + m) | (uv[:, 1] > feat.h + m)
    uv[bad] = np.nan
    return uv


def _calibrate_image(img: np.ndarray, image_size: tuple[int, int],
                     diag: dict | None = None, focal_hint: float | None = None,
                     veto_dominated: bool = True, colour: str = "white") -> CameraModel | None:
    """Automatic calibration of one (median) image; None when not confident.
    `colour`: line paint model (LINE_COLOURS) of the line evidence.
    `diag["reason"]` is one of REASONS. `veto_dominated` (default on, median and single
    frames): reject a dominated winner — see the comment at the veto ("ambiguous",
    diag["dominated_by"] = the dominating rival's explained length)."""
    def fail(reason: str) -> None:
        if diag is not None:
            diag["reason"] = reason
        return None

    feat = _Features(img, colour)
    lines = _merge_segments(_segments(feat), feat.D)
    fams = [f for f in _vp_families(lines) if len(f[1]) >= 2]
    if len(fams) < 2:
        return fail("no_line_families")
    Hs, Ss, Rs = [], [], []
    for a in range(len(fams)):
        for b in range(len(fams)):
            if a != b:
                H, s, r = _hypotheses(feat, fams[a], fams[b])
                Hs.append(H), Ss.append(s), Rs.append(r)
    H, S, REF = np.concatenate(Hs), np.concatenate(Ss), np.concatenate(Rs)
    if len(H) == 0:
        return fail("no_hypotheses")
    top = np.argsort(-S, kind="stable")[:2000]
    H, REF = H[top], REF[top]
    chosen = _choose_hypotheses(feat, H)
    s_chk = max(3.0, 0.003 * feat.D)
    tol = max(3.0, 0.004 * feat.D)
    cands = _quick_candidates(feat, H, REF, chosen, image_size)
    if not cands:
        return fail("no_candidates")
    cands.sort(key=lambda x: -x[0])
    # full refinement of the top 3 and of every close contender, so the margin
    # compares fully refined cameras (a half-refined rival would look weaker)
    finals: list[tuple[float, CameraModel, dict]] = []
    others: list[tuple[float, CameraModel, dict]] = []
    for k, (e0, c0, st0) in enumerate(cands):
        if k >= 3 and (e0 < 0.7 * cands[0][0] or len(finals) >= 6):
            others.append((e0, c0, st0))
            continue
        out = _refine(feat, c0, s_chk, fit_k1="auto", n_iter=4)
        if out is None:
            continue
        c, st = out
        finals.append((st.get("explained", 0.0), c, st))
    # valid cameras ranked by NET evidence = explained − unexplained visible model line length
    # (2·explained − explained / explained_frac). Explained length alone rewards a camera that
    # puts more model line into the frame: sweep2 #1426, a court shifted 3.96 m whose near
    # long-service line sits on a colour-blend ridge at a mat edge (explained 5280 vs the right
    # camera's 4828, net 2285 vs 3644); R7 doubles_s16, a telephoto camera fitted into the
    # horizon band (6418 vs 6313, net 1477 vs 5321). Sweep2 (2156): FA 1 → 0, #1426 right,
    # lost #406 (right camera 0.125 m, margin 0.12 → 0 → low_confidence).
    valid = sorted([(_net(x[2]), x[1], x[2]) for x in finals if _valid(x[2])], key=lambda x: -x[0])
    if diag is not None:
        diag.update(n_hyp=int(len(S)), n_lines=len(lines), n_valid=len(valid))
        if finals:
            diag.update(max(finals, key=lambda x: x[0])[2])
        if diag.get("keep_candidates"):
            diag["candidates"] = [(s, c, st, _confidence(st, 1.0, feat.D))
                                  for s, c, st in finals + others]
    if not valid:
        return fail("evidence_gate")
    s1, cam, stats = valid[0]
    # margin against DISTINCT alternatives that also pass the evidence gate:
    # wrong cameras that leave model lines on bare floor are no ambiguity
    kb = _kp_image(cam.floor_homography(), feat)
    s2 = 0.0
    for s, c, st in valid[1:] + [(_net(x[2]), x[1], x[2]) for x in others if _valid(x[2])]:
        if not _same_solution(kb, _kp_image(c.floor_homography(), feat), tol):
            s2 = max(s2, s)
    margin = 1.0 if s1 <= 0 else float(np.clip(1 - s2 / s1, 0, 1))
    conf = _confidence(stats, margin, feat.D)
    f = cam.K[0, 0]
    if diag is not None:
        diag.update(stats, margin=margin, confidence=conf, score=s1, score2=s2)
    # Dominated winner → "ambiguous" (diag["dominated_by"] = the rival's explained length):
    # a DISTINCT final camera explains more line length but failed the evidence gate on one
    # statistic — the right camera, typically: single frames (sweep case 3, frames 2-4:
    # coverage 0.62-0.69 < 0.70) and low tripods on the median (sweep2 #384/#421/#508/#576:
    # right camera n_t == 2 without a net straddle — the far lines are sub-pixel — expl
    # 6220-6627 vs the valid winner's 4693-5689, 2-13.6 m off). Only rivals with NO visible
    # model line on bare floor count: n_absent > 0 is evidence against the rival, not a near
    # miss (1080p zoom clip: the right winner, conf 0.70, was vetoed by rivals with 2-3 absent
    # lines explaining 5777-5967 vs 5477).
    dominated = 0.0
    e1 = stats.get("explained", 0.0)
    if veto_dominated:
        for s, c, st in finals:
            if s > e1 and not _valid(st) and st.get("n_absent", 0) == 0 \
                    and not _same_solution(kb, _kp_image(c.floor_homography(), feat), tol):
                dominated = max(dominated, s)
        if diag is not None:
            diag["dominated_by"] = dominated
    # the focal prior is a VETO on the winner, never a selector: a wrong hint can only turn
    # an accept into a reject (as a selector, 0.6x the true focal picked a 3.4 m-off camera)
    for bad, reason in ((not _focal_ok(cam, focal_hint), "focal_mismatch"), (dominated > 0, "ambiguous"),
                        (conf < MIN_CONFIDENCE, "low_confidence"), (margin < MIN_MARGIN, "ambiguous"),
                        (stats["coverage"] < 0.4, "low_coverage"),
                        (not (0.25 * feat.D < f < 5 * feat.D), "implausible_focal"),
                        (stats["rmse"] > max(1.0, 0.0007 * feat.D), "high_rmse")):
        if bad:
            return fail(reason)
    if cam.center[0] > 0:
        cam = _flip_180(cam)
    cam.confidence = conf
    if diag is not None:
        diag["reason"] = "ok"
        diag["far_m_per_px"] = far_m_per_px(cam)      # precision diagnostic (accepted cameras)
    return cam


def calibrate_from_frames(frames: list[np.ndarray], image_size: tuple[int, int],
                          doubles_hint: bool | None = None,
                          diagnostics: dict | None = None,
                          focal_hint_px: float | None = None) -> CameraModel | None:
    """Fully automatic calibration from frames sampled across the video.

    Temporal median (removes moving players) → line mask → segments → two
    vanishing-point families → line-quad × model-rectangle homography
    hypotheses (vectorised, ordering + chirality + camera-plausibility
    constraints) → model/mask agreement scoring → dense refinement.
    Returns None when not confident (never a guess). If the median image fails
    (e.g. camera moved, a player standing still on a line) and there are > 2 frames,
    single frames are tried; their camera is accepted only when >= 2 different frames
    give it within FRAME_AGREE_PX ("inconsistent_frames" otherwise).
    `doubles_hint` is accepted for API compatibility: singles and doubles
    lines are both always painted, so the full model is always used.
    `focal_hint_px`: optional focal length in pixels at `image_size` (EXIF / device
    intrinsics); the winning camera is rejected ("focal_mismatch") when more than
    FOCAL_HINT_TOL off it — a veto only, it never selects a different camera.
    `diagnostics["reason"]` (one of REASONS) says why the reported attempt failed.
    """
    frames = [f for f in (frames or []) if f is not None]
    if not frames:
        if diagnostics is not None:
            diagnostics["reason"] = "no_frames"
        return None
    median = _to_image(frames, image_size)
    first: dict | None = None
    # line paint colour models in order (white first: white-line courts behave exactly as
    # before); a colour model is tried only when every earlier one gave no camera. Colour
    # images only (a grey image has no hue to model).
    for colour in (LINE_COLOURS if median.ndim == 3 else ("white",)):
        cam, d = _calibrate_colour(frames, median, image_size, diagnostics, focal_hint_px, colour)
        if first is None:
            first = d
        if cam is not None:
            if median.ndim == 3:
                cam, d = _check_paint(frames, median, image_size, diagnostics, focal_hint_px, colour, cam, d)
            break
        # a focal veto ends the search: no other attempt may win by agreeing with the hint
        if d.get("reason") == "focal_mismatch":
            break
    if cam is None and d.get("reason") != "paint_mismatch":
        d = first                            # rejected: report the white-model attempt
    if diagnostics is not None:              # the attempt that succeeded, else the median
        diagnostics.update(d)
    return cam


def _paint_fit(img: np.ndarray, cam: CameraModel) -> dict[str, float]:
    """How well each paint colour model fits the lines under `cam`: per model, the median over
    the visible model-line samples (FINE; projected line >= 1 px wide, sample and both floor
    points 0.25 m either side inside the image) of the model's score on the line minus the
    larger of its two floor scores (nearest pixel). -inf with < 20 usable samples.
    Venue sweep + sweep2 colour grid + R7: the painted colour wins in every case checked,
    white by >= 26 over the next model on white-line scenes."""
    P, Dir, _ = _FINE
    nrm = np.column_stack([-Dir[:, 1], Dir[:, 0], np.zeros(len(Dir))])
    uv, a, b = cam.project(P), cam.project(P + 0.25 * nrm), cam.project(P - 0.25 * nrm)
    w1 = cam.project(P + 0.5 * cm.LINE_WIDTH * nrm)
    w0 = cam.project(P - 0.5 * cm.LINE_WIDTH * nrm)
    h, w = img.shape[:2]

    def inside(x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.all(np.isfinite(x), 1) & (x[:, 0] >= 0) & (x[:, 1] >= 0) \
                & (x[:, 0] <= w - 1) & (x[:, 1] <= h - 1)
    ok = inside(uv) & inside(a) & inside(b) & (np.linalg.norm(w1 - w0, axis=1) >= 1.0)
    if ok.sum() < 20:
        return {c: -np.inf for c in LINE_COLOURS}
    idx = [(np.floor(x[ok, 1] + 0.5).astype(np.int64), np.floor(x[ok, 0] + 0.5).astype(np.int64))
           for x in (uv, a, b)]
    out = {}
    for c in LINE_COLOURS:
        W = _whiteness(img, c)
        L, A, B = (W[r, q] for r, q in idx)
        out[c] = float(np.median(np.minimum(L - A, L - B)))
    return out


def _check_paint(frames, median, image_size, diagnostics, focal_hint_px, colour, cam, d):
    """An accepted camera's paint model must be the one that fits its lines: when another
    model's _paint_fit beats it by > _MIN_CONTRAST, calibrate with that model instead and
    use its camera ("paint_mismatch" when it gives none); when that model is white (which
    was tried first and gave nothing) the camera is dropped and the white rejection stands. A weaker model sees the paint
    faintly and can pass the gate with a worse camera (sweep2 colour grid #2219: red lines
    accepted by the yellow model 0.145 m off, red model 0.008 m; #2221: red on a blue mat by
    the white model 0.102 m off, red model 0.011 m)."""
    fit = _paint_fit(median, cam)
    best = max(LINE_COLOURS, key=lambda c: fit[c])        # first maximum in LINE_COLOURS order
    fit_json = {c: v for c, v in fit.items() if np.isfinite(v)}      # diagnostics go to JSON
    d["paint_fit"] = fit_json
    if best == colour or not fit[best] > fit[colour] + _MIN_CONTRAST:
        return cam, d
    if best == "white":
        # white paint, and the white model already gave no camera: a colour model seeing the
        # white lines faintly must not overrule it (sweep2 #1426: white 'ambiguous') — the
        # white attempt's rejection stands
        d["reason"] = "paint_white"
        return None, d
    cam2, d2 = _calibrate_colour(frames, median, image_size, diagnostics, focal_hint_px, best)
    d2["paint_fit"] = fit_json
    if cam2 is None:
        d2["reason"] = "paint_mismatch"
        d2["line_colour"] = best
        return None, d2
    return cam2, d2


def _calibrate_colour(frames: list[np.ndarray], median: np.ndarray, image_size: tuple[int, int],
                      diagnostics: dict | None, focal_hint_px: float | None, colour: str
                      ) -> tuple[CameraModel | None, dict]:
    """One line colour model: the median image, then the single-frame fallback."""
    d = dict(diagnostics or {})              # carries request flags (keep_candidates)
    d["source"] = "median"
    cam = _calibrate_image(median, image_size, d, focal_hint=focal_hint_px, colour=colour)
    # a focal veto ends the search: no single frame may win by agreeing with the hint
    if cam is None and len(frames) > 2 and d.get("reason") != "focal_mismatch":
        cam, fd = _frame_fallback(frames, image_size, diagnostics, focal_hint_px, colour)
        if cam is not None:
            d = fd
        else:
            d["n_frame_accepts"] = fd["n_frame_accepts"]
            d["frames_tried"] = fd["frames_tried"]
            if fd["n_frame_accepts"] > 0:    # some frame gave a camera no other frame confirms
                d["reason"] = "inconsistent_frames"
    if cam is not None and colour != "white":
        d["line_colour"] = colour            # advice / evidence: which paint model found it
    return cam, d


def _fallback_indices(n: int) -> list[int]:
    """Frames tried after the median fails: 1/2, 1/4, 3/4, 1/8, 7/8 of the list (distinct)."""
    out: list[int] = []
    for num, den in ((1, 2), (1, 4), (3, 4), (1, 8), (7, 8)):
        i = (n * num) // den
        if i not in out:
            out.append(i)
    return out[:FALLBACK_FRAMES]


def _camera_agreement_px(a: CameraModel, b: CameraModel) -> float:
    """ShuttlPresent CameraAgreement.maxDisagreementPx: max distance between the two
    cameras' projections of the court keypoints + net-post tops, over the points inside
    either image (inf when a point visible in one camera does not project in the other)."""
    P = np.vstack([cm.keypoints_array()[1], cm.NET_POST_TOPS])

    def one(x: CameraModel, y: CameraModel) -> float:
        w, h = x.image_size
        u, v = x.project(P), y.project(P)
        vis = np.all(np.isfinite(u), 1) & (u[:, 0] >= 0) & (u[:, 1] >= 0) & (u[:, 0] <= w) & (u[:, 1] <= h)
        if not vis.any():
            return np.inf
        dd = np.where(np.all(np.isfinite(v[vis]), 1), np.linalg.norm(u[vis] - v[vis], axis=1), np.inf)
        return float(dd.max())

    return max(one(a, b), one(b, a))


def far_m_per_px(cam: CameraModel, h: float = 1e-3) -> float:
    """Precision diagnostic: floor metres per image pixel at the far-baseline centre (the
    baseline on the side of the net away from the camera), worst direction =
    1 / smallest singular value of d(u, v)/d(X, Y) there. Central differences of
    `cam.project` (includes k1); closed-form 2x2 singular value (same arithmetic in Swift).
    A right camera's far keypoint error of e px is up to e·far_m_per_px metres on the floor
    (0.8 m high, 5 m back: ~0.3 m/px → landing calls ~±10 cm)."""
    xf = cm.HALF_LENGTH if cam.center[0] < 0 else -cm.HALF_LENGTH
    P = np.array([[xf + h, 0.0, 0.0], [xf - h, 0.0, 0.0], [xf, h, 0.0], [xf, -h, 0.0]])
    uv = cam.project(P)
    if not np.all(np.isfinite(uv)):
        return float("inf")
    a, c = (uv[0] - uv[1]) / (2 * h)            # d(u, v)/dX
    b, d = (uv[2] - uv[3]) / (2 * h)            # d(u, v)/dY
    s = a * a + b * b + c * c + d * d
    det = abs(a * d - b * c)
    smax = np.sqrt(0.5 * s + np.sqrt(max(0.25 * s * s - det * det, 0.0)))
    return float(smax / det) if det > 0 else float("inf")      # 1 / s_min = s_max / |det|


def _frame_fallback(frames: list[np.ndarray], image_size: tuple[int, int], diagnostics: dict | None,
                    focal_hint: float | None, colour: str = "white") -> tuple[CameraModel | None, dict]:
    """Calibrate single frames until two DIFFERENT frames agree (earliest pair in try
    order; the earlier frame's camera is returned). The focal veto applies per frame.
    Stops as soon as a pair can no longer be reached."""
    idx = _fallback_indices(len(frames))
    accepted: list[tuple[int, CameraModel, dict]] = []
    tried: list[int] = []
    for k, i in enumerate(idx):
        if len(accepted) + (len(idx) - k) < 2:
            break
        tried.append(i)
        d = dict(diagnostics or {})
        d["source"] = f"frame{i}"
        cam = _calibrate_image(_to_image([frames[i]], image_size), image_size, d, focal_hint=focal_hint,
                               veto_dominated=True, colour=colour)
        if cam is None:
            continue
        for j, cj, dj in accepted:
            px = _camera_agreement_px(cj, cam)
            if px <= FRAME_AGREE_PX:
                dj.update(frames_agree=[j, i], frame_agree_px=px, n_frame_accepts=len(accepted) + 1,
                          frames_tried=tried)
                return cj, dj
        accepted.append((i, cam, d))
    return None, {"n_frame_accepts": len(accepted), "frames_tried": tried}
