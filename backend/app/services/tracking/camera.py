"""
Pinhole camera model shared by calibration, 3-D reconstruction, simulation.

Pixel convention everywhere in the tracking stack: full-resolution pixels of
the upright (rotation-applied) video frame, origin top-left, x right, y down.
World frame: see court_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray                    # 3x3 intrinsics
    R: np.ndarray                    # 3x3 world→camera rotation
    t: np.ndarray                    # (3,) world→camera translation
    image_size: tuple[int, int]      # (width, height)
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # OpenCV k1 k2 p1 p2 k3
    reproj_rmse_px: float = float("nan")   # calibration fit quality
    confidence: float = 0.0                # 0..1, calibration self-assessed

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """(N,3) world → (N,2) pixels. Points behind the camera give NaN."""
        pts = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
        cam = pts @ self.R.T + self.t
        z = cam[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = cam[:, :2] / z
        if np.any(self.dist):
            xy = _distort(xy, self.dist)
        uv = xy @ self.K[:2, :2].T + self.K[:2, 2]
        uv[(z[:, 0] <= 1e-9)] = np.nan
        return uv

    def ray(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N,2) → (origin (3,), unit directions (N,3)) in world."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        K_inv, R_T_K_inv, centre, distorted = self._ray_cache()
        if distorted:
            xy = (uv - self.K[:2, 2]) @ K_inv[:2, :2].T
            xy = _undistort(xy, self.dist)
            d = xy @ self.R[:2] + self.R[2]
        else:
            d = uv @ R_T_K_inv[:, :2].T + R_T_K_inv[:, 2]
        return centre, d / np.linalg.norm(d, axis=1, keepdims=True)

    def _ray_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        # Keyed on the array contents, so in-place edits of K/R/t/dist invalidate it.
        key = (self.K.tobytes(), self.R.tobytes(), np.asarray(self.t).tobytes(),
               np.asarray(self.dist).tobytes())
        cached = self.__dict__.get("_ray_cached")
        if cached is None or cached[0] != key:
            K_inv = np.linalg.inv(self.K)
            cached = (key, (K_inv, self.R.T @ K_inv, -self.R.T @ self.t, bool(np.any(self.dist))))
            self.__dict__["_ray_cached"] = cached
        return cached[1]

    def pixel_to_floor(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the plane Z=z. (N,2) → (N,3); NaN if parallel/behind."""
        o, d = self.ray(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - o[2]) / d[:, 2]
        pts = o + d * s[:, None]
        pts[~(s > 0)] = np.nan
        return pts

    def floor_homography(self) -> np.ndarray:
        """3x3 H mapping floor (X, Y, 1) → pixels (ignores lens distortion)."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return H / H[2, 2]

    @staticmethod
    def look_at(eye: np.ndarray, target: np.ndarray, focal_px: float,
                image_size: tuple[int, int], roll_deg: float = 0.0) -> "CameraModel":
        """Build a camera at `eye` looking at `target` (world up = +z)."""
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.vstack([right, down, fwd])          # rows = camera axes in world
        if roll_deg:
            c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
        w, h = image_size
        K = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1.0]])
        return CameraModel(K=K, R=R, t=-R @ eye, image_size=image_size)

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist(),
                "dist": self.dist.tolist(), "image_size": list(self.image_size),
                "reproj_rmse_px": self.reproj_rmse_px, "confidence": self.confidence}

    @staticmethod
    def from_dict(d: dict) -> "CameraModel":
        return CameraModel(K=np.array(d["K"]), R=np.array(d["R"]), t=np.array(d["t"]),
                           dist=np.array(d.get("dist", np.zeros(5))),
                           image_size=tuple(d["image_size"]),
                           reproj_rmse_px=d.get("reproj_rmse_px", float("nan")),
                           confidence=d.get("confidence", 0.0))


def _distort(xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([xd, yd])


def _undistort(xy_d: np.ndarray, dist: np.ndarray, iters: int = 8) -> np.ndarray:
    xy = xy_d.copy()
    for _ in range(iters):   # fixed-point inversion, adequate for phone lenses
        xy = xy + (xy_d - _distort(xy, dist))
    return xy
