"""
BWF badminton court geometry — the single source of truth for every module.

World frame (metres, right-handed):
  origin  = court centre on the floor, directly under the net
  +x      = along the court length, towards the FAR baseline (away from a camera
            placed behind the near baseline). Near half: x < 0, far half: x > 0.
  +y      = across the court, pointing LEFT as seen from behind the near baseline
            looking at the net (right-handed: x forward, y left, z up)
  +z      = up
Lines are 40 mm wide and belong to the area they bound (a shuttle touching the
line is IN). Coordinates below are line centre-lines; use LINE_WIDTH when judging.
"""

from __future__ import annotations

import numpy as np

HALF_LENGTH = 6.70            # net → baseline
DOUBLES_HALF_WIDTH = 3.05
SINGLES_HALF_WIDTH = 2.59
SHORT_SERVICE_X = 1.98        # net → short service line
DOUBLES_LONG_SERVICE_X = 5.94 # net → doubles long service line (0.76 inside baseline)
LINE_WIDTH = 0.04
NET_HEIGHT_CENTRE = 1.524
NET_HEIGHT_POST = 1.55

COURT_LENGTH = 2 * HALF_LENGTH        # 13.40
DOUBLES_WIDTH = 2 * DOUBLES_HALF_WIDTH  # 6.10
SINGLES_WIDTH = 2 * SINGLES_HALF_WIDTH  # 5.18

# Every painted line segment as ((x1, y1), (x2, y2)) on the floor (z = 0).
_L, _W, _S = HALF_LENGTH, DOUBLES_HALF_WIDTH, SINGLES_HALF_WIDTH
COURT_LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_baseline":        ((-_L, -_W), (-_L, _W)),
    "far_baseline":         ((_L, -_W), (_L, _W)),
    "near_long_service":    ((-DOUBLES_LONG_SERVICE_X, -_W), (-DOUBLES_LONG_SERVICE_X, _W)),
    "far_long_service":     ((DOUBLES_LONG_SERVICE_X, -_W), (DOUBLES_LONG_SERVICE_X, _W)),
    "near_short_service":   ((-SHORT_SERVICE_X, -_W), (-SHORT_SERVICE_X, _W)),
    "far_short_service":    ((SHORT_SERVICE_X, -_W), (SHORT_SERVICE_X, _W)),
    "left_doubles_side":    ((-_L, -_W), (_L, -_W)),
    "right_doubles_side":   ((-_L, _W), (_L, _W)),
    "left_singles_side":    ((-_L, -_S), (_L, -_S)),
    "right_singles_side":   ((-_L, _S), (_L, _S)),
    "near_centre":          ((-_L, 0.0), (-SHORT_SERVICE_X, 0.0)),
    "far_centre":           ((SHORT_SERVICE_X, 0.0), (_L, 0.0)),
}

# Named floor keypoints = every line intersection (x, y). Stable names so
# detectors, calibration and tests agree.
_XS = {"nb": -_L, "nl": -DOUBLES_LONG_SERVICE_X, "ns": -SHORT_SERVICE_X,
       "fs": SHORT_SERVICE_X, "fl": DOUBLES_LONG_SERVICE_X, "fb": _L}
_YS = {"ld": -_W, "ls": -_S, "rs": _S, "rd": _W}
COURT_KEYPOINTS: dict[str, tuple[float, float]] = {
    f"{xn}_{yn}": (xv, yv) for xn, xv in _XS.items() for yn, yv in _YS.items()
}
for _xn in ("nb", "ns", "fs", "fb"):
    COURT_KEYPOINTS[f"{_xn}_c"] = (_XS[_xn], 0.0)

# Net posts stand on the doubles side lines; top of posts is a 3-D keypoint.
NET_POST_TOPS = np.array([[0.0, -_W, NET_HEIGHT_POST], [0.0, _W, NET_HEIGHT_POST]])


def keypoints_array(names: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Return (names, Nx3 world points with z=0)."""
    names = names or list(COURT_KEYPOINTS)
    pts = np.array([[*COURT_KEYPOINTS[n], 0.0] for n in names], dtype=np.float64)
    return names, pts


def side_of(x: float) -> str:
    """Which half a floor x coordinate lies in: 'near' (x<0) or 'far' (x>0)."""
    return "near" if x < 0 else "far"


def in_out_margin(x: float, y: float, doubles: bool = False) -> float:
    """Signed distance (m) from the outer edge of the boundary lines.

    Positive = inside (line counts as in), negative = outside. Uses the singles
    or doubles side lines; baselines are shared for rally play.
    """
    half_w = DOUBLES_HALF_WIDTH if doubles else SINGLES_HALF_WIDTH
    edge_x = HALF_LENGTH + LINE_WIDTH / 2
    edge_y = half_w + LINE_WIDTH / 2
    return float(min(edge_x - abs(x), edge_y - abs(y)))
