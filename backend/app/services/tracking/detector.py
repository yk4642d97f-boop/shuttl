"""Per-frame shuttlecock detection.

Backends (all emit top-k ``ShuttleCandidate`` per frame in FULL-RES upright px):

* ``tracknetv3`` — TrackNetV3 (Chen et al. 2023, github qaz812345/TrackNetV3, MIT).
  U-Net, input = median background + 8 RGB frames (27 ch), output 8 heatmaps.
* ``wasb`` — WASB-SBDT HRNet (Tarashima et al., BMVC 2023, github nttcom/WASB-SBDT, MIT),
  badminton weights. Input 3 RGB frames (ImageNet norm), output 3 heatmap logits.
* ``motion`` — weight-free fallback: running-median background + 3-frame differencing.

Architectures are faithful ports; official checkpoints load with ``strict=True``.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.services.tracking.data_types import ShuttleCandidate, VideoMeta
from app.services.tracking.video_io import iter_frames

log = logging.getLogger(__name__)

NET_W, NET_H = 512, 288
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "ml" / "weights"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ProgressFn = Callable[[float], None]
FrameSource = Callable[[], Iterable[tuple[int, np.ndarray]]]   # re-iterable BGR frames


# ══════════════════════════════════════════════════════════════════════════════
# TrackNetV3 (module/parameter names match the official checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

class _Conv2DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _ConvStack(nn.Module):
    """``n`` × Conv2DBlock with attributes conv_1..conv_n (official naming)."""

    def __init__(self, in_dim: int, out_dim: int, n: int):
        super().__init__()
        self.n = n
        for i in range(n):
            setattr(self, f"conv_{i + 1}", _Conv2DBlock(in_dim if i == 0 else out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n):
            x = getattr(self, f"conv_{i + 1}")(x)
        return x


class TrackNetV3(nn.Module):
    """TrackNetV3 tracking module. In: (N, 3*(S+1), 288, 512) in [0,1], background first.
    Out: (N, S, 288, 512) sigmoid heatmaps."""

    def __init__(self, in_dim: int = 27, out_dim: int = 8):
        super().__init__()
        self.down_block_1 = _ConvStack(in_dim, 64, 2)
        self.down_block_2 = _ConvStack(64, 128, 2)
        self.down_block_3 = _ConvStack(128, 256, 3)
        self.bottleneck = _ConvStack(256, 512, 3)
        self.up_block_1 = _ConvStack(768, 256, 3)
        self.up_block_2 = _ConvStack(384, 128, 2)
        self.up_block_3 = _ConvStack(192, 64, 2)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(self.pool(x1))
        x3 = self.down_block_3(self.pool(x2))
        x = self.bottleneck(self.pool(x3))
        x = self.up_block_1(torch.cat([self.up(x), x3], dim=1))
        x = self.up_block_2(torch.cat([self.up(x), x2], dim=1))
        x = self.up_block_3(torch.cat([self.up(x), x1], dim=1))
        return self.sigmoid(self.predictor(x))


# ══════════════════════════════════════════════════════════════════════════════
# WASB HRNet (config src/configs/model/wasb.yaml; names match official checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def _cbr(i: int, o: int, k: int, s: int, relu: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _HRModule(nn.Module):
    """HighResolutionModule with BasicBlock branches and SUM fusion."""

    def __init__(self, n_blocks: list[int], in_ch: list[int], ch: list[int]):
        super().__init__()
        nb = len(ch)
        self.num_branches = nb
        branches = []
        for b in range(nb):
            ds = None if in_ch[b] == ch[b] else _cbr(in_ch[b], ch[b], 1, 1, relu=False)
            blocks = [_BasicBlock(in_ch[b], ch[b], 1, ds)]
            blocks += [_BasicBlock(ch[b], ch[b]) for _ in range(1, n_blocks[b])]
            branches.append(nn.Sequential(*blocks))
        self.branches = nn.ModuleList(branches)
        self.fuse_layers = None
        if nb > 1:
            fuse = []
            for i in range(nb):
                row: list[nn.Module | None] = []
                for j in range(nb):
                    if j > i:
                        row.append(nn.Sequential(
                            nn.Conv2d(ch[j], ch[i], 1, 1, 0, bias=False), nn.BatchNorm2d(ch[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest")))
                    elif j == i:
                        row.append(None)
                    else:
                        convs = [_cbr(ch[j], ch[j], 3, 2) for _ in range(i - j - 1)]
                        convs.append(_cbr(ch[j], ch[i], 3, 2, relu=False))
                        row.append(nn.Sequential(*convs))
                fuse.append(nn.ModuleList(row))
            self.fuse_layers = nn.ModuleList(fuse)
        self.relu = nn.ReLU(True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [br(x) for br, x in zip(self.branches, xs)]
        if self.fuse_layers is None:
            return xs
        out = []
        for i, row in enumerate(self.fuse_layers):
            y = xs[0] if i == 0 else row[0](xs[0])
            for j in range(1, self.num_branches):
                y = y + (xs[j] if i == j else row[j](xs[j]))
            out.append(self.relu(y))
        return out


class WASBHRNet(nn.Module):
    """WASB HRNet (stride-1 stem, widths 16/32/64/128). In: (N, 3*F, 288, 512)
    ImageNet-normalised RGB. Out: (N, F, 288, 512) heatmap LOGITS (apply sigmoid)."""

    def __init__(self, frames_in: int = 3, frames_out: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3 * frames_in, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(_Bottleneck(64, 32, 1, _cbr(64, 128, 1, 1, relu=False)))
        self.transition1 = nn.ModuleList([_cbr(128, 16, 3, 1), nn.Sequential(_cbr(128, 32, 3, 2))])
        self.stage2 = nn.Sequential(_HRModule([2, 2], [16, 32], [16, 32]))
        self.transition2 = nn.ModuleList([None, None, nn.Sequential(_cbr(32, 64, 3, 2))])
        self.stage3 = nn.Sequential(_HRModule([2, 2, 2], [16, 32, 64], [16, 32, 64]))
        self.transition3 = nn.ModuleList([None, None, None, nn.Sequential(_cbr(64, 128, 3, 2))])
        self.stage4 = nn.Sequential(_HRModule([2, 2, 2, 2], [16, 32, 64, 128], [16, 32, 64, 128]))
        self.final_layers = nn.ModuleList([nn.Conv2d(16, frames_out, 1)])

    @staticmethod
    def _transition(trans: nn.ModuleList, ys: list[torch.Tensor]) -> list[torch.Tensor]:
        # official forward feeds y_list[-1] to every non-identity transition
        return [ys[i] if t is None else t(ys[-1]) for i, t in enumerate(trans)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        ys = self.stage2([t(x) for t in self.transition1])
        ys = self.stage3(self._transition(self.transition2, ys))
        ys = self.stage4(self._transition(self.transition3, ys))
        return self.final_layers[0](ys[0])


# ══════════════════════════════════════════════════════════════════════════════
# Geometry: full-res frame <-> network canvas (exact, affine)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FrameGeometry:
    """Aspect-preserving letterbox of a (width, height) frame into the net canvas.

    Mapping (continuous coords, pixel-centre convention unless ``corner=True``):
        net = full * scale + offset          full = (net - offset) / scale
    With pixel centres, full px centre i maps to net (i + .5) * s - .5 + pad, which is what
    cv2.resize / INTER_AREA sampling implements, so decoding is unbiased.
    ``corner=True`` reproduces WASB's centre-affine convention (net = s*full + pad).
    """
    width: int
    height: int
    net_w: int = NET_W
    net_h: int = NET_H
    corner: bool = False

    @property
    def scale(self) -> float:
        return min(self.net_w / self.width, self.net_h / self.height)

    @property
    def content_wh(self) -> tuple[int, int]:
        """Size of the resized image inside the canvas."""
        s = self.scale
        return (min(self.net_w, round(self.width * s)), min(self.net_h, round(self.height * s)))

    @property
    def pad(self) -> tuple[float, float]:
        if self.corner:
            s = self.scale
            return ((self.net_w - self.width * s) / 2.0, (self.net_h - self.height * s) / 2.0)
        cw, ch = self.content_wh
        return (float((self.net_w - cw) // 2), float((self.net_h - ch) // 2))

    def _axis(self) -> tuple[float, float, float, float]:
        """(sx, sy, ox, oy) with net = full*s + o."""
        px, py = self.pad
        if self.corner:
            s = self.scale
            return s, s, px, py
        cw, ch = self.content_wh
        sx, sy = cw / self.width, ch / self.height
        return sx, sy, px + 0.5 * sx - 0.5, py + 0.5 * sy - 0.5

    def to_net(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple:
        sx, sy, ox, oy = self._axis()
        return np.asarray(x) * sx + ox, np.asarray(y) * sy + oy

    def to_full(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple:
        sx, sy, ox, oy = self._axis()
        return (np.asarray(x) - ox) / sx, (np.asarray(y) - oy) / sy

    def peak_to_full(self, x: np.ndarray | float, y: np.ndarray | float,
                     peak_offset: float = 0.0) -> tuple:
        """Decoded heatmap peak (net px) → full-res px, removing the net's learned offset
        ``peak_offset`` = (target peak − image content position) in net px (see NetSpec)."""
        return self.to_full(np.asarray(x) - peak_offset, np.asarray(y) - peak_offset)

    def warp(self, frame: np.ndarray, interp: int = cv2.INTER_AREA) -> np.ndarray:
        """Resize + letterbox (black) a HxWx3 frame to the net canvas."""
        if self.corner:
            s = self.scale
            px, py = self.pad
            m = np.array([[s, 0, px], [0, s, py]], dtype=np.float64)
            # exactly WASB's preprocessing (utils/image.get_affine_transform + warpAffine)
            return cv2.warpAffine(frame, m, (self.net_w, self.net_h), flags=cv2.INTER_LINEAR)
        cw, ch = self.content_wh
        img = cv2.resize(frame, (cw, ch), interpolation=interp)
        if (cw, ch) == (self.net_w, self.net_h):
            return img
        out = np.zeros((self.net_h, self.net_w, frame.shape[2]), dtype=frame.dtype)
        px, py = int(self.pad[0]), int(self.pad[1])
        out[py:py + ch, px:px + cw] = img
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap decoding: peak NMS → top-k → subpixel refinement
# ══════════════════════════════════════════════════════════════════════════════

def _blob(hm: np.ndarray, cy: int, cx: int, r: int) -> tuple[int, int, np.ndarray]:
    """The peak's blob: weights ``(hm - peak/2)+`` on the 8-connected component of
    ``{hm > peak/2}`` that contains the peak pixel, inside a (2r+1)² box → (y0, x0, weights).

    Disconnected neighbours above half-peak are excluded. The box starts at the seed; if the
    blob touches its edge (the seed of a plateau is its first pixel in scan order, i.e. its
    top edge) it is re-centred on the blob centroid, at most twice.
    """
    h, w = hm.shape
    half = 0.5 * float(hm[cy, cx])
    for _ in range(3):
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
        wts = hm[y0:y1, x0:x1].astype(np.float64) - half
        np.maximum(wts, 0.0, out=wts)
        n, lab = cv2.connectedComponents((wts > 0).astype(np.uint8), connectivity=8)
        if n > 2:
            wts[lab != lab[cy - y0, cx - x0]] = 0.0
        m = wts > 0
        if not ((y0 > 0 and m[0].any()) or (y1 < h and m[-1].any())
                or (x0 > 0 and m[:, 0].any()) or (x1 < w and m[:, -1].any())):
            break
        tot = float(wts.sum())
        ny = int(round(float((wts.sum(1) * np.arange(y0, y1)).sum()) / tot))
        nx = int(round(float((wts.sum(0) * np.arange(x0, x1)).sum()) / tot))
        if (ny, nx) == (cy, cx) or not m[ny - y0, nx - x0]:
            break
        cy, cx = ny, nx
    return y0, x0, wts


def _refine_centroid(hm: np.ndarray, cy: int, cx: int, r: int) -> tuple[float, float]:
    """Weighted centroid of the peak's above-half-maximum blob (see ``_blob``).

    Unlike a fixed window around the seed pixel this is unbiased on plateaus — TrackNet-style
    disc targets, or sigmoid outputs saturated at 1.0 in fp16 — where the seed is merely the
    first maximal pixel in scan order (the plateau's top edge).
    """
    y0, x0, wts = _blob(hm, cy, cx, r)
    tot = float(wts.sum())
    if tot <= 0:
        return float(cx), float(cy)
    ys, xs = np.mgrid[y0:y0 + wts.shape[0], x0:x0 + wts.shape[1]]
    return float((wts * xs).sum() / tot), float((wts * ys).sum() / tot)


def _refine_quadratic(hm: np.ndarray, cy: int, cx: int) -> tuple[float, float]:
    """Separable log-parabola (Gaussian) fit on the 3×3 neighbourhood."""
    h, w = hm.shape
    if not (0 < cx < w - 1 and 0 < cy < h - 1):
        return float(cx), float(cy)
    p = np.log(np.maximum(hm[cy - 1:cy + 2, cx - 1:cx + 2].astype(np.float64), 1e-6))

    def off(a: float, b: float, c: float) -> float:
        d = a - 2 * b + c
        return 0.0 if d >= 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))

    return cx + off(p[1, 0], p[1, 1], p[1, 2]), cy + off(p[0, 1], p[1, 1], p[2, 1])


def _plateau_extent(hm: np.ndarray, cy: int, cx: int, cap: int = 16) -> int:
    """Chebyshev extent (px) from the seed of the connected set of pixels tied with it."""
    h, w = hm.shape
    y0, y1, x0, x1 = max(cy - cap, 0), min(cy + cap + 1, h), max(cx - cap, 0), min(cx + cap + 1, w)
    tie = (hm[y0:y1, x0:x1] >= hm[cy, cx]).astype(np.uint8)
    _, lab = cv2.connectedComponents(tie, connectivity=8)
    yy, xx = np.nonzero(lab == lab[cy - y0, cx - x0])
    return int(max(np.abs(yy + y0 - cy).max(), np.abs(xx + x0 - cx).max()))


def decode_heatmap(hm: np.ndarray, k: int = 3, min_score: float = 0.1, nms_radius: int = 6,
                   refine: str = "centroid", refine_radius: int = 4
                   ) -> list[tuple[float, float, float]]:
    """Top-k peaks of a (H, W) heatmap in [0,1] → [(x, y, score)] in heatmap px, score-sorted.

    Peaks are local maxima (5×5) above ``min_score``, taken greedily by score. A seed is
    suppressed if it lies within ``nms_radius`` of a kept seed or refined position, or inside a
    kept peak's above-half-maximum blob — so a plateau (all of whose pixels are local maxima)
    yields one peak. Refinement: 'centroid' (blob centroid, default), 'quadratic' (3×3
    log-parabola; plateaus fall back to the centroid, the fit being flat there) or 'none'.
    The blob box has radius ``refine_radius`` (covers the half-maximum disc of both nets'
    r=2.5 px targets), grown by the plateau's own extent for tied peaks. score = peak value.
    """
    hm = np.asarray(hm, dtype=np.float32)
    if hm.size == 0 or float(hm.max()) < min_score:
        return []
    dil = cv2.dilate(hm, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((hm >= dil) & (hm >= min_score))
    if len(ys) == 0:
        return []
    sc = hm[ys, xs]
    order = np.argsort(-sc, kind="stable")
    out: list[tuple[float, float, float]] = []
    kept: list[tuple[float, float]] = []
    taken: list[tuple[int, int, np.ndarray]] = []           # kept blobs (y0, x0, mask)
    r2 = nms_radius * nms_radius
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any((y - ky) ** 2 + (x - kx) ** 2 <= r2 for ky, kx in kept) or any(
                0 <= y - by < m.shape[0] and 0 <= x - bx < m.shape[1] and m[y - by, x - bx]
                for by, bx, m in taken):
            continue
        plateau = int((hm[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] == sc[i]).sum()) > 1
        by, bx, wts = _blob(hm, y, x, refine_radius + (_plateau_extent(hm, y, x) if plateau
                                                         else 0))
        if refine == "quadratic" and not plateau:
            fx, fy = _refine_quadratic(hm, y, x)
        elif refine == "none":
            fx, fy = float(x), float(y)
        else:
            tot = float(wts.sum())
            gy, gx = np.mgrid[by:by + wts.shape[0], bx:bx + wts.shape[1]]
            fx, fy = float((wts * gx).sum() / tot), float((wts * gy).sum() / tot)
        kept += [(y, x), (fy, fx)]
        taken.append((by, bx, wts > 0))
        out.append((fx, fy, float(sc[i])))
        if len(out) >= k:
            break
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════════════

class ShuttleDetector(Protocol):
    def detect_video(self, path: str, meta: VideoMeta, progress: ProgressFn | None = None
                     ) -> list[list[ShuttleCandidate]]:
        """One list per frame (len == meta.n_frames), top-k candidates sorted by score desc,
        full-resolution upright px, subpixel."""
        ...


# A re-iterable frame source: ``source(step)`` yields (index, BGR frame) for every
# ``step``-th frame, indices consecutive from 0 when step == 1.
SteppedSource = Callable[[int], Iterable[tuple[int, np.ndarray]]]


def video_source(path: str) -> SteppedSource:
    return lambda step: iter_frames(path, step=step)


def array_source(frames: list[np.ndarray] | np.ndarray) -> SteppedSource:
    return lambda step: ((i, frames[i]) for i in range(0, len(frames), step))


def _to_rgb_net(frame_bgr: np.ndarray, geom: FrameGeometry, interp: int) -> np.ndarray:
    return cv2.cvtColor(geom.warp(frame_bgr, interp), cv2.COLOR_BGR2RGB)


@dataclass
class BackgroundPlan:
    """Per-segment median backgrounds (net-canvas RGB uint8). Segments split at scene cuts
    and every ``max_segment_s`` seconds."""
    starts: list[int] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)

    def index_for(self, frame: int) -> int:
        i = int(np.searchsorted(np.asarray(self.starts), frame, side="right")) - 1
        return max(i, 0)


def estimate_backgrounds(source: SteppedSource, geom: FrameGeometry, fps: float,
                         interp: int = cv2.INTER_AREA, sample_hz: float = 4.0,
                         max_samples: int = 64, max_segment_s: float = 120.0,
                         cut_frac: float = 0.45, progress: ProgressFn | None = None,
                         n_frames: int | None = None) -> BackgroundPlan:
    """Temporal median of sampled frames per scene segment (bounded memory)."""
    step = max(1, int(round(fps / sample_hz)))
    seg_len = max(1, int(max_segment_s * fps))
    plan = BackgroundPlan()
    samples: list[np.ndarray] = []
    stride = 1                     # reservoir: keep every `stride`-th sample of segment
    seen = 0
    prev_thumb: np.ndarray | None = None
    seg_start = 0

    def close() -> None:
        if samples:
            plan.starts.append(seg_start)
            plan.images.append(np.median(np.stack(samples), axis=0).astype(np.uint8))

    for idx, frame in source(step):
        img = _to_rgb_net(frame, geom, interp)
        thumb = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_RGB2GRAY).astype(np.int16)
        is_cut = prev_thumb is not None and np.mean(np.abs(thumb - prev_thumb) > 40) > cut_frac
        prev_thumb = thumb
        if samples and (is_cut or idx - seg_start >= seg_len):
            close()
            samples, stride, seen, seg_start = [], 1, 0, idx
        if seen % stride == 0:
            samples.append(img)
            if len(samples) >= 2 * max_samples:
                samples = samples[::2]
                stride *= 2
        seen += 1
        if progress and n_frames:
            progress(min(idx / n_frames, 1.0))
    close()
    if not plan.images:
        plan.starts, plan.images = [0], [np.zeros((geom.net_h, geom.net_w, 3), np.uint8)]
    plan.starts[0] = 0
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# Neural heatmap detector (streaming, batched, overlapping-window fusion)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NetSpec:
    name: str
    seq_len: int            # frames in == heatmaps out
    uses_background: bool   # median background prepended as an extra RGB frame
    imagenet_norm: bool     # else x/255
    logits: bool            # output needs sigmoid
    corner_geometry: bool   # WASB centre-affine convention
    # Learned (target − content) offset in net px. Both repos draw targets at int(coord*s)
    # (floor → −0.5). WASB content is index-affine (warpAffine) → −0.5. TrackNetV3 content is
    # pixel-centre resized (PIL) while targets are index-scaled at s=0.4 (1280→512) → −0.5·0.4.
    peak_offset: float = -0.5

    @property
    def in_channels(self) -> int:
        return 3 * (self.seq_len + int(self.uses_background))


SPECS = {
    "tracknetv3": NetSpec("tracknetv3", 8, True, False, False, False, -0.2),
    "wasb": NetSpec("wasb", 3, False, True, True, True, -0.5),
}


def ensemble_weights(seq_len: int, mode: str) -> np.ndarray:
    """Positional weights for overlapping windows ('weight' = TrackNetV3 triangular)."""
    if mode == "average":
        return np.ones(seq_len, np.float32)
    w = np.ones(seq_len, np.float32)
    for i in range(math.ceil(seq_len / 2)):
        w[i] = w[seq_len - i - 1] = i + 1
    return w


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prefetch(it: Iterable, maxsize: int) -> Iterator:
    """Run an iterator in a daemon thread with a bounded queue (decode ‖ inference)."""
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=maxsize)
    end = object()
    err: list[BaseException] = []

    def work() -> None:
        try:
            for item in it:
                q.put(item)
        except BaseException as e:   # noqa: BLE001 - re-raised in consumer
            err.append(e)
        finally:
            q.put(end)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is end:
            if err:
                raise err[0]
            return
        yield item


class HeatmapDetector:
    """Streaming neural detector over a heatmap net (TrackNetV3 or WASB).

    Memory: holds at most ``seq_len + stride`` preprocessed frames plus one pending batch and
    the per-frame fusion accumulators of frames still covered by unrun windows.
    """

    def __init__(self, model: nn.Module, spec: NetSpec, device: str | None = None,
                 half: bool | None = None, stride: int | None = None, batch_size: int = 8,
                 k: int = 3, min_score: float = 0.1, tta_flip: bool = False,
                 ensemble: str = "weight", interp: int | None = None,
                 refine: str = "centroid", peak_offset: float | None = None,
                 prefetch: bool = True):
        """
        stride: frames between window starts (default: seq_len, i.e. non-overlapping).
        half: fp16 inference (default on for mps/cuda, never on cpu).
        peak_offset: override ``spec.peak_offset`` (learned target−content offset, net px).
        interp: resize filter (default INTER_AREA for TrackNetV3, INTER_LINEAR for WASB,
            matching each repo's training preprocessing).
        """
        self.spec = spec
        self.device = pick_device(device)
        self.half = (self.device in ("mps", "cuda")) if half is None else bool(half)
        if self.device == "cpu":
            self.half = False
        self._dtype = torch.float16 if self.half else torch.float32
        self.model = model.eval().to(self.device, self._dtype)
        self.stride = max(1, min(stride or spec.seq_len, spec.seq_len))
        self.batch_size = max(1, batch_size)
        self.k, self.min_score, self.refine = k, min_score, refine
        self.tta_flip = tta_flip
        self.peak_offset = spec.peak_offset if peak_offset is None else peak_offset
        self.prefetch = prefetch
        self.interp = interp if interp is not None else (
            cv2.INTER_LINEAR if spec.corner_geometry else cv2.INTER_AREA)
        self.weights = ensemble_weights(spec.seq_len, ensemble if self.stride < spec.seq_len
                                        else "average")
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device,
                                  dtype=self._dtype).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device,
                                 dtype=self._dtype).view(1, 1, 3, 1, 1)
        self.max_buffered = 0          # instrumentation (tests): peak frames held
        self.max_lag = 0               # instrumentation: newest decoded − oldest unyielded
        self.last_stats: dict = {}

    # ---------------------------------------------------------------- public
    def detect_video(self, path: str, meta: VideoMeta, progress: ProgressFn | None = None
                     ) -> list[list[ShuttleCandidate]]:
        return self.detect_frames(video_source(path), meta.width, meta.height, meta.fps,
                                  meta.n_frames, progress)

    def detect_frames(self, source: SteppedSource, width: int, height: int, fps: float,
                      n_frames: int | None = None, progress: ProgressFn | None = None
                      ) -> list[list[ShuttleCandidate]]:
        geom = self.geometry(width, height)
        out: list[list[ShuttleCandidate]] = []
        for idx, hm in self.iter_heatmaps(source, width, height, fps, n_frames, progress):
            while len(out) < idx:
                out.append([])
            out.append(self.candidates(hm, geom))
        if n_frames is not None:
            out = (out + [[] for _ in range(n_frames - len(out))])[:n_frames]
        return out

    def geometry(self, width: int, height: int) -> FrameGeometry:
        return FrameGeometry(width, height, corner=self.spec.corner_geometry)

    def candidates(self, hm: np.ndarray, geom: FrameGeometry) -> list[ShuttleCandidate]:
        res = []
        for x, y, s in decode_heatmap(hm, self.k, self.min_score, refine=self.refine):
            fx, fy = geom.peak_to_full(x, y, self.peak_offset)
            if -1.0 <= fx <= geom.width and -1.0 <= fy <= geom.height:   # drop letterbox hits
                res.append(ShuttleCandidate(float(fx), float(fy), float(s)))
        return res

    # ------------------------------------------------------------- streaming core
    def iter_heatmaps(self, source: SteppedSource, width: int, height: int, fps: float,
                      n_frames: int | None = None, progress: ProgressFn | None = None
                      ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, fused heatmap (288,512) float32 in [0,1]) in frame order.

        Windows start every ``stride`` frames. If the last full window does not reach the last
        frame, one tail window over the last ``seq_len`` real frames (clips shorter than
        ``seq_len``: padded with the last frame, as upstream) covers the rest, contributing only
        to frames not yet finalised. Overlapping predictions are fused with positional weights;
        a frame is finalised (yielded exactly once) as soon as no later window covers it.
        """
        spec, S, stride = self.spec, self.spec.seq_len, self.stride
        geom = self.geometry(width, height)
        prog = _Throttle(progress)
        t0 = time.perf_counter()
        bg_share = 0.15 if spec.uses_background else 0.0
        plan = None
        if spec.uses_background:
            plan = estimate_backgrounds(source, geom, fps, self.interp, n_frames=n_frames,
                                        progress=lambda f: prog(bg_share * f))
        buf: deque[tuple[int, np.ndarray]] = deque(maxlen=S)
        pending: list[tuple[int, np.ndarray, int, int]] = []   # start, frames, bg, lowest f
        acc: dict[int, list] = {}              # frame -> [weighted sum, weight]
        wts = self.weights
        next_start, total, n_windows, covered = 0, 0, 0, -1
        self.max_buffered = 0
        self.max_lag = 0

        def run(limit: int | None) -> None:
            if not pending:
                return
            bgs = np.stack([plan.images[p[2]] for p in pending]) if plan else None
            y = self._forward(np.stack([p[1] for p in pending]), bgs)
            for b, (start, _, _, lo) in enumerate(pending):
                for s in range(S):
                    f = start + s
                    if f < lo:
                        continue
                    if limit is not None and f >= limit:
                        break
                    a = acc.get(f)
                    if a is None:
                        acc[f] = [y[b, s] * wts[s], float(wts[s])]
                    else:
                        a[0] += y[b, s] * wts[s]
                        a[1] += float(wts[s])
            pending.clear()

        def finalize(upto: int) -> Iterator[tuple[int, np.ndarray]]:
            for f in sorted(f for f in acc if f < upto):
                s, w = acc.pop(f)
                yield f, s / w

        def window(start: int, frames: list[np.ndarray], lo: int) -> None:
            nonlocal n_windows
            pending.append((start, np.stack(frames), plan.index_for(start) if plan else 0, lo))
            n_windows += 1

        pre = ((i, _to_rgb_net(f, geom, self.interp)) for i, f in source(1))
        for idx, img in (_prefetch(pre, maxsize=2 * S) if self.prefetch else pre):
            if idx != total:
                raise ValueError(f"frame source must yield consecutive indices (got {idx})")
            buf.append((idx, img))
            total = idx + 1
            if len(buf) == S and buf[0][0] == next_start:
                window(next_start, [im for _, im in buf], next_start)
                covered = next_start + S - 1
                next_start += stride
                self.max_buffered = max(self.max_buffered, len(buf) + len(pending) * S)
                if len(pending) >= self.batch_size:
                    run(None)
                    for f, hm in finalize(next_start):
                        self.max_lag = max(self.max_lag, total - 1 - f)
                        yield f, hm
            if n_frames:
                prog(bg_share + (1 - bg_share) * min(total / n_frames, 1.0))
        if total and covered < total - 1:                 # tail / clip shorter than S
            frames = [im for _, im in buf]                # = frames max(total-S,0) .. total-1
            while len(frames) < S:
                frames.append(frames[-1])                 # upstream: repeat last frame
            window(max(total - S, 0), frames, min(next_start, covered + 1))
        run(total)
        yield from finalize(total)
        dt = time.perf_counter() - t0
        self.last_stats = {"frames": total, "seconds": dt, "fps": total / dt if dt else 0.0,
                           "windows": n_windows, "device": self.device, "half": self.half}
        if progress:
            progress(1.0)

    # ------------------------------------------------------------------ network
    @torch.inference_mode()
    def _forward(self, windows: np.ndarray, bgs: np.ndarray | None) -> np.ndarray:
        """windows (B,S,H,W,3) uint8 RGB [+ bgs (B,H,W,3)] → heatmaps (B,S,H,W) float32."""
        x = torch.from_numpy(windows).to(self.device)
        if bgs is not None:
            x = torch.cat([torch.from_numpy(bgs).to(self.device)[:, None], x], dim=1)
        x = x.permute(0, 1, 4, 2, 3).to(self._dtype) / 255.0          # (B,T,3,H,W)
        if self.spec.imagenet_norm:
            x = (x - self._mean) / self._std
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])       # frame-major channels
        y = self._net(x)
        if self.tta_flip:
            y = 0.5 * (y + self._net(x.flip(-1)).flip(-1))
        return y.float().cpu().numpy()

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if self.spec.logits:
            y = torch.sigmoid(y.float())
        return y.float()


class _Throttle:
    """Rate-limit a progress callback (≥0.5 % change or ≥0.25 s apart)."""

    def __init__(self, fn: ProgressFn | None):
        self.fn, self.last_v, self.last_t = fn, -1.0, 0.0

    def __call__(self, v: float) -> None:
        if self.fn is None:
            return
        now = time.perf_counter()
        if v - self.last_v >= 0.005 or now - self.last_t >= 0.25 or v >= 1.0:
            self.last_v, self.last_t = v, now
            self.fn(float(v))


# ══════════════════════════════════════════════════════════════════════════════
# Weights / model construction (also imported by ml/train_detector.py, ml/export_coreml.py)
# ══════════════════════════════════════════════════════════════════════════════

# Candidate files per kind, searched in order inside ``weights_dir``. Fine-tuned outputs of
# ml/train_detector.py (``*_finetuned*``) take precedence over the official checkpoints.
WEIGHT_FILES: dict[str, tuple[str, ...]] = {
    "wasb": ("wasb_badminton_finetuned.pth.tar", "wasb_badminton_finetuned.pt",
             "wasb_badminton_best.pth.tar"),
    "tracknetv3": ("ckpts/TrackNet_finetuned.pt", "TrackNet_finetuned.pt",
                   "ckpts/TrackNet_best.pt", "TrackNet_best.pt"),
}
# Server default = TrackNetV3: on the held-out Test split (29 clips, official protocol) it beat
# WASB on every match (F1 0.976 vs 0.966, recall 0.961 vs 0.943, p95 err 3.9 vs 4.1 px) and ran
# faster on MPS (~36 vs ~26 fps) and CPU (~4.4 vs ~2.3 fps). It needs a median background, i.e. a
# static camera per segment; WASB (no background, 6 MB) is the on-device model and next in line.
AUTO_ORDER = ("tracknetv3", "wasb")


def find_weights(kind: str, weights_dir: str | Path | None = None) -> Path | None:
    """First existing checkpoint for ``kind`` (``weights_dir`` may also be a file)."""
    d = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    if d.is_file():
        return d
    for name in WEIGHT_FILES.get(kind, ()):
        if (d / name).is_file():
            return d / name
    return None


def load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Read a checkpoint of either official format → (state_dict, extra metadata).

    Accepts TrackNetV3 (``{'model', 'param_dict'}``), WASB (``{'model_state_dict'}``),
    ``{'state_dict'}`` and bare state dicts; strips DataParallel ``module.`` prefixes.
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    meta: dict = {}
    sd = ck
    if isinstance(ck, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in ck and isinstance(ck[key], dict):
                sd = ck[key]
                meta = {k: v for k, v in ck.items()
                        if k not in (key, "optimizer", "optimizer_state_dict", "scheduler")}
                break
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    return sd, meta


def infer_kind(state_dict: dict[str, torch.Tensor]) -> str:
    if "down_block_1.conv_1.conv.weight" in state_dict:
        return "tracknetv3"
    if "stage4.0.branches.0.0.conv1.weight" in state_dict:
        return "wasb"
    raise ValueError("unrecognised detector checkpoint")


def spec_from_state_dict(state_dict: dict[str, torch.Tensor], kind: str | None = None) -> NetSpec:
    """NetSpec with seq_len / background usage read from the weight shapes."""
    kind = kind or infer_kind(state_dict)
    base = SPECS[kind]
    if kind == "tracknetv3":
        c_in = state_dict["down_block_1.conv_1.conv.weight"].shape[1]
        s_out = state_dict["predictor.weight"].shape[0]
        uses_bg = c_in == 3 * (s_out + 1)
        if not uses_bg and c_in != 3 * s_out:
            raise ValueError(f"unsupported TrackNetV3 bg_mode (in={c_in}, out={s_out})")
        return NetSpec(base.name, s_out, uses_bg, base.imagenet_norm, base.logits,
                       base.corner_geometry, base.peak_offset)
    f_in = state_dict["conv1.weight"].shape[1] // 3
    f_out = state_dict["final_layers.0.weight"].shape[0]
    if f_in != f_out:
        raise ValueError(f"WASB frames_in != frames_out ({f_in} vs {f_out}) unsupported")
    return NetSpec(base.name, f_out, False, base.imagenet_norm, base.logits, base.corner_geometry,
                   base.peak_offset)


def build_model(spec: NetSpec) -> nn.Module:
    """Uninitialised network for ``spec`` (random weights)."""
    if spec.name == "tracknetv3":
        return TrackNetV3(spec.in_channels, spec.seq_len)
    if spec.name == "wasb":
        return WASBHRNet(spec.seq_len, spec.seq_len)
    raise ValueError(f"unknown net {spec.name!r}")


def load_model(path: str | Path, kind: str | None = None) -> tuple[nn.Module, NetSpec]:
    """Build the architecture from checkpoint shapes and load weights with ``strict=True``."""
    sd, _ = load_checkpoint(path)
    spec = spec_from_state_dict(sd, kind)
    model = build_model(spec)
    model.load_state_dict(sd, strict=True)
    return model.eval(), spec


def build_detector(weights_dir: str | None = None, device: str | None = None,
                   kind: str = "auto", **kwargs) -> ShuttleDetector:
    """Best available detector: pretrained net (``AUTO_ORDER``) > motion fallback.

    ``kind``: 'auto' | 'wasb' | 'tracknetv3' | 'motion'. Env ``SHUTTL_DETECTOR`` overrides
    'auto'. Extra kwargs go to the detector constructor (e.g. stride, batch_size, tta_flip).
    """
    import os

    if kind == "auto":
        kind = os.environ.get("SHUTTL_DETECTOR", "auto").strip().lower() or "auto"
    if kind == "motion":
        return MotionDetector(**kwargs)
    order = AUTO_ORDER if kind == "auto" else (kind,)
    if kind != "auto" and kind not in SPECS:
        raise ValueError(f"unknown detector kind {kind!r}")
    for k in order:
        path = find_weights(k, weights_dir)
        if path is None:
            continue
        try:
            model, spec = load_model(path, k)
        except Exception:
            if kind != "auto":
                raise
            log.exception("failed to load %s weights from %s", k, path)
            continue
        log.info("shuttle detector: %s (%s) on %s", k, path.name, pick_device(device))
        opts = {**DETECTOR_DEFAULTS.get(k, {}), **kwargs}
        return HeatmapDetector(model, spec, device=device, **opts)
    if kind != "auto":
        raise FileNotFoundError(f"no {kind} weights in {weights_dir or DEFAULT_WEIGHTS_DIR}")
    log.warning("no detector weights found; using weight-free MotionDetector")
    return MotionDetector(**kwargs)


# Per-model defaults for build_detector: each repo's own non-overlapping inference mode
# (WASB detector ``step: 3``; TrackNetV3 ``eval_mode=nonoverlap``). Overlap (stride 1 / 4)
# costs 2-3x and gave no F1 gain on a 3-clip probe; see the module report.
DETECTOR_DEFAULTS: dict[str, dict] = {
    "wasb": {"stride": 3, "batch_size": 8},
    "tracknetv3": {"stride": 8, "batch_size": 4},
}


# ══════════════════════════════════════════════════════════════════════════════
# Motion fallback (no weights): running-median background + 3-frame differencing
# ══════════════════════════════════════════════════════════════════════════════

class MotionDetector:
    """Weight-free shuttle candidates from motion.

    Per frame t (one frame latency): at a reduced processing resolution,
    ``s = min(max(|g_t-g_{t-1}|, |g_t-g_{t+1}|), |g_t-bg|)`` — moving now AND different from
    the background, which rejects the "ghost" left where the shuttle was. ``bg`` is an
    approximate running median (±1 grey level per frame) initialised from a sparse temporal
    median and reset at scene cuts. Blobs are filtered by area / streak length (scaled to
    resolution) and scored by contrast, size, polarity (white shuttle) and a static-blob
    suppressor (repeated hits in the same cell = lights, lines, logos). Positions are refined
    at full resolution (thresholded weighted centroid = mid-exposure point of a blur streak).
    """

    def __init__(self, k: int = 3, proc_width: int = 640, min_contrast: float = 10.0,
                 noise_k: float = 3.0, bg_init_frames: int = 15,
                 bg_init_seconds: float = 10.0, static_decay: float = 0.95,
                 static_tolerance: float = 3.0, min_score: float = 0.02,
                 max_components: int = 400):
        self.k, self.proc_width = k, proc_width
        self.min_contrast, self.noise_k = min_contrast, noise_k
        self.bg_init_frames, self.bg_init_seconds = bg_init_frames, bg_init_seconds
        self.static_decay, self.static_tolerance = static_decay, static_tolerance
        self.min_score, self.max_components = min_score, max_components
        self.last_stats: dict = {}
        self.max_buffered = 0

    def detect_video(self, path: str, meta: VideoMeta, progress: ProgressFn | None = None
                     ) -> list[list[ShuttleCandidate]]:
        return self.detect_frames(video_source(path), meta.width, meta.height, meta.fps,
                                  meta.n_frames, progress)

    # ---------------------------------------------------------------- helpers
    def _proc_size(self, width: int, height: int) -> tuple[int, int, float]:
        r = min(1.0, self.proc_width / max(width, height))
        return max(1, round(width * r)), max(1, round(height * r)), r

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _small(self, gray: np.ndarray, pw: int, ph: int) -> np.ndarray:
        if gray.shape[1] == pw and gray.shape[0] == ph:
            return gray
        return cv2.resize(gray, (pw, ph), interpolation=cv2.INTER_AREA)

    def _init_background(self, source: SteppedSource, pw: int, ph: int, fps: float
                         ) -> np.ndarray | None:
        step = max(1, int(round(self.bg_init_seconds * fps / self.bg_init_frames)))
        smalls = []
        for _, f in source(step):
            smalls.append(self._small(self._gray(f), pw, ph))
            if len(smalls) >= self.bg_init_frames:
                break
        return np.median(np.stack(smalls), axis=0).astype(np.uint8) if smalls else None

    # ------------------------------------------------------------------- main
    def detect_frames(self, source: SteppedSource, width: int, height: int, fps: float,
                      n_frames: int | None = None, progress: ProgressFn | None = None
                      ) -> list[list[ShuttleCandidate]]:
        pw, ph, r = self._proc_size(width, height)
        prog = _Throttle(progress)
        t0 = time.perf_counter()
        bg = self._init_background(source, pw, ph, fps)
        cell = 8
        self._static = np.zeros(((ph + cell - 1) // cell, (pw + cell - 1) // cell), np.float32)
        self._cell = cell
        self._thr = self.min_contrast
        self._fast = 0
        win: deque[tuple[int, np.ndarray, np.ndarray]] = deque(maxlen=3)
        out: list[list[ShuttleCandidate]] = []
        t_proc = 0.0

        for idx, frame in source(1):
            G = self._gray(frame)
            win.append((idx, G, self._small(G, pw, ph)))
            self.max_buffered = max(self.max_buffered, len(win))
            if bg is None:
                bg = win[-1][2].copy()
            if len(win) >= 2:
                tp = time.perf_counter()
                prev = win[-3] if len(win) == 3 else None
                out.append(self._process(prev, win[-2], win[-1], bg, r))
                bg = self._update_bg(bg, win[-2][2], prev[2] if prev else None)
                t_proc += time.perf_counter() - tp
            if n_frames:
                prog(min(len(out) / n_frames, 1.0))
        if win:
            prev = win[-2] if len(win) >= 2 else None
            out.append(self._process(prev, win[-1], None, bg, r) if prev else [])
        if n_frames is not None:
            out = (out + [[] for _ in range(n_frames - len(out))])[:n_frames]
        dt = time.perf_counter() - t0
        self.last_stats = {"frames": len(out), "seconds": dt,
                           "fps": len(out) / dt if dt else 0.0,
                           "proc_fps": len(out) / t_proc if t_proc else 0.0}
        if progress:
            progress(1.0)
        return out

    def _update_bg(self, bg: np.ndarray, g: np.ndarray, gp: np.ndarray | None) -> np.ndarray:
        if gp is not None and float(cv2.absdiff(g, gp)[::4, ::4].mean()) > 30.0:
            self._fast = 30                    # scene cut: re-seed and adapt quickly
            return g.copy()
        step = 8 if self._fast > 0 else 1
        self._fast = max(0, self._fast - 1)
        cv2.add(bg, step, dst=bg, mask=cv2.compare(g, bg, cv2.CMP_GT))
        cv2.subtract(bg, step, dst=bg, mask=cv2.compare(g, bg, cv2.CMP_LT))
        return bg

    def _process(self, prev, cur, nxt, bg: np.ndarray, r: float) -> list[ShuttleCandidate]:
        idx, G, g = cur
        nbrs = [n for n in (prev, nxt) if n is not None]
        if not nbrs:
            return []
        mot = cv2.absdiff(g, nbrs[0][2])
        if len(nbrs) == 2:
            mot = cv2.max(mot, cv2.absdiff(g, nbrs[1][2]))
        fgd = cv2.absdiff(g, bg)
        s = cv2.blur(cv2.min(mot, fgd), (3, 3))
        if idx % 15 == 0 or self._thr is None:     # noise-adaptive threshold
            med = float(np.median(s[::4, ::4]))
            self._thr = max(self.min_contrast, self.noise_k * med + 4.0)
        thr = self._thr
        _, mask = cv2.threshold(s, thr, 255, cv2.THRESH_BINARY)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        self._static *= self.static_decay
        if n <= 1 or n - 1 > self.max_components:   # nothing / global change (pan, flash)
            return []
        ph, pw = g.shape
        diag = math.hypot(pw, ph)
        max_len, a0 = 0.08 * diag, max(4.0, (0.012 * diag) ** 2)
        cands: list[tuple[float, float, float]] = []
        cells: list[tuple[int, int]] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if max(w, h) > max_len or area > 6 * a0:
                continue
            roi_lab = lab[y:y + h, x:x + w] == i
            roi_s = s[y:y + h, x:x + w]
            contrast = float(roi_s[roi_lab].mean()) - thr * 0.5
            brighter = float(g[y:y + h, x:x + w][roi_lab].mean()) >= float(
                bg[y:y + h, x:x + w][roi_lab].mean())
            size_term = 1.0 if area <= a0 else math.exp(-(area - a0) / (2 * a0))
            sc = (1.0 - math.exp(-max(contrast, 0.0) / 25.0)) * size_term * (1.0 if brighter else 0.5)
            cy, cx = min((y + h // 2) // self._cell, self._static.shape[0] - 1), \
                min((x + w // 2) // self._cell, self._static.shape[1] - 1)
            c = float(self._static[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].max())
            sc *= math.exp(-max(0.0, c - self.static_tolerance) / self.static_tolerance)
            cells.append((cy, cx))
            if sc < self.min_score:
                continue
            fx, fy = self._refine(G, prev, nxt, x, y, w, h, r, (x + w / 2) / r, (y + h / 2) / r)
            cands.append((fx, fy, sc))
        for cy, cx in cells:
            self._static[cy, cx] += 1.0
        cands.sort(key=lambda c: -c[2])
        return [ShuttleCandidate(float(x), float(y), float(min(sc, 1.0)))
                for x, y, sc in cands[:self.k]]

    @staticmethod
    def _refine(G: np.ndarray, prev, nxt, x: int, y: int, w: int, h: int, r: float,
                fx0: float, fy0: float) -> tuple[float, float]:
        """Thresholded weighted centroid of the full-res motion energy inside the blob ROI."""
        H, W = G.shape
        m = 2
        x0, y0 = max(int(x / r) - m, 0), max(int(y / r) - m, 0)
        x1, y1 = min(int(math.ceil((x + w) / r)) + m, W), min(int(math.ceil((y + h) / r)) + m, H)
        if x1 <= x0 or y1 <= y0:
            return fx0, fy0
        roi = G[y0:y1, x0:x1]
        e = None                    # 3-frame differencing: min(|G-prev|, |G-next|)
        for n in (prev, nxt):
            if n is not None:
                d = cv2.absdiff(roi, n[1][y0:y1, x0:x1])
                e = d if e is None else cv2.min(e, d)
        if e is None:
            return fx0, fy0
        e = e.astype(np.float32)
        e -= 0.35 * float(e.max())
        np.maximum(e, 0.0, out=e)
        tot = float(e.sum())
        if tot <= 0:
            return fx0, fy0
        ys, xs = np.mgrid[y0:y1, x0:x1]
        return float((e * xs).sum() / tot), float((e * ys).sum() / tot)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation on labelled clips (TrackNet CSV: Frame,Visibility,X,Y) + benchmark CLI
#   python -m app.services.tracking.detector eval --kind wasb --clips ... [--max-frames N]
#   python -m app.services.tracking.detector bench --kind wasb --size 1920x1080 --device cpu
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(csv_path: str | Path, n_frames: int | None = None) -> np.ndarray:
    """(N, 3) float array [visible, x, y] indexed by frame (missing rows = invisible)."""
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    n = int(max(rows[:, 0].max() + 1, n_frames or 0))
    lab = np.zeros((n, 3), np.float64)
    lab[rows[:, 0].astype(int)] = rows[:, 1:4]
    lab[:, 0] = lab[:, 0] > 0
    return lab


def score_candidates(cands: list[list[ShuttleCandidate]], labels: np.ndarray,
                     width: int, height: int, frames: Iterable[int] | None = None,
                     threshold: float = 0.5, tol_px: float = 10.0, net_tol: float = 4.0,
                     k: int = 3) -> dict:
    """Frame-level TP/TN/FP1/FP2/FN of the top-1 candidate with score ≥ threshold.

    ``official``: TrackNetV3 test.py protocol — coordinates truncated to the 512×288 net grid
    (``int(v / scaler)``), TP if distance ≤ ``net_tol`` (4) net px. ``fixed``: full-res
    Euclidean distance ≤ ``tol_px``. ``px4``: WASB evaluator (full-res distance < 4 px).
    Also px errors of the top-1 candidate and recall of any top-k candidate.
    """
    sx, sy = width / NET_W, height / NET_H
    res = {m: dict(TP=0, TN=0, FP1=0, FP2=0, FN=0) for m in ("official", "fixed", "px4")}
    err, dxy, hit_k, n_vis = [], [], 0, 0
    for f in (range(len(labels)) if frames is None else frames):
        vis, gx, gy = labels[f] if f < len(labels) else (0.0, 0.0, 0.0)
        cs = [c for c in (cands[f] if f < len(cands) else []) if c.score >= threshold]
        if vis:
            n_vis += 1
            hit_k += any(math.hypot(c.x - gx, c.y - gy) <= tol_px for c in cs[:k])
        if not cs:
            for m in res:
                res[m]["FN" if vis else "TN"] += 1
            continue
        if not vis:
            for m in res:
                res[m]["FP2"] += 1
            continue
        c = cs[0]
        d = math.hypot(c.x - gx, c.y - gy)
        err.append(d)
        dxy.append((c.x - gx, c.y - gy))
        dn = math.hypot(int(c.x / sx) - int(gx / sx), int(c.y / sy) - int(gy / sy))
        res["official"]["TP" if dn <= net_tol else "FP1"] += 1
        res["fixed"]["TP" if d <= tol_px else "FP1"] += 1
        res["px4"]["TP" if d < 4.0 else "FP1"] += 1          # WASB evaluator: < 4 full-res px
    for r in res.values():
        tp, fp, fn = r["TP"], r["FP1"] + r["FP2"], r["FN"]
        r["precision"] = tp / (tp + fp) if tp + fp else 0.0
        r["recall"] = tp / (tp + fn) if tp + fn else 0.0
        r["f1"] = (2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"])
                   if r["precision"] + r["recall"] else 0.0)
        r["accuracy"] = (tp + r["TN"]) / max(1, tp + r["TN"] + fp + fn)
    e = np.asarray(err)
    res["err_median_px"] = float(np.median(e)) if len(e) else float("nan")
    res["err_p95_px"] = float(np.percentile(e, 95)) if len(e) else float("nan")
    res["bias_dxy_px"] = [float(v) for v in np.median(np.asarray(dxy), 0)] if dxy else []
    res[f"recall_top{k}"] = hit_k / n_vis if n_vis else 0.0
    res["n_frames"] = sum(res["fixed"][t] for t in ("TP", "TN", "FP1", "FP2", "FN"))
    res["n_visible"] = n_vis
    return res


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import pickle

    ap = argparse.ArgumentParser(prog="python -m app.services.tracking.detector")
    ap.add_argument("cmd", choices=["eval", "bench"])
    ap.add_argument("--kind", default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--clips", nargs="*", default=[], help="videos (labels: ../csv or corrected)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--size", default="1280x720", help="bench: WxH of synthetic input")
    ap.add_argument("--frames", type=int, default=120, help="bench: frames")
    ap.add_argument("--cache", default=None, help="eval: pickle dir for raw candidates")
    ap.add_argument("--tag", default="", help="eval: suffix for cache files")
    a = ap.parse_args(argv)
    kw = {k: v for k, v in dict(stride=a.stride, batch_size=a.batch_size,
                                half=None if a.half is None else bool(a.half)).items()
          if v is not None}
    if a.kind == "motion":
        kw = {}
    det = build_detector(a.weights_dir, a.device, a.kind, **kw)
    if a.cmd == "bench":
        w, h = (int(v) for v in a.size.lower().split("x"))
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (h, w, 3), np.uint8)
        frames = [np.roll(base, 3 * i, axis=1) for i in range(8)]
        src: SteppedSource = lambda step: ((i, frames[i % 8]) for i in range(0, a.frames, step))
        det.detect_frames(src, w, h, 30.0, a.frames)            # warm-up (kernels, caches)
        det.detect_frames(src, w, h, 30.0, a.frames)
        print(json.dumps({"kind": a.kind, "size": a.size, **det.last_stats}))
        return
    from app.services.tracking import video_io
    for clip in a.clips:
        p = Path(clip)
        rally, match_dir = p.stem, p.parent.parent
        corr = (match_dir.parents[2] / "corrected" / match_dir.name / "corrected_csv"
                / f"{rally}_ball.csv")
        csv = corr if corr.is_file() else match_dir / "csv" / f"{rally}_ball.csv"
        meta = video_io.probe(str(p))
        stop = a.max_frames
        src = (lambda step, s=str(p), stop=stop: iter_frames(s, step=step, stop=stop))
        n = min(meta.n_frames, stop or meta.n_frames)
        t0 = time.perf_counter()
        cands = det.detect_frames(src, meta.width, meta.height, meta.fps, n)
        dt = time.perf_counter() - t0
        labels = load_labels(csv, n)
        frames = range(n)
        drop = match_dir.parents[2] / "corrected" / "drop_frame.json"
        key = f"{match_dir.name.replace('match', '')}_{rally}"
        if drop.is_file():
            dj = json.loads(drop.read_text())
            if key in dj.get("start", {}):
                frames = range(dj["start"][key], min(dj["end"][key], n))
        if a.cache:
            Path(a.cache).mkdir(parents=True, exist_ok=True)
            with open(Path(a.cache) / f"{a.kind}{a.tag}_{key}.pkl", "wb") as fh:
                pickle.dump({"cands": [[(c.x, c.y, c.score) for c in cs] for cs in cands],
                             "labels": labels, "frames": list(frames),
                             "wh": (meta.width, meta.height), "seconds": dt}, fh)
        r = score_candidates(cands, labels, meta.width, meta.height, frames, a.threshold)
        print(json.dumps({"clip": key, "kind": a.kind, "labels": csv.parent.name,
                          "fps": round(n / dt, 1), **{m: {kk: (round(vv, 4) if isinstance(
                              vv, float) else vv) for kk, vv in r[m].items()}
                                                      for m in ("official", "fixed", "px4")},
                          **{kk: vv for kk, vv in r.items()
                             if kk not in ("official", "fixed", "px4")}}))


if __name__ == "__main__":
    _main()
