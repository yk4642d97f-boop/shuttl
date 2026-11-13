"""Portable HitNet features, TCN and peak decoder.

Copied from research/hitnet/common.py (2026-10-07); no dataset/scorer imports.
Player slot 0 is far/Top, slot 1 near/Bottom. Both outputs depend on slot order.
Runtime parity against the frozen research checkpoint is checked separately.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ASPECT = 1280.0 / 720.0
HALF_L, HALF_W = 6.7, 3.05

def _fwd(x):
    o = np.full_like(x, np.nan)
    o[:-1] = x[1:] - x[:-1]
    return o


def _bwd(x):
    o = np.full_like(x, np.nan)
    o[1:] = x[1:] - x[:-1]
    return o


def _cen(x):
    o = np.full_like(x, np.nan)
    o[1:-1] = (x[2:] - x[:-2]) / 2
    return o


def _acc(x):
    o = np.full_like(x, np.nan)
    o[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]
    return o


def _z(x):
    return np.nan_to_num(x, nan=0.0)


def _dist_to_vis(vis):
    T = len(vis)
    idx = np.arange(T)
    last = np.where(vis, idx, -10 ** 6)
    last = np.maximum.accumulate(last)
    nxt = np.where(vis, idx, 10 ** 6)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return np.minimum(idx - last, 30) / 30.0, np.minimum(nxt - idx, 30) / 30.0


def derive(raw, fps):
    """(T,16) normalised raw -> (T, C) float32 model input. Pure function of raw+fps."""
    T = len(raw)
    ps = fps / 30.0  # velocities are expressed per 1/30 s so 25 and 30 fps look alike
    sx, sy = raw[:, 0].astype(np.float64), raw[:, 1].astype(np.float64)
    vis = np.isfinite(sx)
    f = []

    def add(*cols):
        for c in cols:
            f.append(np.asarray(c, np.float32))

    add(_z(sx), _z(sy), vis, _z(raw[:, 3]), _z(raw[:, 2]))
    # shuttle kinematics (zero where any needed sample is missing) + validity flags
    vfx, vfy = _fwd(sx), _fwd(sy)
    vbx, vby = _bwd(sx), _bwd(sy)
    vcx, vcy = _cen(sx), _cen(sy)
    ax, ay = _acc(sx), _acc(sy)
    k1, k2 = 30.0 * ps, 100.0 * ps * ps
    sp = np.hypot(vcx, vcy)
    # direction change at t: angle between backward and forward step, plus speed ratio
    dot = vbx * vfx + vby * vfy
    nb, nf = np.hypot(vbx, vby), np.hypot(vfx, vfy)
    cosang = dot / np.maximum(nb * nf, 1e-6)
    cosang = np.where((nb * nf) > 1e-8, cosang, np.nan)
    c = lambda a, k: np.clip(_z(a) * k, -6, 6)  # noqa: E731
    add(c(vfx, k1), c(vfy, k1), c(vbx, k1), c(vby, k1), c(vcx, k1), c(vcy, k1),
        c(ax, k2), c(ay, k2), c(sp, k1), _z(cosang),
        np.isfinite(vfx), np.isfinite(vbx), np.isfinite(ax))
    d0, d1 = _dist_to_vis(vis)
    add(d0, d1)
    # players
    hH = None
    for p in range(2):
        o = 4 + 4 * p
        cx, cy, w, h = (raw[:, o + i].astype(np.float64) for i in range(4))
        okb = np.isfinite(cx)
        add(_z(cx), _z(cy), _z(w) * 4, _z(h) * 2, okb)
        add(c(_cen(cx), 100 * ps), c(_cen(cy), 100 * ps))
        fx, fy = raw[:, 12 + 2 * p].astype(np.float64), raw[:, 13 + 2 * p].astype(np.float64)
        okf = np.isfinite(fx)
        add(_z(fx) / HALF_L, _z(fy) / HALF_W, okf)
        add(c(_cen(fx), 3 * ps), c(_cen(fy), 3 * ps))
        okr = okb & vis
        dx = np.where(okr, (sx - cx) * ASPECT / np.maximum(h, 1e-3), 0.0)
        dy = np.where(okr, (sy - cy) / np.maximum(h, 1e-3), 0.0)
        add(np.clip(dx, -6, 6) / 3, np.clip(dy, -6, 6) / 3, okr)
    add(np.full(T, fps / 30.0))
    X = np.stack(f, 1).astype(np.float32)
    return X


class Block(nn.Module):
    def __init__(self, h, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(h, h, 3, padding=dil, dilation=dil)
        self.b1 = nn.BatchNorm1d(h)
        self.c2 = nn.Conv1d(h, h, 1)
        self.b2 = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = F.gelu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return x + self.drop(y)


class HitTCN(nn.Module):
    """Non-causal dilated residual TCN, per-frame outputs [hit logit, hitter-is-Top logit]."""

    def __init__(self, n_in, hidden=96, dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32), drop=0.15):
        super().__init__()
        self.inp = nn.Conv1d(n_in, hidden, 1)
        self.bn = nn.BatchNorm1d(hidden)
        self.blocks = nn.ModuleList([Block(hidden, d, drop) for d in dils])
        self.out = nn.Conv1d(hidden, 2, 1)

    def forward(self, x):  # x (B,T,C)
        h = F.gelu(self.bn(self.inp(x.transpose(1, 2))))
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2)


def local_peaks(p, floor):
    T = len(p)
    pl = np.concatenate([[-1], p[:-1]])
    pr = np.concatenate([p[1:], [-1]])
    idx = np.where((p > pl) & (p >= pr) & (p >= floor))[0]
    return idx


def decode_nms(p, fps, thr, gap_s):
    gap = max(1, int(round(gap_s * fps)))
    idx = local_peaks(p, thr)
    order = idx[np.argsort(-p[idx], kind="stable")]
    kept = []
    for i in order:
        if all(abs(i - j) >= gap for j in kept):
            kept.append(int(i))
    return sorted(kept)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
