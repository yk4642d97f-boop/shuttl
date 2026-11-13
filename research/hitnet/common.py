"""Shared code for hitnet: data cache, feature builder, model, decoder, scorer glue.

Everything (including the cache) lives under research/hitnet/.  Reads, never writes,
research/extract/out and research/hits/score.py.
"""
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
EXTRACT = os.path.join(RESEARCH, "extract", "out")
LABELS_ROOT = os.path.join(RESEARCH, "shuttleset", "labels", "ShuttleSet", "set")
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(RESEARCH, "hits"))
import score as scorelib  # noqa: E402  (read-only import of the official scorer)

TEST_IDS = ["039", "040", "042", "043", "044"]
VAL_IDS = ["041"]
HELD_OUT = set(TEST_IDS) | set(VAL_IDS)
ASPECT = 1280.0 / 720.0
HALF_L, HALF_W = 6.7, 3.05

# raw per-frame channel layout (T, 16), normalised, NaN = missing
# 0 sx, 1 sy, 2 conf, 3 interp, 4-7 box0 (cx,cy,w,h), 8-11 box1, 12-13 foot0, 14-15 foot1
NRAW = 16


# --------------------------------------------------------------------------- data
def _side_labels(label_dir):
    """{(set, rally, frame): 1 if hitter is Top/far else 0} from ShuttleSet csv.
    Top = smaller player_location_y than the opponent (verified on the first hits of 041)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(LABELS_ROOT, label_dir, "set*.csv"))):
        s = int(os.path.basename(path)[3:-4])
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    fr = int(round(float(r["frame_num"])))
                    rl = int(float(r["rally"]))
                    py = float(r["player_location_y"])
                    oy = float(r["opponent_location_y"])
                except (TypeError, ValueError, KeyError):
                    continue
                out[(s, rl, fr)] = 1 if py < oy else 0
    return out


def build_cache(mid):
    """npz -> list of raw rally dicts (cached as pickle keyed by npz mtime+size)."""
    src = os.path.join(EXTRACT, mid + ".npz")
    st = os.stat(src)
    key = (st.st_mtime_ns, st.st_size)
    dst = os.path.join(CACHE, mid + ".pkl")
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            d = pickle.load(f)
        if d["key"] == key:
            return d
    z = np.load(src, allow_pickle=True)
    fps = float(z["fps"])
    W, H = float(z["width"]), float(z["height"])
    info = json.loads(str(z["info_json"][0]))
    label_dir = os.path.basename(info["labels"].rstrip("/"))
    sides = _side_labels(label_dir)
    frame = z["frame"]
    sxy = z["shuttle_xy"].astype(np.float64).copy()
    vis = z["shuttle_visible"] & np.isfinite(sxy).all(1)
    conf = z["shuttle_conf"].astype(np.float64)
    interp = z["shuttle_interp"].astype(np.float64)
    box = z["player_box"].astype(np.float64)
    foot = z["player_foot_world"].astype(np.float64)
    rs = z["rally_row_start"]
    hf, hs = z["rally_hit_frames"], z["rally_hit_start"]
    rallies = []
    for r in range(len(rs) - 1):
        a, b = int(rs[r]), int(rs[r + 1])
        T = b - a
        raw = np.full((T, NRAW), np.nan, np.float32)
        v = vis[a:b]
        raw[v, 0] = sxy[a:b][v, 0] / W
        raw[v, 1] = sxy[a:b][v, 1] / H
        raw[v, 2] = conf[a:b][v]
        raw[:, 3] = interp[a:b]
        for p in range(2):
            bx = box[a:b, p]
            ok = np.isfinite(bx).all(1)
            o = 4 + 4 * p
            raw[ok, o + 0] = (bx[ok, 0] + bx[ok, 2]) / 2 / W
            raw[ok, o + 1] = (bx[ok, 1] + bx[ok, 3]) / 2 / H
            raw[ok, o + 2] = (bx[ok, 2] - bx[ok, 0]) / W
            raw[ok, o + 3] = (bx[ok, 3] - bx[ok, 1]) / H
            ft = foot[a:b, p]
            ok = np.isfinite(ft).all(1)
            raw[ok, 12 + 2 * p] = ft[ok, 0]
            raw[ok, 13 + 2 * p] = ft[ok, 1]
        hits = hf[hs[r]:hs[r + 1]].astype(np.int64)
        idx = (hits - frame[a]).astype(np.float64)
        sd = np.array([sides.get((int(z["rally_set"][r]), int(z["rally_id"][r]), int(h)), -1)
                       for h in hits], np.int64)
        rallies.append(dict(raw=raw, fps=fps, hits=idx, side=sd, set=int(z["rally_set"][r]),
                            rally=int(z["rally_id"][r]), frames=frame[a:b].astype(np.int64),
                            mid=mid))
    d = dict(key=key, mid=mid, fps=fps, label_dir=label_dir, rallies=rallies)
    os.makedirs(CACHE, exist_ok=True)
    with open(dst + ".tmp", "wb") as f:
        pickle.dump(d, f)
    os.replace(dst + ".tmp", dst)
    return d


def scan_ids():
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(EXTRACT, "*.npz")))
    return ids


def load_match(mid):
    try:
        return build_cache(mid)
    except Exception as e:  # half-written npz from a running extract
        print("skip %s: %s" % (mid, e), file=sys.stderr)
        return None


# --------------------------------------------------------------------------- features
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


N_FEAT = None


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


def n_features():
    global N_FEAT
    if N_FEAT is None:
        N_FEAT = derive(np.full((5, NRAW), np.nan, np.float32), 30.0).shape[1]
    return N_FEAT


def mirror(raw):
    raw = raw.copy()
    for col in (0, 4, 8):
        raw[:, col] = 1.0 - raw[:, col]
    raw[:, 13] = -raw[:, 13]
    raw[:, 15] = -raw[:, 15]
    return raw


def stretch(raw, hits, s):
    """Time-stretch by factor s (>1 = more frames). Linear for values, nearest for NaN-edges."""
    T = len(raw)
    T2 = int(round((T - 1) * s)) + 1
    u = np.arange(T2) / s
    u = np.minimum(u, T - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    a = (u - i0)[:, None].astype(np.float32)
    near = np.where(a[:, 0] < 0.5, i0, i1)
    lin = (1 - a) * raw[i0] + a * raw[i1]
    out = np.where(np.isfinite(lin), lin, raw[near])
    out[:, 3] = raw[near, 3]
    # visibility: follow nearest sample (so a lone visible neighbour does not leak)
    miss = ~np.isfinite(raw[near, 0])
    out[miss, 0:3] = np.nan
    return out, hits * s


def drop_shuttle(raw, rng, max_frac=0.15):
    raw = raw.copy()
    T = len(raw)
    n = rng.integers(0, 6)
    for _ in range(n):
        L = int(rng.integers(3, 25))
        a = int(rng.integers(0, max(1, T - L)))
        raw[a:a + L, 0:3] = np.nan
    return raw


# --------------------------------------------------------------------------- targets
def make_targets(T, hits, side, sigma):
    t = np.arange(T, dtype=np.float32)
    y = np.zeros(T, np.float32)
    sd = np.zeros(T, np.float32)
    sm = np.zeros(T, np.float32)
    for h, s in zip(hits, side):
        y = np.maximum(y, np.exp(-((t - h) ** 2) / (2 * sigma ** 2)))
        if s >= 0:
            lo, hi = int(np.floor(h - 1)), int(np.ceil(h + 1))
            for k in range(max(lo, 0), min(hi, T - 1) + 1):
                if abs(k - h) <= 1.0:
                    sd[k] = s
                    sm[k] = 1
    return y.astype(np.float32), sd, sm


class WindowDS(torch.utils.data.Dataset):
    def __init__(self, rallies, n_samples, win=384, sigma=1.0, aug=True, stretch_range=(0.8, 1.3),
                 seed=0, use_stretch=True):
        self.r = rallies
        self.n = n_samples
        self.win = win
        self.sigma = sigma
        self.aug = aug
        self.sr = stretch_range
        self.use_stretch = use_stretch
        self.seed = seed
        lens = np.array([len(x["raw"]) for x in rallies], np.float64)
        self.p = lens / lens.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1000003 + i) % (2 ** 32) + int(torch.initial_seed()) % 997)
        r = self.r[rng.choice(len(self.r), p=self.p)]
        raw, hits, side, fps = r["raw"], r["hits"].copy(), r["side"], r["fps"]
        if self.aug:
            if rng.random() < 0.5:
                raw = mirror(raw)
            if self.use_stretch and rng.random() < 0.8:
                s = float(np.exp(rng.uniform(np.log(self.sr[0]), np.log(self.sr[1]))))
                raw, hits = stretch(raw, hits, s)
                fps = fps * s
            raw = drop_shuttle(raw, rng)
            raw = raw.copy()
            m = np.isfinite(raw[:, 0])
            raw[m, 0] += rng.normal(0, 0.0007, m.sum())
            raw[m, 1] += rng.normal(0, 0.0007, m.sum())
        X = derive(raw, fps)
        T = len(X)
        y, sd, sm = make_targets(T, hits, side, self.sigma)
        W = self.win
        if T >= W:
            a = int(rng.integers(0, T - W + 1))
            X, y, sd, sm = X[a:a + W], y[a:a + W], sd[a:a + W], sm[a:a + W]
            valid = np.ones(W, np.float32)
        else:
            pad = W - T
            valid = np.concatenate([np.ones(T, np.float32), np.zeros(pad, np.float32)])
            X = np.pad(X, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad)); sd = np.pad(sd, (0, pad)); sm = np.pad(sm, (0, pad))
        return (torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(sd),
                torch.from_numpy(sm), torch.from_numpy(valid))


# --------------------------------------------------------------------------- model
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
        return self.out(h).transpose(1, 2)  # (B,T,2)


class HitGRU(nn.Module):
    def __init__(self, n_in, hidden=96, layers=2, drop=0.15):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        # a small local conv front-end gives the GRU kink-scale context
        self.conv = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        h = F.gelu(self.inp(x))
        h = h + F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h)


def make_model(arch, n_in, hidden):
    if arch == "tcn":
        return HitTCN(n_in, hidden)
    if arch == "gru":
        return HitGRU(n_in, hidden)
    raise ValueError(arch)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_rally(models, raw, fps, device):
    """Mean logits over models. Returns (T,2) numpy."""
    X = torch.from_numpy(derive(raw, fps))[None].to(device)
    outs = []
    for m in models:
        outs.append(m(X)[0].float().cpu().numpy())
    return np.mean(outs, 0)


# --------------------------------------------------------------------------- decoding
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


def decode_alt(p, side_top, fps, thr, gap_s, lam, floor=None):
    """Max-sum subset of candidate peaks: gain (logit(p) - logit(thr)) per pick, hard min
    gap, and a soft MonoTrack-style alternation reward: consecutive picks gain
    lam*(P(alternate) - P(same side)) using the side head probabilities."""
    gap = max(1, int(round(gap_s * fps)))
    cand = local_peaks(p, floor if floor is not None else min(thr, 0.05))
    n = len(cand)
    if n == 0:
        return []
    lg = np.log(np.clip(p[cand], 1e-6, 1 - 1e-6) / (1 - np.clip(p[cand], 1e-6, 1 - 1e-6)))
    lt = np.log(thr / (1 - thr))
    gain = lg - lt
    ps = side_top[cand]
    best = np.full(n, -1e18)
    prev = np.full(n, -1)
    start = gain.copy()  # chain starting here
    for i in range(n):
        best[i] = start[i]
        for j in range(i):
            if cand[i] - cand[j] < gap:
                continue
            same = ps[i] * ps[j] + (1 - ps[i]) * (1 - ps[j])
            pair = lam * (1 - 2 * same)
            v = best[j] + gain[i] + pair
            if v > best[i]:
                best[i] = v
                prev[i] = j
    # chain may be empty if everything negative
    e = int(np.argmax(best))
    if best[e] <= 0:
        return []
    out = []
    while e >= 0:
        out.append(int(cand[e]))
        e = prev[e]
    out = out[::-1]
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def detections_from_logits(logits, fps, frames, params):
    """logits (T,2) -> list of dict(frame,t,confidence,source,hitter)."""
    p = sigmoid(logits[:, 0])
    st = sigmoid(logits[:, 1])
    if params.get("lam", 0.0) > 0:
        idx = decode_alt(p, st, fps, params["thr"], params["gap_s"], params["lam"])
    else:
        idx = decode_nms(p, fps, params["thr"], params["gap_s"])
    out = []
    for i in idx:
        out.append(dict(frame=int(frames[i]), t=float(i / fps), row=int(i), confidence=float(p[i]),
                        source="hitnet", hitter="far" if st[i] > 0.5 else "near"))
    return out


def write_match_json(path, mid, label_dir, rallies, dets, thr):
    obj = dict(source=mid + ".npz", labelDir=label_dir, threshold=float(thr), usedCamera=True,
               usedPlayers=True,
               rallies=[dict(set=r["set"], rally=r["rally"], videoStart=int(r["frames"][0]),
                             videoStop=int(r["frames"][-1]), hits=d)
                        for r, d in zip(rallies, dets)])
    with open(path, "w") as f:
        json.dump(obj, f)


# --------------------------------------------------------------------------- quick in-process eval
def eval_dets(rallies, dets, tols=(1, 2, 4)):
    """Score detections vs npz labels with the official pairing; returns dict tol -> (tp,fp,fn),
    plus serve/rally recall at +-2."""
    res = {t: [0, 0, 0] for t in tols}
    srv = [0, 0]
    rly = [0, 0]
    for r, d in zip(rallies, dets):
        gf = sorted(int(r["frames"][int(round(h))]) for h in r["hits"])
        dd = sorted(d, key=lambda x: x["frame"])
        kept, miss, free = scorelib.pair(gf, dd, max(tols))
        for t in tols:
            k = [x for x in kept if abs(x[2]) <= t]
            res[t][0] += len(k)
            res[t][1] += len(dd) - len(k)
            res[t][2] += len(gf) - len(k)
        k2 = {x[0] for x in kept if abs(x[2]) <= 2}
        for gi in range(len(gf)):
            tgt = srv if gi == 0 else rly
            tgt[1] += 1
            tgt[0] += gi in k2
    return res, srv, rly


def f1_of(res, t=2):
    tp, fp, fn = res[t]
    return scorelib.prf(tp, fp, fn)[2]
