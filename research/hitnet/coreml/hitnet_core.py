"""Shared pieces of the hitnet CoreML port.

Run with the coremltools venv: /path/to/shuttl/research/coreml/venv/bin/python
(coremltools 9.0, torch 2.14.1).  `common.py` from the parent directory is imported as-is — its
`derive()`, `make_model()`, decoder and scorer are the reference and are never re-spelled here.

The shipped model = the 3-TCN logit-average ensemble of REPORT.md
(`ckpt/tcn_s0.pt`, `ckpt/tcn_s1.pt`, `ckpt/tcn_lr.pt`), whole-rally non-causal, decoder
thr 0.4 / gap 0.30 s / lam 0 -> pooled test F1@+-2 = 0.9212 on 039 040 042 043 044.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HITNET = HERE.parent
sys.path.insert(0, str(HITNET))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import common as C  # noqa: E402

MEMBERS = ["ckpt/tcn_s0.pt", "ckpt/tcn_s1.pt", "ckpt/tcn_lr.pt"]
TEST_IDS = C.TEST_IDS                                    # 039 040 042 043 044
PARAMS = dict(thr=0.4, gap_s=0.3, lam=0.0)               # tuned on val 041 (tune.py), shipped default
REF_F1_2 = 0.9212                                        # REPORT.md, pooled over the 5 test matches
WIN, STRIDE = 384, 192                                   # infer.predict_windowed


# --------------------------------------------------------------------------- model
class Ensemble(nn.Module):
    """Mean of the members' logits. (B,T,51) -> (B,T,2), all fp32."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList([n.eval() for n in nets])

    def forward(self, x):
        out = None
        for net in self.nets:
            y = net(x)
            out = y if out is None else out + y
        return out / float(len(self.nets))


def ckpt_path(rel):
    return str(HITNET / rel)


def load_net(path):
    ck = torch.load(path, map_location="cpu")
    net = C.make_model(ck["arch"], ck["n_in"], ck["hidden"])
    net.load_state_dict(ck["state"])
    return net.eval()


def load_ensemble(members=None):
    members = MEMBERS if members is None else members
    nets = [load_net(ckpt_path(p) if not os.path.isabs(p) else p) for p in members]
    return Ensemble(nets).eval(), members


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def logits_full(ens, raw, fps):
    """Whole-rally (non-causal) logits, i.e. common.predict_rally. (T,2) float32."""
    X = torch.from_numpy(C.derive(raw, fps))[None]
    return ens(X)[0].float().numpy()


@torch.no_grad()
def logits_windowed(ens, raw, fps, win=WIN, stride=STRIDE):
    """infer.predict_windowed: overlapping windows, centre-only. (T,2) float32.

    For T <= win this is exactly logits_full (infer falls back to the whole-rally path).
    The CoreML model sees a whole rally in one shot, so this is the input the port is
    compared against on real data; the two spellings differ by <= 0.32 logit (REPORT.md).
    """
    T = len(raw)
    X = torch.from_numpy(C.derive(raw, fps))
    if T <= win:
        return ens(X[None])[0].float().numpy()
    out = np.zeros((T, 2), np.float32)
    starts = list(range(0, T - win + 1, stride))
    if starts[-1] != T - win:
        starts.append(T - win)
    margin = (win - stride) // 2
    for k, a in enumerate(starts):
        lg = ens(X[a:a + win][None])[0].float().numpy()
        lo = 0 if k == 0 else margin
        hi = win if k == len(starts) - 1 else win - margin
        out[a + lo:a + hi] = lg[lo:hi]
    return out


# --------------------------------------------------------------------------- decoding / scoring
def decode(logits, fps, frames, params=None):
    return C.detections_from_logits(logits, fps, frames, PARAMS if params is None else params)


def _pooled(per_match):
    """per_match: {mid: (rallies, dets)} -> (tols -> (tp,fp,fn), serve, rally, f1@2)."""
    res = {t: [0, 0, 0] for t in (1, 2, 4)}
    srv, rly = [0, 0], [0, 0]
    for rallies, dets in per_match.values():
        r, s, l = C.eval_dets(rallies, dets)
        for t in res:
            for i in range(3):
                res[t][i] += r[t][i]
        srv[0] += s[0]; srv[1] += s[1]
        rly[0] += l[0]; rly[1] += l[1]
    return res, srv, rly


def score_dets(per_match):
    res, srv, rly = _pooled(per_match)
    p, r, f = C.scorelib.prf(*res[2])
    return dict(f1_2=float(f), p_2=float(p), r_2=float(r), counts_2=[int(x) for x in res[2]],
                counts_1=[int(x) for x in res[1]], counts_4=[int(x) for x in res[4]],
                f1_1=float(C.scorelib.prf(*res[1])[2]), f1_4=float(C.scorelib.prf(*res[4])[2]),
                serve_recall=[int(x) for x in srv], rally_recall=[int(x) for x in rly],
                n_detections=int(res[2][0] + res[2][1]))
