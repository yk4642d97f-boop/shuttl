"""Shared pieces of the BST-CG-AP seq30 CoreML port.

Run with the coremltools venv (research/coreml/venv: coremltools 9.0, torch 2.14.1). The BST repo
imports `positional_encodings` / `torchinfo`, which only the BST venv has, so that venv's
site-packages is APPENDED to sys.path (it can only supply modules the coreml venv lacks; torch and
numpy still come from the coreml venv).

Everything here mirrors the reference exactly:
  - class order        = repo get_merged_stroke_types()  (asserted against convert.CLASSES)
  - seq-len transform  = repo make_seq_len_same(30, ...)   (truncate / zero-pad at the end)
  - bones              = repo create_bones(joints, coco pairs), concatenated after the joints
  - network            = repo BST_CG_AP(in_dim=72, n_class=25, seq_len=30, depth_tem=2, depth_inter=1)
"""
import glob
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BST = HERE.parent
ROOT = BST.parent.parent
REPO = BST / "repo" / "stroke_classification"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BST))
_bst_site = BST / "venv" / "lib" / "python3.12" / "site-packages"
if str(_bst_site) not in sys.path:
    sys.path.append(str(_bst_site))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from preparing_data.shuttleset_dataset import (  # noqa: E402
    create_bones, get_bone_pairs, get_merged_stroke_types, make_seq_len_same)
from model.bst import BST_CG_AP  # noqa: E402

SEQ_LEN = 30
N_CLASS = 25
N_JOINTS = 17
PAIRS = get_bone_pairs("coco")
IN_DIM = (N_JOINTS + len(PAIRS)) * 2          # 72
CLASSES = get_merged_stroke_types()
WEIGHTS = sorted(glob.glob(str(BST / "weights" / "bst_CG_AP_JnB_bone_merged*.pt")),
                 key=lambda p: (len(Path(p).stem), Path(p).stem))
M039 = ROOT / "research" / "e2e" / "m039_ctrl30"
M039_COLLATED = ROOT / "research" / "e2e" / "m039_ctrl30_collated"


def _check_classes():
    # convert.py writes the class folders; the collate step indexes them with
    # get_merged_stroke_types(). Both must agree or every label is silently wrong.
    import importlib.util
    spec = importlib.util.spec_from_file_location("bst_convert", BST / "convert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(mod.CLASSES) == list(CLASSES), (mod.CLASSES, CLASSES)
    assert len(CLASSES) == N_CLASS


_check_classes()


def load_net(path):
    net = BST_CG_AP(in_dim=IN_DIM, n_class=N_CLASS, seq_len=SEQ_LEN, depth_tem=2, depth_inter=1)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval()


NEG = -1e9   # stands in for masked_fill(-inf): exp(NEG - max) is exactly 0 in fp32 (fp16: -inf)


class StaticBST(nn.Module):
    """BST_CG_AP.forward re-spelled with static shapes (batch 1, 2 players, 30 frames).

    coremltools cannot convert the repo forward as traced (shape arithmetic such as
    `b*n` from `JnB.shape` becomes aten::Int on a 1-element tensor). This module re-uses the
    original net's own submodules and parameters and changes only the spelling: constant shapes,
    the attention mask as `where(mask, dots, NEG)` instead of masked_fill(-inf), and the cosine
    similarity written out. `reference.py`/`convert_bst.py` assert it equals the repo forward.
    """
    T, N, D, H = SEQ_LEN, 2, 100, 6

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.dh = net.encoder_tem.layers[0].attn.to_qkv.out_features // (3 * self.H)
        self.scale = self.dh ** -0.5

    def self_attn(self, attn, x, mask):
        # x (S, 31, D); mask (1, 1, 1, 31) bool
        S, L = x.shape[0], self.T + 1
        qkv = attn.to_qkv(x).reshape(S, L, self.H, 3 * self.dh)
        q, k, v = qkv[..., :self.dh], qkv[..., self.dh:2 * self.dh], qkv[..., 2 * self.dh:]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)        # (S,H,L,dh)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = torch.where(mask, dots, torch.full_like(dots, NEG))
        a = torch.matmul(torch.softmax(dots, dim=-1), v)
        return attn.tail(a.transpose(1, 2).reshape(S, L, self.H * self.dh))

    def encoder(self, enc, x, mask):
        for layer in enc.layers:
            x = self.self_attn(layer.attn, layer.layer_norm1(x), mask) + x
            x = layer.ff(layer.layer_norm2(x)) + x
        return x

    def cross(self, layer, x1, x2, mask):
        # x1, x2 (1, 30, D); mask (1, 1, 1, 30) bool
        ca = layer.cross_attn
        x1, x2 = layer.layer_norm1_x1(x1), layer.layer_norm1_x2(x2)
        q = ca.to_q(x1).reshape(1, self.T, self.H, self.dh).transpose(1, 2)
        kv = ca.to_kv(x2).reshape(1, self.T, self.H, 2 * self.dh)
        k, v = kv[..., :self.dh].transpose(1, 2), kv[..., self.dh:].transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = torch.where(mask, dots, torch.full_like(dots, NEG))
        a = torch.matmul(torch.softmax(dots, dim=-1), v)
        x = ca.tail(a.transpose(1, 2).reshape(1, self.T, self.H * self.dh))
        return layer.ff(layer.layer_norm2(x)) + x

    @staticmethod
    def cos(x, y):
        # torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        num = (x * y).sum(dim=1)
        den = torch.sqrt(torch.clamp((x * x).sum(dim=1) * (y * y).sum(dim=1), min=1e-16))
        return num / den

    def forward(self, jnb, shuttle, pos, valid):
        net, T, N, D = self.net, self.T, self.N, self.D
        x = jnb.permute(0, 2, 3, 1).reshape(N, IN_DIM, T)
        pi = net.mlp_positions(pos).permute(0, 2, 3, 1).reshape(N, IN_DIM, T)
        x = x * pi + x
        x = net.tcn_pose(x).reshape(1, N, D, T).transpose(-2, -1)            # (1,2,T,D)
        s = net.tcn_shuttle(shuttle.transpose(1, 2)).unsqueeze(1).transpose(-2, -1)   # (1,1,T,D)
        x = torch.cat((x, s), dim=1).reshape(N + 1, T, D)
        tok = net.learned_token_tem.reshape(1, 1, D).expand(N + 1, 1, D)
        x = torch.cat((tok, x), dim=1) + net.embedding_tem                    # (3,31,D)
        m31 = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1) > 0.5  # (1,31)
        m31 = m31.reshape(1, 1, 1, T + 1)
        x = self.encoder(net.encoder_tem, x, m31)
        p1, p2, sh = x[0:1], x[1:2], x[2:3]                                   # (1,31,D)
        p1_cls, p2_cls, sh_cls = p1[:, 0], p2[:, 0], sh[:, 0]
        p1 = p1[:, 1:] + net.embedding_cross
        p2 = p2[:, 1:] + net.embedding_cross
        sh = sh[:, 1:] + net.embedding_cross
        m30 = (valid > 0.5).reshape(1, 1, 1, T)
        p1s = self.cross(net.cross_trans, p1, sh, m30)
        p2s = self.cross(net.cross_trans, p2, sh, m30)
        ti = net.learned_token_inter.reshape(1, 1, D)
        p1s = torch.cat((ti, p1s), dim=1) + net.embedding_inter
        p2s = torch.cat((ti, p2s), dim=1) + net.embedding_inter
        p1s = self.encoder(net.encoder_inter, p1s, m31)[:, 0]
        p2s = self.encoder(net.encoder_inter, p2s, m31)[:, 0]
        alpha = ((self.cos(p1s, sh_cls) - self.cos(p2s, sh_cls) + 2) / 4).unsqueeze(1)
        p1c = alpha * (p1_cls + p1s)
        p2c = (1 - alpha) * (p2_cls + p2s)
        sh_cls = sh_cls - net.mlp_clean(torch.minimum(p1s, p2s))
        return net.mlp_head(torch.cat((p1c, p2c, sh_cls), dim=1))


class BSTCoreML(nn.Module):
    """The shipped graph: raw convert.py-layout clip in, mean logits of the members out.

    Inputs (batch 1, every array zero-padded at the END to 30 frames, as make_seq_len_same does):
      joints  (1, 30, 2, 17, 2)  float32
      pos     (1, 30, 2, 2)      float32
      shuttle (1, 30, 2)         float32
      valid   (1, 30)            float32  1 for the first video_len frames, 0 for padding
    Output:
      logits  (1, 25)            float32  mean over members of the raw (pre-softmax) logits

    Bones are built inside, with create_bones' rule: per coordinate, end - start where BOTH
    joints' coordinate is non-zero, else 0.
    """

    def __init__(self, nets, static=False):
        super().__init__()
        self.static = static
        self.nets = nn.ModuleList([StaticBST(n) for n in nets] if static else nets)
        self.register_buffer("i0", torch.tensor([a for a, _ in PAIRS], dtype=torch.long))
        self.register_buffer("i1", torch.tensor([b for _, b in PAIRS], dtype=torch.long))

    def forward(self, joints, pos, shuttle, valid):
        s = torch.index_select(joints, 3, self.i0)
        e = torch.index_select(joints, 3, self.i1)
        keep = (s != 0.0) & (e != 0.0)
        bones = torch.where(keep, e - s, torch.zeros_like(s))
        jnb = torch.cat((joints, bones), dim=3)                       # (1,30,2,36,2)
        jnb = jnb.reshape(1, SEQ_LEN, 2, IN_DIM)                       # (1,30,2,72)
        video_len = valid.sum(dim=1)                                   # (1,)
        out = None
        for net in self.nets:
            y = net(jnb, shuttle, pos, valid) if self.static else net(jnb, shuttle, pos, video_len)
            out = y if out is None else out + y
        return out / float(len(self.nets))


# ---------------------------------------------------------------- data
def raw_clips(root=M039, split="test"):
    """[(name, class_index, joints, pos, shuttle)] in collate_npy's order (iterdir, sorted names)."""
    out = []
    target = Path(root) / split
    for typ in target.iterdir():                     # same iteration as collate_npy
        shots = sorted(str(s).replace("_pos.npy", "") for s in typ.glob("*_pos.npy"))
        for b in shots:
            out.append((os.path.basename(b), CLASSES.index(typ.name),
                        np.load(b + "_joints.npy"), np.load(b + "_pos.npy"), np.load(b + "_shuttle.npy")))
    return out


def model_inputs(joints, pos, shuttle):
    """convert.py arrays -> the CoreML input dict (make_seq_len_same, then the valid mask)."""
    j, p, s, n = make_seq_len_same(SEQ_LEN, joints.astype(np.float32), pos.astype(np.float32),
                                   shuttle.astype(np.float32))
    valid = np.zeros((1, SEQ_LEN), np.float32)
    valid[0, :n] = 1.0
    return {"joints": j[None].astype(np.float32), "pos": p[None].astype(np.float32),
            "shuttle": s[None].astype(np.float32), "valid": valid}, n


def reference_jnb(joints):
    """The repo's own JnB_bone transform of one padded clip (t, 2, 36, 2)."""
    return np.concatenate((joints, create_bones(joints, PAIRS)), axis=-2)


def reference_logits(joints, pos, shuttle, nets=None):
    """Mean logits of the released checkpoints for raw convert.py arrays, via the repo forward.

    `joints`/`pos`/`shuttle` are the convert.py arrays of ONE clip (t, ...) — make_seq_len_same
    truncates a longer clip and zero-pads a shorter one, so this is the forward the measured
    numbers came from. Returns (logits (25,), video_len).
    """
    d, n = model_inputs(joints, pos, shuttle)
    # eval_bst.py flattens the last two axes: JnB (b, t, n, J+B, 2) -> (b, t, n, in_dim)
    jnb = reference_jnb(d["joints"][0])[None].reshape(1, SEQ_LEN, 2, IN_DIM).astype(np.float32)
    args = (torch.from_numpy(jnb), torch.from_numpy(d["shuttle"]), torch.from_numpy(d["pos"]),
            torch.from_numpy(np.array([n], dtype=np.int64)))
    assert jnb.shape == (1, SEQ_LEN, 2, IN_DIM), jnb.shape
    outs = []
    with torch.no_grad():
        for net in (nets if nets is not None else [load_net(w) for w in WEIGHTS]):
            outs.append(net(*args).numpy()[0])
    return np.mean(outs, 0), n
