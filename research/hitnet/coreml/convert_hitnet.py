"""Convert the shipped hitnet detector (3-TCN logit-average ensemble) to a CoreML mlprogram.

  /path/to/shuttl/research/coreml/venv/bin/python convert_hitnet.py

Input : features (1,T,51) float32, T = whole-rally frame count (RangeDim 1..4096; the longest
        rally in the 5 held-out matches is 1328 frames).
Output: logits   (1,T,2) float32, mean over the 3 members, computed inside the traced graph.

The traced graph is asserted equal to eager, input-dependent, and equal on many real rallies with
differing T before conversion.  fp32 only — the sibling BST port measured 12-54% argmax flips in
fp16, and this decoder thresholds probabilities, so fp16 is not shipped (--fp16-too exists only to
re-measure that claim).
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch
import coremltools as ct

import hitnet_core as H

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

MAX_T = 4096
N_FEAT = H.C.derive(np.full((5, H.C.NRAW), np.nan, np.float32), 30.0).shape[1]   # 51


def real_inputs(limit=6):
    """Feature matrices of real rallies from the 5 held-out matches, longest-first variety."""
    out = []
    for mid in H.TEST_IDS:
        m = H.C.load_match(mid)
        lens = np.array([len(r["raw"]) for r in m["rallies"]])
        for pick in (int(np.argmax(lens)), int(np.argmin(lens)), len(lens) // 2):
            r = m["rallies"][pick]
            out.append((f"{mid}/{r['set']}.{r['rally']}",
                        H.C.derive(r["raw"], r["fps"])[None].astype(np.float32)))
    out.sort(key=lambda kv: kv[1].shape[1])
    keep = out[:limit // 2] + out[-(limit - limit // 2):]
    return keep


def build(members):
    ens, paths = H.load_ensemble(members)
    rows = real_inputs()
    xs = [torch.from_numpy(x) for _, x in rows]
    with torch.no_grad():
        traced = torch.jit.trace(ens, (xs[-1],), check_trace=False)
        errs, shapes = [], []
        for (name, _), x in zip(rows, xs):
            a, t = ens(x), traced(x)
            err = float((a - t).abs().max())
            assert err < 1e-5, (name, err)
            assert a.shape == (1, x.shape[1], 2), a.shape
            errs.append(err)
            shapes.append(int(x.shape[1]))
        assert not torch.allclose(ens(xs[0])[0, :32], ens(xs[-1])[0, :32]), "input ignored"
    print(f"traced == eager on {len(rows)} real rallies, T {shapes[0]}..{shapes[-1]}: "
          f"max|diff| {max(errs):.2e}")
    return ens, traced, paths, rows[0][1]


def convert(ens, traced, paths, example, out, fp16):
    x = torch.from_numpy(example)
    with torch.no_grad():
        ex = traced(x)
    inputs = [ct.TensorType(name="features", shape=(1, ct.RangeDim(1, MAX_T), N_FEAT),
                            dtype=np.float32)]
    m = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                   convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                   compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32)
    m.short_description = ("hitnet badminton hit-frame detector (bidirectional dilated TCN, 12 blocks, "
                           f"96 ch), mean logits of {len(paths)} checkpoints.")
    m.author = "Shuttl research/hitnet"
    meta = {
        "members": ",".join(os.path.basename(p) for p in paths),
        "n_members": str(len(paths)),
        "n_in": str(example.shape[2]),
        "n_out": "2",
        "precision": "fp16" if fp16 else "fp32",
        "pooling": "none (whole-rally, non-causal); for T > 384 use overlapping 384/192 windows, centre-only",
        "input_contract": ("features (1,T,51) float32, one row per video frame, from common.derive(raw, fps); "
                           "see coreml/README.md for the exact 51-channel layout and scaling"),
        "output": "logits (1,T,2) raw pre-softmax: [hit, hitter-is-far/Top]; sigmoid, threshold 0.4, "
                  "NMS min gap 0.30 s",
        "decoder": json.dumps(H.PARAMS),
        "reference_f1_pm2": str(H.REF_F1_2),
    }
    for k, v in meta.items():
        m.user_defined_metadata[k] = v
    m.input_description["features"] = "per-frame feature matrix (1,T,51), common.derive(raw,fps), float32"
    m.output_description["logits"] = "per-frame logits (1,T,2): hit, hitter-is-far"
    p = H.HERE / out
    m.save(str(p))
    print("wrote", p)
    return p, ex.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=None, help="comma-separated ckpt paths (default: shipped 3)")
    ap.add_argument("--out", default="HitNetDetector.mlpackage")
    ap.add_argument("--fp16-too", action="store_true")
    a = ap.parse_args()
    members = a.members.split(",") if a.members else None
    ens, traced, paths, example = build(members)
    p, shape = convert(ens, traced, paths, example, a.out, fp16=False)
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"{a.out}: {size / 1e6:.2f} MB, example output {shape}")
    if a.fp16_too:
        convert(ens, traced, paths, example, a.out.replace(".mlpackage", "_fp16.mlpackage"), fp16=True)


if __name__ == "__main__":
    main()
