"""CoreML vs PyTorch parity + accuracy on the 749 match-039 clips (and a 25-frame masked variant).

  ../../coreml/venv/bin/python parity_bst.py [--units CPU_ONLY,CPU_AND_NE,ALL] [--models ...]

Reference logits: reference_logits.npz (reference.py, repo forward, CPU fp32).
  ensemble model  <- mean over the 10 checkpoints
  single0 model   <- checkpoint bst_CG_AP_JnB_bone_merged.pt
The "t25" set re-uses every clip's first 25 frames (5 end-padded, masked frames): the 30-fps match
never exercises the padding path that 25-fps video takes; its reference is computed here with the
repo forward and video_len = 25.
writes parity.json
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch
import coremltools as ct

import bstcore as B

warnings.filterwarnings("ignore")
torch.set_num_threads(2)


def t25_reference(clips, members):
    ins = [B.model_inputs(j[:25], p[:25], s[:25])[0] for _, _, j, p, s in clips]
    jnb = np.stack([B.reference_jnb(d["joints"][0]) for d in ins]).reshape(len(ins), B.SEQ_LEN, 2, -1)
    pos = np.concatenate([d["pos"] for d in ins]); sh = np.concatenate([d["shuttle"] for d in ins])
    vl = torch.full((len(ins),), 25, dtype=torch.long)
    outs = []
    with torch.no_grad():
        for w in members:
            net = B.load_net(w)
            outs.append(net(torch.from_numpy(jnb), torch.from_numpy(sh), torch.from_numpy(pos), vl).numpy())
    return ins, np.stack(outs)


def run(model, ins):
    out = np.zeros((len(ins), B.N_CLASS), np.float32)
    t = time.time()
    for i, d in enumerate(ins):
        out[i] = model.predict(d)["logits"].reshape(-1)
    return out, (time.time() - t) / len(ins)


def margins(ref):
    s = np.sort(ref, 1)
    return s[:, -1] - s[:, -2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="CPU_ONLY,CPU_AND_NE,ALL")
    ap.add_argument("--models", default="BSTStroke,BSTStroke_fp16,BSTStroke_single0,BSTStroke_single0_fp16")
    a = ap.parse_args()

    ref = np.load(B.HERE / "reference_logits.npz")
    per, labels = ref["per"], ref["labels"]
    clips = B.raw_clips()
    assert [c[0] for c in clips] == list(ref["names"])
    full = [B.model_inputs(j, p, s)[0] for _, _, j, p, s in clips]
    t25_in, t25_per = t25_reference(clips, B.WEIGHTS)
    refs = {"ensemble": (per.mean(0), t25_per.mean(0)), "single0": (per[0], t25_per[0])}
    print(f"torch: ensemble acc {np.mean(per.mean(0).argmax(1) == labels):.4f}, "
          f"single0 acc {np.mean(per[0].argmax(1) == labels):.4f}, mean single "
          f"{np.mean(per.argmax(2) == labels[None]):.4f}; t25 ensemble acc {np.mean(t25_per.mean(0).argmax(1) == labels):.4f}")
    mg = margins(per.mean(0))
    print(f"ensemble top1-top2 logit margin: min {mg.min():.4f}, p1 {np.percentile(mg, 1):.4f}, median {np.median(mg):.3f}")

    res = {"n_clips": len(clips), "torch_acc_ensemble": float(np.mean(per.mean(0).argmax(1) == labels)),
           "torch_acc_mean_single": float(np.mean(per.argmax(2) == labels[None])),
           "torch_acc_single0": float(np.mean(per[0].argmax(1) == labels)),
           "ensemble_margin_min": float(mg.min()), "ensemble_margin_p1": float(np.percentile(mg, 1)),
           "runs": []}
    for name in a.models.split(","):
        kind = "single0" if "single0" in name else "ensemble"
        r_full, r_t25 = refs[kind]
        for unit in a.units.split(","):
            m = ct.models.MLModel(str(B.HERE / f"{name}.mlpackage"), compute_units=getattr(ct.ComputeUnit, unit))
            got, sec = run(m, full)
            g25, _ = run(m, t25_in)
            row = {"model": name, "units": unit,
                   "max_abs_logit_diff": float(np.abs(got - r_full).max()),
                   "argmax_agree": float(np.mean(got.argmax(1) == r_full.argmax(1))),
                   "n_flips": int(np.sum(got.argmax(1) != r_full.argmax(1))),
                   "acc": float(np.mean(got.argmax(1) == labels)),
                   "t25_max_abs_logit_diff": float(np.abs(g25 - r_t25).max()),
                   "t25_argmax_agree": float(np.mean(g25.argmax(1) == r_t25.argmax(1))),
                   "ms_per_clip": 1000 * sec}
            res["runs"].append(row)
            print(f"{name:24s} {unit:11s} max|dlogit| {row['max_abs_logit_diff']:.2e}  argmax agree "
                  f"{row['argmax_agree']:.4f} ({row['n_flips']} flips)  acc {row['acc']:.4f}  | t25 max|d| "
                  f"{row['t25_max_abs_logit_diff']:.2e} agree {row['t25_argmax_agree']:.4f}  | {row['ms_per_clip']:.1f} ms/clip",
                  flush=True)
    json.dump(res, open(B.HERE / "parity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
