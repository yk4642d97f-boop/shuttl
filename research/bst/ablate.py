#!/usr/bin/env python3
"""Ablation: which of the three BST inputs breaks the released weights on our features?

Builds hybrid clip sets that mix our extracted features with the official ShuttleSet
features for the SAME clips (matched by clip name), so a drop can be attributed to one
input branch.  Every variant is then collated and evaluated with the released weights.

  official_all   official joints + official pos + official shuttle   (control)
  ours_all       ours + ours + ours
  ours_joints    ours joints + official pos/shuttle
  ours_pos       official joints + ours pos + official shuttle
  ours_shuttle   official joints/pos + ours shuttle

usage: venv/bin/python ablate.py [--limit N]
"""
import argparse, glob, os, random, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = os.path.join(HERE, "data/ShuttleSet_data_merged/dataset_npy_between_2_hits_with_max_limits/train")
OUR = "/tmp/bst_c1b/train"
WORK = "/tmp/ablate"
COLL = "/tmp/ablate_coll"

VARIANTS = {
    "official_all": (0, 0, 0),
    "ours_all": (1, 1, 1),
    "ours_joints": (1, 0, 0),
    "ours_pos": (0, 1, 0),
    "ours_shuttle": (0, 0, 1),
}


def pairs():
    out = []
    for f in sorted(glob.glob(OUR + "/*/*_pos.npy")):
        name = os.path.basename(f)[:-8]
        cls = os.path.basename(os.path.dirname(f))
        g = os.path.join(OFF, cls, name)
        if os.path.exists(g + "_pos.npy"):
            out.append((name, cls, g, f[:-8]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ps = pairs()
    if a.limit:
        ps = ps[:a.limit]
    print(f"clips available for matching: {len(ps)}")
    if not ps:
        sys.exit("nothing to compare")

    lens_o, lens_f = [], []
    for name, cls, g, m in ps:
        lens_o.append(len(np.load(g + "_joints.npy")))
        lens_f.append(len(np.load(m + "_joints.npy")))
    print(f"clip length: ours median {np.median(lens_f):.0f}, official median {np.median(lens_o):.0f}")

    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(COLL, ignore_errors=True)
    counts = {k: 0 for k in VARIANTS}
    for name, cls, g, m in ps:
        oj = np.load(g + "_joints.npy")
        op = np.load(g + "_pos.npy")
        os_ = np.load(g + "_shuttle.npy")
        nj = np.load(m + "_joints.npy")
        np_ = np.load(m + "_pos.npy")
        ns = np.load(m + "_shuttle.npy")
        T = min(len(oj), len(nj), len(op), len(np_), len(os_), len(ns))
        if T < 8:
            continue
        zo = np.abs(oj[:T]).sum(axis=(1, 2, 3)) > 0
        zn = np.abs(nj[:T]).sum(axis=(1, 2, 3)) > 0
        keep = zo & zn
        if keep.sum() < 8:
            continue
        for v, (uj, up, us) in VARIANTS.items():
            d = os.path.join(WORK, v, "train", cls)
            os.makedirs(d, exist_ok=True)
            base = os.path.join(d, name)
            np.save(base + "_joints.npy", (nj if uj else oj)[:T][keep].astype(np.float32))
            np.save(base + "_pos.npy", (np_ if up else op)[:T][keep].astype(np.float32))
            np.save(base + "_shuttle.npy", (ns if us else os_)[:T][keep].astype(np.float32))
            counts[v] += 1
    print("clips written per variant:", counts)

    for v in VARIANTS:
        subprocess.run([sys.executable, os.path.join(HERE, "collate.py"),
                        os.path.join(WORK, v), os.path.join(COLL, v), "100", "train"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        out = f"/tmp/ablate_eval_{v}.json"
        subprocess.run([sys.executable, os.path.join(HERE, "eval_bst.py"),
                        "--data", os.path.join(COLL, v), "--seq-len", "100", "--set", "train",
                        "--device", "mps", "--out", out,
                        "--weights", *glob.glob(os.path.join(HERE,
                            "weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt"))],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        import json
        r = list(json.load(open(out))["results"].values())
        print(f"{v:15s} acc {np.mean([x['acc'] for x in r]):.4f}  "
              f"macroF1 {np.mean([x['macro_f1'] for x in r]):.4f}  "
              f"top2 {np.mean([x['top2'] for x in r]):.4f}")


if __name__ == "__main__":
    main()
