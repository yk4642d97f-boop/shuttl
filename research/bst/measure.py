#!/usr/bin/env python3
"""End-to-end stroke-classification accuracy harness for the Shuttl/BST claim.

Claim under test
    Features produced by OUR iPhone pipeline achieve the same stroke-classification
    accuracy as the published ShuttleSet features, when BST's expected pose (RTMPose-m)
    is used.

One command, one report:

    bst/venv/bin/python bst/measure.py                  # measure everything that exists now
    bst/venv/bin/python bst/measure.py --wait           # keep polling, re-report on new data
    bst/venv/bin/python bst/measure.py --self-test-318  # reproduce the 318-clip match-1 numbers

Writes bst/ACCURACY.md and bst/accuracy.json.  Drives convert.py / collate.py / eval_bst's
forward pass and the released weights; trains nothing, edits none of them, redoes no cached
work.

Arms — within a match every arm is collated and scored on EXACTLY the same clip names, so
every comparison is clip-for-clip and no number compares across different clip sets:
    ours_rtm   our pipeline's features, pose = RTMPose-m    <- the claim's setting
    ours_vis   our pipeline's features, pose = Apple Vision
    offi       official ShuttleSet features, same clip names, as published
A match's comparison set is the intersection of the clip names the arms can produce for it.
A clip whose raw joints are all zero in any arm is dropped from every arm (counted, reported).

Held-out matches: test = 39, 40, 42, 43, 44.  Aggregates are over held-out matches only;
val = 41 and the training-side matches are reported separately as regression figures.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../research/bst
RESEARCH = HERE.parent                          # .../research
VENV_PY = HERE / "venv" / "bin" / "python"

if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

TEST_IDS = (39, 40, 42, 43, 44)
VAL_IDS = (41,)
SEQ_LEN = 100
N_CLASSES = 25
ARMS = ("ours_rtm", "ours_vis", "offi")
OFFICIAL_RAW = HERE / "data" / "ShuttleSet_data_merged" / "dataset_npy_between_2_hits_with_max_limits"
OFFICIAL_COLL = (HERE / "data" / "ShuttleSet_data_merged"
                 / "dataset_npy_collated_between_2_hits_with_max_limits_seq_100")

OFFICIAL_MATCH_IDS = set()   # match ids the released ShuttleSet data covers; filled at startup

REPO = HERE / "repo" / "stroke_classification"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from preparing_data.shuttleset_dataset import get_merged_stroke_types  # noqa: E402

CLASSES = get_merged_stroke_types()
ENGLISH = {
    "未知球種": "none", "放小球": "net shot", "擋小球": "return net", "殺球": "smash",
    "挑球": "lob", "長球": "clear", "平球": "drive", "切球": "drop", "推球": "push",
    "撲球": "rush", "勾球": "cross-court net shot", "發短球": "short service",
    "發長球": "long service",
}


def english(name: str) -> str:
    if "_" in name:
        side, typ = name.split("_", 1)
        return f"{side} {ENGLISH.get(typ, typ)}"
    return ENGLISH.get(name, name)


def split_of(mid: int) -> str:
    return "test" if mid in TEST_IDS else "val" if mid in VAL_IDS else "train"


def log(msg: str, logfile: "Path | None" = None):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as fh:
            fh.write(line + "\n")


# --------------------------------------------------------------------------- metrics

def metrics_from_preds(pred, gt, top2hit, ncls=N_CLASSES):
    """Accuracy / macro-F1 / top-2 / per-class table — same definitions as eval_bst.metrics."""
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    top2hit = np.asarray(top2hit, dtype=bool)
    n = len(gt)
    f1 = np.zeros(ncls)
    prec = np.zeros(ncls)
    rec = np.zeros(ncls)
    sup = np.zeros(ncls, dtype=np.int64)
    for c in range(ncls):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        sup[c] = int((gt == c).sum())
        prec[c] = tp / (tp + fp) if tp + fp else 0.0
        rec[c] = tp / (tp + fn) if tp + fn else 0.0
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"n": int(n), "acc": float((pred == gt).mean()) if n else 0.0,
            "macro_f1": float(f1.mean()), "min_f1": float(f1.min()),
            "top2": float(top2hit.mean()) if n else 0.0,
            "f1_each": f1.tolist(), "precision_each": prec.tolist(),
            "recall_each": rec.tolist(), "support_each": sup.tolist()}


def row_from_weights(per_weight_metrics):
    """One report row: mean over the 10 released checkpoints, plus their spread."""
    row = {k: float(np.mean([p[k] for p in per_weight_metrics]))
           for k in ("acc", "macro_f1", "top2", "min_f1")}
    row["acc_std"] = float(np.std([p["acc"] for p in per_weight_metrics]))
    row["n"] = per_weight_metrics[0]["n"]
    return row


# --------------------------------------------------------------------------- plumbing

def run(cmd, logfile=None):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
                           f"--- stdout ---\n{p.stdout[-4000:]}\n--- stderr ---\n{p.stderr[-4000:]}")
    return p.stdout


def fingerprint(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        try:
            st = os.stat(p)
            h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
        except FileNotFoundError:
            h.update(f"{p}:missing".encode())
    return h.hexdigest()[:16]


def clip_bases(raw_dir: Path):
    """Clip base paths in a BST raw dir, in collate.py's own glob order."""
    return sorted(str(f)[:-8] for f in raw_dir.rglob("*_pos.npy"))


def is_degenerate(base: str) -> bool:
    try:
        return bool(np.abs(np.load(base + "_joints.npy")).sum() == 0)
    except FileNotFoundError:
        return True


def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def weights():
    ws = sorted(glob.glob(str(
        HERE / "weights" / "bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt")))
    if len(ws) != 10:
        raise RuntimeError(f"expected the 10 released checkpoints, found {len(ws)}")
    return ws


# --------------------------------------------------------------------------- data prep

def official_index(cache: Path):
    """name -> official raw base path, for every released clip (built once, cached)."""
    idx_file = cache / "official_index.json"
    stamp = fingerprint([OFFICIAL_RAW / s for s in ("train", "val", "test")])
    if idx_file.exists():
        try:
            d = json.load(open(idx_file))
            if d.get("stamp") == stamp:
                return d["index"]
        except Exception:                                            # noqa: BLE001
            pass
    index = {os.path.basename(str(p))[:-8]: str(p)[:-8] for p in OFFICIAL_RAW.rglob("*_pos.npy")}
    json.dump({"stamp": stamp, "index": index}, open(idx_file, "w"))
    log(f"official index: {len(index)} released clips")
    return index


def build_converted(mid, npz, pose_shards, cache, arm, force, logfile):
    """convert.py for one match and one pose source (cached by input fingerprint)."""
    d = cache / arm / f"{mid:03d}"
    raw, key_file = d / "raw", d / ".key"
    key = fingerprint([npz] + (list(pose_shards) if arm == "ours_rtm" else []))
    if not force and key_file.exists() and key_file.read_text().strip() == key and raw.is_dir():
        return raw
    if d.exists():
        shutil.rmtree(d)
    raw.mkdir(parents=True)
    one = d / "extract_one"
    one.mkdir(parents=True)
    symlink(npz, one / npz.name)                  # convert.py globs by the NNN prefix
    cmd = [sys.executable, HERE / "convert.py", "--extract", str(one), "--out", str(raw),
           "--labels", str(RESEARCH / "shuttleset" / "labels" / "ShuttleSet" / "set")]
    if arm == "ours_rtm":
        pd = d / "pose_one"
        pd.mkdir(parents=True)
        for s in pose_shards:
            symlink(Path(s), pd / Path(s).name)
        cmd += ["--rtm", str(pd)]
    out = run(cmd, logfile)
    for ln in out.strip().splitlines():
        if ln.startswith(f"{mid:03d}") or ln.startswith("skipped") or ln.startswith("summary"):
            log(f"    convert[{arm}] {ln}", logfile)
    key_file.write_text(key)
    return raw


def build_filtered(mid, arm, source_raw, keep, cache, official, force, logfile):
    """Raw dir holding exactly `keep` clips, from ours (source_raw) or the official data."""
    d = cache / arm / f"{mid:03d}"
    filt, key_file = d / "filtered", d / ".fkey"
    # Key on the clip names AND the source each one resolves to, so a rebuilt `raw` (same
    # names, new files) still invalidates the symlinks rather than leaving them dangling.
    key = hashlib.sha256("\n".join(keep).encode()).hexdigest()[:16]
    if not force and key_file.exists() and key_file.read_text().strip() == key and filt.is_dir():
        # The name list can be unchanged while the sources behind it moved; verify that every
        # symlink still resolves to a live file before trusting the cache.
        broken = [f for f in filt.rglob("*.npy") if not f.resolve().exists()]
        if not broken:
            return filt
        log(f"    filtered[{arm}] {mid:03d}: {len(broken)} stale links — rebuilding", logfile)
    if filt.exists():
        shutil.rmtree(filt)
    split = split_of(mid)

    if arm == "offi":
        src = {}
        for b in keep:
            ob = official.get(b)
            if ob is None:
                raise RuntimeError(f"match {mid}: official data has no clip {b}, but the same "
                                   f"clip name is required in every arm")
            src[b] = ob
    else:
        have = {os.path.basename(b): b for b in clip_bases(source_raw)}
        missing = [b for b in keep if b not in have]
        if missing:
            raise RuntimeError(f"match {mid} / {arm}: {len(missing)} clips missing from the "
                               f"converted data, e.g. {missing[:3]}")
        src = {b: have[b] for b in keep}

    for name, base in src.items():
        cls = os.path.basename(os.path.dirname(base))
        for suffix in ("_joints.npy", "_pos.npy", "_shuttle.npy"):
            s = Path(base + suffix)
            if not s.exists():
                raise RuntimeError(f"{arm} match {mid}: source missing {s}")
            symlink(s, filt / split / cls / (name + suffix))
    key_file.write_text(key)
    return filt


def collate(mid, arm, filt, cache, force, logfile):
    coll = cache / arm / f"{mid:03d}" / "collated"
    if force and coll.exists():
        shutil.rmtree(coll)
    sets = sorted(p.name for p in filt.glob("*") if p.is_dir())
    if not sets:
        return None
    # The cached arrays must still describe the clips currently in `filt`.  A stale collated
    # dir (from a run whose clip set has since changed) would otherwise be reused silently and
    # hand eval_arm a row count that disagrees with the files on disk.  Compare row counts to
    # the live tree and rebuild on any disagreement.
    want = {s: len(clip_bases(filt / s)) for s in sets}
    if not force and (coll / sets[0] / "labels.npy").exists():
        have = {}
        ok = True
        for s in sets:
            f = coll / s / "labels.npy"
            if not f.exists():
                ok = False
                break
            have[s] = int(np.load(f, mmap_mode="r").shape[0])
        if ok and have == want:
            return coll
        log(f"    collate[{arm}] {mid:03d}: cached arrays {have} disagree with the filtered "
            f"clips {want} — re-collating", logfile)
    if coll.exists():
        shutil.rmtree(coll)
    run([sys.executable, HERE / "collate.py", str(filt), str(coll), str(SEQ_LEN)] + sets, logfile)
    return coll


# --------------------------------------------------------------------------- eval

def eval_arm(coll_dir: Path, raw_dir: Path, device="mps", flat_set=None):
    """Forward the 10 released checkpoints over one collated dir.

    collate.py flattens the class directories into per-set arrays, so the clip names are
    read back from the raw dir that was collated; the arrays' leading dimension is
    asserted to equal the number of raw clips in each set.  Returns, per checkpoint, the
    pooled (pred, gt, top2hit).  Metric definitions match eval_bst.py.

    `flat_set` handles the released layout, where the arrays sit directly in
    `<coll_dir>/<set>/` rather than in a per-set subdirectory of a common root.
    """
    import torch
    from torch.utils.data import DataLoader
    from eval_bst import arch_from_name, run as fwd
    from preparing_data.shuttleset_dataset import Dataset_npy_collated, get_bone_pairs

    if flat_set:
        # Released layout: arrays live in `<coll_dir>/<flat_set>/`.
        sets = [flat_set]
        root = coll_dir if (coll_dir / flat_set / "labels.npy").exists() else coll_dir.parent
        want_raw = raw_dir
    else:
        sets = sorted(p.name for p in coll_dir.glob("*") if p.is_dir())
        root, want_raw = coll_dir, raw_dir
    if not sets:
        return None
    names, per_set = [], {}
    for s in sets:
        nm = [os.path.basename(b) for b in clip_bases(want_raw / s if not flat_set else want_raw)]
        per_set[s] = nm
        names += nm

    in_dim = (17 + len(get_bone_pairs())) * 2
    per_weight = []
    for w in weights():
        net = arch_from_name(Path(w).name)(
            in_dim=in_dim, n_class=N_CLASSES, seq_len=SEQ_LEN, depth_tem=2, depth_inter=1)
        net.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        net.to(device)
        rec = {}
        for s in sets:
            ds = Dataset_npy_collated(root, s, "JnB_bone")
            logits, gt = fwd(net, DataLoader(ds, batch_size=128), device)
            if len(gt) != len(per_set[s]):
                raise RuntimeError(f"{coll_dir}/{s}: {len(gt)} clips in the collated arrays but "
                                   f"{len(per_set[s])} raw clip files on disk")
            top2 = torch.topk(logits, 2, dim=1).indices
            rec[s] = (logits.argmax(1).numpy().astype(np.int16),
                      gt.numpy().astype(np.int16),
                      (top2 == gt[:, None]).any(1).numpy())
        net.cpu()
        per_weight.append(rec)

    pooled = [(np.concatenate([rec[s][0] for s in sets]),
               np.concatenate([rec[s][1] for s in sets]),
               np.concatenate([rec[s][2] for s in sets])) for rec in per_weight]
    return {"sets": sets, "names": names, "sets_sizes": {s: len(per_set[s]) for s in sets},
            "pooled": pooled}


# --------------------------------------------------------------------------- one match

def official_reference(device="mps", cache=None):
    """The published number, re-measured here on its own terms.

    The released collated test split is the full 3,499-stroke official test set (matches
    39, 40, 42, 43, 44).  Scoring it with the same 10 checkpoints the arms use is what the
    published 0.8322 refers to, so it is the fair anchor for a subset that only holds the
    clips our pipeline also produced.
    """
    coll = OFFICIAL_COLL / "test"
    if not (coll / "labels.npy").exists():
        return None
    res = eval_arm(coll, OFFICIAL_RAW / "test", device=device, flat_set="test")
    if res is None:
        return None
    mets = [metrics_from_preds(P, G, T) for P, G, T in res["pooled"]]
    return row_from_weights(mets)


def measure_match(mid, npz, pose_shards, cache, a, official, logfile):
    raw = {"ours_vis": build_converted(mid, npz, [], cache, "ours_vis", a.force, logfile)}
    if pose_shards and not a.no_pose:
        raw["ours_rtm"] = build_converted(mid, npz, pose_shards, cache, "ours_rtm",
                                          a.force, logfile)

    name_sets = {arm: {os.path.basename(b) for b in clip_bases(r)} for arm, r in raw.items()}
    common = set.intersection(*name_sets.values())
    # The arms we can score for this match are clipped to the same names; the official arm
    # joins only if the released data covers this match at all.  A match with no official
    # counterpart (e.g. 9, 10) still yields a valid ours_rtm-vs-ours_vis comparison, so it
    # is kept rather than dropped — it just cannot carry an official column.  Official clip
    # names are unpadded (`1_1_5_13`), so the match id is compared as an integer.
    has_official = mid in OFFICIAL_MATCH_IDS
    if not a.no_official and has_official:
        common &= set(official)
    if not common:
        log(f"  match {mid}: no clip name common to {sorted(name_sets)} — nothing to do", logfile)
        return None
    common = sorted(common)

    dropped = [n for n in common
               if any(is_degenerate(b)
                      for arm, r in raw.items()
                      for b in clip_bases(r) if os.path.basename(b) == n)]
    keep = [n for n in common if n not in set(dropped)]
    log(f"  match {mid}: {len(common)} common clips "
        f"({len(dropped)} degenerate dropped), arms {sorted(raw)}", logfile)
    if not keep:
        return None

    arms = {}
    for arm, r in raw.items():
        filt = build_filtered(mid, arm, r, keep, cache, official, a.force, logfile)
        coll = collate(mid, arm, filt, cache, a.force, logfile)
        if coll is None:
            continue
        res = eval_arm(coll, filt, device=a.device)
        if res is None:
            continue
        if sorted(res["names"]) != keep:
            raise RuntimeError(f"{arm} match {mid}: evaluated {len(res['names'])} clips, "
                               f"expected {len(keep)} — clip set drift")
        arms[arm] = res
    filt = None
    if not a.no_official and has_official:
        filt = build_filtered(mid, "offi", None, keep, cache, official, a.force, logfile)
        coll = collate(mid, "offi", filt, cache, a.force, logfile)
        if coll is not None:
            res = eval_arm(coll, filt, device=a.device)
            if res is not None:
                if sorted(res["names"]) != keep:
                    raise RuntimeError(f"official arm match {mid} does not hold exactly our clip "
                                       f"names ({len(res['names'])} vs {len(keep)})")
                arms["offi"] = res

    rows = {}
    for arm, res in arms.items():
        mets = [metrics_from_preds(P, G, T) for P, G, T in res["pooled"]]
        rows[arm] = {"per_weight_metrics": mets, "n": mets[0]["n"], "match": mid,
                     "split": split_of(mid), "n_dropped_degenerate": len(dropped),
                     "clip_names": list(keep), "pooled": res["pooled"]}
    return {"rows": rows, "n_common": len(common), "n_dropped": len(dropped), "kept": keep}


# --------------------------------------------------------------------------- aggregate

def aggregate(rows_by_match, matches, arm):
    """Pool the per-checkpoint arrays across matches, then average the 10 metric rows."""
    pool = [rows_by_match[m][arm] for m in matches
            if arm in rows_by_match.get(m, {})]
    if not pool:
        return None
    n_w = len(pool[0]["pooled"])
    per_w = []
    for w in range(n_w):
        P = np.concatenate([r["pooled"][w][0] for r in pool])
        G = np.concatenate([r["pooled"][w][1] for r in pool])
        T = np.concatenate([r["pooled"][w][2] for r in pool])
        per_w.append(metrics_from_preds(P, G, T))
    row = row_from_weights(per_w)
    row["n_matches"] = len(pool)
    row["matches"] = sorted({r["match"] for r in pool})
    for key in ("f1_each", "precision_each", "recall_each", "support_each"):
        row[key] = np.mean([p[key] for p in per_w], axis=0).tolist()
    return row


# --------------------------------------------------------------------------- report

def write_report(arms, matches, manifest, checks, args, logfile, official_ref=None):
    held = [m for m in sorted(TEST_IDS) if m in arms]
    val = [m for m in sorted(VAL_IDS) if m in arms]
    train_side = [m for m in sorted(arms) if split_of(m) == "train"]
    agg_held = {arm: aggregate(arms, held, arm) for arm in ARMS}
    agg_val = {arm: aggregate(arms, val, arm) for arm in ARMS}
    agg_train = {arm: aggregate(arms, train_side, arm) for arm in ARMS}

    def fmt(row):
        if not row:
            return "— | — | — | — | — | —"
        return (f"{row['n_matches']} | {row['n']} | {row['acc']:.4f} | {row['macro_f1']:.4f} | "
                f"{row['top2']:.4f} | {row['acc_std']:.4f}")

    md = ["# Stroke classification accuracy — BST on our pipeline's features", "",
          f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `measure.py`.  "
          f"Released ShuttleSet weights (`bst_CG_AP_JnB_bone…seq_100_merged*`, "
          f"{len(weights())} checkpoints averaged), seq_len {SEQ_LEN}, {N_CLASSES} classes.  "
          f"**Nothing was trained for this report.**", "",
          "Claim under test: *features from our pipeline reach the same stroke-classification "
          "accuracy as the published ShuttleSet features when BST's expected pose (RTMPose-m) is "
          "used.*", ""]

    md += ["## Inputs available at this run", "",
           "| match | split | extract | pose | clips kept | degenerate dropped | kept in arm(s) | note |",
           "|---|---|---|---|---|---|---|---|"]
    for m in sorted(manifest):
        e = manifest[m]
        md.append(f"| {m} | {e['split']} | yes | {'RTMPose-m' if e['pose'] else '—'} | "
                  f"{e['n_common']} | {e['n_dropped']} | {', '.join(e['arms']) or '—'} | "
                  f"{e.get('note') or ''} |")
    have_held = [m for m in held if manifest.get(m, {}).get("n_common")]
    md += [""]
    if len(have_held) < len(TEST_IDS):
        md += [f"> **HEADLINE NUMBER PENDING.** {len(have_held)} of the {len(TEST_IDS)} held-out "
               f"matches (39, 40, 42, 43, 44) have data"
               + (f": {', '.join(map(str, have_held))}" if have_held else "")
               + ". The held-out aggregates below are therefore not the claim's number yet; use "
                 "the training-side regression table for now."]
    else:
        md += ["All five held-out matches are present; the held-out aggregate is the headline."]
    md += [""]

    md += ["## Arms", "",
           "| arm | features | pose |", "|---|---|---|",
           "| `ours_rtm` | our pipeline | RTMPose-m (BST's expected pose) |",
           "| `ours_vis` | our pipeline | Apple Vision |",
           "| `offi` | official ShuttleSet, as published | as published |", "",
           "Within each match every arm is evaluated on exactly the same clip names, so all rows "
           "are clip-for-clip.", ""]

    md += ["## Published reference", ""]
    if official_ref:
        md += [f"The published number is the official test split as released: 3,499 strokes over "
               f"matches 39, 40, 42, 43, 44. Re-measured here with the same 10 released "
               f"checkpoints, it gives **acc {official_ref['acc']:.4f} / macro-F1 "
               f"{official_ref['macro_f1']:.4f} / top-2 {official_ref['top2']:.4f}** "
               f"({official_ref['n']} clips). `bst/REPORT.md` records 0.8322 for the same split, "
               f"so this is a reproduction of the published figure, not a new one.", ""]
    else:
        md += ["The released collated test split is not present, so the published figure could "
               "not be re-measured this run.", ""]
    md += ["A held-out aggregate over fewer clips than the full split is not directly comparable "
           "to it — the `offi` arm's held-out row above is the like-for-like comparison, because "
           "it holds exactly the clips `ours_rtm` scored.", ""]

    md += ["## Held-out aggregate (matches 39, 40, 42, 43, 44 only)", "",
           "| arm | matches | clips | acc | macro-F1 | top-2 | acc std over the 10 checkpoints |",
           "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        md.append(f"| `{arm}` | {fmt(agg_held[arm])} |")
    md += [""]
    if agg_held["ours_rtm"] and agg_held["offi"]:
        d = agg_held["ours_rtm"]["acc"] - agg_held["offi"]["acc"]
        spread = max(agg_held["ours_rtm"]["acc_std"], agg_held["offi"]["acc_std"])
        verdict = ("at least as accurate" if d >= 0 else "less accurate")
        md += [f"Our features under RTMPose-m vs the official features, same clips: "
               f"**{d:+.4f} accuracy** ({verdict})"
               + (f" — within the {spread:.4f} spread across the 10 checkpoints." if abs(d) <= spread
                  else f" — outside the {spread:.4f} spread across the 10 checkpoints."), ""]

    for title, aggs, note in (
            ("val = 41 (reported separately, never pooled with the held-out set)",
             agg_val, "Single match; treat as indicative."),
            ("training-side matches (regression figure, NOT the claim)",
             agg_train, "These matches are in BST's training split, so this is a reproduction "
                        "check, not a held-out result.")):
        md += [f"## Aggregate — {title}", "",
               "| arm | matches | clips | acc | macro-F1 | top-2 | acc std |",
               "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            md.append(f"| `{arm}` | {fmt(aggs[arm])} |")
        md += ["", note, ""]

    md += ["## Per-match", "",
           "| match | split | arm | clips | acc | macro-F1 | top-2 |", "|---|---|---|---|---|---|---|"]
    for m in sorted(arms):
        for arm in ARMS:
            r = arms[m].get(arm)
            if not r:
                continue
            row = row_from_weights(r["per_weight_metrics"])
            md.append(f"| {m} | {split_of(m)} | `{arm}` | {r['n']} | {row['acc']:.4f} | "
                      f"{row['macro_f1']:.4f} | {row['top2']:.4f} |")
    md += [""]

    # Clip-for-clip head-to-head against the official features, on the identical clip names
    # every arm in that match holds.  A positive delta means our features scored higher.
    head = [m for m in sorted(arms) if "offi" in arms[m]]
    if head:
        md += ["## Clip-for-clip vs the official features", "",
               "Same clip names, same checkpoints, same metric code — the only difference is "
               "whose features the classifier saw.", "",
               "| match | split | arm | clips | acc | Δ acc vs `offi` | macro-F1 | Δ F1 vs `offi` |",
               "|---|---|---|---|---|---|---|---|"]
        for m in head:
            base = row_from_weights(arms[m]["offi"]["per_weight_metrics"])
            for arm in ("ours_rtm", "ours_vis"):
                r = arms[m].get(arm)
                if not r:
                    continue
                row = row_from_weights(r["per_weight_metrics"])
                md.append(f"| {m} | {split_of(m)} | `{arm}` | {r['n']} | {row['acc']:.4f} | "
                          f"{row['acc'] - base['acc']:+.4f} | {row['macro_f1']:.4f} | "
                          f"{row['macro_f1'] - base['macro_f1']:+.4f} |")
            md.append(f"| {m} | {split_of(m)} | `offi` (baseline) | {arms[m]['offi']['n']} | "
                      f"{base['acc']:.4f} | — | {base['macro_f1']:.4f} | — |")
        if not any(split_of(m) == "test" for m in head):
            md += ["", "Every match above is training-side, so these deltas are a reproduction "
                       "check on matches BST's weights were fitted on — **not** the claim. The "
                       "held-out head-to-head appears once matches 39, 40, 42, 43, 44 land."]
        md += [""]

    if agg_held["ours_rtm"]:
        a = agg_held["ours_rtm"]
        md += ["## Per-class, held-out, our features under RTMPose-m (worst F1 first)", "",
               "| class | support | F1 | precision | recall |", "|---|---|---|---|---|"]
        for i in sorted(range(N_CLASSES), key=lambda i: (a["f1_each"][i], -a["support_each"][i])):
            md.append(f"| {english(CLASSES[i])} | {int(a['support_each'][i])} | {a['f1_each'][i]:.3f} "
                      f"| {a['precision_each'][i]:.3f} | {a['recall_each'][i]:.3f} |")
        md += ["", "The same table for the official features, for contrast:", "",
               "| class | support | F1 | precision | recall |", "|---|---|---|---|---|"]
        b = agg_held["offi"] or a
        for i in sorted(range(N_CLASSES), key=lambda i: (b["f1_each"][i], -b["support_each"][i])):
            md.append(f"| {english(CLASSES[i])} | {int(b['support_each'][i])} | {b['f1_each'][i]:.3f} "
                      f"| {b['precision_each'][i]:.3f} | {b['recall_each'][i]:.3f} |")
        md += [""]

    md += ["## Sanity checks", ""] + [f"- {c}" for c in checks]
    md += ["", "Any violation of these aborts the affected arm or the run; a number is never "
           "reported from a check that did not run.", ""]

    md += ["## How to re-run", "", "```sh",
           f"cd {RESEARCH} && bst/venv/bin/python bst/measure.py {' '.join(args)}".rstrip(),
           "```", "",
           "`--wait` polls the extract and pose directories and re-reports whenever new data "
           "lands.  Converted, filtered and collated dirs are cached per (match, arm) and keyed "
           "on the inputs' size+mtime, so a re-run does only what is new or changed; `--force` "
           "ignores the cache.  Held-out aggregates stay pending until matches 39, 40, 42, 43, 44 "
           "arrive.", f"Log: `{HERE / 'measure.log'}`.", ""]

    (HERE / "ACCURACY.md").write_text("\n".join(md) + "\n")

    def clean(row):
        if not row:
            return None
        out = {k: v for k, v in row.items() if k != "matches"}
        out["classes"] = CLASSES
        out["classes_en"] = [english(c) for c in CLASSES]
        return out

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim": "our pipeline's features reach the published ShuttleSet features' "
                 "stroke-classification accuracy under BST's expected pose (RTMPose-m)",
        "seq_len": SEQ_LEN, "n_weights": len(weights()), "n_classes": N_CLASSES,
        "classes": CLASSES, "classes_en": [english(c) for c in CLASSES],
        "held_out_matches": list(TEST_IDS), "val_matches": list(VAL_IDS),
        "headline_ready": len(have_held) == len(TEST_IDS),
        "held_out_matches_measured": have_held,
        "published_reference": official_ref,
        "inputs": manifest,
        "aggregates": {"held_out": {arm: clean(agg_held[arm]) for arm in ARMS},
                       "val_41": {arm: clean(agg_val[arm]) for arm in ARMS},
                       "train_side": {arm: clean(agg_train[arm]) for arm in ARMS}},
        "per_match": {str(m): {arm: {**row_from_weights(r["per_weight_metrics"]),
                                     "n": r["n"], "split": r["split"],
                                     "n_dropped_degenerate": r["n_dropped_degenerate"]}
                               for arm, r in arms[m].items() if arm in ARMS}
                      for m in sorted(arms)},
        "checks": checks,
    }
    json.dump(payload, open(HERE / "accuracy.json", "w"), ensure_ascii=False, indent=1)
    log(f"wrote {HERE / 'ACCURACY.md'} and {HERE / 'accuracy.json'}", logfile)
    return payload


# --------------------------------------------------------------------------- checks

def sanity_checks(arms, manifest, per_match):
    """Fail loudly on anything that would make a reported number meaningless."""
    problems, notes = [], []

    # 1. every clip evaluated belongs to the match it is filed under
    for m, info in sorted(per_match.items()):
        cross = sorted({n.split("_")[0] for n in info["kept"]} - {str(m)})
        if cross:
            problems.append(f"match {m} holds clips from match(es) {cross} — SPLIT LEAK")
        notes.append(f"match {m}: all {len(info['kept'])} evaluated clips belong to match {m}")

    # 2. a held-out match's clips must never sit inside a training-side arm's data
    test_names = {n for m, i in per_match.items() if split_of(m) == "test" for n in i["kept"]}
    train_names = {n for m, i in per_match.items() if split_of(m) == "train" for n in i["kept"]}
    both = test_names & train_names
    if both:
        problems.append(f"{len(both)} clip names appear in both a held-out and a training-side "
                        f"match: {sorted(both)[:5]} — SPLIT LEAK")
    else:
        notes.append(f"no held-out clip appears in any training-side arm "
                     f"({len(test_names)} held-out vs {len(train_names)} training-side clip names)")

    # 3. the official baseline holds exactly our clip names, per match
    for m in sorted(arms):
        if "offi" in arms[m] and "ours_vis" in arms[m]:
            a, b = sorted(arms[m]["offi"]["clip_names"]), sorted(arms[m]["ours_vis"]["clip_names"])
            if a != b:
                problems.append(f"match {m}: official arm clip names differ from ours "
                                f"({len(a)} vs {len(b)}) — BASELINE NOT CLIP-FOR-CLIP")
    notes.append("official baseline uses exactly the clip names present in our set "
                 "(asserted per match; the arm aborts if not)")

    # 4. clip counts are asserted against the files on disk inside eval_arm
    notes.append("clip counts per split match the files on disk (asserted on every collated dir, "
                 "including the arrays-vs-disk count inside eval_arm)")

    # 5. degenerate clips
    dropped = sum(i["n_dropped"] for i in per_match.values())
    notes.append(f"degenerate all-zero-pose clips are counted and dropped from every arm in that "
                 f"match ({dropped} dropped this run)")

    # 6. cache fingerprints present
    notes.append("every cached (match, arm) directory is keyed on its inputs' size+mtime; a "
                 "changed extract or pose file re-converts only that match")

    if problems:
        raise RuntimeError("SANITY CHECK FAILED:\n  " + "\n  ".join(problems))
    return notes


# --------------------------------------------------------------------------- driver

def argv_strings(a):
    parts = []
    if Path(a.extract) != RESEARCH / "extract" / "out":
        parts += ["--extract", a.extract]
    if Path(a.pose) != RESEARCH / "rtmpose" / "pose":
        parts += ["--pose", a.pose]
    if Path(a.cache) != HERE / "acc_cache":
        parts += ["--cache", a.cache]
    if a.match:
        parts += ["--match", *map(str, a.match)]
    if a.device != "mps":
        parts += ["--device", a.device]
    if a.no_official:
        parts += ["--no-official"]
    if a.no_pose:
        parts += ["--no-pose"]
    return parts


def measure_all(a, logfile, cache=None, out_dir=HERE):
    cache = Path(cache or a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    official = {} if a.no_official else official_index(cache)
    OFFICIAL_MATCH_IDS.clear()
    OFFICIAL_MATCH_IDS.update(int(k.split("_")[0]) for k in official)
    found = {}
    for f in sorted(Path(a.extract).glob("*.npz")):
        mid = int(f.name[:3])
        if a.match and mid not in set(a.match):
            continue
        # run_pose.py writes ONE merged file per match (`NNN.npz`, keys kp/box/frame);
        # older runs produced per-shard `NNN_sh*.npz`.  Take whichever exists, preferring
        # the merged file so a fully-finished match is never mixed with stale shards.
        pose_dir = Path(a.pose)
        merged = pose_dir / f"{mid:03d}.npz"
        if merged.exists():
            poses = [str(merged)]
        else:
            poses = sorted(glob.glob(str(pose_dir / f"{mid:03d}_sh*.npz")))
        found[mid] = {"npz": f, "pose": poses}
    log(f"{len(found)} match(es) with extract output: {sorted(found)}", logfile)
    log(f"pose available for: {sorted(m for m, v in found.items() if v['pose'])}", logfile)

    arms, manifest, per_match = {}, {}, {}
    official_matches = sorted({int(k.split("_")[0]) for k in official}) if official else []
    for mid, v in sorted(found.items()):
        entry = {"split": split_of(mid), "pose": bool(v["pose"]), "n_common": 0, "n_dropped": 0,
                 "arms": [], "note": ""}
        # A match the released data does not cover can never carry an official column; say so
        # on its row rather than leaving the absence to look like a failure.
        if official and mid not in OFFICIAL_MATCH_IDS:
            entry["note"] = "no official ShuttleSet counterpart — no official arm"
        try:
            res = measure_match(mid, v["npz"], v["pose"], cache, a, official, logfile)
        except Exception as exc:                                     # noqa: BLE001
            log(f"  match {mid} FAILED: {type(exc).__name__}: {exc}", logfile)
            entry["arms"] = [f"FAILED: {type(exc).__name__}"]
            entry["note"] = "FAILED"
            manifest[mid] = entry
            continue
        manifest[mid] = entry
        if not res:
            entry["note"] = entry["note"] or "no common clips"
            continue
        entry.update(n_common=res["n_common"], n_dropped=res["n_dropped"],
                     arms=sorted(res["rows"]))
        arms[mid] = res["rows"]
        per_match[mid] = {"kept": res["kept"], "n_dropped": res["n_dropped"]}

    checks = sanity_checks(arms, manifest, per_match)
    ref = None
    if not a.no_official:
        try:
            ref = official_reference(device=a.device)
            if ref:
                log(f"published reference (official test split, {ref['n']} clips): "
                    f"acc={ref['acc']:.4f} macro_f1={ref['macro_f1']:.4f} top2={ref['top2']:.4f}",
                    logfile)
        except Exception as exc:                                     # noqa: BLE001
            log(f"published reference not measured: {type(exc).__name__}: {exc}", logfile)
    return write_report(arms, list(arms), manifest, checks, argv_strings(a), logfile, ref)


def snapshot(a):
    h = hashlib.sha256()
    for d, pat in ((a.extract, "*.npz"), (a.pose, "*.npz")):
        for f in sorted(glob.glob(os.path.join(d, pat))):
            h.update(f"{f}:{os.stat(f).st_size}:{int(os.stat(f).st_mtime)}".encode())
    return h.hexdigest()


def self_test(a, logfile):
    """Reproduce the 318-clip match-1 subset: RTMPose ~0.873, Vision ~0.53."""
    log("self-test: match 1, RTMPose subset (/tmp/rtm/ab2) vs Vision, identical clip names",
        logfile)
    a.match = [1]
    a.pose = "/tmp/rtm/ab2"
    payload = measure_all(a, logfile, cache=str(HERE / "acc_cache_selftest"))
    for arm in ARMS:
        row = (payload["per_match"].get("1", {}) or {}).get(arm)
        log(f"self-test {arm}: " + (f"n={row['n']} acc={row['acc']:.4f} "
                                    f"macro_f1={row['macro_f1']:.4f} top2={row['top2']:.4f}"
                                    if row else "no result"), logfile)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", default=str(RESEARCH / "extract" / "out"))
    ap.add_argument("--pose", default=str(RESEARCH / "rtmpose" / "pose"))
    ap.add_argument("--cache", default=str(HERE / "acc_cache"))
    ap.add_argument("--match", type=int, nargs="*", default=[])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--wait", action="store_true", help="poll and re-report when new data lands")
    ap.add_argument("--poll", type=float, default=180.0, help="seconds between polls")
    ap.add_argument("--no-pose", dest="no_pose", action="store_true",
                    help="skip the RTMPose arm (Vision + official only)")
    ap.add_argument("--no-official", dest="no_official", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--self-test-318", action="store_true",
                    help="reproduce the 318-clip match-1 subset numbers and exit")
    ap.add_argument("--log", default=str(HERE / "measure.log"))
    a = ap.parse_args()
    logfile = Path(a.log)
    log(f"measure.py start  extract={a.extract}  pose={a.pose}  cache={a.cache}  "
        f"device={a.device}", logfile)

    if a.self_test_318:
        self_test(a, logfile)
        return

    while True:
        try:
            measure_all(a, logfile)
        except Exception as exc:                                     # noqa: BLE001
            import traceback
            log(f"ERROR: {type(exc).__name__}: {exc}", logfile)
            log(traceback.format_exc()[-1500:], logfile)
        if not a.wait:
            return
        last = snapshot(a)
        log(f"wait: report written; polling every {a.poll:.0f}s for new extract/pose data", logfile)
        while snapshot(a) == last:
            time.sleep(a.poll)
        log("wait: new data detected, re-measuring", logfile)


if __name__ == "__main__":
    main()
