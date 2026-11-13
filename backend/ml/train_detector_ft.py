#!/usr/bin/env python3
"""Fine-tune entry point for the shuttle heatmap detectors on the TrackNetV2 Train split.

Thin wrapper around ml/train_detector.py (its recipe, loss, targets, augmentation, decode and
checkpoint format are used unchanged). This file adds only what task 6 needs around it:

  prepare  unpack the official TrackNetV2.zip (HackMD https://hackmd.io/@TUIK/rJkRW54cU ->
           NYCU OneDrive, see ml/runs/fetch_tracknetv2.sh) into the trainer's layout
           <root>/Train/match{1..26}/{csv,video}  (Professional 1-23, Amateur -> 24-26, as the
           TrackNetV3 README), verify labels load with benchmark.load_label_csv, and prove the
           Train clips do not overlap the held-out Test clips (video sha256, csv sha256, first-
           frame thumbnail distance). Writes a manifest JSON.
  split    seeded BY-MATCH validation split (professional matches only, like Test).
  train    train_detector.main(...) with the split as --data-root / --val-root, plus early
           stopping (patience on val F1) and a wall-clock budget; never touches Test. Copies
           best.pth.tar to a NEW ml/weights/ckpts/*_ft.* file (refuses to overwrite).
  score    per-clip TrackNet 'tn4' counts from benchmark.py caches (the same parser, drop
           ranges and matching rule as `benchmark.py --stage real`), before vs after: pooled F1,
           pre-registered weak subsets, paired clip bootstrap CI and sign test.

Usage (from backend/):
  .venv/bin/python ml/train_detector_ft.py prepare --zip ml/data/raw/shuttlecock_trajectory/_download/TrackNetV2.zip
  .venv/bin/python ml/train_detector_ft.py split
  .venv/bin/python ml/train_detector_ft.py train --arch wasb --budget-min 100
  .venv/bin/python ml/train_detector_ft.py score --kind wasb --before /tmp/shuttl_ft_bench/base \
        --after /tmp/shuttl_ft_bench/wasb_ft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
ML_DIR = BACKEND / "ml"
for _p in (str(BACKEND), str(ML_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RAW = ML_DIR / "data" / "raw" / "shuttlecock_trajectory"
TRAIN_ROOT = RAW / "TrackNetV2" / "Train"
TEST_ROOT = RAW / "TrackNetV2" / "Test"
RUN_DIR = ML_DIR / "runs" / "detector_ft"
SPLIT_JSON = RUN_DIR / "split.json"
MANIFEST_JSON = RUN_DIR / "data_manifest.json"
CKPT_DIR = ML_DIR / "weights" / "ckpts"
FT_NAMES = {"wasb": "wasb_badminton_ft.pth.tar", "tracknetv3": "TrackNet_ft.pt"}
N_PRO = 23                      # HackMD: professional matches 1-23, amateur 1-3 (-> 24-26)
# HackMD "Rally of each match (train professional matches)" table: rallies per match 1..23
HACKMD_PRO_RALLIES = [12, 8, 7, 8, 6, 5, 5, 8, 9, 5, 5, 5, 6, 5, 5, 6, 6, 6, 5, 6, 7, 5, 6]
VIDEO_DIRS = ("video", "rally_video")
CSV_DIRS = ("csv", "ball_trajectory")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bench():
    """scripts/benchmark.py (its label parser, drop ranges and clip list are THE Test rule)."""
    from scripts import benchmark
    return benchmark


# ══════════════════════════════════════════════════════════════════════════════
# prepare: unpack + verify + overlap check
# ══════════════════════════════════════════════════════════════════════════════

def archive_group(rel: Path) -> str:
    """'pro' | 'amateur' | 'test' | 'unknown' for a path inside the official archive."""
    parts = [p.lower() for p in rel.parts]
    if any(p == "test" or p.startswith("test_") for p in parts):
        return "test"
    if any("amateur" in p for p in parts):
        return "amateur"
    if any("profession" in p or p == "train" for p in parts):
        return "pro"
    return "unknown"


def match_number(name: str) -> int | None:
    m = re.fullmatch(r"match(\d+)", name)
    return int(m.group(1)) if m else None


def train_match_name(group: str, n: int) -> str:
    """TrackNetV3 README: Professional match1-23 keep their names, Amateur match1-3 -> 24-26."""
    if group == "pro":
        return f"match{n}"
    if group == "amateur":
        return f"match{N_PRO + n}"
    raise ValueError(group)


def find_clip_media(match_dir: Path, clip: str) -> Path | None:
    for d in VIDEO_DIRS:
        for ext in (".mp4", ".MP4", ".avi", ".mov"):
            p = match_dir / d / f"{clip}{ext}"
            if p.is_file():
                return p
    p = match_dir / "frame" / clip
    return p if p.is_dir() else None


def scan_archive_tree(stage: Path) -> list[dict]:
    """Every labelled clip in an extracted archive: {group, match, clip, csv, media}."""
    out = []
    for csv in sorted(stage.rglob("*_ball.csv")):
        if csv.parent.name not in CSV_DIRS or "__MACOSX" in csv.parts:
            continue
        match_dir = csv.parent.parent
        n = match_number(match_dir.name)
        if n is None:
            continue
        clip = csv.name[: -len("_ball.csv")]
        out.append({"group": archive_group(csv.relative_to(stage)), "match": match_dir.name,
                    "n": n, "clip": clip, "csv": csv, "media": find_clip_media(match_dir, clip)})
    return out


def first_frame_thumb(media: Path, size: tuple[int, int] = (64, 36)) -> np.ndarray | None:
    import cv2
    if media.is_dir():
        files = sorted(media.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
        img = cv2.imread(str(files[0])) if files else None
    else:
        cap = cv2.VideoCapture(str(media))
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def media_frames(media: Path) -> tuple[int, int, int]:
    """(n_frames, width, height) by decoding (CAP_PROP_FRAME_COUNT is unreliable)."""
    import cv2
    if media.is_dir():
        files = list(media.glob("*.png"))
        img = cv2.imread(str(files[0])) if files else None
        h, w = (img.shape[:2] if img is not None else (0, 0))
        return len(files), w, h
    cap = cv2.VideoCapture(str(media))
    n, w, h = 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.grab():
        n += 1
    cap.release()
    return n, w, h


def label_summary(csv: Path, n_frames: int, width: int, height: int) -> dict:
    """Load with benchmark.load_label_csv (the Test parser) and train_detector.load_label_csv."""
    import train_detector as TD
    xy, vis = bench().load_label_csv(str(csv), n_frames)
    lab = TD.load_label_csv(csv, n_frames)
    rows = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
    agree = bool(np.array_equal(vis, lab[:, 0] > 0)
                 and np.allclose(xy[vis], lab[vis, 1:3]))
    inside = (xy[vis] >= 0).all(1) & (xy[vis][:, 0] < width) & (xy[vis][:, 1] < height)
    return {"rows": int(len(rows)), "visible": int(vis.sum()),
            "max_label_frame": int(rows[:, 0].max()) if len(rows) else -1,
            "parsers_agree": agree, "xy_in_image": bool(np.all(inside)),
            "_xy": xy, "_vis": vis}


def label_content_match(a_xy: np.ndarray, a_vis: np.ndarray, b_xy: np.ndarray, b_vis: np.ndarray,
                        max_shift: int = 3) -> tuple[float, int]:
    """Smallest mean |dxy| (px) between two label tracks over frame shifts in +-max_shift,
    on frames visible in both (needs >= 30 of them) -> (distance, overlapping frames)."""
    best = (float("inf"), 0)
    for s in range(-max_shift, max_shift + 1):
        lo, hi = max(0, s), min(len(a_xy), len(b_xy) + s)
        if hi - lo < 30:
            continue
        va, vb = a_vis[lo:hi], b_vis[lo - s:hi - s]
        both = va & vb
        if both.sum() < 30:
            continue
        d = float(np.abs(a_xy[lo:hi][both] - b_xy[lo - s:hi - s][both]).mean())
        best = min(best, (d, int(both.sum())))
    return best


def overlap_report(train: list[dict], test: list[dict]) -> dict:
    """Exact (sha256 video/csv) and near (first-frame thumbnail) duplicates Train vs Test."""
    test_v = {t["video_sha256"]: t["name"] for t in test}
    test_c = {t["csv_sha256"]: t["name"] for t in test}
    exact_v = [(r["name"], test_v[r["video_sha256"]]) for r in train if r["video_sha256"] in test_v]
    exact_c = [(r["name"], test_c[r["csv_sha256"]]) for r in train if r["csv_sha256"] in test_c]
    near = []
    tt = [(t["name"], t["_thumb"]) for t in test if t.get("_thumb") is not None]
    for r in train:
        if r.get("_thumb") is None or not tt:
            continue
        d = [(float(np.abs(r["_thumb"] - th).mean()), n) for n, th in tt]
        best = min(d)
        near.append((best[0], r["name"], best[1]))
    near.sort()
    lab = []                                    # same trajectory = same rally, whatever the video
    for r in train:
        for t in test:
            if "_xy" in r and "_xy" in t:
                d, k = label_content_match(r["_xy"], r["_vis"], t["_xy"], t["_vis"])
                lab.append((d, k, r["name"], t["name"]))
    lab.sort()
    return {"exact_video_matches": exact_v, "exact_csv_matches": exact_c,
            "label_track_min_mean_px": round(lab[0][0], 2) if lab and np.isfinite(lab[0][0]) else None,
            "label_track_closest_pairs": [(round(d, 2), k, a, b) for d, k, a, b in lab[:5]
                                          if np.isfinite(d)],
            "label_track_below_5px": [(round(d, 2), k, a, b) for d, k, a, b in lab if d < 5.0][:20],
            "thumb_l1_min": round(near[0][0], 3) if near else None,
            "thumb_closest_pairs": [(round(a, 3), b, c) for a, b, c in near[:5]],
            "thumb_l1_below_3": [(round(a, 3), b, c) for a, b, c in near if a < 3.0][:20]}
def _link_or_copy(src: Path, dst: Path) -> None:
    import os
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)                       # same volume: no extra disk, stage untouched
    except OSError:
        shutil.copy2(src, dst)


def extract_all(zpath: Path, stage: Path) -> None:
    """Extract the archive (and any archives nested inside it) into ``stage``, once."""
    done = stage / ".extracted.json"
    if done.is_file():
        return
    stage.mkdir(parents=True, exist_ok=True)
    todo, seen = [zpath], set()
    while todo:
        z = todo.pop()
        dest = stage if z == zpath else z.with_suffix("")
        log(f"[prepare] extracting {z.name} -> {dest}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        seen.add(z.resolve())
        todo += [p for p in stage.rglob("*.zip") if p.resolve() not in seen and "__MACOSX" not in p.parts]
    done.write_text(json.dumps({"archive": str(zpath), "size": zpath.stat().st_size,
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}))


def clip_record(name: str, csv: Path, media: Path, thumb: bool = True) -> dict:
    n, w, h = media_frames(media)
    rec = {"name": name, "csv": str(csv), "media": str(media), "frames": n, "width": w,
           "height": h, "csv_sha256": sha256(csv),
           "video_sha256": sha256(media) if media.is_file() else None,
           **label_summary(csv, n, w, h)}
    if thumb:
        rec["_thumb"] = first_frame_thumb(media)
    return rec


def cmd_prepare(a: argparse.Namespace) -> dict:
    import train_detector as TD
    zpath, stage, root = Path(a.zip), Path(a.stage_dir), Path(a.train_root)
    extract_all(zpath, stage)
    clips = scan_archive_tree(stage)
    groups: dict[str, list[dict]] = {}
    for c in clips:
        groups.setdefault(c["group"], []).append(c)
    log(f"[prepare] archive clips by group: { {k: len(v) for k, v in groups.items()} }")
    if groups.get("unknown"):
        log(f"[prepare] WARNING unknown-group clips (not used): "
            f"{[str(c['csv']) for c in groups['unknown']][:5]}")
    train_recs = []
    for c in groups.get("pro", []) + groups.get("amateur", []):
        if c["media"] is None:
            log(f"[prepare] no media for {c['csv']}, skipped")
            continue
        m = train_match_name(c["group"], c["n"])
        dst_csv = root / m / "csv" / f"{c['clip']}_ball.csv"
        _link_or_copy(c["csv"], dst_csv)
        if c["media"].is_file():
            dst_media = root / m / "video" / f"{c['clip']}.mp4"
            _link_or_copy(c["media"], dst_media)
        else:
            dst_media = root / m / "frame" / c["clip"]
            if not dst_media.exists():
                shutil.copytree(c["media"], dst_media)
        rec = clip_record(f"{m}/{c['clip']}", dst_csv, dst_media)
        rec.update({"group": c["group"], "archive_path": str(c["csv"].relative_to(stage))})
        train_recs.append(rec)
    # held-out Test clips as measured by benchmark.py (local copy) + the archive's own Test copy
    test_recs = []
    for t in bench().real_clips():
        rec = clip_record(f"{t['match']}/{t['clip']}", Path(t["labels"]), Path(t["video"]))
        rec["orig_csv_sha256"] = sha256(Path(t["video"]).parent.parent / "csv" / f"{t['clip']}_ball.csv")
        test_recs.append(rec)
    local_test = {r["name"]: r for r in test_recs}
    arch_test = []
    for c in groups.get("test", []):
        if c["media"] is None or not c["media"].is_file():
            continue
        name = f"{c['match']}/{c['clip']}"
        loc = local_test.get(name)
        arch_test.append({"name": name, "video_identical": bool(loc and sha256(c["media"]) == loc["video_sha256"]),
                          "csv_identical_to_local_original": bool(loc and sha256(c["csv"]) == loc["orig_csv_sha256"])})
    ov = overlap_report(train_recs, test_recs)
    per_match: dict[str, dict] = {}
    for r in train_recs:
        pm = per_match.setdefault(r["name"].split("/")[0], {"clips": 0, "frames": 0, "visible": 0})
        pm["clips"] += 1
        pm["frames"] += r["frames"]
        pm["visible"] += r["visible"]
    pro = {k: v for k, v in per_match.items() if match_number(k) <= N_PRO}
    rally_mismatch = {f"match{i + 1}": (per_match.get(f"match{i + 1}", {}).get("clips", 0), n)
                      for i, n in enumerate(HACKMD_PRO_RALLIES)
                      if per_match.get(f"match{i + 1}", {}).get("clips", 0) != n}
    found = TD.discover([root], allow_held_out=False)
    checks = {
        "pro_matches": len(pro), "amateur_matches": len(per_match) - len(pro),
        "train_clips": len(train_recs), "pro_clips": sum(v["clips"] for v in pro.values()),
        "pro_frames": sum(v["frames"] for v in pro.values()), "hackmd_pro_frames": 68675,
        "train_frames": sum(r["frames"] for r in train_recs),
        "train_visible_labels": sum(r["visible"] for r in train_recs),
        "hackmd_rally_count_mismatch": rally_mismatch,
        "all_labels_parse_with_benchmark_parser": all(r["parsers_agree"] for r in train_recs),
        "all_label_xy_inside_image": all(r["xy_in_image"] for r in train_recs),
        "resolutions": sorted({f"{r['width']}x{r['height']}" for r in train_recs}),
        "label_rows_vs_frames_mismatch": [r["name"] for r in train_recs
                                          if r["max_label_frame"] >= r["frames"]][:20],
        "discover_finds": len(found), "discover_held_out": sum(c.held_out for c in found),
        "archive_test_clips": len(arch_test),
        "archive_test_video_identical_to_local": sum(t["video_identical"] for t in arch_test),
        "archive_test_csv_identical_to_local": sum(t["csv_identical_to_local_original"] for t in arch_test),
        "no_exact_overlap": not ov["exact_video_matches"] and not ov["exact_csv_matches"],
    }
    man = {"source": {
        "dataset": "Shuttlecock Trajectory Dataset (TrackNetV2), NYCU CoachAI",
        "page": "https://hackmd.io/@TUIK/rJkRW54cU (= https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw, "
                "linked from github.com/qaz812345/TrackNetV3 README)",
        "archive_url": "https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/"
                       "EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g?e=S0AB1Z",
        "archive_sha256": a.zip_sha256 or sha256(zpath), "archive_bytes": zpath.stat().st_size,
        "licence": "none stated on the dataset page or in the archive (see licence_files); "
                   "broadcast footage copyright stays with the broadcasters; research use only",
        "licence_files": sorted(str(p.relative_to(stage)) for p in stage.rglob("*")
                                if p.is_file() and re.search(r"licen|readme|copying", p.name, re.I))},
        "checks": checks, "per_match": per_match, "overlap": ov, "archive_test": arch_test,
        "train": [{k: v for k, v in r.items() if not k.startswith("_")} for r in train_recs],
        "test": [{k: v for k, v in r.items() if not k.startswith("_")} for r in test_recs]}
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(man, indent=1, default=str))
    log(json.dumps({"checks": checks, "overlap": {k: v for k, v in ov.items()}}, indent=1,
                   default=str))
    log(f"[prepare] wrote {MANIFEST_JSON}")
    return man
# ══════════════════════════════════════════════════════════════════════════════
# split: by match, professional only for validation
# ══════════════════════════════════════════════════════════════════════════════

def make_split(matches: list[str], n_val: int, seed: int) -> dict:
    """Seeded by-match split. Validation = ``n_val`` professional matches (Test is
    professional broadcast too); everything else (incl. amateur 24-26) trains."""
    nums = {m: match_number(m) for m in matches}
    if any(v is None for v in nums.values()):
        raise ValueError(f"not match<N> names: {[m for m, v in nums.items() if v is None]}")
    pro = sorted((m for m in matches if nums[m] <= N_PRO), key=nums.get)
    if len(pro) <= n_val:
        raise ValueError(f"only {len(pro)} professional matches for {n_val} validation matches")
    val = sorted(random.Random(seed).sample(pro, n_val), key=nums.get)
    train = sorted((m for m in matches if m not in val), key=nums.get)
    assert not set(val) & set(train) and len(val) + len(train) == len(set(matches))
    return {"seed": seed, "n_val": n_val, "val_matches": val, "train_matches": train}


def cmd_split(a: argparse.Namespace) -> dict:
    import train_detector as TD
    root = Path(a.train_root)
    if TD.is_held_out(root):
        raise SystemExit(f"{root} is (below) a held-out Test directory")
    matches = sorted((d.name for d in root.iterdir() if d.is_dir() and match_number(d.name)),
                     key=match_number)
    sp = make_split(matches, a.n_val, a.seed)
    per = {}
    for m in matches:
        cl = TD.discover([root / m], allow_held_out=False)
        per[m] = len(cl)
    sp.update({"train_root": str(root), "clips_per_match": per,
               "val_clips": sum(per[m] for m in sp["val_matches"]),
               "train_clips": sum(per[m] for m in sp["train_matches"]),
               "note": "Test (TrackNetV2/Test) is never part of this split; it is measured only by "
                       "scripts/benchmark.py --stage real"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(sp, indent=1))
    log(json.dumps(sp, indent=1))
    return sp


# ══════════════════════════════════════════════════════════════════════════════
# train: train_detector.main + early stopping + wall-clock budget
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStop:
    """Wraps train_detector._evaluate (called once per epoch on the validation clips only).
    After each validation it may lower ``args.epochs`` to the epoch just finished, which makes
    train_detector's ``while st.epoch < a.epochs`` loop exit cleanly after saving last.pt.
    Stops when val F1 has not beaten the best for ``patience`` evaluations, or when the next
    epoch would end past ``budget_s`` of wall time."""

    def __init__(self, patience: int, budget_s: float, min_delta: float = 0.0,
                 state_path: Path | None = None):
        self.patience, self.budget_s, self.min_delta = patience, budget_s, min_delta
        self.state_path = state_path
        self.best, self.bad, self.history, self.reason = -1.0, 0, [], None
        self.prior_s = 0.0
        self.t0 = time.time()
        self.t_last = self.t0
        if state_path and state_path.is_file():          # resumed run: keep patience/budget
            s = json.loads(state_path.read_text())
            self.best, self.bad, self.history = s["best"], s["bad"], s["history"]
            self.prior_s = s["elapsed_s"]

    def elapsed(self) -> float:
        return self.prior_s + time.time() - self.t0

    def after_eval(self, f1: float, args: argparse.Namespace) -> None:
        now = time.time()
        chunk = now - self.t_last
        self.t_last = now
        epoch = len(self.history) + 1
        improved = f1 > self.best + self.min_delta
        self.best, self.bad = (f1, 0) if improved else (self.best, self.bad + 1)
        self.history.append({"epoch": epoch, "val_f1": f1, "elapsed_min": round(self.elapsed() / 60, 1),
                             "chunk_min": round(chunk / 60, 1), "improved": improved})
        if self.bad >= self.patience:
            self.reason = f"early stop: no val F1 gain for {self.patience} evals (best {self.best})"
        elif self.elapsed() + chunk > self.budget_s:
            self.reason = (f"budget: {self.elapsed() / 60:.0f} min used, next epoch ~{chunk / 60:.0f} min "
                           f"> {self.budget_s / 60:.0f} min")
        if self.reason and args.epochs > epoch:
            args.epochs = epoch
        if self.state_path:
            self.state_path.write_text(json.dumps({"best": self.best, "bad": self.bad,
                                                   "history": self.history,
                                                   "elapsed_s": self.elapsed(), "reason": self.reason}))
        log(f"[ft] epoch {epoch}: val F1 {f1:.4f} best {self.best:.4f} bad {self.bad}/{self.patience} "
            f"elapsed {self.elapsed() / 60:.1f} min" + (f" -> {self.reason}" if self.reason else ""))


def install_checkpoint(best: Path, arch: str, dest: Path | None = None) -> Path:
    """Copy ``best`` to a NEW file (never overwrites) and prove it loads strict=True with
    identical tensors through detector.load_model (what build_detector/benchmark use)."""
    import torch
    import train_detector as TD
    dest = Path(dest) if dest else CKPT_DIR / FT_NAMES[arch]
    if dest.exists():
        raise FileExistsError(f"{dest} exists; refusing to overwrite (pass another --install-path)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    net, spec = TD.det().load_model(dest, arch)
    sd_src, _ = TD.det().load_checkpoint(best)
    same = all(torch.equal(v, sd_src[k]) for k, v in net.state_dict().items())
    if spec.name != arch or not same:
        dest.unlink()
        raise RuntimeError(f"round-trip failed for {dest}: spec {spec.name}, identical={same}")
    return dest


def _read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def split_roots(sp: dict) -> tuple[list[str], list[str]]:
    """(train match dirs, val match dirs) of a split; refuses anything held out."""
    import train_detector as TD
    root = Path(sp["train_root"])
    tr = [str(root / m) for m in sp["train_matches"]]
    va = [str(root / m) for m in sp["val_matches"]]
    bad = [p for p in tr + va if TD.is_held_out(Path(p))]
    if bad or set(tr) & set(va):
        raise SystemExit(f"invalid split: held-out {bad}, overlap {set(tr) & set(va)}")
    return tr, va


def cmd_train(a: argparse.Namespace) -> dict:
    import train_detector as TD
    sp = json.loads(Path(a.split).read_text())
    tr, va = split_roots(sp)
    out_dir = Path(a.out_dir or ML_DIR / "runs" / f"{a.arch}_ft")
    out_dir.mkdir(parents=True, exist_ok=True)
    common = ["--arch", a.arch, "--data-root", *tr, "--val-root", *va,
              "--cache-dir", a.cache_dir, "--workers", str(a.workers), "--seed", str(a.seed),
              "--eval-batch-size", str(a.eval_batch_size)] + (["--device", a.device] if a.device else [])
    t_all = time.time()
    summary: dict = {"arch": a.arch, "split": str(a.split), "val_matches": sp["val_matches"],
                     "train_matches": sp["train_matches"], "out_dir": str(out_dir)}
    # 1) pretrained reference on the SAME validation clips (also builds the frame caches)
    pre_p = out_dir / "pretrained_val.json"
    if pre_p.is_file():
        summary["pretrained_val"] = json.loads(pre_p.read_text())
    else:
        t = time.time()
        r = TD.main(common + ["--eval-only", "--init", "pretrained"])
        summary["pretrained_val"] = r["val"]
        summary["prep_and_pretrained_eval_s"] = round(time.time() - t, 1)
        pre_p.write_text(json.dumps(r["val"], indent=1))
    log(f"[ft] pretrained val: {summary['pretrained_val']}")
    if a.prep_only:
        (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary
    # 2) fine-tune with early stopping + budget (checkpoint selection on validation only)
    stop = EarlyStop(a.patience, a.budget_min * 60, a.min_delta, out_dir / "early_stop.json")
    orig = TD._evaluate

    def _evaluate(net, norm, clips, arch, args, device):
        m = orig(net, norm, clips, arch, args, device)
        if m is not None and clips and not any(c.held_out for c in clips):
            stop.after_eval(float(m["f1"]), args)
        return m

    train_argv = common + ["--init", a.init, "--epochs", str(a.epochs),
                           "--epoch-size", str(a.epoch_size), "--out-dir", str(out_dir),
                           "--log-every", "50", "--save-every", "250"]
    if a.batch_size:
        train_argv += ["--batch-size", str(a.batch_size)]
    if a.lr:
        train_argv += ["--lr", str(a.lr)]
    last = out_dir / "last.pt"
    if a.resume and last.is_file():
        train_argv = ["--resume", str(last), "--workers", str(a.workers)]
    TD._evaluate = _evaluate
    t = time.time()
    try:
        res = TD.main(train_argv)
    finally:
        TD._evaluate = orig
    train_s = time.time() - t
    recs = _read_jsonl(out_dir / "log.jsonl")
    start = [r for r in recs if r.get("event") == "start"]
    summary.update({
        "train_wall_s": round(train_s, 1), "train_wall_s_total_incl_resumes": round(stop.elapsed(), 1),
        "global_step": res["global_step"], "epochs_done": res["epoch"],
        "epoch_size_windows": a.epoch_size, "batch_size": a.batch_size or TD.ARCHS[a.arch].batch_size,
        "train_clips": start[-1]["train_clips"] if start else None,
        "val_clips": start[-1]["val_clips"] if start else None,
        "train_windows": start[-1]["train_windows"] if start else None,
        "lr": start[-1].get("lr") if start else None, "milestones": start[-1].get("milestones") if start else None,
        "best_val_f1": res["best_val_f1"], "val_history": stop.history, "stop_reason": stop.reason,
        "last_losses": res["losses"][-5:], "wall_s_total": round(time.time() - t_all, 1)})
    best = out_dir / "best.pth.tar"
    pre_f1 = float(summary["pretrained_val"]["f1"])
    summary["beats_pretrained_on_val"] = bool(best.is_file() and res["best_val_f1"] > pre_f1)
    if a.install and best.is_file():
        if summary["beats_pretrained_on_val"] or a.install_anyway:
            summary["installed"] = str(install_checkpoint(best, a.arch, a.install_path))
        else:
            summary["installed"] = None
            summary["install_skipped"] = (f"best val F1 {res['best_val_f1']} <= pretrained {pre_f1}; "
                                          "keep the released weights (--install-anyway overrides)")
    (out_dir / "ft_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(json.dumps(summary, indent=1, default=str))
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# score: Test-split before/after from benchmark.py caches (measurement only)
# ══════════════════════════════════════════════════════════════════════════════

def f1_counts(tp, fp, fn):
    """TrackNet F1 from pooled counts (fp = fp1 + fp2); == benchmark.add_tn4. Array-friendly."""
    tp, fp, fn = (np.asarray(v, np.float64) for v in (tp, fp, fn))
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def per_clip_counts(cache_dir: Path, kind: str, score_thresh: float = 0.5,
                    linker: bool = False) -> list[dict]:
    """benchmark.run_real's detector scoring (or, ``linker``, its real_linker scoring), clip by
    clip, from its npz cache."""
    from app.services.tracking import metrics as M
    from app.services.tracking import video_io
    B = bench()
    rows = []
    for clip in B.real_clips():
        meta = video_io.probe(clip["video"])
        n = meta.n_frames
        cache = Path(cache_dir) / f"{clip['match']}_{clip['clip']}_{kind}.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"{cache}: run scripts/benchmark.py --stage real --cache-dir {cache_dir}")
        z = np.load(cache)
        if int(z["n_done"]) < n:
            raise ValueError(f"{cache}: incomplete ({int(z['n_done'])} < {n})")
        cands = B._arr_to_cands(z["cands"][:n])
        true_xy, vis = B.load_label_csv(clip["labels"], n)
        dm = B.drop_mask(clip["match"], clip["clip"], n)
        mask = np.ones(n, bool) if dm is None else dm
        top, sc = M.candidates_to_xy(cands, score_thresh)
        if linker:                          # benchmark's real_linker: link the full candidate lists
            from app.services.tracking import linker as LK
            top = LK.link_candidates(cands, meta).xy
        r = M.detection_metrics(top, true_xy, vis, 4.0, mask=mask, tracknet_int=True,
                                image_size=(meta.width, meta.height))
        seen = vis & mask
        rows.append({"clip": f"{clip['match']}/{clip['clip']}", "eval_frames": int(mask.sum()),
                     "visible": int(seen.sum()), "tp": r["tp"], "fp": r["fp1"] + r["fp2"],
                     "fp1": r["fp1"], "fp2": r["fp2"], "fn": r["fn"], "f1": float(r["f1"]),
                     "median_top1_score_visible": float(np.median(sc[seen])) if seen.any() else 0.0})
    return rows


def pooled(rows: list[dict], names: set[str] | None = None) -> dict:
    sel = [r for r in rows if names is None or r["clip"] in names]
    tp, fp, fn = (sum(r[k] for r in sel) for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"clips": len(sel), "tp": tp, "fp": fp, "fn": fn, "precision": round(p, 5),
            "recall": round(rc, 5), "f1": float(f1_counts(tp, fp, fn))}


def weak_sets(base_rows: list[dict], k: int = 7) -> dict:
    """Pre-registered from the BASELINE run only (fixed before any fine-tuned number exists):
    low_confidence = k clips with the lowest median top-1 heatmap score on visible frames;
    long_rally = k clips with the most evaluated frames; low_f1 = k lowest baseline F1 (subject
    to regression to the mean, reported for completeness)."""
    by = lambda key, rev=False: [r["clip"] for r in sorted(base_rows, key=lambda r: (r[key], r["clip"]),  # noqa: E731
                                                                   reverse=rev)[:k]]
    return {"low_confidence": by("median_top1_score_visible"), "long_rally": by("eval_frames", True),
            "low_f1": by("f1")}


def paired_bootstrap(before: list[dict], after: list[dict], n_boot: int = 10000, seed: int = 0) -> dict:
    """Resample CLIPS with replacement (frames within a clip are correlated), recompute pooled
    F1 before and after on the same resample -> 95% CI of the difference."""
    idx = {r["clip"]: i for i, r in enumerate(before)}
    aft = sorted(after, key=lambda r: idx[r["clip"]])
    b = np.array([[r["tp"], r["fp"], r["fn"]] for r in before], np.float64)
    c = np.array([[r["tp"], r["fp"], r["fn"]] for r in aft], np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(b), size=(n_boot, len(b)))
    bs, cs = b[pick].sum(1), c[pick].sum(1)
    d = f1_counts(*cs.T) - f1_counts(*bs.T)
    obs = float(f1_counts(*c.sum(0)) - f1_counts(*b.sum(0)))
    return {"delta_f1": obs, "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_delta_le_0": float(np.mean(d <= 0)), "n_boot": n_boot}


def sign_test(before: list[dict], after: list[dict], eps: float = 1e-9) -> dict:
    fa = {r["clip"]: r["f1"] for r in after}
    diffs = [fa[r["clip"]] - r["f1"] for r in before]
    up, down = sum(d > eps for d in diffs), sum(d < -eps for d in diffs)
    n = up + down
    k = min(up, down)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"improved": up, "worse": down, "tied": len(diffs) - n, "p_two_sided": p,
            "median_clip_delta": float(np.median(diffs)),
            "clip_delta_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0}
def cmd_score(a: argparse.Namespace) -> dict:
    stage = "real_linker" if a.linker else "real_detector"
    before = per_clip_counts(Path(a.before), a.kind, linker=a.linker)
    wpath = Path(a.weak_json or RUN_DIR / f"weak_clips_{a.kind}.json")
    if wpath.is_file():
        weak = json.loads(wpath.read_text())
    else:                                   # freeze on the first (baseline) scoring
        weak = weak_sets(before, a.weak_k)
        wpath.parent.mkdir(parents=True, exist_ok=True)
        wpath.write_text(json.dumps(weak, indent=1))
        log(f"[score] froze weak clip sets -> {wpath}")
    rep: dict = {"kind": a.kind, "stage": stage, "before_cache": a.before, "after_cache": a.after,
                 "rule": "benchmark.py --stage real: top-1 score>=0.5, tn4 (4 px @512x288, int "
                         "truncation), corrected labels, drop_frame.json ranges, pooled counts",
                 "weak_sets": weak, "before": {"all": pooled(before)}}
    for k, names in weak.items():
        rep["before"][k] = pooled(before, set(names))
    if a.check_json:
        ref = json.loads(Path(a.check_json).read_text())["results"][stage]["tn4_f1"]
        rep["before_matches_benchmark_json"] = abs(ref - rep["before"]["all"]["f1"]) < 1e-12
    if a.after:
        after = per_clip_counts(Path(a.after), a.kind, linker=a.linker)
        rep["after"] = {"all": pooled(after)}
        for k, names in weak.items():
            rep["after"][k] = pooled(after, set(names))
        rep["delta"] = {k: rep["after"][k]["f1"] - rep["before"][k]["f1"] for k in rep["after"]}
        rep["bootstrap_all"] = paired_bootstrap(before, after, a.n_boot)
        for k, names in weak.items():
            s = set(names)
            rep[f"bootstrap_{k}"] = paired_bootstrap([r for r in before if r["clip"] in s],
                                                    [r for r in after if r["clip"] in s], a.n_boot)
        rep["sign_test"] = sign_test(before, after)
        rep["per_match_delta"] = {}
        for m in sorted({r["clip"].split("/")[0] for r in before}):
            names = {r["clip"] for r in before if r["clip"].startswith(m + "/")}
            rep["per_match_delta"][m] = pooled(after, names)["f1"] - pooled(before, names)["f1"]
        fa = {r["clip"]: r for r in after}
        rep["per_clip"] = [{"clip": r["clip"], "f1_before": r["f1"], "f1_after": fa[r["clip"]]["f1"],
                            "fn_before": r["fn"], "fn_after": fa[r["clip"]]["fn"],
                            "fp_before": r["fp"], "fp_after": fa[r["clip"]]["fp"]} for r in before]
        ci = rep["bootstrap_all"]["ci95"]
        rep["verdict"] = ("real gain" if ci[0] > 0 else "real loss" if ci[1] < 0 else
                          "within noise (95% clip-bootstrap CI spans 0)")
    else:
        rep["per_clip"] = before
    out = Path(a.out or RUN_DIR / f"score_{a.kind}{'_' + a.tag if a.tag else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str))
    short = {k: v for k, v in rep.items() if k not in ("per_clip", "weak_sets")}
    log(json.dumps(short, indent=1, default=str))
    log(f"[score] wrote {out}")
    return rep


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="unpack + verify the Train split, check Test overlap")
    p.add_argument("--zip", required=True)
    p.add_argument("--zip-sha256", default=None, help="skip re-hashing the archive")
    p.add_argument("--stage-dir", default=str(RAW / "_download" / "extracted"))
    p.add_argument("--train-root", default=str(TRAIN_ROOT))
    p = sub.add_parser("split", help="seeded by-match validation split")
    p.add_argument("--train-root", default=str(TRAIN_ROOT))
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--out", default=str(SPLIT_JSON))
    p = sub.add_parser("train", help="fine-tune with early stopping + wall-clock budget")
    p.add_argument("--arch", choices=("wasb", "tracknetv3"), default="wasb")
    p.add_argument("--split", default=str(SPLIT_JSON))
    p.add_argument("--init", default="pretrained")
    p.add_argument("--out-dir", default=None, help="default ml/runs/<arch>_ft")
    p.add_argument("--cache-dir", default=str(ML_DIR / "data" / "processed" / "detector_cache"))
    p.add_argument("--epochs", type=int, default=10, help="planned epochs (LR milestones at 1/2, 3/4)")
    p.add_argument("--epoch-size", type=int, default=6000, help="training windows per epoch")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None, help="default: trainer's fine-tune 1e-4")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--budget-min", type=float, default=100.0, help="wall-clock training budget")
    p.add_argument("--resume", action="store_true", help="continue <out-dir>/last.pt")
    p.add_argument("--prep-only", action="store_true",
                   help="only build frame caches + pretrained val F1 (no training)")
    p.add_argument("--install", action="store_true",
                   help="copy best.pth.tar to ml/weights/ckpts/*_ft.* if it beats pretrained on val")
    p.add_argument("--install-anyway", action="store_true")
    p.add_argument("--install-path", default=None)
    p = sub.add_parser("score", help="Test-split before/after from benchmark.py caches")
    p.add_argument("--kind", choices=("wasb", "tracknetv3"), required=True)
    p.add_argument("--before", required=True, help="benchmark --cache-dir of the baseline run")
    p.add_argument("--after", default=None, help="benchmark --cache-dir of the fine-tuned run")
    p.add_argument("--check-json", default=None, help="baseline benchmark JSON to cross-check")
    p.add_argument("--weak-json", default=None)
    p.add_argument("--weak-k", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--linker", action="store_true",
                   help="score the linked track (benchmark real_linker) instead of top-1 detections")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    return ap


def main(argv: list[str] | None = None):
    a = make_parser().parse_args(argv)
    return {"prepare": cmd_prepare, "split": cmd_split, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
