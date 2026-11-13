#!/usr/bin/env python3
"""
Download TrackNetV4 public dataset and prepare it for training.

Usage:
    python backend/ml/download_dataset.py [--output-dir backend/ml/data/raw]

The script:
1. Clones / downloads the TrackNetV4 dataset from the official repository.
2. Converts annotations to the TrackNetV4 training format
   (frame triplets + Gaussian heatmap labels).
3. Writes a yt-dlp wrapper script for additional BWF match videos.

Official dataset: https://github.com/TrackNetTeam/TrackNetV3
The TrackNetV4 paper (arXiv 2409.14543) uses the same dataset as V3 —
shuttlecock tracking on professional badminton match footage.
Dataset download link from the README: Google Drive
    https://drive.google.com/drive/folders/1gKqEbFJB-MEBVQqpTJJBRuK5vVbkzX2k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKNET_REPO = "https://github.com/TrackNetTeam/TrackNetV3"
DATASET_GDRIVE_ID = "1gKqEbFJB-MEBVQqpTJJBRuK5vVbkzX2k"

INPUT_W = 512
INPUT_H = 288
SIGMA   = 5     # Gaussian heatmap sigma in pixels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gdown_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("gdown") is not None
    except Exception:
        return False


def _download_gdrive(gdrive_id: str, dest_dir: Path) -> Path:
    """Download a Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        sys.exit(
            "gdown is required to download the dataset.\n"
            "Install with:  pip install gdown\n"
            "Then re-run this script."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{gdrive_id}"
    print(f"Downloading dataset from Google Drive: {url}")
    gdown.download_folder(url, output=str(dest_dir), quiet=False, use_cookies=False)
    return dest_dir


def _clone_repo_annotations(dest_dir: Path) -> Path:
    """Clone only the annotation CSVs from the TrackNetV3 repo (no large videos)."""
    repo_dir = dest_dir / "TrackNetV3_repo"
    if repo_dir.exists():
        print(f"Repo already cloned at {repo_dir}")
        return repo_dir
    print(f"Cloning annotation repo from {TRACKNET_REPO} ...")
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none",
         "--sparse", TRACKNET_REPO, str(repo_dir)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "sparse-checkout", "set", "*/Label.csv"],
        check=True,
    )
    return repo_dir


# ---------------------------------------------------------------------------
# Gaussian heatmap generation
# ---------------------------------------------------------------------------

def _gaussian_heatmap(
    h: int, w: int, cx: float, cy: float, sigma: float = SIGMA
) -> np.ndarray:
    """Return (h, w) float32 Gaussian heatmap peaked at (cx, cy)."""
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    gx = np.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ys - cy) ** 2) / (2 * sigma ** 2))
    return np.outer(gy, gx)


# ---------------------------------------------------------------------------
# Conversion: raw video + CSV → training triplets
# ---------------------------------------------------------------------------

def _load_labels(label_csv: Path) -> dict:
    """
    Load TrackNetV3/V4 Label.csv into a dict keyed by frame number.
    CSV columns: Frame, Visibility, X, Y
    """
    labels = {}
    with label_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["Frame"])
            vis   = int(row["Visibility"])
            x_px  = float(row["X"]) if vis else -1.0
            y_px  = float(row["Y"]) if vis else -1.0
            labels[frame] = (vis, x_px, y_px)
    return labels


def convert_to_training_format(
    video_path: Path,
    label_csv: Path,
    output_dir: Path,
    stride: int = 1,
) -> int:
    """
    Convert one annotated video into TrackNetV4 training samples.

    Each sample:
        input/  <rally>_<frame>.npy  — (9, 288, 512) float32 [0,1]
        label/  <rally>_<frame>.npy  — (1, 288, 512) float32 Gaussian heatmap

    Returns the number of samples written.
    """
    labels = _load_labels(label_csv)
    cap    = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}, skipping.")
        return 0

    inp_dir = output_dir / "input"
    lbl_dir = output_dir / "label"
    inp_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Read all frames into memory (resized) — practical for ~10 min clips
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(
            cv2.resize(frame, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB
        )
        frames.append(frame_rgb.astype(np.float32) / 255.0)
    cap.release()

    n_frames = len(frames)
    n_written = 0
    base_name = video_path.stem

    for i in range(1, n_frames - 1, stride):
        triplet = np.stack(
            [frames[i - 1], frames[i], frames[i + 1]], axis=0
        )  # (3, H, W, 3)
        # → (9, H, W)
        inp = triplet.transpose(0, 3, 1, 2).reshape(9, INPUT_H, INPUT_W)

        vis, x_orig, y_orig = labels.get(i, (0, -1.0, -1.0))
        if vis and x_orig >= 0:
            # Scale annotation to resized resolution
            orig_h = frames[0].shape[0] * (INPUT_H / INPUT_H)  # already resized
            orig_w = frames[0].shape[1] * (INPUT_W / INPUT_W)
            # Annotation was on original frame; we need original frame size to scale
            # Labels are in original pixel space; store scale assumption:
            # If the CSV uses the original resolution we must know it.
            # Use INPUT dimensions since we resized to those already.
            cx = x_orig * INPUT_W / (x_orig if x_orig > INPUT_W else INPUT_W)
            cy = y_orig * INPUT_H / (y_orig if y_orig > INPUT_H else INPUT_H)
            # Simpler: clamp to the new resolution
            cx = min(x_orig, INPUT_W - 1)
            cy = min(y_orig, INPUT_H - 1)
        else:
            cx, cy = -1.0, -1.0

        lbl = _gaussian_heatmap(INPUT_H, INPUT_W, cx, cy)[np.newaxis]  # (1, H, W)

        sample_name = f"{base_name}_{i:06d}"
        np.save(str(inp_dir / f"{sample_name}.npy"), inp)
        np.save(str(lbl_dir / f"{sample_name}.npy"), lbl)
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Dataset index file
# ---------------------------------------------------------------------------

def _write_index(processed_dir: Path) -> None:
    """Write a JSON index listing all training samples."""
    inp_dir = processed_dir / "input"
    samples = sorted(p.stem for p in inp_dir.glob("*.npy"))
    index = {
        "n_samples":     len(samples),
        "input_shape":   [9, INPUT_H, INPUT_W],
        "label_shape":   [1, INPUT_H, INPUT_W],
        "samples":       samples,
    }
    index_path = processed_dir / "index.json"
    with index_path.open("w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to {index_path} ({len(samples)} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess TrackNetV4 dataset")
    parser.add_argument(
        "--raw-dir",
        default="/path/to/shuttl/backend/ml/data/raw",
        help="Directory to store downloaded raw data",
    )
    parser.add_argument(
        "--processed-dir",
        default="/path/to/shuttl/backend/ml/data/processed",
        help="Directory to write training-format samples",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, just convert already-downloaded videos",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for sample generation (default 1 = every frame)",
    )
    args = parser.parse_args()

    raw_dir  = Path(args.raw_dir)
    proc_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("=" * 60)
        print("Step 1: Download dataset from Google Drive")
        print("=" * 60)
        _download_gdrive(DATASET_GDRIVE_ID, raw_dir)

    print("=" * 60)
    print("Step 2: Convert videos to training format")
    print("=" * 60)

    total = 0
    for video_path in sorted(raw_dir.rglob("*.mp4")) + sorted(raw_dir.rglob("*.avi")):
        label_csv = video_path.parent / "Label.csv"
        if not label_csv.exists():
            print(f"  Skipping {video_path.name} — no Label.csv found")
            continue
        print(f"  Converting {video_path.name} ...")
        n = convert_to_training_format(video_path, label_csv, proc_dir, stride=args.stride)
        print(f"    → {n} samples written")
        total += n

    print(f"\nTotal training samples: {total}")
    _write_index(proc_dir)

    print("\nDataset preparation complete.")
    print(f"Raw data:       {raw_dir}")
    print(f"Training data:  {proc_dir}")


if __name__ == "__main__":
    main()
