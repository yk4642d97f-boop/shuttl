#!/usr/bin/env python3
"""Fine-tune (or train) the shuttle heatmap detectors: WASB HRNet (primary, the on-device model)
and TrackNetV3 (the server default), on TrackNetV3-format labels + sim.py renders.

DATA (TrackNetV2 / TrackNetV3 "Shuttlecock Trajectory Dataset" layout, any depth below a root):
    .../<match>/csv/<clip>_ball.csv     Frame,Visibility,X,Y  (full-res px, Visibility 0 = unseen)
    .../<match>/video/<clip>.mp4        or  .../<match>/frame/<clip>/<index>.{png,jpg}
  Clips below a directory named "Test" are the HELD-OUT split: they are refused for training and
  validation and only scored after training with --test-eval (report only, never used to select
  checkpoints or tune anything). Validation = a seeded clip-level --val-frac slice of the real
  training clips (or --val-root). --synthetic N renders N rallies with sim.py (random_camera ->
  simulate_rally -> render_video) as one more TrackNetV3-format match; skipped with a warning if
  sim.py lacks them. Frames are decoded once, warped to the 512x288 canvas exactly like inference
  and cached as JPEG q95 (+ a per-clip median background) under --cache-dir.

TRAINING mirrors the official code:
  WASB (nttcom/WASB-SBDT: dataloaders/dataset_loader.py, seq_transforms.py, utils/heatmap.py,
  losses/*.py, configs/{transform/full,optimizer/adam_multistep,dataloader/default}.yaml, and
  arXiv:2311.05237 sec. 3-4): 3 frames aspect-fitted by cv2.warpAffine INTER_LINEAR in the corner
  convention net = s*full + pad, ImageNet mean/std; target = binary disc r=2.5 per frame; loss =
  quality focal loss beta=2 on sigmoid clamped to [1e-4, 1-1e-4], mean over pixels (== WBCE ==
  focal gamma=2 on binary targets); Adam lr 1e-3, MultiStepLR [10, 20] x0.1, 30 epochs, batch 8;
  HLSM (--hlsm-epoch, paper: 20): frames the model mislocalises (TrackNet FP1) get the
  real-valued target min(C exp(-d^2/r^2), 1) zeroed below 0.5 with c_min 0.7.
  TrackNetV3 (qaz812345/TrackNetV3: train.py, dataset.py, model.py): median background + 8
  frames, x/255, pixel-centre resize (detector.py: INTER_AREA); target = the same r=2.5 disc at
  int(x * 512 / w); loss = WBCE on the model's sigmoid output with logs clamped at 1e-7; Adam
  lr 1e-3; README recipe uses sample mixup alpha 0.5.
  Augmentation (both): WASB 'full' - colour jitter p=.5 (brightness/contrast/saturation/hue 0.1,
  one draw per sample), horizontal flip p=.5, x1.125 bilinear rescale + random crop p=.5 applied
  to images and targets alike; opt-in --blur-p; --mixup-alpha (TrackNetV3 sample mixup).
  Fine-tuning (default: from the released checkpoint, ml/weights/wasb_badminton_best.pth.tar or
  ml/weights/ckpts/TrackNet_best.pt): lr 1e-4, milestones at 1/2 and 3/4 of --epochs.

CENTROID-OFFSET CONSISTENCY. The shipped ShuttleTracker.mlpackage (metadata centroid_offset=0.5)
and detector.py (NetSpec.peak_offset = -0.5 WASB, -0.2 TrackNetV3) decode position = heatmap
centroid - peak_offset, then the inverse letterbox. --target-mode official (default) draws the
disc at the upstream integer pixel int(s*x + pad), so the net learns to peak ~0.5 px (WASB) /
0.5*s px (TrackNetV3, s = 512/1280 -> 0.2) low, exactly the offsets baked into the decoders.
--target-mode subpixel draws it at content position + peak_offset (continuous), which the same
decode inverts without bias and without the +-0.5 px integer-grid quantisation. Augmentations move the continuous content
position with the image (flip x -> W-1-x; rescale+crop by the half-pixel mapping) and targets are
drawn afterwards, so the offset survives them (flipping a drawn target would mirror it to +0.5).
So checkpoints from this script keep centroid_offset=0.5 and export with the same contract:
  .venv/bin/python ml/export_coreml.py --weights ml/runs/wasb_ft/best.pth.tar --output X.mlpackage

EVAL: per-frame precision / recall / F1 at 4 px in 512x288 net space (TrackNet standard;
FP = far detections + detections on invisible frames), non-overlapping windows (WASB step=S),
the iOS decode (peak >= 0.5, largest-mass connected blob, weighted centroid - peak_offset);
median error in full-res px. Test clips use corrected labels and drop_frame.json ranges.

Usage (from backend/):
  .venv/bin/python ml/train_detector.py --data-root /data/TrackNetV2/Train          # fine-tune WASB
  .venv/bin/python ml/train_detector.py --arch tracknetv3 --data-root D --batch-size 4
  .venv/bin/python ml/train_detector.py --data-root D --init scratch --epochs 30 --hlsm-epoch 20
  .venv/bin/python ml/train_detector.py --data-root D --synthetic 200               # + sim renders
  .venv/bin/python ml/train_detector.py --resume ml/runs/wasb_ft/last.pt [--epochs 40]
  .venv/bin/python ml/train_detector.py --eval-only --test-eval [--test-max-clips 6]
  .venv/bin/python ml/train_detector.py --smoke [--arch tracknetv3]          # 20-iter self-test
Outputs (--out-dir, default ml/runs/<arch>_ft): last.pt (resumable: model, optimizer,
scheduler, RNG, sampler position, HLSM set), best.pth.tar (official checkpoint format + 'shuttl'
metadata; loads with strict=True in detector.py and export_coreml.py), log.jsonl. --install
copies best to ml/weights/{wasb_badminton_finetuned.pth.tar, ckpts/TrackNet_finetuned.pt},
which detector.build_detector prefers over the released weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Sampler

BACKEND = Path(__file__).resolve().parents[1]
ML_DIR = BACKEND / "ml"
for _p in (str(BACKEND), str(ML_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Same decode + detector import helper that validated the shipped CoreML package.
from export_coreml import CARDS, NET_H, NET_W, blob_decode, import_detector  # noqa: E402
from export_coreml import _drop_range as _test_drop_range  # noqa: E402

WEIGHTS_DIR = ML_DIR / "weights"
DATA_ROOT = ML_DIR / "data" / "raw" / "shuttlecock_trajectory"
DEFAULT_DATA = DATA_ROOT / "TrackNetV2"
DEFAULT_TEST = DATA_ROOT / "TrackNetV2" / "Test"
DEFAULT_CACHE = ML_DIR / "data" / "processed" / "detector_cache"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HELD_OUT_DIRS = {"test"}
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

_DET = None


def det():
    """app.services.tracking.detector (architectures, FrameGeometry, load_model), imported once."""
    global _DET
    if _DET is None:
        _DET = import_detector()
    return _DET


def log(msg: str) -> None:
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Architectures (conventions must match detector.SPECS and the CoreML contract)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Arch:
    name: str
    seq_len: int
    uses_background: bool
    imagenet_norm: bool
    logits: bool                # net outputs logits (WASB) vs sigmoid probabilities (TrackNetV3)
    corner: bool                # FrameGeometry convention of the preprocessing
    interp: int
    peak_offset: float          # learned (target - content) offset in net px; decode subtracts it
    loss: str
    batch_size: int
    mixup_alpha: float
    pretrained: Path
    install: Path

    @property
    def frames_in(self) -> int:
        return self.seq_len + int(self.uses_background)

    @property
    def cache_tag(self) -> str:
        return f"{'corner' if self.corner else 'center'}-interp{self.interp}-v1"


ARCHS: dict[str, Arch] = {
    "wasb": Arch("wasb", 3, False, True, True, True, cv2.INTER_LINEAR, -0.5, "qfl", 8, 0.0,
                 WEIGHTS_DIR / "wasb_badminton_best.pth.tar",
                 WEIGHTS_DIR / "wasb_badminton_finetuned.pth.tar"),
    # batch 10 upstream (CUDA); 4 keeps a shared 24 GB Mac responsive
    "tracknetv3": Arch("tracknetv3", 8, True, False, False, False, cv2.INTER_AREA, -0.2, "wbce",
                       4, 0.5, WEIGHTS_DIR / "ckpts" / "TrackNet_best.pt",
                       WEIGHTS_DIR / "ckpts" / "TrackNet_finetuned.pt"),
}


def check_conventions(arch: Arch) -> None:
    """Fail loudly if detector.py / the CoreML card drifted from what we train."""
    spec = getattr(det(), "SPECS", {}).get(arch.name)
    if spec is not None:
        mine = (arch.seq_len, arch.uses_background, arch.imagenet_norm, arch.logits, arch.corner,
                arch.peak_offset)
        theirs = (spec.seq_len, spec.uses_background, spec.imagenet_norm, spec.logits,
                  spec.corner_geometry, spec.peak_offset)
        if mine != theirs:
            raise RuntimeError(f"{arch.name}: training conventions {mine} != detector {theirs}")
    card = CARDS.get(arch.name)
    if arch.name == "wasb" and card is not None and abs(card.centroid_offset + arch.peak_offset) > 1e-9:
        raise RuntimeError("WASB peak_offset disagrees with the CoreML centroid_offset metadata")


# ══════════════════════════════════════════════════════════════════════════════
# Clip discovery (TrackNetV3 CSV format) + held-out guard
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClipSource:
    name: str                  # "<match>/<clip>"
    labels_csv: Path
    media: Path                # video file or frame directory
    held_out: bool


def is_held_out(path: Path) -> bool:
    return any(part.lower() in HELD_OUT_DIRS for part in Path(path).resolve().parts)


def _corrected_csv(csv: Path) -> Path | None:
    """TrackNetV3's corrected Test labels <...>/corrected/<match>/corrected_csv/<clip>_ball.csv
    (held-out clips only: Train and Test both use match1..N names)."""
    match = csv.parent.parent.name
    for up in list(csv.parents)[2:5]:
        p = up / "corrected" / match / "corrected_csv" / csv.name
        if p.is_file():
            return p
    return None


def discover(roots: list, allow_held_out: bool) -> list[ClipSource]:
    """All labelled clips below ``roots``; held-out (Test) clips dropped unless allowed."""
    out: list[ClipSource] = []
    seen: set[Path] = set()
    refused = 0
    for root in map(Path, roots):
        if not root.exists():
            log(f"[data] missing root {root}")
            continue
        for csv in sorted(root.rglob("*_ball.csv")):
            if csv.parent.name != "csv" or csv.resolve() in seen:
                continue
            seen.add(csv.resolve())
            held = is_held_out(csv)
            if held and not allow_held_out:
                refused += 1
                continue
            match_dir, clip = csv.parent.parent, csv.name[: -len("_ball.csv")]
            media = next((match_dir / "video" / f"{clip}{e}" for e in VIDEO_EXTS
                          if (match_dir / "video" / f"{clip}{e}").is_file()), None)
            if media is None and (match_dir / "frame" / clip).is_dir():
                media = match_dir / "frame" / clip
            if media is None:
                log(f"[data] no video/frames for {csv}, skipped")
                continue
            labels = (_corrected_csv(csv) or csv) if held else csv
            out.append(ClipSource(f"{match_dir.name}/{clip}", labels, media, held))
    if refused:
        log(f"[data] refused {refused} held-out Test clips for training/validation")
    assert allow_held_out or not any(c.held_out for c in out)
    return out


def load_label_csv(path: Path, n: int) -> np.ndarray:
    """(n, 3) float64 [visible, x, y] per frame; missing rows = not visible."""
    lab = np.zeros((n, 3), np.float64)
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if rows.size:
        rows = rows[(rows[:, 0] >= 0) & (rows[:, 0] < n)]
        lab[rows[:, 0].astype(int)] = rows[:, 1:4]
    lab[:, 0] = lab[:, 0] > 0
    return lab


# ══════════════════════════════════════════════════════════════════════════════
# Net-resolution frame cache
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CachedClip:
    name: str
    dir: Path
    n: int
    width: int
    height: int
    fps: float
    labels: np.ndarray         # (n, 3) [visible, x, y] full-res px
    held_out: bool
    corner: bool
    synthetic: bool = False
    _bg: np.ndarray | None = field(default=None, repr=False)

    @property
    def geom(self):
        return det().FrameGeometry(self.width, self.height, corner=self.corner)

    @property
    def corner_geom(self):
        return det().FrameGeometry(self.width, self.height, corner=True)

    def content_xy(self, f: int) -> np.ndarray:
        """Where frame ``f``'s labelled shuttle sits on the net canvas (pixel-index coords)."""
        return np.array(self.geom.to_net(self.labels[f, 1], self.labels[f, 2]), np.float64)

    def frame_rgb(self, f: int) -> np.ndarray:
        img = cv2.imread(str(self.dir / f"{f:06d}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"missing cached frame {self.dir}/{f:06d}.jpg")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def background_rgb(self) -> np.ndarray:
        if self._bg is None:
            self._bg = cv2.cvtColor(cv2.imread(str(self.dir / "bg.png"), cv2.IMREAD_COLOR),
                                    cv2.COLOR_BGR2RGB)
        return self._bg


def _iter_media(media: Path):
    if media.is_dir():
        files = sorted((p for p in media.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                       key=lambda p: (0, int(p.stem), "") if p.stem.isdigit() else (1, 0, p.stem))
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(str(media))
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
    finally:
        cap.release()


def _media_fps(media: Path) -> float:
    if media.is_dir():
        return 30.0
    cap = cv2.VideoCapture(str(media))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps) if 1.0 < fps < 1000.0 else 30.0


def cache_clip(src: ClipSource, cache_dir: Path, arch: Arch, quality: int = 95) -> CachedClip:
    """Decode once, warp to the canvas exactly like detector.py inference for ``arch``, store
    JPEGs + bg.png = median of frames sampled at ~4 Hz (<= 64, as detector.estimate_backgrounds)."""
    st = src.media.stat()
    key = hashlib.sha1(f"{src.media.resolve()}|{st.st_size}|{st.st_mtime_ns}|{NET_W}x{NET_H}|"
                       f"{arch.cache_tag}|q{quality}".encode()).hexdigest()[:16]
    d = Path(cache_dir) / key
    meta_p = d / "meta.json"
    if not meta_p.is_file():
        tmp = Path(cache_dir) / f"{key}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        fps = _media_fps(src.media)
        step = max(1, int(round(fps / 4.0)))
        n, w, h, geom, samples, stride = 0, 0, 0, None, [], 1
        for frame in _iter_media(src.media):
            if geom is None:
                h, w = frame.shape[:2]
                geom = det().FrameGeometry(w, h, corner=arch.corner)
            net = geom.warp(frame, arch.interp)
            cv2.imwrite(str(tmp / f"{n:06d}.jpg"), net, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if n % (step * stride) == 0:
                samples.append(net)
                if len(samples) > 128:
                    samples, stride = samples[::2], stride * 2
            n += 1
        if n == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise IOError(f"no frames decoded from {src.media}")
        samples = samples[:: max(1, len(samples) // 64)][:64]
        cv2.imwrite(str(tmp / "bg.png"), np.median(np.stack(samples), 0).astype(np.uint8))
        (tmp / "meta.json").write_text(json.dumps(
            {"n": n, "width": w, "height": h, "fps": fps, "source": str(src.media)}))
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
    meta = json.loads(meta_p.read_text())
    return CachedClip(src.name, d, meta["n"], meta["width"], meta["height"], meta["fps"],
                      load_label_csv(src.labels_csv, meta["n"]), src.held_out, arch.corner,
                      synthetic=src.name.startswith("sim_"))


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic renders (sim.py, imported defensively)
# ══════════════════════════════════════════════════════════════════════════════

def render_synthetic(n: int, root: Path, seed: int, size: tuple[int, int], fps: float) -> Path | None:
    """Render ``n`` sim rallies as a TrackNetV3-format match below ``root``; None if unavailable."""
    if n <= 0:
        return None
    try:
        from app.services.tracking import sim
    except Exception as e:  # noqa: BLE001 - sibling module may be mid-edit
        log(f"[synthetic] sim.py not importable ({e!r}); skipping synthetic data")
        return None
    if not (hasattr(sim, "simulate_rally") and hasattr(sim, "render_video")):
        log("[synthetic] sim.py has no simulate_rally/render_video yet; skipping synthetic data")
        return None
    match = Path(root) / f"sim_{size[0]}x{size[1]}_{fps:g}fps"
    (match / "video").mkdir(parents=True, exist_ok=True)
    (match / "csv").mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(n):
        clip = f"rally_{seed + i:06d}"
        vid, csv = match / "video" / f"{clip}.mp4", match / "csv" / f"{clip}_ball.csv"
        if vid.is_file() and csv.is_file():
            continue
        try:
            cam = None
            if hasattr(sim, "random_camera"):
                cam = sim.random_camera(np.random.default_rng(seed + i), image_size=size)
            r = sim.simulate_rally(seed + i, camera=cam, fps=fps)
            sim.render_video(r, str(vid), image_size=size)
            k = np.array(size, np.float64) / np.asarray(r.camera.image_size, np.float64)
            px = np.asarray(r.shuttle_px, np.float64) * k
            vis = np.asarray(r.visible, bool) & np.isfinite(px).all(1)
            with open(csv, "w") as fh:
                fh.write("Frame,Visibility,X,Y\n")
                for f in range(len(vis)):
                    x, y = px[f] if vis[f] else (0.0, 0.0)
                    fh.write(f"{f},{int(vis[f])},{x:.3f},{y:.3f}\n")
            made += 1
        except Exception as e:  # noqa: BLE001 - never let the renderer kill training
            log(f"[synthetic] rally {seed + i} failed: {e!r}")
            vid.unlink(missing_ok=True)
            csv.unlink(missing_ok=True)
    log(f"[synthetic] {made} new rallies rendered into {match}")
    return Path(root)


# ══════════════════════════════════════════════════════════════════════════════
# Targets (utils/heatmap.py gen_binary_map / gen_heatmap) + augmentation (seq_transforms.py)
# ══════════════════════════════════════════════════════════════════════════════

def target_xy(content_xy, arch: Arch, mode: str) -> tuple[float, float]:
    """Disc centre (pixel-index coords) for a shuttle whose image content sits at ``content_xy``
    on the (possibly augmented) canvas. 'subpixel': content + peak_offset. 'official':
    floor(content + peak_offset + 0.5), which equals upstream exactly on un-augmented frames:
    WASB int(s*x + pad) (content = s*x + pad - 0.5, offset -0.5) and TrackNetV3 int(x*512/w)
    (content = 0.4x - 0.3, offset -0.2 at 1280 px wide; the TrackNetV2 data are all 1280x720).
    Working from the content position keeps the offset right under flip/crop, where
    transforming a drawn target would mirror it (+0.5 instead of -0.5 in x after a flip)."""
    cx, cy = float(content_xy[0]) + arch.peak_offset, float(content_xy[1]) + arch.peak_offset
    if mode in ("official", "wasb"):
        return float(math.floor(cx + 0.5)), float(math.floor(cy + 0.5))
    if mode == "subpixel":
        return cx, cy
    raise ValueError(f"unknown target mode {mode!r}")


def draw_target(cx: float, cy: float, r: float = 2.5, binary: bool = True,
                cmin: float = 0.7) -> np.ndarray:
    """(H, W) float32. binary: 1 where |p - c| <= r (WASB eq. 1 / TrackNetV3 _get_heatmap).
    Else the HLSM map exp(-d^2/r^2) / e^-1 * cmin, clipped to 1 and zeroed below 0.5 (eq. 2)."""
    hm = np.zeros((NET_H, NET_W), np.float32)
    m = 1.5 * r + 1.0
    x0, x1 = max(int(math.floor(cx - m)), 0), min(int(math.ceil(cx + m)) + 1, NET_W)
    y0, y1 = max(int(math.floor(cy - m)), 0), min(int(math.ceil(cy + m)) + 1, NET_H)
    if x0 >= x1 or y0 >= y1:
        return hm
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if binary:
        hm[y0:y1, x0:x1] = d2 <= r * r
    else:
        v = np.exp(-d2 / (r * r)) / math.exp(-1.0) * cmin
        v[v < 0.5] = 0.0
        hm[y0:y1, x0:x1] = np.minimum(v, 1.0)
    return hm


@dataclass
class AugConfig:
    """Defaults = WASB-SBDT configs/transform/full.yaml; blur is an opt-in extra."""
    jitter_p: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.1
    hflip_p: float = 0.5
    crop_p: float = 0.5
    max_rescale: float = 0.125
    blur_p: float = 0.0


def augment(x: torch.Tensor, pts: np.ndarray, a: AugConfig, rnd: random.Random | None = None
            ) -> tuple[torch.Tensor, np.ndarray]:
    """x (F, 3, H, W) RGB in [0, 1] (background included); pts (S, 2) content positions in
    pixel-index coords (NaN = invisible). Photometric factors are drawn once per sample (as
    upstream). Geometry (WASB seq_transforms: hflip, x1.125 bilinear rescale + random crop) is
    applied to the images and, exactly, to the points; targets are drawn afterwards."""
    rnd = rnd or random
    pts = np.array(pts, np.float64, copy=True)
    if rnd.random() < a.jitter_p:
        b = rnd.uniform(max(0.0, 1 - a.brightness), 1 + a.brightness)
        c = rnd.uniform(max(0.0, 1 - a.contrast), 1 + a.contrast)
        s = rnd.uniform(max(0.0, 1 - a.saturation), 1 + a.saturation)
        h = rnd.uniform(-a.hue, a.hue)
        x = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(x, b), c), s), h)
    if a.blur_p > 0 and rnd.random() < a.blur_p:
        sig = rnd.uniform(0.3, 1.2)
        k = 2 * math.ceil(3 * sig) + 1
        x = TF.gaussian_blur(x, [k, k], [sig, sig])
    H, W = x.shape[-2:]
    if rnd.random() < a.hflip_p:
        x = x.flip(-1)
        pts[:, 0] = (W - 1) - pts[:, 0]
    if rnd.random() < a.crop_p:
        nw, nh = int(W * (1 + a.max_rescale)), int(H * (1 + a.max_rescale))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x1, y1 = int(round(rnd.uniform(0, nw - W))), int(round(rnd.uniform(0, nh - H)))
        x = x[..., y1:y1 + H, x1:x1 + W]
        # half-pixel (align_corners=False) resize: i -> (i + .5) * n / N - .5, then crop shift
        pts[:, 0] = (pts[:, 0] + 0.5) * (nw / W) - 0.5 - x1
        pts[:, 1] = (pts[:, 1] + 0.5) * (nh / H) - 0.5 - y1
    return x.clamp(0.0, 1.0).contiguous(), pts


# ══════════════════════════════════════════════════════════════════════════════
# Datasets / sampler
# ══════════════════════════════════════════════════════════════════════════════

def slide_starts(clips: list[CachedClip], S: int) -> list[tuple[int, int]]:
    """Every window start (training, sliding step 1 as both upstream trainers)."""
    return [(ci, s) for ci, c in enumerate(clips) for s in range(c.n - S + 1)]


def tile_starts(clips: list[CachedClip], S: int) -> list[tuple[int, int]]:
    """Non-overlapping windows covering every frame (WASB step=S); tail window re-anchored."""
    out = []
    for ci, c in enumerate(clips):
        if c.n < S:
            continue
        st = list(range(0, c.n - S + 1, S))
        if st[-1] + S < c.n:
            st.append(c.n - S)
        out += [(ci, s) for s in st]
    return out


def collate_np(batch: list[tuple]) -> tuple[np.ndarray, ...]:
    """Stack numpy items. Workers never hand torch tensors across processes: on macOS that
    starts torch_shm_manager daemons which inherit multiprocessing's resource_tracker pipe and
    make the main process hang at exit (observed with --workers 2)."""
    return tuple(np.stack(f) for f in zip(*batch))


def batch_to(x: np.ndarray, device: str) -> torch.Tensor:
    """uint8 (N, 3F, H, W) batch -> float [0, 1] on ``device`` (uint8 over the bus)."""
    return torch.from_numpy(x).to(device).float().div_(255.0)


class WindowDataset(Dataset):
    """Item = (frames uint8 (3F, H, W) RGB [bg first], targets float32 (S, H, W),
    int64 [clip, start]) as numpy; use with ``collate_np`` and ``batch_to``. Augmented frames
    are re-quantised to uint8 (upstream jitters PIL uint8 images too)."""

    def __init__(self, clips: list[CachedClip], arch: Arch, starts: list[tuple[int, int]],
                 aug: AugConfig | None = None, target_mode: str = "official", radius: float = 2.5,
                 cmin: float = 0.7):
        self.clips, self.arch, self.S, self.starts = clips, arch, arch.seq_len, starts
        self.aug, self.target_mode, self.radius, self.cmin = aug, target_mode, radius, cmin
        self.soft: set[tuple[int, int]] = set()     # HLSM frames (clip, frame)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        ci, s = self.starts[i]
        c = self.clips[ci]
        imgs = [c.frame_rgb(s + k) for k in range(self.S)]
        if self.arch.uses_background:
            imgs = [c.background_rgb()] + imgs
        x = np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2))      # (F, 3, H, W) uint8
        pts = np.full((self.S, 2), np.nan)
        for k in range(self.S):
            if c.labels[s + k, 0] > 0:
                pts[k] = c.content_xy(s + k)
        if self.aug is not None:
            xt, pts = augment(torch.from_numpy(x).float().div_(255.0), pts, self.aug)
            x = xt.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).numpy()
        hm = np.zeros((self.S, NET_H, NET_W), np.float32)
        for k in range(self.S):
            if np.isfinite(pts[k]).all():
                cx, cy = target_xy(pts[k], self.arch, self.target_mode)
                hm[k] = draw_target(cx, cy, self.radius, (ci, s + k) not in self.soft, self.cmin)
        return x.reshape(-1, NET_H, NET_W), hm, np.array([ci, s], np.int64)


class ResumableSampler(Sampler[int]):
    """Seeded per-epoch permutation (optionally truncated to ``epoch_size``) that can start
    mid-epoch, so a resumed run sees exactly the samples it had not seen yet."""

    def __init__(self, n: int, seed: int, epoch_size: int | None = None):
        self.n, self.seed, self.epoch_size = n, seed, epoch_size
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def full_len(self) -> int:
        return min(self.n, self.epoch_size) if self.epoch_size else self.n

    def order(self) -> list[int]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(self.n, generator=g).tolist()[: self.full_len()]

    def __iter__(self):
        return iter(self.order()[self.skip:])

    def __len__(self) -> int:
        return max(0, self.full_len() - self.skip)


# ══════════════════════════════════════════════════════════════════════════════
# Model, loss (upstream formulas), evaluation
# ══════════════════════════════════════════════════════════════════════════════

def hrnet_init(net: nn.Module) -> None:
    """Upstream HRNet init_weights: conv ~ N(0, 0.001), BN = (1, 0)."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def build_net(arch: Arch, init: str) -> tuple[nn.Module, str]:
    """'pretrained' (released checkpoint), 'scratch', or a checkpoint path (any format
    detector.load_checkpoint reads). Weights always load with strict=True."""
    D = det()
    if str(init) == "scratch":
        if arch.name == "wasb":
            net = D.WASBHRNet(arch.seq_len, arch.seq_len)
            hrnet_init(net)
        else:
            net = D.TrackNetV3(3 * arch.frames_in, arch.seq_len)   # PyTorch default init, as upstream
        return net, "scratch"
    path = arch.pretrained if str(init) == "pretrained" else Path(init)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    net, spec = D.load_model(path, arch.name)
    if (spec.seq_len, spec.uses_background) != (arch.seq_len, arch.uses_background):
        raise ValueError(f"{path}: seq_len/background {spec.seq_len}/{spec.uses_background} "
                         f"!= {arch.name} {arch.seq_len}/{arch.uses_background}")
    return net.train(), str(path)


class InputNorm(nn.Module):
    """0-1 RGB -> ImageNet mean/std (WASB; baked into the CoreML export) or identity."""

    def __init__(self, arch: Arch):
        super().__init__()
        self.on = arch.imagenet_norm
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).repeat(arch.frames_in).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).repeat(arch.frames_in).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std if self.on else x


class HeatmapLoss(nn.Module):
    """Upstream heatmap losses, mean over all pixels.
    from_logits (WASB): p = clamp(sigmoid(z), 1e-4, 1-1e-4)  (utils._sigmoid)
    else (TrackNetV3, net ends in sigmoid): logs clamped at 1e-7 (utils/metric.py WBCELoss)
      qfl   -|y-p|^beta [(1-y) log(1-p) + y log p]     quality_focal_loss.py, WASB eq. 3
      wbce  -[(1-p)^2 y log p + p^2 (1-y) log(1-p)]    wbce.py / TrackNetV2
      focal -[y (1-p)^g log p + (1-y) p^g log(1-p)]    focal_loss.py
      bce   -[y log p + (1-y) log(1-p)]"""

    def __init__(self, kind: str, from_logits: bool, gamma: float = 2.0, beta: float = 2.0):
        super().__init__()
        if kind not in ("qfl", "wbce", "focal", "bce"):
            raise ValueError(f"unknown loss {kind!r}")
        self.kind, self.from_logits, self.gamma, self.beta = kind, from_logits, gamma, beta

    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = out.float()
        if self.from_logits:
            p = torch.clamp(out.sigmoid(), 1e-4, 1 - 1e-4)
            lp, lq = torch.log(p), torch.log(1 - p)
        else:
            p = out
            lp, lq = torch.log(torch.clamp(p, 1e-7, 1)), torch.log(torch.clamp(1 - p, 1e-7, 1))
        if self.kind == "qfl":
            loss = (p - y).abs() ** self.beta * ((1 - y) * lq + y * lp)
        elif self.kind == "wbce":
            loss = (1 - p) ** 2 * y * lp + p ** 2 * (1 - y) * lq
        elif self.kind == "focal":
            loss = y * (1 - p) ** self.gamma * lp + (1 - y) * p ** self.gamma * lq
        else:
            loss = y * lp + (1 - y) * lq
        return -loss.mean()


def probs(net: nn.Module, norm: nn.Module, x: torch.Tensor, arch: Arch) -> torch.Tensor:
    out = net(norm(x)).float()
    return out.sigmoid() if arch.logits else out


@torch.no_grad()
def predict_frames(net: nn.Module, norm: nn.Module, ds: WindowDataset, device: str,
                   batch_size: int, workers: int, thr: float = 0.5) -> dict:
    """{(clip, frame): ((x, y) content net px, peak) | None}; first window covering a frame wins."""
    was_training = net.training
    net.eval()
    out: dict = {}
    off = -ds.arch.peak_offset
    for x, _, idx in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                                collate_fn=collate_np):
        y = probs(net, norm, batch_to(x, device), ds.arch).cpu().numpy()
        for b, (ci, s) in enumerate(idx.tolist()):
            for k in range(ds.S):
                if (ci, s + k) not in out:
                    out[(ci, s + k)] = blob_decode(y[b, k], thr, off)
    net.train(was_training)
    return out


def _eval_range(clip: CachedClip) -> tuple[int, int]:
    if not clip.held_out:
        return 0, clip.n
    match, name = clip.name.split("/", 1)
    return _test_drop_range(match, name, clip.n)


def score(dets: dict, clips: list[CachedClip], tol: float = 4.0) -> tuple[dict, set]:
    """TrackNet-style frame metrics at ``tol`` net px. Returns (metrics, FP1 frame set)."""
    tp = fp1 = fp2 = fn = tn = 0
    errs: list[float] = []
    fp1_keys: set[tuple[int, int]] = set()
    ranges = {ci: _eval_range(c) for ci, c in enumerate(clips)}
    for (ci, f), d in dets.items():
        lo, hi = ranges[ci]
        if not lo <= f < hi:
            continue
        c = clips[ci]
        vis = bool(c.labels[f, 0] > 0)
        if d is None:
            fn, tn = fn + vis, tn + (not vis)
        elif not vis:
            fp2 += 1
        else:
            dist = float(np.hypot(*(d[0] - c.content_xy(f))))
            errs.append(dist / c.geom.scale)
            if dist <= tol:
                tp += 1
            else:
                fp1 += 1
                fp1_keys.add((ci, f))
    p = tp / (tp + fp1 + fp2) if tp + fp1 + fp2 else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    n = tp + fp1 + fp2 + fn + tn
    return ({"frames": n, "tp": tp, "fp1": fp1, "fp2": fp2, "fn": fn, "tn": tn,
             "precision": round(p, 4), "recall": round(r, 4),
             "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
             "accuracy": round((tp + tn) / n, 4) if n else 0.0,
             "median_err_fullres_px": round(float(np.median(errs)), 3) if errs else None,
             "tol_net_px": tol}, fp1_keys)


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def _cache_all(srcs: list[ClipSource], a: argparse.Namespace, arch: Arch, what: str
               ) -> list[CachedClip]:
    out, t0 = [], time.time()
    for i, s in enumerate(srcs):
        out.append(cache_clip(s, Path(a.cache_dir), arch))
        if (i + 1) % 10 == 0 or i + 1 == len(srcs):
            log(f"[data] {what}: cached {i + 1}/{len(srcs)} clips ({time.time() - t0:.0f} s)")
    return out


def prepare_data(a: argparse.Namespace, arch: Arch, require_train: bool = True
                 ) -> tuple[list[CachedClip], list[CachedClip]]:
    """(train clips, val clips); never contains held-out Test clips."""
    srcs = discover(a.data_root, allow_held_out=False)
    if a.synthetic:
        syn = render_synthetic(a.synthetic, Path(a.cache_dir) / "synthetic_src", a.seed,
                               (a.synthetic_width, a.synthetic_height), a.synthetic_fps)
        if syn is not None:
            srcs += discover([syn], allow_held_out=False)
    is_syn = {s.name: s.name.startswith("sim_") for s in srcs}
    if a.val_root:
        val = discover(a.val_root, allow_held_out=False)
        vmedia = {v.media.resolve() for v in val}
        train = [s for s in srcs if s.media.resolve() not in vmedia]
    else:
        real = [s for s in srcs if not is_syn[s.name]]
        pool = sorted(real if len(real) >= 2 else srcs, key=lambda s: s.name)
        random.Random(a.seed).shuffle(pool)
        k = max(1, round(len(pool) * a.val_frac)) if len(pool) >= 2 and a.val_frac > 0 else 0
        val = pool[:k]
        train = [s for s in srcs if s not in val]
    if require_train and not train:
        raise SystemExit(
            "no training clips found (held-out Test clips are refused). Get the Shuttlecock "
            "Trajectory Dataset Train split (TrackNetV3 README -> TrackNetV2 dataset, layout "
            "<root>/Train/match*/{csv,video}) and pass --data-root, and/or --synthetic N.")
    log(f"[data] train clips {len(train)} (synthetic {sum(is_syn.get(s.name, False) for s in train)})"
        f", val clips {len(val)}: {[v.name for v in val][:8]}")
    return _cache_all(train, a, arch, "train"), _cache_all(val, a, arch, "val")


def test_clips(a: argparse.Namespace, arch: Arch) -> list[CachedClip]:
    """Held-out Test clips (evaluation only): --test-clips names or an even --test-max-clips
    spread."""
    srcs = [s for s in discover([a.test_root], allow_held_out=True) if s.held_out]
    if a.test_clips:
        want = set(a.test_clips)
        srcs = [s for s in srcs if s.name in want or s.name.split("/")[1] in want]
    elif a.test_max_clips and len(srcs) > a.test_max_clips:
        step = len(srcs) / a.test_max_clips
        srcs = [srcs[int(i * step)] for i in range(a.test_max_clips)]
    return _cache_all(srcs, a, arch, "test")


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoints
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunState:
    epoch: int = 0                 # next epoch to run (or the one in progress)
    step_in_epoch: int = 0         # batches of ``epoch`` already done
    global_step: int = 0
    best_f1: float = -1.0
    hlsm_done: bool = False
    hlsm_keys: list | None = None  # [(clip index, frame)] with real-valued targets


def weights_payload(net: nn.Module, arch: Arch) -> dict:
    """Official checkpoint layout, so detector.load_checkpoint and export_coreml read it."""
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    if arch.name == "wasb":
        return {"model_state_dict": sd}
    return {"model": sd, "param_dict": {"model_name": "TrackNet", "seq_len": arch.seq_len,
                                        "bg_mode": "concat" if arch.uses_background else ""}}


def _atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_last(path: Path, net, arch, opt, sched, st: RunState, a: argparse.Namespace) -> None:
    _atomic_save({**weights_payload(net, arch),
                  "optimizer_state_dict": opt.state_dict(),
                  "scheduler_state_dict": sched.state_dict(),
                  "run_state": vars(st).copy(), "args": dict(vars(a)),
                  "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                          "random": random.getstate()}}, path)


def save_best(path: Path, net, arch: Arch, a: argparse.Namespace, epoch: int, metrics: dict,
              init_desc: str) -> None:
    _atomic_save({**weights_payload(net, arch), "shuttl": {
        "arch": arch.name, "seq_len": arch.seq_len, "in_channels": 3 * arch.frames_in,
        "uses_background": arch.uses_background,
        "normalize": "imagenet" if arch.imagenet_norm else "0-1",
        "centroid_offset": -arch.peak_offset, "target_mode": a.target_mode, "radius": a.radius,
        "loss": a.loss, "epoch": epoch, "val": metrics, "init": init_desc,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S")}}, path)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def _pick_device(dev: str | None) -> str:
    if dev:
        return dev
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _evaluate(net, norm, clips, arch, a, device) -> dict | None:
    if not clips:
        return None
    ds = WindowDataset(clips, arch, tile_starts(clips, arch.seq_len))
    t0 = time.time()
    m, _ = score(predict_frames(net, norm, ds, device, a.eval_batch_size, a.workers), clips, a.tol)
    m["seconds"] = round(time.time() - t0, 1)
    _empty_cache(device)
    return m


def eval_test(net: nn.Module, arch: Arch, a: argparse.Namespace, device: str) -> dict:
    """Held-out Test split: reporting only, never used for selection or tuning."""
    clips = test_clips(a, arch)
    m = _evaluate(net.to(device), InputNorm(arch).to(device), clips, arch, a, device) or {}
    m["clips"] = [c.name for c in clips]
    m["note"] = "held-out TrackNetV2 Test (corrected labels, drop_frame ranges): report only"
    return m


def train(a: argparse.Namespace, resume_ck: dict | None = None) -> dict:
    arch = ARCHS[a.arch]
    check_conventions(arch)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = _pick_device(a.device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.jsonl", "a")

    def emit(rec: dict) -> None:
        line = json.dumps({"time": time.strftime("%H:%M:%S"), **rec}, default=str)
        logf.write(line + "\n")
        logf.flush()
        log(line)

    train_clips, val_clips = prepare_data(a, arch)
    net, init_desc = build_net(arch, "scratch" if resume_ck else a.init)
    if resume_ck:
        net.load_state_dict(resume_ck.get("model_state_dict") or resume_ck["model"], strict=True)
        init_desc = f"resumed ({a.init})"
    net.to(device)
    norm = InputNorm(arch).to(device)
    loss_fn = HeatmapLoss(a.loss, arch.logits, a.gamma, a.beta)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=a.milestones, gamma=0.1)
    aug = AugConfig(a.jitter_p, 0.1, 0.1, 0.1, 0.1, a.hflip_p, a.crop_p, 0.125, a.blur_p)
    train_ds = WindowDataset(train_clips, arch, slide_starts(train_clips, arch.seq_len), aug,
                             a.target_mode, a.radius, a.hlsm_cmin)
    sampler = ResumableSampler(len(train_ds), a.seed, a.epoch_size)
    st = RunState()
    if resume_ck:
        opt.load_state_dict(resume_ck["optimizer_state_dict"])
        sched.load_state_dict(resume_ck["scheduler_state_dict"])
        st = RunState(**resume_ck["run_state"])
        rng = resume_ck.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["random"])
        if st.hlsm_keys:
            train_ds.soft = {tuple(k) for k in st.hlsm_keys}
    batches_per_epoch = sampler.full_len() // a.batch_size
    if batches_per_epoch == 0:
        raise SystemExit(f"only {len(train_ds)} training windows < batch size {a.batch_size}")
    emit({"event": "start", "arch": arch.name, "device": device, "init": init_desc,
          "lr": a.lr, "milestones": a.milestones, "loss": a.loss, "target_mode": a.target_mode,
          "train_clips": len(train_clips), "train_windows": len(train_ds),
          "batches_per_epoch": batches_per_epoch, "val_clips": len(val_clips),
          "resume_at": [st.epoch, st.step_in_epoch, st.global_step]})

    losses: list[float] = []
    last_val: dict | None = None
    stopped = False
    t_log = time.time()
    while st.epoch < a.epochs:
        epoch = st.epoch
        if a.hlsm_epoch is not None and epoch >= a.hlsm_epoch and not st.hlsm_done:
            tds = WindowDataset(train_clips, arch, tile_starts(train_clips, arch.seq_len))
            _, hard = score(predict_frames(net, norm, tds, device, a.eval_batch_size, a.workers),
                            train_clips, a.tol)
            train_ds.soft = hard
            st.hlsm_done, st.hlsm_keys = True, sorted(hard)
            emit({"event": "hlsm", "epoch": epoch, "hard_frames": len(hard)})
            _empty_cache(device)
        sampler.set_epoch(epoch, st.step_in_epoch * a.batch_size)
        loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=sampler, drop_last=True,
                            num_workers=a.workers, collate_fn=collate_np)
        net.train()
        if a.freeze_bn:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        for x, y, _ in loader:
            x, y = batch_to(x, device), torch.from_numpy(y).to(device)
            if a.mixup_alpha > 0:       # TrackNetV3 train.py mixup (sample level)
                lam = np.random.beta(a.mixup_alpha, a.mixup_alpha, size=x.shape[0])
                lam = torch.from_numpy(np.maximum(lam, 1 - lam)).float().to(device).view(-1, 1, 1, 1)
                perm = torch.randperm(x.shape[0]).to(device)
                x, y = x * lam + x[perm] * (1 - lam), y * lam + y[perm] * (1 - lam)
            loss = loss_fn(net(norm(x)), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {st.global_step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            st.step_in_epoch += 1
            st.global_step += 1
            losses.append(float(loss))
            if st.global_step % a.log_every == 0:
                rec = {"event": "iter", "epoch": epoch, "step": st.global_step,
                       "loss": round(float(np.mean(losses[-a.log_every:])), 6),
                       "lr": opt.param_groups[0]["lr"],
                       "s_per_iter": round((time.time() - t_log) / a.log_every, 3)}
                t_log = time.time()
                if device == "mps":
                    rec["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
                emit(rec)
            if a.save_every and st.global_step % a.save_every == 0:
                save_last(out_dir / "last.pt", net, arch, opt, sched, st, a)
            if a.max_iters and st.global_step >= a.max_iters:
                stopped = True
                break
        if stopped and st.step_in_epoch < batches_per_epoch:   # mid-epoch stop: keep position
            save_last(out_dir / "last.pt", net, arch, opt, sched, st, a)
            emit({"event": "stopped", "epoch": epoch, "step_in_epoch": st.step_in_epoch,
                  "step": st.global_step})
            break
        # ── epoch end ──
        sched.step()
        st.epoch, st.step_in_epoch = epoch + 1, 0
        if val_clips and (st.epoch % a.eval_every == 0 or st.epoch == a.epochs):
            last_val = _evaluate(net, norm, val_clips, arch, a, device)
            improved = last_val["f1"] > st.best_f1
            if improved:
                st.best_f1 = last_val["f1"]
                save_best(out_dir / "best.pth.tar", net, arch, a, st.epoch, last_val, init_desc)
            emit({"event": "val", "epoch": st.epoch, "step": st.global_step, **last_val,
                  "best": improved})
        elif not val_clips:
            save_best(out_dir / "best.pth.tar", net, arch, a, st.epoch, {}, init_desc)
        save_last(out_dir / "last.pt", net, arch, opt, sched, st, a)
        if stopped:
            break

    if stopped and val_clips:
        last_val = _evaluate(net, norm, val_clips, arch, a, device)
        emit({"event": "val", "epoch": st.epoch, "step": st.global_step, "partial": True,
              **last_val})
    summary = {"arch": arch.name, "global_step": st.global_step, "epoch": st.epoch,
               "step_in_epoch": st.step_in_epoch, "best_val_f1": st.best_f1,
               "losses": [round(v, 6) for v in losses], "val": last_val, "device": device,
               "out_dir": str(out_dir)}
    best = out_dir / "best.pth.tar"
    if a.test_eval:
        tnet = build_net(arch, str(best))[0] if best.is_file() else net
        summary["test_heldout"] = eval_test(tnet, arch, a, device)
        emit({"event": "test_heldout", **summary["test_heldout"]})
    if a.install and best.is_file():
        shutil.copy2(best, arch.install)
        emit({"event": "install", "path": str(arch.install)})
    logf.close()
    return summary


def evaluate_only(a: argparse.Namespace) -> dict:
    arch = ARCHS[a.arch]
    check_conventions(arch)
    device = _pick_device(a.device)
    net, desc = build_net(arch, a.init)
    net.to(device).eval()
    res: dict = {"arch": arch.name, "weights": desc, "device": device}
    _, val = prepare_data(a, arch, require_train=False)
    res["val"] = _evaluate(net, InputNorm(arch).to(device), val, arch, a, device)
    if a.test_eval:
        res["test_heldout"] = eval_test(net, arch, a, device)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test: fabricated clips (+ one sim render), 10 iters, resume, 10 more
# ══════════════════════════════════════════════════════════════════════════════

def fabricate_clip(match_dir: Path, clip: str, size: tuple[int, int], n: int,
                   rng: np.random.Generator, fps: float = 30.0) -> None:
    """Textured court-ish background + lines + a white 'shuttle' on a parabola (some frames
    hidden), written in TrackNetV3 format."""
    w, h = size
    (match_dir / "video").mkdir(parents=True, exist_ok=True)
    (match_dir / "csv").mkdir(parents=True, exist_ok=True)
    base = cv2.resize(rng.random((h // 24 + 2, w // 24 + 2, 3)).astype(np.float32),
                      (w, h), interpolation=cv2.INTER_CUBIC)
    base = np.clip(base * 110 + 50, 0, 255).astype(np.uint8)
    for _ in range(5):
        p = rng.integers(0, [w, h, w, h])
        cv2.line(base, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (210, 210, 210), 2,
                 cv2.LINE_AA)
    vw = cv2.VideoWriter(str(match_dir / "video" / f"{clip}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    x0, y0 = rng.uniform(0.1, 0.3) * w, rng.uniform(0.35, 0.6) * h
    vx, vy, g = rng.uniform(0.012, 0.02) * w, -0.03 * h, 0.0016 * h
    rad = max(2.0, 0.006 * w)
    rows = []
    for f in range(n):
        x, y = x0 + vx * f, y0 + vy * f + 0.5 * g * f * f
        vis = 0 <= x < w and 0 <= y < h and f % 13 != 6
        img = np.clip(base.astype(np.int16) + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
        if vis:
            cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), int(rad * 16),
                       (245, 245, 245), -1, cv2.LINE_AA, shift=4)
        vw.write(img)
        rows.append(f"{f},{int(vis)},{x if vis else 0:.3f},{y if vis else 0:.3f}")
    vw.release()
    (match_dir / "csv" / f"{clip}_ball.csv").write_text("Frame,Visibility,X,Y\n" + "\n".join(rows) + "\n")


def convention_checks(arch: Arch, n: int = 200) -> dict:
    """Target drawing vs the shipped decode (blob centroid - peak_offset), and the augmentation
    point mapping vs the warped image content. No model involved.

    official_*: 1280x720 labels, upstream-quantised targets decoded with the shipped offset
    (mean bias should be ~0, |err| ~0.25 px from the integer grid). subpixel_*: continuous
    targets. aug_*: a Gaussian dot at a known content position, flipped/rescaled/cropped; its
    intensity centroid vs the mapped point (net px)."""
    rng = np.random.default_rng(1)
    clip = CachedClip("x/y", Path("."), 1, 1280, 720, 30.0, np.zeros((1, 3)), False, arch.corner)
    res: dict = {}
    for mode in ("official", "subpixel"):
        e = []
        for _ in range(n):
            clip.labels[0] = (1, rng.uniform(50, 1230), rng.uniform(50, 670))
            content = clip.content_xy(0)
            d = blob_decode(draw_target(*target_xy(content, arch, mode)), 0.5, -arch.peak_offset)
            e.append(d[0] - content)
        e = np.asarray(e)
        res[f"{mode}_mean_bias_net_px"] = [round(float(v), 4) for v in e.mean(0)]
        res[f"{mode}_mean_abs_err_net_px"] = round(float(np.abs(e).mean()), 4)
    errs = []
    yy, xx = np.mgrid[0:NET_H, 0:NET_W]
    for i in range(24):
        p = np.array([[rng.uniform(60, NET_W - 60), rng.uniform(40, NET_H - 40)]])
        img = np.exp(-((xx - p[0, 0]) ** 2 + (yy - p[0, 1]) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
        x = torch.from_numpy(img)[None, None].repeat(1, 3, 1, 1)
        cfg = AugConfig(jitter_p=0.0, hflip_p=float(i % 2), crop_p=float(i % 3 != 0))
        xa, pa = augment(x, p, cfg, random.Random(i))
        im = xa[0, 0].numpy().astype(np.float64)
        im[im < 0.05] = 0
        c = np.array([(im * xx).sum() / im.sum(), (im * yy).sum() / im.sum()])
        errs.append(float(np.hypot(*(c - pa[0]))))
    res["aug_point_max_err_net_px"] = round(max(errs), 4)
    return res


def smoke(a: argparse.Namespace) -> dict:
    arch = ARCHS[a.arch]
    tmp = Path(tempfile.mkdtemp(prefix="shuttl_train_smoke_"))
    t0 = time.time()
    try:
        rng = np.random.default_rng(0)
        data = tmp / "data"
        n = 40
        fabricate_clip(data / "Train" / "match1", "c0", (640, 360), n, rng)
        fabricate_clip(data / "Train" / "match1", "c1", (640, 360), n, rng)
        fabricate_clip(data / "Train" / "match2", "c2", (480, 360), n, rng)      # 4:3 letterbox
        fabricate_clip(data / "Test" / "match1", "t0", (640, 360), n, rng)       # must be refused
        found = discover([data], allow_held_out=False)
        assert [c.name for c in found] == ["match1/c0", "match1/c1", "match2/c2"], found
        common = ["--arch", arch.name, "--data-root", str(data), "--cache-dir", str(tmp / "cache"),
                  "--out-dir", str(tmp / "run"), "--val-frac", "0.34", "--batch-size", "2",
                  "--eval-batch-size", "2", "--workers", "0", "--epochs", "3",
                  "--epoch-size", "16", "--hlsm-epoch", "2", "--log-every", "5", "--seed", "0",
                  "--synthetic", str(a.smoke_synthetic), "--synthetic-width", "640",
                  "--synthetic-height", "360"] + (["--device", a.device] if a.device else [])
        s1 = main(common + ["--max-iters", "10"])
        s2 = main(["--resume", str(tmp / "run" / "last.pt"), "--max-iters", "20"])
        best = tmp / "run" / "best.pth.tar"
        net, spec = det().load_model(best, arch.name)          # strict=True
        meta = torch.load(best, map_location="cpu", weights_only=False)["shuttl"]
        losses = s1["losses"] + s2["losses"]
        rep = {"arch": arch.name, "iterations": len(losses), "phase1_steps": s1["global_step"],
               "resumed_at": [s1["epoch"], s1["step_in_epoch"]], "final_step": s2["global_step"],
               "loss_first5": round(float(np.mean(losses[:5])), 6),
               "loss_last5": round(float(np.mean(losses[-5:])), 6),
               "all_finite": bool(np.isfinite(losses).all()), "val": s2["val"],
               "best_ckpt_strict_load": spec.name, "best_meta_centroid_offset": meta["centroid_offset"],
               "held_out_refused": True, "conventions": convention_checks(arch),
               "seconds": round(time.time() - t0, 1)}
        assert rep["iterations"] == 20 and rep["final_step"] == 20 and rep["all_finite"], rep
        assert s1["global_step"] == 10 and s2["global_step"] == 20
        return rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def make_parser(suppress: bool = False) -> argparse.ArgumentParser:
    """``suppress``: a twin parser that only reports explicitly given options (for --resume)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    def add(*names, **kw):
        if suppress:
            kw["default"] = argparse.SUPPRESS
        ap.add_argument(*names, **kw)

    add("--arch", choices=sorted(ARCHS), default="wasb")
    add("--init", default="pretrained", help="'pretrained' (released weights), 'scratch' or a path")
    add("--data-root", nargs="+", default=[str(DEFAULT_DATA)])
    add("--val-root", nargs="+", default=None)
    add("--val-frac", type=float, default=0.1)
    add("--synthetic", type=int, default=0, metavar="N", help="render N sim.py rallies")
    add("--synthetic-width", type=int, default=1280)
    add("--synthetic-height", type=int, default=720)
    add("--synthetic-fps", type=float, default=30.0)
    add("--cache-dir", default=str(DEFAULT_CACHE))
    add("--out-dir", default=None)
    add("--device", default=None, help="mps | cuda | cpu (default: best available)")
    add("--epochs", type=int, default=None, help="default 30 from scratch, 10 fine-tuning")
    add("--epoch-size", type=int, default=None, help="windows per epoch (default: all)")
    add("--max-iters", type=int, default=0, help="stop (resumably) after this many steps")
    add("--batch-size", type=int, default=None)
    add("--eval-batch-size", type=int, default=4)
    add("--workers", type=int, default=2)
    add("--lr", type=float, default=None, help="default 1e-3 scratch, 1e-4 fine-tune")
    add("--milestones", type=int, nargs="*", default=None)
    add("--weight-decay", type=float, default=0.0)
    add("--loss", choices=("qfl", "wbce", "focal", "bce"), default=None)
    add("--gamma", type=float, default=2.0)
    add("--beta", type=float, default=2.0)
    add("--target-mode", choices=("official", "subpixel"), default="official")
    add("--radius", type=float, default=2.5)
    add("--hlsm-epoch", type=int, default=None, help="WASB hard-to-localise mining epoch")
    add("--hlsm-cmin", type=float, default=0.7)
    add("--jitter-p", type=float, default=0.5)
    add("--hflip-p", type=float, default=0.5)
    add("--crop-p", type=float, default=0.5)
    add("--blur-p", type=float, default=0.0)
    add("--mixup-alpha", type=float, default=None)
    add("--freeze-bn", action="store_true")
    add("--tol", type=float, default=4.0, help="TP distance in 512x288 net px")
    add("--eval-every", type=int, default=1)
    add("--save-every", type=int, default=500)
    add("--log-every", type=int, default=20)
    add("--seed", type=int, default=0)
    add("--resume", default=None, help="last.pt of a previous run")
    add("--eval-only", action="store_true")
    add("--test-eval", action="store_true", help="score the held-out Test split at the end")
    add("--test-root", default=str(DEFAULT_TEST))
    add("--test-clips", nargs="*", default=None)
    add("--test-max-clips", type=int, default=0)
    add("--install", action="store_true", help="copy best.pth.tar where detector.py finds it")
    add("--smoke", action="store_true", help="20-iteration self-test on fabricated clips")
    add("--smoke-synthetic", type=int, default=1)
    return ap


def resolve_defaults(a: argparse.Namespace) -> argparse.Namespace:
    arch = ARCHS[a.arch]
    scratch = str(a.init) == "scratch"
    a.epochs = a.epochs if a.epochs is not None else (30 if scratch else 10)
    a.batch_size = a.batch_size or arch.batch_size
    a.lr = a.lr if a.lr is not None else (1e-3 if scratch else 1e-4)
    if a.milestones is None:
        a.milestones = ([10, 20] if scratch and a.epochs == 30 else
                        [max(1, a.epochs // 2), max(1, (3 * a.epochs) // 4)])
    a.loss = a.loss or arch.loss
    a.mixup_alpha = arch.mixup_alpha if a.mixup_alpha is None else a.mixup_alpha
    a.out_dir = a.out_dir or str(ML_DIR / "runs" / f"{arch.name}_ft")
    return a


def main(argv: list[str] | None = None) -> dict:
    a = make_parser().parse_args(argv)
    if a.smoke:
        rep = smoke(a)
        print(json.dumps({"smoke": rep}, indent=2, default=str))
        return rep
    resume_ck = None
    if a.resume:
        resume_ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        explicit = vars(make_parser(suppress=True).parse_args(argv))
        a = argparse.Namespace(**{**vars(a), **resume_ck["args"], **explicit})
    resolve_defaults(a)
    if a.eval_only:
        res = evaluate_only(a)
        print(json.dumps(res, indent=2, default=str))
        return res
    return train(a, resume_ck)


if __name__ == "__main__":
    main()
