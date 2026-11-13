#!/usr/bin/env python3
"""Export the on-device shuttle detector (WASB HRNet by default) to a CoreML .mlpackage.

iOS CONTRACT (the exported graph bakes in normalisation and the sigmoid):
  input  "frames"  MLMultiArray Float32 (1, C, H, W), H=288, W=512.
         C = 3 * (seq_len + uses_background). Channel blocks of 3 = one RGB frame in [0, 1]
         (plain /255, no mean/std), ordered [background (if used)], oldest ... newest.
  output "heatmap" MLMultiArray Float32 (1, S, H, W), S = seq_len; values are probabilities
         in [0, 1]; heatmap[0, i] belongs to input frame i (same order as the input).
  Geometry: the full frame is aspect-fitted into WxH (scale s = min(W/fw, H/fh)), centred,
  black padding. Continuous-coordinate affine (WASB convention, pixel corners):
        net = full * s + pad,   pad = ((W - fw*s)/2, (H - fh*s)/2)
        full = (net - pad) / s
  Detection: shuttle present if heatmap peak >= score_threshold (0.5); position = heatmap-
  weighted centroid of the largest-mass connected blob (>= threshold), plus centroid_offset
  (0.5 px for WASB, both axes), then mapped back with the inverse affine above.

Metadata (``user_defined``): seq_len, in_channels, uses_background, input_order, normalize,
input_width, input_height, output_activation, score_threshold, centroid_offset, frame_order,
resize, model.

Usage (from backend/):
    .venv/bin/python ml/export_coreml.py                      # WASB, fp16, validate + bench
    .venv/bin/python ml/export_coreml.py --model tracknetv3 --output /tmp/tnv3.mlpackage
    .venv/bin/python ml/export_coreml.py --no-validate --precision fp32
    .venv/bin/python ml/export_coreml.py --weights ml/runs/wasb_ft/best.pth.tar --output X.mlpackage

Requires coremltools >= 8.1 with native libs (8.0 in this venv was a pure-python build that
cannot save ML programs or run predictions).
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

WEIGHTS_DIR = BACKEND / "ml" / "weights"
DEFAULT_OUTPUT = BACKEND.parent / "iOS" / "Resources" / "Models" / "ShuttleTracker.mlpackage"
TEST_ROOT = BACKEND / "ml" / "data" / "raw" / "shuttlecock_trajectory"
NET_W, NET_H = 512, 288
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# Model cards + loading (architectures are imported from the detector module)
# ──────────────────────────────────────────────────────────────────────────────

def import_detector(retries: int = 6, wait_s: float = 10.0):
    """Import app.services.tracking.detector, retrying while a sibling is mid-edit."""
    err: BaseException | None = None
    for i in range(retries):
        try:
            mod = importlib.import_module("app.services.tracking.detector")
            if hasattr(mod, "WASBHRNet") and hasattr(mod, "TrackNetV3"):
                return mod
            err = ImportError("detector module lacks WASBHRNet/TrackNetV3")
        except Exception as e:  # noqa: BLE001 - file may be half-written
            err = e
        sys.modules.pop("app.services.tracking.detector", None)
        if i + 1 < retries:
            time.sleep(wait_s)
    raise ImportError(f"cannot import detector architectures: {err!r}")


@dataclass(frozen=True)
class ModelCard:
    """What the exported graph expects/produces (drives the metadata contract)."""
    kind: str
    seq_len: int
    uses_background: bool
    imagenet_norm: bool      # net was trained on ImageNet mean/std input (else plain 0-1)
    logits: bool             # net emits logits (sigmoid gets baked in)
    weights: str             # relative to WEIGHTS_DIR
    state_key: str
    licence: str
    description: str
    # Added to the decoded blob centroid (net px) before the inverse affine. Derived from the
    # training code, not tuned: WASB draws its target disc at int(label * s + pad), i.e. the
    # net learns to peak ~0.5 px below the true position on average.
    centroid_offset: float = 0.0

    @property
    def in_channels(self) -> int:
        return 3 * (self.seq_len + int(self.uses_background))


CARDS: dict[str, ModelCard] = {
    "wasb": ModelCard(
        "wasb", 3, False, True, True, "wasb_badminton_best.pth.tar", "model_state_dict",
        "MIT (code and checkpoints), NTT Communications - github.com/nttcom/WASB-SBDT",
        "WASB-SBDT HRNet (BMVC 2023) badminton weights: 3 RGB frames -> 3 heatmaps", 0.5),
    "tracknetv3": ModelCard(
        "tracknetv3", 8, True, False, False, "ckpts/TrackNet_best.pt", "model",
        "MIT, qaz812345 - github.com/qaz812345/TrackNetV3",
        "TrackNetV3 tracking module: median background + 8 RGB frames -> 8 heatmaps"),
}


def load_net(card: ModelCard, weights_dir: Path = WEIGHTS_DIR,
             weights_file: Path | None = None) -> nn.Module:
    """Build the architecture and load the official (or a ml/train_detector.py fine-tuned,
    same format) checkpoint with strict=True."""
    det = import_detector()
    path = Path(weights_file) if weights_file else weights_dir / card.weights
    if not path.is_file():
        raise FileNotFoundError(f"weights not found: {path} (refusing to export random init)")
    net = det.WASBHRNet() if card.kind == "wasb" else det.TrackNetV3()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get(card.state_key, ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net.eval()


class OnDeviceWrapper(nn.Module):
    """Plain 0-1 RGB in, probability heatmaps out: normalisation + sigmoid inside the graph.

    Mean/std is applied as an explicit per-channel affine (not folded into conv1) so zero
    padding at image borders behaves exactly as in training.
    """

    def __init__(self, net: nn.Module, card: ModelCard):
        super().__init__()
        self.net = net
        self.logits = card.logits
        n = card.seq_len + int(card.uses_background)
        if card.imagenet_norm:
            mean = torch.tensor(IMAGENET_MEAN).repeat(n)
            inv_std = 1.0 / torch.tensor(IMAGENET_STD).repeat(n)
        else:
            mean, inv_std = torch.zeros(3 * n), torch.ones(3 * n)
        self.norm = card.imagenet_norm
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("inv_std", inv_std.view(1, -1, 1, 1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = (frames - self.mean) * self.inv_std if self.norm else frames
        y = self.net(x)
        return torch.sigmoid(y) if self.logits else y


def contract_metadata(card: ModelCard) -> dict[str, str]:
    """``user_defined`` metadata read by the iOS inference code (all values strings)."""
    return {
        "model": card.kind,
        "seq_len": str(card.seq_len),
        "in_channels": str(card.in_channels),
        "uses_background": "1" if card.uses_background else "0",
        "input_order": "rgb",
        "normalize": "0-1",
        "input_width": str(NET_W),
        "input_height": str(NET_H),
        "frame_order": ("background," if card.uses_background else "") + "oldest_to_newest",
        "output_activation": "sigmoid",
        "score_threshold": "0.5",
        "centroid_offset": f"{card.centroid_offset:g}",
        "resize": "aspect_fit_center_black_pad",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────

def trace(card: ModelCard, weights_dir: Path = WEIGHTS_DIR, weights_file: Path | None = None
          ) -> tuple[torch.jit.ScriptModule, nn.Module]:
    wrapper = OnDeviceWrapper(load_net(card, weights_dir, weights_file), card).eval()
    example = torch.rand(1, card.in_channels, NET_H, NET_W)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    return traced, wrapper


def convert(traced: torch.jit.ScriptModule, card: ModelCard, precision: str = "fp16"):
    """TorchScript → ML program (iOS 17). ``precision`` = compute/storage precision."""
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="frames", shape=(1, card.in_channels, NET_H, NET_W),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name="heatmap", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=(ct.precision.FLOAT16 if precision == "fp16"
                           else ct.precision.FLOAT32),
        compute_units=ct.ComputeUnit.ALL,
    )
    bg = "median background RGB, then " if card.uses_background else ""
    mlmodel.author = "Shuttl (weights: " + card.licence + ")"
    mlmodel.license = card.licence
    mlmodel.version = "2.0.0"
    mlmodel.short_description = (
        f"{card.description}. Normalisation and sigmoid are baked in; feed plain 0-1 RGB.")
    mlmodel.input_description["frames"] = (
        f"(1,{card.in_channels},{NET_H},{NET_W}) float: {bg}{card.seq_len} RGB frames oldest->"
        f"newest, each 3 channels R,G,B in [0,1], aspect-fit into {NET_W}x{NET_H}, black pad.")
    mlmodel.output_description["heatmap"] = (
        f"(1,{card.seq_len},{NET_H},{NET_W}) float probabilities; channel i = input frame i. "
        "Shuttle present if peak >= 0.5.")
    mlmodel.user_defined_metadata.update(contract_metadata(card))
    return mlmodel


def save(mlmodel, out: Path) -> float:
    """Save (replacing any previous package); returns package size in MB."""
    out = Path(out)
    if out.suffix != ".mlpackage":
        raise ValueError("output must end with .mlpackage")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    return package_mb(out)


def package_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


# ──────────────────────────────────────────────────────────────────────────────
# Reference preprocessing / decoding (what the iOS side must reproduce)
# ──────────────────────────────────────────────────────────────────────────────

def fit_affine(fw: int, fh: int, net_w: int = NET_W, net_h: int = NET_H) -> np.ndarray:
    """2x3 full→net affine: aspect fit, centred, corner convention (== WASB get_transform)."""
    s = min(net_w / fw, net_h / fh)
    return np.array([[s, 0.0, (net_w - fw * s) / 2.0], [0.0, s, (net_h - fh * s) / 2.0]])


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 frame → (3, H, W) float32 RGB in [0,1] on the letterboxed net canvas."""
    h, w = frame_bgr.shape[:2]
    img = cv2.warpAffine(frame_bgr, fit_affine(w, h), (NET_W, NET_H), flags=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1),
                                dtype=np.float32) / 255.0


def to_full(xy_net: np.ndarray, fw: int, fh: int) -> np.ndarray:
    m = fit_affine(fw, fh)
    return (np.asarray(xy_net, np.float64) - m[:, 2]) / m[0, 0]


def blob_decode(hm: np.ndarray, thr: float = 0.5, offset: float = 0.0
                ) -> tuple[np.ndarray, float] | None:
    """WASB 'concomp' decode: threshold, connected components, heatmap-weighted centroid of
    the blob with the largest mass (+ ``offset``). Returns ((x, y) net px, peak) or None."""
    if float(hm.max()) < thr:
        return None
    n, lab = cv2.connectedComponents((hm >= thr).astype(np.uint8))
    best, best_mass = None, -1.0
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        w = hm[ys, xs].astype(np.float64)
        if w.sum() > best_mass:
            best_mass = float(w.sum())
            best = (np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()]) + offset,
                    float(w.max()))
    return best


@dataclass
class Window:
    clip: str
    start: int
    frames: np.ndarray               # (C, H, W) float32 model input
    gt: list[tuple[int, float, float]]   # per seq frame: (visibility, x, y) full-res
    size: tuple[int, int]            # (fw, fh)


def _labels(match: str, clip: str) -> np.ndarray | None:
    for p in (TEST_ROOT / "corrected" / match / "corrected_csv" / f"{clip}_ball.csv",
              TEST_ROOT / "TrackNetV2" / "Test" / match / "csv" / f"{clip}_ball.csv"):
        if p.is_file():
            return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    return None


def real_windows(card: ModelCard, n_clips: int = 6, per_clip: int = 4,
                 seed: int = 0) -> list[Window]:
    """Model inputs cut from held-out TrackNetV2 Test clips (evaluation only, never training)."""
    vids = sorted((TEST_ROOT / "TrackNetV2" / "Test").glob("match*/video/*.mp4"))
    if not vids:
        return []
    rng = np.random.default_rng(seed)
    # spread across matches: round-robin over match dirs
    by_match: dict[str, list[Path]] = {}
    for v in vids:
        by_match.setdefault(v.parts[-3], []).append(v)
    picked: list[Path] = []
    while len(picked) < n_clips and any(by_match.values()):
        for m in sorted(by_match):
            if by_match[m] and len(picked) < n_clips:
                picked.append(by_match[m].pop(int(rng.integers(len(by_match[m])))))
    S, out = card.seq_len, []
    for v in picked:
        lab = _labels(v.parts[-3], v.stem)
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if lab is None or n < S + 1:
            cap.release()
            continue
        starts = sorted(int(s) for s in rng.choice(np.arange(0, n - S), per_clip, replace=False))
        need = {s + i for s in starts for i in range(S)}
        frames: dict[int, np.ndarray] = {}
        thumbs: list[np.ndarray] = []
        fw = fh = 0
        for idx in range(n):
            ok, f = cap.read()
            if not ok:
                break
            fh, fw = f.shape[:2]
            if idx in need:
                frames[idx] = preprocess(f)
            if card.uses_background and idx % 15 == 0:
                thumbs.append(preprocess(f))
            if idx > max(need) and not card.uses_background:
                break
        cap.release()
        bg = np.median(np.stack(thumbs), axis=0).astype(np.float32) if thumbs else None
        rows = {int(r[0]): (int(r[1]), float(r[2]), float(r[3])) for r in lab}
        for s in starts:
            if any(s + i not in frames for i in range(S)):
                continue
            seq = [frames[s + i] for i in range(S)]
            if card.uses_background:
                seq = [bg] + seq
            out.append(Window(v.stem, s, np.concatenate(seq, 0),
                              [rows.get(s + i, (0, 0.0, 0.0)) for i in range(S)], (fw, fh)))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Validation (numeric parity vs PyTorch fp32 + sanity vs ground truth) and benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def gt_score(hms: list[np.ndarray], wins: list[Window], tol: float = 4.0,
             offset: float = 0.0) -> dict:
    """TrackNet-style TP/FP/FN at ``tol`` px in the 512x288 net space (TrackNetV3 standard)."""
    tp = fp = fn = tn = 0
    errs = []
    for hm, w in zip(hms, wins):
        m = fit_affine(*w.size)
        for s, (vis, gx, gy) in enumerate(w.gt):
            det = blob_decode(hm[s], offset=offset)
            if det is None:
                fn += vis > 0
                tn += vis == 0
                continue
            if vis == 0:
                fp += 1
                continue
            g = np.array([gx, gy]) * m[0, 0] + m[:, 2]
            d = float(np.hypot(*(det[0] - g)))
            errs.append(d / m[0, 0])
            tp, fp = (tp + 1, fp) if d <= tol else (tp, fp + 1)
    res = {"frames": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn, **_prf(tp, fp, fn)}
    res["median_err_fullres_px"] = round(float(np.median(errs)), 3) if errs else None
    return res


def parity(ref: list[np.ndarray], got: list[np.ndarray]) -> dict:
    """Heatmap difference + peak/decode agreement between two runs over the same windows."""
    diff = np.concatenate([np.abs(a - b).ravel() for a, b in zip(ref, got)])
    same_argmax = n_peak = agree = both = 0
    dists = []
    for a, b in zip(ref, got):
        for s in range(a.shape[0]):
            ha, hb = a[s], b[s]
            if ha.max() >= 0.5:
                n_peak += 1
                same_argmax += int(np.argmax(ha) == np.argmax(hb))
            da, db = blob_decode(ha), blob_decode(hb)
            agree += int((da is None) == (db is None))
            if da is not None and db is not None:
                both += 1
                dists.append(float(np.hypot(*(da[0] - db[0]))))
    n_hm = sum(a.shape[0] for a in ref)
    return {
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "p99_9_abs_diff": float(np.quantile(diff, 0.999)),
        "argmax_equal_on_peaks": f"{same_argmax}/{n_peak}",
        "detection_agreement": f"{agree}/{n_hm}",
        "decoded_xy_max_dist_net_px": round(max(dists), 4) if dists else None,
        "decoded_xy_mean_dist_net_px": round(float(np.mean(dists)), 5) if dists else None,
        "both_detected": both,
    }


def _predict(ml, frames: np.ndarray) -> np.ndarray:
    return np.asarray(ml.predict({"frames": frames[None].astype(np.float32)})["heatmap"])[0]


def validate(wrapper: nn.Module, card: ModelCard, packages: dict[str, tuple[Path, str]],
             n_clips: int = 6, per_clip: int = 4) -> dict:
    """``packages``: label -> (mlpackage path, compute unit name)."""
    import coremltools as ct

    wins = real_windows(card, n_clips, per_clip)
    source = "TrackNetV2 Test split (held out, eval only)"
    if not wins:   # no data on disk: fall back to noise (weak check, flagged in the report)
        rng = np.random.default_rng(0)
        wins = [Window("random", 0, rng.random((card.in_channels, NET_H, NET_W), np.float32),
                       [(0, 0.0, 0.0)] * card.seq_len, (NET_W, NET_H)) for _ in range(4)]
        source = "random noise (no real clips found)"
    with torch.no_grad():
        ref = [wrapper(torch.from_numpy(w.frames)[None]).numpy()[0] for w in wins]
    report: dict = {"data": source, "windows": len(wins),
                    "clips": sorted({w.clip for w in wins}),
                    "heatmaps": sum(r.shape[0] for r in ref),
                    "pytorch_fp32_vs_gt": gt_score(ref, wins, offset=card.centroid_offset)}
    for label, (path, cu) in packages.items():
        ml = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, cu))
        got = [_predict(ml, w.frames) for w in wins]
        report[label] = {"compute_units": cu, **parity(ref, got),
                         "vs_gt": gt_score(got, wins, offset=card.centroid_offset)}
        del ml
    return report


def _drop_range(match: str, clip: str, n: int) -> tuple[int, int]:
    """TrackNetV3's corrected/drop_frame.json evaluation range [start, end) for a clip."""
    p = TEST_ROOT / "corrected" / "drop_frame.json"
    if p.is_file():
        d = json.loads(p.read_text())
        key = f"{match.removeprefix('match')}_{clip}"
        if key in d.get("start", {}):
            return int(d["start"][key]), int(d["end"][key])
    return 0, n


def clip_eval(path: Path, card: ModelCard, n_clips: int = 6, cu: str = "ALL",
              seed: int = 0) -> dict:
    """Run an exported package over WHOLE held-out Test clips with non-overlapping windows
    (stride = seq_len, tail window re-anchored to the last frames) and score every labelled
    frame in the drop_frame range. Background (TrackNetV3) = median of every 15th frame."""
    import coremltools as ct

    vids = sorted((TEST_ROOT / "TrackNetV2" / "Test").glob("match*/video/*.mp4"))
    rng = np.random.default_rng(seed)
    by_match: dict[str, list[Path]] = {}
    for v in vids:
        by_match.setdefault(v.parts[-3], []).append(v)
    picked: list[Path] = []
    while len(picked) < n_clips and any(by_match.values()):
        for m in sorted(by_match):
            if by_match[m] and len(picked) < n_clips:
                picked.append(by_match[m].pop(int(rng.integers(len(by_match[m])))))
    ml = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, cu))
    S = card.seq_len
    tp = fp = fn = tn = 0
    errs: list[float] = []
    t_pred = 0.0
    n_pred = 0
    for v in picked:
        lab = _labels(v.parts[-3], v.stem)
        if lab is None:
            continue
        rows = {int(r[0]): (int(r[1]), float(r[2]), float(r[3])) for r in lab}
        cap = cv2.VideoCapture(str(v))
        frames = []
        while True:   # clips are ~10-20 s; keep net-size float frames (~1.8 MB each)
            ok, f = cap.read()
            if not ok:
                break
            fh, fw = f.shape[:2]
            frames.append(preprocess(f))
        cap.release()
        n = len(frames)
        if n < S:
            continue
        bg = [np.median(np.stack(frames[::15]), axis=0)] if card.uses_background else []
        starts = list(range(0, n - S + 1, S))
        if starts[-1] + S < n:
            starts.append(n - S)
        hms: dict[int, np.ndarray] = {}
        for s in starts:
            x = np.concatenate(bg + frames[s:s + S], 0)
            t = time.perf_counter()
            y = _predict(ml, x)
            t_pred += time.perf_counter() - t
            n_pred += 1
            for i in range(S):
                hms.setdefault(s + i, y[i])
        m = fit_affine(fw, fh)
        lo, hi = _drop_range(v.parts[-3], v.stem, n)
        for f_idx in range(lo, min(hi, n)):
            vis, gx, gy = rows.get(f_idx, (0, 0.0, 0.0))
            det = blob_decode(hms[f_idx], offset=card.centroid_offset)
            if det is None:
                fn += vis > 0
                tn += vis == 0
            elif vis == 0:
                fp += 1
            else:
                d = float(np.hypot(*(det[0] - (np.array([gx, gy]) * m[0, 0] + m[:, 2]))))
                errs.append(d / m[0, 0])
                tp, fp = (tp + 1, fp) if d <= 4.0 else (tp, fp + 1)
        del frames, hms
    del ml
    return {"clips": [v.stem for v in picked], "compute_units": cu,
            "frames": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            **_prf(tp, fp, fn),
            "median_err_fullres_px": round(float(np.median(errs)), 3) if errs else None,
            "mean_predict_ms": round(1e3 * t_pred / max(n_pred, 1), 2)}


def benchmark(path: Path, card: ModelCard, units: tuple[str, ...] = (
        "ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"), iters: int = 20) -> dict:
    """Median/p90 wall-clock predict latency on this Mac per compute-unit setting."""
    import coremltools as ct

    x = {"frames": np.random.default_rng(0).random(
        (1, card.in_channels, NET_H, NET_W), dtype=np.float32)}
    out: dict = {}
    for cu in units:
        t0 = time.perf_counter()
        ml = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, cu))
        load_s = time.perf_counter() - t0
        for _ in range(3):
            ml.predict(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            ml.predict(x)
            ts.append((time.perf_counter() - t) * 1e3)
        med = float(np.median(ts))
        out[cu] = {"median_ms": round(med, 2), "p90_ms": round(float(np.quantile(ts, 0.9)), 2),
                   "ms_per_frame": round(med / card.seq_len, 2), "load_s": round(load_s, 2)}
        del ml
    return out


def compute_plan(path: Path) -> dict:
    """Per-op preferred device + estimated cost share under ComputeUnit.ALL (macOS 14.4+)."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
    plan = MLComputePlan.load_from_path(path=ml.get_compiled_model_path(),
                                        compute_units=ct.ComputeUnit.ALL)
    ops = plan.model_structure.program.functions["main"].block.operations
    count: dict[str, int] = {}
    cost: dict[str, float] = {}
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = type(usage.preferred_compute_device).__name__ if usage else "none"
        dev = dev.replace("ML", "").replace("ComputeDevice", "")
        est = plan.get_estimated_cost_for_mlprogram_operation(op)
        count[dev] = count.get(dev, 0) + 1
        cost[dev] = cost.get(dev, 0.0) + (est.weight if est else 0.0)
    tot = sum(cost.values()) or 1.0
    return {"ops": count, "cost_share": {k: round(v / tot, 4) for k, v in cost.items()}}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=sorted(CARDS), default="wasb")
    ap.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    ap.add_argument("--weights", type=Path, default=None,
                    help="checkpoint file overriding the card's release weights (e.g. "
                         "ml/runs/wasb_ft/best.pth.tar from ml/train_detector.py)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                    help="precision of the shipped package (the other one is built in /tmp "
                         "for validation only)")
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--no-bench", dest="bench", action="store_false")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--windows-per-clip", type=int, default=4)
    ap.add_argument("--clip-eval", type=int, default=0, metavar="N",
                    help="also run the shipped package over N whole held-out Test clips")
    ap.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    a = ap.parse_args(argv)

    card = CARDS[a.model]
    torch.set_grad_enabled(False)
    traced, wrapper = trace(card, a.weights_dir, a.weights)
    params = sum(p.numel() for p in wrapper.net.parameters())
    report: dict = {"model": card.kind, "params": params, "output": str(a.output),
                    "metadata": contract_metadata(card),
                    "io": {"frames": [1, card.in_channels, NET_H, NET_W],
                           "heatmap": [1, card.seq_len, NET_H, NET_W]}}
    ml = convert(traced, card, a.precision)
    report["package_mb"] = round(save(ml, a.output), 3)
    del ml
    print(f"saved {a.output} ({report['package_mb']:.2f} MB, {params / 1e6:.2f} M params)")

    tmp = Path(tempfile.mkdtemp(prefix="shuttl_coreml_"))
    try:
        if a.validate:
            other = "fp32" if a.precision == "fp16" else "fp16"
            other_path = tmp / f"{card.kind}_{other}.mlpackage"
            report[f"{other}_package_mb"] = round(save(convert(traced, card, other), other_path), 3)
            pk = {a.precision: a.output, other: other_path}
            report["validation"] = validate(
                wrapper, card,
                {"coreml_fp32_cpu": (pk["fp32"], "CPU_ONLY"),
                 "coreml_fp16_all": (pk["fp16"], "ALL"),
                 "coreml_fp16_cpu_ne": (pk["fp16"], "CPU_AND_NE")},
                a.clips, a.windows_per_clip)
        if a.bench:
            report["latency_mac"] = benchmark(a.output, card)
            try:
                report["compute_plan_all"] = compute_plan(a.output)
            except Exception as e:  # noqa: BLE001 - needs macOS 14.4+
                report["compute_plan_all"] = f"unavailable: {e!r}"
        if a.clip_eval:
            report["clip_eval"] = clip_eval(a.output, card, a.clip_eval)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if a.report:
        a.report.write_text(text)
    return report


if __name__ == "__main__":
    main()
