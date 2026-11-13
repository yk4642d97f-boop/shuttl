"""Reference values for the ShuttlDetector Swift package (port of tracking/detector.py).

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.detector [--no-nets] [--score-swift DIR]

Writes iOS/Packages/ShuttlDetector/Tests/ShuttlDetectorTests/Fixtures/*.json:
  geometry.json    FrameGeometry scale / content / pad / axis / mappings (SIZES x corner)
  images.json      cv2 warpAffine / resize / RGB2GRAY outputs on formula images (row checksums)
  decode.json      decode_heatmap on sparse heatmaps (synthetic + real torch fp32 outputs)
  fusion.json      iter_heatmaps fused values with a slot-constant net (windowing contract)
  background.json  estimate_backgrounds on a formula scene with a cut (starts + row checksums)
  real.json        TrackNetV2 Test split (held out, eval only): labels, drop masks, cached Python
                   candidates per model (/tmp/shuttl_bench_cache, benchmark.py --stage real) and the
                   benchmark's pooled TrackNet-protocol F1.

``--score-swift DIR`` instead scores Swift candidate dumps (DIR/<match>_<clip>_<kind>.json written by
the Swift real-clip test) with benchmark.py's own metric code and prints F1 per model.
Formula images are generated identically in Swift (see FixtureImages.swift).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from app.services.tracking import detector as D
from app.services.tracking import metrics as M
from scripts.swift_fixtures.common import BACKEND, SWIFT_PACKAGES, TEST_SPLIT, dump

OUT = SWIFT_PACKAGES / "ShuttlDetector/Tests/ShuttlDetectorTests/Fixtures"
REAL_ROOT = BACKEND / "ml/data/raw/shuttlecock_trajectory"
CACHE = Path("/tmp/shuttl_bench_cache")
SPARSE_MIN = 0.049         # values below min_score/2 (>= 0.05 in every case) never influence decode_heatmap


# ───────────────────────────────────────────── formula images (mirrored in Swift)

def hash_image(w: int, h: int, seed: int = 0) -> np.ndarray:
    """(h, w, 3) uint8 RGB pseudo-random texture from uint32 integer hashing."""
    y, x, c = np.meshgrid(np.arange(h, dtype=np.uint64), np.arange(w, dtype=np.uint64),
                          np.arange(3, dtype=np.uint64), indexing="ij")
    m = np.uint64(0xFFFFFFFF)
    hh = ((x * np.uint64(73856093)) ^ (y * np.uint64(19349663)) ^ ((c + np.uint64(seed)) * np.uint64(83492791))) & m
    hh = ((hh ^ (hh >> np.uint64(13))) * np.uint64(1274126177)) & m
    return ((hh >> np.uint64(16)) & np.uint64(255)).astype(np.uint8)


def smooth_image(w: int, h: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    r = x * 255 // max(w - 1, 1)
    g = y * 255 // max(h - 1, 1)
    b = (x + y) * 255 // max(w + h - 2, 1)
    return np.dstack([r, g, b]).astype(np.uint8)


def mixed_image(w: int, h: int, seed: int = 0) -> np.ndarray:
    n = hash_image(w, h, seed).astype(np.int64) % 32 - 16
    return np.clip(smooth_image(w, h).astype(np.int64) + n, 0, 255).astype(np.uint8)


def make_image(kind: str, w: int, h: int, seed: int = 0) -> np.ndarray:
    return {"hash": hash_image, "smooth": lambda w, h, s: smooth_image(w, h),
            "mixed": mixed_image}[kind](w, h, seed)


def row_checksums(img: np.ndarray) -> dict:
    """Per row: plain sum and sum of v * ((i % 251) + 1), i = index inside the row (x*3+c)."""
    a = img.reshape(img.shape[0], -1).astype(np.int64)
    wts = (np.arange(a.shape[1]) % 251 + 1).astype(np.int64)
    return {"sum": a.sum(1), "wsum": (a * wts).sum(1)}


# ───────────────────────────────────────────── geometry + images

SIZES = [(1920, 1080), (1280, 720), (640, 480), (1024, 768), (1080, 1920), (720, 1280), (640, 360),
         (100, 60), (3000, 1000)]


def geometry_cases() -> list[dict]:
    out = []
    rng = np.random.default_rng(11)
    for w, h in SIZES:
        for corner in (False, True):
            g = D.FrameGeometry(w, h, corner=corner)
            px, py = rng.uniform(-5, w + 5, 6), rng.uniform(-5, h + 5, 6)
            nx, ny = g.to_net(px, py)
            fx, fy = g.to_full(px * 0.4, py * 0.4)
            qx, qy = g.peak_to_full(px * 0.4, py * 0.4, -0.2)
            out.append({"w": w, "h": h, "corner": corner, "scale": g.scale, "content": list(g.content_wh),
                        "pad": list(g.pad), "axis": list(g._axis()), "px": px, "py": py, "net_x": nx,
                        "net_y": ny, "full_x": fx, "full_y": fy, "peak_x": qx, "peak_y": qy})
    return out


def image_cases() -> dict:
    warp, area, linear = [], [], []
    for (w, h), kind in [((1280, 720), "hash"), ((1920, 1080), "mixed"), ((640, 480), "hash"),
                         ((1080, 1920), "mixed"), ((720, 1280), "hash"), ((640, 360), "mixed"),
                         ((1366, 768), "smooth"), ((100, 60), "hash"), ((3000, 1000), "mixed")]:
        img = make_image(kind, w, h, seed=w % 7)
        gc = D.FrameGeometry(w, h, corner=True)
        warp.append({"w": w, "h": h, "kind": kind, "seed": w % 7, **row_checksums(gc.warp(img))})
        gp = D.FrameGeometry(w, h)
        area.append({"w": w, "h": h, "kind": kind, "seed": w % 7, **row_checksums(gp.warp(img, cv2.INTER_AREA))})
        linear.append({"w": w, "h": h, "kind": kind, "seed": w % 7,
                       **row_checksums(gp.warp(img, cv2.INTER_LINEAR))})
    canvas = mixed_image(512, 288, 3)
    thumb = cv2.cvtColor(cv2.resize(canvas, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray_src = hash_image(97, 31, 5)
    return {"warp": warp, "area": area, "linear": linear,
            "thumb": {"kind": "mixed", "seed": 3, "gray": thumb.ravel()},
            "gray": {"w": 97, "h": 31, "kind": "hash", "seed": 5,
                     "gray": cv2.cvtColor(gray_src, cv2.COLOR_RGB2GRAY).ravel()},
            "ensemble": {str(s): {m: D.ensemble_weights(s, m) for m in ("weight", "average")}
                         for s in (1, 2, 3, 5, 8)}}


# ───────────────────────────────────────────── decoding

def _gauss(h: int, w: int, cx: float, cy: float, sigma: float, amp: float = 1.0) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return (amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def sparse(hm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Zero values < SPARSE_MIN (decode-neutral) and encode the rest as flat index/value."""
    hm = np.where(hm >= SPARSE_MIN, hm, 0).astype(np.float32)
    idx = np.flatnonzero(hm)
    return hm, {"idx": idx, "val": [float(v) for v in hm.ravel()[idx]]}


def decode_case(name: str, hm: np.ndarray, params: list[dict]) -> dict:
    hm, enc = sparse(hm)
    res = []
    for p in params:
        out = D.decode_heatmap(hm, **p)
        res.append({**p, "out": [list(o) for o in out]})
    return {"name": name, "h": int(hm.shape[0]), "w": int(hm.shape[1]), **enc, "runs": res}


STD = [dict(k=3, min_score=0.1, refine=r) for r in ("centroid", "quadratic", "none")]


def synthetic_heatmaps() -> list[dict]:
    H, W = D.NET_H, D.NET_W
    cases = []
    for sigma in (1.5, 2.0, 2.5, 3.0):
        rng = np.random.default_rng(int(sigma * 10))
        for j in range(6):
            cx, cy = rng.uniform(20, 490), rng.uniform(20, 265)
            hm = _gauss(H, W, cx, cy, sigma, amp=rng.uniform(0.4, 1.0))
            cases.append(decode_case(f"gauss_s{sigma}_{j}", hm, [dict(k=1, min_score=0.1, refine=r)
                                                                 for r in ("centroid", "quadratic")]))
    hm = _gauss(H, W, 100.3, 50.7, 2.0, 0.9) + _gauss(H, W, 300.0, 200.0, 2.0, 0.6) \
        + _gauss(H, W, 450.0, 30.0, 2.0, 0.05)
    cases.append(decode_case("topk_nms_threshold", hm, STD))
    for r in (2, 3, 8):
        hm = np.zeros((H, W), np.float32)
        cv2.circle(hm, (200, 100), r, 1.0, -1)
        cases.append(decode_case(f"disc_{r}", hm, STD))
    for amp in (3.0, 20.0):
        hm = np.minimum(_gauss(H, W, 200.3, 100.6, 2.5, amp), 1.0).astype(np.float16).astype(np.float32)
        cases.append(decode_case(f"clipped_{amp:g}", hm, STD))
    rng = np.random.default_rng(5)
    for j in range(12):                       # clutter: overlapping peaks, borders, plateaus, ties
        hm = np.zeros((H, W), np.float32)
        for _ in range(int(rng.integers(2, 9))):
            cx = float(rng.choice([rng.uniform(-2, 3), rng.uniform(508, 513), rng.uniform(0, 512)]))
            cy = float(rng.choice([rng.uniform(-2, 3), rng.uniform(285, 290), rng.uniform(0, 288)]))
            hm = np.maximum(hm, _gauss(H, W, cx, cy, float(rng.uniform(1.0, 4.0)), float(rng.uniform(0.08, 1.6))))
        if j % 3 == 0:
            hm = np.minimum(hm, 1.0)
        if j % 4 == 1:
            hm = hm.astype(np.float16).astype(np.float32)
        if j % 5 == 2:                       # a pair of touching blobs (NMS by blob membership)
            hm = np.maximum(hm, _gauss(H, W, 250.0, 140.0, 2.5, 0.9) + _gauss(H, W, 257.0, 140.0, 2.5, 0.7))
        if j % 6 == 3:                       # exact ties at distinct places (stable ordering)
            hm[50:53, 60:63] = 0.75
            hm[200, 400] = 0.75
        params = STD + [dict(k=5, min_score=0.3, refine="centroid"), dict(k=2, min_score=0.1, nms_radius=10,
                                                                          refine="centroid", refine_radius=3)]
        cases.append(decode_case(f"clutter_{j}", hm, params))
    return cases


def _clip_frames(n: int, clip: Path) -> list[np.ndarray]:
    from app.services.tracking.video_io import iter_frames
    return [f for _, f in iter_frames(str(clip), stop=n)]


def real_heatmaps() -> list[dict]:
    """Torch fp32 CPU heatmaps of both nets on the first frames of a held-out Test clip."""
    torch.set_grad_enabled(False)
    clip = TEST_SPLIT / "match1" / "video" / "1_05_02.mp4"
    cases = []
    if not clip.is_file():
        return cases
    frames = _clip_frames(40, clip)
    for kind, pick in (("wasb", range(30, 36)), ("tracknetv3", range(24, 40))):
        path = D.find_weights(kind)
        if path is None:
            continue
        model, spec = D.load_model(path, kind)
        det = D.HeatmapDetector(model, spec, device="cpu", prefetch=False)
        hms = dict(det.iter_heatmaps(D.array_source(frames), 1280, 720, 30.0, len(frames)))
        for f in pick:
            cases.append(decode_case(f"real_{kind}_{f}", hms[f], STD + [dict(k=3, min_score=0.5, refine="centroid")]))
    return cases


# ───────────────────────────────────────────── fusion + background

class _SlotNet(torch.nn.Module):
    """Every heatmap = constant (slot + 1) / 16 (test_detector._SlotNet)."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.shape
        v = (torch.arange(self.seq_len, dtype=x.dtype) + 1) / 16
        return v.view(1, -1, 1, 1).expand(b, self.seq_len, h, w) + 0 * self.dummy


def fusion_cases() -> list[dict]:
    out = []
    for S, stride, bg, n, bs in [(3, 3, False, 50, 3), (3, 1, False, 31, 3), (8, 8, True, 45, 2),
                                 (8, 4, True, 43, 3), (8, 3, False, 30, 1), (8, 1, False, 20, 4),
                                 (8, 8, False, 5, 3), (3, 2, False, 2, 3), (8, 8, False, 64, 4),
                                 (8, 5, False, 17, 2)]:
        spec = D.NetSpec("slot", S, bg, False, False, False)
        det = D.HeatmapDetector(_SlotNet(S), spec, device="cpu", stride=stride, batch_size=bs, prefetch=False)
        frames = [np.full((72, 128, 3), 100, np.uint8) for _ in range(n)]
        got = list(det.iter_heatmaps(D.array_source(frames), 128, 72, 30.0, n))
        out.append({"seq_len": S, "stride": stride, "uses_background": bg, "n": n, "batch_size": bs,
                    "frames": [f for f, _ in got], "values": [float(h[0, 0]) for _, h in got],
                    "max_buffered": det.max_buffered, "max_lag": det.max_lag})
    return out


def scene_frame(i: int, w: int, h: int, cut: int | None) -> np.ndarray:
    """Formula scene: static texture (a second one after ``cut``) + a bright 12 px square moving
    9 px per frame. Mirrored in Swift."""
    base = mixed_image(w, h, 1 if cut is not None and i >= cut else 0).copy()
    x = (9 * i) % (w - 12)
    y = (h // 3 + 5 * i) % (h - 12)
    base[y:y + 12, x:x + 12] = (250, 240, 230)
    return base


def background_cases() -> list[dict]:
    out = []
    for w, h, n, cut, kw in [(640, 360, 200, 104, dict(max_samples=4)),
                             (640, 360, 200, None, dict(max_segment_s=2.0)),
                             (1280, 720, 64, None, {}),
                             (640, 480, 120, 56, dict(max_samples=3, cut_frac=0.3))]:
        geom = D.FrameGeometry(w, h)

        def source(step: int, w=w, h=h, n=n, cut=cut):
            for i in range(0, n, step):
                yield i, scene_frame(i, w, h, cut)[..., ::-1].copy()     # BGR like a video decoder
        plan = D.estimate_backgrounds(source, geom, 30.0, **kw)
        out.append({"w": w, "h": h, "n": n, "cut": cut, "fps": 30.0, "kwargs": kw, "starts": plan.starts,
                    "images": [row_checksums(im) for im in plan.images]})
    return out


# ───────────────────────────────────────────── real clips

def real_cases() -> dict:
    from scripts.benchmark import drop_mask, load_label_csv, real_clips
    from app.services.tracking import video_io
    clips, models = [], {}
    pooled: dict[str, list] = {}
    for clip in real_clips():
        meta = video_io.probe(clip["video"])
        n = meta.n_frames
        xy, vis = load_label_csv(clip["labels"], n)
        dm = drop_mask(clip["match"], clip["clip"], n)
        mask = np.ones(n, bool) if dm is None else dm
        entry = {"match": clip["match"], "clip": clip["clip"],
                 "video": str(Path(clip["video"]).relative_to(BACKEND.parent)),
                 "n": n, "width": meta.width, "height": meta.height,
                 "vis_frames": np.flatnonzero(vis), "vis_xy": xy[vis],
                 "mask_range": [int(np.argmax(mask)), int(len(mask) - np.argmax(mask[::-1]))] if mask.any() else [0, 0]}
        assert mask[entry["mask_range"][0]:entry["mask_range"][1]].all()
        for kind in ("tracknetv3", "wasb"):
            f = CACHE / f"{clip['match']}_{clip['clip']}_{kind}.npz"
            if not f.exists():
                continue
            arr = np.load(f)["cands"][:n]
            cands = [[(round(float(x), 3), round(float(y), 3), round(float(s), 5)) for x, y, s in row
                      if np.isfinite(s)] for row in arr]
            entry.setdefault("cands", {})[kind] = cands
            top, _ = M.candidates_to_xy(D_cands(arr), 0.5)
            r = M.detection_metrics(top, xy, vis, 4.0, mask=mask, tracknet_int=True,
                                    image_size=(meta.width, meta.height))
            entry.setdefault("f1_tn4", {})[kind] = r["f1"]
            pooled.setdefault(kind, []).append(r)
        clips.append(entry)
    for kind, rs in pooled.items():
        tp, fp1, fp2, fn, tn = (sum(r[k] for r in rs) for k in ("tp", "fp1", "fp2", "fn", "tn"))
        p, rc = M._ratio(tp, tp + fp1 + fp2), M._ratio(tp, tp + fn)
        models[kind] = {"tn4_f1": M._f1(p, rc), "tn4_precision": p, "tn4_recall": rc, "n_clips": len(rs),
                        "tp": tp, "fp1": fp1, "fp2": fp2, "fn": fn, "tn": tn}
    return {"clips": clips, "pooled": models}


def D_cands(arr: np.ndarray):
    from app.services.tracking.data_types import ShuttleCandidate
    return [[ShuttleCandidate(float(x), float(y), float(s)) for x, y, s in row if np.isfinite(s)] for row in arr]


def score_swift(d: Path) -> None:
    """Score Swift candidate dumps with benchmark.py's exact matching rule."""
    from scripts.benchmark import drop_mask, load_label_csv, real_clips
    from app.services.tracking import video_io
    res: dict[str, list] = {}
    for clip in real_clips():
        for kind in ("tracknetv3", "wasb"):
            f = d / f"{clip['match']}_{clip['clip']}_{kind}.json"
            if not f.exists():
                continue
            cands = json.loads(f.read_text())["cands"]
            meta = video_io.probe(clip["video"])
            n = meta.n_frames
            xy, vis = load_label_csv(clip["labels"], n)
            dm = drop_mask(clip["match"], clip["clip"], n)
            mask = np.ones(n, bool) if dm is None else dm
            arr = np.full((n, 3, 3), np.nan, np.float32)
            for i, row in enumerate(cands[:n]):
                for j, c in enumerate(row[:3]):
                    arr[i, j] = c
            top, _ = M.candidates_to_xy(D_cands(arr), 0.5)
            res.setdefault(kind, []).append(M.detection_metrics(top, xy, vis, 4.0, mask=mask, tracknet_int=True,
                                                                image_size=(meta.width, meta.height)))
    for kind, rs in res.items():
        tp, fp1, fp2, fn = (sum(r[k] for r in rs) for k in ("tp", "fp1", "fp2", "fn"))
        p, rc = M._ratio(tp, tp + fp1 + fp2), M._ratio(tp, tp + fn)
        print(json.dumps({"kind": kind, "clips": len(rs), "tn4_f1": M._f1(p, rc), "precision": p, "recall": rc}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-nets", action="store_true", help="skip real torch heatmaps")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--score-swift", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.score_swift:
        score_swift(a.score_swift)
        return
    todo = a.only or ["geometry", "images", "decode", "fusion", "background", "real"]
    if "geometry" in todo:
        print(dump(geometry_cases(), OUT / "geometry.json"))
    if "images" in todo:
        print(dump(image_cases(), OUT / "images.json"))
    if "decode" in todo:
        cases = synthetic_heatmaps() + ([] if a.no_nets else real_heatmaps())
        print(dump(cases, OUT / "decode.json"), len(cases), "cases")
    if "fusion" in todo:
        print(dump(fusion_cases(), OUT / "fusion.json"))
    if "background" in todo:
        print(dump(background_cases(), OUT / "background.json"))
    if "real" in todo:
        r = real_cases()
        print(dump(r, OUT / "real.json"), json.dumps(r["pooled"]))
    for f in sorted(OUT.glob("*.json")):
        print(f"{f.name}: {f.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    sys.exit(main())
