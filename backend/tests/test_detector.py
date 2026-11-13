"""Shuttle detector tests: decoding, geometry, streaming/fusion, motion fallback, weights."""

from __future__ import annotations

import math
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.services.tracking import detector as D
from app.services.tracking.data_types import ShuttleCandidate

BACKEND = Path(__file__).resolve().parents[1]
TEST_SPLIT = BACKEND / "ml" / "data" / "raw" / "shuttlecock_trajectory"
CLIP = TEST_SPLIT / "TrackNetV2" / "Test" / "match1" / "video" / "1_05_02.mp4"
CLIP_CSV = TEST_SPLIT / "corrected" / "match1" / "corrected_csv" / "1_05_02_ball.csv"


def _gauss(h: int, w: int, cx: float, cy: float, sigma: float, amp: float = 1.0) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return (amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


# ───────────────────────────────────────────────────────────────── decoding

@pytest.mark.parametrize("refine", ["centroid", "quadratic"])
@pytest.mark.parametrize("sigma", [1.5, 2.0, 2.5, 3.0])
def test_decode_subpixel_gaussian(refine: str, sigma: float) -> None:
    rng = np.random.default_rng(int(sigma * 10))
    errs = []
    for _ in range(40):
        cx, cy = rng.uniform(20, 490), rng.uniform(20, 265)
        hm = _gauss(D.NET_H, D.NET_W, cx, cy, sigma, amp=rng.uniform(0.4, 1.0))
        (x, y, s), = D.decode_heatmap(hm, k=1, min_score=0.1, refine=refine)
        errs.append(math.hypot(x - cx, y - cy))
        assert 0.4 * 0.95 <= s <= 1.0
    assert max(errs) < 0.2, f"max subpixel error {max(errs):.3f}"


def test_decode_topk_sorted_nms_and_threshold() -> None:
    hm = _gauss(D.NET_H, D.NET_W, 100.3, 50.7, 2.0, 0.9) + _gauss(D.NET_H, D.NET_W, 300.0, 200.0,
                                                                  2.0, 0.6)
    hm += _gauss(D.NET_H, D.NET_W, 450.0, 30.0, 2.0, 0.05)          # below min_score
    c = D.decode_heatmap(hm, k=3, min_score=0.1)
    assert len(c) == 2 and c[0][2] > c[1][2]
    assert abs(c[0][0] - 100.3) < 0.2 and abs(c[1][1] - 200.0) < 0.2
    assert D.decode_heatmap(np.zeros((D.NET_H, D.NET_W), np.float32)) == []


@pytest.mark.parametrize("refine", ["centroid", "quadratic"])
@pytest.mark.parametrize("shape,arg", [("disc", 2), ("disc", 3), ("disc", 8), ("clipped", 3.0),
                                       ("clipped", 20.0)])
def test_decode_plateau_gives_one_centred_peak(shape: str, arg: float, refine: str) -> None:
    """Plateaus — TrackNet-style binary disc targets, or sigmoid outputs saturated at 1.0 in
    fp16 — must decode to ONE peak at the blob centre, whichever plateau pixel seeds it
    (the seed is the first maximal pixel in scan order, i.e. the plateau's top edge)."""
    if shape == "disc":
        cx, cy = 200.0, 100.0
        hm = np.zeros((D.NET_H, D.NET_W), np.float32)
        cv2.circle(hm, (200, 100), int(arg), 1.0, -1)
    else:
        cx, cy = 200.3, 100.6
        hm = np.minimum(_gauss(D.NET_H, D.NET_W, cx, cy, 2.5, arg), 1.0)
        hm = hm.astype(np.float16).astype(np.float32)                # fp16 quantisation
    p = D.decode_heatmap(hm, k=3, min_score=0.1, refine=refine)
    assert len(p) == 1, p
    assert math.hypot(p[0][0] - cx, p[0][1] - cy) < 0.1, (p, cx, cy)


# ───────────────────────────────────────────────────────────────── geometry

SIZES = [(1920, 1080), (1280, 720), (640, 480), (1024, 768), (1080, 1920), (720, 1280)]


@pytest.mark.parametrize("corner", [False, True])
@pytest.mark.parametrize("wh", SIZES)
def test_geometry_roundtrip_and_warp(wh: tuple[int, int], corner: bool) -> None:
    w, h = wh
    g = D.FrameGeometry(w, h, corner=corner)
    rng = np.random.default_rng(w + h)
    px, py = rng.uniform(0.2 * w, 0.8 * w, 10), rng.uniform(0.2 * h, 0.8 * h, 10)
    bx, by = g.to_full(*g.to_net(px, py))
    assert np.allclose(bx, px, atol=1e-9) and np.allclose(by, py, atol=1e-9)
    # a blob rendered at full res lands where to_net says (sampling convention is exact)
    sigma = 2.5 / g.scale
    for x, y in zip(px[:3], py[:3]):
        img = np.repeat(_gauss(h, w, x, y, sigma, 250.0)[..., None], 3, 2).astype(np.uint8)
        net = g.warp(img, cv2.INTER_LINEAR if corner else cv2.INTER_AREA)[..., 0].astype(float)
        ex, ey = g.to_net(x, y)
        ys, xs = np.mgrid[0:net.shape[0], 0:net.shape[1]]
        wts = np.where(net > 0.1 * net.max(), net, 0)
        cx, cy = (wts * xs).sum() / wts.sum(), (wts * ys).sum() / wts.sum()
        assert math.hypot(cx - ex, cy - ey) < 0.15, (cx, cy, ex, ey)
    # letterbox: content fills one axis, padding is black and symmetric
    net = g.warp(np.full((h, w, 3), 200, np.uint8))
    cols, rows = np.nonzero(net[..., 0].max(0))[0], np.nonzero(net[..., 0].max(1))[0]
    assert (cols[0] == 0 and cols[-1] == D.NET_W - 1) or (rows[0] == 0 and rows[-1] == D.NET_H - 1)
    assert abs(cols[0] - (D.NET_W - 1 - cols[-1])) <= 1 and abs(rows[0] - (D.NET_H - 1 - rows[-1])) <= 1


@pytest.mark.parametrize("kind", ["wasb", "tracknetv3"])
def test_peak_offset_inverts_training_targets(kind: str) -> None:
    """At the training resolution (1280x720, s=0.4) both repos draw the target at
    int(X*0.4), whose mean position is X*0.4-0.5; decoding must return X."""
    spec = D.SPECS[kind]
    g = D.FrameGeometry(1280, 720, corner=spec.corner_geometry)
    X = np.array([3.0, 101.7, 640.2, 1279.0])
    fx, fy = g.peak_to_full(X * 0.4 - 0.5, X * 0.4 - 0.5, spec.peak_offset)
    assert np.allclose(fx, X, atol=1e-9) and np.allclose(fy, X, atol=1e-9)


def test_candidates_drop_letterbox_hits() -> None:
    det = D.HeatmapDetector(_EchoNet(3, False), _echo_spec(3, False), device="cpu")
    g = det.geometry(640, 480)                        # 4:3 → pillarbox, pad ≈ 64 px each side
    hm = _gauss(D.NET_H, D.NET_W, 10.0, 144.0, 2.0) + _gauss(D.NET_H, D.NET_W, 256.0, 144.0,
                                                             2.0, 0.8)
    c = det.candidates(hm, g)
    assert len(c) == 1 and abs(c[0].x - 320) < 1.5 and abs(c[0].y - 240) < 1.5


# ───────────────────────────────────────────────────── streaming + window fusion

class _EchoNet(nn.Module):
    """Heatmap s of every window = red channel of input frame s (background slot skipped)."""

    def __init__(self, seq_len: int, uses_bg: bool):
        super().__init__()
        self.seq_len, self.off = seq_len, int(uses_bg)
        self.dummy = nn.Parameter(torch.zeros(1))          # gives the module a dtype/device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        fr = x.view(b, -1, 3, h, w)[:, self.off:self.off + self.seq_len, 0]
        return fr + 0 * self.dummy


def _echo_spec(seq_len: int, uses_bg: bool) -> D.NetSpec:
    return D.NetSpec("echo", seq_len, uses_bg, False, False, False)   # positional (compat)


def _frame_value(i: int) -> int:
    return 10 + (7 * i) % 230


@pytest.mark.parametrize("seq_len,stride,uses_bg,n", [
    (3, 3, False, 50), (3, 1, False, 31), (8, 8, True, 45), (8, 4, True, 43),
    (8, 1, False, 20), (8, 8, False, 5), (3, 2, False, 2)])
def test_streaming_fusion_alignment_and_bounded_memory(seq_len: int, stride: int, uses_bg: bool,
                                                       n: int) -> None:
    alive: weakref.WeakSet = weakref.WeakSet()

    def source(step: int):
        for i in range(0, n, step):
            f = np.empty((72, 128, 3), np.uint8)   # 16:9 → fills the canvas exactly
            f[:] = (0, 0, _frame_value(i))         # BGR: red channel carries the frame id
            alive.add(_Tracked(f))
            yield i, f

    det = D.HeatmapDetector(_EchoNet(seq_len, uses_bg), _echo_spec(seq_len, uses_bg),
                            device="cpu", stride=stride, batch_size=2, prefetch=False)
    got = list(det.iter_heatmaps(source, 128, 72, 30.0, n))
    assert [f for f, _ in got] == list(range(n))            # every frame exactly once, in order
    for f, hm in got:
        assert hm.shape == (D.NET_H, D.NET_W)
        assert abs(float(hm.mean()) - _frame_value(f) / 255.0) < 1e-5, f
    # bounded: ≤ one window + one batch of windows, independent of video length
    assert det.max_buffered <= seq_len + 2 * seq_len
    assert det.max_lag <= seq_len + 2 * stride + seq_len


class _Tracked:
    """Weak-referenceable holder so the test can count live raw frames."""

    def __init__(self, a: np.ndarray):
        self.a = a


def test_streaming_does_not_hold_more_than_window() -> None:
    """Raw frames are released as soon as they are preprocessed; preprocessed frames held
    stay bounded by window + batch regardless of video length."""
    n, S = 400, 3
    live: list[int] = []
    refs: list[weakref.ref] = []

    class Frame(np.ndarray):
        pass

    def source(step: int):
        for i in range(0, n, step):
            f = np.full((72, 128, 3), i % 250, np.uint8).view(Frame)
            refs.append(weakref.ref(f))
            live.append(sum(r() is not None for r in refs[-64:]))
            yield i, f

    det = D.HeatmapDetector(_EchoNet(S, False), _echo_spec(S, False), device="cpu",
                            stride=S, batch_size=4, prefetch=False)
    got = sum(1 for _ in det.iter_heatmaps(source, 128, 72, 30.0, n))
    assert got == n
    assert max(live) <= 2, f"raw frames alive: {max(live)}"
    assert det.max_buffered <= S + 4 * S
    assert det.max_lag <= S + 4 * S


class _BgEcho(_EchoNet):
    """Every output heatmap = green channel of input slot 0 (the background slot)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        bg = x.view(b, -1, 3, h, w)[:, 0, 1]
        return bg[:, None].expand(b, self.seq_len, h, w) + 0 * self.dummy


def _scene_source(n: int, cut: int | None = None):
    """Static scene (green 100; 200 from frame ``cut`` on) + a small bright blob moving 5 px
    per frame, so every pixel is foreground in at most a minority of the median's samples.
    A frame in slot 0 would carry the blob, so it is told apart from the median."""

    def source(step: int):
        for i in range(0, n, step):
            f = np.zeros((72, 128, 3), np.uint8)
            f[..., 1] = 200 if cut is not None and i >= cut else 100
            x = (5 * i) % 120
            f[30:38, x:x + 8, 1] = 250
            yield i, f
    return source


def test_background_is_prepended_first() -> None:
    """TrackNetV3 order: [median background, f0..f7]; the echo of slot 0 must be the median
    (moving blob removed), not a frame."""
    n = 40
    det = D.HeatmapDetector(_BgEcho(8, True), _echo_spec(8, True), device="cpu", prefetch=False)
    got = list(det.iter_heatmaps(_scene_source(n), 128, 72, 30.0, n))
    assert [f for f, _ in got] == list(range(n))
    assert all(abs(float(h.mean()) - 100 / 255) < 1e-5 for _, h in got)


def test_background_follows_scene_cuts() -> None:
    """A global change (scene cut) starts a new median segment; windows inside the second
    scene use its own background (the 4 Hz samples land on the cut at frame 24)."""
    n, cut = 48, 24
    det = D.HeatmapDetector(_BgEcho(8, True), _echo_spec(8, True), device="cpu", prefetch=False)
    got = dict(det.iter_heatmaps(_scene_source(n, cut), 128, 72, 30.0, n))
    for f, h in got.items():
        assert abs(float(h.mean()) - (200 if f >= cut else 100) / 255) < 1e-5, f


def test_detect_frames_output_contract() -> None:
    """End to end through HeatmapDetector with an oracle net: one list per frame, top-k
    sorted, full-res subpixel coordinates (corner geometry, zero peak offset → identity)."""
    W, H, n = 640, 360, 12
    pts = [(100.25 + 30 * i, 80.75 + 11 * i) for i in range(n)]

    class Oracle(_EchoNet):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            return x.view(b, -1, 3, h, w)[:, :self.seq_len, 0] + 0 * self.dummy

    def source(step: int):
        for i in range(0, n, step):
            g = _gauss(H, W, *pts[i], sigma=2.5 * W / D.NET_W, amp=230.0)
            yield i, np.dstack([np.zeros_like(g), np.zeros_like(g), g]).astype(np.uint8)

    spec = D.NetSpec("oracle", 3, False, False, False, True, 0.0)
    det = D.HeatmapDetector(Oracle(3, False), spec, device="cpu", stride=3, prefetch=False)
    out = det.detect_frames(source, W, H, 30.0, n)
    assert len(out) == n
    for i, cs in enumerate(out):
        assert 1 <= len(cs) <= 3 and all(isinstance(c, ShuttleCandidate) for c in cs)
        assert [c.score for c in cs] == sorted((c.score for c in cs), reverse=True)
        assert math.hypot(cs[0].x - pts[i][0], cs[0].y - pts[i][1]) < 0.35, (i, cs[0], pts[i])


class _SlotNet(_EchoNet):
    """Every heatmap = constant (slot + 1) / 16 — exposes the fusion weights per slot."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        v = (torch.arange(self.seq_len, dtype=x.dtype) + 1) / 16
        return v.view(1, -1, 1, 1).expand(b, self.seq_len, h, w) + 0 * self.dummy


def _reference_fusion(n: int, S: int, stride: int) -> list[float]:
    """Independent re-statement of the windowing contract: full windows every ``stride``;
    if frames remain uncovered, one tail window over the last S real frames (shorter clips:
    padded) contributing only to frames at or after the next unrun window start."""
    w = D.ensemble_weights(S, "weight" if stride < S else "average")
    num, den = np.zeros(n), np.zeros(n)
    starts = list(range(0, n - S + 1, stride)) if n >= S else []
    for st in starts:
        for s in range(S):
            num[st + s] += w[s] * (s + 1) / 16
            den[st + s] += w[s]
    if not starts or starts[-1] + S < n:
        lo, tail = (starts[-1] + stride if starts else 0), max(n - S, 0)
        for s in range(S):
            if lo <= tail + s < n:
                num[tail + s] += w[s] * (s + 1) / 16
                den[tail + s] += w[s]
    return list(num / den)


@pytest.mark.parametrize("seq_len,stride,uses_bg,n", [
    (3, 3, False, 50), (3, 1, False, 31), (8, 8, True, 45), (8, 4, True, 43), (8, 3, False, 30),
    (8, 1, False, 20), (8, 8, False, 5), (3, 2, False, 2), (8, 8, False, 64)])
def test_window_fusion_weights_and_tail(seq_len: int, stride: int, uses_bg: bool, n: int) -> None:
    det = D.HeatmapDetector(_SlotNet(seq_len, uses_bg), _echo_spec(seq_len, uses_bg),
                            device="cpu", stride=stride, batch_size=3, prefetch=False)
    got = list(det.iter_heatmaps(_scene_source(n), 128, 72, 30.0, n))
    ref = _reference_fusion(n, seq_len, stride)
    assert [f for f, _ in got] == list(range(n))
    for f, hm in got:
        assert abs(float(hm[0, 0]) - ref[f]) < 1e-6, (f, float(hm[0, 0]), ref[f])
    if stride == seq_len:        # non-overlapping: every frame from exactly one window slot
        assert all(any(abs(r - (s + 1) / 16) < 1e-9 for s in range(seq_len)) for r in ref)


def test_progress_is_monotonic_and_completes() -> None:
    seen: list[float] = []
    det = D.HeatmapDetector(_EchoNet(8, True), _echo_spec(8, True), device="cpu", prefetch=True)
    out = det.detect_frames(_scene_source(100), 128, 72, 30.0, 100, progress=seen.append)
    assert len(out) == 100 and seen and seen[-1] == 1.0
    assert all(b >= a for a, b in zip(seen, seen[1:])) and 0.0 <= seen[0]


# ───────────────────────────────────────────────────── build_detector / weights

def test_build_detector_falls_back_without_weights(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUTTL_DETECTOR", raising=False)
    assert isinstance(D.build_detector(str(tmp_path), "cpu"), D.MotionDetector)
    with pytest.raises(FileNotFoundError):
        D.build_detector(str(tmp_path), "cpu", kind="wasb")
    with pytest.raises(ValueError):
        D.build_detector(str(tmp_path), "cpu", kind="bogus")
    monkeypatch.setenv("SHUTTL_DETECTOR", "motion")
    assert isinstance(D.build_detector(None, "cpu"), D.MotionDetector)


@pytest.mark.parametrize("kind,seq_len,in_ch", [("wasb", 3, 9), ("tracknetv3", 8, 27)])
def test_official_weights_load_strict(kind: str, seq_len: int, in_ch: int) -> None:
    path = D.find_weights(kind)
    if path is None:
        pytest.skip(f"{kind} weights not present")
    sd, _ = D.load_checkpoint(path)
    spec = D.spec_from_state_dict(sd)
    assert spec.name == kind and spec.seq_len == seq_len and spec.in_channels == in_ch
    res = D.build_model(spec).load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model, spec2 = D.load_model(path)
    assert spec2 == spec
    with torch.inference_mode():
        y = model(torch.rand(1, in_ch, D.NET_H, D.NET_W))
    assert y.shape == (1, seq_len, D.NET_H, D.NET_W) and torch.isfinite(y).all()


def _clip_frames(n: int) -> list[np.ndarray]:
    from app.services.tracking.video_io import iter_frames
    return [f for _, f in iter_frames(str(CLIP), stop=n)]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("kind", ["wasb", "tracknetv3"])
def test_fp16_mps_matches_fp32_cpu(kind: str) -> None:
    path = D.find_weights(kind)
    if path is None or not CLIP.is_file():
        pytest.skip("weights or Test clip not present")
    model, spec = D.load_model(path)
    frames = _clip_frames(2 * spec.seq_len)
    hms = {}
    for dev, half in (("cpu", False), ("mps", True)):
        det = D.HeatmapDetector(model, spec, device=dev, half=half, prefetch=False)
        hms[dev] = dict(det.iter_heatmaps(D.array_source(frames), 1280, 720, 30.0, len(frames)))
        model = model.float().cpu()
    for f, ref in hms["cpu"].items():
        assert float(np.abs(hms["mps"][f] - ref).max()) < 0.02, f
        a, b = D.decode_heatmap(ref, k=1), D.decode_heatmap(hms["mps"][f], k=1)
        assert len(a) == len(b)
        if a:
            assert math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]) < 0.1


@pytest.mark.parametrize("kind", ["wasb", "tracknetv3"])
def test_real_clip_smoke(kind: str) -> None:
    """First 96 frames of a held-out Test clip (no tuning happens here; regression guard)."""
    path = D.find_weights(kind)
    if path is None or not CLIP.is_file() or not CLIP_CSV.is_file():
        pytest.skip("weights or Test clip not present")
    det = D.build_detector(None, None, kind=kind)
    n = 96
    frames = _clip_frames(n)
    cands = det.detect_frames(D.array_source(frames), 1280, 720, 30.0, n)
    r = D.score_candidates(cands, D.load_labels(CLIP_CSV, n), 1280, 720, range(28, n))
    assert r["fixed"]["f1"] > 0.9 and r["err_median_px"] < 3.0, r


# ───────────────────────────────────────────────────── motion fallback

def _motion_clip(n: int = 90, w: int = 1280, h: int = 720, seed: int = 0):
    """Textured static background with court-like lines and a flickering light, per-frame
    sensor noise (σ=4), and a small white shuttle (Gaussian σ=1.8 px) on a parabolic path."""
    rng = np.random.default_rng(seed)
    bg = cv2.resize(rng.uniform(60, 140, (h // 16, w // 16)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    for y in (int(0.3 * h), int(0.7 * h)):
        bg[y:y + 3, int(0.1 * w):int(0.9 * w)] = 230.0                 # static white lines
    t = np.arange(n)
    xs = 0.15 * w + t * (0.7 * w / n)
    ys = 0.2 * h + 0.6 * h * ((t - n / 2) / (n / 2)) ** 2
    ys = ys + rng.uniform(-0.5, 0.5, n)
    frames = []
    for i in range(n):
        f = bg + rng.normal(0, 4.0, (h, w)).astype(np.float32)
        if i % 4 < 2:
            f[40:48, 40:48] += 60.0                                   # flickering light
        x0, y0 = int(xs[i]) - 8, int(ys[i]) - 8
        f[y0:y0 + 17, x0:x0 + 17] += _gauss(17, 17, xs[i] - x0, ys[i] - y0, 1.8, 150.0)
        g = np.clip(f, 0, 255).astype(np.uint8)
        frames.append(np.dstack([g, g, g]))
    return frames, np.stack([xs, ys], 1)


def test_motion_detector_finds_moving_white_dot() -> None:
    frames, truth = _motion_clip()
    det = D.MotionDetector()
    out = det.detect_frames(D.array_source(frames), 1280, 720, 30.0, len(frames))
    assert len(out) == len(frames)
    errs = [math.hypot(cs[0].x - tx, cs[0].y - ty) if cs else math.inf
            for cs, (tx, ty) in zip(out, truth)]
    hits = [e for e in errs if e <= 5.0]
    assert len(hits) / len(errs) >= 0.95, f"recall {len(hits) / len(errs):.3f}"
    assert float(np.mean(hits)) <= 1.5, f"mean err {np.mean(hits):.2f}"
    assert det.max_buffered <= 3


def test_motion_detector_speed_720p_cpu() -> None:
    """≥150 fps at 1280x720 on CPU, measured in process CPU time (all threads) so the check
    is conservative for an idle machine and not flaky on a shared, oversubscribed one."""
    import time

    frames, _ = _motion_clip(n=120, seed=1)
    det = D.MotionDetector()
    c0 = time.process_time()
    det.detect_frames(D.array_source(frames), 1280, 720, 30.0, len(frames))
    fps = len(frames) / (time.process_time() - c0)
    assert fps >= 150, f"{fps:.0f} fps (CPU time)"
