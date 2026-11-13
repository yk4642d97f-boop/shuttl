"""Fast checks of the shipped CoreML contract and of the training-target conventions that must
stay consistent with it (ml/export_coreml.py, ml/train_detector.py).

CoreML-dependent tests skip cleanly when coremltools' native libs or the package are missing.
Run: cd backend && .venv/bin/python -m pytest tests/test_export_coreml.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ML = BACKEND / "ml"
for p in (str(BACKEND), str(ML)):
    if p not in sys.path:
        sys.path.insert(0, p)

E = pytest.importorskip("export_coreml")
PACKAGE = E.DEFAULT_OUTPUT
EXPECTED_META = {
    "model": "wasb", "seq_len": "3", "in_channels": "9", "uses_background": "0",
    "input_order": "rgb", "normalize": "0-1", "input_width": "512", "input_height": "288",
    "frame_order": "oldest_to_newest", "output_activation": "sigmoid",
    "score_threshold": "0.5", "centroid_offset": "0.5",
    "resize": "aspect_fit_center_black_pad",
}


def _coremltools():
    ct = pytest.importorskip("coremltools")
    try:
        import coremltools.libcoremlpython  # noqa: F401
        import coremltools.libmilstoragepython  # noqa: F401
    except ImportError:
        pytest.skip("coremltools native libs unavailable (pure-python build)")
    return ct


def _weights_or_skip() -> None:
    if not (E.WEIGHTS_DIR / E.CARDS["wasb"].weights).is_file():
        pytest.skip("WASB weights not on disk")


# ── contract (no model load) ─────────────────────────────────────────────────

def test_metadata_contract_matches_card():
    assert E.contract_metadata(E.CARDS["wasb"]) == EXPECTED_META


def test_shipped_package_spec_and_metadata():
    ct = _coremltools()
    if not PACKAGE.exists():
        pytest.skip("ShuttleTracker.mlpackage not built")
    m = ct.models.MLModel(str(PACKAGE), skip_model_load=True)
    spec = m.get_spec()
    (inp,), (out,) = spec.description.input, spec.description.output
    assert inp.name == "frames" and list(inp.type.multiArrayType.shape) == [1, 9, 288, 512]
    assert out.name == "heatmap"
    meta = dict(m.user_defined_metadata)
    for k, v in EXPECTED_META.items():
        assert meta.get(k) == v, k
    assert E.package_mb(PACKAGE) < 10.0


def test_geometry_roundtrip_and_blob_decode():
    for fw, fh in ((1280, 720), (1920, 1080), (640, 480), (720, 1280)):
        m = E.fit_affine(fw, fh)
        pts = np.array([[0.0, 0.0], [fw - 1.0, fh - 1.0], [fw / 3.0, fh / 5.0]])
        net = pts * m[0, 0] + m[:, 2]
        assert np.allclose(E.to_full(net, fw, fh), pts)
        assert (net >= -1e-9).all() and (net[:, 0] <= 512).all() and (net[:, 1] <= 288).all()
    hm = np.zeros((288, 512), np.float32)
    yy, xx = np.mgrid[0:288, 0:512]
    # grid-symmetric centres (the thresholded centroid is exact there): 100.5 and 50.0
    hm += np.exp(-((xx - 100.5) ** 2 + (yy - 50.0) ** 2) / 4.0) * 0.9
    hm += np.exp(-((xx - 300.0) ** 2 + (yy - 200.0) ** 2) / 1.0) * 0.6     # less mass
    (x, y), peak = E.blob_decode(hm, 0.5, 0.5)
    assert abs(x - 101.0) < 1e-6 and abs(y - 50.5) < 1e-6 and 0.84 < peak < 0.85
    assert E.blob_decode(hm * 0.4) is None


# ── training conventions stay consistent with the shipped decode ─────────────

def _train_module():
    try:
        import train_detector as T
        T.det()
    except Exception as e:  # noqa: BLE001 - detector.py owned by a sibling agent
        pytest.skip(f"train_detector / detector import failed: {e!r}")
    return T


@pytest.mark.parametrize("arch", ["wasb", "tracknetv3"])
def test_targets_match_upstream_and_decode_unbiased(arch):
    T = _train_module()
    a = T.ARCHS[arch]
    T.check_conventions(a)
    rng = np.random.default_rng(0)
    clip = T.CachedClip("m/c", Path("."), 1, 1280, 720, 30.0, np.zeros((1, 3)), False, a.corner)
    errs = []
    for _ in range(300):
        x, y = rng.uniform(20, 1260), rng.uniform(20, 700)
        clip.labels[0] = (1, x, y)
        t = T.target_xy(clip.content_xy(0), a, "official")
        if arch == "wasb":     # dataset_loader.py: affine_transform(...).astype(np.int32)
            up = tuple(float(np.int32(v)) for v in clip.corner_geom.to_net(x, y))
        else:                  # dataset.py: int(x / w_scaler), w_scaler = 1280 / 512
            up = (float(int(x / 2.5)), float(int(y / 2.5)))
        assert t == up
        d = E.blob_decode(T.draw_target(*t), 0.5, -a.peak_offset)
        errs.append(d[0] - clip.content_xy(0))
    errs = np.asarray(errs)
    assert np.abs(errs.mean(0)).max() < 0.05          # offset removes the floor bias
    assert np.abs(errs).max() <= 0.5 + 1e-6


def test_wasb_offset_equals_coreml_centroid_offset():
    T = _train_module()
    assert -T.ARCHS["wasb"].peak_offset == E.CARDS["wasb"].centroid_offset == 0.5


def test_augmentation_moves_points_with_pixels():
    T = _train_module()
    torch = pytest.importorskip("torch")
    yy, xx = np.mgrid[0:288, 0:512]
    rng = np.random.default_rng(3)
    for i in range(8):
        p = np.array([[rng.uniform(80, 430), rng.uniform(60, 230)], [np.nan, np.nan]])
        img = np.exp(-((xx - p[0, 0]) ** 2 + (yy - p[0, 1]) ** 2) / 8.0).astype(np.float32)
        x = torch.from_numpy(img)[None, None].repeat(2, 3, 1, 1)
        cfg = T.AugConfig(jitter_p=0.0, hflip_p=float(i % 2), crop_p=float(i >= 4))
        xa, pa = T.augment(x, p, cfg, random.Random(i))
        im = xa[0, 0].numpy().astype(np.float64)
        im[im < 0.05] = 0
        c = np.array([(im * xx).sum() / im.sum(), (im * yy).sum() / im.sum()])
        assert np.hypot(*(c - pa[0])) < 0.1
        assert np.isnan(pa[1]).all()


def test_held_out_test_split_is_refused(tmp_path):
    T = _train_module()
    for split in ("Train", "Test"):
        d = tmp_path / split / "match1"
        (d / "csv").mkdir(parents=True)
        (d / "video").mkdir()
        (d / "csv" / "c_ball.csv").write_text("Frame,Visibility,X,Y\n0,1,10,10\n")
        (d / "video" / "c.mp4").write_bytes(b"")
    assert [c.held_out for c in T.discover([tmp_path], allow_held_out=False)] == [False]
    assert sorted(c.held_out for c in T.discover([tmp_path], allow_held_out=True)) == [False, True]


# ── numeric parity of the shipped package (one real forward, ~2 s) ────────────

def test_package_matches_pytorch_on_noise():
    ct = _coremltools()
    _weights_or_skip()
    if not PACKAGE.exists():
        pytest.skip("ShuttleTracker.mlpackage not built")
    torch = pytest.importorskip("torch")
    card = E.CARDS["wasb"]
    wrapper = E.OnDeviceWrapper(E.load_net(card), card).eval()
    x = np.random.default_rng(0).random((1, 9, 288, 512), dtype=np.float32)
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(x)).numpy()
    ml = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_ONLY)
    got = np.asarray(ml.predict({"frames": x})["heatmap"])
    assert got.shape == ref.shape == (1, 3, 288, 512)
    assert float(np.abs(got - ref).max()) < 2e-2          # fp16 package vs fp32 PyTorch
