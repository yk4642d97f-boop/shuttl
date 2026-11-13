"""Tests for the portable learned hit re-scorer artifact and data path."""

from __future__ import annotations

import math

import numpy as np

from app.services.tracking import hits
from app.services.tracking.data_types import Track2D, VideoMeta
from ml import hit_rescorer_data as D


def _real_candidates():
    clip = D.real_test_clips()[0]
    meta = D._probe(clip["video"])
    xy, visible = D.load_tracknet_csv(clip["labels"], meta.n_frames)
    track = Track2D(t=meta.timestamps, xy=xy, conf=visible.astype(float),
                    visible=visible, interpolated=np.zeros(meta.n_frames, bool))
    candidates = hits.detect_candidates(track, meta)
    return candidates, math.hypot(meta.width, meta.height)


def test_real_candidate_features_follow_hit_features_order():
    candidates, diag = _real_candidates()
    assert len(candidates) >= 3
    assert len(hits.HIT_FEATURES) == 13
    for c in candidates[:5]:
        expected = np.array([
            math.log(max(c.lam, 1e-6)),
            math.log(max(c.lam_v, 1e-6)),
            np.linalg.norm(c.v_in) / diag,
            np.linalg.norm(c.v_out) / diag,
            math.log((np.linalg.norm(c.v_out) / diag + 1e-3) /
                     (np.linalg.norm(c.v_in) / diag + 1e-3)),
            np.dot(c.v_in, c.v_out) /
            (np.linalg.norm(c.v_in) * np.linalg.norm(c.v_out) + 1e-12),
            float(np.sign(c.v_in[1]) != np.sign(c.v_out[1])),
            float(c.static_before),
            float(c.static_after),
            float(c.outside),
            c.gap_s,
            c.reach,
            float(c.is_start),
        ])
        assert tuple(hits.HIT_FEATURES) == D.FEATURES
        np.testing.assert_allclose(c.feats, expected, rtol=1e-7, atol=1e-9)


def test_npz_loader_with_and_without_standardization(tmp_path):
    n = len(hits.HIT_FEATURES)
    with_stats = tmp_path / "with_stats.npz"
    without_stats = tmp_path / "without_stats.npz"
    np.savez(with_stats, w=np.ones(n), b=2.0, mu=np.arange(n), sd=np.full(n, 2.0))
    np.savez(without_stats, w=np.ones(n), b=2.0)
    X = np.arange(2 * n, dtype=float).reshape(2, n)
    try:
        assert hits.load_learned_hit_model(str(with_stats))
        np.testing.assert_allclose(hits._LEARNED(X), ((X - np.arange(n)) / 2.0).sum(1) + 2.0)
        assert hits.load_learned_hit_model(str(without_stats))
        np.testing.assert_allclose(hits._LEARNED(X), X.sum(1) + 2.0)
    finally:
        hits.load_learned_hit_model(None)


def test_exported_npz_loads_and_has_runtime_shape():
    path = D.DEFAULT_TEST_ROOT.parents[3] / "ml/weights/hit_rescorer.npz"
    # The repository artifact is the acceptance deliverable; this test gives a
    # direct failure if it is missing or no longer matches the runtime contract.
    assert path.exists(), path
    data = np.load(path)
    assert data["w"].shape == (len(hits.HIT_FEATURES),)
    assert data["mu"].shape == data["w"].shape
    assert data["sd"].shape == data["w"].shape
    try:
        assert hits.load_learned_hit_model(str(path))
    finally:
        hits.load_learned_hit_model(None)


def test_all_zero_model_is_harmless(tmp_path):
    from tests.test_hits import _arc_track

    track, meta = _arc_track(noise=1.0, seed=4)
    baseline = hits.detect_hits(track, meta)
    path = tmp_path / "zero.npz"
    np.savez(path, w=np.zeros(len(hits.HIT_FEATURES)), b=0.0)
    try:
        assert hits.load_learned_hit_model(str(path))
        zero = hits.detect_hits(track, meta)
        assert len(zero) == len(baseline)
        assert [h.frame for h in zero] == [h.frame for h in baseline]
    finally:
        hits.load_learned_hit_model(None)
