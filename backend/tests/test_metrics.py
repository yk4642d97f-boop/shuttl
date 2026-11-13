"""Evaluation metrics (metrics.py): perfect input → perfect scores, known counts."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.tracking import metrics as M
from app.services.tracking.data_types import HitEvent, ShuttleCandidate, Track2D


def _truth(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=float)
    xy = np.column_stack([100 + 5 * t, 300 - 0.2 * (t - 25) ** 2])
    vis = np.ones(n, bool)
    vis[[3, 4, 30]] = False
    xy[~vis] = np.nan
    return xy, vis


def test_scaled_tolerance():
    assert M.scaled_tolerance(4.0, (1280, 720)) == pytest.approx(10.0)
    assert M.scaled_tolerance(4.0, (1920, 1080)) == pytest.approx(15.0)


def test_detection_perfect_and_counts():
    xy, vis = _truth()
    r = M.detection_metrics(xy, xy, vis)
    assert r["precision"] == r["recall"] == r["f1"] == r["accuracy"] == 1.0
    assert r["tp"] == 47 and r["tn"] == 3 and r["err_median"] == 0.0
    p = xy.copy()
    p[10] += [20.0, 0.0]                  # FP1 (too far)
    p[11] = np.nan                        # FN
    p[3] = [5.0, 5.0]                     # FP2 (nothing there)
    p[12] += [3.0, 4.0]                   # 5 px: TP at tol 10
    r = M.detection_metrics(p, xy, vis, tol_px=10)
    assert (r["tp"], r["fp1"], r["fp2"], r["fn"], r["tn"]) == (45, 1, 1, 1, 2)
    assert r["precision"] == pytest.approx(45 / 47) and r["recall"] == pytest.approx(45 / 46)
    assert M.detection_metrics(p, xy, vis, tol_px=4)["tp"] == 44
    m = np.zeros(len(xy), bool)
    m[20:40] = True
    assert M.detection_metrics(p, xy, vis, mask=m)["n_frames"] == 20


def test_detection_tracknet_int_rule():
    true = np.array([[640.0, 360.0]])
    # 4 net px at 512×288 == 10 full px at 1280×720 (x)
    ok = M.detection_metrics(np.array([[649.9, 360.0]]), true, tracknet_int=True,
                             image_size=(1280, 720), tol_px=4.0)
    bad = M.detection_metrics(np.array([[652.6, 360.0]]), true, tracknet_int=True,
                              image_size=(1280, 720), tol_px=4.0)
    assert ok["tp"] == 1 and bad["fp1"] == 1


def test_candidates_to_xy_threshold():
    c = [[ShuttleCandidate(1, 2, 0.9), ShuttleCandidate(5, 5, 0.95)], [], [ShuttleCandidate(3, 3, 0.3)]]
    xy, sc = M.candidates_to_xy(c, 0.5)
    assert np.allclose(xy[0], [5, 5]) and np.isnan(xy[1]).all() and np.isnan(xy[2]).all()
    assert sc[2] == pytest.approx(0.3)
    r = M.detection_metrics(c, np.array([[5, 5], [np.nan, np.nan], [3, 3]], float))
    assert (r["tp"], r["tn"], r["fn"]) == (1, 1, 1)


def test_track_metrics():
    xy, vis = _truth()
    tr = Track2D(t=np.arange(50.0), xy=xy.copy(), conf=vis.astype(float), visible=vis,
                 interpolated=np.zeros(50, bool))
    r = M.track_metrics(tr, xy, vis)
    assert r["f1"] == 1.0 and r["jitter_px"] == 0.0 and r["jumps"] == 0
    p = xy.copy()
    p[20:23] += [200.0, 0.0]              # one excursion onto something else
    p[40] += [0.0, 1.0]                   # small noise → jitter
    r = M.track_metrics(p, xy, vis, tol_px=10)
    assert r["jumps"] == 1 and r["wrong_frames"] == 3
    assert 0 < r["jitter_px"] < 2.0
    filled = xy.copy()
    filled[[3, 4]] = [[115, 200], [120, 200]]
    tr2 = Track2D(t=np.arange(50.0), xy=filled, conf=vis.astype(float), visible=vis,
                  interpolated=np.isin(np.arange(50), [3, 4]))
    r2 = M.track_metrics(tr2, xy, vis)
    assert r2["fp2"] == 2 and r2["interp_frames"] == 2


def test_hit_metrics():
    true = [{"frame": 10, "hitter": "near"}, {"frame": 50, "hitter": "far"}, {"frame": 90, "hitter": "near"}]
    perfect = [HitEvent(frame=h["frame"], t=h["frame"] / 30, xy_px=(0, 0), hitter=h["hitter"]) for h in true]
    r = M.hit_metrics(perfect, true)
    assert r["f1"] == 1.0 and r["offset_abs_mean"] == 0.0 and r["hitter_acc"] == 1.0
    pred = [{"frame": 12, "hitter": "near"}, {"frame": 11, "hitter": "far"},
            {"frame": 49, "hitter": "unknown"}, {"frame": 70, "hitter": "far"}]
    r = M.hit_metrics(pred, true, tol_frames=3)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 1)       # one-to-one: 11 ↔ 10 (closest)
    assert r["offset_abs_mean"] == pytest.approx(1.0)
    assert r["hitter_acc"] == 0.0 and r["hitter_acc_known"] == 0.0   # 'unknown' excluded


def test_segment_and_landing_metrics():
    shots = [{"start_frame": 10, "p0": np.array([-5.0, 0, 2.8]), "v0": np.array([80.0, 0, -10]),
              "net_crossing_height": 2.0, "landing_xy": (3.0, 1.0)},
             {"start_frame": 60, "p0": np.array([4.0, 1, 1.0]), "v0": np.array([-20.0, 0, 5]),
              "net_crossing_height": 1.8, "landing_xy": None}]
    r = M.segment_metrics(shots, shots)
    assert r["recall"] == 1.0 and r["speed_err_kmh_median"] == 0.0 and r["landing_err_m_n"] == 1
    pred = [dict(shots[0], v0=np.array([80.0, 0, -10]) * 1.1, landing_xy=(3.3, 1.4), start_frame=12)]
    r = M.segment_metrics(pred, shots)
    assert r["matched"] == 1 and r["recall"] == 0.5
    assert r["speed_err_kmh_median"] == pytest.approx(0.1 * np.linalg.norm([80, 0, -10]) * 3.6)
    assert r["landing_err_m_median"] == pytest.approx(0.5)
    assert M.landing_metrics([(0, 0), None], [(0, 1), (2, 2)])["landing_err_m_n"] == 1


def test_in_out_metrics():
    r = M.in_out_metrics(["in", "out", "net", "unsure"], ["in", "in", "net", "out"], [0.05, 0.1, None, -0.5])
    assert r["accuracy"] == pytest.approx(0.5) and r["accuracy_decided"] == pytest.approx(2 / 3)
    assert r["n_close"] == 2 and r["accuracy_close"] == pytest.approx(0.5)
    M_ = np.array(r["confusion"]["matrix"])
    assert M_[0, 0] == 1 and M_[0, 1] == 1 and M_[1, 3] == 1
    assert M.in_out_metrics(["in", "out"], ["in", "out"])["accuracy"] == 1.0


def test_shot_type_metrics():
    r = M.shot_type_metrics(["smash", "clear", "drop"], ["smash", "clear", "net"])
    assert r["accuracy"] == pytest.approx(2 / 3) and r["per_class_recall"]["net"] == 0.0
    assert M.shot_type_metrics(["a", "b"], ["a", "b"])["accuracy"] == 1.0


def test_rally_metrics():
    true = [{"start_frame": 0, "end_frame": 100, "end_reason": "in", "winner": "near",
             "landing_xy": (1.0, 2.0), "margin": 0.1},
            {"start_frame": 300, "end_frame": 400, "end_reason": "net", "winner": "far",
             "landing_xy": None, "margin": None}]
    r = M.rally_metrics(true, true)
    assert r["f1"] == r["iou_mean"] == r["end_reason_acc"] == r["winner_acc"] == 1.0
    assert r["accuracy_close"] == 1.0 and r["landing_err_m_median"] == 0.0
    pred = [dict(true[0], start_frame=10, end_reason="out", winner="far"),
            {"start_frame": 150, "end_frame": 200, "end_reason": "in", "winner": "near"}]
    r = M.rally_metrics(pred, true)
    assert (r["tp"], r["n_pred"], r["n_true"]) == (1, 2, 2)
    assert r["iou_mean"] == pytest.approx(91 / 101)
    assert r["end_reason_acc"] == 0.0 and r["winner_acc"] == 0.0 and r["accuracy_close"] == 0.0


def test_format_table():
    s = M.format_table({"linker": {"f1": 0.9, "jumps": 2}, "hits": {"skipped": "no module"}})
    assert "f1=0.900" in s and "jumps=2" in s and "SKIPPED" in s


def test_perfect_sim_scores():
    from app.services.tracking import sim
    s = sim.simulate_match(3, 2, fps=30)
    tr = sim.truth_track(s)
    assert M.track_metrics(tr, s.shuttle_px, s.visible)["f1"] == 1.0
    assert M.hit_metrics(sim.truth_hits(s), s.hits)["f1"] == 1.0
    assert M.segment_metrics(sim.truth_segments(s), s.shots)["speed_err_kmh_median"] == 0.0
    r = M.rally_metrics(sim.truth_rallies(s), s.rallies)
    assert r["f1"] == 1.0 and r["end_reason_acc"] == 1.0 and r["winner_acc"] == 1.0
    types = [h["shot_type"] for h in s.hits]
    assert M.shot_type_metrics(types, [sh.shot_type for sh in s.shots])["accuracy"] == 1.0
