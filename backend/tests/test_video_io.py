"""video_io: upright frames (rotation tag applied, like the phone's VideoSource), real
presentation clock (only duplicates re-timed) and the PTS-gap rule."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.tracking import video_io as V

CORPUS = Path(__file__).resolve().parents[2] / "tools/robustness/corpus"
TEST_CLIP = (Path(__file__).resolve().parents[1]
             / "ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_09_06.mp4")


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p.name} not available")
    return p


# ── rotation (F1) ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("codec", ["h264", "hevc"])
@pytest.mark.parametrize("rot,size", [(0, (640, 360)), (90, (360, 640)),
                                      (180, (640, 360)), (270, (360, 640))])
def test_rotated_corpus_is_upright(codec, rot, size):
    p = str(_need(CORPUS / f"{codec}_rot{rot}.mov"))
    m = V.probe(p)
    assert (m.width, m.height) == size and m.rotation_deg == rot and m.n_frames == 12
    # the stored frame turned `rot` degrees clockwise == FFmpeg's own display-matrix rotation
    ref = cv2.VideoCapture(p)
    ref.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    want = [ref.read()[1] for _ in range(12)]
    got = [f for _, f in V.iter_frames(p)]
    assert len(got) == 12 and all(np.array_equal(a, b) for a, b in zip(want, got))
    assert got[0].shape[:2] == (size[1], size[0])
    raw = cv2.VideoCapture(p)
    raw.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    stored = raw.read()[1]
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)
    assert np.array_equal(got[0], stored if code is None else cv2.rotate(stored, code))
    # every reader: sampled (sequential), stepped and seeked frames are upright too
    s = V.sample_frames(p, 3, m.n_frames)
    assert [i for i, _ in s] == [0, 5, 11] and all(f.shape[:2] == (size[1], size[0]) for _, f in s)
    assert [i for i, _ in V.iter_frames(p, start=7, stop=9)] == [7, 8]
    assert np.array_equal(dict(V.iter_frames(p, start=7, stop=9))[7], want[7])


def test_real_portrait_recording_is_upright():
    p = str(_need(CORPUS / "real_rot90.mov"))
    m = V.probe(p)
    assert (m.width, m.height, m.rotation_deg, m.n_frames) == (720, 1280, 90, 133)
    f = next(V.iter_frames(p))[1]
    assert f.shape[:2] == (1280, 720)


def test_unrotated_clip_unchanged():
    p = str(_need(TEST_CLIP))
    m = V.probe(p)
    cap = cv2.VideoCapture(p)
    ts = []
    while cap.grab():
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    assert (m.width, m.height, m.rotation_deg) == (1280, 720, 0)
    assert np.array_equal(m.timestamps, np.asarray(ts))          # bit-identical clock
    assert len(V.pts_gaps(m.timestamps)) == 0


# ── clock ────────────────────────────────────────────────────────────────────

def test_probe_keeps_real_clock_with_gap():
    m = V.probe(str(_need(CORPUS / "real_jump.mov")))
    d = np.diff(m.timestamps)
    assert m.n_frames == 133 and np.all(d > 0)
    assert V.pts_gaps(m.timestamps).tolist() == [66] and d[65] == pytest.approx(5.0333, abs=1e-3)
    m = V.probe(str(_need(CORPUS / "real_drop.mov")))
    assert V.pts_gaps(m.timestamps).tolist() == [60]
    assert np.diff(m.timestamps)[59] == pytest.approx(13 / 30, abs=1e-6)


def test_probe_retimes_duplicate_pts_only():
    m = V.probe(str(_need(CORPUS / "real_duppts.mp4")))
    d = np.diff(m.timestamps)
    assert m.n_frames == 133 and np.all(d > 0)
    # each duplicate sits before a 2-step hole (a frame moved onto its neighbour's PTS):
    # spreading the run evenly to the next distinct time restores the 1/30 s grid
    assert np.allclose(d, 1 / 30, atol=1e-9)
    assert len(V.pts_gaps(m.timestamps)) == 0


def test_retime_duplicates_all_zero_clock_is_nominal():
    t = V.retime_duplicates(np.zeros(5), 1 / 30)
    assert np.allclose(t, np.arange(5) / 30) and np.all(np.diff(t) > 0)


# ── gap rule ─────────────────────────────────────────────────────────────────

def test_pts_gap_rule():
    t = np.arange(100) / 30.0
    assert V.pts_gaps(t).size == 0
    j = t.copy(); j[40:] += 5.0
    assert V.pts_gaps(j).tolist() == [40]
    # threshold: max(0.30 s, 4 x median step)
    for dt, gap in [(0.29, False), (0.2999, False), (0.3001, True), (0.31, True)]:
        u = np.r_[np.arange(50) / 60.0, 49 / 60.0 + dt + np.arange(50) / 60.0]
        assert (V.pts_gaps(u).tolist() == [50]) == gap, dt
    slow = np.arange(40) * 0.1                                   # 10 fps: 4 x 0.1 = 0.4 s
    s2 = slow.copy(); s2[20:] += 0.25                            # 0.35 s step: not a gap
    assert V.pts_gaps(s2).size == 0
    s2[20:] += 0.1                                               # 0.45 s: gap
    assert V.pts_gaps(s2).tolist() == [20]
    # iPhone-like 60 fps VFR with 30 fps stretches (low light) and 34 ms hitches
    rng = np.random.default_rng(0)
    steps = np.where(rng.random(5000) < 0.3, 1 / 30, 1 / 60) + rng.normal(0, 5e-4, 5000)
    steps[rng.integers(0, 5000, 40)] = 0.034
    assert V.pts_gaps(np.r_[0, np.cumsum(steps)]).size == 0
    # degenerate inputs never raise / never fire
    for bad in ([], [0.0], [0.0, 1.0], [0.0, 0.0, 0.0], [0, np.nan, 5.0, 5.1],
                [0, 1 / 30, 2 / 30, np.inf], None):
        assert V.pts_gaps(bad).size == 0, bad


def test_pts_gap_info_keys():
    t = np.r_[np.arange(10) / 30.0, 5 + np.arange(10) / 30.0]
    info = V.pts_gap_info(t)
    assert info == [{"frame": 10, "t_before": t[9], "t_after": t[10], "dt_s": t[10] - t[9]}]
    assert all(type(v) in (int, float) for v in info[0].values())
