"""Synthetic ground truth (sim.py): physics, cameras, rally logic, noise, video."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.tracking import court_model as cm
from app.services.tracking import physics
from app.services.tracking import sim

SEEDS = range(8)


@pytest.fixture(scope="module")
def rallies() -> list[sim.SimRally]:
    out = [sim.simulate_rally(s, doubles=(s % 4 == 3)) for s in SEEDS]
    out += [sim.simulate_rally(100 + s, n_shots=1) for s in range(6)]      # serve-only finals
    return out


def test_deterministic():
    a, b = sim.simulate_rally(5), sim.simulate_rally(5)
    np.testing.assert_array_equal(a.shuttle_world, b.shuttle_world)
    assert [h["frame"] for h in a.hits] == [h["frame"] for h in b.hits]


def test_simulate_positions_matches_physics():
    p0, v0 = np.array([-5.0, 1.0, 2.7]), np.array([60.0, -5.0, -8.0])
    ts = np.linspace(0, 1.2, 37)
    np.testing.assert_allclose(sim.simulate_positions(p0, v0, ts, 4.5),
                               physics.simulate(p0, v0, ts, 4.5), atol=1e-9)


def test_projection_consistent_with_camera(rallies):
    for s in rallies:
        m = s.in_play
        np.testing.assert_allclose(s.shuttle_px[m], s.camera.project(s.shuttle_world[m]), atol=1e-6)
        assert not np.isfinite(s.shuttle_px[~m]).any()
        w, h = s.image_size
        v = s.visible
        assert np.all((s.shuttle_px[v] >= 0) & (s.shuttle_px[v] < [w, h]))
        assert not (s.visible & s.occluded).any()
        for p in s.players.values():               # feet pixel inside the player box
            fpx = s.camera.project(np.column_stack([p["foot"], np.zeros(s.n)]))
            ok = np.all(np.isfinite(fpx), axis=1)
            b = p["bbox"][ok]
            assert np.all((fpx[ok, 0] >= b[:, 0] - 1e-6) & (fpx[ok, 0] <= b[:, 2] + 1e-6))
            assert np.all(b[:, 3] - b[:, 1] > 0)


def test_shots_land_where_intended(rallies):
    n = 0
    for s in rallies:
        for sh in s.shots:
            if sh.in_out == "net":
                continue
            err = np.hypot(sh.landing_xy[0] - sh.target_xy[0], sh.landing_xy[1] - sh.target_xy[1])
            assert err < 0.05, (s.seed, sh.index, sh.shot_type, err)
            n += 1
    assert n > 30


def test_trajectory_is_the_drag_ode(rallies):
    s = rallies[0]
    for sh in s.shots:
        fr = np.arange(sh.frame, sh.end_frame)
        fr = fr[s.timestamps[fr] >= sh.t0]
        exp = physics.simulate(sh.p0, sh.v0, s.timestamps[fr] - sh.t0, sh.drag_length)
        np.testing.assert_allclose(s.shuttle_world[fr], exp, atol=1e-6)
    assert all(abs(sh.drag_length / physics.DRAG_LENGTH - 1) <= 0.0801 for r in rallies for sh in r.shots)


def test_terminal_velocity():
    L = physics.DRAG_LENGTH
    P = sim.simulate_positions(np.array([0.0, 0.0, 60.0]), np.zeros(3), np.array([4.0, 4.01]), L)
    assert abs(np.linalg.norm(P[1] - P[0]) / 0.01 - physics.TERMINAL_VELOCITY) < 0.05
    # deep clears/lifts come down steeply at close to terminal speed
    for s in [sim.simulate_rally(k) for k in range(12)]:
        for sh in s.shots:
            if sh.shot_type in ("clear", "lift") and np.isfinite(sh.landing_t):
                tl = sh.landing_t - sh.t0
                P = sim.simulate_positions(sh.p0, sh.v0, np.array([tl - 0.02, tl - 0.01]), sh.drag_length)
                v = np.linalg.norm(P[1] - P[0]) / 0.01
                vt = np.sqrt(physics.G * sh.drag_length)
                assert 0.5 * vt < v < 1.15 * vt


def test_shot_parameters_realistic():
    smash, clears, serves = [], [], []
    for k in range(30):
        for sh in sim.simulate_rally(k).shots:
            if sh.shot_type == "smash":
                smash.append((sh.speed0_kmh, sh.p0[2]))
            if sh.shot_type == "clear":
                clears.append(sh.speed0_kmh)
            if sh.shot_type == "serve":
                serves.append(sh.p0[2])
    sp = np.array([s for s, _ in smash])
    assert len(sp) >= 5 and np.all((sp > 140) & (sp <= 421)) and np.median(sp) > 200
    assert np.all(np.array([z for _, z in smash]) > 1.9)
    assert np.all((np.array(clears) > 50) & (np.array(clears) < 300))
    assert np.all(np.array(serves) <= 1.15)


def test_hits_alternate_and_rally_truth(rallies):
    for s in rallies:
        sides = [h["hitter"] for h in s.hits]
        assert all(a != b for a, b in zip(sides, sides[1:]))
        ts = [h["t"] for h in s.hits]
        assert all(b > a for a, b in zip(ts, ts[1:]))
        assert s.shots[0].shot_type == "serve"
        held = {f: p for a, b, p in s.resting for f in range(a, b + 1)}
        for h in s.hits:                           # contact lies on the shuttle trajectory
            p = s.shuttle_world[h["frame"]]
            p = p if np.isfinite(p).all() else held[h["frame"]]      # serve frame just before contact
            d = np.linalg.norm(p - h["pos"])
            assert d < 60 / s.fps + 0.05
        r = s.rallies[0]
        last = s.shots[-1]
        assert r.end_reason == last.outcome == last.in_out
        if r.end_reason in ("in", "out"):
            assert (r.margin >= 0) == (r.end_reason == "in")
            assert r.winner == (last.hitter if r.end_reason == "in" else ("far" if last.hitter == "near" else "near"))
        else:
            assert r.margin is None and last.net_crossing_height < cm.NET_HEIGHT_CENTRE
        for sh in s.shots[:-1]:                    # returned shots clear the net
            assert sh.outcome == "returned" and sh.net_crossing_height > cm.NET_HEIGHT_CENTRE
        if r.end_reason == "net":                  # dead shuttle drops off the net, then rests
            after = np.nonzero(s.in_play[r.end_frame + 2:])[0]
            assert len(after) == 0 or after[-1] < 1.5 * s.fps
        else:
            assert not s.in_play[r.end_frame + 2:].any()


def test_serve_margin():
    # server near (court +1 → stands at y>0), receiver's box: far half, y<0
    assert sim.serve_margin(3.0, -1.0, 1.0, 1.0, doubles=False) > 0
    assert sim.serve_margin(1.5, -1.0, 1.0, 1.0, doubles=False) < 0        # short
    assert sim.serve_margin(3.0, 1.0, 1.0, 1.0, doubles=False) < 0         # wrong box
    assert sim.serve_margin(6.3, -1.0, 1.0, 1.0, doubles=False) > 0
    assert sim.serve_margin(6.3, -1.0, 1.0, 1.0, doubles=True) < 0         # doubles long line


def test_n_shots_and_doubles():
    s = sim.simulate_rally(3, n_shots=6, doubles=True)
    assert len(s.shots) <= 6 and set(s.players) == {"near", "far", "near2", "far2"}
    assert len(s.shots) >= 4


@pytest.mark.parametrize("style", ["behind", "elevated_side", "broadcast"])
def test_random_camera_styles(style):
    for k in range(5):
        cam = sim.random_camera(np.random.default_rng(k), (1280, 720), style)
        c = cam.center
        uv = cam.project(np.array([[cm.HALF_LENGTH, 0, 0], [0, 0, cm.NET_HEIGHT_CENTRE]]))
        assert np.all((uv >= 0) & (uv < [1280, 720])), (style, uv)
        assert c[0] < 0 and c[2] > 0.9
        if style == "behind":
            assert 1.0 <= c[2] <= 2.0 and -cm.HALF_LENGTH - 6.01 <= c[0] <= -cm.HALF_LENGTH - 0.99
            assert 0.8 * 1280 <= cam.K[0, 0] <= 1.0 * 1280
        if style == "broadcast":                   # like the real TrackNetV2 clips
            corners = cam.project(sim._CORNERS)
            assert 6.0 <= c[2] <= 10.0 and cam.K[0, 0] > 1.5 * 1280
            assert np.all((corners >= 0) & (corners < [1280, 720]))
            near_w = (corners[1, 0] - corners[0, 0]) if corners[1, 0] > corners[0, 0] else -1
            assert near_w > 0.5 * 1280 or abs(corners[1, 0] - corners[0, 0]) > 0.5 * 1280


def test_corrupt_detections_noise_model():
    s = sim.simulate_match(2, 3, camera=sim.random_camera(1, (1280, 720), "broadcast"), fps=30)
    vis = np.nonzero(s.visible)[0]
    rates, errs = [], []
    for seed in range(3):
        c = sim.corrupt_detections(s, np.random.default_rng(seed), sigma_px=1.5, miss_rate=0.1,
                                   false_positive_rate=0.03)
        assert len(c) == s.n and all(len(x) <= 3 for x in c)
        assert all(all(a.score >= b.score for a, b in zip(x, x[1:])) for x in c)
        d = np.array([min([np.hypot(q.x - s.shuttle_px[i, 0], q.y - s.shuttle_px[i, 1]) for q in c[i]]
                          or [np.inf]) for i in vis])
        rates.append(np.mean(d < 8))
        errs.extend(d[d < 8])
    assert len(vis) > 150 and 0.82 < np.mean(rates) < 0.96
    assert 1.0 < np.median(errs) < 3.5
    clean = sim.corrupt_detections(s, 0, sigma_px=0.0, miss_rate=0.0, false_positive_rate=0.0,
                                   dead_shuttle_rate=0.0)
    assert all((len(clean[i]) == 1) == bool(s.visible[i]) for i in range(s.n))


def test_truth_converters():
    s = sim.simulate_rally(4, fps=30)
    tr = sim.truth_track(s)
    assert tr.n == s.n and np.array_equal(tr.visible, s.visible)
    hits = sim.truth_hits(s)
    assert [h.hitter for h in hits] == [h["hitter"] for h in s.hits]
    segs = sim.truth_segments(s)
    assert len(segs) == len(s.shots) and all(np.allclose(a.v0, b.v0) for a, b in zip(segs, s.shots))
    pl = sim.truth_players(s, every=5)
    assert {p.side for p in pl} == {"near", "far"}
    r = sim.truth_rallies(s)[0]
    assert r.end_reason == s.end_reason and len(r.shots) == len(s.shots)
    px, m = s.image_truth(occluded=True)
    assert (m >= (s.visible | s.occluded)).all() and m.sum() > s.visible.sum()


def test_simulate_match():
    m = sim.simulate_match(7, 3, fps=30)
    assert len(m.rallies) == 3
    assert sum(r.n_shots for r in m.rallies) == len(m.shots)
    for a, b in zip(m.rallies, m.rallies[1:]):
        gap = m.timestamps[b.start_frame] - m.timestamps[a.end_frame]
        assert gap >= 3.0 and not m.in_play[a.end_frame + 3:b.start_frame - 1].any()
        assert m.shots[b.first_shot].hitter == a.winner          # winner serves
    assert m.meta.n_frames == m.n


def test_render_video(tmp_path):
    from app.services.tracking import video_io
    s = sim.simulate_rally(1, camera=sim.random_camera(2, (320, 180), "broadcast"), fps=30, n_shots=2)
    path = sim.render_video(s, str(tmp_path / "r.mp4"))
    meta = video_io.probe(path)
    assert meta.n_frames == s.n and (meta.width, meta.height) == (320, 180)
    frames = dict(video_io.sample_frames(path, 3))
    f = next(iter(frames.values()))
    assert f.shape == (180, 320, 3) and f.std() > 5
