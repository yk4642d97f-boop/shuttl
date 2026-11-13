"""pipeline_v2 orchestration with stub stage modules: stage isolation, degraded
paths (no track / no camera), manual-corner calibration, and that the result
always serialises to strict JSON. Real stage modules are tested on their own."""

from __future__ import annotations

import json
import sys
import types

import cv2
import numpy as np
import pytest

import app.services.tracking as tracking_pkg
from app.services.tracking import serialize
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, RallyV2, ShotEventV2, ShotSegment3D, ShuttleCandidate, Track2D,
)
from app.services.tracking.pipeline_v2 import PipelineOptions, analyze_video, player_step

W, H, N = 320, 180, 40
STAGES = ("court_calibration", "detector", "hits", "linker", "players", "rally",
          "reconstruct3d", "shots")


def test_hitnet_success_empty_is_not_rules_fallback(video, stubs, monkeypatch):
    from app.services.tracking import hitnet
    monkeypatch.setattr(hitnet, 'load', lambda *a: types.SimpleNamespace(
        sha256='verified-model', detect=lambda *a: []))
    res = analyze_video(video, PipelineOptions(hitnet_weights='candidate.pt'))
    assert res.hits == [] and res.diagnostics['hit_model']['used'] == 'hitnet_single_v2'
    assert res.diagnostics['stages']['hitnet']['ok']
    assert 'hits' not in res.diagnostics['stages']


def test_hitnet_failure_and_doubles_are_explicit(video, stubs, monkeypatch):
    from app.services.tracking import hitnet
    def fail(*a):
        raise FileNotFoundError('candidate absent')
    monkeypatch.setattr(hitnet, 'load', fail)
    res = analyze_video(video, PipelineOptions(hitnet_weights='missing.pt'))
    assert res.hits and res.diagnostics['hit_model']['used'] == 'rules'
    assert res.diagnostics['hit_model']['fallback_reason'] == 'hitnet_stage_failed'
    assert not res.diagnostics['stages']['hitnet']['ok']
    res = analyze_video(video, PipelineOptions(hitnet_weights='missing.pt',doubles=True))
    assert res.hits and res.diagnostics['hit_model']['fallback_reason'] == 'doubles_not_validated'
    assert 'hitnet' not in res.diagnostics['stages']


@pytest.fixture
def video(tmp_path) -> str:
    path = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i in range(N):
        f = np.full((H, W, 3), 60, np.uint8)
        cv2.circle(f, (20 + 6 * i, 90), 3, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    return path


def _camera() -> CameraModel:
    return CameraModel.look_at([-10.5, 0, 4.0], [0, 0, 0], 300, (W, H))


class _Detector:
    def detect_video(self, path, meta, progress=None):
        if progress:
            progress(1.0)
        return [[ShuttleCandidate(20 + 6 * i, 90, 0.9)] for i in range(meta.n_frames)]


def _stubs(calls: dict) -> dict[str, types.ModuleType]:
    def mod(name, **fns):
        m = types.ModuleType(f"app.services.tracking.{name}")
        for k, fn in fns.items():
            setattr(m, k, fn)
        return m

    def link(cands, meta):
        xy = np.array([[c[0].x, c[0].y] for c in cands], float)
        n = len(cands)
        return Track2D(t=meta.timestamps, xy=xy, conf=np.ones(n), visible=np.ones(n, bool),
                       interpolated=np.zeros(n, bool))

    def calibrate_from_frames(frames, size, doubles_hint=None):
        calls["auto_frames"] = len(frames)
        return _camera()

    def calibrate_from_points(pts, size):
        calls["points"] = pts
        return _camera()

    def refine_camera(cam, frames):
        calls["refined"] = True
        return cam

    def track_players(path, meta, camera, step=2, device=None, doubles=None):
        calls["player_step"] = step
        return [PlayerObservation(frame=0, track_id=1, bbox=np.array([0, 0, 10, 30.]),
                                  score=0.9, side="near", foot_world=np.array([-4.0, 0.0]))]

    def detect_hits(track, meta, camera, players):
        return [HitEvent(frame=f, t=float(meta.timestamps[f]), xy_px=tuple(track.xy[f]),
                         hitter=side, confidence=0.9) for f, side in ((2, "near"), (20, "far"))]

    def reconstruct_segments(track, hits, camera, meta, players):
        P = np.array([[-4.0, 0, 2.5], [4.0, 0, 0.0]])
        return [ShotSegment3D(start_frame=2, end_frame=19, t0=0.0, p0=P[0],
                              v0=np.array([30.0, 0, -3]), drag_length=4.71, rmse_px=0.8,
                              frames=np.array([2, 19]), positions=P, landing_xy=(4.0, 0.0),
                              landing_t=0.6, net_crossing_height=1.8, confidence=0.9)]

    def segment_rallies(track, hits, segments, meta, camera, players=None):
        seg = {s.start_frame: s for s in segments}
        shots = [ShotEventV2(hit=h, segment=seg.get(h.frame)) for h in hits]
        return [RallyV2(start_frame=0, end_frame=meta.n_frames - 1, shots=shots,
                        end_reason="in" if segments else "unknown", winner="far")]

    def score_match(rallies, doubles=False):
        calls["score_doubles"] = doubles
        return [{"near": 0, "far": 1}]

    def detect_doubles(players):
        return None if not players else False

    def classify_shots(rally, camera, players):
        for s in rally.shots:
            s.shot_type, s.shot_type_confidence = "smash", 0.8
        return rally

    return {
        "detector": mod("detector", build_detector=lambda weights_dir, device: _Detector()),
        "linker": mod("linker", link_candidates=link),
        "court_calibration": mod("court_calibration", calibrate_from_frames=calibrate_from_frames,
                                 calibrate_from_points=calibrate_from_points,
                                 refine_camera=refine_camera),
        "players": mod("players", track_players=track_players),
        "hits": mod("hits", detect_hits=detect_hits),
        "reconstruct3d": mod("reconstruct3d", reconstruct_segments=reconstruct_segments),
        "rally": mod("rally", segment_rallies=segment_rallies, score_match=score_match,
                     detect_doubles=detect_doubles),
        "shots": mod("shots", classify_shots=classify_shots),
    }


@pytest.fixture
def stubs(monkeypatch):
    calls: dict = {}
    mods = _stubs(calls)
    for name in STAGES:
        monkeypatch.setitem(sys.modules, f"app.services.tracking.{name}", mods[name])
        monkeypatch.setattr(tracking_pkg, name, mods[name], raising=False)
    return mods, calls


def _stage_ok(res) -> dict[str, bool]:
    return {k: v["ok"] for k, v in res.diagnostics["stages"].items()}


def _strict_json(res) -> dict:
    doc = serialize.to_dict(res)
    json.dumps(doc, allow_nan=False)
    return doc


def test_full_chain(video, stubs):
    res = analyze_video(video)
    assert all(_stage_ok(res).values()) and len(_stage_ok(res)) == 9
    assert res.meta.n_frames == N and res.track is not None and res.camera is not None
    assert [s.shot_type for s in res.rallies[0].shots] == ["smash", "smash"]
    assert res.rallies[0].shots[0].segment is not None and res.score
    doc = _strict_json(res)
    assert doc["calibrated"] and doc["summary"]["total_shots"] == 2


def test_progress_is_monotonic(video, stubs):
    seen: list[float] = []
    analyze_video(video, progress=lambda p, s: seen.append(p))
    assert seen == sorted(seen) and seen[-1] <= 100


def test_detector_failure_is_isolated(video, stubs, monkeypatch):
    mods, _ = stubs

    def boom(weights_dir, device):
        raise RuntimeError("no weights")

    monkeypatch.setattr(mods["detector"], "build_detector", boom)
    res = analyze_video(video)
    ok = _stage_ok(res)
    assert ok["detect"] is False and "no weights" in res.diagnostics["stages"]["detect"]["error"]
    assert ok["calibrate"] and ok["players"] and "hits" not in ok
    assert res.track is None and res.rallies == []
    _strict_json(res)


def test_no_camera_skips_3d(video, stubs, monkeypatch):
    mods, _ = stubs
    monkeypatch.setattr(mods["court_calibration"], "calibrate_from_frames",
                        lambda frames, size, doubles_hint=None: None)
    res = analyze_video(video)
    assert res.camera is None and res.segments == []
    assert "reconstruct3d" not in res.diagnostics["stages"]
    assert res.rallies and res.rallies[0].end_reason == "unknown"
    assert not _strict_json(res)["calibrated"]
    # A calibrator without the diagnostics hook records no evidence (and doesn't crash).
    assert "calibration_auto" not in res.diagnostics


def test_rejected_calibration_records_its_evidence(video, stubs, monkeypatch):
    mods, _ = stubs

    def reject(frames, size, doubles_hint=None, diagnostics=None):
        diagnostics.update(source="median", n_t=2, n_l=5, n_absent=0, explained_frac=0.77,
                           margin=np.float64(0.4), reason="few_transverse", candidates=[("not", "json")])
        return None

    monkeypatch.setattr(mods["court_calibration"], "calibrate_from_frames", reject)
    res = analyze_video(video)
    auto = res.diagnostics["calibration_auto"]
    assert (auto["n_t"], auto["n_l"], auto["explained_frac"]) == (2, 5, 0.77)
    assert auto["reason"] == "few_transverse"
    assert isinstance(auto["margin"], float)
    assert "candidates" not in auto
    assert "calibration" not in res.diagnostics       # only written for an accepted camera
    assert _strict_json(res)["diagnostics"]["calibration_auto"]["n_t"] == 2


def test_accepted_calibration_recorded_and_manual_has_none(video, stubs, monkeypatch):
    mods, _ = stubs

    def accept(frames, size, doubles_hint=None, diagnostics=None):
        diagnostics.update(n_t=5, n_l=5, confidence=0.93)
        return _camera()

    monkeypatch.setattr(mods["court_calibration"], "calibrate_from_frames", accept)
    auto = analyze_video(video).diagnostics["calibration_auto"]
    assert auto["confidence"] == 0.93 and "reason" not in auto
    corners = {"nb_ld": (0.8, 0.9), "nb_rd": (0.2, 0.9), "fb_ld": (0.6, 0.3), "fb_rd": (0.4, 0.3)}
    res = analyze_video(video, PipelineOptions(manual_corners=corners, calibration_frames=4))
    assert "calibration_auto" not in res.diagnostics   # same as the phone: absent for tapped corners


def test_manual_corners_are_scaled_and_refined(video, stubs):
    _, calls = stubs
    corners = {"nb_ld": (0.1, 0.9), "nb_rd": (0.9, 0.9), "fb_ld": (0.3, 0.3), "fb_rd": (0.7, 0.3)}
    res = analyze_video(video, PipelineOptions(manual_corners=corners, calibration_frames=4))
    assert calls["points"]["nb_ld"] == pytest.approx((0.1 * W, 0.9 * H))
    assert calls.get("refined") and "auto_frames" not in calls
    assert res.camera is not None


def test_refine_failure_keeps_manual_camera(video, stubs, monkeypatch):
    mods, _ = stubs

    def bad_refine(cam, frames):
        raise ValueError("lines not found")

    monkeypatch.setattr(mods["court_calibration"], "refine_camera", bad_refine)
    corners = {"nb_ld": (32.0, 162.0), "nb_rd": (288.0, 162.0),
               "fb_ld": (96.0, 54.0), "fb_rd": (224.0, 54.0)}
    res = analyze_video(video, PipelineOptions(manual_corners=corners, corners_normalized=False))
    assert res.camera is not None and _stage_ok(res)["calibrate"]


def test_camera_hint_fallback(video, stubs, monkeypatch):
    mods, _ = stubs
    monkeypatch.setattr(mods["court_calibration"], "calibrate_from_frames",
                        lambda frames, size, doubles_hint=None: None)
    hint = _camera().to_dict()
    res = analyze_video(video, PipelineOptions(camera_hint=hint))
    assert res.camera is not None
    np.testing.assert_allclose(res.camera.K, _camera().K)


@pytest.mark.parametrize("fps,override,expected", [
    (30, None, 2), (29.97, None, 2), (60, None, 4), (59.94, None, 4), (120, None, 8),
    (10, None, 1), (None, None, 2), (60, 2, 2), (60, 0, 1),
])
def test_player_step_is_fps_aware(fps, override, expected):
    assert player_step(fps, override) == expected


def test_pipeline_passes_fps_aware_player_step(video, stubs):
    _, calls = stubs
    analyze_video(video)                          # 30 fps clip → pose at ~15 Hz
    assert calls["player_step"] == 2
    analyze_video(video, PipelineOptions(player_step=3))
    assert calls["player_step"] == 3


def test_bst_requires_validated_hit_and_pose_path(video, stubs):
    res = analyze_video(video,PipelineOptions(bst_manifest='manifest.json'))
    assert res.diagnostics['stroke_model']['fallback_reason']=='validated_hitnet_required'
    assert 'bst' not in res.diagnostics['stages']


def test_bst_and_pose_pipeline_wiring(video, stubs, monkeypatch):
    from app.services.tracking import hitnet,bst,rtmpose
    def learned(track,meta,players):
        return [HitEvent(2,float(meta.timestamps[2]),tuple(track.xy[2]),'far',source='hitnet')]
    monkeypatch.setattr(hitnet,'load',lambda *a:types.SimpleNamespace(sha256='hit-hash',detect=learned))
    monkeypatch.setattr(rtmpose,'load',lambda *a:types.SimpleNamespace(
        sha256='pose-hash',providers=['CPUExecutionProvider'],refine=lambda path,meta,players,hits:players))
    def refine(rallies,track,meta,players):
        assert players and rallies[0].shots[0].hit.source=='hitnet'
        rallies[0].shots[0].shot_type='cross_net'
        return {'used':'BST_CG_AP_seq30','refined':1}
    monkeypatch.setattr(bst,'load',lambda *a:types.SimpleNamespace(refine=refine))
    seen=[]
    res=analyze_video(video,PipelineOptions(hitnet_weights='hit.pt',bst_manifest='bst.json',rtmpose_weights='pose.onnx'),
                      progress=lambda p,s:seen.append(p))
    assert seen==sorted(seen)
    assert res.rallies[0].shots[0].shot_type=='cross_net'
    assert res.diagnostics['stroke_model']['used']=='BST_CG_AP_seq30'
    _strict_json(res)
