from __future__ import annotations

import json

import numpy as np

from app.services.tracking import physics, serialize
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, RallyV2, ShotEventV2, ShotSegment3D, Track2D, VideoMeta,
)
from app.services.tracking.pipeline_v2 import MatchAnalysis


def _analysis() -> MatchAnalysis:
    cam = CameraModel.look_at([-10.5, 0, 1.6], [0, 0, 1], 1500, (1920, 1080))
    t = np.linspace(0, 0.5, 31)
    P, _ = physics.simulate_state([-4, 0.5, 2.8], [80, -2, -12], t)
    seg = ShotSegment3D(start_frame=10, end_frame=40, t0=0.0, p0=P[0], v0=np.array([80, -2, -12.]),
                        drag_length=physics.DRAG_LENGTH, rmse_px=1.1, frames=np.arange(10, 41),
                        positions=P, landing_xy=(5.1, 0.2), landing_t=0.6,
                        net_crossing_height=1.9, confidence=0.9)
    hit = HitEvent(frame=10, t=0.1667, xy_px=(900.0, 400.0), hitter="near", confidence=0.95)
    shot = ShotEventV2(hit=hit, segment=seg, shot_type="kill", shot_type_confidence=0.8)
    rally = RallyV2(start_frame=10, end_frame=60, shots=[shot], end_reason="in",
                    landing_xy=(5.1, 0.2), in_out_margin_m=1.6, winner="near")
    n = 100
    xy = np.full((n, 2), np.nan)
    xy[10:41] = cam.project(P)
    track = Track2D(t=np.arange(n) / 60, xy=xy, conf=np.ones(n), visible=np.isfinite(xy[:, 0]),
                    interpolated=np.zeros(n, bool))
    meta = VideoMeta(fps=60, width=1920, height=1080, n_frames=n, timestamps=np.arange(n) / 60)
    return MatchAnalysis(meta=meta, track=track, camera=cam, rallies=[rally],
                         diagnostics={"stages": {}})


def test_document_is_json_and_consistent():
    doc = serialize.to_dict(_analysis())
    json.dumps(doc, allow_nan=False)            # strict JSON, no NaN leaks
    assert doc["schema_version"] == 2 and doc["calibrated"]
    shot = doc["rallies"][0]["shots"][0]
    assert shot["legacy_shot_type"] == "smash"
    assert 250 < shot["flight"]["speed0_kmh"] < 320
    assert shot["flight"]["net_clearance_m"] == round(1.9 - 1.524, 2)
    assert all(0 <= u <= 1 and 0 <= v <= 1 for u, v in shot["flight"]["path_norm"])
    assert doc["track"][0]["start_frame"] == 10 and len(doc["track"][0]["xy"]) == 31
    assert doc["summary"]["max_smash_kmh"] == shot["flight"]["speed0_kmh"]


def test_empty_analysis():
    res = _analysis()
    res.rallies, res.track, res.camera = [], None, None
    doc = serialize.to_dict(res)
    json.dumps(doc, allow_nan=False)
    assert doc["summary"]["total_rallies"] == 0 and doc["track"] == [] and not doc["calibrated"]
