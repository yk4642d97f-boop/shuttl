"""End-to-end reference for iOS/Packages/ShuttlLive: the whole pipeline_v2 on real
Test clips with user-style corner taps, so Swift can run the same video with the
same taps and be compared stage by stage.

Taps = the four outer court corners projected through Python's automatically
calibrated camera (what a careful user would tap). Clips where automatic
calibration fails are skipped.

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.live_e2e [n_clips]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from app.services.tracking import court_calibration, video_io
from app.services.tracking import serialize
from app.services.tracking.pipeline_v2 import PipelineOptions, analyze_video
from scripts.swift_fixtures.common import REPO, SWIFT_PACKAGES, TEST_SPLIT, dump

OUT = SWIFT_PACKAGES / "ShuttlLive/Tests/ShuttlLiveTests/Fixtures"
CORNERS = ("nb_ld", "nb_rd", "fb_ld", "fb_rd")


def corner_taps(path: str) -> tuple[dict[str, list[float]], dict] | None:
    meta = video_io.probe(path)
    frames = [f for _, f in video_io.sample_frames(path, 24, meta.n_frames)]
    cam = court_calibration.calibrate_from_frames(frames, (meta.width, meta.height))
    if cam is None:
        return None
    from app.services.tracking import court_model as cm
    pts = np.array([[*cm.COURT_KEYPOINTS[k], 0.0] for k in CORNERS])
    uv = cam.project(pts)
    if not np.all(np.isfinite(uv)):
        return None
    taps = {k: [float(u / meta.width), float(v / meta.height)] for k, (u, v) in zip(CORNERS, uv)}
    return taps, cam.to_dict()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    clips = sorted(TEST_SPLIT.glob("match*/video/*.mp4"), key=lambda p: p.stat().st_size)
    cases = []
    for path in clips:
        if len(cases) >= n:
            break
        got = corner_taps(str(path))
        if got is None:
            print("skip (no auto camera):", path.name)
            continue
        taps, auto_cam = got
        t0 = time.perf_counter()
        res = analyze_video(str(path), PipelineOptions(manual_corners=taps, corners_normalized=True))
        secs = time.perf_counter() - t0
        doc = serialize.to_dict(res)
        cases.append({
            "relative_path": str(path.relative_to(REPO)),
            "taps": taps,
            "auto_camera": auto_cam,
            "python_seconds": secs,
            "camera": res.camera.to_dict() if res.camera is not None else None,
            "n_visible": int(np.sum(res.track.visible)) if res.track is not None else 0,
            "hits": [{"frame": h.frame, "hitter": h.hitter} for h in res.hits],
            "n_segments": len(res.segments),
            "rallies": [{"start_frame": r.start_frame, "end_frame": r.end_frame, "end_reason": r.end_reason,
                         "winner": r.winner, "n_shots": len(r.shots),
                         "shot_types": [s.shot_type for s in r.shots]} for r in res.rallies],
            "summary": doc["summary"],
            "n_players": len(res.players),
            "stages": res.diagnostics.get("stages", {}),
        })
        print(f"{path.name}: {len(res.hits)} hits, {len(res.rallies)} rallies, {secs:.1f}s")
    dump({"cases": cases}, OUT / "live_e2e.json")
    print("wrote", OUT / "live_e2e.json", len(cases), "cases")


if __name__ == "__main__":
    main()
