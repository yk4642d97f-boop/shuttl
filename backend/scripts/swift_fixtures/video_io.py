"""ShuttlCore VideoIO fixtures: upright frames of rotated files + real presentation clocks.

    cd backend && .venv/bin/python -m scripts.swift_fixtures.video_io

Writes iOS/Packages/ShuttlCore/Tests/ShuttlCoreTests/Fixtures/video_rotation.json:
per corpus file (h264/hevc rot0/90/180/270, real_rot90, real_jump, real_drop): the
Python probe (size, rotation, frame count, timestamps, pts_gaps / pts_gap_info) and
frames 0 and last as 64x36-ish gray thumbnails of the UPRIGHT frame (INTER_AREA), so
the Swift VideoSource is checked for the same orientation and the same clock.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.services.tracking import video_io
from scripts.swift_fixtures.common import REPO, SWIFT_PACKAGES, dump

CORPUS = REPO / "tools/robustness/corpus"
OUT = SWIFT_PACKAGES / "ShuttlCore/Tests/ShuttlCoreTests/Fixtures/video_rotation.json"
CLIPS = [f"{c}_rot{r}.mov" for c in ("h264", "hevc") for r in (0, 90, 180, 270)] + \
        ["real_rot90.mov", "real_jump.mov", "real_drop.mov"]


def thumb(f: np.ndarray, long_side: int = 64) -> dict:
    h, w = f.shape[:2]
    tw, th = (long_side, max(1, round(long_side * h / w))) if w >= h else \
             (max(1, round(long_side * w / h)), long_side)
    g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (tw, th), interpolation=cv2.INTER_AREA)
    return {"width": tw, "height": th, "gray": g.astype(int).ravel().tolist()}


def main() -> None:
    out = []
    for name in CLIPS:
        p = CORPUS / name
        if not p.exists():
            print("skip (missing)", name)
            continue
        m = video_io.probe(str(p))
        frames = dict(video_io.iter_frames(str(p)))
        last = max(frames)
        out.append({"relative_path": str(p.relative_to(REPO)), "width": m.width, "height": m.height,
                    "rotation_deg": m.rotation_deg, "n_frames": m.n_frames, "fps": m.fps,
                    "timestamps": [float(v) for v in m.timestamps],
                    "pts_gaps": video_io.pts_gaps(m.timestamps).tolist(),
                    "pts_gap_info": video_io.pts_gap_info(m.timestamps),
                    "thumbs": {"0": thumb(frames[0]), str(last): thumb(frames[last])}})
        print(f"  {name}: {m.width}x{m.height} rot {m.rotation_deg}, {m.n_frames} fr, gaps {out[-1]['pts_gaps']}")
    dump(out, OUT)
    print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
