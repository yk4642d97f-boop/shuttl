# Feature extraction from ShuttleSet with the phone's own analysis code

Status 2026-10-05.  Everything here runs on the Mac (Apple M4, 24 GB) from
`research/extract/`.  The point is that the training features come from the *same*
Swift code the iPhone runs (TrackNetV3 detection + linker + ShuttlCourt automatic
calibration + Apple Vision pose + the player tracker), so nothing has to be
re-implemented at inference time and there is no train/serve mismatch.

## 1. What runs

```
research/extract/
  Sources/extract/main.swift          CLI: --video --labels --out [--cache --pose-step --max-rallies]
  Sources/extract/Labels.swift        reads ShuttleSet set{1,2,3}.csv, builds rally spans
  Sources/extract/RallyExtractor.swift per-rally detect + link + calibrate + pose + track
  Sources/extract/Output.swift        writes one .npz per match
  Sources/extract/Court.swift, NPZ.swift
  drive.py                            worker pool; extracts every match whose video is complete
  validate.py                         compares an .npz against the ShuttleSet labels
  out/NNN.npz                         the extracted features (NNN = manifest match id)
  extract.log                         progress
```

It links the existing packages by path: ShuttlCore, ShuttlDetector, ShuttlLinker,
ShuttlCourt, ShuttlPlayers.  No existing package or app file was modified.

Build and run:

```bash
cd research/extract && swift build -c release
python3 drive.py --workers 3            # all complete matches, 3 at a time
python3 drive.py --only 7               # a single match
```

`drive.py` re-reads `manifest.csv` every pass, so matches that finish downloading
(~0.4 MB/s from the Seoul relay, about 14 h for all 21.8 GB) get picked up
automatically.  Per-rally results are cached in `out/NNN.parts/*.plist`, so stopping
and restarting never repeats a finished rally.  Each worker keeps its own CoreML
cache (`--cache .mlcache-wN`) so the compiled model is not recompiled per match.

## 2. Output format (`out/NNN.npz`)

One file per match, everything in the video's own 1280x720 pixel frame, y down from
the top-left, unless stated otherwise.  Read it with `numpy`; scalars are 0-d arrays.

| key | shape / dtype | meaning |
|---|---|---|
| `frame` | (N,) int64 | video frame number (the same convention as the labels' `frame_num`) |
| `t` | (N,) float64 | timestamp, seconds from the video start |
| `rally_index` | (N,) int32 | index into the per-rally arrays below |
| `shuttle_xy` | (N,2) float32 | linked shuttle position, px; NaN where none |
| `shuttle_visible` | (N,) bool | a detection was accepted this frame |
| `shuttle_interp` | (N,) bool | position interpolated across a short gap |
| `shuttle_conf` | (N,) float32 | linker confidence |
| `shuttle_cand` | (N,3,3) float32 | top-3 raw detector candidates: x, y, score; NaN padded |
| `personStart` | (N+1,) int64 | offsets into the per-frame person arrays |
| `person_kp` | (P,19,3) float32 | every Vision pose in the frame: 19 joints, x, y, confidence |
| `person_box` | (P,4) float32 | x1, y1, x2, y2 |
| `person_score` | (P,) float32 | |
| `person_source` | (P,) int8 | 0 = full frame, i>0 = far-court crop pass i |
| `player_person` | (N,2) int64 | index into `person_*` for far/near, −1 if none |
| `player_track` | (N,2) int32 | tracker id, −1 if none |
| `player_box` | (N,2,4) float32 | |
| `player_kp` | (N,2,17,3) float32 | tracker keypoints, COCO-17 order, px + confidence |
| `player_foot_world` | (N,2,2) float32 | ankle midpoint on the floor, metres, our world frame |
| `rally_set`, `rally_id` | (R,) int32 | the label's set number and rally number |
| `rally_span` | (R,2) int64 | first/last video frame of the rally span (first hit − 2 s … last hit + 2 s) |
| `rally_row_start` | (R+1,) int64 | offsets into the per-frame arrays |
| `rally_hit_frames` | (H,) int64 | every labelled hit's video frame |
| `rally_hit_start` | (R+1,) int64 | which hit belongs to which rally |
| `rally_cam_ok`, `rally_cam_reason`, `rally_cam_conf`, `rally_cam_diff_px` | (R,) | per-rally calibration outcome, and its median keypoint difference (px) from the match-level camera |
| `rally_cam_K/R/t/H` | (R,3,3)/(R,3,3)/(R,3)/(R,3,3) | per-rally camera |
| `rally_error` | (R,) str | failure reason, empty on success |
| `rally_timing` | (R,6) float64 | seconds per stage |
| `rally_timing_keys` | (6,) str | calibrate, detect, link, pose, track, total |
| `cam_ok`, `cam_K/R/t/dist/H`, `cam_frames` | — | match-level automatic calibration and the frames it used |
| `joint_names` | (19,) str | Vision joint order |
| `fps`, `width`, `height`, `n_video_frames` | scalar | video metadata |
| `info_json` | (1,) str | JSON: creation time, detector, label path, match-camera confidence/coverage/rmse, pose step |

World frame (`cam_*`, and `player_foot_world`): origin at the court centre under the
net, `+x` towards the far baseline, `+y` to the left as seen from the near baseline,
`+z` up.  Identical to `CourtModel` in ShuttlCore and to the Python stack.

## 3. Validation against the labels (match 1, 3 rallies, 1117 frames)

`validate.py out/NNN.npz` reports the following.  Numbers below are from the smoke
run on match 1 (Kento Momota vs Chou Tien Chen, Fuzhou Open 2019 final, 25 fps).

**Player positions: good.**  At the 35 labelled hit frames, our near/far ankle
midpoints sit a median 19.3 px from the label's `player_location_x/y` (p10 14, p90
37), i.e. about 1.5 % of the 1280 px width.  This confirms the pixel convention and
the frame alignment: the labels really are in the 1280x720 frame and our frame
numbering matches theirs.

**Court calibration: good.**  The match-level automatic calibration is accepted
(`reason=ok`, `confidence 0.92`, `rmse 0.43`) and the per-rally calibrations agree
with it to a median 0.02 px.

**Shuttle: the first comparison was mine, not the data's, that was wrong.**  Comparing
our airborne track with the labels' `hit_x/hit_y` gives a median 185 px — but
`hit_x/hit_y` is not the airborne shuttle.  Two independent proofs:

- It duplicates the *previous* shot's landing point.  Row `rally 2, ball_round 4,
  hit_x/hit_y = 527,627` equals row `rally 2, ball_round 3, landing_x/y = 527,627`;
  the first shot of every rally has an empty `hit_x` because there is no previous
  shot.  So the column is a floor/contact point, not a flight position.
- The residual is exactly a height.  Our horizontal offset from the label point has
  a median of only **4.1 px** (p10 −76, p90 +98), and the vertical offset is
  **negative in 100 % of frames** with a median of −167 px.  At focal 3110 px and
  35.3 m camera distance, 1 m of height projects to 88 px, so −167 px is a shuttle
  about 1.9 m above the floor, which is where a contact happens.  The track is
  therefore correct and sits above the label's floor point by exactly the shuttle's
  height.

A direct check of the same idea — projecting our tracked pixel onto the floor with
our own camera and comparing with the label — is *not* conclusive yet (median 7.4 m
disagreement), but that projection depends on the sign/axis conventions of our world
frame relative to the dataset's coordinate system, and the pixel-only evidence above
does not.  The pixel evidence is the one to trust.

**Conclusion for training.**  `hit_x/hit_y` must not be used as a shuttle-position
supervision target.  It is usable as a *floor* annotation only after our own
projection conventions are pinned down.  The labels we do rely on are exact and
verified: the hit frame, the stroke type, the hitter, `player_location_*`,
`opponent_location_*`, `landing_*` (for rally-final shots), `aroundhead`, `backhand`.

## 4. Throughput

Measured on match 1 with one process: 45 frames/s over 22 131 frames in 491 s.
Stage split: **pose 77 %**, detect 17 %, calibrate 6 %, link and track ~0 %.  Pose is
the bottleneck, so on a 10-core M4 running three matches at once is roughly a 3x
speed-up; 43 matches of about 10 min each fit in a few hours of wall-clock, running
unattended.

`--pad-before` (default 2 s) and `--pad-after` (default 6 s) set the rally span.  The
tail is deliberately longer than the head: the shuttle keeps flying after the last
hit and reaches the floor — the label's `landing_x/y` — about 2–3 s later.  The
running batch uses the default 2 s tail, which is enough for the hit/stroke work; a
landing study needs the 6 s tail (a separate pass, or `--pad-after 6` on a rebuilt
`out/`).  The per-rally cache is keyed on the span, so changing the pad recomputes
every rally of a match.

## 4b. Landing check (6 s tail)

With a 6 s tail, the tracked shuttle descends into the labelled landing point on 2 of
3 sampled rallies: rally 1's lowest point is (801,672) against the label's (825,679),
25 px; rally 3's is (813,456) against (800,465), 16 px.  Rally 2 fails: the track
stops at y = 401 in mid-descent and then jumps to the scoreboard (y ≈ 55).  So on
broadcast footage a steep near-court smash loses the shuttle for about a second.  A
3-rally sample proves nothing on its own; the full pass is needed before deciding
whether the linker needs a descent prior.

## 5. Open items

- The floor-projection comparison needs our world axes pinned against the dataset's
  `homography.csv` before `landing_*` can be used as a ground-truth target.
- Only rally spans are extracted (first hit − 2 s … last hit + 2 s); the gap between
  rallies is not, so a hit detector trained on this would need negative windows from
  inside the spans or a second pass over the gaps.
- Broadcast 25/30 fps only.  The phone records 60 fps, so the hit detector must be
  trained or adapted at the phone's rate.
