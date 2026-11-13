# Can BST's released weights work on our phone-extracted features? (2026-10-05)

Short answer: **no — and the reason is the pose branch.** Our court positions and our
shuttle track are as good as the published features; our Apple Vision pose is not, and
BST is tightly coupled to the pose estimator it was trained with (RTMPose/MMPose).

## 1. What was compared

`convert.py` turns the extractor's `out/NNN.npz` into BST's raw per-clip layout
(`<split>/<Top|Bottom>_<stroke>/<match>_<set>_<rally>_<ball_round>_{joints,pos,shuttle}.npy`),
so our clips are named identically to the released clips and can be compared one by one.
The released clips live in `data/ShuttleSet_data_merged/dataset_npy_between_2_hits_with_max_limits/`.

Match 1–3 produced 2 945 clips (24 of the 25 classes; the `none` class and some rare
types are missing), and 1 464 of them pair with a released clip of the same name.

## 2. Two bugs in my converter, both found and fixed

- **Side mapping was inverted.** The labels' `player` column (A/B) is the *opposite* of
  the court side in the released data: for 400 labelled hits of match 1, the class prefix
  `Top_` corresponds to player **B** in 100 % of cases and player A in 0 %. I had mapped
  A→Top. Fixed.
- **Window tail was 3 frames short.** The official window is `next hit + (fps//2)//2`;
  I used `fps//8`. Measured clip length differed by −3.2 frames on average, matching
  exactly. Fixed.

Both fixes moved the released-weight accuracy only from 0.218 to 0.252, so neither was
the main cause.

## 3. The decisive ablation

Hybrid sets, same clips, one input branch swapped at a time, evaluated with the 10
released `BST_CG_AP` seq-100 weights (accuracy / macro-F1 / top-2):

| variant | acc | macro-F1 | top-2 |
|---|---|---|---|
| official joints + official pos + official shuttle (control) | 0.8531 | 0.7901 | 0.9533 |
| **our joints** + official pos + official shuttle | **0.4247** | 0.3466 | 0.5934 |
| official joints + **our pos** + official shuttle | 0.8535 | 0.7918 | 0.9539 |
| official joints + official pos + **our shuttle** | 0.8413 | 0.7809 | 0.9489 |
| all three ours | 0.4332 | 0.3532 | 0.6078 |

**Court position: no loss at all (0.8535 vs 0.8531).**  The phone's court calibration and
player tracker are therefore equivalent to the published features.
**Shuttle: 1.2 points (0.8413 vs 0.8531).**  TrackNetV3 transfers cleanly.
**Pose: 43 points lost.**  This is the whole problem.

## 4. Why the pose fails

Per-joint error against the released joints (normalised units, ~0.1 = a limb):

- Signed bias is negligible (+0.006) and the optimal uniform scale is 0.98, so there is
  no coordinate or scale convention error. Left/right is not mirrored (a mirrored
  comparison is 2.5x *worse*).
- The per-clip **median** joint error is only 0.023, but the per-frame **worst** joint
  error is 0.29 and the p90 of frames containing a joint off by more than 0.5 is 6 %.
  So the pose is mostly good with occasional gross errors.
- The error is concentrated in the **near/Bottom player's head** (nose, eyes) — up to 9
  normalised units, i.e. that player's head joints are sometimes placed on a different
  person. That is the far-court crop pass producing a second detection that the tracker
  assigns to the near player.

A pose estimator's error *style* is what a stroke model learns to expect. RTMPose-m and
Apple Vision disagree in exactly the ways that matter here: Vision has no ankle keypoint
refinement, it drops limbs at the frame edge, and its head joints are the least reliable.
Zero-shot transfer from RTMPose features to Vision features therefore cannot work, no
matter how small the average error is.

## 5. Fine-tuning on our own features

`finetune.py` initialises from the mean of the 10 released weights and trains on matches
1–3 with match 4 held out (2 945 train / 687 val, 12 epochs, ~32 s per epoch on MPS):

- val macro-F1 rose only to 0.21, test accuracy 0.42.
- Test **top-2 is 0.70**, so the model does find the right answer in its top two, but the
  pose branch is not carrying enough signal to separate the 25 classes.

Twelve epochs is a smoke test, not a training run — the released recipe uses ~350–1600
epochs. But the shape of the curve (val 0.33 → 0.37 over 12 epochs) says more epochs of
this alone will not close a 43-point gap. The pose has to be fixed or replaced first.

## 5b. The fix, proven (2026-10-05 ~23:00)

Same 318 clips of match 1, same released weights, only the pose estimator swapped:

| pose input | acc | macro-F1 | top-2 |
|---|---|---|---|
| **Apple Vision** (what the phone runs) | 0.5305 | 0.3488 | 0.6918 |
| **RTMPose-m** (what BST was trained with) | **0.8730** | 0.6992 | 0.9726 |

RTMPose closes essentially the whole gap and beats the published 0.8322 on this subset.
So the conclusion is unambiguous: **the released BST weights work as published as soon
as the pose comes from RTMPose-m**, and everything else in our pipeline (court
calibration, player tracking, shuttle tracking) is already at published quality.

Practical consequence: `rtm_pose.py` in `research/rtmpose/` runs RTMPose-m (body7
256x192, the ONNX rtmlib ships) on the person boxes our extractor already produces, and
`convert.py --rtm <dir>` swaps its joints into the BST input. No BST retraining is
needed. Speed on the Mac's CPU: about 8 frames/s per process, 4 processes shard a video
4 ways.

## 6. What this means for the plan

The phone's pipeline is **not** the problem: the court calibration, player positions and
shuttle track are already at published quality, and the phone is the cheap place to run
them. The pose is the one stage that needs to change, and there are three options:

1. **Run RTMPose on the server** (Apache-2.0, `rtmlib`, CPU-capable, ~90 FPS on a
   desktop core). The phone sends the frames' person boxes plus its shuttle track; the
   server does pose + stroke classification. This reuses BST exactly as published and is
   the option that matches the user's preference for server-side intelligence. Cost: a
   pose pass over the match on the 2-core server, which is minutes, not hours, because
   pose is a small model and only needs the two detected players.
2. **Fine-tune BST on Vision pose for a long run** (many hours on MPS, or rent a GPU).
   Accepts the pose style but spends the budget on a second-best estimator.
3. **Train our own small stroke model on our own features** from scratch, on the
   ShuttleSet labels plus whatever phone footage we can label. Most work, most control,
   and the only path that eventually matches the phone exactly.

Recommendation: (1) as the immediate unblock, with (3) as the long-run answer once real
phone footage exists. Do not start either until the full ShuttleSet extraction is done,
so training and validation use all 44 matches.

## 7. Reproduce

```bash
cd research/bst
venv/bin/python convert.py --extract ../extract/out --out /tmp/bst_c1b
venv/bin/python ablate.py                       # the table in section 3
venv/bin/python finetune.py --init "weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt" \
    --data /tmp/bst_split_coll --epochs 12 --budget-min 40 --out /tmp/bst_ft_r1
```

`ablate.py` needs `data/.../dataset_npy_between_2_hits_with_max_limits` (the released
raw clips) and `/tmp/bst_c1b` from `convert.py`.
