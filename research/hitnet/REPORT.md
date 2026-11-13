# hitnet — a learned badminton hit-frame detector vs the hand-written rule

Everything lives in `research/hitnet/` (nothing outside was modified). Train = every match under
`research/extract/out/` **except** 039–044; model selection and decoder tuning = **val 041 only**;
test = 039, 040, 042, 043, 044 (044's npz was present). Both detectors are scored by the same
`research/hits/score.py` one-to-one greedy pairing on the same matches — learned with `--shift 0`,
rule with `--shift 2`.

Files: `common.py` (cache, features, model, decoder, eval glue), `train.py`, `infer.py`, `tune.py`,
`evaluate.py`, `out/<tag>/NNN.hits.json`, `ckpt/*.pt`, `logs/*.log`.

## Model

**Features (51/frame, npz only, no keypoints):** shuttle `x/W, y/H, visible, interp, conf`;
forward/backward/central finite differences of `x,y`, second difference, speed, cosine of the
direction change (the "kink" the rule uses); finite/missing flags; frames to nearest visible sample;
per player box centre/w/h + central differences; foot-world `x,y` scaled by half-length 6.7 m /
half-width 3.05 m + differences; shuttle-to-box offsets in box-height units; `fps/30`. All
differences are scaled per 1/30 s (`× fps/30`) so 25 and 30 fps rallies are comparable (12/25 train
matches are 25 fps, all of 039–044 are 30 fps). Missing values are 0 with a companion mask — no
silent imputation.

**Net:** bidirectional dilated residual TCN (12 blocks, dilations 1-2-4-8-16-32 twice, 96 channels,
GELU+BN+dropout 0.15) → 2 logits/frame (hit; hitter-is-Top). **454,658 params.** MPS training, CPU
inference.

**Targets:** Gaussian soft targets σ=1 frame, 4× positive weight outside the Gaussian; side head
supervised only at ±1 frame of a hit.

**Augmentation:** horizontal mirror (shuttle `x→W−x`, foot world `y→−y`), time stretch 0.8–1.3× with
labels moved along, random shuttle dropout (≤5 gaps of ≤24 frames) with probability, ±0.7 px jitter.
Missing players/shuttle/stubs all occur.

**Decoding:** sigmoid → local maxima → NMS with minimum gap at one threshold. Optional alternation
decoder (`--lam`): DP over peaks adding `lam·(P(alternate) − P(same side))` using the side head
(MonoTrack-style) — implemented and measured, not assumed.

**Inference:** whole-rally (non-causal) default; `--windowed` = overlapping 384-frame windows,
centre-only. Full-rally vs windowed logits: max |Δ| 0.316, mean 0.019, identical detections on 041;
256/384/512-frame windows give identical val numbers, so it works streaming.

Training: 2400 windows of 384 frames/epoch (rally-proportional), AdamW lr 2e-3 one-cycle, wd 1e-2,
batch 32, best-on-val-041 checkpoint. 40 epochs ≈ 38 min on MPS.

## Held-out result — pooled over the 5 test matches

| tol | rule (shift +2) | learned (shift 0) |
|---|---|---|
| ±1 | 0.625 (P 0.561 / R 0.706) | **0.839 (P 0.859 / R 0.820)** |
| ±2 | 0.692 (P 0.621 / R 0.782) | **0.928 (P 0.951 / R 0.907)** |
| ±4 | 0.743 (P 0.666 / R 0.839) | **0.952 (P 0.975 / R 0.930)** |

Rule 5919 detections / 2244 FP vs learned 4482 / 221 FP against 4699 labelled hits. Errors moved
from FP-dominated (1024 misses, 2244 FPs) to miss-dominated (438 misses, 221 FPs).

Per match ±2 (F1/P/R), same matches and scorer:

| match | rule | learned |
|---|---|---|
| 039 | 0.676 / 0.610 / 0.757 | 0.885 / 0.911 / 0.861 |
| 040 | 0.465 / 0.374 / 0.616 | 0.927 / 0.955 / 0.900 |
| 042 | 0.733 / 0.660 / 0.825 | 0.938 / 0.972 / 0.907 |
| 043 | 0.721 / 0.658 / 0.797 | 0.922 / 0.942 / 0.903 |
| 044 | 0.721 / 0.653 / 0.804 | 0.938 / 0.953 / 0.923 |

Val 041 (tuned on this match, optimistic): rule 0.777, learned **0.949** (±1 0.870/0.832,
±4 0.966/0.966).

## Serve vs rally recall (±2, 5 test matches pooled)

| | serve | non-serve |
|---|---|---|
| rule | 125/426 = 0.293 | 2553/3073 = 0.831 |
| learned | 215/426 = 0.505 | 2929/3073 = 0.953 |

Misses by rally position went 311 serve / 446 rally (rule) → 222 / 107 (learned): the serve is now
the dominant remaining miss type, matching the reported anatomy (static shuttle, no kink — the rule
fires late or not at all; the learned detector can use the receiver's movement and the shuttle
leaving the racket).

## Offset distribution, and no shift is needed

learned (shift 0), matched pairs ±4: `0:2032, +1:1390, −1:430, +2:309, −2:100, +3:59, …` — centred
on 0/±1, no systematic lead/lag; `score.py` independently reports per-match best shifts of 0/−1, so a
fixed shift would hurt. Rule (shift +2): `0:1253, +1:469, −1:654, …` — the shift removes a mean bias
but leaves a wide spread.

## Decoder ablation (val 041; test where run)

| config | val F1@2 | test F1@2 |
|---|---|---|
| NMS thr 0.35–0.45, gap 0.30 s | 0.940–0.946 | **0.921** (thr 0.4) |
| NMS thr 0.6, gap 0.30 s | 0.940 | — |
| NMS thr 0.6, gap 0.15 s + alternation lam 1.0 | 0.941 | 0.922 |

Alternation buys nothing measurable (test ±2 0.921→0.922, ±1 identical to 3 dp, serve recall
unchanged, val +0.001), so the shipped default is `lam 0`; it is still available via `--lam`. F1@±2
is a plateau over thr 0.35–0.6, so the operating point is not delicate.

## Variants

| variant | val F1@2 | test F1@2 | test F1@1 |
|---|---|---|---|
| single TCN (seed 0) | 0.940 | 0.908 | 0.792 |
| 3-TCN logit-average ensemble | **0.949** | **0.921** | 0.829 |
| ensemble, windowed 384 | 0.949 | 0.921 | 0.828 |
| train without 009/010 (poor tracking) | 0.914 | 0.889 | 0.766 |
| no time-stretch augmentation | 0.896 | — | — |
| σ 0.8 + 8× positive weight | 0.919 | — | — |

Keeping 009/010 is better (0.921 vs 0.889 test F1@2), so they stay in training. Time stretch is
worth ~0.045 val F1. Ensembling 3 seeds is worth ~0.016 test F1@2 for 3× inference.

## Cost

454,658 params (1.36 M for the ensemble). CPU (M4), 1 thread, whole 31.3-min match of 041:
features 0.006 s/min of video, features+model **0.29 s/min** (band, single model; 0.85 s/min for the
3-ensemble), decode 0.001 s/min. Four threads was *slower* (0.40 s/min) for this 3-conv-channel net.
A 30-minute match is ~9 s of one CPU core, without MPS and without keypoints.

Confidence is usable as a filter: TP mean 0.790 (p50 0.802) vs FP mean 0.639 (p50 0.647); the rule's
is not (TP 0.949 vs FP 0.820, both p90 ≈ 0.99).

## Not verified

- Keypoints untested by design (v1 uses none, so RTMPose in v2 is an extension, not a fix).
- Camera/venue/label drift across matches not measured.
- Matches extracted after this run are not in these numbers; `scan_ids()` re-scans the directory.
- Cross-fps generalisation rests on the fps scalings plus the 25/30 fps training mix — all test
  matches are 30 fps.
- The BiGRU comparison (alternative architecture) was still training when this stopped (~11/40
  epochs, val F1@2 0.926 at epoch 10, 1.5 h/epoch); no BiGRU test number is claimed.
- CoreML export not attempted (not needed yet).
