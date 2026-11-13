# Hit detection: accuracy against ShuttleSet labels

The shipping app hit detector (`ShuttlHits.detectHits`, the same public entry point
`LiveStages().detectHits` wraps) run over the extracted per-rally shuttles of 13
ShuttleSet matches, scored against the ShuttleSet `frame_num` labels.

The detector runs out-of-process so the numbers come from the real code path, not a
re-implementation:

```
../extract/out/NNN.npz --dump.py--> out/NNN.json --hitseval--> out/NNN.hits.json
                                                                    |
                  ../shuttleset/labels/.../set*.csv --score.py------+
```

Reproduce everything:

```sh
cd /path/to/shuttl/research/hits
swift build -c release                       # if .build/release/hitseval is missing
./run.sh                                     # default --threshold -4, scores all 13
../bst/venv/bin/python score.py --hits 'out/*.hits.json' \
    --labels-dir ../shuttleset/labels/ShuttleSet/set \
    --shift 2 --exclude 009 010 --tols 1 2 4
```

`run.sh` re-dumps, re-detects and re-scores; the detector is ~2 s/match.

## Headline numbers

**Operating point: `min_hit_logodds = 1`, all detections shifted +2 frames, 11
matches (009 and 010 excluded, see below).**

| tolerance | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| +/-1 frame | 8067 | 3578 | 3329 | 0.6927 | 0.7079 | 0.7002 |
| **+/-2 frames** | **8947** | **2698** | **2449** | **0.7683** | **0.7851** | **0.7766** |
| +/-4 frames | 9553 | 2092 | 1843 | 0.8204 | 0.8383 | 0.8292 |

11396 labelled hits, 11645 detections in the pool. Matching is the T-DEED/F3Set
protocol: one detection pairs with at most one label, greedy on |offset|, and a pair
kept at +/-4 but outside +/-2 counts as both an FP and an FN. At +/-2, 21% of
detections are spurious and 21% of labelled hits are missed.

### At the current default threshold (-4), same 11 matches, shift +2

| tolerance | P | R | F1 |
|---|---|---|---|
| +/-1 | 0.6511 | 0.7339 | 0.6901 |
| +/-2 | 0.7241 | 0.8162 | 0.7674 |
| +/-4 | 0.7792 | 0.8783 | 0.8258 |

Threshold 1 buys +0.009 F1 at +/-2 (vs -4), +0.003 at +/-4, +0.010 at +/-1, by
trading 1200 detections for 602 fewer FPs and 354 more FNs. The curve is flat on top
(0.7766 at 1, 0.7726 at 2), so the exact choice inside 0-2 matters little.

The -4 column is what the detector ships with (`HitConfig().minHitLogodds`); quote it
if no config change is made.

## Recommended fixed shift: +2 frames

The detector fires early: `detection - label` over matched pairs peaks sharply at
**-2**. Adding +2 to every detection on every match:

| shift | -2 | -1 | 0 | +1 | **+2** | +3 | +4 |
|---|---|---|---|---|---|---|---|
| pooled F1@+/-2 (thr 1) | 0.090 | 0.229 | 0.680 | 0.765 | **0.7766** | 0.749 | 0.610 |

After the shift the residual offset histogram centres on 0 (mean +0.088):
`{-2:397, -1:1086, 0:5277, +1:1704, +2:483}`. One constant applied to every match,
never per-match tuning; per-match best shifts span 0..+3 and would inflate the pooled
number by ~0.01, not worth over-fitting. Best shift is +2 at thr -4, 0, 1 and 2.

## Per-match spread (thr 1, shift +2)

| match | GT | DET | F1@+/-1 | F1@+/-2 | F1@+/-4 | visible | conf | best shift |
|---|---|---|---|---|---|---|---|---|
| 001 Fuzhou 2019 Final | 1644 | 1518 | 0.514 | 0.716 | 0.830 | 75.8% | 0.56 | +3 |
| 002 WT Finals Group | 860 | 833 | 0.884 | 0.903 | 0.917 | 81.0% | 0.61 | +1 |
| 003 Korea Open Final | 928 | 881 | 0.851 | 0.878 | 0.884 | 79.4% | 0.59 | +2 |
| 004 Denmark QF | 818 | 873 | 0.222 | 0.543 | 0.781 | 66.6% | 0.44 | 0 |
| 005 Fuzhou 2018 Final | 1166 | 1184 | 0.643 | 0.706 | 0.778 | 60.7% | 0.38 | +2 |
| 006 Denmark 2018 Final | 1311 | 1384 | 0.782 | 0.807 | 0.828 | 71.2% | 0.49 | +1 |
| 007 Malaysia Open QF | 972 | 1037 | 0.827 | 0.842 | 0.856 | 70.4% | 0.48 | +1 |
| 008 Fuzhou 2019 SF | 704 | 756 | 0.673 | 0.736 | 0.774 | 63.6% | 0.44 | +2 |
| 011 Indonesia Open QF | 1219 | 1201 | 0.804 | 0.829 | 0.840 | 58.9% | 0.38 | +2 |
| 013 Malaysia Masters Final | 756 | 946 | 0.740 | 0.761 | 0.777 | 65.4% | 0.45 | +1 |
| 014 Indonesia Masters QF | 1018 | 1032 | 0.794 | 0.827 | 0.852 | 69.1% | 0.47 | +1 |

Pooled F1@+/-2 = 0.777; min 0.543 (004), median 0.807, max 0.903 (002). Match 004 is
the outlier: also the only match whose best shift is 0, on a sparse track (66.6%,
conf 0.44). It costs the pool ~0.02 F1; nothing else is that weak.

## Dominant error mode: extra detections inside a rally, not missed serves

At thr 1, shift +2, +/-2, the 2698 FPs decompose as:

| kind | count | share of FPs |
|---|---|---|
| double-count: extra detection <=10 frames from an already-matched label | 414 | 15% |
| near-miss FP: 11-12 frames from a label (outside +/-2, but a real contact) | 1035 | 38% |
| far FP: >=13 frames from any label in that rally | 1249 | 46% |

and the 2449 FNs: 681 on the serve (first label of a rally), 1768 later. Missed serves
are the largest *named* bucket (28% of FNs) but not a majority; the FPs are dominated
by detections that are near *something* but outside tolerance.

By stroke, misses concentrate on soft/deceptive shots where the shuttle barely
touches the racket: 發短球 (short serve) 511, 放小球 (drop) 274, 擋小球 (net block)
255 - 42% of all FNs. Fast kills (殺球, 157) and late-rally exchanges (挑球 137,
推球 136) far less. Double-counts come in pairs 6-8 frames apart:

```
001 s1 r3:  label 12783  <- matched det 12781, extra det 12787 (conf 0.87)
001 s1 r3:  label 12847  <- matched det 12847, extra det 12841 (conf 0.84)
001 s1 r9:  label 16588  <- matched det 16586, extra det 16580 (conf 0.86)
001 s1 r11: label 18160  <- matched det 18158, extra det 18166 (conf 0.80)
```

Typical misses (short serves and net exchanges, early in the rally where the track is
often lost):

```
001 s1 r1 frame 11496  發短球  (serve, 1st of 2)
001 s1 r2 frame 11881  發短球  (serve, 1st of 6)
001 s1 r2 frame 11900  推球    (2nd of 6)
001 s1 r3 frame 12400  挑球    (5th of 27)
```

Confidence does not separate TP from FP cleanly: at thr 1, TP median 0.993 vs FP
median 0.907; FP rate in the top bucket (>=0.99) is 11% vs 35% in 0.70-0.80. Raising
to logodds 2 gives P 0.822 / R 0.729 / F1 0.773; to 3 gives P 0.854 / R 0.653 /
F1 0.740. Precision improves, recall collapses - the residual error is a peak-picking
problem (right place, wrong count), not a scoring problem.

## Excluded matches: 009 and 010

**Both excluded. Their numbers are not meaningful and no threshold or shift rescues
them.**

| match | F1@+/-2 (thr -4) | F1@+/-2 (thr 1) | visible | conf |
|---|---|---|---|---|
| 009 Sudirman Cup QF | 0.085 | 0.074 | 33.2% | 0.215 |
| 010 Sudirman Cup Group | 0.112 | 0.098 | 42.1% | 0.262 |

F1 is flat in shift (009: 0.085 at 0, 0.092 at +4; 010: 0.112 at 0, 0.113 at +5) and
flat in threshold - the detections carry no alignment information. Three hypotheses
tested and dismissed:

1. **Wrong threshold.** Both re-run at `--threshold -4` and compared to stored output:
   identical hits. F1 moves only 0.085 -> 0.074 from thr -4 to thr 1. Not the cause.
2. **Video/label frame-rate mismatch (30 vs 29.97).** Disproved directly. Every match
   is 25.0 fps on both sides: `../extract/out/NNN.npz` records `fps = 25.0` for all 13,
   and `manifest.csv` records `label_fps = 25.0 / 24.999 / 25.001`. Regressing the
   detection-to-nearest-label offset on absolute video frame gives slopes of
   +0.000..-0.03 frames per 1000 frames for healthy matches and +0.019 for 010 - at
   most 4 frames drift across a 100k-frame match, R^2 <= 0.005 (pure noise). A
   30-vs-29.97 mismatch would show ~1.0 f/1000. 009's +0.32 f/1000 over 50k frames is
   16 frames total at R^2 = 0.04, indistinguishable from healthy matches' flat
   histograms. Every rally's extracted window also starts exactly 50 frames before its
   first label, in 009 and 010 as in the rest. No correction warranted for any match.
3. **Genuinely bad shuttle tracking.** The cause. Tracks are sparse and low-confidence:
   33.2% / 42.1% visible vs 58.9-81.0% for the other eleven, mean shuttle confidence
   0.215 / 0.262 vs 0.378-0.605. Per-rally it is worse: 23 of 009's 66 rallies are
   below 25% visible and 13 below 10%.

Symptom: their detections sit *uniformly* within +/-6 frames of labels (009 counts
8/9/10/10/5/7/8/10/9/10/8/8/8 at offsets -6..+6) instead of peaking at -2 as healthy
matches do (002: 556 of 893 detections at exactly -2). With the shuttle unobserved the
detector falls back to weak cues and emits approximate contacts.

## What the number rests on

Pooled F1 **0.7766 at +/-2, 0.8292 at +/-4** (thr 1, shift +2) rests on **11 matches**:
001, 002, 003, 004, 005, 006, 007, 008, 011, 013, 014 - 11396 labelled hits, 11645
detections. 009 and 010 are excluded as unscorable. The earlier hand-recorded 0.702 at
+/-2 came from a 12-match pool that also omitted 014 (12720 = 13848-1128 detections,
11599 = 12617-1018 labels, matching the 13-match totals minus 014's); 014 is healthy
and fully scorable, so that number is superseded. Spread across the eleven
(0.543-0.903) is real per-match variation, not a measurement artefact.

## Caveats

- The +2 shift corrects a detector timing bias, not a label property: applied live it
  delays every hit by 80 ms at 25 fps. The real fix belongs in the detector's
  peak-picking, not in post-processing.
- A few labels are marked 未知球種 (unknown stroke); they stay in the pool as ordinary
  ground truth, adding ~1.4% noise to the stroke-type breakdown but nothing to P/R/F1.
- `ffprobe`/`ffmpeg` are not installed here (confirmed), so frame counts come from the
  `.npz` and the manifest; the two agree to within 0.004% on fps and every label
  window is consistent with them.
