# End-to-end accuracy: OUR detector's hit frames -> BST stroke labels

Measured 2026-10-06 in `research/e2e/`. Nothing trained; released weights only
(`bst_CG_AP_JnB_bone_...seq_100_merged*`, 10 checkpoints averaged, seq_len 100, 25 classes).

**Scope: match 1 only.** It is the only match with pose. `rtmpose/pose/` is still being
filled by another agent, and the only pose that exists is the 12000-frame RTMPose subset at
`/tmp/rtm/ab2/001_sh{0..3}.npz` (12000 of 48311 frames = 24.8%, in 30 contiguous runs).
All numbers below are therefore one match, and only over the frames that subset covers.

## Headline

| arm | window centre | window neighbours | clips | acc | macro-F1 | top-2 |
|---|---|---|---|---|---|---|
| **control** (labelled hits) | labelled hit | labelled | 318 | **0.8730** | 0.6992 | 0.9726 |
| **detected** (ours) | detected hit | detected | 231 | **0.7779** | 0.5612 | 0.8848 |
| detwin (ours, centre moved onto the matched label) | labelled hit | detected | 231 | 0.7814 | 0.5823 | 0.8900 |
| control restricted to the 224 clip keys the detector reaches | labelled hit | labelled | 224 | 0.8830 | 0.6806 | 0.9804 |
| detected, all detections shifted **+2** | detected+2 | detected+2 | 257 | 0.7911 | 0.6043 | 0.9019 |

The control reproduces the documented 0.8730 / 0.6992 / 0.9726 **exactly**, and its 318 clip
names are **identical** to `bst/acc_cache_selftest/ours_rtm/001/filtered` -- so the control is
the same arm, not a reimplementation.

**End-to-end cost of using our detector: 0.8730 -> 0.7779, -9.5 points** (macro-F1 -0.138,
top-2 -0.088).

## Detection quality (whole match, from `hits/out/001.hits.json` vs labels)

| quantity | value |
|---|---|
| labelled hits | 1644 (1642 distinct frames) |
| detected hits | 1707 |
| detections matching a label within +-4 | 1288 (75.4%) |
| **false-positive detections** (no label within +-4) | **419 (24.5%)** |
| labelled hits reached by a detection | 1266 (**77.0%**) |
| **labelled hits missed entirely** | **378 (23.0%)** |
| matched detections more than 2 frames off | 676 (52.5%) |
| offset `label - detected`, median | **+2** (detections fire 2 frames early) |

## Error decomposition

Sign convention: `offset = label - detected`; median +2 means a detection fires 2 frames
*before* the stroke.

1. **Missed strokes -- 378 of 1644 (23.0%).** These never get a window, so in deployment the
   stroke is simply never emitted. Not folded into the acc above (which scores emitted clips
   only); it is the separate, larger loss.
2. **False-positive detections -- 419 of 1707 (24.5%).** Dropped as unscoreable (no label
   within +-4). They would become spurious strokes in deployment.
3. **Mis-centred windows -- ~10 points, and it is NOT the centre.** The two arms that differ
   *only* in window centre (detected vs detwin: same detections, same classes, window moved so
   the matched label sits at the centre) score 0.7779 vs 0.7814 -- **0.35 points**. Moving the
   centre off by the detector's offset costs almost nothing. The rest of the gap is the window
   *boundaries*: detwin (centre correct, boundaries from detected neighbours) vs control on the
   same 224 clip keys = 0.7814 vs 0.8830 -> **10.2 points**. Building the window from the
   detected hit sequence at all -- not its centre -- is the dominant cost.
4. **Genuinely wrong stroke labels.** On matched clips the class is the nearest labelled hit's
   type, so the *given label* is right by construction; the residual is the classifier's own
   error, 1 - 0.8830 = **11.7% even in the control**. The detector does not add to this beyond
   (3).

**The -2/+2 bias:** re-running detected with every detection shifted **+2** moves detections
onto the labels -- matches rise 1288 -> 1419, false positives fall 419 -> 288, clips rise
231 -> 257 -- and accuracy rises 0.7779 -> **0.7911 (+1.3 points)**. So the bias explains about
**14% of the 9.5-point drop**, not most of it. It is worth correcting, but correcting it does
not close the gap.

**Biggest single remaining error source:** the window boundaries built from the detected hit
sequence (10.2 points), *not* the centre offset (0.35) and *not* the +2 bias (1.3).

## What is missing

13 matches have both `extract/out/NNN.npz` and `hits/out/NNN.hits.json`, but only match 1 has
any pose at all. `rtmpose/pose/` currently holds one failed shard (`001.sh0.npz`); the other
agent's driver is retrying. This measurement therefore rests on **one match**, and only on the
24.8% of its frames that RTMPose has covered. The same pipeline should be re-run per match as
pose lands -- the tooling below takes `--limit N` and globs shards, so it scales unchanged.

## Reproduce

```bash
cd /path/to/shuttl/research/e2e

# control (labelled hits) -- reproduces 0.8730 / 0.6992 / 0.9726 on 318 clips
../bst/venv/bin/python -u e2e_convert.py --extract ../extract/out \
  --labels ../shuttleset/labels/ShuttleSet/set --hits ../hits/out \
  --arm control --rtm /tmp/rtm/ab2 --limit 1 --out e2e_ctrl_rtm --stats stats_ctrl_rtm.json
ln -sfn train e2e_ctrl_rtm/test
../bst/venv/bin/python -u ../bst/collate.py e2e_ctrl_rtm e2e_ctrl_rtm_collated 100 train
../bst/venv/bin/python -u ../bst/eval_bst.py --data e2e_ctrl_rtm_collated --seq-len 100 \
  --set train --device mps --out eval_ctrl_rtm.json \
  --weights ../bst/weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt

# detected (ours) -- 0.7779 on 231 clips
../bst/venv/bin/python -u e2e_convert.py --extract ../extract/out \
  --labels ../shuttleset/labels/ShuttleSet/set --hits ../hits/out \
  --arm detected --rtm /tmp/rtm/ab2 --limit 1 --out e2e_det_rtm --stats stats_det_rtm.json
ln -sfn train e2e_det_rtm/test
../bst/venv/bin/python -u ../bst/collate.py e2e_det_rtm e2e_det_rtm_collated 100 train
../bst/venv/bin/python -u ../bst/eval_bst.py --data e2e_det_rtm_collated --seq-len 100 \
  --set train --device mps --out eval_det_rtm.json \
  --weights ../bst/weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt

# detwin: same detections, window centred on the matched label -- 0.7814; detwin vs detected = mis-centring
../bst/venv/bin/python -u e2e_convert.py ... --arm detwin --out e2e_detwin_rtm ...

# detected with the +2 bias corrected -- 0.7911 on 257 clips
../bst/venv/bin/python -u e2e_convert.py ... --arm detected --shift 2 --out e2e_detp2_rtm ...
```

`e2e_convert.py --arm control` is `bst/convert.py` re-implemented with the arm switch; the
window rule (`clip_window`), joint/pos/shuttle normalisation, class naming, split naming, the
"both players resolved" gate and the RTMPose full-clip-coverage gate are copied verbatim, so
the two arms differ only in which hit frames drive the window. `--shift N` adds N to every
detected frame; `--no-rtm` drops the pose swap.

---

# Follow-up: the seq30 fixed-window protocol (`--arm fixed`)

Weights `bst/weights/bst_CG_AP_JnB_bone_merged*.pt` (10 ckpts, seq_len 30); window
`[hit - fps//2, hit + fps//2]` (confirmed: released seq30 raw clips are 25 frames at fps 25).
Same match 1, same `/tmp/rtm/ab2` RTMPose, same clip keys.

## Full arms

| arm | centre | bounds | model | clips | acc | macro-F1 | top-2 |
|---|---|---|---|---|---|---|---|
| control | labelled | labelled | seq100 | 318 | 0.8730 | 0.6992 | 0.9726 |
| detected | detected | detected | seq100 | 231 | 0.7779 | 0.5612 | 0.8848 |
| **fixed** | **detected** | **fixed ±12** | **seq30** | **233** | **0.8682** | 0.6392 | 0.9670 |
| fixed_ctrl | labelled | fixed ±12 | seq30 | 319 | 0.8696 | 0.6742 | 0.9752 |
| fixed (+2 bias corrected) | detected+2 | fixed ±12 | seq30 | 259 | 0.8846 | 0.6716 | 0.9834 |

## Key-matched (same clip keys, so arms are directly comparable)

| arm | centre | bounds | model | clips | acc | macro-F1 | top-2 |
|---|---|---|---|---|---|---|---|
| control | labelled | labelled | seq100 | 223 | **0.8870** | 0.6838 | 0.9803 |
| detected | detected | detected | seq100 | 222 | **0.7932** | 0.5885 | 0.8950 |
| fixed_ctrl | labelled | fixed ±12 | seq30 | 224 | **0.8839** | 0.6757 | 0.9826 |
| **fixed** | **detected** | **fixed ±12** | **seq30** | **224** | **0.8786** | 0.6635 | 0.9723 |

**The fixed window recovers essentially the whole loss.** 0.7932 -> 0.8786 = **+8.5 points**,
against a seq30 ceiling of 0.8839 — only **0.53 points** left on the table, versus 9.4 points
for the seq100 variable window.

**And seq30 costs almost nothing on this subset**: 0.8870 (seq100 control) vs 0.8839 (seq30
control) = **0.3 points**, well inside the 0.02 checkpoint std. So the fixed window more than
pays for the weaker model. The +2 correction still stacks a little (0.8682 -> 0.8846 on the
full arm, +1.6 points).

## Recommendation

Ship the **seq30 fixed-window path** (`bst_CG_AP_JnB_bone_merged*`, `[hit ± fps//2]`) and use
the detected hit as the centre. It is also cheaper (30-frame clips, no neighbour lookup, so the
detector's errors in neighbouring hits cannot move the window at all).
