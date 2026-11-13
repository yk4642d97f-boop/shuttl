# Stroke classification accuracy — BST on our pipeline's features

Generated 2026-10-06T15:51:47+00:00 by `measure.py`.  Released ShuttleSet weights (`bst_CG_AP_JnB_bone…seq_100_merged*`, 10 checkpoints averaged), seq_len 100, 25 classes.  **Nothing was trained for this report.**

Claim under test: *features from our pipeline reach the same stroke-classification accuracy as the published ShuttleSet features when BST's expected pose (RTMPose-m) is used.*

## Inputs available at this run

| match | split | extract | pose | clips kept | degenerate dropped | kept in arm(s) | note |
|---|---|---|---|---|---|---|---|
| 39 | test | yes | RTMPose-m | 748 | 0 | offi, ours_rtm, ours_vis |  |
| 40 | test | yes | RTMPose-m | 525 | 0 | offi, ours_rtm, ours_vis |  |
| 41 | val | yes | RTMPose-m | 1128 | 0 | offi, ours_rtm, ours_vis |  |
| 42 | test | yes | RTMPose-m | 571 | 0 | offi, ours_rtm, ours_vis |  |
| 43 | test | yes | RTMPose-m | 507 | 0 | offi, ours_rtm, ours_vis |  |
| 44 | test | yes | RTMPose-m | 858 | 0 | offi, ours_rtm, ours_vis |  |

All five held-out matches are present; the held-out aggregate is the headline.

## Arms

| arm | features | pose |
|---|---|---|
| `ours_rtm` | our pipeline | RTMPose-m (BST's expected pose) |
| `ours_vis` | our pipeline | Apple Vision |
| `offi` | official ShuttleSet, as published | as published |

Within each match every arm is evaluated on exactly the same clip names, so all rows are clip-for-clip.

## Published reference

The published number is the official test split as released: 3,499 strokes over matches 39, 40, 42, 43, 44. Re-measured here with the same 10 released checkpoints, it gives **acc 0.8322 / macro-F1 0.8097 / top-2 0.9593** (3499 clips). `bst/REPORT.md` records 0.8322 for the same split, so this is a reproduction of the published figure, not a new one.

A held-out aggregate over fewer clips than the full split is not directly comparable to it — the `offi` arm's held-out row above is the like-for-like comparison, because it holds exactly the clips `ours_rtm` scored.

## Held-out aggregate (matches 39, 40, 42, 43, 44 only)

| arm | matches | clips | acc | macro-F1 | top-2 | acc std over the 10 checkpoints |
|---|---|---|---|---|---|---|
| `ours_rtm` | 5 | 3209 | 0.8363 | 0.7894 | 0.9595 | 0.0039 |
| `ours_vis` | 5 | 3209 | 0.5159 | 0.4433 | 0.7030 | 0.0482 |
| `offi` | 5 | 3209 | 0.8354 | 0.7875 | 0.9608 | 0.0036 |

Our features under RTMPose-m vs the official features, same clips: **+0.0009 accuracy** (at least as accurate) — within the 0.0039 spread across the 10 checkpoints.

## Aggregate — val = 41 (reported separately, never pooled with the held-out set)

| arm | matches | clips | acc | macro-F1 | top-2 | acc std |
|---|---|---|---|---|---|---|
| `ours_rtm` | 1 | 1128 | 0.8598 | 0.7322 | 0.9723 | 0.0074 |
| `ours_vis` | 1 | 1128 | 0.4830 | 0.4060 | 0.6893 | 0.0486 |
| `offi` | 1 | 1128 | 0.8647 | 0.7336 | 0.9748 | 0.0049 |

Single match; treat as indicative.

## Aggregate — training-side matches (regression figure, NOT the claim)

| arm | matches | clips | acc | macro-F1 | top-2 | acc std |
|---|---|---|---|---|---|---|
| `ours_rtm` | — | — | — | — | — | — |
| `ours_vis` | — | — | — | — | — | — |
| `offi` | — | — | — | — | — | — |

These matches are in BST's training split, so this is a reproduction check, not a held-out result.

## Per-match

| match | split | arm | clips | acc | macro-F1 | top-2 |
|---|---|---|---|---|---|---|
| 39 | test | `ours_rtm` | 748 | 0.8116 | 0.7300 | 0.9548 |
| 39 | test | `ours_vis` | 748 | 0.4773 | 0.4103 | 0.6742 |
| 39 | test | `offi` | 748 | 0.8183 | 0.7379 | 0.9592 |
| 40 | test | `ours_rtm` | 525 | 0.8752 | 0.8046 | 0.9665 |
| 40 | test | `ours_vis` | 525 | 0.5897 | 0.4790 | 0.7604 |
| 40 | test | `offi` | 525 | 0.8731 | 0.8044 | 0.9655 |
| 41 | val | `ours_rtm` | 1128 | 0.8598 | 0.7322 | 0.9723 |
| 41 | val | `ours_vis` | 1128 | 0.4830 | 0.4060 | 0.6893 |
| 41 | val | `offi` | 1128 | 0.8647 | 0.7336 | 0.9748 |
| 42 | test | `ours_rtm` | 571 | 0.8699 | 0.8281 | 0.9734 |
| 42 | test | `ours_vis` | 571 | 0.4996 | 0.3996 | 0.6792 |
| 42 | test | `offi` | 571 | 0.8536 | 0.7997 | 0.9680 |
| 43 | test | `ours_rtm` | 507 | 0.8059 | 0.7408 | 0.9462 |
| 43 | test | `ours_vis` | 507 | 0.4604 | 0.3550 | 0.6631 |
| 43 | test | `offi` | 507 | 0.8071 | 0.7433 | 0.9485 |
| 44 | test | `ours_rtm` | 858 | 0.8296 | 0.7529 | 0.9580 |
| 44 | test | `ours_vis` | 858 | 0.5481 | 0.4276 | 0.7324 |
| 44 | test | `offi` | 858 | 0.8319 | 0.7606 | 0.9617 |

## Clip-for-clip vs the official features

Same clip names, same checkpoints, same metric code — the only difference is whose features the classifier saw.

| match | split | arm | clips | acc | Δ acc vs `offi` | macro-F1 | Δ F1 vs `offi` |
|---|---|---|---|---|---|---|---|
| 39 | test | `ours_rtm` | 748 | 0.8116 | -0.0067 | 0.7300 | -0.0079 |
| 39 | test | `ours_vis` | 748 | 0.4773 | -0.3410 | 0.4103 | -0.3275 |
| 39 | test | `offi` (baseline) | 748 | 0.8183 | — | 0.7379 | — |
| 40 | test | `ours_rtm` | 525 | 0.8752 | +0.0021 | 0.8046 | +0.0002 |
| 40 | test | `ours_vis` | 525 | 0.5897 | -0.2834 | 0.4790 | -0.3254 |
| 40 | test | `offi` (baseline) | 525 | 0.8731 | — | 0.8044 | — |
| 41 | val | `ours_rtm` | 1128 | 0.8598 | -0.0049 | 0.7322 | -0.0014 |
| 41 | val | `ours_vis` | 1128 | 0.4830 | -0.3817 | 0.4060 | -0.3276 |
| 41 | val | `offi` (baseline) | 1128 | 0.8647 | — | 0.7336 | — |
| 42 | test | `ours_rtm` | 571 | 0.8699 | +0.0163 | 0.8281 | +0.0284 |
| 42 | test | `ours_vis` | 571 | 0.4996 | -0.3539 | 0.3996 | -0.4001 |
| 42 | test | `offi` (baseline) | 571 | 0.8536 | — | 0.7997 | — |
| 43 | test | `ours_rtm` | 507 | 0.8059 | -0.0012 | 0.7408 | -0.0025 |
| 43 | test | `ours_vis` | 507 | 0.4604 | -0.3467 | 0.3550 | -0.3883 |
| 43 | test | `offi` (baseline) | 507 | 0.8071 | — | 0.7433 | — |
| 44 | test | `ours_rtm` | 858 | 0.8296 | -0.0023 | 0.7529 | -0.0077 |
| 44 | test | `ours_vis` | 858 | 0.5481 | -0.2838 | 0.4276 | -0.3331 |
| 44 | test | `offi` (baseline) | 858 | 0.8319 | — | 0.7606 | — |

## Per-class, held-out, our features under RTMPose-m (worst F1 first)

| class | support | F1 | precision | recall |
|---|---|---|---|---|
| none | 0 | 0.000 | 0.000 | 0.000 |
| Top drive | 90 | 0.595 | 0.670 | 0.538 |
| Top cross-court net shot | 43 | 0.650 | 0.572 | 0.765 |
| Bottom push | 125 | 0.662 | 0.720 | 0.629 |
| Bottom rush | 15 | 0.664 | 0.808 | 0.573 |
| Bottom drive | 110 | 0.667 | 0.759 | 0.597 |
| Top push | 145 | 0.683 | 0.674 | 0.698 |
| Bottom cross-court net shot | 61 | 0.728 | 0.733 | 0.731 |
| Top return net | 181 | 0.780 | 0.899 | 0.690 |
| Bottom return net | 166 | 0.809 | 0.783 | 0.838 |
| Bottom lob | 221 | 0.822 | 0.793 | 0.860 |
| Top lob | 226 | 0.824 | 0.790 | 0.863 |
| Bottom drop | 152 | 0.831 | 0.767 | 0.908 |
| Top rush | 34 | 0.832 | 0.850 | 0.824 |
| Top drop | 159 | 0.836 | 0.837 | 0.837 |
| Bottom smash | 219 | 0.872 | 0.907 | 0.841 |
| Top net shot | 216 | 0.881 | 0.870 | 0.893 |
| Bottom net shot | 207 | 0.883 | 0.904 | 0.864 |
| Top smash | 227 | 0.890 | 0.883 | 0.899 |
| Bottom long service | 49 | 0.921 | 0.978 | 0.871 |
| Top clear | 186 | 0.966 | 0.955 | 0.978 |
| Bottom short service | 88 | 0.974 | 0.968 | 0.981 |
| Bottom clear | 189 | 0.978 | 0.976 | 0.979 |
| Top long service | 28 | 0.989 | 0.990 | 0.989 |
| Top short service | 72 | 0.999 | 1.000 | 0.999 |

The same table for the official features, for contrast:

| class | support | F1 | precision | recall |
|---|---|---|---|---|
| none | 0 | 0.000 | 0.000 | 0.000 |
| Top drive | 90 | 0.590 | 0.669 | 0.530 |
| Bottom push | 125 | 0.631 | 0.732 | 0.573 |
| Top cross-court net shot | 43 | 0.643 | 0.602 | 0.702 |
| Bottom drive | 110 | 0.657 | 0.782 | 0.569 |
| Top push | 145 | 0.672 | 0.686 | 0.663 |
| Bottom rush | 15 | 0.680 | 0.704 | 0.673 |
| Bottom cross-court net shot | 61 | 0.690 | 0.730 | 0.661 |
| Top return net | 181 | 0.792 | 0.879 | 0.722 |
| Bottom return net | 166 | 0.805 | 0.795 | 0.817 |
| Bottom lob | 221 | 0.821 | 0.769 | 0.887 |
| Top lob | 226 | 0.826 | 0.777 | 0.882 |
| Top drop | 159 | 0.832 | 0.825 | 0.840 |
| Top rush | 34 | 0.834 | 0.892 | 0.788 |
| Bottom drop | 152 | 0.844 | 0.789 | 0.910 |
| Bottom net shot | 207 | 0.870 | 0.874 | 0.867 |
| Top smash | 227 | 0.881 | 0.868 | 0.896 |
| Bottom smash | 219 | 0.883 | 0.906 | 0.863 |
| Top net shot | 216 | 0.886 | 0.873 | 0.900 |
| Bottom long service | 49 | 0.938 | 0.993 | 0.890 |
| Top clear | 186 | 0.960 | 0.959 | 0.961 |
| Bottom clear | 189 | 0.978 | 0.974 | 0.981 |
| Bottom short service | 88 | 0.984 | 0.969 | 0.999 |
| Top long service | 28 | 0.992 | 0.984 | 1.000 |
| Top short service | 72 | 1.000 | 1.000 | 1.000 |

## Sanity checks

- match 39: all 748 evaluated clips belong to match 39
- match 40: all 525 evaluated clips belong to match 40
- match 41: all 1128 evaluated clips belong to match 41
- match 42: all 571 evaluated clips belong to match 42
- match 43: all 507 evaluated clips belong to match 43
- match 44: all 858 evaluated clips belong to match 44
- no held-out clip appears in any training-side arm (3209 held-out vs 0 training-side clip names)
- official baseline uses exactly the clip names present in our set (asserted per match; the arm aborts if not)
- clip counts per split match the files on disk (asserted on every collated dir, including the arrays-vs-disk count inside eval_arm)
- degenerate all-zero-pose clips are counted and dropped from every arm in that match (0 dropped this run)
- every cached (match, arm) directory is keyed on its inputs' size+mtime; a changed extract or pose file re-converts only that match

Any violation of these aborts the affected arm or the run; a number is never reported from a check that did not run.

## How to re-run

```sh
cd /path/to/shuttl/research && bst/venv/bin/python bst/measure.py --cache bst/acc_cache_v2 --match 39 40 41 42 43 44
```

`--wait` polls the extract and pose directories and re-reports whenever new data lands.  Converted, filtered and collated dirs are cached per (match, arm) and keyed on the inputs' size+mtime, so a re-run does only what is new or changed; `--force` ignores the cache.  Held-out aggregates stay pending until matches 39, 40, 42, 43, 44 arrive.
Log: `/path/to/shuttl/research/bst/measure.log`.

