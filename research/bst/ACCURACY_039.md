# Stroke classification accuracy — BST on our pipeline's features

Generated 2026-10-06T05:32:14+00:00 by `measure.py`.  Released ShuttleSet weights (`bst_CG_AP_JnB_bone…seq_100_merged*`, 10 checkpoints averaged), seq_len 100, 25 classes.  **Nothing was trained for this report.**

Claim under test: *features from our pipeline reach the same stroke-classification accuracy as the published ShuttleSet features when BST's expected pose (RTMPose-m) is used.*

## Inputs available at this run

| match | split | extract | pose | clips kept | degenerate dropped | kept in arm(s) | note |
|---|---|---|---|---|---|---|---|
| 39 | test | yes | RTMPose-m | 748 | 0 | offi, ours_rtm, ours_vis |  |

> **HEADLINE NUMBER PENDING.** 1 of the 5 held-out matches (39, 40, 42, 43, 44) have data: 39. The held-out aggregates below are therefore not the claim's number yet; use the training-side regression table for now.

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
| `ours_rtm` | 1 | 748 | 0.8116 | 0.7300 | 0.9548 | 0.0061 |
| `ours_vis` | 1 | 748 | 0.4773 | 0.4103 | 0.6742 | 0.0459 |
| `offi` | 1 | 748 | 0.8183 | 0.7379 | 0.9592 | 0.0087 |

Our features under RTMPose-m vs the official features, same clips: **-0.0067 accuracy** (less accurate) — within the 0.0087 spread across the 10 checkpoints.

## Aggregate — val = 41 (reported separately, never pooled with the held-out set)

| arm | matches | clips | acc | macro-F1 | top-2 | acc std |
|---|---|---|---|---|---|---|
| `ours_rtm` | — | — | — | — | — | — |
| `ours_vis` | — | — | — | — | — | — |
| `offi` | — | — | — | — | — | — |

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

## Clip-for-clip vs the official features

Same clip names, same checkpoints, same metric code — the only difference is whose features the classifier saw.

| match | split | arm | clips | acc | Δ acc vs `offi` | macro-F1 | Δ F1 vs `offi` |
|---|---|---|---|---|---|---|---|
| 39 | test | `ours_rtm` | 748 | 0.8116 | -0.0067 | 0.7300 | -0.0079 |
| 39 | test | `ours_vis` | 748 | 0.4773 | -0.3410 | 0.4103 | -0.3275 |
| 39 | test | `offi` (baseline) | 748 | 0.8183 | — | 0.7379 | — |

## Per-class, held-out, our features under RTMPose-m (worst F1 first)

| class | support | F1 | precision | recall |
|---|---|---|---|---|
| none | 0 | 0.000 | 0.000 | 0.000 |
| Top long service | 0 | 0.000 | 0.000 | 0.000 |
| Top cross-court net shot | 8 | 0.572 | 0.629 | 0.537 |
| Bottom push | 41 | 0.587 | 0.841 | 0.461 |
| Top push | 59 | 0.640 | 0.831 | 0.529 |
| Top drive | 29 | 0.648 | 0.801 | 0.548 |
| Bottom rush | 3 | 0.663 | 0.602 | 0.767 |
| Bottom cross-court net shot | 10 | 0.674 | 0.811 | 0.590 |
| Bottom lob | 33 | 0.698 | 0.569 | 0.909 |
| Top lob | 37 | 0.702 | 0.556 | 0.959 |
| Bottom return net | 33 | 0.712 | 0.770 | 0.664 |
| Top rush | 11 | 0.750 | 0.735 | 0.782 |
| Top return net | 40 | 0.774 | 0.864 | 0.702 |
| Bottom drive | 26 | 0.780 | 0.753 | 0.812 |
| Bottom drop | 35 | 0.830 | 0.718 | 0.983 |
| Bottom smash | 54 | 0.840 | 0.993 | 0.730 |
| Top drop | 31 | 0.847 | 0.834 | 0.861 |
| Top smash | 38 | 0.883 | 0.878 | 0.889 |
| Bottom net shot | 63 | 0.906 | 0.877 | 0.937 |
| Top net shot | 65 | 0.916 | 0.869 | 0.968 |
| Top clear | 33 | 0.936 | 0.948 | 0.924 |
| Bottom long service | 1 | 0.940 | 0.925 | 1.000 |
| Bottom clear | 32 | 0.953 | 0.981 | 0.928 |
| Bottom short service | 36 | 1.000 | 1.000 | 1.000 |
| Top short service | 30 | 1.000 | 1.000 | 1.000 |

The same table for the official features, for contrast:

| class | support | F1 | precision | recall |
|---|---|---|---|---|
| none | 0 | 0.000 | 0.000 | 0.000 |
| Top long service | 0 | 0.000 | 0.000 | 0.000 |
| Bottom push | 41 | 0.519 | 0.805 | 0.398 |
| Top cross-court net shot | 8 | 0.602 | 0.669 | 0.562 |
| Bottom rush | 3 | 0.622 | 0.587 | 0.700 |
| Top push | 59 | 0.634 | 0.837 | 0.515 |
| Top drive | 29 | 0.636 | 0.797 | 0.531 |
| Top lob | 37 | 0.695 | 0.546 | 0.962 |
| Bottom return net | 33 | 0.698 | 0.764 | 0.645 |
| Bottom lob | 33 | 0.700 | 0.559 | 0.948 |
| Bottom cross-court net shot | 10 | 0.701 | 0.810 | 0.620 |
| Bottom drive | 26 | 0.797 | 0.790 | 0.808 |
| Top rush | 11 | 0.798 | 0.778 | 0.836 |
| Top return net | 40 | 0.829 | 0.855 | 0.805 |
| Bottom drop | 35 | 0.843 | 0.746 | 0.971 |
| Bottom smash | 54 | 0.862 | 0.989 | 0.765 |
| Top drop | 31 | 0.874 | 0.851 | 0.900 |
| Top smash | 38 | 0.882 | 0.874 | 0.892 |
| Bottom net shot | 63 | 0.911 | 0.884 | 0.940 |
| Top net shot | 65 | 0.928 | 0.899 | 0.958 |
| Top clear | 33 | 0.957 | 0.964 | 0.952 |
| Bottom clear | 32 | 0.959 | 0.997 | 0.925 |
| Bottom short service | 36 | 1.000 | 1.000 | 1.000 |
| Top short service | 30 | 1.000 | 1.000 | 1.000 |
| Bottom long service | 1 | 1.000 | 1.000 | 1.000 |

## Sanity checks

- match 39: all 748 evaluated clips belong to match 39
- no held-out clip appears in any training-side arm (748 held-out vs 0 training-side clip names)
- official baseline uses exactly the clip names present in our set (asserted per match; the arm aborts if not)
- clip counts per split match the files on disk (asserted on every collated dir, including the arrays-vs-disk count inside eval_arm)
- degenerate all-zero-pose clips are counted and dropped from every arm in that match (0 dropped this run)
- every cached (match, arm) directory is keyed on its inputs' size+mtime; a changed extract or pose file re-converts only that match

Any violation of these aborts the affected arm or the run; a number is never reported from a check that did not run.

## How to re-run

```sh
cd /path/to/shuttl/research && bst/venv/bin/python bst/measure.py --cache acc_cache_v2 --match 39
```

`--wait` polls the extract and pose directories and re-reports whenever new data lands.  Converted, filtered and collated dirs are cached per (match, arm) and keyed on the inputs' size+mtime, so a re-run does only what is new or changed; `--force` ignores the cache.  Held-out aggregates stay pending until matches 39, 40, 42, 43, 44 arrive.
Log: `/path/to/shuttl/research/bst/measure.log`.

