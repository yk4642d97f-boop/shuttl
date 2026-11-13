# BST reproduction on Apple M4 (2026-10-05)

- **Repo:** Va6lue/BST-Badminton-Stroke-type-Transformer @ fb9b310b, copied into `repo/`.
- **Paper:** arXiv 2502.21085 v4 (CVPRW 2026).
- **Environment:** Homebrew Python 3.12 venv in `venv/`, torch 2.14.1 with MPS.

## 1. Results: ShuttleSet merged, 25 classes, test split

The test split is matches 39, 40, 42, 43 and 44, with 3,499 strokes.

| BST-CG-AP | Acc | Macro-F1 | Min-F1 | Top-2 |
|---|---|---|---|---|
| Paper, seq100 | 0.8322 | 0.8097 | 0.5762 | 0.9594 |
| **Mean of 10 released weights, seq100** | 0.8322 | 0.8097 | 0.5762 | 0.9593 |
| Std of the 10 | 0.0036 | 0.0054 | 0.0318 | 0.0030 |
| Paper, seq30 | 0.8254 | 0.7983 | 0.5196 | 0.9503 |
| Mean of 10 released weights, seq30 | 0.8254 | 0.7982 | 0.5197 | 0.9501 |

- MPS and CPU give identical results.
- The official `bst_main.py`, with only `seq_len` and `train_partial` changed, prints the same per-model values. Log: `results/official_bst_main_seq100_cpu.log`.

**Weak classes** (seq100, F1 averaged over the 10 models):

| Class | F1 |
|---|---|
| Top drive | .590 |
| Bottom push | .635 |
| Top cross-court net shot | .643 |
| Bottom drive | .652 |
| Bottom rush | .675 (only 15 test strokes) |
| Top push | .682 |
| Bottom cross-court net shot | .687 |

- **Push vs lob:** 31–39 % of pushes are predicted as lob.
- **Cross-court net shot vs net shot:** 15–23 % confused.
- **Wrong player (Top↔Bottom):** 0.43 % of strokes.

The confusion matrix summed over the 10 models is `results/seq100_confusion_sum10.npy`.

## 2. Training on MPS

- **Speed:** about 172 s per epoch, roughly 18× the paper's RTX 4090. A full run of about 350+ epochs would take about 17 h on the Mac.
- **30-epoch run** (91 min): the best epoch was 20, with test acc .8228, macro-F1 .8010, min-F1 .5783 and top-2 .9603.
- Artifacts: `results/train_mps_bst_cg_ap_seq100_30ep.*`.

## 3. Re-run

Commands are run from `research/bst/`. Set up the environment:

```bash
/opt/homebrew/bin/python3.12 -m venv venv
venv/bin/pip install torch torchvision torcheval transformers positional-encodings torchinfo pandas scikit-learn matplotlib openpyxl "numpy<2.3"
```

The raw clip trees can be rebuilt from the packed copies:

```bash
venv/bin/python pack_raw.py unpack data/packed100 data/ShuttleSet_data_merged/dataset_npy_between_2_hits_with_max_limits
```

Collate the clips:

```bash
venv/bin/python collate.py data/ShuttleSet_data_merged/dataset_npy_between_2_hits_with_max_limits data/ShuttleSet_data_merged/dataset_npy_collated_between_2_hits_with_max_limits_seq_100 100
venv/bin/python collate.py data/ShuttleSet_data_merged/dataset_npy data/ShuttleSet_data_merged/dataset_npy_collated 30
```

Evaluate the released weights:

```bash
venv/bin/python eval_bst.py --seq-len 100 --device mps --data data/ShuttleSet_data_merged/dataset_npy_collated_between_2_hits_with_max_limits_seq_100 --out results/seq100_bst_cg_ap_mps.json --weights weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt
venv/bin/python eval_bst.py --seq-len 30 --device mps --data data/ShuttleSet_data_merged/dataset_npy_collated --out results/seq30_bst_cg_ap_mps.json --weights weights/bst_CG_AP_JnB_bone_merged*.pt
```

Official script. **Warning:** it trains for 1,600 epochs if a weight file it expects is missing.

```bash
cd repo/stroke_classification && ../../venv/bin/python main_on_shuttleset/bst_main_seq100_eval.py
```

Short training run:

```bash
venv/bin/python train_short.py --seq-len 100 --model BST_CG_AP --epochs 30 --budget-min 95 --data data/ShuttleSet_data_merged/dataset_npy_collated_between_2_hits_with_max_limits_seq_100 --out results/train_mps_bst_cg_ap_seq100_30ep.json
```

## 4. Input spec (BST-CG-AP seq100, JnB_bone, 2D)

**Model call:**
- `net = BST_CG_AP(in_dim=72, n_class=25, seq_len=100, depth_tem=2, depth_inter=1)`
- `logits = net(JnB, shuttle, pos, video_len)` returns (B, 25).

**Tensors:**

| Name | Type and shape |
|---|---|
| JnB | float32 (B,100,2,72) |
| shuttle | float32 (B,100,2) |
| pos | float32 (B,100,2,2) |
| video_len | int64 (B,), number of real frames |

- Padding is zeros at the end and is masked out.
- **Player order:** index 0 is the Top (far) player, index 1 the Bottom (near) player.
- **Classes:** id 0 is none (未知球種). Then Top_ and Bottom_ each get, in this order: 放小球, 擋小球, 殺球, 挑球, 長球, 平球, 切球, 推球, 撲球, 勾球, 發短球, 發長球. The label therefore includes who hit.

**Clip window** (native fps):
- t = fps//2, limit = fps*3//2, eps = t//2. At 30 fps that is 15, 45 and 7.
- start = previous hit, or hit−t if there is none. end = next hit + eps, or hit+t if there is none.
- start is clamped to ≥ hit−limit, and end to ≤ hit+limit+eps. The clip is frames [start, end).
- There is no resampling. A clip over 100 frames is subsampled with stride T//100, +1 if T%100 > 50. For 60 fps input, take every 2nd frame yourself.
- **The hit frames are dataset ground truth.** No evaluation with detected hits exists.

**Pose:**
- Model: MMPose `'human'`, i.e. RTMPose-m body8-256x192 (COCO-17) with the RTMDet-m person detector. Keypoint scores are unused.
- **Choosing the two players:** map each person's ankle midpoint (keypoints 15 and 16) to normalised court coordinates and keep those inside (−0.01, 1.01). Exactly 2 must remain. Otherwise the whole frame (joints, position and shuttle) is set to zero; about 8 % of test frames are all-zero.
- **Joints:** (x − bbox_cx)/diag and (y − bbox_cy)/diag, where diag is the box diagonal and image y points down.
- **Bones:** the 19 COCO pairs (0,1)(0,2)(1,2)(1,3)(2,4)(3,5)(4,6)(5,7)(7,9)(6,8)(8,10)(5,6)(5,11)(6,12)(11,12)(11,13)(13,15)(12,14)(14,16). Each bone is end − start on the normalised joints, or 0 if either endpoint is 0.
- The 72 numbers per player are 17 joints × 2 followed by 19 bones × 2.

**Shuttle:**
- TrackNetV3, the alenzenx attention variant: pixel x/W and y/H.
- Not visible = (0,0). No interpolation is applied.

**Player position:**
- Rescale the ankle midpoint to 1280×720 and map it through the per-match homography.
- Normalise to the outer doubles court: x runs 0→1 from left to right sideline, y runs 0 at the far baseline to 1 at the near baseline.

## 5. Provenance and notes

- **Weights:** Drive folder 1D4172WZDJWPvpJdpaHDhy_cA-s8F-zR5. 20 BST-CG-AP files were copied and their sha256 was verified. The folder listing may be incomplete because of a gdown timeout.
- **Data:** the seq30 RAR (sha256 039de3b7…) and the seq100 RAR (sha256 484254b5…).
  - Python's libarchive failed on them; RARLAB unrar extracted them fine.
  - They were repacked as lossless float32 npz with xz for transfer.
- **Repo changes:** only `bst_main_seq100_eval.py` and two symlinks were added.
- **Scope:** only BST-CG-AP was evaluated. `eval_bst.py` also supports BST, BST_CG and BST_AP.
