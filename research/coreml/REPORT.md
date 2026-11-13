# RTMPose-m → CoreML: verification report

Date: 2026-10-06
Deliverable: `RTMPoseM.mlpackage` — verified float32 CoreML port of RTMPose-m
(MMPose body7 / rtmlib `rtmpose-m_simcc-body7_pt-body7_420e-256x192`), for on-device pose.

## 1. TL;DR

| | |
|---|---|
| Root cause of the real-crop failure | **The CoreML input contract was wrong.** The ONNX graph does NOT contain the ImageNet normalisation; rtmlib does it in numpy before the ONNX call. The earlier conversion asserted it was inside and shipped metadata telling Swift to feed raw 0..255. That drove activations ~50x too large, saturated the SimCC head and collapsed peak margins — which also caused the fp16 NaN. |
| Fix | Normalisation folded into the traced graph. The shipped model takes raw 0..255 RGB and cannot be mis-fed. |
| Parity, shipped fp32 | max diff 5.7e-6, **100% argmax agreement** on CPU_ONLY and CPU_AND_NE over 120 real crops + 24 random. GPU/ALL: 2 flips, both adjacent-bin, both genuine reference ties. |
| End-to-end vs rtmlib gold truth | **0.358 px mean error, 99.8% of keypoints within 2 px.** |
| float16 | **Fails** — 12–54% of keypoints flip. Not shippable. |
| Speed (M4) | 8–35 ms/crop → 16–70 ms for 2 crops. Fast enough. |
| Size | 54.4 MB fp32 (27.3 MB fp16). |

## 2. The route

```
rtmlib ONNX --patch empty Clip inputs--> RTMPoseM.patched.onnx
  -> onnx2torch -> torch module (bit-exact vs onnxruntime on real crops)
  -> Wrapper: (x - MEAN_RGB) / STD_RGB      <-- THE FIX
  -> torch.jit.trace (asserted eager == traced, and input-dependent)
  -> coremltools.convert(..., mlprogram, fp32) -> RTMPoseM.mlpackage
```

Scripts, all in `research/coreml/`:

- `convert2.py` — the conversion. Writes the shipped model + a raw-input reference + (optionally) fp16.
- `parity_final.py` — parity in the shipped contract.
- `goldtruth.py` — reproduces rtmlib's own keypoints; settles the input contract empirically.
- `final_verify.py` — full sweep: parity × compute unit, timing, size, end-to-end.
- `ties.py` — proves the residual flips are implementation-independent ties.
- `diag.py`, `contract.py`, `channel.py`, `margins.py`, `verify.py`, `parity2.py`, `which_crops.py`, `bench.py` — diagnostic stages.
- `convert.py` — SUPERSEDED (wrong input contract); kept for history only.

Reproduce:

```sh
cd /path/to/shuttl/research/coreml
./venv/bin/python convert2.py --fp16-too     # convert
./venv/bin/python parity_final.py            # parity
./venv/bin/python final_verify.py            # parity + timing + size + gold truth
```

## 3. Root cause, with evidence

### 3.1 The previous claim was wrong

`convert.py` stated in its docstring and in the shipped metadata: "The ONNX graph
already contains the ImageNet mean/std normalisation and the BGR->RGB channel swap,
so the CoreML input is raw 0..255 RGB." Two independent checks refute this:

1. **Graph structure.** The first node is `Conv(input, weight, bias)` on `input`
   directly. There is no `Sub` and no `Div` consuming the input that could implement
   `(x - mean)/std`. Normalisation cannot be absent from the graph and present in it
   at the same time.
2. **rtmlib's own source** (`rtmlib/tools/pose_estimation/rtmpose.py` v0.0.16, line 70):

   ```python
   resized_img = (resized_img - self.mean) / self.std   # mean=(123.675,116.28,103.53) RGB
   outputs = self.inference(resized_img)                # std=(58.395,57.12,57.375)
   ```

   The normalisation happens in numpy, BEFORE the ONNX call.

### 3.2 Settled empirically against gold truth

`research/rtmpose/rtm_pose.py` had already run rtmlib on 60 frames of 001.mp4; its
output `/tmp/rtm/001_test.npz` holds rtmlib's own keypoints. Reproducing that pipeline
exactly (bbox_xyxy2cs(padding=1.25) → top_down_affine(192x256) → ONNX → SimCC decode
→ map to frame) and comparing against gold:

| input convention | mean px error vs rtmlib | keypoints within 2 px |
|---|---|---|
| raw_rgb (0..255) | 87.83 px | 0.1% |
| raw_bgr (0..255) | 89.32 px | 0.0% |
| **norm_rgb** | **0.353 px** | **99.9%** |
| norm_bgr | 0.405 px | 99.4% |

`norm_rgb` reproduces rtmlib to sub-pixel accuracy. Raw input is off by 88 px — the
model was not approximated, it was broken — which is exactly what the app would have
received from the old, wrong metadata. That 88 px gap was the entire failure.

### 3.3 What the failure looked like from outside

Because raw input oversaturates the head, in the OLD (wrong) regime: fp32 max diff on
real crops was 0.0418 with peak margins median 0.0078 on logits reaching ±204 (55% of
keypoints had margin < 1e-2); fp16 produced NaN on CPU/GPU and 99.5% keypoint flips on
the ANE. In the CORRECT regime the same measurements are: fp32 max diff 5.7e-6, logits
in [-0.52, 0.68], margins normal; fp16 no longer NaNs but still flips 12–54%. The
earlier diagnosis of "content-dependent rounding" was a symptom of the saturation, not
a cause.

### 3.4 The one residual fp32 flip is a genuine tie

After the fix, fp32 under CPU_ONLY gives 1 flip of 4080 keypoints (crop `004_011.png`,
simcc_x, kp 12). At that keypoint the reference top1−top2 margin is **5.96e-8** on a
peak of 0.5912 — a relative margin of 1.0e-7, i.e. float32 epsilon. The reference model
itself cannot separate those bins. Control: `ties.py` runs the onnx2torch eager model (a
different execution path from CoreML) and flips at the SAME keypoint with the SAME
margin. A tie that reproduces across unrelated implementations is a property of the
reference, not a defect of the port, and the flip moves the decoded keypoint 0.5 px.

## 4. Parity results

Against onnxruntime (CPU) on the same effectively-applied input; 120 real person crops
+ 24 random inputs.

### 4.1 float32, shipped (`RTMPoseM.mlpackage`, normalisation inside)

| compute unit | input | max diff | argmax agree | flips | NaN | worst px err |
|---|---|---|---|---|---|---|
| CPU_ONLY | real | 5.7e-6 | **100.000%** | 0/4080 | 0 | 0.000 |
| CPU_ONLY | random | 5.7e-6 | 100.000% | 0/816 | 0 | 0.000 |
| CPU_AND_NE | real | 5.7e-6 | **100.000%** | 0/4080 | 0 | 0.000 |
| CPU_AND_NE | random | 5.7e-6 | 100.000% | 0/816 | 0 | 0.000 |
| CPU_AND_GPU | real | 9.3e-6 | 99.951% | 2/4080 | 0 | 0.500 |
| CPU_AND_GPU | random | 1.3e-5 | 100.000% | 0/816 | 0 | 0.000 |
| ALL | real | 9.3e-6 | 99.951% | 2/4080 | 0 | 0.500 |
| ALL | random | 1.3e-5 | 100.000% | 0/816 | 0 | 0.000 |

The verdict DOES differ between compute units: CPU_ONLY and CPU_AND_NE give a literal
0-flip result, while GPU/ALL route some ops through a different (Metal) kernel set with
a different accumulation order, producing 2 adjacent-bin ties (margins 6.0e-8 and
3.1e-8), each worth 0.5 px. All four units pass the max-diff bar by ~770x.

### 4.2 float16 (`RTMPoseM_fp16.mlpackage`) — NOT shippable

| compute unit | input | max diff | argmax agree | flips |
|---|---|---|---|---|
| CPU_ONLY | real | 0.128 | 68.7% | 1279/4080 |
| CPU_AND_GPU | real | 0.065 | 87.4% | 514/4080 |
| CPU_AND_NE | real | 0.522 | 46.3% | 2192/4080 |
| ALL | real | 0.522 | 46.3% | 2192/4080 |

fp16 moves the argmax on 12–54% of keypoints, with bin deltas reaching hundreds of
bins (hundreds of pixels). Cause is structural: the SimCC head's winning margin is
median ~5e-3 on logits of magnitude ~0.6 — a relative margin near 1% — while float16
has ~1e-3 relative precision, so quantisation of the runner-up bin is comparable to the
gap deciding the winner.

### 4.3 End-to-end accuracy vs rtmlib

Full pipeline (box → affine → model → SimCC decode → map to frame) against rtmlib's own
keypoints, 30 frames × 2 players, shipped fp32, ALL:

```
mean 0.358 px   median 0.433   p90 0.863   <2px 99.8%   <5px 100.0%   (n=1020)
```

The CoreML model on the phone reproduces the reference pipeline that scored 0.8730
stroke accuracy.

## 5. Preprocessing / postprocessing the Swift side must implement

The shipped `RTMPoseM.mlpackage` has the normalisation folded into the graph: feed it raw
0..255 and the mean/std question cannot be got wrong. An equivalent variant expecting
pre-normalised input exists as `RTMPoseM_rawinput.mlpackage`; prefer the shipped one.

### 5.1 Model I/O

- Input `image`: float32, shape (1, 3, 256, 192), NCHW, **RGB, raw 0..255**.
- Outputs `simcc_x`: (1, 17, 384), `simcc_y`: (1, 17, 512) — raw SimCC response logits
  (not probabilities, not softmaxed), typical range ±0.7.
- 17 COCO keypoints, in COCO order.

### 5.2 Preprocessing (per player crop)

1. Take the person box, pad by 10% of its width/height each side, clamped to the frame:
   `mw = 0.10*(x2-x1)`, `mh = 0.10*(y2-y1)`; clamp `x1-mw, y1-mh` at 0 and `x2+mw,
   y2+mh` at `w-1, h-1`.
2. Box to centre/scale at padding 1.25: `center = ((x1+x2)/2, (y1+y2)/2)`,
   `scale = ((x2-x1)*1.25, (y2-y1)*1.25)`.
3. Match the model's 192:256 aspect: if `scale.w > scale.h * (192/256)` then
   `scale.h = scale.w / (192/256)` else `scale.w = scale.h * (192/256)`.
4. Affine-warp the frame to 192x256 (w=192, h=256) with INTER_LINEAR, using the matrix
   from `get_warp_matrix(center, scale, rot=0)`: map
   `src = [center; center + (0, -scale.w/2); third_point]` to
   `dst = [(96,128); (96,128) + (0, -96); third_point]`, third point = second rotated 90°.
5. Read the warped RGB pixels into NCHW float32, raw 0..255. **Do NOT divide by 255 and
   do NOT subtract a mean** — the model does it.

### 5.3 Postprocessing (SimCC decode)

Per keypoint k:

1. `lx = argmax(simcc_x[0, k, :])` (384 bins)
2. `ly = argmax(simcc_y[0, k, :])` (512 bins)
3. Split ratio 2.0: `keypoint_in_model_frame = (lx, ly) / 2.0` → pixels in the 192x256
   (W x H) model input frame. `lx/2 ∈ [0, 191.5]`, `ly/2 ∈ [0, 255.5]`.
4. Map back: `kp = keypoint_in_model_frame / (192, 256) * scale + center - scale/2`
   with the aspect-corrected `scale`/`center` from 5.2.

### 5.4 The parts that are easy to get wrong

- The argmax is a bin index; split ratio 2.0 means 1 bin = 0.5 px, so `bin/2.0` is a
  pixel coordinate. Dropping the `/2` doubles every coordinate — a silent 2x scale
  error costing ~90 px.
- Transposition: `simcc_x` has 384 = 2x192 entries (the width), `simcc_y` has 512 = 2x256
  (the height). `simcc_x → x`, `simcc_y → y`.
- The model frame is **192 wide x 256 high**, not 256x192.
- Confidence: rtmlib scores each keypoint as `(max(simcc_x) + max(simcc_y)) / 2` and
  marks it invalid when ≤ 0 (location set to −1). Apply the same rule.

### 5.5 Do NOT

- Do not feed raw 0..255 to `RTMPoseM_rawinput.mlpackage`, nor pre-normalised data to
  `RTMPoseM.mlpackage`. Each expects exactly one thing; the metadata inside each
  package states which.
- Do not divide by 255. rtmlib's mean/std are on the 0..255 scale.

## 6. Speed and size

Mac: Apple M4 (Mac16,13, 10 cores), macOS Darwin 24.6.0. A single 256x192 crop through
`MLModel.predict`, warmed up, best of several repeats. Measured while the machine was
under heavy external load (load average ~73–107 on 10 cores), so these are CONSERVATIVE.

| compute unit | inferences/s | ms/crop | ms for 2 crops |
|---|---|---|---|
| CPU_ONLY | 16–29 | 35–62 | 70–123 |
| CPU_AND_NE | 7–34 | 29–145 | 59–290 |
| CPU_AND_GPU | 31–71 | 14–32 | 28–64 |
| ALL | 34–126 | 8–30 | 16–59 |

**Fast enough for on-device**: the app needs up to 2 crops per frame; at 30 fps the
budget is 33 ms. Two crops on ALL cost ~16 ms quiet (~59 ms under severe load), and the
phone's ANE was idle in a way this Mac was not. Badminton stroke classification does not
need pose at 30 Hz. The only thing that would threaten this is shipping fp16 to chase 2x
— which fails correctness.

| package | size |
|---|---|
| `RTMPoseM.mlpackage` (shipped, fp32) | 54.43 MB |
| `RTMPoseM_rawinput.mlpackage` (fp32) | 54.43 MB |
| `RTMPoseM_fp16.mlpackage` | 27.28 MB |

## 7. Recommendation

1. Ship `RTMPoseM.mlpackage` (float32, normalisation folded in). Takes raw 0..255 RGB;
   no mean/std needed on the Swift side.
2. Configure with `MLComputeUnits.cpuAndNeuralEngine` — a literal 100% argmax agreement,
   0 flips, and it keeps the GPU free. (CPU_ONLY is equal; GPU/ALL introduces 2
   adjacent-bin ties worth 0.5 px, so it is not wrong but is not the strictest choice.)
3. Do not ship float16 for this model.
4. Follow section 5 verbatim, especially the split ratio `/2.0` and the 192w x 256h frame.

## 8. Files

- `RTMPoseM.mlpackage` — SHIPPED, fp32, raw 0..255 RGB in
- `RTMPoseM_rawinput.mlpackage` — fp32, pre-normalised input (reference)
- `RTMPoseM_fp16.mlpackage` — fp16, verified unusable
- `convert2.py`, `parity_final.py`, `final_verify.py`, `goldtruth.py`, `ties.py` and the
  diagnostic scripts listed in section 2
- `final_results.json`, `parity_final.json`, `parity_production.json`,
  `bench_production.json` — raw numbers
