"""Convert RTMPose-m (MMPose body7 / rtmlib rtmpose-m) to CoreML -- CORRECTED.

THE BUG THIS FIXES
------------------
The previous version asserted the ONNX contains the ImageNet normalisation and
therefore told the iOS side to feed raw 0..255 RGB. It does NOT. rtmlib's own
source (rtmlib/tools/pose_estimation/rtmpose.py, v0.0.16) normalises in numpy
BEFORE calling the ONNX:

    resized_img = (resized_img - mean) / std
    mean = (123.675, 116.28, 103.53)   # RGB order
    std  = (58.395,  57.12,  57.375)
    outputs = self.inference(resized_img)   # <-- ONNX sees NORMALISED input

Feeding raw 0..255 to this graph pushes every activation ~50x too large, the
SimCC response saturates, and peak margins collapse (measured mean keypoint
error 87.8 px against rtmlib gold truth, vs 0.35 px when normalised).

Verified against real gold truth: /tmp/rtm/001_test.npz holds rtmlib's own
keypoints for 60 frames of 001.mp4. Reproducing them through

    bbox_xyxy2cs(padding=1.25) -> top_down_affine(192x256) -> normalise -> model

gives 0.353 px mean error for 'norm_rgb' vs 87.8 px for 'raw_rgb'.

WHAT WE SHIP
------------
We fold the normalisation INTO the traced graph, so the CoreML input is raw
0..255 RGB NCHW -- the natural output of a CVPixelBuffer -- and the Swift side
cannot get the mean/std wrong. An equivalent raw-input variant is kept as
RTMPoseM_rawinput.mlpackage for reference.

Route: ONNX -> onnx2torch -> [normalise] -> torch.jit.trace -> mlprogram fp32.
float16 is NOT usable (see REPORT.md): the SimCC head's winning margin is a
median ~5e-3 on logits of magnitude ~0.6, so fp16 quantisation flips the argmax
on 12-54% of keypoints.

  python convert2.py                # convert, write RTMPoseM.mlpackage (fp32)
  python convert2.py --verify       # convert, then run the production parity suite
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
SRC_ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")
ORIG_ONNX = os.path.expanduser(
    "~/.cache/rtmlib/hub/checkpoints/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx")
OUT = os.path.join(HERE, "RTMPoseM.mlpackage")
OUT_RAW = os.path.join(HERE, "RTMPoseM_rawinput.mlpackage")
OUT_FP16 = os.path.join(HERE, "RTMPoseM_fp16.mlpackage")

# rtmlib's constants, RGB order (rtmlib/tools/pose_estimation/rtmpose.py)
MEAN_RGB = (123.675, 116.28, 103.53)
STD_RGB = (58.395, 57.12, 57.375)


def patch_onnx(src, dst):
    """rtmlib's export leaves optional Clip inputs with an empty name; both
    onnx2torch and coremltools reject empty tensor names."""
    import onnx
    m = onnx.load(src)
    fixed = 0
    for n in m.graph.node:
        keep = [i for i in n.input if i != ""]
        if len(keep) != len(n.input):
            fixed += len(n.input) - len(keep)
            del n.input[:]
            n.input.extend(keep)
    onnx.save(m, dst)
    print(f"[patch] dropped {fixed} empty optional inputs -> {dst}")
    return dst


def build_traced(onnx_path, normalise):
    """onnx2torch -> (optional normalise wrapper) -> traced torch module."""
    import onnx
    import torch
    from onnx2torch import convert

    net = convert(onnx.load(onnx_path)).eval()

    class Wrapper(torch.nn.Module):
        """Optional (x - mean)/std with rtmlib's per-channel constants."""

        def __init__(self, inner, on):
            super().__init__()
            self.inner = inner
            self.on = on
            if on:
                m = torch.tensor(MEAN_RGB, dtype=torch.float32).view(1, 3, 1, 1)
                s = torch.tensor(STD_RGB, dtype=torch.float32).view(1, 3, 1, 1)
                self.register_buffer("mean", m)
                self.register_buffer("std", s)

        def forward(self, x):
            if self.on:
                x = (x - self.mean) / self.std
            return self.inner(x)

    model = Wrapper(net, normalise).eval()

    # pick a representative input scale for tracing
    x = (torch.rand(1, 3, H, W) * 255.0) if normalise else torch.rand(1, 3, H, W)

    with torch.no_grad():
        eager = model(x)
    eager = list(eager) if isinstance(eager, (list, tuple)) else [eager]

    traced = torch.jit.trace(model, x, check_trace=False)
    with torch.no_grad():
        t = traced(x)
    t = list(t) if isinstance(t, (list, tuple)) else [t]
    for i, (e, g) in enumerate(zip(eager, t)):
        d = float((e - g).abs().max())
        assert d < 1e-5, f"trace changed output {i} by {d}"

    # tracing must not bake in a constant
    x2 = (torch.rand(1, 3, H, W) * 255.0) if normalise else torch.rand(1, 3, H, W)
    with torch.no_grad():
        e2 = model(x2)
        t2 = traced(x2)
    e2 = e2 if isinstance(e2, (list, tuple)) else (e2,)
    t2 = t2 if isinstance(t2, (list, tuple)) else (t2,)
    for i, (e, g) in enumerate(zip(e2, t2)):
        d = float((e - g).abs().max())
        assert d < 1e-5, f"trace is not input-dependent at output {i} (diff {d})"

    print(f"[trace] eager == traced, input-dependent (normalise={normalise})")
    return traced, [tuple(tt.shape) for tt in t]


def to_coreml(traced, dst, precision, normalise):
    import coremltools as ct
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, H, W))],
        outputs=[ct.TensorType(name="simcc_x"), ct.TensorType(name="simcc_y")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=precision,
    )
    ml.user_defined_metadata["author"] = "MMPose RTMPose-m (body7) via rtmlib ONNX"
    if normalise:
        ml.user_defined_metadata["input_contract"] = (
            f"raw 0..255 RGB, NCHW 1x3x256x192, float32. Normalisation "
            f"((x - mean)/std, mean={MEAN_RGB}, std={STD_RGB}, RGB order) is "
            f"FOLDED INTO this graph.")
    else:
        ml.user_defined_metadata["input_contract"] = (
            f"PRE-NORMALISED RGB, NCHW 1x3x256x192, float32. Feed "
            f"(rgb - {MEAN_RGB}) / {STD_RGB}. The graph does NOT normalise.")
    ml.user_defined_metadata["description"] = (
        "Top-down RTMPose-m, 17 COCO keypoints. Outputs are SimCC response maps "
        "(logits), simcc_x 17x384, simcc_y 17x512. Decode: "
        "loc = argmax(row); pixel = loc / 2.0 (split ratio 2.0) in the 192x256 "
        "model input frame; map to the crop with "
        "/model_input_size*scale + center - scale/2.")
    ml.user_defined_metadata["simcc_split_ratio"] = "2.0"
    ml.user_defined_metadata["mean_rgb"] = str(MEAN_RGB)
    ml.user_defined_metadata["std_rgb"] = str(STD_RGB)
    ml.save(dst)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(dst) for f in fs)
    print(f"[save] {dst}  ({size/1e6:.2f} MB on disk)")
    return ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fp16-too", action="store_true")
    a = ap.parse_args()

    src = SRC_ONNX
    if not os.path.exists(src):
        src = patch_onnx(ORIG_ONNX, SRC_ONNX)

    # 1. the shipped model: normalisation folded in, raw 0..255 RGB input
    traced, shapes = build_traced(src, normalise=True)
    print("[shape] traced outputs", shapes)
    to_coreml(traced, OUT, __import__("coremltools").precision.FLOAT32, True)
    print("[save] metadata written")

    # 2. reference variant: expects pre-normalised input (graph identical to ONNX)
    traced_raw, _ = build_traced(src, normalise=False)
    to_coreml(traced_raw, OUT_RAW, __import__("coremltools").precision.FLOAT32, False)

    if a.fp16_too:
        import coremltools as ct
        to_coreml(traced, OUT_FP16, ct.precision.FLOAT16, True)

    if a.verify:
        import parity2
        parity2.main()


if __name__ == "__main__":
    main()
