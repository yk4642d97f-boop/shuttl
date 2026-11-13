"""Convert RTMPose-m (MMPose body7 / rtmlib rtmpose-m) to CoreML for on-device use.

Route: rtmlib's ONNX checkpoint -> onnx2torch (bit-exact, verified 0.0 vs
onnxruntime) -> torch.jit.trace -> coremltools mlprogram, **float32**.

Why float32 and not float16
---------------------------
The SimCC head output is 17x384 / 17x512 and is *already softmaxed* by the
exported graph: values live in [-0.28, 0.28] and each row sums to -12..-17
(i.e. they are log-probabilities). The winning-peak margin (top1 - top2) has a
median of ~4e-3, while float16 quantisation error on these magnitudes is
~6e-5-1e-4 *per element* and the converted fp16 graph amplifies that to a
0.24 max abs deviation -- larger than the entire output range. Result: the
argmax flips constantly. Measured on one random input:

    fp32: maxdiff 0.00000, argmax agreement 100.0% / 100.0%
    fp16: maxdiff 0.24396, argmax agreement  41.2% /  17.6%

A pose model whose peak moves is broken regardless of mean error, so we ship
float32. It costs ~2x model size and some ANE throughput; see REPORT.md.

The ONNX graph already contains the ImageNet mean/std normalisation and the
BGR->RGB channel swap, so the CoreML input is **raw 0..255 RGB** in NCHW
(verified: raw-255 input produces ~32 peak activations, normalised input ~0.15).

  python convert.py                     # convert + write RTMPoseM.mlpackage
  python convert.py --verify            # convert, then run the parity suite
"""
import argparse
import json
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
OUT_FP16 = os.path.join(HERE, "RTMPoseM_fp16.mlpackage")


def patch_onnx(src, dst):
    """The rtmlib export leaves optional Clip inputs with an empty name; both
    onnx2torch and coremltools reject those empty tensor names."""
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


def build_traced(onnx_path):
    """onnx2torch -> traced torch module, with an eager-vs-traced assertion."""
    import onnx
    import torch
    from onnx2torch import convert

    m = convert(onnx.load(onnx_path)).eval()
    x = torch.rand(1, 3, H, W)
    with torch.no_grad():
        eager = m(x)
    eager = list(eager) if isinstance(eager, (list, tuple)) else [eager]

    traced = torch.jit.trace(m, x, check_trace=False)
    with torch.no_grad():
        t = traced(x)
    t = list(t) if isinstance(t, (list, tuple)) else [t]
    for i, (e, g) in enumerate(zip(eager, t)):
        d = float((e - g).abs().max())
        assert d < 1e-5, f"trace changed output {i} by {d}"

    # tracing must not have baked in a constant: check a second input
    x2 = torch.rand(1, 3, H, W)
    with torch.no_grad():
        e2 = m(x2)
        t2 = traced(x2)
    e2 = e2 if isinstance(e2, (list, tuple)) else (e2,)
    t2 = t2 if isinstance(t2, (list, tuple)) else (t2,)
    for i, (e, g) in enumerate(zip(e2, t2)):
        d = float((e - g).abs().max())
        assert d < 1e-5, f"trace is input-dependent at output {i} (diff {d})"
    print("[trace] eager == traced on 2 inputs (max diff < 1e-5)")
    return traced, [tuple(t.shape) for t in t]


def to_coreml(traced, dst, precision):
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
    ml.user_defined_metadata["description"] = (
        "Top-down RTMPose-m. Input: raw 0-255 NCHW RGB (normalisation is inside "
        "the graph). Outputs: SimCC log-probs 17x384 (x) and 17x512 (y). Decode: "
        "argmax/2.0 -> pixel in the 192x256 input frame.")
    ml.user_defined_metadata["simcc_split_ratio"] = "2.0"
    ml.save(dst)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(dst) for f in fs)
    print(f"[save] {dst}  ({size/1e6:.1f} MB on disk)")
    return ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fp16-too", action="store_true",
                    help="also emit the float16 build (for the report only)")
    a = ap.parse_args()

    src = SRC_ONNX
    if not os.path.exists(src):
        src = patch_onnx(ORIG_ONNX, SRC_ONNX)

    traced, shapes = build_traced(src)
    print("[shape] traced outputs", shapes)
    ml = to_coreml(traced, OUT, __import__("coremltools").precision.FLOAT32)
    print("[save] metadata written")

    if a.fp16_too:
        import coremltools as ct
        to_coreml(traced, OUT_FP16, ct.precision.FLOAT16)

    if a.verify:
        import parity
        parity.run_all()


if __name__ == "__main__":
    main()
