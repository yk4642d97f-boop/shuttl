"""Alternative route: convert the ONNX graph DIRECTLY with coremltools' ONNX
frontend (no onnx2torch, no torch.jit.trace). Compare numerics against the
torch route on the real crops.

Rationale: the torch route forces a trace and a graph rewrite; the native
frontend keeps the ONNX ops mostly intact and may reassociate the reductions
differently.
"""
import glob
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")
OUT = os.path.join(HERE, "RTMPoseM_direct.mlpackage")

import onnx
import onnxruntime as ort
import coremltools as ct


def load(f):
    im = cv2.imread(f)
    r = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)


def main():
    print("[1] converting ONNX -> mlprogram directly (fp32) ...")
    onnx_model = onnx.load(ONNX)
    for i in onnx_model.graph.input:
        i.type.tensor_type.shape.dim[0].dim_param = "1"
    ml = ct.convert(
        onnx_model,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        inputs=[ct.TensorType(name="image", shape=(1, 3, H, W))],
    )
    ml.save(OUT)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(OUT) for f in fs)
    print(f"[1] saved {OUT}  {size/1e6:.1f} MB")
    print("[1] spec outputs:", [o.name for o in ml.get_spec().description.output])

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))

    for cu_name, cu in (("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                        ("ALL", ct.ComputeUnit.ALL)):
        m = ct.models.MLModel(OUT, compute_units=cu)
        worst = 0.0
        flips = 0
        nkp = 0
        margin_of_flips = []
        for f in files:
            x = load(f)
            rx, ry = sess.run(None, {iname: x})
            o = m.predict({"image": x})
            gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
            for ref, got in ((rx, gx), (ry, gy)):
                worst = max(worst, float(np.abs(ref.astype(np.float64)
                                                - got.astype(np.float64)).max()))
                a, b = np.argmax(ref, -1), np.argmax(got, -1)
                d = a != b
                nkp += a.size
                flips += int(d.sum())
                if d.any():
                    s = np.sort(ref, axis=-1)
                    mrg = (s[..., -1] - s[..., -2]).ravel()
                    margin_of_flips += mrg[d.ravel()].tolist()
        print(f"[direct/{cu_name}] crops={len(files)} worst_maxdiff={worst:.6f} "
              f"argmax_flips={flips}/{nkp} ({100*(1-flips/nkp):.3f}% agree)")
        if margin_of_flips:
            print(f"            flip margins: max={max(margin_of_flips):.6f} "
                  f"min={min(margin_of_flips):.6f}")


if __name__ == "__main__":
    main()
