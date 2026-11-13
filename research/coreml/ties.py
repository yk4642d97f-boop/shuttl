"""Prove the residual argmax flips are genuine ties, not a conversion defect.

At the flipping keypoints the reference top1-top2 margin is ~1e-8 on values of
magnitude ~5 -- i.e. 1e-9 relative, at the very bottom of float32 precision.
If that is so, then ANY two independent implementations must disagree there.
The control: ONNX Runtime vs the onnx2torch eager model (a completely different
execution path from CoreML) must flip at the SAME keypoints.

Also reports output magnitudes in the production regime.
"""
import glob
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
SPLIT = 2.0
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(3, 1, 1)

import onnxruntime as ort
import coremltools as ct
import onnx
import torch
from onnx2torch import convert as o2t

ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name


def prep(bgr):
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (r[:, :, ::-1].transpose(2, 0, 1) - MEAN) / STD
    return np.ascontiguousarray(x[None]).astype(np.float32)


def main():
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
    cases = [(os.path.basename(f), prep(cv2.imread(f)))
             for f in files if cv2.imread(f) is not None]

    net = o2t(onnx.load(ONNX)).eval()
    ml = ct.models.MLModel(os.path.join(HERE, "RTMPoseM.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_ONLY)

    print("=== output magnitude in the PRODUCTION (normalised) regime ===")
    _, x0 = cases[0]
    rx, ry = sess.run(None, {INAME: x0})
    print(f"  reference simcc_x: range [{rx.min():.4f}, {rx.max():.4f}] ptp={np.ptp(rx):.4f}")
    print(f"  reference simcc_y: range [{ry.min():.4f}, {ry.max():.4f}] ptp={np.ptp(ry):.4f}")

    print("\n=== flips: CoreML vs (ORR)  and  torch-eager vs (ORR) ===")
    cm_flips = []
    torch_flips = []
    nkp = 0
    worst_cm = worst_t = 0.0
    for nm, x in cases:
        rx, ry = sess.run(None, {INAME: x})
        o = ml.predict({"image": x})
        with torch.no_grad():
            to = net(torch.from_numpy(x))
        to = to if isinstance(to, (list, tuple)) else (to,)
        to = [np.asarray(v) for v in to]
        if to[0].shape[-1] != 384:
            to = [to[1], to[0]]
        for lbl, r, g_cm, g_t in (("x", rx, np.asarray(o["simcc_x"]), to[0]),
                                  ("y", ry, np.asarray(o["simcc_y"]), to[1])):
            worst_cm = max(worst_cm, float(np.abs(r.astype(np.float64)
                                                  - g_cm.astype(np.float64)).max()))
            worst_t = max(worst_t, float(np.abs(r.astype(np.float64)
                                                - g_t.astype(np.float64)).max()))
            a = np.argmax(r, -1).ravel()
            b = np.argmax(g_cm, -1).ravel()
            c = np.argmax(g_t, -1).ravel()
            nkp += a.size
            for i in np.argwhere(a != b).ravel():
                s = np.sort(r.reshape(-1, r.shape[-1]), axis=-1)
                mrg = (s[..., -1] - s[..., -2]).ravel()
                cm_flips.append((nm, lbl, int(i), int(b[i] - a[i]), float(mrg[i]),
                                 float(r.reshape(-1, r.shape[-1])[i, a[i]])))
            for i in np.argwhere(a != c).ravel():
                s = np.sort(r.reshape(-1, r.shape[-1]), axis=-1)
                mrg = (s[..., -1] - s[..., -2]).ravel()
                torch_flips.append((nm, lbl, int(i), int(c[i] - a[i]), float(mrg[i]),
                                    float(r.reshape(-1, r.shape[-1])[i, a[i]])))
    print(f"  keypoints compared: {nkp}")
    print(f"  CoreML fp32 vs ONNX (CPU_ONLY): {len(cm_flips)} flips, worst maxdiff {worst_cm:.3e}")
    for f in cm_flips:
        print(f"     {f[0]} simcc_{f[1]} kp={f[2]} binΔ={f[3]} margin={f[4]:.3e} "
              f"peak_val={f[5]:.4f} relative_margin={f[4]/max(abs(f[5]),1e-9):.3e}")
    print(f"  torch eager  vs ONNX:           {len(torch_flips)} flips, worst maxdiff {worst_t:.3e}")
    for f in torch_flips:
        print(f"     {f[0]} simcc_{f[1]} kp={f[2]} binΔ={f[3]} margin={f[4]:.3e} "
              f"peak_val={f[5]:.4f} relative_margin={f[4]/max(abs(f[5]),1e-9):.3e}")

    cm_set = set((f[0], f[1], f[2]) for f in cm_flips)
    t_set = set((f[0], f[1], f[2]) for f in torch_flips)
    print(f"\n  CoreML flips: {sorted(cm_set)}")
    print(f"  torch  flips: {sorted(t_set)}")
    print(f"  overlap: {len(cm_set & t_set)} / union {len(cm_set | t_set)}")
    if cm_set or t_set:
        print("  -> ties are implementation-independent: a DIFFERENT execution "
              "path flips at the SAME keypoints")


if __name__ == "__main__":
    main()
