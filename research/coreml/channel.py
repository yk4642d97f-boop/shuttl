"""Settle the RGB vs BGR input convention by pose plausibility.

The graph has ImageNet normalisation folded into conv1 (verified: first node is
Conv on raw `input`, and no Sub/Div node consumes the input). What is NOT yet
settled is the CHANNEL ORDER, because the three ImageNet stds (58.395/57.12/
57.375) are nearly equal, so weight-energy tests cannot separate them.

Method: feed 24 real person crops under each convention, decode the pose, and
score anatomical plausibility. A wrong channel order swaps the R and B
responses, which visibly distorts the skeleton (and the model was trained with
one specific order). The correct convention should give tighter, more upright
skeletons.
"""
import glob
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
H, W = 256, 192
SPLIT = 2.0

import onnxruntime as ort

sess = ort.InferenceSession("RTMPoseM.patched.onnx", providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name

SK = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
      (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
# COCO-17 plausible edge lengths as a fraction of the 256-px crop height
# (head-torso ~0.15H, upper arm ~0.13H, thigh ~0.17H)
EXPECTED = {"torso": (0.05, 0.35), "arm": (0.06, 0.30), "leg": (0.08, 0.40)}


def decode(rx, ry):
    kx = np.argmax(rx, -1).ravel() / SPLIT
    ky = np.argmax(ry, -1).ravel() / SPLIT
    return np.stack([kx, ky], -1)


def edge_stats(kp):
    L = np.array([np.hypot(*(kp[a] - kp[b])) for a, b in SK])
    return L


def run(swap_to_rgb, n=24):
    rows = []
    for f in sorted(glob.glob("real_crops/*.png"))[:n]:
        im = cv2.imread(f)
        r = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        img = r[:, :, ::-1] if swap_to_rgb else r
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)
        rx, ry = sess.run(None, {INAME: x})
        kp = decode(rx, ry)
        L = edge_stats(kp)
        rows.append(dict(
            bone_mean=float(L.mean()), bone_max=float(L.max()),
            bone_std=float(L.std()),
            span_x=float(np.ptp(kp[:, 0])), span_y=float(np.ptp(kp[:, 1])),
            # anatomical sanity: how many bones fall in a plausible range
            in_range=float(((L > 0.05 * H) & (L < 0.55 * H)).mean()),
            head_top=float(kp[[0, 1, 2, 3, 4], 1].mean()),
            foot_bot=float(kp[[15, 16], 1].mean()),
            # a person is taller than wide
            upright=float(np.ptp(kp[:, 1]) / max(np.ptp(kp[:, 0]), 1e-6)),
        ))
    return rows


def report(tag, rows):
    a = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print(f"  {tag:16s} bone_mean={a['bone_mean']:6.2f} bone_max={a['bone_max']:7.2f} "
          f"bone_std={a['bone_std']:6.2f} spanX={a['span_x']:6.1f} spanY={a['span_y']:6.1f}")
    print(f"  {'':16s} bones_in_plausible_range={a['in_range']*100:5.1f}%  "
          f"upright_ratio={a['upright']:5.2f}  head_y={a['head_top']:6.1f} foot_y={a['foot_bot']:6.1f}")
    return a


print("=== channel-order / scaling probe on 24 real crops ===")
rgb = report("input=RGB 0-255", run(True))
bgr = report("input=BGR 0-255", run(False))

print("\n=== verdict ===")
# The correct convention gives a person-shaped skeleton: y-span > x-span and a
# majority of bones inside the plausible range.
for tag, a in (("RGB", rgb), ("BGR", bgr)):
    ok = a["upright"] > 1.08 and a["in_range"] > 0.6
    print(f"  {tag}: upright={a['upright']:.2f} (>1.08?) bones_ok={a['in_range']*100:.0f}% "
          f"(>60%?) -> {'PLAUSIBLE' if ok else 'implausible'}")
