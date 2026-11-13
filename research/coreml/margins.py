"""Peak-margin analysis: for the failing keypoint, how close are top1/top2?

If margin >> observed diff, the flip is a real model ambiguity; if margin is
comparable to the diff, it's a near-tie that any precision change flips.
Also reports fp16 margins to show why fp16 cannot work at all.
"""
import glob
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192

import onnxruntime as ort
import coremltools as ct

sess = ort.InferenceSession(os.path.join(HERE, "RTMPoseM.patched.onnx"),
                            providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name


def top2(vals):
    """margin = top1 - top2 along last axis, per row."""
    s = np.sort(vals, axis=-1)
    return s[..., -1] - s[..., -2]


def load(f):
    im = cv2.imread(f)
    r = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)


ml32 = ct.models.MLModel(os.path.join(HERE, "RTMPoseM.mlpackage"),
                         compute_units=ct.ComputeUnit.CPU_ONLY)
ml16 = ct.models.MLModel(os.path.join(HERE, "RTMPoseM_fp16.mlpackage"),
                         compute_units=ct.ComputeUnit.CPU_ONLY)

files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
focus = os.path.join(HERE, "real_crops", "002_006.png")

print("=== 002_006.png : the single fp32 argmax flip ===")
x = load(focus)
rx, ry = sess.run(None, {iname: x})
o = ml32.predict({"image": x})
gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])

for lbl, ref, got in (("simcc_x", rx, gx), ("simcc_y", ry, gy)):
    a, b = np.argmax(ref, -1), np.argmax(got, -1)
    m = top2(ref).ravel()
    for kp in np.argwhere(a != b):
        k = kp[0]
        p = int(b[k[0]])
        print(f" {lbl} kp={k[0]} refpeak={a[k]} coremlpeak={b[k]} "
              f"ref_val={ref[k[0],a[k[0]]]:.6f} coreml_val={got[k[0],b[k[0]]]:.6f} "
              f"ref_at_coreml_peak={ref[k[0],p]:.6f} MARGIN={m[k[0]]:.6f} "
              f"valdiff_at_coreml_peak={abs(float(ref[k[0],p])-float(got[k[0],p])):.6f}")
    print(f" {lbl}: min margin over all 17 kp = {m.min():.6f}, "
          f"2nd min = {np.sort(m)[1]:.6f}")

print("\n=== global margin stats over 120 crops (fp32 candidate) ===")
margins = []
for f in files:
    x = load(f)
    o = ml32.predict({"image": x})
    for k in ("simcc_x", "simcc_y"):
        margins.append(top2(np.asarray(o[k])).ravel())
margins = np.concatenate(margins)
print(f" n={margins.size} min={margins.min():.6f} p1={np.percentile(margins,1):.6f} "
      f"median={np.median(margins):.6f} max={margins.max():.6f}")
print(f" #margins < 1e-2 (the maxdiff bar): {(margins < 1e-2).sum()} "
      f"({(margins < 1e-2).mean()*100:.2f}%)")
print(f" #margins < 1e-3: {(margins < 1e-3).sum()}")

print("\n=== fp16 sanity: 002_006 ===")
o16 = ml16.predict({"image": x})
for lbl, ref, got in (("simcc_x", rx, o16["simcc_x"]), ("simcc_y", ry, o16["simcc_y"])):
    got = np.asarray(got)
    print(f" {lbl}: range got [{got.min():.3f},{got.max():.3f}] "
          f"ref [{ref.min():.3f},{ref.max():.3f}] "
          f"argmax agree {(np.argmax(ref,-1)==np.argmax(got,-1)).mean()*100:.1f}% "
          f"head={got.ravel()[:6]}")
