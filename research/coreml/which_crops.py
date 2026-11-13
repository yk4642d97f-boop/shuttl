"""Find exactly which real crops break fp32 parity, and characterise them."""
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
ml = ct.models.MLModel(os.path.join(HERE, "RTMPoseM.mlpackage"),
                       compute_units=ct.ComputeUnit.CPU_ONLY)

files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
print(f"{len(files)} crops")
rows = []
for f in files:
    im = cv2.imread(f)
    if im is None:
        print("unreadable", f)
        continue
    r = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
    x = np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)
    rx, ry = sess.run(None, {iname: x})
    o = ml.predict({"image": x})
    gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
    d = max(np.abs(gx.astype(np.float64) - rx.astype(np.float64)).max(),
            np.abs(gy.astype(np.float64) - ry.astype(np.float64)).max())
    ax = (np.argmax(gx, -1) == np.argmax(rx, -1)).mean()
    ay = (np.argmax(gy, -1) == np.argmax(ry, -1)).mean()
    rows.append((os.path.basename(f), d, ax, ay, im.shape, float(im.mean()),
                 float(im.std()), float(rx.max()), float(np.ptp(rx))))

bad = [r for r in rows if r[1] >= 1e-2 or r[2] < 1 or r[3] < 1]
good = [r for r in rows if r not in bad]
print(f"\nFAIL {len(bad)} / {len(rows)}   PASS {len(good)}")
print("\nfailures (name, maxdiff, agreeX, agreeY, shape, mean, std, refmax, refptp):")
for r in bad[:40]:
    print(f"  {r[0]:14s} d={r[1]:9.5f} ax={r[2]:.2f} ay={r[3]:.2f} "
          f"shape={r[4]} mean={r[5]:6.2f} std={r[6]:6.2f} refmax={r[7]:8.2f} ptp={r[8]:7.2f}")
print("\npasses sample:")
for r in good[:5]:
    print(f"  {r[0]:14s} d={r[1]:9.5f} ax={r[2]:.2f} ay={r[3]:.2f} "
          f"shape={r[4]} mean={r[5]:6.2f} std={r[6]:6.2f} refmax={r[7]:8.2f} ptp={r[8]:7.2f}")

# relation between maxdiff and reference activation magnitude?
import statistics
print("\nmedian maxdiff pass: %.6f  fail: %.6f" % (
    statistics.median([r[1] for r in good]) if good else -1,
    statistics.median([r[1] for r in bad]) if bad else -1))
