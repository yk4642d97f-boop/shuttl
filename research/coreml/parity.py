"""Parity test: CoreML RTMPoseM vs the ONNX reference (run by onnxruntime).

The bar that matters is **100% per-keypoint argmax agreement** -- a pose model
that picks a different peak is broken no matter how small the mean error is.
We also report max abs diff on the SimCC tensors, and, because the argmax alone
is a weak summary, the decoded keypoint pixel error in the 192x256 input frame.

Test sets:
  * >=20 random uniform inputs (1,3,256,192) in 0..255
  * >=20 real cropped-person inputs taken from the production extractor's
    person boxes (coreml/real_crops/*.png, see make_crops.py)

  python parity.py                 # verify RTMPoseM.mlpackage (float32)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
SPLIT = 2.0
ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")


def preprocess_crop(img_bgr):
    """Resize a person crop to 192x256 and return (1,3,256,192) RGB 0..255.

    Note: this is the *test-harness* preprocessing for real crops. It is NOT
    the full rtmlib top-down affine (no aspect-ratio padding, no bbox
    expansion) -- it only exists to feed plausible real image content through
    both backends. The 0..255 RGB convention is what the CoreML model needs.
    """
    r = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = r[:, :, ::-1]                       # BGR -> RGB
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)
    return x


def load_real(maxn=40):
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
    out = []
    for f in files[:maxn]:
        im = cv2.imread(f)
        if im is None:
            continue
        out.append((os.path.basename(f), preprocess_crop(im)))
    return out


def decode(sx, sy):
    """SimCC decode -> pixel coords in the 192x256 input frame (rtmlib's formula)."""
    lx = np.argmax(sx, axis=-1).astype(np.float32) / SPLIT
    ly = np.argmax(sy, axis=-1).astype(np.float32) / SPLIT
    return np.stack([lx, ly], axis=-1), np.maximum(sx.max(-1), sy.max(-1))


def evaluate(tag, cases, ref_runs, ml):
    print(f"\n=== {tag} ===")
    worst_diff = 0.0
    agree_all = True
    px_errs = []
    for name, x in cases:
        r = ref_runs(x)
        o = ml.predict({"image": x})
        gx = np.array(o["simcc_x"])
        gy = np.array(o["simcc_y"])
        rx, ry = r[0], r[1]
        for lbl, g, rr in (("simcc_x", gx, rx), ("simcc_y", gy, ry)):
            if g.shape != rr.shape:
                print(f"  {name} {lbl}: SHAPE {g.shape} vs {rr.shape}")
                agree_all = False
                continue
            d = np.abs(g.astype(np.float64) - rr.astype(np.float64)).max()
            worst_diff = max(worst_diff, float(d))
            a = np.argmax(g, axis=-1)
            b = np.argmax(rr, axis=-1)
            agree = (a == b).mean()
            if agree < 1.0:
                agree_all = False
                bad = np.argwhere(a != b)
                print(f"  {name} {lbl}: maxdiff {d:.5f} argmax agree {agree*100:.1f}% "
                      f"({len(bad)} mismatched keypoints, e.g. kp {bad[0].tolist()})")
        kp_g, _ = decode(gx, gy)
        kp_r, _ = decode(rx, ry)
        px_errs.append(np.linalg.norm(kp_g - kp_r, axis=-1).max())
    print(f"  worst max|diff| over set: {worst_diff:.6f}")
    print(f"  worst keypoint pixel error: {max(px_errs):.4f} px (of {W}x{H} input)")
    print(f"  ALL ARGMAX AGREE: {agree_all}")
    return worst_diff, agree_all


def run_all():
    import coremltools as ct
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ref = lambda x: sess.run(None, {iname: x})

    rng = np.random.default_rng(12345)
    rand_cases = [(f"rand{i}", (rng.random((1, 3, H, W), dtype=np.float32)
                              * 255.0).astype(np.float32)) for i in range(24)]
    real = load_real(40)
    print(f"real crops available: {len(real)}")

    results = {}
    targets = [("float32", os.path.join(HERE, "RTMPoseM.mlpackage"))]
    fp16 = os.path.join(HERE, "RTMPoseM_fp16.mlpackage")
    if os.path.exists(fp16):
        targets.append(("float16", fp16))

    for label, path in targets:
        if not os.path.exists(path):
            continue
        ml = ct.models.MLModel(path)
        results[label] = {}
        for tag, cases in (("random inputs", rand_cases), ("real person crops", real)):
            results[label][tag] = evaluate(tag, cases, ref, ml)

    print("\n=== SUMMARY ===")
    for label, r in results.items():
        ok = all(v[1] for v in r.values())
        print(f"{label}: " + "  ".join(f"{k}: maxdiff {v[0]:.5f} agree {v[1]}"
                                       for k, v in r.items()))
        print(f"  -> {'PASS' if ok else 'FAIL'}")
    json.dump({k: {kk: [vv[0], bool(vv[1])] for kk, vv in v.items()}
               for k, v in results.items()},
              open(os.path.join(HERE, "parity_results.json"), "w"), indent=1)
    return results


if __name__ == "__main__":
    run_all()
