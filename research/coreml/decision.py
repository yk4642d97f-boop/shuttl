"""Decision-relevance analysis.

An absolute 1e-2 bar on SimCC *logits* (range +-204) is scale-inappropriate.
What actually matters for pose is the decoded pixel location. With split ratio
2.0 the bins are 0.5 px apart, so:
  * a flip between ADJACENT bins costs 0.5 px  (negligible)
  * a flip to a bin k away costs k*0.5 px      (a real error)
We report, per package/precision, the pixel distance distribution of every
argmax disagreement, plus the keypoint-space error (PCK-style at 3 px / 5 px).
"""
import glob
import json
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
SPLIT = 2.0

import onnxruntime as ort
import coremltools as ct

sess = ort.InferenceSession(os.path.join(HERE, "RTMPoseM.patched.onnx"),
                            providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name


def load(f):
    im = cv2.imread(f)
    r = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)


def make_cases():
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
    real = [(os.path.basename(f), load(f)) for f in files if cv2.imread(f) is not None]
    rng = np.random.default_rng(12345)
    rand = [(f"rand{i}", (rng.random((1, 3, H, W), dtype=np.float32) * 255).astype(np.float32))
            for i in range(24)]
    return real, rand


def analyse(ml, cases, ref):
    """Return dict with bin-distance histogram and pixel errors of flips."""
    bin_deltas = []
    kp_px = []          # per-keypoint (per crop) euclidean error
    nkp = 0
    worst = 0.0
    for nm, x in cases:
        rx, ry = ref(x)
        o = ml.predict({"image": x})
        gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
        if np.isnan(gx).any() or np.isnan(gy).any():
            return dict(bad_nan=True)
        for r, g, axis in ((rx, gx, "x"), (ry, gy, "y")):
            worst = max(worst, float(np.abs(r.astype(np.float64)
                                            - g.astype(np.float64)).max()))
            a, b = np.argmax(r, -1), np.argmax(g, -1)
            nkp += int(a.size)
            for d in np.argwhere((a != b).ravel()).ravel():
                bin_deltas.append(int(abs(int(b.ravel()[d]) - int(a.ravel()[d]))))
        # full 2D keypoint error
        kx, ky = np.argmax(gx, -1) / SPLIT, np.argmax(gy, -1) / SPLIT
        rkx, rky = np.argmax(rx, -1) / SPLIT, np.argmax(ry, -1) / SPLIT
        kp_px.append(np.hypot(kx - rkx, ky - rky).ravel())
    kp_px = np.concatenate(kp_px)
    bd = np.array(bin_deltas) if bin_deltas else np.array([0])
    return dict(bad_nan=False, worst=worst, nkp=nkp, nflips=len(bin_deltas),
                bin_delta_hist={int(k): int((bd == k).sum())
                                for k in sorted(set(bd.tolist()))},
                px_max=float(kp_px.max()), px_p99=float(np.percentile(kp_px, 99)),
                px_mean=float(kp_px.mean()),
                pck3=float((kp_px <= 3).mean()), pck5=float((kp_px <= 5).mean()))


def main():
    real, rand = make_cases()
    print(f"cases: {len(real)} real, {len(rand)} random\n")
    units = [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
             ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
             ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
             ("ALL", ct.ComputeUnit.ALL)]
    out = {}
    for pname, ppath in (("float32", "RTMPoseM.mlpackage"),
                         ("float16", "RTMPoseM_fp16.mlpackage")):
        p = os.path.join(HERE, ppath)
        if not os.path.exists(p):
            continue
        for uname, cu in units:
            try:
                ml = ct.models.MLModel(p, compute_units=cu)
            except Exception as e:
                print(f"{pname}/{uname}: LOAD FAIL {str(e)[:80]}")
                continue
            for csname, cases in (("real", real), ("rand", rand)):
                r = analyse(ml, cases, lambda x: sess.run(None, {INAME: x}))
                k = f"{pname}/{uname}/{csname}"
                out[k] = r
                if r.get("bad_nan"):
                    print(f"{k:30s} NaN present -> unusable")
                    continue
                print(f"{k:30s} maxdiff={r['worst']:9.5f} flips={r['nflips']:5d}/{r['nkp']:5d} "
                      f"binΔ={r['bin_delta_hist']} px_max={r['px_max']:.2f} "
                      f"px_p99={r['px_p99']:.3f} PCK@3={r['pck3']*100:.2f}% "
                      f"PCK@5={r['pck5']*100:.2f}%")
    json.dump(out, open(os.path.join(HERE, "decision_results.json"), "w"), indent=1)
    print("\nwrote decision_results.json")


if __name__ == "__main__":
    main()
