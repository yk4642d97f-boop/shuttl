"""FINAL verification suite.

For each package (fp32, fp16/direct if present) x each compute unit:
  * 121 real person crops + 24 random 0..255 inputs
  * max |diff| on simcc_x and simcc_y vs onnxruntime CPU reference
  * per-keypoint argmax agreement
  * NaN/Inf count
  * decoded keypoint pixel error (split ratio 2.0)
Reports the verdict per (package, unit) and writes verify_results.json.
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
ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")

import onnxruntime as ort
import coremltools as ct

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
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


def run_one(ml, cases, ref_fn):
    worst = 0.0
    nkp = 0
    flips = 0
    nans = 0
    pxworst = 0.0
    flip_info = []
    for nm, x in cases:
        rx, ry = ref_fn(x)
        o = ml.predict({"image": x})
        gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
        nans += int(np.isnan(gx).sum() + np.isnan(gy).sum())
        for lbl, r, g in (("x", rx, gx), ("y", ry, gy)):
            if r.shape != g.shape:
                return dict(worst=float("inf"), agree=0.0, nkp=0, flips=999,
                            nan=nans, px=float("inf"), notes=f"shape {g.shape} vs {r.shape}")
            worst = max(worst, float(np.abs(r.astype(np.float64)
                                            - g.astype(np.float64)).max()))
            a, b = np.argmax(r, -1), np.argmax(g, -1)
            d = a != b
            nkp += int(a.size)
            flips += int(d.sum())
            if d.any():
                s = np.sort(r, axis=-1)
                mrg = (s[..., -1] - s[..., -2]).ravel()
                for idx in np.argwhere(d.ravel()).ravel():
                    flip_info.append((nm, lbl, int(idx), float(mrg[idx])))
        kx = np.argmax(gx, -1) / SPLIT
        ky = np.argmax(gy, -1) / SPLIT
        rkx = np.argmax(rx, -1) / SPLIT
        rky = np.argmax(ry, -1) / SPLIT
        pxworst = max(pxworst, float(np.hypot(kx - rkx, ky - rky).max()))
    return dict(worst=worst, agree=flips == 0, nkp=nkp, flips=flips,
                nan=nans, px=pxworst, flips_info=flip_info)


def main():
    real, rand = make_cases()
    print(f"cases: {len(real)} real crops, {len(rand)} random")
    units = [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
             ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
             ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
             ("ALL", ct.ComputeUnit.ALL)]
    pkgs = [(n, os.path.join(HERE, p)) for n, p in
            (("float32", "RTMPoseM.mlpackage"),
             ("float16", "RTMPoseM_fp16.mlpackage"))]
    pkgs = [(n, p) for n, p in pkgs if os.path.exists(p)]

    results = {}
    for pname, ppath in pkgs:
        for uname, cu in units:
            try:
                ml = ct.models.MLModel(ppath, compute_units=cu)
            except Exception as e:
                print(f"[{pname}/{uname}] LOAD FAIL {type(e).__name__}: {str(e)[:120]}")
                continue
            for csname, cases in (("real", real), ("rand", rand)):
                r = run_one(ml, cases, lambda x: sess.run(None, {INAME: x}))
                key = f"{pname}/{uname}/{csname}"
                results[key] = {k: v for k, v in r.items() if k != "flips_info"}
                ok = (r["worst"] < 1e-2) and r["agree"] and r["nan"] == 0
                print(f"[{key:28s}] maxdiff={r['worst']:10.6f} "
                      f"argmax_agree={100*(1-r['flips']/max(r['nkp'],1)):8.3f}% "
                      f"flips={r['flips']:3d}/{r['nkp']:4d} nan={r['nan']:4d} "
                      f"px_err={r['px']:7.3f}  -> {'PASS' if ok else 'FAIL'}")
                if r.get("flips_info"):
                    for fi in r["flips_info"][:8]:
                        print(f"      flip {fi[0]} simcc_{fi[1]} kp={fi[2]} "
                              f"ref_margin={fi[3]:.8f}")
    json.dump(results, open(os.path.join(HERE, "verify_results.json"), "w"), indent=1)
    print("\nwrote verify_results.json")


if __name__ == "__main__":
    main()
