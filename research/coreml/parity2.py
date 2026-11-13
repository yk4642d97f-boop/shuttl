"""Parity and decision-relevance, measured in the PRODUCTION input regime.

Contract (verified against rtmlib gold truth, mean 0.35 px):
    x = (resized_RGB_float32 - mean_RGB) / std_RGB
    mean_RGB = (123.675, 116.28, 103.53)
    std_RGB  = (58.395,  57.12,  57.375)
Input  NCHW float32, 1x3x256x192.  Values are ~0-centred, roughly [-2.2, 2.6].

Reports per package x compute unit:
  max|diff| on simcc, argmax agreement, decoded pixel error, NaN count.
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
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(3, 1, 1)

import onnxruntime as ort
import coremltools as ct

sess = ort.InferenceSession(os.path.join(HERE, "RTMPoseM.patched.onnx"),
                            providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name


def prep(bgr):
    """Person crop -> normalised RGB NCHW, exactly the rtmlib contract."""
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = r[:, :, ::-1]                       # BGR -> RGB
    x = (rgb.transpose(2, 0, 1) - MEAN) / STD
    return np.ascontiguousarray(x[None]).astype(np.float32)


def make_cases():
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
    real = [(os.path.basename(f), prep(cv2.imread(f)))
            for f in files if cv2.imread(f) is not None]
    rng = np.random.default_rng(12345)
    rand = []
    for i in range(24):
        # random pixels in 0..255, then the SAME normalisation
        im = (rng.random((H, W, 3), dtype=np.float32) * 255.0)
        rand.append((f"rand{i}", np.ascontiguousarray(
            ((im.transpose(2, 0, 1) - MEAN) / STD)[None]).astype(np.float32)))
    return real, rand


def evaluate(ml, cases, ref):
    worst = 0.0
    nflips = 0
    nkp = 0
    nans = 0
    px_max = 0.0
    bin_deltas = []
    flips = []
    for nm, x in cases:
        rx, ry = ref(x)
        o = ml.predict({"image": x})
        gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
        nans += int(np.isnan(gx).sum() + np.isnan(gy).sum())
        for lbl, r, g in (("x", rx, gx), ("y", ry, gy)):
            worst = max(worst, float(np.abs(r.astype(np.float64)
                                            - g.astype(np.float64)).max()))
            a, b = np.argmax(r, -1), np.argmax(g, -1)
            d = a != b
            nkp += int(a.size)
            nflips += int(d.sum())
            for idx in np.argwhere(d.ravel()).ravel():
                bd = abs(int(b.ravel()[idx]) - int(a.ravel()[idx]))
                bin_deltas.append(bd)
                s = np.sort(r, axis=-1)
                mrg = (s[..., -1] - s[..., -2]).ravel()
                flips.append((nm, lbl, int(idx), bd, float(mrg[idx])))
        kx = np.argmax(gx, -1) / SPLIT
        ky = np.argmax(gy, -1) / SPLIT
        rkx = np.argmax(rx, -1) / SPLIT
        rky = np.argmax(ry, -1) / SPLIT
        px_max = max(px_max, float(np.hypot(kx - rkx, ky - rky).max()))
    return dict(worst=worst, nflips=nflips, nkp=nkp, nans=nans, px_max=px_max,
                bin_deltas=bin_deltas, flips=flips)


def main():
    real, rand = make_cases()
    print(f"cases: {len(real)} real crops, {len(rand)} random (normalised RGB)")
    units = [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
             ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
             ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
             ("ALL", ct.ComputeUnit.ALL)]
    pkgs = [(n, os.path.join(HERE, p)) for n, p in
            (("float32", "RTMPoseM.mlpackage"),
             ("float16", "RTMPoseM_fp16.mlpackage"))]
    out = {}
    for pn, pp in pkgs:
        if not os.path.exists(pp):
            continue
        for un, cu in units:
            try:
                ml = ct.models.MLModel(pp, compute_units=cu)
            except Exception as e:
                print(f"{pn}/{un}: LOAD FAIL {str(e)[:70]}")
                continue
            for cn, cases in (("real", real), ("rand", rand)):
                r = evaluate(ml, cases, lambda x: sess.run(None, {INAME: x}))
                key = f"{pn}/{un}/{cn}"
                ok = (r["worst"] < 1e-2 and r["nflips"] == 0 and r["nans"] == 0)
                print(f"{key:28s} maxdiff={r['worst']:10.6f} "
                      f"agree={100*(1-r['nflips']/max(r['nkp'],1)):7.3f}% "
                      f"flips={r['nflips']:3d}/{r['nkp']:4d} nan={r['nans']:6d} "
                      f"px_err={r['px_max']:7.3f} -> {'PASS' if ok else 'FAIL'}")
                for fi in r["flips"][:5]:
                    print(f"      flip {fi[0]} simcc_{fi[1]} kp={fi[2]} "
                          f"bin_delta={fi[3]} ref_margin={fi[4]:.8f}")
                out[key] = {k: v for k, v in r.items() if k != "flips"}
    json.dump(out, open(os.path.join(HERE, "parity_production.json"), "w"), indent=1)
    print("\nwrote parity_production.json")


if __name__ == "__main__":
    main()
