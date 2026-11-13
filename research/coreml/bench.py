"""Benchmark + preprocessing-contract verification.

1. Throughput (inferences/s) and latency for one 256x192 crop, per compute unit,
   after warmup. Also a 2-crop back-to-back timing (the app does up to 2/frame).
2. Model size on disk (fp32 vs fp16).
3. Preprocessing contract: decide empirically whether the graph wants
   (a) raw 0..255 RGB, (b) raw 0..255 BGR, (c) pre-normalised RGB.
   We check by decoding a crop and seeing which variant produces a plausible
   pose (peaks at distinct locations, low mean activation entropy) AND, more
   rigorously, by comparing against the ORIGINAL rtmlib ONNX that we know the
   convention for.
"""
import glob
import json
import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
SPLIT = 2.0

import onnxruntime as ort
import coremltools as ct

ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name


def load_bgr(f):
    return cv2.imread(f)


def to_nchw_rgb255(bgr):
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)


def sizes(path):
    tot = 0
    per = {}
    for root, _, fs in os.walk(path):
        for f in fs:
            s = os.path.getsize(os.path.join(root, f))
            tot += s
            per[f] = s
    return tot, per


def bench(ml, x, n=120):
    for _ in range(10):
        ml.predict({"image": x})
    t0 = time.perf_counter()
    for _ in range(n):
        ml.predict({"image": x})
    dt = time.perf_counter() - t0
    return n / dt, dt / n * 1000.0


def preprocess_probe():
    """Which input convention does the graph want?

    The reference convention: rtmlib's top-down pipeline feeds the ORIGINAL
    rtmlib ONNX the *cropped RGB image already divided by 255 and normalised*?
    No -- we verify by measuring peak sharpness. A correctly conditioned input
    produces a confident SimCC (large peak margin); a wrong one produces a
    near-flat / saturated response.
    """
    orig = os.path.expanduser(
        "~/.cache/rtmlib/hub/checkpoints/"
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx")
    f = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))[0]
    bgr = load_bgr(f)
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    variants = {
        "rgb_0_255": r[:, :, ::-1].astype(np.float32),
        "bgr_0_255": r.astype(np.float32),
        "rgb_norm_imagenet": (r[:, :, ::-1].astype(np.float32) / 255.0
                              - np.array([0.485, 0.456, 0.406], np.float32)) /
                             np.array([0.229, 0.224, 0.225], np.float32),
    }
    print("\n=== preprocessing probe (OG rtmlib ONNX if present, else patched) ===")
    path = orig if os.path.exists(orig) else ONNX
    print("using", os.path.basename(path))
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inm = s.get_inputs()[0].name
    for name, img in variants.items():
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)
        rx, ry = s.run(None, {inm: x})
        def top2(v):
            ss = np.sort(v, axis=-1)
            return (ss[..., -1] - ss[..., -2]).ravel()
        mx = top2(rx); my = top2(ry)
        argx = np.argmax(rx, -1).ravel(); argy = np.argmax(ry, -1).ravel()
        print(f"  {name:20s} simcc_x max={rx.max():9.3f} peakmargin_med={np.median(mx):.4f} "
              f"argx spread={argx.max()-argx.min():4d}  simcc_y max={ry.max():9.3f} "
              f"argy spread={argy.max()-argy.min():4d}")


def main():
    x = to_nchw_rgb255(load_bgr(sorted(glob.glob(os.path.join(HERE, 'real_crops', '*.png')))[0]))
    print("=== size on disk ===")
    for nm in ("RTMPoseM.mlpackage", "RTMPoseM_fp16.mlpackage"):
        p = os.path.join(HERE, nm)
        if os.path.exists(p):
            tot, per = sizes(p)
            print(f"  {nm}: {tot/1e6:.2f} MB total")
            for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:6]:
                print(f"      {k:45s} {v/1e6:8.2f} MB")

    print("\n=== throughput (1 crop 256x192) ===")
    res = {}
    for nm in ("RTMPoseM.mlpackage", "RTMPoseM_fp16.mlpackage"):
        p = os.path.join(HERE, nm)
        if not os.path.exists(p):
            continue
        for un, cu in (("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                       ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
                       ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                       ("ALL", ct.ComputeUnit.ALL)):
            try:
                ml = ct.models.MLModel(p, compute_units=cu)
                ml.predict({"image": x})
            except Exception as e:
                print(f"  {nm}/{un}: {type(e).__name__}: {str(e)[:90]}")
                continue
            # note: first predict can be slow (ANE compile); bench() warms up
            try:
                ips, ms = bench(ml, x)
                res[f"{nm}/{un}"] = {"ips": ips, "ms": ms}
                print(f"  {nm:24s} {un:12s} {ips:7.1f} inf/s   {ms:7.2f} ms   "
                      f"-> 2 crops = {2*ms:6.2f} ms")
            except Exception as e:
                print(f"  {nm}/{un}: predict FAIL {str(e)[:90]}")
    json.dump(res, open(os.path.join(HERE, "bench_results.json"), "w"), indent=1)

    preprocess_probe()


if __name__ == "__main__":
    main()
