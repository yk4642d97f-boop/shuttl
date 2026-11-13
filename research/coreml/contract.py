"""Determine the TRUE input contract, against rtmlib's own source.

rtmlib (rtmlib/tools/pose_estimation/rtmpose.py, v0.0.16):

    img, center, scale = self.preprocess(image, bbox)
      -> bbox_xyxy2cs(bbox, padding=1.25)
      -> top_down_affine(input_size=(192,256), ...)   # aspect-preserving affine
      -> resized_img = (resized_img - mean) / std     # mean=(123.675,116.28,103.53)
                                                      # std =(58.395,57.12,57.375)
    outputs = self.inference(img)                     # <-- ONNX gets NORMALISED input

    # postprocess
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs / simcc_split_ratio              # split ratio 2.0
    keypoints = keypoints / model_input_size * scale
    keypoints = keypoints + center - scale / 2

So the ONNX input is normalised (mean 0, std 1), NOT raw 0..255. The graph's
first node is Conv on `input` with no Sub, confirming no in-graph normalisation.

This script measures which (channel order, scaling) combination produces a
correct pose, using anatomical plausibility as ground truth.
"""
import glob
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
H, W = 256, 192
SPLIT = 2.0
MEAN_RGB = np.array([123.675, 116.28, 103.53], np.float32)
STD_RGB = np.array([58.395, 57.12, 57.375], np.float32)
MEAN_BGR = MEAN_RGB[::-1]
STD_BGR = STD_RGB[::-1]

import onnxruntime as ort

sess = ort.InferenceSession("RTMPoseM.patched.onnx", providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name

SK = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
      (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def decode_rtmlib(sx, sy):
    """get_simcc_maximum + scaling, exactly as rtmlib does (batch=1)."""
    N, K, _ = sx.shape
    fx = sx.reshape(N * K, -1)
    fy = sy.reshape(N * K, -1)
    xl = np.argmax(fx, axis=1)
    yl = np.argmax(fy, axis=1)
    locs = np.stack((xl, yl), axis=-1).astype(np.float32)
    vx = np.amax(fx, axis=1)
    vy = np.amax(fy, axis=1)
    vals = (vx + vy) / 2.0
    locs[vals <= 0.0] = -1
    return locs.reshape(N, K, 2) / SPLIT, vals.reshape(N, K), locs.reshape(N, K, 2)


def variants(bgr):
    """Return {name: (1,3,256,192) float32} candidate inputs."""
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = r[:, :, ::-1]
    out = {}
    out["raw_rgb_0_255"] = rgb
    out["raw_bgr_0_255"] = r
    out["norm_rgb"] = (rgb - MEAN_RGB) / STD_RGB
    out["norm_bgr"] = (r - MEAN_BGR) / STD_BGR
    # a deliberately wrong-but-plausible one: normalised with /255 first
    out["norm_rgb_div255"] = ((rgb / 255.0) - MEAN_RGB / 255.0) / (STD_RGB / 255.0)
    return {k: np.ascontiguousarray(v.transpose(2, 0, 1)[None]).astype(np.float32)
            for k, v in out.items()}


def plaus(kp):
    """Anatomical plausibility of a 17x2 pose inside a 192x256 crop."""
    L = np.array([np.hypot(*(kp[a] - kp[b])) for a, b in SK])
    full = L[3:]                      # exclude tiny eye/ear links
    ok = ((full > 0.04 * H) & (full < 0.60 * H)).mean()
    return dict(bone_med=float(np.median(full)), bone_max=float(L.max()),
                bones_ok=float(ok), upright=float(np.ptp(kp[:, 1]) / max(np.ptp(kp[:, 0]), 1e-6)),
                ymin=float(kp[:, 1].min()), ymax=float(kp[:, 1].max()))


def main():
    files = sorted(glob.glob("real_crops/*.png"))[:40]
    names = list(variants(cv2.imread(files[0])).keys())
    print(f"testing {len(names)} input conventions on {len(files)} real crops\n")

    agg = {n: [] for n in names}
    for f in files:
        bgr = cv2.imread(f)
        for n, x in variants(bgr).items():
            sx, sy = sess.run(None, {INAME: x})
            kp, vals, _ = decode_rtmlib(sx, sy)
            agg[n].append(plaus(kp[0]))

    print(f"{'convention':20s} {'bones_ok':>9s} {'bone_med':>9s} {'bone_max':>9s} "
          f"{'upright':>8s} {'y-span':>7s}")
    ranked = []
    for n in names:
        a = {k: np.mean([r[k] for r in agg[n]]) for k in agg[n][0]}
        print(f"{n:20s} {a['bones_ok']*100:8.1f}% {a['bone_med']:9.2f} {a['bone_max']:9.2f} "
              f"{a['upright']:8.2f} {a['ymax']-a['ymin']:7.1f}")
        ranked.append((a["bones_ok"], n))
    ranked.sort(reverse=True)
    print(f"\nbest by anatomical plausibility: {ranked[0][1]} "
          f"({ranked[0][0]*100:.1f}% plausible bones)")


if __name__ == "__main__":
    main()
