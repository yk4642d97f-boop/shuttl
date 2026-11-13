"""SETTLE the input contract by exact reproduction of rtmlib.

We have real gold truth in /tmp/rtm/001_test.npz: rtmlib's own keypoints for 60
frames of 001.mp4, produced by research/rtmpose/rtm_pose.py (which calls
rtmlib.RTMPose with this exact ONNX). If we can re-derive those keypoints from
the video + boxes, we have proven the whole preprocessing chain.

For each candidate input convention we run:
    bbox_xyxy2cs(bbox, padding=1.25)        [rtmlib/pre_processings.py]
    top_down_affine(input_size=(192,256))   [aspect-preserving affine, INTER_LINEAR]
    <candidate channel order + normalisation>
    ONNX -> simcc
    get_simcc_maximum -> locs/2.0 -> /input_size*scale + center - scale/2

and compare against gold keypoints. The convention with near-zero pixel error
is the truth; anything else is wrong.
"""
import argparse
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
H, W = 256, 192
SPLIT = 2.0
MEAN_RGB = np.array([123.675, 116.28, 103.53], np.float32)
STD_RGB = np.array([58.395, 57.12, 57.375], np.float32)

import onnxruntime as ort


# ---------------- rtmlib reference implementations (verbatim behaviour) ----
def bbox_xyxy2cs(bbox, padding=1.0):
    """Verbatim rtmlib behaviour. Always returns 2-D center/scale of shape (1,2)
    -- rtmlib's top_down_affine does np.hsplit(bbox_scale,[1]) which requires
    2-D, so the 1-D branch is effectively dead code in the shipped pipeline."""
    bbox = np.asarray(bbox, np.float32)
    if bbox.ndim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    return center, scale


def affine_one(img, center2d, scale2d, input_size=(W, H)):
    """Play the role of rtmlib's top_down_affine for a single bbox."""
    w, h = input_size
    ar = w / h
    b_w, b_h = np.hsplit(scale2d, [1])
    scale = np.where(b_w > b_h * ar, np.hstack([b_w, b_w / ar]),
                     np.hstack([b_h * ar, b_h]))
    warp_mat = get_warp_matrix(center2d[0], scale[0], 0, output_size=(w, h))
    warped = cv2.warpAffine(img, warp_mat, (int(w), int(h)),
                            flags=cv2.INTER_LINEAR)
    return warped, scale[0]


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([[cs, -sn], [sn, cs]]) @ pt


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_warp_matrix(center, scale, rot, output_size, shift=(0.0, 0.0)):
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(input_size, bbox_scale, bbox_center, img):
    w, h = input_size
    warp_size = (int(w), int(h))
    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(b_w > b_h * aspect_ratio,
                          np.hstack([b_w, b_w / aspect_ratio]),
                          np.hstack([b_h * aspect_ratio, b_h]))
    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale


def get_simcc_maximum(sx, sy, split=2.0):
    N, K, _ = sx.shape
    fx = sx.reshape(N * K, -1)
    fy = sy.reshape(N * K, -1)
    locs = np.stack((np.argmax(fx, 1), np.argmax(fy, 1)), -1).astype(np.float32)
    vals = (np.amax(fx, 1) + np.amax(fy, 1)) / 2.0
    locs[vals <= 0.0] = -1
    return locs.reshape(N, K, 2) / split, vals.reshape(N, K)


# ---------------- candidate input conventions ------------------------------
def make_input(warped_bgr, conv):
    """warped_bgr: uint8/float BGR image from warpAffine. Returns NCHW f32."""
    if conv == "raw_rgb":
        img = warped_bgr[:, :, ::-1].astype(np.float32)
    elif conv == "raw_bgr":
        img = warped_bgr.astype(np.float32)
    elif conv == "norm_rgb":
        img = (warped_bgr[:, :, ::-1].astype(np.float32) - MEAN_RGB) / STD_RGB
    elif conv == "norm_bgr":
        img = (warped_bgr.astype(np.float32) - MEAN_RGB[::-1]) / STD_RGB[::-1]
    else:
        raise ValueError(conv)
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/rtm/001_test.npz")
    ap.add_argument("--video", default="/path/to/shuttl/research/shuttleset/videos/001.mp4")
    ap.add_argument("--onnx", default="RTMPoseM.patched.onnx")
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args()

    import zipfile
    z = zipfile.ZipFile(a.npz)
    d = {n[:-4]: np.load(z.open(n)) for n in z.namelist()}
    frames, boxes, gold = d["frame"], d["box"], d["kp"]
    n = min(a.max, len(frames))

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    inm = sess.get_inputs()[0].name

    # read the needed frames from the video
    cap = cv2.VideoCapture(a.video)
    imgs = {}
    want = set(int(f) for f in frames[:n])
    idx = 0
    while len(imgs) < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            imgs[idx] = fr
        idx += 1
    cap.release()
    print(f"gold frames {n}; read {len(imgs)} frames from video")

    convs = ["raw_rgb", "raw_bgr", "norm_rgb", "norm_bgr"]
    print(f"\n{'conv':10s} {'mean_px_err':>12s} {'median':>9s} {'p90':>9s} "
          f"{'frac<2px':>9s} {'frac<5px':>9s}")
    results = {}
    for conv in convs:
        errs = []
        for i in range(n):
            fid = int(frames[i])
            if fid not in imgs:
                continue
            fr = imgs[fid]
            hh, ww = fr.shape[:2]
            for p in range(2):
                box = boxes[i, p]
                if not np.isfinite(box).all():
                    continue
                # rtm_pose.py pads the saved box by 10% of its size before
                # calling rtmlib (it saves the UNPADDED box in the npz).
                x1, y1, x2, y2 = box
                mw, mh = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
                pad = np.array([max(x1 - mw, 0), max(y1 - mh, 0),
                                min(x2 + mw, ww - 1), min(y2 + mh, hh - 1)],
                               dtype=np.float32)
                cx, sc = bbox_xyxy2cs(pad, padding=1.25)
                warped, sc2 = affine_one(fr, cx, sc)
                x = make_input(warped, conv)
                sx, sy = sess.run(None, {inm: x})
                locs, vals = get_simcc_maximum(sx, sy)
                kp = locs[0] / np.array([W, H]) * sc2 + cx[0] - sc2 / 2
                g = gold[i, p, :, :2]
                ok = np.isfinite(g).all(-1) & (g >= 0).all(-1)
                if ok.any():
                    errs.append(np.linalg.norm(kp[ok] - g[ok], axis=-1))
        if not errs:
            print(f"{conv:10s}  no comparable keypoints")
            continue
        e = np.concatenate(errs)
        results[conv] = dict(mean=float(e.mean()), median=float(np.median(e)),
                             p90=float(np.percentile(e, 90)),
                             f2=float((e < 2).mean()), f5=float((e < 5).mean()),
                             n=int(e.size))
        print(f"{conv:10s} {e.mean():12.3f} {np.median(e):9.3f} "
              f"{np.percentile(e,90):9.3f} {(e<2).mean()*100:8.1f}% {(e<5).mean()*100:8.1f}%")

    if results:
        best = min(results.items(), key=lambda kv: kv[1]["mean"])
        print(f"\nBEST: {best[0]}  mean px error {best[1]['mean']:.3f} "
              f"({best[1]['f2']*100:.1f}% of keypoints within 2 px of rtmlib)")
        if best[1]["mean"] < 1.0:
            print(f"-> VERIFIED: input contract is '{best[0]}'")
        else:
            print("-> no convention reproduces rtmlib; investigate further")


if __name__ == "__main__":
    main()
