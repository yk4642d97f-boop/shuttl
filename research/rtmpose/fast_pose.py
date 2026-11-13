"""Fast batched RTMPose path + benchmark against the plain CPU path.

The reference (rtm_pose.py) calls rtmlib's RTMPose once per player box: 2 model
forwards per frame, no session tuning.  This module exposes the same
pre/post-processing (rtmlib's own helpers) but packs the 2 boxes of a frame into
a single (2,3,256,192) forward pass, and lets the execution provider be chosen.
Numerics therefore stay comparable to the reference.
"""
import os, time
import numpy as np
import cv2
from rtmlib.tools.pose_estimation.rtmpose import (bbox_xyxy2cs, top_down_affine,
                                                  get_simcc_maximum)

MODEL_BODY7 = os.path.join(
    os.path.expanduser("~/.cache/rtmlib/hub/checkpoints"),
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx")
INPUT_SIZE = (192, 256)          # (w, h) -- must not change, reference uses this
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def make_session(onnx_path, providers, provider_options=None, threads=None,
                 batch_hint=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    if batch_hint:                       # fixed batch so CoreML can fuse
        fn = getattr(so, "add_free_dimension_override_by_name", None)
        if fn is None:
            raise RuntimeError("this onnxruntime has no free-dimension override API")
        fn("batch", batch_hint)
    kw = dict(sess_options=so, providers=providers)
    if provider_options is not None:
        # onnxruntime requires provider_options to be the same length as
        # providers, otherwise it silently falls back to CPU only.
        po = list(provider_options)
        while len(po) < len(providers):
            po.append({})
        kw["provider_options"] = po
    return ort.InferenceSession(onnx_path, **kw)


def run_batch(sess, inp_name, out_names, img, boxes_pad):
    """boxes_pad: (P,4) already padded xyxy.  Returns kp (P,17,2) px, sc (P,17)."""
    P = len(boxes_pad)
    x = np.empty((P, 3, INPUT_SIZE[1], INPUT_SIZE[0]), dtype=np.float32)
    centers = np.empty((P, 2), dtype=np.float32)
    scales = np.empty((P, 2), dtype=np.float32)
    for j in range(P):
        center, scale = bbox_xyxy2cs(boxes_pad[j], padding=1.25)
        crop, scale = top_down_affine(INPUT_SIZE, scale, center, img)
        crop = (crop.astype(np.float32) - MEAN) / STD
        x[j] = crop.transpose(2, 0, 1)
        centers[j] = center
        scales[j] = scale
    outs = sess.run(out_names, {inp_name: x})
    locs, scores = get_simcc_maximum(outs[0], outs[1])   # (P,17,2) (P,17)
    kp = locs / 2.0
    kp = kp / np.array(INPUT_SIZE, dtype=np.float32) * scales[:, None, :]
    kp = kp + centers[:, None, :] - scales[:, None, :] / 2
    return kp.astype(np.float32), scores.astype(np.float32)


def pad_box(b, w, h):
    x1, y1, x2, y2 = b
    mw, mh = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
    return np.array([max(x1 - mw, 0), max(y1 - mh, 0),
                     min(x2 + mw, w - 1), min(y2 + mh, h - 1)], dtype=np.float32)


# ---------------------------------------------------------------- benchmark --
def load_frames(npz_path, video_path, lo_frame_idx, hi_frame_idx):
    """Decode video for input frame-index range and return {frame_id: img}."""
    import zipfile
    z = zipfile.ZipFile(npz_path)
    d = {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}
    fids = d["frame"].astype(int)
    boxes = d["player_box"]
    want = set(int(f) for f in fids[lo_frame_idx:hi_frame_idx])
    cap = cv2.VideoCapture(video_path)
    out = {}
    pos = int(fids[lo_frame_idx])
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    last = int(fids[hi_frame_idx - 1])
    while pos <= last:
        ok, img = cap.read()
        if not ok:
            break
        if pos in want:
            out[pos] = img
        pos += 1
    cap.release()
    return d, fids, boxes, out


def bench(name, sess, inp_name, out_names, frames, boxes_by_fid, fids_slice, n_batches):
    ids = [f for f in fids_slice if f in frames]
    ids = ids[:n_batches]
    t0 = time.time()
    used = 0
    for fid in ids:
        img = frames[fid]
        h, w = img.shape[:2]
        bs = boxes_by_fid[fid]
        pb = [pad_box(b, w, h) for b in bs if np.all(np.isfinite(b))]
        if not pb:
            continue
        run_batch(sess, inp_name, out_names, img, np.stack(pb))
        used += 1
    dt = time.time() - t0
    print(f"RESULT {name}: {used} frames in {dt:.2f}s = {used/dt:.1f} frames/s "
          f"({2*used/dt:.1f} poses/s)", flush=True)
    return used / dt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/path/to/shuttl/research/extract/out/001.npz")
    ap.add_argument("--video", default="/path/to/shuttl/research/shuttleset/videos/001.mp4")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()

    d, fids, boxes, frames = load_frames(a.npz, a.video, 0, 12000)
    fids_slice = list(fids[:12000])
    boxes_by_fid = {int(fids[i]): boxes[i] for i in range(12000)}
    print(f"cached {len(frames)} frames for the benchmark", flush=True)

    import onnxruntime as ort
    import json
    print("available providers:", ort.get_available_providers(), flush=True)

    results = {}

    # 1. reference-style: one box at a time, default CPU session
    s_cpu = make_session(MODEL_BODY7, ["CPUExecutionProvider"])
    inp = s_cpu.get_inputs()[0].name
    outs = [o.name for o in s_cpu.get_outputs()]
    print("model input:", s_cpu.get_inputs()[0].shape, flush=True)

    def bench_onebyone(name, sess, n):
        inp = sess.get_inputs()[0].name
        outs = [o.name for o in sess.get_outputs()]
        ids = [f for f in fids_slice if f in frames][:n]
        t0 = time.time(); used = 0
        for fid in ids:
            img = frames[fid]; h, w = img.shape[:2]
            for b in boxes_by_fid[fid]:
                if not np.all(np.isfinite(b)):
                    continue
                run_batch(sess, inp, outs, img, pad_box(b, w, h)[None, :])
            used += 1
        dt = time.time() - t0
        print(f"RESULT {name}: {used} frames in {dt:.2f}s = {used/dt:.1f} frames/s", flush=True)
        return used / dt

    results["cpu_default_1by1"] = bench_onebyone("cpu_default_1by1", s_cpu, 200)
    results["cpu_default_batch2"] = bench("cpu_default_batch2", s_cpu, inp, outs,
                                          frames, boxes_by_fid, fids_slice, a.n)
    del s_cpu

    for threads in (4, 6):
        try:
            s = make_session(MODEL_BODY7, ["CPUExecutionProvider"], threads=threads)
            results[f"cpu_t{threads}_batch2"] = bench(f"cpu_t{threads}_batch2", s,
                                                      s.get_inputs()[0].name,
                                                      [o.name for o in s.get_outputs()],
                                                      frames, boxes_by_fid, fids_slice, a.n)
            del s
        except Exception as e:
            print(f"RESULT cpu_t{threads}_batch2 FAILED: {e}", flush=True)

    coreml_variants = [
        ("coreml_all_b1", {"MLComputeUnits": "ALL"}, None),
        ("coreml_all_b2", {"MLComputeUnits": "ALL"}, 2),
        ("coreml_cpuonly_b2", {"MLComputeUnits": "CPUOnly"}, 2),
    ]
    for name, popts, bh in coreml_variants:
        try:
            opts = [dict(popts)] if bh is None else [dict(popts), {"__override": bh}]
            if bh is None:
                s = make_session(MODEL_BODY7, ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                                 provider_options=[popts])
            else:
                s = make_session(MODEL_BODY7, ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                                 provider_options=[popts], batch_hint=bh)
            results[name] = bench(name, s, s.get_inputs()[0].name,
                                  [o.name for o in s.get_outputs()],
                                  frames, boxes_by_fid, fids_slice, a.n)
            del s
        except Exception as e:
            print(f"RESULT {name} FAILED: {type(e).__name__}: {e}", flush=True)

    print("SUMMARY " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
