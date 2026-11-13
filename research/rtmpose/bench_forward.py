"""Timing benchmark: warm up, cache preprocessed model inputs, time ONLY the
forward pass.  This isolates the execution provider / batching / thread-count
effect from video decode and from the crop+affine preprocessing (which is pure
numpy+cv2 and identical in every configuration).
"""
import os, sys, time, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_pose import (MODEL_BODY7, INPUT_SIZE, MEAN, STD, make_session,
                       pad_box)

NPZ = "/path/to/shuttl/research/extract/out/001.npz"
VID = "/path/to/shuttl/research/shuttleset/videos/001.mp4"
NF = int(os.environ.get("BENCH_FRAMES", "300"))
ITERS = int(os.environ.get("BENCH_ITERS", "3"))


def preprocess(img, bbox):
    from rtmlib.tools.pose_estimation.rtmpose import bbox_xyxy2cs, top_down_affine
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    crop, scale = top_down_affine(INPUT_SIZE, scale, center, img)
    crop = (crop.astype(np.float32) - MEAN) / STD
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def load_inputs():
    import zipfile
    z = zipfile.ZipFile(NPZ)
    d = {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}
    fids, boxes = d["frame"].astype(int), d["player_box"]
    want = {int(f): i for i, f in enumerate(fids[:12000])}
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fids[0]))
    pos, out = int(fids[0]), []
    last = int(fids[min(11999, len(fids) - 1)])
    while pos <= last and len(out) < NF:
        ok, img = cap.read()
        if not ok:
            break
        if pos in want:
            i = want[pos]
            h, w = img.shape[:2]
            bs = [pad_box(b, w, h) for b in boxes[i] if np.all(np.isfinite(b))]
            if bs:
                out.append(np.stack([preprocess(img, b) for b in bs]))
        pos += 1
    cap.release()
    return out


def time_cfg(sess, frames, reps=ITERS):
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    for f in frames[:20]:                       # warm up (CoreML compiles here)
        sess.run(outs, {inp: f})
    ts = []
    for _ in range(reps):
        t0 = time.time()
        for f in frames:
            sess.run(outs, {inp: f})
        ts.append(time.time() - t0)
    return min(ts)


def main():
    frames = load_inputs()
    print(f"prepared {len(frames)} frame inputs, avg {np.mean([len(f) for f in frames]):.2f} boxes/frame", flush=True)
    b1 = [f[:1] for f in frames]
    b2 = [np.concatenate([f, f[-1:]], axis=0)[:2] for f in frames]

    CM = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    cfgs = [
        ("cpu_default_b1", dict(providers=["CPUExecutionProvider"]), b1),
        ("cpu_default_b2", dict(providers=["CPUExecutionProvider"]), b2),
        ("cpu_t2_b2", dict(providers=["CPUExecutionProvider"], threads=2), b2),
        ("cpu_t4_b2", dict(providers=["CPUExecutionProvider"], threads=4), b2),
        ("coreml_all_b1", dict(providers=CM, provider_options=[{"MLComputeUnits": "ALL"}]), b1),
        ("coreml_all_b2", dict(providers=CM, provider_options=[{"MLComputeUnits": "ALL"}]), b2),
        ("coreml_ane_b1", dict(providers=CM, provider_options=[{"MLComputeUnits": "CPUAndNeuralEngine"}]), b1),
        ("coreml_ane_b2", dict(providers=CM, provider_options=[{"MLComputeUnits": "CPUAndNeuralEngine"}]), b2),
        ("coreml_fixed_b2", dict(providers=CM, provider_options=[{"MLComputeUnits": "ALL"}], batch_hint=2), b2),
        ("coreml_fixed_b1", dict(providers=CM, provider_options=[{"MLComputeUnits": "ALL"}], batch_hint=1), b1),
    ]
    res = {}
    for name, kw, fr in cfgs:
        try:
            sess = make_session(MODEL_BODY7, kw.pop("providers"), **kw)
            ep = sess.get_providers()[0]
            t = time_cfg(sess, fr)
            fps = len(frames) / t
            npose = sum(len(f) for f in fr)
            res[name] = fps
            print(f"RESULT {name:18s} ep={ep:26s} {fps:6.1f} frames/s  "
                  f"({npose/t:6.1f} poses/s)", flush=True)
            del sess
        except Exception as e:
            print(f"RESULT {name} FAILED: {type(e).__name__}: {e}", flush=True)
    print("SUMMARY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
