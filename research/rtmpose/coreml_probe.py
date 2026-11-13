"""Decide whether CoreML is usable: compare its keypoints against the CPU path
and the shipped reference over a contiguous window, and report the diff
DISTRIBUTION (how many keypoints exceed 1 px, and where they sit), not just max.

Also times each config end-to-end (decode + preprocess + forward) on the same
window, which is the number that actually matters for the wall clock.
"""
import os, sys, time, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_pose import MODEL_BODY7, make_session, run_batch, pad_box

NPZ = "/path/to/shuttl/research/extract/out/001.npz"
VID = "/path/to/shuttl/research/shuttleset/videos/001.mp4"
REF = "/tmp/rtm/ab2/001_sh{}.npz"
CM = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def load_ref_finite():
    """(fid,p) -> (kp(17,3), box(4)) keeping the first FINITE value per slot."""
    recs = {}
    for s in range(4):
        d = np.load(REF.format(s))
        fids, kp, box = d["frame"], d["kp"], d["box"]
        for i in range(len(fids)):
            fid = int(fids[i])
            for p in range(2):
                k = kp[i, p]
                prev = recs.get((fid, p))
                if prev is not None and np.isfinite(prev[0][:, 0]).any():
                    continue
                if prev is not None and not np.isfinite(k[:, 0]).any():
                    continue
                recs[(fid, p)] = (k.copy(), box[i, p].copy())
    return recs


def decode(fids):
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fids[0])
    want, frames, pos = set(fids), {}, fids[0]
    while pos <= fids[-1]:
        ok, img = cap.read()
        if not ok:
            break
        if pos in want:
            frames[pos] = img
        pos += 1
    cap.release()
    return frames


def run_cfg(name, sess, slots, frames, batch=False):
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    res = {}
    t0 = time.time()
    if not batch:
        for fid, p, bx in slots:
            k, s = run_batch(sess, inp, outs, frames[fid], bx[None, :])
            res[(fid, p)] = (k[0], s[0])
    else:
        byf = {}
        for fid, p, bx in slots:
            byf.setdefault(fid, []).append((p, bx))
        for fid, items in byf.items():
            bx = np.stack([b for _, b in items])
            if len(bx) == 1:
                bx = np.concatenate([bx, bx[:1]], axis=0)
            k, s = run_batch(sess, inp, outs, frames[fid], bx[:2])
            for j, (p, _) in enumerate(items[:2]):
                res[(fid, p)] = (k[j], s[j])
    dt = time.time() - t0
    print(f"SPEED {name}: {len(slots)} slots in {dt:.1f}s = {len(slots)/dt:.1f} slots/s", flush=True)
    return res, len(slots) / dt


def dist(a, b, label):
    """Distribution of x,y differences between two result dicts."""
    ds = []
    for k in b:
        if k not in a:
            continue
        ka, kb = a[k][0][:, :2], b[k][0][:, :2]
        fa, fb = np.isfinite(ka[:, 0]), np.isfinite(kb[:, 0])
        m = fa & fb
        if m.any():
            d = np.abs(ka[m] - kb[m]).max(axis=1)
            ds.append(d)
    if not ds:
        print(f"DIST {label}: no overlap")
        return
    d = np.concatenate(ds)
    n = len(d)
    over = d > 1.0
    print(f"DIST {label}: n={n} max={d.max():.3f} mean={d.mean():.4f} "
          f"p50={np.percentile(d,50):.3f} p99={np.percentile(d,99):.3f} "
          f"p99.9={np.percentile(d,99.9):.3f}")
    print(f"DIST {label}: >1px {over.sum()} ({over.mean()*100:.3f}%)  "
          f">5px {(d>5).sum()}  >20px {(d>20).sum()}")
    return d


def main():
    NF = int(os.environ.get("PARITY_MAXF", "4000"))
    MAXS = int(os.environ.get("PARITY_SLOTS", "2500"))
    ref = load_ref_finite()
    fids_all = sorted({k[0] for k in ref})
    hi = fids_all[min(NF, len(fids_all)) - 1]
    slots_all = [(fid, p, bx) for (fid, p), (k, bx) in ref.items() if fid <= hi]
    slots_all.sort()
    fids = sorted({s[0] for s in slots_all})
    print(f"window {fids[0]}..{hi}: {len(slots_all)} slots, {len(fids)} frames", flush=True)
    frames = decode(fids)
    print(f"decoded {len(frames)} frames", flush=True)

    slots = []
    for fid, p, bx in slots_all:
        if fid not in frames or not np.all(np.isfinite(bx)):
            continue
        h, w = frames[fid].shape[:2]
        slots.append((fid, p, pad_box(bx, w, h)))
    slots = slots[:MAXS]
    print(f"using {len(slots)} slots {slots[0][0]}..{slots[-1][0]}", flush=True)

    res = {}
    s = make_session(MODEL_BODY7, ["CPUExecutionProvider"], threads=2)
    res["cpu_t2_b1"], _ = run_cfg("cpu_t2_b1", s, slots, frames, batch=False)
    del s
    s = make_session(MODEL_BODY7, CM, provider_options=[{"MLComputeUnits": "ALL"}],
                     batch_hint=1)
    res["coreml_fixb1"], fps_cm = run_cfg("coreml_fixb1", s, slots, frames, batch=False)
    del s
    s = make_session(MODEL_BODY7, CM, provider_options=[{"MLComputeUnits": "ALL"}],
                     batch_hint=2)
    res["coreml_fixb2"], _ = run_cfg("coreml_fixb2", s, slots, frames, batch=True)
    del s

    refsub = {k: (v[0], None) for k, v in ref.items()}
    for name in res:
        dist(res[name], res["cpu_t2_b1"], f"{name}-vs-cpu")
        dist(res[name], refsub, f"{name}-vs-REFERENCE")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
