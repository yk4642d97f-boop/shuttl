"""Parity check: the fast CoreML/batched path vs the shipped CPU reference.

Ground truth is rtm_pose.py's own output on match 1 (/tmp/rtm/ab2/001_sh*.npz,
union of 4 shards covering the first 12000 slots).  For each reference slot we
re-run the SAME padded box through the fast path and compare the 17 keypoints in
full-frame pixels.  Also compares a fresh CPU run of the same fast code, so the
"fast path" scaffolding itself is validated separately from CoreML's float16.
"""
import os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_pose import MODEL_BODY7, INPUT_SIZE, MEAN, STD, make_session, pad_box

NPZ = "/path/to/shuttl/research/extract/out/001.npz"
VID = "/path/to/shuttl/research/shuttleset/videos/001.mp4"
REF = "/tmp/rtm/ab2/001_sh{}.npz"
CM = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def load_ref(fid_lo=None, fid_hi=None):
    """Union of the 4 reference shards -> per-slot records.

    A record is (kp (17,3) px, score (17,), box (4,)) so it lines up with the
    fast path's own (kp, score) tuple -- the reference npz stores (box, kp), and
    the score lives in kp[..., 2].  The box is kept so the fast path can be fed
    the SAME padded box the reference used.
    """
    recs = {}
    for s in range(4):
        d = np.load(REF.format(s))
        fids, kp, box = d["frame"], d["kp"], d["box"]
        for i in range(len(fids)):
            fid = int(fids[i])
            if fid_lo is not None and not (fid_lo <= fid <= fid_hi):
                continue
            for p in range(2):
                k = kp[i, p]
                # The 4 shards are blocks of the SAME slot grid, so a later shard
                # can hold an out-of-slot NaN for a (fid,p) an earlier shard
                # computed for real.  Keep the first FINITE value; only let a
                # NaN in when nothing finite exists for that slot.
                prev = recs.get((fid, p))
                if prev is not None and np.isfinite(prev[0][:, 0]).any():
                    continue
                if prev is not None and not np.isfinite(k[:, 0]).any():
                    continue
                recs[(fid, p)] = (k.copy(), k[:, 2].copy(), box[i, p].copy())
    return recs


def run_slots(sess, sess_batch, slots, frames, mode):
    """slots: list of (fid, p, box_padded).  mode: 'b1' or 'b2'."""
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    from fast_pose import run_batch
    res = {}
    t0 = time.time()
    if mode == "b1":
        for fid, p, bx in slots:
            kp, sc = run_batch(sess, inp, outs, frames[fid], bx[None, :])
            res[(fid, p)] = (kp[0], sc[0])
    else:
        # group consecutive slots by frame so each frame is one batch-2 call
        byf = {}
        for fid, p, bx in slots:
            byf.setdefault(fid, []).append((p, bx))
        for fid, items in byf.items():
            bx = np.stack([b for _, b in items])
            if len(bx) == 1:
                bx = np.concatenate([bx, bx[:1]], axis=0)
            kp, sc = run_batch(sess, inp, outs, frames[fid], bx[:2])
            for j, (p, _) in enumerate(items[:2]):
                res[(fid, p)] = (kp[j], sc[j])
    return res, time.time() - t0


def _score(v):
    """Score vector for either layout: kp[...,2] when present, else the 2nd item."""
    return v[0][:, 2] if v[0].shape[1] == 3 else v[1]


def compare(a, b, label, keep=None):
    """a,b: dict slot->(kp,sc). Compare keypoints where the reference is finite."""
    diffs, n_fin_a, n_fin_b, n_tot, worst = [], 0, 0, 0, []
    for k in b:
        if keep is not None and k not in keep:
            continue
        ka = a[k][0][:, :2]
        kb = b[k][0][:, :2]
        fa = np.isfinite(ka[:, 0])
        fb = np.isfinite(kb[:, 0])
        n_tot += 1
        n_fin_a += fa.sum(); n_fin_b += fb.sum()
        m = fa & fb
        if m.any():
            d = np.abs(ka[m] - kb[m]).max(axis=1)
            diffs.append(d)
            if d.max() > 1.0:
                worst.append((k, float(d.max())))
    if not diffs:
        print(f"PARITY {label}: no comparable keypoints"); return
    d = np.concatenate(diffs)
    print(f"PARITY {label}: slots={n_tot} finite A={n_fin_a} B={n_fin_b} "
          f"return_rate A={n_fin_a/(n_tot*17):.4f} B={n_fin_b/(n_tot*17):.4f}")
    print(f"PARITY {label}: kp px diff max={d.max():.4f} mean={d.mean():.4f} "
          f"p99={np.percentile(d,99):.4f}  (n={len(d)})")
    # score agreement
    sc = np.concatenate([np.abs(_score(a[k]) - _score(b[k])) for k in b
                         if keep is None or k in keep])
    print(f"PARITY {label}: score diff max={np.nanmax(sc):.4f} mean={np.nanmean(sc):.4f}")
    return float(d.max()), float(d.mean())


def main():
    t0 = time.time()
    # Restrict to an early contiguous window so the parity run decodes a
    # contiguous span (the video is sequential-decode only) instead of seeking
    # across all 12 000 reference frames.
    MAXF = int(os.environ.get("PARITY_MAXF", "4000"))
    ref_all = load_ref()
    fids_all = sorted({k[0] for k in ref_all})
    fid_hi = fids_all[min(MAXF, len(fids_all)) - 1]
    ref = load_ref(fids_all[0], fid_hi)
    print(f"reference slots: {len(ref)} in frames {fids_all[0]}..{fid_hi} "
          f"({time.time()-t0:.0f}s to load)", flush=True)

    fids = sorted({k[0] for k in ref})
    # decode only those frames
    import zipfile
    z = zipfile.ZipFile(NPZ)
    d = {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in ["frame.npy", "player_box.npy"]}
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fids[0])
    frames, pos = {}, fids[0]
    last = fids[-1]
    want = set(fids)
    while pos <= last:
        ok, img = cap.read()
        if not ok:
            break
        if pos in want:
            frames[pos] = img
        pos += 1
    cap.release()
    print(f"decoded {len(frames)}/{len(fids)} frames", flush=True)

    # build padded boxes from the reference's own box (so the input is identical)
    slots = []
    for (fid, p), (kp, sc, bx) in ref.items():
        if fid not in frames or not np.all(np.isfinite(bx)):
            continue
        h, w = frames[fid].shape[:2]
        slots.append((fid, p, pad_box(bx, w, h)))
    slots.sort()
    print(f"comparable slots with a valid box: {len(slots)}", flush=True)

    # subsample for speed -- take a contiguous run so the fast path's frames stay
    # contiguous too (keeps decode caches warm and mirrors a real shard)
    N = int(os.environ.get("PARITY_N", "1500"))
    sub = slots[:N]
    print(f"using {len(sub)} slots ({sub[0][0]}..{sub[-1][0]})", flush=True)

    # A: fresh CPU run of the fast scaffolding (batch 1)
    t = time.time()
    s_cpu = make_session(MODEL_BODY7, ["CPUExecutionProvider"])
    res_cpu, tc0 = run_slots(s_cpu, s_cpu, sub, frames, "b1")
    print(f"cpu b1: {time.time()-t:.1f}s", flush=True)
    del s_cpu

    # A2: CPU, batch 2 (the reference-style pairing) -- also gives the CPU parity
    # of the batching itself, independent of any provider float difference.
    t = time.time()
    s_cpu2 = make_session(MODEL_BODY7, ["CPUExecutionProvider"])
    res_cpu2, tc2 = run_slots(s_cpu2, s_cpu2, sub, frames, "b2")
    print(f"cpu b2: {time.time()-t:.1f}s", flush=True)
    del s_cpu2

    # B: CoreML, dynamic batch (b1-shaped calls)
    s_cm1 = make_session(MODEL_BODY7, CM, provider_options=[{"MLComputeUnits": "ALL"}])
    res_cm1, t1 = run_slots(s_cm1, s_cm1, sub, frames, "b1")
    print(f"coreml dynamic b1: {t1:.1f}s", flush=True)
    del s_cm1

    # C: CoreML, fixed batch 1
    s_cmf1 = make_session(MODEL_BODY7, CM, provider_options=[{"MLComputeUnits": "ALL"}], batch_hint=1)
    res_cmf1, t2 = run_slots(s_cmf1, s_cmf1, sub, frames, "b1")
    print(f"coreml fixed b1: {t2:.1f}s", flush=True)
    del s_cmf1

    # D: CoreML, fixed batch 2 (real pairing when 2 slots share a frame)
    s_cmf2 = make_session(MODEL_BODY7, CM, provider_options=[{"MLComputeUnits": "ALL"}], batch_hint=2)
    res_cmf2, t3 = run_slots(s_cmf2, s_cmf2, sub, frames, "b2")
    print(f"coreml fixed b2: {t3:.1f}s for {len(sub)} slots", flush=True)

    refsub = {k: ref[k] for k in [s[:2] for s in sub]}
    compare(res_cpu, refsub, "fast-CPU-b1-vs-REFERENCE")
    compare(res_cpu2, refsub, "fast-CPU-b2-vs-REFERENCE")
    compare(res_cm1, res_cpu, "coreml-dynamicB1-vs-fastCPU")
    compare(res_cmf1, res_cpu, "coreml-fixedB1-vs-fastCPU")
    compare(res_cmf2, res_cpu, "coreml-fixedB2-vs-fastCPU")
    compare(res_cmf2, refsub, "coreml-fixedB2-vs-REFERENCE")

    n = len(sub)
    print(f"SPEED_CPU_B1 {n/tc0:.1f} slots/s")
    print(f"SPEED_CPU_B2 {n/tc2:.1f} slots/s")
    print(f"SPEED_COREML_DYN_B1 {n/t1:.1f} slots/s")
    print(f"SPEED_COREML_FIX_B1 {n/t2:.1f} slots/s")
    print(f"SPEED_COREML_FIX_B2 {n/t3:.1f} slots/s")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
