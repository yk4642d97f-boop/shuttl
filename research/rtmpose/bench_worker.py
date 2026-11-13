"""One benchmark shard: same decode+pose work as run_pose.py's worker, but it
prints its own running frame count every second so the aggregate benchmark can
read throughput without waiting for a shard to finish.
"""
import os, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_pose import MODEL_BODY7, make_session, run_batch, pad_box
cv2.setNumThreads(1)


def main():
    npz, video, si, sn, threads = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                   int(sys.argv[4]), int(sys.argv[5]))
    import zipfile
    z = zipfile.ZipFile(npz)
    fids = np.load(z.open("frame.npy")).astype(int)
    boxes = np.load(z.open("player_box.npy"))
    z.close()
    n = len(fids)
    block = (n + sn - 1) // sn
    lo, hi = si * block, min(n, (si + 1) * block)
    if lo >= hi:
        return
    sess = make_session(MODEL_BODY7, ["CPUExecutionProvider"], threads=threads)
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fids[lo]))
    pos = int(fids[lo])
    want = {int(fids[i]): i for i in range(lo, hi)}
    last = int(fids[hi - 1])
    t0 = time.time()
    tlast = t0
    done = 0
    while want and pos <= last:
        ok, img = cap.read()
        if not ok:
            break
        i = want.pop(pos, None)
        pos += 1
        if i is None:
            continue
        h, w = img.shape[:2]
        pbs, ps = [], []
        for p in range(2):
            if np.all(np.isfinite(boxes[i, p])):
                pbs.append(pad_box(boxes[i, p], w, h))
                ps.append(p)
        if pbs:
            run_batch(sess, inp, outs, img, np.stack(pbs))
        done += 1
        now = time.time()
        if now - tlast >= 1.0:
            print(f"TICK {si} {done}", flush=True)
            tlast = now
    print(f"SHARD_DONE {si} {done} {time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
