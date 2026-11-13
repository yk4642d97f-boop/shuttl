"""Batch RTMPose driver for every extracted match.

Walks ../extract/out/*.npz, runs the PROVEN-CORRECT CPU path (fast_pose's
batched forward, which the parity check matched to rtm_pose.py's reference at
max 0.41 px over 10200 keypoints), and writes rtmpose/pose/NNN.npz with the same
layout as rtm_pose.py: kp (N,2,17,3) full-frame px + score, box (N,2,4), frame (N,).

Design notes
  * CPU only.  CoreML is 3-4x faster but its fp16 numerics move keypoints by up
    to 143 px vs the reference -- far past the 1 px bar -- so it is discarded.
  * intra_op_num_threads=2 is the sweet spot for one process; the driver runs
    several single-purpose shard workers instead of one fat process, because the
    work is memory-bandwidth bound and does not scale with threads.
  * Each match is split into S block shards so a match finishes even if it is
    long, and the caller can keep MAX_WORKERS processes busy across matches.
  * Match 1's first 12000 slots are seeded from the shipped reference shards in
    /tmp/rtm/ab2 instead of being recomputed.
  * New extract/*.npz files are picked up by polling; matches already present in
    pose/ are skipped, failures are retried and then marked.

Usage:
  python run_pose.py --hours 10 --workers 6 --shards 4
"""
import argparse, os, sys, time, json, subprocess, signal, glob
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_pose import MODEL_BODY7, make_session, run_batch, pad_box

EXTRACT = os.path.join(HERE, "..", "extract", "out")
VIDEOS = os.path.join(HERE, "..", "shuttleset", "videos")
POSE = os.path.join(HERE, "pose")
REFSHARDS = "/tmp/rtm/ab2/{:03d}_sh{}.npz"
SEED_LIMIT = 12000          # slots seeded from the reference shards for match 1

# The held-out accuracy number rests on these matches only, so pose them first.
# 039/040 are test, 041 val, 042-044 test; everything else is a regression figure.
PRIORITY = ["039", "040", "041", "042", "043", "044"]

# keep OpenCV from spawning its own thread pool in every worker
cv2.setNumThreads(1)


def order_key(path):
    """Sort matches: the priority six first (in PRIORITY order), then the rest."""
    mid = match_id(path)
    if mid in PRIORITY:
        return (0, PRIORITY.index(mid), 0)
    try:
        return (1, 0, int(mid))
    except ValueError:
        return (1, 1, 0)


def ordered(files):
    return sorted(files, key=order_key)


def match_id(path):
    return os.path.basename(path).split(".")[0]


def mid_from_path(path):
    return match_id(path)


CHECKPOINT_EVERY = 250          # frames between shard checkpoint writes


def _checkpoint(prog, kp, shard_fids, lo, done_upto):
    """Write this shard's partial result plus a done-prefix flag.

    done_upto is an ABSOLUTE slot index: slots [lo, done_upto) are finished.
    The decode is sequential, so the finished slots always form a prefix.
    """
    flags = np.zeros(len(shard_fids), dtype=bool)
    flags[:max(0, done_upto - lo)] = True
    tmp = prog + ".tmp.npz"
    np.savez(tmp, kp=kp, frame=shard_fids, done=flags)
    os.replace(tmp, prog)


def read_meta(npz_path):
    """frame ids and boxes, without pulling the pose arrays into memory."""
    import zipfile
    z = zipfile.ZipFile(npz_path)
    fids = np.load(z.open("frame.npy")).astype(int)
    boxes = np.load(z.open("player_box.npy"))
    z.close()
    return fids, boxes


def slot_bounds(n, shards, i):
    block = (n + shards - 1) // shards
    return i * block, min(n, (i + 1) * block)


# ------------------------------------------------------------------ worker --
def worker(npz, video, out, shard, threads):
    """One shard: decode sequentially, pose both boxes, save this shard's slice."""
    si, sn = shard
    fids, boxes = read_meta(npz)
    n = len(fids)
    lo, hi = slot_bounds(n, sn, si)
    if mid_from_path(npz) == "001":
        # slots [0, SEED_LIMIT) are taken from the shipped reference shards
        lo = max(lo, min(SEED_LIMIT, hi))
    kp = np.full((hi - lo, 2, 17, 3), np.nan, dtype=np.float32)
    used = np.array(boxes[lo:hi], dtype=np.float32)
    sess = make_session(MODEL_BODY7, ["CPUExecutionProvider"], threads=threads)
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    cap = cv2.VideoCapture(video)
    if lo >= hi:
        np.savez_compressed(out, kp=kp, box=used, frame=fids[lo:hi])
        return 0
    # Resume from this shard's own checkpoint if one exists: it holds a done-flag
    # per slot, so a SIGTERM from anything on the box costs at most
    # CHECKPOINT_EVERY frames of work instead of the whole shard.
    prog = out + ".prog.npz"
    start_slot = lo
    if os.path.exists(prog):
        try:
            d = np.load(prog)
            if d["frame"].shape[0] == hi - lo and np.array_equal(d["frame"], fids[lo:hi]):
                kp = d["kp"].copy()
                dflags = d["done"]
                n_done = int(dflags.sum())
                if dflags.all():
                    np.savez_compressed(out, kp=kp, box=used, frame=fids[lo:hi])
                    os.remove(prog)
                    print(f"RESUME {match_id(npz)} shard {si}: already complete", flush=True)
                    return 0
                if n_done:
                    start_slot = lo + int(np.argmin(dflags))
                    print(f"RESUME {match_id(npz)} shard {si} at slot {start_slot} "
                          f"({n_done}/{hi-lo} already done)", flush=True)
        except Exception as e:
            print(f"RESUME {match_id(npz)} shard {si}: unreadable checkpoint: {e}", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fids[start_slot]))
    pos = int(fids[start_slot])
    want = {int(fids[i]): i for i in range(start_slot, hi)}
    last = int(fids[hi - 1])
    t0 = time.time()
    done = 0
    while want and pos <= last:
        ok, img = cap.read()
        if not ok:
            break
        i = want.pop(pos, None)
        pos += 1
        if i is None:
            continue
        j = i - lo
        h, w = img.shape[:2]
        pbs, ps = [], []
        for p in range(2):
            if np.all(np.isfinite(boxes[i, p])):
                pbs.append(pad_box(boxes[i, p], w, h))
                ps.append(p)
        if pbs:
            try:
                k, s = run_batch(sess, inp, outs, img, np.stack(pbs))
            except Exception as e:
                print(f"ERROR {match_id(npz)} shard {si}: frame {fids[i]}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            for g, p in enumerate(ps):
                kp[j, p, :, :2] = k[g]
                kp[j, p, :, 2] = s[g]
        done += 1
        if done % CHECKPOINT_EVERY == 0:
            _checkpoint(prog, kp, fids[lo:hi], lo, start_slot + done)
            dt = time.time() - t0
            print(f"PROGRESS {match_id(npz)} shard {si} {(start_slot-lo)+done}/{hi-lo} "
                  f"{done/dt:.1f} frames/s eta {len(want)/max(done/dt,1e-9)/60:.1f} min", flush=True)
    cap.release()
    np.savez_compressed(out, kp=kp, box=used, frame=fids[lo:hi])
    if os.path.exists(prog):
        os.remove(prog)
    print(f"SHARD_DONE {match_id(npz)} {si}/{sn} {(start_slot-lo)+done} frames "
          f"{done/max(time.time()-t0,1e-9):.1f} frames/s", flush=True)
    return 0


# ------------------------------------------------------------------- merge --
def merge(mid, n, shards, threads):
    """Combine shard files into pose/NNN.npz, seeded where reference shards exist."""
    fids, boxes = read_meta(os.path.join(EXTRACT, f"{mid}.npz"))
    kp = np.full((n, 2, 17, 3), np.nan, dtype=np.float32)
    have = np.zeros(n, dtype=bool)
    for i in range(shards):
        p = os.path.join(POSE, ".shards", f"{mid}.sh{i}.npz")
        if not os.path.exists(p):
            return None, f"missing shard file {p}"
        d = np.load(p)
        lo, hi = slot_bounds(n, shards, i)
        if mid == "001":
            lo = max(lo, min(SEED_LIMIT, hi))
        sub = d["kp"]
        kp[lo:lo + len(sub)] = sub[:hi - lo]
        have[lo:hi] = np.isfinite(d["kp"][:hi - lo]).any(axis=(1, 2, 3))
        os.remove(p)
    # Seed match 1's first 12000 slots from the shipped reference shards.  Each
    # shard i holds real values only inside its own slot window [i*3000,(i+1)*3000)
    # and NaN elsewhere, so take a value only when it is finite.
    seed = 0
    if mid == "001":
        for s in range(4):
            p = REFSHARDS.format(1, s)
            if not os.path.exists(p):
                break
            d = np.load(p)
            rf, rkp = d["frame"], d["kp"]
            m = min(len(rf), SEED_LIMIT, n)
            if not np.array_equal(rf[:m], fids[:m]):
                print(f"SEED {mid}: shard {s} frame ids do not match, skipping", flush=True)
                continue
            for j in range(m):
                if rf[j] != fids[j]:
                    continue
                for p in range(2):
                    good = np.isfinite(rkp[j, p, :, 0])
                    if good.any():
                        kp[j, p, good] = rkp[j, p, good]
                        seed += 1
        print(f"SEED {mid}: {seed} player slots taken from the reference shards", flush=True)
    out = os.path.join(POSE, f"{mid}.npz")
    np.savez_compressed(out, kp=kp, box=np.array(boxes[:n], dtype=np.float32), frame=fids[:n])
    fin = np.isfinite(kp[:, :, 0, 0])
    anyf = fin.any(axis=1)
    return dict(match=mid, frames=int(n), slots_with_box=int(np.isfinite(boxes[:n, :, 0]).sum()),
                slots_with_pose=int(fin.sum()), frames_with_pose=int(anyf.sum()),
                seeded=seed), None


# ----------------------------------------------------------------- manager --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    os.makedirs(POSE, exist_ok=True)
    os.makedirs(os.path.join(POSE, ".shards"), exist_ok=True)
    logdir = os.path.join(POSE, ".logs")
    os.makedirs(logdir, exist_ok=True)
    # A finished shard file could have been left behind by an earlier attempt
    # that was killed before its merge, and shards/ was sized for that attempt;
    # drop those so a stale file is never merged as a fresh result.  Resume
    # checkpoints (*.prog.npz) are KEPT -- they carry real partial work and the
    # worker picks them up, so a SIGTERM costs at most CHECKPOINT_EVERY frames.
    shard_dir = os.path.join(POSE, ".shards")
    for sp in glob.glob(os.path.join(shard_dir, "*.npz")):
        if sp.endswith(".prog.npz"):
            print(f"resume checkpoint kept: {os.path.basename(sp)}", flush=True)
            continue
        try:
            os.remove(sp)
            print(f"cleaned stale shard {os.path.basename(sp)}", flush=True)
        except OSError as e:
            print(f"could not clean {sp}: {e}", flush=True)
    print(f"run_pose: workers={a.workers} shards/match={a.shards} threads={a.threads} "
          f"hours={a.hours}", flush=True)

    deadline = time.time() + a.hours * 3600
    attempts = {}         # mid -> attempts so far
    # a "task" is one (match, shard) worker process; the global cap is a.workers
    tasks = {}            # (mid, si) -> (proc, logpath, t0)
    done_shards = {}      # mid -> set of finished shard indices
    failed_shards = {}    # mid -> set of shard indices that exited non-zero
    merged = set()
    done_stats = []
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))

    while True:
        files = ordered(glob.glob(os.path.join(EXTRACT, "*.npz")))

        # ---- fill the pool with the highest-priority (match, shard) tasks ----
        for p in files:
            if stop["v"] or len(tasks) >= a.workers:
                break
            mid = match_id(p)
            if mid in merged or attempts.get(mid, 0) >= a.retries:
                continue
            video = os.path.join(VIDEOS, f"{mid}.mp4")
            if not os.path.exists(video):
                if attempts.get(mid, 0) == 0:
                    print(f"WAIT {mid}: video missing {video}", flush=True)
                attempts[mid] = attempts.get(mid, 0) + 1
                continue
            if mid not in done_shards:
                done_shards[mid] = set()
                attempts[mid] = attempts.get(mid, 0) + 1
                print(f"START {mid}: {a.shards} shards, attempt {attempts[mid]}", flush=True)
            for si in range(a.shards):
                if len(tasks) >= a.workers:
                    break
                if si in done_shards[mid] or (mid, si) in tasks:
                    continue
                lp = os.path.join(logdir, f"{mid}.sh{si}.log")
                cmd = [sys.executable, "-u",
                       os.path.join(HERE, "shard_worker.py"),
                       p, video, str(si), str(a.shards), str(a.threads)]
                with open(lp, "ab") as f:
                    f.write(f"\n=== attempt {attempts[mid]} {time.strftime('%H:%M:%S')} ===\n".encode())
                    pr = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE,
                                          start_new_session=True)
                tasks[(mid, si)] = (pr, lp, time.time())

        # ---- reap finished shard processes ----
        changed = False
        for key in list(tasks):
            pr, lp, t0 = tasks[key]
            code = pr.poll()
            if code is None:
                continue
            mid, si = key
            del tasks[key]
            changed = True
            if code != 0:
                failed_shards.setdefault(mid, set()).add(si)
                print(f"FAIL {mid}: shard {si} exited {code}; see {lp}", flush=True)
            else:
                done_shards[mid].add(si)

        # ---- merge any match whose shards are all present ----
        for p in files:
            mid = match_id(p)
            if mid in merged or mid in failed_shards:
                continue
            if mid not in done_shards or len(done_shards[mid]) < a.shards:
                continue
            n = len(read_meta(p)[0])
            stats, err = merge(mid, n, a.shards, a.threads)
            if err:
                print(f"FAIL {mid}: merge: {err}", flush=True)
                failed_shards[mid] = set(range(a.shards))
                continue
            merged.add(mid)
            done_stats.append(dict(stats, wall_s=0.0))
            print(f"DONE {mid} {json.dumps(stats)}", flush=True)

        # ---- retry matches whose shards failed, then give up on them ----
        for mid in list(failed_shards):
            if mid in merged:
                continue
            if attempts.get(mid, 0) >= a.retries:
                merged.add(mid)          # do not look at it again
                print(f"GIVEUP {mid}: {a.retries} attempts, shards failed "
                      f"{sorted(failed_shards[mid])}", flush=True)
                continue
            done_shards.pop(mid, None)
            failed_shards.pop(mid, None)
            attempts[mid] = attempts.get(mid, 0) + 1
            print(f"RETRY {mid}: attempt {attempts[mid]}", flush=True)
            changed = True

        if tasks:
            time.sleep(5)
            continue

        # ---- pool empty: done, or waiting for new matches ----
        remaining = [match_id(p) for p in files
                     if not os.path.exists(os.path.join(POSE, f"{match_id(p)}.npz"))
                     and match_id(p) not in merged]
        if a.once:
            print(f"ONCE exit: {len(remaining)} pending", flush=True)
            return
        if time.time() >= deadline:
            print(f"HOURS elapsed ({a.hours}h); pending={remaining}", flush=True)
            return
        print(f"idle: {len(merged)} finished, {len(remaining)} pending "
              f"({','.join(remaining[:12])}), polling in {a.poll}s", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once-worker":
        _, _, npz, video, si, sn, th = sys.argv
        out = os.path.join(POSE, ".shards", f"{match_id(npz)}.sh{si}.npz")
        sys.exit(worker(npz, video, out, (int(si), int(sn)), int(th)))
    main()
