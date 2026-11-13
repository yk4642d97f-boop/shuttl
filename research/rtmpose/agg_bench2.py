"""Control the confound: run every config on the SAME matches so box density is
held constant, and report frames/s and boxes/s separately.  Uses a long enough
window that the 1-second ticks settle, and warms up before timing.
"""
import os, sys, time, subprocess, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "..", "extract", "out")
VIDEOS = os.path.join(HERE, "..", "shuttleset", "videos")
BDIR = os.path.join(HERE, "pose", ".bench")


def density(mid):
    import numpy as np, zipfile
    z = zipfile.ZipFile(os.path.join(EXTRACT, f"{mid}.npz"))
    b = np.load(z.open("player_box.npy"))
    z.close()
    return 2 * float(np.isfinite(b[:, :, 0]).mean())


def run_config(workers, threads, secs, matches, warm=12):
    os.makedirs(BDIR, exist_ok=True)
    procs, logs, mids = [], [], []
    for i in range(workers):
        mid = matches[i % len(matches)]
        npz = os.path.join(EXTRACT, f"{mid}.npz")
        video = os.path.join(VIDEOS, f"{mid}.mp4")
        if not (os.path.exists(npz) and os.path.exists(video)):
            continue
        si, sn = i, max(workers, 30)
        lp = os.path.join(BDIR, f"w{workers}t{threads}_{i}.log")
        f = open(lp, "wb")
        cmd = [sys.executable, "-u", os.path.join(HERE, "bench_worker.py"),
               npz, video, str(si), str(sn), str(threads)]
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE))
        logs.append(lp)
        mids.append(mid)
    time.sleep(max(0, warm))
    # measure over a clean window AFTER warmup
    base = {}
    for lp in logs:
        base[lp] = _last(lp)
    t0 = time.time()
    time.sleep(secs)
    dt = time.time() - t0
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
    total, per = 0, []
    for lp in logs:
        n = _last(lp) - base[lp]
        per.append(n)
        total += n
    fps = total / dt if dt else 0
    dens = sum(density(m) for m in mids) / max(len(mids), 1)
    print(f"AGG workers={workers} threads={threads}: per-worker={per} "
          f"total={total} in {dt:.0f}s = {fps:.1f} frames/s ({fps*dens:.1f} boxes/s, "
          f"density {dens:.2f})", flush=True)
    return fps


def _last(lp):
    try:
        txt = open(lp, "r", errors="replace").read()
    except Exception:
        return 0
    last = 0
    for line in txt.splitlines():
        for tag in ("TICK", "SHARD_DONE"):
            if line.startswith(tag):
                try:
                    v = int(line.split()[2] if tag == "TICK" else line.split()[1])
                    # TICK prints "TICK <si> <done>"
                    if tag == "TICK":
                        v = int(line.split()[2])
                    last = max(last, v)
                except Exception:
                    pass
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=50)
    ap.add_argument("--warm", type=int, default=12)
    ap.add_argument("--configs", default="4x1,6x1,8x1,4x2")
    a = ap.parse_args()
    # fixed match set, held constant across configs
    matches = ["001", "002", "003", "005", "006", "007", "008", "010"]
    res = {}
    for cfg in a.configs.split(","):
        w, t = cfg.split("x")
        res[cfg] = round(run_config(int(w), int(t), a.secs, matches, a.warm), 2)
    print("SUMMARY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
