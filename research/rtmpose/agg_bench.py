"""Aggregate throughput sweep: launch N copies of bench_worker.py on real
matches, let them run a fixed number of seconds, and sum the frames they
completed.  This measures what the MACHINE delivers for a given
(workers, threads) config, so we keep the config that maximises output.
"""
import os, sys, time, subprocess, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "..", "extract", "out")
VIDEOS = os.path.join(HERE, "..", "shuttleset", "videos")
BDIR = os.path.join(HERE, "pose", ".bench")


def run_config(workers, threads, secs, matches):
    os.makedirs(BDIR, exist_ok=True)
    procs, logs = [], []
    for i in range(workers):
        mid = matches[i % len(matches)]
        npz = os.path.join(EXTRACT, f"{mid}.npz")
        video = os.path.join(VIDEOS, f"{mid}.mp4")
        if not (os.path.exists(npz) and os.path.exists(video)):
            continue
        # each proc takes a distinct block so no two decode the same span
        si, sn = i, max(workers, 30)
        lp = os.path.join(BDIR, f"w{workers}t{threads}_{i}.log")
        f = open(lp, "wb")
        cmd = [sys.executable, "-u", os.path.join(HERE, "bench_worker.py"),
               npz, video, str(si), str(sn), str(threads)]
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=HERE))
        logs.append(lp)
    t0 = time.time()
    time.sleep(secs)
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
    dt = time.time() - t0
    # take each worker's LAST tick; they are ~1s apart so the sum of last ticks
    # divided by the window is the aggregate rate
    total = 0
    per = []
    for lp in logs:
        try:
            txt = open(lp, "r", errors="replace").read()
        except Exception:
            continue
        last = 0
        for line in txt.splitlines():
            if line.startswith("TICK"):
                try:
                    last = max(last, int(line.split()[2]))
                except Exception:
                    pass
            if line.startswith("SHARD_DONE"):
                try:
                    last = max(last, int(line.split()[2]))
                except Exception:
                    pass
        per.append(last)
        total += last
    agg = total / dt if dt else 0
    print(f"AGG workers={workers} threads={threads}: procs={len(procs)} "
          f"per-worker={per} total={total} in {dt:.0f}s = {agg:.1f} frames/s",
          flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=75)
    ap.add_argument("--configs", default="4x1,4x2,6x1,2x2")
    a = ap.parse_args()
    matches = ["001", "002", "003", "004", "005", "006", "007", "008"]
    res = {}
    for cfg in a.configs.split(","):
        w, t = cfg.split("x")
        res[cfg] = round(run_config(int(w), int(t), a.secs, matches), 2)
    print("SUMMARY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
