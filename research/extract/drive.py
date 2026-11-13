#!/usr/bin/env python3
"""Drive `extract` over every ShuttleSet match whose video is complete, N matches at a time.

Each worker gets its own CoreML cache directory (the compiled TrackNetV3 model is
re-compiled per worker, not per match).  Per-rally results are cached under
<out>.parts/, so stopping and restarting never repeats finished rallies.

  python3 drive.py --workers 3            # run until every downloaded match is extracted
  python3 drive.py --workers 3 --only 7   # one match
"""
import argparse, csv, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, ".build/release/extract")
MANIFEST = os.path.join(ROOT, "../shuttleset/manifest.csv")
VIDEOS = os.path.join(ROOT, "../shuttleset/videos")
LABELS = os.path.join(ROOT, "../shuttleset/labels/ShuttleSet/set")
OUT = os.path.join(ROOT, "out")

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# The official held-out matches whose accuracy the headline number rests on.  They are
# extracted first once their videos land, because everything downstream waits on them.
PRIORITY = (39, 40, 41, 42, 43, 44)


def ready_matches():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    out = []
    for r in rows:
        if r["status"] != "done":
            continue
        vid = os.path.join(VIDEOS, f"{int(r['id']):03d}.mp4")
        lab = os.path.join(LABELS, r["video"])
        out.append((int(r["id"]), r["video"], vid, lab, os.path.exists(vid) and os.path.isdir(lab)))
    return sorted(out, key=lambda m: (m[0] not in PRIORITY, m[0]))


def extract_one(mid, name, vid, lab, worker):
    out = os.path.join(OUT, f"{mid:03d}.npz")
    if os.path.exists(out):
        return True
    cache = os.path.join(ROOT, f".mlcache-w{worker}")
    os.makedirs(cache, exist_ok=True)
    t = time.time()
    r = subprocess.run([BIN, "--video", vid, "--labels", lab, "--out", out, "--cache", cache],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    tail = lines[-1] if lines else ""
    ok = r.returncode == 0 and os.path.exists(out)
    log(f"w{worker} {mid:03d} {'done' if ok else 'FAILED rc=%d' % r.returncode} "
        f"in {(time.time()-t)/60:.1f} min  {name[:38]}  {tail[:70]}")
    if not ok:
        for l in lines[-6:]:
            log(f"w{worker} {mid:03d} | {l[:150]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--idle-wait", type=int, default=600)
    ap.add_argument("--only", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    while True:
        ready = [m for m in ready_matches() if m[4] and (not a.only or m[0] == a.only)]
        pending = [m for m in ready if not os.path.exists(os.path.join(OUT, f"{m[0]:03d}.npz"))]
        if a.only:
            for m in pending:
                extract_one(m[0], m[1], m[2], m[3], 0)
            return
        if pending:
            log(f"{len(pending)} matches to do with {a.workers} workers: "
                + ", ".join(f"{m[0]:03d}" for m in pending))
            lock = threading.Lock()
            queue = list(pending)

            def run(worker):
                while True:
                    with lock:
                        if not queue:
                            return
                        m = queue.pop(0)
                    extract_one(m[0], m[1], m[2], m[3], worker)

            threads = [threading.Thread(target=run, args=(i + 1,)) for i in range(a.workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            continue
        still = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["status"] != "done"]
        if not still:
            done = sum(1 for m in ready if os.path.exists(os.path.join(OUT, f"{m[0]:03d}.npz")))
            log(f"nothing left to download; {done} matches extracted")
            return
        log(f"up to date; {len(still)} matches still downloading; sleeping {a.idle_wait}s")
        time.sleep(a.idle_wait)


if __name__ == "__main__":
    main()
