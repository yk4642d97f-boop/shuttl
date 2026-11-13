#!/usr/bin/env python3
"""Build BST raw clips whose windows are driven by OUR DETECTED hit frames,
for a true end-to-end measurement against the labelled-hit control.

Two arms, identical in every respect except which hit list drives the clip window:

  control  : hits = extract npz `rally_hit_frames` (human labels); class from the
             labelled hit at the window centre.  This reproduces bst/convert.py.
  detected : hits = ours, from hits/out/NNN.hits.json (per set/rally).  Window is
             centred on a DETECTED hit, and its neighbours in the window rule are
             the DETECTED neighbours (what would really happen).  Class = stroke type
             of the nearest LABELLED hit within +-4 frames; if none, the detection is
             a false positive and is dropped (counted separately).

A third arm, `detwin`, isolates the mis-centring cost:
  detwin   : the SAME matched detections as `detected` (same clip set, same classes),
             but the window is centred on the matched LABELLED hit instead of the
             detected one.  `detected` minus `detwin` is therefore exactly the cost of
             mis-centred windows: only the window centre moves between the two.

Everything else (normalisation, window rule, class naming, split naming, the
"both players resolved" gate, the RTMPose full-clip-coverage gate) is copied
verbatim from bst/convert.py so the arms differ only in the hit frames.

usage:
  python3 e2e_convert.py --extract ../extract/out --labels ../shuttleset/labels/ShuttleSet/set \
      --hits ../hits/out --out e2e_data_detected --arm detected \
      --rtm '/tmp/rtm/ab2' [--limit 1] [--shift 2] [--no-rtm]
"""
import argparse, csv, json, os, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bst"))
from convert import official_classes, striker_side   # ONE ground-truth rule for every harness
from sides import top_letter_map

COURT_HALF_LEN, COURT_HALF_WID = 6.7, 3.05
TEST_IDS = {39, 40, 42, 43, 44}
VAL_IDS = {41}

MERGED = ["未知球種", "放小球", "擋小球", "殺球", "挑球", "長球", "平球", "切球", "推球", "撲球", "勾球",
          "發短球", "發長球"]
CLASSES = ["未知球種"] + [f"{side}_{t}" for side in ("Top", "Bottom") for t in MERGED[1:]]

MATCH_TOL = 4  # +-4 frames: a detection matches a labelled hit within this


def load_npz(path):
    import zipfile
    z = zipfile.ZipFile(path)
    return {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}


def label_rows(label_dir):
    out = {}
    for f in sorted(os.listdir(label_dir)):
        if not (f.startswith("set") and f.endswith(".csv")):
            continue
        s = int(f[3:-4])
        for r in csv.DictReader(open(os.path.join(label_dir, f), encoding="utf-8")):
            fn = (r.get("frame_num") or "").strip()
            if not fn:
                continue
            r["_set"], r["_frame"] = s, int(float(fn))
            r["_rally"] = int(float(r["rally"]))
            r["_round"] = int(float(r["ball_round"]))
            out.setdefault((s, r["_rally"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["_round"])
    return out


def clip_window(hits, i, fps):
    """bst/convert.py:clip_window, verbatim."""
    fps = int(round(fps))
    t, limit, eps = fps // 2, fps * 3 // 2, (fps // 2) // 2
    start = hits[i - 1] if i > 0 else hits[i] - t
    end = hits[i + 1] + eps if i + 1 < len(hits) else hits[i] + t
    start = max(start, hits[i] - limit)
    end = min(end, hits[i] + limit + eps)
    return start, end


def fixed_window(centre, fps):
    """The seq30 '1-second fixed clips' protocol: [hit - fps//2, hit + fps//2].

    Confirmed against the released seq30 data: raw clip lengths are 25 frames at fps 25,
    i.e. the inclusive range hit-12 .. hit+12.  No neighbour hits are consulted, so a
    detected neighbour's error cannot move the window.
    """
    t = int(round(fps)) // 2
    return centre - t, centre + t + 1


def detected_hits(hits_path, mid):
    """{ (set, rally): [frame, ...] } from hits/out/NNN.hits.json, ordered."""
    if not os.path.exists(hits_path):
        return None
    h = json.load(open(hits_path))
    out = {}
    for rr in h["rallies"]:
        key = (int(rr["set"]), int(rr["rally"]))
        out[key] = sorted(int(x["frame"]) for x in rr["hits"])
    return out


def nearest_label(frame, frames_sorted, rows_by_frame, tol=MATCH_TOL):
    """Labelled row nearest to `frame` within +-tol, else None. O(log n)."""
    import bisect
    j = bisect.bisect_left(frames_sorted, frame)
    best, bestd = None, tol + 1
    for k in (j - 1, j):
        if 0 <= k < len(frames_sorted):
            f = frames_sorted[k]
            d = abs(f - frame)
            if d < bestd:
                best, bestd = rows_by_frame[f][0], d
    return best if bestd <= tol else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default="../extract/out")
    ap.add_argument("--out", default="e2e_data")
    ap.add_argument("--labels", default="../shuttleset/labels/ShuttleSet/set")
    ap.add_argument("--hits", default="../hits/out")
    ap.add_argument("--rtm", default="")
    ap.add_argument("--arm", default="control",
                    choices=["control", "detected", "detwin", "fixed", "fixed_ctrl"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match", type=int, nargs="*", default=[], help="only these match ids")
    ap.add_argument("--shift", type=int, default=0, help="add N to every detected hit frame")
    ap.add_argument("--no-rtm", action="store_true", help="ignore --rtm (Vision pose)")
    ap.add_argument("--stats", default="", help="write per-clip match stats to this json")
    a = ap.parse_args()

    rtm_dir = "" if a.no_rtm else a.rtm
    files = sorted(f for f in os.listdir(a.extract) if f.endswith(".npz"))
    if a.match:
        files = [f for f in files if f[:3].isdigit() and int(f[:3]) in a.match]
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no .npz under {a.extract}")

    # a clip is only kept when EVERY frame of its window has pose.  Without a pose
    # source that gate is simply "the frame exists in the extract".  With RTMPose we
    # require RTMPose on every frame, which is exactly bst/convert.py's rule and keeps
    # the two arms comparable (both see the same estimator on the same frames).
    stats = {"arm": a.arm, "shift": a.shift,
             "coverage_gate": "rtm_full" if rtm_dir else "none (vision)",
             "clips": [], "totals": {}}
    tot = dict(detected=0, matched=0, false_positive=0, matched_offset_gt2=0,
               clips_written=0, clips_window_ok=0, clips_pose_ok=0, labelled_hits=0)
    counts = {}
    official = official_classes()

    for fn in files:
        mid = int(fn[:3])
        split = "test" if mid in TEST_IDS else "val" if mid in VAL_IDS else "train"
        d = load_npz(os.path.join(a.extract, fn))
        info = json.loads(str(d["info_json"][0]))
        label_dir = os.path.join(os.path.dirname(os.path.abspath(a.labels)), "set", info["labels"].split("/")[-1]) \
            if not os.path.isabs(info["labels"]) else info["labels"]
        if not os.path.isdir(label_dir):
            print(f"{fn}: labels not found ({label_dir})")
            continue
        rows = label_rows(label_dir)
        top_map = top_letter_map(label_dir)
        fps = float(d["fps"])
        W, H = float(d["width"]), float(d["height"])
        frame, kp = d["frame"], d["player_kp"]
        box, shuttle, vis = d["player_box"], d["shuttle_xy"], d["shuttle_visible"]

        rtm_cover = None
        if rtm_dir:
            shards = sorted(glob.glob(os.path.join(rtm_dir, f"{mid:03d}_sh*.npz")))
            if not shards and os.path.exists(os.path.join(rtm_dir, f"{mid:03d}.npz")):
                shards = [os.path.join(rtm_dir, f"{mid:03d}.npz")]   # run_pose.py's merged file
            if not shards:
                print(f"{fn}: no RTMPose output under {rtm_dir}")
                continue
            seen = {}
            for s in shards:
                z = np.load(s)
                z_kp, z_box, z_frame = z["kp"], z["box"], z["frame"]
                fin = np.isfinite(z_kp[:, :, 0, 0]).any(axis=1)
                for k_i in np.nonzero(fin)[0]:
                    seen[int(z_frame[k_i])] = (z_kp[k_i], z_box[k_i])
            rtm_cover = set(seen)
            kp = np.full_like(kp, np.nan)
            box = np.full_like(box, np.nan)
            for k_i, f_i in enumerate(frame):
                if int(f_i) in seen:
                    kp[k_i], box[k_i] = seen[int(f_i)]
            print(f"{fn}: RTMPose covers {len(seen)}/{len(frame)} frames")

        fidx = {int(f): i for i, f in enumerate(frame)}
        hit_start, hit_frames, sets = d["rally_hit_start"], d["rally_hit_frames"], d["rally_set"]
        ral_ids = d["rally_id"]

        det = detected_hits(os.path.join(a.hits, f"{mid:03d}.hits.json"), mid) \
            if a.arm in ("detected", "detwin", "fixed") else None
        if a.arm in ("detected", "detwin", "fixed") and det is None:
            print(f"{fn}: no hits json at {a.hits}/{mid:03d}.hits.json")
            continue

        n_hit = 0
        for k in range(len(hit_start) - 1):
            st, rl = int(sets[k]), int(ral_ids[k])
            rr = rows.get((st, rl))
            if not rr:
                continue
            by_frame = {}
            for r in rr:
                by_frame.setdefault(r["_frame"], []).append(r)
            label_frames = sorted(by_frame)
            tot["labelled_hits"] += len(label_frames)

            if a.arm in ("control", "fixed_ctrl"):
                hits = [int(h) for h in hit_frames[hit_start[k]:hit_start[k + 1]]]
            else:
                hits = [f + a.shift for f in det.get((st, rl), [])]
            if not hits:
                continue

            for i, hf in enumerate(hits):
                if a.arm in ("control", "fixed_ctrl"):
                    r = by_frame.get(hf, [None])[0]
                    if r is None:
                        continue
                    centre_row = r
                    win_hits, win_i = hits, i
                    tot["detected"] += 1
                    tot["matched"] += 1
                else:
                    tot["detected"] += 1
                    lab = nearest_label(hf, label_frames, by_frame)
                    if lab is None:
                        tot["false_positive"] += 1
                        continue
                    tot["matched"] += 1
                    if abs(hf - lab["_frame"]) > 2:
                        tot["matched_offset_gt2"] += 1
                    centre_row = lab
                    if a.arm == "detwin":
                        # same clip set and classes as `detected`, but the whole window is
                        # shifted so the MATCHED LABELLED hit sits at the centre: the only
                        # thing that differs from `detected` is the window centre.
                        shiftby = lab["_frame"] - hf
                        win_hits = [f + shiftby for f in hits]
                        win_i = i
                    else:
                        win_hits, win_i = hits, i

                # ground truth exactly as bst/convert.py: the released class when the clip is
                # in the released data, else the striker's court side from the labels
                gt = official.get(f"{mid}_{st}_{rl}_{centre_row['_round']}", "")
                if "_" in gt:
                    side, typ = gt.split("_", 1)
                else:
                    typ = centre_row.get("type", "")
                    if not typ or typ not in MERGED[1:]:
                        continue
                    side = striker_side(centre_row, top_map)

                if a.arm in ("fixed", "fixed_ctrl"):
                    # seq30 protocol: fixed +-fps//2 window centred on the hit.  `fixed` is
                    # centred on the DETECTED hit, `fixed_ctrl` on the labelled one.
                    frame_lo, frame_hi = fixed_window(hf, fps)
                else:
                    frame_lo, frame_hi = clip_window(win_hits, win_i, fps)
                idx = [fidx[f] for f in range(frame_lo, frame_hi) if f in fidx]
                if len(idx) < 2:
                    continue
                if rtm_cover is not None and any(int(frame[fi]) not in rtm_cover for fi in idx):
                    # bst/convert.py: a partly covered clip would mix two pose estimators
                    continue
                tot["clips_window_ok"] += 1

                j = np.zeros((len(idx), 2, 17, 2), dtype=np.float32)
                pos = np.zeros((len(idx), 2, 2), dtype=np.float32)
                sh = np.zeros((len(idx), 2), dtype=np.float32)
                for t_i, fi in enumerate(idx):
                    both = True
                    for p in range(2):
                        kx = kp[fi, p]
                        b = box[fi, p]
                        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(kx[:17, :2])) \
                                or kx[15, 2] <= 0 or kx[16, 2] <= 0:
                            both = False
                            break
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        diag = max(np.hypot(b[2] - b[0], b[3] - b[1]), 1e-6)
                        j[t_i, p] = (kx[:17, :2] - [cx, cy]) / diag
                    if not both:
                        continue
                    if vis[fi]:
                        sh[t_i] = [shuttle[fi, 0] / W, shuttle[fi, 1] / H]
                foot = d["player_foot_world"]
                for t_i, fi in enumerate(idx):
                    for p in range(2):
                        fx, fy = foot[fi, p]
                        if np.isfinite(fx):
                            pos[t_i, p] = [COURT_HALF_WID / (2 * COURT_HALF_WID) - fy / (2 * COURT_HALF_WID),
                                           (COURT_HALF_LEN - fx) / (2 * COURT_HALF_LEN)]
                        else:
                            pos[t_i, p] = 0.0

                rnd = centre_row["_round"]
                suffix = ""
                if a.arm in ("detected", "detwin", "fixed"):
                    # detected hits may collide on round number; disambiguate by frame.
                    # NOTE: the suffix uses the DETECTED frame in both arms so the
                    # `detected` and `detwin` clip sets line up name-for-name.
                    suffix = f"_{hf}"
                name = f"{mid}_{st}_{rl}_{rnd}{suffix}"
                cls = f"{side}_{typ}"
                outdir = os.path.join(a.out, split, cls)
                os.makedirs(outdir, exist_ok=True)
                base = os.path.join(outdir, name)
                np.save(base + "_joints.npy", j)
                np.save(base + "_pos.npy", pos)
                np.save(base + "_shuttle.npy", sh)
                counts[cls] = counts.get(cls, 0) + 1
                tot["clips_written"] += 1
                n_hit += 1
                if a.stats:
                    stats["clips"].append({
                        "name": name, "cls": cls, "match": mid, "set": st, "rally": rl,
                        "det_frame": hf, "lab_frame": centre_row["_frame"],
                        "offset": hf - centre_row["_frame"],
                        "n_frames": len(idx),
                        "start": frame_lo, "end": frame_hi,
                    })
        print(f"{fn}: {n_hit} clips ({a.arm})", flush=True)

    tot["clips_pose_ok"] = tot["clips_written"]
    stats["totals"] = tot
    print(f"\n[{a.arm} shift={a.shift}] {json.dumps(tot)}")
    print(f"clips by class: {len(counts)} classes, {sum(counts.values())} clips")
    if a.stats:
        json.dump(stats, open(a.stats, "w"))
        print("wrote stats", a.stats)


if __name__ == "__main__":
    main()
