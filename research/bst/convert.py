#!/usr/bin/env python3
"""Convert the Swift extract output (out/NNN.npz) into BST's raw per-clip layout.

    <out>/<split>/<Top|Bottom>_<stroke>/<match>_<set>_<rally>_<ball_round>_{joints,pos,shuttle}.npy

Shapes, exactly as the released weights expect (verified against the released data):
    joints   (T, 2, 17, 2) float32   joint x, y normalised by the person box centre and diagonal
    pos      (T, 2, 2)     float32   ankle midpoint on the doubles court, 0..1, y = 0 at the far baseline
    shuttle  (T, 2)        float32   x/W, y/H in the camera frame; (0, 0) when not visible

Player 0 is always the far/Top player, player 1 the near/Bottom one.  The clip window is
the one BST was trained with: from the opponent's previous hit (or hit − fps//2) to the
next hit + fps//8 (or hit + fps//2), clamped to ±3*fps/2 around the hit.

Split boundaries come from the paper: test = matches 39, 40, 42, 43, 44 and val = 41;
everything else that has labels is train.

usage:  python3 convert.py --extract ../extract/out --out bst_data [--pad-frames N] [--limit N]
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sides import top_letter_map, side_of

# our world frame: origin under the net, +x towards the far baseline, +y to the left
COURT_HALF_LEN, COURT_HALF_WID = 6.7, 3.05
TEST_IDS = {39, 40, 42, 43, 44}
VAL_IDS = {41}
JOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle"]

MERGED = ["未知球種", "放小球", "擋小球", "殺球", "挑球", "長球", "平球", "切球", "推球", "撲球", "勾球",
          "發短球", "發長球"]
CLASSES = ["未知球種"] + [f"{side}_{t}" for side in ("Top", "Bottom") for t in MERGED[1:]]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def load_npz(path):
    import zipfile
    z = zipfile.ZipFile(path)
    return {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}


def label_rows(label_dir):
    """All labelled hits of one match, grouped by (set, rally) and ordered."""
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


OFFICIAL_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ShuttleSet_data_merged",
                            "dataset_npy_between_2_hits_with_max_limits")


def official_classes(root=OFFICIAL_RAW):
    """clip name -> class folder of the released ShuttleSet data (empty if not present)."""
    import glob
    out = {}
    for sp in ("train", "val", "test"):
        for d in glob.glob(os.path.join(root, sp, "*")):
            cls = os.path.basename(d)
            for f in glob.glob(os.path.join(d, "*_joints.npy")):
                out[os.path.basename(f)[:-len("_joints.npy")]] = cls
    return out


def striker_side(r, top_map):
    """Top/Bottom of the striker.  The A/B column alone does NOT give it: players change
    ends every set and at 11 in set 3.  The hitter's pixel position is the direct
    answer (agrees with the released data on 99.2% of 32k strokes); BST's own
    set/score rule is the fallback when the location is missing."""
    try:
        py, oy = float(r["player_location_y"]), float(r["opponent_location_y"])
        if py != oy:
            return "Top" if py < oy else "Bottom"
    except (KeyError, TypeError, ValueError):
        pass
    return side_of(top_map, r["_set"], r["_rally"], r.get("player", ""))


def clip_window(hits, i, fps, seq_limit=None):
    fps = int(round(fps))
    t, limit, eps = fps // 2, fps * 3 // 2, (fps // 2) // 2
    start = hits[i - 1] if i > 0 else hits[i] - t
    end = hits[i + 1] + eps if i + 1 < len(hits) else hits[i] + t
    start = max(start, hits[i] - limit)
    end = min(end, hits[i] + limit + eps)
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default="../extract/out")
    ap.add_argument("--out", default="bst_data")
    ap.add_argument("--labels", default="../shuttleset/labels/ShuttleSet/set")
    ap.add_argument("--rtm", default="", help="glob of rtm_pose.py outputs; use RTMPose joints instead of Apple Vision")
    ap.add_argument("--limit", type=int, default=0, help="only this many matches")
    ap.add_argument("--stride-report", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.extract) if f.endswith(".npz"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no .npz under {a.extract}")

    # Ground truth: when a clip is in the released data, take its class from there, so
    # every arm of a clip-for-clip comparison is scored against the SAME label.
    official = official_classes()
    gt_source = {"official": 0, "derived": 0, "type differs from csv": 0}
    counts = {"train": {}, "val": {}, "test": {}}
    clips = {"train": 0, "val": 0, "test": 0}
    skipped = {}
    for fn in files:
        mid = int(fn[:3])
        split = "test" if mid in TEST_IDS else "val" if mid in VAL_IDS else "train"
        d = load_npz(os.path.join(a.extract, fn))
        info = json.loads(str(d["info_json"][0]))
        label_dir = os.path.join(os.path.dirname(os.path.abspath(a.labels)), "set", info["labels"].split("/")[-1]) \
            if not os.path.isabs(info["labels"]) else info["labels"]
        if not os.path.isdir(label_dir):
            skipped[fn] = "labels not found"
            continue
        rows = label_rows(label_dir)
        top_map = top_letter_map(label_dir)
        fps = float(d["fps"])
        W, H = float(d["width"]), float(d["height"])
        frame, kp = d["frame"], d["player_kp"]
        box, shuttle, vis = d["player_box"], d["shuttle_xy"], d["shuttle_visible"]
        rtm_cover = None
        if a.rtm:
            import glob as _glob
            # run_pose.py writes the merged pose as `NNN.npz`; older/sharded runs used
            # `NNN.shK.npz`.  Accept both, and let the merged file win when present.
            shards = sorted(_glob.glob(os.path.join(a.rtm, f"{mid:03d}_sh*.npz")))
            merged = os.path.join(a.rtm, f"{mid:03d}.npz")
            if not shards and os.path.exists(merged):
                shards = [merged]
            if not shards:
                skipped[fn] = "no RTMPose output"
                continue
            seen = {}
            for s in shards:
                z = np.load(s)
                # hoist the arrays: indexing an NpzFile inside a loop re-reads and
                # decompresses the whole member every time (~100x slower)
                zkp, zbox, zframe = z["kp"], z["box"], z["frame"]
                for k_i, f_i in enumerate(zframe):
                    if np.isfinite(zkp[k_i, :, 0, 0]).any():
                        seen[int(f_i)] = (zkp[k_i], zbox[k_i])
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
        n_hit = 0
        for k in range(len(hit_start) - 1):
            st, rl = int(sets[k]), int(ral_ids[k])
            rr = rows.get((st, rl))
            if not rr:
                skipped[f"{fn} s{st}r{rl}"] = "no labels"
                continue
            hits = [int(h) for h in hit_frames[hit_start[k]:hit_start[k + 1]]]
            if not hits:
                continue
            hit_rows = {}
            for r in rr:
                hit_rows.setdefault(r["_frame"], []).append(r)
            for i, hf in enumerate(hits):
                r = hit_rows.get(hf, [None])[0]
                if r is None:
                    continue
                typ = r.get("type", "")
                name = f"{mid}_{st}_{rl}_{r['_round']}"
                if name in official and "_" in official[name]:
                    side, otyp = official[name].split("_", 1)
                    if otyp != typ:
                        gt_source["type differs from csv"] += 1
                    typ = otyp
                    gt_source["official"] += 1
                else:
                    if not typ or typ not in MERGED[1:]:
                        skipped.setdefault("unknown stroke type", 0)
                        skipped["unknown stroke type"] += 1
                        continue
                    side = striker_side(r, top_map)
                    gt_source["derived"] += 1
                frame_lo, frame_hi = clip_window(hits, i, fps)
                idx = [fidx[f] for f in range(frame_lo, frame_hi) if f in fidx]
                if len(idx) < 2:
                    continue
                if rtm_cover is not None and any(int(frame[fi]) not in rtm_cover for fi in idx):
                    # a partly covered clip would mix two pose estimators; skip it, and the
                    # Vision run is restricted to the same clip names for a fair comparison
                    skipped["rtm partial clip"] = skipped.get("rtm partial clip", 0) + 1
                    continue
                # zero the whole frame when the two players are not both resolved
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
                cls = f"{side}_{typ}"
                outdir = os.path.join(a.out, split, cls)
                os.makedirs(outdir, exist_ok=True)
                base = os.path.join(outdir, name)
                np.save(base + "_joints.npy", j)
                np.save(base + "_pos.npy", pos)
                np.save(base + "_shuttle.npy", sh)
                counts[split][cls] = counts[split].get(cls, 0) + 1
                clips[split] += 1
                n_hit += 1
        print(f"{fn}: {n_hit} clips", flush=True)
    print("\nsummary:", clips)
    print("ground truth:", gt_source)
    for sp in counts:
        if counts[sp]:
            print(f"  {sp}: {len(counts[sp])} classes")
    if skipped:
        print("skipped:", dict(list(skipped.items())[:8]))


if __name__ == "__main__":
    main()
