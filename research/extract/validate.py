"""Validate an extract .npz against ShuttleSet labels.

Usage: python validate.py <out.npz> [--limit N]

At every labelled hit frame inside the extracted rallies, compares our shuttle
track and our two tracked players against the label's hit_x/hit_y and
player/opponent locations (all in 1280x720 px).  Because the label's A/B player
column is not known to map onto near/far, the two players are matched to the
label's two positions by best assignment; the striker error uses whichever of
our players sits closest to the label's `player` position.
"""
import argparse, csv, json, os, zipfile
import numpy as np


def load_npz(path):
    z = zipfile.ZipFile(path)
    return {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}


def label_rows(dirpath):
    rows = []
    for f in sorted(os.listdir(dirpath)):
        if f.startswith("set") and f.endswith(".csv"):
            s = int(f[3:-4])
            for r in csv.DictReader(open(os.path.join(dirpath, f), encoding="utf-8")):
                if (r.get("frame_num") or "").strip():
                    r["_set"], r["_frame"] = s, int(float(r["frame_num"]))
                    rows.append(r)
    return rows


def num(r, k):
    try:
        return float((r.get(k) or "").strip())
    except ValueError:
        return None


def rep(name, vals, extra=""):
    if not vals:
        print(f"{name}: no samples{extra}")
        return
    v = np.array(vals)
    print(f"{name}: n={len(v)} median={np.median(v):.1f} p10={np.percentile(v,10):.1f} "
          f"p90={np.percentile(v,90):.1f} <=10px {100*(v<=10).mean():3.0f}% "
          f"<=20px {100*(v<=20).mean():3.0f}% <=40px {100*(v<=40).mean():3.0f}%{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = load_npz(a.npz)
    info = json.loads(str(d["info_json"][0]))
    laps = info["labels"]
    if not os.path.isabs(laps):
        laps = os.path.join("/path/to/shuttl/research", laps)
    rows = label_rows(laps)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["_set"], r["_frame"]), []).append(r)

    frame = d["frame"]
    fidx = {int(f): i for i, f in enumerate(frame)}
    xy, vis = d["shuttle_xy"], d["shuttle_visible"]
    kp = d["player_kp"]                     # (N,2,17,3)
    hits, sets = d["rally_hit_frames"], d["rally_set"]
    hit_start = d["rally_hit_start"]        # (R+1,) index of each rally's first hit

    # expand each hit frame to its rally's set number
    n_rallies = len(hit_start) - 1
    hit_rally = np.zeros(len(hits), dtype=int)
    for r in range(n_rallies):
        hit_rally[hit_start[r]:hit_start[r + 1]] = r
    if a.limit:
        hits, hit_rally = hits[:a.limit], hit_rally[:a.limit]

    sh_err, sh_missing, sh_nolabel = [], 0, 0
    striker_err, bi_err, side_ok, side_tot, no_pose = [], [], 0, 0, 0
    for hf, r in zip(hits, hit_rally):
        st = int(sets[r])
        i = fidx.get(int(hf))
        if i is None:
            continue
        L = by_key.get((st, int(hf)))
        if not L:
            continue
        lab = L[0]
        hx, hy = num(lab, "hit_x"), num(lab, "hit_y")
        if hx is None or hy is None:
            sh_nolabel += 1
        elif not vis[i]:
            sh_missing += 1
        else:
            sh_err.append(float(np.hypot(xy[i, 0] - hx, xy[i, 1] - hy)))

        px, py = num(lab, "player_location_x"), num(lab, "player_location_y")
        ox, oy = num(lab, "opponent_location_x"), num(lab, "opponent_location_y")
        if None in (px, py, ox, oy):
            continue
        ank = []
        for p in range(2):
            k = kp[i, p]
            if k[15, 2] > 0 and k[16, 2] > 0:
                ank.append(np.array([(k[15, 0] + k[16, 0]) / 2, (k[15, 1] + k[16, 1]) / 2]))
            else:
                ank.append(None)
        if ank[0] is None or ank[1] is None:
            no_pose += 1
            continue
        P = np.array([px, py]); O = np.array([ox, oy])
        d_striker = [np.linalg.norm(a - P) for a in ank]
        j = int(np.argmin(d_striker))
        striker_err.append(float(d_striker[j]))
        # best assignment of our two players to the label's two positions
        straight = np.linalg.norm(ank[0] - P) + np.linalg.norm(ank[1] - O)
        swap = np.linalg.norm(ank[0] - O) + np.linalg.norm(ank[1] - P)
        bi_err.append(float(min(straight, swap)) / 2)
        # label's `player` is the striker; check our two players are on the same
        # court half-image side as the label's two positions do not require it.
        side_tot += 1
        same = (np.linalg.norm(ank[0] - P) < np.linalg.norm(ank[1] - P)) == (ank[0][1] < ank[1][1])
        side_ok += int(same or abs(ank[0][1] - ank[1][1]) < 20)

    print(f"video={info['video']} fps={float(d['fps'])} cam_ok={bool(d['cam_ok'])} "
          f"label_rows={len(rows)} frames={len(frame)} hits={len(hits)}")
    rep("shuttle err px", sh_err, f" [no shuttle {sh_missing}, no label {sh_nolabel}]")
    rep("striker err px", striker_err, f" [no pose {no_pose}]")
    rep("player pair err px", bi_err)
    if side_tot:
        print(f"near/far ordering consistent: {100*side_ok/side_tot:.0f}% of {side_tot}")


if __name__ == "__main__":
    main()
