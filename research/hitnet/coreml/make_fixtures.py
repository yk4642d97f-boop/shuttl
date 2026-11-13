"""Write golden fixtures for the Swift port of the hitnet detector.

  /path/to/shuttl/research/coreml/venv/bin/python make_fixtures.py

Picks 8 real rallies spanning the 5 held-out matches (039 040 042 043 044), including the longest
and the shortest, and dumps per rally under fixtures/:
  raw      (T,16) float32   the normalised per-frame inputs derive() reads (NaN = missing)
  features (T,51) float32   common.derive(raw, fps) — the CoreML model input
  logits   (T,2)  float32   PyTorch reference, 3-checkpoint ensemble, whole rally
  logits_windowed (T,2)     PyTorch reference through infer.predict_windowed (384/192, centre-only)
plus fps, frame indices, hits, side, set, rally.  A Swift implementation must be able to rebuild
(51) from (16)+fps alone — every array derive() touches is here, no keypoints, no video.

The same numbers are re-checked against the CoreML model right after writing, so the fixtures cannot
silently drift from the shipped .mlpackage.
"""
import json
import warnings

import numpy as np
import torch
import coremltools as ct

import hitnet_core as H

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
FIX = H.HERE / "fixtures"
SEED = 20261006
# one pool per match, then a fixed spread; longest and shortest rally always included
PER_MATCH = {"039": 2, "040": 2, "042": 2, "043": 1, "044": 1}

# provenance of the 16 raw columns, as build_cache() fills them (common.py:88-105)
RAW_LAYOUT = [
    dict(i=0, name="shuttle_x", desc="shuttle x / W (W = 1280, already divided); W in .npz 'width'",
         nan_means="shuttle not visible this frame (shuttle_visible False or non-finite xy)"),
    dict(i=1, name="shuttle_y", desc="shuttle y / H (H = 720, already divided)",
         nan_means="shuttle not visible this frame"),
    dict(i=2, name="shuttle_conf", desc="tracker confidence 0..1 (raw shuttle_conf)",
         nan_means="shuttle not visible this frame"),
    dict(i=3, name="shuttle_interp", desc="1 where the extractor interpolated the shuttle position, else 0",
         nan_means="never NaN (array, not masked)"),
    dict(i=4, name="box0_cx", desc="player 0 box centre x / W", nan_means="player 0 box not finite"),
    dict(i=5, name="box0_cy", desc="player 0 box centre y / H", nan_means="player 0 box not finite"),
    dict(i=6, name="box0_w", desc="player 0 box width / W", nan_means="player 0 box not finite"),
    dict(i=7, name="box0_h", desc="player 0 box height / H", nan_means="player 0 box not finite"),
    dict(i=8, name="box1_cx", desc="player 1 box centre x / W", nan_means="player 1 box not finite"),
    dict(i=9, name="box1_cy", desc="player 1 box centre y / H", nan_means="player 1 box not finite"),
    dict(i=10, name="box1_w", desc="player 1 box width / W", nan_means="player 1 box not finite"),
    dict(i=11, name="box1_h", desc="player 1 box height / H", nan_means="player 1 box not finite"),
    dict(i=12, name="foot0_x", desc="player 0 ankle-midpoint world x, metres (court-centred frame)",
         nan_means="player 0 foot not resolved"),
    dict(i=13, name="foot0_y", desc="player 0 ankle-midpoint world y, metres (sign carries near/far)",
         nan_means="player 0 foot not resolved"),
    dict(i=14, name="foot1_x", desc="player 1 ankle-midpoint world x, metres", nan_means="player 1 foot not resolved"),
    dict(i=15, name="foot1_y", desc="player 1 ankle-midpoint world y, metres", nan_means="player 1 foot not resolved"),
]



def choose():
    picks = []
    for mid, k in PER_MATCH.items():
        m = H.C.load_match(mid)
        lens = np.array([len(r["raw"]) for r in m["rallies"]])
        rng = np.random.default_rng(SEED + int(mid))
        cand = {int(np.argmax(lens)), int(np.argmin(lens))}
        order = np.argsort([-lens[i] for i in range(len(lens))])   # longest first, then fill spread
        for i in order:
            if len(cand) >= k:
                break
            if int(i) not in cand:
                cand.add(int(i))
        while len(cand) < k:                                       # k > 2: take random extras
            i = int(rng.integers(0, len(lens)))
            if i not in cand:
                cand.add(i)
        for i in sorted(cand):
            picks.append((mid, i))
    return picks


def main():
    FIX.mkdir(exist_ok=True)
    ens, members = H.load_ensemble()
    picks = choose()
    model = ct.models.MLModel(str(H.HERE / "HitNetDetector.mlpackage"),
                              compute_units=ct.ComputeUnit.CPU_ONLY)
    index = []
    for k, (mid, i) in enumerate(picks, 1):
        m = H.C.load_match(mid)
        r = m["rallies"][i]
        raw = r["raw"].astype(np.float32)
        X = H.C.derive(r["raw"], r["fps"]).astype(np.float32)
        lg_full = H.logits_full(ens, r["raw"], r["fps"]).astype(np.float32)
        lg_win = H.logits_windowed(ens, r["raw"], r["fps"]).astype(np.float32)
        got = np.asarray(model.predict({"features": X[None]})["logits"])[0].astype(np.float32)
        d_full = float(np.abs(got - lg_full).max())
        d_win = float(np.abs(got - lg_win).max())
        name = "rally_%02d_%s_set%d_r%d" % (k, mid, r["set"], r["rally"])
        path = FIX / (name + ".npz")
        np.savez_compressed(
            path, raw=raw, features=X, logits=lg_full, logits_windowed=lg_win,
            coreml_logits=got, fps=np.float64(m["fps"]), frames=r["frames"].astype(np.int64),
            hits=r["hits"].astype(np.float64), side=r["side"].astype(np.int64),
            set=np.int64(r["set"]), rally=np.int64(r["rally"]), n_frames=np.int64(len(raw)))
        index.append(dict(name=name, file=path.name, match=mid, set=int(r["set"]), rally=int(r["rally"]),
                          T=int(len(raw)), fps=float(m["fps"]),
                          coreml_vs_torch_full_max=round(d_full, 6),
                          coreml_vs_torch_windowed_max=round(d_win, 6),
                          n_hits=int(len(r["hits"])), first_frame=int(r["frames"][0])))
        print("%-28s T %5d  detections-relevant max|d| coreml/full %.2e coreml/windowed %.2e"
              % (name, len(raw), d_full, d_win), flush=True)
    (FIX / "index.json").write_text(json.dumps(
        dict(model="HitNetDetector.mlpackage", members=members, params=H.PARAMS,
             feature_fn="common.derive(raw, fps)", raw_layout=RAW_LAYOUT, rallies=index), indent=1))
    print("wrote", FIX / "index.json", "with", len(index), "rallies")


if __name__ == "__main__":
    main()
