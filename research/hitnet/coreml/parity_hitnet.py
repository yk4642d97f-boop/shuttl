"""CoreML vs PyTorch parity for hitnet, on REAL rallies of the 5 held-out matches.

  /path/to/shuttl/research/coreml/venv/bin/python parity_hitnet.py [--units ...]

Reference: PyTorch, fp32, CPU, infer.predict_windowed (384/192, centre-only) over the 3-checkpoint
logit-average ensemble — the exact pipeline whose pooled test F1@+-2 is 0.9212.  The CoreML model is
run whole-rally (one prediction per rally, T frames) because that is how the app will call it.

Reported: max/total abs logit difference, per-frame classifier agreement, and — the number that
matters — agreement of the FINAL detections after sigmoid + threshold 0.4 + NMS (gap 0.30 s), plus
counts and ms/clip for each compute-unit setting.  Writes parity.json.
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch
import coremltools as ct

import hitnet_core as H

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
MK = H.HERE / "HitNetDetector.mlpackage"


def torch_logits(ens, m):
    """windowed logits per rally (the reference pipeline)."""
    out = []
    t0 = time.time()
    for r in m["rallies"]:
        out.append(H.logits_windowed(ens, r["raw"], r["fps"]))
    return out, time.time() - t0


def coreml_logits(model, m):
    """whole-rally logits per rally, one predict() call each."""
    out = []
    t0 = time.time()
    for r in m["rallies"]:
        X = H.C.derive(r["raw"], r["fps"])[None].astype(np.float32)
        out.append(np.asarray(model.predict({"features": X})["logits"])[0].astype(np.float64))
    return out, time.time() - t0


def side_stats(per_match):
    """per_match: {mid: (rallies, dets)} -> pooled F1 + detection counts."""
    s = H.score_dets(per_match)
    s["n_rallies"] = int(sum(len(r) for r, _ in per_match.values()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="CPU_ONLY,CPU_AND_NE")
    ap.add_argument("--matches", default=",".join(H.TEST_IDS))
    a = ap.parse_args()
    ids = a.matches.split(",")

    ens, members = H.load_ensemble()
    res = {"model": str(MK), "members": members, "params": H.PARAMS,
           "input": "features (1,T,51) fp32, whole rally", "reference": "torch fp32 CPU, predict_windowed 384/192",
           "reference_f1_pm2_report_md": H.REF_F1_2, "matches": ids, "per_match": {}, "runs": []}

    t_lg, t_re, t_de, c_lg, c_de = {}, {}, {}, {}, {}
    ref_dets, ref_logits = {}, {}
    for mid in ids:
        m = H.C.load_match(mid)
        lg, sec = torch_logits(ens, m)
        ref_logits[mid] = lg
        t0 = time.time()
        dets = [H.decode(l, r["fps"], r["frames"]) for l, r in zip(lg, m["rallies"])]
        t_de[mid] = time.time() - t0
        t_lg[mid] = sec
        ref_dets[mid] = dets
        n = sum(len(r["raw"]) for r in m["rallies"])
        res["per_match"][mid] = {"n_rallies": len(m["rallies"]), "n_frames": int(n), "fps": m["fps"],
                                 "torch_detections": int(sum(len(d) for d in dets)),
                                 "torch_s_per_match": sec, "torch_ms_per_clip": 1000 * sec / len(m["rallies"])}
        print(f"torch {mid}: {len(m['rallies'])} rallies, {n} frames, "
              f"{res['per_match'][mid]['torch_detections']} detections, {sec:.2f} s", flush=True)

    ref = side_stats({mid: (H.C.load_match(mid)["rallies"], ref_dets[mid]) for mid in ids})
    res["torch"] = dict(ref, s_infer=sum(t_lg.values()), s_decode=sum(t_de.values()))

    for unit in a.units.split(","):
        cu = getattr(ct.ComputeUnit, unit)
        model = ct.models.MLModel(str(MK), compute_units=cu)
        run = {"units": unit, "per_match": {}, "logit_diff": {}, "detections": {}}
        tot_diff, tot_abs, tot_n = 0.0, 0.0, 0
        agree = tot = 0
        cml_dets = {}
        t_all = 0.0
        for mid in ids:
            m = H.C.load_match(mid)
            lg, sec = coreml_logits(model, m)
            c_lg[mid], c_de[mid] = lg, sec
            t_all += sec
            diff = [np.abs(a - b) for a, b in zip(lg, ref_logits[mid])]
            n = sum(d.size for d in diff)
            mx = max(float(d.max()) for d in diff)
            tot_diff = max(tot_diff, mx)
            tot_abs += sum(float(d.sum()) for d in diff)
            tot_n += n
            # per-frame classifier agreement at the operating threshold
            for (aa, bb) in zip(lg, ref_logits[mid]):
                agree += int(((aa[:, 0] > 0) == (bb[:, 0] > 0)).sum() + ((aa[:, 1] > 0) == (bb[:, 1] > 0)).sum())
                tot += 2 * len(aa)
            dets = [H.decode(l, r["fps"], r["frames"]) for l, r in zip(lg, m["rallies"])]
            cml_dets[mid] = dets
            a_fr = [d["frame"] for per_rally in dets for d in per_rally]
            b_fr = [d["frame"] for per_rally in ref_dets[mid] for d in per_rally]
            a_set, b_set = set(a_fr), set(b_fr)
            run["per_match"][mid] = {
                "max_abs_logit_diff": mx,
                "mean_abs_logit_diff": float(sum(d.sum() for d in diff) / n),
                "coreml_detections": int(sum(len(d) for d in dets)),
                "torch_detections": int(sum(len(d) for d in ref_dets[mid])),
                "frame_agree": int(len(a_set & b_set)), "only_coreml": int(len(a_set - b_set)),
                "only_torch": int(len(b_set - a_set)),
                "ordered_frames_identical": bool(a_fr == b_fr),
                "n_frames_in_order_match": int(sum(1 for x, y in zip(a_fr, b_fr) if x == y)),
                "coreml_ms_per_clip": 1000 * sec / len(m["rallies"]),
            }
            v = run["per_match"][mid]
            print(f"{unit} {mid}: max|dlogit| {mx:.2e}  dets {v['coreml_detections']}"
                  f" vs {v['torch_detections']}  frame agree {v['frame_agree']}"
                  f" (+{v['only_coreml']}/-{v['only_torch']})  ordered match "
                  f"{v['n_frames_in_order_match']}/{len(b_fr)}", flush=True)
        run["max_abs_logit_diff"] = tot_diff
        run["mean_abs_logit_diff"] = tot_abs / tot_n
        run["frame_classifier_agreement"] = agree / tot
        run["total_detections"] = int(sum(v["coreml_detections"] for v in run["per_match"].values()))
        run["frame_agree"] = int(sum(v["frame_agree"] for v in run["per_match"].values()))
        run["only_coreml"] = int(sum(v["only_coreml"] for v in run["per_match"].values()))
        run["only_torch"] = int(sum(v["only_torch"] for v in run["per_match"].values()))
        run["ordered_match"] = int(sum(v["n_frames_in_order_match"] for v in run["per_match"].values()))
        run["all_ordered_identical"] = bool(all(v["ordered_frames_identical"] for v in run["per_match"].values()))
        run["ms_per_clip"] = 1000 * t_all / res["torch"]["n_rallies"]
        run["score"] = side_stats({mid: (H.C.load_match(mid)["rallies"], cml_dets[mid]) for mid in ids})
        # graph-only error: CoreML (whole rally) vs torch run the SAME way (whole rally), so this
        # isolates the mlprogram from the windowed-vs-whole-rally approximation the numbers above mix in
        g_max = 0.0
        for mid in ids:
            m = H.C.load_match(mid)
            for lg, r in zip(c_lg[mid], m["rallies"]):
                with torch.no_grad():
                    ref = ens(torch.from_numpy(H.C.derive(r["raw"], r["fps"]))[None])[0].float().numpy()
                g_max = max(g_max, float(np.abs(np.asarray(lg) - ref).max()))
        run["graph_max_abs_logit_diff_whole_rally"] = g_max
        res["runs"].append(run)
        print(f"{unit}: max|dlogit| {tot_diff:.2e} mean {run['mean_abs_logit_diff']:.2e} "
              f"frame-clf agree {run['frame_classifier_agreement']:.6f} | dets {run['total_detections']} vs "
              f"{res['torch']['n_detections']}, frames {run['frame_agree']} exact "
              f"(+{run['only_coreml']}/-{run['only_torch']}) | F1@2 {run['score']['f1_2']:.4f} "
              f"vs torch {res['torch']['f1_2']:.4f} | {run['ms_per_clip']:.1f} ms/clip | "
              f"graph-only max|d| {g_max:.2e}", flush=True)

    json.dump(res, open(H.HERE / "parity.json", "w"), indent=1)
    print("torch pooled F1@2 %.4f (REPORT.md %.4f)" % (res["torch"]["f1_2"], H.REF_F1_2))


if __name__ == "__main__":
    main()
