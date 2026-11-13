"""Score the CoreML detections with the OFFICIAL scorer path and compare with the shipped
PyTorch ensemble output.

  /path/to/shuttl/research/coreml/venv/bin/python score_official.py

Writes out_coreml/NNN.hits.json (the same JSON schema infer.py writes) and score_official.json.
Both sides go through evaluate.tally -> research/hits/score.py pairing against the ShuttleSet CSVs
with shift 0, which is the protocol behind the reported pooled F1@+-2 = 0.9212.
"""
import json
import os
import sys
import warnings

import numpy as np
import coremltools as ct

import hitnet_core as H

sys.path.insert(0, str(H.HITNET))
warnings.filterwarnings("ignore")
import evaluate as E  # noqa: E402  (parent-repo scorer glue; imported, never modified)

OUT = H.HERE / "out_coreml"
REF_DIR = os.path.join(H.C.OUT, "ens3_windowed")   # the shipped PyTorch ensemble's own output


def run_coreml():
    OUT.mkdir(exist_ok=True)
    model = ct.models.MLModel(str(H.HERE / "HitNetDetector.mlpackage"),
                              compute_units=ct.ComputeUnit.CPU_ONLY)
    thr_logit = float(np.log(H.PARAMS["thr"] / (1 - H.PARAMS["thr"])))
    per_match = {}
    for mid in H.TEST_IDS:
        m = H.C.load_match(mid)
        dets = []
        for r in m["rallies"]:
            X = H.C.derive(r["raw"], r["fps"])[None].astype(np.float32)
            lg = np.asarray(model.predict({"features": X})["logits"])[0].astype(np.float64)
            dets.append(H.decode(lg, r["fps"], r["frames"]))
        H.C.write_match_json(str(OUT / (mid + ".hits.json")), mid, m["label_dir"],
                             m["rallies"], dets, thr_logit)
        per_match[mid] = int(sum(len(d) for d in dets))
    return per_match


def tally(dirpath):
    files = [(mid, os.path.join(dirpath, mid + ".hits.json")) for mid in H.TEST_IDS]
    per, pool, srv, rly, off = E.tally(files, 0)          # shift 0 = learned detector convention
    out = {"pooled": {}, "n_detections": int(sum(per[m]["c"][2][0] + per[m]["c"][2][1] for m in H.TEST_IDS))}
    for t in (1, 2, 4):
        p, r, f = H.C.scorelib.prf(*pool[t])
        out["pooled"]["tol%d" % t] = dict(tp=int(pool[t][0]), fp=int(pool[t][1]), fn=int(pool[t][2]),
                                          precision=round(p, 6), recall=round(r, 6), f1=round(f, 6))
    out["per_match"] = {m: dict(
        tp=int(per[m]["c"][2][0]), fp=int(per[m]["c"][2][1]), fn=int(per[m]["c"][2][2]),
        f1_2=round(H.C.scorelib.prf(*per[m]["c"][2])[2], 6),
        f1_1=round(H.C.scorelib.prf(*per[m]["c"][1])[2], 6),
        f1_4=round(H.C.scorelib.prf(*per[m]["c"][4])[2], 6),
        detections=int(per[m]["c"][2][0] + per[m]["c"][2][1])) for m in H.TEST_IDS}
    out["serve_recall_2"] = [int(srv[0]), int(srv[1])]
    out["nonserve_recall_2"] = [int(rly[0]), int(rly[1])]
    out["offset_histogram_pm4"] = {str(k): int(v) for k, v in sorted(off.items())}
    return out


def main():
    per_match = run_coreml()
    print("CoreML detections per match:", per_match, "total", sum(per_match.values()))
    coreml = tally(str(OUT))
    torch_ = tally(str(REF_DIR))
    coreml["reference_dir"] = str(REF_DIR)
    coreml["report_md_f1_pm2"] = H.REF_F1_2
    res = dict(coreml=coreml, torch_ens3_windowed=torch_,
               identical=coreml["pooled"] == torch_["pooled"] and coreml["per_match"] == torch_["per_match"])
    (H.HERE / "score_official.json").write_text(json.dumps(res, indent=1))
    for name, s in (("coreml", coreml), ("torch", torch_)):
        q = s["pooled"]["tol2"]
        print(f"{name:6s} F1@2 {q['f1']:.4f}  P {q['precision']:.4f} R {q['recall']:.4f}  "
              f"tp/fp/fn {q['tp']}/{q['fp']}/{q['fn']}  dets {s['n_detections']}  "
              f"@1 {s['pooled']['tol1']['f1']:.4f} @4 {s['pooled']['tol4']['f1']:.4f}")
    print("per-match F1@2 identical:", coreml["per_match"] == torch_["per_match"])
    print("pooled counts identical:", coreml["pooled"] == torch_["pooled"])


if __name__ == "__main__":
    main()
