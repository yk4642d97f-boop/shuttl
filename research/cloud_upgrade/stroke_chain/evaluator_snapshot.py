"""Evaluate frozen HitNet -> fixed-window BST with misses and false alarms retained.

Only annotated singles rally spans are available. This is a two-stage component
chain audit, not full-video scoring/rally/court/3D validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import hit_experiment as E

sys.path.insert(0, str(E.HERE.parent / "bst/coreml"))
import bstcore as B
import convert as CV

OUT = E.HERE / "stroke_chain"
POSE = E.HERE.parent / "rtmpose/pose"


class MatchFrames:
    """Reference feature contract, independently checked against existing raw clips."""
    def __init__(self, mid):
        self.mid = mid
        self.data = CV.load_npz(E.HERE.parent / "extract/out" / f"{mid}.npz")
        d = self.data
        self.frames = d["frame"].astype(np.int64)
        assert len(np.unique(self.frames)) == len(self.frames)
        # Extracts concatenate rally rows; their global storage order need not
        # be chronological (044 has out-of-order rallies). Look up by frame ID.
        self.fps = float(d["fps"])
        self.indices = {int(f): i for i, f in enumerate(self.frames)}
        with np.load(POSE / f"{mid}.npz") as z:
            frame, kp, box = z["frame"], z["kp"], z["box"]
        assert np.array_equal(frame, self.frames), "Pose/extract frame alignment changed"
        kp, box = kp.astype(np.float32), box.astype(np.float32)
        self.covered = np.isfinite(kp[:, :, 0, 0]).any(1)
        good = np.isfinite(box).all(-1) & np.isfinite(kp[:, :, :17, :2]).all((-1, -2))
        good &= ~(kp[:, :, 15, 2] <= 0) & ~(kp[:, :, 16, 2] <= 0)
        # The reference checks the far player first and keeps it if only near fails.
        good[:, 1] &= good[:, 0]
        self.resolved = good.all(1)
        centre = (box[:, :, :2] + box[:, :, 2:]) / np.float32(2)
        diag = np.maximum(np.hypot(box[:, :, 2] - box[:, :, 0], box[:, :, 3] - box[:, :, 1]), 1e-6)
        self.joints = np.where(good[:, :, None, None],
                               (kp[:, :, :17, :2] - centre[:, :, None]) / diag[:, :, None, None],
                               0).astype(np.float32)
        foot = d["player_foot_world"].astype(np.float32)
        fx, fy = foot[:, :, 0], foot[:, :, 1]
        self.pos = np.stack((np.float32(.5) - fy / np.float32(6.1),
                             (np.float32(6.7) - fx) / np.float32(13.4)), -1)
        self.pos[~np.isfinite(fx)] = 0
        self.shuttle = np.zeros((len(frame), 2), np.float32)
        visible = self.resolved & d["shuttle_visible"]
        self.shuttle[visible] = d["shuttle_xy"][visible] / np.array([d["width"], d["height"]], np.float32)
        assert all(np.isfinite(a).all() for a in (self.joints, self.pos, self.shuttle))
        info = json.loads(str(d["info_json"][0]))
        self.label_dir = Path(E.C.LABELS_ROOT) / Path(info["labels"]).name
        self.rows = CV.label_rows(str(self.label_dir))
        self.top_map = CV.top_letter_map(str(self.label_dir))

    def raw_clip(self, frame):
        half = int(round(self.fps)) // 2
        indices = [self.indices[f] for f in range(frame - half, frame + half + 1) if f in self.indices]
        if len(indices) < 2:
            return None, "missing_window"
        if not self.covered[indices].all():
            return None, "missing_pose_coverage"
        if not self.resolved[indices[:B.SEQ_LEN]].any():
            return None, "no_resolved_players"
        return (self.joints[indices], self.pos[indices], self.shuttle[indices]), None

    def labels(self, rally, official):
        rows = self.rows[(rally["set"], rally["rally"])]
        by_frame = {}
        for row in rows:
            by_frame.setdefault(row["_frame"], []).append(row)
        def row_class(row):
            cls = 0
            if row:
                name = f"{int(self.mid)}_{rally['set']}_{rally['rally']}_{row['_round']}"
                label = official.get(name)
                if label in B.CLASSES:
                    cls = B.CLASSES.index(label)
                elif row.get("type") in CV.MERGED[1:]:
                    side = CV.striker_side(row, self.top_map)
                    cls = B.CLASSES.index(f"{side}_{row['type']}")
            return cls
        out = []
        for h in rally["hits"]:
            frame = int(rally["frames"][int(round(h))])
            rr = by_frame.get(frame, [])
            classes = {row_class(row) for row in rr}
            # Some source rows assign different strokes to the very same frame.
            # A frame-only detector cannot disambiguate them. Retain these hits
            # in detection scoring but mark their stroke labels unknown, rather
            # than silently choosing whichever CSV row happened to be last.
            ambiguous = len(classes) > 1
            cls = next(iter(classes)) if classes and not ambiguous else 0
            label = {"frame": frame, "class": cls}
            if ambiguous:
                label.update(annotation_issue="conflicting_same_frame_classes", source_classes=sorted(classes))
            out.append(label)
        return out


def check_reference_features():
    frames = MatchFrames("039")
    differences = Counter()
    paths = sorted((E.HERE.parent / "e2e/m039_ctrl30/test").glob("*/*_joints.npy"))
    assert paths, "Existing reference clips are missing"
    for path in paths[::max(1, len(paths) // 24)]:
        match, set_no, rally_no, round_no = map(int, path.name.removesuffix("_joints.npy").split("_"))
        row = next(r for r in frames.rows[(set_no, rally_no)] if r["_round"] == round_no)
        raw, reason = frames.raw_clip(row["_frame"])
        assert raw is not None, reason
        for name, actual in zip(("joints", "pos", "shuttle"), raw):
            reference = np.load(str(path).replace("_joints.npy", f"_{name}.npy"))
            assert reference.shape == actual.shape
            delta = float(np.max(np.abs(reference - actual)))
            differences[name] = max(differences[name], delta)
            assert np.allclose(reference, actual, atol=2e-7, rtol=0), (path, name, delta)
    return {"clips_checked": len(paths[::max(1, len(paths) // 24)]), "max_abs_error": dict(differences)}


def type_index(cls):
    return (cls - 1) % 12 + 1 if cls else 0


def score_events(labels, predictions, tolerance=2):
    """Pair by time only, one-to-one, then evaluate class correctness.

    Unknown ground-truth classes are excluded from class denominators, while
    missed known labels remain false negatives. Unmatched classified predictions
    remain false positives. Abstentions remain in the detection report.
    """
    pairs, missing, extra = E.C.scorelib.pair([x["frame"] for x in labels], predictions, tolerance)
    known = sum(x["class"] > 0 for x in labels)
    matched_unknown = {pi for gi, pi, _ in pairs if labels[gi]["class"] == 0}
    n_pred = sum(p.get("class", 0) > 0 and pi not in matched_unknown for pi, p in enumerate(predictions))
    exact = types = classified_pairs = 0
    for gi, pi, _ in pairs:
        gc, pc = labels[gi]["class"], predictions[pi].get("class", 0)
        if gc and pc:
            classified_pairs += 1
            exact += gc == pc
            types += type_index(gc) == type_index(pc)
    return {"detection": [len(pairs), len(extra), len(missing)],
            "joint_25class": [exact, n_pred - exact, known - exact],
            "joint_type": [types, n_pred - types, known - types],
            "known_labels": known, "unknown_labels": len(labels) - known,
            "classified_predictions": n_pred, "matched_classified": classified_pairs,
            "correct_type": types, "correct_25class": exact}


@torch.inference_mode()
def classify(clips, device):
    if not clips:
        return np.zeros((0, 25), np.float32)
    inputs = [B.model_inputs(*clip)[0] for clip in clips]
    joints = np.concatenate([d["joints"] for d in inputs])
    jnb = np.stack([B.reference_jnb(j) for j in joints]).reshape(-1, B.SEQ_LEN, 2, B.IN_DIM)
    arrays = [jnb, np.concatenate([d["shuttle"] for d in inputs]),
              np.concatenate([d["pos"] for d in inputs]),
              np.concatenate([d["valid"] for d in inputs]).sum(1)]
    logits = np.zeros((len(clips), B.N_CLASS), np.float32)
    for path in B.WEIGHTS:
        net = B.load_net(path).to(device)
        for start in range(0, len(clips), 128):
            batch = [torch.from_numpy(a[start:start + 128]).to(device) for a in arrays]
            logits[start:start + 128] += net(*batch).float().cpu().numpy() / len(B.WEIGHTS)
        del net
    assert np.isfinite(logits).all()
    return logits


def aggregate(rows):
    totals = {}
    for key in ("detection", "joint_25class", "joint_type"):
        counts = np.sum([r[key] for r in rows], axis=0).astype(int).tolist()
        totals[key] = {"counts": counts, "f1": E.C.scorelib.prf(*counts)[2]}
    for key in ("known_labels", "unknown_labels", "classified_predictions", "matched_classified", "correct_type", "correct_25class"):
        totals[key] = sum(r[key] for r in rows)
    totals["conditional_type_accuracy"] = totals["correct_type"] / max(1, totals["matched_classified"])
    return totals


def evaluate_match(mid, models, shifts, device, official):
    match = E.C.load_match(mid)
    frames = MatchFrames(mid)
    rally_labels = [frames.labels(r, official) for r in match["rallies"]]
    output = {name: [] for name in shifts}
    inputs, locations, cache = [], [], {}
    for model_name, networks in models.items():
        all_dets = []
        for rally in match["rallies"]:
            logits = E.infer.predict_windowed(networks, rally["raw"], rally["fps"], "cpu")
            all_dets.append(E.C.detections_from_logits(logits, rally["fps"], rally["frames"], E.DECODER))
        for arm, (which, shift) in shifts.items():
            if which != model_name:
                continue
            for ri, (rally, detections) in enumerate(zip(match["rallies"], all_dets)):
                items = [dict(d) for d in detections]
                output[arm].append(items)
                for di, det in enumerate(items):
                    centre = det["frame"] + shift
                    if centre not in cache:
                        clip, reason = frames.raw_clip(centre)
                        if clip is None:
                            cache[centre] = (None, reason)
                        else:
                            cache[centre] = (len(inputs), None)
                            inputs.append(clip)
                    index, reason = cache[centre]
                    det["classifier_centre"] = centre
                    det["class"] = 0
                    if reason:
                        det["abstain_reason"] = reason
                    else:
                        locations.append((arm, ri, di, index))
    logits = classify(inputs, device)
    for arm, ri, di, index in locations:
        output[arm][ri][di]["class"] = int(logits[index].argmax())
    rows = {}
    for arm, per_rally in output.items():
        scores = [score_events(labels, preds) for labels, preds in zip(rally_labels, per_rally)]
        rows[arm] = {"metrics": aggregate(scores),
                     "ambiguous_label_events": sum("annotation_issue" in x for labels in rally_labels for x in labels),
                     "abstentions": dict(Counter(d.get("abstain_reason", "model_unknown")
                                                 for dets in per_rally for d in dets if not d["class"])),
                     "rallies": [{"set": r["set"], "rally": r["rally"], "labels": lab, "predictions": pred}
                                 for r, lab, pred in zip(match["rallies"], rally_labels, per_rally)]}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["val", "test"])
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    a = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    report_path = OUT / f"{a.stage}.json"
    if report_path.exists():
        raise SystemExit("Report exists; do not silently rerun or overwrite this audit.")
    selected = json.loads((E.HERE / "selection.json").read_text())
    candidate = Path(selected["candidate"])
    assert E.sha(candidate) == selected["sha256"]
    old = [E.HIT / "ckpt" / (n + ".pt") for n in ("tcn_s0", "tcn_s1", "tcn_lr")]
    protocol = {"scope": "Annotated singles rally spans; HitNet + BST only; one-to-one temporal pairing at +/-2 frames",
                "selection_metric": "Validation match 041 joint type F1; pose extraction for 033 is unavailable",
                "candidate_hit_sha256": selected["sha256"],
                "old_hit_sha256": {p.name: E.sha(p) for p in old},
                "bst_weights_sha256": {Path(p).name: E.sha(p) for p in B.WEIGHTS},
                "window": "fixed +/-round(fps)//2, existing feature contract and seq30 resampling",
                "baselines": "old ensemble3 and current +2 frame stroke lead; no rule fallback or score logic",
                "limitation": "Previously used matches, broadcast views, annotated spans, no doubles or continuous negatives"}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "Protocol changed after validation"
    else:
        E.write_json(protocol_path, protocol)
    parity = check_reference_features()
    print("FEATURE_PARITY", json.dumps(parity), flush=True)
    models = {"old": E.load_models(old), "candidate": E.load_models([candidate])}
    for nets in models.values():
        for net in nets:
            net.eval()
    shifts = {"old_current_plus2": ("old", 2), "candidate_plus2": ("candidate", 2),
              "candidate_no_shift": ("candidate", 0)}
    if a.stage == "test":
        choice = json.loads((OUT / "selection.json").read_text())
        assert choice["protocol_sha256"] == E.sha(protocol_path)
        shifts = {k: v for k, v in shifts.items() if k in ("old_current_plus2", choice["selected_arm"])}
    mids = ["041"] if a.stage == "val" else E.TEST
    official = CV.official_classes()
    result = {"protocol": protocol, "feature_parity": parity, "per_match": {}}
    t0 = time.monotonic()
    for mid in mids:
        details = evaluate_match(mid, models, shifts, a.device, official)
        E.write_json(OUT / f"{a.stage}_{mid}_events.json", details)
        result["per_match"][mid] = {arm: {"metrics": row["metrics"], "abstentions": row["abstentions"],
                                          "ambiguous_label_events": row["ambiguous_label_events"]}
                                    for arm, row in details.items()}
        print(mid, json.dumps({arm: row["metrics"] for arm, row in details.items()}), flush=True)
    pooled = {}
    for arm in shifts:
        rr = [v[arm]["metrics"] for v in result["per_match"].values()]
        pooled[arm] = aggregate([{k: v["counts"] if isinstance(v, dict) and "counts" in v else v
                                 for k, v in r.items()} for r in rr])
    result["pooled"] = pooled
    result["seconds"] = time.monotonic() - t0
    E.write_json(report_path, result)
    if a.stage == "val":
        # Tie keeps the existing +2 behavior. Test never chooses the shift.
        choice = max(("candidate_plus2", "candidate_no_shift"), key=lambda k: pooled[k]["joint_type"]["f1"])
        E.write_json(OUT / "selection.json", {"selected_arm": choice,
                                              "protocol_sha256": E.sha(protocol_path),
                                              "validation_joint_type_f1": pooled[choice]["joint_type"]["f1"]})
    print("POOLED", json.dumps(pooled), flush=True)


if __name__ == "__main__":
    main()
