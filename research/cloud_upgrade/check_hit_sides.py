"""Audit the frozen detector's hitter-side accuracy with a fresh feature cache.

This is a post-selection diagnostic, not another candidate search.
"""
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import hit_experiment as E


def main():
    torch.set_num_threads(2)
    selected = json.loads((E.HERE / "selection.json").read_text())
    candidate = Path(selected["candidate"])
    assert E.sha(candidate) == selected["sha256"]
    previous = json.loads((E.HERE / "comparison.json").read_text())
    models = {
        "old_ensemble3": E.load_models([E.HIT / "ckpt" / (n + ".pt")
                                       for n in ("tcn_s0", "tcn_s1", "tcn_lr")]),
        "candidate_single": E.load_models([candidate]),
    }
    for networks in models.values():
        for network in networks:
            network.eval()
    result = {"candidate_sha256": selected["sha256"],
              "scope": "Singles, matched hits within 2 frames; fresh cache, no tuning"}
    # Rebuild from the current NPZ + CSV labels, independently of the old cache key.
    with tempfile.TemporaryDirectory(prefix="shuttl-hit-audit-") as cache:
        E.C.CACHE = cache
        for partition in ("audit", "test"):
            _, rallies = E.load_partition(partition)
            arms = {}
            for name, networks in models.items():
                counts = Counter()
                detections = []
                for rally in rallies:
                    logits = E.infer.predict_windowed(networks, rally["raw"], rally["fps"], "cpu")
                    dets = E.C.detections_from_logits(logits, rally["fps"], rally["frames"], E.DECODER)
                    detections.append(dets)
                    frames = [int(rally["frames"][int(round(h))]) for h in rally["hits"]]
                    pairs, _, _ = E.C.scorelib.pair(frames, dets, 2)
                    for gi, di, _ in pairs:
                        side = int(rally["side"][gi])
                        if side < 0:
                            counts["unknown_label"] += 1
                            continue
                        predicted = int(dets[di]["hitter"] == "far")
                        counts["correct"] += int(predicted == side)
                        counts["matched_with_side"] += 1
                        counts[f"true_{side}_pred_{predicted}"] += 1
                detection_counts, _, _ = E.C.eval_dets(rallies, detections)
                assert list(detection_counts[2]) == previous[partition][name]["counts"]["2"], \
                    "Fresh-cache detections differ from recorded comparison"
                arms[name] = {**counts,
                              "side_accuracy": counts["correct"] / counts["matched_with_side"],
                              "detection_counts_at_2": detection_counts[2],
                              "fresh_cache_matches_prior_result": True}
            result[partition] = arms
            print(partition, json.dumps(arms), flush=True)
    E.write_json(E.HERE / "side_audit.json", result)


if __name__ == "__main__":
    main()
