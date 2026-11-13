#!/usr/bin/env python
"""Train and evaluate the learned hit candidate re-scorer.

Run from ``backend/`` with the project venv, for example:

  .venv/bin/python ml/train_hit_rescorer.py --sim-train 160 --sim-val 40 \
      --shuttleset-root /tmp/shuttleset-coachai/ShuttleSet/set

The script never reads TrackNetV2/Test while fitting.  Test clips are loaded
only by the final ``--evaluate-real`` pass.  It writes a JSON run record under
``backend/ml/runs`` and exports the NPZ only when the selected model improves
both required held-out F1 measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.tracking import hits  # noqa: E402
from app.services.tracking.data_types import HitEvent, Track2D, VideoMeta  # noqa: E402
from ml import hit_rescorer_data as D  # noqa: E402


DEFAULT_SHUTTLESET = Path("/tmp/shuttleset-coachai/ShuttleSet/set")
DEFAULT_OUT = ROOT / "ml/weights/hit_rescorer.npz"
DEFAULT_RUNS = ROOT / "ml/runs"


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    with np.errstate(over="ignore"):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 max_iter: int = 100, class_weight: float | None = None,
                 ) -> tuple[np.ndarray, float, int]:
    """Small deterministic Newton solver; no sklearn object is serialized."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, f = X.shape
    if class_weight is None:
        cw = np.ones(n)
    else:
        cw = np.where(y > 0.5, float(class_weight), 1.0)
    A = np.column_stack([X, np.ones(n)])
    theta = np.zeros(f + 1, float)
    reg = np.zeros(f + 1, float)
    reg[:f] = float(l2)
    for it in range(max_iter):
        z = np.clip(A @ theta, -40.0, 40.0)
        p = sigmoid(z)
        grad = A.T @ (cw * (p - y)) + reg * theta
        W = cw * p * (1.0 - p)
        hess = (A.T * W) @ A + np.diag(reg + 1e-8)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        theta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            return theta[:f], float(theta[-1]), it + 1
    return theta[:f], float(theta[-1]), max_iter


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-7), sd, 1.0)
    return (np.asarray(X, float) - mu) / sd, mu, sd


def scores_for_model(X: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray,
                     sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, float) - mu) / sd) @ w + float(b)


def candidate_threshold_scores(cands: list[Any], logits: np.ndarray,
                               blend: float = 0.5) -> list[Any]:
    """Install a model result in candidate copies without mutating runtime state."""
    out = []
    for c, z in zip(cands, logits):
        q = type(c)(**{k: getattr(c, k) for k in c.__dataclass_fields__})
        q.kin = (1.0 - blend) * float(c.kin) + blend * float(z)
        out.append(q)
    return out


def _hit_events(cands: list[Any], config: hits.HitConfig | None = None,
                pair_w: float | None = None) -> list[HitEvent]:
    if not cands:
        return []
    cfg = config or hits.HitConfig()
    diag = math.hypot(1280, 720)
    dec = hits._Decoder(cands, cfg, diag,
                        cfg.pair_y_weight if pair_w is None else pair_w)
    chain = [(k, s) for k, s in dec.viterbi() if dec.hit[k] >= cfg.min_hit_logodds]
    idx = [k for k, _ in chain]
    p_near = dec.posteriors(idx)
    out = []
    for k, pn in zip(idx, p_near):
        c = cands[k]
        side = "near" if pn >= 0.6 else ("far" if pn <= 0.4 else "unknown")
        out.append(HitEvent(frame=int(c.frame), t=float(c.tau),
                            xy_px=(float(c.xy[0]), float(c.xy[1])), hitter=side,
                            confidence=float(1.0 / (1.0 + math.exp(-dec.hit[k])))))
    return out


def _matched(pred: list[int], truth: list[int], tol: int = 3) -> list[tuple[int, int]]:
    return D.event_pairs(pred, truth, tol)


def event_report(pred: list[HitEvent], truth: list[dict[str, Any] | HitEvent],
                 tol: int = 3) -> dict[str, Any]:
    pf = [int(x.frame) for x in pred]
    tf = [int(x["frame"] if isinstance(x, dict) else x.frame) for x in truth]
    pairs = _matched(pf, tf, tol)
    tp, fp, fn = len(pairs), len(pf) - len(pairs), len(tf) - len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    sides = []
    for i, j in pairs:
        ph = pred[i].hitter
        th = truth[j]["hitter"] if isinstance(truth[j], dict) else truth[j].hitter
        if th in ("near", "far"):
            sides.append(ph == th)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_truth = {j for _, j in pairs}
    truth_types = {
        j: (str(row.get("shot_type", "unknown")) if isinstance(row, dict) else "unknown")
        for j, row in enumerate(truth)
    }
    for typ in sorted(set(truth_types.values())):
        tp_type = sum(1 for _, j in pairs if truth_types[j] == typ)
        fn_type = sum(1 for j in range(len(truth))
                      if truth_types[j] == typ and j not in matched_truth)
        # The runtime emits a hit/no-hit decision, not a shot classifier.  For
        # this one-vs-rest diagnostic every detected event outside this type is
        # a false positive for the type-specific hit detector.
        fp_type = len(pf) - tp_type
        by_type[typ] = {"tp": tp_type, "fp": fp_type, "fn": fn_type}
    per_type = {}
    for typ, v in sorted(by_type.items()):
        tpp, fpp, fnt = v["tp"], v["fp"], v["fn"]
        pp = tpp / (tpp + fpp) if tpp + fpp else 0.0
        rec = tpp / (tpp + fnt) if tpp + fnt else 0.0
        per_type[typ] = {"tp": tpp, "fp": fpp, "fn": fnt,
                         "precision": pp, "recall": rec,
                         "f1": 2 * pp * rec / (pp + rec) if pp + rec else 0.0}
    return {"tp": tp, "fp": fp, "fn": fn, "missed_hits": fn,
            "spurious_hits": fp, "n_pred": len(pf), "n_true": len(tf),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            "side_accuracy": float(np.mean(sides)) if sides else None,
            "side_correct": int(np.sum(sides)), "side_count": len(sides),
            "per_shot_type": per_type}


def _truth_sim(r: Any) -> list[dict[str, Any]]:
    return [{"frame": int(h["frame"]), "hitter": str(h["hitter"]),
             "shot_type": str(h.get("shot_type", "unknown"))} for h in r.hits]


def evaluate_sim(seeds: list[int], image_size: tuple[int, int], fps: float,
                 sigma: float, miss_rate: float, fp_rate: float, blend: float,
                 w: np.ndarray | None = None, b: float = 0.0,
                 mu: np.ndarray | None = None, sd: np.ndarray | None = None,
                 rallies: int = 1, style: str = "random") -> dict[str, Any]:
    """Evaluate full sim hit events on fresh seeds, with per-shot diagnostics."""
    aggregate = {"tp": 0, "fp": 0, "fn": 0, "missed_hits": 0,
                 "spurious_hits": 0, "n_pred": 0, "n_true": 0}
    side_ok = side_n = 0
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    baseline_times, learned_times = [], []
    per_sample = []
    for seed in seeds:
        fps_i = float(fps if fps else (30.0 + 30.0 * (seed % 3)))
        r, track, players, _ = D.make_sim_sample(seed, image_size, fps_i, style, rallies,
                                                  sigma, miss_rate, fp_rate)
        meta = r.meta
        pair_w = hits.HitConfig().pair_y_weight * (0.5 if (r.camera is not None or players) else 1.0)
        t0 = time.perf_counter()
        base_cands = hits.detect_candidates(hits._prefer_raw(track), meta, r.camera, players)
        candidate_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        base_pred = _hit_events(base_cands, pair_w=pair_w)
        baseline_times.append(candidate_seconds + time.perf_counter() - t0)
        learned_pred = base_pred
        if w is not None:
            t0 = time.perf_counter()
            X = np.stack([np.asarray(c.feats, float) for c in base_cands]) if base_cands else np.empty((0, len(D.FEATURES)))
            logits = scores_for_model(X, w, b, mu, sd) if len(X) else np.empty(0)
            learned_cands = candidate_threshold_scores(base_cands, logits, blend)
            learned_pred = _hit_events(learned_cands, hits.HitConfig(learned_blend=blend), pair_w)
            learned_times.append(candidate_seconds + time.perf_counter() - t0)
        else:
            learned_times.append(baseline_times[-1])
        truth = _truth_sim(r)
        report = event_report(learned_pred, truth)
        for key in aggregate:
            aggregate[key] += int(report[key])
        side_n += report["side_count"]
        side_ok += report["side_correct"]
        for typ, vals in report["per_shot_type"].items():
            for key in ("tp", "fp", "fn"):
                type_counts[typ][key] += vals[key]
        per_sample.append({"seed": seed, **report})
    p = aggregate["tp"] / aggregate["n_pred"] if aggregate["n_pred"] else 0.0
    r = aggregate["tp"] / aggregate["n_true"] if aggregate["n_true"] else 0.0
    aggregate.update({"precision": p, "recall": r,
                      "f1": 2 * p * r / (p + r) if p + r else 0.0,
                      "side_accuracy": side_ok / side_n if side_n else None,
                      "latency_ms_mean": 1000 * float(np.mean(learned_times)),
                      "latency_ms_median": 1000 * float(np.median(learned_times)),
                      "latency_ms_p95": 1000 * float(np.percentile(learned_times, 95)),
                      "baseline_latency_ms_mean": 1000 * float(np.mean(baseline_times)),
                      "learned_over_baseline_latency_ratio": float(np.mean(learned_times) / max(np.mean(baseline_times), 1e-9)),
                      "per_shot_type": {typ: {**v,
                                                "precision": v["tp"] / (v["tp"] + v["fp"])
                                                if v["tp"] + v["fp"] else 0.0,
                                                "recall": v["tp"] / (v["tp"] + v["fn"])
                                                if v["tp"] + v["fn"] else 0.0,
                                                "f1": 2 * (v["tp"] / (v["tp"] + v["fp"])) * (v["tp"] / (v["tp"] + v["fn"])) /
                                                ((v["tp"] / (v["tp"] + v["fp"])) + (v["tp"] / (v["tp"] + v["fn"])))
                                                if v["tp"] and v["tp"] + v["fp"] and v["tp"] + v["tp"] + v["fn"] else 0.0}
                                        for typ, v in sorted(type_counts.items())},
                      "per_sample": per_sample})
    return aggregate


def evaluate_real(root: Path, blend: float, w: np.ndarray | None = None,
                  b: float = 0.0, mu: np.ndarray | None = None,
                  sd: np.ndarray | None = None) -> dict[str, Any]:
    aggregate = {"tp": 0, "fp": 0, "fn": 0, "missed_hits": 0,
                 "spurious_hits": 0, "n_pred": 0, "n_true": 0}
    lat_base, lat_learned = [], []
    per_clip = []
    for c in D.real_test_clips(root):
        meta0 = D._probe(c["video"])
        xy, visible = D.load_tracknet_csv(c["labels"], meta0.n_frames)
        track = Track2D(t=meta0.timestamps, xy=xy, conf=visible.astype(float),
                        visible=visible, interpolated=np.zeros(meta0.n_frames, bool))
        meta = meta0
        truth_frames = D.reversal_labels(xy, visible, meta.fps)
        t0 = time.perf_counter()
        base_cands = hits.detect_candidates(hits._prefer_raw(track), meta)
        candidate_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        base_pred = _hit_events(base_cands)
        lat_base.append(candidate_seconds + time.perf_counter() - t0)
        learned_pred = base_pred
        if w is not None:
            t0 = time.perf_counter()
            X = np.stack([np.asarray(x.feats, float) for x in base_cands]) if base_cands else np.empty((0, len(D.FEATURES)))
            logits = scores_for_model(X, w, b, mu, sd) if len(X) else np.empty(0)
            learned_pred = _hit_events(candidate_threshold_scores(base_cands, logits, blend),
                                       hits.HitConfig(learned_blend=blend), hits.HitConfig().pair_y_weight)
            lat_learned.append(candidate_seconds + time.perf_counter() - t0)
        else:
            lat_learned.append(lat_base[-1])
        report = event_report(learned_pred, [{"frame": x, "hitter": "unknown",
                                              "shot_type": "unknown"} for x in truth_frames])
        for key in aggregate:
            aggregate[key] += int(report[key])
        per_clip.append({"match": c["match"], "clip": c["clip"], **report,
                         "truth_source": "independent direction reversal"})
    p = aggregate["tp"] / aggregate["n_pred"] if aggregate["n_pred"] else 0.0
    r = aggregate["tp"] / aggregate["n_true"] if aggregate["n_true"] else 0.0
    aggregate.update({"precision": p, "recall": r,
                      "f1": 2 * p * r / (p + r) if p + r else 0.0,
                      "side_accuracy": None,
                      "latency_ms_mean": 1000 * float(np.mean(lat_learned)) if lat_learned else float("nan"),
                      "latency_ms_median": 1000 * float(np.median(lat_learned)) if lat_learned else float("nan"),
                      "latency_ms_p95": 1000 * float(np.percentile(lat_learned, 95)) if lat_learned else float("nan"),
                      "baseline_latency_ms_mean": 1000 * float(np.mean(lat_base)) if lat_base else float("nan"),
                      "per_clip": per_clip})
    return aggregate


def _split_sim(rows: D.CandidateRows, val_start: int, val_n: int) -> tuple[D.CandidateRows, D.CandidateRows]:
    groups = np.array([int(str(x).split(":")[-1]) for x in rows.groups])
    val_groups = set(range(val_start, val_start + val_n))
    val = np.isin(groups, list(val_groups))
    return rows.take(~val), rows.take(val)


def _candidate_auc(y: np.ndarray, z: np.ndarray) -> float:
    order = np.argsort(-z)
    y = np.asarray(y)[order]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if not pos or not neg:
        return float("nan")
    rank = np.flatnonzero(y == 1) + 1
    return float(1.0 - (rank.sum() - pos * (pos + 1) / 2) / (pos * neg))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--sim-train", type=int, default=160)
    ap.add_argument("--sim-val", type=int, default=40)
    ap.add_argument("--sim-test", type=int, default=20)
    ap.add_argument("--eval-seed", type=int, default=0,
                    help="held-out sim seed start; these seeds are never fitted")
    ap.add_argument("--sim-image-size", default="1280x720")
    ap.add_argument("--sim-rallies", type=int, default=1)
    ap.add_argument("--shuttleset-root", type=Path, default=DEFAULT_SHUTTLESET)
    ap.add_argument("--shuttleset-rallies", type=int, default=150,
                    help="rallies per match split pass; 0 means all annotation rallies")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--positive-class-weight", type=float, default=1.0)
    ap.add_argument("--logit-scale", type=float, default=1.0)
    ap.add_argument("--no-real", dest="evaluate_real", action="store_false",
                    help="skip the final held-out TrackNetV2 Test evaluation")
    ap.set_defaults(evaluate_real=True)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-shuttleset", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    np.random.seed(args.seed)
    w_img, h_img = (int(x) for x in args.sim_image_size.lower().split("x"))
    run: dict[str, Any] = {"seed": args.seed, "features": list(D.FEATURES),
                           "args": {k: str(v) for k, v in vars(args).items()},
                           "source": {"shuttleset_root": str(args.shuttleset_root),
                                      "test_root": str(D.DEFAULT_TEST_ROOT)}}
    print("collecting simulated candidate rows")
    sim_train, sim_train_info = D.collect_sim_rows(args.seed, args.sim_train,
                                                    (w_img, h_img), rallies=args.sim_rallies)
    sim_val, sim_val_info = D.collect_sim_rows(args.seed + args.sim_train, args.sim_val,
                                                (w_img, h_img), rallies=args.sim_rallies)
    sim_test, sim_test_info = D.collect_sim_rows(args.eval_seed, args.sim_test,
                                                  (w_img, h_img), rallies=args.sim_rallies)
    train = sim_train
    ss_info = {}
    if not args.no_shuttleset and args.shuttleset_root.exists():
        cap = args.shuttleset_rallies or None
        ss_train, ss_train_info = D.collect_shuttleset_rows(args.shuttleset_root, "train",
                                                            args.seed, cap)
        ss_val, ss_val_info = D.collect_shuttleset_rows(args.shuttleset_root, "val",
                                                        args.seed, cap)
        train = train.concat(ss_train)
        ss_info = {"train": ss_train_info, "val": ss_val_info,
                   "train_rows": len(ss_train), "val_rows": len(ss_val),
                   "train_groups": sorted(set(map(str, ss_train.groups))),
                   "val_groups": sorted(set(map(str, ss_val.groups)))}
    run["sample_info"] = {"sim_train": sim_train_info, "sim_val": sim_val_info,
                           "sim_test": sim_test_info}
    run["rows"] = {"sim_train": len(sim_train), "sim_val": len(sim_val),
                    "sim_test": len(sim_test), "train_total": len(train),
                    "sim_train_positive": int(sim_train.y.sum()),
                    "sim_val_positive": int(sim_val.y.sum()),
                    "sim_test_positive": int(sim_test.y.sum()), "shuttleset": ss_info}
    print("fitting standardized logistic model")
    X, mu, sd = standardize(train.X)
    w, b, iterations = fit_logistic(X, train.y, l2=args.l2,
                                    class_weight=args.positive_class_weight)
    w = w * args.logit_scale
    b = b * args.logit_scale
    val_X = (sim_val.X - mu) / sd
    auc = _candidate_auc(sim_val.y, val_X @ w + b)
    run["model"] = {"l2": args.l2,
                    "positive_class_weight": args.positive_class_weight,
                    "logit_scale": args.logit_scale, "iterations": iterations,
                    "validation_candidate_auc": auc, "mu": mu.tolist(), "sd": sd.tolist(),
                    "w": w.tolist(), "b": float(b)}
    print(f"selected l2={args.l2:g} positive_weight={args.positive_class_weight:g} "
          f"scale={args.logit_scale:g} validation_candidate_auc={auc:.4f}")

    val_seeds = list(range(args.seed + args.sim_train,
                           args.seed + args.sim_train + args.sim_val))
    val_base = evaluate_sim(val_seeds, (w_img, h_img), 60.0, 1.5, 0.1, 0.03,
                            args.blend, rallies=args.sim_rallies)
    val_model = evaluate_sim(val_seeds, (w_img, h_img), 60.0, 1.5, 0.1, 0.03,
                             args.blend, w, b, mu, sd, args.sim_rallies)
    run["sim_validation_baseline"] = val_base
    run["sim_validation_model"] = val_model

    # Baseline and learned event metrics are both measured on the untouched sim test.
    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.sim_test))
    baseline = evaluate_sim(eval_seeds,
                            (w_img, h_img), 60.0, 1.5, 0.1, 0.03, args.blend,
                            rallies=args.sim_rallies)
    learned = evaluate_sim(eval_seeds,
                           (w_img, h_img), 60.0, 1.5, 0.1, 0.03, args.blend, w, b, mu, sd,
                           args.sim_rallies)
    run["sim_test_baseline"] = baseline
    run["sim_test_learned"] = learned
    if args.evaluate_real:
        print("evaluating real Test clips (labels are independent direction reversals)")
        real_base = evaluate_real(D.DEFAULT_TEST_ROOT, args.blend)
        real_learned = evaluate_real(D.DEFAULT_TEST_ROOT, args.blend, w, b, mu, sd)
        run["real_test_baseline"] = real_base
        run["real_test_learned"] = real_learned
    else:
        real_base = real_learned = None
    install_ok = (learned["f1"] > baseline["f1"] and
                  (real_learned is None or real_learned["f1"] > real_base["f1"]))
    run["exported"] = False
    run["install_gate_passed"] = bool(install_ok)
    if install_ok and not args.no_export:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, w=np.asarray(w, float), b=float(b), mu=np.asarray(mu, float),
                  sd=np.asarray(sd, float), feature_names=np.asarray(D.FEATURES))
        run["exported"] = True
        print(f"exported {args.out}")
    else:
        print("model not exported: held-out gate did not pass" if not install_ok
              else "model export disabled")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.runs_dir / f"hit_rescorer_{args.seed}.json"
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=True, allow_nan=False))
    print(json.dumps({"sim_baseline": baseline, "sim_learned": learned,
                      "real_baseline": real_base, "real_learned": real_learned,
                      "exported": run["exported"], "run": str(run_path)},
                     indent=2, ensure_ascii=True, allow_nan=True))
    return run


if __name__ == "__main__":
    main()
