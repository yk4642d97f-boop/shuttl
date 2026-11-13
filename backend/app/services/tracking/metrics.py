"""
Evaluation metrics for the v2 tracking stack (pure numpy; duck-typed inputs so
dataclasses from data_types.py, sim.py truth dicts and plain arrays all work).

  detection_metrics(pred_xy, true_xy, true_vis=None, tol_px=10.0)      per-frame shuttle
  candidates_to_xy(cands, score_thresh=0.5)                           top-1 detector output
  scaled_tolerance(tol_ref=4.0, image_size, ref_size=(512, 288))      TrackNet-style tolerance
  track_metrics(track, true_xy, true_vis=None, tol_px=10.0)           + jitter / jumps / fill
  hit_metrics(pred_hits, true_hits, tol_frames=3)                      F1, offset, hitter acc
  segment_metrics(pred_segments, true_shots, tol_frames=4)             speed / height / landing
  in_out_metrics(pred_calls, true_calls, true_margins=None, close_m=0.2)
  landing_metrics(pred_xy, true_xy)                                   metric landing error
  shot_type_metrics(pred_types, true_types, labels=None)
  rally_metrics(pred_rallies, true_rallies, iou_thresh=0.5)            segmentation + outcome
  confusion_matrix(pred, true, labels=None)
  format_table(results)                                                pretty print nested dicts

Per-frame detection classification follows TrackNetV3 (test.py `evaluate`):
  pred ∧ truth: TP if ‖pred − truth‖ ≤ tol else FP1;  pred ∧ ¬truth: FP2;
  ¬pred ∧ truth: FN;  neither: TN.
  precision = TP/(TP+FP1+FP2), recall = TP/(TP+FN), accuracy = (TP+TN)/N.
TrackNetV3 measures 4 px on the 512×288 heatmap grid with integer-truncated
coordinates; `scaled_tolerance(4, (1280, 720))` = 10 px is the full-resolution
equivalent (use `tracknet_int=True` to also mimic the truncation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

REF_SIZE = (512, 288)


# ── helpers ──────────────────────────────────────────────────────────────────

def _get(o: Any, name: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(name, default)
    return getattr(o, name, default)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _f1(p: float, r: float) -> float:
    return float(2 * p * r / (p + r)) if (p + r) > 0 and np.isfinite(p + r) else 0.0


def _stats(err: np.ndarray, prefix: str = "err") -> dict[str, float]:
    err = np.asarray(err, float)
    err = err[np.isfinite(err)]
    if not len(err):
        return {f"{prefix}_mean": float("nan"), f"{prefix}_median": float("nan"),
                f"{prefix}_p95": float("nan"), f"{prefix}_n": 0}
    return {f"{prefix}_mean": float(err.mean()), f"{prefix}_median": float(np.median(err)),
            f"{prefix}_p95": float(np.percentile(err, 95)), f"{prefix}_n": int(len(err))}


def scaled_tolerance(tol_ref: float = 4.0, image_size: tuple[int, int] = (1280, 720),
                     ref_size: tuple[int, int] = REF_SIZE) -> float:
    """A tolerance defined on the ref grid (TrackNet: 4 px @ 512×288) in image px."""
    return float(tol_ref * image_size[0] / ref_size[0])


def candidates_to_xy(cands: Sequence[Sequence[Any]], score_thresh: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Top-scoring candidate per frame → ((N,2) xy with NaN below threshold, (N,) score)."""
    n = len(cands)
    xy, sc = np.full((n, 2), np.nan), np.zeros(n)
    for i, cs in enumerate(cands):
        if not cs:
            continue
        best = max(cs, key=lambda c: _get(c, "score", 0.0))
        s = float(_get(best, "score", 0.0))
        sc[i] = s
        if s >= score_thresh:
            xy[i] = (_get(best, "x"), _get(best, "y"))
    return xy, sc


def _as_xy(pred: Any) -> np.ndarray:
    """Track2D | (N,2) array | candidate lists → (N,2) with NaN for 'no prediction'."""
    if hasattr(pred, "xy") and hasattr(pred, "t"):
        return np.asarray(pred.xy, float)
    if isinstance(pred, (list, tuple)) and pred and isinstance(pred[0], (list, tuple)) \
            and (not pred[0] or not np.isscalar(pred[0][0])):
        return candidates_to_xy(pred)[0]
    return np.asarray(pred, float).reshape(-1, 2)


def _truth_vis(true_xy: np.ndarray, true_vis: np.ndarray | None) -> np.ndarray:
    ok = np.all(np.isfinite(true_xy), axis=1)
    return ok if true_vis is None else ok & np.asarray(true_vis, bool)


# ── per-frame detection / track ──────────────────────────────────────────────

def detection_metrics(pred_xy: Any, true_xy: np.ndarray, true_vis: np.ndarray | None = None,
                      tol_px: float = 10.0, mask: np.ndarray | None = None,
                      tracknet_int: bool = False, image_size: tuple[int, int] | None = None
                      ) -> dict[str, float]:
    """TrackNetV3-style TP/TN/FP1/FP2/FN counts and P/R/F1 + pixel error stats.

    pred_xy: (N,2) px with NaN where nothing is predicted (or a Track2D or
    per-frame candidate lists). true_vis defaults to 'true_xy finite'. mask
    restricts the evaluated frames (e.g. TrackNetV3 drop-frame ranges).
    tracknet_int: compare int-truncated coordinates on the 512×288 grid (needs
    image_size; tol_px is then interpreted on that grid, i.e. use 4)."""
    p = _as_xy(pred_xy)
    t = np.asarray(true_xy, float).reshape(-1, 2)
    if len(p) != len(t):
        raise ValueError(f"length mismatch: pred {len(p)} vs truth {len(t)}")
    tv = _truth_vis(t, true_vis)
    pv = np.all(np.isfinite(p), axis=1)
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)
    if tracknet_int:
        if image_size is None:
            raise ValueError("tracknet_int needs image_size")
        s = np.array([REF_SIZE[0] / image_size[0], REF_SIZE[1] / image_size[1]])
        d = np.full(len(t), np.inf)
        both = pv & tv
        d[both] = np.linalg.norm(np.trunc(p[both] * s) - np.trunc(t[both] * s), axis=1)
    else:
        d = np.linalg.norm(p - t, axis=1)
    both = pv & tv & m
    tp = int(np.sum(both & (d <= tol_px)))
    fp1 = int(np.sum(both & ~(d <= tol_px)))
    fp2 = int(np.sum(pv & ~tv & m))
    fn = int(np.sum(~pv & tv & m))
    tn = int(np.sum(~pv & ~tv & m))
    prec, rec = _ratio(tp, tp + fp1 + fp2), _ratio(tp, tp + fn)
    out = {"tp": tp, "tn": tn, "fp1": fp1, "fp2": fp2, "fn": fn, "n_frames": int(m.sum()),
           "precision": prec, "recall": rec, "f1": _f1(prec, rec),
           "accuracy": _ratio(tp + tn, int(m.sum())), "miss_rate": _ratio(fn, tp + fn),
           "tol_px": float(tol_px)}
    out.update(_stats(np.linalg.norm(p - t, axis=1)[both]))
    return out


def track_metrics(track: Any, true_xy: np.ndarray, true_vis: np.ndarray | None = None,
                  tol_px: float = 10.0, jump_factor: float = 3.0,
                  mask: np.ndarray | None = None) -> dict[str, float]:
    """detection_metrics on a Track2D (filled frames count as predictions) plus:

    jitter_px     RMS 2nd difference of the error (pred − truth) over runs of 3
                  consecutive frames on the right object (error ≤ jump_factor·tol) —
                  noise independent of true motion; wrong frames count as jumps
    accel_rms_px  RMS 2nd difference of the predicted track itself
    jumps         number of maximal runs where the track sits on something else
                  (error > jump_factor·tol, or a prediction where truth is absent)
    wrong_frames  frames inside those runs
    interp_*      accuracy of gap-filled frames (if the track marks them)"""
    p = _as_xy(track)
    t = np.asarray(true_xy, float).reshape(-1, 2)
    out = detection_metrics(p, t, true_vis, tol_px, mask=mask)
    tv = _truth_vis(t, true_vis)
    pv = np.all(np.isfinite(p), axis=1)
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)
    e = p - t
    with np.errstate(invalid="ignore"):
        ok = pv & tv & m & (np.linalg.norm(e, axis=1) <= jump_factor * tol_px)
    run3 = ok[:-2] & ok[1:-1] & ok[2:]
    dd = e[2:] - 2 * e[1:-1] + e[:-2]
    out["jitter_px"] = float(np.sqrt(np.mean(np.sum(dd[run3] ** 2, axis=1)))) if run3.any() else float("nan")
    okp = pv[:-2] & pv[1:-1] & pv[2:] & m[1:-1]
    da = p[2:] - 2 * p[1:-1] + p[:-2]
    out["accel_rms_px"] = float(np.sqrt(np.mean(np.sum(da[okp] ** 2, axis=1)))) if okp.any() else float("nan")
    with np.errstate(invalid="ignore"):
        wrong = pv & m & (~tv | (np.linalg.norm(e, axis=1) > jump_factor * tol_px))
    starts = wrong & ~np.concatenate([[False], wrong[:-1]])
    out["jumps"] = int(starts.sum())
    out["wrong_frames"] = int(wrong.sum())
    interp = np.asarray(_get(track, "interpolated", np.zeros(len(p), bool)), bool) \
        if hasattr(track, "interpolated") else np.zeros(len(p), bool)
    if interp.any():
        sub = interp & pv & tv & m
        d = np.linalg.norm(e, axis=1)
        out["interp_frames"] = int(interp.sum())
        out["interp_within_tol"] = _ratio(int(np.sum(d[sub] <= tol_px)), int(sub.sum()))
        out.update(_stats(d[sub], "interp_err"))
    return out


# ── hits ─────────────────────────────────────────────────────────────────────

def _match_events(pf: np.ndarray, tf: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """One-to-one matching minimising |Δframe| (greedy on sorted distances)."""
    if not len(pf) or not len(tf):
        return []
    d = np.abs(pf[:, None] - tf[None, :]).astype(float)
    pairs = np.argwhere(d <= tol)
    order = np.argsort(d[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_p, used_t, out = set(), set(), []
    for k in order:
        i, j = pairs[k]
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        out.append((int(i), int(j)))
    return out


def hit_metrics(pred_hits: Sequence[Any], true_hits: Sequence[Any], tol_frames: int = 3
                ) -> dict[str, float]:
    """Event F1 at ±tol_frames, mean |Δframe| of matches, hitter side accuracy
    (over matches; 'unknown' counts as wrong — hitter_acc_known excludes it)."""
    pf = np.array([int(_get(h, "frame")) for h in pred_hits], int)
    tf = np.array([int(_get(h, "frame")) for h in true_hits], int)
    pairs = _match_events(pf, tf, tol_frames)
    tp = len(pairs)
    prec, rec = _ratio(tp, len(pf)), _ratio(tp, len(tf))
    out = {"tp": tp, "fp": len(pf) - tp, "fn": len(tf) - tp, "n_pred": len(pf), "n_true": len(tf),
           "precision": prec, "recall": rec, "f1": _f1(prec, rec), "tol_frames": tol_frames}
    off = np.array([pf[i] - tf[j] for i, j in pairs], float)
    out["offset_abs_mean"] = float(np.mean(np.abs(off))) if tp else float("nan")
    out["offset_mean"] = float(np.mean(off)) if tp else float("nan")
    ph = [str(_get(pred_hits[i], "hitter", "unknown")) for i, _ in pairs]
    th = [str(_get(true_hits[j], "hitter", "unknown")) for _, j in pairs]
    ok = [a == b for a, b in zip(ph, th)]
    known = [a == b for a, b in zip(ph, th) if a != "unknown"]
    out["hitter_acc"] = float(np.mean(ok)) if ok else float("nan")
    out["hitter_acc_known"] = float(np.mean(known)) if known else float("nan")
    return out


# ── 3-D ──────────────────────────────────────────────────────────────────────

def landing_metrics(pred_xy: Sequence[Any], true_xy: Sequence[Any]) -> dict[str, float]:
    """Euclidean floor error (m) for paired landing points (None/NaN skipped)."""
    err = []
    for p, t in zip(pred_xy, true_xy):
        if p is None or t is None:
            continue
        p, t = np.asarray(p, float)[:2], np.asarray(t, float)[:2]
        if np.all(np.isfinite(p)) and np.all(np.isfinite(t)):
            err.append(float(np.linalg.norm(p - t)))
    return _stats(np.array(err), "landing_err_m")


def confusion_matrix(pred: Sequence[str], true: Sequence[str], labels: Sequence[str] | None = None
                     ) -> dict[str, Any]:
    """{'labels': [...], 'matrix': rows = truth, cols = prediction}."""
    labels = list(labels) if labels is not None else sorted(set(true) | set(pred))
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for p, t in zip(pred, true):
        if p in idx and t in idx:
            M[idx[t], idx[p]] += 1
    return {"labels": labels, "matrix": M.tolist()}


def in_out_metrics(pred_calls: Sequence[str], true_calls: Sequence[str],
                   true_margins: Sequence[float | None] | None = None, close_m: float = 0.2
                   ) -> dict[str, Any]:
    """Call accuracy ('in'|'out'|'net'; anything else = 'unknown' = wrong),
    confusion matrix, and accuracy on close calls (|true margin| < close_m)."""
    pred = [p if p in ("in", "out", "net") else "unknown" for p in pred_calls]
    true = list(true_calls)
    ok = np.array([p == t for p, t in zip(pred, true)], bool)
    out: dict[str, Any] = {"n": len(true), "accuracy": float(ok.mean()) if len(ok) else float("nan"),
                           "confusion": confusion_matrix(pred, true, ["in", "out", "net", "unknown"])}
    decided = np.array([p != "unknown" for p in pred], bool)
    out["accuracy_decided"] = float(ok[decided].mean()) if decided.any() else float("nan")
    out["coverage"] = float(decided.mean()) if len(decided) else float("nan")
    if true_margins is not None:
        mg = np.array([np.nan if m is None else float(m) for m in true_margins])
        close = np.isfinite(mg) & (np.abs(mg) < close_m)
        out["n_close"] = int(close.sum())
        out["accuracy_close"] = float(ok[close].mean()) if close.any() else float("nan")
    return out


def segment_metrics(pred_segments: Sequence[Any], true_shots: Sequence[Any], tol_frames: int = 4
                    ) -> dict[str, float]:
    """Match predicted ShotSegment3D to true shots (SimShot / ShotSegment3D) by
    start frame; errors on initial speed (km/h), contact height (m), contact
    position (m), net-crossing height (m) and landing point (both present)."""
    pf = np.array([int(_get(s, "start_frame", _get(s, "frame", -1))) for s in pred_segments], int)
    tf = np.array([int(_get(s, "start_frame", _get(s, "frame", -1))) for s in true_shots], int)
    pairs = _match_events(pf, tf, tol_frames)
    out: dict[str, float] = {"n_pred": len(pf), "n_true": len(tf), "matched": len(pairs),
                             "recall": _ratio(len(pairs), len(tf))}
    sp, sp_rel, hz, pos, znet, land = [], [], [], [], [], []
    for i, j in pairs:
        ps, ts = pred_segments[i], true_shots[j]
        pv, tv = np.asarray(_get(ps, "v0"), float), np.asarray(_get(ts, "v0"), float)
        pp, tp = np.asarray(_get(ps, "p0"), float), np.asarray(_get(ts, "p0"), float)
        ts_kmh = np.linalg.norm(tv) * 3.6
        sp.append(abs(np.linalg.norm(pv) * 3.6 - ts_kmh))
        sp_rel.append(sp[-1] / max(ts_kmh, 1e-6))
        hz.append(abs(pp[2] - tp[2]))
        pos.append(np.linalg.norm(pp - tp))
        a, b = _get(ps, "net_crossing_height"), _get(ts, "net_crossing_height")
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            znet.append(abs(a - b))
        a, b = _get(ps, "landing_xy"), _get(ts, "landing_xy")
        if a is not None and b is not None and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            land.append(np.linalg.norm(np.subtract(a, b)))
    out.update(_stats(np.array(sp), "speed_err_kmh"))
    out.update(_stats(np.array(sp_rel), "speed_err_rel"))
    out.update(_stats(np.array(hz), "height_err_m"))
    out.update(_stats(np.array(pos), "contact_err_m"))
    out.update(_stats(np.array(znet), "net_height_err_m"))
    out.update(_stats(np.array(land), "landing_err_m"))
    return out


# ── shot type / rally ────────────────────────────────────────────────────────

def shot_type_metrics(pred_types: Sequence[str], true_types: Sequence[str],
                      labels: Sequence[str] | None = None) -> dict[str, Any]:
    ok = [p == t for p, t in zip(pred_types, true_types)]
    cmx = confusion_matrix(list(pred_types), list(true_types), labels)
    M = np.array(cmx["matrix"], float)
    per = {}
    for k, l in enumerate(cmx["labels"]):
        n = M[k].sum()
        if n:
            per[l] = float(M[k, k] / n)
    return {"n": len(ok), "accuracy": float(np.mean(ok)) if ok else float("nan"),
            "per_class_recall": per, "confusion": cmx}


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return max(inter, 0) / union if union > 0 else 0.0


def rally_metrics(pred_rallies: Sequence[Any], true_rallies: Sequence[Any], iou_thresh: float = 0.5
                  ) -> dict[str, Any]:
    """Rally segmentation F1 at interval IoU ≥ iou_thresh (greedy by IoU), mean
    IoU of matches, start/end frame offsets, end-reason and winner accuracy
    over matched rallies, the end-reason confusion matrix, landing error of
    matched rallies and end-reason accuracy on close calls (|true margin| <
    0.2 m; truth margin from `margin` or `in_out_margin_m`)."""
    P = [(int(_get(r, "start_frame")), int(_get(r, "end_frame"))) for r in pred_rallies]
    T = [(int(_get(r, "start_frame")), int(_get(r, "end_frame"))) for r in true_rallies]
    cand = sorted(((_interval_iou(p, t), i, j) for i, p in enumerate(P) for j, t in enumerate(T)),
                  reverse=True)
    up, ut, pairs = set(), set(), []
    for iou, i, j in cand:
        if iou < iou_thresh:
            break
        if i in up or j in ut:
            continue
        up.add(i)
        ut.add(j)
        pairs.append((i, j, iou))
    tp = len(pairs)
    prec, rec = _ratio(tp, len(P)), _ratio(tp, len(T))
    out: dict[str, Any] = {"n_pred": len(P), "n_true": len(T), "tp": tp, "precision": prec,
                           "recall": rec, "f1": _f1(prec, rec)}
    out["iou_mean"] = float(np.mean([p[2] for p in pairs])) if pairs else float("nan")
    out["start_offset_abs_mean"] = float(np.mean([abs(P[i][0] - T[j][0]) for i, j, _ in pairs])) if pairs else float("nan")
    out["end_offset_abs_mean"] = float(np.mean([abs(P[i][1] - T[j][1]) for i, j, _ in pairs])) if pairs else float("nan")
    pe = [str(_get(pred_rallies[i], "end_reason", "unknown")) for i, _, _ in pairs]
    te = [str(_get(true_rallies[j], "end_reason", "unknown")) for _, j, _ in pairs]
    pw = [str(_get(pred_rallies[i], "winner", "unknown")) for i, _, _ in pairs]
    tw = [str(_get(true_rallies[j], "winner", "unknown")) for _, j, _ in pairs]
    out["end_reason_acc"] = float(np.mean([a == b for a, b in zip(pe, te)])) if pairs else float("nan")
    out["winner_acc"] = float(np.mean([a == b for a, b in zip(pw, tw)])) if pairs else float("nan")
    out["end_reason_confusion"] = confusion_matrix(pe, te, ["in", "out", "net", "unknown"])
    land_p = [_get(pred_rallies[i], "landing_xy") for i, _, _ in pairs]
    land_t = [_get(true_rallies[j], "landing_xy") for _, j, _ in pairs]
    out.update(landing_metrics(land_p, land_t))
    mg = []
    for _, j, _ in pairs:
        m = _get(true_rallies[j], "margin", _get(true_rallies[j], "in_out_margin_m"))
        mg.append(np.nan if m is None else float(m))
    close = np.isfinite(mg) & (np.abs(np.array(mg, float)) < 0.2) if pairs else np.zeros(0, bool)
    ok = np.array([a == b for a, b in zip(pe, te)], bool)
    out["n_close"] = int(close.sum())
    out["accuracy_close"] = float(ok[close].mean()) if close.any() else float("nan")
    return out


# ── reporting ────────────────────────────────────────────────────────────────

_TABLE_KEYS = ("f1", "precision", "recall", "accuracy", "err_median", "err_p95", "jitter_px",
               "jumps", "offset_abs_mean", "hitter_acc", "speed_err_kmh_median",
               "height_err_m_median", "landing_err_m_median", "accuracy_close",
               "end_reason_acc", "winner_acc", "iou_mean", "n_frames", "n_true", "n")


def format_table(results: dict[str, Any], keys: Sequence[str] = _TABLE_KEYS) -> str:
    """Compact text table: one row per stage (nested dicts), selected metrics."""
    rows = []
    for stage, res in results.items():
        if not isinstance(res, dict):
            rows.append(f"{stage:<22} {res}")
            continue
        if "skipped" in res:
            rows.append(f"{stage:<22} SKIPPED ({res['skipped']})")
            continue
        cells = []
        for k in keys:
            v = res.get(k)
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                cells.append(f"{k}={int(v)}")
            elif isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append(f"{k}={v:.3f}")
        rows.append(f"{stage:<22} " + "  ".join(cells))
    return "\n".join(rows)
