#!/usr/bin/env python
"""Score the detector's hit frames against the ShuttleSet labels.

Protocol (T-DEED / F3Set style event detection): a detection matches at most one
labelled hit and vice versa, and a pair may be matched only if their video frames
are within `tol` frames.  Pairing is greedy on |offset|, so each labelled hit
takes its closest still-free detection.  Reported at +-1, +-2 and +-4 frames.

Reads the detector output of `hitseval` (one JSON per match) and the ShuttleSet
set*.csv labels; writes a JSON summary and prints a table.

A detection is scored at `--shift` frames later than the raw detector output: the
detector's matched offset distribution is sharply peaked at -2 (it fires ~2 frames
early relative to the ShuttleSet labels), so a single fixed --shift 2 is applied to
every match, never a per-match shift.

Usage:
  score.py --hits 'out/*.hits.json' --labels-dir <label root>
           [--tols 1 2 4] [--shift 2] [--exclude 009 010]
           [--json out/summary.json] [--examples 20]
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

TOLS = (1, 2, 4)


def load_labels(root, label_dir):
    """[(set, rally, frame, type)] for one match's label directory."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, label_dir, "set*.csv"))):
        key = int(os.path.basename(path)[3:-4])
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    fr = int(round(float(r["frame_num"])))
                    rl = int(float(r["rally"]))
                except (TypeError, ValueError):
                    continue
                out.append((key, rl, fr, r.get("type", "")))
    return out


def pair(gf, d, tol):
    """Greedy nearest pairing within `tol`.

    gf: sorted ground-truth frames; d: detections (list of dicts) sorted by frame.
    Returns (kept, miss_gt, free_det) as index lists.
    """
    cands = []
    for gi, f in enumerate(gf):
        for di, dd in enumerate(d):
            off = dd["frame"] - f
            if abs(off) <= tol:
                cands.append((abs(off), gi, di, off))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    free_gt = set(range(len(gf)))
    free_det = set(range(len(d)))
    kept = []
    for _, gi, di, off in cands:
        if gi in free_gt and di in free_det:
            free_gt.discard(gi)
            free_det.discard(di)
            kept.append((gi, di, off))
    return kept, sorted(free_gt), sorted(free_det)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", nargs="+", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--tols", nargs="+", type=int, default=list(TOLS))
    ap.add_argument("--shift", type=int, default=0,
                    help="add this many frames to every detection before pairing "
                         "(the detector fires ~2 frames early; see module docstring)")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="match ids to drop entirely (e.g. 009 010, whose shuttle "
                         "tracking is broken so they cannot be scored)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--cuts", nargs="+", type=float, default=None)
    args = ap.parse_args()

    files = []
    for pat in args.hits:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        sys.exit("no hit files matched %r" % (args.hits,))
    if args.exclude:
        drop = {e.strip() for e in args.exclude}
        files = [p for p in files
                 if os.path.basename(p).split(".")[0] not in drop]
        if not files:
            sys.exit("every hit file was excluded")

    matches = []
    thr_seen = {}
    for path in files:
        with open(path) as f:
            m = json.load(f)
        gts = load_labels(args.labels_dir, m["labelDir"])
        gt_by = collections.defaultdict(list)
        for s, rl, fr, ty in gts:
            gt_by[(s, rl)].append((fr, ty))
        det_by = collections.defaultdict(list)
        for r in m["rallies"]:
            for h in r["hits"]:
                h = dict(h, set=r["set"], rally=r["rally"], frame=h["frame"] + args.shift)
                det_by[(r["set"], r["rally"])].append(h)
        for k in det_by:
            det_by[k].sort(key=lambda x: x["frame"])
        thr_seen.setdefault(m["threshold"], []).append(m["source"])
        matches.append(dict(path=path, source=m["source"], labelDir=m["labelDir"],
                            threshold=m["threshold"], gt=gt_by, det=det_by))

    # Every match must have been detected at the same operating point, or the
    # pooled number would mix thresholds.  Refuse rather than average silently.
    if len(thr_seen) > 1:
        detail = "; ".join("%g: %d match(es) e.g. %s"
                           % (t, len(v), ", ".join(sorted(v)[:3]))
                           for t, v in sorted(thr_seen.items()))
        sys.exit("hits files disagree on the detector threshold (min_hit_logodds): %s\n"
                 "re-run hitseval with one --threshold for every match" % detail)

    max_tol = max(args.tols)
    summary = dict(matches=len(matches), threshold=matches[0]["threshold"],
                   shift=args.shift,
                   per_match=[], pooled={}, offset_hist={}, sweep=[],
                   misses_by_type={}, fp_offset={}, examples=[],
                   total_gt=0, total_det=0)

    pooled = {t: dict(tp=0, fp=0, fn=0) for t in args.tols}
    off_hist = {t: collections.Counter() for t in args.tols}
    misses = collections.Counter()
    miss_by_len = collections.Counter()      # service / rally-position of the miss
    fp_off = collections.Counter()           # FP offset to the nearest labelled hit
    examples = []

    for mt in matches:
        row = dict(source=mt["source"], labelDir=mt["labelDir"], gt=0, det=0,
                   by_tol={t: dict(tp=0, fp=0, fn=0) for t in args.tols})
        for key in sorted(set(mt["gt"]) | set(mt["det"])):
            g = sorted(mt["gt"].get(key, []))
            gf = [x[0] for x in g]
            types = [x[1] for x in g]
            d = mt["det"].get(key, [])
            kept, miss, free = pair(gf, d, max_tol)
            row["gt"] += len(gf)
            row["det"] += len(d)
            summary["total_gt"] += len(gf)
            summary["total_det"] += len(d)
            for t in args.tols:
                k = [x for x in kept if abs(x[2]) <= t]
                for gi, di, off in k:
                    off_hist[t][off] += 1
                # A detection/GT pair kept at max_tol but outside +-t is an FP and
                # an FN: both sides are scored against the *+-t* matching, NOT
                # against the max_tol pairing (using len(k) here would credit the
                # far-apart pairs and understate both errors).
                ntp = len(k)
                s = row["by_tol"][t]
                s["tp"] += ntp
                s["fp"] += len(d) - ntp
                s["fn"] += len(gf) - ntp
                pooled[t]["tp"] += ntp
                pooled[t]["fp"] += len(d) - ntp
                pooled[t]["fn"] += len(gf) - ntp
            # misses: which stroke, and where in the rally
            for gi in miss:
                misses[types[gi] or "(unlabelled)"] += 1
                miss_by_len["serve" if gi == 0 else "rally"] += 1
                if args.examples and len(examples) < args.examples:
                    examples.append(dict(kind="MISS", match=mt["source"], set=key[0],
                                         rally=key[1], gt_frame=gf[gi], stroke=types[gi],
                                         nth_in_rally=gi, rally_hits=len(gf)))
            # false positives: how near the next labelled hit
            for di in free:
                dd = d[di]
                near = min((dd["frame"] - f for f in gf), key=abs) if gf else None
                bucket = "none" if near is None else ("%+d" % near if abs(near) <= 12 else "far")
                fp_off[bucket] += 1
                if args.examples and len(examples) < args.examples:
                    examples.append(dict(kind="FP", match=mt["source"], set=key[0],
                                         rally=key[1], det_frame=dd["frame"],
                                         conf=round(dd["confidence"], 3),
                                         nearest_gt_offset=near, src=dd["source"]))
        for t in args.tols:
            s = row["by_tol"][t]
            p, r, f = prf(s["tp"], s["fp"], s["fn"])
            s.update(P=round(p, 4), R=round(r, 4), F1=round(f, 4))
        # what fixed shift would have been best for this match alone (+-2)?
        # (reported as a diagnostic of spread; we never tune per match)
        best, bestf1 = args.shift, -1.0
        for sh in range(-4, 7):
            tp = fp = fn = 0
            for key in sorted(set(mt["gt"]) | set(mt["det"])):
                gf = [x[0] for x in sorted(mt["gt"].get(key, []))]
                d = [dict(x, frame=x["frame"] - args.shift + sh)
                     for x in mt["det"].get(key, [])]
                d.sort(key=lambda x: x["frame"])
                kept, _, _ = pair(gf, d, 2)
                tp += len(kept)
                fn += len(gf) - len(kept)
                fp += len(d) - len(kept)
            f1 = prf(tp, fp, fn)[2]
            if f1 > bestf1:
                best, bestf1 = sh, f1
        row["best_shift"] = best
        summary["per_match"].append(row)

    f1s = [row["by_tol"][2]["F1"] for row in summary["per_match"]]
    summary["f1_min"] = round(min(f1s), 4)
    summary["f1_max"] = round(max(f1s), 4)

    for t in args.tols:
        p, r, f = prf(pooled[t]["tp"], pooled[t]["fp"], pooled[t]["fn"])
        summary["pooled"][t] = dict(tp=pooled[t]["tp"], fp=pooled[t]["fp"], fn=pooled[t]["fn"],
                                    P=round(p, 4), R=round(r, 4), F1=round(f, 4))
        summary["offset_hist"][t] = dict(sorted(off_hist[t].items()))
    summary["misses_by_type"] = dict(misses.most_common())
    summary["fp_offset"] = dict(fp_off.most_common())
    summary["miss_serve_split"] = dict(miss_by_len)

    # ---- confidence / threshold sweep (+-2), recomputed at each cut
    cuts = args.cuts or [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    sweep = []
    for cut in cuts:
        tp = fp = fn = 0
        for mt in matches:
            for key in sorted(set(mt["gt"]) | set(mt["det"])):
                gf = [x[0] for x in sorted(mt["gt"].get(key, []))]
                d = [x for x in mt["det"].get(key, []) if x["confidence"] >= cut]
                kept, _, _ = pair(gf, d, 2)
                tp += len(kept)
                fn += len(gf) - len(kept)
                fp += len(d) - len(kept)
        p, r, f = prf(tp, fp, fn)
        sweep.append(dict(cut=cut, tp=tp, fp=fp, fn=fn,
                          P=round(p, 4), R=round(r, 4), F1=round(f, 4),
                          logodds=(round(math.log(cut / (1 - cut)), 3) if 0 < cut < 1 else None)))
    summary["sweep"] = sweep

    # ---- per-match confidence split: how well does `confidence` separate TP/FP
    tp_conf, fp_conf = [], []
    for mt in matches:
        for key in sorted(set(mt["gt"]) | set(mt["det"])):
            gf = [x[0] for x in sorted(mt["gt"].get(key, []))]
            d = mt["det"].get(key, [])
            kept, _, free = pair(gf, d, 2)
            tp_conf += [d[di]["confidence"] for _, di, _ in kept]
            fp_conf += [d[di]["confidence"] for di in free]
    def q(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return dict(n=len(xs), mean=round(sum(xs) / len(xs), 4),
                    p10=round(xs[int(0.1 * (len(xs) - 1))], 4),
                    p50=round(xs[len(xs) // 2], 4),
                    p90=round(xs[int(0.9 * (len(xs) - 1))], 4))
    summary["confidence"] = dict(tp=q(tp_conf), fp=q(fp_conf))

    if args.examples:
        summary["examples"] = examples

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1, default=str)

    # ---- print
    print("matches: %d   min_hit_logodds: %s   detection shift: %+d frames"
          % (summary["matches"], summary["threshold"], summary["shift"]))
    print("detections: %d   labelled hits: %d" % (summary["total_det"], summary["total_gt"]))
    print()
    print("POOLED       TP     FP     FN       P       R      F1")
    for t in args.tols:
        s = summary["pooled"][t]
        print("  +-%d frames %5d  %5d  %5d  %.4f  %.4f  %.4f"
              % (t, s["tp"], s["fp"], s["fn"], s["P"], s["R"], s["F1"]))
    print()
    print("PER MATCH (+-2)      GT   DET      P       R      F1")
    for row in summary["per_match"]:
        s = row["by_tol"][2]
        print("  %-11s %5d %5d  %.4f  %.4f  %.4f"
              % (row["source"][:11], row["gt"], row["det"], s["P"], s["R"], s["F1"]))
    print()
    print("PER MATCH best fixed shift (pooled F1 over shifts -4..+6, +-2):")
    print("  " + "  ".join("%s:%+d" % (row["source"][:11], row["best_shift"])
                           for row in summary["per_match"]))
    print("  spread of F1@2 across matches (shift %+d): min %.4f  max %.4f"
          % (args.shift, summary["f1_min"], summary["f1_max"]))
    print()
    print("OFFSET HISTOGRAM (+-%d):" % max_tol)
    print("  " + "  ".join("%+d:%d" % kv for kv in summary["offset_hist"][max(args.tols)].items()))
    print("  (+-2 only): " + "  ".join("%+d:%d" % kv for kv in summary["offset_hist"][2].items()))
    print()
    print("CONFIDENCE  TP: %s" % summary["confidence"]["tp"])
    print("CONFIDENCE  FP: %s" % summary["confidence"]["fp"])
    print()
    best = max(sweep, key=lambda s: s["F1"])
    print("THRESHOLD SWEEP (+-2)   cut   logodds    TP    FP    FN      P      R     F1")
    for s in sweep:
        lo = "%.2f" % s["logodds"] if s["logodds"] is not None else "   -inf"
        print("  %8.3f  %7s  %5d %5d %5d  %.4f %.4f %.4f%s"
              % (s["cut"], lo, s["tp"], s["fp"], s["fn"], s["P"], s["R"], s["F1"],
                 "  <- best" if s is best else ""))
    print()
    print("MISSES BY STROKE: " + ", ".join("%s:%d" % kv for kv in summary["misses_by_type"].items()))
    print("MISSES serve/rally: " + ", ".join("%s:%d" % kv for kv in summary["miss_serve_split"].items()))
    print("FP offset to nearest label: " + ", ".join("%s:%d" % kv for kv in summary["fp_offset"].items()))
    if summary.get("examples"):
        print()
        print("EXAMPLES:")
        for e in summary["examples"]:
            print("  " + json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
