#!/usr/bin/env python
"""Per-match stroke types for the labelled hits (for the error analysis).

Reads the ShuttleSet set*.csv files and writes a small JSON:

  { "set1": { "r1": [[frame, type, player], ...] }, ... }

score.py merges this with the detector output so misses and false positives can
be broken down by stroke type (serve vs fast exchange) without carrying the full
label tables around.

Usage: labels.py <label dir> <out.json>
"""
import csv
import glob
import json
import os
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    out = {}
    for path in sorted(glob.glob(os.path.join(src, "set*.csv"))):
        key = os.path.basename(path)[:-4]        # "set1"
        rows = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    fr = int(round(float(r["frame_num"])))
                    rally = int(float(r["rally"]))
                except (TypeError, ValueError):
                    continue
                rows.append({"set": int(key[3:]), "rally": rally, "frame": fr,
                             "type": r.get("type", ""), "player": r.get("player", "")})
        out[key] = rows
    with open(dst, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("%s -> %s (%d hits)" % (src, dst, sum(len(v) for v in out.values())))


if __name__ == "__main__":
    main()
