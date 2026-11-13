#!/bin/bash
# Run the whole hit-eval pipeline over every extracted match.
#
#   research/hits/run.sh [--threshold LOGODDS] [--shift N] [--wait] [--rebuild]
#
# npz -> JSON (dump.py, python venv) -> detect hits (hitseval, Swift) -> score.py
#
# The threshold defaults to the app's own HitConfig().minHitLogodds (-4): runs at
# -1000000000 (accept every Viterbi contact) are NOT a valid operating point
# because no decoded hit ever scores below -4 anyway, so the two are identical in
# count -- but recording -4 keeps the reported `threshold` honest.
#
# --wait loops until no new .npz appears for PASSES consecutive sweeps, so more
# matches arriving during an extract run get picked up.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../bst/venv/bin/python"
OUT="$HERE/out"
SRC="$HERE/../extract/out"
LABELS="$HERE/../shuttleset/labels/ShuttleSet/set"
EXTRA=()
SCORE_EXTRA=()
WAIT=0
PASSES=3
while [ $# -gt 0 ]; do
  case "$1" in
    --wait) WAIT=1; shift ;;
    --rebuild) swift build -c release --package-path "$HERE"; shift ;;
    --threshold) EXTRA+=(--threshold "$2"); shift 2 ;;
    --shift) SCORE_EXTRA+=(--shift "$2"); shift 2 ;;
    --exclude) SCORE_EXTRA+=(--exclude "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
# 009 and 010 keep their hits.json but are excluded from the scoreable pool:
# their shuttle track is too sparse to score at all (see REPORT.md).
: "${EXTRA:=--threshold -4}"

mkdir -p "$OUT"

run_once() {
  local n=0
  for npz in "$SRC"/*.npz; do
    [ -e "$npz" ] || continue
    local base ts jh
    base="$(basename "$npz" .npz)"
    ts="$OUT/$base.json"
    jh="$OUT/$base.hits.json"
    n=$((n + 1))
    if [ ! -f "$ts" ] || [ "$npz" -nt "$ts" ]; then
      "$PY" "$HERE/dump.py" "$npz" "$ts" || { echo "dump failed: $npz" >&2; rm -f "$ts"; continue; }
    fi
    # Always re-detect: the detector is ~2 s/match and touching it is what
    # guarantees the numbers track the current code.
    "$HERE/.build/release/hitseval" "$ts" "$jh" "${EXTRA[@]+"${EXTRA[@]}"}"
  done
  echo "$n"
}

if [ "$WAIT" = "1" ]; then
  stable=0
  last=0
  while [ "$stable" -lt "$PASSES" ]; do
    cur=$(run_once)
    if [ "$cur" = "$last" ]; then stable=$((stable + 1)); else stable=0; fi
    last="$cur"
    [ "$stable" -lt "$PASSES" ] && sleep 30
  done
  echo "no new matches for $PASSES sweeps; $last matches"
else
  run_once >/dev/null
fi

"$PY" "$HERE/score.py" --hits "$OUT/*.hits.json" --labels-dir "$LABELS" \
  --json "$OUT/summary.json"
