#!/bin/bash
# End-to-end table for one match: labelled-hit controls vs OUR detected hits,
# seq100 variable window vs seq30 fixed window.   usage: e2e/run_match.sh 39 [shift]
set -euo pipefail
cd "$(dirname "$0")"
M=$1; S=${2:-2}; MM=$(printf %03d "$M")
# HITS=<dir> TAG=<suffix> let the same table be recomputed against another hit detector
HITS=${HITS:-../hits/out}
TAG=${TAG:-}
PY=../bst/venv/bin/python
SPLIT=train; case $M in 39|40|42|43|44) SPLIT=test;; 41) SPLIT=val;; esac
W100="../bst/weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt"
W30="../bst/weights/bst_CG_AP_JnB_bone_merged*.pt"
run() {  # name arm seq shift
  local out="m${MM}_$1$TAG"
  rm -rf "$out" "${out}_collated"
  $PY -u e2e_convert.py --match "$M" --arm "$2" --shift "$4" --rtm ../rtmpose/pose --hits "$HITS" \
      --out "$out" --stats "stats_${out}.json" | tail -2
  $PY -u ../bst/collate.py "$out" "${out}_collated" "$3" "$SPLIT" > /dev/null
  $PY -u ../bst/eval_bst.py --data "${out}_collated" --seq-len "$3" --set "$SPLIT" \
      --device mps --out "eval_${out}.json" --weights $( [ "$3" = 100 ] && echo $W100 || echo $W30 ) \
      2>&1 | grep -E '^MEAN' | sed "s/^/$1 ($3): /"
}
if [ -z "$TAG" ]; then run ctrl100 control 100 0; run ctrl30 fixed_ctrl 30 0; fi
run det100     detected   100 "$S"
run ctrl30     fixed_ctrl 30  0
run fixed30    fixed      30  "$S"
run fixed30_s0 fixed      30  0
