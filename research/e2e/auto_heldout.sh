#!/bin/bash
# For each held-out/val match: once extract + RTMPose pose exist, make sure the hit
# detector has run, then build the end-to-end table.  Results -> e2e/heldout.log
cd "$(dirname "$0")/.."
LOG=e2e/heldout.log
for M in 40 42 41 43 44; do
  MM=$(printf %03d $M)
  [ -f "e2e/eval_m${MM}_fixed30.json" ] && continue
  until [ -f "extract/out/$MM.npz" ] && [ -f "rtmpose/pose/$MM.npz" ]; do sleep 120; done
  [ -f "hits/out/$MM.hits.json" ] || hits/run.sh > /dev/null 2>&1
  echo "=== $MM  $(date '+%m-%d %H:%M') ===" >> $LOG
  e2e/run_match.sh $M 2 2>&1 | grep -E '^(ctrl|det|fixed)' >> $LOG
done
echo "=== all done $(date '+%m-%d %H:%M') ===" >> $LOG
