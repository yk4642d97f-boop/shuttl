#!/bin/bash
# End-to-end with the LEARNED hit detector (hitnet) instead of the rule detector.
cd "$(dirname "$0")/.."
LOG=e2e/learned.log
: > $LOG
for M in 39 40 42 43 44 41; do
  MM=$(printf %03d $M)
  echo "=== $MM $(date '+%H:%M') ===" >> $LOG
  HITS=../hitnet/out/ens3_windowed TAG=_net e2e/run_match.sh $M 0 2>&1 | grep -E '^(fixed30|det100)' >> $LOG
done
echo "=== done $(date '+%H:%M') ===" >> $LOG
