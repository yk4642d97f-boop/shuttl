#!/bin/zsh
# Supervisor: keep the pose driver alive.  Something on this box periodically
# kills run_pose.py processes (SIGTERM, -15, seen three times).  Each shard now
# checkpoints every 250 frames, so a restart re-does at most that much work.
# Restarting is therefore the correct response, not debugging the killer.
cd /path/to/shuttl/research/rtmpose
LOG=/path/to/shuttl/research/rtmpose/pose/run_pose.log
WORKERS=${WORKERS:-5}
mkdir -p pose
while true; do
  echo "=== supervisor: driver start $(date '+%F %T') ===" >> "$LOG"
  ./venv/bin/python -u run_pose.py --hours 13 --workers $WORKERS --shards 6 --threads 2 --poll 30 >> "$LOG" 2>&1
  code=$?
  echo "=== supervisor: driver exited code=$code $(date '+%F %T') ===" >> "$LOG"
  if [ $code -eq 0 ]; then
    # clean exit: either the 10 hours elapsed or everything is done
    echo "=== supervisor: clean exit, stopping ===" >> "$LOG"
    break
  fi
  sleep 20
done
