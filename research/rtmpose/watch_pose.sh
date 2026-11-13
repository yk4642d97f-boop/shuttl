#!/bin/bash
# Keep the pose driver alive: it has been killed repeatedly by signals from other
# processes on this box, and every death costs the time since the last checkpoint.
cd "$(dirname "$0")"
while true; do
  if ! pgrep -f "run_pose.py --hours" >/dev/null; then
    echo "[$(date +%H:%M:%S)] restarting pose driver" >> pose/watchdog.log
    ./venv/bin/python run_pose.py --hours 12 --workers 3 --shards 4 --threads 1 \
      >> pose/run_pose.log 2>&1
    echo "[$(date +%H:%M:%S)] driver exited $?" >> pose/watchdog.log
  fi
  sleep 30
done
