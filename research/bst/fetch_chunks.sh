#!/bin/bash
# Pull every file of a remote directory with N parallel scp streams (the
# Seoul relay throttles each TCP connection to ~100 KB/s, so parallelism is
# what makes the transfer usable). Retries until local sizes match remote.
# usage: fetch_chunks.sh <remote_dir> <local_dir> [parallel]
set -u
REMOTE_DIR="$1"; LOCAL_DIR="$2"; P="${3:-8}"
KEY="$HOME/.ssh/shuttl_deploy"; HOST=root@43.108.9.91
mkdir -p "$LOCAL_DIR"; cd "$LOCAL_DIR" || exit 1
for round in 1 2 3 4 5 6 7 8; do
  ssh -i "$KEY" "$HOST" "cd $REMOTE_DIR && stat -c '%n %s' *" > .remote_sizes || { sleep 5; continue; }
  todo=$(while read -r n s; do
    l=$(stat -f %z "$n" 2>/dev/null || echo -1)
    [ "$l" = "$s" ] || echo "$n"
  done < .remote_sizes)
  [ -z "$todo" ] && { echo "all $(wc -l < .remote_sizes) files complete"; exit 0; }
  echo "round $round: $(echo "$todo" | wc -l) files to fetch"
  echo "$todo" | xargs -P "$P" -S 4096 -I{} sh -c "scp -q -o ConnectTimeout=20 -i '$KEY' '$HOST:$REMOTE_DIR/{}' '{}.tmp' && mv '{}.tmp' '{}' || { rm -f '{}.tmp'; sleep 2; }"
done
echo "INCOMPLETE after retries"; exit 1
