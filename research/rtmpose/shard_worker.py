"""One pose shard, as its own process.

Deliberately NOT named run_pose.py and launched with this filename: a stray
`pkill -f run_pose.py` from another session then cannot match these children
(that is exactly how a previous run lost every shard to SIGTERM).

Usage (the driver builds this line):
    python shard_worker.py <extract npz> <video mp4> <shard i> <shards n> <threads>
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_pose import worker, match_id, POSE


def main():
    npz, video, si, sn, th = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = os.path.join(POSE, ".shards", f"{match_id(npz)}.sh{si}.npz")
    sys.exit(worker(npz, video, out, (si, sn), th))


if __name__ == "__main__":
    main()
