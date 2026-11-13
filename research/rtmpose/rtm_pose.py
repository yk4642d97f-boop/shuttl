"""Run RTMPose-m (the pose model the released ShuttleSet features were made with)
on frames of a ShuttleSet match, using the person boxes our extractor already
produced, so the only variable is the pose estimator.

  python3 rtm_pose.py --npz out/001.npz --video ../shuttleset/videos/001.mp4 \
      --out /tmp/rtm/001_rtm.npz [--max-frames N] [--mode balanced|lightweight]

Output arrays (same frame order as the input npz's `frame`):
  kp   (N,2,17,3)  COCO-17 x, y (full-frame px) and score, NaN when undetected
  box  (N,2,4)     the box actually used
"""
import argparse, os, time, zipfile
import numpy as np

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def load(path):
    z = zipfile.ZipFile(path)
    return {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="balanced", choices=["balanced", "lightweight", "performance"])
    ap.add_argument("--onnx", default=os.path.join(
        os.path.expanduser("~/.cache/rtmlib/hub/checkpoints"),
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx"),
        help="RTMPose ONNX (body7 256x192: the estimator the released ShuttleSet features used)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th block of frames, for parallel runs")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import cv2
    from rtmlib import RTMPose

    d = load(a.npz)
    frame_ids = d["frame"].astype(int)
    boxes = d["player_box"]                      # (N,2,4) x1,y1,x2,y2
    poses = d["player_kp"]                       # (N,2,17,3) Vision, for a fallback
    n = len(frame_ids)
    if a.max_frames:
        n = min(n, a.max_frames)
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    shard_n = max(shard_n, 1)
    # block-wise sharding keeps the video seeks monotonic inside each worker
    block = (n + shard_n - 1) // shard_n
    lo, hi = shard_i * block, min(n, (shard_i + 1) * block)
    print(f"{n} frames total; this shard {shard_i}/{shard_n} covers {lo}..{hi}")
    print(f"{n} frames, boxes present {np.isfinite(boxes[:n,:,0]).mean()*100:.0f}%")

    model = RTMPose(onnx_model=a.onnx, model_input_size=(192, 256),
                    backend="onnxruntime", device=a.device)
    cap = cv2.VideoCapture(a.video)
    kp = np.full((n, 2, 17, 3), np.nan, dtype=np.float32)
    used = np.array(boxes[:n], dtype=np.float32)
    # the extractor's box order is [far/Top, near/Bottom]; RTMPose is run per box, so the
    # output keeps that order.  A padded box (a bit of margin) helps when the box is tight.
    #
    # Decode sequentially instead of seeking per frame: the frame ids are increasing and
    # mostly contiguous (rally spans), and a seek costs far more than a decode.
    want = {int(frame_ids[i]): i for i in range(lo, hi)}
    t0 = time.time()
    if lo < hi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[lo]))
    pos = int(frame_ids[lo]) if lo < hi else 0
    done = 0
    while lo < hi and pos <= int(frame_ids[hi - 1]):
        ok, img = cap.read()
        if not ok:
            break
        i = want.pop(pos, None)
        pos += 1
        if i is None:
            continue
        h, w = img.shape[:2]
        for p in range(2):
            b = boxes[i, p]
            if not np.all(np.isfinite(b)):
                continue
            x1, y1, x2, y2 = b
            mw, mh = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
            pad = np.array([max(x1 - mw, 0), max(y1 - mh, 0), min(x2 + mw, w - 1), min(y2 + mh, h - 1)],
                           dtype=np.float32)
            try:
                k, s = model(img, bboxes=pad[None, :])
            except Exception as e:
                print(f"frame {frame_ids[i]} player {p}: {e}")
                continue
            if len(k) == 0:
                continue
            kp[i, p, :, :2] = k[0][:, :2]
            kp[i, p, :, 2] = s[0]
        done += 1
        if done % 200 == 0:
            dt = time.time() - t0
            left = len(want)
            print(f"  {done}/{hi-lo}  {done/dt:.1f} frames/s  eta {left/max(done/dt,1e-9)/60:.1f} min", flush=True)
    dt = time.time() - t0
    print(f"done: {done} frames in {dt:.0f}s = {done/max(dt,1e-9):.1f} frames/s ({2*done/max(dt,1e-9):.1f} poses/s)")
    covered = np.isfinite(kp[:n, :, 0, 0]).mean()
    print(f"poses returned for {covered*100:.0f}% of player slots (Vision had "
          f"{np.isfinite(poses[:n,:,0,0]).mean()*100:.0f}%)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, kp=kp, box=used, frame=frame_ids[:n])


if __name__ == "__main__":
    main()
