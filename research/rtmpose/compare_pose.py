"""Draw Apple Vision and RTMPose keypoints on the same frames, side by side per player,
so the pose difference can be seen rather than inferred from numbers.

  python3 compare_pose.py --npz ../extract/out/001.npz --rtm /tmp/rtm/ab --video ../shuttleset/videos/001.mp4 \
      --out /tmp/rtm/fig --frames 11600 11890 12300
"""
import argparse, glob, os
import numpy as np
import cv2

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLOR = {"vision": (0, 165, 255), "rtm": (0, 255, 0)}   # BGR: orange, green


def draw(img, kp, color, label, thick=2):
    for a, b in EDGES:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, thick)
    for j in range(17):
        if kp[j, 2] > 0:
            cv2.circle(img, tuple(kp[j, :2].astype(int)), 3, color, -1)
    if np.isfinite(kp[0, 0]):
        cv2.putText(img, label, tuple(np.array([kp[0, 0], kp[0, 1] - 26]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--rtm", required=True, help="dir with NNN_sh*.npz")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--rtm-frame-offset", type=int, default=0,
                    help="add this to an rtm frame id to map it to a video frame, if the rtm run used a different base")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import zipfile
    z = zipfile.ZipFile(a.npz)
    d = {n[:-4]: np.load(z.open(n), allow_pickle=True) for n in z.namelist()}
    mid = int(os.path.basename(a.npz)[:3])
    rtm_kp, rtm_box = {}, {}
    for s in sorted(glob.glob(os.path.join(a.rtm, f"{mid:03d}_sh*.npz"))):
        zz = np.load(s)
        for i, f in enumerate(zz["frame"]):
            rtm_kp[int(f)] = zz["kp"][i]
            rtm_box[int(f)] = zz["box"][i]
    print(f"vision frames {len(d['frame'])}, rtm frames {len(rtm_kp)}")
    fidx = {int(f): i for i, f in enumerate(d["frame"])}

    cap = cv2.VideoCapture(a.video)
    for f in a.frames:
        if f not in fidx:
            print(f"{f}: not in the extracted span"); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            print(f"{f}: cannot read"); continue
        i = fidx[f]
        right = img.copy()
        names = ["far/Top", "near/Bottom"]
        for p in range(2):
            b = d["player_box"][i, p]
            if np.all(np.isfinite(b)):
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)
                cv2.rectangle(right, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)
            draw(img, d["player_kp"][i, p], COLOR["vision"], f"Vision {names[p]}")
            if f + a.rtm_frame_offset in rtm_kp:
                draw(right, rtm_kp[f + a.rtm_frame_offset][p], COLOR["rtm"], f"RTMPose {names[p]}")
        cv2.putText(img, f"frame {f}  Apple Vision", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(right, f"frame {f}  RTMPose-m", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out = np.concatenate([img, right], axis=0)
        path = os.path.join(a.out, f"pose_{f}.png")
        cv2.imwrite(path, out)
        print("wrote", path)


if __name__ == "__main__":
    main()
