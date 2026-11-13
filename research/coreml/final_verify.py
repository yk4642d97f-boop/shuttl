"""FINAL verification of the SHIPPED model.

RTMPoseM.mlpackage now has the normalisation FOLDED IN, so it takes raw 0..255
RGB NCHW. This suite checks:

  1. raw-input correctness -- feed raw crops, compare against the rtmlib
     reference pipeline fed with (rgb-mean)/std. This is the contract the Swift
     side will use.
  2. end-to-end pose error against rtmlib gold truth (/tmp/rtm/001_test.npz).
  3. fp32 vs fp16, every compute unit.
  4. timing and model size.

Run: ./venv/bin/python final_verify.py
"""
import glob
import json
import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
SPLIT = 2.0
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(3, 1, 1)

import onnxruntime as ort
import coremltools as ct

sess = ort.InferenceSession(os.path.join(HERE, "RTMPoseM.patched.onnx"),
                            providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name

UNITS = [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
         ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
         ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
         ("ALL", ct.ComputeUnit.ALL)]


def raw_input(bgr):
    """What Swift feeds: raw 0..255 RGB, NCHW."""
    r = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)


def ref_normalised(x_raw):
    """The ONNX expects normalised input; apply rtmlib's maths to the raw tensor."""
    return np.ascontiguousarray((x_raw - MEAN) / STD).astype(np.float32)


def cases():
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))
    real = [(os.path.basename(f), raw_input(cv2.imread(f)))
            for f in files if cv2.imread(f) is not None]
    rng = np.random.default_rng(12345)
    rand = []
    for i in range(24):
        x = (rng.random((1, 3, H, W), dtype=np.float32) * 255.0).astype(np.float32)
        rand.append((f"rand{i}", x))
    return real, rand


def evaluate(ml, cases_, ref):
    worst = 0.0
    nflips = 0
    nkp = 0
    nans = 0
    px_max = 0.0
    flips = []
    for nm, xr in cases_:
        rx, ry = ref(xr)
        o = ml.predict({"image": xr})
        gx, gy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
        nans += int(np.isnan(gx).sum() + np.isnan(gy).sum())
        for lbl, r, g in (("x", rx, gx), ("y", ry, gy)):
            worst = max(worst, float(np.abs(r.astype(np.float64)
                                            - g.astype(np.float64)).max()))
            a, b = np.argmax(r, -1), np.argmax(g, -1)
            d = a != b
            nkp += int(a.size)
            nflips += int(d.sum())
            for i in np.argwhere(d.ravel()).ravel():
                s = np.sort(r, axis=-1)
                mrg = (s[..., -1] - s[..., -2]).ravel()
                flips.append((nm, lbl, int(i),
                              abs(int(b.ravel()[i]) - int(a.ravel()[i])),
                              float(mrg[i])))
        kx, ky = np.argmax(gx, -1) / SPLIT, np.argmax(gy, -1) / SPLIT
        rkx, rky = np.argmax(rx, -1) / SPLIT, np.argmax(ry, -1) / SPLIT
        px_max = max(px_max, float(np.hypot(kx - rkx, ky - rky).max()))
    return dict(worst=worst, nflips=nflips, nkp=nkp, nans=nans, px_max=px_max,
                flips=flips)


def bench(ml, x, n=60):
    for _ in range(8):
        ml.predict({"image": x})
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            ml.predict({"image": x})
        dt = time.perf_counter() - t0
        ips = n / dt
        best = ips if best is None else max(best, ips)
    return best


def goldtruth_check(ml):
    """End-to-end: reproduce rtmlib's own keypoints from the video."""
    import zipfile
    p = "/tmp/rtm/001_test.npz"
    v = "/path/to/shuttl/research/shuttleset/videos/001.mp4"
    if not (os.path.exists(p) and os.path.exists(v)):
        return None
    import goldtruth as G
    z = zipfile.ZipFile(p)
    d = {n[:-4]: np.load(z.open(n)) for n in z.namelist()}
    frames, boxes, gold = d["frame"], d["box"], d["kp"]
    n = min(30, len(frames))
    cap = cv2.VideoCapture(v)
    want = set(int(f) for f in frames[:n])
    imgs, idx = {}, 0
    while len(imgs) < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            imgs[idx] = fr
        idx += 1
    cap.release()
    errs = []
    for i in range(n):
        fid = int(frames[i])
        if fid not in imgs:
            continue
        fr = imgs[fid]
        hh, ww = fr.shape[:2]
        for pl in range(2):
            b = boxes[i, pl]
            if not np.isfinite(b).all():
                continue
            x1, y1, x2, y2 = b
            mw, mh = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
            pad = np.array([max(x1 - mw, 0), max(y1 - mh, 0),
                            min(x2 + mw, ww - 1), min(y2 + mh, hh - 1)], np.float32)
            cx, sc = G.bbox_xyxy2cs(pad, padding=1.25)
            warped, sc2 = G.affine_one(fr, cx, sc)
            # SHIPPED model wants RAW 0..255 RGB (normalisation is inside)
            xr = np.ascontiguousarray(
                warped[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None])
            o = ml.predict({"image": xr})
            sx, sy = np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])
            locs, vals = G.get_simcc_maximum(sx, sy)
            kp = locs[0] / np.array([W, H]) * sc2 + cx[0] - sc2 / 2
            g = gold[i, pl, :, :2]
            ok = np.isfinite(g).all(-1) & (g >= 0).all(-1)
            if ok.any():
                errs.append(np.linalg.norm(kp[ok] - g[ok], axis=-1))
    if not errs:
        return None
    e = np.concatenate(errs)
    return dict(mean=float(e.mean()), median=float(np.median(e)),
                p90=float(np.percentile(e, 90)),
                frac2=float((e < 2).mean()), frac5=float((e < 5).mean()),
                n=int(e.size))


def main():
    real, rand = cases()
    print(f"cases: {len(real)} real crops, {len(rand)} random (raw 0..255 RGB in)\n")
    ref = lambda xr: sess.run(None, {INAME: ref_normalised(xr)})

    paths = [("float32 (shipped)", "RTMPoseM.mlpackage"),
             ("float32 (raw-input ref)", "RTMPoseM_rawinput.mlpackage"),
             ("float16", "RTMPoseM_fp16.mlpackage")]
    out = {}
    for pn, pf in paths:
        p = os.path.join(HERE, pf)
        if not os.path.exists(p):
            continue
        for un, cu in UNITS:
            try:
                ml = ct.models.MLModel(p, compute_units=cu)
            except Exception as e:
                print(f"{pn}/{un}: LOAD FAIL {str(e)[:60]}")
                continue
            for cn, cs in (("real", real), ("rand", rand)):
                # RTMPoseM_rawinput.mlpackage does NOT normalise internally, so
                # it must be fed the normalised tensor; the shipped model takes
                # raw 0..255. Each is compared against the ONNX on the SAME
                # effectively-applied input (ref() normalises what it receives).
                if "rawinput" in pf:
                    csx = [(nm, ref_normalised(x)) for nm, x in cs]
                    refn = lambda x_norm: sess.run(None, {INAME: x_norm})
                else:
                    csx = cs
                    refn = ref
                r = evaluate(ml, csx, refn)
                key = f"{pn}/{un}/{cn}"
                ok = (r["worst"] < 1e-2 and r["nflips"] == 0 and r["nans"] == 0)
                print(f"{key:36s} maxdiff={r['worst']:10.6f} "
                      f"agree={100*(1-r['nflips']/max(r['nkp'],1)):7.3f}% "
                      f"flips={r['nflips']:3d}/{r['nkp']:4d} nan={r['nans']:6d} "
                      f"px_max={r['px_max']:6.3f} -> {'PASS' if ok else 'FAIL'}")
                for f in r["flips"][:3]:
                    print(f"      flip {f[0]} simcc_{f[1]} kp={f[2]} binΔ={f[3]} "
                          f"margin={f[4]:.2e}")
                out[key] = {k: v for k, v in r.items() if k != "flips"}

    print("\n=== timing (best of 3 x 60, one crop) ===")
    x = raw_input(cv2.imread(sorted(glob.glob(os.path.join(HERE, 'real_crops', '*.png')))[0]))
    tim = {}
    for pn, pf in paths:
        p = os.path.join(HERE, pf)
        if not os.path.exists(p):
            continue
        for un, cu in UNITS:
            try:
                ml = ct.models.MLModel(p, compute_units=cu)
            except Exception:
                continue
            ips = bench(ml, x)
            tim[f"{pn}/{un}"] = dict(ips=ips, ms=1000 / ips, ms2=2000 / ips)
            print(f"  {pn:26s} {un:12s} {ips:7.1f} inf/s  {1000/ips:7.2f} ms  "
                  f"2 crops {2000/ips:7.2f} ms")
    out["timing"] = tim

    print("\n=== end-to-end vs rtmlib gold truth (shipped model) ===")
    ml = ct.models.MLModel(os.path.join(HERE, "RTMPoseM.mlpackage"),
                           compute_units=ct.ComputeUnit.ALL)
    g = goldtruth_check(ml)
    if g:
        print(f"  mean {g['mean']:.3f} px  median {g['median']:.3f}  p90 {g['p90']:.3f}  "
              f"<2px {g['frac2']*100:.1f}%  <5px {g['frac5']*100:.1f}%  (n={g['n']})")
        out["goldtruth"] = g

    print("\n=== size ===")
    for pn, pf in paths:
        p = os.path.join(HERE, pf)
        if os.path.exists(p):
            tot = sum(os.path.getsize(os.path.join(r, f))
                      for r, _, fs in os.walk(p) for f in fs)
            print(f"  {pf}: {tot/1e6:.2f} MB")
            out[f"size_{pf}"] = tot
    json.dump(out, open(os.path.join(HERE, "final_results.json"), "w"), indent=1,
              default=str)
    print("\nwrote final_results.json")


if __name__ == "__main__":
    main()
