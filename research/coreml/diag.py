"""Isolate the real-crop parity failure. Stages:

  A. ONNX graph introspection (input contract, normalisation fingerprint, node mix)
  B. onnx2torch eager   vs onnxruntime   (suspect c: mis-conversion)
  C. torch.jit.trace    vs onnxruntime   (trace baking / branch)
  D. CoreML per compute unit vs onnxruntime (suspect a: fp16 compute, b: branch)

Prints a compact table. Run: ./venv/bin/python diag.py
"""
import glob
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 256, 192
ONNX = os.path.join(HERE, "RTMPoseM.patched.onnx")
N_REAL = 24
N_RAND = 8


# ---------------------------------------------------------------- stage A
def stage_a():
    import onnx
    from onnx import numpy_helper

    m = onnx.load(ONNX)
    g = m.graph
    ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
           for i in g.input]
    outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
            for o in g.output]
    print(f"[A] nodes={len(g.node)}  inputs={ins}  outputs={outs}")

    from collections import Counter
    ops = Counter(n.op_type for n in g.node)
    print("[A] op histogram:", dict(ops.most_common(20)))

    # ImageNet normalisation fingerprint: 1/std or mean values present as consts
    inits = {}
    for t in g.initializer:
        inits[t.name] = numpy_helper.to_array(t)
    fp = {"mean_123.675": 123.675, "std_58.395": 58.395,
          "inv_std_0.017125": 1 / 58.395, "inv_std_0.017507": 1 / 57.12}
    for k, v in fp.items():
        for name, arr in inits.items():
            if arr.size and arr.dtype.kind == "f" and np.isclose(arr, v, atol=1e-4).any():
                print(f"[A] normalisation fingerprint {k} -> initializer {name} "
                      f"shape {arr.shape}")
                break
        else:
            print(f"[A] normalisation fingerprint {k}: NOT FOUND")

    print("[A] first 14 nodes:")
    for n in g.node[:14]:
        print(f"      {n.op_type:12s} in={list(n.input)[:4]} out={list(n.output)[:2]}")

    # Any data-dependent control flow?
    for n in g.node:
        if n.op_type in ("If", "Loop", "Scan", "NonZero", "Where", "TopK",
                         "Unique", "Compress", "NonMaxSuppression"):
            print(f"[A] DATA-DEPENDENT op: {n.op_type} ({n.name})")
    print("[A] done")


# ---------------------------------------------------------------- helpers
def cases(n_real=N_REAL, n_rand=N_RAND):
    files = sorted(glob.glob(os.path.join(HERE, "real_crops", "*.png")))[:n_real]
    real = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        r = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = r[:, :, ::-1]
        real.append((os.path.basename(f),
                     np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)))
    rng = np.random.default_rng(12345)
    rand = [(f"rand{i}", (rng.random((1, 3, H, W), dtype=np.float32) * 255).astype(np.float32))
            for i in range(n_rand)]
    return real, rand


def cmp_pair(name, ga, gb, tag, worst):
    """ga = candidate outputs, gb = reference outputs. Returns (maxdiff, all_agree)."""
    ok = True
    for lbl, a, b in zip(("simcc_x", "simcc_y"), ga, gb):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            print(f"  !! {tag}/{name} {lbl} SHAPE {a.shape} vs {b.shape}")
            ok = False
            continue
        d = float(np.abs(a - b).max())
        ag = (np.argmax(a, -1) == np.argmax(b, -1)).mean()
        if ag < 1.0:
            bad = np.argwhere(np.argmax(a, -1) != np.argmax(b, -1))
            ok = False
            if worst[1] is None or d > worst[0]:
                worst[0], worst[1] = d, f"{tag}/{name} {lbl} agree={ag*100:.1f}% "\
                                           f"kp={bad[0].tolist()}"
        if not ok and d >= 0:
            pass
    return ok


def summarize(tag, cases_, cand_fn, ref_fn, worstbox):
    ok_all = True
    for nm, x in cases_:
        ok = cmp_pair(nm, cand_fn(x), ref_fn(x), tag, worstbox)
        ok_all = ok_all and ok
    if not ok_all:
        print(f"  {tag}: FAIL  worst -> {worstbox[1]}")
    else:
        print(f"  {tag}: PASS")
    return ok_all


def main():
    stage_a()
    real, rand = cases()
    print(f"[cases] {len(real)} real crops, {len(rand)} random")

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ref = lambda x: sess.run(None, {iname: np.ascontiguousarray(x, dtype=np.float32)})

    # ---- sanity: what does the ONNX give on raw vs normalised input? -------
    xm = real[0][1]
    raw = ref(xm)
    norm = (xm / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]) \
        / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
    nrm = ref(norm.astype(np.float32))
    print(f"[probe] raw0..255  simcc_x max={raw[0].max():.4f} min={raw[0].min():.4f} "
          f"ptp={np.ptp(raw[0]):.4f}  argmax_unique={len(set(np.argmax(raw[0],-1).ravel()))}")
    print(f"[probe] normalised simcc_x max={nrm[0].max():.4f} min={nrm[0].min():.4f} "
          f"ptp={np.ptp(nrm[0]):.4f}  argmax_unique={len(set(np.argmax(nrm[0],-1).ravel()))}")

    # ---- stage B/C: onnx2torch -------------------------------------------
    import torch
    import onnx
    from onnx2torch import convert as o2t

    print("[B] building onnx2torch model ...")
    net = o2t(onnx.load(ONNX)).eval()
    xt = torch.from_numpy(real[0][1])

    def eager(x):
        with torch.no_grad():
            o = net(torch.from_numpy(x))
        o = o if isinstance(o, (list, tuple)) else (o,)
        o = [np.asarray(v) for v in o]
        return [o[0], o[1]] if o[0].shape[-1] == 384 else [o[1], o[0]]

    traced = torch.jit.trace(net, xt, check_trace=False)

    def trac(x):
        with torch.no_grad():
            o = traced(torch.from_numpy(x))
        o = o if isinstance(o, (list, tuple)) else (o,)
        o = [np.asarray(v) for v in o]
        return [o[0], o[1]] if o[0].shape[-1] == 384 else [o[1], o[0]]

    print("\n=== B/C: onnx2torch vs onnxruntime ===")
    wb = [0.0, None]
    summarize("eager", real, eager, ref, wb)
    wb = [0.0, None]
    summarize("traced", real, trac, ref, wb)

    # ---- stage D: CoreML per compute unit --------------------------------
    import coremltools as ct
    path = os.path.join(HERE, "RTMPoseM.mlpackage")
    units = [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
             ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
             ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
             ("ALL", ct.ComputeUnit.ALL)]
    print("\n=== D: CoreML RTMPoseM.mlpackage per compute unit ===")
    for lbl, cu in units:
        try:
            ml = ct.models.MLModel(path, compute_units=cu)
        except Exception as e:
            print(f"  {lbl}: load failed {e}")
            continue

        def cand(x, ml=ml):
            o = ml.predict({"image": x})
            return [np.asarray(o["simcc_x"]), np.asarray(o["simcc_y"])]

        wb = [0.0, None]
        a = summarize(f"{lbl}/real", real, cand, ref, wb)
        wb = [0.0, None]
        b = summarize(f"{lbl}/rand", rand, cand, ref, wb)
        # also direct vs traced torch, to separate coremltools from onnx2torch
        wb = [0.0, None]
        summarize(f"{lbl}/real_vs_traced", real, cand, trac, wb)


if __name__ == "__main__":
    main()
