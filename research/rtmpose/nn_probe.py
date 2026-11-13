"""Compare CoreML ModelFormat variants against the CPU path + reference.

The default MLProgram format is fp16-only, which flips simcc argmax on
near-tied peaks (wrist/elbow), producing >1 px outliers.  The legacy
NeuralNetwork format can run fp32, so it may keep CoreML's speed with CPU
numerics.  This measures both, plus CPU, on one window.
"""
import os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_pose import MODEL_BODY7, make_session, run_batch, pad_box
import coreml_probe as cp


def main():
    NF = int(os.environ.get("PARITY_MAXF", "3000"))
    MAXS = int(os.environ.get("PARITY_SLOTS", "1200"))
    ref = cp.load_ref_finite()
    fids_all = sorted({k[0] for k in ref})
    hi = fids_all[min(NF, len(fids_all)) - 1]
    slots_all = sorted((f, p, bx) for (f, p), (k, bx) in ref.items() if f <= hi)
    fids = sorted({s[0] for s in slots_all})
    frames = cp.decode(fids)
    slots = []
    for fid, p, bx in slots_all:
        if fid not in frames or not np.all(np.isfinite(bx)):
            continue
        h, w = frames[fid].shape[:2]
        slots.append((fid, p, pad_box(bx, w, h)))
    slots = slots[:MAXS]
    print(f"{len(slots)} slots {slots[0][0]}..{slots[-1][0]}", flush=True)

    refsub = {k: (v[0], None) for k, v in ref.items()}

    s = make_session(MODEL_BODY7, ["CPUExecutionProvider"], threads=2)
    cpu, _ = cp.run_cfg("cpu_t2_b1", s, slots, frames, False)
    del s

    cfgs = [
        ("coreml_nn_b1", {"MLComputeUnits": "ALL", "ModelFormat": "NeuralNetwork"}, 1, False),
        ("coreml_nn_b2", {"MLComputeUnits": "ALL", "ModelFormat": "NeuralNetwork"}, 2, True),
        ("coreml_mlprog_b2", {"MLComputeUnits": "ALL"}, 2, True),
    ]
    for name, po, bh, bt in cfgs:
        try:
            s = make_session(MODEL_BODY7, cp.CM, provider_options=[po], batch_hint=bh)
            res, fps = cp.run_cfg(name, s, slots, frames, batch=bt)
            del s
            cp.dist(res, cpu, f"{name}-vs-cpu")
            cp.dist(res, refsub, f"{name}-vs-REFERENCE")
        except Exception as e:
            print(f"SPEED {name} FAILED: {type(e).__name__}: {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
