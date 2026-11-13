"""Convert BST-CG-AP seq30 (25 merged classes, JnB_bone) to a CoreML mlprogram.

  ../../coreml/venv/bin/python convert_bst.py [--members all|0] [--fp16-too]

--members all : the averaged-logit ensemble of the 10 released checkpoints (the shipped model)
--members 0   : the single checkpoint bst_CG_AP_JnB_bone_merged.pt (for comparison)

Writes BSTStroke.mlpackage (ensemble, fp32) / BSTStroke_single.mlpackage, plus *_fp16 variants.
The traced graph is asserted equal to eager and input-dependent before conversion.
"""
import argparse
import json
import warnings

import numpy as np
import torch
import coremltools as ct

import bstcore as B

warnings.filterwarnings("ignore")
torch.set_num_threads(2)


def example(seed=0):
    g = np.random.default_rng(seed)
    clips = B.raw_clips()
    _, _, j, p, s = clips[seed % len(clips)]
    d, _ = B.model_inputs(j, p, s)
    return {k: torch.from_numpy(v) for k, v in d.items()}


def build(members):
    paths = B.WEIGHTS if members == "all" else [B.WEIGHTS[int(members)]]
    nets = [B.load_net(p) for p in paths]
    eager = B.BSTCoreML(nets).eval()
    static = B.BSTCoreML(nets, static=True).eval()
    # the static spelling must equal the repo forward, on full clips AND on 25-frame
    # (end-padded, masked) clips, which the 30-fps match-039 clips never exercise
    clips = B.raw_clips()
    worst = 0.0
    with torch.no_grad():
        for i in range(0, len(clips), 25):
            _, _, j, p, s = clips[i]
            for n in (len(j), 25, 13):
                d, _ = B.model_inputs(j[:n], p[:n], s[:n])
                a = [torch.from_numpy(d[k]) for k in ("joints", "pos", "shuttle", "valid")]
                e, t = eager(*a), static(*a)
                worst = max(worst, float((e - t).abs().max()))
                assert e.argmax() == t.argmax()
    assert worst < 1e-4, worst
    print(f"static spelling == repo forward: max|diff| {worst:.2e} over {3 * len(range(0, len(clips), 25))} clips")
    return static, paths


def convert(model, paths, out, fp16):
    ex = example(0)
    args = (ex["joints"], ex["pos"], ex["shuttle"], ex["valid"])
    with torch.no_grad():
        traced = torch.jit.trace(model, args, check_trace=False)
        e0, t0 = model(*args), traced(*args)
        assert torch.allclose(e0, t0, atol=1e-5), (e0 - t0).abs().max()
        ex2 = example(7)
        args2 = (ex2["joints"], ex2["pos"], ex2["shuttle"], ex2["valid"])
        assert not torch.allclose(traced(*args2), t0), "traced graph ignores its input"
        # masking must be live too: a 25-frame clip (5 padded frames) must change the output
        v25 = ex2["valid"].clone(); v25[0, 25:] = 0
        j25 = ex2["joints"].clone(); j25[0, 25:] = 0
        p25 = ex2["pos"].clone(); p25[0, 25:] = 0
        s25 = ex2["shuttle"].clone(); s25[0, 25:] = 0
        assert torch.allclose(model(j25, p25, s25, v25), traced(j25, p25, s25, v25), atol=1e-5)
    inputs = [ct.TensorType(name="joints", shape=(1, B.SEQ_LEN, 2, 17, 2), dtype=np.float32),
              ct.TensorType(name="pos", shape=(1, B.SEQ_LEN, 2, 2), dtype=np.float32),
              ct.TensorType(name="shuttle", shape=(1, B.SEQ_LEN, 2), dtype=np.float32),
              ct.TensorType(name="valid", shape=(1, B.SEQ_LEN), dtype=np.float32)]
    m = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                   convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                   compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32)
    m.short_description = ("BST-CG-AP badminton stroke classifier, seq_len 30, 25 merged ShuttleSet classes; "
                           f"mean logits of {len(paths)} released checkpoint(s).")
    m.author = "BST (Chang et al.) released weights; CoreML port for Shuttl"
    meta = {
        "classes_json": json.dumps(B.CLASSES, ensure_ascii=False),
        "n_members": str(len(paths)),
        "members": ",".join(p.split("/")[-1] for p in paths),
        "seq_len": str(B.SEQ_LEN),
        "precision": "fp16" if fp16 else "fp32",
        "input_contract": ("joints (1,30,2,17,2) (kp - box centre)/box diagonal, player 0 = far/Top, 1 = near/Bottom, "
                           "frame all-zero unless both players resolved; pos (1,30,2,2) court 0..1; "
                           "shuttle (1,30,2) x/W,y/H, 0 if unseen; valid (1,30) 1 for the first video_len "
                           "frames; every array zero beyond video_len. Bones are built in-graph."),
        "output": "logits (1,25) raw mean logits; class order = classes_json",
    }
    for k, v in meta.items():
        m.user_defined_metadata[k] = v
    m.input_description["joints"] = "COCO-17 joints, box-normalised, (1,30,2,17,2)"
    m.input_description["pos"] = "ankle-midpoint court position 0..1, (1,30,2,2)"
    m.input_description["shuttle"] = "shuttle x/W, y/H, (1,30,2)"
    m.input_description["valid"] = "1 for real frames, 0 for end padding, (1,30)"
    m.output_description["logits"] = "mean pre-softmax logits over the members, (1,25)"
    m.save(str(B.HERE / out))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="all")
    ap.add_argument("--fp16-too", action="store_true")
    a = ap.parse_args()
    model, paths = build(a.members)
    base = "BSTStroke" if a.members == "all" else f"BSTStroke_single{a.members}"
    convert(model, paths, base + ".mlpackage", fp16=False)
    if a.fp16_too:
        convert(model, paths, base + "_fp16.mlpackage", fp16=True)


if __name__ == "__main__":
    main()
