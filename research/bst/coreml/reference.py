"""PyTorch reference on the 749 match-039 clips (research/e2e/m039_ctrl30).

1. Proves the raw-clip -> model-input path used for CoreML (bstcore.model_inputs + the in-graph
   bones) reproduces the repo's collated arrays (m039_ctrl30_collated) bit for bit.
2. Runs every checkpoint through the repo forward (batch of clips, CPU, fp32) and through the
   BSTCoreML wrapper (one clip at a time), checks they agree, and stores the logits.
3. Reports per-checkpoint accuracy (must reproduce eval_m039_ctrl30.json), the averaged-logit
   ensemble, and the side-free stroke-type accuracy under two decision rules.

usage:  ../../coreml/venv/bin/python reference.py
writes: reference_logits.npz, reference.json
"""
import json
import time

import numpy as np
import torch

import bstcore as B

torch.set_num_threads(2)          # the machine is shared with a long CPU batch job


def type_of(ci):
    """class index -> side-free type index (0 = unknown, 1..12)."""
    return 0 if ci == 0 else (ci - 1) % 12 + 1


def main():
    clips = B.raw_clips()
    names = [c[0] for c in clips]
    labels = np.array([c[1] for c in clips])
    col = B.M039_COLLATED / "test"
    c_jnb, c_pos, c_sh = np.load(col / "JnB_bone.npy"), np.load(col / "pos.npy"), np.load(col / "shuttle.npy")
    c_len, c_lab = np.load(col / "videos_len.npy"), np.load(col / "labels.npy")
    assert len(clips) == len(c_lab), (len(clips), len(c_lab))
    assert np.array_equal(labels, c_lab), "clip order / labels differ from the collated arrays"

    inputs, lens = [], []
    for i, (_, _, j, p, s) in enumerate(clips):
        d, n = B.model_inputs(j, p, s)
        inputs.append(d)
        lens.append(n)
        assert n == c_len[i]
        assert np.array_equal(B.reference_jnb(d["joints"][0]), c_jnb[i]), f"JnB differs at {i}"
        assert np.array_equal(d["pos"][0], c_pos[i]) and np.array_equal(d["shuttle"][0], c_sh[i])
    print(f"{len(clips)} clips: model inputs reproduce the collated JnB_bone/pos/shuttle/videos_len exactly; "
          f"video_len values {sorted(set(lens))}")

    X = {k: torch.from_numpy(np.concatenate([d[k] for d in inputs])) for k in inputs[0]}
    jnb_t = torch.from_numpy(c_jnb).reshape(len(clips), B.SEQ_LEN, 2, -1)
    per = []
    t0 = time.time()
    with torch.no_grad():
        for w in B.WEIGHTS:
            net = B.load_net(w)
            ref = net(jnb_t, torch.from_numpy(c_sh), torch.from_numpy(c_pos), torch.from_numpy(c_len)).numpy()
            wrap = B.BSTCoreML([net]).eval()
            mine = np.concatenate([wrap(X["joints"][i:i + 1], X["pos"][i:i + 1], X["shuttle"][i:i + 1],
                                        X["valid"][i:i + 1]).numpy() for i in range(len(clips))])
            d = np.abs(ref - mine).max()
            assert d < 1e-4 and (ref.argmax(1) == mine.argmax(1)).all(), (w, d)
            per.append(ref)
            print(f"  {w.split('/')[-1]:36s} acc={np.mean(ref.argmax(1) == labels):.4f}  wrapper max|diff|={d:.2e}")
    per = np.stack(per)                               # (10, N, 25)
    print(f"torch forward: {time.time() - t0:.1f} s")

    accs = (per.argmax(2) == labels[None]).mean(1)
    ens = per.mean(0)
    ens_acc = float(np.mean(ens.argmax(1) == labels))
    # side-free stroke type: (a) type of the 25-way argmax, (b) argmax of the side-marginal probs
    lt = np.array([type_of(c) for c in labels])
    prob = torch.softmax(torch.from_numpy(ens), 1).numpy()
    marg = np.zeros((len(clips), 13))
    for c in range(B.N_CLASS):
        marg[:, type_of(c)] += prob[:, c]
    type_a = float(np.mean(np.array([type_of(c) for c in ens.argmax(1)]) == lt))
    type_b = float(np.mean(marg.argmax(1) == lt))
    side_ok = float(np.mean(((ens.argmax(1) - 1) // 12) == ((labels - 1) // 12)))
    single_type = [float(np.mean(np.array([type_of(c) for c in per[k].argmax(1)]) == lt)) for k in range(len(per))]

    rep = {
        "n_clips": len(clips),
        "weights": [w.split("/")[-1] for w in B.WEIGHTS],
        "acc_per_checkpoint": accs.tolist(),
        "acc_mean_over_checkpoints": float(accs.mean()),
        "acc_std_over_checkpoints": float(accs.std()),
        "acc_ensemble_mean_logits": ens_acc,
        "type_acc_ensemble_25argmax": type_a,
        "type_acc_ensemble_side_marginal": type_b,
        "type_acc_per_checkpoint_25argmax": single_type,
        "side_acc_ensemble": side_ok,
    }
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)}, indent=1))
    np.savez_compressed(B.HERE / "reference_logits.npz", per=per.astype(np.float32), labels=labels,
                        names=np.array(names))
    json.dump(rep, open(B.HERE / "reference.json", "w"), indent=1)


if __name__ == "__main__":
    main()
