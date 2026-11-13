"""Fine-tune a BST model on OUR extracted features, starting from a released checkpoint.

train_short.py trains from scratch; this only adds `--init` (load a released state dict
before the first epoch) and otherwise reuses exactly the same loop, optimiser, schedule
and augmentation, so the two are directly comparable.

usage:
  venv/bin/python finetune.py --init weights/bst_CG_AP_..._merged.pt \
      --data /tmp/bst_split_coll --epochs 12 --budget-min 40 --out /tmp/bst_ft/r1
"""
import argparse, os, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "repo" / "stroke_classification"))

import main_on_shuttleset.bst_main as bm  # noqa: E402
from preparing_data.shuttleset_dataset import (  # noqa: E402
    RandomTranslation_batch, get_bone_pairs, prepare_npy_collated_loaders)
from model.bst import BST, BST_AP, BST_CG, BST_CG_AP  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402
from eval_bst import metrics as eval_metrics, run as eval_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path, nargs="+")
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--model", default="BST_CG_AP")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--budget-min", type=float, default=40)
    ap.add_argument("--lr", type=float, default=None, help="default: the released 5e-4")
    ap.add_argument("--sched-epochs", type=int, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    train_loader, val_loader, test_loader = prepare_npy_collated_loaders(
        root_dir=a.data, pose_style="JnB_bone", batch_size=128, use_cuda=False,
        num_workers=(0, 0, 0), train_partial=1.0)
    print(f"train {len(train_loader.dataset)}  val {len(val_loader.dataset)}  test {len(test_loader.dataset)}")

    n_bones = len(get_bone_pairs())
    Arch = {"BST": BST, "BST_CG": BST_CG, "BST_AP": BST_AP, "BST_CG_AP": BST_CG_AP}[a.model]
    net = Arch(in_dim=(17 + n_bones) * 2, n_class=25, seq_len=a.seq_len, depth_tem=2, depth_inter=1).to(a.device)

    # initialise from the mean of the released checkpoints (their own recipe averages 10)
    states = [torch.load(str(w), map_location="cpu", weights_only=True) for w in a.init]
    if len(states) > 1:
        avg = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    else:
        avg = states[0]
    missing = net.load_state_dict(avg, strict=False)
    print("loaded released weights; missing:", list(missing.missing_keys),
          "unexpected:", list(missing.unexpected_keys))

    lr = a.lr or bm.hyp.lr
    epochs = a.epochs
    sched_epochs = a.sched_epochs or epochs
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(net.parameters(), lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(bm.hyp.warm_up_step, 20),
                                           num_training_steps=sched_epochs * max(len(train_loader), 1),
                                           num_cycles=0.25)
    shift = RandomTranslation_batch()
    log, best, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        loss = bm.train_one_epoch(net, train_loader, shift, n_bones, loss_fn, opt, sched, a.device)
        if a.device == "mps":
            torch.mps.synchronize()
        m = eval_metrics(*eval_run(net, val_loader, a.device), 25)
        dt = time.time() - te
        print(f"ep {ep:2d}  train loss {loss:.4f}  val acc {m['acc']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"top2 {m['top2']:.4f}  ({dt:.0f}s)", flush=True)
        log.append({"epoch": ep, "loss": loss, **m, "seconds": dt})
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], deepcopy(net.state_dict())
        if time.time() - t0 > a.budget_min * 60:
            print("budget reached")
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    test = eval_metrics(*eval_run(net, test_loader if len(test_loader.dataset) else val_loader, a.device), 25)
    print(f"\nBEST val macroF1 {best:.4f}; test acc {test['acc']:.4f} macroF1 {test['macro_f1']:.4f} top2 {test['top2']:.4f}")
    torch.save(net.state_dict(), a.out / "bst_ft.pt")
    import json
    json.dump({"log": log, "best_val_macro_f1": best, "test": {k: v for k, v in test.items() if not isinstance(v, list)}},
              open(a.out / "train.json", "w"), indent=1)
    print("wrote", a.out / "bst_ft.pt")


if __name__ == "__main__":
    main()
