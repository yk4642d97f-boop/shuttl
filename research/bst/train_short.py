"""Short from-scratch training run of a BST model on Apple MPS, using the repo's own
train_one_epoch / validate functions and the same optimiser, LR schedule and loss as
repo/stroke_classification/main_on_shuttleset/bst_main.py (AdamW lr 5e-4, cosine with
400 warm-up steps over hyp.n_epochs*len(loader) steps, num_cycles 0.25, CE with
label smoothing 0.1, RandomTranslation_batch on joints, batch 128).

Differences from the official script (all deliberate, all logged):
  * device = mps (official: cuda or cpu)
  * stops after --epochs epochs or --budget-min minutes (official: 1600 epochs, early stop 300)
  * the LR schedule length still uses --sched-epochs (default 1600 = official), so the
    LR curve for the epochs we do run is identical to the official one
  * best-on-val (macro-F1) state is tested at the end, like the official script

usage: venv/bin/python train_short.py --data <collated_dir> --seq-len 100 --model BST_CG_AP --epochs 40
"""
import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

REPO = Path(__file__).resolve().parent / 'repo' / 'stroke_classification'
sys.path.insert(0, str(REPO))

import main_on_shuttleset.bst_main as bm  # noqa: E402  (module-level hyp: n_classes=25)
from preparing_data.shuttleset_dataset import (  # noqa: E402
    RandomTranslation_batch, get_bone_pairs, prepare_npy_collated_loaders)
from model.bst import BST, BST_AP, BST_CG, BST_CG_AP  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bst import metrics, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--seq-len', type=int, default=100)
    ap.add_argument('--model', default='BST_CG_AP')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--budget-min', type=float, default=110)
    ap.add_argument('--sched-epochs', type=int, default=1600)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    assert bm.hyp.n_classes == 25 and bm.hyp.batch_size == 128 and bm.hyp.lr == 5e-4

    train_loader, val_loader, test_loader = prepare_npy_collated_loaders(
        root_dir=a.data, pose_style='JnB_bone', batch_size=128, use_cuda=False,
        num_workers=(0, 0, 0), train_partial=1.0)
    print('train/val/test:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    n_bones = len(get_bone_pairs())
    Arch = {'BST': BST, 'BST_CG': BST_CG, 'BST_AP': BST_AP, 'BST_CG_AP': BST_CG_AP}[a.model]
    net = Arch(in_dim=(17 + n_bones) * 2, n_class=25, seq_len=a.seq_len, depth_tem=2, depth_inter=1).to(a.device)
    print(a.model, 'params:', sum(p.numel() for p in net.parameters()))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(net.parameters(), lr=bm.hyp.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=bm.hyp.warm_up_step,
        num_training_steps=a.sched_epochs * len(train_loader), num_cycles=0.25)
    shift = RandomTranslation_batch()

    log, best, best_state = [], -1.0, None
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss = bm.train_one_epoch(net, train_loader, shift, n_bones, loss_fn, optimizer, scheduler, a.device)
        if a.device == 'mps':
            torch.mps.synchronize()
        t1 = time.time()
        va_loss, f1_avg, f1_min = bm.validate(net, loss_fn, val_loader, a.device)
        t2 = time.time()
        rec = dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss, val_macro_f1=float(f1_avg),
                   val_min_f1=float(f1_min), train_s=t1 - t0, val_s=t2 - t1, lr=scheduler.get_last_lr()[0])
        log.append(rec)
        picked = float(f1_avg) > best
        if picked:
            best, best_state = float(f1_avg), deepcopy(net.state_dict())
        print(f'ep {ep:3d} train_loss={tr_loss:.3f} val_loss={va_loss:.3f} val_macroF1={f1_avg:.3f} '
              f'minF1={f1_min:.3f} train {t1 - t0:.1f}s val {t2 - t1:.1f}s{" *" if picked else ""}', flush=True)
        if (time.time() - t_start) / 60 > a.budget_min:
            print('time budget reached'); break

    total = time.time() - t_start
    net.load_state_dict(best_state)
    logits, gt = run(net, test_loader, a.device)
    m = metrics(logits, gt, 25)
    print(f'TEST (best-val state, val macroF1={best:.4f}): acc={m["acc"]:.4f} macroF1={m["macro_f1"]:.4f} '
          f'minF1={m["min_f1"]:.4f} top2={m["top2"]:.4f}; wall {total / 60:.1f} min for {len(log)} epochs')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, a.out.with_suffix('.pt'))
    json.dump({'args': {k: str(v) for k, v in vars(a).items()}, 'log': log, 'test': m,
               'wall_s': total, 'torch': torch.__version__}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
