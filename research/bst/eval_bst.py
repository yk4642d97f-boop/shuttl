"""Evaluate released BST weights on the ShuttleSet (merged, 25 classes) test split.

Mirrors the metric code of repo/stroke_classification/main_on_shuttleset/bst_main.py:
  - accuracy = mean(argmax == gt)
  - macro-F1 = torcheval multiclass_f1_score(average=None).mean()   (show_f1_results)
  - min-F1   = min of the per-class F1
  - top-2    = gt in torch.topk(logits, 2)
Additionally dumps per-class F1 / precision / recall / support to JSON.

usage:
  venv/bin/python eval_bst.py --data <collated_dir> --seq-len 100 --device mps \
      --weights weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged*.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

REPO = Path(__file__).resolve().parent / 'repo' / 'stroke_classification'
sys.path.insert(0, str(REPO))

from preparing_data.shuttleset_dataset import (  # noqa: E402
    Dataset_npy_collated, get_bone_pairs, get_merged_stroke_types)
from model.bst import BST, BST_AP, BST_CG, BST_CG_AP  # noqa: E402

ENGLISH = {  # paper Table G
    '未知球種': 'none', '放小球': 'net shot', '擋小球': 'return net', '殺球': 'smash',
    '挑球': 'lob', '長球': 'clear', '平球': 'drive', '切球': 'drop', '推球': 'push',
    '撲球': 'rush', '勾球': 'cross-court net shot', '發短球': 'short service', '發長球': 'long service',
}


def english(name: str) -> str:
    if '_' in name:
        side, typ = name.split('_', 1)
        return f'{side} {ENGLISH[typ]}'
    return ENGLISH[name]


def arch_from_name(fname: str):
    if fname.startswith('bst_CG_AP_'):
        return BST_CG_AP
    if fname.startswith('bst_CG_'):
        return BST_CG
    if fname.startswith('bst_AP_'):
        return BST_AP
    if fname.startswith('bst_JnB') or fname.startswith('bst_J_only'):
        return BST
    raise ValueError(f'unknown architecture for {fname}')


@torch.no_grad()
def run(net, loader, device):
    net.eval()
    logits_ls, gt_ls = [], []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose = human_pose.to(device)
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = net(human_pose, shuttle.to(device), pos.to(device), video_len.to(device))
        logits_ls.append(logits.float().cpu())
        gt_ls.append(labels)
    return torch.cat(logits_ls), torch.cat(gt_ls)


def metrics(logits, gt, n_classes):
    pred = logits.argmax(1)
    f1_each = multiclass_f1_score(pred, gt, num_classes=n_classes, average=None)
    top2 = torch.topk(logits, 2, dim=1).indices
    oh_p = F.one_hot(pred, n_classes).bool()
    oh_g = F.one_hot(gt, n_classes).bool()
    tp = (oh_p & oh_g).sum(0).float()
    prec = tp / oh_p.sum(0).clamp(min=1)
    rec = tp / oh_g.sum(0).clamp(min=1)
    return {
        'n': int(len(gt)),
        'acc': float((pred == gt).float().mean()),
        'macro_f1': float(f1_each.mean()),
        'min_f1': float(f1_each.min()),
        'top2': float((top2 == gt[:, None]).any(1).float().mean()),
        'f1_each': f1_each.tolist(),
        'precision_each': prec.tolist(),
        'recall_each': rec.tolist(),
        'support_each': oh_g.sum(0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--set', default='test')
    ap.add_argument('--seq-len', type=int, default=100)
    ap.add_argument('--n-classes', type=int, default=25)
    ap.add_argument('--pose-style', default='JnB_bone')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--weights', nargs='+', required=True, type=Path)
    a = ap.parse_args()

    ds = Dataset_npy_collated(a.data, a.set, a.pose_style)
    loader = DataLoader(ds, batch_size=a.batch_size)
    print(f'{a.set}: {len(ds)} strokes; pose {ds.human_pose.shape} pos {ds.pos.shape} '
          f'shuttle {ds.shuttle.shape} videos_len {ds.videos_len.shape} device={a.device}')

    n_joints = 17
    n_bones = len(get_bone_pairs())
    extra = {'J_only': 0, 'JnB_bone': 1, 'JnB_interp': 1, 'Jn2B': 2}[a.pose_style]
    in_dim = (n_joints + n_bones * extra) * 2
    classes = get_merged_stroke_types()

    results = {}
    for w in sorted(a.weights, key=lambda p: (len(p.stem), p.stem)):
        Arch = arch_from_name(w.name)
        net = Arch(in_dim=in_dim, n_class=a.n_classes, seq_len=a.seq_len, depth_tem=2, depth_inter=1)
        net.load_state_dict(torch.load(str(w), map_location='cpu', weights_only=True))
        net.to(a.device)
        t0 = time.time()
        logits, gt = run(net, loader, a.device)
        dt = time.time() - t0
        m = metrics(logits, gt, a.n_classes)
        m['seconds'] = dt
        m['arch'] = Arch.__name__
        results[w.name] = m
        print(f'{w.name:75s} acc={m["acc"]:.4f} macroF1={m["macro_f1"]:.4f} '
              f'minF1={m["min_f1"]:.4f} top2={m["top2"]:.4f}  ({dt:.1f} s)')

    keys = ['acc', 'macro_f1', 'min_f1', 'top2']
    arr = np.array([[r[k] for k in keys] for r in results.values()])
    print(f'MEAN over {len(results)} weights: ' + '  '.join(f'{k}={v:.4f}' for k, v in zip(keys, arr.mean(0))))
    print(f'STD  over {len(results)} weights: ' + '  '.join(f'{k}={v:.4f}' for k, v in zip(keys, arr.std(0))))

    f1 = np.array([r['f1_each'] for r in results.values()])
    print('\nper-class F1 (mean over weights, min..max):')
    sup = next(iter(results.values()))['support_each']
    for i, c in enumerate(classes):
        print(f'  {i:2d} {english(c):28s} n={sup[i]:4d}  F1={f1[:, i].mean():.3f}  ({f1[:, i].min():.3f}..{f1[:, i].max():.3f})')

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'args': {k: str(v) for k, v in vars(a).items()}, 'classes': classes,
                   'classes_en': [english(c) for c in classes], 'results': results},
                  open(a.out, 'w'), ensure_ascii=False, indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
