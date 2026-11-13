#!/usr/bin/env python
"""Train the hit-frame detector.  Train = every extracted match except 039-044;
early stopping / model selection on val 041 only (F1@+-2 with a fixed default decoder).

  train.py --name tcn_s0 [--arch tcn|gru] [--hidden 96] [--epochs 60] [--seed 0]
           [--no-stretch] [--exclude 009 010]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stretch", action="store_true")
    ap.add_argument("--side-w", type=float, default=0.3)
    ap.add_argument("--pos-w", type=float, default=4.0)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--eval-every", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ids = [i for i in C.scan_ids() if i not in C.HELD_OUT and i not in set(args.exclude)]
    train_r = []
    used = []
    for i in ids:
        m = C.load_match(i)
        if m is None:
            continue
        used.append(i)
        train_r += m["rallies"]
    val = C.load_match("041")
    val_r = val["rallies"]
    print("train matches (%d): %s" % (len(used), " ".join(used)))
    print("train rallies %d, val rallies %d, features %d" % (len(train_r), len(val_r), C.n_features()))

    model = C.make_model(args.arch, C.n_features(), args.hidden).to(dev)
    print("params", C.n_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * (args.samples // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ds = C.WindowDS(train_r, args.samples, sigma=args.sigma, seed=args.seed,
                    use_stretch=not args.no_stretch)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=2,
                                     drop_last=True, persistent_workers=False)

    default_dec = dict(thr=0.4, gap_s=0.45, lam=0.0)

    def val_eval():
        model.eval()
        dets = []
        for r in val_r:
            lg = C.predict_rally([model], r["raw"], r["fps"], dev)
            dets.append(C.detections_from_logits(lg, r["fps"], r["frames"], default_dec))
        res, srv, rly = C.eval_dets(val_r, dets)
        return C.f1_of(res, 2), res

    best = -1
    log = []
    ck = os.path.join(C.HERE, "ckpt", args.name + ".pt")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for X, y, sd, sm, valid in dl:
            X, y, sd, sm, valid = (t.to(dev) for t in (X, y, sd, sm, valid))
            out = model(X)
            w = (1 + args.pos_w * y) * valid
            lh = (F.binary_cross_entropy_with_logits(out[..., 0], y, reduction="none") * w).sum() / w.sum()
            ls = (F.binary_cross_entropy_with_logits(out[..., 1], sd, reduction="none") * sm).sum() / sm.sum().clamp(min=1)
            loss = lh + args.side_w * ls
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            n += 1
        msg = "ep %3d loss %.4f  %.0fs" % (ep, tot / n, time.time() - t0)
        if ep % args.eval_every == 0 or ep == args.epochs:
            f1, res = val_eval()
            msg += "  val F1@2 %.4f (@1 %.4f @4 %.4f)" % (f1, C.f1_of(res, 1), C.f1_of(res, 4))
            log.append((ep, tot / n, f1))
            if f1 > best:
                best = f1
                torch.save(dict(state=model.state_dict(), arch=args.arch, hidden=args.hidden,
                                n_in=C.n_features(), epoch=ep, val_f1=f1, args=vars(args),
                                train_matches=used), ck)
                msg += "  *"
        print(msg, flush=True)
    print("best val F1@2 (default decoder) %.4f -> %s" % (best, ck))
    with open(os.path.join(C.HERE, "ckpt", args.name + ".log.json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, best=best, train_matches=used), f)


if __name__ == "__main__":
    main()
