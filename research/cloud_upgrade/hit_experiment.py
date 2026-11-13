"""Reproducible HitNet expansion experiment; never selects on the test partition.

Run from any directory with research/bst/venv/bin/python. Outputs remain in this folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
HIT = HERE.parent / "hitnet"
sys.path.insert(0, str(HIT))
import common as C
import infer

VAL = ["033", "041"]
AUDIT = ["036", "037"]
TEST = ["039", "040", "042", "043", "044"]
DECODER = {"thr": 0.4, "gap_s": 0.30, "lam": 0.0}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audit():
    rows = []
    for mid in C.scan_ids():
        m = C.load_match(mid)
        if m is None:
            raise RuntimeError(f"Unreadable match {mid}; do not silently change the experiment")
        rallies = m["rallies"]
        for r in rallies:
            assert len(r["raw"]) == len(r["frames"]) > 0
            assert np.all(np.diff(r["frames"]) == 1), (mid, "non-contiguous rally")
            assert np.isfinite(C.derive(r["raw"], r["fps"])).all()
            assert np.all((r["hits"] >= 0) & (r["hits"] < len(r["raw"]))), mid
        p = Path(C.EXTRACT) / (mid + ".npz")
        labels = sorted((Path(C.LABELS_ROOT) / m["label_dir"]).glob("set*.csv"))
        rows.append({"id": mid, "partition": "val" if mid in VAL else "audit" if mid in AUDIT
                     else "test" if mid in TEST else "train", "fps": m["fps"],
                     "rallies": len(rallies), "frames": sum(len(r["raw"]) for r in rallies),
                     "hits": sum(len(r["hits"]) for r in rallies), "bytes": p.stat().st_size,
                     "sha256": sha(p), "labels": {p.name: sha(p) for p in labels}})
        print(mid, rows[-1]["partition"], rows[-1]["rallies"], flush=True)
    assert set(VAL + AUDIT + TEST) <= {r["id"] for r in rows}
    manifest = {"version": 1, "created": "2026-10-07", "decoder": DECODER,
                "scope": "annotated singles rally spans; not full-match/doubles validation",
                "matches": rows}
    write_json(HERE / "data_manifest.json", manifest)
    print("PARTITIONS", {p: [r["id"] for r in rows if r["partition"] == p]
                          for p in ("train", "val", "audit", "test")}, flush=True)


def load_partition(name):
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    ids = [r["id"] for r in manifest["matches"] if r["partition"] == name]
    rallies = []
    for mid in ids:
        match = C.load_match(mid)
        if match is None:
            raise RuntimeError(mid)
        rallies += match["rallies"]
    return ids, rallies


class Windows(torch.utils.data.Dataset):
    """Same features/augmentation as v1; epoch RNG and optional first-hit oversampling."""
    def __init__(self, rallies, samples, seed, serve_fraction=0.0, win=384):
        self.rallies, self.samples, self.seed = rallies, samples, seed
        self.serve_fraction, self.win, self.epoch = serve_fraction, win, 0
        lengths = np.array([len(r["raw"]) for r in rallies], dtype=float)
        self.prob = lengths / lengths.sum()

    def __len__(self):
        return self.samples

    def __getitem__(self, i):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
        serve = rng.random() < self.serve_fraction
        ri = rng.integers(len(self.rallies)) if serve else rng.choice(len(self.rallies), p=self.prob)
        r = self.rallies[ri]
        raw, hits, fps = r["raw"].copy(), r["hits"].copy(), r["fps"]
        if rng.random() < 0.5:
            raw = C.mirror(raw)
        if rng.random() < 0.8:
            scale = float(np.exp(rng.uniform(np.log(0.8), np.log(1.3))))
            raw, hits = C.stretch(raw, hits, scale)
            fps *= scale
        raw = C.drop_shuttle(raw, rng)
        visible = np.isfinite(raw[:, 0])
        raw[visible, 0:2] += rng.normal(0, 0.0007, (visible.sum(), 2))
        x = C.derive(raw, fps)
        y, side, mask = C.make_targets(len(raw), hits, r["side"], 1.0)
        if len(raw) >= self.win:
            if serve and len(hits):
                # Random location within the crop avoids learning a fixed centre position.
                start = int(np.clip(hits[0] - rng.integers(32, self.win - 32), 0, len(raw) - self.win))
            else:
                start = int(rng.integers(0, len(raw) - self.win + 1))
            z = slice(start, start + self.win)
            values = (x[z], y[z], side[z], mask[z], np.ones(self.win, np.float32))
        else:
            pad = self.win - len(raw)
            values = (np.pad(x, ((0, pad), (0, 0))), np.pad(y, (0, pad)),
                      np.pad(side, (0, pad)), np.pad(mask, (0, pad)),
                      np.r_[np.ones(len(raw), np.float32), np.zeros(pad, np.float32)])
        return tuple(torch.from_numpy(np.asarray(v, np.float32)) for v in values)


@torch.no_grad()
def evaluate(models, rallies):
    # CPU inference is fast for this small network and avoids unbounded MPS shape caches.
    models = [m.to("cpu").eval() for m in models]
    dets = [C.detections_from_logits(infer.predict_windowed(models, r["raw"], r["fps"], "cpu"),
                                     r["fps"], r["frames"], DECODER) for r in rallies]
    counts, serve, rally = C.eval_dets(rallies, dets)
    return {"f1_at_1": C.f1_of(counts, 1), "f1_at_2": C.f1_of(counts, 2),
            "f1_at_4": C.f1_of(counts, 4), "counts": counts,
            "serve": serve, "rally": rally}


def load_models(paths):
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = C.make_model(ck["arch"], ck["n_in"], ck["hidden"])
        model.load_state_dict(ck["state"], strict=True)
        models.append(model)
    return models


def run_train(a):
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=False)  # refuse accidental checkpoint overwrite
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    ids, training = load_partition("train")
    val_ids, validation = load_partition("val")
    assert not set(ids) & set(VAL + AUDIT + TEST)
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    assert not set(ck["train_matches"]) & set(VAL + AUDIT + TEST), "initial weights leak into this split"
    model = load_models([a.init])[0]
    baseline = evaluate([model], validation)
    config = {**vars(a), "device": device, "train_ids": ids, "val_ids": val_ids,
              "manifest_sha256": sha(HERE / "data_manifest.json"), "initial_sha256": sha(a.init),
              "torch": torch.__version__, "decoder": DECODER, "baseline": baseline}
    write_json(out / "config.json", config)
    (out / "trainer_snapshot.py").write_text(Path(__file__).read_text())
    print("BASELINE", json.dumps(baseline), flush=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    ds = Windows(training, a.samples, a.seed, a.serve_fraction)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(loader), eta_min=a.lr / 10)
    best, t0 = baseline["f1_at_2"], time.monotonic()
    torch.save({**ck, "experiment": config, "epoch": 0, "val_f1": best}, out / "best.pt")
    for epoch in range(1, a.epochs + 1):
        ds.epoch = epoch
        model.to(device).train()
        losses = []
        for x, y, side, mask, valid in loader:
            x, y, side, mask, valid = [v.to(device) for v in (x, y, side, mask, valid)]
            logits = model(x)
            weight = (1 + 4 * y) * valid
            hit_loss = (F.binary_cross_entropy_with_logits(logits[..., 0], y, reduction="none") * weight).sum() / weight.sum()
            side_loss = (F.binary_cross_entropy_with_logits(logits[..., 1], side, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
            loss = hit_loss + 0.3 * side_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate([model], validation)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic() - t0, **metrics}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        state = {"state": model.state_dict(), "arch": ck["arch"], "hidden": ck["hidden"],
                 "n_in": ck["n_in"], "epoch": epoch, "train_matches": ids,
                 "val_f1": metrics["f1_at_2"], "experiment": config,
                 "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "torch_rng": torch.get_rng_state()}
        torch.save(state, out / "last.pt")
        if metrics["f1_at_2"] > best:
            best = metrics["f1_at_2"]
            torch.save(state, out / "best.pt")
        write_json(out / "status.json", {"state": "running", "pid": os.getpid(), "best_val_f1": best, **row})
        print(json.dumps(row), flush=True)
    write_json(out / "status.json", {"state": "complete", "pid": os.getpid(), "best_val_f1": best,
                                     "epochs": a.epochs, "seconds": time.monotonic() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "baseline", "train"])
    ap.add_argument("--name", default="expanded_v1")
    ap.add_argument("--init", default=str(HIT / "ckpt" / "tcn_s0.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--serve-fraction", type=float, default=0.0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.mode == "audit":
        audit()
    elif a.mode == "baseline":
        _, rallies = load_partition("val")
        paths = [HIT / "ckpt" / (x + ".pt") for x in ("tcn_s0", "tcn_s1", "tcn_lr")]
        results = {"single": evaluate(load_models(paths[:1]), rallies),
                   "ensemble3": evaluate(load_models(paths), rallies)}
        write_json(HERE / "baseline_val.json", results)
        print(json.dumps(results), flush=True)
    else:
        run_train(a)


if __name__ == "__main__":
    main()
