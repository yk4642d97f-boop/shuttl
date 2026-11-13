"""Convert a learned hit re-scorer (.npz with w, b[, mu, sd] over hits.HIT_FEATURES, as
read by hits.load_learned_hit_model) to the JSON the Swift port loads
(ShuttlHits.LinearHitModel / HitModelRegistry.load(path:)).

    .venv/bin/python -m scripts.swift_fixtures.hits_model_to_json model.npz model.json

JSON: {"w": [F], "b": x, "mu": [F] | x, "sd": [F] | x, "features": [...HIT_FEATURES]}
(absent mu/sd mean 0 / 1, exactly like Python). Floats are written with repr precision
(round-trip exact). TorchScript (.pt) models are not convertible here — export those to
CoreML and plug them in via ShuttlHits.CoreMLHitRescorer.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from app.services.tracking.hits import HIT_FEATURES


def _vec_or_scalar(a: np.ndarray):
    a = np.asarray(a, float)
    return float(a) if a.ndim == 0 else [float(v) for v in a.reshape(-1)]


def convert(npz_path: str, json_path: str) -> dict:
    d = np.load(npz_path)
    w = np.asarray(d["w"], float).reshape(-1)
    if len(w) != len(HIT_FEATURES):
        raise ValueError(f"w has {len(w)} weights, HIT_FEATURES has {len(HIT_FEATURES)}")
    out = {"w": [float(v) for v in w], "b": float(d["b"]), "features": list(HIT_FEATURES)}
    if "mu" in d:
        out["mu"] = _vec_or_scalar(d["mu"])
    if "sd" in d:
        out["sd"] = _vec_or_scalar(d["sd"])
    flat = out["w"] + [out["b"]] + [v for k in ("mu", "sd") if k in out
                                    for v in (out[k] if isinstance(out[k], list) else [out[k]])]
    if not all(math.isfinite(v) for v in flat):
        raise ValueError("model contains non-finite values")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}")
