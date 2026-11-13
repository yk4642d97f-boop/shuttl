"""Export frozen weights without optimizer or Python-specific checkpoint objects."""
import json
from pathlib import Path

import numpy as np
import torch

import hit_experiment as E


def main():
    torch.set_num_threads(2)
    selection = json.loads((E.HERE / "selection.json").read_text())
    source = Path(selection["candidate"])
    assert E.sha(source) == selection["sha256"], "Frozen checkpoint changed"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    payload = {"state": checkpoint["state"], "arch": str(checkpoint["arch"]),
               "n_in": int(checkpoint["n_in"]), "hidden": int(checkpoint["hidden"]),
               "decoder": E.DECODER, "source_sha256": selection["sha256"]}
    destination = E.HERE / "artifacts/hitnet_single_v2.pt"
    if destination.exists():
        raise SystemExit("Export already exists; inspect rather than silently overwrite.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    # PyTorch 2.6+ defaults to restricted weight loading. Confirm the export supports it.
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    original = E.load_models([source])[0].eval()
    exported = E.C.make_model(reloaded["arch"], reloaded["n_in"], reloaded["hidden"]).eval()
    exported.load_state_dict(reloaded["state"], strict=True)
    _, rallies = E.load_partition("val")
    sample = rallies[0]
    with torch.no_grad():
        old = E.infer.predict_windowed([original], sample["raw"], sample["fps"], "cpu")
        new = E.infer.predict_windowed([exported], sample["raw"], sample["fps"], "cpu")
    assert np.array_equal(old, new), "Export changed predictions"
    E.write_json(destination.with_suffix(".json"), {
        "file": destination.name, "sha256": E.sha(destination),
        "source_sha256": selection["sha256"], "decoder": E.DECODER,
        "parity": {"max_abs_logit_error": float(np.max(np.abs(old - new))),
                   "frames": len(old), "match": sample["mid"]},
        "scope": "Singles hit detection candidate; full-video and doubles validation pending",
        "weights_only_load": True,
    })
    print(destination)


if __name__ == "__main__":
    main()
