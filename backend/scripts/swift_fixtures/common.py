"""Shared helpers for exporting Python-reference fixtures to the Swift packages.

Convention (mirrors ShuttlCore.ShuttlJSON): keys are the Python field names,
non-finite floats become null, numpy arrays become nested lists, dataclasses
become dicts of their fields. Output is compact, sorted JSON.

Usage from a stage exporter:
    from scripts.swift_fixtures.common import dump, REPO, SWIFT_PACKAGES
    dump(obj, SWIFT_PACKAGES / "ShuttlLinker/Tests/ShuttlLinkerTests/Fixtures/x.json")
Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.<module>
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
SWIFT_PACKAGES = REPO / "iOS" / "Packages"
TEST_SPLIT = BACKEND / "ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test"


def _omit_default(obj: Any, f: dataclasses.Field) -> bool:
    """HitEvent.source is written only when it isn't the default "kink" (Swift does the same)."""
    return type(obj).__name__ == "HitEvent" and f.name == "source" and getattr(obj, "source") == "kink"


def to_jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)
                if not _omit_default(x, f)}
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def dump(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return path
