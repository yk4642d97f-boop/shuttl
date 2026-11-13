"""BWF scoring conformance fixture for ShuttlRally (BWFConformanceTests): the same
DELIVERY_R2 vectors backend/tests/test_bwf_conformance.py runs (tests/data/bwf_r2).

  full    hand_singles, hand_doubles, app_probes: expected rows in the reference shape
  digest  random_singles, random_doubles: sha256 of the canonical row lines (canon())
          of the reference rows, so 4000 cases (~368k rallies) fit in a small fixture
  infer   infer.jsonl: history, next observation, expected winner

Canonical line per rally (Swift BWFConformanceTests.canon must match byte for byte):
  index|game|server|court|winner|score n,f|games n,f|by_player A,B|games_by_player A,B|
  game_over|game_winner|match_over|ends_changed|next_server|uncertain|reasons (,-joined)
  [|server_player|receiver_player|A1,A2,B1,B2 positions]   (doubles only)
Booleans 1/0, null as '-'.

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.bwf
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.swift_fixtures.common import SWIFT_PACKAGES  # noqa: E402
from scripts.swift_fixtures.rally import dump_z  # noqa: E402
from tests.test_bwf_conformance import app_rows, first_diff, load  # noqa: E402

OUT = SWIFT_PACKAGES / "ShuttlRally/Tests/ShuttlRallyTests/Fixtures/bwf_conformance.json.zz"


def _b(v) -> str:
    return "-" if v is None else ("1" if v is True else ("0" if v is False else str(v)))


def canon(row: dict) -> str:
    f = [row["index"], row["game"], row["server"], row["service_court"], row["winner"],
         f"{row['score']['near']},{row['score']['far']}", f"{row['games']['near']},{row['games']['far']}",
         f"{row['score_by_player']['A']},{row['score_by_player']['B']}",
         f"{row['games_by_player']['A']},{row['games_by_player']['B']}",
         row["game_over"], row["game_winner"], row["match_over"], row["ends_changed"], row["next_server"],
         row["uncertain"], ",".join(row["reasons"])]
    if "positions" in row:
        p = row["positions"]
        f += [row["server_player"], row["receiver_player"], ",".join(p[k] for k in ("A1", "A2", "B1", "B2"))]
    return "|".join(_b(x) for x in f)


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(canon(r) for r in rows).encode()).hexdigest()


def _inp(c: dict) -> dict:
    st = c.get("starters")
    st = None if st is None else {k: ([v] if isinstance(v, str) else list(v)) for k, v in st.items()}
    return {"id": c["id"], "mode": c.get("mode", "singles"), "config": c.get("config") or {},
            "first_server": c.get("first_server", "near"), "starters": st}


def main() -> None:
    doc = {"full": [], "digest": [], "infer": []}
    for name in ("hand_singles.jsonl", "hand_doubles.jsonl", "app_probes.jsonl"):
        for c in load(name):
            assert first_diff(app_rows(c), c["expected"]) is None, c["id"]     # the Python gate holds
            exp = [{**r, "reasons": ["winner_unknown" if x == "unknown_winner" else x for x in r["reasons"]]}
                   for r in c["expected"]]
            doc["full"].append({**_inp(c), "winners": c["winners"], "expected": exp})
    for name in ("random_singles.jsonl.gz", "random_doubles.jsonl.gz"):
        for c in load(name):
            assert digest(app_rows(c)) == digest(c["expected"]), c["id"]
            exp = [{**r, "reasons": ["winner_unknown" if x == "unknown_winner" else x for x in r["reasons"]]}
                   for r in c["expected"]]
            doc["digest"].append({**_inp(c), "winners": c["winners"], "n_rows": len(exp), "sha256": digest(exp)})
    for c in load("infer.jsonl"):
        doc["infer"].append({**_inp(c), "history": c["history"], "next": c["next"],
                             "expected": c["expected"]["winner"]})
    dump_z(doc, OUT)
    print(f"{OUT.name}: full {len(doc['full'])}, digest {len(doc['digest'])}, infer {len(doc['infer'])}, "
          f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
