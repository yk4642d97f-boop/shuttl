"""BWF scoring conformance gate: rally.score_match / _implied_winner_obs against the
independent reference of deliverables/DELIVERY_R2 (bwf_ref, written from the BWF Laws
of Badminton). Every score vector (hand-written singles 40 + doubles 22, random
singles 2000 + doubles 2000, 5 new-match probes) must agree on EVERY field of EVERY
rally, and every infer vector (200) on the implied winner. Vectors are copied into
tests/data/bwf_r2 (sha256 in SHA256SUMS = DELIVERY_R2 MANIFEST.json).

App <-> reference: 'rally' -> 'index'; start_frame, end_frame, observed_server are
app-only; reason 'winner_unknown' = 'unknown_winner'. The app's other reasons
(low_confidence, server_mismatch, winner_inferred_from_next_serve) need shots or
confidences, which winner-only vectors do not have.
The same cases run in Swift: ShuttlRally BWFConformanceTests on the fixture from
scripts/swift_fixtures/bwf.py.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from app.services.tracking import rally
from app.services.tracking.data_types import RallyV2

DATA = Path(__file__).parent / "data" / "bwf_r2"
SCORE_FILES = ("hand_singles.jsonl", "hand_doubles.jsonl", "app_probes.jsonl",
               "random_singles.jsonl.gz", "random_doubles.jsonl.gz")
ROW_KEYS = ("index", "game", "server", "service_court", "winner", "score", "games",
            "score_by_player", "games_by_player", "game_over", "game_winner", "match_over",
            "ends_changed", "next_server", "uncertain", "reasons")
DOUBLES_KEYS = ("server_player", "receiver_player", "positions")


def load(name: str) -> list[dict]:
    p = DATA / name
    f = gzip.open(p, "rt", encoding="utf-8") if name.endswith(".gz") else open(p, encoding="utf-8")
    with f:
        return [json.loads(line) for line in f if line.strip()]


def app_kwargs(case: dict) -> dict:
    c = case.get("config") or {}
    return {"first_server": case.get("first_server", "near"), "games_to_win": c.get("games_to_win", 2),
            "points": c.get("points", 21), "cap": c.get("cap", 30), "interval": c.get("interval", 11),
            "doubles": case.get("mode", "singles") == "doubles", "starters": case.get("starters")}


def app_rows(case: dict) -> list[dict]:
    """The app's rows for a winners-only case, in the reference's shape."""
    rs = [RallyV2(start_frame=i, end_frame=i, winner=w, confidence=1.0)
          for i, w in enumerate(case["winners"])]
    out = []
    for r in rally.score_match(rs, **app_kwargs(case)):
        row = {"index": r["rally"], **{k: r[k] for k in ROW_KEYS[1:] if k in r}}
        row["reasons"] = ["unknown_winner" if x == "winner_unknown" else x for x in r["reasons"]]
        for k in DOUBLES_KEYS:
            if k in r:
                row[k] = r[k]
        out.append(row)
    return out


def first_diff(got: list[dict], want: list[dict]) -> str | None:
    if len(got) != len(want):
        return f"{len(got)} rows, expected {len(want)}"
    for g, w in zip(got, want):
        for k in sorted(set(g) | set(w)):
            if g.get(k) != w.get(k):
                return f"rally {w.get('index')}: {k} = {g.get(k)!r}, expected {w.get(k)!r}"
    return None


def test_vectors_are_the_delivered_ones():
    want = dict(line.split()[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines() if line.strip())
    assert set(want) == set(SCORE_FILES) | {"infer.jsonl"}
    for name, sha in want.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == sha, name


@pytest.mark.parametrize("name", SCORE_FILES)
def test_score_vectors_agree_on_every_field(name):
    cases = load(name)
    assert cases, name
    bad = []
    for c in cases:
        assert c.get("kind", "score") == "score"
        d = first_diff(app_rows(c), c["expected"])
        if d is not None:
            bad.append(f"{c['id']}: {d}")
    n_rows = sum(len(c["expected"]) for c in cases)
    print(f"\n[bwf] {name}: {len(cases) - len(bad)}/{len(cases)} cases, {n_rows} rallies")
    assert not bad, f"{len(bad)}/{len(cases)} cases differ, first: " + "; ".join(bad[:10])


def test_infer_vectors_implied_winner():
    """Winner implied by what is seen at the start of the next rally (server side,
    service court, doubles server / receiver player), after replaying `history`."""
    cases = load("infer.jsonl")
    bad = []
    for c in cases:
        kw = app_kwargs(c)
        st = rally._Score(kw["first_server"], kw["games_to_win"], kw["points"], kw["cap"],
                          kw["interval"], kw["doubles"], kw["starters"])
        for w in c["history"]:
            if w in rally.SIDES:
                st.point(w)
            else:
                st.begin()                      # an unknown rally after match over opens the new match
        got = rally._implied_winner_obs(st, c["next"])
        if got != c["expected"]["winner"]:
            bad.append(f"{c['id']}: {got} != {c['expected']['winner']} (next {c['next']})")
    print(f"\n[bwf] infer.jsonl: {len(cases) - len(bad)}/{len(cases)}")
    assert len(cases) == 200 and not bad, "; ".join(bad[:10])


# ── the four DELIVERY_R2 findings, stated directly ───────────────────────────

def _rows(winners, **kw):
    return rally.score_match([RallyV2(start_frame=i, end_frame=i, winner=w, confidence=1.0)
                              for i, w in enumerate(winners)], **kw)


def test_d1_new_match_serves_from_the_right_court():
    """Law 10.1.1: 0-0 -> right court. Before the fix the first rally of a new
    match read the court from the old match's final score (21 odd -> 'left')."""
    rows = _rows(["near"] * 21 + ["far"] * 21 + ["far"])   # A wins 21-0 21-0 (ends change), one more
    r = rows[42]
    assert rows[41]["match_over"] and r["reasons"] == ["new_match"]
    assert (r["game"], r["server"], r["service_court"]) == (1, "far", "right")


def test_d2_players_keep_their_names_across_matches():
    rows = _rows(["near"] * 21 + ["far"] * 21 + ["far"])
    assert rows[42]["score_by_player"] == {"A": 1, "B": 0}   # A (now at the far end) scored
    assert rows[42]["score"] == {"near": 0, "far": 1}


def test_d3_new_match_alone_is_not_uncertain():
    rows = _rows(["near"] * 21 + ["far"] * 21 + ["far"])
    assert rows[42]["uncertain"] is False
    low = rally.score_match([RallyV2(start_frame=0, end_frame=1, winner="near", confidence=0.3)])
    assert low[0]["uncertain"] and low[0]["reasons"] == ["low_confidence"]


def test_d4_unknown_after_match_over_opens_the_new_match():
    rows = _rows(["near"] * 21 + ["far"] * 21 + ["unknown", "near"])
    r = rows[42]
    assert r["reasons"] == ["new_match", "winner_unknown"] and r["uncertain"]
    assert (r["game"], r["match_over"], r["service_court"]) == (1, False, "right")
    assert r["score"] == {"near": 0, "far": 0} and r["games"] == {"near": 0, "far": 0}
    assert rows[43]["reasons"] == [] and rows[43]["game"] == 1     # not a second new match


def test_doubles_service_order():
    """Law 11: the serving pair swaps courts when it scores (A1 serves again, from
    the left); the receiver is the diagonal opponent (same-named court); a side-out
    passes the serve without anyone moving. Expected rows = bwf_ref.doubles."""
    rows = _rows(["near", "near", "far", "far", "near"], doubles=True)
    got = [(r["server_player"], r["receiver_player"], r["service_court"]) for r in rows]
    assert got == [("A1", "B1", "right"), ("A1", "B2", "left"), ("A1", "B1", "right"),
                   ("B2", "A2", "left"), ("B2", "A1", "right")]
    assert rows[4]["positions"] == {"A1": "right", "A2": "left", "B1": "left", "B2": "right"}
    assert rows[0]["positions"] == {"A1": "right", "A2": "left", "B1": "right", "B2": "left"}
    assert all("server_player" not in r for r in _rows(["near"]))       # singles rows unchanged
