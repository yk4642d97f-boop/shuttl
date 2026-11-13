"""ml/train_detector_ft.py: logic around the training loop (no training here).

Split-by-match correctness, held-out guards, Test scoring maths (== benchmark.add_tn4),
bootstrap/sign test, early-stop bookkeeping, overlap detection and checkpoint round-trip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(BACKEND / "ml")):
    if p not in sys.path:
        sys.path.insert(0, p)

FT = pytest.importorskip("train_detector_ft")


# ── split by match ───────────────────────────────────────────────────────────

MATCHES = [f"match{i}" for i in range(1, 27)]


def test_split_is_by_match_disjoint_and_complete():
    sp = FT.make_split(MATCHES, 2, seed=20260927)
    assert len(sp["val_matches"]) == 2
    assert not set(sp["val_matches"]) & set(sp["train_matches"])
    assert sorted(sp["val_matches"] + sp["train_matches"], key=FT.match_number) == MATCHES


def test_split_validation_is_professional_only_and_amateur_trains():
    for seed in range(50):
        sp = FT.make_split(MATCHES, 3, seed)
        assert all(FT.match_number(m) <= FT.N_PRO for m in sp["val_matches"])
        assert {"match24", "match25", "match26"} <= set(sp["train_matches"])


def test_split_is_deterministic_and_seed_dependent():
    a, b = FT.make_split(MATCHES, 2, 1), FT.make_split(MATCHES, 2, 1)
    assert a == b
    assert len({tuple(FT.make_split(MATCHES, 2, s)["val_matches"]) for s in range(20)}) > 1


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        FT.make_split(["match1", "clipX"], 1, 0)
    with pytest.raises(ValueError):
        FT.make_split(["match1", "match2"], 2, 0)


def test_split_roots_refuse_held_out_and_overlap(tmp_path):
    ok = {"train_root": str(tmp_path / "Train"), "train_matches": ["match1", "match3"],
          "val_matches": ["match2"]}
    tr, va = FT.split_roots(ok)
    assert [Path(p).name for p in tr] == ["match1", "match3"] and [Path(p).name for p in va] == ["match2"]
    with pytest.raises(SystemExit):
        FT.split_roots({**ok, "train_root": str(tmp_path / "Test")})
    with pytest.raises(SystemExit):
        FT.split_roots({**ok, "val_matches": ["match1"]})


def test_cmd_split_on_disk_counts_clips_per_match(tmp_path):
    root = tmp_path / "Train"
    for m, n in (("match1", 2), ("match2", 1), ("match3", 1), ("match24", 3)):
        for k in range(n):
            (root / m / "csv").mkdir(parents=True, exist_ok=True)
            (root / m / "video").mkdir(parents=True, exist_ok=True)
            (root / m / "csv" / f"c{k}_ball.csv").write_text("Frame,Visibility,X,Y\n0,0,0,0\n")
            (root / m / "video" / f"c{k}.mp4").write_bytes(b"x")
    out = tmp_path / "split.json"
    sp = FT.cmd_split(argparse.Namespace(train_root=str(root), n_val=1, seed=0, out=str(out)))
    assert json.loads(out.read_text()) == sp
    assert sp["clips_per_match"] == {"match1": 2, "match2": 1, "match3": 1, "match24": 3}
    assert sp["val_clips"] + sp["train_clips"] == 7
    assert sp["val_matches"][0] in ("match1", "match2", "match3")


# ── archive mapping ──────────────────────────────────────────────────────────

def test_archive_groups_and_match_renaming():
    assert FT.archive_group(Path("TrackNetV2/Professional/match3/csv/a_ball.csv")) == "pro"
    assert FT.archive_group(Path("TrackNetV2/Amateur/match1/csv/a_ball.csv")) == "amateur"
    assert FT.archive_group(Path("TrackNetV2/Test/match1/csv/a_ball.csv")) == "test"
    assert FT.train_match_name("pro", 23) == "match23"
    assert FT.train_match_name("amateur", 1) == "match24"
    assert FT.train_match_name("amateur", 3) == "match26"
    with pytest.raises(ValueError):
        FT.train_match_name("test", 1)
    assert sum(FT.HACKMD_PRO_RALLIES) == 146 and len(FT.HACKMD_PRO_RALLIES) == FT.N_PRO


# ── overlap detection ────────────────────────────────────────────────────────

def _track(n, seed, shift=0):
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 5, (n + 10, 2)), 0) + 400
    vis = np.ones(n + 10, bool)
    return xy[shift:shift + n], vis[shift:shift + n]


def test_label_content_match_finds_shifted_copy_and_not_others():
    a_xy, a_vis = _track(200, 1)
    b_xy, b_vis = _track(200, 1, shift=2)          # same rally, 2-frame offset
    c_xy, c_vis = _track(200, 2)
    assert FT.label_content_match(a_xy, a_vis, b_xy, b_vis)[0] < 1e-9
    assert FT.label_content_match(a_xy, a_vis, c_xy, c_vis)[0] > 20


def test_overlap_report_flags_exact_and_near_duplicates():
    xy, vis = _track(100, 3)
    th = np.full((36, 64), 100.0, np.float32)
    train = [{"name": "match1/a", "video_sha256": "v1", "csv_sha256": "c1", "_thumb": th,
              "_xy": xy, "_vis": vis},
             {"name": "match2/b", "video_sha256": "v2", "csv_sha256": "c2", "_thumb": th + 50,
              "_xy": xy + 300, "_vis": vis}]
    test = [{"name": "match1/t", "video_sha256": "v1", "csv_sha256": "cX", "_thumb": th + 1,
             "_xy": xy, "_vis": vis}]
    ov = FT.overlap_report(train, test)
    assert ov["exact_video_matches"] == [("match1/a", "match1/t")]
    assert ov["exact_csv_matches"] == []
    assert ov["thumb_closest_pairs"][0][1:] == ("match1/a", "match1/t")
    assert ov["label_track_below_5px"][0][2:] == ("match1/a", "match1/t")
    assert all(p[2] != "match2/b" for p in ov["label_track_below_5px"])


# ── scoring maths ────────────────────────────────────────────────────────────

def _rows(counts):
    return [{"clip": f"m/c{i}", "tp": tp, "fp": fp, "fn": fn, "f1": float(FT.f1_counts(tp, fp, fn)),
             "eval_frames": tp + fp + fn + i, "median_top1_score_visible": 0.9 - 0.01 * i}
            for i, (tp, fp, fn) in enumerate(counts)]


def test_f1_counts_matches_benchmark_add_tn4_formula():
    from app.services.tracking import metrics as M
    for tp, fp, fn in ((9020, 72, 365), (1, 0, 0), (0, 3, 4), (5, 5, 5)):
        p, r = M._ratio(tp, tp + fp), M._ratio(tp, tp + fn)
        assert float(FT.f1_counts(tp, fp, fn)) == pytest.approx(M._f1(p, r), abs=1e-12)
    assert float(FT.f1_counts(0, 0, 0)) == 0.0


def test_pooled_sums_counts_not_clip_f1():
    rows = _rows([(90, 5, 5), (10, 0, 30)])
    pm = FT.pooled(rows)
    assert (pm["tp"], pm["fp"], pm["fn"]) == (100, 5, 35)
    assert pm["f1"] == pytest.approx(200 / 240)
    assert FT.pooled(rows, {"m/c1"})["f1"] == pytest.approx(20 / 50)


def test_bootstrap_identical_runs_give_zero_and_real_gain_excludes_zero():
    before = _rows([(90, 5, 5), (80, 10, 10), (95, 2, 3), (70, 15, 15)] * 5)
    same = FT.paired_bootstrap(before, before, n_boot=500)
    assert same["delta_f1"] == 0 and same["ci95"] == [0.0, 0.0]
    after = [{**r, "fn": max(0, r["fn"] - 3), "tp": r["tp"] + 3} for r in before]
    gain = FT.paired_bootstrap(before, after, n_boot=2000)
    assert gain["delta_f1"] > 0 and gain["ci95"][0] > 0 and gain["p_delta_le_0"] == 0.0


def test_sign_test_counts_and_p_value():
    before = _rows([(90, 5, 5)] * 10)
    after = [{**r, "f1": r["f1"] + (0.01 if i < 9 else -0.01)} for i, r in enumerate(before)]
    st = FT.sign_test(before, after)
    assert (st["improved"], st["worse"], st["tied"]) == (9, 1, 0)
    assert st["p_two_sided"] == pytest.approx(2 * 11 / 1024)
    assert FT.sign_test(before, before)["p_two_sided"] == 1.0


def test_weak_sets_come_from_baseline_only_and_are_deterministic():
    rows = _rows([(90, 5, 5), (50, 20, 30), (95, 2, 3), (70, 15, 15), (60, 1, 1)])
    w = FT.weak_sets(rows, k=2)
    assert w["low_confidence"] == ["m/c4", "m/c3"]          # lowest median score
    assert w["low_f1"][0] == "m/c1"
    assert len(w["long_rally"]) == 2
    assert FT.weak_sets(list(reversed(rows)), k=2) == w


# ── early stopping bookkeeping ───────────────────────────────────────────────

def test_early_stop_patience_lowers_epochs_and_persists(tmp_path):
    st = FT.EarlyStop(patience=2, budget_s=1e9, state_path=tmp_path / "es.json")
    a = argparse.Namespace(epochs=10)
    for f1 in (0.90, 0.92, 0.91):
        st.after_eval(f1, a)
    assert a.epochs == 10 and st.reason is None and st.best == 0.92
    st.after_eval(0.915, a)
    assert a.epochs == 4 and "early stop" in st.reason
    st2 = FT.EarlyStop(patience=2, budget_s=1e9, state_path=tmp_path / "es.json")
    assert st2.best == 0.92 and st2.bad == 2 and len(st2.history) == 4


def test_early_stop_budget():
    st = FT.EarlyStop(patience=99, budget_s=0.0)
    a = argparse.Namespace(epochs=10)
    st.after_eval(0.5, a)
    assert a.epochs == 1 and st.reason.startswith("budget")


# ── checkpoint round-trip ────────────────────────────────────────────────────

def test_install_checkpoint_round_trip_and_never_overwrites(tmp_path):
    torch = pytest.importorskip("torch")
    import train_detector as TD
    arch = TD.ARCHS["wasb"]
    net, _ = TD.build_net(arch, "scratch")
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    best = tmp_path / "best.pth.tar"
    a = argparse.Namespace(target_mode="official", radius=2.5, loss="qfl")
    TD.save_best(best, net, arch, a, 3, {"f1": 0.9}, "test")
    dest = tmp_path / "ckpts" / FT.FT_NAMES["wasb"]
    got = FT.install_checkpoint(best, "wasb", dest)
    assert got == dest and dest.is_file()
    loaded, spec = TD.det().load_model(dest, "wasb")
    assert spec.name == "wasb"
    ref = net.state_dict()
    assert all(torch.equal(v, ref[k]) for k, v in loaded.state_dict().items())
    before = dest.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        FT.install_checkpoint(best, "wasb", dest)
    assert dest.stat().st_mtime_ns == before


def test_ft_names_do_not_shadow_detector_defaults():
    """build_detector prefers *_finetuned*; the _ft files must only load when asked for."""
    import train_detector as TD
    names = {n for v in TD.det().WEIGHT_FILES.values() for n in v}
    for fn in FT.FT_NAMES.values():
        assert fn not in names and f"ckpts/{fn}" not in names
