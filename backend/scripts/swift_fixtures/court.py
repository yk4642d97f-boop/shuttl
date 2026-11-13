"""Reference values for the Swift ShuttlCourt package (court_calibration.py port).

Sections (run all, or name some on the command line):
  primitives  cv2 / numpy building blocks on small random inputs (bit-level checks)
  manual      calibrate_from_points cases ported from tests/test_court_calibration.py
  images      exact-input parity: median images saved as PNG + Python intermediates and
              outputs of refine_camera / the automatic path on them
  real        the 29 held-out TrackNetV2 Test clips: Python calibrate_from_frames result
              on the pipeline's 24 sampled frames (the Swift test decodes the clip itself)

Run from backend/:  .venv/bin/python -m scripts.swift_fixtures.court [section ...]
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from app.services.tracking import court_calibration as cc
from app.services.tracking import court_model as cm
from app.services.tracking import video_io
from app.services.tracking.camera import CameraModel
from scripts.swift_fixtures.common import REPO, SWIFT_PACKAGES, TEST_SPLIT, dump

OUT = SWIFT_PACKAGES / "ShuttlCourt/Tests/ShuttlCourtTests/Fixtures"


# ----------------------------------------------------------------------------- primitives

def primitives() -> dict:
    rng = np.random.default_rng(11)
    out: dict = {}

    rv = [rng.normal(0, 1.2, 3) for _ in range(8)] + [np.array([np.pi, 0, 0]),
                                                      np.array([0.0, 0.0, 0.0]), np.array([0, 3.1, 0.4])]
    out["rodrigues"] = [{"rvec": r, "R": cv2.Rodrigues(r.reshape(3, 1))[0],
                         "back": cv2.Rodrigues(cv2.Rodrigues(r.reshape(3, 1))[0])[0].ravel()} for r in rv]

    homs = []
    for n in (4, 4, 6, 9):
        src = rng.uniform(-6.7, 6.7, (n, 2))
        if n == 4:
            dst = rng.uniform(0, 1280, (n, 2))
        else:                       # consistent with a real camera + 1 px noise (tapped keypoints)
            cam = CameraModel.look_at([-14.0, 0.5, 5.0], [0.0, 0.0, 0.0], 1500.0, (1280, 720))
            dst = cam.project(np.column_stack([src, np.zeros(n)])) + rng.normal(0, 1.0, (n, 2))
        H, _ = cv2.findHomography(src, dst, 0)
        homs.append({"src": src, "dst": dst, "H": H})
    out["find_homography"] = homs

    img = rng.uniform(-100, 255, (29, 37)).astype(np.float32)
    out["gauss"] = {"w": 37, "h": 29, "src": img.ravel(), "dst": cv2.GaussianBlur(img, (0, 0), 0.8).ravel(),
                    "kernel": cv2.getGaussianKernel(7, 0.8, cv2.CV_32F).ravel()}

    m = (rng.random((40, 53)) < 0.04).astype(np.uint8) * 255
    m[10:14, 5:40] = 255
    out["dist"] = {"w": 53, "h": 40, "mask": (m > 0).astype(int).ravel(),
                   "dt_out": cv2.distanceTransform(255 - m, cv2.DIST_L2, 3).ravel(),
                   "dt_in": cv2.distanceTransform(m, cv2.DIST_L2, 3).ravel()}

    s = rng.uniform(0, 60, (20, 30)).astype(np.float32)
    X = rng.uniform(-3, 33, (15, 11)).astype(np.float32)
    Y = rng.uniform(-3, 23, (15, 11)).astype(np.float32)
    out["remap"] = {"w": 30, "h": 20, "src": s.ravel(), "X": X.ravel(), "Y": Y.ravel(), "rows": 15, "cols": 11,
                    "dst": cv2.remap(s, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0).ravel()}

    th = np.zeros((300, 400), np.uint8)
    for _ in range(9):
        p = tuple(int(v) for v in rng.integers(0, [400, 300]))
        q = tuple(int(v) for v in rng.integers(0, [400, 300]))
        cv2.line(th, p, q, 255, 1)
    th[rng.random(th.shape) < 0.01] = 255
    D = 400.0
    segs = cv2.HoughLinesP(th, 1, np.pi / 720, threshold=max(12, int(0.012 * D)),
                           minLineLength=0.03 * D, maxLineGap=0.006 * D)
    out["hough"] = {"w": 400, "h": 300, "thin": (th > 0).astype(int).ravel(),
                    "segments": segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4))}

    covs = []
    for k in range(40):
        if k < 5:
            a = np.diag(rng.uniform(0, 5, 2))
        else:
            v = rng.normal(size=(2, 2)) * rng.uniform(0.1, 300)
            a = v @ v.T
        if k % 7 == 0:
            a[0, 1] = a[1, 0] = a[0, 1] * 1e-9
        covs.append({"a": a, "evecs": np.linalg.eigh(a)[1]})
    out["eigh2"] = covs

    svs = []
    for k in range(30):
        A = rng.normal(size=(int(rng.integers(2, 7)), 3))
        A[:, 2] *= rng.uniform(1, 800)
        svs.append({"A": A, "v": np.linalg.svd(A)[2][-1]})
    out["svd_last"] = svs

    strength = np.zeros((60, 80), np.float32)
    strength[rng.random(strength.shape) < 0.08] = 30
    strength[20:22, 3:70] = 25
    strength[30:50, 40] = 19
    out["mask"] = {"w": 80, "h": 60, "strength": strength.ravel(), "min_len": 0.012 * 1280,
                   "mask": (cc._mask_from_strength(strength, 0.012 * 1280) > 0).astype(int).ravel()}

    out["scales"] = {str(D): sorted({int(s) for s in np.round(np.geomspace(2, max(4, int(round(0.012 * D))), 6))})
                     for D in (540.0, 720.0, 960.0, 1280.0, 1920.0, 2560.0, 3840.0)}
    return out


# ----------------------------------------------------------------------------- manual path

def _test_module():
    from tests import test_court_calibration as t
    return t


def manual() -> dict:
    t = _test_module()
    cases = []

    def add(kind: str, true: CameraModel, pts: dict, size=t.SIZE, expect_error: bool = False) -> None:
        c = {"kind": kind, "size": list(size), "true": true.to_dict(),
             "points": {k: list(v) for k, v in pts.items()}, "order": list(pts)}
        try:
            est = cc.calibrate_from_points(pts, size)
            c["camera"] = est.to_dict()
            c["errors"] = t.calib_errors(true, est)
        except ValueError as e:
            c["error"] = str(e)
        cases.append(c)

    for seed in range(10):                                        # test_points_exact
        rng = np.random.default_rng(seed)
        true = [t.behind_camera, t.broadcast_camera, t.side_camera][seed % 3](rng)
        add("exact", true, t._pts(true, t.CORNERS))
    for seed in range(30):                                        # test_points_click_noise
        rng = np.random.default_rng(100 + seed)
        true = [t.behind_camera, t.broadcast_camera][seed % 2](rng)
        add("noise", true, t._pts(true, t.CORNERS, rng, 2.0))
    rng = np.random.default_rng(7)                               # mirrored / far-side labels
    true = t.broadcast_camera(rng)
    pts = t._pts(true, t.CORNERS)
    swapped = {"nb_ld": "nb_rd", "nb_rd": "nb_ld", "fb_ld": "fb_rd", "fb_rd": "fb_ld"}
    add("mirrored", true, {swapped[k]: v for k, v in pts.items()})
    near_far = {"nb_ld": "fb_rd", "nb_rd": "fb_ld", "fb_ld": "nb_rd", "fb_rd": "nb_ld"}
    add("far_side", true, {near_far[k]: v for k, v in pts.items()})
    rng = np.random.default_rng(3)                               # extra keypoints + errors
    true = t.behind_camera(rng)
    names = t.CORNERS + ["ns_ld", "ns_rd", "fs_c", "nl_ls"]
    add("extra", true, {**t._pts(true, names), "bogus": (1.0, 2.0)})
    add("three", true, t._pts(true, t.CORNERS[:3]))
    add("collinear", true, {n: (float(i), float(i)) for i, n in enumerate(t.CORNERS)})
    return {"cases": cases}


# ----------------------------------------------------------------------------- exact-input images

def save_planes(img: np.ndarray, name: str, out_dir=None) -> str:
    """Store the (min, max)-over-channels planes of a BGR uint8 image as raw DEFLATE
    (Swift: NSData.decompressed(using: .zlib)). The calibration only ever uses the
    per-pixel channel min / max (whiteness), so this is an exact, compact stand-in."""
    import zlib
    mm = np.stack([img.min(2), img.max(2)], -1).astype(np.uint8) if img.ndim == 3 else img[..., None]
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = co.compress(np.ascontiguousarray(mm).tobytes()) + co.flush()
    ((out_dir or OUT) / name).write_bytes(raw)
    return name


def _stats_json(st: dict) -> dict:
    return {k: st.get(k) for k in ("coverage", "rmse", "n_t", "n_l", "n_vis", "n_inl", "n_absent",
                                   "absent_frac", "explained", "explained_frac", "t_straddle")}


def trace_calibrate_image(img: np.ndarray, image_size) -> dict:
    """cc._calibrate_image step by step (same private calls), recording intermediates."""
    tr: dict = {}
    feat = cc._Features(img)
    tr["features"] = {"mask_n": int((feat.mask > 0).sum()), "thin_n": int((feat.thin > 0).sum()),
                      "strength_sum": float(feat.strength.astype(np.float64).sum()),
                      "dt_sum": float(np.minimum(feat.dt, 100).astype(np.float64).sum())}
    segs = cc._segments(feat)
    tr["segments"] = segs
    lines = cc._merge_segments(segs, feat.D)
    tr["lines"] = [{"l": ln.l, "p0": ln.p0, "p1": ln.p1, "weight": ln.weight} for ln in lines]
    fams_all = cc._vp_families(lines)
    lid = {id(ln): k for k, ln in enumerate(lines)}
    tr["families"] = [{"v": v, "members": [lid[id(m)] for m in mem]} for v, mem in fams_all]
    fams = [f for f in fams_all if len(f[1]) >= 2]
    out = {"trace": tr, "camera": None}
    if len(fams) < 2:
        return out
    Hs, Ss, Rs = [], [], []
    for a in range(len(fams)):
        for b in range(len(fams)):
            if a != b:
                H, s, r = cc._hypotheses(feat, fams[a], fams[b])
                Hs.append(H), Ss.append(s), Rs.append(r)
    H, S, REF = np.concatenate(Hs), np.concatenate(Ss), np.concatenate(Rs)
    tr["n_hyp"] = int(len(S))
    tr["n_hyp_per_pair"] = [int(len(s)) for s in Ss]
    if len(H) == 0:
        return out
    top = np.argsort(-S, kind="stable")[:2000]
    tr["top_scores"] = S[top[:50]]
    H, REF = H[top], REF[top]
    ls = cc._line_score(feat, H)
    chosen = cc._choose_hypotheses(feat, H)
    tr["chosen"] = [{"H": H[i] / H[i][2, 2], "ref": REF[i], "ls": float(ls[i])} for i in chosen]
    cands = cc._quick_candidates(feat, H, REF, chosen, image_size)
    tr["cands"] = [{"explained": e, "camera": c.to_dict(), "stats": _stats_json(st)} for e, c, st in cands]
    ref_cam = cc._calibrate_image(img, image_size, d := {})
    out["camera"] = None if ref_cam is None else ref_cam.to_dict()
    out["diag"] = {k: v for k, v in d.items() if not isinstance(v, (list, dict))}
    return out


def images() -> dict:
    t = _test_module()
    out: dict = {"sim": [], "real": []}
    size = (1280, 720)
    for seed, kind, style in [(300, "behind", "green"), (301, "broadcast", "wood"), (304, "behind", "blue")]:
        rng = np.random.default_rng(seed)
        true = {"behind": t.behind_camera, "broadcast": t.broadcast_camera}[kind](rng, size)
        taps = t._pts(true, t.CORNERS, rng, 2.0)
        est = cc.calibrate_from_points(taps, size)
        frames = t.render_frames(true, rng, style=style)
        med = cc._to_image(frames, size)
        ref = cc.refine_camera(est, [med])
        fk = cc.refine_camera(est, [med], fit_k1=False)
        auto = trace_calibrate_image(med, size)
        name = save_planes(med, f"sim_{seed}.deflate")
        out["sim"].append({"seed": seed, "kind": kind, "style": style, "planes": name, "size": list(size),
                           "true": true.to_dict(), "taps": {k: list(v) for k, v in taps.items()},
                           "est": est.to_dict(), "refined": ref.to_dict(), "refined_no_k1": fk.to_dict(),
                           "est_errors": t.calib_errors(true, est), "refined_errors": t.calib_errors(true, ref),
                           "auto": auto})
    for match in ("match1", "match2", "match3"):
        clip = sorted(TEST_SPLIT.glob(f"{match}/video/*.mp4"))[0]
        frames = [f for _, f in video_io.sample_frames(str(clip), 24)]
        med = cc._to_image(frames, size)
        auto = trace_calibrate_image(med, size)
        name = save_planes(med, f"real_{match}.deflate")
        out["real"].append({"clip": str(clip.relative_to(REPO)), "planes": name, "size": list(size), "auto": auto})
        if auto["camera"] is not None:
            cam = CameraModel.from_dict(auto["camera"]) if hasattr(CameraModel, "from_dict") else None
            if cam is not None:
                out["real"][-1]["refined"] = cc.refine_camera(cam, [med]).to_dict()
    return out


# ----------------------------------------------------------------------------- evidence gates

def gates() -> dict:
    """Exact-input cases for the gate changes (n_t == 2 exception, focal hint, reasons):
    venue sweep cases 1 and 3 (1920x1080, 0.8 m high / 0.5 m behind: n_t == 2 → formerly a
    false reject and a false accept), a grid of non-badminton lines, a blank image, and
    focal-hint variants. Same trace as `images` plus diagnostics["reason"]."""
    sys.path.insert(0, str(REPO / "tools/venue"))
    import calib_sweep as cs
    t = _test_module()
    out: dict = {"cases": []}

    def add(name: str, med: np.ndarray, size, true: CameraModel | None, hints=()) -> None:
        auto = trace_calibrate_image(med, size)
        d: dict = {}
        cc._calibrate_image(med, size, d)
        auto.setdefault("diag", {})["reason"] = d.get("reason")
        hinted = []
        for h in hints:
            dh: dict = {}
            cam = cc._calibrate_image(med, size, dh, focal_hint=h)
            hinted.append({"focal_hint": h, "camera": None if cam is None else cam.to_dict(),
                           "reason": dh.get("reason")})
        out["cases"].append({"name": name, "planes": save_planes(med, f"gate_{name}.deflate"),
                             "size": list(size), "true": None if true is None else true.to_dict(),
                             "auto": auto, "hinted": hinted})
        print(f"  gates {name}: {d.get('reason')} {[x['reason'] for x in hinted]}", flush=True)

    cases = {c["case"]: c for c in cs.build_cases(3, 5, False)}
    for cid in (1, 3):
        c = cases[cid]
        true = cs.make_camera(c["height_m"], c["dist_m"], c["lat_m"], c["roll_deg"])
        frames = t.render_frames(true, np.random.default_rng(c["seed"]), n=c["frames"], style=c["style"])
        f = float(true.K[0, 0])
        add(f"sweep{cid}", cc._to_image(frames, (cs.W, cs.H)), (cs.W, cs.H), true, hints=(f, 0.6 * f))
    rng = np.random.default_rng(1011)
    true = t.behind_camera(rng)
    frames = t.render_frames(true, rng, style="wood", lines=t._grid_lines(rng))
    add("grid", cc._to_image(frames, t.SIZE), t.SIZE, None)
    add("blank", np.full((t.SIZE[1], t.SIZE[0], 3), 120, np.uint8), t.SIZE, None)
    return out


# ----------------------------------------------------------------------------- single-frame fallback

def fallback() -> dict:
    """calibrate_from_frames' single-frame fallback on exact inputs: per scene the median
    image and every frame the fallback tries (min/max planes), Python's per-frame result
    (veto_dominated=True) and the final outcome. Scenes: sweep case 3 with the median
    FORCED to fail (its frames 2-4 agree on a camera 3.35 m wrong), a player standing still
    on the near baseline in frames 4-20 (median fails, frames 3 & 21 agree), and still for
    the whole clip (every frame occluded → reject)."""
    t = _test_module()
    size = t.SIZE
    out: dict = {"cases": []}
    true3, f3 = t._low_tripod(0.5, 10000002, "green")
    cam_b = t._behind(1.6, 3.0)
    scenes = [("case3_forced", true3, f3, True),
              ("still_4_20", cam_b, t.still_player_frames(cam_b, np.random.default_rng(4), "blue", range(4, 21)), False),
              ("still_all", cam_b, t.still_player_frames(cam_b, np.random.default_rng(7), "green", range(24)), False)]
    for name, true, frames, forced in scenes:
        med = cc._to_image(frames, size)
        dm: dict = {}
        cm_cam = cc._calibrate_image(med, size, dm)
        entry = {"name": name, "size": list(size), "n_frames": len(frames), "forced_median_fail": forced,
                 "true": true.to_dict(), "median_planes": save_planes(med, f"fb_{name}_median.deflate"),
                 "median": {"reason": "evidence_gate" if forced else dm.get("reason"),
                            "camera": None if (forced or cm_cam is None) else cm_cam.to_dict()},
                 "frames": []}
        for i in cc._fallback_indices(len(frames)):
            d: dict = {"source": f"frame{i}"}
            cam = cc._calibrate_image(cc._to_image([frames[i]], size), size, d, veto_dominated=True)
            entry["frames"].append({"i": i, "planes": save_planes(frames[i], f"fb_{name}_f{i}.deflate"),
                                    "reason": d.get("reason"), "dominated_by": d.get("dominated_by"),
                                    "camera": None if cam is None else cam.to_dict()})
        if forced:
            cam, fd = cc._frame_fallback(frames, size, None, None)
            res = {"camera": None if cam is None else cam.to_dict(), "n_frame_accepts": fd["n_frame_accepts"],
                   "frames_tried": fd["frames_tried"], "frames_agree": fd.get("frames_agree"),
                   "reason": fd.get("reason") if cam is not None else
                   ("inconsistent_frames" if fd["n_frame_accepts"] else "evidence_gate")}
        else:
            d = {}
            cam = cc.calibrate_from_frames(frames, size, diagnostics=d)
            res = {"camera": None if cam is None else cam.to_dict(), "reason": d.get("reason"),
                   "n_frame_accepts": d.get("n_frame_accepts"), "frames_tried": d.get("frames_tried"),
                   "frames_agree": d.get("frames_agree")}
        entry["final"] = res
        out["cases"].append(entry)
        print(f"  fallback {name}: median {entry['median']['reason']} frames "
              f"{[(f['i'], f['reason']) for f in entry['frames']]} → {res['reason']}", flush=True)
    return out


# ----------------------------------------------------------------------------- sweep2 subset

SWEEP2_DIR = SWIFT_PACKAGES / "ShuttlCourt/TestData/sweep2"      # outside the test bundle (size)


def sweep2(max_cases: int = 72) -> None:
    """Exact-input parity on a stratified subset of calib_sweep2 (tools/venue/calib_sweep2/
    results_after.csv): every reject reason and hard kind, every grid, all false accepts,
    then accepted cases spread over the grid cells. Per case: the median image (planes),
    Python _calibrate_image on it (reason, camera, n_t/n_l/margin) and the full
    calibrate_from_frames outcome. Writes TestData/sweep2/manifest.json itself."""
    import csv
    src = REPO / "tools/venue/calib_sweep2/results_after.csv"
    if not src.exists():
        print(f"sweep2: skipped ({src} missing — run tools/venue/calib_sweep2/run.py first)")
        return None
    sys.path.insert(0, str(REPO / "tools/venue/calib_sweep2"))
    import cases as s2c
    import scenes as s2s
    rows = list(csv.DictReader(open(src)))
    by_id = {c["id"]: c for c in s2c.build()}
    rng = np.random.default_rng(2026)
    pick: list[int] = []
    strata: dict[tuple, list[int]] = {}
    for r in rows:
        acc = r["accepted"] == "True"
        fa = acc and float(r["floor_max_m"] or 0) > 0.10
        key = (r["grid"], r["kind"], "FA" if fa else ("acc" if acc else r["reason"]))
        strata.setdefault(key, []).append(int(r["id"]))
    for key in sorted(strata):                       # >= 1 per stratum, rejects/FA: up to 4
        ids = sorted(strata[key])
        k = 4 if key[2] != "acc" else 2
        pick += [int(i) for i in rng.choice(ids, min(k, len(ids)), replace=False)]
    # step 32-35 decisions (former wrong cameras → reject / right; recovered accepts)
    pick += [i for i in (576, 1426, 508, 507, 421, 384, 1192, 395, 56, 1937, 2034, 128, 471)
             if i not in pick]
    rest = [int(r["id"]) for r in rows if int(r["id"]) not in pick]
    pick += [int(i) for i in rng.choice(rest, max(0, max_cases - len(pick)), replace=False)]
    SWEEP2_DIR.mkdir(parents=True, exist_ok=True)
    out = {"cases": []}
    for i in sorted(set(pick)):
        c = by_id[i]
        cam = s2c.make_camera(c["height_m"], c["dist_m"], c["lat_m"], c["roll_deg"], c["pitch_deg"], c["focal"])
        frames = s2s.render(c, cam)
        med = cc._to_image(frames, (s2c.W, s2c.H))
        d: dict = {}
        mc = cc._calibrate_image(med, (s2c.W, s2c.H), d)
        dfull: dict = {}
        full = cc.calibrate_from_frames(frames, (s2c.W, s2c.H), diagnostics=dfull)
        out["cases"].append({"id": i, "grid": c["grid"], "kind": c["kind"], "size": [s2c.W, s2c.H],
                             "true": cam.to_dict(), "planes": save_planes(med, f"s2_{i}.deflate", SWEEP2_DIR),
                             "median": {"reason": d.get("reason"), "camera": None if mc is None else mc.to_dict(),
                                        "n_t": d.get("n_t"), "n_l": d.get("n_l"), "margin": d.get("margin"),
                                        "n_valid": d.get("n_valid")},
                             "full": {"reason": dfull.get("reason"), "source": dfull.get("source"),
                                      "accepted": full is not None}})
        print(f"  sweep2 #{i} {c['grid']}/{c['kind']}: median {d.get('reason')}, full {dfull.get('reason')}",
              flush=True)
    # R7 phonesim clips (tools/venue/r7set, frames in /tmp/r7set): Swift spot-check on the
    # phone-view set — ids 100000+k, grid "r7", kind = config
    import glob
    from pathlib import Path
    import json as _json
    r7 = REPO / "tools/venue/r7set/manifest.json"
    if r7.exists():
        man = _json.loads(r7.read_text())
        for k, name in enumerate(("doubles_s11", "doubles_s16", "hard_lights_s23", "closeup_s43",
                                  "multicourt_s31", "batch_013")):
            d = Path(man["frames_dir"]) / name
            files = sorted(glob.glob(str(d / "frame_*.jpg")))
            if not files:
                continue
            frames = [cv2.imread(f) for f in files]
            size = (frames[0].shape[1], frames[0].shape[0])
            med = cc._to_image(frames, size)
            dd: dict = {}
            mc = cc._calibrate_image(med, size, dd)
            cfg = next(c for c in man["cases"] if c["name"] == name)
            out["cases"].append({"id": 100000 + k, "grid": "r7", "kind": name, "size": list(size),
                                 "true": cfg["camera"], "planes": save_planes(med, f"r7_{name}.deflate", SWEEP2_DIR),
                                 "median": {"reason": dd.get("reason"), "camera": None if mc is None else mc.to_dict(),
                                            "n_t": dd.get("n_t"), "n_l": dd.get("n_l"), "margin": dd.get("margin"),
                                            "n_valid": dd.get("n_valid")},
                                 "full": None})
            print(f"  r7 {name}: median {dd.get('reason')}", flush=True)
    dump(out, SWEEP2_DIR / "manifest.json")
    print(f"sweep2: {len(out['cases'])} cases → {SWEEP2_DIR}")


# ----------------------------------------------------------------------------- real clips

def real() -> None:
    """Python calibrate_from_frames on the pipeline's 24 sampled frames of every Test clip.

    Runs ONE clip at a time (the machine is shared) and saves after each clip, so an
    interrupted run resumes where it stopped. Writes real.json itself (returns None)."""
    import json
    path = OUT / "real.json"
    done = json.loads(path.read_text()) if path.exists() else {"clips": []}
    have = {c["clip"] for c in done["clips"]}
    for clip in sorted(TEST_SPLIT.glob("match*/video/*.mp4")):
        rel = str(clip.relative_to(REPO))
        if rel in have:
            continue
        meta = video_io.probe(str(clip))
        size = (meta.width, meta.height)
        t0 = time.perf_counter()
        sampled = video_io.sample_frames(str(clip), 24, meta.n_frames)
        t1 = time.perf_counter()
        diag: dict = {}
        cam = cc.calibrate_from_frames([f for _, f in sampled], size, diagnostics=diag)
        t2 = time.perf_counter()
        done["clips"].append({
            "clip": rel, "size": list(size), "n_frames": meta.n_frames,
            "indices": [i for i, _ in sampled], "accepted": cam is not None,
            "camera": None if cam is None else cam.to_dict(),
            "diag": {k: v for k, v in diag.items() if not isinstance(v, (list, dict))},
            "decode_s": t1 - t0, "calibrate_s": t2 - t1})
        dump(done, path)
        print(f"{rel}: {'accepted' if cam is not None else 'rejected'} in {t2 - t1:.1f}s", flush=True)
    return None


# ----------------------------------------------------------------------------- colour lines

COLOUR_DIR = SWIFT_PACKAGES / "ShuttlCourt/TestData/colour"      # outside the test bundle (size)


def colour() -> None:
    """Exact-input parity of the line paint colour models: full-RGB medians (raw DEFLATE,
    RGB interleaved = the Swift ImageU8 order) of the R7 yellow-line clips, sweep2 grid
    'colour' cases (yellow / light blue / red / orange) and a white control. Per case and
    per colour model: whiteness checksum, mask pixel count, _calibrate_image reason and
    camera, and the model-loop outcome (first model that accepts). Writes
    TestData/colour/manifest.json itself."""
    import glob
    import json as _json
    import zlib
    from pathlib import Path
    sys.path.insert(0, str(REPO / "tools/venue/calib_sweep2"))
    import cases as s2c
    import scenes as s2s
    COLOUR_DIR.mkdir(parents=True, exist_ok=True)
    meds: list[tuple[str, np.ndarray, dict]] = []
    man = _json.loads((REPO / "tools/venue/r7set/manifest.json").read_text())
    for name in ("batch_006", "batch_014", "batch_016", "batch_018", "batch_013", "default_s7"):
        files = sorted(glob.glob(str(Path(man["frames_dir"]) / name / "frame_*.jpg")))
        if files:
            frames = [cv2.imread(f) for f in files]
            cfg = next(c for c in man["cases"] if c["name"] == name)
            meds.append((f"r7_{name}", cc._to_image(frames, (frames[0].shape[1], frames[0].shape[0])), cfg["camera"]))
    col = s2c.build(("colour",))
    for line in ("yellow", "light_blue", "red", "orange"):
        for c in [x for x in col if x["line"] == line][:2]:
            cam = s2c.make_camera(c["height_m"], c["dist_m"], c["lat_m"], c["roll_deg"], c["pitch_deg"], c["focal"])
            meds.append((f"s2_{c['id']}_{line}", cc._to_image(s2s.render(c, cam), (s2c.W, s2c.H)), cam.to_dict()))
    out = {"cases": []}
    for name, med, true in meds:
        co = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = co.compress(np.ascontiguousarray(med[..., ::-1]).tobytes()) + co.flush()
        (COLOUR_DIR / f"{name}.deflate").write_bytes(raw)
        size = (med.shape[1], med.shape[0])
        per = {}
        for colour_ in cc.LINE_COLOURS:
            w = cc._whiteness(med, colour_)
            feat = cc._Features(med, colour_)
            d: dict = {}
            cam = cc._calibrate_image(med, size, d, colour=colour_)
            per[colour_] = {"whiteness_sum": float(w.astype(np.float64).sum()), "whiteness_min": float(w.min()),
                            "whiteness_max": float(w.max()), "mask_px": int((feat.mask > 0).sum()),
                            "reason": d.get("reason"), "camera": None if cam is None else cam.to_dict(),
                            "paint_fit": None if cam is None else
                            {k: v for k, v in cc._paint_fit(med, cam).items() if np.isfinite(v)}}
        first = next((c_ for c_ in cc.LINE_COLOURS if per[c_]["camera"] is not None), None)
        # manual path: 4 corner taps with ±2 px noise under the TRUE camera → refine_camera
        true_cam = CameraModel.from_dict(true)
        names, P = cm.keypoints_array()
        kp = dict(zip(names, true_cam.project(P)))
        trng = np.random.default_rng(sum(map(ord, name)))          # deterministic per case
        taps = {n: [float(v) for v in kp[n] + trng.uniform(-2, 2, 2)] for n in ("nb_ld", "nb_rd", "fb_ld", "fb_rd")}
        tapped = cc.calibrate_from_points({k: tuple(v) for k, v in taps.items()}, size)
        refined = cc.refine_camera(tapped, [med])
        refine = {"taps": taps, "tapped": tapped.to_dict(), "paint_fit": {k: v for k, v in cc._paint_fit(med, tapped).items()
                                                                          if np.isfinite(v)},
                  "refined": refined.to_dict(), "moved": refined is not tapped}
        dl: dict = {}
        lc = cc.calibrate_from_frames([med], size, diagnostics=dl)      # 1 frame: no fallback
        out["cases"].append({"name": name, "size": list(size), "planes": f"{name}.deflate", "true": true,
                             "per_colour": per, "first_accept": first, "refine": refine,
                             "loop": {"reason": dl.get("reason"), "line_colour": dl.get("line_colour"),
                                      "paint_fit": dl.get("paint_fit"),
                                      "camera": None if lc is None else lc.to_dict()}})
        print(f"  colour {name}: " + ", ".join(f"{k} {v['reason']}" for k, v in per.items()), flush=True)
    dump(out, COLOUR_DIR / "manifest.json")
    print(f"colour: {len(out['cases'])} cases → {COLOUR_DIR}")
    return None


# ----------------------------------------------------------------------------- ShuttlLive

def calibration_attempt() -> None:
    """ShuttlLive CalibrationAttemptTests fixture: Python calibrate_from_frames diagnostics
    (key set; "candidates" dropped) on the pipeline's 24 sampled frames of two Test clips,
    plus a blank 1280x720 frame x3. Writes ShuttlLive/Tests/ShuttlLiveTests/Fixtures/
    calibration_attempt.json itself."""
    out = {"cases": []}
    for rel in ("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4",
                "backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/2_02_07.mp4"):
        path = str(REPO / rel)
        meta = video_io.probe(path)
        frames = [f for _, f in video_io.sample_frames(path, 24, meta.n_frames)]
        d: dict = {}
        cam = cc.calibrate_from_frames(frames, (meta.width, meta.height), diagnostics=d)
        d.pop("candidates", None)
        out["cases"].append({"relative_path": rel, "accepted": cam is not None, "diagnostics": d})
        print(f"  calibration_attempt {rel[-12:]}: {d.get('reason')} keys {sorted(d)}", flush=True)
    d = {}
    blank = [np.full((720, 1280, 3), 90, np.uint8)] * 3
    cam = cc.calibrate_from_frames(blank, (1280, 720), diagnostics=d)
    out["cases"].append({"blank": True, "relative_path": None, "accepted": cam is not None, "diagnostics": d})
    dump(out, SWIFT_PACKAGES / "ShuttlLive/Tests/ShuttlLiveTests/Fixtures/calibration_attempt.json")
    return None


# ----------------------------------------------------------------------------- main

SECTIONS = {"primitives": primitives, "manual": manual, "images": images, "gates": gates,
            "fallback": fallback, "sweep2": sweep2, "real": real,
            "calibration_attempt": calibration_attempt, "colour": colour}


def main(argv: list[str]) -> None:
    names = argv or list(SECTIONS)
    for n in names:
        t0 = time.time()
        res = SECTIONS[n]()
        if res is not None:
            p = dump(res, OUT / f"{n}.json")
            print(f"{n}: {p.stat().st_size / 1e3:.0f} kB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
