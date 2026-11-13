"""Reference values for ShuttlCore (camera, physics, court model, numerics, video)."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.tracking import court_model as cm
from app.services.tracking import physics, video_io
from app.services.tracking.camera import CameraModel
from scripts.swift_fixtures.common import REPO, SWIFT_PACKAGES, TEST_SPLIT, dump

OUT = SWIFT_PACKAGES / "ShuttlCore/Tests/ShuttlCoreTests/Fixtures"


def camera_cases() -> list[dict]:
    rng = np.random.default_rng(7)
    cams = [
        CameraModel.look_at([-11.5, 0.4, 3.2], [0.5, 0.0, 0.0], 1450.0, (1920, 1080)),
        CameraModel.look_at([-9.0, -2.5, 1.4], [1.0, 0.3, 0.8], 1100.0, (1280, 720), roll_deg=2.5),
    ]
    distorted = CameraModel.look_at([-10.0, 1.0, 2.6], [0.0, 0.0, 0.0], 1300.0, (1920, 1080))
    distorted.dist = np.array([0.08, -0.21, 0.0012, -0.0007, 0.05])
    cams.append(distorted)
    out = []
    for cam in cams:
        world = np.column_stack([rng.uniform(-6.7, 6.7, 12), rng.uniform(-3, 3, 12), rng.uniform(0, 5, 12)])
        world = np.vstack([world, [[-12.0, 0.0, 3.0]]])      # behind-ish point
        uv = cam.project(world)
        pix = np.column_stack([rng.uniform(0, cam.image_size[0], 10), rng.uniform(0, cam.image_size[1], 10)])
        o, d = cam.ray(pix)
        floor = cam.pixel_to_floor(pix)
        floor_z1 = cam.pixel_to_floor(pix, z=1.0)
        out.append({"camera": cam.to_dict(), "world": world, "uv": uv, "pixels": pix,
                    "ray_origin": o, "ray_dirs": d, "floor": floor, "floor_z1": floor_z1,
                    "center": cam.center, "floor_homography": cam.floor_homography()})
    return out


def physics_cases() -> list[dict]:
    cases = []
    for p0, v0 in [([-5.0, 0.5, 2.6], [30.0, -1.0, 8.0]),
                   ([-1.0, 0.0, 2.8], [60.0, 2.0, -12.0]),
                   ([4.5, -1.5, 0.4], [-12.0, 1.5, 9.0])]:
        times = np.array([-0.05, 0.0, 0.01, 0.1, 0.35, 0.8, 1.5])
        P, V = physics.simulate_state(np.array(p0), np.array(v0), np.sort(np.abs(times)))
        Pn, Vn = physics.simulate_state(np.array(p0), np.array(v0), -np.sort(np.abs(times)))
        cases.append({"p0": p0, "v0": v0, "times": np.sort(np.abs(times)), "P": P, "V": V,
                      "neg_times": -np.sort(np.abs(times)), "P_neg": Pn, "V_neg": Vn,
                      "t_ground": physics.time_to_ground(np.array(p0), np.array(v0))})
    return cases


def court_cases() -> dict:
    names, pts = cm.keypoints_array()
    probes = [(0.0, 0.0), (6.7, 2.59), (6.73, 0.0), (-6.8, 1.0), (3.0, 2.61), (3.0, 3.06), (-2.0, -3.1)]
    return {"names": names, "points": pts,
            "lines": {k: [list(a), list(b)] for k, (a, b) in cm.COURT_LINES.items()},
            "margins": [{"x": x, "y": y, "singles": cm.in_out_margin(x, y, False),
                         "doubles": cm.in_out_margin(x, y, True)} for x, y in probes]}


def numerics_cases() -> dict:
    rng = np.random.default_rng(3)
    xs = rng.normal(size=37)
    a = rng.normal(size=(9, 4))
    b = rng.normal(size=9)
    sym = rng.normal(size=(5, 5)); sym = sym + sym.T
    dlt = rng.normal(size=(8, 9))
    xp = np.sort(rng.uniform(0, 10, 12)); fp = rng.normal(size=12)
    probe = np.array([-1.0, 0.0, 2.5, 5.0, 9.99, 11.0])
    u, s, vt = np.linalg.svd(dlt)
    return {
        "xs": xs, "median": float(np.median(xs)),
        "percentiles": {str(q): float(np.percentile(xs, q)) for q in (0, 5, 25, 50, 75, 90, 95, 100)},
        "std": float(np.std(xs)), "std_ddof1": float(np.std(xs, ddof=1)),
        "xp": xp, "fp": fp, "probe": probe, "interp": np.interp(probe, xp, fp),
        "lstsq_A": a, "lstsq_b": b, "lstsq_x": np.linalg.lstsq(a, b, rcond=None)[0],
        "sym": sym, "eigvalsh": np.linalg.eigvalsh(sym),
        "solve_x": np.linalg.solve(sym, b[:5]),
        "dlt": dlt, "dlt_singular": s, "dlt_null": vt[-1],
        "convolve_a": xs[:11], "convolve_v": [0.25, 0.5, 0.25], "convolve_same": np.convolve(xs[:11], [0.25, 0.5, 0.25], "same"),
        "convolve_v4": [0.1, 0.2, 0.3, 0.4], "convolve_same4": np.convolve(xs[:11], [0.1, 0.2, 0.3, 0.4], "same"),
    }


def video_case() -> dict | None:
    clips = sorted(TEST_SPLIT.glob("match1/video/*.mp4"))
    if not clips:
        return None
    path = clips[0]
    meta = video_io.probe(str(path))
    frames = video_io.sample_frames(str(path), 5, meta.n_frames)
    grays = []
    for idx, f in frames:
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA)
        grays.append({"index": idx, "gray_160x90": g.astype(int).ravel().tolist()})
    return {"relative_path": str(path.relative_to(REPO)),
            "fps": meta.fps, "width": meta.width, "height": meta.height, "n_frames": meta.n_frames,
            "timestamps_head": meta.timestamps[:10], "timestamps_tail": meta.timestamps[-5:],
            "samples": grays}


def main() -> None:
    dump(camera_cases(), OUT / "camera.json")
    dump(physics_cases(), OUT / "physics.json")
    dump(court_cases(), OUT / "court.json")
    dump(numerics_cases(), OUT / "numerics.json")
    v = video_case()
    if v:
        dump(v, OUT / "video.json")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
