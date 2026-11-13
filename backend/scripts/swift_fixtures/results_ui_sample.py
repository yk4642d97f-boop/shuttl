"""Export compact MatchAnalysis fixtures for ShuttlResultsUI.

Run from ``backend/`` one clip at a time:

    .venv/bin/python -m scripts.swift_fixtures.results_ui_sample \
        --clip match2/video/1_04_04.mp4

The analysis itself is kept in the ShuttlCore shape. Large per-frame shuttle
arrays, fitted flight samples, and pose keypoints are deliberately removed;
the results UI only needs rally events, metric flight endpoints, player feet,
and video metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

from app.services.tracking import players as players_module
from app.services.tracking import pipeline_v2, serialize
from scripts.swift_fixtures.common import REPO, SWIFT_PACKAGES, TEST_SPLIT, dump

OUT = SWIFT_PACKAGES / "ShuttlResultsUI/Tests/ShuttlResultsUITests/Fixtures"
DEFAULT_CLIP = "match2/video/1_04_04.mp4"


def _empty_segment(segment):
    """Keep segment identity and metric fields, but drop dense fitted samples."""
    return dataclasses.replace(
        segment,
        frames=np.empty(0, dtype=np.int64),
        positions=np.empty((0, 3), dtype=np.float64),
    )


def _compact_shot(shot):
    return dataclasses.replace(
        shot,
        segment=None if shot.segment is None else _empty_segment(shot.segment),
    )


def _compact_rally(rally):
    return dataclasses.replace(rally, shots=[_compact_shot(shot) for shot in rally.shots])


def compact_analysis(result: pipeline_v2.MatchAnalysis) -> pipeline_v2.MatchAnalysis:
    """Return a small, still-Codable MatchAnalysis for the Swift package."""
    return dataclasses.replace(
        result,
        track=None,
        players=[dataclasses.replace(player, keypoints=None) for player in result.players],
        hits=[],
        segments=[_empty_segment(segment) for segment in result.segments],
        rallies=[_compact_rally(rally) for rally in result.rallies],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        default=DEFAULT_CLIP,
        help="path relative to TrackNetV2/Test, or an absolute mp4 path",
    )
    parser.add_argument("--device", default=None, help="torch device, for example mps or cpu")
    parser.add_argument(
        "--weights-dir",
        default=str(REPO / "backend/ml/weights"),
        help="tracking model directory",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="fixture prefix; defaults to results_<match>_<clip>",
    )
    return parser.parse_args(argv)


def resolve_clip(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    path = TEST_SPLIT / path
    if path.is_file():
        return path
    raise FileNotFoundError(f"clip not found: {path}")


def progress(percent: float, stage: str) -> None:
    print(f"  {percent:5.1f}% {stage}", flush=True)


def run(args: argparse.Namespace) -> list[Path]:
    clip = resolve_clip(args.clip)
    print(f"analyzing {clip.relative_to(REPO)}", flush=True)
    result = pipeline_v2.analyze_video(
        str(clip),
        opts=pipeline_v2.PipelineOptions(
            weights_dir=args.weights_dir,
            device=args.device,
            player_step=2,
        ),
        progress=progress,
    )

    # Compute the Python reference values from the full pipeline result before
    # compacting it. These are the values the Swift view model must reproduce.
    summary = serialize.summarize(result.rallies)
    movement = {
        side: players_module.movement_stats(result.players, side)
        for side in ("near", "far")
    }
    compact = compact_analysis(result)

    stem = clip.stem.replace("-", "_")
    match = clip.parent.parent.name
    prefix = args.output_prefix or f"results_{match}_{stem}"
    paths = [
        dump(compact, OUT / f"{prefix}_analysis.json"),
        dump(summary, OUT / f"{prefix}_summary.json"),
        dump(movement["near"], OUT / f"{prefix}_movement_near.json"),
        dump(movement["far"], OUT / f"{prefix}_movement_far.json"),
    ]
    print(
        f"result rallies={len(result.rallies)} shots={sum(len(r.shots) for r in result.rallies)} "
        f"camera={result.camera is not None} players={len(result.players)}",
        flush=True,
    )
    for path in paths:
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)", flush=True)
    return paths


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
