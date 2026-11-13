"""
Analysis computation and transaction-scoped persistence helpers.
The durable job lifecycle is in app.services.analysis_jobs.

The full analysis document goes into matches.analysis (JSON); one ShotPoint
row per hit is kept in shot_points for the legacy /trajectory + /statistics
endpoints. psycopg2 (sync) throughout — Celery workers run synchronously.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras

from app.core.config import settings
from app.services.storage import local_storage
from app.services.tracking import serialize
from app.services.tracking.pipeline_v2 import MatchAnalysis, PipelineOptions, analyze_video
from app.worker import celery_app

logger = logging.getLogger(__name__)
psycopg2.extras.register_uuid()      # let psycopg2 bind uuid.UUID parameters


def _options(calibration: dict | None) -> PipelineOptions:
    calibration = calibration or {}
    return PipelineOptions(
        weights_dir=os.environ.get("SHUTTL_WEIGHTS_DIR"),
        device=os.environ.get("INFERENCE_DEVICE") or None,
        hitnet_weights=os.environ.get("SHUTTL_HITNET_WEIGHTS") or None,
        bst_manifest=os.environ.get("SHUTTL_BST_MANIFEST") or None,
        rtmpose_weights=os.environ.get("SHUTTL_RTMPOSE_WEIGHTS") or None,
        doubles=calibration.get("doubles"),
        manual_corners={k: tuple(v) for k, v in calibration.get("corners", {}).items()} or None,
        corners_normalized=calibration.get("corners_normalized", True),
        camera_hint=calibration.get("camera"),
    )


# Same wording and thresholds as the app's on-device verdict (ShuttlAnalysis AnalysisOutcome).
NO_SHUTTLE_REASON = "没有看到羽毛球。请确认整片球场都在画面里，光线充足。"
NO_RALLIES_REASON = "看到了羽毛球，但没有找到完整的回合。请拍下完整的来回，并让两边球员都在画面里。"


class AnalysisUnusable(Exception):
    """The analysis ran but found nothing to show. Deterministic: never retried."""


def unusable_reason(res: MatchAnalysis) -> str | None:
    """None when there are rallies; otherwise the sentence the app shows."""
    if res.rallies:
        return None
    n = res.meta.n_frames or 0
    needed = max(30, -(-n // 100))                   # 30 frames or 1 %, whichever is larger
    visible = int(res.track.visible.sum()) if res.track is not None else 0
    return NO_SHUTTLE_REASON if visible < needed else NO_RALLIES_REASON


def compute_analysis(video_oss_key: str, calibration: dict | None, progress, *, video_identity=None):
    """Compute without mutating job state; durable job runner owns persistence/retries."""
    tmp_path = None
    try:
        if settings.use_local_storage:
            video_path = str(local_storage.path_for_key(video_oss_key))
            if not Path(video_path).is_file():
                raise FileNotFoundError(f"local video missing: {video_path}")
        else:
            from app.services.storage.oss_service import OSSService, ObjectChanged
            if not video_identity:
                raise AnalysisUnusable('旧版任务缺少录像版本信息，请重新提交分析')
            with tempfile.NamedTemporaryFile(suffix=Path(video_oss_key).suffix or ".mov", delete=False,
                                             dir=local_storage.work_directory(video_oss_key)) as tmp:
                tmp_path = tmp.name
            progress(2, "download")
            try:
                OSSService().download_to_path(video_oss_key, tmp_path, identity=video_identity)
            except ObjectChanged as exc:
                raise AnalysisUnusable(str(exc)) from exc
            video_path = tmp_path
        # Keep the validated 5090 precision policy identical to the offline preflight.
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        res = analyze_video(video_path, _options(calibration),
                            progress=lambda p, stage: progress(int(5 + p * .9), stage))
        reason = unusable_reason(res)
        if reason is not None:
            raise AnalysisUnusable(reason)
        doc = serialize.to_dict(res)
        doc["mobile_archive"] = serialize.mobile_archive(res)
        return res, doc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@celery_app.task(name="app.services.tracking.tasks.analyze_match", ignore_result=True)
def analyze_match(*args, **kwargs):
    """Reject pre-upgrade queue messages instead of allowing an unfenced stale write."""
    raise RuntimeError("旧版分析任务已停用，请从比赛页面重新提交分析")


def _save_results(cur, match_id, res: MatchAnalysis, doc: dict) -> None:
    """Called inside the job runner's fenced transaction; never commits independently."""
    mid = uuid.UUID(str(match_id))
    w, h = res.meta.width or 1, res.meta.height or 1
    cur.execute("DELETE FROM rallies WHERE match_id = %s", (mid,))   # cascades shots
    for r in res.rallies:
        rally_id = uuid.uuid4()
        cur.execute(
            "INSERT INTO rallies (id, match_id, start_frame, end_frame, winner) "
            "VALUES (%s, %s, %s, %s, %s)",
            (rally_id, mid, r.start_frame, r.end_frame,
             r.winner if r.winner in ("near", "far") else "unknown"))
        rows = []
        for s in r.shots:
            seg = s.segment
            pos = s.hitter_pos_world
            if pos is None:
                pos = seg.p0[:2] if seg is not None else (0.0, 0.0)
            rows.append((uuid.uuid4(), rally_id, s.hit.frame,
                         s.hit.xy_px[0] / w, s.hit.xy_px[1] / h,
                         float(pos[0]), float(pos[1]),
                         seg.speed0_kmh if seg is not None else 0.0,
                         serialize.legacy_shot_type(s.shot_type)))
        if rows:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO shot_points (id, rally_id, frame, x_px, y_px, x_meters, "
                "y_meters, speed_kmh, shot_type) VALUES %s", rows)
    cur.execute(
        "UPDATE matches SET status = %s, duration_seconds = %s, fps = %s, analysis = %s "
        "WHERE id = %s",
        ("done", float(res.meta.timestamps[-1]) if res.meta.n_frames else 0.0,
         res.meta.fps, json.dumps(doc), mid))
