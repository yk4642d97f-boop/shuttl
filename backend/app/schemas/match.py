import uuid
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class MatchCreate(BaseModel):
    id: uuid.UUID | None = None  # stable client ID makes create retries and an in-flight delete safe
    duration_seconds: float = 0.0


class MatchResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    status: str
    duration_seconds: float
    video_oss_key: Optional[str] = None
    celery_job_id: Optional[str] = None
    created_at: datetime


class UploadUrlResponse(BaseModel):
    upload_url: str
    oss_key: str


class TrajectoryPointSchema(BaseModel):
    frame: int
    x_px: float
    y_px: float
    x_meters: float
    y_meters: float
    speed_kmh: float
    shot_type: str


class TrajectoryResponseSchema(BaseModel):
    match_id: uuid.UUID
    fps: float
    points: list[TrajectoryPointSchema]


class StatisticsResponseSchema(BaseModel):
    match_id: uuid.UUID
    total_rallies: int
    total_shots: int
    max_speed_kmh: float
    avg_speed_kmh: float
    shot_type_counts: dict[str, int]


class AnalyzeRequest(BaseModel):
    """Optional calibration hints from the app. Corner keys are court_model
    keypoint names (e.g. nb_ld, nb_rd, fb_ld, fb_rd) → [u, v] normalised 0–1."""
    corners: Optional[dict[str, tuple[float, float]]] = None
    corners_normalized: bool = True
    doubles: Optional[bool] = None
    camera: Optional[dict] = None
    force: bool = False  # completed result is otherwise reused; active jobs are never duplicated


class CancelRequest(BaseModel):
    job_id: uuid.UUID | None = None  # fence delayed cancellation against a newer analysis
