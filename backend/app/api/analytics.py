import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from collections import Counter

from app.core.database import get_db
from app.core.auth_deps import get_current_user_id
from app.models.match import Match, Rally, ShotPoint, ShotType, MatchStatus
from app.schemas.match import TrajectoryPointSchema, TrajectoryResponseSchema, StatisticsResponseSchema

router = APIRouter()


@router.get("/{match_id}/trajectory", response_model=TrajectoryResponseSchema)
async def get_trajectory(
    match_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    match = await _get_match_or_404(match_id, user_id, db)
    if match.status != MatchStatus.done:
        raise HTTPException(status_code=400, detail="分析尚未完成")

    match_with_data = await db.scalar(
        select(Match)
        .where(Match.id == match_id)
        .options(selectinload(Match.rallies).selectinload(Rally.shots))
    )
    all_shots = [shot for rally in (match_with_data.rallies or []) for shot in rally.shots]
    all_shots.sort(key=lambda s: s.frame)

    points = [
        TrajectoryPointSchema(
            frame=s.frame,
            x_px=s.x_px, y_px=s.y_px,
            x_meters=s.x_meters, y_meters=s.y_meters,
            speed_kmh=s.speed_kmh,
            shot_type=s.shot_type.value,
        )
        for s in all_shots
    ]
    return TrajectoryResponseSchema(
        match_id=match_id,
        fps=match.fps or 60.0,
        points=points,
    )


@router.get("/{match_id}/analysis")
async def get_analysis(
    match_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Full v2 analysis document: rallies, per-shot 3-D flights, in/out calls,
    score, player movement. See services/tracking/serialize.py for the schema."""
    match = await _get_match_or_404(match_id, user_id, db)
    if match.status != MatchStatus.done:
        raise HTTPException(status_code=400, detail="分析尚未完成")
    if not match.analysis:
        raise HTTPException(status_code=404, detail="该比赛没有 v2 分析数据")
    return {"match_id": str(match_id), **{k: v for k, v in match.analysis.items() if k != "mobile_archive"}}


@router.get("/{match_id}/analysis/archive")
async def get_analysis_archive(
    match_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Native report archive, subject to the same ownership checks as every result."""
    match = await _get_match_or_404(match_id, user_id, db)
    if match.status != MatchStatus.done:
        raise HTTPException(status_code=400, detail="分析尚未完成")
    archive = (match.analysis or {}).get("mobile_archive")
    if archive is None:
        raise HTTPException(status_code=404, detail="该比赛没有完整分析存档")
    return archive


@router.get("/{match_id}/statistics", response_model=StatisticsResponseSchema)
async def get_statistics(
    match_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    match = await _get_match_or_404(match_id, user_id, db)
    if match.status != MatchStatus.done:
        raise HTTPException(status_code=400, detail="分析尚未完成")

    match_data = await db.scalar(
        select(Match)
        .where(Match.id == match_id)
        .options(selectinload(Match.rallies).selectinload(Rally.shots))
    )
    all_shots = [s for r in (match_data.rallies or []) for s in r.shots]
    speeds = [s.speed_kmh for s in all_shots if s.speed_kmh > 0]
    shot_type_counts = Counter(s.shot_type.value for s in all_shots)

    return StatisticsResponseSchema(
        match_id=match_id,
        total_rallies=len(match_data.rallies or []),
        total_shots=len(all_shots),
        max_speed_kmh=max(speeds, default=0.0),
        avg_speed_kmh=sum(speeds) / len(speeds) if speeds else 0.0,
        shot_type_counts=dict(shot_type_counts),
    )


async def _get_match_or_404(match_id: uuid.UUID, user_id: str, db: AsyncSession) -> Match:
    match = await db.scalar(
        select(Match).where(Match.id == match_id, Match.user_id == uuid.UUID(user_id))
    )
    if not match:
        raise HTTPException(status_code=404, detail="比赛记录不存在")
    return match
