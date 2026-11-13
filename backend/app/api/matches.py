import asyncio
import logging
import uuid
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select, text
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_upload_token, decode_upload_token
from app.core.auth_deps import get_current_user_id
from app.models.match import Match, MatchStatus
from app.models.media_deletion import MediaDeletion
from app.models.user import User
from app.schemas.match import AnalyzeRequest, CancelRequest, MatchCreate, MatchResponse, UploadUrlResponse
from app.services.storage import local_storage

logger = logging.getLogger(__name__)
router = APIRouter()
_oss = None


def _oss_service():
    """Created lazily so local-storage mode never touches oss2 credentials."""
    global _oss
    if _oss is None:
        from app.services.storage.oss_service import OSSService
        _oss = OSSService()
    return _oss


@router.post("", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
async def create_match(
    payload: MatchCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    mid = payload.id or uuid.uuid4()
    await _lifecycle_lock(db, mid)
    if await db.get(MediaDeletion, mid) is not None:
        raise HTTPException(409, '比赛已删除，不能重新创建')
    prior = await db.get(Match, mid)
    if prior is not None:
        if prior.user_id != uuid.UUID(user_id):
            raise HTTPException(409, '比赛编号已被使用')
        await db.commit()
        return MatchResponse.model_validate(prior)
    match = Match(
        id=mid,
        user_id=uuid.UUID(user_id),
        duration_seconds=payload.duration_seconds,
        status=MatchStatus.uploading,
    )
    db.add(match)
    await db.commit()
    await db.refresh(match)
    return MatchResponse.model_validate(match)


async def _lifecycle_lock(db, match_id):
    # Separate from the worker lock: deletion must not wait for a long GPU stage.
    # Unlike a row lock, this also serializes create/delete when no row exists yet.
    await db.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))'),
                     {'key': f'shuttl-match-lifecycle:{match_id}'})


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(
    match_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    match = await db.scalar(
        select(Match)
        .where(Match.id == match_id, Match.user_id == uuid.UUID(user_id))
        .options(selectinload(Match.rallies))
    )
    if not match:
        raise HTTPException(status_code=404, detail="比赛记录不存在")
    return MatchResponse.model_validate(match)


@router.post("/{match_id}/upload", response_model=UploadUrlResponse)
async def get_upload_url(
    match_id: uuid.UUID,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    match = await db.scalar(
        select(Match).where(Match.id == match_id, Match.user_id == uuid.UUID(user_id)).with_for_update()
    )
    if not match:
        raise HTTPException(status_code=404, detail="比赛记录不存在")
    oss_key = f"videos/{user_id}/{match_id}.mov"
    if settings.use_local_storage:
        # Absolute URL on this server, built from the host the client actually used.
        token = create_upload_token(str(match_id), user_id, oss_key)
        base = (settings.public_base_url or str(request.base_url)).rstrip("/")
        upload_url = f"{base}/matches/{match_id}/video?token={quote(token)}"
    else:
        upload_url = _oss_service().generate_presigned_put_url(oss_key)
    match.video_oss_key = oss_key
    await db.commit()
    return UploadUrlResponse(upload_url=upload_url, oss_key=oss_key)


@router.put("/{match_id}/video")
async def put_video(match_id: uuid.UUID, request: Request, token: str = Query(...),
                    db: AsyncSession = Depends(get_db)):
    """Local-storage upload target (replaces the OSS presigned PUT). Auth is the
    signed, expiring, match-bound token in the query string; body is the raw video."""
    if not settings.use_local_storage:
        raise HTTPException(status_code=404, detail="本地存储未启用")
    try:
        claims = decode_upload_token(token, str(match_id))
        owner = uuid.UUID(claims['sub'])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="上传凭证无效或已过期")
    owned = select(Match).where(Match.id == match_id, Match.user_id == owner,
                               Match.video_oss_key == claims['key'])
    if await db.scalar(owned) is None:
        raise HTTPException(404, '比赛记录已删除或上传已失效')
    # No database lock/transaction held while the client streams a long recording.
    await db.rollback()
    try:
        async with local_storage.stage_stream(claims['key'], request.stream()) as staged:
            if await db.scalar(owned.with_for_update()) is None:
                raise HTTPException(404, '比赛记录已删除或上传已失效')
            size = await staged.publish()
            await db.commit()
    except local_storage.UploadTooLarge:
        raise HTTPException(status_code=413, detail="视频文件过大")
    except local_storage.UploadEmpty:
        raise HTTPException(status_code=400, detail="上传内容为空")
    except local_storage.UploadConflict:
        raise HTTPException(status_code=409, detail="该比赛已上传其他录像，请创建新比赛后上传")
    except ValueError:
        raise HTTPException(status_code=400, detail="非法存储路径")
    logger.info("stored %s (%d bytes)", claims["key"], size)
    return {"oss_key": claims["key"], "size": size}


@router.post("/{match_id}/analyze")
async def trigger_analysis(
    match_id: uuid.UUID,
    payload: AnalyzeRequest | None = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key", min_length=1, max_length=128),
):
    match = await db.scalar(
        select(Match).where(Match.id == match_id, Match.user_id == uuid.UUID(user_id)).with_for_update()
    )
    if not match:
        raise HTTPException(status_code=404, detail="比赛记录不存在")
    if not match.video_oss_key:
        raise HTTPException(status_code=400, detail="视频尚未上传")
    if settings.use_local_storage and not local_storage.path_for_key(match.video_oss_key).is_file():
        raise HTTPException(status_code=400, detail="视频尚未上传")

    video_identity = None
    if not settings.use_local_storage:
        try:
            video_identity = await asyncio.to_thread(_oss_service().input_identity, match.video_oss_key)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.warning('Cannot verify uploaded video revision', exc_info=True)
            raise HTTPException(503, '暂时无法核验上传录像，请稍后重试') from exc

    from app.services import analysis_jobs
    payload = payload or AnalyzeRequest()
    # Compare the same JSON representation that the database stores (corners are lists).
    calibration = payload.model_dump(mode="json", exclude_none=True, exclude={"force"})
    job = await analysis_jobs.submit(db, match, calibration, force=payload.force,
                                     idempotency_key=idempotency_key, video_identity=video_identity)
    # Submission was committed already. If delivery fails, the dispatcher retries it.
    if job.state in analysis_jobs.ACTIVE:
        try:
            await asyncio.to_thread(analysis_jobs.dispatch_due, job.id)
        except Exception:
            logger.warning("Durable analysis %s awaits dispatcher recovery", job.id, exc_info=True)
    return {"job_id": str(job.id), "status": job.state}


@router.post("/{match_id}/cancel")
async def cancel_analysis(
    match_id: uuid.UUID,
    payload: CancelRequest | None = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    from app.services import analysis_jobs
    match = await db.scalar(select(Match).where(
        Match.id == match_id, Match.user_id == uuid.UUID(user_id)).with_for_update())
    if match is None:
        raise HTTPException(404, "比赛记录不存在")
    return analysis_jobs.status(await analysis_jobs.cancel(db, match, expected_job_id=payload.job_id if payload else None))


@router.delete('/{match_id}', status_code=202)
async def delete_match(
    match_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Remove results and enqueue file cleanup atomically. Repeat requests are safe."""
    owner = uuid.UUID(user_id)
    await _lifecycle_lock(db, match_id)
    match = await db.scalar(select(Match).where(Match.id == match_id).with_for_update())
    if match is not None and match.user_id != owner:
        raise HTTPException(404, '比赛记录不存在')
    if match is None:
        tombstone = await db.get(MediaDeletion, match_id)
        if tombstone is not None and tombstone.user_id != owner:
            raise HTTPException(404, '比赛记录不存在')
        if tombstone is not None:
            return {'match_id': str(match_id), 'status': 'deleted', 'media_cleanup': tombstone.state}
    db.add(MediaDeletion(match_id=match_id, user_id=owner,
                         video_key=(match.video_oss_key if match else None) or f'videos/{owner}/{match_id}.mov',
                         storage_backend='local' if settings.use_local_storage else 'oss'))
    await db.flush()
    # DB cascade removes jobs/rallies/shots; live workers can no longer pass their write fence.
    if match is not None:
        await db.execute(delete(Match).where(Match.id == match.id))
    await db.commit()
    return {'match_id': str(match_id), 'status': 'deleted', 'media_cleanup': 'pending'}


@router.get("/{match_id}/job/{job_id}")
async def poll_job(
    match_id: uuid.UUID,
    job_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    match = await db.scalar(select(Match).where(
        Match.id == match_id, Match.user_id == uuid.UUID(user_id),
        Match.celery_job_id == job_id))
    if match is None:
        raise HTTPException(status_code=404, detail="分析任务不存在")
    from app.models.analysis_job import AnalysisJob
    from app.services import analysis_jobs
    try:
        job = await db.get(AnalysisJob, uuid.UUID(job_id))
    except ValueError:
        job = None
    if job:
        return analysis_jobs.status(job)
    # Pre-upgrade results can still be retrieved after Redis expires them.
    if match.status == MatchStatus.done:
        return {"job_id": job_id, "state": "SUCCESS", "info": None}
    from celery.result import AsyncResult
    from app.worker import celery_app
    result = AsyncResult(job_id, app=celery_app)
    return {"job_id": job_id, "state": result.state, "info": str(result.info)}
