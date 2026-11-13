"""Transactional outbox, duplicate delivery guard, cooperative cancellation and durable status.

Run a dispatcher with `python -m app.services.analysis_jobs dispatch`. It retries
pending delivery even after Redis/API restarts. Workers receive only a job UUID.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import signal
import threading
import uuid

import psycopg2
import psycopg2.extras
from fastapi import HTTPException
from sqlalchemy import select

from app.core.config import settings
from app.models.analysis_job import AnalysisJob
from app.models.match import MatchStatus
from app.worker import celery_app

logger = logging.getLogger(__name__)
psycopg2.extras.register_uuid()
ACTIVE = {'queued', 'running', 'retry'}
MAX_ATTEMPTS = 3
DISPATCH_INTERVAL = 60
STALE_HEARTBEAT = 120
TASK_NAME = 'app.services.analysis_jobs.execute_analysis'


def connect():
    dsn = settings.database_url.replace('postgresql+asyncpg://', 'postgresql://').replace('postgresql+psycopg2://', 'postgresql://')
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor,
                            connect_timeout=5, keepalives=1, keepalives_idle=10,
                            keepalives_interval=5, keepalives_count=3,
                            options='-c statement_timeout=15000 -c lock_timeout=5000')


async def submit(db, match, calibration, *, force=False, idempotency_key=None, video_identity=None):
    """Caller holds this owned match's row lock; commit the job and pointer together."""
    payload = {'video_oss_key': match.video_oss_key, 'calibration': calibration}
    if video_identity is not None:
        payload['video_identity'] = video_identity
    if idempotency_key:
        prior = await db.scalar(select(AnalysisJob).where(
            AnalysisJob.match_id == match.id, AnalysisJob.idempotency_key == idempotency_key))
        if prior:
            if prior.payload != payload:
                raise HTTPException(409, '同一请求编号不能用于不同的分析参数')
            if match.celery_job_id != str(prior.id):
                raise HTTPException(409, '该请求已被更新的分析任务替代')
            await db.commit()
            return prior
    prior = None
    if match.celery_job_id:
        try:
            prior = await db.get(AnalysisJob, uuid.UUID(match.celery_job_id))
        except ValueError:
            pass  # pre-upgrade legacy Celery identifier
    if prior and (prior.state in ACTIVE or (prior.state == 'done' and not force)):
        if idempotency_key and prior.idempotency_key != idempotency_key:
            raise HTTPException(409, '比赛已有分析任务，请查询现有任务，或结束后明确重新分析')
        if prior.payload != payload:
            raise HTTPException(409, '已有分析使用不同参数，请先取消进行中的任务，或明确重新分析')
        await db.commit()
        return prior
    job = AnalysisJob(id=uuid.uuid4(), match_id=match.id, payload=payload,
                      idempotency_key=idempotency_key)
    # An old completed match stays completed unless reanalysis was explicit.
    if prior is None and match.status == MatchStatus.done and not force:
        job.state = 'done'
        job.stage = 'legacy_done'
        job.percent = 100
        job.finished_at = datetime.now(timezone.utc)
    else:
        match.status = MatchStatus.analyzing
    db.add(job)
    match.celery_job_id = str(job.id)
    await db.commit()
    return job


async def cancel(db, match, *, expected_job_id=None):
    """Caller owns and locks match. Result commit and cancellation serialize on it."""
    if not match.celery_job_id:
        raise HTTPException(409, '没有可取消的分析任务')
    if expected_job_id is not None and match.celery_job_id != str(expected_job_id):
        raise HTTPException(409, '该分析任务已被更新的任务替代')
    try:
        job = await db.get(AnalysisJob, uuid.UUID(match.celery_job_id), with_for_update=True)
    except ValueError:
        job = None
    if not job:
        raise HTTPException(409, '旧版任务需要重新提交')
    if job.state == 'done':
        raise HTTPException(409, '分析已完成')
    if job.state in ACTIVE:
        job.state = 'cancelled'
        job.stage = 'cancelled'
        job.error = '分析已取消'
        job.finished_at = datetime.now(timezone.utc)
        match.status = MatchStatus.failed
    await db.commit()
    return job


def status(job):
    state = {'queued':'PENDING', 'running':'PROGRESS', 'retry':'RETRY',
             'done':'SUCCESS', 'failed':'FAILURE', 'cancelled':'REVOKED'}[job.state]
    info = job.error if job.state in {'failed','cancelled'} else json.dumps({
        'match_id':str(job.match_id), 'percent':job.percent, 'stage':job.stage,
        'attempt':job.attempt}, ensure_ascii=False)
    return {'job_id':str(job.id), 'state':state, 'info':info}


def publish(job_id):
    celery_app.send_task(TASK_NAME, args=[str(job_id)], task_id=str(job_id), retry=False)


def dispatch_due(job_id=None, *, publisher=publish, connection_factory=connect):
    """Publish committed rows. A crash after publish may redeliver; worker fences it."""
    conn = connection_factory()
    sent = 0
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT j.id FROM analysis_jobs j JOIN matches m ON m.id=j.match_id
                WHERE m.celery_job_id=j.id::text AND
                  ((j.state IN ('queued','retry') AND j.available_at<=now()) OR
                   (j.state='running' AND j.heartbeat_at<now() - %s * interval '1 second'))
                  AND (j.dispatched_at IS NULL OR j.dispatched_at<now() - %s * interval '1 second')
                  AND (%s::uuid IS NULL OR j.id=%s::uuid)
                ORDER BY j.created_at LIMIT 20 FOR UPDATE OF j SKIP LOCKED
                """, (STALE_HEARTBEAT,DISPATCH_INTERVAL,job_id,job_id))
            ids = [row['id'] for row in cur.fetchall()]
            for ident in ids:
                try:
                    publisher(ident)
                    sent += 1
                except Exception:
                    logger.warning('Queue unavailable; durable request retained', exc_info=True)
                    # Record a delivery attempt, not success, to bound retry churn.
                    cur.execute('UPDATE analysis_jobs SET dispatched_at=now() WHERE id=%s', (ident,))
                    break
                cur.execute('UPDATE analysis_jobs SET dispatched_at=now() WHERE id=%s', (ident,))
        return sent
    finally:
        conn.close()


class JobInterrupted(Exception):
    """Cancellation, superseded request, or lost database ownership."""


def lock_key(match_id):
    raw = hashlib.blake2b(('shuttl-analysis:'+str(match_id)).encode(), digest_size=8).digest()
    return int.from_bytes(raw, 'big', signed=True)


def locked_current(cur, job):
    # All state-changing code locks match before job to avoid inverse lock order.
    cur.execute('SELECT celery_job_id FROM matches WHERE id=%s FOR UPDATE', (job['match_id'],))
    match = cur.fetchone()
    if not match or match['celery_job_id'] != str(job['id']):
        raise JobInterrupted('Analysis request was superseded or deleted')
    cur.execute('SELECT * FROM analysis_jobs WHERE id=%s FOR UPDATE', (job['id'],))
    current = cur.fetchone()
    if not current or current['state'] != 'running' or current['claim_token'] != job['claim_token']:
        raise JobInterrupted('Analysis request is no longer running under this claim')
    return current


def claim(conn, job_id):
    with conn, conn.cursor() as cur:
        cur.execute('SELECT match_id FROM analysis_jobs WHERE id=%s', (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        key = lock_key(row['match_id'])
        cur.execute('SELECT pg_try_advisory_lock(%s) AS acquired', (key,))
        if not cur.fetchone()['acquired']:
            return None
        cur.execute('SELECT celery_job_id FROM matches WHERE id=%s FOR UPDATE', (row['match_id'],))
        match = cur.fetchone()
        cur.execute('SELECT *, available_at<=now() AS available FROM analysis_jobs WHERE id=%s FOR UPDATE', (job_id,))
        job = cur.fetchone()
        if not job or not match or match['celery_job_id'] != str(job_id) or job['state'] not in ACTIVE:
            return None
        if job['state'] != 'running' and not job['available']:
            return None
        if job['attempt'] >= MAX_ATTEMPTS:
            cur.execute("UPDATE analysis_jobs SET state='failed',stage='failed',error=%s,finished_at=now() WHERE id=%s",
                        ('分析多次中断，请重试', job_id))
            cur.execute("UPDATE matches SET status='failed' WHERE id=%s", (job['match_id'],))
            return None
        token = uuid.uuid4()
        cur.execute("""UPDATE analysis_jobs SET state='running',stage='starting',percent=0,
                       attempt=attempt+1,claim_token=%s,heartbeat_at=now(),error=NULL
                       WHERE id=%s RETURNING *""", (token,job_id))
        return cur.fetchone()


class RunningJob:
    def __init__(self, conn, job):
        self.conn, self.job = conn, job
        self.guard = threading.RLock()
        self.stop = threading.Event()
        self.interrupted = threading.Event()
        self.last_progress = None
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)

    def _touch(self, percent=None, stage=None):
        with self.guard, self.conn, self.conn.cursor() as cur:
            cur.execute("""UPDATE analysis_jobs SET heartbeat_at=now(),
                       percent=COALESCE(%s,percent),stage=COALESCE(%s,stage)
                       WHERE id=%s AND state='running' AND claim_token=%s
                       AND EXISTS (SELECT 1 FROM matches m WHERE m.id=analysis_jobs.match_id
                                   AND m.celery_job_id=analysis_jobs.id::text)
                       RETURNING id""", (percent,stage,self.job['id'],self.job['claim_token']))
            if not cur.fetchone():
                raise JobInterrupted('Analysis cancelled, superseded or deleted')

    def _heartbeat_loop(self):
        while not self.stop.wait(5):
            try:
                self._touch()
            except Exception:
                self.interrupted.set()
                return

    def progress(self, percent, stage):
        if self.interrupted.is_set():
            raise JobInterrupted('Analysis lost its database claim')
        value = (min(99,max(0,int(percent))),stage[:64])
        if self.last_progress != value:
            self._touch(*value)
            self.last_progress = value

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)


def execute(job_id, *, compute=None, connection_factory=connect):
    """One actual attempt. Duplicate deliveries do no model work; results are fenced."""
    from app.services.tracking import tasks
    job_id = uuid.UUID(str(job_id))
    conn = connection_factory()
    job = None
    try:
        job = claim(conn,job_id)
        if job is None:
            return {'skipped':True}
        with RunningJob(conn,job) as running:
            try:
                args = (job['payload']['video_oss_key'],job['payload']['calibration'],running.progress)
                if compute is None:
                    res,doc = tasks.compute_analysis(*args, video_identity=job['payload'].get('video_identity'))
                else:
                    res,doc = compute(*args)
                running.progress(96,'persist')
                with running.guard, conn, conn.cursor() as cur:
                    locked_current(cur,job)
                    tasks._save_results(cur,job['match_id'],res,doc)
                    cur.execute("""UPDATE analysis_jobs SET state='done',stage='done',percent=100,
                                   finished_at=now(),heartbeat_at=now() WHERE id=%s""", (job_id,))
                return {'status':'done','job_id':str(job_id)}
            except JobInterrupted:
                return {'status':'interrupted','job_id':str(job_id)}
            except Exception as exc:
                logger.exception('Analysis job %s attempt %s failed',job_id,job['attempt'])
                deterministic = isinstance(exc,(FileNotFoundError,tasks.AnalysisUnusable))
                terminal = deterministic or job['attempt'] >= MAX_ATTEMPTS
                try:
                    with running.guard, conn, conn.cursor() as cur:
                        locked_current(cur,job)
                        cur.execute("""UPDATE analysis_jobs SET state=%s,stage=%s,error=%s,
                            available_at=now()+interval '30 seconds',dispatched_at=NULL,
                            finished_at=CASE WHEN %s THEN now() ELSE NULL END WHERE id=%s""",
                            ('failed' if terminal else 'retry','failed' if terminal else 'retry',
                             str(exc)[:2000],terminal,job_id))
                        if terminal:
                            cur.execute("UPDATE matches SET status='failed' WHERE id=%s", (job['match_id'],))
                except JobInterrupted:
                    return {'status':'interrupted','job_id':str(job_id)}
                return {'status':'failed' if terminal else 'retry','job_id':str(job_id)}
    finally:
        # Session-level advisory lock is released on close, including process crash.
        conn.close()


@celery_app.task(name=TASK_NAME, ignore_result=True, acks_late=True, reject_on_worker_lost=True)
def execute_analysis(job_id):
    return execute(job_id)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['dispatch'])
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    stopped=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    signal.signal(signal.SIGINT,lambda *_:stopped.set())
    logging.basicConfig(level=logging.INFO)
    while not stopped.is_set():
        try:
            count=dispatch_due()
            if count:logger.info('Dispatched %s durable analysis request(s)',count)
            from app.services.media_cleanup import collect_due
            collect_due()
        except Exception:
            logger.exception('Dispatcher unavailable; persisted requests remain pending')
            if args.once:raise
        if args.once:break
        stopped.wait(5)

if __name__ == '__main__':main()
