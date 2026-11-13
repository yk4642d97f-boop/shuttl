"""Real PostgreSQL concurrency/recovery contracts. Uses an explicitly isolated test DB.

SHUTTL_TEST_JOB_DSN=postgresql://.../shuttl_job_test pytest tests/test_analysis_jobs_postgres.py
No compute models or cloud resources are used. Never point this at application data.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

import numpy as np
import httpx
import psycopg2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import Session

from app.api import matches
from app.core.auth_deps import get_current_user_id
from app.core.config import settings
from app.core.database import Base, get_db
from app.models.analysis_job import AnalysisJob
from app.models.match import Match
from app.models.user import User
from app.services import analysis_jobs as jobs
from app.services.tracking import tasks, serialize
from app.services.tracking.data_types import HitEvent, RallyV2, ShotEventV2, VideoMeta
from app.services.tracking.pipeline_v2 import MatchAnalysis

DSN = os.environ.get('SHUTTL_TEST_JOB_DSN')
pytestmark = pytest.mark.skipif(not DSN, reason='Set an isolated SHUTTL_TEST_JOB_DSN to run PostgreSQL contracts')


@pytest.fixture(scope='module')
def database():
    url=make_url(DSN)
    assert url.database=='shuttl_job_test' and url.host in {'127.0.0.1','localhost'}, 'Refusing non-test database'
    engine=create_engine(url.set(drivername='postgresql+psycopg2'),poolclass=NullPool)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def env(database,monkeypatch,tmp_path):
    with database.begin() as c:
        c.execute(text('TRUNCATE users,matches,analysis_jobs,media_deletions CASCADE'))
    monkeypatch.setattr(settings,'database_url',DSN)
    monkeypatch.setattr(settings,'storage_backend','local')
    monkeypatch.setattr(settings,'media_dir',str(tmp_path))
    owner,other,mid=uuid.uuid4(),uuid.uuid4(),uuid.uuid4()
    key=f'videos/{owner}/{mid}.mov'
    video=tmp_path/key;video.parent.mkdir(parents=True);video.write_bytes(b'fixture-only')
    with Session(database) as db:
        db.add_all([User(id=owner,email=f'{owner}@test.invalid',hashed_password='unused'),
                    User(id=other,email=f'{other}@test.invalid',hashed_password='unused')])
        db.flush();db.add(Match(id=mid,user_id=owner,video_oss_key=key));db.commit()
    ae=create_async_engine(DSN.replace('postgresql://','postgresql+asyncpg://'),poolclass=NullPool)
    factory=async_sessionmaker(ae,expire_on_commit=False)
    async def dependency():
        async with factory() as db:yield db
    app=FastAPI();app.include_router(matches.router,prefix='/matches')
    app.dependency_overrides[get_db]=dependency
    app.dependency_overrides[get_current_user_id]=lambda:str(owner)
    published=[]
    def publish(name,args,task_id,**kw):
        # A worker can already see both committed request and match pointer.
        with closing(jobs.connect()) as c,c.cursor() as cur:
            cur.execute('SELECT state FROM analysis_jobs WHERE id=%s',(uuid.UUID(args[0]),))
            assert cur.fetchone()['state'] in jobs.ACTIVE
        published.append((name,args,task_id))
    monkeypatch.setattr(jobs.celery_app,'send_task',publish)
    return {'app':app,'client':TestClient(app),'mid':mid,'owner':owner,'other':other,'published':published,'engine':database}


def submit(env,**kw):
    response=env['client'].post(f"/matches/{env['mid']}/analyze",**kw)
    assert response.status_code==200,response.text
    return uuid.UUID(response.json()['job_id'])


def row(ident):
    with closing(jobs.connect()) as c,c.cursor() as cur:
        cur.execute('SELECT * FROM analysis_jobs WHERE id=%s',(ident,));return cur.fetchone()


def sql(command,parameters=()):
    with closing(jobs.connect()) as c,c,c.cursor() as cur:
        cur.execute(command,parameters)
        return cur.fetchall() if cur.description else None


def output(label='verified'):
    meta=VideoMeta(30.,320,180,4,np.arange(4)/30)
    rally=RallyV2(0,3,[ShotEventV2(HitEvent(1,1/30,(20.,30.),'near',.9),None,'smash')],winner='near')
    res=MatchAnalysis(meta,rallies=[rally]);doc=serialize.to_dict(res);doc['test_marker']=label
    return res,doc


def test_parallel_submissions_have_one_committed_job(env):
    async def concurrent_requests():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env['app']), base_url='http://test') as client:
            async def request():
                response=await client.post(f"/matches/{env['mid']}/analyze")
                assert response.status_code==200,response.text
                return response.json()['job_id']
            return await asyncio.wait_for(asyncio.gather(*(request() for _ in range(12))),timeout=20)
    ids=asyncio.run(concurrent_requests())
    assert len(set(ids))==1 and len(env['published'])==1
    assert sql('SELECT count(*) AS n FROM analysis_jobs')[0]['n']==1
    assert row(uuid.UUID(ids[0]))['state']=='queued'


def test_lost_broker_delivery_is_recoverable_without_resubmission(env,monkeypatch):
    def unavailable(*a,**kw):raise ConnectionError('isolated broker unavailable')
    monkeypatch.setattr(jobs.celery_app,'send_task',unavailable)
    ident=submit(env)
    assert row(ident)['state']=='queued'
    sql('UPDATE analysis_jobs SET dispatched_at=now()-interval \'61 seconds\' WHERE id=%s',(ident,))
    seen=[]
    assert jobs.dispatch_due(publisher=seen.append)==1 and seen==[ident]
    assert row(ident)['state']=='queued'  # delivery is not completion


def test_duplicate_delivery_and_redelivery_do_not_repeat_compute(env):
    ident=submit(env);entered=threading.Event();release=threading.Event();calls=[]
    def compute(*args):
        calls.append(1);entered.set();assert release.wait(10);return output()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(jobs.execute,ident,compute=compute)
        assert entered.wait(5)
        assert jobs.execute(ident,compute=compute)=={'skipped':True}
        release.set();assert first.result(timeout=10)['status']=='done'
    assert jobs.execute(ident,compute=compute)=={'skipped':True} and calls==[1]
    assert row(ident)['attempt']==1
    assert sql('SELECT count(*) AS n FROM rallies')[0]['n']==1
    assert sql('SELECT count(*) AS n FROM shot_points')[0]['n']==1


def test_deterministic_failure_is_not_retried(env):
    ident=submit(env)
    def compute(*args):raise tasks.AnalysisUnusable(tasks.NO_RALLIES_REASON)
    assert jobs.execute(ident,compute=compute)['status']=='failed'
    assert row(ident)['attempt']==1 and row(ident)['error']==tasks.NO_RALLIES_REASON
    assert jobs.execute(ident,compute=lambda *a:pytest.fail('recomputed deterministic failure'))=={'skipped':True}
    assert sql('SELECT status FROM matches')[0]['status']=='failed'


def test_transient_failure_budget_and_delay(env):
    ident=submit(env);calls=[]
    def compute(*args):calls.append(1);raise RuntimeError('temporary stage error')
    for attempt in range(1,4):
        result=jobs.execute(ident,compute=compute)
        assert result['status']==('failed' if attempt==3 else 'retry')
        assert row(ident)['attempt']==attempt
        assert jobs.execute(ident,compute=compute)=={'skipped':True}
        sql('UPDATE analysis_jobs SET available_at=now()-interval \'1 second\' WHERE id=%s',(ident,))
    assert len(calls)==3


def test_worker_crash_releases_lock_and_recovers_same_job(env):
    ident=submit(env)
    connection=jobs.connect();first=jobs.claim(connection,ident)
    assert first['attempt']==1
    connection.close()  # process crash equivalent for the PostgreSQL session lock
    assert jobs.execute(ident,compute=lambda *a:output())['status']=='done'
    assert row(ident)['attempt']==2 and row(ident)['claim_token']!=first['claim_token']


def test_lost_database_session_cannot_commit_and_next_attempt_recovers(env):
    ident = submit(env)
    connection = jobs.connect()
    def compute(*args):
        assert sql('SELECT pg_terminate_backend(%s) AS stopped', (connection.get_backend_pid(),))[0]['stopped']
        return output('lost-session')
    with pytest.raises(psycopg2.Error):
        jobs.execute(ident, compute=compute, connection_factory=lambda: connection)
    assert sql('SELECT analysis FROM matches')[0]['analysis'] is None
    assert row(ident)['state'] == 'running'
    assert jobs.execute(ident, compute=lambda *a: output('recovered'))['status'] == 'done'
    assert row(ident)['attempt'] == 2
    assert sql('SELECT analysis FROM matches')[0]['analysis']['test_marker'] == 'recovered'


def test_result_and_terminal_state_commit_atomically(env, monkeypatch):
    ident = submit(env)
    save = tasks._save_results
    def interrupted_commit(*args):
        save(*args)
        raise RuntimeError('failure after result SQL but before commit')
    monkeypatch.setattr(tasks, '_save_results', interrupted_commit)
    assert jobs.execute(ident, compute=lambda *a: output())['status'] == 'retry'
    assert sql('SELECT count(*) AS n FROM rallies')[0]['n'] == 0
    assert sql('SELECT analysis, status FROM matches')[0] == {'analysis': None, 'status': 'analyzing'}
    monkeypatch.setattr(tasks, '_save_results', save)
    sql("UPDATE analysis_jobs SET available_at=now()-interval '1 second' WHERE id=%s", (ident,))
    assert jobs.execute(ident, compute=lambda *a: output())['status'] == 'done'
    assert row(ident)['state'] == 'done'
    assert sql('SELECT count(*) AS n FROM rallies')[0]['n'] == 1


def test_cancel_queued_never_starts_compute(env):
    ident=submit(env)
    result=env['client'].post(f"/matches/{env['mid']}/cancel")
    assert result.status_code==200 and result.json()['state']=='REVOKED'
    assert jobs.execute(ident,compute=lambda *a:pytest.fail('cancelled job ran'))=={'skipped':True}


def test_cancel_running_fences_late_result_and_allows_fresh_job(env):
    ident=submit(env);entered=threading.Event();release=threading.Event()
    def compute(*args):entered.set();assert release.wait(10);return output('stale')
    with ThreadPoolExecutor(max_workers=1) as pool:
        running=pool.submit(jobs.execute,ident,compute=compute)
        assert entered.wait(5)
        assert env['client'].post(f"/matches/{env['mid']}/cancel").status_code==200
        new_id=submit(env)
        assert new_id!=ident
        assert jobs.execute(new_id,compute=lambda *a:pytest.fail('overlapped old GPU work'))=={'skipped':True}
        release.set();assert running.result(timeout=10)['status']=='interrupted'
    assert sql('SELECT analysis FROM matches')[0]['analysis'] is None
    assert sql('SELECT count(*) AS n FROM rallies')[0]['n']==0
    assert jobs.execute(new_id,compute=lambda *a:output('fresh'))['status']=='done'
    assert sql('SELECT analysis FROM matches')[0]['analysis']['test_marker']=='fresh'
    assert row(ident)['state']=='cancelled'


def test_claim_token_fences_stale_attempt_even_if_state_still_running(env):
    ident=submit(env)
    def compute(*args):
        sql('UPDATE analysis_jobs SET claim_token=%s WHERE id=%s',(uuid.uuid4(),ident))
        return output('stale')
    assert jobs.execute(ident,compute=compute)['status']=='interrupted'
    assert sql('SELECT analysis FROM matches')[0]['analysis'] is None


def test_poll_uses_database_after_redis_result_expiry_and_checks_owner(env,monkeypatch):
    ident=submit(env);jobs.execute(ident,compute=lambda *a:output())
    import celery.result
    monkeypatch.setattr(celery.result,'AsyncResult',lambda *a,**kw:pytest.fail('polled Redis as authority'))
    path=f"/matches/{env['mid']}/job/{ident}"
    assert env['client'].get(path).json()['state']=='SUCCESS'
    assert env['client'].get(f"/matches/{env['mid']}/job/{uuid.uuid4()}").status_code==404
    env['app'].dependency_overrides[get_current_user_id]=lambda:str(env['other'])
    assert env['client'].get(path).status_code==404
    assert env['client'].post(f"/matches/{env['mid']}/cancel").status_code==404


def test_completed_result_reused_until_explicit_force(env):
    ident=submit(env);jobs.execute(ident,compute=lambda *a:output())
    assert submit(env)==ident
    assert env['client'].post(f"/matches/{env['mid']}/cancel").status_code==409
    new=submit(env,json={'force':True});assert new!=ident
    assert sql('SELECT count(*) AS n FROM analysis_jobs')[0]['n']==2


def test_idempotency_key_replay_and_payload_conflict(env):
    headers={'Idempotency-Key':'persisted-request-1'}
    ident=submit(env,headers=headers,json={'doubles':False})
    assert submit(env,headers=headers,json={'doubles':False})==ident
    assert env['client'].post(f"/matches/{env['mid']}/analyze",headers=headers,json={'doubles':True}).status_code==409
    assert sql('SELECT count(*) AS n FROM analysis_jobs')[0]['n']==1


def test_calibration_json_round_trip_reuses_job_even_when_force_is_set(env):
    calibration = {'corners': {'nb_ld': [.1, .9], 'fb_ld': [.3, .2]}, 'doubles': False}
    ident = submit(env, json=calibration)
    assert submit(env, json=calibration) == ident
    assert submit(env, json={**calibration, 'force': True}) == ident
    assert row(ident)['payload']['calibration']['corners'] == calibration['corners']
    assert sql('SELECT count(*) AS n FROM analysis_jobs')[0]['n'] == 1


def test_oss_revision_is_committed_with_request_and_replacement_conflicts(env,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(settings, 'storage_backend', 'oss')
    current = {'etag': 'original', 'size': 100}
    monkeypatch.setattr(matches, '_oss_service', lambda: SimpleNamespace(input_identity=lambda key: dict(current)))
    ident = submit(env)
    assert row(ident)['payload']['video_identity'] == current
    current['etag'] = 'replacement'
    assert env['client'].post(f"/matches/{env['mid']}/analyze").status_code == 409
    assert row(ident)['payload']['video_identity']['etag'] == 'original'


def test_dispatcher_recovers_running_job_after_missing_heartbeat(env):
    ident=submit(env)
    with closing(jobs.connect()) as c:jobs.claim(c,ident)
    sql("UPDATE analysis_jobs SET heartbeat_at=now()-interval '121 seconds',dispatched_at=now()-interval '61 seconds' WHERE id=%s",(ident,))
    seen=[]
    assert jobs.dispatch_due(publisher=seen.append)==1 and seen==[ident]


def test_application_schema_upgrade_is_repeatable_and_keeps_matches(env):
    sql('DROP TABLE analysis_jobs')
    sql('ALTER TABLE matches DROP COLUMN analysis, DROP COLUMN fps')
    process_env = dict(os.environ, DATABASE_URL=DSN.replace('postgresql://', 'postgresql+asyncpg://'))
    command = [sys.executable, '-c', 'import asyncio; from app.core.database import create_tables; asyncio.run(create_tables())']
    for _ in range(2):
        subprocess.run(command, env=process_env, cwd=Path(__file__).resolve().parents[1],
                       check=True, capture_output=True, text=True, timeout=15)
    assert sql('SELECT id, fps, analysis FROM matches')[0] == {'id': env['mid'], 'fps': 60.0, 'analysis': None}
    assert sql('SELECT count(*) AS n FROM analysis_jobs')[0]['n'] == 0


@pytest.mark.skipif(not os.environ.get('SHUTTL_TEST_JOB_REDIS'), reason='Requires a disposable local Redis database 15')
def test_real_dispatcher_worker_crash_and_restart(env, tmp_path):
    """Real Celery/Redis/processes; only video inference is replaced by a fixture."""
    broker = os.environ['SHUTTL_TEST_JOB_REDIS']
    url = urlsplit(broker)
    assert url.scheme == 'redis' and url.hostname in {'127.0.0.1', 'localhost'} and url.path == '/15'
    marker = tmp_path / 'compute-starts.txt'
    proc_env = dict(os.environ, DATABASE_URL=DSN.replace('postgresql://', 'postgresql+asyncpg://'),
                    REDIS_URL=broker, STORAGE_BACKEND='local', MEDIA_DIR=str(tmp_path),
                    SHUTTL_JOB_PROCESS_MARKER=str(marker))
    workdir = Path(__file__).resolve().parents[1]
    processes, logs = [], []

    def start(name, *arguments):
        handle = (tmp_path / f'{name}.log').open('w')
        logs.append(handle)
        proc = subprocess.Popen([sys.executable, *arguments], cwd=workdir, env=proc_env,
                                stdout=handle, stderr=subprocess.STDOUT)
        processes.append(proc)
        return proc

    def wait_until(condition, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.1)
        diagnostics = '\n'.join(p.read_text()[-5000:] for p in tmp_path.glob('*.log'))
        pytest.fail('Process condition timed out:\n' + diagnostics)

    try:
        ident = submit(env)  # parent publisher is mocked; the durable request is not in Redis yet
        sql('UPDATE analysis_jobs SET dispatched_at=NULL WHERE id=%s', (ident,))
        dispatcher = start('dispatcher', '-m', 'app.services.analysis_jobs', 'dispatch')
        first = start('first-worker', '-m', 'tests.job_worker_fixture')
        wait_until(lambda: marker.exists() and marker.stat().st_size > 0)
        assert row(ident)['state'] == 'running' and row(ident)['attempt'] == 1
        first.kill()
        first.wait(timeout=10)
        dispatcher.terminate()
        dispatcher.wait(timeout=10)
        # Advance eligibility timestamps instead of sleeping two minutes in the test.
        sql("UPDATE analysis_jobs SET heartbeat_at=now()-interval '121 seconds',dispatched_at=NULL WHERE id=%s", (ident,))
        start('restarted-dispatcher', '-m', 'app.services.analysis_jobs', 'dispatch')
        start('replacement-worker', '-m', 'tests.job_worker_fixture')
        wait_until(lambda: row(ident)['state'] == 'done')
        assert row(ident)['attempt'] == 2
        assert len(marker.read_text().splitlines()) == 2
        assert sql('SELECT analysis FROM matches')[0]['analysis']['process_test'] == 'recovered'
        assert sql('SELECT count(*) AS n FROM rallies')[0]['n'] == 1
        assert sql('SELECT count(*) AS n FROM shot_points')[0]['n'] == 1
        assert env['client'].get(f"/matches/{env['mid']}/job/{ident}").json()['state'] == 'SUCCESS'
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        for handle in logs:
            handle.close()
