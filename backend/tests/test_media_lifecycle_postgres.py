"""Real DB/file contracts for deletion while workers and uploads are active."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import uuid

import httpx
import pytest

from app.api import matches
from app.core.auth_deps import get_current_user_id
from app.core.config import settings
from app.core.security import create_upload_token
from app.services import analysis_jobs as jobs, media_cleanup
from app.services.storage import local_storage
from tests.test_analysis_jobs_postgres import DSN, database, env, output, row, sql, submit

pytestmark = pytest.mark.skipif(not DSN, reason='Requires isolated local shuttl_job_test DB')


def video(env):
    return Path(settings.media_dir)/f"videos/{env['owner']}/{env['mid']}.mov"


def make_due(env):
    sql("UPDATE media_deletions SET available_at=now()-interval '1 second' WHERE match_id=%s", (env['mid'],))


def test_delete_completed_match_cascades_results_and_retains_cleanup_request(env):
    ident = submit(env)
    assert jobs.execute(ident, compute=lambda *a: output())['status'] == 'done'
    response = env['client'].delete(f"/matches/{env['mid']}")
    assert response.status_code == 202 and response.json()['media_cleanup'] == 'pending'
    for table in ('matches', 'rallies', 'shot_points', 'analysis_jobs'):
        assert sql(f'SELECT count(*) AS n FROM {table}')[0]['n'] == 0
    assert video(env).exists()
    assert media_cleanup.collect_due() == 1
    assert not video(env).exists()
    assert sql('SELECT state FROM media_deletions')[0]['state'] == 'watching'
    assert env['client'].delete(f"/matches/{env['mid']}").status_code == 202
    assert sql('SELECT count(*) AS n FROM media_deletions')[0]['n'] == 1


def test_delete_other_owners_match_does_not_enqueue_or_remove_anything(env):
    env['app'].dependency_overrides[get_current_user_id] = lambda: str(env['other'])
    assert env['client'].delete(f"/matches/{env['mid']}").status_code == 404
    assert sql('SELECT count(*) AS n FROM matches')[0]['n'] == 1
    assert sql('SELECT count(*) AS n FROM media_deletions')[0]['n'] == 0
    assert video(env).exists()


def test_delete_during_compute_fences_results_and_defers_file_cleanup(env):
    ident = submit(env)
    entered, release = threading.Event(), threading.Event()
    def compute(*args):
        entered.set()
        assert release.wait(10)
        return output('must-not-reappear')
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(jobs.execute, ident, compute=compute)
        assert entered.wait(5)
        assert env['client'].delete(f"/matches/{env['mid']}").status_code == 202
        assert media_cleanup.collect_due() == 0
        assert video(env).exists(), 'do not unlink input being read by a live worker'
        release.set()
        assert active.result(timeout=10)['status'] == 'interrupted'
    assert row(ident) is None
    assert sql('SELECT count(*) AS n FROM rallies')[0]['n'] == 0
    make_due(env)
    assert media_cleanup.collect_due() == 1 and not video(env).exists()


def test_deleted_match_invalidates_old_local_upload_token(env):
    key = f"videos/{env['owner']}/{env['mid']}.mov"
    token = create_upload_token(str(env['mid']), str(env['owner']), key)
    assert env['client'].delete(f"/matches/{env['mid']}").status_code == 202
    media_cleanup.collect_due()
    assert env['client'].put(f"/matches/{env['mid']}/video", params={'token': token}, content=b'resurrection').status_code == 404
    assert not video(env).exists()
    assert env['client'].post(f"/matches/{env['mid']}/analyze").status_code == 404
    assert env['client'].post(f"/matches/{env['mid']}/upload").status_code == 404


def test_delete_while_streaming_does_not_wait_for_body_and_blocks_publication(env):
    video(env).unlink()
    key = f"videos/{env['owner']}/{env['mid']}.mov"
    token = create_upload_token(str(env['mid']), str(env['owner']), key)
    async def race():
        entered, release = asyncio.Event(), asyncio.Event()
        async def body():
            yield b'first-part'
            entered.set()
            await release.wait()
            yield b'last-part'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env['app']), base_url='http://test') as client:
            uploading = asyncio.create_task(client.put(f"/matches/{env['mid']}/video", params={'token': token}, content=body()))
            await asyncio.wait_for(entered.wait(), 5)
            deletion = await asyncio.wait_for(client.delete(f"/matches/{env['mid']}"), 3)
            assert deletion.status_code == 202
            release.set()
            assert (await asyncio.wait_for(uploading, 5)).status_code == 404
    asyncio.run(race())
    assert not video(env).exists()
    assert not list(Path(settings.media_dir).rglob('*.part'))


def test_cleanup_failure_is_retained_and_late_files_are_collected_again(env):
    assert env['client'].delete(f"/matches/{env['mid']}").status_code == 202
    def unavailable(row):
        raise OSError('fixture storage offline')
    assert media_cleanup.collect_due(remover=unavailable) == 0
    assert sql('SELECT state,attempts FROM media_deletions')[0] == {'state': 'pending', 'attempts': 1}
    make_due(env)
    assert media_cleanup.collect_due() == 1
    video(env).write_bytes(b'late-direct-object-upload')
    work = local_storage.work_directory(f"videos/{env['owner']}/{env['mid']}.mov")
    (work/'orphaned-worker.mov').write_bytes(b'worker-crash-copy')
    make_due(env)
    assert media_cleanup.collect_due() == 1
    assert not video(env).exists() and not work.exists()


def test_old_queued_cancel_cannot_cancel_a_newer_job(env):
    original = submit(env)
    assert env['client'].post(f"/matches/{env['mid']}/cancel", json={'job_id': str(original)}).status_code == 200
    new = submit(env)
    response = env['client'].post(f"/matches/{env['mid']}/cancel", json={'job_id': str(original)})
    assert response.status_code == 409
    assert row(new)['state'] == 'queued'


def test_delete_before_delayed_create_prevents_resurrection(env):
    mid = uuid.uuid4()
    assert env['client'].delete(f'/matches/{mid}').status_code == 202
    assert env['client'].post('/matches', json={'id': str(mid)}).status_code == 409
    assert sql('SELECT id FROM matches WHERE id=%s', (mid,)) == []


def test_parallel_create_retries_share_client_id_and_delete_wins_final_state(env):
    mid = uuid.uuid4()
    async def concurrent():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env['app']), base_url='http://test') as client:
            created = await asyncio.gather(*(client.post('/matches', json={'id': str(mid), 'duration_seconds': 10}) for _ in range(8)))
            assert all(response.status_code == 201 and response.json()['id'] == str(mid) for response in created)
            result = await asyncio.gather(client.post('/matches', json={'id': str(mid)}), client.delete(f'/matches/{mid}'))
            assert result[0].status_code in {201, 409} and result[1].status_code == 202
    asyncio.run(concurrent())
    assert sql('SELECT id FROM matches WHERE id=%s', (mid,)) == []
    assert len(sql('SELECT match_id FROM media_deletions WHERE match_id=%s', (mid,))) == 1
