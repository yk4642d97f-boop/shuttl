"""Local-storage video upload (used when Aliyun OSS isn't configured)."""

from __future__ import annotations

import os
import asyncio
import uuid
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import matches as matches_api
from app.core.auth_deps import get_current_user_id
from app.core.config import Settings, settings
from app.core.database import get_db
from app.core.security import create_access_token, create_upload_token
from app.services.storage import local_storage

USER_ID = str(uuid.uuid4())
BASE = "http://10.0.0.5:8000"


class _FakeSession:
    def __init__(self, match):
        self.match = match
        self.commits = 0

    async def scalar(self, _stmt):
        return self.match

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass

    async def get(self, *args, **kwargs):
        return None


@pytest.fixture
def local_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "local")
    monkeypatch.setattr(settings, "media_dir", str(tmp_path / "media"))
    return tmp_path / "media"


@pytest.fixture
def match():
    mid = uuid.uuid4()
    return SimpleNamespace(id=mid, user_id=uuid.UUID(USER_ID), video_oss_key=None,
                           celery_job_id=None, status="uploading")


@pytest.fixture
def client(local_mode, match):
    app = FastAPI()
    app.include_router(matches_api.router, prefix="/matches")
    session = _FakeSession(match)

    async def _db():
        yield session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user_id] = lambda: USER_ID
    c = TestClient(app, base_url=BASE)
    c.session = session
    return c


def _key(match) -> str:
    return f"videos/{USER_ID}/{match.id}.mov"


def _path_qs(url: str) -> str:
    u = urlsplit(url)
    return f"{u.path}?{u.query}"


def test_upload_url_is_absolute_on_request_host(client, match):
    r = client.post(f"/matches/{match.id}/upload")
    assert r.status_code == 200
    body = r.json()
    assert body["oss_key"] == _key(match)
    assert body["upload_url"].startswith(f"{BASE}/matches/{match.id}/video?token=")
    assert match.video_oss_key == _key(match) and client.session.commits == 1


def test_round_trip_streams_body_to_disk(client, match, local_mode):
    url = client.post(f"/matches/{match.id}/upload").json()["upload_url"]
    payload = os.urandom(3 * 1024 * 1024 + 17)
    chunks = [payload[i:i + 256 * 1024] for i in range(0, len(payload), 256 * 1024)]
    r = client.put(_path_qs(url), content=iter(chunks),
                   headers={"Content-Type": "video/quicktime"})
    assert r.status_code == 200, r.text
    assert r.json() == {"oss_key": _key(match), "size": len(payload)}
    dest = local_mode / _key(match)
    assert dest.read_bytes() == payload
    assert not [p for p in dest.parent.iterdir() if p.name.endswith(".part")]


@pytest.mark.parametrize("make_token", [
    lambda m: "not-a-jwt",
    lambda m: create_upload_token(str(uuid.uuid4()), USER_ID, _key(m)),           # other match
    lambda m: create_upload_token(str(m.id), USER_ID, _key(m), timedelta(seconds=-5)),  # expired
    lambda m: create_access_token(USER_ID, timedelta(minutes=5)),                   # wrong type
], ids=["garbage", "other-match", "expired", "access-token"])
def test_put_rejects_bad_token(client, match, local_mode, make_token):
    r = client.put(f"/matches/{match.id}/video", params={"token": make_token(match)},
                   content=b"x" * 100)
    assert r.status_code == 403
    assert not (local_mode / _key(match)).exists()


def test_put_requires_token(client, match):
    assert client.put(f"/matches/{match.id}/video", content=b"x").status_code == 422


def test_put_rejects_oversized_upload_and_cleans_up(client, match, local_mode, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 1000)
    token = create_upload_token(str(match.id), USER_ID, _key(match))
    r = client.put(f"/matches/{match.id}/video", params={"token": token},
                   content=iter([b"x" * 600, b"x" * 600]))
    assert r.status_code == 413
    d = local_mode / f"videos/{USER_ID}"
    assert not d.exists() or not list(d.iterdir())


def test_put_rejects_empty_body(client, match):
    token = create_upload_token(str(match.id), USER_ID, _key(match))
    r = client.put(f"/matches/{match.id}/video", params={"token": token}, content=b"")
    assert r.status_code == 400


def test_upload_retry_cannot_change_a_published_video(client, match, local_mode):
    url = client.post(f"/matches/{match.id}/upload").json()['upload_url']
    path = _path_qs(url)
    assert client.put(path, content=b'original').status_code == 200
    assert client.put(path, content=b'original').status_code == 200
    assert client.put(path, content=b'changed!').status_code == 409
    assert client.put(path, content=b'').status_code == 400
    assert (local_mode / _key(match)).read_bytes() == b'original'
    assert not list(local_mode.rglob('*.part'))


def test_concurrent_different_uploads_publish_only_one_complete_file(local_mode):
    key = f'videos/{USER_ID}/concurrent.mov'
    async def run():
        async def chunks(data):
            for block in (data[:4], data[4:]):
                await asyncio.sleep(0)
                yield block
        return await asyncio.gather(
            local_storage.save_stream(key, chunks(b'first-video')),
            local_storage.save_stream(key, chunks(b'second-video')), return_exceptions=True)
    results = asyncio.run(run())
    assert sum(isinstance(r, int) for r in results) == 1
    assert sum(isinstance(r, local_storage.UploadConflict) for r in results) == 1
    assert (local_mode / key).read_bytes() in (b'first-video', b'second-video')
    assert not list(local_mode.rglob('*.part'))


def test_put_disabled_in_oss_mode(client, match, monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "oss")
    token = create_upload_token(str(match.id), USER_ID, _key(match))
    r = client.put(f"/matches/{match.id}/video", params={"token": token}, content=b"x")
    assert r.status_code == 404


def test_analyze_requires_local_file(client, match, local_mode, monkeypatch):
    from app.services import analysis_jobs

    match.video_oss_key = _key(match)
    queued = []
    async def submit(db, owned, calibration, **kwargs):
        queued.append((str(owned.id), owned.video_oss_key, calibration))
        return SimpleNamespace(id="job-1", state="queued")
    monkeypatch.setattr(analysis_jobs, "submit", submit)
    monkeypatch.setattr(analysis_jobs, "dispatch_due", lambda *a: 1)
    assert client.post(f"/matches/{match.id}/analyze").status_code == 400
    assert not queued
    dest = local_mode / _key(match)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"video")
    r = client.post(f"/matches/{match.id}/analyze")
    assert r.status_code == 200 and r.json()["job_id"] == "job-1"
    assert queued == [(str(match.id), _key(match), {"corners_normalized": True})]


def test_task_analyses_local_file_in_place_and_keeps_it(local_mode, monkeypatch):
    from app.services.tracking import tasks

    key = f"videos/{USER_ID}/{uuid.uuid4()}.mov"
    dest = local_mode / key
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"video")
    seen = {}

    def fake_analyze(path, opts, progress=None):
        seen["path"] = path
        # One rally: a result with none is a failure now (tasks.unusable_reason, tested in test_tasks.py).
        return SimpleNamespace(rallies=[object()], processing_time_seconds=0.1)

    monkeypatch.setattr(tasks, "analyze_video", fake_analyze)
    monkeypatch.setattr(tasks.serialize, "to_dict", lambda res: {})
    monkeypatch.setattr(tasks.serialize, "mobile_archive", lambda res: {"fixture": True})
    result, doc = tasks.compute_analysis(key, None, lambda *a: None)
    assert result.rallies and doc == {"mobile_archive": {"fixture": True}}
    assert seen["path"] == str(dest.resolve())
    assert dest.read_bytes() == b"video"          # original never deleted


def test_task_fails_fast_when_local_file_missing(local_mode, monkeypatch):
    from app.services.tracking import tasks

    with pytest.raises(FileNotFoundError):
        tasks.compute_analysis(f"videos/{USER_ID}/missing.mov", None, lambda *a: None)


@pytest.mark.parametrize("key", ["../evil.mov", "videos/../../evil.mov", "/etc/passwd", ""])
def test_path_for_key_rejects_escape(local_mode, key):
    with pytest.raises(ValueError):
        local_storage.path_for_key(key)


@pytest.mark.parametrize("kid,secret,backend,expected", [
    ("", "", "auto", True),
    ("your-access-key-id", "your-secret", "auto", True),
    ("LTAI5tRealLookingId", "realSecretValue", "auto", False),
    ("LTAI5tRealLookingId", "realSecretValue", "local", True),
    ("", "", "oss", False),
])
def test_storage_mode_selection(kid, secret, backend, expected):
    s = Settings(_env_file=None, oss_access_key_id=kid, oss_access_key_secret=secret,
                 storage_backend=backend)
    assert s.use_local_storage is expected


def test_task_module_lets_psycopg2_bind_uuids():
    """_write_results passes uuid.UUID params; psycopg2 needs register_uuid() for that."""
    import psycopg2.extensions

    from app.services.tracking import tasks  # noqa: F401  (registers the adapter)

    u = uuid.uuid4()
    assert psycopg2.extensions.adapt(u).getquoted() == f"'{u}'::uuid".encode()


def test_job_poll_checks_owner_match_and_job_before_result(client, match, monkeypatch):
    import celery.result
    calls=[]
    monkeypatch.setattr(celery.result,'AsyncResult',lambda job_id,**kw:
                        calls.append(job_id) or SimpleNamespace(state='PROGRESS',info={'percent':42}))
    async def allowed(stmt):
        from sqlalchemy.dialects import postgresql
        compiled=stmt.compile(dialect=postgresql.dialect())
        assert 'matches.user_id' in str(compiled) and 'matches.celery_job_id' in str(compiled)
        assert set(compiled.params.values())=={match.id,uuid.UUID(USER_ID),'owned-job'}
        return match
    client.session.scalar=allowed
    r=client.get(f'/matches/{match.id}/job/owned-job')
    assert r.status_code==200 and calls==['owned-job']
    async def denied(stmt): return None
    client.session.scalar=denied
    r=client.get(f'/matches/{match.id}/job/another-users-job')
    assert r.status_code==404 and calls==['owned-job']
