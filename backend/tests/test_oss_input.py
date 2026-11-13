"""Offline storage protocol checks: a replaced object must never become job input."""
from pathlib import Path
from types import SimpleNamespace

import oss2
import pytest

from app.services.storage.oss_service import OSSService, ObjectChanged
from app.services.tracking import tasks


class Bucket:
    def __init__(self, versioned=False):
        self.current = 'original'
        self.versioned = versioned
        self.videos = {'original': b'original-video', 'replacement': b'new-video'}
        self.targets = []

    def get_object_meta(self, key):
        return SimpleNamespace(etag=self.current, content_length=len(self.videos[self.current]),
                               headers={'x-oss-version-id': self.current} if self.versioned else {})

    def get_object_to_file(self, key, target, *, headers, params):
        revision = (params or {}).get('versionId', self.current)
        if headers.get('If-Match') != f'"{revision}"':
            raise oss2.exceptions.ServerError(412, {}, b'', {'Code': 'PreconditionFailed'})
        self.targets.append(Path(target))
        Path(target).write_bytes(self.videos[revision])
        return SimpleNamespace(etag=revision)


def service(bucket):
    instance = object.__new__(OSSService)
    instance._bucket = bucket
    return instance


def test_replacement_between_metadata_and_download_is_rejected(tmp_path):
    bucket = Bucket()
    storage = service(bucket)
    identity = storage.input_identity('video')
    bucket.current = 'replacement'
    with pytest.raises(ObjectChanged):
        storage.download_to_path('video', str(tmp_path/'download.mov'), identity=identity)
    assert not (tmp_path/'download.mov').exists()


def test_versioned_bucket_can_recover_original_input_after_replacement(tmp_path):
    bucket = Bucket(versioned=True)
    storage = service(bucket)
    identity = storage.input_identity('video')
    bucket.current = 'replacement'
    storage.download_to_path('video', str(tmp_path/'download.mov'), identity=identity)
    assert (tmp_path/'download.mov').read_bytes() == b'original-video'


def test_download_length_mismatch_is_rejected(tmp_path):
    storage = service(Bucket())
    identity = storage.input_identity('video')
    identity['size'] += 1
    with pytest.raises(ObjectChanged):
        storage.download_to_path('video', str(tmp_path/'download.mov'), identity=identity)


def test_changed_object_fails_before_model_inference_and_cleans_temp(monkeypatch, tmp_path):
    import app.services.storage.oss_service as module
    storage = service(Bucket())
    identity = storage.input_identity('video')
    storage._bucket.current = 'replacement'
    monkeypatch.setattr(module, 'OSSService', lambda: storage)
    monkeypatch.setattr(tasks.settings, 'storage_backend', 'oss')
    monkeypatch.setattr(tasks.settings, 'media_dir', str(tmp_path))
    monkeypatch.setattr(tasks.tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(tasks, 'analyze_video', lambda *a, **kw: pytest.fail('wrong video reached inference'))
    with pytest.raises(tasks.AnalysisUnusable, match='替换或删除'):
        tasks.compute_analysis('video.mov', None, lambda *a: None, video_identity=identity)
    assert not list(tmp_path.rglob('*.mov'))


def test_cleanup_deletes_exact_key_versions_and_markers_but_keeps_neighbors():
    class VersionBucket:
        def __init__(self):
            self.entries = [('video', 'one'), ('video', 'two'), ('video-copy', 'neighbor')]
            self.deleted = []
        def delete_object(self, key, params=None):
            if params:
                self.deleted.append((key, params['versionId']))
                self.entries.remove((key, params['versionId']))
        def list_object_versions(self, **kwargs):
            assert kwargs['prefix'] == 'video'
            return SimpleNamespace(versions=[SimpleNamespace(key=k, versionid=v) for k,v in self.entries],
                                   delete_marker=[], is_truncated=False)
    bucket = VersionBucket()
    service(bucket).delete_all_versions('video')
    assert bucket.deleted == [('video', 'one'), ('video', 'two')]
    assert bucket.entries == [('video-copy', 'neighbor')]


def test_unversioned_legacy_request_cannot_download_arbitrary_current_object(monkeypatch):
    monkeypatch.setattr(tasks.settings, 'storage_backend', 'oss')
    with pytest.raises(tasks.AnalysisUnusable, match='缺少录像版本信息'):
        tasks.compute_analysis('video.mov', None, lambda *a: None)
