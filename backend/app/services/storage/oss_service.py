import oss2
from pathlib import Path
from app.core.config import settings


class ObjectChanged(Exception):
    """The submitted video revision is no longer available."""


class OSSService:
    """Thin wrapper around oss2 for presigned URL generation and object operations."""

    def __init__(self):
        auth = oss2.Auth(settings.oss_access_key_id, settings.oss_access_key_secret)
        self._bucket = oss2.Bucket(auth, settings.oss_endpoint, settings.oss_bucket_name, connect_timeout=10)

    def generate_presigned_put_url(self, key: str, expires: int = 3600) -> str:
        """Return a presigned URL that the iOS client can PUT directly to OSS."""
        return self._bucket.sign_url("PUT", key, expires, slash_safe=True)

    def generate_presigned_get_url(self, key: str, expires: int = 3600) -> str:
        return self._bucket.sign_url("GET", key, expires, slash_safe=True)

    def input_identity(self, key: str) -> dict:
        """Freeze the opaque ETag and length; retain a version ID when bucket versioning exists."""
        try:
            result = self._bucket.get_object_meta(key)
        except oss2.exceptions.ServerError as exc:
            if exc.status == 404:
                raise FileNotFoundError('视频尚未上传') from exc
            raise
        if result.content_length == 0:
            raise ValueError('上传内容为空')
        if not result.etag or result.content_length is None:
            raise RuntimeError('对象存储没有返回录像版本信息')
        identity = {'etag': result.etag, 'size': result.content_length}
        version = result.headers.get('x-oss-version-id')
        if version and version != 'null':
            identity['version_id'] = version
        return identity

    def download_to_path(self, key: str, local_path: str, *, identity: dict) -> None:
        # If-Match is checked by OSS during GET, closing the HEAD/download race.
        params = {'versionId': identity['version_id']} if identity.get('version_id') else None
        try:
            result = self._bucket.get_object_to_file(
                key, local_path, headers={'If-Match': '"' + identity['etag'].strip('"') + '"'}, params=params)
        except oss2.exceptions.ServerError as exc:
            if exc.status in {404, 412}:
                raise ObjectChanged('已提交的录像已被替换或删除，请重新提交分析') from exc
            raise
        if result.etag != identity['etag'] or Path(local_path).stat().st_size != identity['size']:
            raise ObjectChanged('下载录像与提交时的版本不一致，请重新提交分析')

    def object_exists(self, key: str) -> bool:
        return self._bucket.object_exists(key)

    def delete_all_versions(self, key: str, *, max_pages: int = 20) -> None:
        """Purge this exact key, including historical versions and delete markers.

        Tombstones are rechecked later: an already-issued PUT can finish after deletion.
        Bound each sweep; another sweep continues if a key has many versions.
        """
        # Delete current data promptly even when listing old versions later fails.
        self._bucket.delete_object(key)
        for _ in range(max_pages):
            result = self._bucket.list_object_versions(prefix=key, max_keys=100)
            entries = [entry for entry in result.versions + result.delete_marker if entry.key == key]
            if not entries:
                return
            for entry in entries:
                self._bucket.delete_object(key, params={'versionId': entry.versionid})
            if not result.is_truncated:
                return
        raise RuntimeError('录像历史版本尚未全部清理，将继续重试')
