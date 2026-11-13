"""Local-disk video storage, used when Aliyun OSS isn't configured.

Keys have the same shape as OSS keys (videos/<user>/<match>.mov) and map to
files under settings.media_dir. Uploads stream straight to disk.
"""

from __future__ import annotations

import os
import hashlib
import uuid
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import AsyncIterator
from pathlib import Path

import anyio

from app.core.config import settings


class UploadTooLarge(Exception):
    pass


class UploadEmpty(ValueError):
    pass


class UploadConflict(Exception):
    """A different video already owns this immutable key."""


def media_root() -> Path:
    root = Path(settings.media_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def path_for_key(key: str) -> Path:
    """Resolve a storage key to a path inside media_root; rejects traversal."""
    root = media_root()
    p = (root / key).resolve()
    if p == root or root not in p.parents:
        raise ValueError(f"Invalid storage key: {key!r}")
    return p


@dataclass
class StagedUpload:
    temporary: Path
    destination: Path
    digest: str
    size: int

    async def publish(self) -> int:
        await anyio.to_thread.run_sync(_publish_once, self.temporary, self.destination, self.digest, self.size)
        return self.size


@asynccontextmanager
async def stage_stream(key: str, chunks: AsyncIterator[bytes], max_bytes: int | None = None):
    """Stream a complete video to a private temp file; caller authorizes final publication."""
    dest = path_for_key(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.part")
    limit = max_bytes or settings.max_upload_bytes
    written = 0
    digest = hashlib.sha256()
    f = await anyio.to_thread.run_sync(open, tmp, "wb")
    try:
        async for chunk in chunks:
            if not chunk:
                continue
            written += len(chunk)
            if written > limit:
                raise UploadTooLarge(f"upload exceeds {limit} bytes")
            digest.update(chunk)
            await anyio.to_thread.run_sync(f.write, chunk)
        if not written:
            raise UploadEmpty('upload body is empty')
        await anyio.to_thread.run_sync(f.flush)
        await anyio.to_thread.run_sync(os.fsync, f.fileno())
        await anyio.to_thread.run_sync(f.close)
        yield StagedUpload(tmp, dest, digest.hexdigest(), written)
    finally:
        f.close()
        tmp.unlink(missing_ok=True)


async def save_stream(key: str, chunks: AsyncIterator[bytes], max_bytes: int | None = None) -> int:
    """Publish once; API callers use stage_stream to lock/check the match before publishing."""
    async with stage_stream(key, chunks, max_bytes) as staged:
        return await staged.publish()


def work_directory(key: str) -> Path:
    """Worker downloads remain attributable to a key, including after a worker crash."""
    directory = media_root() / '.analysis' / hashlib.sha256(key.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def delete_files(key: str) -> None:
    """Caller holds the analysis advisory lock so no worker is using these files."""
    path = path_for_key(key)
    path.unlink(missing_ok=True)
    # An in-flight uploader may still own a .part inode; its final DB check will reject it.
    for temporary in path.parent.glob(f'.{path.name}.*.part'):
        temporary.unlink(missing_ok=True)
    work = media_root() / '.analysis' / hashlib.sha256(key.encode()).hexdigest()
    if work.exists():
        shutil.rmtree(work)


def _publish_once(tmp: Path, dest: Path, digest: str, size: int) -> None:
    try:
        os.link(tmp, dest)
    except FileExistsError:
        with dest.open('rb') as existing:
            if os.fstat(existing.fileno()).st_size != size or hashlib.file_digest(existing, 'sha256').hexdigest() != digest:
                raise UploadConflict('this match already has a different video')
    tmp.unlink()
    directory = os.open(dest.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
