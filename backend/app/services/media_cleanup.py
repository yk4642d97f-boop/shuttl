"""Persistent cleanup of deleted matches. Run by the existing dispatcher.

The tombstone stays after a successful sweep: late direct OSS PUTs or abandoned
temporary files must be collected on subsequent sweeps too.
"""
import logging

from app.services.analysis_jobs import connect, lock_key
from app.services.storage import local_storage

logger = logging.getLogger(__name__)
RECHECK_SECONDS = 3600
RETRY_SECONDS = 60


def remove_files(row):
    if row['storage_backend'] == 'oss':
        from app.services.storage.oss_service import OSSService
        OSSService().delete_all_versions(row['video_key'])
    elif row['storage_backend'] != 'local':
        raise ValueError('Unknown storage backend in deletion record')
    local_storage.delete_files(row['video_key'])


def collect_due(match_id=None, *, remover=remove_files, connection_factory=connect):
    """DB records survive process/broker failures; deletion itself is safely repeatable."""
    conn = connection_factory()
    cleaned = 0
    try:
        with conn, conn.cursor() as cur:
            cur.execute('''SELECT * FROM media_deletions WHERE available_at<=now()
                           AND (%s::uuid IS NULL OR match_id=%s::uuid)
                           ORDER BY available_at LIMIT 10 FOR UPDATE SKIP LOCKED''', (match_id, match_id))
            for row in cur.fetchall():
                key = lock_key(row['match_id'])
                cur.execute('SELECT pg_try_advisory_lock(%s) AS acquired', (key,))
                if not cur.fetchone()['acquired']:
                    cur.execute("UPDATE media_deletions SET available_at=now()+interval '5 seconds' WHERE match_id=%s", (row['match_id'],))
                    continue
                try:
                    remover(row)
                except Exception as exc:
                    logger.warning('Deleted-match media cleanup remains pending for %s', row['match_id'], exc_info=True)
                    cur.execute('''UPDATE media_deletions SET state='pending',attempts=attempts+1,last_error=%s,
                                   available_at=now()+%s*interval '1 second' WHERE match_id=%s''',
                                (str(exc)[:2000], RETRY_SECONDS, row['match_id']))
                else:
                    cleaned += 1
                    cur.execute('''UPDATE media_deletions SET state='watching',attempts=attempts+1,last_error=NULL,
                                   cleaned_at=now(),available_at=now()+%s*interval '1 second' WHERE match_id=%s''',
                                (RECHECK_SECONDS, row['match_id']))
                finally:
                    cur.execute('SELECT pg_advisory_unlock(%s)', (key,))
        return cleaned
    finally:
        conn.close()
