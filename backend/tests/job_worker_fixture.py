"""Disposable process-test worker. The real queue/job runner uses synthetic computation.

Never imported by production; launched only by test_analysis_jobs_postgres.py.
"""
from pathlib import Path
import os
import time
from urllib.parse import urlsplit

import numpy as np

from app.core.config import settings
from app.services.tracking import tasks, serialize
from app.services.tracking.data_types import HitEvent, RallyV2, ShotEventV2, VideoMeta
from app.services.tracking.pipeline_v2 import MatchAnalysis
from app.worker import celery_app


def main():
    db, broker = urlsplit(settings.database_url), urlsplit(settings.redis_url)
    assert db.hostname in {'127.0.0.1', 'localhost'} and db.path == '/shuttl_job_test'
    assert broker.hostname in {'127.0.0.1', 'localhost'} and broker.path == '/15'
    marker = Path(os.environ['SHUTTL_JOB_PROCESS_MARKER'])

    def compute(*args, **kwargs):
        first = not marker.exists()
        with marker.open('a') as f:
            f.write(f'{os.getpid()}\n')
            f.flush()
        if first:
            # The parent kills this process while it owns the database lock.
            until = time.monotonic() + 60
            while time.monotonic() < until:
                time.sleep(.1)
            raise RuntimeError('parent failed to interrupt fixture worker')
        meta = VideoMeta(30., 320, 180, 4, np.arange(4) / 30)
        rally = RallyV2(0, 3, [ShotEventV2(HitEvent(1, 1/30, (20., 30.), 'near', .9), None, 'smash')], winner='near')
        result = MatchAnalysis(meta, rallies=[rally])
        document = serialize.to_dict(result)
        document['process_test'] = 'recovered'
        return result, document

    tasks.compute_analysis = compute
    celery_app.worker_main(['worker', '--pool=solo', '--concurrency=1', '--loglevel=info',
                            '--without-gossip', '--without-mingle', '--without-heartbeat',
                            f'--hostname=contract-{os.getpid()}@localhost'])


if __name__ == '__main__':
    main()
