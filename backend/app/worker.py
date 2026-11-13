from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "shuttlevision",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.services.tracking.tasks", "app.services.analysis_jobs"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"socket_connect_timeout": 3, "socket_timeout": 5,
                              "visibility_timeout": 6 * 60 * 60},
)
