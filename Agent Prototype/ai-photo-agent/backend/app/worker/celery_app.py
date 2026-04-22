"""
Celery Application — task queue for async photo processing.
"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "photo_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,           # Re-queue if worker crashes
    worker_prefetch_multiplier=1,  # Fair task distribution
    task_routes={
        "app.worker.tasks.process_photo_job": {"queue": "photo_processing"},
        "app.worker.tasks.process_single_image": {"queue": "photo_processing"},
    },
    beat_schedule={
        # Cleanup old pending jobs every hour
        "cleanup-stale-jobs": {
            "task": "app.worker.tasks.cleanup_stale_jobs",
            "schedule": 3600.0,
        },
    },
)
