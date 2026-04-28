"""
Celery Application — task queue for async photo processing.
"""
import structlog
from celery import Celery
from celery.signals import worker_process_init

from app.core.config import settings
from app.services.inference_engine import ModelRegistry

log = structlog.get_logger()

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
    task_acks_late=True,                    # Re-queue if worker crashes
    worker_prefetch_multiplier=1,           # Fair task distribution
    broker_connection_retry_on_startup=True,  # Suppress Celery 6.0 deprecation
    task_routes={
        # Orchestration task gets its own queue checked FIRST so a new job
        # is never blocked behind a backlog of per-image tasks.
        "app.worker.tasks.process_photo_job": {"queue": "job_orchestration"},
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


@worker_process_init.connect
def warm_up_models(**kwargs):
    """
    Load heavy models once per worker process so the first task does not pay
    the full cold-start penalty.
    """
    try:
        ModelRegistry.get_clip()
        ModelRegistry.get_yolo()
        ModelRegistry.get_safety()
        if settings.GROQ_ENABLED:
            ModelRegistry.get_groq()
        log.info("inference.warmup_complete")
    except Exception as exc:
        log.warning("inference.warmup_failed", error=str(exc))
