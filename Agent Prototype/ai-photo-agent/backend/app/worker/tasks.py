"""
Celery Tasks — the main processing pipeline orchestrated as async tasks.

Flow:
  process_photo_job
    ↳ fetch image list from Drive
    ↳ dispatch process_single_image per image (parallel)
    ↳ aggregate results → generate report
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.worker.celery_app import celery_app
from app.models.models import (
    ProcessingJob, ImageRecord, AuditLog, JobStatus, ImageDecision, PolicyConfig
)
from app.services.drive_service import GoogleDriveService
from app.services.preprocessing import PreprocessingPipeline
from app.services.inference_engine import InferenceEngine
from app.services.scoring_engine import ScoringEngine, Decision
from app.services.storage_service import StorageService
from app.services.policy_engine import PolicyEngine, DEFAULT_POLICY

log = structlog.get_logger()

# ── Sync DB Engine for Celery (sync context) ──────────────────────────────
_sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
sync_engine = create_engine(_sync_db_url, pool_pre_ping=True)
SyncSession = sessionmaker(bind=sync_engine)

# ── Service singletons (per worker process) ───────────────────────────────
drive_svc = GoogleDriveService()
preproc = PreprocessingPipeline()
inference = InferenceEngine()
scoring = ScoringEngine()
storage = StorageService()


# ─── Main Job Task ────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.worker.tasks.process_photo_job",
    max_retries=2,
    default_retry_delay=30,
)
def process_photo_job(self: Task, job_id: str):
    """
    Orchestrates the full processing pipeline for a Drive folder.
    1. Fetch image list
    2. Dispatch parallel per-image tasks via a Celery group
    3. On completion, generate report
    """
    log.info("job.started", job_id=job_id)
    with SyncSession() as db:
        job: ProcessingJob = db.query(ProcessingJob).get(uuid.UUID(job_id))
        if not job:
            log.error("job.not_found", job_id=job_id)
            return

        job.status = JobStatus.FETCHING
        job.started_at = datetime.now(timezone.utc)
        job.celery_task_id = self.request.id
        db.commit()

        try:
            storage.ensure_buckets()

            # Load and snapshot the active policy at job submission time
            active_policy = db.query(PolicyConfig).filter(
                PolicyConfig.is_active == True
            ).first()
            if active_policy:
                job.policy_id = active_policy.id
                job.policy_snapshot = active_policy.rules
                log.info("job.policy_loaded", policy_name=active_policy.name, version=active_policy.version)
            else:
                job.policy_snapshot = DEFAULT_POLICY
                log.info("job.policy_default", msg="No active policy; using built-in defaults")
            db.commit()

            # Fetch image list
            files = list(drive_svc.list_images(job.drive_folder_id))
            job.total_images = len(files)
            job.status = JobStatus.PROCESSING
            db.commit()

            log.info("job.fetched_images", job_id=job_id, count=len(files))

            # 2. Process each image (can parallelise with Celery chord/group)
            for file_meta in files:
                process_single_image.apply_async(
                    args=[job_id, file_meta, job.policy_snapshot],
                    queue="photo_processing",
                )

        except Exception as exc:
            log.exception("job.failed", job_id=job_id, error=str(exc))
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            db.commit()
            raise self.retry(exc=exc)


# ─── Per-Image Task ───────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.worker.tasks.process_single_image",
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def process_single_image(self: Task, job_id: str, file_meta: dict, policy_rules: dict = None):
    """
    Full pipeline for a single image:
    download → preprocess → inference → policy → score → store → persist
    """
    filename = file_meta.get("name", "unknown")
    file_id = file_meta.get("id")
    log.info("image.processing", filename=filename, file_id=file_id)

    with SyncSession() as db:
        job: ProcessingJob = db.query(ProcessingJob).get(uuid.UUID(job_id))
        if not job or job.status == JobStatus.CANCELLED:
            return

        # Create DB record
        record = ImageRecord(
            job_id=uuid.UUID(job_id),
            drive_file_id=file_id,
            filename=filename,
            mime_type=file_meta.get("mime_type"),
            file_size_bytes=file_meta.get("size"),
        )
        db.add(record)
        db.flush()

        try:
            # ── Step 1: Download ──────────────────────────────────
            image_bytes = drive_svc.download_image(file_id)

            # ── Step 2: Preprocess ────────────────────────────────
            img_cv, prep_result = preproc.evaluate(image_bytes, filename)
            record.width = prep_result.width
            record.height = prep_result.height
            record.blur_variance = prep_result.blur_variance
            record.brightness = prep_result.brightness
            record.aspect_ratio = prep_result.aspect_ratio

            # ── Step 3: AI Inference (skip if hard-rejected) ──────
            inf_result = None
            if prep_result.ok:
                inf_result = inference.run(image_bytes, filename)
                record.people_count = inf_result.yolo.people_count
                record.phone_detected = inf_result.yolo.phone_detected
                record.detected_objects = inf_result.yolo.detections
                record.nsfw_detected = inf_result.safety.nsfw_detected
                record.nsfw_score = inf_result.safety.nsfw_score

            # ── Step 3b: Policy Engine ────────────────────────────────
            policy_result = None
            if inf_result is not None:
                engine = PolicyEngine(policy_rules or {})
                policy_result = engine.evaluate(
                    yolo_result=inf_result.yolo,
                    clip_result=inf_result.clip,
                    preprocess_result=prep_result,
                )
                record.policy_violations = [
                    {
                        "rule_name": v.rule_name,
                        "severity": v.severity,
                        "description": v.description,
                        "measured_value": v.measured_value,
                        "threshold": v.threshold,
                    }
                    for v in policy_result.violations
                ]
                record.policy_compliance_score = policy_result.compliance_score
                if policy_result.violations:
                    log.info(
                        "image.policy_violations",
                        filename=filename,
                        violations=len(policy_result.violations),
                        hard_rejected=policy_result.hard_rejected,
                    )

            # ── Step 4: Score ─────────────────────────────────────
            score_result = scoring.score(prep_result, inf_result, policy_result)
            record.clip_score = score_result.clip_score
            record.quality_score = score_result.quality_score
            record.resolution_score = score_result.resolution_score
            record.object_compliance_score = score_result.object_compliance_score
            record.aesthetic_score = score_result.aesthetic_score
            record.final_score = score_result.final_score
            record.decision = ImageDecision(score_result.decision)
            record.rejection_reasons = score_result.reasons
            record.score_breakdown = score_result.breakdown
            record.processed_at = datetime.now(timezone.utc)

            # ── Step 5: Store image ───────────────────────────────
            storage_path = storage.upload_image(
                image_bytes,
                f"{job_id}/{filename}",
                score_result.decision,
                content_type=file_meta.get("mime_type", "image/jpeg"),
            )
            record.storage_url = storage_path

            # ── Step 6: Update job counters ───────────────────────
            job.processed_images = (job.processed_images or 0) + 1
            if score_result.decision == Decision.APPROVED:
                job.approved_count = (job.approved_count or 0) + 1
            elif score_result.decision == Decision.REJECTED:
                job.rejected_count = (job.rejected_count or 0) + 1
            else:
                job.review_count = (job.review_count or 0) + 1

            # Mark job completed if all images processed
            if job.processed_images >= job.total_images:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                _generate_and_store_report(db, job)

            # Audit log
            _audit(db, "image", str(record.id), "processed", {
                "decision": score_result.decision,
                "score": score_result.final_score,
            })

            db.commit()
            log.info(
                "image.done",
                filename=filename,
                decision=score_result.decision,
                score=score_result.final_score,
            )

        except Exception as exc:
            db.rollback()
            record.decision = ImageDecision.REJECTED
            record.rejection_reasons = [f"Processing error: {str(exc)}"]
            db.add(record)
            db.commit()
            log.exception("image.failed", filename=filename, error=str(exc))
            raise self.retry(exc=exc)


# ─── Report Generation ────────────────────────────────────────────────────

def _generate_and_store_report(db: Session, job: ProcessingJob):
    """Build and upload a JSON summary report for the completed job."""
    images = db.query(ImageRecord).filter(ImageRecord.job_id == job.id).all()
    report = {
        "job_id": str(job.id),
        "drive_folder_url": job.drive_folder_url,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": job.total_images,
            "approved": job.approved_count,
            "rejected": job.rejected_count,
            "review": job.review_count,
        },
        "images": [
            {
                "filename": img.filename,
                "decision": img.decision.value if img.decision else None,
                "final_score": img.final_score,
                "reasons": img.rejection_reasons or [],
                "scores": {
                    "clip": img.clip_score,
                    "quality": img.quality_score,
                    "resolution": img.resolution_score,
                    "object_compliance": img.object_compliance_score,
                    "aesthetic": img.aesthetic_score,
                },
                "storage_url": img.storage_url,
            }
            for img in images
        ],
    }
    try:
        url = storage.upload_report(str(job.id), report)
        job.report_url = url
        log.info("report.uploaded", job_id=str(job.id), url=url)
    except Exception as e:
        log.error("report.upload_failed", error=str(e))


# ─── Beat Tasks ───────────────────────────────────────────────────────────

@celery_app.task(name="app.worker.tasks.cleanup_stale_jobs")
def cleanup_stale_jobs():
    """
    Mark jobs that have been pending/processing for >2h as failed.
    Runs on a schedule via Celery Beat.
    """
    from datetime import timedelta
    with SyncSession() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
        stale = db.query(ProcessingJob).filter(
            ProcessingJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.FETCHING]),
            ProcessingJob.created_at < cutoff,
        ).all()
        for job in stale:
            job.status = JobStatus.FAILED
            job.error_message = "Job timed out (>2 hours)"
            _audit(db, "job", str(job.id), "timeout", {})
        db.commit()
        log.info("cleanup.stale_jobs", count=len(stale))


# ─── Helpers ──────────────────────────────────────────────────────────────

def _audit(db, entity_type: str, entity_id: str, action: str, details: dict):
    log_entry = AuditLog(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        details=details,
    )
    db.add(log_entry)
