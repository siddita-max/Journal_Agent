"""Jobs API — create, list, cancel and monitor batch photo processing jobs."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import ProcessingJob, JobStatus
from app.services.drive_service import GoogleDriveService, extract_folder_id
from app.worker.tasks import process_photo_job

log = structlog.get_logger()
router = APIRouter()


# ─── Schemas ──────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    drive_folder_url: str = Field(
        ...,
        description="Full Google Drive folder URL. The service account must have Viewer access.",
        examples=["https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J"],
    )

    @field_validator("drive_folder_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if "drive.google.com" not in v and len(v) < 20:
            raise ValueError("Must be a valid Google Drive folder URL")
        return v


class JobResponse(BaseModel):
    id: str = Field(..., description="Unique job UUID", examples=["550e8400-e29b-41d4-a716-446655440000"])
    drive_folder_url: str = Field(..., description="Original Drive folder URL submitted")
    status: str = Field(..., description="Job lifecycle status", examples=["processing"])
    total_images: int = Field(..., description="Total image files found in the Drive folder")
    processed_images: int = Field(..., description="Images fully processed so far")
    approved_count: int = Field(..., description="Images classified as Approved")
    rejected_count: int = Field(..., description="Images classified as Rejected")
    review_count: int = Field(..., description="Images flagged for human review (score 0.60–0.74)")
    progress_pct: float = Field(..., description="Processing progress as a percentage (0–100)", examples=[42.5])
    error_message: Optional[str] = Field(None, description="Error details if job failed")
    report_url: Optional[str] = Field(None, description="Presigned URL to download the JSON report (available when status=completed)")
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new photo evaluation job",
    response_description="Job accepted and queued for async processing. Poll the returned job ID for status.",
    responses={
        202: {
            "description": "Job accepted — poll `/jobs/{job_id}` for progress",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "drive_folder_url": "https://drive.google.com/drive/folders/...",
                        "status": "pending",
                        "total_images": 0,
                        "processed_images": 0,
                        "approved_count": 0,
                        "rejected_count": 0,
                        "review_count": 0,
                        "progress_pct": 0.0,
                        "error_message": None,
                        "report_url": None,
                        "created_at": "2026-04-16T10:00:00Z",
                        "started_at": None,
                        "completed_at": None,
                    }
                }
            },
        },
        400: {"description": "Invalid Drive URL or folder not accessible by service account"},
    },
)
async def create_job(
    payload: CreateJobRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a Google Drive folder for batch photo evaluation.

    **Steps triggered asynchronously:**
    1. Validate Drive folder access using the configured service account
    2. Fetch all image files (JPEG, PNG, WEBP, BMP, GIF, TIFF) in the folder
    3. Run preprocessing (blur / brightness / resolution / aspect ratio)
    4. Run AI inference: CLIP semantic scoring, YOLOv8 object detection, NSFW safety check
    5. Apply weighted scoring and classify each image: `approved`, `review`, or `rejected`
    6. Store images in MinIO buckets and generate a downloadable JSON report

    **Prerequisite:** Share the Drive folder with the service account email configured in your Drive credential JSON.
    """
    drive = GoogleDriveService()
    try:
        folder_id = extract_folder_id(payload.drive_folder_url)
        await _validate_folder_async(drive, folder_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot access Drive folder: {e}. "
                "Ensure the folder is shared with the service account."
            ),
        )

    job = ProcessingJob(
        drive_folder_url=payload.drive_folder_url,
        drive_folder_id=folder_id,
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    process_photo_job.apply_async(args=[str(job.id)], queue="photo_processing")

    log.info("api.job_created", job_id=str(job.id), folder_id=folder_id)
    return _to_response(job)


@router.get(
    "/",
    response_model=List[JobResponse],
    summary="List all jobs",
    response_description="Paginated list of processing jobs, newest first.",
)
async def list_jobs(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by job status",
        examples=["completed"],
    ),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all photo processing jobs with optional status filtering.

    **Possible status values:** `pending` · `fetching` · `processing` · `completed` · `failed` · `cancelled`
    """
    stmt = select(ProcessingJob).order_by(desc(ProcessingJob.created_at)).limit(limit).offset(offset)
    if status_filter:
        try:
            stmt = stmt.where(ProcessingJob.status == JobStatus(status_filter))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: '{status_filter}'. Valid values: pending, fetching, processing, completed, failed, cancelled")

    result = await db.execute(stmt)
    return [_to_response(j) for j in result.scalars().all()]


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status and progress",
    responses={404: {"description": "Job not found"}},
)
async def get_job(
    job_id: str = Path(..., description="Job UUID returned from POST /jobs/"),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve status, progress counters, and metadata for a specific job.

    Poll this endpoint to track processing. When `status` is `completed`,
    use `report_url` to download the full JSON report.
    """
    job = await _get_or_404(job_id, db)
    return _to_response(job)


@router.post(
    "/{job_id}/cancel",
    response_model=JobResponse,
    summary="Cancel a running job",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job already in a terminal state (completed / failed / cancelled)"},
    },
)
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Cancel a pending or processing job.
    Already-dispatched Celery subtasks will detect the cancellation flag and stop gracefully.
    """
    job = await _get_or_404(job_id, db)
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(status_code=409, detail=f"Job already in terminal state: {job.status.value}")
    job.status = JobStatus.CANCELLED
    await db.commit()
    await db.refresh(job)
    log.info("api.job_cancelled", job_id=job_id)
    return _to_response(job)


@router.get(
    "/{job_id}/report",
    summary="Get the JSON report URL for a completed job",
    responses={
        200: {
            "description": "Presigned MinIO URL valid for 7 days",
            "content": {
                "application/json": {
                    "example": {"report_url": "http://minio:9000/reports/job-id/report.json?X-Amz-Signature=..."}
                }
            },
        },
        404: {"description": "Job not found or report not yet generated"},
        409: {"description": "Job not yet completed"},
    },
)
async def get_report(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns a presigned URL to the structured JSON evaluation report.

    The report contains:
    - Job summary (total / approved / rejected / review counts)
    - Per-image entry with decision, score breakdown, reasons, and storage URL

    **Only available when `status == completed`.**
    """
    job = await _get_or_404(job_id, db)
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job not completed yet (current status: {job.status.value})")
    if not job.report_url:
        raise HTTPException(status_code=404, detail="Report not yet available — try again shortly")
    return {"report_url": job.report_url}


# ─── Helpers ──────────────────────────────────────────────────────────────

async def _validate_folder_async(drive: GoogleDriveService, folder_id: str):
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, drive.validate_folder_access, folder_id)



async def _get_or_404(job_id: str, db: AsyncSession) -> ProcessingJob:
    try:
        uid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID — must be a UUID")
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == uid))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _to_response(job: ProcessingJob) -> dict:
    return {
        "id": str(job.id),
        "drive_folder_url": job.drive_folder_url,
        "status": job.status.value,
        "total_images": job.total_images or 0,
        "processed_images": job.processed_images or 0,
        "approved_count": job.approved_count or 0,
        "rejected_count": job.rejected_count or 0,
        "review_count": job.review_count or 0,
        "progress_pct": job.progress_pct,
        "error_message": job.error_message,
        "report_url": job.report_url,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }
