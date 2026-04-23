"""Images API — browse, filter, inspect, and override image evaluation results."""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import ImageRecord, ImageDecision, AuditLog, FeedbackRecord
from app.services.storage_service import StorageService

log = structlog.get_logger()
router = APIRouter()
storage = StorageService()


# ─── Schemas ──────────────────────────────────────────────────────────────

class ImageSummary(BaseModel):
    id: str = Field(..., description="Unique image record UUID")
    job_id: str = Field(..., description="Parent job UUID")
    filename: str = Field(..., description="Original filename from Google Drive")
    decision: Optional[str] = Field(None, description="AI classification: approved | rejected | review | pending")
    effective_decision: Optional[str] = Field(None, description="Decision after applying any human override")
    final_score: Optional[float] = Field(None, description="Weighted composite score (0.0–1.0)")
    rejection_reasons: Optional[List[str]] = Field(None, description="Human-readable list of reasons for rejection or low score")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    people_count: Optional[int] = Field(None, description="Number of people detected by YOLOv8")
    face_count: Optional[int] = Field(None, description="Alias for people_count (for frontend compatibility)")
    student_count: Optional[int] = Field(None, description="Estimated number of students detected")
    teacher_count: Optional[int] = Field(None, description="Estimated number of teachers detected")
    phone_detected: Optional[bool] = Field(None, description="Whether a mobile phone was detected in the frame")
    nsfw_detected: bool = Field(..., description="Whether NSFW content was detected (always triggers rejection)")
    storage_url: Optional[str] = Field(None, description="MinIO object path: bucket/filename")
    preview_url: Optional[str] = Field(None, description="Temporary presigned URL for previewing the image (uses public MinIO endpoint)")
    db_image_url: Optional[str] = Field(None, description="Stable fallback URL serving image bytes directly from the API/DB — always works regardless of MinIO reachability")
    human_override: Optional[str] = Field(None, description="Override decision set by a human reviewer")
    processed_at: Optional[datetime] = Field(None, description="Timestamp when AI processing completed")
    ai_reason: Optional[str] = Field(None, description="Human-readable reason for the AI decision (computed from rejection_reasons or score_breakdown)")
    role_classification: Optional[str] = Field(None, description="Detected role breakdown (e.g., 'Students: 5, Teachers: 1')")
    detected_activity: Optional[str] = Field(None, description="Main activity detected by AI (e.g., classroom, outdoor, etc.)")
    activity_description: Optional[str] = Field(None, description="Natural language description of the scene")

    class Config:
        from_attributes = True


class ImageDetail(ImageSummary):
    """Extended image record including full score breakdown and detection details."""
    blur_variance: Optional[float] = Field(None, description="Laplacian variance (blur measure). Higher = sharper.")
    brightness: Optional[float] = Field(None, description="Mean pixel brightness (0–255). Ideal: 60–200.")
    aspect_ratio: Optional[float] = Field(None, description="Width / Height ratio")
    clip_score: Optional[float] = Field(None, description="CLIP semantic similarity score (professional photo prompts)")
    quality_score: Optional[float] = Field(None, description="Composite quality: blur + brightness (0–1)")
    resolution_score: Optional[float] = Field(None, description="Normalised resolution score (0–1)")
    object_compliance_score: Optional[float] = Field(None, description="YOLO / policy compliance score (0–1)")
    aesthetic_score: Optional[float] = Field(None, description="Heuristic aesthetic: aspect ratio + CLIP positive signal")
    score_breakdown: Optional[Dict[str, Any]] = Field(None, description="Full scoring breakdown including weights and thresholds")
    detected_objects: Optional[List[Dict[str, Any]]] = Field(None, description="YOLO detected objects: [{class, confidence, class_id}]")
    policy_violations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Policy violations for this image. Each entry: {rule_name, severity (hard|soft), description, measured_value, threshold}"
    )
    policy_compliance_score: Optional[float] = Field(
        None,
        description="Aggregate policy compliance score (0–1). Factored into the object compliance component."
    )

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    decisions: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Count and average score per decision category",
        examples=[{
            "approved": {"count": 142, "avg_score": 0.821},
            "rejected": {"count": 58, "avg_score": 0.312},
            "review": {"count": 17, "avg_score": 0.671},
        }],
    )


class OverrideRequest(BaseModel):
    decision: str = Field(
        ...,
        description="New decision to apply",
        examples=["approved"],
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for the override (stored for audit and feedback loop)",
        examples=["Photo is of acceptable quality despite low CLIP score"],
    )
    annotator: Optional[str] = Field(
        "api_user",
        description="Name or ID of the person making the override",
        examples=["john.doe@company.com"],
    )


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=List[ImageSummary],
    summary="List image evaluation results",
    response_description="Paginated list of image records with scores and decisions.",
)
async def list_images(
    job_id: Optional[str] = Query(None, description="Filter by parent job UUID"),
    decision: Optional[str] = Query(
        None,
        description="Filter by decision: `approved` | `rejected` | `review` | `pending`",
        examples=["rejected"],
    ),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum final score (inclusive)"),
    max_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum final score (inclusive)"),
    limit: int = Query(50, ge=1, le=200, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
):
    """
    Browse image evaluation results with flexible filtering.

    **Combine filters** to find specific images:
    - `?job_id=...&decision=rejected` — all rejected images in a job
    - `?decision=review&min_score=0.60&max_score=0.74` — borderline images needing human review
    - `?decision=approved&min_score=0.90` — only high-confidence approvals
    """
    stmt = select(ImageRecord).order_by(desc(ImageRecord.processed_at))
    conditions = []

    if job_id:
        try:
            conditions.append(ImageRecord.job_id == uuid.UUID(job_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job_id — must be a UUID")

    if decision:
        try:
            conditions.append(func.coalesce(ImageRecord.human_override, ImageRecord.decision) == ImageDecision(decision))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid decision: '{decision}'. Valid: approved, rejected, review, pending")

    if min_score is not None:
        conditions.append(ImageRecord.final_score >= min_score)
    if max_score is not None:
        conditions.append(ImageRecord.final_score <= max_score)

    if conditions:
        stmt = stmt.where(and_(*conditions))

    result = await db.execute(stmt.limit(limit).offset(offset))
    return [_to_summary(img) for img in result.scalars().all()]


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Aggregate decision statistics",
    response_description="Count and average score grouped by decision, optionally scoped to a job.",
)
async def get_stats(
    job_id: Optional[str] = Query(None, description="Scope stats to a specific job UUID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns aggregate counts and average scores grouped by decision category.

    Use this to quickly assess the quality distribution of a job before diving into individual images.
    """
    stmt = select(
        ImageRecord.decision,
        func.count(ImageRecord.id).label("count"),
        func.avg(ImageRecord.final_score).label("avg_score"),
    ).group_by(ImageRecord.decision)

    if job_id:
        try:
            stmt = stmt.where(ImageRecord.job_id == uuid.UUID(job_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job_id — must be a UUID")

    result = await db.execute(stmt)
    return {
        "decisions": {
            (r.decision.value if r.decision else "unknown"): {
                "count": r.count,
                "avg_score": round(float(r.avg_score or 0), 3),
            }
            for r in result.all()
        }
    }


@router.get(
    "/{image_id}",
    response_model=ImageDetail,
    summary="Get full image detail with score breakdown",
    responses={404: {"description": "Image not found"}},
)
async def get_image(
    image_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the complete evaluation record for an image.

    Includes:
    - All component scores (CLIP, quality, resolution, object compliance, aesthetic)
    - YOLO detection details (class, confidence)
    - Laplacian blur variance and brightness measurements
    - Full score breakdown with weights and threshold info
    - Human override history
    """
    img = await _get_image_or_404(image_id, db)
    return _to_detail(img)


@router.get(
    "/{image_id}/data",
    summary="Stream raw image bytes from the database",
    response_description="Raw image content (JPEG/PNG/WEBP) served directly from the DB.",
    responses={
        200: {"content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}}},
        404: {"description": "Image not found or no image data stored"},
    },
)
async def get_image_data(
    image_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Serve the raw image bytes stored in the database.

    Use this as a reliable alternative to MinIO presigned URLs when the
    MinIO hostname is not reachable from the browser (e.g. Docker-internal
    hostname `minio:9000` producing `ERR_NAME_NOT_RESOLVED`).

    The frontend can reference images as:
      `<img src="/api/v1/images/{id}/data" />`
    """
    img = await _get_image_or_404(image_id, db)
    if not img.image_data:
        raise HTTPException(status_code=404, detail="No image data stored in database for this record")
    content_type = img.mime_type or "image/jpeg"
    return Response(content=img.image_data, media_type=content_type)


@router.post(
    "/{image_id}/override",
    response_model=ImageDetail,
    summary="Override the AI decision for an image",
    responses={
        200: {"description": "Override applied. Feedback stored for threshold tuning."},
        400: {"description": "Invalid decision value"},
        404: {"description": "Image not found"},
    },
)
async def override_decision(
    image_id: str,
    payload: OverrideRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Apply a human override to an AI-generated decision.

    **Why this matters:**
    - All overrides are stored in the `feedback_records` table
    - The `GET /feedback/threshold-suggestion` endpoint analyses these overrides
      to recommend scoring threshold adjustments
    - Override data can be exported for future model fine-tuning

    The original AI decision is preserved alongside the override for full auditability.
    """
    img = await _get_image_or_404(image_id, db)

    try:
        new_decision = ImageDecision(payload.decision)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision: '{payload.decision}'. Valid values: approved, rejected, review",
        )

    original = img.decision
    img.human_override = new_decision
    img.override_reason = payload.reason
    img.overridden_by = payload.annotator
    img.overridden_at = datetime.now(timezone.utc)

    db.add(FeedbackRecord(
        image_id=img.id,
        original_decision=original,
        override_decision=new_decision,
        reason=payload.reason,
        annotator=payload.annotator,
    ))
    db.add(AuditLog(
        entity_type="image",
        entity_id=str(img.id),
        action="human_override",
        actor=payload.annotator,
        details={
            "original": original.value if original else None,
            "override": payload.decision,
            "reason": payload.reason,
        },
    ))

    await db.commit()
    await db.refresh(img)
    log.info("api.override", image_id=image_id, from_=original, to=new_decision)
    return _to_detail(img)


# ─── Helpers ──────────────────────────────────────────────────────────────

async def _get_image_or_404(image_id: str, db: AsyncSession) -> ImageRecord:
    try:
        uid = uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image ID — must be a UUID")
    result = await db.execute(select(ImageRecord).where(ImageRecord.id == uid))
    img = result.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="Image record not found")
    return img


def _to_summary(img: ImageRecord) -> dict:
    ai_reason = ""
    if img.rejection_reasons and isinstance(img.rejection_reasons, list):
        ai_reason = "; ".join(img.rejection_reasons)
    if not ai_reason:
        score_val = img.final_score if img.final_score is not None else 0.0
        if score_val >= 0.75:
            ai_reason = "Strong overall score. Image is high quality and matches policy."
        elif score_val >= 0.60:
            # Try to identify the bottleneck
            scores = (img.score_breakdown or {}).get("scores", {})
            if scores:
                # Filter out non-numeric values (like 'resolution_tier') for comparison
                numeric_scores = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
                if numeric_scores:
                    lowest_key = min(numeric_scores, key=numeric_scores.get)
                    lowest_val = numeric_scores[lowest_key]
                    ai_reason = f"Borderline result ({score_val:.3f}). Pulled down by {lowest_key} ({lowest_val:.2f})."
                else:
                    ai_reason = f"Borderline evaluation score ({score_val:.3f})."
            else:
                ai_reason = f"Borderline evaluation score ({score_val:.3f})."
        else:
            ai_reason = f"Poor evaluation score ({score_val:.3f}). Multiple quality/policy flags."
    
    role_classification = "Unknown"
    if img.student_count and img.teacher_count:
        role_classification = f"Students: {img.student_count}, Teachers: {img.teacher_count}"
    elif img.student_count:
        role_classification = f"Students ({img.student_count})"
    elif img.teacher_count:
        role_classification = f"Teachers ({img.teacher_count})"
    
    return {
        "id": str(img.id),
        "job_id": str(img.job_id),
        "filename": img.filename,
        "decision": img.decision.value if img.decision else None,
        "effective_decision": (img.human_override.value if img.human_override else img.decision.value if img.decision else None),
        "final_score": img.final_score,
        "rejection_reasons": img.rejection_reasons or [],
        "width": img.width,
        "height": img.height,
        "people_count": img.people_count,
        "face_count": img.people_count,  # Alias for frontend
        "phone_detected": img.phone_detected,
        "nsfw_detected": img.nsfw_detected or False,
        "storage_url": img.storage_url,
        "preview_url": _preview_url(img.storage_url),
        "db_image_url": f"/api/v1/images/{img.id}/data" if img.image_data else None,
        "human_override": img.human_override.value if img.human_override else None,
        "processed_at": img.processed_at,
        "ai_reason": ai_reason,  # 🤖 NEW: Computed from reasons
        "role_classification": role_classification,  # 🤖 NEW: Computed from student/teacher counts
        "detected_activity": img.detected_activity or "Unknown",  # 🤖 NEW: Main activity detected
        "activity_description": img.activity_description,        # 🤖 NEW: Description from Qwen
        "student_count": img.student_count,  # 🤖 NEW: Raw counts for reference
        "teacher_count": img.teacher_count,
    }


def _to_detail(img: ImageRecord) -> dict:
    d = _to_summary(img)
    d.update({
        "blur_variance": img.blur_variance,
        "brightness": img.brightness,
        "aspect_ratio": img.aspect_ratio,
        "clip_score": img.clip_score,
        "quality_score": img.quality_score,
        "resolution_score": img.resolution_score,
        "object_compliance_score": img.object_compliance_score,
        "aesthetic_score": img.aesthetic_score,
        "score_breakdown": img.score_breakdown or {},
        "detected_objects": img.detected_objects or [],
        "policy_violations": img.policy_violations or [],
        "policy_compliance_score": img.policy_compliance_score,
        "preview_url": _preview_url(img.storage_url),
        "db_image_url": f"/api/v1/images/{img.id}/data" if img.image_data else None,
        "ai_reason": d.get("ai_reason", ""),  # Include from summary
        "role_classification": d.get("role_classification", "Unknown"),  # Include from summary
    })
    return d


def _preview_url(storage_url: Optional[str]) -> Optional[str]:
    if not storage_url or "/" not in storage_url:
        return None
    bucket, object_name = storage_url.split("/", 1)
    try:
        return storage.get_presigned_url(bucket, object_name)
    except Exception:
        return None
