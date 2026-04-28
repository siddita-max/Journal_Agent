"""
Upload API — direct file upload endpoint for testing without Google Drive.
Accepts multipart image uploads, runs the full pipeline (preprocess → AI → score → store),
and returns the evaluation result immediately.

This is ideal for:
- Local testing with a folder of images
- CI/CD validation
- Single-image debugging
"""
import io
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import (
    ImageDecision, ImageRecord, AuditLog, PolicyConfig
)
from app.services.policy_engine import PolicyEngine, get_school_child_safety_policy
from app.services.preprocessing import PreprocessingPipeline
from app.services.inference_engine import InferenceEngine
from app.services.scoring_engine import ScoringEngine
from app.services.storage_service import StorageService

log = structlog.get_logger()
router = APIRouter()

SUPPORTED_TYPES = {
    "image/jpeg", "image/png", "image/webp",
}

DECISION_TO_STATUS = {
    "approved": "approved",
    "review": "rejected",   # review tier removed — collapsed into rejected
    "rejected": "rejected",
}

# ── Lazy singletons (loaded once per API process) ─────────────────────────
_preproc: Optional[PreprocessingPipeline] = None
_inference: Optional[InferenceEngine] = None
_scoring: Optional[ScoringEngine] = None
_storage: Optional[StorageService] = None


def _get_services():
    global _preproc, _inference, _scoring, _storage
    if _preproc is None:
        _preproc = PreprocessingPipeline()
        _inference = InferenceEngine()
        _scoring = ScoringEngine()
        _storage = StorageService()
        _storage.ensure_buckets()
    return _preproc, _inference, _scoring, _storage


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    status_code=status.HTTP_200_OK,
    summary="Upload and evaluate a single image (no Google Drive needed)",
    response_description="Full AI evaluation result for the uploaded image.",
    responses={
        200: {
            "description": "Evaluation complete",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "team_photo.jpg",
                        "decision": "approved",
                        "final_score": 0.823,
                        "scores": {
                            "clip": 0.712,
                            "quality": 0.891,
                            "resolution": 0.950,
                            "object_compliance": 0.800,
                            "aesthetic": 0.750,
                        },
                        "preprocessing": {
                            "width": 1920,
                            "height": 1080,
                            "blur_variance": 312.4,
                            "brightness": 142.1,
                            "aspect_ratio": 1.778,
                        },
                        "yolo_detections": {
                            "people_count": 3,
                            "phone_detected": False,
                            "objects": [],
                        },
                        "safety": {
                            "nsfw_detected": False,
                            "nsfw_score": 0.02,
                        },
                        "policy_violations": [],
                        "policy_compliance_score": 1.0,
                        "rejection_reasons": [],
                        "hard_rejected": False,
                    }
                }
            },
        },
        400: {"description": "Unsupported file type"},
        413: {"description": "File too large (max 20 MB)"},
    },
)
async def evaluate_image(
    file: UploadFile = File(
        ...,
        description="Image file to evaluate (JPEG, PNG, WEBP, BMP, GIF, TIFF). Max 20 MB.",
    ),
    user_text: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    **Upload and evaluate a single image directly — no Google Drive required.**

    This endpoint runs the complete evaluation pipeline synchronously and returns
    the result immediately. Perfect for:
    - Testing with a local folder of images
    - Debugging individual images
    - Validating model behaviour before running a full job

    **Pipeline:**
    `upload → preprocess → CLIP + YOLOv8 + Safety → Policy Engine → Scoring → result`

    The result is **also persisted** to the database as a standalone image record
    (not linked to any job) so you can retrieve it later via `GET /images/{id}`.
    """
    # ── Validation ────────────────────────────────────────────────────────
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file.content_type}'. Supported: {', '.join(SUPPORTED_TYPES)}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20 MB.")

    filename = file.filename or f"upload_{uuid.uuid4().hex[:8]}.jpg"
    log.info("upload.received", filename=filename, size=len(image_bytes))

    result = await db.execute(
        select(PolicyConfig).where(
            PolicyConfig.name == "school_journal_policy_v1",
            PolicyConfig.is_active == True,
        ).limit(1)
    )
    stored_policy = result.scalar_one_or_none()
    active_policy_rules = stored_policy.rules if stored_policy else get_school_child_safety_policy()

    # ── Pipeline ──────────────────────────────────────────────────────────
    import asyncio
    loop = asyncio.get_event_loop()

    def _run_pipeline():
        preproc, inference_engine, scoring_engine, storage_svc = _get_services()

        # 1. Preprocess
        _, prep = preproc.evaluate(image_bytes, filename)

        # 2. Step-by-Step Gatekeeper: Skip AI if quality is too low
        inf = None
        if prep.ok:
            inf = inference_engine.run(
                image_bytes,
                filename,
                user_text=user_text,
                yolo_augment=False,
                mime_type=file.content_type,
            )
        else:
            log.info("pipeline.skipping_ai", reason=prep.rejection_reasons)

        # 3. Fixed school/children safety policy
        policy_rules = active_policy_rules

        policy_result = None
        if inf is not None:
            engine = PolicyEngine(policy_rules)
            policy_result = engine.evaluate(
                yolo_result=inf.yolo,
                clip_result=inf.clip,
                preprocess_result=prep,
            )

        # 4. Score
        score = scoring_engine.score(prep, inf, policy_result)

        # 5. Store image
        try:
            storage_path = storage_svc.upload_image(
                image_bytes,
                f"direct-upload/{filename}",
                score.decision,
                content_type=file.content_type or "image/jpeg",
            )
        except Exception:
            storage_path = None

        return prep, inf, policy_result, score, storage_path

    prep, inf, policy_result, score, storage_path = await loop.run_in_executor(None, _run_pipeline)

    # ── Build a rich activity description preferring Groq output ──────────
    activity_description = None
    detected_activity_override = None
    if inf is not None:
        if inf.groq and not inf.groq.error:
            activity_text = (
                inf.groq.activity_detail
                or inf.groq.activity_label
                or inf.groq.activity
                or ""
            )
            surroundings_text = inf.groq.surroundings or ""
            role_text = inf.groq.role_summary or ""
            parts = []
            if role_text:
                parts.append(f"Role: {role_text}")
            if activity_text:
                parts.append(f"Activity: {activity_text}")
            if surroundings_text:
                parts.append(f"Surroundings: {surroundings_text}")
            activity_description = "\n".join(parts) or None
            if inf.groq.activity_label:
                detected_activity_override = inf.groq.activity_label

    # ── Persist to DB ─────────────────────────────────────────────────────
    record = ImageRecord(
        job_id=None,                         # standalone upload — no job
        drive_file_id=f"upload:{uuid.uuid4().hex}",
        filename=filename,
        mime_type=file.content_type,
        file_size_bytes=len(image_bytes),
        width=prep.width,
        height=prep.height,
        blur_variance=prep.blur_variance,
        brightness=prep.brightness,
        aspect_ratio=prep.aspect_ratio,
        people_count=inf.yolo.people_count if inf else None,
        student_count=inf.yolo.student_count if inf else None,
        teacher_count=inf.yolo.teacher_count if inf else None,
        detected_activity=(detected_activity_override or (inf.clip.detected_activity if inf else None)),
        activity_description=activity_description,
        phone_detected=inf.yolo.phone_detected if inf else None,
        detected_objects=inf.yolo.detections if inf else None,
        nsfw_detected=inf.safety.nsfw_detected if inf else False,
        nsfw_score=inf.safety.nsfw_score if inf else None,
        clip_score=score.clip_score,
        quality_score=score.quality_score,
        resolution_score=score.resolution_score,
        object_compliance_score=score.object_compliance_score,
        aesthetic_score=score.aesthetic_score,
        final_score=score.final_score,
        decision=ImageDecision(score.decision),
        rejection_reasons=score.reasons,
        score_breakdown=score.breakdown,
        policy_violations=[
            {
                "rule_name": v.rule_name,
                "severity": v.severity,
                "description": v.description,
                "measured_value": v.measured_value,
                "threshold": v.threshold,
            }
            for v in (policy_result.violations if policy_result else [])
        ],
        policy_compliance_score=policy_result.compliance_score if policy_result else None,
        storage_url=storage_path,
        image_data=image_bytes,
        processed_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.add(AuditLog(
        entity_type="image",
        entity_id=str(record.id),
        action="direct_upload_evaluated",
        details={"filename": filename, "decision": score.decision, "score": score.final_score},
    ))
    await db.commit()
    await db.refresh(record)

    # ── Response ──────────────────────────────────────────────────────────
    # Focus only on high-level results, hiding all internal ML details
    decision = (score.decision or "rejected").lower()
    if decision == "review":
        decision = "rejected"
    status = DECISION_TO_STATUS.get(decision, "rejected")
    reason = None
    if status == "rejected":
        reason = "; ".join(score.reasons) or "Image rejected by policy"

    is_approved = decision == "approved"

    groq_payload = None
    if is_approved and inf and inf.groq and not inf.groq.error:
        groq_payload = {
            "activity": inf.groq.activity,
            "activity_label": inf.groq.activity_label,
            "activity_detail": inf.groq.activity_detail,
            "surroundings": inf.groq.surroundings,
            "people_count": inf.groq.people_count,
            "role_summary": inf.groq.role_summary,
            "matches_journal": inf.groq.matches_journal,
            "match_reason": inf.groq.match_reason,
            "confidence": inf.groq.confidence,
            "safe": inf.groq.safe,
            "quality_note": inf.groq.quality_note,
        }

    scene_context = {
        "total_people": inf.yolo.people_count if inf else 0,
        "students": inf.yolo.student_count if inf else 0,
        "teachers": inf.yolo.teacher_count if inf else 0,
        "match_score": round(inf.clip.semantic_score * 100, 1) if inf else None,
    }
    if is_approved and inf:
        scene_context["activity"] = (
            (inf.groq.activity_label if inf.groq and not inf.groq.error and inf.groq.activity_label else None)
            or inf.clip.detected_activity
            or "Unknown"
        )
        scene_context["activity_detail"] = inf.groq.activity_detail if inf.groq and not inf.groq.error else None
        scene_context["surroundings"] = inf.groq.surroundings if inf.groq and not inf.groq.error else None

    return {
        "image_id": str(record.id),
        "filename": filename,
        "decision": decision,
        "status": status,
        "reason": reason,
        "scene_context": scene_context,
        "groq": groq_payload,
    }


@router.post(
    "/evaluate-batch",
    status_code=status.HTTP_200_OK,
    summary="Upload and evaluate multiple images at once (no Google Drive needed)",
    response_description="List of evaluation results, one per uploaded file.",
)
async def evaluate_batch(
    files: List[UploadFile] = File(
        ...,
        description="Batch of 5 to 20 image files. Select multiple files in the Swagger file picker.",
    ),
    user_text: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    **Batch upload and evaluate 5 to 20 images at once — no Google Drive required.**

    Each image runs through the full pipeline independently.
    Results are returned in the same order as the uploaded files.
    """
    if len(files) < 5:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch evaluation requires at least 5 images (received {len(files)}). For fewer images, please use the single upload endpoint."
        )
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch upload.")

    results = []
    for f in files:
        # Re-use single evaluate logic by building a fake request
        if f.content_type not in SUPPORTED_TYPES:
            results.append({
                "filename": f.filename,
                "decision": "rejected",
                "error": f"Unsupported type: {f.content_type}",
            })
            continue

        # Call evaluate_image by re-using its internals
        image_bytes = await f.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            results.append({
                "filename": f.filename,
                "decision": "rejected",
                "error": "File too large (max 20 MB)",
            })
            continue

        # Monkey-patch read so we can reuse the endpoint logic
        f_copy = UploadFile(
            filename=f.filename,
            file=io.BytesIO(image_bytes),
            headers=f.headers,
        )
        try:
            result = await evaluate_image(file=f_copy, user_text=user_text, db=db)
            results.append(result)
        except HTTPException as e:
            results.append({"filename": f.filename, "error": e.detail, "decision": "rejected"})
        except Exception as e:
            log.exception("batch.image_failed", filename=f.filename, error=str(e))
            results.append({"filename": f.filename, "error": str(e), "decision": "rejected"})

    summary = {
        "total": len(results),
        "approved": sum(1 for r in results if (r.get("decision") or r.get("status")) == "approved"),
        "rejected": sum(1 for r in results if (r.get("decision") or r.get("status")) == "rejected"),
    }

    return {"summary": summary, "results": results}
