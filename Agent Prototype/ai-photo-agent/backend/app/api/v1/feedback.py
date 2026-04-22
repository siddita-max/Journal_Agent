"""Feedback API — human override history and AI-driven threshold suggestions."""
from typing import List, Optional
import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.models import FeedbackRecord, ImageDecision

log = structlog.get_logger()
router = APIRouter()


# ─── Schemas ──────────────────────────────────────────────────────────────

class FeedbackItem(BaseModel):
    id: str = Field(..., description="Feedback record UUID")
    image_id: str = Field(..., description="Image record UUID")
    original_decision: Optional[str] = Field(None, description="AI-generated decision before override")
    override_decision: Optional[str] = Field(None, description="Human override decision")
    reason: Optional[str] = Field(None, description="Annotator-provided reason for the override")
    annotator: Optional[str] = Field(None, description="Email or ID of the annotator")
    used_for_training: bool = Field(..., description="Whether this record has been used for model training")


class ThresholdSuggestion(BaseModel):
    current_approved_threshold: float = Field(..., description="Currently active approval threshold")
    current_review_threshold: float = Field(..., description="Currently active review threshold")
    suggested_approved_threshold: float = Field(..., description="Suggested new approval threshold based on feedback analysis")
    suggested_review_threshold: float = Field(..., description="Suggested new review threshold based on feedback analysis")
    feedback_samples: int = Field(..., description="Total number of feedback records analysed")
    false_approvals: int = Field(..., description="Cases where AI approved but human rejected")
    false_rejections: int = Field(..., description="Cases where AI rejected but human approved")
    override_rate_pct: float = Field(..., description="Overall human override rate as a percentage")
    note: str = Field(..., description="Human-readable explanation of the suggestion")
    action_required: bool = Field(..., description="True if the suggestion differs meaningfully from current thresholds")


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=List[FeedbackItem],
    summary="List all human feedback records",
    response_description="All override records ordered by most recent first.",
)
async def list_feedback(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve all human decision overrides stored in the system.

    These records are:
    - Written every time `POST /images/{id}/override` is called
    - Used by `/feedback/threshold-suggestion` for adaptive threshold tuning
    - Exportable for future supervised fine-tuning of CLIP or custom classifiers
    """
    result = await db.execute(
        select(FeedbackRecord)
        .order_by(FeedbackRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    records = result.scalars().all()
    return [
        {
            "id": str(r.id),
            "image_id": str(r.image_id),
            "original_decision": r.original_decision.value if r.original_decision else None,
            "override_decision": r.override_decision.value if r.override_decision else None,
            "reason": r.reason,
            "annotator": r.annotator,
            "used_for_training": r.used_for_training,
        }
        for r in records
    ]


@router.get(
    "/threshold-suggestion",
    response_model=ThresholdSuggestion,
    summary="Get AI-driven scoring threshold suggestions",
    response_description="Threshold adjustment recommendations based on accumulated feedback patterns.",
    responses={
        200: {
            "description": "Threshold analysis result",
            "content": {
                "application/json": {
                    "example": {
                        "current_approved_threshold": 0.75,
                        "current_review_threshold": 0.60,
                        "suggested_approved_threshold": 0.80,
                        "suggested_review_threshold": 0.60,
                        "feedback_samples": 45,
                        "false_approvals": 8,
                        "false_rejections": 2,
                        "override_rate_pct": 22.2,
                        "note": "High false approval rate (18%). Consider raising the approved threshold (+0.05).",
                        "action_required": True,
                    }
                }
            },
        }
    },
)
async def suggest_thresholds(db: AsyncSession = Depends(get_db)):
    """
    Analyse accumulated human feedback to recommend scoring threshold adjustments.

    **Algorithm:**
    1. Count overrides where AI approved → human rejected (`false_approvals`)
    2. Count overrides where AI rejected → human approved (`false_rejections`)
    3. If `false_approvals / total > 15%` → suggest raising `approved_threshold` by 0.05
    4. If `false_rejections / total > 15%` → suggest lowering `approved_threshold` by 0.05

    **When to apply:** Manually update `SCORE_APPROVED_THRESHOLD` in `.env` and redeploy,
    or call the `PATCH /api/v1/config/thresholds` endpoint (enterprise extension).

    Requires at least **10 feedback samples** before providing meaningful suggestions.
    """
    # Aggregate override patterns
    result = await db.execute(
        select(
            FeedbackRecord.original_decision,
            FeedbackRecord.override_decision,
            func.count(FeedbackRecord.id).label("cnt"),
        ).group_by(
            FeedbackRecord.original_decision,
            FeedbackRecord.override_decision,
        )
    )
    rows = result.all()

    false_approvals = 0
    false_rejections = 0
    total = 0

    for row in rows:
        total += row.cnt
        if (
            row.original_decision == ImageDecision.APPROVED
            and row.override_decision == ImageDecision.REJECTED
        ):
            false_approvals += row.cnt
        elif (
            row.original_decision == ImageDecision.REJECTED
            and row.override_decision == ImageDecision.APPROVED
        ):
            false_rejections += row.cnt

    current_approved = settings.SCORE_APPROVED_THRESHOLD
    current_review = settings.SCORE_REVIEW_THRESHOLD
    suggested_approved = current_approved
    suggested_review = current_review
    action_required = False

    if total >= 10:
        fa_rate = false_approvals / total
        fr_rate = false_rejections / total
        if fa_rate > 0.15:
            suggested_approved = min(0.95, round(current_approved + 0.05, 2))
            action_required = True
        if fr_rate > 0.15:
            suggested_approved = max(0.50, round(current_approved - 0.05, 2))
            action_required = True

    override_rate = round((false_approvals + false_rejections) / max(total, 1) * 100, 1)

    if total < 10:
        note = f"Insufficient feedback ({total} samples). Collect at least 10 overrides for meaningful suggestions."
    elif not action_required:
        note = f"Thresholds appear well-calibrated based on {total} feedback samples. No adjustment needed."
    elif false_approvals > false_rejections:
        note = f"High false-approval rate ({round(false_approvals/total*100,1)}%). Raising the approved threshold to {suggested_approved} is recommended."
    else:
        note = f"High false-rejection rate ({round(false_rejections/total*100,1)}%). Lowering the approved threshold to {suggested_approved} is recommended."

    return {
        "current_approved_threshold": current_approved,
        "current_review_threshold": current_review,
        "suggested_approved_threshold": suggested_approved,
        "suggested_review_threshold": suggested_review,
        "feedback_samples": total,
        "false_approvals": false_approvals,
        "false_rejections": false_rejections,
        "override_rate_pct": override_rate,
        "note": note,
        "action_required": action_required,
    }
