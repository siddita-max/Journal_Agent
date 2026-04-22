
"""Policies API — CRUD for company policy configurations."""
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import PolicyConfig, AuditLog
from app.services.policy_engine import PolicyEngine, DEFAULT_POLICY

log = structlog.get_logger()
router = APIRouter()


# ─── Schemas ──────────────────────────────────────────────────────────────

class PolicyCreateRequest(BaseModel):
    name: str = Field(
        ...,
        description="Human-readable policy name",
        examples=["Standard Corporate Policy v2"],
    )
    description: Optional[str] = Field(
        None,
        description="What this policy is for",
        examples=["Used for all internal recruitment photo submissions"],
    )
    rules: Dict[str, Any] = Field(
        ...,
        description="""
Policy rules document. All fields are optional; omitted fields fall back to defaults.

**Supported fields:**
```json
{
  "max_people": 10,
  "prohibited_objects": ["cell phone", "laptop"],
  "min_professionalism_score": 0.55,
  "required_context_prompts": ["corporate office", "professional setting"],
  "dress_code": {
    "enabled": true,
    "required_prompts": ["formal attire", "business dress"],
    "min_score": 0.40
  },
  "custom_hard_rules": [
    {
      "name": "no_outdoor",
      "description": "Must be indoor/office setting",
      "clip_negative_prompts": ["outdoor photo", "street photography"],
      "threshold": 0.55
    }
  ],
  "custom_soft_rules": [
    {
      "name": "brand_background",
      "description": "Prefer clean/branded background",
      "clip_positive_prompts": ["clean background", "office background"],
      "weight": 0.10,
      "min_score": 0.30
    }
  ]
}
```
**Hard rules** → immediate rejection regardless of score.
**Soft rules** → reduce the object_compliance_score component.
        """,
        examples=[{
            "max_people": 5,
            "prohibited_objects": ["cell phone"],
            "min_professionalism_score": 0.55,
            "dress_code": {"enabled": True, "required_prompts": ["formal attire"], "min_score": 0.40},
        }],
    )
    activate: bool = Field(
        False,
        description="If true, immediately set this policy as the active one after creation",
    )


class PolicyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    rules: Dict[str, Any]
    is_active: bool
    version: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class PolicyUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="New policy name")
    description: Optional[str] = Field(None, description="Updated description")
    rules: Optional[Dict[str, Any]] = Field(None, description="New rules document (replaces existing)")


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PolicyResponse,
    status_code=201,
    summary="Create a new company policy",
    responses={
        201: {"description": "Policy created"},
        422: {"description": "Invalid policy document — validation errors returned"},
    },
)
async def create_policy(
    payload: PolicyCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new versioned company policy configuration.

    The `rules` document is validated before saving. Any validation errors are
    returned as a structured 422 response so you can fix them in Swagger.

    If `activate=true`, all other policies are deactivated and this one becomes live immediately.
    Running Celery workers pick up the new active policy on the **next job** submitted.

    **Policy enforcement happens at inference time, per job.** The active policy
    at job submission time is snapshotted into the job and applied to every image in that batch.
    """
    # Validate rules doc
    errors = PolicyEngine.validate_policy_document(payload.rules)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "Policy document validation failed", "errors": errors},
        )

    # Version = max existing + 1
    result = await db.execute(select(PolicyConfig).order_by(PolicyConfig.version.desc()).limit(1))
    latest = result.scalar_one_or_none()
    next_version = (latest.version + 1) if latest else 1

    # Deactivate all if this one should be active
    if payload.activate:
        existing = await db.execute(select(PolicyConfig).where(PolicyConfig.is_active == True))
        for p in existing.scalars().all():
            p.is_active = False

    policy = PolicyConfig(
        name=payload.name,
        description=payload.description,
        rules=payload.rules,
        is_active=payload.activate,
        version=next_version,
    )
    db.add(policy)
    db.add(AuditLog(
        entity_type="policy",
        entity_id=str(policy.id) if policy.id else "new",
        action="created",
        details={"name": payload.name, "version": next_version, "activated": payload.activate},
    ))
    await db.commit()
    await db.refresh(policy)

    log.info("api.policy_created", name=payload.name, version=next_version, active=payload.activate)
    return _to_response(policy)


@router.get(
    "/",
    response_model=List[PolicyResponse],
    summary="List all policy configurations",
)
async def list_policies(
    active_only: bool = Query(False, description="Return only the currently active policy"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all saved policy configurations, ordered by version (newest first).
    At most one policy is active at a time.
    """
    stmt = select(PolicyConfig).order_by(PolicyConfig.version.desc())
    if active_only:
        stmt = stmt.where(PolicyConfig.is_active == True)
    result = await db.execute(stmt)
    return [_to_response(p) for p in result.scalars().all()]


@router.get(
    "/active",
    response_model=PolicyResponse,
    summary="Get the currently active policy",
    responses={404: {"description": "No active policy set — the system uses built-in defaults"}},
)
async def get_active_policy(db: AsyncSession = Depends(get_db)):
    """
    Returns the currently active policy.

    If no policy is active, the system uses internal defaults:
    - `max_people`: 10
    - `prohibited_objects`: `["cell phone"]`
    - `min_professionalism_score`: 0.0 (not enforced)
    - Dress code: disabled
    """
    result = await db.execute(select(PolicyConfig).where(PolicyConfig.is_active == True).limit(1))
    policy = result.scalar_one_or_none()
    if not policy:
        raise HTTPException(
            status_code=404,
            detail="No active policy found. System is using built-in defaults. POST /policies/ with activate=true to set one.",
        )
    return _to_response(policy)


@router.get(
    "/default",
    summary="Get the built-in default policy document",
    response_description="The default policy rules applied when no DB policy is active.",
)
async def get_default_policy():
    """
    Returns the built-in default policy rules.
    Useful as a starting template when creating a new policy via POST /policies/.
    """
    return {"default_policy": DEFAULT_POLICY}


@router.get(
    "/{policy_id}",
    response_model=PolicyResponse,
    summary="Get a specific policy by ID",
    responses={404: {"description": "Policy not found"}},
)
async def get_policy(policy_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve a specific policy configuration by its UUID."""
    policy = await _get_or_404(policy_id, db)
    return _to_response(policy)


@router.patch(
    "/{policy_id}",
    response_model=PolicyResponse,
    summary="Update a policy (name, description, or rules)",
    responses={
        404: {"description": "Policy not found"},
        422: {"description": "Invalid rules document"},
    },
)
async def update_policy(
    policy_id: str,
    payload: PolicyUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Update a policy's name, description, or rules document.
    Validation runs on the new rules before saving.

    **Note:** Updating a policy does not automatically reprocess existing jobs.
    The new rules take effect on the next submitted job.
    """
    policy = await _get_or_404(policy_id, db)

    if payload.rules is not None:
        errors = PolicyEngine.validate_policy_document(payload.rules)
        if errors:
            raise HTTPException(
                status_code=422,
                detail={"message": "Policy document validation failed", "errors": errors},
            )
        policy.rules = payload.rules

    if payload.name is not None:
        policy.name = payload.name
    if payload.description is not None:
        policy.description = payload.description

    policy.updated_at = datetime.now(timezone.utc)
    db.add(AuditLog(
        entity_type="policy",
        entity_id=str(policy.id),
        action="updated",
        details={"fields_changed": list(payload.model_fields_set)},
    ))
    await db.commit()
    await db.refresh(policy)
    return _to_response(policy)


@router.post(
    "/{policy_id}/activate",
    response_model=PolicyResponse,
    summary="Activate a policy (deactivates all others)",
    responses={404: {"description": "Policy not found"}},
)
async def activate_policy(policy_id: str, db: AsyncSession = Depends(get_db)):
    """
    Set a policy as the active one. All other policies are deactivated atomically.
    The newly activated policy applies to all jobs submitted after this call.
    """
    policy = await _get_or_404(policy_id, db)

    # Deactivate all
    existing = await db.execute(select(PolicyConfig).where(PolicyConfig.is_active == True))
    for p in existing.scalars().all():
        p.is_active = False

    policy.is_active = True
    policy.updated_at = datetime.now(timezone.utc)
    db.add(AuditLog(
        entity_type="policy",
        entity_id=str(policy.id),
        action="activated",
        details={"policy_name": policy.name, "version": policy.version},
    ))
    await db.commit()
    await db.refresh(policy)
    log.info("api.policy_activated", policy_id=policy_id, name=policy.name)
    return _to_response(policy)


@router.delete(
    "/{policy_id}",
    status_code=204,
    summary="Delete a policy",
    responses={
        400: {"description": "Cannot delete the currently active policy"},
        404: {"description": "Policy not found"},
    },
)
async def delete_policy(policy_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a policy configuration. Active policies cannot be deleted —
    activate another policy first, then delete this one.
    """
    policy = await _get_or_404(policy_id, db)
    if policy.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the active policy. Activate a different policy first.",
        )
    db.add(AuditLog(
        entity_type="policy",
        entity_id=str(policy.id),
        action="deleted",
        details={"name": policy.name},
    ))
    await db.delete(policy)
    await db.commit()
    log.info("api.policy_deleted", policy_id=policy_id)


# ─── Helpers ──────────────────────────────────────────────────────────────

async def _get_or_404(policy_id: str, db: AsyncSession) -> PolicyConfig:
    try:
        uid = uuid.UUID(policy_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid policy ID — must be a UUID")
    result = await db.execute(select(PolicyConfig).where(PolicyConfig.id == uid))
    policy = result.scalar_one_or_none()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


def _to_response(p: PolicyConfig) -> dict:
    return {
        "id": str(p.id),
        "name": p.name,
        "description": p.description,
        "rules": p.rules or {},
        "is_active": p.is_active,
        "version": p.version,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
    }
