"""Policies API - fixed school journal policy storage."""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import AuditLog, PolicyConfig
from app.services.policy_engine import get_school_child_safety_policy

router = APIRouter()


class PolicyResponse(BaseModel):
    id: Optional[str] = Field(None, description="Database policy UUID when stored")
    stored: bool = Field(..., description="Whether the policy is persisted in DB")
    active: bool = Field(..., description="Whether this is the active DB policy")
    policy: Dict[str, Any] = Field(..., description="School journal policy JSON")


@router.get(
    "/",
    response_model=PolicyResponse,
    summary="Get the school journal policy",
)
async def get_policy(db: AsyncSession = Depends(get_db)):
    """
    Return the exact fixed school journal policy.

    If it has already been stored through `POST /api/v1/policies/`, this returns
    the active DB copy. Otherwise, it returns the built-in policy JSON.
    """
    result = await db.execute(
        select(PolicyConfig)
        .where(PolicyConfig.name == "school_journal_policy_v1")
        .order_by(PolicyConfig.created_at.desc())
        .limit(1)
    )
    stored_policy = result.scalar_one_or_none()
    if stored_policy:
        return {
            "id": str(stored_policy.id),
            "stored": True,
            "active": stored_policy.is_active,
            "policy": stored_policy.rules,
        }

    return {
        "id": None,
        "stored": False,
        "active": False,
        "policy": get_school_child_safety_policy(),
    }


@router.post(
    "/",
    response_model=PolicyResponse,
    summary="Store the school journal policy in DB",
)
async def store_policy(db: AsyncSession = Depends(get_db)):
    """
    Store or refresh the exact fixed `school_journal_policy_v1` JSON in DB.

    This endpoint does not accept arbitrary policy input; it persists the
    application-owned policy so jobs can snapshot it consistently.
    """
    policy_json = get_school_child_safety_policy()

    existing_active = await db.execute(select(PolicyConfig).where(PolicyConfig.is_active == True))
    for policy in existing_active.scalars().all():
        policy.is_active = False

    result = await db.execute(
        select(PolicyConfig)
        .where(PolicyConfig.name == policy_json["name"])
        .order_by(PolicyConfig.created_at.desc())
        .limit(1)
    )
    stored_policy = result.scalar_one_or_none()

    if stored_policy:
        stored_policy.description = policy_json["description"]
        stored_policy.rules = policy_json
        stored_policy.is_active = True
        stored_policy.updated_at = datetime.now(timezone.utc)
        action = "refreshed"
    else:
        latest = await db.execute(select(PolicyConfig).order_by(PolicyConfig.version.desc()).limit(1))
        latest_policy = latest.scalar_one_or_none()
        next_version = (latest_policy.version + 1) if latest_policy else 1
        stored_policy = PolicyConfig(
            name=policy_json["name"],
            description=policy_json["description"],
            rules=policy_json,
            is_active=True,
            version=next_version,
        )
        db.add(stored_policy)
        action = "stored"

    db.add(AuditLog(
        entity_type="policy",
        entity_id=str(stored_policy.id) if stored_policy.id else "school_journal_policy_v1",
        action=action,
        details={"name": policy_json["name"], "version": policy_json["version"]},
    ))
    await db.commit()
    await db.refresh(stored_policy)

    return {
        "id": str(stored_policy.id),
        "stored": True,
        "active": stored_policy.is_active,
        "policy": stored_policy.rules,
    }
