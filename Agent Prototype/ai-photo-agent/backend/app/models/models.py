"""
SQLAlchemy ORM models for the Photo Agent system.
"""
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, Enum, func, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


# ─── Policy Config ────────────────────────────────────────────────────────

class PolicyConfig(Base):
    """
    Versioned company photo policy stored as a JSON document.
    Exactly one policy can be active at a time.
    Active policy is applied to every image in a submitted job.
    """
    __tablename__ = "policy_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    rules = Column(JSON, nullable=False)   # Full policy document
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    version = Column(Integer, nullable=False, unique=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=True)

    jobs = relationship("ProcessingJob", back_populates="policy")


# ─── Enums ────────────────────────────────────────────────────────────────

class JobStatus(str, PyEnum):
    PENDING = "pending"
    FETCHING = "fetching"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImageDecision(str, PyEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REVIEW = "review"
    PENDING = "pending"


# ─── Processing Job ───────────────────────────────────────────────────────

class ProcessingJob(Base):
    """Represents a batch photo evaluation job."""
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drive_folder_url = Column(String(1024), nullable=False)
    drive_folder_id = Column(String(256), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    celery_task_id = Column(String(256), nullable=True)

    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    approved_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    review_count = Column(Integer, default=0)

    error_message = Column(Text, nullable=True)
    report_url = Column(String(1024), nullable=True)

    # Policy snapshot: which policy was active when this job was submitted
    policy_id = Column(UUID(as_uuid=True), ForeignKey("policy_configs.id"), nullable=True)
    policy_snapshot = Column(JSON, nullable=True)  # Copy of rules at job submission time

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    images = relationship("ImageRecord", back_populates="job", cascade="all, delete-orphan")
    policy = relationship("PolicyConfig", back_populates="jobs")

    @property
    def progress_pct(self) -> float:
        if self.total_images == 0:
            return 0.0
        return round((self.processed_images / self.total_images) * 100, 1)


# ─── Image Record ─────────────────────────────────────────────────────────

class ImageRecord(Base):
    """Stores per-image metadata, scores, and decision."""
    __tablename__ = "image_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("processing_jobs.id"), nullable=True, index=True)

    # Source metadata
    drive_file_id = Column(String(256), nullable=False)
    filename = Column(String(512), nullable=False)
    mime_type = Column(String(128), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    # Preprocessing results
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    blur_variance = Column(Float, nullable=True)
    brightness = Column(Float, nullable=True)
    aspect_ratio = Column(Float, nullable=True)

    # AI scores
    clip_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    resolution_score = Column(Float, nullable=True)
    object_compliance_score = Column(Float, nullable=True)
    aesthetic_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True, index=True)

    # YOLO detections
    people_count = Column(Integer, nullable=True)
    student_count = Column(Integer, nullable=True)
    teacher_count = Column(Integer, nullable=True)
    detected_activity = Column(String(256), nullable=True)
    phone_detected = Column(Boolean, nullable=True)
    detected_objects = Column(JSON, nullable=True)  # list of {class, confidence}

    # Safety
    nsfw_detected = Column(Boolean, default=False, nullable=False)
    nsfw_score = Column(Float, nullable=True)

    # Decision
    decision = Column(Enum(ImageDecision), default=ImageDecision.PENDING, nullable=False, index=True)
    rejection_reasons = Column(JSON, nullable=True)   # List[str]
    score_breakdown = Column(JSON, nullable=True)     # Full scoring dict

    # Policy enforcement
    policy_violations = Column(JSON, nullable=True)   # List of {rule_name, severity, description, measured_value, threshold}
    policy_compliance_score = Column(Float, nullable=True)  # 0–1 from PolicyEngine

    # Storage
    storage_url = Column(String(1024), nullable=True)  # MinIO object path

    # Human override
    human_override = Column(Enum(ImageDecision), nullable=True)
    override_reason = Column(Text, nullable=True)
    overridden_by = Column(String(256), nullable=True)
    overridden_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("ProcessingJob", back_populates="images")
    feedback = relationship("FeedbackRecord", back_populates="image", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_image_records_job_decision", "job_id", "decision"),
    )


# ─── Audit Log ────────────────────────────────────────────────────────────

class AuditLog(Base):
    """Immutable audit trail for all system decisions."""
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(64), nullable=False)  # "image" | "job"
    entity_id = Column(String(256), nullable=False)
    action = Column(String(128), nullable=False)
    actor = Column(String(256), nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
    )


# ─── Feedback Record ──────────────────────────────────────────────────────

class FeedbackRecord(Base):
    """Stores human feedback for threshold tuning & future training."""
    __tablename__ = "feedback_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("image_records.id"), nullable=False, index=True)

    original_decision = Column(Enum(ImageDecision), nullable=False)
    override_decision = Column(Enum(ImageDecision), nullable=False)
    reason = Column(Text, nullable=True)
    annotator = Column(String(256), nullable=True)

    # For future model training
    used_for_training = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    image = relationship("ImageRecord", back_populates="feedback")


# ─── Threshold Config ─────────────────────────────────────────────────────

class ThresholdConfig(Base):
    """Versioned scoring threshold configuration."""
    __tablename__ = "threshold_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    approved_threshold = Column(Float, default=0.75)
    review_threshold = Column(Float, default=0.60)
    weight_clip = Column(Float, default=0.35)
    weight_quality = Column(Float, default=0.25)
    weight_resolution = Column(Float, default=0.15)
    weight_object = Column(Float, default=0.15)
    weight_aesthetic = Column(Float, default=0.10)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)
