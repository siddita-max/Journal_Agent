"""
Centralized configuration using pydantic-settings.
All values are sourced from environment variables / .env file.
"""
from functools import lru_cache
from typing import List, Union, Any
import json
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    AUDIT_LOG_ENABLED: bool = True

    # ── Security ─────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me"
    ALLOWED_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("ALLOWED_ORIGINS", "GOOGLE_DRIVE_SCOPES", mode="before")
    @classmethod
    def _parse_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    # Fallback to comma split if it looks like JSON but isn't valid
                    pass
            return [v.strip() for v in v.split(",") if v.strip()]
        return v

    # ── Google Drive ─────────────────────────────────────────────────
    GOOGLE_SERVICE_ACCOUNT_FILE: str = "/app/credentials/photo-agent-494106-a73062df921d.json"
    GOOGLE_DRIVE_SCOPES: Union[List[str], str] = ["https://www.googleapis.com/auth/drive.readonly"]

    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://photo_agent:photo_agent_secret@localhost:5432/photo_agent_db"

    # ── Redis / Celery ───────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_CONCURRENCY: int = 4

    # ── MinIO ────────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_PUBLIC_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET_APPROVED: str = "approved-images"
    MINIO_BUCKET_REJECTED: str = "rejected-images"
    MINIO_SECURE: bool = False

    def get_minio_public_endpoint(self) -> str:
        """Return the public MinIO endpoint (for presigned URLs the browser opens)."""
        return self.MINIO_PUBLIC_ENDPOINT.strip() or self.MINIO_ENDPOINT

    # ── AI Models ────────────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "ViT-B/32"
    YOLO_MODEL_PATH: str = "yolov8s.pt"
    YOLO_AUGMENT: bool = False
    DEVICE: str = "cpu"

    # ── Groq Vision API (primary multimodal analyser) ────────────────
    # When GROQ_ENABLED=true and GROQ_API_KEY is provided, the inference
    # pipeline calls Groq's chat-completions vision endpoint for activity
    # detection, surroundings description, and journal-title matching.
    GROQ_ENABLED: bool = True
    GROQ_API_KEY: str = ""
    # Comma-separated extra keys for round-robin rotation across free-tier quotas.
    # E.g. GROQ_API_KEYS=gsk_key2,gsk_key3  (GROQ_API_KEY is always key #1)
    GROQ_API_KEYS: str = ""
    GROQ_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GROQ_TIMEOUT_S: float = 45.0
    GROQ_MAX_TOKENS: int = 200
    GROQ_CACHE_ENABLED: bool = True
    GROQ_MATCH_REQUIRED: bool = True  # Hard-reject photos that don't match the journal title
    # Skip Groq when YOLO detects zero people — saves ~30-50 % of API calls.
    GROQ_YOLO_PREFILTER: bool = True
    # Optional inter-call delay (seconds) to stay under per-minute rate limits.
    GROQ_CALL_DELAY_S: float = 0.0

    # ── Scoring Thresholds ───────────────────────────────────────────
    SCORE_APPROVED_THRESHOLD: float = 0.65
    SCORE_REVIEW_THRESHOLD: float = 0.50

    # ── Preprocessing Limits ─────────────────────────────────────────
    # Social-media uploads (Facebook/Instagram) are typically 600×600 px.
    # Use 480×360 as minimum to pass those while still rejecting tiny thumbnails.
    MIN_RESOLUTION_WIDTH: int = 480
    MIN_RESOLUTION_HEIGHT: int = 360
    MIN_BLUR_VARIANCE: float = 80.0   # Rejects clearly blurry shots; sharp activity photos score 200+
    MIN_BRIGHTNESS: int = 45          # Reject underexposed / very dark frames
    MAX_BRIGHTNESS: int = 215         # Reject blown-out / heavily overexposed frames
    MIN_ASPECT_RATIO: float = 0.5
    MAX_ASPECT_RATIO: float = 2.5     # Reject unusually wide panoramas

    # ── Scoring Weights ──────────────────────────────────────────────
    WEIGHT_CLIP: float = 0.35
    WEIGHT_QUALITY: float = 0.25
    WEIGHT_RESOLUTION: float = 0.15
    WEIGHT_OBJECT: float = 0.15
    WEIGHT_AESTHETIC: float = 0.10

    def get_device(self) -> str:
        return "cpu"

    def get_clip_model_name(self) -> str:
        return (self.CLIP_MODEL_NAME or "ViT-L/14@336px").strip()

    def get_yolo_model_path(self) -> str:
        return (self.YOLO_MODEL_PATH or "yolov8s.pt").strip()


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
