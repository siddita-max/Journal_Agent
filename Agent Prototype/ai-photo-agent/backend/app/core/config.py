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
    GOOGLE_SERVICE_ACCOUNT_FILE: str = "/app/credentials/photo-agent-494106-23b977507358.json"
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
    CLIP_MODEL_NAME: str = "ViT-L/14@336px"
    YOLO_MODEL_PATH: str = "yolov8s.pt"
    YOLO_AUGMENT: bool = False
    QWEN_ENABLED: bool = True
    QWEN_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"  # 2B model fits in 4GB VRAM (RTX 3050)
    QWEN_DEVICE: str = "auto"  # auto | cpu | cuda
    QWEN_CACHE_ENABLED: bool = True
    DEVICE: str = "auto"  # auto | cpu | cuda

    # ── Scoring Thresholds ───────────────────────────────────────────
    SCORE_APPROVED_THRESHOLD: float = 0.75
    SCORE_REVIEW_THRESHOLD: float = 0.60

    # ── Preprocessing Limits ─────────────────────────────────────────
    MIN_RESOLUTION_WIDTH: int = 640
    MIN_RESOLUTION_HEIGHT: int = 480
    MIN_BLUR_VARIANCE: float = 100.0  # Increased for "clear not blur" requirement
    MIN_BRIGHTNESS: int = 40
    MAX_BRIGHTNESS: int = 220
    MIN_ASPECT_RATIO: float = 0.5
    MAX_ASPECT_RATIO: float = 3.0

    # ── Scoring Weights ──────────────────────────────────────────────
    WEIGHT_CLIP: float = 0.35
    WEIGHT_QUALITY: float = 0.25
    WEIGHT_RESOLUTION: float = 0.15
    WEIGHT_OBJECT: float = 0.15
    WEIGHT_AESTHETIC: float = 0.10

    # ── YOLO Policy ──────────────────────────────────────────────────
    def get_device(self) -> str:
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE

    def get_clip_model_name(self) -> str:
        value = (self.CLIP_MODEL_NAME or "").strip()
        if value.lower() != "auto":
            return value
        return "ViT-L/14@336px" if self.get_device() == "cuda" else "ViT-B/32"

    def get_yolo_model_path(self) -> str:
        value = (self.YOLO_MODEL_PATH or "").strip()
        if value.lower() != "auto":
            return value
        return "/app/models/yolov8s.pt" if self.get_device() == "cuda" else "/app/models/yolov8n.pt"  # nano on CPU, small on GPU

    def get_qwen_device(self) -> str:
        if self.QWEN_DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.QWEN_DEVICE

    def get_qwen_model_name(self) -> str:
        value = (self.QWEN_MODEL_NAME or "").strip()
        if value.lower() != "auto":
            return value
        return "Qwen/Qwen2-VL-2B-Instruct"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
