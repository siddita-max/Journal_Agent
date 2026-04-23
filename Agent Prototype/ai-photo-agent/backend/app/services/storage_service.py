"""
Storage Service — manages object storage (MinIO / S3-compatible).
Handles upload/download of processed images and JSON reports.
"""
import io
import json
from datetime import timedelta
from typing import Optional
import structlog
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

log = structlog.get_logger()


class StorageService:
    """MinIO wrapper for image and report storage."""

    def __init__(self):
        self._client: Optional[Minio] = None
        self._public_client: Optional[Minio] = None
        self._initialized = False

    def _get_client(self) -> Minio:
        """Return the internal client (uses Docker hostname — for server-side ops)."""
        if self._client is None:
            endpoint = settings.MINIO_ENDPOINT
            log.info("storage.init_client", endpoint=endpoint)
            self._client = Minio(
                endpoint,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                region="us-east-1",
            )
        return self._client

    def _get_public_client(self) -> Minio:
        """
        Return a client pointed at the *public* MinIO endpoint.
        This is used exclusively for generating presigned URLs that the
        browser will open. When MINIO_PUBLIC_ENDPOINT is not set it falls
        back to MINIO_ENDPOINT (same behaviour as before).
        """
        if self._public_client is None:
            self._public_client = Minio(
                settings.get_minio_public_endpoint(),
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                region="us-east-1",
            )
        return self._public_client

    def ensure_buckets(self):
        """Create buckets if they don't exist (idempotent)."""
        client = self._get_client()
        for bucket in [settings.MINIO_BUCKET_APPROVED, settings.MINIO_BUCKET_REJECTED, "reports"]:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                log.info("storage.bucket_created", bucket=bucket)

    def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        decision: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """
        Upload image to the appropriate bucket based on decision.
        Returns the object path (bucket/filename).
        """
        client = self._get_client()
        bucket = (
            settings.MINIO_BUCKET_APPROVED
            if decision == "approved"
            else settings.MINIO_BUCKET_REJECTED
        )
        object_name = filename
        try:
            client.put_object(
                bucket,
                object_name,
                io.BytesIO(image_bytes),
                length=len(image_bytes),
                content_type=content_type,
            )
            path = f"{bucket}/{object_name}"
            log.info("storage.uploaded", path=path, size=len(image_bytes))
            return path
        except S3Error as e:
            log.error("storage.upload_failed", filename=filename, error=str(e))
            raise

    def upload_report(self, job_id: str, report: dict) -> str:
        """Upload JSON report for a completed job. Returns presigned URL."""
        client = self._get_client()
        payload = json.dumps(report, indent=2, default=str).encode()
        object_name = f"{job_id}/report.json"
        try:
            client.put_object(
                "reports",
                object_name,
                io.BytesIO(payload),
                length=len(payload),
                content_type="application/json",
            )
            # Use public client so the URL is browser-accessible
            url = self._get_public_client().presigned_get_object(
                "reports", object_name, expires=timedelta(days=7)
            )
            log.info("storage.report_uploaded", job_id=job_id)
            return url
        except S3Error as e:
            log.error("storage.report_failed", job_id=job_id, error=str(e))
            raise

    def get_presigned_url(self, bucket: str, object_name: str, expires_hours: int = 24) -> str:
        """
        Generate a presigned URL for temporary browser access.
        Uses the *public* MinIO endpoint so the URL resolves in the browser
        (avoids the Docker-internal 'minio:9000' hostname leaking to clients).
        """
        return self._get_public_client().presigned_get_object(
            bucket, object_name, expires=timedelta(hours=expires_hours)
        )
