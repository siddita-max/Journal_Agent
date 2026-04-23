"""
Google Drive Service — fetches image files from a Drive folder.
Uses service account credentials for secure, server-side access.
"""
import re
import io
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Generator, Optional
import structlog
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from app.core.config import settings

log = structlog.get_logger()

# Supported image MIME types
IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

HD_MIN_BYTES = 500_000
HD_PREFERRED_BYTES = 1_500_000
LOCAL_TZ = ZoneInfo("Asia/Kolkata")

FOLDER_ID_PATTERNS = [
    r"(?:/folders/|id=)([a-zA-Z0-9_-]{25,})",
    r"([a-zA-Z0-9_-]{25,})",
]


def extract_folder_id(drive_url: str) -> str:
    """Extract the folder ID from a Google Drive URL."""
    for pattern in FOLDER_ID_PATTERNS:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract folder ID from URL: {drive_url}")


class GoogleDriveService:
    """
    Service class for interacting with Google Drive API.
    
    Supports:
    - Listing image files in a folder
    - Streaming file downloads (memory-efficient)
    - Batch fetching with pagination
    """

    def __init__(self):
        self._service = None
        self._credential_file = self._resolve_credential_file()

    def _get_service(self):
        if self._service is None:
            if not self._credential_file.exists():
                raise FileNotFoundError(
                    f"Google Drive service account file not found: {self._credential_file}"
                )
            creds = service_account.Credentials.from_service_account_file(
                str(self._credential_file),
                scopes=settings.GOOGLE_DRIVE_SCOPES,
            )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
            log.info("drive.service_initialized")
        return self._service

    @staticmethod
    def _resolve_credential_file() -> Path:
        """Find the uploaded service account JSON in Docker or local workspace."""
        candidate_paths = [
            Path(settings.GOOGLE_SERVICE_ACCOUNT_FILE),
            Path("/app/credentials/photo-agent-494106-23b977507358.json"),
            Path("/app/credentials/service_account.json"),
            Path("backend/credentials/photo-agent-494106-23b977507358.json"),
            Path("credentials/service_account.json"),
        ]
        for path in candidate_paths:
            if path.exists():
                return path
        return candidate_paths[0]

    def list_images(
        self,
        folder_id: str,
        page_size: int = 100,
        max_retries: int = 3,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields image file metadata dicts from a Drive folder.
        Handles pagination automatically.

        Enforces a strict "today only" rule:
        - only files created/modified on the current local date are accepted
        - if any older image exists in the folder, the call fails with a clear error
        File size is treated as a ranking signal, not a hard reject.
        """
        service = self._get_service()
        mime_query = " or ".join(
            [f"mimeType='{m}'" for m in IMAGE_MIME_TYPES]
        )
        query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"

        page_token: Optional[str] = None
        total_fetched = 0
        all_files = []
        today = datetime.now(LOCAL_TZ).date()
        low_quality_today = []
        missing_time_meta = []

        while True:
            for attempt in range(max_retries):
                try:
                    response = service.files().list(
                        q=query,
                        pageSize=page_size,
                        fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime)",
                        pageToken=page_token,
                    ).execute()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait = 2 ** attempt
                    log.warning("drive.list_retry", attempt=attempt + 1, wait=wait, error=str(e))
                    time.sleep(wait)

            files = response.get("files", [])
            for f in files:
                created_time = self._parse_drive_timestamp(f.get("createdTime"))
                modified_time = self._parse_drive_timestamp(f.get("modifiedTime"))
                effective_time = created_time or modified_time
                if effective_time is None:
                    missing_time_meta.append(f.get("name", f.get("id", "unknown")))
                    continue

                # Date validation: Check if image is from today (local time)
                effective_date = effective_time.astimezone(LOCAL_TZ).date()
                is_wrong_date = (effective_date != today)
                size_val = int(f.get("size", 0))

                all_files.append({
                    "id": f["id"],
                    "name": f["name"],
                    "mime_type": f.get("mimeType", ""),
                    "size": size_val,
                    "created_time": (created_time or effective_time).isoformat(),
                    "is_wrong_date": is_wrong_date,
                    "actual_date": str(effective_date),
                })

            page_token = response.get("nextPageToken")
            if not page_token:
                if not all_files:
                    if missing_time_meta:
                        meta_names = ", ".join(missing_time_meta[:5])
                        raise ValueError(
                            "Drive returned image files without created/modified timestamps, so date validation failed. "
                            f"Examples: {meta_names}"
                        )
                    raise ValueError(
                        f"No images (JPEG, PNG, WEBP) found in the Drive folder. "
                        f"Ensure the folder contains images and the service account has Viewer access."
                    )
                if low_quality_today and len(low_quality_today) == len(all_files):
                    log.warning(
                        "drive.all_today_files_below_min_size",
                        minimum_size=HD_MIN_BYTES,
                        total_today=len(all_files),
                    )
                all_files.sort(key=self._drive_quality_score, reverse=True)
                for f in all_files:
                    yield f
                    total_fetched += 1
                log.info("drive.list_complete", total=total_fetched, folder_id=folder_id)
                break

    @staticmethod
    def _drive_quality_score(file_meta: dict) -> float:
        size = int(file_meta.get("size", 0))
        name = file_meta.get("name", "").lower()
        penalty = 0.5 if any(x in name for x in ["thumb", "small", "compressed", "_s."]) else 1.0
        preferred_bonus = 1.1 if size >= HD_PREFERRED_BYTES else 1.0
        return size * penalty * preferred_bonus

    @staticmethod
    def _parse_drive_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def download_image(
        self,
        file_id: str,
        max_retries: int = 3,
    ) -> bytes:
        """
        Download a Drive file and return raw bytes.
        Uses exponential backoff on transient errors.
        """
        service = self._get_service()

        for attempt in range(max_retries):
            try:
                request = service.files().get_media(fileId=file_id)
                buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(buffer, request, chunksize=4 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return buffer.getvalue()
            except Exception as e:
                if attempt == max_retries - 1:
                    log.error("drive.download_failed", file_id=file_id, error=str(e))
                    raise
                wait = 2 ** attempt
                log.warning("drive.download_retry", attempt=attempt + 1, file_id=file_id, wait=wait)
                time.sleep(wait)

    def validate_folder_access(self, folder_id: str) -> dict:
        """
        Verify the service account can access the given folder.
        Returns folder metadata if accessible.
        """
        service = self._get_service()
        try:
            meta = service.files().get(
                fileId=folder_id,
                fields="id, name, mimeType"
            ).execute()
            if meta.get("mimeType") != "application/vnd.google-apps.folder":
                raise ValueError(f"ID '{folder_id}' is not a folder.")
            return meta
        except Exception as e:
            log.error("drive.folder_access_failed", folder_id=folder_id, error=str(e))
            raise
