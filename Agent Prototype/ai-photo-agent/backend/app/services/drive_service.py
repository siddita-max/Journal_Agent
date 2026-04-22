"""
Google Drive Service — fetches image files from a Drive folder.
Uses service account credentials for secure, server-side access.
"""
import re
import io
import time
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
    "image/bmp",
    "image/gif",
    "image/tiff",
}

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

    def _get_service(self):
        if self._service is None:
            creds = service_account.Credentials.from_service_account_file(
                settings.GOOGLE_SERVICE_ACCOUNT_FILE,
                scopes=settings.GOOGLE_DRIVE_SCOPES,
            )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
            log.info("drive.service_initialized")
        return self._service

    def list_images(
        self,
        folder_id: str,
        page_size: int = 100,
        max_retries: int = 3,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields image file metadata dicts from a Drive folder.
        Handles pagination automatically.
        """
        service = self._get_service()
        mime_query = " or ".join(
            [f"mimeType='{m}'" for m in IMAGE_MIME_TYPES]
        )
        query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"

        page_token: Optional[str] = None
        total_fetched = 0

        while True:
            for attempt in range(max_retries):
                try:
                    response = service.files().list(
                        q=query,
                        pageSize=page_size,
                        fields="nextPageToken, files(id, name, mimeType, size)",
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
                yield {
                    "id": f["id"],
                    "name": f["name"],
                    "mime_type": f.get("mimeType", ""),
                    "size": int(f.get("size", 0)),
                }
                total_fetched += 1

            page_token = response.get("nextPageToken")
            if not page_token:
                log.info("drive.list_complete", total=total_fetched, folder_id=folder_id)
                break

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
