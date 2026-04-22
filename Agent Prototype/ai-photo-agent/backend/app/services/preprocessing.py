"""
Preprocessing Layer — fast CPU-based image quality checks.
Runs BEFORE AI inference to reject obviously bad images cheaply.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import structlog

from app.core.config import settings

log = structlog.get_logger()


@dataclass
class PreprocessResult:
    """Outcome of the preprocessing pipeline for a single image."""
    ok: bool                              # False → hard reject, skip AI stage
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    blur_variance: float = 0.0
    brightness: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)


class PreprocessingPipeline:
    """
    Fast image quality checks that run on CPU.

    Checks (in order):
    1. Decodeable image
    2. Minimum resolution
    3. Aspect ratio
    4. Blur (Laplacian variance)
    5. Brightness (mean pixel value of grayscale)
    """

    def __init__(self):
        cfg = settings
        self.min_w = cfg.MIN_RESOLUTION_WIDTH
        self.min_h = cfg.MIN_RESOLUTION_HEIGHT
        self.min_blur = cfg.MIN_BLUR_VARIANCE
        self.min_bright = cfg.MIN_BRIGHTNESS
        self.max_bright = cfg.MAX_BRIGHTNESS
        self.min_ar = cfg.MIN_ASPECT_RATIO
        self.max_ar = cfg.MAX_ASPECT_RATIO

    # ── Public API ──────────────────────────────────────────────────

    def evaluate(self, image_bytes: bytes, filename: str = "") -> Tuple[Optional[np.ndarray], PreprocessResult]:
        """
        Run all preprocessing checks.

        Returns:
            (cv2_image_or_None, PreprocessResult)
            If result.ok is False, the image should be rejected immediately.
        """
        result = PreprocessResult(ok=True)
        reasons = result.rejection_reasons

        # ── 1. Decode ──
        img_pil, img_cv = self._decode(image_bytes)
        if img_cv is None:
            result.ok = False
            reasons.append("Cannot decode image file")
            return None, result

        h, w = img_cv.shape[:2]
        result.width = w
        result.height = h
        result.aspect_ratio = round(w / h, 3) if h > 0 else 0.0

        # ── 2. Resolution ──
        if w < self.min_w or h < self.min_h:
            result.ok = False
            reasons.append(f"Low resolution ({w}×{h}, minimum {self.min_w}×{self.min_h})")

        # ── 3. Aspect Ratio ──
        ar = result.aspect_ratio
        if ar < self.min_ar or ar > self.max_ar:
            result.ok = False
            reasons.append(f"Invalid aspect ratio ({ar:.2f}, allowed {self.min_ar}–{self.max_ar})")

        # ── 4. Blur ──
        blur_var = self._laplacian_variance(img_cv)
        result.blur_variance = round(blur_var, 2)
        if blur_var < self.min_blur:
            result.ok = False
            reasons.append(f"Blurry image (variance={blur_var:.1f}, minimum={self.min_blur})")

        # ── 5. Brightness ──
        brightness = self._mean_brightness(img_cv)
        result.brightness = round(brightness, 2)
        if brightness < self.min_bright:
            result.ok = False
            reasons.append(f"Image too dark (brightness={brightness:.1f}, minimum={self.min_bright})")
        elif brightness > self.max_bright:
            result.ok = False
            reasons.append(f"Image overexposed (brightness={brightness:.1f}, maximum={self.max_bright})")

        log.debug(
            "preprocess.result",
            filename=filename,
            ok=result.ok,
            size=f"{w}x{h}",
            blur=result.blur_variance,
            brightness=result.brightness,
        )
        return img_cv, result

    # ── Private Helpers ─────────────────────────────────────────────

    @staticmethod
    def _decode(image_bytes: bytes) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Decode bytes to both PIL and OpenCV formats."""
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_cv = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img_cv is None:
                # Fall back: convert PIL → numpy
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_pil, img_cv
        except (UnidentifiedImageError, Exception):
            return None, None

    @staticmethod
    def _laplacian_variance(img_cv: np.ndarray) -> float:
        """
        Measure blur using Laplacian variance.
        Higher = sharper; lower = blurrier.
        """
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _mean_brightness(img_cv: np.ndarray) -> float:
        """Mean pixel value of the grayscale image (0–255)."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
