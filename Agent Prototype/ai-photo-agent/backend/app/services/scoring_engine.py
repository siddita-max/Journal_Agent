"""
Scoring Engine — the heart of the decision system.

Implements:
  A. Hard rules → immediate rejection
  B. Soft weighted scoring (0–1)
  C. Threshold-based decision classification
  D. Full explainability in every result
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from app.core.config import settings
from app.services.preprocessing import PreprocessResult
from app.services.inference_engine import InferenceResult
from app.services.policy_engine import PolicyResult

log = structlog.get_logger()


# ─── Decision Enum ────────────────────────────────────────────────────────

class Decision:
    APPROVED = "approved"
    REVIEW = "review"
    REJECTED = "rejected"


# ─── Score Result ─────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Complete scoring output for a single image."""
    final_score: float
    decision: str
    reasons: List[str] = field(default_factory=list)

    # Component scores
    clip_score: float = 0.0
    quality_score: float = 0.0
    resolution_score: float = 0.0
    object_compliance_score: float = 0.0
    aesthetic_score: float = 0.0

    # Hard-rule flags
    hard_rejected: bool = False
    nsfw_detected: bool = False

    # Full breakdown for transparency
    breakdown: Dict = field(default_factory=dict)


# ─── Scoring Engine ───────────────────────────────────────────────────────

class ScoringEngine:
    """
    Hybrid scoring system: rule-based hard filters + weighted AI scores.

    Weights (configurable via settings):
        CLIP semantic       → 0.35
        Image quality       → 0.25
        Resolution          → 0.15
        Object compliance   → 0.15
        Aesthetic           → 0.10
    """

    def __init__(self):
        self.approved_threshold = settings.SCORE_APPROVED_THRESHOLD
        self.review_threshold = settings.SCORE_REVIEW_THRESHOLD

        self.w_clip = settings.WEIGHT_CLIP
        self.w_quality = settings.WEIGHT_QUALITY
        self.w_resolution = settings.WEIGHT_RESOLUTION
        self.w_object = settings.WEIGHT_OBJECT
        self.w_aesthetic = settings.WEIGHT_AESTHETIC

    # ── Public API ──────────────────────────────────────────────────

    def score(
        self,
        preprocess: PreprocessResult,
        inference: Optional[InferenceResult] = None,
        policy_result: Optional[PolicyResult] = None,
    ) -> ScoreResult:
        """
        Compute the complete score for an image.

        Args:
            preprocess:    CPU preprocessing result (blur, brightness, resolution, aspect ratio)
            inference:     AI model results (CLIP, YOLO, Safety). None = preprocessing-only path.
            policy_result: Output from PolicyEngine.evaluate(). None = no policy applied.
        """
        # ── 1. Policy Violations (Top Priority) ─────────────────────
        policy_reasons = []
        nsfw_detected = False
        
        if inference:
            nsfw_detected = inference.safety.nsfw_detected
            if nsfw_detected:
                policy_reasons.append("NSFW content detected")

        if policy_result and policy_result.hard_rejected:
            policy_reasons.extend(policy_result.reasons)

        if policy_reasons:
            return ScoreResult(
                final_score=0.0,
                decision=Decision.REJECTED,
                reasons=policy_reasons,
                hard_rejected=True,
                nsfw_detected=nsfw_detected
            )

        # ── 1.5. User Match Check ──────────────────────────────────
        # If the user provided a search text and it matches poorly (< 30%)
        # We check match_active to ensure we don't reject photos when no search was requested
        if inference and inference.clip.match_active and inference.clip.semantic_score < 0.28:
            return ScoreResult(
                final_score=0.0,
                decision=Decision.REJECTED,
                reasons=["Does not match your specific search description"],
                hard_rejected=True
            )

        # ── 2. Quality Thresholds (Secondary) ───────────────────────
        quality_reasons = list(preprocess.rejection_reasons)
        if not preprocess.ok:
            return ScoreResult(
                final_score=0.0,
                decision=Decision.REJECTED,
                reasons=quality_reasons,
                hard_rejected=True
            )

        # ── 3. Accept (Default if no violations) ───────────────────
        clip_score = self._clip_score(inference)
        quality_score = self._quality_score(preprocess)
        resolution_score = self._resolution_score(preprocess)
        object_score = policy_result.compliance_score if policy_result else self._object_score(inference)
        aesthetic_score = self._aesthetic_score(preprocess, inference)
        composition_score = (resolution_score + aesthetic_score) / 2.0

        final_score = (
            clip_score * self.w_clip
            + quality_score * self.w_quality
            + object_score * self.w_object
            + composition_score * self.w_aesthetic
        )
        final_score = max(0.0, min(1.0, round(final_score, 4)))

        hard_rejected = False
        reasons = self._generate_soft_reasons(
            clip_score,
            quality_score,
            resolution_score,
            object_score,
            inference,
        )
        if policy_result and policy_result.flagged:
            reasons.extend(policy_result.reasons)
            decision = Decision.REVIEW
        elif final_score >= self.approved_threshold:
            decision = Decision.APPROVED
        elif final_score >= self.review_threshold:
            decision = Decision.REVIEW
        else:
            decision = Decision.REJECTED

        breakdown = {
            "weights": {
                "clip_semantic_match": self.w_clip,
                "image_quality": self.w_quality,
                "safety_compliance": self.w_object,
                "composition_score": self.w_aesthetic,
            },
            "scores": {
                "clip": round(clip_score, 4),
                "quality": round(quality_score, 4),
                "resolution": round(resolution_score, 4),
                "safety": round(object_score, 4),
                "aesthetic": round(aesthetic_score, 4),
                "composition": round(composition_score, 4),
            },
            "thresholds": {
                "approved": self.approved_threshold,
                "review": self.review_threshold,
            },
            "hard_rejected": hard_rejected,
        }

        log.debug(
            "scoring.result",
            decision=decision,
            final_score=final_score,
            reasons=reasons,
        )

        return ScoreResult(
            final_score=final_score,
            decision=decision,
            reasons=reasons,
            clip_score=round(clip_score, 4),
            quality_score=round(quality_score, 4),
            resolution_score=round(resolution_score, 4),
            object_compliance_score=round(object_score, 4),
            aesthetic_score=round(aesthetic_score, 4),
            hard_rejected=hard_rejected,
            nsfw_detected=nsfw_detected,
            breakdown=breakdown,
        )

    # ── Component Score Formulas ──────────────────────────────────

    @staticmethod
    def _clip_score(inference: Optional[InferenceResult]) -> float:
        if inference is None:
            return 0.5
        return inference.clip.semantic_score

    @staticmethod
    def _quality_score(prep: PreprocessResult) -> float:
        """
        Composite quality score from blur and brightness.
        Both components range 0–1 and are averaged.
        """
        # Blur: logistic function centred at min_blur
        blur_norm = min(1.0, prep.blur_variance / (settings.MIN_BLUR_VARIANCE * 4))

        # Brightness: penalty for deviation from ideal centre (128)
        ideal = 128.0
        bright_dev = abs(prep.brightness - ideal) / ideal
        bright_norm = max(0.0, 1.0 - bright_dev)

        return (blur_norm + bright_norm) / 2.0

    @staticmethod
    def _resolution_score(prep: PreprocessResult) -> float:
        """Normalised resolution score: 1.0 at 4× minimum, 0 at or below minimum."""
        megapixels = (prep.width * prep.height) / 1_000_000
        min_mp = (settings.MIN_RESOLUTION_WIDTH * settings.MIN_RESOLUTION_HEIGHT) / 1_000_000
        target_mp = min_mp * 4  # 4× minimum = perfect score
        return min(1.0, megapixels / target_mp)

    @staticmethod
    def _object_score(inference: Optional[InferenceResult]) -> float:
        if inference is None:
            return 0.5
        return inference.yolo.compliance_score

    @staticmethod
    def _aesthetic_score(prep: PreprocessResult, inference: Optional[InferenceResult]) -> float:
        """
        Heuristic aesthetic score combining aspect ratio quality
        and CLIP professional signal.
        """
        # Aspect ratio: standard photo ratios (4:3, 16:9, 1:1) get bonus
        ar = prep.aspect_ratio
        GOOD_RATIOS = [1.0, 1.333, 1.5, 1.778]
        ar_score = max(0.0, 1.0 - min(abs(ar - r) for r in GOOD_RATIOS))

        if inference:
            return (ar_score + inference.clip.positive_score) / 2.0
        return ar_score

    @staticmethod
    def _generate_soft_reasons(
        clip_score: float,
        quality_score: float,
        resolution_score: float,
        object_score: float,
        inference: Optional[InferenceResult],
    ) -> List[str]:
        reasons = []
        if clip_score < 0.4:
            reasons.append("Low professional appearance (CLIP score)")
        if quality_score < 0.4:
            reasons.append("Poor image quality (blur/brightness)")
        if resolution_score < 0.3:
            reasons.append("Resolution below recommended threshold")
        if object_score < 0.5 and inference:
            if inference.yolo.phone_detected:
                reasons.append("Phone visible in frame")
            if inference.yolo.people_count == 0:
                reasons.append("No people detected")
        return reasons
