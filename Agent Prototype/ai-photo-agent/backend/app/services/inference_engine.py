"""
AI Inference Layer — CLIP, YOLOv8, Qwen, and Safety Model inference.
Models are singletons loaded once at worker startup.
GPU batching is used for throughput efficiency.
Qwen provides verification & activity refinement when CLIP/YOLO disagree.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import io
import hashlib
import gc

import clip
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import structlog

from app.core.config import settings
from app.services.groq_vision_service import GroqVisionResult, GroqVisionService

log = structlog.get_logger()

# ─── CLIP Prompt Set ──────────────────────────────────────────────────────
POSITIVE_PROMPTS = [
    "professional corporate photo",
    "high quality business portrait",
    "professional team photo",
]
NEGATIVE_PROMPTS = [
    "low quality blurry image",
    "casual selfie photo",
    "personal photo",
    "inappropriate content",
]

JOURNAL_CLIP_PROMPTS = {
    "setting_school": [
        "a photo taken inside a school classroom",
        "students sitting at desks in a classroom",
        "a teacher standing in front of a whiteboard",
        "children learning in a bright classroom environment",
        "a school hallway with students walking",
        "children playing outside on a school playground",
        "students doing an outdoor school group activity",
        "a supervised school sports day outdoors",
    ],
    "setting_nonschool_private": [
        "a person alone in a private room",
        "an adult and child alone indoors",
        "a private office or home setting",
    ],
    "activity_learning": [
        "students engaged in a group activity",
        "children reading books together",
        "a collaborative classroom project",
        "students raising hands during a lesson",
    ],
    "quality_positive": [
        "a well-lit, high quality photograph",
        "a sharp, clear, professional photo",
    ],
    "safety_flag": [
        "a knife or sharp object visible",
        "scissors being held by a child",
        "a dangerous object near children",
    ],
}

# ─── YOLO Class Labels (COCO) ─────────────────────────────────────────────
YOLO_PERSON_CLASS = 0
YOLO_PHONE_CLASS = 67   # "cell phone"
YOLO_LAPTOP_CLASS = 63
SHARP_OBJECT_CLASSES = {"scissors", "knife", "fork"}
ROLE_CONFIDENCE_MARGIN = 3.0
ROLE_MIN_VOTE = 22.0

# ─── Canonical Activities ──────────────────────────────────────────────────
CANONICAL_ACTIVITIES = {
    "classroom_learning",
    "outdoor_play",
    "lunch_dining",
    "sports_activity",
    "assembly_event",
    "library_reading",
    "hallway_movement",
    "group_project",
    "teacher_meeting",
    "arts_crafts",
    "computer_lab",
    "pe_gym",
    "other",
}

def normalise_activity(raw_label: str) -> str:
    """Map Qwen's free-form label to a canonical category for filtering."""
    raw = raw_label.lower()
    if any(w in raw for w in ["cook", "food", "kitchen", "bak"]):
        return "arts_crafts"
    if any(w in raw for w in ["class", "lesson", "desk", "board", "studying"]):
        return "classroom_learning"
    if any(w in raw for w in ["play", "outside", "recess", "yard", "playground"]):
        return "outdoor_play"
    if any(w in raw for w in ["lunch", "eat", "dining", "cafeteria"]):
        return "lunch_dining"
    if any(w in raw for w in ["sport", "football", "soccer", "basketball", "field"]):
        return "sports_activity"
    if any(w in raw for w in ["gym", "pe", "physical"]):
        return "pe_gym"
    if any(w in raw for w in ["assembly", "theater", "performance", "stage"]):
        return "assembly_event"
    if any(w in raw for w in ["library", "book", "reading"]):
        return "library_reading"
    if any(w in raw for w in ["computer", "lab", "coding", "typing"]):
        return "computer_lab"
    if any(w in raw for w in ["art", "craft", "painting", "drawing"]):
        return "arts_crafts"
    if any(w in raw for w in ["hallway", "corridor", "walking"]):
        return "hallway_movement"
    if any(w in raw for w in ["meeting", "staff", "training"]):
        return "teacher_meeting"
    return "other"


# ─── Result Dataclasses ───────────────────────────────────────────────────

@dataclass
class CLIPResult:
    positive_score: float        # Similarity to positive prompts (0–1)
    negative_score: float        # Similarity to negative prompts (0–1)
    semantic_score: float        # Final combined score
    detected_activity: str = "unknown"
    match_active: bool = False   # True if user_text was provided
    prompt_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class YOLOResult:
    people_count: int = 0
    student_count: int = 0
    teacher_count: int = 0
    uncertain_count: int = 0
    person_estimates: List[Dict[str, Any]] = field(default_factory=list)
    phone_detected: bool = False
    laptop_detected: bool = False
    sharp_object_detected: bool = False
    flagged_objects: List[Dict[str, Any]] = field(default_factory=list)
    detected_activity: List[Dict[str, Any]] = field(default_factory=list)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    compliance_score: float = 1.0


@dataclass
class SafetyResult:
    nsfw_detected: bool
    nsfw_score: float
    labels: List[str] = field(default_factory=list)


@dataclass
class QwenResult:
    """Qwen vision-language model results for verification & activity refinement."""
    activity_classification: str = "other"  # Canonical category
    activity_label: str = ""               # 3-6 words descriptive label
    activity_description: str = ""         # One specific sentence (max 20 words)
    activity_confidence: str = "low"       # high | medium | low
    people_count_estimate: Optional[int] = None
    safety_assessment: str = "safe"        # safe | review
    rejection_explanation: str = ""
    notes: str = ""
    used_for_verification: bool = False
    verification_reason: str = ""
    processing_time_ms: float = 0.0


@dataclass
class InferenceResult:
    clip: CLIPResult
    yolo: YOLOResult
    safety: SafetyResult
    qwen: Optional[QwenResult] = None  # Filled when CLIP/YOLO disagree
    groq: Optional[GroqVisionResult] = None  # Primary multimodal analyser when GROQ_ENABLED


# ─── Model Registry (Singleton) ───────────────────────────────────────────

class ModelRegistry:
    """Lazy-loaded, singleton model registry. Thread/process-safe for Celery workers."""

    _clip_model = None
    _clip_preprocess = None
    _yolo_model = None
    _safety_model = None
    _qwen_model = None
    _qwen_processor = None
    _device: str = None
    _qwen_cache: Dict[str, QwenResult] = {}  # Simple hash-based cache for Qwen results
    _groq_service: Optional[GroqVisionService] = None

    @classmethod
    def get_device(cls) -> str:
        if cls._device is None:
            cls._device = settings.get_device()
            log.info("inference.device_selected", device=cls._device)
        return cls._device

    @classmethod
    def get_clip(cls):
        if cls._clip_model is None:
            device = cls.get_device()
            clip_model_name = settings.get_clip_model_name()
            log.info("inference.loading_clip", model=clip_model_name)
            cls._clip_model, cls._clip_preprocess = clip.load(
                clip_model_name, device=device
            )
            cls._clip_model.eval()
        return cls._clip_model, cls._clip_preprocess

    @classmethod
    def get_yolo(cls):
        if cls._yolo_model is None:
            yolo_model_path = settings.get_yolo_model_path()
            log.info("inference.loading_yolo", path=yolo_model_path)
            cls._yolo_model = YOLO(yolo_model_path)
        return cls._yolo_model

    @classmethod
    def get_safety(cls):
        if cls._safety_model is None:
            log.info("inference.loading_safety_model")
            try:
                from nudenet import NudeDetector
                cls._safety_model = NudeDetector()
            except ImportError:
                log.warning("inference.nudenet_unavailable", msg="Install nudenet for NSFW detection")
                cls._safety_model = None
        return cls._safety_model

    @classmethod
    def get_qwen(cls):
        """Lazy-load Qwen2-VL model if enabled. Returns (model, processor) tuple or (None, None)."""
        if not settings.QWEN_ENABLED:
            return None, None
        
        if cls._qwen_model is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                device = settings.get_qwen_device()
                model_name = settings.get_qwen_model_name()
                log.info("inference.loading_qwen", model=model_name, device=device)
                
                cls._qwen_processor = AutoProcessor.from_pretrained(
                    model_name,
                    min_pixels=256 * 28 * 28,
                    max_pixels=512 * 28 * 28,  # Keep memory low: max ~512 visual tokens
                )

                if device == "cpu":
                    cls._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                    )
                else:
                    # 4-bit quantization: 2B model uses ~2 GB VRAM, well within RTX 3050 4 GB
                    try:
                        from transformers import BitsAndBytesConfig
                        bnb_cfg = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        cls._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_name,
                            quantization_config=bnb_cfg,
                            device_map="auto",
                            # Reserve 1.2GB for CLIP/YOLO/Context on 4GB cards
                            max_memory={0: "2800MiB", "cpu": "16GiB"}
                        )
                        log.info("inference.qwen_4bit_quantized", max_mem="2800MiB")
                    except ImportError:
                        # bitsandbytes not available — fall back to fp16
                        log.warning("inference.qwen_bnb_unavailable", msg="Install bitsandbytes for 4-bit quant")
                        cls._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            max_memory={0: "2800MiB", "cpu": "16GiB"}
                        )
                
                cls._qwen_model.eval()
                log.info("inference.qwen_ready", model=model_name)
            except ImportError as e:
                log.warning("inference.qwen_unavailable", error=str(e))
                cls._qwen_model = None
                cls._qwen_processor = None
            except Exception as e:
                log.error("inference.qwen_load_failed", error=str(e))
                cls._qwen_model = None
                cls._qwen_processor = None
        
        return cls._qwen_model, cls._qwen_processor

    @classmethod
    def clear_qwen_cache(cls):
        """Clear the Qwen result cache."""
        cls._qwen_cache.clear()
        log.info("inference.qwen_cache_cleared")

    @classmethod
    def get_groq(cls) -> Optional[GroqVisionService]:
        """Lazy-init the Groq Vision client. Returns None when disabled / no key."""
        if cls._groq_service is None:
            svc = GroqVisionService()
            cls._groq_service = svc
            if svc.enabled:
                log.info("inference.groq_ready", model=svc.model)
        return cls._groq_service if cls._groq_service and cls._groq_service.enabled else None

    @classmethod
    def preload_all(cls) -> None:
        """
        Load the core models eagerly so the first request does not pay the cold-start cost.
        """
        cls.get_device()
        cls.get_clip()
        cls.get_yolo()
        cls.get_safety()
        if settings.QWEN_ENABLED:
            cls.get_qwen()
        if settings.GROQ_ENABLED:
            cls.get_groq()
        log.info("inference.models_ready")


def ensemble_clip_score(image_features, prompts: List[str], model, preprocess=None) -> float:
    """Average cosine similarity across multiple prompts for robustness."""
    device = ModelRegistry.get_device()
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)
    return float(similarities.mean().item())


# ─── Qwen Verification Engine ────────────────────────────────────────────

class QwenVerifier:
    """
    Uses Qwen for verification when CLIP and YOLO disagree significantly.
    Also refines activity classification when confidence is low.
    Implements efficient caching and early-exit logic.
    """

    DISAGREEMENT_THRESHOLD = 0.25  # Disagreement if predictions differ by >25%
    LOW_CONFIDENCE_THRESHOLD = 0.50  # Use Qwen if confidence < 50%
    
    @staticmethod
    def _get_cache_key(image_bytes: bytes) -> str:
        """Generate a hash-based cache key for an image."""
        return hashlib.md5(image_bytes).hexdigest()
    
    @classmethod
    def _check_disagreement(cls, clip_result: CLIPResult, yolo_result: YOLOResult) -> Tuple[bool, str]:
        """
        Detect significant disagreement between CLIP and YOLO.
        Returns (has_disagreement, reason).
        """
        # Check 1: Activity type mismatch
        clip_activity = clip_result.detected_activity.lower()
        yolo_people = yolo_result.people_count
        
        # If CLIP says indoor classroom but YOLO detects 0 people
        if "classroom" in clip_activity and yolo_people == 0:
            return True, "clip_indoor_vs_yolo_no_people"
        
        # If CLIP says playground but YOLO detects minimal movement/people
        if "playground" in clip_activity and yolo_people < 2:
            return True, "clip_outdoor_vs_yolo_few_people"
        
        # Check 2: Safety flag mismatch
        if yolo_result.sharp_object_detected and clip_result.semantic_score > 0.75:
            return True, "yolo_risk_vs_clip_confidence"
        
        # Check 3: Confidence level mismatch
        if clip_result.semantic_score < cls.LOW_CONFIDENCE_THRESHOLD:
            if yolo_people > 0:  # YOLO has clear detections
                return True, "clip_low_confidence_vs_yolo_detections"
        
        # Check 4: Activity confidence vs people count
        if "unknown" in clip_activity and yolo_people >= 3:
            return True, "clip_unknown_vs_yolo_group"
        
        return False, ""

    @classmethod
    def should_use_qwen(cls, clip_result: CLIPResult, yolo_result: YOLOResult) -> Tuple[bool, str]:
        """
        Determine if Qwen should be invoked.
        In the new architecture, Qwen is the primary source for activity labeling.
        """
        if not settings.QWEN_ENABLED:
            return False, "qwen_disabled"
        
        # We always use Qwen for activity and description in the Right Architecture
        return True, "activity_labeling"

    @classmethod
    def verify_and_refine(
        cls,
        image_bytes: bytes,
        img_pil: Image.Image,
        clip_result: CLIPResult,
        yolo_result: YOLOResult,
    ) -> Optional[QwenResult]:
        """
        Use Qwen to verify/refine results when CLIP and YOLO disagree.
        Implements caching for efficiency.
        Returns QwenResult or None.
        """
        should_use, reason = cls.should_use_qwen(clip_result, yolo_result)
        if not should_use:
            return None
        
        # Check cache
        if settings.QWEN_CACHE_ENABLED:
            cache_key = cls._get_cache_key(image_bytes)
            if cache_key in ModelRegistry._qwen_cache:
                cached = ModelRegistry._qwen_cache[cache_key]
                log.info("inference.qwen_cache_hit", reason=reason)
                return cached
        
        # Invoke Qwen
        try:
            import time
            start_time = time.time()
            
            qwen_model, qwen_processor = ModelRegistry.get_qwen()
            if qwen_model is None:
                return None
            
            # Build the prompt
            prompt = cls._build_prompt(clip_result, yolo_result)
            device = settings.get_qwen_device()

            # ── Qwen2-VL message format ──────────────────────────────────
            from qwen_vl_utils import process_vision_info
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_input = qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = qwen_processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if device == "cuda":
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = qwen_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                )
                # Trim prompt tokens from output
                trimmed = outputs[:, inputs["input_ids"].shape[-1]:]
                response = qwen_processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            # Clean up after heavy Qwen run
            del inputs, outputs, trimmed
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Parse response
            qwen_result = cls._parse_qwen_response(response, time.time() - start_time, reason)
            
            # Cache result
            if settings.QWEN_CACHE_ENABLED:
                cache_key = cls._get_cache_key(image_bytes)
                ModelRegistry._qwen_cache[cache_key] = qwen_result
            
            log.info(
                "inference.qwen_verification_complete",
                reason=reason,
                activity=qwen_result.activity_classification,
                confidence=qwen_result.activity_confidence,
            )
            
            return qwen_result
            
        except Exception as e:
            log.warning("inference.qwen_verification_failed", error=str(e), reason=reason)
            return None

    @staticmethod
    def _build_prompt(clip_result: CLIPResult, yolo_result: YOLOResult) -> str:
        """Build a targeted prompt for Qwen based on YOLO facts."""
        flagged = [d["class"] for d in yolo_result.flagged_objects]
        facts = {
            "total_people": yolo_result.people_count,
            "students": yolo_result.student_count,
            "teachers": yolo_result.teacher_count,
            "flagged_items": flagged,
            "has_laptop": yolo_result.laptop_detected,
            "has_phone": yolo_result.phone_detected,
        }
        
        return f"""You are analysing a school photograph.

Automated detection found:
- People: {facts['total_people']} total
- Students: {facts['students']}, Teachers: {facts['teachers']}
- Objects visible: {', '.join(facts['flagged_items']) or 'none flagged'}
- Equipment: {'laptop ' if facts['has_laptop'] else ''}{'phone ' if facts['has_phone'] else ''}

Respond ONLY in this exact format:
ACTIVITY: <single lowercase snake_case label you determine yourself>
LABEL: <3-6 words describing the activity>
DESCRIPTION: <one specific sentence, max 20 words>
CONFIDENCE: <high|medium|low>
SAFE: <yes|no>"""

    @staticmethod
    def _parse_qwen_response(response: str, processing_time: float, reason: str) -> QwenResult:
        """Parse Qwen's response into structured QwenResult."""
        lines = response.strip().split("\n")
        parsed = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                parsed[key.strip().upper()] = val.strip()
        
        raw_activity = parsed.get("ACTIVITY", "other").lower()
        activity_label = parsed.get("LABEL", "Unknown Activity")
        description = parsed.get("DESCRIPTION", "")
        confidence = parsed.get("CONFIDENCE", "low").lower()
        safe_str = parsed.get("SAFE", "no").lower()
        
        # Normalise the raw activity label to a canonical category
        category = normalise_activity(raw_activity)
        if category == "other":
            # Try normalising the descriptive label too
            category = normalise_activity(activity_label)

        return QwenResult(
            activity_classification=category,
            activity_label=activity_label,
            activity_description=description,
            activity_confidence=confidence,
            people_count_estimate=None, # Qwen no longer asked for count in this format
            safety_assessment="safe" if "yes" in safe_str else "review",
            rejection_explanation="",
            notes=response[:500],
            used_for_verification=True,
            verification_reason=reason,
            processing_time_ms=processing_time * 1000,
        )


# ─── Inference Engine ─────────────────────────────────────────────────────

class InferenceEngine:
    """
    Runs CLIP, YOLOv8, and Safety inference on images.
    Designed for batched GPU processing.
    """

    def run(
        self,
        image_bytes: bytes,
        filename: str = "",
        user_text: Optional[str] = None,
        yolo_augment: Optional[bool] = None,
        mime_type: Optional[str] = None,
    ) -> InferenceResult:
        """Run all models on a single image. Groq Vision is the primary multimodal
        analyser when enabled; Qwen-VL is used as a local fallback."""
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            # Clear cache before starting inference to maximize headroom
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            clip_result = self._run_clip(img_pil, user_text=user_text)
        except Exception as exc:
            log.exception(
                "inference.clip_failed",
                filename=filename or "unknown",
                error=str(exc),
            )
            clip_result = self._empty_clip_result()

        yolo_result = self._run_yolo(img_pil, augment=yolo_augment)
        yolo_result = self._maybe_refine_yolo_for_school_context(
            img_pil,
            image_bytes,
            yolo_result,
            clip_result,
            augment=yolo_augment,
            filename=filename,
        )

        # 🌐 PRIMARY: Groq Vision API — full activity + surroundings + journal-title match
        groq_result = self._run_groq(image_bytes, user_text, mime_type, filename)

        # Fallback: local Qwen-VL only when Groq is unavailable / returned an error
        qwen_result = None
        groq_usable = groq_result is not None and not groq_result.error
        if not groq_usable and settings.QWEN_ENABLED:
            qwen_result = QwenVerifier.verify_and_refine(
                image_bytes, img_pil, clip_result, yolo_result
            )
            if qwen_result:
                clip_result.detected_activity = qwen_result.activity_label
                log.info(
                    "inference.activity_labeled_by_qwen",
                    label=qwen_result.activity_label,
                    category=qwen_result.activity_classification,
                )

        if groq_usable:
            self._apply_groq_labels(clip_result, yolo_result, groq_result)

        return InferenceResult(
            clip=clip_result,
            yolo=yolo_result,
            safety=SafetyResult(nsfw_detected=False, nsfw_score=0.0),
            qwen=qwen_result,
            groq=groq_result,
        )

    def run_batch(
        self,
        image_batch: List[bytes],
        user_text: Optional[str] = None,
        yolo_augment: Optional[bool] = None,
        mime_type: Optional[str] = None,
    ) -> List[InferenceResult]:
        """
        GPU-batched inference for throughput efficiency.
        Falls back to sequential if batch fails.
        Integrates Groq Vision (primary) and Qwen verification (fallback).
        """
        results = []
        try:
            pil_images = [
                Image.open(io.BytesIO(b)).convert("RGB") for b in image_batch
            ]
            try:
                clip_results = self._run_clip_batch(pil_images, user_text=user_text)
            except Exception as exc:
                log.exception(
                    "inference.clip_batch_failed",
                    batch_size=len(pil_images),
                    error=str(exc),
                )
                clip_results = [self._empty_clip_result() for _ in pil_images]

            for i, img_bytes in enumerate(image_batch):
                clip_res = clip_results[i]
                yolo_res = self._run_yolo(pil_images[i], augment=yolo_augment)

                yolo_res = self._maybe_refine_yolo_for_school_context(
                    pil_images[i],
                    img_bytes,
                    yolo_res,
                    clip_res,
                    augment=yolo_augment,
                    image_index=i,
                )

                # 🌐 Groq Vision (primary)
                groq_res = self._run_groq(
                    img_bytes, user_text, mime_type, filename=f"batch[{i}]"
                )
                groq_usable = groq_res is not None and not groq_res.error

                # Qwen fallback only when Groq is unavailable
                qwen_res = None
                if not groq_usable and settings.QWEN_ENABLED:
                    qwen_res = QwenVerifier.verify_and_refine(
                        img_bytes, pil_images[i], clip_res, yolo_res
                    )
                    if qwen_res:
                        clip_res.detected_activity = qwen_res.activity_label

                if groq_usable:
                    self._apply_groq_labels(clip_res, yolo_res, groq_res)

                safety_res = self._run_safety(img_bytes)

                # 🏫 Unified School Shield
                school_keywords = [
                    "Classroom", "Playground", "Lunch", "Assembly", "Math", "Desk",
                    "Storytime", "Teacher", "Meeting", "Training", "Library",
                    "Reading", "Gym", "Hallway", "Life", "Cafeteria",
                    "Sensory", "Toddler", "Preschool", "Daycare", "Childcare",
                ]
                activity_text = clip_res.detected_activity or ""
                is_school = any(k.lower() in activity_text.lower() for k in school_keywords)
                if groq_usable and groq_res.safe == "yes":
                    is_school = True

                if is_school and safety_res.nsfw_score < 0.99:
                    safety_res.nsfw_detected = False

                results.append(InferenceResult(
                    clip=clip_res,
                    yolo=yolo_res,
                    safety=safety_res,
                    qwen=qwen_res,
                    groq=groq_res,
                ))
        except Exception as e:
            log.warning("inference.batch_fallback", error=str(e))
            results = [self.run(b, user_text=user_text, mime_type=mime_type) for b in image_batch]
        return results

    # ── Groq Vision ──────────────────────────────────────────────────

    def _run_groq(
        self,
        image_bytes: bytes,
        user_text: Optional[str],
        mime_type: Optional[str],
        filename: Optional[str] = None,
    ) -> Optional[GroqVisionResult]:
        """Run the Groq Vision API on a single image. Returns None if disabled."""
        groq = ModelRegistry.get_groq()
        if groq is None:
            return None
        try:
            return groq.analyse(
                image_bytes=image_bytes,
                journal_title=user_text,
                mime_type=mime_type,
                filename=filename,
            )
        except Exception as exc:
            log.warning("inference.groq_failed", filename=filename, error=str(exc))
            return None

    def _apply_groq_labels(
        self,
        clip_result: CLIPResult,
        yolo_result: YOLOResult,
        groq_result: GroqVisionResult,
    ) -> None:
        """Push Groq's label / people-count into the CLIP & YOLO result objects so
        downstream code (scoring, persistence, frontend) sees the unified value."""
        if groq_result.activity_label and groq_result.activity_label != "Unknown Activity":
            clip_result.detected_activity = groq_result.activity_label

        # If YOLO missed everyone but Groq sees people, trust Groq for the count.
        if (
            groq_result.people_count is not None
            and yolo_result.people_count == 0
            and groq_result.people_count > 0
        ):
            yolo_result.people_count = groq_result.people_count
            yolo_result.uncertain_count = groq_result.people_count

    # ── CLIP ─────────────────────────────────────────────────────────

    def _run_clip(self, img_pil: Image.Image, user_text: Optional[str] = None) -> CLIPResult:
        return self._run_clip_batch([img_pil], user_text=user_text)[0]

    @staticmethod
    def _empty_clip_result() -> CLIPResult:
        return CLIPResult(
            positive_score=0.0,
            negative_score=0.0,
            semantic_score=0.0,
            detected_activity="unknown",
            match_active=False,
            prompt_scores={},
        )

    def _run_clip_batch(self, images: List[Image.Image], user_text: Optional[str] = None) -> List[CLIPResult]:
        model, preprocess = ModelRegistry.get_clip()
        device = ModelRegistry.get_device()

        all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
        text_tokens = clip.tokenize(all_prompts).to(device)

        image_tensors = torch.stack([preprocess(img) for img in images]).to(device)

        results = []

        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            
            journal_scores = {}
            for key, prompts in JOURNAL_CLIP_PROMPTS.items():
                prompt_tokens = clip.tokenize(prompts).to(device)
                prompt_features = model.encode_text(prompt_tokens)
                prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
                scores = (image_features @ prompt_features.T).mean(dim=-1).cpu().numpy()
                journal_scores[key] = scores

        for i, sim in enumerate(similarity):
            # CLIP now only provides quality signals
            pos_score = float(np.sum(sim[:len(POSITIVE_PROMPTS)]))
            neg_score = float(np.sum(sim[len(POSITIVE_PROMPTS):]))

            # User Match Score fallback (semantic similarity)
            prompt_scores = {
                key: round(float(scores[i]), 4)
                for key, scores in journal_scores.items()
            }
            
            # Use activity_learning score as a proxy for semantic match if no user_text
            raw_al = prompt_scores.get("activity_learning", 0.0)
            match_score = max(0.0, min(1.0, (raw_al + 1.0) / 2.0))

            results.append(CLIPResult(
                positive_score=round(pos_score, 4),
                negative_score=round(neg_score, 4),
                semantic_score=round(match_score, 4),
                detected_activity="Unknown", # Labeling now handled by Qwen
                match_active=(user_text is not None),
                prompt_scores=prompt_scores,
            ))
        return results

    def _maybe_refine_yolo_for_school_context(
        self,
        img_pil: Image.Image,
        image_bytes: bytes,
        yolo_result: YOLOResult,
        clip_result: CLIPResult,
        augment: Optional[bool] = None,
        filename: str = "",
        image_index: Optional[int] = None,
    ) -> YOLOResult:
        """
        Retry YOLO at lower confidence when the image looks like a school scene
        but the first pass only found a single person. This helps recover
        occluded students in classroom and outdoor school activity shots.
        """
        school_score = float((clip_result.prompt_scores or {}).get("setting_school", 0.0))
        if yolo_result.people_count != 1 or school_score < 0.28:
            return yolo_result

        try:
            model = ModelRegistry.get_yolo()
            use_augment = settings.YOLO_AUGMENT if augment is None else augment
            fallback_results = model(
                img_pil,
                verbose=False,
                conf=0.20,
                iou=0.45,
                imgsz=1280,
                augment=use_augment,
            )
            fallback = self._process_yolo_results(
                img_pil=img_pil,
                results=fallback_results,
                clip_result=clip_result,
                filename=filename,
                image_index=image_index,
                augment=augment
            )
            
            # 🤖 Qwen verification boost for school context refinement
            if settings.QWEN_ENABLED:
                qwen_res = QwenVerifier.verify_and_refine(
                    image_bytes, img_pil, clip_result, fallback
                )
                if qwen_res and qwen_res.used_for_verification:
                    # If Qwen confirms more people than YOLO, trust Qwen
                    if qwen_res.people_count_estimate is not None and qwen_res.people_count_estimate > fallback.people_count:
                        fallback.people_count = qwen_res.people_count_estimate
                        fallback.uncertain_count = max(0, fallback.people_count - fallback.student_count - fallback.teacher_count)

            if fallback.people_count > yolo_result.people_count:
                log.info(
                    "inference.yolo_fallback_applied",
                    filename=filename or None,
                    image_index=image_index,
                    school_score=round(school_score, 4),
                    original_people_count=yolo_result.people_count,
                    fallback_people_count=fallback.people_count,
                )
                return fallback
        except Exception as exc:
            log.warning(
                "inference.yolo_fallback_failed",
                filename=filename or None,
                image_index=image_index,
                error=str(exc),
            )
        return yolo_result

    def _run_yolo(self, img_pil: Image.Image, augment: Optional[bool] = None) -> YOLOResult:
        """Run standard YOLO detection pass."""
        model = ModelRegistry.get_yolo()
        use_augment = settings.YOLO_AUGMENT if augment is None else augment
        results = model(
            img_pil,
            verbose=False,
            conf=0.35,
            iou=0.45,
            imgsz=1280,
            augment=use_augment,
        )
        return self._process_yolo_results(img_pil, results, augment=augment)

    def _process_yolo_results(
        self,
        img_pil: Image.Image,
        results: Any,
        clip_result: Optional[CLIPResult] = None,
        filename: str = "",
        image_index: Optional[int] = None,
        augment: Optional[bool] = None
    ) -> YOLOResult:
        """Unified processing logic for YOLO raw results, handling role detection and compliance."""
        model = ModelRegistry.get_yolo()
        detections = []
        flagged_objects = []
        people_count = 0
        phone_detected = False
        laptop_detected = False
        person_estimates: List[Dict[str, Any]] = []
        person_boxes = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                coords = box.xyxy[0].cpu().numpy().tolist()

                # Anti-Ghosting: Ignore very small detections
                img_area = img_pil.width * img_pil.height
                box_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                if label.lower() == "person" and (box_area / img_area) < 0.02:
                    continue

                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "class_id": cls_id,
                    "box": coords,
                })

                if cls_id == YOLO_PERSON_CLASS:
                    people_count += 1
                    person_boxes.append(coords)
                elif cls_id == YOLO_PHONE_CLASS:
                    phone_detected = True
                elif cls_id == YOLO_LAPTOP_CLASS:
                    laptop_detected = True

                if label.lower() in SHARP_OBJECT_CLASSES:
                    flagged_objects.append({
                        "class": label,
                        "confidence": round(conf, 3),
                    })

        student_count = 0
        teacher_count = 0
        if people_count > 0:
            clip_model, preprocess = ModelRegistry.get_clip()
            device = ModelRegistry.get_device()
            role_prompts = [
                "a young primary school child, student, small elementary pupil",
                "a small kid with a young round juvenile face and child features",
                "a young child sitting at a school desk",
                "a fully grown adult teacher, professional faculty, mature face",
                "a mature adult woman or man with adult facial features",
                "a professional grown person in adult teacher attire",
            ]
            role_tokens = clip.tokenize(role_prompts).to(device)

            if people_count <= 8:
                for coords in person_boxes:
                    try:
                        w, h = coords[2] - coords[0], coords[3] - coords[1]
                        pad_w, pad_h = w * 0.15, h * 0.15
                        person_crop = img_pil.crop((
                            max(0, coords[0] - pad_w),
                            max(0, coords[1] - pad_h),
                            min(img_pil.width, coords[2] + pad_w),
                            min(img_pil.height, coords[3] + pad_h)
                        ))
                        
                        estimated_age = self._estimate_age(person_crop)
                        if estimated_age is not None:
                            role = "student" if estimated_age < 18 else "teacher"
                            person_estimates.append({
                                "box": [round(v, 2) for v in coords],
                                "estimated_age": round(float(estimated_age), 1),
                                "role": role,
                                "confidence": "high" if (estimated_age < 14 or estimated_age > 24) else "medium",
                                "source": "deepface",
                            })
                            if role == "student": student_count += 1
                            else: teacher_count += 1
                            continue

                        img_tensor = preprocess(person_crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            img_f = clip_model.encode_image(img_tensor)
                            role_f = clip_model.encode_text(role_tokens)
                            img_f /= img_f.norm(dim=-1, keepdim=True)
                            role_f /= role_f.norm(dim=-1, keepdim=True)
                            sims = (100.0 * img_f @ role_f.T).cpu().numpy()[0]

                        student_vote = np.mean(sims[:3])
                        teacher_vote = np.mean(sims[3:])
                        
                        # Fix Bug 2: Check clip_result.detected_activity (string) instead of list
                        if clip_result and "Teacher meeting" in clip_result.detected_activity:
                            teacher_vote += 5.0

                        if max(student_vote, teacher_vote) < ROLE_MIN_VOTE or abs(student_vote - teacher_vote) < ROLE_CONFIDENCE_MARGIN:
                            role, source = "uncertain", "clip_fallback"
                        elif student_vote > teacher_vote:
                            role, source, student_count = "student", "clip_fallback", student_count + 1
                        else:
                            role, source, teacher_count = "teacher", "clip_fallback", teacher_count + 1
                        
                        person_estimates.append({
                            "box": [round(v, 2) for v in coords],
                            "role": role,
                            "confidence": "medium",
                            "source": source,
                        })
                    except Exception as exc:
                        log.warning("inference.person_proc_failed", error=str(exc))
                        person_estimates.append({"box": coords, "role": "uncertain", "source": "error"})
            else:
                # Batch fallback for crowds
                img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_f = clip_model.encode_image(img_tensor)
                    role_f = clip_model.encode_text(role_tokens)
                    img_f /= img_f.norm(dim=-1, keepdim=True)
                    role_f /= role_f.norm(dim=-1, keepdim=True)
                    sims = (100.0 * img_f @ role_f.T).cpu().numpy()[0]
                
                student_vote = np.mean(sims[:3])
                teacher_vote = np.mean(sims[3:])
                if student_vote > teacher_vote:
                    student_count = people_count
                    teacher_count = 0
                else:
                    teacher_count = 1
                    student_count = max(0, people_count - 1)

        return YOLOResult(
            people_count=people_count,
            student_count=student_count,
            teacher_count=teacher_count,
            uncertain_count=max(0, people_count - student_count - teacher_count),
            person_estimates=person_estimates,
            phone_detected=phone_detected,
            laptop_detected=laptop_detected,
            sharp_object_detected=len(flagged_objects) > 0,
            flagged_objects=flagged_objects,
            detections=detections,
            compliance_score=0.0 if len(flagged_objects) > 0 else 1.0,
        )

    # ── Safety ───────────────────────────────────────────────────────

    @staticmethod
    def _estimate_age(person_crop: Image.Image) -> Optional[float]:
        """
        Estimate age using DeepFace when available.
        Returns None when the dependency is unavailable or inference fails.
        """
        try:
            from deepface import DeepFace
        except ImportError:
            return None

        try:
            analysis = DeepFace.analyze(
                np.array(person_crop),
                actions=["age"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(analysis, list) and analysis:
                analysis = analysis[0]
            if not isinstance(analysis, dict):
                return None
            age = analysis.get("age")
            return float(age) if age is not None else None
        except Exception as exc:
            log.warning("inference.age_estimation_failed", error=str(exc))
            return None

    def _run_safety(self, image_bytes: bytes) -> SafetyResult:
        detector = ModelRegistry.get_safety()
        if detector is None:
            return SafetyResult(nsfw_detected=False, nsfw_score=0.0)

        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            detections = detector.detect(tmp_path)
            os.unlink(tmp_path)

            nsfw_labels = {d["class"] for d in detections}
            nsfw_score = max((d["score"] for d in detections), default=0.0)
            # Increased threshold to 0.7 to avoid false positives in messy school group shots
            nsfw_detected = nsfw_score > 0.7

            return SafetyResult(
                nsfw_detected=nsfw_detected,
                nsfw_score=round(nsfw_score, 4),
                labels=list(nsfw_labels),
            )
        except Exception as e:
            log.warning("inference.safety_failed", error=str(e))
            return SafetyResult(nsfw_detected=False, nsfw_score=0.0)
