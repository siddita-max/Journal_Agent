"""
AI Inference Layer — CLIP, YOLOv8, and Safety Model inference.
Models are singletons loaded once at worker startup.
GPU batching is used for throughput efficiency.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import io

import clip
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import structlog

from app.core.config import settings

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
class InferenceResult:
    clip: CLIPResult
    yolo: YOLOResult
    safety: SafetyResult


# ─── Model Registry (Singleton) ───────────────────────────────────────────

class ModelRegistry:
    """Lazy-loaded, singleton model registry. Thread/process-safe for Celery workers."""

    _clip_model = None
    _clip_preprocess = None
    _yolo_model = None
    _safety_model = None
    _device: str = None

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
    def preload_all(cls) -> None:
        """
        Load the core models eagerly so the first request does not pay the cold-start cost.
        """
        cls.get_device()
        cls.get_clip()
        cls.get_yolo()
        cls.get_safety()
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
    ) -> InferenceResult:
        """Run all models on a single image."""
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
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
            yolo_result,
            clip_result,
            augment=yolo_augment,
            filename=filename,
        )
        safety_result = self._run_safety(image_bytes)

        # 🏫 School-Context Safeguard:
        # If we are highly confident this is a school activity, we suppress
        # NSFW false positives.
        school_keywords = [
            "Classroom", "Playground", "Lunch", "Assembly", "Math", "Desk", 
            "Storytime", "Teacher", "Meeting", "Training", "Library", 
            "Reading", "Gym", "Hallway", "Life", "Cafeteria"
        ]
        is_school_activity = any(k.lower() in clip_result.detected_activity.lower() for k in school_keywords)
        
        if is_school_activity and safety_result.nsfw_score < 0.99:
            # Downgrade detection unless it's absolutely certain (>0.95)
            safety_result.nsfw_detected = False
            log.info("safety.suppressed_by_context", 
                     activity=clip_result.detected_activity, 
                     original_score=safety_result.nsfw_score)

        return InferenceResult(
            clip=clip_result,
            yolo=yolo_result,
            safety=safety_result,
        )

    def run_batch(
        self,
        image_batch: List[bytes],
        user_text: Optional[str] = None,
        yolo_augment: Optional[bool] = None,
    ) -> List[InferenceResult]:
        """
        GPU-batched inference for throughput efficiency.
        Falls back to sequential if batch fails.
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
                    yolo_res,
                    clip_res,
                    augment=yolo_augment,
                    image_index=i,
                )
                safety_res = self._run_safety(img_bytes)

                # 🏫 Unified School Shield
                school_keywords = [
                    "Classroom", "Playground", "Lunch", "Assembly", "Math", "Desk", 
                    "Storytime", "Teacher", "Meeting", "Training", "Library", 
                    "Reading", "Gym", "Hallway", "Life", "Cafeteria"
                ]
                is_school = any(k.lower() in clip_res.detected_activity.lower() for k in school_keywords)
                
                # Trust school context up to 99% certainty
                if is_school and safety_res.nsfw_score < 0.99:
                    safety_res.nsfw_detected = False

                results.append(InferenceResult(
                    clip=clip_res,
                    yolo=yolo_res,
                    safety=safety_res,
                ))
        except Exception as e:
            log.warning("inference.batch_fallback", error=str(e))
            results = [self.run(b) for b in image_batch]
        return results

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
        n_pos = len(POSITIVE_PROMPTS)

        # School-specific activity prompts
        activity_prompts = [
            "students studying in a classroom",
            "children playing outdoors on a playground",
            "students having lunch in the cafeteria",
            "teacher explaining at a whiteboard",
            "school assembly or theater event",
            "students working on a creative math project",
            "kids sitting on a rug for storytime",
            "students focused at desks with books",
            "physical education in a gym or sports field",
            "students reading in a library",
            "adults having a teacher meeting or training",
            "general school corridor or hallway activity",
        ]
        # Add user text if provided
        final_act_prompts = list(activity_prompts)
        user_text_idx = -1
        if user_text:
            final_act_prompts.append(user_text)
            user_text_idx = len(final_act_prompts) - 1

        activity_tokens = clip.tokenize(final_act_prompts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)
            act_features = model.encode_text(activity_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            act_features = act_features / act_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            # Raw similarity for activities
            act_similarity = (100.0 * image_features @ act_features.T).cpu().numpy()
            journal_scores = {}
            for key, prompts in JOURNAL_CLIP_PROMPTS.items():
                prompt_tokens = clip.tokenize(prompts).to(device)
                prompt_features = model.encode_text(prompt_tokens)
                prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
                scores = (image_features @ prompt_features.T).mean(dim=-1).cpu().numpy()
                journal_scores[key] = scores

        for i, sim in enumerate(similarity):
            # Detect activity (excluding user_text)
            img_act_sims = act_similarity[i][:len(activity_prompts)]
            best_act_idx = np.argmax(img_act_sims)
            conf = img_act_sims[best_act_idx]
            
            if conf < 22.0:
                detected_activity = "General School Life"
            else:
                detected_activity = activity_prompts[best_act_idx].split(" ")[1:]
                detected_activity = " ".join(detected_activity).capitalize()

            # User Match Score
            match_score = 0.0
            if user_text_idx != -1:
                raw_score = act_similarity[i][user_text_idx]
                match_score = min(1.0, max(0.0, (raw_score - 20.0) / 15.0))

            prompt_scores = {
                key: round(float(scores[i]), 4)
                for key, scores in journal_scores.items()
            }
            if user_text_idx == -1:
                match_score = max(0.0, min(1.0, prompt_scores.get("activity_learning", 0.0) + 0.5))

            results.append(CLIPResult(
                positive_score=prompt_scores.get("quality_positive", 0.0),
                negative_score=prompt_scores.get("setting_nonschool_private", 0.0),
                semantic_score=round(match_score, 4),
                detected_activity=detected_activity,
                match_active=(user_text is not None),
                prompt_scores=prompt_scores,
            ))
        return results

    def _maybe_refine_yolo_for_school_context(
        self,
        img_pil: Image.Image,
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
            fallback = self._build_yolo_result_from_raw(
                img_pil=img_pil,
                raw_results=fallback_results,
            )
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

    def _build_yolo_result_from_raw(
        self,
        img_pil: Image.Image,
        raw_results,
    ) -> YOLOResult:
        model = ModelRegistry.get_yolo()
        detections = []
        flagged_objects = []
        people_count = 0
        phone_detected = False
        laptop_detected = False
        person_estimates: List[Dict[str, Any]] = []
        detected_activity = []
        person_boxes = []

        for result in raw_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                coords = box.xyxy[0].cpu().numpy().tolist()

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
                    detected_activity.append({"class": "person", "confidence": round(conf, 3)})
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
                        w = coords[2] - coords[0]
                        h = coords[3] - coords[1]
                        pad_w, pad_h = w * 0.15, h * 0.15

                        left = max(0, coords[0] - pad_w)
                        top = max(0, coords[1] - pad_h)
                        right = min(img_pil.width, coords[2] + pad_w)
                        bottom = min(img_pil.height, coords[3] + pad_h)

                        person_crop = img_pil.crop((left, top, right, bottom))
                        estimated_age = self._estimate_age(person_crop)
                        if estimated_age is not None:
                            role = "student" if estimated_age < 18 else "teacher"
                            confidence = "low" if 15 <= estimated_age <= 21 else "high"
                            person_estimates.append({
                                "box": [round(v, 2) for v in coords],
                                "estimated_age": round(float(estimated_age), 1),
                                "role": role,
                                "confidence": confidence,
                                "source": "deepface",
                            })
                            if role == "student":
                                student_count += 1
                            else:
                                teacher_count += 1
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
                        vote_gap = abs(float(student_vote) - float(teacher_vote))
                        best_vote = max(float(student_vote), float(teacher_vote))
                        if best_vote < ROLE_MIN_VOTE or vote_gap < ROLE_CONFIDENCE_MARGIN:
                            person_estimates.append({
                                "box": [round(v, 2) for v in coords],
                                "estimated_age": None,
                                "role": "uncertain",
                                "confidence": "low",
                                "source": "clip_fallback",
                            })
                        elif student_vote > teacher_vote:
                            student_count += 1
                            person_estimates.append({
                                "box": [round(v, 2) for v in coords],
                                "estimated_age": None,
                                "role": "student",
                                "confidence": "medium",
                                "source": "clip_fallback",
                            })
                        else:
                            teacher_count += 1
                            person_estimates.append({
                                "box": [round(v, 2) for v in coords],
                                "estimated_age": None,
                                "role": "teacher",
                                "confidence": "medium",
                                "source": "clip_fallback",
                            })
                    except Exception as exc:
                        log.warning("inference.crop_failed", error=str(exc))
                        person_estimates.append({
                            "box": [round(v, 2) for v in coords],
                            "estimated_age": None,
                            "role": "uncertain",
                            "confidence": "low",
                            "source": "error",
                        })
            else:
                img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_features = clip_model.encode_image(img_tensor)
                    role_features = clip_model.encode_text(role_tokens)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    role_features /= role_features.norm(dim=-1, keepdim=True)
                    probs = (img_features @ role_features.T).softmax(dim=-1).cpu().numpy()[0]

                if abs(float(probs[0]) - float(probs[1])) < 0.15:
                    pass
                elif probs[0] > probs[1]:
                    student_count = people_count
                    teacher_count = 0
                else:
                    teacher_count = 1
                    student_count = people_count - 1

        compliance = 1.0
        if flagged_objects:
            compliance = 0.0
        compliance = max(0.0, compliance)

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
            detected_activity=detected_activity,
            detections=detections,
            compliance_score=round(compliance, 4),
        )


    # ── YOLO ─────────────────────────────────────────────────────────

    def _run_yolo(self, img_pil: Image.Image, augment: Optional[bool] = None) -> YOLOResult:
        model = ModelRegistry.get_yolo()
        detected_activity = []
        use_augment = settings.YOLO_AUGMENT if augment is None else augment
        results = model(
            img_pil,
            verbose=False,
            conf=0.35,
            iou=0.45,
            imgsz=1280,
            augment=use_augment,
        )

        detections = []
        flagged_objects = []
        people_count = 0
        phone_detected = False
        laptop_detected = False
        person_estimates: List[Dict[str, Any]] = []

        # Prepare for role classification (Students vs Teachers)
        person_boxes = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                coords = box.xyxy[0].cpu().numpy().tolist()

                # Anti-Ghosting: Ignore very small detections (likely posters or background clutter)
                # Area check: if person box is less than 2% of the total image area, ignore it
                img_area = img_pil.width * img_pil.height
                box_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                
                if label.lower() == "person" and (box_area / img_area) < 0.02:
                    continue

                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "class_id": cls_id,
                    "box": coords
                })

                if cls_id == YOLO_PERSON_CLASS:
                    people_count += 1
                    person_boxes.append(coords)
                    detected_activity.append({"class": "person", "confidence": round(conf, 3)})
                elif cls_id == YOLO_PHONE_CLASS:
                    phone_detected = True
                elif cls_id == YOLO_LAPTOP_CLASS:
                    laptop_detected = True

                if label.lower() in SHARP_OBJECT_CLASSES:
                    flagged_objects.append({
                        "class": label,
                        "confidence": round(conf, 3),
                    })

        # Role Classification: Student vs Teacher
        student_count = 0
        teacher_count = 0

        if people_count > 0:
            clip_model, preprocess = ModelRegistry.get_clip()
            device = ModelRegistry.get_device()
            # High-contrast Ensemble Prompts for definitive Role Detection
            role_prompts = [
                "a young primary school child, student, small elementary pupil",
                "a small kid with a young round juvenile face and child features",
                "a young child sitting at a school desk",
                "a fully grown adult teacher, professional faculty, mature face",
                "a mature adult woman or man with adult facial features",
                "a professional grown person in adult teacher attire"
            ]
            role_tokens = clip.tokenize(role_prompts).to(device)

            # For each detected person, crop and classify
            if people_count <= 8:
                for d in detections:
                    if d["class"].lower() != "person":
                        continue
                    
                    try:
                        box = d["box"]
                        # Pad the crop slightly to see height/clothing
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        pad_w, pad_h = w * 0.15, h * 0.15
                        
                        left = max(0, box[0] - pad_w)
                        top = max(0, box[1] - pad_h)
                        right = min(img_pil.width, box[2] + pad_w)
                        bottom = min(img_pil.height, box[3] + pad_h)
                        
                        person_crop = img_pil.crop((left, top, right, bottom))
                        estimated_age = self._estimate_age(person_crop)
                        if estimated_age is not None:
                            role = "student" if estimated_age < 18 else "teacher"
                            confidence = "low" if 15 <= estimated_age <= 21 else "high"
                            d["estimated_age"] = round(float(estimated_age), 1)
                            d["age_confidence"] = confidence
                            d["role_source"] = "deepface"
                            d["suggested_role"] = role
                            person_estimates.append({
                                "box": [round(v, 2) for v in box],
                                "estimated_age": round(float(estimated_age), 1),
                                "role": role,
                                "confidence": confidence,
                                "source": "deepface",
                            })
                            if role == "student":
                                student_count += 1
                            else:
                                teacher_count += 1
                            continue

                        img_tensor = preprocess(person_crop).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            img_f = clip_model.encode_image(img_tensor)
                            role_f = clip_model.encode_text(role_tokens)
                            img_f /= img_f.norm(dim=-1, keepdim=True)
                            role_f /= role_f.norm(dim=-1, keepdim=True)
                            
                            # Calculate raw similarities
                            sims = (100.0 * img_f @ role_f.T).cpu().numpy()[0]
                        
                        # Ensemble Vote: First 3 are Students, Last 3 are Teachers
                        student_vote = np.mean(sims[:3])
                        teacher_vote = np.mean(sims[3:])
                        
                        # Context-Aware Boost: If it's a teacher meeting, adults are highly likely
                        if "Teacher meeting" in detected_activity:
                            teacher_vote += 5.0 # Significant boost for adult roles in professional settings
                        
                        vote_gap = abs(float(student_vote) - float(teacher_vote))
                        best_vote = max(float(student_vote), float(teacher_vote))
                        if best_vote < ROLE_MIN_VOTE or vote_gap < ROLE_CONFIDENCE_MARGIN:
                            d["suggested_role"] = "unknown"
                            person_estimates.append({
                                "box": [round(v, 2) for v in box],
                                "estimated_age": None,
                                "role": "uncertain",
                                "confidence": "low",
                                "source": "clip_fallback",
                            })
                        elif student_vote > teacher_vote:
                            d["suggested_role"] = "student"
                            student_count += 1
                            person_estimates.append({
                                "box": [round(v, 2) for v in box],
                                "estimated_age": None,
                                "role": "student",
                                "confidence": "medium",
                                "source": "clip_fallback",
                            })
                        else:
                            d["suggested_role"] = "teacher"
                            teacher_count += 1
                            person_estimates.append({
                                "box": [round(v, 2) for v in box],
                                "estimated_age": None,
                                "role": "teacher",
                                "confidence": "medium",
                                "source": "clip_fallback",
                            })
                    except Exception as e:
                        log.warning("inference.crop_failed", error=str(e))
                        d["suggested_role"] = "unknown"
                        # Keep the role conservative when the crop-based heuristic is unsure.
                        # The policy engine will treat unassigned people as uncategorized.
                        person_estimates.append({
                            "box": [round(v, 2) for v in d.get("box", [])],
                            "estimated_age": None,
                            "role": "uncertain",
                            "confidence": "low",
                            "source": "error",
                        })
                        pass
            else:
                # Fallback for crowded photos
                img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_features = clip_model.encode_image(img_tensor)
                    role_features = clip_model.encode_text(role_tokens)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    role_features /= role_features.norm(dim=-1, keepdim=True)
                    probs = (img_features @ role_features.T).softmax(dim=-1).cpu().numpy()[0]
                
                if abs(float(probs[0]) - float(probs[1])) < 0.15:
                    pass
                elif probs[0] > probs[1]:
                    student_count = people_count
                    teacher_count = 0
                else:
                    teacher_count = 1
                    student_count = people_count - 1

        # Compliance score
        compliance = 1.0
        if flagged_objects:
            compliance = 0.0
        compliance = max(0.0, compliance)

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
            detected_activity=detected_activity,
            detections=detections,
            compliance_score=round(compliance, 4),
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
