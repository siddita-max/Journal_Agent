"""
Company Policy Engine — DB-backed, JSON-driven policy enforcement.

Policies are stored as versioned JSON documents in PostgreSQL.
The active policy set is loaded per job and applied to every image.

Supports:
  Hard constraints  → immediate rejection regardless of score
  Soft penalties    → deducted from the object_compliance_score component

Policy document schema (JSON):
{
  "max_people": 10,
  "prohibited_objects": ["cell phone"],
  "min_professionalism_score": 0.55,
  "required_context_prompts": ["corporate office", "professional setting"],
  "dress_code": {
    "enabled": true,
    "required_prompts": ["formal attire", "business dress"],
    "min_score": 0.40
  },
  "custom_hard_rules": [
    {
      "name": "no_outdoor",
      "description": "Must be indoor/office setting",
      "clip_negative_prompts": ["outdoor photo", "street photo"],
      "threshold": 0.60
    }
  ],
  "custom_soft_rules": [
    {
      "name": "brand_environment",
      "description": "Should show branded or neutral background",
      "clip_positive_prompts": ["clean background", "office background"],
      "weight": 0.10,
      "min_score": 0.30
    }
  ]
}
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

log = structlog.get_logger()

# ─── Default policy (used if no DB policy is active) ─────────────────────

DEFAULT_POLICY: Dict[str, Any] = {
    "max_people": 100,
    "prohibited_objects": [],
    "min_professionalism_score": 0.0,
    "required_context_prompts": [],
    "dress_code": {
        "enabled": False,
        "required_prompts": ["formal attire", "business dress"],
        "min_score": 0.35,
    },
    "custom_hard_rules": [],
    "custom_soft_rules": [],
}


SCHOOL_CHILD_SAFETY_POLICY: Dict[str, Any] = {
    "name": "school_journal_policy_v1",
    "version": "1.1",
    "description": "Journal image selection policy for school settings",
    "max_people": 100,
    "prohibited_objects": [],
    "quality_gates": {
        "min_resolution": [100, 100],
        "preferred_resolution": [800, 600],
        "min_sharpness_score": 30,
        "min_brightness": 40,
        "max_brightness": 220,
        "blur_threshold": 30,
        "allowed_formats": ["jpeg", "jpg", "png", "webp"],
        "reject_if_any_fail": True,
    },
    "safety_rules": [
        {
            "id": "SR-03",
            "name": "sharp_objects_present",
            "description": "Sharp objects (scissors, knives, blades) must not appear",
            "yolo_classes": ["scissors", "knife", "blade", "cutter"],
            "action": "REJECT",
            "priority": "CRITICAL",
        },
    ],
    "journal_matching": {
        "enabled": True,
        "clip_similarity_threshold": 0.22,
        "reject_below_threshold": True,
        "strategy": "per_image_independent",
        "note": "Each image is scored against journal description independently — no dependency on previous photos",
    },
    "scoring_weights": {
        "clip_semantic_match": 0.35,
        "image_quality": 0.25,
        "safety_compliance": 0.30,
        "composition_score": 0.10,
    },
    "decision_thresholds": {
        "approve": 0.70,
        "review": 0.50,
        "reject": 0.0,
    },
}


def get_school_child_safety_policy() -> Dict[str, Any]:
    """Return a copy of the fixed school journal policy used by the pipeline."""
    return copy.deepcopy(SCHOOL_CHILD_SAFETY_POLICY)


# ─── Result types ─────────────────────────────────────────────────────────

@dataclass
class PolicyViolation:
    rule_name: str
    severity: str           # "hard" | "soft"
    description: str
    measured_value: Any = None
    threshold: Any = None

    def to_reason(self) -> str:
        parts = [f"[Policy] {self.description}"]
        if self.measured_value is not None and self.threshold is not None:
            parts.append(f"(got {self.measured_value}, required {self.threshold})")
        return " ".join(parts)


@dataclass
class PolicyResult:
    hard_rejected: bool = False
    flagged: bool = False
    compliance_score: float = 1.0           # 0–1, factored into object_compliance_score
    violations: List[PolicyViolation] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        return [v.to_reason() for v in self.violations]

    @property
    def hard_violations(self) -> List[PolicyViolation]:
        return [v for v in self.violations if v.severity == "hard"]

    @property
    def soft_violations(self) -> List[PolicyViolation]:
        return [v for v in self.violations if v.severity == "soft"]


# ─── Policy Engine ────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Stateless engine that evaluates an image against a policy document.

    Usage:
        engine = PolicyEngine(policy_dict)
        result = engine.evaluate(yolo_result, clip_result, preprocess_result)
    """

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        policy = policy or {}
        if "quality_gates" in policy and "safety_rules" in policy:
            self.policy = copy.deepcopy(policy)
        else:
            self.policy = self._merge_with_defaults(policy)

    # ── Public API ──────────────────────────────────────────────────

    def evaluate(
        self,
        yolo_result=None,       # YOLOResult
        clip_result=None,       # CLIPResult (with prompt_scores)
        preprocess_result=None, # PreprocessResult
    ) -> PolicyResult:
        """
        Run all configured policy rules against the image data.
        Returns a PolicyResult with violation list and aggregate compliance score.
        """
        result = PolicyResult()
        penalty = 0.0  # Accumulated soft penalty

        if "safety_rules" in self.policy:
            penalty += self._evaluate_school_journal_rules(
                result,
                yolo_result=yolo_result,
                clip_result=clip_result,
            )

        # ── 0. Quality Gates ──────────────────────────────────────────
        gates = self.policy.get("quality_gates", {})
        if gates and preprocess_result:
            min_res = gates.get("min_resolution", [0, 0])
            if preprocess_result.width < min_res[0] or preprocess_result.height < min_res[1]:
                result.violations.append(PolicyViolation(
                    rule_name="min_resolution",
                    severity="hard",
                    description=f"Resolution below policy minimum ({preprocess_result.width}x{preprocess_result.height})",
                    measured_value=f"{preprocess_result.width}x{preprocess_result.height}",
                    threshold=f"{min_res[0]}x{min_res[1]}",
                ))
                result.hard_rejected = True

            min_bright = gates.get("min_brightness", 0)
            if preprocess_result.brightness < min_bright:
                result.violations.append(PolicyViolation(
                    rule_name="min_brightness",
                    severity="hard",
                    description="Image too dark",
                    measured_value=round(preprocess_result.brightness, 1),
                    threshold=min_bright,
                ))
                result.hard_rejected = True

        # ── 1. People count ──────────────────────────────────────────
        max_people = self.policy.get("max_people", 10)
        if yolo_result and yolo_result.people_count > max_people:
            result.violations.append(PolicyViolation(
                rule_name="max_people",
                severity="hard",
                description=f"Too many people in frame ({yolo_result.people_count} detected, max {max_people})",
                measured_value=yolo_result.people_count,
                threshold=max_people,
            ))
            result.hard_rejected = True

        # ── 2. Prohibited objects ────────────────────────────────────
        prohibited = [o.lower() for o in self.policy.get("prohibited_objects", [])]
        if yolo_result and prohibited:
            detected_classes = {d["class"].lower() for d in (yolo_result.detections or [])}
            for obj in prohibited:
                if obj in detected_classes:
                    result.violations.append(PolicyViolation(
                        rule_name=f"prohibited_object_{obj.replace(' ', '_')}",
                        severity="hard",
                        description=f"Prohibited object detected: '{obj}'",
                        measured_value=obj,
                        threshold="not allowed",
                    ))
                    result.hard_rejected = True

        # ── 3. Minimum professionalism score (CLIP) ──────────────────
        min_prof = self.policy.get("min_professionalism_score", 0.0)
        if min_prof > 0.0 and clip_result:
            prof_score = clip_result.positive_score
            if prof_score < min_prof:
                severity = "hard" if prof_score < min_prof * 0.7 else "soft"
                if severity == "hard":
                    result.hard_rejected = True
                else:
                    penalty += 0.25
                result.violations.append(PolicyViolation(
                    rule_name="min_professionalism_score",
                    severity=severity,
                    description=f"Professionalism score below policy minimum",
                    measured_value=round(prof_score, 3),
                    threshold=min_prof,
                ))

        # ── 4. Required context (CLIP semantic) ──────────────────────
        required_contexts = self.policy.get("required_context_prompts", [])
        if required_contexts and clip_result and clip_result.prompt_scores:
            for ctx_prompt in required_contexts:
                ctx_score = clip_result.prompt_scores.get(ctx_prompt, 0.0)
                if ctx_score < 0.25:
                    penalty += 0.15
                    result.violations.append(PolicyViolation(
                        rule_name=f"required_context_{ctx_prompt[:30].replace(' ', '_')}",
                        severity="soft",
                        description=f"Required context not detected: '{ctx_prompt}'",
                        measured_value=round(ctx_score, 3),
                        threshold=0.25,
                    ))

        # ── 5. Dress code compliance ─────────────────────────────────
        dress_code = self.policy.get("dress_code", {})
        if dress_code.get("enabled", False) and clip_result and clip_result.prompt_scores:
            dc_prompts = dress_code.get("required_prompts", [])
            dc_min = dress_code.get("min_score", 0.35)
            dc_scores = [
                clip_result.prompt_scores.get(p, 0.0)
                for p in dc_prompts
                if p in clip_result.prompt_scores
            ]
            if dc_scores:
                avg_dc_score = sum(dc_scores) / len(dc_scores)
                if avg_dc_score < dc_min:
                    penalty += 0.20
                    result.violations.append(PolicyViolation(
                        rule_name="dress_code_compliance",
                        severity="soft",
                        description="Dress code policy not met — formal/business attire not detected",
                        measured_value=round(avg_dc_score, 3),
                        threshold=dc_min,
                    ))

        # ── 6. Custom hard rules ──────────────────────────────────────
        for rule in self.policy.get("custom_hard_rules", []):
            violation = self._evaluate_custom_hard_rule(rule, clip_result)
            if violation:
                result.violations.append(violation)
                result.hard_rejected = True

        # ── 7. Custom soft rules ──────────────────────────────────────
        for rule in self.policy.get("custom_soft_rules", []):
            violation, rule_penalty = self._evaluate_custom_soft_rule(rule, clip_result)
            if violation:
                result.violations.append(violation)
                penalty += rule_penalty

        # ── Final compliance score ────────────────────────────────────
        result.compliance_score = max(0.0, round(1.0 - penalty, 4))

        log.debug(
            "policy.evaluated",
            hard_rejected=result.hard_rejected,
            violations=len(result.violations),
            compliance_score=result.compliance_score,
        )
        return result

    # ── Custom rule evaluators ────────────────────────────────────

    def _evaluate_school_journal_rules(
        self,
        result: PolicyResult,
        yolo_result=None,
        clip_result=None,
    ) -> float:
        """Evaluate school_journal_policy_v1 safety rules."""
        if yolo_result is None:
            return 0.0

        penalty = 0.0
        rules = {rule.get("id"): rule for rule in self.policy.get("safety_rules", [])}
        person_estimates = getattr(yolo_result, "person_estimates", []) or []
        if person_estimates:
            adults = sum(1 for p in person_estimates if p.get("role") == "teacher")
            children = sum(1 for p in person_estimates if p.get("role") == "student")
            uncertain = sum(1 for p in person_estimates if p.get("role") == "uncertain")
        else:
            adults = int(getattr(yolo_result, "teacher_count", 0) or 0)
            children = int(getattr(yolo_result, "student_count", 0) or 0)
            uncertain = int(getattr(yolo_result, "uncertain_count", 0) or 0)
            if children == 0 and adults == 0:
                children = int(getattr(yolo_result, "people_count", 0) or 0)
        effective_children = children + uncertain

        school_setting = self._is_formal_school_setting(clip_result)

        # Isolation and Ratio checks removed (User request: 'rest all is fine')

        sharp_rule = rules.get("SR-03")
        if sharp_rule:
            sharp_classes = {c.lower() for c in sharp_rule.get("yolo_classes", [])}
            flagged_objects = getattr(yolo_result, "flagged_objects", []) or []
            detected_classes = {d.get("class", "").lower() for d in flagged_objects}
            detected_classes.update({
                d.get("class", "").lower()
                for d in (getattr(yolo_result, "detections", []) or [])
                if d.get("class", "").lower() in sharp_classes
            })
            if detected_classes:
                result.hard_rejected = True
                result.violations.append(PolicyViolation(
                    rule_name=sharp_rule["id"],
                    severity="hard",
                    description=sharp_rule["description"],
                    measured_value=sorted(detected_classes),
                    threshold="not allowed",
                ))

        return penalty

    @staticmethod
    def _is_formal_school_setting(clip_result) -> Optional[bool]:
        if clip_result is None:
            return None

        prompt_scores = getattr(clip_result, "prompt_scores", {}) or {}
        school_score = float(prompt_scores.get("setting_school", 0.0) or 0.0)
        private_score = float(prompt_scores.get("setting_nonschool_private", 0.0) or 0.0)
        if school_score < 0.20 and private_score < 0.20:
            return None
        if school_score >= 0.28 and school_score >= private_score:
            return True
        if private_score >= 0.28 and private_score > school_score:
            return False
        return None

    @staticmethod
    def _evaluate_custom_hard_rule(
        rule: Dict[str, Any],
        clip_result,
    ) -> Optional[PolicyViolation]:
        """
        Hard rule: if CLIP score for negative prompts exceeds threshold → reject.
        """
        if clip_result is None:
            return None
        name = rule.get("name", "custom_hard")
        desc = rule.get("description", "Custom policy violation")
        neg_prompts = rule.get("clip_negative_prompts", [])
        threshold = rule.get("threshold", 0.5)

        scores = [
            clip_result.prompt_scores.get(p, 0.0)
            for p in neg_prompts
            if p in (clip_result.prompt_scores or {})
        ]
        if not scores:
            log.warning(
                "policy.hard_rule_no_matching_prompts",
                rule_name=name,
                prompts=neg_prompts,
            )
            return None
        avg = sum(scores) / len(scores)
        if avg > threshold:
            return PolicyViolation(
                rule_name=name,
                severity="hard",
                description=desc,
                measured_value=round(avg, 3),
                threshold=threshold,
            )
        return None

    @staticmethod
    def _evaluate_custom_soft_rule(
        rule: Dict[str, Any],
        clip_result,
    ) -> Tuple[Optional[PolicyViolation], float]:
        """
        Soft rule: if CLIP score for positive prompts is below minimum → apply penalty.
        Returns (violation_or_None, penalty_amount).
        """
        if clip_result is None:
            return None, 0.0
        name = rule.get("name", "custom_soft")
        desc = rule.get("description", "Custom soft policy")
        pos_prompts = rule.get("clip_positive_prompts", [])
        weight = float(rule.get("weight", 0.10))
        min_score = float(rule.get("min_score", 0.30))

        scores = [
            clip_result.prompt_scores.get(p, 0.0)
            for p in pos_prompts
            if p in (clip_result.prompt_scores or {})
        ]
        if not scores:
            return None, 0.0
        avg = sum(scores) / len(scores)
        if avg < min_score:
            penalty = min(max(weight, 0.0), 1.0)
            return PolicyViolation(
                rule_name=name,
                severity="soft",
                description=desc,
                measured_value=round(avg, 3),
                threshold=min_score,
            ), penalty
        return None, 0.0

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _merge_with_defaults(policy: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge user policy on top of DEFAULT_POLICY."""
        merged = dict(DEFAULT_POLICY)
        for key, value in policy.items():
            if isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = list(dict.fromkeys([*merged[key], *value]))
            elif isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = PolicyEngine._merge_nested_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _merge_nested_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = list(dict.fromkeys([*merged[key], *value]))
            elif isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = PolicyEngine._merge_nested_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    # ── Validation ───────────────────────────────────────────────────

    @classmethod
    def validate_policy_document(cls, policy: Dict[str, Any]) -> List[str]:
        """
        Validate a policy document before saving to DB.
        Returns a list of validation errors (empty = valid).
        """
        errors = []
        if "max_people" in policy:
            if not isinstance(policy["max_people"], int) or policy["max_people"] < 1:
                errors.append("max_people must be a positive integer")

        if "prohibited_objects" in policy:
            if not isinstance(policy["prohibited_objects"], list):
                errors.append("prohibited_objects must be a list of strings")

        if "min_professionalism_score" in policy:
            v = policy["min_professionalism_score"]
            if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                errors.append("min_professionalism_score must be a float between 0.0 and 1.0")

        if "dress_code" in policy:
            dc = policy["dress_code"]
            if not isinstance(dc, dict):
                errors.append("dress_code must be an object")
            elif "min_score" in dc:
                if not isinstance(dc["min_score"], (int, float)) or not (0.0 <= dc["min_score"] <= 1.0):
                    errors.append("dress_code.min_score must be a float between 0.0 and 1.0")

        if "journal_matching" in policy:
            jm = policy["journal_matching"]
            if not isinstance(jm, dict):
                errors.append("journal_matching must be an object")
            else:
                t = jm.get("clip_similarity_threshold")
                if t is not None:
                    try:
                        threshold = float(t)
                        if not (0.0 <= threshold <= 1.0):
                            errors.append("journal_matching.clip_similarity_threshold must be 0.0-1.0")
                    except (TypeError, ValueError):
                        errors.append("journal_matching.clip_similarity_threshold must be numeric")

        if "decision_thresholds" in policy:
            dt = policy["decision_thresholds"]
            if not isinstance(dt, dict):
                errors.append("decision_thresholds must be an object")
            else:
                try:
                    approve = float(dt.get("approve", 0))
                    review = float(dt.get("review", 0))
                    reject = float(dt.get("reject", 0))
                    if not all(0.0 <= v <= 1.0 for v in [approve, review, reject]):
                        errors.append("decision_thresholds values must be 0.0-1.0")
                    if approve <= review:
                        errors.append("decision_thresholds.approve must be greater than review")
                    if review < reject:
                        errors.append("decision_thresholds.review must be greater than or equal to reject")
                except (TypeError, ValueError):
                    errors.append("decision_thresholds values must be numeric")

        if "scoring_weights" in policy:
            sw = policy["scoring_weights"]
            if not isinstance(sw, dict):
                errors.append("scoring_weights must be an object")
            else:
                try:
                    total = sum(float(v) for v in sw.values())
                    if not (0.99 <= total <= 1.01):
                        errors.append(f"scoring_weights must sum to 1.0, got {round(total, 3)}")
                except (TypeError, ValueError):
                    errors.append("scoring_weights values must be numeric")

        for rule in policy.get("custom_hard_rules", []):
            if "name" not in rule:
                errors.append("Each custom_hard_rule must have a 'name' field")
            if "clip_negative_prompts" not in rule or not isinstance(rule["clip_negative_prompts"], list):
                errors.append(f"custom_hard_rule '{rule.get('name', '?')}' must have 'clip_negative_prompts' list")
            t = rule.get("threshold", 0.5)
            try:
                threshold = float(t)
                if not (0.0 <= threshold <= 1.0):
                    errors.append(f"Rule '{rule.get('name')}' threshold must be 0.0-1.0")
            except (TypeError, ValueError):
                errors.append(f"Rule '{rule.get('name')}' threshold must be numeric")

        for rule in policy.get("custom_soft_rules", []):
            if "name" not in rule:
                errors.append("Each custom_soft_rule must have a 'name' field")
            w = rule.get("weight", 0)
            if not isinstance(w, (int, float)) or not (0.0 <= w <= 1.0):
                errors.append(f"custom_soft_rule '{rule.get('name', '?')}' weight must be 0.0–1.0")

        return errors
