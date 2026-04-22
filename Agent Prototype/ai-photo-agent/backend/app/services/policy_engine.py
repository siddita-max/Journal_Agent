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
  "prohibited_objects": ["cell phone", "laptop"],
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

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

log = structlog.get_logger()

# ─── Default policy (used if no DB policy is active) ─────────────────────

DEFAULT_POLICY: Dict[str, Any] = {
    "max_people": 10,
    "prohibited_objects": ["cell phone"],
    "min_professionalism_score": 0.0,       # 0.0 = not enforced by default
    "required_context_prompts": [],
    "dress_code": {
        "enabled": False,
        "required_prompts": ["formal attire", "business dress"],
        "min_score": 0.35,
    },
    "custom_hard_rules": [],
    "custom_soft_rules": [],
}


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
        self.policy = self._merge_with_defaults(policy or {})

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
            return PolicyViolation(
                rule_name=name,
                severity="soft",
                description=desc,
                measured_value=round(avg, 3),
                threshold=min_score,
            ), weight
        return None, 0.0

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _merge_with_defaults(policy: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge user policy on top of DEFAULT_POLICY."""
        merged = dict(DEFAULT_POLICY)
        for key, value in policy.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
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

        for rule in policy.get("custom_hard_rules", []):
            if "name" not in rule:
                errors.append("Each custom_hard_rule must have a 'name' field")
            if "clip_negative_prompts" not in rule or not isinstance(rule["clip_negative_prompts"], list):
                errors.append(f"custom_hard_rule '{rule.get('name', '?')}' must have 'clip_negative_prompts' list")

        for rule in policy.get("custom_soft_rules", []):
            if "name" not in rule:
                errors.append("Each custom_soft_rule must have a 'name' field")
            w = rule.get("weight", 0)
            if not isinstance(w, (int, float)) or not (0.0 <= w <= 1.0):
                errors.append(f"custom_soft_rule '{rule.get('name', '?')}' weight must be 0.0–1.0")

        return errors
