"""
Groq Vision Service — primary multimodal analyzer for the journal pipeline.

Sends each image (base64) to Groq's chat-completions endpoint with a vision-capable
Llama-4 model (`meta-llama/llama-4-scout-17b-16e-instruct`) and asks for a strict
JSON response describing:
    - what the person/people are doing (in full detail)
    - their surroundings (indoor/outdoor, room type, props, colours, lighting)
    - whether the photo matches the user-supplied journal title
    - safety / quality flags
This replaces the on-device Qwen2-VL model so the agent works on lightweight
hosts and surfaces real "API activity" the user can see in the live stream.

The response feeds directly into the scoring pipeline:
    journal_match == "no" or "unrelated"  →  hard rejection
    journal_match == "yes"                →  approved (subject to quality gates)

Notes
-----
Uses ``httpx`` (already in requirements) — no extra dependency required.
Errors / missing API key fall back to ``None`` so the pipeline stays usable.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
import structlog

from app.core.config import settings

log = structlog.get_logger()

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an expert school/childcare photo curator. For every image you "
    "receive you MUST: (1) describe in full detail what the person or people "
    "are doing — their pose, action, expression, props they hold, what they "
    "are looking at; (2) describe their surroundings — indoor vs outdoor, "
    "room type, walls, floor, furniture, plants, lighting, dominant colours, "
    "any visible signage; (3) decide whether the scene clearly relates to the "
    "given JOURNAL TITLE. Be strict — if the journal title is e.g. \"Robotics "
    "Workshop\" and the image is a child eating lunch, mark it as not "
    "matching. Always reply with a single JSON object and nothing else."
)

USER_PROMPT_TEMPLATE = (
    "JOURNAL TITLE: \"{journal_title}\"\n\n"
    "Analyse the attached photograph and respond with a JSON object that has "
    "EXACTLY these keys:\n"
    "  - activity:           short snake_case label (e.g. \"sensory_play\", "
    "\"art_and_craft\", \"outdoor_exploration\").\n"
    "  - activity_label:     3-6 word human readable phrase (e.g. \"Toddler "
    "Exploring Sensory Toys\").\n"
    "  - activity_detail:    one or two sentences (max 40 words) describing "
    "in vivid detail what the subject is DOING — pose, hands, expression, "
    "props, gaze.\n"
    "  - surroundings:       one or two sentences (max 40 words) describing "
    "the environment — room/outdoor type, walls, floor, furniture, lighting, "
    "background objects, dominant colours.\n"
    "  - people_count:       integer count of people clearly visible.\n"
    "  - role_summary:       short string e.g. \"1 toddler\", \"2 children + "
    "1 teacher\".\n"
    "  - matches_journal:    \"yes\" if the scene CLEARLY relates to the "
    "journal title above, otherwise \"no\".\n"
    "  - match_reason:       one short sentence justifying the matches_journal "
    "decision.\n"
    "  - confidence:         one of \"high\" | \"medium\" | \"low\".\n"
    "  - safe:               \"yes\" if the photo is safe for a school journal, "
    "otherwise \"no\".\n"
    "  - quality_note:       one short sentence about photographic quality "
    "(sharpness, lighting, composition).\n"
    "Respond with ONLY the JSON object. No markdown, no commentary."
)


# ─── Result dataclass ─────────────────────────────────────────────────────


@dataclass
class GroqVisionResult:
    """Structured Groq vision-API response."""

    activity: str = "unknown"
    activity_label: str = "Unknown Activity"
    activity_detail: str = ""
    surroundings: str = ""
    people_count: Optional[int] = None
    role_summary: str = ""
    matches_journal: str = "unknown"  # "yes" | "no" | "unknown"
    match_reason: str = ""
    confidence: str = "low"
    safe: str = "yes"
    quality_note: str = ""

    raw_response: str = ""
    model: str = ""
    processing_time_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        return self.matches_journal.strip().lower().startswith("y")

    @property
    def journal_provided(self) -> bool:
        return bool(self.extra.get("journal_title"))


# ─── Service ──────────────────────────────────────────────────────────────


class GroqVisionService:
    """
    Stateless wrapper around Groq's chat-completions vision endpoint.

    Public methods
    --------------
    analyse(image_bytes, journal_title, mime_type=None) -> GroqVisionResult | None
        Run a single image through Groq and return a parsed result. Returns
        ``None`` if Groq is disabled or no API key is configured.
    """

    _cache: Dict[str, GroqVisionResult] = {}

    def __init__(self) -> None:
        self.api_key: str = (settings.GROQ_API_KEY or "").strip()
        self.model: str = (settings.GROQ_MODEL or "meta-llama/llama-4-scout-17b-16e-instruct").strip()
        self.enabled: bool = bool(settings.GROQ_ENABLED) and bool(self.api_key)
        self.timeout: float = float(settings.GROQ_TIMEOUT_S or 45.0)
        self.max_tokens: int = int(settings.GROQ_MAX_TOKENS or 700)
        self.cache_enabled: bool = bool(settings.GROQ_CACHE_ENABLED)

        if settings.GROQ_ENABLED and not self.api_key:
            log.warning(
                "groq.disabled_no_key",
                msg="GROQ_ENABLED=true but GROQ_API_KEY is empty — Groq calls will be skipped.",
            )

    # ── Public API ───────────────────────────────────────────────────────

    def analyse(
        self,
        image_bytes: bytes,
        journal_title: Optional[str] = None,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[GroqVisionResult]:
        """Send the image to Groq and return the parsed analysis."""
        if not self.enabled:
            return None
        if not image_bytes:
            return None

        title = (journal_title or "").strip() or "(no journal title supplied)"
        cache_key = self._cache_key(image_bytes, title)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.cached = True
            log.info("groq.cache_hit", filename=filename, title=title[:60])
            return cached

        start = time.time()
        mime = mime_type or "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_PROMPT_TEMPLATE.format(journal_title=title),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(GROQ_ENDPOINT, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPStatusError as exc:
            err_msg = self._extract_http_error(exc)
            log.warning(
                "groq.http_error",
                filename=filename,
                status=exc.response.status_code,
                error=err_msg,
            )
            return GroqVisionResult(
                error=f"Groq HTTP {exc.response.status_code}: {err_msg}",
                model=self.model,
                processing_time_ms=(time.time() - start) * 1000.0,
                extra={"journal_title": title},
            )
        except Exception as exc:
            log.warning("groq.request_failed", filename=filename, error=str(exc))
            return GroqVisionResult(
                error=f"Groq request failed: {exc}",
                model=self.model,
                processing_time_ms=(time.time() - start) * 1000.0,
                extra={"journal_title": title},
            )

        elapsed_ms = (time.time() - start) * 1000.0
        text = self._extract_text(body)
        result = self._parse(text)
        result.raw_response = text
        result.model = self.model
        result.processing_time_ms = elapsed_ms
        result.extra["journal_title"] = title
        result.extra["usage"] = body.get("usage", {})

        log.info(
            "groq.analysed",
            filename=filename,
            model=self.model,
            elapsed_ms=round(elapsed_ms, 1),
            activity=result.activity_label,
            matches=result.matches_journal,
            confidence=result.confidence,
        )

        if self.cache_enabled:
            self._cache[cache_key] = result

        return result

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(image_bytes: bytes, title: str) -> str:
        h = hashlib.md5()
        h.update(image_bytes)
        h.update(b"||")
        h.update(title.encode("utf-8", "ignore"))
        return h.hexdigest()

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> str:
        try:
            choices = body.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # OpenAI-compat sometimes returns a list of parts
                texts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "\n".join(t for t in texts if t)
        except Exception:
            return ""
        return ""

    @staticmethod
    def _extract_http_error(exc: httpx.HTTPStatusError) -> str:
        try:
            data = exc.response.json()
            if isinstance(data, dict):
                err = data.get("error") or {}
                if isinstance(err, dict):
                    return err.get("message") or json.dumps(err)
                return str(err)
            return exc.response.text[:300]
        except Exception:
            return exc.response.text[:300] if exc.response.text else str(exc)

    @staticmethod
    def _parse(text: str) -> GroqVisionResult:
        """Parse the Groq JSON response into a GroqVisionResult."""
        if not text:
            return GroqVisionResult(error="Empty response from Groq")

        cleaned = text.strip()
        # Strip markdown fences if the model wrapped JSON in ```json ... ```
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        data: Dict[str, Any] = {}
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Best-effort: pull the first JSON object out of the string
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {}

        if not isinstance(data, dict):
            return GroqVisionResult(error="Groq response was not a JSON object")

        def _str(key: str, default: str = "") -> str:
            v = data.get(key, default)
            if isinstance(v, str):
                return v.strip()
            if v is None:
                return default
            return str(v).strip()

        people_count: Optional[int] = None
        raw_count = data.get("people_count")
        if isinstance(raw_count, int):
            people_count = raw_count
        elif isinstance(raw_count, str):
            digits = re.findall(r"\d+", raw_count)
            if digits:
                people_count = int(digits[0])
        elif isinstance(raw_count, float):
            people_count = int(raw_count)

        matches = _str("matches_journal", "unknown").lower()
        if matches.startswith("y"):
            matches = "yes"
        elif matches.startswith("n"):
            matches = "no"
        else:
            matches = "unknown"

        confidence = _str("confidence", "low").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        safe = _str("safe", "yes").lower()
        safe = "yes" if safe.startswith("y") else "no"

        return GroqVisionResult(
            activity=_str("activity", "unknown") or "unknown",
            activity_label=_str("activity_label", "Unknown Activity") or "Unknown Activity",
            activity_detail=_str("activity_detail", ""),
            surroundings=_str("surroundings", ""),
            people_count=people_count,
            role_summary=_str("role_summary", ""),
            matches_journal=matches,
            match_reason=_str("match_reason", ""),
            confidence=confidence,
            safe=safe,
            quality_note=_str("quality_note", ""),
        )
