"""
Groq Vision Service — primary multimodal analyzer for the journal pipeline.

Sends each image (base64) to Groq's chat-completions endpoint with a vision-capable
Llama-4 model (`meta-llama/llama-4-scout-17b-16e-instruct`) and asks for a strict
JSON response describing:
    - what the person/people are doing (in full detail)
    - their surroundings (indoor/outdoor, room type, props, colours, lighting)
    - whether the photo matches the user-supplied journal title
    - safety / quality flags
Works on lightweight hosts and surfaces real "API activity" the user can see in the live stream.

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
from typing import Any, Dict, List, Optional

import httpx
import structlog

from app.core.config import settings

log = structlog.get_logger()

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a school-photo curator. Return ONLY a JSON object — no markdown, no text outside JSON. "
    "Keys: activity (snake_case), activity_label (3-6 words), activity_detail (≤20 words: what subjects do), "
    "surroundings (≤15 words: setting), people_count (int), role_summary (e.g. '2 children+1 teacher'), "
    "scene_summary (one full sentence ≤30 words starting with the people present e.g. "
    "'Two teachers and five students are engaged in outdoor nature exploration with magnifying glasses.'), "
    "matches_journal (yes/no: does scene relate to JOURNAL TITLE?), match_reason (≤10 words), "
    "confidence (high/medium/low), safe (yes/no for school), "
    "quality (good=sharp+well-lit / acceptable=minor issues / poor=blurry or unusable), "
    "quality_note (≤10 words)."
)

USER_PROMPT_TEMPLATE = (
    "JOURNAL TITLE: \"{journal_title}\"\n"
    "Analyse the photo. Return ONLY the JSON object."
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
    scene_summary: str = ""  # One full sentence: who is present + what they're doing
    matches_journal: str = "unknown"  # "yes" | "no" | "unknown"
    match_reason: str = ""
    confidence: str = "low"
    safe: str = "yes"
    quality_note: str = ""
    quality: str = "acceptable"   # "good" | "acceptable" | "poor"

    raw_response: str = ""
    model: str = ""
    processing_time_ms: float = 0.0
    cached: bool = False
    rate_limited: bool = False  # True when the API returned HTTP 429
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
        # Build the key pool: primary key + any extras from GROQ_API_KEYS
        primary = (settings.GROQ_API_KEY or "").strip()
        extras_raw = (getattr(settings, "GROQ_API_KEYS", "") or "").strip()
        extras = [k.strip() for k in extras_raw.split(",") if k.strip()]
        seen: set = set()
        self._key_pool: List[str] = []
        for k in ([primary] + extras):
            if k and k not in seen:
                seen.add(k)
                self._key_pool.append(k)
        self._key_idx: int = 0
        # Keys that have hit their daily token limit — skipped until restart.
        self._exhausted_keys: set = set()

        # Keep api_key pointing at the first key for compatibility
        self.api_key: str = self._key_pool[0] if self._key_pool else ""
        self.model: str = (settings.GROQ_MODEL or "meta-llama/llama-4-scout-17b-16e-instruct").strip()
        self.enabled: bool = bool(settings.GROQ_ENABLED) and bool(self._key_pool)
        self.timeout: float = float(settings.GROQ_TIMEOUT_S or 45.0)
        self.max_tokens: int = int(settings.GROQ_MAX_TOKENS or 700)
        self.cache_enabled: bool = bool(settings.GROQ_CACHE_ENABLED)
        self.call_delay_s: float = float(getattr(settings, "GROQ_CALL_DELAY_S", 0.0))

        if settings.GROQ_ENABLED and not self._key_pool:
            log.warning(
                "groq.disabled_no_key",
                msg="GROQ_ENABLED=true but GROQ_API_KEY is empty — Groq calls will be skipped.",
            )
        elif len(self._key_pool) > 1:
            log.info("groq.key_pool_ready", key_count=len(self._key_pool))

    def _next_key(self) -> str:
        """Return the next non-exhausted API key (round-robin), skipping daily-limit keys."""
        if not self._key_pool:
            return ""
        # Try every key in the pool before giving up
        for _ in range(len(self._key_pool)):
            key = self._key_pool[self._key_idx % len(self._key_pool)]
            self._key_idx += 1
            if key not in self._exhausted_keys:
                return key
        # All keys exhausted — return empty so caller can signal failure cleanly
        return ""

    def _active_key_count(self) -> int:
        return len(self._key_pool) - len(self._exhausted_keys)

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

        if self.call_delay_s > 0:
            time.sleep(self.call_delay_s)

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

        # Try each non-exhausted key; on daily-limit 429, mark key and move on.
        body = None
        last_error: Optional[str] = None
        all_rate_limited = False
        attempts = self._active_key_count() or len(self._key_pool)
        for attempt in range(attempts):
            active_key = self._next_key()
            if not active_key:
                all_rate_limited = True
                break
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(GROQ_ENDPOINT, json=payload, headers=headers)
                    resp.raise_for_status()
                    body = resp.json()
                    break  # success
            except httpx.HTTPStatusError as exc:
                err_msg = self._extract_http_error(exc)
                last_error = f"Groq HTTP {exc.response.status_code}: {err_msg}"
                if exc.response.status_code == 429:
                    is_daily = "tokens per day" in err_msg.lower() or "tpd" in err_msg.lower()
                    if is_daily:
                        self._exhausted_keys.add(active_key)
                        log.warning(
                            "groq.key_exhausted_daily",
                            filename=filename,
                            active_keys_remaining=self._active_key_count(),
                            error=err_msg,
                        )
                        if self._active_key_count() == 0:
                            all_rate_limited = True
                            break
                        continue  # retry with next key
                    else:
                        log.warning(
                            "groq.rate_limited",
                            filename=filename,
                            status=exc.response.status_code,
                            error=err_msg,
                            key_pool_size=len(self._key_pool),
                        )
                else:
                    log.warning(
                        "groq.http_error",
                        filename=filename,
                        status=exc.response.status_code,
                        error=err_msg,
                    )
                break
            except Exception as exc:
                last_error = f"Groq request failed: {exc}"
                log.warning("groq.request_failed", filename=filename, error=str(exc))
                break

        if body is None:
            if all_rate_limited:
                log.warning("groq.all_keys_exhausted", filename=filename)
            return GroqVisionResult(
                error=last_error or "All Groq keys exhausted for today",
                rate_limited=True,
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

        quality = _str("quality", "acceptable").lower()
        if quality not in {"good", "acceptable", "poor"}:
            # Coerce synonyms the model might use
            if quality.startswith("g") or "high" in quality or "great" in quality or "sharp" in quality:
                quality = "good"
            elif quality.startswith("p") or "blur" in quality or "bad" in quality or "low" in quality:
                quality = "poor"
            else:
                quality = "acceptable"

        return GroqVisionResult(
            activity=_str("activity", "unknown") or "unknown",
            activity_label=_str("activity_label", "Unknown Activity") or "Unknown Activity",
            activity_detail=_str("activity_detail", ""),
            surroundings=_str("surroundings", ""),
            people_count=people_count,
            role_summary=_str("role_summary", ""),
            scene_summary=_str("scene_summary", ""),
            matches_journal=matches,
            match_reason=_str("match_reason", ""),
            confidence=confidence,
            safe=safe,
            quality_note=_str("quality_note", ""),
            quality=quality,
        )
