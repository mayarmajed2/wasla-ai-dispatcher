"""Option C — Hybrid Egyptian Arabic Voice-First Audio Pipeline.

Supports:
- Mode 'direct' (Option A): Sends validated WhatsApp audio/ogg directly to Gemini 3.5/3.8 Flash.
- Mode 'stt'    (Option B): Transcribes audio via Groq Whisper Large v3 Turbo (or OpenAI gpt-transcribe).
- Mode 'hybrid' (Option C - Recommended): Direct Gemini Audio primary with automatic Whisper fallback
  when the primary result is empty, has a provider-supplied low confidence, or processing fails.
"""

from __future__ import annotations

import base64
from collections import Counter
import re
import time
import unicodedata

import httpx

from ai_service.config import WaslaConfig
from ai_service.schemas.models import AudioInput, AudioTranscriptionResult
from ai_service.security.guardrails import validate_audio_bytes

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"

_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")
_TRANSCRIPT_TOKENS = re.compile(r"[^\W\d_]+", re.UNICODE)
_INCOMPLETE_REQUEST_STARTERS = {
    "عايز",
    "عايزة",
    "عايزه",
    "عايزين",
    "عاوز",
    "عاوزة",
    "عاوزه",
    "عاوزين",
    "هات",
    "هاتي",
    "هاتلي",
    "ممكن",
    "اريد",
    "أريد",
}
_INCOMPLETE_FILLERS = {"بقى", "بس", "لو", "سمحت", "من", "فضلك"}
_TRAILING_CONNECTORS = {
    "و",
    "أو",
    "او",
    "من",
    "في",
    "على",
    "علي",
    "ب",
    "ل",
    "عن",
    "لو",
    "إلى",
    "الى",
}


def _has_sufficient_transcript_quality(transcript: str) -> bool:
    """Reject clearly incomplete or corrupted text; this is not a recognition confidence score."""
    normalized = unicodedata.normalize("NFKC", transcript).casefold()
    if "\ufffd" in normalized or any(
        ord(char) < 32 and char not in "\t\r\n" for char in normalized
    ):
        return False
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    analysis_text = re.sub(r"(.)\1{2,}", r"\1", normalized)
    tokens = _TRANSCRIPT_TOKENS.findall(normalized)
    analysis_tokens = _TRANSCRIPT_TOKENS.findall(analysis_text)
    letters = [char for char in normalized if char.isalpha()]
    visible = [char for char in normalized if not char.isspace()]
    if (
        len(letters) < 4
        or not visible
        or len(letters) / len(visible) < 0.75
        or not tokens
    ):
        return False
    if any(len(token) > 32 for token in tokens):
        return False
    if any(
        len(token) >= 4 and max(Counter(token).values()) / len(token) >= 0.75
        for token in tokens
    ):
        return False
    if analysis_tokens[-1] in _TRAILING_CONNECTORS:
        return False
    if analysis_tokens[0] in _INCOMPLETE_REQUEST_STARTERS and all(
        token in _INCOMPLETE_FILLERS for token in analysis_tokens[1:]
    ):
        return False
    return True


class HybridAudioPipeline:
    """Validates and transcribes/processes Egyptian Arabic WhatsApp voice notes."""

    def __init__(self, config: WaslaConfig, confidence_threshold: float = 0.72):
        self.config = config
        self.confidence_threshold = confidence_threshold

    async def process(self, audio: AudioInput) -> AudioTranscriptionResult:
        validate_audio_bytes(audio.audio_bytes, audio.mime_type, max_mb=self.config.max_audio_file_mb)
        mode = self.config.ai_audio_mode.lower()

        if mode == "stt":
            return self._assess_result(await self._transcribe_stt(audio))

        if mode == "direct":
            return self._assess_result(await self._transcribe_gemini_direct(audio))

        # Default: Option C (Hybrid)
        primary_error: Exception | None = None
        try:
            primary = await self._transcribe_gemini_direct(audio)
            assessed_primary = self._assess_result(primary)
            if assessed_primary.is_usable:
                return assessed_primary
        except Exception as exc:
            primary_error = exc
            primary = None

        if self.config.groq_api_key or self.config.openai_api_key:
            fallback = await self._transcribe_stt(audio)
            return self._assess_result(
                fallback.model_copy(update={"used_fallback": True})
            )

        if primary is not None:
            return self._assess_result(primary)
        if primary_error is not None:
            raise RuntimeError("Gemini audio processing failed and no STT fallback provider is configured.") from primary_error
        raise RuntimeError("No active audio provider configured. Set GEMINI_API_KEY or GROQ_API_KEY in .env.")

    def _assess_result(self, result: AudioTranscriptionResult) -> AudioTranscriptionResult:
        quality_ok = _has_sufficient_transcript_quality(result.transcript)
        confidence_ok = result.confidence is None or result.confidence >= self.confidence_threshold
        return result.model_copy(update={"is_usable": quality_ok and confidence_ok})

    async def _transcribe_gemini_direct(self, audio: AudioInput) -> AudioTranscriptionResult:
        if not self.config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured for direct audio processing.")

        started = time.perf_counter()
        b64_audio = base64.b64encode(audio.audio_bytes).decode("ascii")
        url = f"{GEMINI_API_BASE}/{self.config.ai_primary_model}:generateContent"
        headers = {"x-goog-api-key": self.config.gemini_api_key, "Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "استمع للتسجيل الصوتي بالعامية المصرية لعميل بيطلب من مطعم MOKA. "
                                "اكتب النص الدقيق للرسالة بالعامية المصرية كما نطقها العميل بدون إضافة أو تفسير."
                            )
                        },
                        {"inlineData": {"mimeType": audio.mime_type, "data": b64_audio}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        transcript = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return AudioTranscriptionResult(
            transcript=transcript,
            provider=f"gemini:{self.config.ai_primary_model}",
            latency_ms=round(latency_ms, 2),
            used_fallback=False,
        )

    async def _transcribe_stt(self, audio: AudioInput) -> AudioTranscriptionResult:
        started = time.perf_counter()
        if self.config.groq_api_key:
            url = GROQ_TRANSCRIPTION_URL
            headers = {"Authorization": f"Bearer {self.config.groq_api_key}"}
            model = self.config.groq_whisper_model
            provider = f"groq:{model}"
        elif self.config.openai_api_key:
            url = OPENAI_TRANSCRIPTION_URL
            headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
            model = "gpt-transcribe"
            provider = f"openai:{model}"
        else:
            raise RuntimeError("Neither GROQ_API_KEY nor OPENAI_API_KEY is configured for STT.")

        files = {"file": (audio.filename, audio.audio_bytes, audio.mime_type)}
        form_data = {
            "model": model,
            "language": "ar",
            "prompt": "طلب مطعم مصري موكا: كريب، بيتزا، برجر، قشطوطة، وافل، بان كيك، ميلك شيك، حبة كاملة، ستربس، رانش، باربيكيو.",
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, files=files, data=form_data)
            resp.raise_for_status()
            data = resp.json()

        transcript = data.get("text", "").strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return AudioTranscriptionResult(
            transcript=transcript,
            provider=provider,
            latency_ms=round(latency_ms, 2),
            used_fallback=False,
        )
