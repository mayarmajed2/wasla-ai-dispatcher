"""Security, PII redaction, prompt-injection defense, and audio validation."""

from __future__ import annotations

import re

# Egyptian mobile numbers (010, 011, 012, 015 with optional +20 / 0020) and international phone numbers
_PHONE_RE = re.compile(r"(?:\+?20|0020)?\s*0?1[0125][\s\-]?\d{4}[\s\-]?\d{4}\b|\b\d{10,15}\b")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)", re.IGNORECASE),
    re.compile(r"تجاهل\s+(كل\s+)?التعليمات", re.IGNORECASE),
    re.compile(r"(اعمل|اديني)\s+خصم\s*100%|ببلاش|سعره\s*صفر", re.IGNORECASE),
]

SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
}


class AudioValidationError(ValueError):
    """Raised when an incoming audio payload fails security or format validation."""


def redact_pii(text: str) -> str:
    """Mask phone numbers and emails before writing logs or evaluation traces."""
    if not text:
        return text
    masked = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    masked = _EMAIL_RE.sub("[REDACTED_EMAIL]", masked)
    return masked


def detect_prompt_injection(text: str) -> bool:
    """Detect common prompt-injection or unauthorized price-override attempts."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def validate_audio_bytes(data: bytes, mime_type: str, max_mb: float = 10.0) -> None:
    """Validate incoming audio payload size, MIME type, and file header magic bytes."""
    if not data:
        raise AudioValidationError("Audio payload is empty.")

    max_bytes = int(max_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise AudioValidationError(f"Audio payload exceeds max allowed size ({max_mb} MB).")

    normalized_mime = mime_type.split(";")[0].strip().lower()
    if normalized_mime not in SUPPORTED_AUDIO_MIME_TYPES:
        raise AudioValidationError(f"Unsupported audio MIME type: {mime_type}")

    if len(data) < 4:
        raise AudioValidationError("Audio payload is too short to be valid.")

    header = data[:12]
    is_ogg = header.startswith(b"OggS")
    is_wav = header.startswith(b"RIFF") and b"WAVE" in header
    is_mp3 = header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
    is_mp4_m4a = len(header) >= 8 and header[4:8] == b"ftyp"
    is_webm = header.startswith(b"\x1a\x45\xdf\xa3")

    if not (is_ogg or is_wav or is_mp3 or is_mp4_m4a or is_webm):
        raise AudioValidationError("Audio payload failed magic-byte header verification.")
