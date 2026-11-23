"""Centralized environment-driven configuration for Wasla AI Service.

Never hardcodes API keys or secrets. Reads strictly from environment variables / local .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class WaslaConfig:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Verified active Gemini models on Google AI Studio (Nov 2026)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    ai_primary_model: str = field(default_factory=lambda: os.getenv("AI_PRIMARY_MODEL", "gemini-3.5-flash-lite"))
    ai_fallback_model: str = field(default_factory=lambda: os.getenv("AI_FALLBACK_MODEL", "gemini-3.8-flash"))

    # Hybrid Audio (Option C: Direct Gemini Audio + Groq Whisper Large v3 Turbo fallback)
    ai_audio_mode: str = field(default_factory=lambda: os.getenv("AI_AUDIO_MODE", "hybrid"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_whisper_model: str = field(default_factory=lambda: os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Backend Integration
    backend_adapter_mode: str = field(default_factory=lambda: os.getenv("BACKEND_ADAPTER_MODE", "mock"))
    backend_base_url: str = field(default_factory=lambda: os.getenv("BACKEND_BASE_URL", "http://localhost:8000"))
    backend_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("BACKEND_TIMEOUT_SECONDS", "10.0")))

    # Conversation, Upsell & Safety
    message_debounce_seconds: float = field(default_factory=lambda: float(os.getenv("MESSAGE_DEBOUNCE_SECONDS", "3.0")))
    session_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", "3600")))
    redact_pii_in_logs: bool = field(default_factory=lambda: os.getenv("REDACT_PII_IN_LOGS", "true").lower() == "true")
    upsell_enabled: bool = field(default_factory=lambda: os.getenv("UPSELL_ENABLED", "true").lower() == "true")
    complaint_escalation_enabled: bool = field(
        default_factory=lambda: os.getenv("COMPLAINT_ESCALATION_ENABLED", "true").lower() == "true"
    )
    max_audio_file_mb: float = field(default_factory=lambda: float(os.getenv("MAX_AUDIO_FILE_MB", "10.0")))
    menu_json_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MENU_JSON_PATH", str(Path(__file__).resolve().parent.parent / "menu" / "menu.json"))
        )
    )
