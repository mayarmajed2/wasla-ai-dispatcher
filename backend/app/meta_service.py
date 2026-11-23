"""Meta WhatsApp Cloud API (Graph API v21.0) Outbound Messaging & Media Download Service.

Handles:
- Outbound text message delivery with retry/backoff.
- Two-step WhatsApp voice-note (audio/ogg) media download via Graph API.
- Safe dry-run reporting when META_ACCESS_TOKEN or META_PHONE_NUMBER_ID is not configured.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ai_service.security.guardrails import redact_pii

logger = logging.getLogger("wasla.meta_whatsapp")


class MetaWhatsAppClient:
    """Async client for official Meta WhatsApp Cloud API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        graph_version: str = "v21.0",
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.access_token = access_token.strip()
        self.phone_number_id = phone_number_id.strip()
        self.graph_base = f"https://graph.facebook.com/{graph_version}"
        self.timeout = timeout
        self._external_client = client

    @property
    def is_configured(self) -> bool:
        return bool(
            self.access_token
            and self.phone_number_id
            and not self.access_token.startswith("your_")
            and not self.phone_number_id.startswith("your_")
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def send_text_message(
        self,
        to_phone: str,
        body_text: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Send a WhatsApp text reply to an authorized recipient number."""
        if not self.is_configured:
            logger.info("Meta WhatsApp dry-run (credentials not configured) to %s", redact_pii(to_phone))
            return {
                "sent": False,
                "mode": "dry_run_unconfigured",
                "reason": "META_ACCESS_TOKEN or META_PHONE_NUMBER_ID is not set",
            }

        url = f"{self.graph_base}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_phone,
            "type": "text",
            "text": {"preview_url": False, "body": body_text},
        }

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if self._external_client is not None:
                    resp = await self._external_client.post(url, headers=self._headers(), json=payload)
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(url, headers=self._headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
                return {"sent": True, "mode": "live", "meta_response": data}
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        logger.error("Meta WhatsApp send failed after retries: %s", last_err)
        return {"sent": False, "mode": "error", "error": str(last_err)}

    async def download_media(self, media_id: str) -> tuple[bytes, str]:
        """Two-step official Meta Graph API media download (e.g., WhatsApp OGG voice note)."""
        if not self.is_configured:
            raise RuntimeError("META_ACCESS_TOKEN is not configured for media download.")

        meta_url = f"{self.graph_base}/{media_id}"
        auth_headers = {"Authorization": f"Bearer {self.access_token}"}

        if self._external_client is not None:
            info_resp = await self._external_client.get(meta_url, headers=auth_headers)
            info_resp.raise_for_status()
            info = info_resp.json()
            media_resp = await self._external_client.get(info["url"], headers=auth_headers)
            media_resp.raise_for_status()
            return media_resp.content, info.get("mime_type", "audio/ogg")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            info_resp = await client.get(meta_url, headers=auth_headers)
            info_resp.raise_for_status()
            info = info_resp.json()
            media_resp = await client.get(info["url"], headers=auth_headers)
            media_resp.raise_for_status()
            return media_resp.content, info.get("mime_type", "audio/ogg")
