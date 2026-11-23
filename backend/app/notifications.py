"""Official Zero-Cost Notification Services & Meta WhatsApp Policy Guard.

Covers:
1. Meta WhatsApp Groups API Eligibility & 2026 24-Hour Customer Service Window Policy Guard
   - Groups API strictly requires Official Business Account (OBA / green checkmark), max 8
     participants, invite-only, and per-recipient per-message billing.
   - Standard Cloud API accounts & test numbers are NOT eligible -> use Telegram Bot API (0 EGP).
   - Free-form customer service replies inside 24h window are capped at 1,000 free/month
     (Oct 1, 2026 Meta pricing); utility templates are billable inside & outside 24h.
   - Never silently send a billable WhatsApp message when `meta_billable_authorized` is False.
2. Telegram Bot API Client (official `https://api.telegram.org/bot{token}/sendMessage`)
   - Instant zero-cost Cashier Group order alerts with minimal PII & secure dashboard link.
   - Zero-cost fallback (and primary when WhatsApp is ineligible/billable) for Owner Complaint Escalations.
3. Full idempotency via `notification_attempts.idempotency_key` — never sends duplicate notifications.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_service.security.guardrails import redact_pii
from app.config import Settings
from app.meta_service import MetaWhatsAppClient
from app.models import (
    ChatSession,
    ComplaintTicket,
    FeedbackRequest,
    NotificationAttempt,
    Order,
)

logger = logging.getLogger("wasla.notifications")

REVIEW_MESSAGE_AR = (
    "يارب الأكل يكون عجبك ❤️\n"
    "تحب تقولنا رأيك في الأوردر؟\n"
    "تقييمك بيفرق معانا جداً 🌟\n"
    "رد برقم من 1 لـ 5 (5 ممتاز) أو ابعتلنا أي ملاحظة."
)

ARRIVAL_CHECK_MESSAGE_AR = (
    "أهلاً بيك من MOKA ❤️\n"
    "بنطمن بس: هل الأوردر وصل لحضرتك بالسلامة؟"
)


def format_restaurant_time(dt: datetime | None, tz_name: str = "Africa/Cairo") -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        local_dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local_dt = dt
    return local_dt.strftime("%Y-%m-%d %H:%M (%Z)")


class MetaPolicyGuard:
    """Enforces Meta WhatsApp Cloud API 2026 rules and zero-cost constraints."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def groups_api_eligibility(self) -> dict[str, Any]:
        eligible = bool(self.settings.meta_is_official_business_account)
        return {
            "eligible": eligible,
            "requires_official_business_account": True,
            "max_participants_per_group": 8,
            "supports_existing_whatsapp_groups": False,
            "invite_only": True,
            "billing_model": "per_message_per_recipient",
            "supports_interactive_messages": False,
            "recommended_zero_cost_channel": "telegram_bot_private_group",
            "reason": (
                "Official Business Account (OBA) enabled."
                if eligible
                else "Account is not an Official Business Account (OBA) and WhatsApp Groups API bills per recipient; using zero-cost Telegram Bot private cashier group."
            ),
        }

    def can_send_free_whatsapp_service_message(
        self,
        db: Session,
        recipient_phone: str,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """Verify 24h customer service window + 1,000/mo free quota + zero-cost constraint."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        session = db.scalar(select(ChatSession).where(ChatSession.phone_number == recipient_phone))
        last_msg = session.last_customer_message_at if session else None

        if last_msg is None:
            if self.settings.meta_billable_messages_authorized:
                return True, "billable_template_authorized"
            return False, "outside_24h_window_no_customer_message_zero_cost_blocked"

        if last_msg.tzinfo is None:
            last_msg = last_msg.replace(tzinfo=timezone.utc)

        if (now - last_msg) > timedelta(hours=24):
            if self.settings.meta_billable_messages_authorized:
                return True, "billable_template_authorized"
            return False, "outside_24h_customer_service_window_zero_cost_blocked"

        # Check monthly free service message usage (Oct 2026 Meta rule: 1,000 free/month)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        sent_this_month = db.scalar(
            select(func.count(NotificationAttempt.id)).where(
                NotificationAttempt.channel == "whatsapp",
                NotificationAttempt.status == "SENT",
                NotificationAttempt.created_at >= month_start,
            )
        ) or 0
        if sent_this_month >= self.settings.meta_monthly_free_service_limit:
            if self.settings.meta_billable_messages_authorized:
                return True, "billable_over_quota_authorized"
            return False, "monthly_1000_free_service_quota_exhausted_zero_cost_blocked"

        return True, "within_24h_free_service_window"


class TelegramBotClient:
    """Official zero-cost Telegram Bot API client (`sendMessage`). Never fabricates credentials."""

    def __init__(
        self,
        bot_token: str,
        timeout: float = 12.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.bot_token = (bot_token or "").strip()
        self.timeout = timeout
        self._external_client = client

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and not self.bot_token.startswith("your_"))

    async def send_message(
        self,
        chat_id: str,
        text: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        target = (chat_id or "").strip()
        if not self.is_configured or not target:
            return {
                "sent": False,
                "mode": "unconfigured",
                "reason": "TELEGRAM_BOT_TOKEN or target chat_id is not configured",
            }

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": target,
            "text": text,
            "disable_web_page_preview": True,
        }
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if self._external_client is not None:
                    resp = await self._external_client.post(url, json=payload)
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok", True):
                    raise RuntimeError(data.get("description", "Telegram API returned ok=false"))
                return {"sent": True, "mode": "live", "telegram_response": data}
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    await asyncio.sleep(0.4 * (attempt + 1))

        logger.error("Telegram sendMessage failed after retries: %s", last_err)
        return {"sent": False, "mode": "error", "error": str(last_err)}


class NotificationService:
    """Coordinates idempotent Cashier Group, Owner Escalation, and Customer Review delivery."""

    def __init__(
        self,
        settings: Settings,
        meta_client: MetaWhatsAppClient,
        telegram_client: TelegramBotClient,
    ):
        self.settings = settings
        self.meta_client = meta_client
        self.telegram_client = telegram_client
        self.policy_guard = MetaPolicyGuard(settings)

    def format_cashier_order_message(self, order: Order) -> str:
        """Format clean cashier group order alert with minimal customer PII + dashboard link."""
        lines = ["NEW MOKA ORDER", "", f"Order: #{order.id}", "Items:"]
        for item in order.items_snapshot or []:
            qty = item.get("quantity", 1)
            name = item.get("name", "Item")
            lines.append(f"- {qty}x {name}")
        lines.extend(
            [
                "",
                f"Total: {order.total_amount} EGP",
                f"Order Type: {getattr(order, 'order_type', 'Delivery')}",
                f"Status: {order.status.value if hasattr(order.status, 'value') else order.status}",
                "Delivery info: Open dashboard (authorized staff only).",
                f"Dashboard: {self.settings.dashboard_public_url.rstrip('/')}/#order-{order.id}",
                f"Timestamp: {format_restaurant_time(order.created_at, self.settings.restaurant_timezone)}",
            ]
        )
        return "\n".join(lines)

    def format_owner_complaint_message(self, ticket: ComplaintTicket) -> str:
        """Format urgent owner alert without exposing sensitive data to unauthorized parties."""
        order_ref = f"#{ticket.order_id}" if ticket.order_id else "N/A"
        contact_ref = redact_pii(ticket.customer_phone)
        return (
            "URGENT CUSTOMER COMPLAINT\n\n"
            f"Ticket: #{ticket.id}\n"
            f"Order: {order_ref}\n"
            f"Issue: {ticket.category}\n"
            f"Severity: {ticket.severity}\n"
            f"Customer Ref: {contact_ref}\n"
            f"Summary: {ticket.summary}\n"
            f"Time: {format_restaurant_time(ticket.created_at, self.settings.restaurant_timezone)}\n\n"
            f"Status: Requires Owner Attention\n"
            f"Dashboard: {self.settings.dashboard_public_url.rstrip('/')}/#complaint-{ticket.id}"
        )

    async def notify_cashier_new_order(self, db: Session, order: Order) -> dict[str, Any]:
        """Deliver confirmed order alert to Cashier Telegram Group (idempotent, fail-safe)."""
        idem_key = f"cashier_order:{order.id}"
        existing = db.scalar(select(NotificationAttempt).where(NotificationAttempt.idempotency_key == idem_key))
        if existing is not None and existing.status == "SENT":
            return {"sent": True, "duplicate_prevented": True, "channel": existing.channel}

        attempt = existing or NotificationAttempt(
            idempotency_key=idem_key,
            notification_type="cashier_order",
            channel="telegram",
            recipient_ref="cashier_group",
            order_id=order.id,
            status="PENDING",
            attempts=0,
        )
        if existing is None:
            db.add(attempt)

        msg = self.format_cashier_order_message(order)
        attempt.attempts = (attempt.attempts or 0) + 1
        result = await self.telegram_client.send_message(self.settings.cashier_telegram_chat_id, msg)
        attempt.provider_response = result
        if result.get("sent"):
            attempt.status = "SENT"
            attempt.last_error = None
        elif result.get("mode") == "unconfigured":
            attempt.status = "DRY_RUN"
            attempt.last_error = result.get("reason")
        else:
            attempt.status = "FAILED"
            attempt.last_error = result.get("error", "Unknown Telegram error")
        db.flush()
        return {
            "sent": bool(result.get("sent")),
            "status": attempt.status,
            "channel": "telegram",
            "error": attempt.last_error,
            "groups_api_eligibility": self.policy_guard.groups_api_eligibility(),
        }

    async def escalate_complaint_to_owner(
        self,
        db: Session,
        ticket: ComplaintTicket,
        *,
        critical_update: bool = False,
    ) -> dict[str, Any]:
        """Escalate complaint to Owner (zero-cost WhatsApp if policy-eligible, else instant Telegram fallback)."""
        if not self.settings.complaint_escalation_enabled:
            return {"escalated": False, "reason": "complaint_escalation_disabled"}

        # Deduplicate repeated notifications for the same unresolved incident unless critical new info arrives
        if ticket.owner_notified and not critical_update:
            return {
                "escalated": False,
                "duplicate_prevented": True,
                "ticket_id": ticket.id,
                "channel": ticket.escalation_channel,
            }

        seq = ticket.escalation_count + 1
        idem_key = f"owner_complaint:{ticket.id}:v{seq}"
        msg = self.format_owner_complaint_message(ticket)

        channel_used = "dashboard_only"
        delivery_res: dict[str, Any] = {}
        wa_blocked_reason: str | None = None

        # 1. Try Owner WhatsApp ONLY if configured AND zero-cost policy permits
        owner_wa = (self.settings.restaurant_owner_whatsapp or "").strip()
        if owner_wa and self.meta_client.is_configured:
            allowed, reason = self.policy_guard.can_send_free_whatsapp_service_message(db, owner_wa)
            if allowed:
                wa_res = await self.meta_client.send_text_message(owner_wa, msg)
                if wa_res.get("sent"):
                    channel_used = "whatsapp"
                    delivery_res = wa_res
                else:
                    wa_blocked_reason = wa_res.get("error") or wa_res.get("reason")
            else:
                wa_blocked_reason = reason

        # 2. Immediate zero-cost Telegram fallback whenever WhatsApp unavailable/blocked/failed
        if channel_used != "whatsapp":
            owner_tg = (
                self.settings.restaurant_owner_telegram_chat_id or self.settings.cashier_telegram_chat_id or ""
            ).strip()
            tg_res = await self.telegram_client.send_message(owner_tg, msg)
            delivery_res = {"whatsapp_skipped_reason": wa_blocked_reason, "telegram": tg_res}
            if tg_res.get("sent"):
                channel_used = "telegram"
            else:
                channel_used = "dashboard_only"

        status = "SENT" if channel_used in ("whatsapp", "telegram") else "FAILED"
        if channel_used == "dashboard_only" and delivery_res.get("telegram", {}).get("mode") == "unconfigured":
            status = "DRY_RUN"

        db.add(
            NotificationAttempt(
                idempotency_key=idem_key,
                notification_type="owner_complaint",
                channel=channel_used,
                recipient_ref="restaurant_owner",
                order_id=ticket.order_id,
                complaint_id=ticket.id,
                status=status,
                attempts=1,
                provider_response=delivery_res,
                last_error=None if status == "SENT" else (wa_blocked_reason or delivery_res.get("telegram", {}).get("reason")),
            )
        )
        ticket.owner_notified = True
        ticket.status = "ESCALATED" if ticket.status == "OPEN" else ticket.status
        ticket.escalation_channel = channel_used
        ticket.escalation_count = seq
        ticket.last_escalated_at = datetime.now(timezone.utc)
        db.flush()

        return {
            "escalated": True,
            "ticket_id": ticket.id,
            "channel": channel_used,
            "status": status,
            "whatsapp_skipped_reason": wa_blocked_reason,
        }

    async def send_customer_review_request(
        self,
        db: Session,
        order: Order,
        feedback: FeedbackRequest,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Send 1-hour post-delivery review request respecting Meta 24h & zero-cost policy."""
        now = now or datetime.now(timezone.utc)
        idem_key = f"customer_review:{order.id}"
        existing = db.scalar(select(NotificationAttempt).where(NotificationAttempt.idempotency_key == idem_key))
        if existing is not None and existing.status == "SENT":
            return {"sent": True, "duplicate_prevented": True}

        allowed, policy_reason = self.policy_guard.can_send_free_whatsapp_service_message(db, order.customer_phone, now=now)
        if not allowed:
            feedback.policy_blocked_reason = policy_reason
            attempt = existing or NotificationAttempt(
                idempotency_key=idem_key,
                notification_type="customer_review",
                channel="whatsapp",
                recipient_ref=redact_pii(order.customer_phone),
                order_id=order.id,
            )
            if existing is None:
                db.add(attempt)
            attempt.status = "POLICY_BLOCKED"
            attempt.attempts = (attempt.attempts or 0) + 1
            attempt.last_error = policy_reason
            db.flush()
            return {"sent": False, "policy_blocked": True, "reason": policy_reason}

        res = await self.meta_client.send_text_message(order.customer_phone, REVIEW_MESSAGE_AR)
        attempt = existing or NotificationAttempt(
            idempotency_key=idem_key,
            notification_type="customer_review",
            channel="whatsapp",
            recipient_ref=redact_pii(order.customer_phone),
            order_id=order.id,
        )
        if existing is None:
            db.add(attempt)
        attempt.attempts = (attempt.attempts or 0) + 1
        attempt.provider_response = res
        if res.get("sent") or res.get("mode") == "dry_run_unconfigured":
            # Mark sent in live mode; in unconfigured dry-run mark DRY_RUN cleanly so tests/ops distinguish them
            attempt.status = "SENT" if res.get("sent") else "DRY_RUN"
            feedback.sent_at = now
            feedback.policy_blocked_reason = None
        else:
            attempt.status = "FAILED"
            attempt.last_error = res.get("error", "WhatsApp send failed")
        db.flush()
        return {"sent": bool(res.get("sent")), "mode": res.get("mode"), "status": attempt.status}

    async def send_arrival_check(
        self,
        db: Session,
        order: Order,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """90-minute fallback arrival check when delivery is still unconfirmed (policy-gated)."""
        now = now or datetime.now(timezone.utc)
        idem_key = f"arrival_check:{order.id}"
        existing = db.scalar(select(NotificationAttempt).where(NotificationAttempt.idempotency_key == idem_key))
        if existing is not None and existing.status == "SENT":
            return {"sent": True, "duplicate_prevented": True}

        allowed, policy_reason = self.policy_guard.can_send_free_whatsapp_service_message(db, order.customer_phone, now=now)
        if not allowed:
            attempt = existing or NotificationAttempt(
                idempotency_key=idem_key,
                notification_type="arrival_check",
                channel="whatsapp",
                recipient_ref=redact_pii(order.customer_phone),
                order_id=order.id,
            )
            if existing is None:
                db.add(attempt)
            attempt.status = "POLICY_BLOCKED"
            attempt.attempts = (attempt.attempts or 0) + 1
            attempt.last_error = policy_reason
            db.flush()
            return {"sent": False, "policy_blocked": True, "reason": policy_reason}

        res = await self.meta_client.send_text_message(order.customer_phone, ARRIVAL_CHECK_MESSAGE_AR)
        attempt = existing or NotificationAttempt(
            idempotency_key=idem_key,
            notification_type="arrival_check",
            channel="whatsapp",
            recipient_ref=redact_pii(order.customer_phone),
            order_id=order.id,
        )
        if existing is None:
            db.add(attempt)
        attempt.attempts = (attempt.attempts or 0) + 1
        attempt.provider_response = res
        attempt.status = "SENT" if res.get("sent") else ("DRY_RUN" if res.get("mode") == "dry_run_unconfigured" else "FAILED")
        attempt.last_error = res.get("error")
        db.flush()
        return {"sent": bool(res.get("sent")), "mode": res.get("mode"), "status": attempt.status}
