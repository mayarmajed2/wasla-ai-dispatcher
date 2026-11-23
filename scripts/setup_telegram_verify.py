"""Official Telegram Bot API Diagnostics, Chat Discovery & Live Delivery CLI.

Usage:
    python scripts/setup_telegram_verify.py              # Single-shot getMe -> getWebhookInfo -> controlled getUpdates
    python scripts/setup_telegram_verify.py --send-test  # Send authorized live alerts to cashier group & owner private chat

Security & Rate-Limit Discipline:
- NEVER prints, logs, or stores raw `TELEGRAM_BOT_TOKEN` or full request URLs.
- Executes ONE controlled `getMe` call first and STOPS immediately on HTTP 429 or 401.
- Checks `getWebhookInfo` BEFORE deciding whether to poll `getUpdates` (never calls `getUpdates` if webhook is active).
- Persists update offsets locally (`.telegram_offset.json`, gitignored) so updates are not reprocessed.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.notifications import TelegramBotClient  # noqa: E402

DEFAULT_OFFSET_FILE = ROOT / ".telegram_offset.json"


def _write_report(output_path: Path | None, report: dict[str, Any]) -> dict[str, Any]:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_offset(offset_file: Path | None) -> int | None:
    if offset_file is None or not offset_file.exists():
        return None
    try:
        data = json.loads(offset_file.read_text(encoding="utf-8"))
        val = data.get("next_offset")
        return int(val) if val is not None else None
    except Exception:
        return None


def _save_offset(offset_file: Path | None, next_offset: int) -> None:
    if offset_file is None:
        return
    try:
        offset_file.write_text(json.dumps({"next_offset": next_offset}, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_trace(
    api_trace: list[dict[str, Any]],
    attempt_counts: dict[str, int],
    method: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    attempt_counts[method] = attempt_counts.get(method, 0) + 1
    meta = response.get("_meta") or {}
    entry = {
        "attempt_number": attempt_counts[method],
        "method": method,
        "http_status": meta.get("http_status", response.get("http_status") or response.get("status_code")),
        "timestamp_utc": meta.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
        "error_description": meta.get("error_description") or (None if response.get("ok") or response.get("sent") else response.get("description") or response.get("error")),
        "retry_after": meta.get("retry_after", response.get("retry_after")),
        "server_header": meta.get("server_header"),
    }
    api_trace.append(entry)
    return entry


async def run_verification(
    send_test: bool = False,
    output_path: Path | None = None,
    *,
    settings: Settings | None = None,
    client: TelegramBotClient | None = None,
    offset_file: Path | None = DEFAULT_OFFSET_FILE,
) -> dict[str, Any]:
    cfg = settings or Settings()
    tg = client or TelegramBotClient(bot_token=cfg.telegram_bot_token)
    attempt_counts: dict[str, int] = {}
    api_trace: list[dict[str, Any]] = []

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "token_configured": tg.is_configured,
        "cashier_chat_configured": bool(cfg.cashier_telegram_chat_id.strip()),
        "owner_chat_configured": bool(cfg.restaurant_owner_telegram_chat_id.strip()),
        "bot_identity": None,
        "webhook_info": None,
        "update_delivery_mode": None,
        "discovered_groups": [],
        "discovered_private_chats": [],
        "discovered_chats": [],
        "cashier_permission_check": None,
        "owner_permission_check": None,
        "live_test_results": {},
        "api_trace": api_trace,
        "status": "BLOCKED_BY_CREDENTIALS",
        "http_status": None,
        "retry_after": None,
        "error": None,
    }

    if not tg.is_configured:
        report["error"] = "TELEGRAM_BOT_TOKEN is not set in local .env."
        return _write_report(output_path, report)

    # Step 1: Strictly ONE controlled getMe call (STOP immediately on 429 or 401!)
    me = await tg.get_me()
    me_trace = _append_trace(api_trace, attempt_counts, "getMe", me)
    report["http_status"] = me_trace["http_status"]
    report["retry_after"] = me_trace["retry_after"]

    if not me.get("ok"):
        sc = me_trace["http_status"]
        report["error"] = me_trace["error_description"] or "getMe failed"
        if sc in (401, 404):
            report["status"] = "INVALID_BOT_TOKEN"
        elif sc == 429:
            report["status"] = "RATE_LIMITED"
            report["root_cause_notes"] = (
                "Telegram returned HTTP 429 directly on the single getMe authentication request with zero retries "
                "and zero local polling processes active. Remaining possibilities: remote consumer/webhook using "
                "this bot token, persistent flood penalty on the bot ID, or shared ISP/CGNAT IP throttling."
            )
        else:
            report["status"] = "TELEGRAM_API_ERROR"
        return _write_report(output_path, report)

    bot_user = me.get("result", {})
    bot_id = bot_user.get("id")
    report["bot_identity"] = {
        "id": bot_id,
        "username": bot_user.get("username"),
        "first_name": bot_user.get("first_name"),
        "can_join_groups": bot_user.get("can_join_groups"),
    }

    # Step 2: Inspect getWebhookInfo BEFORE touching getUpdates (mutually exclusive!)
    wh = await tg.get_webhook_info()
    wh_trace = _append_trace(api_trace, attempt_counts, "getWebhookInfo", wh)
    if wh.get("ok"):
        wh_res = wh.get("result") or {}
        webhook_url = (wh_res.get("url") or "").strip()
        report["webhook_info"] = {
            "active": bool(webhook_url),
            "url": webhook_url,
            "has_custom_certificate": wh_res.get("has_custom_certificate"),
            "pending_update_count": wh_res.get("pending_update_count", 0),
            "last_error_message": wh_res.get("last_error_message"),
        }
    else:
        report["webhook_info"] = {"error": wh_trace["error_description"], "http_status": wh_trace["http_status"]}
        webhook_url = ""

    # Step 3: Capture updates via single getUpdates consumer ONLY if no webhook is configured
    seen_chats: dict[str, dict[str, Any]] = {}
    if webhook_url:
        report["update_delivery_mode"] = "webhook_active"
    else:
        report["update_delivery_mode"] = "polling_getUpdates"
        saved_offset = _load_offset(offset_file)
        updates = await tg.get_updates(limit=25, offset=saved_offset)
        up_trace = _append_trace(api_trace, attempt_counts, "getUpdates", updates)
        if not updates.get("ok"):
            report["http_status"] = up_trace["http_status"]
            report["retry_after"] = up_trace["retry_after"]
            report["error"] = up_trace["error_description"]
            report["status"] = "RATE_LIMITED" if up_trace["http_status"] == 429 else "TELEGRAM_API_ERROR"
            return _write_report(output_path, report)

        max_update_id: int | None = None
        for upd in updates.get("result", []):
            uid = upd.get("update_id")
            if isinstance(uid, int):
                max_update_id = uid if max_update_id is None else max(max_update_id, uid)
            msg = upd.get("message") or upd.get("my_chat_member") or upd.get("channel_post") or {}
            chat = msg.get("chat") or {}
            if chat.get("id") is not None:
                cid = str(chat["id"])
                ctype = chat.get("type", "unknown")
                seen_chats[cid] = {
                    "chat_id": cid,
                    "type": ctype,
                    "title": chat.get("title") or chat.get("username") or chat.get("first_name") or "chat",
                }
        if max_update_id is not None:
            _save_offset(offset_file, max_update_id + 1)

    report["discovered_chats"] = list(seen_chats.values())
    report["discovered_groups"] = [c for c in report["discovered_chats"] if c["type"] in ("group", "supergroup")]
    report["discovered_private_chats"] = [c for c in report["discovered_chats"] if c["type"] == "private"]

    # Step 4: Verify bot membership & send permissions for cashier group (configured or Wasla-AI) & owner chat
    cashier_cid = cfg.cashier_telegram_chat_id.strip()
    if not cashier_cid:
        for grp in report["discovered_groups"]:
            if "wasla" in (grp.get("title") or "").lower():
                cashier_cid = grp["chat_id"]
                break
        if not cashier_cid and len(report["discovered_groups"]) == 1:
            cashier_cid = report["discovered_groups"][0]["chat_id"]

    if cashier_cid and bot_id:
        member = await tg.get_chat_member(cashier_cid, bot_id)
        m_trace = _append_trace(api_trace, attempt_counts, "getChatMember", member)
        res = member.get("result", {})
        report["cashier_permission_check"] = {
            "chat_id": cashier_cid,
            "ok": bool(member.get("ok")),
            "member_status": res.get("status"),
            "can_send_messages": res.get("can_send_messages", res.get("status") in ("administrator", "creator", "member")),
            "error": m_trace["error_description"],
        }

    owner_cid = cfg.restaurant_owner_telegram_chat_id.strip()
    if owner_cid and cashier_cid and owner_cid == cashier_cid:
        report["owner_permission_check"] = {"ok": False, "error": "Owner chat ID must NOT equal cashier group ID"}
    elif owner_cid and bot_id:
        member_o = await tg.get_chat_member(owner_cid, bot_id)
        o_trace = _append_trace(api_trace, attempt_counts, "getChatMember", member_o)
        res_o = member_o.get("result", {})
        report["owner_permission_check"] = {
            "chat_id": owner_cid,
            "ok": bool(member_o.get("ok")),
            "member_status": res_o.get("status"),
            "error": o_trace["error_description"],
        }

    # Step 5: Authorized live delivery verification (only when explicitly requested)
    if send_test:
        if cfg.cashier_telegram_chat_id.strip():
            cashier_msg = (
                "NEW MOKA ORDER (LIVE VERIFICATION TEST)\n\n"
                "Order: #TEST-VERIFY\n"
                "Items:\n- 1x بيتزا مارجريتا (M) + (حبة كاملة)\n"
                "Total: 145.00 EGP\n"
                "Status: CONFIRMED\n"
                f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            res_c = await tg.send_message(cfg.cashier_telegram_chat_id.strip(), cashier_msg, max_retries=0)
            _append_trace(api_trace, attempt_counts, "sendMessage", res_c)
            report["live_test_results"]["cashier_group"] = {
                "sent": bool(res_c.get("sent")),
                "http_status": res_c.get("http_status"),
                "message_id": (res_c.get("telegram_response") or {}).get("result", {}).get("message_id"),
                "error": res_c.get("error"),
            }
        if owner_cid and owner_cid != cfg.cashier_telegram_chat_id.strip():
            owner_msg = (
                "URGENT CUSTOMER COMPLAINT (LIVE VERIFICATION TEST)\n\n"
                "Ticket: #TEST-VERIFY\n"
                "Issue: verification_check\n"
                "Severity: HIGH\n"
                "Status: Owner private Telegram escalation channel verified."
            )
            res_o = await tg.send_message(owner_cid, owner_msg, max_retries=0)
            _append_trace(api_trace, attempt_counts, "sendMessage", res_o)
            report["live_test_results"]["owner_private"] = {
                "sent": bool(res_o.get("sent")),
                "http_status": res_o.get("http_status"),
                "message_id": (res_o.get("telegram_response") or {}).get("result", {}).get("message_id"),
                "error": res_o.get("error"),
            }

        c_ok = report["live_test_results"].get("cashier_group", {}).get("sent", False)
        o_ok = report["live_test_results"].get("owner_private", {}).get("sent", False)
        report["status"] = "LIVE_AND_VERIFIED" if (c_ok and o_ok) else "PARTIALLY_VERIFIED"
    else:
        report["status"] = "TOKEN_VALIDATED_DISCOVERY_COMPLETE"

    return _write_report(output_path, report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Telegram Bot setup and verify live cashier/owner delivery.")
    parser.add_argument("--send-test", action="store_true", help="Send authorized live test alerts to cashier & owner chats.")
    parser.add_argument("--output", default="", help="Optional JSON path to save sanitized verification report.")
    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None
    result = asyncio.run(run_verification(send_test=args.send_test, output_path=out_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("LIVE_AND_VERIFIED", "TOKEN_VALIDATED_DISCOVERY_COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
