"""Official Telegram Bot API Setup, Discovery & Live Delivery Verification CLI.

Usage:
    python scripts/setup_telegram_verify.py              # Validate bot token & discover authorized chat IDs
    python scripts/setup_telegram_verify.py --send-test  # Also send real authorized test messages to cashier & owner chats

Never prints or stores raw `TELEGRAM_BOT_TOKEN`. Reads strictly from environment / local `.env`.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.notifications import TelegramBotClient  # noqa: E402


async def run_verification(send_test: bool, output_path: Path | None) -> dict:
    settings = Settings()
    client = TelegramBotClient(bot_token=settings.telegram_bot_token)
    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "token_configured": client.is_configured,
        "cashier_chat_configured": bool(settings.cashier_telegram_chat_id.strip()),
        "owner_chat_configured": bool(settings.restaurant_owner_telegram_chat_id.strip()),
        "bot_identity": None,
        "discovered_chats": [],
        "cashier_permission_check": None,
        "owner_permission_check": None,
        "live_test_results": {},
        "status": "BLOCKED_BY_CREDENTIALS",
    }

    if not client.is_configured:
        report["next_step"] = (
            "Set TELEGRAM_BOT_TOKEN, CASHIER_TELEGRAM_CHAT_ID, and RESTAURANT_OWNER_TELEGRAM_CHAT_ID in local .env "
            "(never paste tokens in chat or commit .env)."
        )
        return report

    # 1. Validate Bot Token via getMe
    me = await client.get_me()
    if not me.get("ok"):
        report["status"] = "INVALID_BOT_TOKEN"
        report["error"] = me.get("description", "getMe failed")
        return report

    bot_user = me.get("result", {})
    bot_id = bot_user.get("id")
    report["bot_identity"] = {
        "id": bot_id,
        "username": bot_user.get("username"),
        "first_name": bot_user.get("first_name"),
        "can_join_groups": bot_user.get("can_join_groups"),
    }

    # 2. Discover authorized chats via getUpdates (helps owner find group/private chat IDs)
    updates = await client.get_updates(limit=25)
    seen_chats: dict[str, dict] = {}
    for upd in updates.get("result", []):
        msg = upd.get("message") or upd.get("my_chat_member") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            cid = str(chat["id"])
            seen_chats[cid] = {
                "chat_id": cid,
                "type": chat.get("type"),
                "title": chat.get("title") or chat.get("username") or "private_chat",
            }
    report["discovered_chats"] = list(seen_chats.values())

    # 3. Verify bot membership & permissions in configured chats
    if settings.cashier_telegram_chat_id.strip() and bot_id:
        try:
            member = await client.get_chat_member(settings.cashier_telegram_chat_id.strip(), bot_id)
            res = member.get("result", {})
            report["cashier_permission_check"] = {
                "ok": bool(member.get("ok")),
                "member_status": res.get("status"),
                "can_send_messages": res.get("can_send_messages", True),
            }
        except Exception as exc:
            report["cashier_permission_check"] = {"ok": False, "error": str(exc)}

    if settings.restaurant_owner_telegram_chat_id.strip() and bot_id:
        try:
            member = await client.get_chat_member(settings.restaurant_owner_telegram_chat_id.strip(), bot_id)
            res = member.get("result", {})
            report["owner_permission_check"] = {"ok": bool(member.get("ok")), "member_status": res.get("status")}
        except Exception as exc:
            report["owner_permission_check"] = {"ok": False, "error": str(exc)}

    # 4. Optional authorized live test delivery
    if send_test:
        if settings.cashier_telegram_chat_id.strip():
            cashier_msg = (
                "NEW MOKA ORDER (LIVE VERIFICATION TEST)\n\n"
                "Order: #TEST-VERIFY\n"
                "Items:\n- 1x بيتزا مارجريتا (M) + (حبة كاملة)\n"
                "Total: 145.00 EGP\n"
                "Status: CONFIRMED\n"
                f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            res_c = await client.send_message(settings.cashier_telegram_chat_id.strip(), cashier_msg)
            report["live_test_results"]["cashier_group"] = {
                "sent": bool(res_c.get("sent")),
                "message_id": (res_c.get("telegram_response") or {}).get("result", {}).get("message_id"),
                "error": res_c.get("error"),
            }
        if settings.restaurant_owner_telegram_chat_id.strip():
            owner_msg = (
                "URGENT CUSTOMER COMPLAINT (LIVE VERIFICATION TEST)\n\n"
                "Ticket: #TEST-VERIFY\n"
                "Issue: verification_check\n"
                "Severity: HIGH\n"
                "Status: Owner Telegram escalation channel verified."
            )
            res_o = await client.send_message(settings.restaurant_owner_telegram_chat_id.strip(), owner_msg)
            report["live_test_results"]["owner_private"] = {
                "sent": bool(res_o.get("sent")),
                "message_id": (res_o.get("telegram_response") or {}).get("result", {}).get("message_id"),
                "error": res_o.get("error"),
            }

        c_ok = report["live_test_results"].get("cashier_group", {}).get("sent", False)
        o_ok = report["live_test_results"].get("owner_private", {}).get("sent", False)
        report["status"] = "LIVE_AND_VERIFIED" if (c_ok and o_ok) else "PARTIALLY_VERIFIED"
    else:
        report["status"] = "TOKEN_VALIDATED_DISCOVERY_COMPLETE"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


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
