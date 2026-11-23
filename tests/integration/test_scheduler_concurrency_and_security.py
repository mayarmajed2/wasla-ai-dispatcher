"""Phase 9 & 10 Reliability, Concurrency, Lease Locking, Complaint Coverage & Security Tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
from pathlib import Path
import sys

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
import httpx
import pytest
from sqlalchemy import create_engine, inspect, select

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai_service.complaints.detector import assess_complaint
from scripts.verify_postgres_live import run_postgres_verification


def test_all_nine_egyptian_arabic_complaint_scenarios_detected():
    """Verify high-recall detection across all 8 complaint categories + food safety CRITICAL override."""
    cases = [
        ("الأوردر وصل بس ناقص ساندوتش", "missing_item", "HIGH"),
        ("الطلب غلط مش ده اللي طلبته خالص", "wrong_item", "HIGH"),
        ("الأوردر اتأخر ساعة ولسه موصلش", "late_delivery", "HIGH"),
        ("الحساب غلط دفعت اكتر من السعر", "incorrect_billing", "HIGH"),
        ("عايز فلوسي ورجعولي الفلوس ريفند", "refund_request", "HIGH"),
        ("الأكل محروق وطعمه وحش جداً", "food_quality", "HIGH"),
        ("في حشرة في الأكل وبطني وجعتني تسمم", "food_safety", "CRITICAL"),
        ("ايه القرف ده خدمة زبالة اخر مرة اطلب", "aggressive_customer", "HIGH"),
    ]
    for utterance, exp_cat, exp_sev in cases:
        res = assess_complaint(utterance)
        assert res.is_complaint is True, f"Missed complaint: {utterance}"
        assert res.category == exp_cat, f"{utterance}: got {res.category} != {exp_cat}"
        assert res.severity == exp_sev, f"{utterance}: got {res.severity} != {exp_sev}"
        assert res.requires_owner_escalation is True
        assert "فلوسك" not in res.customer_reply_ar  # Never promise unauthorized refund


def test_postgres_live_verifier_strictly_rejects_sqlite():
    """verify_postgres_live.py must NEVER silently fall back to SQLite."""
    with pytest.raises(RuntimeError, match="SQLite is strictly rejected"):
        run_postgres_verification("sqlite:///./restaurant_ai.db")


def test_scheduler_lease_locking_concurrent_workers_cron_tick_and_owner_wa_independence(tmp_path):
    """Tests atomic lease locking, crashed-worker lease recovery, cron-tick auth,
    rich cashier Telegram formatting (sizes + modifiers + notes), and independent owner WA window."""
    isolated_db = tmp_path / "lease_test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{isolated_db.as_posix()}"

    for mod in [
        "app.config",
        "app.database",
        "app.models",
        "app.auth",
        "app.notifications",
        "app.scheduler",
        "app.main",
    ]:
        if mod in sys.modules:
            del sys.modules[mod]

    import app.config as cfg_mod  # noqa: E402
    import app.database as db_mod  # noqa: E402
    import app.main as main_mod  # noqa: E402
    from app.models import ChatSession, ComplaintTicket, Order, OrderStatus, ScheduledJob  # noqa: E402
    from app.scheduler import DurableJobScheduler  # noqa: E402

    main_mod._migrate_sqlite_columns()
    main_mod.seed_moka_menu_if_empty()

    tg_messages: list[str] = []
    wa_messages: list[str] = []

    def tg_handler(req: httpx.Request) -> httpx.Response:
        tg_messages.append(req.read().decode("utf-8"))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(tg_messages)}})

    def wa_handler(req: httpx.Request) -> httpx.Response:
        wa_messages.append(req.read().decode("utf-8"))
        return httpx.Response(200, json={"messages": [{"id": f"wamid.{len(wa_messages)}"}]})

    mock_tg = httpx.AsyncClient(transport=httpx.MockTransport(tg_handler))
    mock_wa = httpx.AsyncClient(transport=httpx.MockTransport(wa_handler))

    cfg_mod.settings.telegram_bot_token = "MOCK_TG_TOKEN"
    cfg_mod.settings.cashier_telegram_chat_id = "-100111"
    cfg_mod.settings.restaurant_owner_telegram_chat_id = "-100222"
    cfg_mod.settings.restaurant_owner_whatsapp = "201099999999"
    cfg_mod.settings.owner_whatsapp_primary = False
    cfg_mod.settings.scheduler_cron_secret = "super-secret-cron-token-123"
    main_mod.telegram_client.bot_token = "MOCK_TG_TOKEN"
    main_mod.telegram_client._external_client = mock_tg
    main_mod.meta_client.access_token = "MOCK_META_TOKEN"
    main_mod.meta_client.phone_number_id = "1515063625018594"
    main_mod.meta_client._external_client = mock_wa

    async def _run():
        now = datetime.now(timezone.utc)
        db = db_mod.SessionLocal()

        # 1. Rich Cashier Telegram formatting: sizes + modifiers + notes + no street address
        order = Order(
            customer_name="منى حسن",
            customer_phone="01011112222",
            backup_phone="01111112222",
            confirmed_by_customer=True,
            delivery_address="شارع التحرير عمارة 7 شقة 4",
            order_type="Delivery",
            subtotal=Decimal("220.00"),
            delivery_fee=Decimal("20.00"),
            total_amount=Decimal("240.00"),
            status=OrderStatus.DELIVERED,
            items_snapshot=[
                {
                    "item_id": 29,
                    "name": "بيتزا مارجريتا (L)",
                    "size_code": "L",
                    "quantity": 1,
                    "modifiers": [{"name": "حبة كاملة", "price": "20.00"}],
                    "notes": "بدون بصل",
                }
            ],
            confirmed_at=now,
            delivered_at=now,
            delivery_confirmed_by="admin",
        )
        db.add(order)
        db.add(ChatSession(phone_number="01011112222", last_customer_message_at=now))
        db.commit()

        msg = main_mod.notification_service.format_cashier_order_message(order)
        assert "بيتزا مارجريتا (L)" in msg
        assert "حبة كاملة" in msg
        assert "بدون بصل" in msg
        assert "240.00 EGP" in msg
        assert "شارع التحرير" not in msg  # Customer home address stays in dashboard only

        # 2. Owner WhatsApp window is strictly independent from customer's active 24h WhatsApp session!
        ticket = ComplaintTicket(
            order_id=order.id,
            customer_phone="01011112222",  # Customer HAS an active 24h WA session
            customer_name="منى حسن",
            category="food_safety",
            severity="CRITICAL",
            summary="Food safety alert",
            sanitized_excerpts=["في شعرة في الأكل"],
            status="OPEN",
            human_takeover_active=True,
        )
        db.add(ticket)
        db.flush()
        esc = await main_mod.notification_service.escalate_complaint_to_owner(db, ticket)
        assert esc["channel"] == "telegram"
        assert len(wa_messages) == 0  # Owner WA MUST NOT borrow customer's 24h window!
        assert len(tg_messages) == 1

        # 3. Concurrent worker lease locking: Worker A claims due job; Worker B gets 0!
        worker_a = DurableJobScheduler(main_mod.notification_service, lease_seconds=120, worker_id="worker-A")
        worker_b = DurableJobScheduler(main_mod.notification_service, lease_seconds=120, worker_id="worker-B")

        job = worker_a.schedule_job(
            db,
            idempotency_key=f"lease_test:{order.id}",
            job_type="cashier_order_notification",
            run_at=now - timedelta(minutes=1),
            order_id=order.id,
        )
        db.commit()

        claimed_a = worker_a.claim_due_jobs(db, now=now)
        assert len(claimed_a) == 1
        assert claimed_a[0].locked_by == "worker-A"

        # Worker B tries to claim during active lease -> MUST get 0 jobs!
        claimed_b = worker_b.claim_due_jobs(db, now=now + timedelta(seconds=30))
        assert len(claimed_b) == 0

        # Simulate Worker A crashing: lease expires at now + 121s -> Worker B reclaims and executes!
        res_b = await worker_b.process_due_jobs(db, now=now + timedelta(seconds=130))
        assert len(res_b) == 1
        assert res_b[0]["worker_id"] == "worker-B"
        db.refresh(job)
        assert job.status == "COMPLETED"
        assert job.locked_by is None
        db.close()

        # 4. External Cron-Tick Endpoint authentication & missed-job execution
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.post("/api/scheduler/cron-tick")).status_code == 401
            assert (
                await client.post("/api/scheduler/cron-tick", headers={"X-Scheduler-Secret": "wrong"})
            ).status_code == 401
            ok_tick = await client.post(
                "/api/scheduler/cron-tick",
                headers={"X-Scheduler-Secret": "super-secret-cron-token-123"},
            )
            assert ok_tick.status_code == 200
            assert ok_tick.json()["ok"] is True

        await mock_tg.aclose()
        await mock_wa.aclose()

    asyncio.run(_run())


def test_alembic_0001_to_0003_includes_scheduler_lease_columns(tmp_path):
    """Verify Alembic migrations 0001 -> 0002 -> 0003 upgrade & downgrade cleanly."""
    mig_db = tmp_path / "alembic_0003.db"
    db_url = f"sqlite:///{mig_db.as_posix()}"
    os.environ["DATABASE_URL"] = db_url
    for mod in ["app.config", "app.database"]:
        if mod in sys.modules:
            del sys.modules[mod]

    cfg = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)

    alembic_command.upgrade(cfg, "head")
    eng = create_engine(db_url)
    insp = inspect(eng)
    cols = {c["name"] for c in insp.get_columns("scheduled_jobs")}
    eng.dispose()
    assert "locked_by" in cols
    assert "locked_until" in cols

    alembic_command.downgrade(cfg, "base")
    alembic_command.upgrade(cfg, "head")
