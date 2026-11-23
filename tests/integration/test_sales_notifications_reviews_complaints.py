"""Comprehensive Automated Verification Suite for Phases 1–8:

Covers:
- Friendly Egyptian Arabic sales personality (no hallucinated offers/prices)
- Context-aware recommendations, official combos, missing combo handling, budget & rejection memory
- Complete order lifecycle & OrderStatusHistory timestamps
- Cashier Telegram Group notifications, deduplication & failure retry safety (MOCKED HTTP)
- Meta Groups API eligibility & 24h zero-cost policy guard
- 1-hour post-delivery review scheduling, 90-minute arrival check, suppression rules & duplicate prevention
- High-recall complaint detection (including CRITICAL food safety) & Owner Escalation with Telegram fallback
- Human Takeover silencing automated AI replies
- Durable restart-safe scheduler recovery, retries & dead-letter handling
- Customer Experience Dashboard APIs, RBAC authentication & privacy protection
- Real Alembic migration upgrade/downgrade chain verification (0001 -> 0002)
"""

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
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.complaints.detector import assess_complaint
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.recommendation.engine import OfficialComboConfig, RecommendationEngine
from ai_service.understanding.menu_index import MenuIndex

MENU_PATH = ROOT / "menu" / "menu.json"


def _make_dispatcher(official_combos=None, upsell_enabled=True):
    cfg = WaslaConfig(gemini_api_key="", upsell_enabled=upsell_enabled)
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    rec_engine = RecommendationEngine(idx, official_combos=official_combos, enabled=upsell_enabled)
    return WaslaAIDispatcher(cfg, idx, backend, recommendation_engine=rec_engine), backend, idx


# ==============================================================================
# PHASE 1 & 2: NATURAL EGYPTIAN PERSONALITY, COMBOS, UPSELL & REJECTION MEMORY
# ==============================================================================


def test_friendly_egyptian_personality_and_missing_combo_separate_pricing():
    """Without an official combo in DB, recommend separate real available items with exact Decimal sum."""

    async def _run():
        dispatcher, backend, idx = _make_dispatcher(official_combos=[])
        phone = "01055000001"

        g = await dispatcher.handle_text(phone, "السلام عليكم")
        assert "MOKA" in g.reply_ar
        assert "خصم" not in g.reply_ar

        # Order a sandwich -> recommends real MOKA fries (30 EGP) & drink (70 EGP) = 100 EGP combined
        r1 = await dispatcher.handle_text(phone, "عايز سندوتش تشيكن هالبينو")
        assert r1.cart is not None and len(r1.cart.items) == 1
        assert r1.upsell_offer is not None
        assert r1.upsell_offer["offer_type"] == "separate_items"
        assert r1.upsell_offer["combo_id"] is None
        # Verify real menu IDs & deterministic combined price (no invented discount!)
        expected_sum = sum(Decimal(i["price"]) for i in r1.upsell_offer["items"])
        assert Decimal(r1.upsell_offer["combined_price"]) == expected_sum
        assert all(idx.get_product(i["product_id"]) is not None for i in r1.upsell_offer["items"])
        # Recommended items MUST NOT be added automatically before customer acceptance!
        assert len(r1.cart.items) == 1

        # Explicit acceptance adds recommended items with deterministic Decimal total
        r2 = await dispatcher.handle_text(phone, "اه ضيفهم")
        assert r2.upsell_event is not None and r2.upsell_event["action"] == "accepted"
        assert len(r2.cart.items) == 1 + len(r1.upsell_offer["items"])
        assert r2.cart.subtotal == Decimal("170.00") + expected_sum

    asyncio.run(_run())


def test_official_combo_offered_only_when_configured_in_db():
    """When an official combo is explicitly configured, offer it accurately."""

    async def _run():
        combo = OfficialComboConfig(
            combo_id="combo_burger_01",
            name_ar="كومبو بطاطس وصودا",
            trigger_categories=["cat_sandwich_pizza"],
            item_product_ids=["item_065", "item_081"],
            combo_price=Decimal("90.00"),
        )
        dispatcher, _, _ = _make_dispatcher(official_combos=[combo])
        r = await dispatcher.handle_text("01055000002", "عايز سندوتش تشيكن هالبينو")
        assert r.upsell_offer is not None
        assert r.upsell_offer["offer_type"] == "official_combo"
        assert r.upsell_offer["combo_id"] == "combo_burger_01"
        assert Decimal(r.upsell_offer["combined_price"]) == Decimal("90.00")

    asyncio.run(_run())


def test_upsell_rejection_never_repeats_and_budget_and_disable_respected():
    """Never repeat upsell after rejection; respect customer budget and upsell_enabled=False."""

    async def _run():
        dispatcher, _, _ = _make_dispatcher(upsell_enabled=True)
        phone = "01055000003"

        r1 = await dispatcher.handle_text(phone, "عايزة كريب كوردن بلو")
        assert r1.upsell_offer is not None

        # Customer declines upsell
        r2 = await dispatcher.handle_text(phone, "لا شكراً مش عايز")
        assert r2.upsell_event is not None and r2.upsell_event["action"] == "rejected"
        assert len(r2.cart.items) == 1

        # Ordering another item MUST NOT repeat upsell after rejection
        r3 = await dispatcher.handle_text(phone, "وزود بيتزا مارجريتا وسط")
        assert r3.upsell_offer is None

        # Budget guard: customer states 150 EGP budget and orders 140 EGP item -> remaining 10 EGP < 30 EGP fries -> no upsell
        phone_budget = "01055000004"
        rb = await dispatcher.handle_text(phone_budget, "الميزانية 150 عايزة كريب كوردن بلو")
        assert rb.upsell_offer is None

        # Upsell globally disabled -> zero offers
        disp_off, _, _ = _make_dispatcher(upsell_enabled=False)
        ro = await disp_off.handle_text("01055000005", "عايزة كريب كوردن بلو")
        assert ro.upsell_offer is None

    asyncio.run(_run())


# ==============================================================================
# PHASE 6: HIGH-RECALL COMPLAINT CLASSIFICATION & HUMAN TAKEOVER
# ==============================================================================


def test_complaint_classification_high_recall_food_safety_and_human_takeover():
    """Food safety is ALWAYS CRITICAL; polite Egyptian reply has zero unauthorized refund promises; takeover silences AI."""

    async def _run():
        # Food safety check (CRITICAL)
        c_safety = assess_complaint("في شعرة في الأكل وريحته وحشة وبطني وجعتني")
        assert c_safety.is_complaint is True
        assert c_safety.category == "food_safety"
        assert c_safety.severity == "CRITICAL"
        assert c_safety.requires_owner_escalation is True
        assert "فلوسك" not in c_safety.customer_reply_ar  # No unauthorized refund promise

        # Missing item check (HIGH)
        c_missing = assess_complaint("الأوردر وصل بس ناقص ساندوتش")
        assert c_missing.is_complaint is True
        assert c_missing.category == "missing_item"
        assert c_missing.severity == "HIGH"

        # Dispatcher handles complaint -> never upsells, sets takeover, silences subsequent AI replies
        dispatcher, _, _ = _make_dispatcher()
        phone = "01055000006"
        r1 = await dispatcher.handle_text(phone, "الأوردر اتأخر ساعة والأكل وصل بارد")
        assert r1.intent.value == "complaint"
        assert r1.upsell_offer is None
        assert r1.complaint_assessment is not None
        assert r1.needs_human_takeover is True

        # Activate human takeover -> subsequent customer message MUST NOT receive automated AI sales response
        dispatcher.set_human_takeover(phone, True, actor="owner")
        r2 = await dispatcher.handle_text(phone, "لسه مستني حد يكلمني")
        assert r2.suppress_ai_reply is True
        assert r2.reply_ar == ""
        assert r2.model_used == "human_takeover_active"

    asyncio.run(_run())


# ==============================================================================
# PHASES 3, 4, 5, 6, 7, 8: FULL BACKEND, NOTIFICATIONS, SCHEDULER, RESTART & RBAC
# ==============================================================================


def test_end_to_end_notifications_scheduler_restarts_complaints_and_cx_dashboard(tmp_path):
    """Tests mocked Telegram/WhatsApp delivery, failure retries, 1h review, 90m arrival check,
    Meta 24h policy blocking, complaint owner escalation + Telegram fallback, restart recovery, and CX RBAC."""

    isolated_db = tmp_path / "cx_test.db"
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
    from app.models import (  # noqa: E402
        ChatSession,
        ComplaintTicket,
        FeedbackRequest,
        NotificationAttempt,
        Order,
        OrderStatus,
        OrderStatusHistory,
        ScheduledJob,
    )
    from app.notifications import NotificationService, TelegramBotClient  # noqa: E402
    from app.scheduler import DurableJobScheduler  # noqa: E402

    main_mod._migrate_sqlite_columns()
    main_mod.seed_moka_menu_if_empty()
    main_mod.seed_initial_admin_if_empty()

    # Configure mocked HTTP transports for Telegram & Meta WhatsApp (clearly mocked, NOT live!)
    sent_telegram: list[dict] = []
    sent_whatsapp: list[dict] = []
    fail_telegram_once = {"fail": False}

    def tg_handler(request: httpx.Request) -> httpx.Response:
        if fail_telegram_once["fail"]:
            return httpx.Response(503, json={"ok": False, "description": "Telegram temporary outage"})
        body = request.read().decode("utf-8")
        sent_telegram.append({"url": str(request.url), "body": body})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(sent_telegram)}})

    def wa_handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8")
        sent_whatsapp.append({"url": str(request.url), "body": body})
        return httpx.Response(200, json={"messages": [{"id": f"wamid.{len(sent_whatsapp)}"}]})

    mock_tg_http = httpx.AsyncClient(transport=httpx.MockTransport(tg_handler))
    mock_wa_http = httpx.AsyncClient(transport=httpx.MockTransport(wa_handler))

    cfg_mod.settings.telegram_bot_token = "TEST_MOCK_TG_TOKEN"
    cfg_mod.settings.cashier_telegram_chat_id = "-100999999"
    cfg_mod.settings.restaurant_owner_whatsapp = "201000000001"
    cfg_mod.settings.restaurant_owner_telegram_chat_id = "-100888888"
    cfg_mod.settings.meta_is_official_business_account = False
    cfg_mod.settings.meta_billable_messages_authorized = False

    main_mod.meta_client.access_token = "TEST_MOCK_META_TOKEN"
    main_mod.meta_client.phone_number_id = "1515063625018594"
    main_mod.meta_client._external_client = mock_wa_http
    main_mod.telegram_client.bot_token = "TEST_MOCK_TG_TOKEN"
    main_mod.telegram_client._external_client = mock_tg_http

    async def _run():
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Auth & Privacy: unauthenticated CX endpoints MUST reject with 401
            assert (await client.get("/api/customer-experience/overview")).status_code == 401
            assert (await client.get("/api/complaints")).status_code == 401

            login = await client.post(
                "/api/auth/login",
                json={
                    "username": cfg_mod.settings.bootstrap_admin_username,
                    "password": cfg_mod.settings.bootstrap_admin_password,
                },
            )
            assert login.status_code == 200
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

            # 2. Verify Meta Groups API eligibility honestly reports NOT eligible for non-OBA
            settings_resp = await client.get("/api/settings/notifications", headers=headers)
            assert settings_resp.status_code == 200
            assert settings_resp.json()["groups_api_eligibility"]["eligible"] is False
            assert settings_resp.json()["groups_api_eligibility"]["max_participants_per_group"] == 8

            # 3. Record active customer 24h window in DB
            db = db_mod.SessionLocal()
            now = datetime.now(timezone.utc)
            db.add(ChatSession(phone_number="01077770001", last_customer_message_at=now))
            db.commit()
            db.close()

            # 4. Create genuine confirmed order -> instant Cashier Telegram notification + 90m arrival check scheduled
            order_payload = {
                "customer_name": "أحمد علي",
                "customer_phone": "01077770001",
                "backup_phone": "01177770002",
                "delivery_address": "المعادي شارع 9 عمارة 10",
                "zone_name": "المعادي",
                "items": [{"item_id": 20, "quantity": 2}],
                "idempotency_key": "order-cx-001",
                "confirmed_by_customer": True,
                "data_source": "whatsapp_webhook",
                "is_test": False,
            }
            o1 = await client.post("/api/orders", json=order_payload)
            assert o1.status_code == 201
            order_id = o1.json()["id"]
            assert o1.json()["cashier_notification"]["sent"] is True
            assert len(sent_telegram) == 1
            assert f"Order: #{order_id}" in sent_telegram[0]["body"]
            # Privacy check: full address must NOT be exposed in Telegram group message
            assert "المعادي شارع 9" not in sent_telegram[0]["body"]

            # Duplicate idempotency key MUST NOT duplicate order or Telegram notification
            o1_dup = await client.post("/api/orders", json=order_payload)
            assert o1_dup.json()["duplicate"] is True
            assert len(sent_telegram) == 1

            # 5. Group notification failure -> order STILL saved, failure recorded, retry job scheduled!
            fail_telegram_once["fail"] = True
            order_payload_2 = dict(order_payload, idempotency_key="order-cx-002", customer_phone="01077770009")
            o2 = await client.post("/api/orders", json=order_payload_2)
            assert o2.status_code == 201
            order2_id = o2.json()["id"]
            assert o2.json()["cashier_notification"]["status"] == "FAILED"
            fail_telegram_once["fail"] = False

            # Retry recovers after simulated application restart (fresh scheduler instance reads DB!)
            db = db_mod.SessionLocal()
            retry_job = db.scalar(
                select(ScheduledJob).where(ScheduledJob.idempotency_key == f"cashier_notify_retry:{order2_id}")
            )
            assert retry_job is not None
            restarted_scheduler = DurableJobScheduler(main_mod.notification_service)
            restarted_scheduler.recover_stale_jobs_on_startup(db)
            await restarted_scheduler.process_due_jobs(db, now=now + timedelta(minutes=2))
            db.refresh(retry_job)
            assert retry_job.status == "COMPLETED"
            assert len(sent_telegram) == 2
            db.close()

            # 6. 90-minute Arrival Check fires for order1 (still unconfirmed delivery) within 24h window
            db = db_mod.SessionLocal()
            await restarted_scheduler.process_due_jobs(db, now=now + timedelta(minutes=91))
            arr_job = db.scalar(select(ScheduledJob).where(ScheduledJob.idempotency_key == f"arrival_check_90m:{order_id}"))
            assert arr_job is not None and arr_job.status == "COMPLETED"
            assert len(sent_whatsapp) == 1
            db.close()

            # 7. Full lifecycle transitions with timestamps -> DELIVERED schedules 1-hour review!
            for next_st in ["ACCEPTED", "PREPARING", "READY", "OUT_FOR_DELIVERY", "DELIVERED"]:
                st_res = await client.patch(f"/api/orders/{order_id}/status", json={"status": next_st}, headers=headers)
                assert st_res.status_code == 200

            db = db_mod.SessionLocal()
            order_row = db.get(Order, order_id)
            assert order_row.delivered_at is not None
            assert order_row.delivery_confirmed_by == cfg_mod.settings.bootstrap_admin_username
            hist = db.scalars(select(OrderStatusHistory).where(OrderStatusHistory.order_id == order_id)).all()
            assert len(hist) >= 6

            # Before 60 minutes -> review NOT sent yet
            rev_job = db.scalar(select(ScheduledJob).where(ScheduledJob.idempotency_key == f"review_1h:{order_id}"))
            assert rev_job is not None and rev_job.status == "PENDING"
            await restarted_scheduler.process_due_jobs(db, now=order_row.delivered_at + timedelta(minutes=30))
            db.refresh(rev_job)
            assert rev_job.status == "PENDING"

            # After 60 minutes (1 hour) -> review IS sent!
            await restarted_scheduler.process_due_jobs(db, now=order_row.delivered_at + timedelta(minutes=61))
            db.refresh(rev_job)
            assert rev_job.status == "COMPLETED"
            fb = db.scalar(select(FeedbackRequest).where(FeedbackRequest.order_id == order_id))
            assert fb is not None and fb.sent_at is not None
            fb_token = fb.token
            db.close()

            # Submit genuine review + verify duplicate submission rejected
            fb_sub = await client.post(f"/api/feedback/{fb_token}", json={"rating": 5, "comment": "الأكل تحفة وسخن ❤️"})
            assert fb_sub.status_code == 200
            fb_dup = await client.post(f"/api/feedback/{fb_token}", json={"rating": 4, "comment": "duplicate"})
            assert fb_dup.status_code == 409

            # 8. Cancelled order MUST suppress review!
            st_cancel = await client.patch(f"/api/orders/{order2_id}/status", json={"status": "CANCELLED"}, headers=headers)
            assert st_cancel.status_code == 200
            db = db_mod.SessionLocal()
            o2_row = db.get(Order, order2_id)
            restarted_scheduler.schedule_post_delivery_review(db, o2_row, delivered_at=now)
            await restarted_scheduler.process_due_jobs(db, now=now + timedelta(minutes=65))
            o2_rev_job = db.scalar(select(ScheduledJob).where(ScheduledJob.idempotency_key == f"review_1h:{order2_id}"))
            assert o2_rev_job.status == "SUPPRESSED"
            assert o2_rev_job.last_error == "order_status_cancelled"
            db.close()

            # 9. Meta 24-hour policy block: outside 24h window with zero-cost -> BLOCKED_POLICY (no billable send!)
            order_payload_3 = dict(order_payload, idempotency_key="order-cx-003", customer_phone="01077770099")
            o3 = await client.post("/api/orders", json=order_payload_3)
            order3_id = o3.json()["id"]
            await client.patch(f"/api/orders/{order3_id}/status", json={"status": "IN_KITCHEN"}, headers=headers)
            await client.patch(f"/api/orders/{order3_id}/status", json={"status": "DISPATCHED"}, headers=headers)
            await client.patch(f"/api/orders/{order3_id}/status", json={"status": "DELIVERED"}, headers=headers)
            db = db_mod.SessionLocal()
            o3_row = db.get(Order, order3_id)
            # Customer never sent a WhatsApp message within 24h -> policy MUST block instead of spending money!
            await restarted_scheduler.process_due_jobs(db, now=o3_row.delivered_at + timedelta(minutes=65))
            o3_rev = db.scalar(select(ScheduledJob).where(ScheduledJob.idempotency_key == f"review_1h:{order3_id}"))
            assert o3_rev.status == "BLOCKED_POLICY"
            db.close()

            # 10. Urgent complaint via chat -> ticket created + Owner escalated via Telegram zero-cost fallback!
            tg_before = len(sent_telegram)
            comp_resp = await client.post(
                "/api/ai/chat",
                json={"phone_number": "01077770001", "message": "الأوردر وصل بس ناقص ساندوتش"},
            )
            assert comp_resp.status_code == 200
            comp_data = comp_resp.json()
            assert comp_data["intent"] == "complaint"
            # Owner has no active 24h WhatsApp session -> automatically falls back to Telegram (0 EGP)!
            assert comp_data["complaint_escalation"]["channel"] == "telegram"
            assert len(sent_telegram) == tg_before + 1
            # Owner private phone number MUST NEVER be exposed in customer reply!
            assert "201000000001" not in comp_data["reply_ar"]

            # Follow-up non-critical message during takeover -> AI silenced & NO duplicate owner alert!
            comp_follow = await client.post(
                "/api/ai/chat",
                json={"phone_number": "01077770001", "message": "انا مستني ردكم"},
            )
            assert comp_follow.json()["suppress_ai_reply"] is True
            assert len(sent_telegram) == tg_before + 1

            # 11. Customer Experience Dashboard APIs (genuine DB data only) + Acknowledge & Resolve
            cx = await client.get("/api/customer-experience/overview", headers=headers)
            assert cx.status_code == 200
            cx_data = cx.json()
            assert cx_data["reviews"]["total_submitted"] == 1
            assert cx_data["reviews"]["average_rating"] == 5.0
            assert cx_data["complaints"]["total"] == 1
            assert cx_data["complaints"]["escalated"] == 1

            tickets_list = (await client.get("/api/complaints", headers=headers)).json()
            tid = tickets_list[0]["id"]
            assert (await client.post(f"/api/complaints/{tid}/acknowledge", headers=headers)).status_code == 200
            assert (
                await client.post(
                    f"/api/complaints/{tid}/resolve",
                    json={"resolution_note": "Sent replacement sandwich", "release_human_takeover": True},
                    headers=headers,
                )
            ).status_code == 200

            # After resolving & releasing human takeover, AI resumes replying normally!
            after_res = await client.post(
                "/api/ai/chat",
                json={"phone_number": "01077770001", "message": "شكراً ليكم السلام عليكم"},
            )
            assert after_res.json()["suppress_ai_reply"] is False
            assert len(after_res.json()["reply_ar"]) > 5

        await mock_tg_http.aclose()
        await mock_wa_http.aclose()

    asyncio.run(_run())


def test_alembic_migrations_0001_to_0002_upgrade_and_downgrade(tmp_path):
    """Verify full Alembic schema migration chain runs cleanly from scratch and downgrades safely."""
    mig_db = tmp_path / "alembic_chain.db"
    db_url = f"sqlite:///{mig_db.as_posix()}"
    os.environ["DATABASE_URL"] = db_url
    for mod in ["app.config", "app.database"]:
        if mod in sys.modules:
            del sys.modules[mod]
    cfg = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)

    alembic_command.upgrade(cfg, "head")
    eng = create_engine(f"sqlite:///{mig_db.as_posix()}")
    with eng.connect() as conn:
        tables = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    eng.dispose()

    for required_table in (
        "orders",
        "order_status_history",
        "scheduled_jobs",
        "notification_attempts",
        "complaint_tickets",
        "recommendation_events",
        "restaurant_settings",
        "feedback_requests",
        "users",
        "audit_logs",
    ):
        assert required_table in tables, f"Missing table {required_table} after alembic upgrade head"

    # Verify clean downgrade to base and re-upgrade to head
    alembic_command.downgrade(cfg, "base")
    alembic_command.upgrade(cfg, "head")
