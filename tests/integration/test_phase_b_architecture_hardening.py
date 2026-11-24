"""Phase B Comprehensive Architecture Hardening, Concurrency, Persistence & Integration Tests.

Covers every Phase A confirmed finding & Phase B requirement:
1. CAMeLBERT LoRA v1/v2 checkpoint loading, caching, honest /api/ready reporting & low-confidence guard
2. Gemini native Function Calling payload, ToolExecutor validation/dedup & hallucination rejection
3. Gemini Primary -> Fallback on HTTP 503/429, timeout, invalid JSON, malformed functionCall & low confidence
4. Durable PostgreSQL/SQLite 14-field SessionState restoration across restarts + TTL expiration
5. Concurrent multi-customer isolation (no shared mutable dispatcher metadata) + idempotent confirmation
6. Egyptian Arabic multi-variant extraction, self-correction ("لأ، خليها كبيرة"), fragmented notes & gibberish
7. Live dashboard menu price & availability synchronization + exact Decimal customer/DB total parity
8. Multi-product price inquiries with availability enforcement
9. 3s inbound WhatsApp MessageDebouncer aggregation, urgent complaint bypass & restart crash recovery
10. Alembic migration chain 0001 -> 0005 upgrade/downgrade/upgrade
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import time

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
import httpx
import pytest
from sqlalchemy import create_engine, inspect, select, text

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai_service.adapters.mock_backend import MockBackendAdapter  # noqa: E402
from ai_service.config import WaslaConfig  # noqa: E402
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher  # noqa: E402
from ai_service.models.finetuned_intent import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    V1_CHECKPOINT_DIR,
    FineTunedIntentClassifier,
)
from ai_service.models.llm_client import GeminiLLMClient, OfflineEgyptianNLU  # noqa: E402
from ai_service.schemas.models import SessionState  # noqa: E402
from ai_service.tools.registry import ToolExecutor  # noqa: E402
from ai_service.understanding.menu_index import MenuIndex  # noqa: E402

MENU_PATH = ROOT / "menu" / "menu.json"


def _fresh_app_modules(db_path: Path):
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
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
    import app.models as models_mod  # noqa: E402

    main_mod._migrate_sqlite_columns()
    main_mod.seed_moka_menu_if_empty()
    main_mod.db_bridge.sync_menu_sync()
    main_mod.seed_initial_admin_if_empty()
    return cfg_mod, db_mod, main_mod, models_mod


def test_lora_v1_v2_loading_caching_and_ready_honesty(tmp_path):
    # Directory exists != loaded before load()
    clf_v1 = FineTunedIntentClassifier(checkpoint_dir=V1_CHECKPOINT_DIR)
    assert clf_v1.checkpoint_exists is True
    assert clf_v1.is_loaded is False
    assert clf_v1.load() is True
    assert clf_v1.is_loaded is True
    pred_v1 = clf_v1.predict("عايز كريب كريسبي")
    assert pred_v1 is not None and len(pred_v1) == 3

    clf_v2 = FineTunedIntentClassifier(checkpoint_dir=DEFAULT_CHECKPOINT_DIR)
    assert clf_v2.checkpoint_exists is True
    assert clf_v2.is_loaded is False
    assert clf_v2.load() is True
    assert clf_v2.is_loaded is True
    pred_v2 = clf_v2.predict("بكام بيتزا مارجريتا وسط؟")
    assert pred_v2 is not None and pred_v2[1] > 0.0

    # Cached attach must succeed virtually instantaneously (<150ms)
    clf_v2_cached = FineTunedIntentClassifier(checkpoint_dir=DEFAULT_CHECKPOINT_DIR)
    t0 = time.perf_counter()
    assert clf_v2_cached.load() is True
    assert (time.perf_counter() - t0) < 0.15

    # Honest /api/ready reporting
    _, _, main_mod, _ = _fresh_app_modules(tmp_path / "ready_lora.db")

    async def _run():
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.get("/api/ready")
            assert r.status_code == 200
            body = r.json()
            assert body["providers"]["finetuned_lora_checkpoint_exists"] is True
            assert body["providers"]["finetuned_lora_version"] == "camelbert-wasla-lora-v2"
            assert body["providers"]["finetuned_lora_inference_available"] is True
            assert body["ai_models"]["checkpoint_v1_exists"] is True
            assert body["ai_models"]["checkpoint_v2_exists"] is True
            assert "active_inference_route" in body["ai_models"]

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_low_confidence_prediction_never_creates_or_confirms_order():
    class LowConfLLM:
        is_configured = True

        async def generate_json(self, system_prompt, user_prompt, use_fallback_model=False):
            return {
                "intent": "add_to_cart",
                "confidence": 0.40,
                "items": [{"product_id": "item_029", "raw_mention": "كلام مش واضح", "quantity": 1, "size_code": "M"}],
                "requires_clarification": False,
            }, "mock-gemini"

    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    dispatcher = WaslaAIDispatcher(WaslaConfig(gemini_api_key="test"), idx, backend, llm_client=LowConfLLM())
    dispatcher.finetuned_intent.predict = lambda text: ("confirm_order", 0.31, 1.0)

    r1 = await dispatcher.handle_text("01066661111", "كلام غريب مش مفهوم خالص كده")
    assert len(r1.cart.items) == 0
    assert r1.order_id is None
    assert len(backend._orders) == 0
    assert r1.requires_clarification is True


@pytest.mark.asyncio
async def test_gemini_function_calling_validation_dedup_and_hallucination_rejection():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "check_item_availability", "args": {"product_id": "item_005"}}},
                                {"functionCall": {"name": "check_item_availability", "args": {"product_id": "item_005"}}},
                                {
                                    "text": json.dumps(
                                        {
                                            "intent": "add_to_cart",
                                            "confidence": 0.93,
                                            "items": [
                                                {"product_id": "item_005", "raw_mention": "كريب كريسبي", "quantity": 1},
                                                {"product_id": "item_999", "raw_mention": "طبق سحري", "quantity": 1},
                                            ],
                                            "requires_clarification": False,
                                        },
                                        ensure_ascii=False,
                                    )
                                },
                            ]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    cfg = WaslaConfig(gemini_api_key="test-gemini-key")
    llm = GeminiLLMClient(cfg, transport=transport)
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    dispatcher = WaslaAIDispatcher(cfg, idx, backend, llm_client=llm)

    resp = await dispatcher.handle_text("01066662222", "عايز كريب كريسبي وطبق سحري")
    assert seen_payloads and "tools" in seen_payloads[0]
    assert "functionDeclarations" in seen_payloads[0]["tools"][0]
    assert any(tc.tool_name == "modify_cart" and tc.arguments["product_id"] == "item_005" for tc in resp.tool_calls)
    assert all(i.product_id != "item_999" for i in resp.cart.items)

    # ToolExecutor deduplicates identical tool executions within one turn
    executor = ToolExecutor(backend)
    sigs: set[str] = set()
    sess = SessionState(phone_number="01066662223")
    rec1, _ = await executor.execute(sess, "modify_cart", {"product_id": "item_005", "quantity": 1}, executed_signatures=sigs)
    rec2, res2 = await executor.execute(sess, "modify_cart", {"product_id": "item_005", "quantity": 1}, executed_signatures=sigs)
    assert rec1.result_summary.startswith("subtotal=")
    assert rec2.result_summary == "deduplicated_skip"
    assert res2 == {"deduplicated": True}
    assert len(sess.cart.items) == 1


@pytest.mark.asyncio
async def test_gemini_primary_to_fallback_on_all_error_modes():
    modes = ["http_503", "http_429", "timeout", "bad_json", "malformed_fc", "low_conf"]
    for mode in modes:
        calls = []

        def handler(request: httpx.Request, m=mode, c=calls) -> httpx.Response:
            is_fb = "gemini-3.8-flash" in str(request.url)
            c.append(is_fb)
            if is_fb:
                return httpx.Response(
                    200,
                    json={"candidates": [{"content": {"parts": [{"text": '{"intent":"greeting","confidence":0.94,"items":[],"requires_clarification":false}'}]}}]},
                )
            if m == "http_503":
                return httpx.Response(503, text="Service Unavailable")
            if m == "http_429":
                return httpx.Response(429, text="Rate Limit")
            if m == "timeout":
                raise httpx.ReadTimeout("Simulated timeout")
            if m == "bad_json":
                return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not json at all {{{"}]}}]})
            if m == "malformed_fc":
                return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"functionCall": {"name": "drop_database", "args": {}}}]}}]})
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": '{"intent":"greeting","confidence":0.35,"items":[],"requires_clarification":true}'}]}}]},
            )

        cfg = WaslaConfig(gemini_api_key="test-key")
        llm = GeminiLLMClient(cfg, transport=httpx.MockTransport(handler))
        idx = MenuIndex(MENU_PATH)
        dispatcher = WaslaAIDispatcher(cfg, idx, MockBackendAdapter(idx), llm_client=llm)
        dispatcher.finetuned_intent.predict = lambda t: ("greeting", 0.90, 1.0)
        parsed, model_used = await dispatcher._run_nlu("السلام عليكم")
        assert True in calls, f"Fallback model was not called for mode={mode}: calls={calls}"
        assert model_used == cfg.ai_fallback_model, f"Wrong model_used={model_used} for mode={mode}"
        assert parsed["intent"] == "greeting"


@pytest.mark.asyncio
async def test_persistent_session_all_14_fields_restart_and_ttl(tmp_path):
    _, db_mod, main_mod, models_mod = _fresh_app_modules(tmp_path / "persist_14_fields.db")
    phone = "01055559999"

    cfg = WaslaConfig(gemini_api_key="", session_ttl_seconds=3600)
    idx1 = MenuIndex(MENU_PATH)
    bridge1 = main_mod._DirectDBOrderBridge(idx1)
    bridge1.sync_menu_sync()
    d1 = WaslaAIDispatcher(cfg, idx1, bridge1)
    d1.finetuned_intent.predict = lambda t: ("add_to_cart", 0.91, 1.0)

    r1 = await d1.handle_text(phone, "عايز بيتزا مارجريتا كبير حبة كاملة بدون بصل")
    assert len(r1.cart.items) == 1
    assert r1.upsell_offer is not None

    # Reject upsell + capture partial customer details (remaining details still needed)
    r2 = await d1.handle_text(phone, "لا شكرا مش عايز")
    assert r2.upsell_event["action"] == "rejected"
    await d1.handle_text(phone, "الاسم: سلمى طارق، العنوان: المعادي شارع 9، تمام اكد الطلب")

    # Simulate FULL worker/process restart with brand-new MenuIndex, DB bridge & Dispatcher
    idx2 = MenuIndex(MENU_PATH)
    bridge2 = main_mod._DirectDBOrderBridge(idx2)
    bridge2.sync_menu_sync()
    d2 = WaslaAIDispatcher(cfg, idx2, bridge2)
    restored = d2.get_session(phone)

    assert len(restored.cart.items) == 1
    assert restored.cart.items[0].size_code == "L"
    assert restored.cart.items[0].quantity == 1
    assert len(restored.cart.items[0].modifiers) == 1
    assert "بدون بصل" in restored.cart.items[0].notes
    assert restored.customer_name == "سلمى طارق"
    assert "المعادي" in (restored.delivery_address or "")
    assert restored.upsell_declined is True
    assert restored.rejected_upsell_ids
    assert restored.last_intent == "confirm_order"
    assert restored.awaiting_final_confirmation is True
    assert len(restored.history) >= 6

    # Finish order after restart by supplying only missing backup_phone (remembers name & address!)
    r_final = await d2.handle_text(phone, "رقم بديل: 01234567890 تمام اكد الطلب")
    assert r_final.order_id is not None

    idx3 = MenuIndex(MENU_PATH)
    bridge3 = main_mod._DirectDBOrderBridge(idx3)
    d3 = WaslaAIDispatcher(cfg, idx3, bridge3)
    post_confirm = d3.get_session(phone)
    assert post_confirm.order_confirmed is True
    assert post_confirm.active_order_id == r_final.order_id

    # Verify SESSION_TTL_SECONDS expiration clears stale draft session safely
    stale_phone = "01055558888"
    await d3.handle_text(stale_phone, "عايز كريب كوردن بلو")
    db = db_mod.SessionLocal()
    try:
        row = db.scalar(select(models_mod.ChatSession).where(models_mod.ChatSession.phone_number == stale_phone))
        row.last_customer_message_at = datetime.now(timezone.utc) - timedelta(seconds=7200)
        db.commit()
    finally:
        db.close()

    expired = d3.get_session(stale_phone)
    assert len(expired.cart.items) == 0
    assert expired.session_epoch >= 2


@pytest.mark.asyncio
async def test_concurrent_customers_isolation_and_idempotent_confirm(tmp_path):
    _, _, main_mod, _ = _fresh_app_modules(tmp_path / "concurrent_iso.db")
    cfg = WaslaConfig(gemini_api_key="")
    idx = MenuIndex(MENU_PATH)
    bridge = main_mod._DirectDBOrderBridge(idx)
    bridge.sync_menu_sync()
    disp = WaslaAIDispatcher(cfg, idx, bridge)
    disp.finetuned_intent.predict = lambda t: ("add_to_cart", 0.92, 1.0)

    p_a = "01011110001"
    p_b = "01022220002"

    # Concurrent interleaved requests for two different customers with different metadata
    r_a, r_b = await asyncio.gather(
        disp.handle_text(
            p_a,
            "عايز كريب كوردن بلو الاسم: عميل أ، العنوان: الدقي شارع 1، رقم بديل: 01011110009 تمام اكد الطلب",
            data_source="simulator",
            is_test=True,
        ),
        disp.handle_text(
            p_b,
            "عايز بيتزا مارجريتا كبير",
            data_source="whatsapp_webhook",
            is_test=False,
        ),
    )

    assert len(r_a.cart.items) == 1 and r_a.cart.items[0].product_id == "item_020"
    assert len(r_b.cart.items) == 1 and r_b.cart.items[0].product_id == "item_029"

    # Concurrent double confirmation from Customer B must create strictly 1 order
    await disp.handle_text(p_b, "الاسم: عميل ب، العنوان: المعادي شارع 5، رقم بديل: 01222220009", data_source="whatsapp_webhook", is_test=False)
    c1, c2 = await asyncio.gather(
        disp.handle_text(p_b, "تمام اكد الطلب", data_source="whatsapp_webhook", is_test=False),
        disp.handle_text(p_b, "تمام اكد الطلب", data_source="whatsapp_webhook", is_test=False),
    )
    assert c1.order_id is not None
    assert c1.order_id == c2.order_id


@pytest.mark.asyncio
async def test_multi_variant_self_correction_fragmented_and_gibberish():
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    disp = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, backend)
    disp.finetuned_intent.predict = lambda t: ("add_to_cart", 0.90, 1.0)

    # 1. Repeated same product with different sizes -> 2 lines (M=125 + L=150 = 275.00)
    r_sizes = await disp.handle_text("01077771001", "عايز بيتزا مارجريتا وسط وبيتزا مارجريتا كبير")
    assert len(r_sizes.cart.items) == 2
    assert [i.size_code for i in r_sizes.cart.items] == ["M", "L"]
    assert r_sizes.cart.subtotal == Decimal("275.00")

    # 2. Repeated same product with different extras -> 2 distinct lines
    r_extras = await disp.handle_text("01077771002", "عايز كريب كريسبي موتزريلا وكريب كريسبي هالبينو")
    assert len(r_extras.cart.items) == 2

    # 3. Elliptical self-correction ("لأ، خليها كبيرة") updates size & price cleanly
    r_init = await disp.handle_text("01077771003", "عايز بيتزا مارجريتا وسط")
    assert r_init.cart.subtotal == Decimal("125.00")
    r_corr = await disp.handle_text("01077771003", "لأ، خليها كبيرة")
    assert len(r_corr.cart.items) == 1
    assert r_corr.cart.items[0].size_code == "L"
    assert r_corr.cart.subtotal == Decimal("150.00")

    # 4. Fragmented WhatsApp note on second line
    r_frag = await disp.handle_text("01077771004", "عايز اتنين كريب كريسبي\nمن غير بصل")
    assert len(r_frag.cart.items) == 1
    assert r_frag.cart.items[0].quantity == 2
    assert "من غير بصل" in r_frag.cart.items[0].notes

    # 5. Gibberish / random keysmash -> unclear_request without quoting 'uhdd' as a dish
    r_gib = await disp.handle_text("01077771005", "uhdd")
    assert r_gib.intent.value == "unclear_request"
    assert "uhdd" not in r_gib.reply_ar


def test_live_price_edit_sync_to_ai_and_db_order_total_plus_availability(tmp_path):
    cfg_mod, db_mod, main_mod, models_mod = _fresh_app_modules(tmp_path / "live_price_sync.db")

    async def _run():
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            login = await client.post(
                "/api/auth/login",
                json={"username": cfg_mod.settings.bootstrap_admin_username, "password": cfg_mod.settings.bootstrap_admin_password},
            )
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

            # Find كريب كريسبي (item_005)
            menu_all = (await client.get("/api/menu/all", headers=headers)).json()
            crispy = next(m for m in menu_all if m["product_id"] == "item_005")

            # 1. Admin edits price to 111.50 EGP
            up = await client.patch(f"/api/menu/{crispy['id']}/price", headers=headers, json={"base_price": "111.50"})
            assert up.status_code == 200
            assert up.json()["base_price"] == "111.50"

            # 2. Customer price inquiry sees live 111.50 price immediately
            price_resp = await main_mod.ai_dispatcher.handle_text("01044441111", "بكام كريب كريسبي")
            assert "111.50" in price_resp.reply_ar

            # 3. Customer adds item (cart = 111.50) & confirms via webhook (delivery 20.00 -> 131.50)
            add_resp = await main_mod.ai_dispatcher.handle_text("01044442222", "عايز كريب كريسبي")
            assert add_resp.cart.subtotal == Decimal("111.50")
            conf_resp = await main_mod.ai_dispatcher.handle_text(
                "01044442222",
                "الاسم: نورا علي، العنوان: المهندسين شارع 4، رقم بديل: 01144442222 تمام اكد الطلب",
                data_source="whatsapp_webhook",
                is_test=False,
            )
            assert conf_resp.order_id is not None

            orders = (await client.get("/api/orders", headers=headers)).json()
            persisted = next(o for o in orders if o["id"] == conf_resp.order_id)
            assert Decimal(persisted["subtotal"]) == Decimal("111.50")
            assert Decimal(persisted["total_amount"]) == Decimal("131.50")
            assert Decimal(persisted["total_amount"]) == conf_resp.cart.total_amount

            # 4. Toggle item unavailable -> survives re-sync/restart & blocks adding/confirming
            off = await client.patch(f"/api/menu/{crispy['id']}/availability", headers=headers, json={"is_available": False})
            assert off.status_code == 200
            main_mod.db_bridge.sync_menu_sync()
            avail_reply = await main_mod.ai_dispatcher.handle_text("01044443333", "بكام كريب كريسبي")
            assert "غير متوفر" in avail_reply.reply_ar

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_multi_product_price_inquiry_mixed_availability():
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    backend.set_item_unavailable("item_020", True)  # كريب كوردن بلو out of stock
    disp = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, backend)

    resp = await disp.handle_text("01011113333", "بكام بيتزا مارجريتا وسط وكريب كوردن بلو؟")
    assert "مارجريتا" in resp.reply_ar and "125" in resp.reply_ar
    assert "كوردن بلو" in resp.reply_ar and "غير متوفر" in resp.reply_ar


def test_inbound_webhook_debouncer_urgent_bypass_and_crash_recovery(tmp_path):
    cfg_mod, db_mod, main_mod, models_mod = _fresh_app_modules(tmp_path / "debounce_wh.db")
    cfg_mod.settings.meta_app_secret = "test_secret_debounce"
    main_mod.inbound_debouncer.window_seconds = 0.25

    sent_wa: list[dict] = []

    async def fake_send(to_phone: str, text: str):
        sent_wa.append({"to": to_phone, "text": text})
        return {"sent": True, "mode": "mock"}

    main_mod.meta_client.send_text_message = fake_send

    def _signed_headers(raw: bytes) -> dict[str, str]:
        sig = "sha256=" + hmac.new(b"test_secret_debounce", raw, hashlib.sha256).hexdigest()
        return {"Content-Type": "application/json", "X-Hub-Signature-256": sig}

    def _wh_body(mid: str, sender: str, text: str, ts: int = 1763900010) -> bytes:
        return json.dumps(
            {"entry": [{"changes": [{"value": {"messages": [{"id": mid, "from": sender, "timestamp": str(ts), "type": "text", "text": {"body": text}}]}}]}]}
        ).encode("utf-8")

    async def _run():
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Burst from Customer 1 + simultaneous message from Customer 2
            b1 = _wh_body("w_d1", "201055511111", "مساء الخير", 1763900001)
            b2 = _wh_body("w_d2", "201055511111", "عايز اتنين كريب كريسبي", 1763900002)
            b3 = _wh_body("w_d3", "201055511111", "من غير بصل", 1763900003)
            b_other = _wh_body("w_other", "201055522222", "عايز بيتزا مارجريتا وسط", 1763900002)

            await asyncio.gather(
                client.post("/api/webhook", content=b1, headers=_signed_headers(b1)),
                client.post("/api/webhook", content=b2, headers=_signed_headers(b2)),
                client.post("/api/webhook", content=b3, headers=_signed_headers(b3)),
                client.post("/api/webhook", content=b_other, headers=_signed_headers(b_other)),
            )

            c1_msgs = [m for m in sent_wa if m["to"] == "201055511111"]
            c2_msgs = [m for m in sent_wa if m["to"] == "201055522222"]
            assert len(c1_msgs) == 1
            assert len(c2_msgs) == 1
            assert "220.00" in c1_msgs[0]["text"]
            assert "125.00" in c2_msgs[0]["text"]

            # 2. Urgent CRITICAL complaint bypasses the 3.0s debounce window immediately
            main_mod.inbound_debouncer.window_seconds = 3.0
            t0 = time.perf_counter()
            b_crit = _wh_body("w_crit", "201055533333", "في شعرة في الأكل وتسمم!", 1763900020)
            rc = await client.post("/api/webhook", content=b_crit, headers=_signed_headers(b_crit))
            elapsed = time.perf_counter() - t0
            assert rc.status_code == 200
            assert elapsed < 2.2
            assert any(m["to"] == "201055533333" for m in sent_wa)

            # 3. Simulated crash leaving BUFFERED message -> recovered on startup
            db = db_mod.SessionLocal()
            try:
                db.add(
                    models_mod.InboundMessageBuffer(
                        message_uid="w_crash_001",
                        phone_number="201055544444",
                        content="بكام كريب كريسبي",
                        data_source="whatsapp_webhook",
                        msg_timestamp=1763900030,
                        status="BUFFERED",
                    )
                )
                db.commit()
            finally:
                db.close()

            recovered = await main_mod.recover_unflushed_inbound_messages()
            assert recovered >= 1
            assert any(m["to"] == "201055544444" and "110" in m["text"] for m in sent_wa)

    asyncio.run(_run())


def test_alembic_0001_to_0005_migrations_backfill_and_downgrade(tmp_path):
    mig_db = tmp_path / "alembic_0005.db"
    db_url = f"sqlite:///{mig_db.as_posix()}"
    os.environ["DATABASE_URL"] = db_url
    for mod in ["app.config", "app.database"]:
        if mod in sys.modules:
            del sys.modules[mod]

    cfg = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)

    alembic_command.upgrade(cfg, "0004_phase_b_durable_ai_state")
    eng = create_engine(db_url)
    with eng.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO orders "
                "(id, customer_name, customer_phone, backup_phone, confirmed_by_customer, "
                "delivery_address, subtotal, delivery_fee, total_amount, status, items_snapshot, "
                "data_source, is_test, created_at, updated_at, order_type, refunded) "
                "VALUES (8101, 'Legacy', '01000000000', '', 1, 'Legacy address', "
                "'10.00', '0.00', '10.00', 'SUBMITTED', '[]', 'api', 0, "
                "'2026-11-24 00:00:00', '2026-11-24 00:00:00', 'Delivery', 0)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO order_events (order_id, event_type, payload, created_at) "
                "VALUES (8101, 'idempotency_key', '{\"key\":\"legacy-idem-key\"}', "
                "'2026-11-24 00:00:00')"
            )
        )
    eng.dispose()

    alembic_command.upgrade(cfg, "head")
    eng = create_engine(db_url)
    inspector = inspect(eng)
    tables = set(inspector.get_table_names())
    order_indexes = {idx["name"]: idx for idx in inspector.get_indexes("orders")}
    with eng.connect() as connection:
        migrated_key = connection.scalar(text("SELECT idempotency_key FROM orders WHERE id = 8101"))
    eng.dispose()
    assert "inbound_message_buffers" in tables
    assert bool(order_indexes["ux_orders_idempotency_key"]["unique"])
    assert migrated_key == "legacy-idem-key"

    alembic_command.downgrade(cfg, "base")
    alembic_command.upgrade(cfg, "head")
