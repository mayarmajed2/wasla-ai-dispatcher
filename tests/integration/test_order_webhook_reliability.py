"""Regression coverage for persistent cart identity, trusted order pricing, and Telegram deduplication."""

from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _fresh_app_modules(db_path: Path):
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    for module in (
        "app.config",
        "app.database",
        "app.models",
        "app.auth",
        "app.notifications",
        "app.scheduler",
        "app.main",
    ):
        sys.modules.pop(module, None)

    import app.config as config
    import app.database as database
    import app.main as main
    import app.models as models

    main._migrate_sqlite_columns()
    main.seed_moka_menu_if_empty()
    main.db_bridge.sync_menu_sync()
    main.seed_initial_admin_if_empty()
    return config, database, main, models


def test_database_cart_removes_only_the_selected_variant(tmp_path):
    _, _, main, _ = _fresh_app_modules(tmp_path / "cart_identity.db")

    async def run():
        from ai_service.understanding.menu_index import MenuIndex

        first = main._DirectDBOrderBridge(MenuIndex(main.MENU_JSON_PATH))
        medium = await first.modify_cart("01055550001", "item_029", 2, size_code="M")
        cart = await first.modify_cart("01055550001", "item_029", 1, size_code="L")
        second = main._DirectDBOrderBridge(MenuIndex(main.MENU_JSON_PATH))
        remaining = await second.remove_cart_item("01055550001", medium.items[0].cart_item_id)
        restored = await second.get_cart_summary("01055550001")

        assert len(remaining.items) == len(restored.items) == 1
        assert restored.items[0].cart_item_id == cart.items[1].cart_item_id
        assert restored.items[0].size_code == "L"
        assert restored.items[0].quantity == 1
        assert restored.subtotal == cart.items[1].line_total

    asyncio.run(run())


def test_database_order_status_is_scoped_to_the_customer_and_active_order(tmp_path):
    _, database, main, models = _fresh_app_modules(tmp_path / "order_tracking.db")
    phone = "01055550009"
    db = database.SessionLocal()
    try:
        orders = [
            models.Order(
                customer_name="عميل التتبع",
                customer_phone=phone,
                backup_phone="01155550009",
                confirmed_by_customer=True,
                delivery_address="المعادي شارع 9",
                subtotal=Decimal("100.00"),
                delivery_fee=Decimal("20.00"),
                total_amount=Decimal("120.00"),
                status="OUT_FOR_DELIVERY",
                items_snapshot=[],
                data_source="api",
                is_test=False,
            ),
            models.Order(
                customer_name="عميل التتبع",
                customer_phone=phone,
                backup_phone="01155550009",
                confirmed_by_customer=True,
                delivery_address="المعادي شارع 9",
                subtotal=Decimal("100.00"),
                delivery_fee=Decimal("20.00"),
                total_amount=Decimal("120.00"),
                status="DELIVERED",
                items_snapshot=[],
                data_source="api",
                is_test=False,
            ),
            models.Order(
                customer_name="عميل آخر",
                customer_phone="01055550010",
                backup_phone="01155550010",
                confirmed_by_customer=True,
                delivery_address="الدقي شارع التحرير",
                subtotal=Decimal("100.00"),
                delivery_fee=Decimal("20.00"),
                total_amount=Decimal("120.00"),
                status="DELIVERED",
                items_snapshot=[],
                data_source="api",
                is_test=False,
            ),
        ]
        db.add_all(orders)
        db.flush()
        active_id = orders[0].id
        other_id = orders[2].id
        db.add(models.ChatSession(phone_number=phone, active_order_id=active_id))
        db.commit()
    finally:
        db.close()

    async def run():
        status = await main.db_bridge.get_customer_order_status(phone)
        assert status == {"id": active_id, "status": "OUT_FOR_DELIVERY"}
        assert await main.db_bridge.get_customer_order_status(phone, other_id) is None
        assert await main.db_bridge.get_customer_order_status("01055550099") is None

    asyncio.run(run())


def test_delivery_quote_uses_configured_fee_and_has_no_unverified_eta(tmp_path):
    config, _, main, _ = _fresh_app_modules(tmp_path / "delivery_quote.db")
    config.settings.delivery_default_fee = Decimal("37.50")
    config.settings.delivery_free_zone_fee = Decimal("0.00")
    config.settings.delivery_estimated_minutes = None

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            paid = await client.get("/api/delivery/quote", params={"zone_name": "nasr-city"})
            free = await client.get("/api/delivery/quote", params={"zone_name": "free_zone"})

        assert paid.json() == {
            "deliverable": True,
            "zone_name": "nasr-city",
            "delivery_fee": "37.50",
            "estimated_minutes": None,
        }
        assert free.json()["delivery_fee"] == "0.00"

    asyncio.run(run())


def test_http_customer_status_requires_service_auth_and_enforces_phone_ownership(tmp_path):
    config, database, main, models = _fresh_app_modules(tmp_path / "authenticated_tracking.db")
    config.settings.backend_service_api_key = ""
    phone = "01055550021"
    db = database.SessionLocal()
    try:
        own = models.Order(
            customer_name="عميل التتبع",
            customer_phone=phone,
            backup_phone="01155550021",
            confirmed_by_customer=True,
            delivery_address="المعادي شارع 9",
            subtotal=Decimal("100.00"),
            delivery_fee=Decimal("20.00"),
            total_amount=Decimal("120.00"),
            status="IN_KITCHEN",
            items_snapshot=[],
            data_source="api",
            is_test=False,
        )
        foreign = models.Order(
            customer_name="عميل آخر",
            customer_phone="01055550022",
            backup_phone="01155550022",
            confirmed_by_customer=True,
            delivery_address="الدقي شارع التحرير",
            subtotal=Decimal("100.00"),
            delivery_fee=Decimal("20.00"),
            total_amount=Decimal("120.00"),
            status="DELIVERED",
            items_snapshot=[],
            data_source="api",
            is_test=False,
        )
        db.add_all([own, foreign])
        db.flush()
        db.add(models.ChatSession(phone_number=phone, active_order_id=own.id))
        db.commit()
        own_id, foreign_id = own.id, foreign.id
    finally:
        db.close()

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            path = "/api/customer/order-status"
            payload = {"customer_phone": phone}
            unconfigured = await client.post(path, json=payload)
            config.settings.backend_service_api_key = "unit-test-service-key"
            unauthenticated = await client.post(path, json=payload)
            own_status = await client.post(
                path,
                headers={"X-Backend-Service-Key": "unit-test-service-key"},
                json=payload,
            )
            foreign_status = await client.post(
                path,
                headers={"X-Backend-Service-Key": "unit-test-service-key"},
                json={"customer_phone": phone, "order_id": foreign_id},
            )
            missing_status = await client.post(
                path,
                headers={"X-Backend-Service-Key": "unit-test-service-key"},
                json={"customer_phone": "01055550023", "order_id": own_id},
            )

        assert unconfigured.status_code == 503
        assert unauthenticated.status_code == 401
        assert own_status.json() == {"id": own_id, "status": "IN_KITCHEN"}
        assert foreign_status.json() == {"id": None, "status": None}
        assert missing_status.json() == {"id": None, "status": None}

    asyncio.run(run())


def test_order_api_uses_database_prices_and_enforces_idempotency(tmp_path):
    config, database, main, models = _fresh_app_modules(tmp_path / "order_integrity.db")
    config.settings.delivery_default_fee = Decimal("37.50")

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            login = await client.post(
                "/api/auth/login",
                json={
                    "username": config.settings.bootstrap_admin_username,
                    "password": config.settings.bootstrap_admin_password,
                },
            )
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

            db = database.SessionLocal()
            try:
                item = next(
                    row
                    for row in db.scalars(select(models.MenuItem)).all()
                    if row.modifiers_json.get("product_id") == "item_037"
                )
                sizes = item.modifiers_json["sizes"]
                extras = item.modifiers_json["extras"]
                size = next(entry for entry in sizes if entry["code"] == "M")
                extra = next(entry for entry in extras if entry.get("modifier_id") == "mod_pizza_whole_wheat_crust")
                expected_subtotal = (Decimal(size["price"]) + Decimal(extra["price"])) * 2
                item_id = item.id
            finally:
                db.close()

            payload = {
                "customer_name": "عميل اختبار",
                "customer_phone": "01055550002",
                "backup_phone": "01155550002",
                "delivery_address": "المعادي شارع 9",
                "zone_name": "default",
                "items": [
                    {
                        "item_id": item_id,
                        "quantity": 2,
                        "size_code": "M",
                        "modifiers": [
                            {
                                "modifier_id": "mod_pizza_whole_wheat_crust",
                                "name": "غير موثوق",
                                "price": "999999.00",
                            }
                        ],
                        "selected_options": {"sauce": "باربيكيو"},
                    }
                ],
                "idempotency_key": "cart-order-retry-001",
                "confirmed_by_customer": True,
            }

            invalid_size = dict(payload)
            invalid_size["idempotency_key"] = "cart-order-invalid-size"
            invalid_size["items"] = [dict(payload["items"][0], size_code="XX")]
            assert (await client.post("/api/orders", headers=headers, json=invalid_size)).status_code == 400
            missing_option = dict(payload)
            missing_option["idempotency_key"] = "cart-order-missing-option"
            missing_option["items"] = [dict(payload["items"][0], selected_options={})]
            assert (await client.post("/api/orders", headers=headers, json=missing_option)).status_code == 400
            invalid_modifier = dict(payload)
            invalid_modifier["idempotency_key"] = "cart-order-invalid-modifier"
            invalid_modifier["items"] = [
                dict(
                    payload["items"][0],
                    modifiers=[{"modifier_id": "mod_crepe_mozzarella", "price": "0"}],
                )
            ]
            assert (await client.post("/api/orders", headers=headers, json=invalid_modifier)).status_code == 400

            created = await client.post("/api/orders", headers=headers, json=payload)
            duplicate = await client.post("/api/orders", headers=headers, json=payload)
            foreign_replay = await client.post(
                "/api/orders",
                headers=headers,
                json={**payload, "customer_phone": "01055550003"},
            )
            assert created.status_code == 201
            assert duplicate.status_code == 201
            assert duplicate.json()["duplicate"] is True
            assert duplicate.json()["id"] == created.json()["id"]
            assert foreign_replay.status_code == 409
            assert "id" not in foreign_replay.json()
            assert Decimal(created.json()["cart"]["subtotal"]) == expected_subtotal
            assert Decimal(created.json()["cart"]["delivery_fee"]) == Decimal("37.50")
            snapshot = created.json()["cart"]["items"][0]
            assert Decimal(snapshot["modifiers"][0]["price"]) == Decimal(extra["price"])

            db = database.SessionLocal()
            try:
                orders = db.scalars(
                    select(models.Order).where(models.Order.idempotency_key == payload["idempotency_key"])
                ).all()
                assert len(orders) == 1
                assert orders[0].items_snapshot[0]["selected_options"] == {"sauce": "باربيكيو"}
                session_row = db.scalar(
                    select(models.ChatSession).where(models.ChatSession.phone_number == payload["customer_phone"])
                )
                assert session_row is not None
                assert session_row.active_order_id == orders[0].id
            finally:
                db.close()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_telegram_update_is_claimed_before_processing_and_bad_json_is_400(tmp_path):
    _, database, main, models = _fresh_app_modules(tmp_path / "telegram_dedup.db")
    from ai_service.schemas.models import CartSnapshot, CustomerIntent, DispatcherResponse

    calls = []
    sent = []

    async def fake_handle_text(*args, **kwargs):
        calls.append((args, kwargs))
        await asyncio.sleep(0.01)
        return DispatcherResponse(
            intent=CustomerIntent.GREETING,
            confidence=0.99,
            cart=CartSnapshot(),
            reply_ar="أهلاً",
            model_used="test",
        )

    async def fake_send(chat_id, text):
        sent.append((chat_id, text))
        return {"sent": True}

    main.ai_dispatcher.handle_text = fake_handle_text
    main.telegram_client.send_message = fake_send
    headers = {"X-Telegram-Bot-Api-Secret-Token": main._telegram_webhook_secret()}
    update = {
        "update_id": 990001,
        "message": {
            "chat": {"id": 123456789, "type": "private"},
            "text": "أهلاً",
        },
    }

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        responses = await asyncio.gather(
            client.post("/api/telegram/webhook", headers=headers, json=update),
            client.post("/api/telegram/webhook", headers=headers, json=update),
        )
        malformed = await client.post("/api/telegram/webhook", headers=headers, content=b"{")
        malformed_whatsapp = await client.post("/api/webhook", content=b"{")

    assert sorted(response.status_code for response in responses) == [200, 200]
    assert sum(response.json().get("duplicate", False) for response in responses) == 1
    assert len(calls) == len(sent) == 1
    assert malformed.status_code == 400
    assert malformed_whatsapp.status_code == 400

    db = database.SessionLocal()
    try:
        row = db.scalar(
            select(models.ProcessedMessage.id).where(
                models.ProcessedMessage.meta_message_id == "tg_upd:990001"
            )
        )
        assert row is not None
    finally:
        db.close()
