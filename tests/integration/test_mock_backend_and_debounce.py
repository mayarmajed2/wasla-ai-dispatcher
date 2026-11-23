"""Tier 2 — Mock Backend Safety, Decimal Math, Idempotency & 3s Debounce Tests."""

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.dispatcher.debounce import MessageDebouncer
from ai_service.schemas.models import BackendCartItemPayload, BackendConfirmOrderPayload
from ai_service.understanding.menu_index import MenuIndex

MENU_PATH = Path(__file__).resolve().parent.parent.parent / "menu" / "menu.json"


def test_mock_backend_exact_decimal_math_and_whole_wheat_modifier():
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)

    async def _run():
        # بيتزا مارجريتا L (150) + حبة كاملة (+20) * qty 2 = 340.00
        cart = await backend.modify_cart(
            phone_number="01000000001",
            product_id="item_029",
            quantity=2,
            size_code="L",
            modifier_ids=["mod_pizza_whole_wheat_crust"],
        )
        assert cart.subtotal == Decimal("340.00")

        res = await backend.resolve_delivery("01000000001", "المعادي")
        assert res.delivery_fee == Decimal("20.00")
        summary = await backend.get_cart_summary("01000000001")
        assert summary.total_amount == Decimal("360.00")

    asyncio.run(_run())


def test_mock_backend_blocks_out_of_stock_and_unconfirmed_orders():
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    backend.set_item_unavailable("item_005", True)

    async def _run():
        with pytest.raises(BackendIntegrationError) as exc:
            await backend.modify_cart("01000000002", "item_005", 1)
        assert exc.value.code == "ITEM_OUT_OF_STOCK"

        unconfirmed = BackendConfirmOrderPayload(
            customer_name="احمد علي",
            customer_phone="01000000002",
            backup_phone="01111111111",
            delivery_address="المعادي شارع 9 عمارة 12",
            items=[BackendCartItemPayload(item_id=1, quantity=1)],
            idempotency_key="idem_test_001",
            confirmed_by_customer=False,
        )
        with pytest.raises(BackendIntegrationError) as exc2:
            await backend.confirm_and_lock_order(unconfirmed)
        assert exc2.value.code == "CONFIRMATION_REQUIRED"

        # Confirmed + Idempotency retry check
        confirmed = unconfirmed.model_copy(update={"confirmed_by_customer": True})
        r1 = await backend.confirm_and_lock_order(confirmed)
        r2 = await backend.confirm_and_lock_order(confirmed)
        assert r1["id"] == r2["id"]
        assert r2.get("duplicate") is True

    asyncio.run(_run())


def test_message_debouncer_merges_fragmented_whatsapp_bursts():
    async def _run():
        debouncer = MessageDebouncer(window_seconds=0.05)
        received: list[str] = []

        async def handler(phone: str, combined: str):
            received.append(combined)
            return combined

        t1 = asyncio.create_task(debouncer.enqueue("01011111111", "مساء الخير", handler))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(debouncer.enqueue("01011111111", "عايز اتنين كريب كريسبي", handler))
        await asyncio.sleep(0.01)
        t3 = asyncio.create_task(debouncer.enqueue("01011111111", "من غير بصل", handler))

        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        assert len(received) == 1
        assert "عايز اتنين كريب كريسبي" in received[0]
        assert "من غير بصل" in received[0]
        assert r1 == r2 == r3

    asyncio.run(_run())
