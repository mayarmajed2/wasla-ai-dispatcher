"""Multi-turn Egyptian Arabic restaurant dialogue & order-safety test suite.

Covers realistic multi-turn restaurant flows against MOKA's real 145-item menu:
1. Ambiguous request -> clarification -> selection -> addition -> total check -> confirm request -> customer details -> confirmed order.
2. Order-safety guarantee: never confirm an order without explicit customer confirmation.
3. Mid-dialogue cancellation clears cart cleanly without creating a confirmed order.
4. Misspelled multi-turn recovery + ambiguous drink clarification -> selection -> confirm.
5. Nonexistent item rejection mid-dialogue without corrupting existing cart items.
"""

import asyncio
from decimal import Decimal
from pathlib import Path

from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.understanding.menu_index import MenuIndex

ROOT = Path(__file__).resolve().parent.parent.parent
MENU_PATH = ROOT / "menu" / "menu.json"


def _make_dispatcher() -> tuple[WaslaAIDispatcher, MockBackendAdapter]:
    cfg = WaslaConfig(gemini_api_key="")
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    return WaslaAIDispatcher(cfg, idx, backend), backend


def test_multi_turn_ambiguous_clarify_add_total_and_confirm():
    """Full realistic dialogue: ambiguous burger -> clarify -> add crepe -> add pizza -> total -> confirm -> details."""

    async def _run():
        dispatcher, backend = _make_dispatcher()
        phone = "01099000001"

        # Turn 1: Ambiguous category request
        r1 = await dispatcher.handle_text(phone, "عايزة برجر")
        assert r1.requires_clarification is True
        assert len(backend._orders) == 0

        # Turn 2: Order confirmed menu item (كريب كوردن بلو = 140.00 EGP)
        r2 = await dispatcher.handle_text(phone, "عايزة كريب كوردن بلو")
        assert r2.cart is not None and len(r2.cart.items) == 1
        assert len(backend._orders) == 0

        # Turn 3: Add pizza (بيتزا مارجريتا وسط = 125.00 EGP)
        r3 = await dispatcher.handle_text(phone, "وزود بيتزا مارجريتا وسط")
        assert r3.cart is not None and len(r3.cart.items) == 2
        subtotal_before_confirm = Decimal(str(r3.cart.subtotal))
        assert subtotal_before_confirm == Decimal("265.00")
        assert len(backend._orders) == 0  # Still unconfirmed

        # Turn 4: Ask for total without confirming
        r4 = await dispatcher.handle_text(phone, "المجموع كام؟")
        assert r4.intent.value in ("get_cart", "ask_price", "add_to_cart", "clarify")
        assert len(backend._orders) == 0  # Asking total must NOT confirm order

        # Turn 5: Explicit confirmation without delivery details -> asks for name, address, backup phone
        r5 = await dispatcher.handle_text(phone, "تمام أكد الأوردر")
        assert r5.intent.value == "confirm_order"
        assert r5.requires_clarification is True
        assert r5.clarification_reason == "missing_order_details"
        assert len(backend._orders) == 0

        # Turn 6: Provide required customer details + confirm -> order confirmed deterministically
        r6 = await dispatcher.handle_text(
            phone,
            "الاسم: سارة محمد، العنوان: المعادي شارع 9 عمارة 12، رقم بديل: 01012345678 تمام أكد الطلب",
        )
        assert r6.order_id is not None
        assert len(backend._orders) == 1

    asyncio.run(_run())


def test_multi_turn_never_confirms_without_explicit_confirmation():
    """Safety invariant: adding items + questions must NEVER finalize an order."""

    async def _run():
        dispatcher, backend = _make_dispatcher()
        phone = "01099000002"

        r1 = await dispatcher.handle_text(phone, "عايزة بيتزا مارجريتا وسط")
        assert r1.cart is not None and len(r1.cart.items) == 1

        r2 = await dispatcher.handle_text(phone, "في عروض النهاردة؟")
        assert r2.order_id is None
        assert len(backend._orders) == 0

        r3 = await dispatcher.handle_text(phone, "التوصيل بياخد قد ايه؟")
        assert r3.order_id is None
        assert len(backend._orders) == 0

    asyncio.run(_run())


def test_multi_turn_cancellation_leaves_zero_confirmed_orders():
    """Mid-dialogue cancellation must not leave a confirmed order."""

    async def _run():
        dispatcher, backend = _make_dispatcher()
        phone = "01099000003"

        r1 = await dispatcher.handle_text(phone, "عايزة كريب كوردن بلو")
        assert r1.cart is not None and len(r1.cart.items) == 1

        r2 = await dispatcher.handle_text(phone, "لا خلاص الغي الطلب كله")
        assert r2.intent.value == "cancel_order"
        assert len(backend._orders) == 0

    asyncio.run(_run())


def test_multi_turn_ambiguous_drink_clarification_and_addition():
    """Ambiguous drink request clarifies variants; specific follow-up adds to cart cleanly."""

    async def _run():
        dispatcher, backend = _make_dispatcher()
        phone = "01099000004"

        r1 = await dispatcher.handle_text(phone, "عايزة بيتزا مارجريتا وسط")
        assert r1.cart is not None and len(r1.cart.items) == 1

        # Generic milkshake triggers specific MOKA variant clarification
        r2 = await dispatcher.handle_text(phone, "وزود ميلك شيك فواكه")
        assert r2.requires_clarification is True
        assert r2.clarification_reason == "ambiguous_product"

        # Customer selects exact variant
        r3 = await dispatcher.handle_text(phone, "ميلك شيك فراولة")
        assert r3.cart is not None and len(r3.cart.items) == 2
        assert Decimal(str(r3.cart.subtotal)) > Decimal("125.00")
        assert len(backend._orders) == 0

    asyncio.run(_run())


def test_multi_turn_nonexistent_item_rejected_without_corrupting_cart():
    """Asking for a nonexistent item mid-dialogue must not remove valid items or add fake items."""

    async def _run():
        dispatcher, backend = _make_dispatcher()
        phone = "01099000005"

        r1 = await dispatcher.handle_text(phone, "عايزة كريب كوردن بلو")
        assert r1.cart is not None and len(r1.cart.items) == 1
        original_subtotal = Decimal(str(r1.cart.subtotal))

        r2 = await dispatcher.handle_text(phone, "وزود سوشي سلمون")
        matched_ids = [i.product_id for i in r2.extracted_items if i.product_id]
        assert matched_ids == []
        assert r2.requires_clarification is True
        assert r2.clarification_reason == "unknown_product"

        # Cart remains intact with only the valid item
        assert r2.cart is not None and len(r2.cart.items) == 1
        assert Decimal(str(r2.cart.subtotal)) == original_subtotal

    asyncio.run(_run())
