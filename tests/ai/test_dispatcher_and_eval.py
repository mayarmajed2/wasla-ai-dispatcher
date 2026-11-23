"""Tier 3 & 4 — AI Dispatcher Evaluation Suite, Multi-Turn Context & Cashier Directives."""

import asyncio
import json
from decimal import Decimal
from pathlib import Path

from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.schemas.models import CashierDirectiveRequest
from ai_service.understanding.menu_index import MenuIndex

ROOT = Path(__file__).resolve().parent.parent.parent
MENU_PATH = ROOT / "menu" / "menu.json"
EVAL_PATH = ROOT / "datasets" / "evaluation" / "moka_eval_cases.json"


def _make_dispatcher() -> WaslaAIDispatcher:
    cfg = WaslaConfig(gemini_api_key="")  # Deterministic offline CI mode
    idx = MenuIndex(MENU_PATH)
    backend = MockBackendAdapter(idx)
    return WaslaAIDispatcher(cfg, idx, backend)


def test_moka_eval_dataset_accuracy():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    async def _run():
        passed = 0
        for case in cases:
            dispatcher = _make_dispatcher()
            resp = await dispatcher.handle_text(f"010999{case['id'][-3:]}", case["utterance"])
            assert resp.intent.value == case["expected_intent"], f"Failed intent on {case['id']}: {resp.intent}"
            assert resp.requires_clarification == case["requires_clarification"], f"Failed clarification on {case['id']}"
            if "expected_clarification_reason" in case:
                assert resp.clarification_reason == case["expected_clarification_reason"]
            passed += 1
        assert passed == len(cases)

    asyncio.run(_run())


def test_multi_turn_clarification_and_full_order_journey():
    dispatcher = _make_dispatcher()
    phone = "01012345678"

    async def _run():
        # Turn 1: Missing size -> asks clarification
        r1 = await dispatcher.handle_text(phone, "عايز بيتزا بيبروني")
        assert r1.requires_clarification is True
        assert r1.clarification_reason == "MISSING_OR_INVALID_SIZE"

        # Turn 2: Customer replies "كبير" -> resolves pending clarification and adds L (200 EGP)
        r2 = await dispatcher.handle_text(phone, "كبير")
        assert r2.requires_clarification is False
        assert r2.cart.subtotal == Decimal("200.00")

        # Turn 3: Delivery resolution
        r3 = await dispatcher.handle_text(phone, "التوصيل بكام للمعادي؟")
        assert r3.cart.delivery_fee == Decimal("20.00")
        assert r3.cart.total_amount == Decimal("220.00")

        # Turn 4: Confirm without customer details -> asks for name, address, backup_phone
        r4 = await dispatcher.handle_text(phone, "تمام اكد الطلب")
        assert r4.requires_clarification is True
        assert r4.order_id is None

        # Turn 5: Provide details + confirm
        r5 = await dispatcher.handle_text(
            phone,
            "الاسم: كريم محمود، العنوان: المعادي شارع 9 عمارة 14، رقم بديل: 01122334455 تمام اكد الطلب",
        )
        assert r5.order_id is not None
        assert r5.requires_clarification is False

    asyncio.run(_run())


def test_cashier_directives_safety_rules():
    dispatcher = _make_dispatcher()

    # Kitchen delay without explicit minutes must NOT auto-assume 15 minutes
    d_delay = dispatcher.interpret_cashier_directive(
        CashierDirectiveRequest(order_id=5, directive_type="kitchen_delay", directive_text="تأخير في المطبخ")
    )
    assert d_delay.validated is True
    assert "15" not in d_delay.customer_message_ar
    assert len(d_delay.warnings) >= 1

    # Free-text substitution with "بنفس السعر" must validate menu item and flag cashier approval
    d_sub = dispatcher.interpret_cashier_directive(
        CashierDirectiveRequest(
            order_id=5,
            directive_type="free_text",
            directive_text="الاستربس خلصان، اعرض زنجر بنفس السعر",
        )
    )
    assert d_sub.validated is True
    assert d_sub.suggested_alternative_product_id == "item_006"
    assert d_sub.requires_cashier_approval is True
