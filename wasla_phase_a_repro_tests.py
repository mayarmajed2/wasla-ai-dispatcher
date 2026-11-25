"""Phase A regression reproductions for Wasla AI.

Run from the repository root:
    pytest -q /path/to/wasla_phase_a_repro_tests.py

These tests express expected behavior. On the audited ZIP, they are expected
to FAIL until the corresponding defects are fixed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(os.environ.get("WASLA_REPO_ROOT", Path.cwd())).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.models.llm_client import GeminiLLMClient, OfflineEgyptianNLU
from ai_service.tools.registry import TOOL_DECLARATIONS
from ai_service.understanding.menu_index import MenuIndex

MENU_PATH = ROOT / "menu" / "menu.json"


def make_index():
    return MenuIndex(MENU_PATH)


def test_same_product_requested_in_two_sizes_is_not_dropped():
    nlu = OfflineEgyptianNLU(make_index())
    parsed = nlu.parse("عايز بيتزا مارجريتا وسط وبيتزا مارجريتا كبير")
    assert len(parsed["items"]) == 2, parsed["items"]
    assert [x["size_code"] for x in parsed["items"]] == ["M", "L"]


@pytest.mark.asyncio
async def test_price_inquiry_must_not_claim_unavailable_item_as_normal_menu_item():
    idx = make_index()
    backend = MockBackendAdapter(idx)
    backend.set_item_unavailable("item_029", True)
    dispatcher = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, backend)
    response = await dispatcher.handle_text("01011112222", "بكام بيتزا مارجريتا وسط")
    assert "غير متوفر" in response.reply_ar or "مش متاح" in response.reply_ar, response.reply_ar


@pytest.mark.asyncio
async def test_price_inquiry_returns_all_requested_products():
    idx = make_index()
    backend = MockBackendAdapter(idx)
    dispatcher = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, backend)
    response = await dispatcher.handle_text(
        "01011112223", "بكام بيتزا مارجريتا وسط وكريب كوردن بلو"
    )
    assert "مارجريتا" in response.reply_ar and "كوردن بلو" in response.reply_ar, response.reply_ar


@pytest.mark.asyncio
async def test_primary_gemini_transport_failure_tries_configured_fallback():
    class FakeLLM:
        is_configured = True

        def __init__(self):
            self.calls = []

        async def generate_json(self, system_prompt, user_prompt, use_fallback_model=False):
            self.calls.append(use_fallback_model)
            if not use_fallback_model:
                raise RuntimeError("simulated primary HTTP 503")
            return {
                "intent": "greeting",
                "confidence": 0.95,
                "items": [],
                "requires_clarification": False,
            }, "fallback-model"

    idx = make_index()
    fake = FakeLLM()
    dispatcher = WaslaAIDispatcher(
        WaslaConfig(gemini_api_key="test-key"),
        idx,
        MockBackendAdapter(idx),
        llm_client=fake,
    )
    _, model_used = await dispatcher._run_nlu("السلام عليكم")
    assert fake.calls == [False, True], fake.calls
    assert model_used == "fallback-model"


@pytest.mark.asyncio
async def test_new_dispatcher_instance_restores_previous_cart_session():
    idx = make_index()
    backend = MockBackendAdapter(idx)
    first = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, backend)
    first_response = await first.handle_text("01099990000", "عايز كريب كوردن بلو")
    assert len(first_response.cart.items) == 1

    second = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, backend)
    restored = second.get_session("01099990000")
    assert len(restored.cart.items) == 1, restored.cart.model_dump()


@pytest.mark.asyncio
async def test_dispatcher_skips_incompatible_finetuned_intent_classifier():
    idx = make_index()
    dispatcher = WaslaAIDispatcher(
        WaslaConfig(gemini_api_key=""), idx, MockBackendAdapter(idx)
    )
    calls = []
    dispatcher.finetuned_intent.predict = lambda text: (
        calls.append(text) or ("greeting", 0.99, 1.0)
    )
    parsed, model_used = await dispatcher._run_nlu("السلام عليكم")
    assert calls == [], "Incompatible classifier must not route application intents"
    assert parsed["intent"] == "greeting"
    assert model_used == "offline_egyptian_nlu"


def test_gemini_function_declarations_are_in_client_payload():
    import inspect

    source = inspect.getsource(GeminiLLMClient.generate_json)
    assert TOOL_DECLARATIONS
    assert '"tools"' in source or "'tools'" in source, (
        "Gemini generateContent payload does not include tools/functionDeclarations"
    )
