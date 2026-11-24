# Wasla AI Regression Test Report

**Date:** 24 November 2026
**Branch:** `mayarmajed2-remaining-fallback-fixes`
**Scope:** Category-aware menu search, verified order tracking, availability/rejection distinctions, and fallback safety.

## Current Full-Suite Result

```text
py -m pytest -q wasla_phase_a_repro_tests.py tests\ backend\test_cart_calculator.py --tb=short
87 passed, 2 skipped in 117.02s
```

The two skips are the real LoRA inference checks in `tests/ai/test_finetuned_lora_model.py` and `tests/integration/test_phase_b_architecture_hardening.py`. Both detect Windows Smart App Control blocking the `regex._regex` native dependency and skip. They are not successful inference results. No test failed.

The final targeted regression/dialogue/database selection passed **28/28** in 24.47 seconds. The clean Python 3.14 environment installed `requirements.txt` successfully, but pytest collection there is blocked when Windows Smart App Control prevents importing `pydantic_core._pydantic_core`.

## Five Customer-Conversation Regressions

`tests/regression/test_critical_dispatcher_regressions.py` includes:

1. `عندكو عصير ايه؟` and `عندكو باستا ايه؟` call `get_active_menu` with the requested category and leave the cart unchanged.
2. Both `الأوردر طلع ولا لسه؟` and `الأوردر اتحرك ولا لسه؟` with no matching order ask for an order number and never call `resolve_delivery`.
3. Tracking for an existing customer returns only that customer's stored `OUT_FOR_DELIVERY` status; a foreign customer's record is not returned, and a missing/unknown status fails closed without an ETA.
4. `مفيش ميلك شيك؟`, `مش عايز ميلك شيك`, `شيل الميلك شيك`, and `الميلك شيك مش موجود` exercise availability lookup, non-mutating rejection, cart removal, and a verified availability response separately. A specific item marked unavailable is checked against backend data.
5. Provider-stub matrix: Gemini configured/available with LoRA disabled; Gemini configured but unavailable with LoRA disabled; and both providers unavailable. Malicious add-to-cart output and unresolved questions do not mutate the cart; order-status questions never route to delivery fees.

`tests/integration/test_order_webhook_reliability.py` additionally verifies that the production DB bridge selects the session's active order only for the matching phone and rejects an order ID belonging to another phone. This is a local SQLite test, not a live PostgreSQL or customer-channel integration.

## Clean-Environment and Live-Provider Limits

- `pip install -r requirements.txt` completed successfully in a new Python 3.14 venv.
- Tests in that venv stop during collection because the native `pydantic_core._pydantic_core` extension is blocked by Windows Smart App Control. Existing installed-environment tests run successfully except for the two explicit LoRA skips above.
- Real LoRA inference could not run: the Windows runtime blocks `regex._regex`; WSL is not installed and Docker is unavailable.
- No live Gemini, Groq, Telegram, WhatsApp/Meta, production PostgreSQL, or delivery-provider call was made. Provider stubs and mocked HTTP tests are not live integration evidence.

The earlier Phase B report's 61-test result is a historical checkpoint, not the current suite result. No production-readiness or live Telegram end-to-end claim is made here.

**Implementation commit:** `e3e4a52dffd931ffef45633d64a731bcef53983c`
**Push/PR:** None.
