# Wasla AI — Phase B Regression Test Report

**Date:** 24 November 2026  
**Environment:** Windows 11, Python 3.14.5 (`C:\Users\Mayoura\AppData\Local\Programs\Python\Python314\python.exe`), `pytest 9.2.0`, `pytest-asyncio 1.5.0`, `torch 2.15.0`, `transformers 5.24.0`, `peft 0.22.0`

---

## 1. Phase A Baseline Reproduction vs. Phase B Repair

| Suite Command | Pre-Fix Phase A Baseline | Post-Fix Phase B Result |
| :--- | :--- | :--- |
| `py -m pytest -q wasla_phase_a_repro_tests.py` | **7 failed** in `1.19s` | **7 passed** (included in 61/61 run) |
| `py -m pytest -v wasla_phase_a_repro_tests.py tests/ backend/test_cart_calculator.py` | 41/43 before Phase B (per Phase A audit on 3.13 without deps) | **61 passed** in `143.59s (0:02:23)` |

Zero assertions in `wasla_phase_a_repro_tests.py` or any pre-existing test file were weakened, skipped, or removed.

---

## 2. Exact Phase A Reproduction Outcomes (Before -> After)

1. `test_same_product_requested_in_two_sizes_is_not_dropped`: **FAILED -> PASSED** (`[M, L]` extracted for `"عايز بيتزا مارجريتا وسط وبيتزا مارجريتا كبير"`).
2. `test_price_inquiry_must_not_claim_unavailable_item_as_normal_menu_item`: **FAILED -> PASSED** (out-of-stock item returns `"غير متوفر"` instead of normal price prompt).
3. `test_price_inquiry_returns_all_requested_products`: **FAILED -> PASSED** (both `بيتزا مارجريتا` and `كريب كوردن بلو` included in reply).
4. `test_primary_gemini_transport_failure_tries_configured_fallback`: **FAILED -> PASSED** (`fake.calls == [False, True]`, `model_used == "fallback-model"`).
5. `test_new_dispatcher_instance_restores_previous_cart_session`: **FAILED -> PASSED** (`second.get_session("01099990000").cart.items` retains `1` item).
6. `test_dispatcher_uses_finetuned_intent_classifier`: **FAILED -> PASSED** (`dispatcher.finetuned_intent.predict` invoked on handle_text).
7. `test_gemini_function_declarations_are_in_client_payload`: **FAILED -> PASSED** (`"tools"` with `functionDeclarations` present in `GeminiLLMClient.generate_json`).

---

## 3. Historical Phase B Test Suite Run (61 / 61 Passed)

- **`wasla_phase_a_repro_tests.py` (7 passed):** All 7 verbatim Phase A reproductions.
- **`tests/integration/test_phase_b_architecture_hardening.py` (11 passed):**
  - `test_lora_v1_v2_loading_caching_and_ready_honesty`: Verifies lazy `is_loaded=False` prior to `.load()`, real v1 & v2 checkpoint inference, sub-150ms cache hit, and `/api/ready` `ai_models` reporting.
  - `test_low_confidence_prediction_never_creates_or_confirms_order`: Verifies low-confidence classifier/LLM outputs cannot mutate cart or confirm orders.
  - `test_gemini_function_calling_validation_dedup_and_hallucination_rejection`: Mocks HTTP Gemini `functionCall` payloads, asserts `"tools"` header/body, deduplicates identical `ToolExecutor` calls, and rejects hallucinated `item_999`.
  - `test_gemini_primary_to_fallback_on_all_error_modes`: Exercises HTTP `503`, HTTP `429`, `ReadTimeout`, invalid JSON, malformed `functionCall`, and `confidence < 0.72` -> all route cleanly to fallback.
  - `test_persistent_session_all_14_fields_restart_and_ttl`: Persists cart items, sizes, modifiers, notes, customer name, address, backup phone, `last_intent`, upsell rejection, `awaiting_final_confirmation`, `order_confirmed`, `active_order_id`, and `history` across fresh DB bridge + dispatcher restarts; verifies `SESSION_TTL_SECONDS` cleanup.
  - `test_concurrent_customers_isolation_and_idempotent_confirm`: Concurrent `asyncio.gather` across 2 customers with separate `data_source`/`is_test` + double concurrent confirm creates strictly 1 order.
  - `test_multi_variant_self_correction_fragmented_and_gibberish`: Verifies same product in two sizes (`M`+`L` = `275.00 EGP`), same product with two extras, `"لأ، خليها كبيرة"` self-correction (`125 -> 150 EGP`), multi-line notes, and `"uhdd"` keysmash handled as `unclear_request`.
  - `test_live_price_edit_sync_to_ai_and_db_order_total_plus_availability`: Admin edits `item_005` price to `111.50` -> AI inquiry returns `111.50` -> cart & webhook confirm produce exact DB `total_amount == 131.50` matching customer cart; toggling unavailable blocks inquiry/confirm across re-sync.
  - `test_multi_product_price_inquiry_mixed_availability`: Queries available + out-of-stock products in one sentence and reports both accurately.
  - `test_inbound_webhook_debouncer_urgent_bypass_and_crash_recovery`: Aggregates 3 rapid WhatsApp webhooks for Customer 1 into 1 AI turn & 1 reply (`220.00`), keeps Customer 2 separate (`125.00`), executes `CRITICAL` food-safety complaint immediately bypassing 3s window, and replays orphaned `InboundMessageBuffer` row on startup recovery.
  - `test_alembic_0001_to_0005_migrations_backfill_and_downgrade`: Upgrades through `0005`, verifies inbound buffering, backfills an existing order idempotency key, checks the unique index, then downgrades and re-upgrades cleanly.
- **Original Pre-Existing Suites (43 passed):**
  - `tests/ai/test_dispatcher_and_eval.py` (3 passed)
  - `tests/ai/test_finetuned_lora_model.py` (2 passed)
  - `tests/ai/test_multi_turn_dialogues.py` (5 passed)
  - `tests/ai/test_typo_tolerance_suite.py` (2 passed, including 154-case Egyptian typo benchmark)
  - `tests/end_to_end/test_real_backend_integration.py` (1 passed)
  - `tests/integration/test_mock_backend_and_debounce.py` (3 passed)
  - `tests/integration/test_production_engineering.py` (4 passed)
  - `tests/integration/test_sales_notifications_reviews_complaints.py` (6 passed)
  - `tests/integration/test_scheduler_concurrency_and_security.py` (10 passed)
  - `tests/unit/test_menu_and_security.py` (5 passed)
  - `backend/test_cart_calculator.py` (2 passed)

## 4. Cart, Order, and Telegram Persistence Regressions

`tests/integration/test_order_webhook_reliability.py` covers database-backed cart restoration and removal by exact cart-line identity; order creation using DB-stored size and modifier prices, required option validation, and idempotent retry; atomic Telegram update claims under concurrent delivery; and malformed webhook JSON returning HTTP 400.

Latest verification after these changes:

- Focused backend regressions, migration backfill, and cart math: **6 passed** in `8.44s`.
- Full command `py -m pytest -q tests/ backend/test_cart_calculator.py`: **55 passed, 2 failed** in `122.74s`. Both failures are the existing LoRA checkpoint-load tests; Windows Application Control blocks the installed `regex` native extension (`_regex`), preventing the model from loading. No backend reliability test failed.
