# Wasla AI — Phase B Implementation Report

**Project:** Wasla AI — Commercial Restaurant AI Dispatcher (`MOKA Restaurant & Cafe`, 145 Menu Items)  
**Repository:** `https://github.com/mayarmajed2/wasla-ai-dispatcher`  
**Phase Scope:** AI architecture repair, LoRA inference activation, Gemini Function Calling & fallback handling, PostgreSQL-durable state & menu sync, concurrency isolation, durable inbound debouncing, Telegram UX code changes, scientific evaluation, and regression suite hardening — **with zero model retraining**. This report does not imply live provider or end-to-end Telegram validation.

---

## 1. Confirmed Phase A Issues & UX Defects Resolved

| Finding / Defect | Root Cause in Phase A | Phase B Repair Implemented | Primary Verification Evidence |
| :--- | :--- | :--- | :--- |
| **A-01:** CAMeLBERT LoRA never called in dispatcher; `torch`/`transformers`/`peft` missing from `requirements.txt` | `WaslaAIDispatcher._run_nlu()` skipped `self.finetuned_intent.predict()`; deps omitted | Pinned CPU PyTorch + `transformers` + `peft` + `safetensors` + `accelerate` in `requirements.txt` & `deploy/hf_requirements.txt`; added thread-safe `_MODEL_CACHE`, lazy loading, offline-first cache check, confidence-routed `predict()` in `_run_nlu()`, and real `ai_models` status on `/api/ready` | `test_dispatcher_uses_finetuned_intent_classifier`, `test_lora_v1_v2_loading_caching_and_ready_honesty`, `test_low_confidence_prediction_never_creates_or_confirms_order` |
| **A-02:** Gemini Function Calling declarations omitted from API request | `GeminiLLMClient.generate_json()` sent no `tools`/`functionDeclarations` | Injected `tools=[{"functionDeclarations": _build_gemini_function_declarations()}]` + `toolConfig` in `llm_client.py`; added `ToolExecutor` validation, single-turn deduplication (`executed_signatures`), and `_sanitize_item_dict()` against `MenuIndex` | `test_gemini_function_declarations_are_in_client_payload`, `test_gemini_function_calling_validation_dedup_and_hallucination_rejection`; mocked HTTP coverage only, not a live-provider verification |
| **A-03:** Primary Gemini transport/API/JSON errors skipped configured fallback | Only `confidence < 0.72` triggered fallback; exceptions jumped directly to `OfflineEgyptianNLU` | `_run_nlu()` invokes `use_fallback_model=True` across HTTP errors, timeouts, rate limits (`429`), invalid JSON, malformed tool calls, and `confidence < 0.72` with bounded retries before degrading to Local Hybrid | `test_primary_gemini_transport_failure_tries_configured_fallback`, `test_gemini_primary_to_fallback_on_all_error_modes` |
| **A-04:** Session state & `SESSION_TTL_SECONDS` not durable or restored across restarts/workers | `self._sessions` and `MockBackendAdapter` were process-local RAM; `ChatSession.pending_cart` write-only | Serialized 14+ fields (`SessionState` items, sizes, modifiers, notes, customer details, `last_intent`, clarification, upsell state, history, confirmation, takeover, `session_epoch`, `updated_at`) to `ChatSession` in PostgreSQL (`WITH FOR UPDATE`) & SQLite with `SESSION_TTL_SECONDS` expiration | `test_new_dispatcher_instance_restores_previous_cart_session`, `test_persistent_session_all_14_fields_restart_and_ttl` |
| **A-05:** Shared mutable request metadata on `WaslaAIDispatcher` | `_current_source` / `_current_is_test` stored on `self` could be overwritten across concurrent turns | Replaced mutable instance attributes with frozen `TurnExecutionContext` passed explicitly and protected turns with per-customer `asyncio.Lock` (`_session_locks`) + idempotent confirmation | `test_concurrent_customers_isolation_and_idempotent_confirm` |
| **A-06:** Same product in two sizes/variants silently dropped (`seen_ids`) | `OfflineEgyptianNLU._extract_items()` skipped repeating `product_id` | Switched to exact `(product_id, size_code, tuple(modifiers))` deduplication, preserving multiple sizes/extras, newline-split notes, elliptical follow-ups, and self-corrections (`"لأ، خليها كبيرة"`) | `test_same_product_requested_in_two_sizes_is_not_dropped`, `test_multi_variant_self_correction_fragmented_and_gibberish` |
| **A-07:** Price inquiry ignored `is_available=False` | Availability flag returned by tool was discarded in rendering | Price inquiry respects availability flags from synced `MenuIndex` and explicitly reports `"غير متوفر حالياً"` for out-of-stock products | `test_price_inquiry_must_not_claim_unavailable_item_as_normal_menu_item`, `test_multi_product_price_inquiry_mixed_availability` |
| **A-08:** Multi-product price inquiry answered only `items[0]` | Dispatcher formatted only the first matched item | Updated `_handle_price_inquiry()` & `persona.render_multi_price_reply()` to format every matched product, size price, and availability state | `test_price_inquiry_returns_all_requested_products`, `test_multi_product_price_inquiry_mixed_availability` |
| **A-09 & A-10:** Static JSON menu prices & lost availability toggles after restart | `MenuIndex` read static JSON once; `MockBackendAdapter._unavailable_ids` reset on restart | Added `MenuIndex.apply_db_items()` + `_DirectDBOrderBridge.sync_menu_sync()` / `revalidate_cart()` on startup, every dashboard edit (`PATCH /price`, `PATCH /availability`), cart updates, and pre-confirmation DB locking so customer cart total matches persisted `Order.total_amount` (`Decimal`) | `test_live_price_edit_sync_to_ai_and_db_order_total_plus_availability` |
| **A-11:** 3-second `MessageDebouncer` orphaned from inbound webhook | Inbound webhook called `ai_dispatcher.handle_text()` directly | Added durable `InboundMessageBuffer` table (`0004_phase_b_durable_ai_state`), wired `MessageDebouncer.enqueue_with_metadata()` with 0ms urgent bypass for CRITICAL/HIGH complaints, chronological sort, and startup crash recovery (`recover_unflushed_inbound_messages`) | `test_inbound_webhook_debouncer_urgent_bypass_and_crash_recovery`, `test_alembic_0001_to_0004_migrations_and_downgrade` |
| **Live Telegram UX Fixes:** Keysmash echoed (`"uhdd"`) & terse menu replies | Latin/arbitrary tokens unmatched on menu triggered missing-item apology | Added `is_gibberish_input()` (`"uhdd"` -> polite `unclear_request`) and category-aware `persona.render_menu_overview()` | `test_multi_variant_self_correction_fragmented_and_gibberish`, existing unit/dialogue tests |

---

## 2. Modified & Created Files Manifest

### Core AI & Backend Implementation
- **Modified:**
  - `ai_service/config.py` — LoRA settings (`lora_enabled`, `lora_checkpoint_dir`, `lora_prefer_gpu`) and `.env` `"disabled"` sentinel handling
  - `ai_service/models/finetuned_intent.py` — Lazy load, thread-safe `_MODEL_CACHE`, offline-first HF loading, CPU/GPU auto-detection, telemetry/status API
  - `ai_service/models/llm_client.py` — Native Gemini `tools` / `functionDeclarations`, bounded retries, `functionCall` parsing, `_sanitize_item_dict()`, multi-variant item extraction & newline-first segmentation
  - `ai_service/dispatcher/dispatcher.py` — Hybrid LoRA + Gemini + Egyptian NLU routing, primary/fallback error cascade, `TurnExecutionContext`, per-phone `asyncio.Lock`, 14-field session rehydration/TTL, live menu sync hooks, multi-product/availability price handling, idempotent order confirmation, deadlock-safe `set_human_takeover(persist=...)`
  - `ai_service/dispatcher/debounce.py` — `enqueue_with_metadata()`, chronological `(ts, seq)` merging, urgent complaint 0ms bypass, `flush_all()`
  - `ai_service/dispatcher/persona.py` — Multi-product/mixed-availability price formatting (`render_multi_price_reply`), rich `render_menu_overview()`, natural upsell decline phrasing
  - `ai_service/schemas/models.py` — Extended `SessionState` persistence schema & frozen `TurnExecutionContext`
  - `ai_service/tools/registry.py` — Strict tool validation & single-turn execution deduplication (`compute_tool_call_signature`)
  - `ai_service/understanding/arabic_nlp.py` — Gibberish/keysmash detector (`is_gibberish_input`), checkout/greeting filler stripping, enhanced self-correction cues
  - `ai_service/understanding/menu_index.py` — Live SQL synchronization (`apply_db_items()`, `all_products()`)
  - `ai_service/adapters/mock_backend.py` — 14-field in-memory persistence/TTL parity & availability/price revalidation helpers
  - `backend/app/models.py` — Added `InboundMessageBuffer` SQLAlchemy model
  - `backend/app/main.py` — Truthful `/api/ready` (`ai_models` telemetry), PostgreSQL row-locked 14-field `ChatSession` bridge, live menu cache sync, durable `InboundMessageBuffer` + debounced webhook pipeline + startup recovery
- **Created:**
  - `backend/migrations/versions/0004_phase_b_durable_ai_state.py` — Alembic migration creating/dropping `inbound_message_buffers`

### Packaging, CI/CD, Testing, Evaluation & Documentation
- **Modified:** `requirements.txt`, `deploy/hf_requirements.txt`, `Dockerfile`, `.github/workflows/ci.yml`, `.github/workflows/deploy-hf-space.yml`, `README.md`
- **Created:**
  - `pytest.ini`
  - `WASLA_AI_PHASE_A_AUDIT_REPORT.md` & `wasla_phase_a_repro_tests.py`
  - `tests/integration/test_phase_b_architecture_hardening.py` (11 comprehensive hardening tests)
  - `training/scripts/evaluate_phase_b_scientific.py`
  - `training/reports/phase_b_scientific_evaluation.json`
  - `docs/AI_ARCHITECTURE.md`
  - `docs/MODEL_INFERENCE_REPORT.md`
  - `docs/REGRESSION_TEST_REPORT.md`
  - `docs/DEPLOYMENT_READINESS.md`
  - `docs/PHASE_B_IMPLEMENTATION_REPORT.md`

---

## 3. Regression & Scientific Verification Summary

| Verification Run | Command | Actual Output |
| :--- | :--- | :--- |
| **Phase A Baseline (Before Fixes)** | `py -m pytest -q wasla_phase_a_repro_tests.py` | **7 failed in 1.19s** |
| **Historical Phase B Suite (24 Nov 2026)** | `$env:GEMINI_API_KEY="disabled"; $env:GROQ_API_KEY="disabled"; py -m pytest -v wasla_phase_a_repro_tests.py tests/ backend/test_cart_calculator.py --tb=short` | **61 passed in 143.59s (0:02:23)** at that earlier checkpoint; this is not the current task's test result. See the current [`FINAL_ENGINEERING_REPORT.md`](../FINAL_ENGINEERING_REPORT.md). |
| **5-Way Scientific Eval (`2026-11-24T19:21:18Z`)** | `py training/scripts/evaluate_phase_b_scientific.py` | Held-Out `test_unseen_106`: Base **39.62%** -> LoRA v1 **79.25%** -> LoRA v2 **82.08%** -> Hybrid **83.02%** (`Macro-P 0.9694`, `Macro-F1 0.7470`); Independent 204: LoRA v2 **60.29%** (`Macro-F1 0.5890`), Gemini 24-sample probe **83.33%**; MOKA Typo 154: **100.00%** intent / **97.52%** exact item / **0.0%** nonexistent FP |

*(Detailed confusion matrices and per-intent metrics are recorded in [`docs/MODEL_INFERENCE_REPORT.md`](docs/MODEL_INFERENCE_REPORT.md) and [`training/reports/phase_b_scientific_evaluation.json`](training/reports/phase_b_scientific_evaluation.json).)*

---

## 4. Honest Remaining External Limitations
1. **External channels/providers:** Live WhatsApp, Telegram, Gemini, and Groq end-to-end behavior has not been verified by this task. Previous claims in this report are historical notes, not current live-readiness evidence; mocked HTTP/provider tests are not live integration.
2. **Gemini quotas:** The primary -> fallback -> local degradation path is covered by mocked/provider-stub tests. Live rate limits and customer delivery behavior remain unverified here.
3. **Cold Container Base Weights:** PEFT LoRA v1 & v2 adapter files are bundled directly in `training/checkpoints/`, while the ~436 MB `CAMeL-Lab/bert-base-arabic-camelbert-mix` base encoder is downloaded lazily into `HF_HOME` on first LoRA use if not pre-cached; in a strictly air-gapped cold start without base weights cached, `FineTunedIntentClassifier` fails safely and keeps the dispatcher operational via `OfflineEgyptianNLU`.

---

## 5. Historical GitHub & Deployment Checkpoint
- **GitHub Repository:** `https://github.com/mayarmajed2/wasla-ai-dispatcher` (default branch `main`)
- **Prior Phase B Commit:** `2e45722` (`2e457228a5ef081e90746435b4cb55fd03701497`)
- **Cloud deployment claims:** A configured `/api/ready` response or CI/Docker check is not equivalent to a live customer/provider end-to-end test. Current-task branch, commit, tests, and limitations are documented in [`FINAL_ENGINEERING_REPORT.md`](../FINAL_ENGINEERING_REPORT.md).
