# Wasla AI — Repaired Production AI Architecture (Phase B)

**System:** Wasla AI Commercial Restaurant Dispatcher (MOKA Pilot — 145 Verified Menu Items)  
**Design Principle:** Hybrid NLU understanding paired with strictly deterministic PostgreSQL pricing, availability, cart arithmetic (`Decimal`), and order confirmation gates. LLMs and neural classifiers never autonomously set prices, invent menu items, or confirm financial transactions without explicit customer consent.

---

## 1. End-to-End Request & Inference Flow

```text
Customer Message (WhatsApp Webhook / Telegram Bot / API)
  │
  ▼ [Meta Webhook only]
HMAC-SHA256 Check + ProcessedMessage Dedup + InboundMessageBuffer (DB: BUFFERED)
  │
  ▼ [urgent CRITICAL/HIGH complaint -> 0ms bypass | normal burst -> 3s debounce]
MessageDebouncer.enqueue_with_metadata()
  │
  ▼
WaslaAIDispatcher.handle_text(phone, text, data_source, is_test)
  ├── Acquires per-phone asyncio.Lock + creates immutable TurnExecutionContext
  ├── Syncs MenuIndex & availability from PostgreSQL (`_DirectDBOrderBridge.sync_menu_sync`)
  ├── Hydrates 14-field SessionState from DB (`ChatSession`), checking SESSION_TTL_SECONDS
  ├── [Gate 0] Human Takeover Active? -> Suppress AI reply cleanly + escalate CRITICAL updates
  ├── [Gate 1] Security Guardrail -> Block prompt-injection / unauthorized discount attempts
  ├── [Gate 2] High-Recall Complaint Detector -> Escalate CRITICAL food safety / HIGH immediately
  ├── [Gate 3] Pending Upsell / Clarification Resolution (memory of prior turns)
  │
  ▼ [NLU Orchestration: `_run_nlu`]
  ├── 1) FineTunedIntentClassifier.predict(text)  [CAMeLBERT-Mix PEFT LoRA v2, cached]
  ├── 2) OfflineEgyptianNLU.parse(text)          [MenuIndex fuzzy/phonetic/multi-variant + corrections]
  ├── 3) If GEMINI_API_KEY configured:
  │      ├── Primary (`gemini-3.5-flash-lite`) with `tools=[{"functionDeclarations": TOOL_DECLARATIONS}]`
  │      └── On error / timeout / 429 / malformed JSON/tool / conf < 0.72 -> Fallback (`gemini-3.8-flash`)
  └── 4) Confidence-Routed Hybrid Arbitration:
         - MenuIndex item matches, ambiguities & self-corrections always authoritative
         - Non-destructive LoRA intents (`greeting`, `menu_inquiry`, `complaint`, `request_human` >= 0.55;
           `delivery_inquiry`, `price_inquiry`, `unclear_request` >= 0.72) when no menu match exists
         - Confirm order blocked if confidence < 0.55 or missing explicit approval
  │
  ▼ [Deterministic Action Layer: `ToolExecutor`]
  ├── Validates tool name (`ALLOWED_TOOL_NAMES`), product_ids against MenuIndex, qty (1..50)
  ├── Deduplicates identical tool executions within the same turn (`executed_signatures`)
  ├── Revalidates DB stock & live Decimal prices before `confirm_and_lock_order`
  ├── Computes idempotency key per `(phone, session_epoch, cart_signature)` -> prevents duplicate orders
  └── Persists updated SessionState + CartSnapshot + history (<=30 turns) back to PostgreSQL `ChatSession`
```

---

## 2. Core Phase B Architectural Components

### A. CAMeLBERT PEFT LoRA Runtime (`ai_service/models/finetuned_intent.py`)
- **Checkpoints Preserved:**
  - `training/checkpoints/camelbert-wasla-lora-v2` (default: rank 16, alpha 32, target `query`/`value`/`key`, 12 intents)
  - `training/checkpoints/camelbert-wasla-lora` (v1: rank 8, alpha 16, target `query`/`value`, preserved for comparison)
- **Lazy Loading & Thread-Safe Cache:** Startup stays sub-second (`/api/ready` reports `checkpoint_exists: true`, `model_loaded: false` until first inference or explicit `.load()`). Once loaded, `_MODEL_CACHE[(ckpt, base_model, device)]` reuses weights in `<2 ms`.
- **Hardware Adaptation:** Uses `cuda` if `torch.cuda.is_available()` and `WASLA_LORA_PREFER_GPU=true`, else CPU. Tries `local_files_only=True` before hitting Hugging Face Hub.

### B. Gemini Native Function Calling & Resilient Fallback (`ai_service/models/llm_client.py`)
- **Tool Declarations in Payload:** `GeminiLLMClient.generate_json()` injects `"tools": [{"functionDeclarations": ...}]` and `"toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}` derived from `ai_service/tools/registry.py`.
- **Bounded Retries & Fallback Matrix:** Retries transient `TimeoutException`/`TransportError`/HTTP `429, 500, 502, 503, 504` once (`0.25s`), validates every `functionCall` part against `ALLOWED_TOOL_NAMES`, and triggers the configured fallback model (`ai_fallback_model`) on HTTP errors, timeouts, invalid JSON, malformed function calls, or `confidence < 0.72`, before safely falling back to Local Hybrid.
- **Hallucination Sanitization:** `_sanitize_item_dict()` drops any invented price/discount keys and rejects any `product_id` absent from `MenuIndex`.

### C. Durable Conversation State & Worker-Safe Concurrency (`ai_service/schemas/models.py`, `backend/app/main.py`)
- **14+ State Fields Persisted:** Cart items, quantities, sizes, modifiers, notes, `customer_name`, `delivery_address`, `backup_phone`, `last_intent`, `pending_clarification`, `pending_upsell`, `upsell_declined`/`rejected_upsell_ids`, `order_confirmed`, `confirmed_cart_sig`, `session_epoch`, `needs_human_takeover`, `history`, `updated_at`.
- **PostgreSQL Authoritative Store:** `_DirectDBOrderBridge` reads/writes `ChatSession` (using `SELECT ... FOR UPDATE` on PostgreSQL) on every turn, enforces `SESSION_TTL_SECONDS` expiration (preserving `needs_human_takeover` chats), and maintains SQLite development compatibility.
- **Zero Mutable Dispatcher Metadata:** Replaced `self._current_source` / `self._current_is_test` with immutable `TurnExecutionContext` and per-customer `asyncio.Lock` (`_session_locks`).

### D. Menu Price & Availability Synchronization (`ai_service/understanding/menu_index.py`)
- `MenuIndex.apply_db_items()` updates `backend_item_id`, `base_price`, size prices, and `is_available` from SQL `menu_items`.
- Every dashboard edit (`PATCH /api/menu/{id}/price` and `PATCH /api/menu/{id}/availability`), startup `lifespan`, price inquiry, cart add, and final order confirmation synchronizes with SQL so AI replies, cart totals, and persisted `Order.total_amount` match using `Decimal` rounding (`0.01`).

### E. Egyptian Arabic Multi-Variant Extraction & Inbound Debouncing
- Removed `seen_ids` product-dropping in `OfflineEgyptianNLU._extract_items()`, allowing identical products with distinct sizes (`وسط` + `كبير`) or distinct extras (`موتزريلا` + `هالبينو`), elliptical continuations, fragmented multi-line notes, and self-corrections (`"لأ، خليها كبيرة"`).
- `is_gibberish_input()` redirects unmapped keysmash tokens (e.g., `"uhdd"`) to polite `unclear_request` instead of echoing them as missing dishes.
- `InboundMessageBuffer` (`0004_phase_b_durable_ai_state`) durably commits incoming Meta WhatsApp text messages prior to awaiting `MessageDebouncer.enqueue_with_metadata()`, bypassing delay (`0ms`) for urgent CRITICAL/HIGH complaints and replaying orphaned `BUFFERED` messages via `recover_unflushed_inbound_messages()` on startup.
