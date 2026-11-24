# Final Engineering Report

**Repository:** `mayarmajed2/wasla-ai-dispatcher`
**Branch:** `mayarmajed2-remaining-fallback-fixes` (based on `main`)
**Report date:** 24 November 2026

## Delivered

- Category questions such as `عندكو عصير ايه؟` and `عندكو باستا ايه؟` are classified as menu inquiries and execute category-filtered menu lookups. Missing categories are reported as absent from the current menu rather than treated as products to order.
- Added read-only `order_status` intent and `get_customer_order_status` backend tool. The production DB bridge selects the session's active order only when the stored customer phone matches; without an active ID, it selects that phone's latest order. No match asks for an order number; missing or unrecognized status is reported as unavailable. Responses use a fixed map of recognized DB statuses and never invent an ETA.
- Availability inquiries, customer reports of missing stock, product rejection, and cart removal are separate paths. Availability and reports query live menu/availability data; rejecting a broad product category does not remove a cart line. Existing explicit negation for a specific item already in the cart still removes that matching line.
- Added a cart-summary intent and a safety gate that prevents unresolved customer questions from becoming cart additions. The non-authenticated `RealBackendAdapter` has no customer-safe status endpoint and fails closed rather than returning mock order data.
- No migration was needed: the lookup uses existing `orders.customer_phone`, `orders.status`, and `chat_sessions.active_order_id`; the current migration head remains `0005_order_idempotency`.

## Verification

| Check | Result |
|---|---|
| Complete suite: `py -m pytest -q wasla_phase_a_repro_tests.py tests\\ backend\\test_cart_calculator.py --tb=short` | **87 passed, 2 skipped, 0 failed** in 117.02s |
| Final targeted regression/dialogue/database selection | **28 passed** in 24.47s |
| LoRA checkpoint/inference-focused run | **12 passed, 2 skipped**; both inference tests skipped because Windows Smart App Control blocks `regex._regex` |
| Clean Python 3.14 venv dependency install (`pip install -r requirements.txt`) | **Succeeded** |
| Test collection in that clean venv | **Blocked**: importing `pydantic_core._pydantic_core` fails because Windows Smart App Control blocks the native extension |
| Linux inference runtime check | WSL is not installed; no Docker executable was available |
| Migration verification | Existing SQLite Alembic migration-chain coverage passed in the full suite; no schema migration was added |

Real LoRA inference was **not verified**. On the installed Windows Python 3.14.5 runtime, `transformers` import fails at `regex._regex` with `DLL load failed ... An Application Control policy has blocked this file`; the clean venv is similarly blocked at `pydantic_core._pydantic_core`. The two real-checkpoint tests therefore skip rather than pass. No compatible Linux runtime was available in this environment.

## External Integration Limits

No live Gemini/Groq provider call, Telegram or WhatsApp end-to-end customer flow, production PostgreSQL lookup, or live delivery-provider tracking was executed. Provider-matrix coverage uses test doubles and does not constitute live integration evidence. The SQLite database-bridge regression verifies phone scoping and active-order selection only in a local test database. This report does not claim production readiness.

**Implementation commit:** `e3e4a52dffd931ffef45633d64a731bcef53983c`
**Push/PR:** None.
