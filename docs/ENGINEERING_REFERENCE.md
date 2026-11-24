# Wasla AI Engineering Reference

**Repository:** `mayarmajed2/wasla-ai-dispatcher`
**Date:** 24 November 2026
**Scope:** Implemented system behavior and locally verifiable QA. This document does not claim live WhatsApp, Telegram, Gemini, Groq, PostgreSQL, or customer traffic verification without credentials and successful external checks.

## System Overview

Wasla AI is a FastAPI and SQLAlchemy restaurant dispatcher for MOKA. It receives text or voice turns from WhatsApp, Telegram, or API callers, resolves intent and menu references through hybrid AI/NLU, mutates a persisted cart, requires explicit confirmation, and creates an order through a backend adapter. The cashier dashboard and notification services consume the same database.

```text
Channel
  WhatsApp / Telegram / API
        |
        v
Webhook security + message dedup + optional debounce
        |
        v
Per-phone asyncio lock + TurnExecutionContext
        |
        v
Session rehydration + human takeover + security guards
        |
        v
LoRA / deterministic Egyptian NLU / Gemini orchestration
        |
        v
Validated ToolExecutor + live MenuIndex
        |
        v
Decimal cart + explicit customer confirmation
        |
        v
Persisted order + cashier/owner notifications
```

## AI Pipeline

1. `WaslaAIDispatcher.handle_text()` acquires a per-phone lock and creates an immutable turn context.
2. The session is loaded from the backend and menu state is synchronized.
3. Human takeover and prompt-injection guards run before normal NLU.
4. High-recall complaint detection checks CRITICAL/HIGH safety signals without relying on model confidence.
5. Pending upsell or clarification state is resolved before a fresh turn.
6. CAMeLBERT-Mix PEFT LoRA classification is attempted lazily and cached by checkpoint/base/device.
7. Deterministic Egyptian Arabic NLU and `MenuIndex` extract products, variants, corrections, and gibberish.
8. Gemini function calling is used only when configured. Primary-to-fallback behavior covers transport errors, malformed JSON, malformed tools, and low confidence.
9. `ToolExecutor` validates tool names, product IDs, quantities, modifiers, and options. It deduplicates tool calls within a turn.
10. Prices, availability, delivery, and cart arithmetic remain authoritative in the backend and `Decimal`.

Gemini and LoRA outputs never set a price or confirm an order autonomously. Hallucinated IDs and unauthorized price/discount fields are sanitized.

## State Model

`SessionState` persists cart contents, customer details, delivery zone/address, active order, human takeover, pending clarification/upsell, upsell rejection memory, budget limit, final-confirmation state, order-confirmed state, confirmed cart signature, session epoch, last intent, directive, bounded history, and update timestamp.

The database-backed adapter uses `ChatSession`; PostgreSQL reads can lock with `SELECT ... FOR UPDATE`. SQLite is the local development compatibility backend. `SESSION_TTL_SECONDS` starts a new session epoch for expired customer sessions; active human takeover is preserved.

The dispatcher keeps only in-memory runtime caches and per-phone locks. Cart and session data are rehydrated by the configured backend.

## Cart and Order Lifecycle

Cart mutations use exact `Decimal` arithmetic with cent rounding. Live menu availability and price revalidation occur on inquiry and confirmation.

Order states are:

```text
DRAFTING
  -> PENDING_CONFIRMATION
  -> CONFIRMED
  -> ACCEPTED
  -> PREPARING
  -> READY
  -> OUT_FOR_DISPATCH / OUT_FOR_DELIVERY
  -> DELIVERED
  or FAILED / CANCELLED
```

Confirmation requires customer details and explicit approval. The confirmation payload requires `confirmed_by_customer=True`. Idempotency is keyed from the phone, session epoch, and cart signature, preventing duplicate creates and replaying an existing order.

## Database

Alembic migrations are `0001` through `0004`:

- `0001`: base menu, order, session, notification, and authentication schema
- `0002`: sales/CX notifications, reviews, and scheduler fields
- `0003`: scheduler leases and database indexes
- `0004`: durable AI session state and inbound message buffer

The inbound WhatsApp buffer is durable; CRITICAL/HIGH complaint messages bypass the normal debounce delay, while normal bursts aggregate into one AI turn.

## API Surface

Important implemented routes include:

- `GET /api/webhook` and `POST /api/webhook`: Meta WhatsApp verification, HMAC, deduplication, and AI dispatch
- `POST /api/telegram/webhook`: secret-verified private-chat text and voice dispatch
- `POST /api/ai/chat`: direct AI dispatcher integration
- `GET /api/ready`: honest model/database readiness telemetry
- `GET /api/menu`, `PATCH /api/menu/{id}/price`, and availability endpoints
- `GET /api/orders`, order status transitions, and cashier directives
- `GET /api/metrics` and `/api/cx/*`: operations and customer-experience APIs
- `/ws/cashier`: live cashier order/complaint updates

Authentication uses the FastAPI JWT/RBAC implementation. Dashboard endpoints and admin operations must use configured secrets and HTTPS in deployment.

## Environment

Copy `.env.example` to `.env` locally. Never commit real credentials.

Required by deployed service:

- `DATABASE_URL`
- `AUTH_SECRET_KEY`
- `APP_ENV`
- channel credentials for enabled integrations

Optional or integration-specific:

- `GEMINI_API_KEY`, `AI_PRIMARY_MODEL`, `AI_FALLBACK_MODEL`
- `GROQ_API_KEY`, `GROQ_WHISPER_MODEL`
- Meta verify token, app secret, access token, and phone number ID
- Telegram bot token and cashier/owner chat IDs
- `MESSAGE_DEBOUNCE_SECONDS`, `SESSION_TTL_SECONDS`
- review, arrival, scheduler, and rate-limit settings

The `.env.example` bootstrap password is development-only and must be overridden.

## Testing

Primary command:

```bash
py -m pytest -q wasla_phase_a_repro_tests.py tests/ backend/test_cart_calculator.py
```

Focused critical regression command:

```bash
py -m pytest -q tests/regression/test_critical_dispatcher_regressions.py
```

Tests cover AI intent/evaluation, menu and security, carts, multi-turn dialogue, persistence, concurrency, idempotency, webhook/HMAC/deduplication, scheduler leases, Telegram 429 behavior, Meta policy guards, reviews, complaints, human takeover, migrations, and dashboard integration.

The suite is deterministic for mocked and local paths. Live API calls are not implied by unit or integration test success.
