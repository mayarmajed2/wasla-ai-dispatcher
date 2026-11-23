# Wasla AI — Sales Assistant, Notifications, Customer Reviews & Owner Escalation Guide

> **Operational Honesty Notice:** All core business logic, database tables, Alembic migrations (`0001` → `0002`), durable scheduler, Meta policy guards, Telegram Bot client, complaint escalation engine, and Customer Experience Dashboard are **implemented and verified via automated tests (33/33 tests)**. Live external Telegram delivery, live PostgreSQL execution, and live public HTTPS Meta webhook delivery remain **blocked by external credentials/infrastructure** until `TELEGRAM_BOT_TOKEN`, `CASHIER_TELEGRAM_CHAT_ID`, `RESTAURANT_OWNER_TELEGRAM_CHAT_ID`, `META_APP_SECRET`, and a hosted PostgreSQL/HTTPS endpoint are provided.

---

## 1. Natural Egyptian Arabic Sales Personality (`ai_service/dispatcher/persona.py`)

- Warm, concise, polite Egyptian Arabic phrasing with occasional subtle emojis (`❤️`, `✨`, `😊`).
- Grounded strictly in verified MOKA 145-item menu database (`menu/menu.json`) and deterministic `Decimal` totals.
- Zero invented menu items, discounts, combos, or unverified delivery promises.
- Uses Gemini (`gemini-3.5-flash-lite` / `gemini-3.8-flash`) for NLU with `OfflineEgyptianNLU` deterministic typo-tolerant fallback; preserves both fine-tuned **CAMeLBERT-Mix PEFT LoRA** checkpoints (`v1` and `v2`, Apache-2.0) and the **TF-IDF + Logistic Regression** intent baseline (`81.37%` Acc / `0.8100` Macro-F1 on the independent 204-example benchmark).

---

## 2. Recommendation & Combo Upsell Engine (`ai_service/recommendation/engine.py`)

- **Real MOKA Menu Grounding:** Every recommended product ID (`item_065` باكت بطاطس 30 EGP, `item_066` بطاطس شيدر هالبينو 65 EGP, `item_081` ميكس صودا تروبيكال 70 EGP, `item_082` ميكس صودا مانجو لوكو 80 EGP, `item_096` ميلك شيك فراولة 80 EGP, `item_075` كابوتشينو 60 EGP) is verified against MOKA's active menu & live availability.
- **Combo Validation Rule:** Official combos are offered **only** when explicitly configured via `OfficialComboConfig`. Because MOKA's current menu defines 0 official combos, Wasla recommends separate available items and calculates their exact combined price via `Decimal` arithmetic — **never inventing discounts**.
- **Safety & Preference Guards:**
  - Configurable via `UPSELL_ENABLED` (env) and `PATCH /api/settings/notifications` (Admin RBAC).
  - Never repeats an offer after customer rejection (`session.upsell_declined` / `rejected_upsell_ids`).
  - Never recommends during complaints, cancellations, item removals, or human handoff.
  - Respects customer-stated budget (e.g., `"الميزانية 150"`).
  - Shows exact EGP prices upfront and adds recommended items **only** after explicit customer acceptance.
- **Genuine Analytics:** Logged in `recommendation_events` (`offered`, `accepted`, `rejected`, `added_revenue`) and surfaced in `/api/customer-experience/overview`.

---

## 3. Complete Order Lifecycle & Timestamps (`backend/app/models.py`)

Supports both backward-compatible 7-state statuses and the full explicit commercial lifecycle:
`PENDING_CONFIRMATION` → `CONFIRMED` / `SUBMITTED` → `ACCEPTED` → `PREPARING` / `IN_KITCHEN` → `READY` → `OUT_FOR_DELIVERY` / `DISPATCHED` → `DELIVERED` / `FULFILLED` (plus `UNREACHABLE_ALERT`, `CANCELLED`, `FAILED`).

- Every transition records an immutable `order_status_history` row (`from_status`, `to_status`, `actor`, `changed_at` in UTC) and updates explicit lifecycle timestamp columns on `orders` (`confirmed_at`, `accepted_at`, `preparing_at`, `ready_at`, `out_for_delivery_at`, `delivered_at`, `cancelled_at`, `failed_at`, `delivery_confirmed_by`).
- An order is **never** marked `DELIVERED` without an authorized staff action (`PATCH /api/orders/{id}/status` or `POST /api/orders/{id}/confirm-arrival`).

---

## 4. Meta WhatsApp Groups API Eligibility & Zero-Cost Telegram Cashier Group

### Official Meta WhatsApp Groups API Verification (Checked Nov 2026)
- **Official Business Account (OBA) Required:** Strictly limited to green-checkmark OBA numbers. Standard Cloud API & Meta Test Numbers (`1515063625018594`) are **NOT eligible**.
- **Group Limits:** Maximum **8 participants** per group, **invite-only** (cannot connect to existing WhatsApp groups), and **interactive buttons/lists are unsupported**.
- **Billing:** Billed **per-message, per-recipient** (e.g., 5 cashiers = 5× per-message cost).
- **Verdict under 0 EGP constraint:** **Ineligible & billable** → Wasla uses the official, free **Telegram Bot API** (`TELEGRAM_BOT_TOKEN` + `CASHIER_TELEGRAM_CHAT_ID`) for instant private cashier group alerts.

### Telegram Cashier Group Setup (Zero Cost)
1. Open `@BotFather` in Telegram → `/newbot` → copy `TELEGRAM_BOT_TOKEN`.
2. Create a private Telegram group for MOKA cashiers and add the bot.
3. Send a message in the group and fetch its `chat.id` (starts with `-100...`) via `https://api.telegram.org/bot<TOKEN>/getUpdates` → set `CASHIER_TELEGRAM_CHAT_ID`.
4. Set both variables in `.env` (never commit `.env`).
5. **Failure & Privacy Guarantees:**
   - Cashier messages omit full home address & backup phone (`Delivery info: Open dashboard (authorized staff only)`) and include a direct dashboard link (`/#order-{id}`).
   - Deduplicated via `notification_attempts.idempotency_key = cashier_order:{order_id}`.
   - If Telegram fails, the order remains safely persisted in DB, a durable retry job (`cashier_notify_retry:{order_id}`) is scheduled in `scheduled_jobs`, and the dashboard receives a live WebSocket `cashier_notification_failed` alert.

---

## 5. Meta 24-Hour Window, Zero-Cost Guard & Durable Review Scheduler (`backend/app/scheduler.py`)

### Verified 2026 Meta Pricing & Policy Rules (`MetaPolicyGuard`)
- **24-Hour Customer Service Window:** Opens when the customer messages the business. Inside 24 hours, free-form service messages have a **1,000 free messages/month** allowance per phone number (Oct 1, 2026 Meta pricing), after which they are billed at utility rates.
- **Templates:** Utility templates are billed per delivered message both inside and outside the 24-hour window.
- **Zero-Cost Protection:** When `META_BILLABLE_MESSAGES_AUTHORIZED=false` (default), if a scheduled review or arrival check falls outside the 24-hour window or exceeds the 1,000 free monthly service quota, Wasla **does NOT silently spend money** — it retains the job in `scheduled_jobs` as `BLOCKED_POLICY` with the exact reason and surfaces it in the dashboard.

### Durable Review & Arrival Scheduling
- Backed by `scheduled_jobs` table (survives restarts via `recover_stale_jobs_on_startup()`).
- **Primary Trigger:** Scheduled for exactly **+60 minutes** (`review_delay_minutes`) after order transitions to `DELIVERED` / `FULFILLED`.
- **Fallback Trigger:** Scheduled for **+90 minutes** (`arrival_check_delay_minutes`) after order confirmation; sends arrival check only if delivery is still unconfirmed and Meta policy allows.
- **Automatic Suppression:** Reviews are automatically suppressed for `CANCELLED`, `FAILED`, `refunded=True`, unresolved `CRITICAL`/`HIGH` complaints, opted-out customers (`opted_out_reviews=True`), or already-sent orders.
- **Dead-Letter & Retries:** Failed jobs retry with exponential backoff up to `max_attempts=5`, then move to `DEAD_LETTER` and emit an `Outbox` alert.

---

## 6. Urgent Complaint Detection, Owner Escalation & Human Takeover

- **High-Recall Detection (`ai_service/complaints/detector.py`):** Covers food safety (`CRITICAL` — escalated immediately with zero confidence barrier), missing items, wrong items, incorrect billing, refund requests, late delivery, cold food, food quality, delivery problems, service issues, and aggressive customers.
- **Owner Escalation (`NotificationService.escalate_complaint_to_owner`):**
  - Creates a durable `complaint_tickets` row immediately.
  - Tries `RESTAURANT_OWNER_WHATSAPP` only if configured AND permitted within zero-cost Meta 24h policy; otherwise **immediately falls back to `RESTAURANT_OWNER_TELEGRAM_CHAT_ID` (Telegram Bot, 0 EGP)** — never silently dropped.
  - Deduplicates repeated alerts for the same unresolved incident unless critical new information (`CRITICAL`) arrives.
  - Never exposes the owner's private number to customers, and never promises unauthorized refunds or compensation deadlines.
- **Human Takeover Hard Gate:**
  - Activated automatically on escalated complaints or manually via `POST /api/conversations/{phone}/takeover`.
  - Once active, `WaslaAIDispatcher` sets `suppress_ai_reply=True` and **stops sending automated AI replies** until staff resolves the complaint or releases takeover.

---

## 7. Database Migrations (`0001` → `0002`)

```bash
alembic -c backend/alembic.ini upgrade head
```

- `0001_initial_schema.py`: Core tables (`menu_items`, `orders`, `chat_sessions`, `order_events`, `processed_messages`, `outbox`, `feedback_requests`, `users`, `audit_logs`).
- `0002_sales_cx_notifications_scheduler.py`: Lifecycle timestamp columns + `order_status_history`, `scheduled_jobs`, `notification_attempts`, `complaint_tickets`, `recommendation_events`, `restaurant_settings`.
