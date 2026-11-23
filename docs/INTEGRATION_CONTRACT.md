# Wasla AI Dispatcher — Verified Backend & Dashboard Integration Contract

**Status:** Verified against actual Backend & Dashboard source (`existing_system/RESTURANT AI/`)  
**Date:** 2026-11-23

---

## 1. Inspected Existing System Summary (Read-Only Audit)

- **Backend:** FastAPI (`existing_system/RESTURANT AI/app/main.py`), SQLAlchemy ORM (`app/models.py`), deterministic `Decimal` calculator (`app/cart_calculator.py`), WebSocket hub (`app/ws_manager.py`).
- **Cashier Dashboard:** Vanilla JS + HTML/CSS (`frontend-cashier-screen/assets/app.js`) connected to `WS /ws/cashier`, listening for `order_created`, `order_modified`, `customer_alert`, and `order_status_changed`.

## 2. Verified Backend Endpoints Used by `RealBackendAdapter`

| Method & Path | Purpose | Exact Request / Response Contract |
|---|---|---|
| `GET /api/menu` | Fetch live available menu items | Returns `[{ id, name, category, base_price, modifiers }]` |
| `POST /api/orders` | Confirm & lock order (201) | Requires `customer_name` (≥2), `customer_phone` (≥8), **`backup_phone` (≥8)**, `delivery_address` (≥5), `zone_name`, `items[]`, **`idempotency_key`**, **`confirmed_by_customer: true`** |
| `PATCH /api/orders/{id}/status` | Cashier status transitions | Enforces 7-state machine (`DRAFTING` → `SUBMITTED` → `IN_KITCHEN` → `DISPATCHED` ↔ `UNREACHABLE_ALERT` → `FULFILLED` / `CANCELLED`) |
| `POST /api/orders/{id}/instruct` | Cashier free-text directive | `{ "directive": "..." }` → broadcasts `customer_alert` over `/ws/cashier` |
| `POST /api/orders/{id}/modification` | Post-submission customer edit | `{ "message": "..." }` → broadcasts `order_modified` over `/ws/cashier` |
| `POST /api/webhook` | Meta WhatsApp webhook | Currently deduplicates `ProcessedMessage` by `meta_message_id` — ready to invoke `WaslaAIDispatcher` |
| `WS /ws/cashier` | Live Dashboard updates | Real-time JSON events with `event_id`, `event_type`, `order_id`, `payload` |

## 3. Critical Alignment Fixes Discovered During Inspection

1. **`backup_phone` Requirement:** The real Backend's `ConfirmOrder` schema strictly requires `backup_phone` (min length 8). `WaslaAIDispatcher` now collects and validates `backup_phone` before calling `confirm_and_lock_order`.
2. **Idempotency Key:** `ToolExecutor.confirm_and_lock_order` deterministically hashes `(phone_number + cart signature)` into `idempotency_key` so retries never duplicate orders.
3. **Menu Seeding:** The Backend's default `seed_menu()` only inserts 3 placeholder items. Use `scripts/seed_moka_menu.py` to seed all 145 confirmed MOKA products into `MenuItem`.
4. **Zero Duplication:** `ai_service/` does NOT duplicate the Backend database, order state machine, or Cashier Dashboard. Switching from Mock testing to the real Backend requires only `BACKEND_ADAPTER_MODE=http` and `BACKEND_BASE_URL=http://localhost:8000`.
