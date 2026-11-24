# Wasla AI — Backend Integration Contract

This contract describes the integrations implemented in this repository. Automated tests use local databases or provider stubs; they are not live production evidence.

## HTTP Backend Adapter

`RealBackendAdapter` talks to the FastAPI backend in this repository:

| Method & path | Purpose | Authorization / source of truth |
|---|---|---|
| `GET /api/menu` | Read available menu items | Backend database |
| `GET /api/delivery/quote?zone_name=...` | Read the restaurant delivery fee and optional configured estimate | Backend `DELIVERY_*` settings; no courier tracking |
| `POST /api/orders` | Submit a confirmed order | Backend validates item prices, availability, and idempotency |
| `POST /api/customer/order-status` | Read a customer's own stored order ID and status | Requires `X-Backend-Service-Key`; lookup is additionally scoped to the exact `customer_phone` |
| `POST /api/orders/{order_id}/modification` | Submit a customer modification request | Existing order endpoint |
| `POST /api/orders/{order_id}/instruct` | Send a cashier directive | Existing order endpoint |

Set the same high-entropy `BACKEND_SERVICE_API_KEY` in both services when using the HTTP status adapter. If it is absent, the adapter fails closed and the backend endpoint returns `503`. A wrong key is rejected. An explicit order ID is never sufficient by itself: the backend requires the requested order's stored phone to match the supplied customer phone. It returns only the order ID and recorded restaurant status, not address, phone, or item details.

The in-process `_DirectDBOrderBridge` also scopes status queries to the inbound phone. Order creation associates the confirmed order ID with that phone's persistent `ChatSession`, so a later "what is my order number?" follow-up can recover it after dispatcher restart. Explicit IDs are checked against the same ownership constraint.

## Delivery and ETA Limits

Delivery fees are configured through `DELIVERY_DEFAULT_FEE`, `DELIVERY_FREE_ZONE`, and `DELIVERY_FREE_ZONE_FEE`. Both quote responses and order totals use backend settings. An estimated duration is omitted unless `DELIVERY_ESTIMATED_MINUTES` is explicitly configured.

No courier provider, authenticated courier API, live courier events, or location feed is configured in this repository. Stored restaurant order states such as `IN_KITCHEN` or `OUT_FOR_DELIVERY` are not evidence of courier events and do not imply an arrival time. Do not present them as such.

## Live Verification

Stubs and SQLite tests verify request shapes, service-key handling, phone ownership, and configured-fee behavior only. Live HTTP, courier, PostgreSQL, WhatsApp/Meta, Telegram, Gemini, and Groq flows require authorized credentials and infrastructure and remain unverified unless separately recorded in `FINAL_RELEASE_VERIFICATION.md`.
