# Wasla AI — Production Deployment Readiness Report

**Service:** Wasla AI — Restaurant AI Dispatcher (`MOKA Restaurant & Cafe`)  
**Target Environments:** Docker (`Dockerfile`), Hugging Face Spaces Free ZeroGPU (`deploy/hf_app.py`), PostgreSQL 16/Neon + External Cron Scheduler

---

## 1. Readiness Endpoint Contract (`GET /api/ready`)

`/api/ready` verifies live PostgreSQL/SQL connectivity (`SELECT 1`) and exposes honest, non-inflated runtime AI & provider telemetry without exposing secrets:

The JSON below is an illustrative schema example, not a health snapshot from a live deployment. A provider's `configured` flag means credentials/settings are present; it does not prove that the provider is reachable or that an end-to-end customer flow has succeeded.

```json
{
  "ready": true,
  "database_dialect": "postgresql",
  "database_connected": true,
  "providers": {
    "gemini_llm_configured": true,
    "groq_whisper_configured": true,
    "meta_whatsapp_configured": true,
    "meta_webhook_hmac_enabled": true,
    "telegram_configured": true,
    "scheduler_cron_configured": true,
    "finetuned_lora_checkpoint_exists": true,
    "finetuned_lora_loaded": false,
    "finetuned_lora_version": "camelbert-wasla-lora-v2",
    "finetuned_lora_inference_available": true,
    "active_inference_route": "gemini+lora+menu_index"
  },
  "ai_models": {
    "enabled": true,
    "checkpoint_exists": true,
    "checkpoint_v1_exists": true,
    "checkpoint_v2_exists": true,
    "checkpoint_path": "...",
    "model_loaded": false,
    "model_version": "camelbert-wasla-lora-v2",
    "base_model_id": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "device": "cpu",
    "inference_available": true,
    "load_time_ms": null,
    "load_error": null,
    "active_inference_route": "gemini+lora+menu_index",
    "telemetry": {
      "turns_total": 0,
      "lora_calls": 0,
      "gemini_primary_calls": 0,
      "gemini_fallback_calls": 0,
      "local_hybrid_calls": 0
    }
  }
}
```

**Important Guarantee:** `model_loaded` (`finetuned_lora_loaded`) remains `false` at cold boot and turns `true` ONLY after `FineTunedIntentClassifier.load()` genuinely loads weights into RAM/VRAM. Startup stays fast (<2s for Docker `HEALTHCHECK`) even with PyTorch + PEFT installed.

---

## 2. Database Schema & Migrations (Alembic `0001` -> `0005`)

| Revision ID | File | Purpose |
| :--- | :--- | :--- |
| `0001_initial_schema` | `backend/migrations/versions/0001_initial_schema.py` | Core menu, sessions, orders, events, processed messages, outbox, feedback, users, audit logs |
| `0002_sales_cx_notifications` | `backend/migrations/versions/0002_sales_cx_notifications_scheduler.py` | Scheduler jobs, notifications, complaint tickets, recommendation analytics, settings |
| `0003_scheduler_leases` | `backend/migrations/versions/0003_scheduler_leases_and_indexes.py` | Multi-worker scheduler lease columns & `must_change_password` |
| `0004_phase_b_durable_ai_state` | `backend/migrations/versions/0004_phase_b_durable_ai_state.py` | Durable `inbound_message_buffers` table for crash-safe WhatsApp debouncing |
| `0005_order_idempotency` | `backend/migrations/versions/0005_order_idempotency.py` | Backfills legacy order keys and adds a unique database constraint for order idempotency |

Command to apply migrations cleanly:

```bash
alembic -c backend/alembic.ini upgrade head
```

`deploy/hf_app.py` executes `alembic upgrade head` automatically before starting uvicorn, while `Dockerfile` executes `alembic -c backend/alembic.ini upgrade head && uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}`.

---

## 3. Container & Cloud Deployment Configuration

- **Dependencies (`requirements.txt` & `deploy/hf_requirements.txt`):**
  - Uses `--extra-index-url https://download.pytorch.org/whl/cpu` so Linux CPU deployments install lightweight PyTorch CPU wheels alongside `transformers`, `peft`, `safetensors`, and `accelerate`.
- **Docker Image (`Dockerfile`):**
  - Bundles `ai_service/`, `backend/`, `dashboard/`, `menu/`, `scripts/`, and `training/checkpoints/` (both v1 & v2).
  - Configures `HF_HOME=/home/wasla/.cache/huggingface` writable by non-root user `wasla` (UID `1000`).
- **Hugging Face Spaces (`deploy-hf-space.yml`):**
  - Automatically stages `deploy/hf_app.py`, `deploy/hf_requirements.txt`, `ai_service/`, `backend/`, `dashboard/`, `menu/`, `scripts/`, and `training/checkpoints/` on push to `main` when `HF_TOKEN` and `HF_SPACE_ID` secrets are configured.
- **Production Safety Gates (`backend/app/main.py`):**
  - When `APP_ENV=production`: forbids default `AUTH_SECRET_KEY`, forbids default `BOOTSTRAP_ADMIN_PASSWORD`, forbids SQLite `DATABASE_URL`, and returns `404` on `/api/ai/chat`.
  - Production startup also requires `META_APP_SECRET`, so WhatsApp webhook requests cannot be accepted without HMAC verification.

---

## 4. External Operational Blockers / Operational Notes

1. **External-provider verification:** This task did not verify live Gemini, Groq, Telegram, WhatsApp/Meta, PostgreSQL, or a full production customer journey. Mock transports and configured flags are not live-integration evidence. Do not represent Telegram or WhatsApp end-to-end ordering as currently verified without a fresh live test.
2. **Order status lookup:** The app's production `DirectDBOrderBridge` reads the active order only when its `customer_phone` matches the inbound customer's number, and otherwise checks that number's latest order. Unknown or absent status fails closed. The separate `RealBackendAdapter` has no authenticated customer-status endpoint and returns an explicit tracking-unavailable error; it never falls back to mock order records.
3. **Base Model Weights on Cold Environment:** LoRA adapter weights are bundled in Git, while the base encoder (`CAMeL-Lab/bert-base-arabic-camelbert-mix`) may download lazily to `HF_HOME` on first inference if not cached. If offline or the model cannot load, `FineTunedIntentClassifier` fails safely and dispatches through `OfflineEgyptianNLU`.
4. **Gemini API Quotas:** The configured primary/fallback/local degradation path is implemented and covered by tests using provider stubs; live quota handling remains unverified until tested with provider credentials.
