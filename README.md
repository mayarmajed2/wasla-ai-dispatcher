# Wasla AI — Commercial Restaurant AI Dispatcher (MOKA Pilot)

**Wasla AI** is a production-oriented, Egyptian Arabic Voice-First AI Ordering & Cashier Operations platform engineered for restaurants and cafes, piloted with **MOKA Restaurant & Cafe** (145 confirmed menu items across 23 categories).

---

## Core Capabilities

- **Natural Egyptian Arabic Sales Assistant & Upsell Engine (`ai_service/`):** Warm, concise Egyptian Arabic personality (`persona.py`), typo-tolerant 145-item MOKA menu matching, context-aware recommendations & official-combo validation (`recommendation/engine.py`) with budget respect and rejection memory.
- **Fine-Tuned Egyptian Arabic Intent Models (`training/`):** Preserves `CAMeL-Lab/bert-base-arabic-camelbert-mix` (verified **Apache-2.0**) PEFT LoRA v1 & v2 (`camelbert-wasla-lora-v2`, `val_macro_f1 = 0.8858`) alongside a fast TF-IDF + Logistic Regression baseline (`81.37%` Acc / `0.8100` Macro-F1 on the 204-example independent benchmark).
- **Option C Hybrid Audio Pipeline:** Validated WhatsApp `audio/ogg` ingestion using **Gemini 3.5 Flash-Lite** and **Groq Whisper Large v3 Turbo** (`AVG_WER = 10.11%`, `AVG_CER = 2.76%` on 20 real Egyptian FLEURS test clips).
- **Deterministic Business Safety & Full Lifecycle:** Exact `Decimal` pricing (`backend/app/cart_calculator.py`), `confirmed_by_customer=True`, `idempotency_key` deduplication, HMAC-SHA256 webhook verification, full commercial order statuses (`PENDING_CONFIRMATION` → `CONFIRMED`/`SUBMITTED` → `ACCEPTED` → `PREPARING` → `READY` → `OUT_FOR_DELIVERY` → `DELIVERED` / `CANCELLED` / `FAILED`) with `order_status_history` timestamps.
- **Zero-Cost Cashier Group, Post-Delivery Reviews & Urgent Owner Escalation (`backend/app/notifications.py`, `backend/app/scheduler.py`):** Instant Telegram Bot cashier group alerts (because Meta Groups API requires OBA + bills per recipient), restart-safe DB-backed scheduler (`scheduled_jobs`) for 1-hour post-delivery reviews & 90-minute arrival checks gated by Meta 24-hour / 1,000-free-service policy, high-recall food-safety & complaint escalation with zero-cost Telegram fallback, and Human Takeover silencing automated AI replies.
- **Commercial Arabic RTL Operations & Customer Experience Console (`dashboard/`):** Admin/Cashier RBAC, zero simulator in production, live WebSocket alerts (`/ws/cashier`), 145-item MOKA inventory toggle, and genuine Customer Experience & Complaints view (`/api/customer-experience/overview`).

---

## Repository Structure

```text
wasla-ai-dispatcher/
├── ai_service/                     # Independent AI Intelligence Layer
│   ├── adapters/                   # Abstract interfaces, MockBackendAdapter, RealBackendAdapter
│   ├── audio/                      # Option C Hybrid OGG/Opus audio pipeline
│   ├── dispatcher/                 # WaslaAIDispatcher, 3s MessageDebouncer, CashierDirectiveInterpreter
│   ├── models/                     # Active Gemini 3.5 Flash-Lite / 3.8 Flash client + OfflineEgyptianNLU
│   ├── schemas/                    # Strict Pydantic v2 DTOs & Backend contracts
│   ├── security/                   # PII log redaction, prompt-injection guardrails, audio validation
│   ├── tools/                      # 8 Function Calling tools & validated ToolExecutor
│   └── understanding/              # MOKA 145-item MenuIndex & Egyptian Arabic disambiguation
├── backend/                        # Integrated FastAPI + SQLAlchemy + WebSockets Backend
│   ├── app/                        # main.py, models.py, cart_calculator.py, ws_manager.py, database.py
│   ├── Dockerfile                  # Production container image
│   ├── render.yaml                 # Render deployment blueprint
│   └── test_cart_calculator.py     # Deterministic Decimal math tests
├── dashboard/                      # Redesigned Commercial Arabic RTL Cashier Console
│   ├── index.html
│   └── assets/                     # styles.css, app.js
├── menu/                           # Canonical MOKA Menu (145 products, 23 categories, 7 extras)
├── datasets/                       # Grounded MOKA evaluation cases + HF real Egyptian audio loader
├── docs/                           # Roadmap PDF, Integration Contract, Audit & Test Results
├── scripts/                        # Menu seeder & Hugging Face Egyptian Arabic ASR loader
├── tests/                          # Unit, Integration, AI Eval, and Real-Backend E2E suites
├── requirements.txt
└── .env.example
```

---

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note on Windows Smart App Control with Python 3.14: install pure-Python SQLAlchemy if Cython `.pyd` extensions are blocked: `DISABLE_SQLALCHEMY_CEXT=1 pip install --no-binary sqlalchemy "sqlalchemy>=2.0.40,<2.1"`.)*

### 2. Configure Environment

Copy `.env.example` to `.env` and populate keys locally (never commit `.env`):

```bash
cp .env.example .env
```

### 3. Run the Integrated Server & Cashier Dashboard

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser to access the **Wasla Cashier Operations Console** (Live Orders, 145-Item MOKA Inventory Toggle, and Live Wasla AI Simulator).

### 4. Run Automated Tests

```bash
pytest -v tests/ backend/test_cart_calculator.py
```

---

## Documentation

- [Sales Assistant, Notifications, Reviews & Owner Escalation Guide](docs/CUSTOMER_EXPERIENCE_AND_OPS.md)
- [Independent Model Evaluation & Audio Benchmark Report](docs/MODEL_EVALUATION.md)
- [Model Card (CAMeLBERT-Mix LoRA v2 — Apache-2.0)](docs/MODEL_CARD.md)
- [Commercial Production Deployment & Zero-Cost Hosting Guide](docs/PRODUCTION_DEPLOYMENT.md)
- [Verified Backend & Dashboard Integration Contract](docs/INTEGRATION_CONTRACT.md)
- [MOKA Menu Extraction Report](menu/extraction_report.md)
- [Project Audit & Test Results](docs/PROJECT_AUDIT_AND_TEST_RESULTS.md)
