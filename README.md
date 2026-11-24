# Wasla AI — Commercial Restaurant AI Dispatcher (MOKA Pilot)

**Wasla AI** is a production-oriented, Egyptian Arabic Voice-First AI Ordering & Cashier Operations platform engineered for restaurants and cafes, piloted with **MOKA Restaurant & Cafe** (145 confirmed menu items across 23 categories).

---

## Core Capabilities

- **Production Hybrid AI Architecture (`ai_service/`):** Confidence-gated hybrid orchestration combining **CAMeLBERT-Mix PEFT LoRA v2** (`FineTunedIntentClassifier` with lazy thread-safe caching & CPU/GPU support), **Gemini Native Function Calling** (`tools` / `functionDeclarations` with automatic primary -> fallback resilience and validated single-turn `ToolExecutor` deduplication), and deterministic **Egyptian Arabic NLU + 145-Item MenuIndex** (handling multi-variant same-product lines, self-corrections, multi-product & availability-aware price queries, gibberish filtering, and rejection-aware recommendations).
- **Fine-Tuned Egyptian Arabic Intent Models (`training/`):** Preserves `CAMeL-Lab/bert-base-arabic-camelbert-mix` (**Apache-2.0**) PEFT LoRA v1 & v2 (`camelbert-wasla-lora-v2`: **82.08%** held-out unseen test accuracy / **60.29%** balanced 12-intent independent benchmark accuracy; Hybrid dispatcher: **83.02%** unseen accuracy and **100.00%** intent / **0.0%** nonexistent FP on the 154-case MOKA typo suite).
- **Durable PostgreSQL State, Live Menu Sync & Inbound Debouncing (`backend/`):** 14-field PostgreSQL/SQLite `ChatSession` rehydration across restarts & workers (`SELECT ... FOR UPDATE`), `SESSION_TTL_SECONDS` cleanup, per-customer `asyncio.Lock` + immutable `TurnExecutionContext`, idempotent confirmation keys, DB-backed 3-second `InboundMessageBuffer` crash recovery (`0004_phase_b_durable_ai_state`) with 0ms urgent complaint bypass, and instant `MenuIndex` synchronization after dashboard price/availability edits.
- **Deterministic Business Safety & Audio Pipeline:** Exact `Decimal` pricing (`backend/app/cart_calculator.py`), `confirmed_by_customer=True`, HMAC-SHA256 webhook verification, Option C Hybrid `audio/ogg` transcription (Gemini + Groq Whisper Large v3 Turbo), full order lifecycle states (`PENDING_CONFIRMATION` → `DELIVERED`), Telegram cashier group & owner urgent escalation (`@wwaslabot`), and scheduled post-delivery reviews.
- **Commercial Arabic RTL Operations Console (`dashboard/`):** Admin/Cashier RBAC, live WebSocket order & complaint alerts (`/ws/cashier`), 145-item inventory & live pricing controls, and honest runtime AI/DB telemetry (`/api/ready`).

---

## Repository Structure

```text
wasla-ai-dispatcher/
├── ai_service/                     # Repaired Hybrid AI Layer (LoRA + Gemini Tools + Egyptian NLU)
│   ├── adapters/                   # Backend abstractions + MockBackendAdapter (14-field TTL parity)
│   ├── audio/                      # Option C Hybrid OGG/Opus audio pipeline
│   ├── dispatcher/                 # WaslaAIDispatcher (per-phone locks, TurnExecutionContext), MessageDebouncer
│   ├── models/                     # FineTunedIntentClassifier (_MODEL_CACHE) + GeminiLLMClient (Function Calling)
│   ├── schemas/                    # Strict Pydantic v2 DTOs & SessionState / TurnExecutionContext
│   ├── security/                   # PII log redaction, prompt-injection guardrails, audio validation
│   ├── tools/                      # 8 Function Calling tools & deduplicated ToolExecutor
│   └── understanding/              # Live DB-synced 145-item MenuIndex & Egyptian Arabic dialect/gibberish NLU
├── backend/                        # FastAPI + SQLAlchemy + Alembic (0001..0004) + WebSockets Backend
│   ├── app/                        # main.py (/api/ready, webhook debouncer, DB bridge), models.py, cart_calculator.py
│   └── migrations/versions/        # 0001..0004 (including InboundMessageBuffer for durable debouncing)
├── dashboard/                      # Commercial Arabic RTL Cashier Console
├── menu/                           # Canonical MOKA Menu (145 products, 23 categories, 7 extras)
├── training/                       # PEFT LoRA v1 & v2 checkpoints, data splits, scientific eval & reports
├── datasets/                       # Grounded MOKA typo/challenge eval cases + HF Egyptian audio loader
├── docs/                           # Phase B Architecture, Model Inference, Regression, Readiness & Ops Docs
├── tests/                          # Unit, AI Eval, Real-Backend E2E, and Phase B Hardening suites
├── WASLA_AI_PHASE_A_AUDIT_REPORT.md
├── wasla_phase_a_repro_tests.py    # 7 Phase A bug reproductions (now 7/7 passing)
└── requirements.txt
```

---

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note on Windows Smart App Control with Python 3.14: install pure-Python SQLAlchemy if Cython `.pyd` extensions are blocked: `DISABLE_SQLALCHEMY_CEXT=1 pip install --no-binary sqlalchemy "sqlalchemy>=2.0.40,<2.1"`.)*

### 2. Configure Environment & Run Database Migrations

Copy `.env.example` to `.env` (never commit `.env`), run Alembic `0001 -> 0004`, and start Uvicorn:

```bash
cp .env.example .env
alembic -c backend/alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Open **http://localhost:8000** for the **Wasla Commercial Operations Console** and **http://localhost:8000/api/ready** for honest AI model & PostgreSQL readiness telemetry.

### 3. Run Automated Tests & Scientific Evaluation

```bash
pytest -v wasla_phase_a_repro_tests.py tests/ backend/test_cart_calculator.py
python training/scripts/evaluate_phase_b_scientific.py
```

*(Verified local result: **61 passed** across Phase A reproductions, Phase B hardening tests, 154-case MOKA typo suite, dialogue flows, cart math, RBAC, and Alembic migrations.)*

---

## Documentation

- [Phase B Implementation Report](docs/PHASE_B_IMPLEMENTATION_REPORT.md)
- [Phase B Repaired AI Architecture](docs/AI_ARCHITECTURE.md)
- [Model & Inference Scientific Evaluation Report (Phase B)](docs/MODEL_INFERENCE_REPORT.md)
- [Phase B Regression Test Report](docs/REGRESSION_TEST_REPORT.md)
- [Production Deployment Readiness & `/api/ready` Contract](docs/DEPLOYMENT_READINESS.md)
- [Phase A AI Audit Report](WASLA_AI_PHASE_A_AUDIT_REPORT.md)
- [Sales Assistant, Notifications, Reviews & Owner Escalation Guide](docs/CUSTOMER_EXPERIENCE_AND_OPS.md)
- [Model Card (CAMeLBERT-Mix LoRA v2 — Apache-2.0)](docs/MODEL_CARD.md)
- [Commercial Production Deployment & Zero-Cost Hosting Guide](docs/PRODUCTION_DEPLOYMENT.md)
- [Verified Backend & Dashboard Integration Contract](docs/INTEGRATION_CONTRACT.md)
- [MOKA Menu Extraction Report](menu/extraction_report.md)
