# Wasla AI — Commercial Restaurant AI Dispatcher (MOKA Pilot)

**Wasla AI** is a production-oriented, Egyptian Arabic Voice-First AI Ordering & Cashier Operations platform engineered for restaurants and cafes, piloted with **MOKA Restaurant & Cafe** (145 confirmed menu items across 23 categories).

---

## Core Capabilities

- **Egyptian Arabic Voice-First & Text Ordering (`ai_service/`):** Understands colloquial Egyptian Arabic, mixed English/Arabic menu names, multi-item orders, sizes (`M/L`, `S/D`, `10PCS/20PCS`), paid extras (`حبة كاملة`, `حشو اطراف`, `موتزريلا`, etc.), and free removals (`من غير بصل`).
- **Option C Hybrid Audio Pipeline:** Validated WhatsApp `audio/ogg` ingestion (magic-byte & size checks) using active **Gemini 3.5 Flash-Lite** direct multimodal audio with automatic fallback to **Groq Whisper Large v3 Turbo** or **OpenAI `gpt-transcribe`**.
- **Deterministic Business Safety:** Prices, modifiers, delivery fees, and stock availability are NEVER invented by the LLM. All financial calculations use exact `Decimal` arithmetic in the authoritative FastAPI Backend (`backend/app/cart_calculator.py`).
- **Transactional Reliability:** Enforces `confirmed_by_customer=True`, `backup_phone`, `idempotency_key` deduplication, Meta webhook `meta_message_id` deduplication, HMAC-SHA256 webhook verification, and a 7-state `OrderStatus` machine (`DRAFTING` → `SUBMITTED` → `IN_KITCHEN` → `DISPATCHED` ↔ `UNREACHABLE_ALERT` → `FULFILLED` / `CANCELLED`).
- **Human-in-the-Loop Cashier Operations Console (`dashboard/`):** Commercial Arabic RTL SaaS dashboard with real-time WebSocket updates (`/ws/cashier`), SQL-backed KPI metrics (`/api/metrics`), 1-click MOKA inventory availability toggles (`/api/menu/{id}/availability`), smart cashier directive presets, and a live Wasla AI chat simulator (`/api/ai/chat`).

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

- [Commercial Project Roadmap (PDF)](docs/RESTAURANT_AI_PROJECT_ROADMAP.pdf)
- [Verified Backend & Dashboard Integration Contract](docs/INTEGRATION_CONTRACT.md)
- [MOKA Menu Extraction Report](menu/extraction_report.md)
- [Project Audit & Test Results](docs/PROJECT_AUDIT_AND_TEST_RESULTS.md)
