# Wasla AI — Commercial Production Deployment & Zero-Cost Hosting Guide

> **Honest Status Notice:** As of this release, Wasla AI is **locally built, tested, and verified on Windows 11 (Python 3.14, SQLite)** with live Gemini, Groq Whisper, and Meta Graph API v27.0 connectivity verified. Live PostgreSQL migration execution, Docker image builds, and public HTTPS deployment have **NOT** yet been executed from this workstation because `docker`, `psql`, `pg_dump`, and a public hosting account are not present locally.

---

## 1. Zero-Cost Free Hosting Analysis (Verified Against Official Provider Documentation)

Because of the strict **0 EGP (no credit card)** requirement, the following free tiers were verified directly from official vendor documentation:

| Provider | Card Required? | Free Tier Limits | Commercial Production Limitations (Honest Assessment) |
|---|---|---|---|
| **Render Web Services (Free)** | No | 750 hours/month, 512 MB RAM, 0.1 CPU, free HTTPS + WebSockets | **Spins down after 15 minutes of inactivity** (~1 minute cold start on next request — breaks real-time WhatsApp webhook latency). Ephemeral filesystem. Free Render Postgres expires after **30 days**. Render official docs explicitly state: *"Do not use them for production applications."* |
| **Neon Serverless Postgres (Free)** | No | 1 GB storage, 100 CU-hours/month, automatic TLS | Scales to zero after 5 minutes of inactivity (~500ms–1s wake latency). Suitable for **staging, UAT, and low-volume pilot**, not guaranteed 24/7 commercial SLA. |
| **GitHub Actions (Free)** | No | 2,000 CI minutes/month (private repos) | Ready via [`docs/ci-workflow-template.yml`](ci-workflow-template.yml); requires a GitHub token with `workflow` scope (`gh auth refresh -s workflow`) to move into `.github/workflows/ci.yml`. |

### Recommended Zero-Cost Staging Architecture
1. **Database:** Neon Free PostgreSQL (`DATABASE_URL=postgresql+psycopg://...?sslmode=require`).
2. **Backend + Dashboard:** Render Free Web Service (connected to GitHub `main`, using `Dockerfile` or `uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT`).
3. **AI Providers (Free Tiers):**
   - Primary LLM: Google Gemini (`gemini-3.5-flash-lite`) via `GEMINI_API_KEY`.
   - Primary Audio STT: Groq Whisper (`whisper-large-v3-turbo`) via `GROQ_API_KEY`.
   - Local Intent Classifier: Fine-tuned CAMeLBERT-Mix LoRA (runs locally on CPU; ~1.2–1.5 MB adapter, Apache-2.0 base).
4. **WhatsApp:** Meta WhatsApp Cloud API (`v27.0`) pointing webhook to `https://<your-service>.onrender.com/api/whatsapp/webhook` with `META_APP_SECRET` set for strict HMAC-SHA256 `X-Hub-Signature-256` verification.

> **Commercial Upgrade Warning:** Before accepting real paying MOKA customers 24/7, upgrade the web service off a sleep-on-idle free tier (so Meta webhooks never time out during a 60s cold start) and enable automated daily PostgreSQL backups.

---

## 2. Required Production Environment Variables

Never commit `.env` to Git. Set these in your hosting provider's Secret/Environment panel:

```env
WASLA_ENV=production
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DBNAME?sslmode=require
WASLA_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_urlsafe(48))">
WASLA_INITIAL_ADMIN_PASSWORD=<strong 14+ char password; forced change on first login>

GEMINI_API_KEY=<rotated Gemini key>
GEMINI_MODEL=gemini-3.5-flash-lite
GROQ_API_KEY=<rotated Groq key>
GROQ_STT_MODEL=whisper-large-v3-turbo

META_ACCESS_TOKEN=<permanent System User token>
META_PHONE_NUMBER_ID=1515063625018594
META_GRAPH_VERSION=v27.0
META_VERIFY_TOKEN=<random 32-char verify token>
META_APP_SECRET=<Meta App Secret for X-Hub-Signature-256 verification>
```

### Startup Security Fail-Closed Guardrails (`backend/app/main.py`)
When `WASLA_ENV=production`, the server **refuses to boot** if:
- `WASLA_SECRET_KEY` is missing, weak, or uses a development default.
- `DATABASE_URL` points to SQLite instead of PostgreSQL.
- `WASLA_INITIAL_ADMIN_PASSWORD` is missing when bootstrapping an empty user table.
- Default development credentials are present.

---

## 3. PostgreSQL Initialization & Alembic Migration Procedure

Once `psql` / PostgreSQL (local Docker or Neon) is reachable:

```bash
# 1. Verify connectivity
python -c "import os, sqlalchemy; e = sqlalchemy.create_engine(os.environ['DATABASE_URL']); print(e.connect().exec_driver_sql('SELECT 1').scalar())"

# 2. Run Alembic migrations
alembic -c backend/alembic.ini upgrade head

# 3. Verify downgrade / re-upgrade safety (staging only)
alembic -c backend/alembic.ini downgrade base
alembic -c backend/alembic.ini upgrade head

# 4. Boot server and check readiness
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
curl -s http://127.0.0.1:8000/api/ready
```

---

## 4. Docker Deployment Procedure

When Docker is installed on the target host:

```bash
docker compose build
docker compose up -d
docker compose ps
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/ready
```

---

## 5. Key Rotation & Security Checklist Before Go-Live

1. **Rotate all API keys** (`GEMINI_API_KEY`, `GROQ_API_KEY`, `META_ACCESS_TOKEN`) that were ever shared in chat or local `.env`.
2. **Set `META_APP_SECRET`** so every inbound `POST /api/whatsapp/webhook` request is cryptographically verified via `X-Hub-Signature-256`.
3. **Complete first-login password change** on the initial admin account (`must_change_password=True` enforced by `POST /api/auth/change-password`).
4. **Verify backup & restore** on your live PostgreSQL database (`pg_dump` + restore to staging DB).
