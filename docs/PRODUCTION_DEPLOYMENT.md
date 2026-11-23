# Wasla AI — Commercial Production Deployment & Zero-Cost Hosting Guide

> **Strict Honesty Notice:** Every provider free tier below was verified directly against official 2026 vendor documentation. No credit card, paid subscription, or temporary tunnel is presented as production.

---

## 1. Official 2026 Zero-Cost Infrastructure Comparison (Verified)

| Provider | Credit Card Required? | Free Tier Specs & Limits | Sleeping / Cold Start | Verdict for 0 EGP Pilot |
|---|---|---|---|---|
| **Neon PostgreSQL (Free)** | **No** | **1 GB storage/project**, **100 CU-hours/mo** (~400h at 0.25 CU), SSL (`sslmode=require`), zero auto-overage | Scales to zero after 5 min idle (~0.5–1s wake) | **SELECTED** for persistent PostgreSQL |
| **Render Web Services (Free)** | **No** | **750 hours/mo**, 512 MB RAM, 0.1 CPU, public HTTPS + WebSockets | **Sleeps after 15 min idle (~30–60s cold start)**; ephemeral disk; Render docs state *"Do not use for production"* | **SELECTED (Controlled Pilot Only)** — NOT guaranteed 24/7 SLA |
| **Cloudflare Workers (Free)** | **No** | **100,000 req/day**, **3 Cron Triggers included**, 10ms CPU/req, 0ms cold start | **Never sleeps** | **SELECTED** as independent zero-cost Cron Trigger (`deploy/cloudflare_cron_worker.js`) calling `/api/scheduler/cron-tick` every 10 min |
| **GitHub Actions (Free)** | **No** | 2,000 CI min/mo (private repos), PostgreSQL 16 service containers | Cron jitter ~5–15 min | **SELECTED** for CI/CD (`docs/ci-workflow-template.yml`) + optional fallback cron |
| **Koyeb (Free)** | **YES (\$29 temp auth hold)** | 512 MB RAM, 0.1 vCPU, 1h idle sleep | 1–5s cold start | **REJECTED** — violates strict 0 EGP / no-card constraint |
| **Hugging Face Spaces** | Docker `cpu-basic` often requires HF PRO / Org | 2 vCPU, 16 GB RAM, non-persistent disk | Sleeps on inactivity (up to 48h) | **NOT SELECTED** — Docker Space creation restrictions & non-persistent disk |

---

## 2. Recommended Zero-Cost Split Architecture (Controlled Pilot)

```text
[WhatsApp Customer] ──(HTTPS Webhook + HMAC-SHA256)──► [Render Free FastAPI + Dashboard] ──► [Neon Free PostgreSQL (SSL)]
                                                              ▲         │
[Cloudflare Worker Cron (every 10m, 0 EGP)] ──(X-Scheduler-Secret)┘     ├──► [Telegram Cashier Group (0 EGP)]
                                                                        ├──► [Telegram Owner Escalation (0 EGP)]
                                                                        └──► [Gemini / Groq Whisper / CAMeLBERT LoRA v2]
```

**Honest Timing & Availability Limitation:**
- Because Render Free sleeps after 15 minutes of inactivity (~30–60s cold start), it **cannot guarantee 24/7 always-on commercial SLA** or exact second-level review timing (~±10 min jitter via Cloudflare Cron + wake latency).
- We explicitly do **NOT** fake always-on behavior with artificial keep-alive ping loops.
- All scheduled jobs are persisted in Neon PostgreSQL (`scheduled_jobs`) with **atomic worker leases (`locked_by`, `locked_until`)**, so sleeping/restarting never loses or duplicates any review, arrival check, or cashier notification.

---

## 3. Exact Live Verification Commands (Run Once Credentials Are Configured)

### A. Real PostgreSQL Verification (Neon Free — Live Verified)
Verified live against **Neon PostgreSQL 18.7** via SSL (`pg8000` / `psycopg`), running Alembic revisions `0001_initial_schema` → `0002_sales_cx_notifications` → `0003_scheduler_leases`, importing all **145 MOKA menu items**, verifying rollback, unique constraints, restart survival, and SHA-256 backup/restore (`reports/postgres_live_verification.json`):
```bash
DATABASE_URL="postgresql://USER:PASS@HOST/neondb?sslmode=require" \
  python scripts/verify_postgres_live.py --output reports/postgres_live_verification.json
```

### B. Render Free Blueprint Deployment (`/render.yaml`)
1. Connect repository `mayarmajed2/wasla-ai-dispatcher` on Render → **New Blueprint Instance** using root [`render.yaml`](../render.yaml) (`plan: free`).
2. Populate all `sync: false` secrets in Render Dashboard (`DATABASE_URL`, `BOOTSTRAP_ADMIN_PASSWORD`, rotated `GEMINI_API_KEY`, `GROQ_API_KEY`, `META_ACCESS_TOKEN`, `META_PHONE_NUMBER_ID`, `META_APP_SECRET`, valid `TELEGRAM_BOT_TOKEN`, `CASHIER_TELEGRAM_CHAT_ID`, `RESTAURANT_OWNER_TELEGRAM_CHAT_ID`).
3. Root [`Dockerfile`](../Dockerfile) automatically executes `alembic -c backend/alembic.ini upgrade head` and binds `uvicorn` to `${PORT:-8000}`.

### C. Real Telegram Bot Verification (Cashier Group + Owner Private Chat)
```bash
# 1. Validate bot token & discover authorized chat IDs (never prints token)
python scripts/setup_telegram_verify.py --output reports/telegram_live_verification.json

# 2. Send authorized live test alerts to cashier group & owner private chat
python scripts/setup_telegram_verify.py --send-test --output reports/telegram_live_verification.json
```

### D. Real Public HTTPS & Meta Webhook Verification
```bash
python scripts/verify_whatsapp_e2e_live.py \
  --base-url "https://<your-render-service>.onrender.com" \
  --verify-token "$META_VERIFY_TOKEN" \
  --app-secret "$META_APP_SECRET" \
  --output reports/whatsapp_e2e_live_verification.json
```

### E. GitHub Actions CI/CD (with PostgreSQL 16 Service Container)
Active in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and [`.github/workflows/scheduler-cron.yml`](../.github/workflows/scheduler-cron.yml).

---

## 4. Security & Key Rotation Checklist Before Accepting Paying Customers

1. **Rotate ALL exposed credentials immediately** (Neon DB password, `@wwaslabot` token, `GEMINI_API_KEY`, `GROQ_API_KEY`, `META_ACCESS_TOKEN`) and update `.env` / Render environment variables.
2. **Set `APP_ENV=production`**, strong 48-char **`AUTH_SECRET_KEY`**, **`SCHEDULER_CRON_SECRET`**, **`META_APP_SECRET`**, and a strong **`BOOTSTRAP_ADMIN_PASSWORD`** (first-login password change is enforced automatically in production).
3. **Never commit `.env`** to Git.
