# Wasla AI — Commercial Operations & Handover Manual

## 1. Restaurant Owner Guide
- **Dashboard URL:** `https://<your-domain>/` (local: `http://127.0.0.1:8000/`)
- **Initial Admin Login:** Configured via `BOOTSTRAP_ADMIN_USERNAME` and `BOOTSTRAP_ADMIN_PASSWORD` in `.env`. Change immediately upon first deployment.
- **Live Operations Views:**
  1. **الطلبات الحية (Live Orders):** Displays genuine operational orders only (`لا توجد طلبات حتى الآن` when empty). No demo/simulator data is mixed into revenue or KPIs.
  2. **محادثات العملاء (Customer Conversations):** Tracks WhatsApp customer sessions and highlights any `مطلوب كاشير` human handoff requests.
  3. **المنيو والتوافر (Menu & Inventory):** Manage all 145 MOKA menu items with 1-click Out-of-Stock toggles.
  4. **تحليلات المبيعات (Sales Analytics):** Real SQL aggregates (total orders, revenue, fulfilled, cancelled, active conversations).
  5. **حالة AI والتكاملات (AI & Integration Health):** Live status of Database, Google Gemini, Groq Whisper, Meta WhatsApp Cloud API (`v27.0`), and HMAC Webhook verification.
  6. **الموظفين والصلاحيات (Staff & Audit Logs — Admin Only):** Create cashier/admin accounts and review security audit logs.

## 2. Cashier Daily Guide
1. Sign in with your Cashier account and click **تفعيل التنبيه الصوتي** at shift start.
2. When a new order arrives via WebSocket, review the itemized list, backup phone, and delivery address, then click **إرسال للمطبخ** → **خرج للتوصيل** → **تم التسليم**.
3. If an item runs out, open **المنيو والتوافر** and click **إيقاف** so Wasla AI stops accepting orders for that item immediately.
4. For delivery issues or kitchen delays, click **توجيه للعميل** on the order card and choose a quick preset or enter a custom instruction.

## 3. Administrator Deployment Guide (Docker + PostgreSQL)
```bash
cp .env.example .env
# Edit .env with real GEMINI_API_KEY, GROQ_API_KEY, META_ACCESS_TOKEN,
# META_PHONE_NUMBER_ID, META_APP_SECRET, AUTH_SECRET_KEY, POSTGRES_PASSWORD
docker compose up -d --build
# Run Alembic migrations against PostgreSQL:
docker compose exec wasla-app bash -c "cd /app/backend && alembic upgrade head"
```

## 4. Backup & Recovery
```bash
# Backup (SQLite or PostgreSQL)
python scripts/backup_restore.py backup --db "$DATABASE_URL" --out backups/
# Restore (SQLite with SHA-256 verification)
python scripts/backup_restore.py restore --backup backups/<file>.db --target restaurant_ai.db
```

## 5. Activating GitHub Actions CI/CD
```bash
gh auth refresh -s workflow
git mv docs/ci-workflow-template.yml .github/workflows/ci.yml
git commit -m "ci: activate GitHub Actions workflow"
git push origin main
```
