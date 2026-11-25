# Wasla AI — Restaurant AI Dispatcher (MOKA Pilot)

Wasla AI is an Egyptian Arabic ordering and cashier-operations application piloted with MOKA Restaurant & Cafe. The repository includes a FastAPI backend, a restaurant operations dashboard, menu-grounded AI/NLU, and optional external-service integrations.

> **Readiness:** The code, CI tests, PostgreSQL service checks, and Docker image have passed the repository's automated checks. This is not evidence of a live staging deployment, provider connectivity, customer traffic, or production readiness. Do not put credentials in this README, source control, or chat.

## What it includes

- **Menu-grounded ordering:** Egyptian Arabic NLU, optional Gemini function calling, menu-backed item matching, deterministic `Decimal` totals, and explicit confirmation before order submission.
- **Safe fallback behavior:** The checked-in LoRA checkpoints use labels incompatible with the application's `CustomerIntent` taxonomy and are excluded from intent routing. Order status responses use stored restaurant order state and phone-scoped lookups; no courier location or ETA is fabricated.
- **Backend and persistence:** FastAPI, SQLAlchemy, Alembic migrations, PostgreSQL or local SQLite, durable sessions and inbound-message debouncing.
- **Operations dashboard:** Arabic RTL cashier interface, staff authentication, order operations, inventory controls, WebSocket notifications, and runtime readiness reporting.
- **Optional integrations:** Gemini, Groq audio transcription, Meta WhatsApp, Telegram, and scheduled jobs. Local mocks and automated tests do not establish live provider connectivity.

## Repository layout

```text
ai_service/       AI dispatcher, NLU, models, audio, tools, adapters
backend/          FastAPI application, persistence, migrations, auth
dashboard/        Arabic RTL cashier and operations dashboard
menu/             Canonical MOKA menu data
training/         LoRA checkpoints, datasets, and evaluation scripts
datasets/         Evaluation data
tests/            Unit, integration, and regression tests
deploy/           Hugging Face Space entry point and deployment assets
docs/             Architecture, operations, integration, and release docs
.github/workflows/ CI, staging deployment, scheduler, and manual probe workflows
```

## Local setup

Use Python 3.12, then install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

For local development, copy `.env.example` to `.env`, review every setting, and keep `.env` out of version control:

```bash
cp .env.example .env
```

The example file is a template, not a credential source. Values shown as examples or defaults are **not working credentials**. For local tests, leave external provider credentials unset unless you are intentionally configuring a test account. Never use exposed or production credentials for staging.

Apply the current Alembic migrations and start the backend from the repository root:

```bash
alembic -c backend/alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Open `http://localhost:8000` for the dashboard and `http://localhost:8000/api/ready` for application readiness information. A local SQLite setup is suitable for development only; production-mode startup requires PostgreSQL, a strong authentication key, a changed bootstrap-admin password, and Meta webhook signature configuration.

## Tests and CI

The same pytest selection used in GitHub Actions can be run locally from PowerShell:

```powershell
$env:PYTHONPATH = ".;backend"
$env:APP_ENV = "development"
$env:GEMINI_API_KEY = ""
$env:GROQ_API_KEY = ""
$env:META_ACCESS_TOKEN = ""
py -m pytest -v tests/ wasla_phase_a_repro_tests.py backend/test_cart_calculator.py
```

The GitHub Actions `test-and-verify` job also checks committed-secret hygiene, menu/model artifacts, frontend syntax, and migrations/persistence against an isolated PostgreSQL 16 service. The separate `docker-production-verify` job builds and boots the production Docker image against its own isolated test database and verifies restart behavior. These CI jobs use test configuration; they are not live production or provider integration tests.

The merged CI correction and staging-workflow configuration passed both jobs on the merge commit `01bf8a9`. The PostgreSQL CI backup/restore round-trip passed; native `pg_dump` was not verified. See [the production release report](PRODUCTION_RELEASE_REPORT.md) for the recorded evidence and limitations.

## Hugging Face staging deployment

`.github/workflows/deploy-hf-space.yml` uploads the application bundle to Hugging Face Spaces. It is **manual-only**; pushing to `main` does not deploy. The job uses the GitHub Actions Environment named `env` and requires explicit confirmation that the selected Space is staging.

Before dispatching **Deploy Wasla AI to Hugging Face Staging Space**, configure these entries in the repository's GitHub Environment named `env`:

| Environment entry | Type | Purpose |
|---|---|---|
| `HF_STAGING_TOKEN` | Secret | Hugging Face token authorized to deploy to the staging Space |
| `HF_STAGING_SPACE_ID` | Variable | Non-production Hugging Face Space identifier |

Select `confirm_staging` only after verifying that `HF_STAGING_SPACE_ID` identifies a staging Space. The workflow fails before upload if confirmation or either setting is missing. Do not substitute repository-level or production Hugging Face credentials.

**The workflow uploads files only.** GitHub Environment settings are available to the workflow job; they are not copied into the Hugging Face Space's runtime environment. Configure the application separately in the staging Space's own Variables and Secrets settings. At minimum, production-mode application startup requires:

| Staging Space setting | Type | Purpose |
|---|---|---|
| `APP_ENV` | Variable | Set to `production` to enable production safety checks |
| `DATABASE_URL` | Secret | Staging PostgreSQL connection string |
| `AUTH_SECRET_KEY` | Secret | Strong staging-only signing key |
| `BOOTSTRAP_ADMIN_PASSWORD` | Secret | Strong staging-only initial admin password |
| `META_APP_SECRET` | Secret | Required at startup for webhook signature verification |

Also configure `DASHBOARD_PUBLIC_URL` and `BOOTSTRAP_ADMIN_USERNAME` as appropriate for the staging Space. Add provider settings only if that staging integration is intentionally enabled—for example `GEMINI_API_KEY`, `GROQ_API_KEY`, `META_VERIFY_TOKEN`, `META_ACCESS_TOKEN`, `META_PHONE_NUMBER_ID`, or Telegram configuration. Use rotated, staging-only credentials from their authorized secret managers. Do not use placeholders or previously exposed values.

Successful workflow completion would confirm that files were uploaded; it would **not** by itself verify that the Space booted, connected to the staging database, or reached any external provider. Verify those separately before treating staging as functional. No staging deployment or live provider connectivity is claimed here.

## Other workflows

- `.github/workflows/ci.yml` runs CI for pushes to `main` and pull requests targeting `main`.
- `.github/workflows/scheduler-cron.yml` periodically calls the configured scheduler endpoint and also supports manual dispatch. It is separate from application deployment.
- `.github/workflows/telegram-cloud-probe.yml` is manual-only; its optional send-test mode sends live Telegram messages and should be used only with authorized staging/test chats.

## Further documentation

- [Production release report and verified limitations](PRODUCTION_RELEASE_REPORT.md)
- [Deployment readiness and `/api/ready` contract](docs/DEPLOYMENT_READINESS.md)
- [Production deployment guide](docs/PRODUCTION_DEPLOYMENT.md)
- [Integration contract and tracking limitations](docs/INTEGRATION_CONTRACT.md)
- [Engineering reference](docs/ENGINEERING_REFERENCE.md)
- [AI architecture](docs/AI_ARCHITECTURE.md)
- [Regression test report](docs/REGRESSION_TEST_REPORT.md)
- [Customer experience and operations guide](docs/CUSTOMER_EXPERIENCE_AND_OPS.md)
- [Model card](docs/MODEL_CARD.md)
- [MOKA menu extraction report](menu/extraction_report.md)
