# Production Release Report

**Release decision: NOT READY for production deployment.** The CI correction is verified, but production secrets were exposed in chat and live integrations cannot safely be used until those credentials are revoked and replaced through authorized secret managers.

**Report date:** 2026-11-25  
**Repository:** `mayarmajed2/wasla-ai-dispatcher`  
**Latest `main` base inspected:** `1a4db9e82e5479fafc22d7d2e52b3fb9d91e1c57`

## Failed workflow and root cause

GitHub Actions run [45508151292](https://github.com/mayarmajed2/wasla-ai-dispatcher/actions/runs/45508151292), on the latest `main` commit above, failed in `test-and-verify`, step 9, **Run full regression & AI evaluation suite**. The suite reported 119 passed and one failed:

`wasla_phase_a_repro_tests.py::test_dispatcher_skips_incompatible_finetuned_intent_classifier`

The test correctly confirmed that the incompatible fine-tuned classifier was not called and that the greeting intent was preserved. Its final assertion expected the obsolete route label `offline_rules`, while the active local Egyptian Arabic NLU returns `offline_egyptian_nlu` (also asserted by the existing dispatcher tests). This was a stale regression expectation, not a runtime routing, dependency, or CI environment failure. Earlier setup, dependency, secret-scan, artifact, and frontend checks passed; the PostgreSQL step was skipped because the test step failed. The independent Docker verification job succeeded.

The test now asserts the current route label; no test was removed, skipped, or weakened, and the classifier-safety and parsed-intent assertions remain unchanged.

## Verification

Correction commit: `8e63792980d14ad638585040b5a3aab6fa42932d` on `mayarmajed2-fix-regression-ci`; [PR #4](https://github.com/mayarmajed2/wasla-ai-dispatcher/pull/4).

GitHub Actions run [45511002675](https://github.com/mayarmajed2/wasla-ai-dispatcher/actions/runs/45511002675) ran against that pushed SHA. Both required jobs passed:

- **`test-and-verify` — passed.** The full regression/evaluation step passed; security, dataset/model-artifact, and frontend checks passed; PostgreSQL 16 migrations and persistence/backup-restore round-trip completed. The verification output reported `backup_restore_roundtrip_verified: true` and `pg_dump_binary_verified: false`; native `pg_dump` therefore remains unverified.
- **`docker-production-verify` — passed.** The production Docker image built, booted against the workflow's isolated PostgreSQL 16 service, passed its synthetic webhook verification, and recovered after container restart. This is CI verification with temporary test configuration, not a production deployment or live provider/customer test.
- **Local full regression command — 120 passed.** The targeted incompatible-classifier regression also passed independently.

The earlier Hugging Face deployment workflow reported that it skipped upload because its required deployment configuration was unavailable at that run. No deployment was performed as part of this correction.

## Staging workflow preparation

The deployment workflow is `.github/workflows/deploy-hf-space.yml`. It now targets only a manually dispatched Hugging Face staging deployment, requires an explicit staging confirmation, uses GitHub Environment `env`, and no longer deploys automatically on pushes to `main`. The staging upload credentials are deliberately distinct from repository-level deployment credentials:

- Add GitHub Environment secret `HF_STAGING_TOKEN`.
- Add GitHub Environment variable `HF_STAGING_SPACE_ID` with the non-production staging Space identifier.
- Add GitHub Environment secret `META_APP_SECRET` for the app's production-mode webhook signature requirement.

The current Environment `env` inventory does not include those three names. Generic `HF_TOKEN` and `HF_SPACE_ID` names exist at repository scope, but the workflow intentionally does not use them: their target and suitability for staging are not established. The environment also has names for `DATABASE_URL`, `AUTH_SECRET_KEY`, and `BOOTSTRAP_ADMIN_PASSWORD`; their values and staging suitability have not been verified. `APP_ENV` exists as a variable; set the staging runtime to `production` mode to enable the app's production safety gates.

The GitHub Actions Environment applies to the workflow job; it does **not** copy application secrets or variables into the Hugging Face Space. The current workflow uploads application files only. The runtime must separately receive `APP_ENV`, `DATABASE_URL`, `AUTH_SECRET_KEY`, `BOOTSTRAP_ADMIN_PASSWORD`, and `META_APP_SECRET` through that staging Space's own authorized Variables and Secrets settings. Configure any provider integrations there as well (for example `GEMINI_API_KEY`, `GROQ_API_KEY`, `META_ACCESS_TOKEN`, `META_VERIFY_TOKEN`, `META_PHONE_NUMBER_ID`, or Telegram settings as needed). No provider configuration or credentials were transmitted by this task.

Repository workflow inventory: `ci.yml` runs CI for pushes to `main` and pull requests; `deploy-hf-space.yml` is now manual-only; `scheduler-cron.yml` periodically calls the configured scheduler endpoint and also supports manual dispatch; `telegram-cloud-probe.yml` is manual-only and can send test messages when explicitly selected. The staging deployment workflow was not dispatched. The pushed branch's PR CI is separate and does not run the deployment, scheduler, or Telegram workflows.

The staging workflow YAML was statically parsed and checked for the manual-only trigger, `env` binding, staging-scoped secret/variable references, and default-off confirmation. Commit `d989d987d422ff1036cd91422f7aaf0a8f6e1c0e` passed both PR CI jobs in [run 45517095649](https://github.com/mayarmajed2/wasla-ai-dispatcher/actions/runs/45517095649). No staging upload or provider connectivity test was run.

## Production blockers and integration status

All credential values previously pasted in chat remain treated as compromised. They were not read from files, used for tests, sent to providers, or included in this report. Rotate them before use; secret names alone do not prove that the configured values are valid, rotated, or staging-only.

No live Gemini, Groq, Meta/WhatsApp, Telegram, Hugging Face runtime, production PostgreSQL, or customer-flow verification was performed. No courier/tracking provider is configured or verified; stored order states must not be presented as courier location or ETA.

**Next release gate:** add the staging-only GitHub Environment entries listed above; configure the app's runtime variables/secrets directly in the staging Hugging Face Space; verify those values are staging-only and rotated; then separately authorize and run the staging deployment and live connectivity checks. Until those steps are completed, this is safe workflow preparation only—not a staging deployment or production readiness sign-off.
