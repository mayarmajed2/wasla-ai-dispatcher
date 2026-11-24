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

## Production blockers and integration status

All credential values previously pasted into chat are treated as compromised. They were not read from files, used for tests, sent to providers, or included in this report. Revoke and rotate them before any use; enter replacements directly in the authorized provider or GitHub secret manager, never in chat or source control.

The GitHub repository's secret-name inventory at inspection time showed Telegram, Hugging Face, scheduler, and public-URL secret names, but no names for Gemini, Groq, Meta access/app secrets, the production database URL, or the backend service API key. A secret name alone does not establish that its value is valid, uncompromised, or usable. No live Gemini, Groq, Meta/WhatsApp, Telegram, production PostgreSQL, or customer-flow verification was performed. No courier/tracking provider is configured or verified; stored order states must not be presented as courier location or ETA.

**Next release gate:** rotate every exposed credential, configure required replacements directly in the relevant authorized secret managers, and then perform separately authorized live integration checks before deploying. Until then, the verified result is CI and Docker readiness only, not production readiness.
