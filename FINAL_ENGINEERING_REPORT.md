# Final Engineering Report

**Repository:** `mayarmajed2/wasla-ai-dispatcher`
**Branch:** `mayarmajed2-qa-docs-release`
**Report Date:** 24 November 2026

## Delivery Summary

This work completes QA, documentation, and release preparation for the current Wasla AI implementation. It adds a deterministic critical-surface regression suite and aligns README/reporting with measured local results.

## Implemented and Verified Locally

- FastAPI, SQLAlchemy, and Alembic application structure
- Hybrid LoRA/deterministic NLU/Gemini dispatcher
- Per-phone lock and immutable turn context
- 14+ field persisted session/cart state
- Deterministic `Decimal` cart arithmetic
- Explicit confirmation and order idempotency
- Human takeover and complaint escalation
- Meta HMAC/deduplication and debouncer behavior
- Telegram private-chat webhook and notification behavior
- Cashier dashboard and WebSocket integration
- Focused critical intent, cart, session, and takeover regressions

## Measured QA

The established repository suite was run on 24 November 2026:

```text
py -m pytest -q wasla_phase_a_repro_tests.py tests/ backend/test_cart_calculator.py
59 passed, 2 failed
```

Both failures are real-LoRA model loading tests. The host reports: `DLL load failed while importing _regex: An Application Control policy has blocked this file.` This is an environment-level Windows policy blocker, not a passing assertion and not evidence of successful live model loading on this host.

The new deterministic suite was run separately:

```text
py -m pytest -q tests/regression/test_critical_dispatcher_regressions.py
10 passed
```

The new suite covers greeting/menu/price/gibberish intents, add/remove/cancel cart lifecycle, explicit confirmation and replay, upsell accept/reject memory, complaint/human takeover, cross-customer isolation, TTL epoch reset, prompt-injection guard behavior, and cart preservation.

## Not Claimed

- No live Gemini or Groq provider call was executed.
- No live Telegram, WhatsApp, Meta, PostgreSQL, or external paid service was verified.
- No live order or live customer-message integration claim is made.
- No production metric, latency, uptime, conversion, or revenue claim is made.

## Release Readiness

The implementation is suitable for local and mocked verification, with external integration credentials still required for live deployment. Production release should remain blocked until:

1. the Windows Smart App Control/native dependency issue is resolved or the host is moved to an allowed environment;
2. real LoRA loading succeeds on the target runtime;
3. Gemini/Groq credentials and provider calls are tested;
4. PostgreSQL migrations, backup/restore, and live webhook delivery are verified;
5. Telegram/WhatsApp targets and secrets are configured from deployment secrets.
