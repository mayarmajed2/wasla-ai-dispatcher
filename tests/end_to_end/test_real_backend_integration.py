"""Tier 5 — Comprehensive End-to-End Integration Tests using an ISOLATED temporary SQLite database.

Guarantees automated tests NEVER pollute the main restaurant_ai.db.
"""

import asyncio
import hashlib
import hmac
import importlib
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_full_integrated_wasla_backend_ai_and_dashboard_e2e(tmp_path):
    isolated_db = tmp_path / "isolated_test_wasla.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{isolated_db.as_posix()}"

    # Reload config, database, models, and main against the isolated test DB
    for mod in ["app.config", "app.database", "app.models", "app.main"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import app.config as cfg_mod  # noqa: E402
    import app.main as main_mod  # noqa: E402

    wasla_app = main_mod.app
    settings = cfg_mod.settings
    main_mod._migrate_sqlite_columns()
    main_mod.seed_moka_menu_if_empty()

    async def _run():
        transport = httpx.ASGITransport(app=wasla_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Dashboard static assets & HTML
            home = await client.get("/")
            assert home.status_code == 200
            assert "Wasla AI" in home.text

            css = await client.get("/assets/styles.css")
            assert css.status_code == 200
            js = await client.get("/assets/app.js")
            assert js.status_code == 200

            # 2. Health & 145 MOKA menu items seeded in isolated DB
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["menu_items_loaded"] == 145

            # 3. Real SQL metrics endpoint starts cleanly at 0 real orders & 0 revenue
            metrics_before = (await client.get("/api/metrics")).json()
            assert metrics_before["menu_total"] == 145
            assert metrics_before["total_orders"] == 0
            assert metrics_before["total_revenue_egp"] == "0.00"

            # 4. Toggle item availability (out-of-stock & back)
            menu_all = (await client.get("/api/menu/all")).json()
            first_id = menu_all[0]["id"]
            t_off = await client.patch(f"/api/menu/{first_id}/availability", json={"is_available": False})
            assert t_off.status_code == 200 and t_off.json()["is_available"] is False
            t_on = await client.patch(f"/api/menu/{first_id}/availability", json={"is_available": True})
            assert t_on.status_code == 200 and t_on.json()["is_available"] is True

            # 5. Live Egyptian Arabic AI Simulator chat -> creates simulator/is_test=True order
            # Verify exact Pizza L price: 150 (L) + 20 (حبة كاملة) + 20 (delivery) = 190.00 EGP
            phone = "01077778888"
            c1 = await client.post("/api/ai/chat", json={"phone_number": phone, "message": "عايز بيتزا مارجريتا كبير حبة كاملة"})
            assert c1.status_code == 200
            assert c1.json()["requires_clarification"] is False

            c2 = await client.post(
                "/api/ai/chat",
                json={
                    "phone_number": phone,
                    "message": "الاسم: سارة حسن، العنوان: الدقي شارع التحرير 22، رقم بديل: 01234567890 تمام اكد الطلب",
                },
            )
            assert c2.status_code == 200
            order_id = c2.json()["order_id"]
            assert order_id is not None

            # 6. Verify exact 190.00 total AND that simulator order is flagged is_test=True
            orders = (await client.get("/api/orders")).json()
            created = next(o for o in orders if o["id"] == order_id)
            assert created["total_amount"] == "190.00"
            assert created["is_test"] is True
            assert created["data_source"] == "simulator"

            # 7. Verify production KPIs strictly exclude simulator/test orders!
            metrics_after = (await client.get("/api/metrics")).json()
            assert metrics_after["total_orders"] == 0
            assert metrics_after["total_revenue_egp"] == "0.00"
            assert metrics_after["test_or_demo_orders"] == 1

            # 8. Order lifecycle transition (SUBMITTED -> IN_KITCHEN)
            st = await client.patch(f"/api/orders/{order_id}/status", json={"status": "IN_KITCHEN"})
            assert st.status_code == 200 and st.json()["status"] == "IN_KITCHEN"

            # 9. Cashier Directive Interpreter integration
            inst = await client.post(
                f"/api/orders/{order_id}/instruct",
                json={"directive": "الاستربس خلصان، اعرض زنجر بنفس السعر", "directive_type": "free_text"},
            )
            assert inst.status_code == 200
            inst_data = inst.json()
            assert "كريب زنجر حار" in inst_data["customer_message_ar"]
            assert inst_data["requires_cashier_approval"] is True

            # 10. Meta WhatsApp Webhook GET challenge + POST deduplication & AI dispatch
            verify = await client.get(
                "/api/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": settings.meta_verify_token,
                    "hub.challenge": "CHALLENGE_123",
                },
            )
            assert verify.status_code == 200 and verify.text == "CHALLENGE_123"

            wh_payload = {
                "entry": [
                    {
                        "changes": [
                            {
                                "value": {
                                    "messages": [
                                        {
                                            "id": f"wamid.test_{order_id}",
                                            "from": "01099887766",
                                            "type": "text",
                                            "text": {"body": "كريب كريسبي بكام؟"},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
            raw_bytes = json.dumps(wh_payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if settings.meta_app_secret:
                headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(
                    settings.meta_app_secret.encode("utf-8"), raw_bytes, hashlib.sha256
                ).hexdigest()

            w1 = await client.post("/api/webhook", content=raw_bytes, headers=headers)
            assert w1.status_code == 200
            assert w1.json()["processed"] == 1
            assert len(w1.json()["replies"]) == 1
            assert "110" in w1.json()["replies"][0]["reply_ar"]

            # Duplicate Meta message ID must be ignored (idempotent)
            w2 = await client.post("/api/webhook", content=raw_bytes, headers=headers)
            assert w2.status_code == 200 and w2.json()["processed"] == 0

    asyncio.run(_run())
    assert isolated_db.exists()
