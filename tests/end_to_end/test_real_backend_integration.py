"""Tier 5 — Comprehensive End-to-End Integration Tests (RBAC Auth, Isolated DB, No Simulator in Prod)."""

import asyncio
import hashlib
import hmac
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

    for mod in ["app.config", "app.database", "app.models", "app.auth", "app.main"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import app.config as cfg_mod  # noqa: E402
    import app.main as main_mod  # noqa: E402

    wasla_app = main_mod.app
    settings = cfg_mod.settings
    main_mod._migrate_sqlite_columns()
    main_mod.seed_moka_menu_if_empty()
    main_mod.seed_initial_admin_if_empty()

    async def _run():
        transport = httpx.ASGITransport(app=wasla_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Dashboard static assets & HTML (NO AI Simulator present!)
            home = await client.get("/")
            assert home.status_code == 200
            assert "Wasla AI" in home.text
            assert "محاكي الوكيل الذكي" not in home.text

            # 2. Unauthenticated access to protected endpoints MUST fail with 401
            assert (await client.get("/api/orders")).status_code == 401
            assert (await client.get("/api/metrics")).status_code == 401
            assert (await client.get("/api/staff")).status_code == 401

            # 3. Admin Login
            login_resp = await client.post(
                "/api/auth/login",
                json={"username": settings.bootstrap_admin_username, "password": settings.bootstrap_admin_password},
            )
            assert login_resp.status_code == 200
            admin_token = login_resp.json()["access_token"]
            admin_headers = {"Authorization": f"Bearer {admin_token}"}

            # 4. Create Cashier account & verify RBAC (Cashier blocked from /api/staff with 403)
            c_staff = await client.post(
                "/api/staff",
                headers=admin_headers,
                json={"username": "cashier1", "full_name": "كاشير أول", "password": "CashierPass123!", "role": "cashier"},
            )
            assert c_staff.status_code == 201
            cashier_login = await client.post("/api/auth/login", json={"username": "cashier1", "password": "CashierPass123!"})
            cashier_headers = {"Authorization": f"Bearer {cashier_login.json()['access_token']}"}
            assert (await client.get("/api/staff", headers=cashier_headers)).status_code == 403

            # 5. Health & 145 MOKA menu items seeded in isolated DB
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["menu_items_loaded"] == 145

            # 6. Metrics start cleanly at 0 real orders & 0.00 revenue
            metrics_before = (await client.get("/api/metrics", headers=admin_headers)).json()
            assert metrics_before["menu_total"] == 145
            assert metrics_before["total_orders"] == 0
            assert metrics_before["total_revenue_egp"] == "0.00"

            # 7. Toggle item availability
            menu_all = (await client.get("/api/menu/all", headers=cashier_headers)).json()
            first_id = menu_all[0]["id"]
            t_off = await client.patch(
                f"/api/menu/{first_id}/availability", headers=cashier_headers, json={"is_available": False}
            )
            assert t_off.status_code == 200 and t_off.json()["is_available"] is False
            await client.patch(f"/api/menu/{first_id}/availability", headers=cashier_headers, json={"is_available": True})

            # 8. Real Webhook Order Journey (is_test=False) -> exact 190.00 EGP total
            wh_msgs = [
                ("wamid.real_1", "عايز بيتزا مارجريتا كبير حبة كاملة"),
                ("wamid.real_2", "الاسم: سارة حسن، العنوان: الدقي شارع التحرير 22، رقم بديل: 01234567890 تمام اكد الطلب"),
            ]
            for mid, txt in wh_msgs:
                wh_payload = {
                    "entry": [{"changes": [{"value": {"messages": [{"id": mid, "from": "01077778888", "type": "text", "text": {"body": txt}}]}}]}]
                }
                raw_bytes = json.dumps(wh_payload).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                if settings.meta_app_secret:
                    headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(
                        settings.meta_app_secret.encode("utf-8"), raw_bytes, hashlib.sha256
                    ).hexdigest()
                wr = await client.post("/api/webhook", content=raw_bytes, headers=headers)
                assert wr.status_code == 200 and wr.json()["processed"] == 1

            # Verify real order appears in GET /api/orders with exact 190.00 total and updates metrics
            orders = (await client.get("/api/orders", headers=cashier_headers)).json()
            assert len(orders) == 1
            order_id = orders[0]["id"]
            assert orders[0]["total_amount"] == "190.00"
            assert orders[0]["is_test"] is False
            assert orders[0]["data_source"] == "whatsapp_webhook"

            metrics_after = (await client.get("/api/metrics", headers=cashier_headers)).json()
            assert metrics_after["total_orders"] == 1
            assert metrics_after["total_revenue_egp"] == "190.00"

            # 9. Order status transition & Cashier Directive
            st = await client.patch(f"/api/orders/{order_id}/status", headers=cashier_headers, json={"status": "IN_KITCHEN"})
            assert st.status_code == 200 and st.json()["status"] == "IN_KITCHEN"

            inst = await client.post(
                f"/api/orders/{order_id}/instruct",
                headers=cashier_headers,
                json={"directive": "الاستربس خلصان، اعرض زنجر بنفس السعر", "directive_type": "free_text"},
            )
            assert inst.status_code == 200
            assert "كريب زنجر حار" in inst.json()["customer_message_ar"]

            # 10. Audit logs captured admin & cashier actions
            audit_logs = (await client.get("/api/audit-logs", headers=admin_headers)).json()
            assert len(audit_logs) >= 4

    asyncio.run(_run())
    assert isolated_db.exists()
