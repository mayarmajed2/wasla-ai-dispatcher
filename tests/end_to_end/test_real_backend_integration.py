"""Tier 5 — Real Backend End-to-End Integration Test against existing_system/RESTURANT AI."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "existing_system" / "RESTURANT AI"
MENU_PATH = ROOT / "menu" / "menu.json"


def test_wasla_dispatcher_dispatches_to_real_existing_fastapi_backend():
    if not (BACKEND_DIR / "app" / "main.py").exists():
        pytest.skip("Existing FastAPI backend not present.")

    try:
        import sqlalchemy  # noqa: F401
        import pydantic_settings  # noqa: F401
    except ImportError:
        pytest.skip("Optional backend deps (sqlalchemy/pydantic-settings) not installed in current Python env.")

    sys.path.insert(0, str(BACKEND_DIR))
    from app.main import app as real_backend_app, seed_menu  # noqa: E402
    from ai_service.adapters.real_backend_adapter import RealBackendAdapter  # noqa: E402
    from ai_service.config import WaslaConfig  # noqa: E402
    from ai_service.dispatcher.dispatcher import WaslaAIDispatcher  # noqa: E402
    from ai_service.understanding.menu_index import MenuIndex  # noqa: E402

    seed_menu()

    async def _run():
        transport = httpx.ASGITransport(app=real_backend_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            idx = MenuIndex(MENU_PATH)
            adapter = RealBackendAdapter(idx, base_url="http://testserver", client=client)
            dispatcher = WaslaAIDispatcher(WaslaConfig(gemini_api_key=""), idx, adapter)

            # Verify live /api/menu retrieval from real FastAPI Backend
            live_menu = await adapter.get_active_menu()
            assert len(live_menu) >= 1

            # Dispatch a full order into real FastAPI POST /api/orders
            phone = "01055556666"
            await dispatcher.handle_text(phone, "عايز واحد كريب بطاطس")
            # Map item_001's backend_item_id to an existing seeded item in real DB
            dispatcher.get_session(phone).cart.items[0].backend_item_id = live_menu[0].backend_item_id
            resp = await dispatcher.handle_text(
                phone,
                "الاسم: سارة حسن، العنوان: الدقي شارع التحرير 22، رقم بديل: 01234567890 تمام اكد الطلب",
            )
            assert resp.order_id is not None

            # Verify order actually persisted in real Backend GET /api/orders
            orders_resp = await client.get("/api/orders")
            assert orders_resp.status_code == 200
            assert any(o["id"] == resp.order_id for o in orders_resp.json())

    asyncio.run(_run())
