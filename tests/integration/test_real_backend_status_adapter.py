from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.adapters.real_backend_adapter import RealBackendAdapter
from ai_service.understanding.menu_index import MenuIndex

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_real_adapter_uses_authenticated_customer_status_and_backend_delivery_quote():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/customer/order-status":
            return httpx.Response(200, json={"id": 41, "status": "IN_KITCHEN"})
        if request.url.path == "/api/delivery/quote":
            return httpx.Response(
                200,
                json={
                    "deliverable": True,
                    "zone_name": "nasr-city",
                    "delivery_fee": "27.50",
                    "estimated_minutes": None,
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = RealBackendAdapter(
            MenuIndex(ROOT / "menu" / "menu.json"),
            base_url="https://backend.invalid",
            client=client,
            service_api_key="test-service-key",
        )
        status = await adapter.get_customer_order_status("01055550041", 41)
        quote = await adapter.resolve_delivery("01055550041", "nasr-city")

    assert status == {"id": 41, "status": "IN_KITCHEN"}
    status_request = next(request for request in seen if request.url.path.endswith("order-status"))
    assert status_request.headers["X-Backend-Service-Key"] == "test-service-key"
    assert status_request.content == b'{"customer_phone":"01055550041","order_id":41}'
    assert quote.delivery_fee == Decimal("27.50")
    assert quote.estimated_minutes is None


@pytest.mark.asyncio
async def test_real_adapter_refuses_customer_status_without_service_auth():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(500))) as client:
        adapter = RealBackendAdapter(
            MenuIndex(ROOT / "menu" / "menu.json"),
            client=client,
            service_api_key="",
        )

        with pytest.raises(BackendIntegrationError) as error:
            await adapter.get_customer_order_status("01055550042")

    assert error.value.code == "ORDER_TRACKING_UNAVAILABLE"
    assert error.value.status_code == 503
