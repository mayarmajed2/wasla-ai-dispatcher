"""Real HTTP Adapter for the Existing FastAPI Restaurant Backend.

Directly integrates with `existing_system/RESTURANT AI/app/main.py` endpoints:
- GET  /api/menu
- POST /api/orders
- POST /api/orders/{order_id}/modification
- POST /api/orders/{order_id}/instruct
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.schemas.models import (
    BackendConfirmOrderPayload,
    CartSnapshot,
    DeliveryResolution,
    MenuItemDTO,
)
from ai_service.understanding.menu_index import MenuIndex


class RealBackendAdapter(MockBackendAdapter):
    """Production HTTP adapter delegating authoritative menu & order operations to FastAPI."""

    IS_MOCK = False

    def __init__(
        self,
        menu_index: MenuIndex,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(menu_index)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external_client = client

    async def _request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> Any:
        if self._external_client is not None:
            resp = await self._external_client.request(method, f"{self.base_url}{path}", json=json_body)
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(method, f"{self.base_url}{path}", json=json_body)

        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise BackendIntegrationError("BACKEND_HTTP_ERROR", str(detail), resp.status_code)
        return resp.json()

    async def get_active_menu(self, category: str | None = None) -> list[MenuItemDTO]:
        rows = await self._request("GET", "/api/menu")
        result: list[MenuItemDTO] = []
        for row in rows:
            if category and category.lower() not in row.get("category", "").lower():
                continue
            result.append(
                MenuItemDTO(
                    product_id=f"backend_{row['id']}",
                    backend_item_id=int(row["id"]),
                    name_ar=row["name"],
                    name_en=row["name"],
                    category_id=row["category"],
                    category_ar=row["category"],
                    base_price=Decimal(str(row["base_price"])),
                    is_available=True,
                )
            )
        return result

    async def resolve_delivery(
        self,
        phone_number: str,
        zone_name: str,
        coordinates: dict[str, float] | None = None,
    ) -> DeliveryResolution:
        # Matches real Backend rule in app/main.py line 227:
        # delivery_fee = "20.00" if payload.zone_name != "free_zone" else "0.00"
        zone_key = zone_name.strip() or "default"
        fee = Decimal("0.00") if zone_key == "free_zone" else Decimal("20.00")
        cart = self._carts.get(phone_number, CartSnapshot())
        updated = self._recalculate(cart.model_copy(update={"delivery_fee": fee, "zone_name": zone_key}))
        self._carts[phone_number] = updated
        return DeliveryResolution(
            deliverable=True,
            zone_name=zone_key,
            delivery_fee=fee,
            estimated_minutes=40,
            message=f"رسوم التوصيل لمنطقة {zone_key}: {fee} ج.م.",
        )

    async def confirm_and_lock_order(self, payload: BackendConfirmOrderPayload) -> dict[str, Any]:
        return await self._request("POST", "/api/orders", json_body=payload.model_dump())

    async def notify_order_modification(self, order_id: int, message: str) -> dict[str, Any]:
        return await self._request("POST", f"/api/orders/{order_id}/modification", json_body={"message": message})

    async def send_cashier_directive(self, order_id: int, directive: str) -> dict[str, Any]:
        return await self._request("POST", f"/api/orders/{order_id}/instruct", json_body={"directive": directive})
