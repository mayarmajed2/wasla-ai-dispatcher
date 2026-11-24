"""Real HTTP Adapter for the FastAPI Restaurant Backend.

Uses these backend endpoints:
- GET  /api/menu
- GET  /api/delivery/quote
- POST /api/orders
- POST /api/customer/order-status (service-key authenticated and phone scoped)
- POST /api/orders/{order_id}/modification
- POST /api/orders/{order_id}/instruct
"""

from __future__ import annotations

from decimal import Decimal
import os
from typing import Any
from urllib.parse import urlencode

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
        service_api_key: str | None = None,
    ):
        super().__init__(menu_index)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external_client = client
        self.service_api_key = (
            service_api_key if service_api_key is not None else os.getenv("BACKEND_SERVICE_API_KEY", "")
        )

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if self._external_client is not None:
            resp = await self._external_client.request(
                method, f"{self.base_url}{path}", json=json_body, headers=headers
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(method, f"{self.base_url}{path}", json=json_body, headers=headers)

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
        zone_key = zone_name.strip() or "default"
        quote = await self._request("GET", f"/api/delivery/quote?{urlencode({'zone_name': zone_key})}")
        if (
            not isinstance(quote, dict)
            or type(quote.get("deliverable")) is not bool
            or not isinstance(quote.get("zone_name"), str)
        ):
            raise BackendIntegrationError("INVALID_BACKEND_RESPONSE", "Delivery quote response is invalid.", 502)
        try:
            fee = Decimal(str(quote["delivery_fee"]))
        except (KeyError, ArithmeticError, ValueError) as exc:
            raise BackendIntegrationError("INVALID_BACKEND_RESPONSE", "Delivery fee response is invalid.", 502) from exc
        if not fee.is_finite() or fee < 0:
            raise BackendIntegrationError("INVALID_BACKEND_RESPONSE", "Delivery fee response is invalid.", 502)
        estimated_minutes = quote.get("estimated_minutes")
        if estimated_minutes is not None and (type(estimated_minutes) is not int or estimated_minutes < 1):
            raise BackendIntegrationError("INVALID_BACKEND_RESPONSE", "Delivery estimate response is invalid.", 502)
        cart = self._carts.get(phone_number, CartSnapshot())
        quoted_zone = quote["zone_name"]
        updated = self._recalculate(cart.model_copy(update={"delivery_fee": fee, "zone_name": quoted_zone}))
        self._carts[phone_number] = updated
        return DeliveryResolution(
            deliverable=bool(quote["deliverable"]),
            zone_name=quoted_zone,
            delivery_fee=fee,
            estimated_minutes=estimated_minutes,
            message=f"رسوم التوصيل لمنطقة {quoted_zone}: {fee} ج.م.",
        )

    async def confirm_and_lock_order(self, payload: BackendConfirmOrderPayload) -> dict[str, Any]:
        return await self._request("POST", "/api/orders", json_body=payload.model_dump())

    async def get_customer_order_status(
        self, phone_number: str, order_id: int | None = None
    ) -> dict[str, Any] | None:
        if not self.service_api_key:
            raise BackendIntegrationError(
                "ORDER_TRACKING_UNAVAILABLE",
                "Backend customer-order tracking requires BACKEND_SERVICE_API_KEY.",
                503,
            )
        result = await self._request(
            "POST",
            "/api/customer/order-status",
            json_body={"customer_phone": phone_number, "order_id": order_id},
            headers={"X-Backend-Service-Key": self.service_api_key},
        )
        if not isinstance(result, dict):
            raise BackendIntegrationError("INVALID_BACKEND_RESPONSE", "Order status response is invalid.", 502)
        result_order_id = result.get("id")
        result_status = result.get("status")
        if result_order_id is None and result_status is None:
            return None
        if (
            type(result_order_id) is not int
            or result_order_id < 1
            or (result_status is not None and not isinstance(result_status, str))
        ):
            raise BackendIntegrationError("INVALID_BACKEND_RESPONSE", "Order status response is invalid.", 502)
        return {"id": result_order_id, "status": result_status}

    async def notify_order_modification(self, order_id: int, message: str) -> dict[str, Any]:
        return await self._request("POST", f"/api/orders/{order_id}/modification", json_body={"message": message})

    async def send_cashier_directive(self, order_id: int, directive: str) -> dict[str, Any]:
        return await self._request("POST", f"/api/orders/{order_id}/instruct", json_body={"directive": directive})
