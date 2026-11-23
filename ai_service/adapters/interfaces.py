"""Abstract Integration Contracts between Wasla AI Layer and the Restaurant Backend."""

from __future__ import annotations

from typing import Any, Protocol

from ai_service.schemas.models import (
    BackendConfirmOrderPayload,
    CartSnapshot,
    DeliveryResolution,
    MenuItemDTO,
)


class BackendIntegrationError(RuntimeError):
    """Structured error raised by any Backend adapter (Mock or Real)."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class MenuProvider(Protocol):
    async def get_active_menu(self, category: str | None = None) -> list[MenuItemDTO]: ...
    async def check_item_availability(self, product_id: str) -> tuple[bool, MenuItemDTO | None]: ...


class CartService(Protocol):
    async def modify_cart(
        self,
        phone_number: str,
        product_id: str,
        quantity: int,
        size_code: str | None = None,
        modifier_ids: list[str] | None = None,
        selected_options: dict[str, str] | None = None,
        notes: str = "",
    ) -> CartSnapshot: ...

    async def remove_cart_item(self, phone_number: str, cart_item_id: str) -> CartSnapshot: ...
    async def get_cart_summary(self, phone_number: str) -> CartSnapshot: ...
    async def clear_cart(self, phone_number: str) -> CartSnapshot: ...


class DeliveryService(Protocol):
    async def resolve_delivery(
        self,
        phone_number: str,
        zone_name: str,
        coordinates: dict[str, float] | None = None,
    ) -> DeliveryResolution: ...


class OrderService(Protocol):
    async def confirm_and_lock_order(self, payload: BackendConfirmOrderPayload) -> dict[str, Any]: ...
    async def notify_order_modification(self, order_id: int, message: str) -> dict[str, Any]: ...


class HumanHandoffService(Protocol):
    async def request_human_assistance(self, phone_number: str, reason: str, urgency: str = "normal") -> dict[str, Any]: ...
