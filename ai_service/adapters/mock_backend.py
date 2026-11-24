"""Isolated In-Memory Mock Backend Adapter for Development & Automated Testing.

IMPORTANT:
- This is strictly a testing/development adapter, NOT the production Backend.
- Mirrors the exact deterministic Decimal arithmetic, session persistence/TTL,
  and validation rules of the FastAPI/PostgreSQL Backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import uuid
from typing import Any

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.schemas.models import (
    BackendConfirmOrderPayload,
    CartItemDraft,
    CartSnapshot,
    DeliveryResolution,
    MenuItemDTO,
    OrderStatus,
    SessionState,
)
from ai_service.understanding.menu_index import MenuIndex

MONEY = Decimal("0.01")


def _money(val: Any) -> Decimal:
    return Decimal(str(val)).quantize(MONEY, rounding=ROUND_HALF_UP)


class MockBackendAdapter:
    """In-memory implementation of all Backend service contracts for isolated QA."""

    IS_MOCK = True

    def __init__(self, menu_index: MenuIndex):
        self.menu_index = menu_index
        self._unavailable_ids: set[str] = set()
        self._carts: dict[str, CartSnapshot] = {}
        self._session_store: dict[str, dict[str, Any]] = {}
        self._orders: dict[int, dict[str, Any]] = {}
        self._idempotency_map: dict[str, int] = {}
        self._handoffs: list[dict[str, Any]] = []
        self._next_order_id = 1

        # Mirrors existing Backend default delivery fee logic (`free_zone` = 0, default = 20 EGP)
        self._zones: dict[str, Decimal] = {
            "default": Decimal("20.00"),
            "free_zone": Decimal("0.00"),
            "المعادي": Decimal("20.00"),
            "مدينة نصر": Decimal("25.00"),
            "التجمع": Decimal("30.00"),
            "الدقي": Decimal("20.00"),
            "المهندسين": Decimal("20.00"),
        }

    def sync_menu_sync(self) -> None:
        """No-op hook overridden by DB adapter to refresh MenuIndex & unavailability from SQL."""
        return None

    def load_session_state(self, phone_number: str, ttl_seconds: int = 3600) -> SessionState | None:
        raw = self._session_store.get(phone_number)
        now = datetime.now(timezone.utc)
        if raw is not None:
            state = SessionState.model_validate(raw)
            updated_at = state.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            if ttl_seconds > 0 and (now - updated_at) > timedelta(seconds=ttl_seconds) and not state.needs_human_takeover:
                self._carts[phone_number] = CartSnapshot()
                expired = SessionState(
                    phone_number=phone_number,
                    customer_name=state.customer_name,
                    backup_phone=state.backup_phone,
                    delivery_address=state.delivery_address,
                    session_epoch=state.session_epoch + 1,
                    updated_at=now,
                )
                self.save_session_state(expired)
                return expired
            if phone_number in self._carts and len(self._carts[phone_number].items) != len(state.cart.items):
                state = state.model_copy(update={"cart": self._carts[phone_number]})
            else:
                self._carts[phone_number] = state.cart
            return state
        if phone_number in self._carts:
            return SessionState(phone_number=phone_number, cart=self._carts[phone_number])
        return None

    def save_session_state(self, session: SessionState) -> None:
        trimmed_history = session.history[-30:] if len(session.history) > 30 else session.history
        to_save = session.model_copy(update={"history": trimmed_history, "updated_at": datetime.now(timezone.utc)})
        self._session_store[session.phone_number] = to_save.model_dump(mode="json")
        self._carts[session.phone_number] = to_save.cart

    def delete_session_state(self, phone_number: str) -> None:
        self._session_store.pop(phone_number, None)
        self._carts.pop(phone_number, None)

    def set_item_unavailable(self, product_id: str, unavailable: bool = True) -> None:
        if unavailable:
            self._unavailable_ids.add(product_id)
        else:
            self._unavailable_ids.discard(product_id)
        prod = self.menu_index.get_product(product_id)
        if prod is not None:
            self.menu_index.products[product_id] = prod.model_copy(update={"is_available": not unavailable})

    async def get_active_menu(self, category: str | None = None) -> list[MenuItemDTO]:
        self.sync_menu_sync()
        items = self.menu_index.list_by_category(category)
        active = []
        for item in items:
            if item.product_id not in self._unavailable_ids and item.is_available is not False:
                active.append(item.model_copy(update={"is_available": True}))
        return active

    async def check_item_availability(self, product_id: str) -> tuple[bool, MenuItemDTO | None]:
        self.sync_menu_sync()
        item = self.menu_index.get_product(product_id)
        if item is None:
            return False, None
        available = product_id not in self._unavailable_ids and item.is_available is not False
        return available, item.model_copy(update={"is_available": available})

    def _recalculate(self, cart: CartSnapshot) -> CartSnapshot:
        subtotal = Decimal("0.00")
        for line in cart.items:
            subtotal += line.line_total
        subtotal = _money(subtotal)
        delivery_fee = _money(cart.delivery_fee)
        return cart.model_copy(
            update={
                "subtotal": subtotal,
                "delivery_fee": delivery_fee,
                "total_amount": _money(subtotal + delivery_fee),
            }
        )

    async def revalidate_cart(self, phone_number: str) -> tuple[CartSnapshot, list[str], bool]:
        """Revalidate prices and availability for every item currently in the cart."""
        self.sync_menu_sync()
        cart = self._carts.get(phone_number, CartSnapshot())
        if not cart.items:
            return cart, [], False

        unavailable_names: list[str] = []
        kept_items: list[CartItemDraft] = []
        price_changed = False

        for line in cart.items:
            available, prod = await self.check_item_availability(line.product_id)
            if not available or prod is None:
                unavailable_names.append(line.name_ar)
                continue
            unit_price, err = self.menu_index.resolve_unit_price(prod, line.size_code)
            if err or unit_price is None:
                unavailable_names.append(line.name_ar)
                continue
            new_unit = _money(unit_price)
            mod_total = sum((_money(m.get("price", "0")) for m in line.modifiers), Decimal("0.00"))
            new_line_total = _money((new_unit + mod_total) * line.quantity)
            if new_unit != _money(line.unit_price) or new_line_total != _money(line.line_total):
                price_changed = True
            kept_items.append(line.model_copy(update={"unit_price": new_unit, "line_total": new_line_total}))

        updated = self._recalculate(cart.model_copy(update={"items": kept_items}))
        self._carts[phone_number] = updated
        return updated, unavailable_names, price_changed

    async def modify_cart(
        self,
        phone_number: str,
        product_id: str,
        quantity: int,
        size_code: str | None = None,
        modifier_ids: list[str] | None = None,
        selected_options: dict[str, str] | None = None,
        notes: str = "",
    ) -> CartSnapshot:
        if quantity < 1:
            raise BackendIntegrationError("INVALID_QUANTITY", "Quantity must be at least 1.", 400)

        available, product = await self.check_item_availability(product_id)
        if product is None:
            raise BackendIntegrationError("PRODUCT_NOT_FOUND", f"Product {product_id} not found.", 404)
        if not available:
            raise BackendIntegrationError("ITEM_OUT_OF_STOCK", f"الصنف {product.name_ar} غير متوفر حالياً.", 409)

        unit_price, err = self.menu_index.resolve_unit_price(product, size_code)
        if err or unit_price is None:
            raise BackendIntegrationError("MISSING_OR_INVALID_SIZE", err or "Invalid size.", 400)

        resolved_opts = selected_options or {}
        for opt in product.options:
            if opt.required and opt.option_group not in resolved_opts:
                choices_str = " / ".join(opt.choices)
                raise BackendIntegrationError(
                    "MISSING_REQUIRED_OPTION",
                    f"محتاج تحدد {opt.option_group} لـ {product.name_ar} ({choices_str}).",
                    400,
                )

        resolved_mods: list[dict[str, Any]] = []
        mod_total = Decimal("0.00")
        for mid in modifier_ids or []:
            mod = self.menu_index.get_modifier(mid)
            if mod is None:
                raise BackendIntegrationError("INVALID_MODIFIER", f"Modifier {mid} does not exist.", 400)
            if product.category_id not in mod.applies_to_categories:
                raise BackendIntegrationError(
                    "MODIFIER_NOT_ALLOWED",
                    f"الإضافة '{mod.name_ar}' غير متاحة لصنف {product.name_ar}.",
                    400,
                )
            resolved_mods.append({"modifier_id": mod.modifier_id, "name": mod.name_ar, "price": str(_money(mod.price))})
            mod_total += _money(mod.price)

        line_total = _money((_money(unit_price) + mod_total) * quantity)
        cart = self._carts.get(phone_number, CartSnapshot())
        new_item = CartItemDraft(
            cart_item_id=f"ci_{uuid.uuid4().hex[:8]}",
            product_id=product.product_id,
            backend_item_id=product.backend_item_id,
            name_ar=product.name_ar if not size_code else f"{product.name_ar} ({size_code})",
            quantity=quantity,
            size_code=size_code,
            unit_price=_money(unit_price),
            modifiers=resolved_mods,
            selected_options=resolved_opts,
            notes=notes.strip(),
            line_total=line_total,
        )
        updated_cart = self._recalculate(cart.model_copy(update={"items": [*cart.items, new_item]}))
        self._carts[phone_number] = updated_cart
        return updated_cart

    async def remove_cart_item(self, phone_number: str, cart_item_id: str) -> CartSnapshot:
        cart = self._carts.get(phone_number, CartSnapshot())
        remaining = [i for i in cart.items if i.cart_item_id != cart_item_id and i.product_id != cart_item_id]
        if len(remaining) == len(cart.items):
            raise BackendIntegrationError("CART_ITEM_NOT_FOUND", "الصنف غير موجود في السلة.", 404)
        updated = self._recalculate(cart.model_copy(update={"items": remaining}))
        self._carts[phone_number] = updated
        return updated

    async def get_cart_summary(self, phone_number: str) -> CartSnapshot:
        self.sync_menu_sync()
        cart, _, _ = await self.revalidate_cart(phone_number)
        return cart

    async def clear_cart(self, phone_number: str) -> CartSnapshot:
        empty = CartSnapshot()
        self._carts[phone_number] = empty
        return empty

    async def resolve_delivery(
        self,
        phone_number: str,
        zone_name: str,
        coordinates: dict[str, float] | None = None,
    ) -> DeliveryResolution:
        zone_key = zone_name.strip() or "default"
        fee = self._zones.get(zone_key, self._zones["default"])
        cart = self._carts.get(phone_number, CartSnapshot())
        updated = self._recalculate(cart.model_copy(update={"delivery_fee": fee, "zone_name": zone_key}))
        self._carts[phone_number] = updated
        return DeliveryResolution(
            deliverable=True,
            zone_name=zone_key,
            delivery_fee=fee,
            estimated_minutes=40,
            message=f"التوصيل متاح لمنطقة {zone_key} بتكلفة {fee} جنيه.",
        )

    async def confirm_and_lock_order(self, payload: BackendConfirmOrderPayload) -> dict[str, Any]:
        if not payload.confirmed_by_customer:
            raise BackendIntegrationError(
                "CONFIRMATION_REQUIRED",
                "The customer must confirm the complete order before dispatch.",
                400,
            )

        if payload.idempotency_key in self._idempotency_map:
            existing_id = self._idempotency_map[payload.idempotency_key]
            existing = self._orders[existing_id]
            return {"id": existing_id, "status": existing["status"], "duplicate": True}

        order_id = self._next_order_id
        self._next_order_id += 1
        cart = self._carts.get(payload.customer_phone, CartSnapshot())
        order_record = {
            "id": order_id,
            "status": OrderStatus.SUBMITTED.value,
            "customer_name": payload.customer_name,
            "customer_phone": payload.customer_phone,
            "backup_phone": payload.backup_phone,
            "delivery_address": payload.delivery_address,
            "zone_name": payload.zone_name,
            "subtotal": str(cart.subtotal),
            "delivery_fee": str(cart.delivery_fee),
            "total_amount": str(cart.total_amount),
            "items": [i.model_dump() for i in payload.items],
        }
        self._orders[order_id] = order_record
        self._idempotency_map[payload.idempotency_key] = order_id
        return {"id": order_id, "status": OrderStatus.SUBMITTED.value, "cart": cart.model_dump()}

    async def notify_order_modification(self, order_id: int, message: str) -> dict[str, Any]:
        if order_id not in self._orders:
            raise BackendIntegrationError("ORDER_NOT_FOUND", f"Order {order_id} not found.", 404)
        return {"order_id": order_id, "status": "modification_received", "message": message}

    async def request_human_assistance(self, phone_number: str, reason: str, urgency: str = "normal") -> dict[str, Any]:
        record = {"phone_number": phone_number, "reason": reason, "urgency": urgency, "needs_human_takeover": True}
        self._handoffs.append(record)
        return record
