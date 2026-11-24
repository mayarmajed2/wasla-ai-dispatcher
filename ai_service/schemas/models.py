"""Strict Pydantic v2 schemas for Wasla AI Dispatcher and Backend Integration.

Aligned with the actual existing FastAPI Backend (`existing_system/RESTURANT AI/app/main.py` & `models.py`):
- 7-state OrderStatus lifecycle
- ConfirmOrder payload requiring customer_name, customer_phone, backup_phone, delivery_address,
  zone_name, items, idempotency_key, and confirmed_by_customer=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


@dataclass(frozen=True)
class TurnExecutionContext:
    """Immutable per-request execution context (eliminates shared mutable dispatcher fields)."""

    data_source: str = "api"
    is_test: bool = False
    request_id: str = ""


class OrderStatus(str, Enum):
    # Legacy / Backward-compatible 7-state values (preserved for existing clients/tests)
    DRAFTING = "DRAFTING"
    SUBMITTED = "SUBMITTED"
    IN_KITCHEN = "IN_KITCHEN"
    DISPATCHED = "DISPATCHED"
    UNREACHABLE_ALERT = "UNREACHABLE_ALERT"
    FULFILLED = "FULFILLED"
    CANCELLED = "CANCELLED"
    # Explicit commercial lifecycle states (Phase 3)
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    ACCEPTED = "ACCEPTED"
    PREPARING = "PREPARING"
    READY = "READY"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


class CustomerIntent(str, Enum):
    GREETING = "greeting"
    MENU_INQUIRY = "menu_inquiry"
    PRODUCT_INQUIRY = "product_inquiry"
    PRICE_INQUIRY = "price_inquiry"
    AVAILABILITY_INQUIRY = "availability_inquiry"
    REPORT_UNAVAILABLE = "report_unavailable"
    REJECT_PRODUCT = "reject_product"
    ORDER_STATUS = "order_status"
    GET_CART = "get_cart"
    ADD_TO_CART = "add_to_cart"
    MODIFY_ORDER = "modify_order"
    REMOVE_ITEM = "remove_item"
    DELIVERY_INQUIRY = "delivery_inquiry"
    PROVIDE_DETAILS = "provide_details"
    CONFIRM_ORDER = "confirm_order"
    CANCEL_ORDER = "cancel_order"
    COMPLAINT = "complaint"
    REQUEST_HUMAN = "request_human"
    UNCLEAR_REQUEST = "unclear_request"


class MenuSizeDTO(BaseModel):
    code: str
    name_ar: str
    name_en: str
    price: Decimal


class MenuOptionGroupDTO(BaseModel):
    option_group: str
    required: bool = True
    choices: list[str] = Field(default_factory=list)


class MenuModifierDTO(BaseModel):
    modifier_id: str
    name_ar: str
    name_en: str
    price: Decimal
    applies_to_categories: list[str] = Field(default_factory=list)


class MenuItemDTO(BaseModel):
    product_id: str
    backend_item_id: int | None = None
    name_ar: str
    name_en: str
    category_id: str
    category_ar: str = ""
    description_ar: str | None = None
    base_price: Decimal | None = None
    sizes: list[MenuSizeDTO] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    options: list[MenuOptionGroupDTO] = Field(default_factory=list)
    is_available: bool | None = None


class ExtractedOrderItem(BaseModel):
    product_id: str | None = None
    raw_mention: str = ""
    quantity: int = Field(default=1, ge=1, le=50)
    size_code: str | None = None
    modifier_ids: list[str] = Field(default_factory=list)
    selected_options: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class CartItemDraft(BaseModel):
    cart_item_id: str
    product_id: str
    backend_item_id: int | None = None
    name_ar: str
    quantity: int = Field(ge=1)
    size_code: str | None = None
    unit_price: Decimal
    modifiers: list[dict[str, Any]] = Field(default_factory=list)
    selected_options: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    line_total: Decimal


class CartSnapshot(BaseModel):
    items: list[CartItemDraft] = Field(default_factory=list)
    subtotal: Decimal = Decimal("0.00")
    delivery_fee: Decimal = Decimal("0.00")
    total_amount: Decimal = Decimal("0.00")
    zone_name: str = "default"


class DeliveryResolution(BaseModel):
    deliverable: bool
    zone_name: str
    delivery_fee: Decimal
    estimated_minutes: int | None = None
    message: str = ""


class BackendCartItemPayload(BaseModel):
    """Exact CartItem schema expected by existing FastAPI Backend POST /api/orders."""

    item_id: int
    quantity: int = Field(ge=1)
    size_code: str | None = None
    modifiers: list[dict[str, str]] = Field(default_factory=list)
    selected_options: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class BackendConfirmOrderPayload(BaseModel):
    """Exact ConfirmOrder schema expected by existing FastAPI Backend POST /api/orders."""

    customer_name: str = Field(min_length=2)
    customer_phone: str = Field(min_length=8)
    backup_phone: str = Field(min_length=8)
    delivery_address: str = Field(min_length=5)
    zone_name: str = "default"
    items: list[BackendCartItemPayload] = Field(min_length=1)
    idempotency_key: str = Field(min_length=4)
    confirmed_by_customer: bool
    data_source: str = "api"
    is_test: bool = False


class AudioInput(BaseModel):
    audio_bytes: bytes
    mime_type: str = "audio/ogg"
    filename: str = "voice_note.ogg"
    duration_seconds: float | None = None


class AudioTranscriptionResult(BaseModel):
    transcript: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provider: str
    latency_ms: float = Field(ge=0.0)
    used_fallback: bool = False


class ConversationMessage(BaseModel):
    role: str = Field(pattern="^(customer|assistant|cashier_directive|system)$")
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_type: str = "text"


class SessionState(BaseModel):
    phone_number: str
    customer_name: str | None = None
    backup_phone: str | None = None
    delivery_address: str | None = None
    zone_name: str = "default"
    cart: CartSnapshot = Field(default_factory=CartSnapshot)
    active_order_id: int | None = None
    needs_human_takeover: bool = False
    human_takeover_by: str | None = None
    pending_clarification: dict[str, Any] | None = None
    pending_upsell: dict[str, Any] | None = None
    upsell_declined: bool = False
    rejected_upsell_ids: list[str] = Field(default_factory=list)
    budget_limit: Decimal | None = None
    awaiting_final_confirmation: bool = False
    order_confirmed: bool = False
    confirmed_cart_sig: str | None = None
    session_epoch: int = 1
    last_intent: str | None = None
    latest_directive: str | None = None
    history: list[ConversationMessage] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    result_summary: str = ""
    error: str | None = None


class DispatcherResponse(BaseModel):
    intent: CustomerIntent
    confidence: float = Field(ge=0.0, le=1.0)
    transcript: str | None = None
    extracted_items: list[ExtractedOrderItem] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_reason: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    cart: CartSnapshot
    order_id: int | None = None
    needs_human_takeover: bool = False
    suppress_ai_reply: bool = False
    upsell_offer: dict[str, Any] | None = None
    upsell_event: dict[str, Any] | None = None
    complaint_assessment: dict[str, Any] | None = None
    reply_ar: str
    model_used: str = "unknown"

    @field_validator("reply_ar")
    @classmethod
    def reply_not_empty(cls, v: str, info) -> str:
        if info.data.get("suppress_ai_reply"):
            return (v or "").strip()
        if not v or not v.strip():
            raise ValueError("reply_ar must not be empty")
        return v.strip()


class CashierDirectiveRequest(BaseModel):
    order_id: int
    directive_type: str = Field(default="free_text", pattern="^(unreachable_client|kitchen_delay|out_of_stock|free_text)$")
    directive_text: str = Field(min_length=2)
    extra_delay_minutes: int | None = Field(default=None, ge=1, le=180)
    authorized_by_cashier: bool = True


class CashierDirectiveResponse(BaseModel):
    order_id: int
    directive_type: str
    validated: bool
    warnings: list[str] = Field(default_factory=list)
    suggested_alternative_product_id: str | None = None
    customer_message_ar: str
    requires_cashier_approval: bool = False
