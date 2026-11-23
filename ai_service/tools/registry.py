"""Function Calling Tool Declarations and Safe Validated Tool Executor."""

from __future__ import annotations

import hashlib
from typing import Any

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.schemas.models import (
    BackendCartItemPayload,
    BackendConfirmOrderPayload,
    SessionState,
    ToolCallRecord,
)

# Gemini / OpenAI compatible Function Calling declarations
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "get_active_menu",
        "description": "جلب الأصناف المتوفرة حالياً من منيو المطعم مع الأسعار والأحجام والإضافات.",
        "parameters": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "اختياري: اسم القسم مثل بيتزا أو كريب أو برجر"}},
        },
    },
    {
        "name": "check_item_availability",
        "description": "التحقق اللحظي من توفر صنف معين في المخزون قبل تأكيده للعميل.",
        "parameters": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
    },
    {
        "name": "modify_cart",
        "description": "إضافة أو تعديل صنف في سلة العميل وإعادة حساب الإجمالي برمجياً في الـ Backend.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
                "size_code": {"type": "string"},
                "modifier_ids": {"type": "array", "items": {"type": "string"}},
                "selected_options": {"type": "object"},
                "notes": {"type": "string"},
            },
            "required": ["product_id", "quantity"],
        },
    },
    {
        "name": "remove_cart_item",
        "description": "حذف صنف محدد من سلة العميل.",
        "parameters": {
            "type": "object",
            "properties": {"cart_item_id": {"type": "string"}},
            "required": ["cart_item_id"],
        },
    },
    {
        "name": "get_cart_summary",
        "description": "عرض ملخص السلة الحالي بالأسعار الرسمية ومصاريف التوصيل قبل التأكيد.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "resolve_delivery",
        "description": "تحديد منطقة التوصيل وإضافة رسوم التوصيل الرسمية للحساب.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone_name": {"type": "string"},
                "coordinates": {"type": "object"},
            },
            "required": ["zone_name"],
        },
    },
    {
        "name": "confirm_and_lock_order",
        "description": "تأكيد وتثبيت الطلب وإرساله لشاشة الكاشير — يُمنع استدعاؤها إلا بعد تأكيد العميل الصريح وتوفر الاسم والعنوان ورقم بديل.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string"},
                "backup_phone": {"type": "string"},
                "delivery_address": {"type": "string"},
                "confirmed_by_customer": {"type": "boolean"},
            },
            "required": ["customer_name", "backup_phone", "delivery_address", "confirmed_by_customer"],
        },
    },
    {
        "name": "request_human_assistance",
        "description": "تحويل المحادثة إلى الكاشير البشري عند الشكاوى أو عدم التأكد.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "urgency": {"type": "string", "enum": ["normal", "high"]},
            },
            "required": ["reason"],
        },
    },
]


class ToolExecutor:
    """Executes validated Function Calls against the injected Backend adapter."""

    def __init__(self, backend: Any):
        self.backend = backend

    async def execute(self, session: SessionState, tool_name: str, args: dict[str, Any]) -> tuple[ToolCallRecord, Any]:
        try:
            if tool_name == "get_active_menu":
                items = await self.backend.get_active_menu(category=args.get("category"))
                return ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"{len(items)} items"), items

            if tool_name == "check_item_availability":
                available, item = await self.backend.check_item_availability(args["product_id"])
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"available={available}"),
                    {"available": available, "item": item},
                )

            if tool_name == "modify_cart":
                cart = await self.backend.modify_cart(
                    phone_number=session.phone_number,
                    product_id=args["product_id"],
                    quantity=int(args.get("quantity", 1)),
                    size_code=args.get("size_code"),
                    modifier_ids=args.get("modifier_ids") or [],
                    selected_options=args.get("selected_options") or {},
                    notes=args.get("notes", ""),
                )
                session.cart = cart
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"subtotal={cart.subtotal}"),
                    cart,
                )

            if tool_name == "remove_cart_item":
                cart = await self.backend.remove_cart_item(session.phone_number, args["cart_item_id"])
                session.cart = cart
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"remaining={len(cart.items)}"),
                    cart,
                )

            if tool_name == "get_cart_summary":
                cart = await self.backend.get_cart_summary(session.phone_number)
                session.cart = cart
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"total={cart.total_amount}"),
                    cart,
                )

            if tool_name == "resolve_delivery":
                res = await self.backend.resolve_delivery(
                    phone_number=session.phone_number,
                    zone_name=args["zone_name"],
                    coordinates=args.get("coordinates"),
                )
                session.zone_name = res.zone_name
                session.cart = await self.backend.get_cart_summary(session.phone_number)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"fee={res.delivery_fee}"),
                    res,
                )

            if tool_name == "confirm_and_lock_order":
                if not args.get("confirmed_by_customer"):
                    raise BackendIntegrationError("CONFIRMATION_REQUIRED", "لا يمكن تأكيد الطلب بدون موافقة العميل الصريحة.", 400)
                if not session.cart.items:
                    raise BackendIntegrationError("EMPTY_CART", "السلة فارغة ولا يمكن تثبيت الطلب.", 400)

                cart_sig = "|".join(f"{i.product_id}:{i.quantity}:{i.size_code}" for i in session.cart.items)
                idem_key = hashlib.sha256(f"{session.phone_number}:{cart_sig}".encode("utf-8")).hexdigest()[:24]

                payload = BackendConfirmOrderPayload(
                    customer_name=args["customer_name"],
                    customer_phone=session.phone_number,
                    backup_phone=args["backup_phone"],
                    delivery_address=args["delivery_address"],
                    zone_name=session.zone_name,
                    items=[
                        BackendCartItemPayload(
                            item_id=i.backend_item_id or 1,
                            quantity=i.quantity,
                            modifiers=[{"name": m["name"], "price": str(m["price"])} for m in i.modifiers],
                            notes=i.notes,
                        )
                        for i in session.cart.items
                    ],
                    idempotency_key=idem_key,
                    confirmed_by_customer=True,
                )
                order_res = await self.backend.confirm_and_lock_order(payload)
                session.active_order_id = order_res.get("id")
                session.awaiting_final_confirmation = False
                return (
                    ToolCallRecord(tool_name=tool_name, arguments={"idempotency_key": idem_key}, success=True, result_summary=f"order_id={order_res.get('id')}"),
                    order_res,
                )

            if tool_name == "request_human_assistance":
                handoff = await self.backend.request_human_assistance(
                    phone_number=session.phone_number,
                    reason=args["reason"],
                    urgency=args.get("urgency", "normal"),
                )
                session.needs_human_takeover = True
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary="escalated_to_cashier"),
                    handoff,
                )

            raise BackendIntegrationError("UNKNOWN_TOOL", f"Unknown tool: {tool_name}", 400)

        except BackendIntegrationError as exc:
            return (
                ToolCallRecord(tool_name=tool_name, arguments=args, success=False, error=f"{exc.code}: {exc.message}"),
                exc,
            )
