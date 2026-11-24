"""Function Calling Tool Declarations and Safe Validated Tool Executor."""

from __future__ import annotations

import hashlib
import json
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
        "description": "التحقق اللحظي من توفر صنف معين في المخزون والسعر الرسمي الحالي قبل تأكيده للعميل.",
        "parameters": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
    },
    {
        "name": "get_customer_order_status",
        "description": (
            "استرجاع الحالة المسجلة فعلياً لآخر أوردر مرتبط برقم العميل الحالي؛ "
            "لا تخمّن الحالة أو وقت الوصول."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "integer"}},
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

ALLOWED_TOOL_NAMES: set[str] = {t["name"] for t in TOOL_DECLARATIONS}


def compute_tool_call_signature(tool_name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class ToolExecutor:
    """Executes validated Function Calls against the injected Backend adapter with deduplication."""

    def __init__(self, backend: Any):
        self.backend = backend

    async def execute(
        self,
        session: SessionState,
        tool_name: str,
        args: dict[str, Any],
        *,
        executed_signatures: set[str] | None = None,
    ) -> tuple[ToolCallRecord, Any]:
        try:
            if not isinstance(args, dict):
                raise BackendIntegrationError("INVALID_TOOL_ARGS", "Tool arguments must be an object.", 400)
            if tool_name not in ALLOWED_TOOL_NAMES:
                raise BackendIntegrationError("UNKNOWN_TOOL", f"Unknown tool: {tool_name}", 400)

            sig = compute_tool_call_signature(tool_name, args)
            if executed_signatures is not None and sig in executed_signatures:
                return (
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments=args,
                        success=True,
                        result_summary="deduplicated_skip",
                    ),
                    {"deduplicated": True},
                )

            if tool_name == "get_active_menu":
                category = args.get("category")
                if category is not None and not isinstance(category, str):
                    raise BackendIntegrationError("INVALID_TOOL_ARGS", "category must be a string.", 400)
                items = await self.backend.get_active_menu(category=category)
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"{len(items)} items"), items

            if tool_name == "check_item_availability":
                pid = str(args.get("product_id") or "").strip()
                if not pid:
                    raise BackendIntegrationError("INVALID_TOOL_ARGS", "product_id is required.", 400)
                available, item = await self.backend.check_item_availability(pid)
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments={"product_id": pid}, success=True, result_summary=f"available={available}"),
                    {"available": available, "item": item},
                )

            if tool_name == "get_customer_order_status":
                raw_order_id = args.get("order_id")
                if raw_order_id is None:
                    order_id = None
                else:
                    try:
                        order_id = int(raw_order_id)
                    except (TypeError, ValueError) as exc:
                        raise BackendIntegrationError("INVALID_TOOL_ARGS", "order_id must be an integer.", 400) from exc
                    if order_id < 1:
                        raise BackendIntegrationError("INVALID_TOOL_ARGS", "order_id must be positive.", 400)
                lookup = getattr(self.backend, "get_customer_order_status", None)
                if lookup is None:
                    raise BackendIntegrationError(
                        "ORDER_TRACKING_UNAVAILABLE", "Order tracking is not available in this backend.", 503
                    )
                result = await lookup(session.phone_number, order_id)
                if result is not None and not isinstance(result, dict):
                    raise BackendIntegrationError(
                        "INVALID_BACKEND_RESPONSE", "Order status lookup returned an invalid response.", 502
                    )
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                summary = "order_not_found" if result is None else f"order_id={result.get('id')}"
                return (
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments={"order_id": order_id} if order_id is not None else {},
                        success=True,
                        result_summary=summary,
                    ),
                    result,
                )

            if tool_name == "modify_cart":
                pid = str(args.get("product_id") or "").strip()
                if not pid:
                    raise BackendIntegrationError("PRODUCT_NOT_FOUND", "product_id is required.", 400)
                try:
                    qty = int(args.get("quantity", 1))
                except (TypeError, ValueError) as exc:
                    raise BackendIntegrationError("INVALID_QUANTITY", "Invalid quantity.", 400) from exc
                if qty < 1 or qty > 50:
                    raise BackendIntegrationError("INVALID_QUANTITY", "Quantity must be between 1 and 50.", 400)
                mod_ids = args.get("modifier_ids") or []
                if not isinstance(mod_ids, list):
                    raise BackendIntegrationError("INVALID_MODIFIER", "modifier_ids must be a list.", 400)
                opts = args.get("selected_options") or {}
                if not isinstance(opts, dict):
                    raise BackendIntegrationError("INVALID_TOOL_ARGS", "selected_options must be an object.", 400)

                cart = await self.backend.modify_cart(
                    phone_number=session.phone_number,
                    product_id=pid,
                    quantity=qty,
                    size_code=args.get("size_code"),
                    modifier_ids=[str(m) for m in mod_ids if m is not None],
                    selected_options={str(k): str(v) for k, v in opts.items() if v is not None},
                    notes=str(args.get("notes") or ""),
                )
                session.cart = cart
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"subtotal={cart.subtotal}"),
                    cart,
                )

            if tool_name == "remove_cart_item":
                target_id = str(args.get("cart_item_id") or "").strip()
                if not target_id:
                    raise BackendIntegrationError("INVALID_TOOL_ARGS", "cart_item_id is required.", 400)
                cart = await self.backend.remove_cart_item(session.phone_number, target_id)
                session.cart = cart
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"remaining={len(cart.items)}"),
                    cart,
                )

            if tool_name == "get_cart_summary":
                cart = await self.backend.get_cart_summary(session.phone_number)
                session.cart = cart
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"total={cart.total_amount}"),
                    cart,
                )

            if tool_name == "resolve_delivery":
                zone_name = str(args.get("zone_name") or "").strip()
                if not zone_name:
                    raise BackendIntegrationError("INVALID_TOOL_ARGS", "zone_name is required.", 400)
                res = await self.backend.resolve_delivery(
                    phone_number=session.phone_number,
                    zone_name=zone_name,
                    coordinates=args.get("coordinates"),
                )
                session.zone_name = res.zone_name
                session.cart = await self.backend.get_cart_summary(session.phone_number)
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments=args, success=True, result_summary=f"fee={res.delivery_fee}"),
                    res,
                )

            if tool_name == "confirm_and_lock_order":
                if not args.get("confirmed_by_customer"):
                    raise BackendIntegrationError("CONFIRMATION_REQUIRED", "لا يمكن تأكيد الطلب بدون موافقة العميل الصريحة.", 400)
                if not session.cart.items:
                    raise BackendIntegrationError("EMPTY_CART", "السلة فارغة ولا يمكن تثبيت الطلب.", 400)

                customer_name = str(args.get("customer_name") or "").strip()
                backup_phone = str(args.get("backup_phone") or "").strip()
                delivery_address = str(args.get("delivery_address") or "").strip()
                if len(customer_name) < 2 or len(backup_phone) < 8 or len(delivery_address) < 5:
                    raise BackendIntegrationError("MISSING_ORDER_DETAILS", "يرجى استكمال الاسم والعنوان ورقم الموبايل البديل.", 400)

                cart_sig = json.dumps(
                    [
                        {
                            "product_id": item.product_id,
                            "quantity": item.quantity,
                            "size_code": item.size_code,
                            "modifier_ids": sorted(m.get("modifier_id", "") for m in item.modifiers),
                            "selected_options": item.selected_options,
                            "notes": item.notes,
                        }
                        for item in session.cart.items
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                idem_key = hashlib.sha256(
                    f"{session.phone_number}:{session.session_epoch}:{cart_sig}".encode("utf-8")
                ).hexdigest()[:24]

                payload = BackendConfirmOrderPayload(
                    customer_name=customer_name,
                    customer_phone=session.phone_number,
                    backup_phone=backup_phone,
                    delivery_address=delivery_address,
                    zone_name=session.zone_name,
                    items=[
                        BackendCartItemPayload(
                            item_id=i.backend_item_id or 1,
                            quantity=i.quantity,
                            size_code=i.size_code,
                            modifiers=[{"modifier_id": m["modifier_id"]} for m in i.modifiers],
                            selected_options=i.selected_options,
                            notes=i.notes,
                        )
                        for i in session.cart.items
                    ],
                    idempotency_key=idem_key,
                    confirmed_by_customer=True,
                    data_source=str(args.get("data_source", "api")),
                    is_test=bool(args.get("is_test", False)),
                )
                order_res = await self.backend.confirm_and_lock_order(payload)
                session.active_order_id = order_res.get("id")
                session.awaiting_final_confirmation = False
                session.order_confirmed = True
                session.confirmed_cart_sig = idem_key
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments={"idempotency_key": idem_key}, success=True, result_summary=f"order_id={order_res.get('id')}"),
                    order_res,
                )

            if tool_name == "request_human_assistance":
                reason = str(args.get("reason") or "").strip() or "customer_request"
                urgency = str(args.get("urgency") or "normal")
                if urgency not in ("normal", "high"):
                    urgency = "normal"
                handoff = await self.backend.request_human_assistance(
                    phone_number=session.phone_number,
                    reason=reason,
                    urgency=urgency,
                )
                session.needs_human_takeover = True
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                return (
                    ToolCallRecord(tool_name=tool_name, arguments={"reason": reason, "urgency": urgency}, success=True, result_summary="escalated_to_cashier"),
                    handoff,
                )

            raise BackendIntegrationError("UNKNOWN_TOOL", f"Unknown tool: {tool_name}", 400)

        except BackendIntegrationError as exc:
            return (
                ToolCallRecord(tool_name=tool_name, arguments=args if isinstance(args, dict) else {}, success=False, error=f"{exc.code}: {exc.message}"),
                exc,
            )
