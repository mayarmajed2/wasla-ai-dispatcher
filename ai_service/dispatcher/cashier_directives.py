"""Human-in-the-Loop Cashier Directive Interpreter.

Enforces safety rules from Phase 9:
- Never auto-applies unapproved delay durations unless explicitly specified by the cashier.
- Validates any alternative product mentioned in a free-text directive against MOKA's real menu.
- Never silently assumes equal prices; flags price mismatches for cashier approval.
"""

from __future__ import annotations

import re

from ai_service.schemas.models import CashierDirectiveRequest, CashierDirectiveResponse
from ai_service.understanding.menu_index import MenuIndex, normalize_ar


class CashierDirectiveInterpreter:
    """Translates cashier presets and free-text directives into validated Egyptian Arabic messages."""

    def __init__(self, menu_index: MenuIndex):
        self.menu_index = menu_index

    def interpret(self, req: CashierDirectiveRequest) -> CashierDirectiveResponse:
        warnings: list[str] = []
        if not req.authorized_by_cashier:
            return CashierDirectiveResponse(
                order_id=req.order_id,
                directive_type=req.directive_type,
                validated=False,
                warnings=["لا يمكن إرسال توجيه للعميل بدون اعتماد الكاشير."],
                customer_message_ar="",
                requires_cashier_approval=True,
            )

        if req.directive_type == "unreachable_client":
            msg = (
                f"أهلاً بحضرتك، كابتن التوصيل وصل لعنوان طلب #{req.order_id} "
                "وبيحاول يتواصل مع حضرتك ومش عارف يوصلك. يا ريت ترد علينا فوراً أو تأكدلنا رقم بديل عشان الأوردر يوصلك سخن."
            )
            return CashierDirectiveResponse(
                order_id=req.order_id,
                directive_type=req.directive_type,
                validated=True,
                customer_message_ar=msg,
                requires_cashier_approval=False,
            )

        if req.directive_type == "kitchen_delay":
            if req.extra_delay_minutes is not None:
                msg = (
                    f"بنعتذر لحضرتك جداً بخصوص طلب #{req.order_id}، فيه ضغط بسيط في المطبخ حالياً "
                    f"والطلب هيحتاج حوالي {req.extra_delay_minutes} دقيقة إضافية عشان يطلع بأعلى جودة. بنشكرك على صبرك معانا."
                )
            else:
                warnings.append("لم يحدد الكاشير مدة التأخير الإضافية؛ لم يتم افتراض 15 دقيقة تلقائياً.")
                msg = (
                    f"بنعتذر لحضرتك جداً بخصوص طلب #{req.order_id}، فيه تأخير بسيط في المطبخ حالياً "
                    "وشغالين نجهز طلبك في أسرع وقت وهنبلغك أول ما يخرج مع الكابتن."
                )
            return CashierDirectiveResponse(
                order_id=req.order_id,
                directive_type=req.directive_type,
                validated=True,
                warnings=warnings,
                customer_message_ar=msg,
                requires_cashier_approval=False,
            )

        # Free-text / out-of-stock directive (e.g., "الاستربس خلصان، اعرض زنجر بنفس السعر")
        norm = normalize_ar(req.directive_text)
        mentioned_products = []
        for p in sorted(self.menu_index.products.values(), key=lambda x: len(x.name_ar), reverse=True):
            if normalize_ar(p.name_ar) in norm:
                mentioned_products.append(p)

        # Also check colloquial mentions like "زنجر" or "ستربس"
        if "زنجر" in norm and not any(p.product_id == "item_006" for p in mentioned_products):
            p_zinger = self.menu_index.get_product("item_006")
            if p_zinger:
                mentioned_products.append(p_zinger)

        suggested_alt = mentioned_products[-1] if mentioned_products else None
        mentions_same_price = "بنفس السعر" in norm or "نفس السعر" in norm
        requires_approval = False

        if mentions_same_price:
            warnings.append("التوجيه يذكر 'بنفس السعر' — يرجى التأكد من اعتماد تعديل السعر في سيستم الكاشير.")
            requires_approval = True

        if suggested_alt:
            price_text = f" ({suggested_alt.base_price} ج.م)" if suggested_alt.base_price is not None else ""
            msg = (
                f"أهلاً بحضرتك بخصوص طلب #{req.order_id}، بنعتذرلك جداً الصنف المطلوب غير متوفر حالياً. "
                f"تحب نبدلهولك بـ {suggested_alt.name_ar}{price_text}؟ يا ريت تأكدلنا عشان نكمل تجهيز الأوردر فوراً."
            )
            return CashierDirectiveResponse(
                order_id=req.order_id,
                directive_type=req.directive_type,
                validated=True,
                warnings=warnings,
                suggested_alternative_product_id=suggested_alt.product_id,
                customer_message_ar=msg,
                requires_cashier_approval=requires_approval,
            )

        msg = (
            f"أهلاً بحضرتك بخصوص طلب #{req.order_id}، رسالة من فريق المطعم: "
            f"{req.directive_text.strip()}. يا ريت ترد علينا بتأكيد حضرتك."
        )
        return CashierDirectiveResponse(
            order_id=req.order_id,
            directive_type=req.directive_type,
            validated=True,
            warnings=warnings,
            customer_message_ar=msg,
            requires_cashier_approval=requires_approval,
        )
