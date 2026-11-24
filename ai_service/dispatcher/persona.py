"""Natural, Warm Egyptian Arabic Conversational Personality Renderer for Wasla AI.

Design Principles:
- Friendly, concise, natural Egyptian Arabic (warm restaurant staff tone, never robotic).
- Occasional appropriate emojis (❤️, ✨, 😊), never excessive.
- Strictly grounded in verified menu items, live DB prices, availability, and cart totals.
- Never invents offers, discounts, items, or delivery times.
"""

from __future__ import annotations

from decimal import Decimal
import re
from typing import Iterable

from ai_service.recommendation.engine import UpsellOffer
from ai_service.schemas.models import CartSnapshot, MenuItemDTO

_GREETING_VARIANTS = [
    "أهلاً بيك في MOKA ❤️ نورتنا! تحب تطلب إيه النهاردة؟",
    "منورنا في MOKA 😊 تحب تبدأ بإيه من المنيو النهاردة؟",
    "أهلاً وسهلاً في MOKA ❤️ تحت أمرك، تحب تطلب إيه؟",
]


def render_greeting(turn_count: int = 0) -> str:
    return _GREETING_VARIANTS[turn_count % len(_GREETING_VARIANTS)]


def _format_product_price(prod: MenuItemDTO, size_code: str | None = None) -> str:
    if prod.sizes:
        if size_code:
            for sz in prod.sizes:
                if sz.code.upper() == size_code.upper():
                    return f"{sz.name_ar}: {sz.price} ج.م"
        return " | ".join(f"{s.name_ar}: {s.price} ج.م" for s in prod.sizes)
    return f"{prod.base_price} ج.م"


def render_menu_overview(
    items_or_count: int | Iterable[MenuItemDTO],
    requested_category: str | None = None,
) -> str:
    if isinstance(items_or_count, int):
        return (
            f"عندنا {items_or_count} صنف جاهزين في منيو MOKA 😊 "
            "بيتزا وساندوتش بيتزا، كريب، برجر لحم وفراخ، باستا، وجبات، ومشروبات وحلويات. "
            "تحب تعرف أسعار قسم معين؟"
        )

    items = list(items_or_count)
    if requested_category and items:
        preview = []
        for p in items[:8]:
            if p.sizes:
                p_price = f"من {min(s.price for s in p.sizes)} ج.م"
            else:
                p_price = f"{p.base_price} ج.م"
            preview.append(f"• {p.name_ar} ({p_price})")
        more = f"\n...وغيرها ({len(items)} صنف متوفرين في القسم ده)." if len(items) > 8 else ""
        return f"أشهر أصناف قسم {requested_category} المتوفرة دلوقتي في MOKA ✨\n" + "\n".join(preview) + more + "\nتحب أضيفلك حاجة منهم؟"

    return (
        f"عندنا {len(items)} صنف متوفرين حالياً في منيو MOKA 😊 "
        "بيتزا وساندوتش بيتزا، كريب، برجر لحم وفراخ، باستا، وجبات، ومشروبات وحلويات. "
        "تحب تعرف أسعار قسم معين أو تطلب صنف محدد؟"
    )


def render_price_reply(prod: MenuItemDTO, available: bool = True, size_code: str | None = None) -> str:
    if not available or prod.is_available is False:
        return f"معلش، {prod.name_ar} غير متوفر حالياً في منيو MOKA 🙏 تحب أرشحلك صنف بديل؟"
    price_text = _format_product_price(prod, size_code=size_code)
    return f"سعر {prod.name_ar}: {price_text} ✨ تحب أضيفهولك في الطلب؟"


def render_multi_price_reply(entries: list[tuple[MenuItemDTO, bool, str | None]]) -> str:
    if len(entries) == 1:
        prod, avail, sc = entries[0]
        return render_price_reply(prod, available=avail, size_code=sc)

    lines: list[str] = []
    any_available = False
    for prod, avail, sc in entries:
        if not avail or prod.is_available is False:
            lines.append(f"• {prod.name_ar}: غير متوفر حالياً")
        else:
            any_available = True
            lines.append(f"• {prod.name_ar}: {_format_product_price(prod, size_code=sc)}")

    header = "أسعار الأصناف في منيو MOKA ✨"
    footer = "تحب أضيفلك حاجة منهم في الطلب؟" if any_available else "تحب أرشحلك بدائل متوفرة حالياً؟"
    return f"{header}\n" + "\n".join(lines) + f"\n{footer}"


def render_item_added(
    added_names: list[str],
    cart: CartSnapshot,
    upsell: UpsellOffer | None = None,
    turn_count: int = 0,
) -> str:
    openers = ["حاضر من عيوني ❤️", "تمام يا فندم ✨", "أحلى اختيار 😊"]
    opener = openers[turn_count % len(openers)]
    items_str = "، ".join(added_names)
    base = f"{opener} ضفتلك ({items_str}) — إجمالي السلة دلوقتي {cart.subtotal} ج.م (قبل التوصيل)."
    if upsell and upsell.prompt_ar:
        return f"{base} {upsell.prompt_ar}"
    return f"{base} تحب تزود حاجة تانية ولا نأكد الأوردر؟"


def render_upsell_declined(cart: CartSnapshot, turn_count: int = 0) -> str:
    summary = "، ".join(f"{i.quantity} × {i.name_ar}" for i in cart.items)
    if summary:
        return (
            f"ولا يهمك خالص 😊 طلبك الحالي ({summary}) بإجمالي {cart.subtotal} ج.م (قبل التوصيل). "
            "تحب نأكد الأوردر دلوقتي ولا في أي تعديل؟"
        )
    return (
        f"ولا يهمك خالص 😊 طلبك زي ما هو بإجمالي {cart.subtotal} ج.م (قبل التوصيل). "
        "تحب نأكد الأوردر دلوقتي؟"
    )


def render_item_removed(cart: CartSnapshot) -> str:
    if not cart.items:
        return "تمام، شيلته من السلة والسلة دلوقتي فاضية. تحب تطلب حاجة تانية؟"
    return f"تمام، شيلت الصنف من السلة 👍 إجمالي السلة دلوقتي {cart.subtotal} ج.م."


def render_order_cancelled() -> str:
    return "حاضر، تم إلغاء الطلب الحالي بالكامل. لو حبيت تطلب في أي وقت إحنا موجودين ❤️"


def render_missing_details(total_amount: Decimal, missing_fields: list[str]) -> str:
    return (
        f"عشان نثبت الأوردر (إجمالي {total_amount} ج.م)، "
        f"محتاجين منك بس: {' و'.join(missing_fields)} 😊"
    )


def render_order_confirmed(order_id: int) -> str:
    return f"تم تأكيد طلبك بنجاح ❤️ رقم الأوردر #{order_id} واتبعت فوراً للكاشير والمطبخ. بألف هنا!"


def render_ambiguous_product(raw_mention: str, options: list[str], is_medium: bool) -> str:
    opts = " / ".join(options)
    if is_medium:
        return f"تقصد أي واحد من دول في منيو MOKA: {opts}؟"
    return f"عندنا أكتر من نوع لـ '{raw_mention}': {opts} — تحب أنهي واحد فيهم؟"


_UNKNOWN_PRODUCT_VARIANTS = [
    "للأسف '{item}' مش موجود في منيو MOKA حالياً 😊 تحب أرشحلك حاجة من البيتزا أو الكريب أو البرجر؟",
    "معلش والله، معندناش '{item}' في MOKA ❤️ إحنا متميزين في البيتزا، الكريب، البرجر، والباستا — تحب تشوف حاجة منهم؟",
    "'{item}' مش متوفر عندنا في المنيو 😊 ممكن نختار حاجة تانية من الكريب أو الساندوتشات أو المشروبات؟",
]

_UNCLEAR_VARIANTS = [
    "ممكن توضحلي تحب تطلب إيه بالظبط؟ 😊 عندنا في MOKA بيتزا، كريب، برجر، باستا، وجبات، ومشروبات وحلويات.",
    "تحت أمرك ❤️ قولي بس نفسك في إيه من المنيو (بيتزا، كريب، برجر، مشروبات أو حلو) وأنا أظبطلك أحلى طلب!",
    "معاك يا فندم 😊 تحب أعرضلك أقسام المنيو ولا في صنف معين على بالك؟",
]


def render_unknown_product(raw_mention: str, turn_count: int = 0) -> str:
    clean = (raw_mention or "").strip()
    tpl = _UNKNOWN_PRODUCT_VARIANTS[turn_count % len(_UNKNOWN_PRODUCT_VARIANTS)]
    return tpl.format(item=clean)


def render_unclear_request(turn_count: int = 0) -> str:
    return _UNCLEAR_VARIANTS[turn_count % len(_UNCLEAR_VARIANTS)]


def render_human_handoff() -> str:
    return "حاضر من عيوني ❤️ حولت المحادثة فوراً للكاشير وهيتواصل معاك حالاً."


def extract_budget_limit(text: str) -> Decimal | None:
    """Extract customer-stated budget in EGP (e.g., 'الميزانية 200', 'مش اكتر من 180 جنيه')."""
    patterns = [
        r"(?:ميزاني[ةت]|الميزاني[ةت]|معايا|بحد\s*أقصى|بحد\s*اقصى|مش\s*اكتر\s*من|مش\s*أكتر\s*من|في\s*حدود)\s*[:\-]?\s*(\d{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                val = Decimal(m.group(1))
                if val > Decimal("0"):
                    return val
            except Exception:
                pass
    return None
