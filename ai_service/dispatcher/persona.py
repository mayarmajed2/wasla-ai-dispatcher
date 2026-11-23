"""Natural, Warm Egyptian Arabic Conversational Personality Renderer for Wasla AI.

Design Principles:
- Friendly, concise, natural Egyptian Arabic (warm restaurant staff tone, never robotic).
- Occasional appropriate emojis (❤️, ✨, 😊), never excessive.
- Strictly grounded in verified menu items, prices, and cart totals.
- Never invents offers, discounts, items, or delivery times.
"""

from __future__ import annotations

from decimal import Decimal
import re

from ai_service.recommendation.engine import UpsellOffer
from ai_service.schemas.models import CartSnapshot, MenuItemDTO

_GREETING_VARIANTS = [
    "أهلاً بيك في MOKA ❤️ نورتنا! تحب تطلب إيه النهاردة؟",
    "منورنا في MOKA 😊 تحب تبدأ بإيه من المنيو النهاردة؟",
    "أهلاً وسهلاً في MOKA ❤️ تحت أمرك، تحب تطلب إيه؟",
]


def render_greeting(turn_count: int = 0) -> str:
    return _GREETING_VARIANTS[turn_count % len(_GREETING_VARIANTS)]


def render_menu_overview(active_count: int) -> str:
    return (
        f"عندنا {active_count} صنف جاهزين في منيو MOKA 😊 "
        "بيتزا وساندوتش بيتزا، كريب، برجر لحم وفراخ، باستا، وجبات، ومشروبات وحلويات. "
        "تحب تعرف أسعار قسم معين؟"
    )


def render_price_reply(prod: MenuItemDTO) -> str:
    if prod.sizes:
        prices = " | ".join(f"{s.name_ar}: {s.price} ج.م" for s in prod.sizes)
        return f"سعر {prod.name_ar}: {prices} ✨ تحب أضيفهولك في الطلب؟"
    return f"سعر {prod.name_ar}: {prod.base_price} ج.م ✨ تحب أضيفهولك في الطلب؟"


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


def render_upsell_declined(cart: CartSnapshot) -> str:
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


def render_unknown_product(raw_mention: str) -> str:
    return (
        f"للأسف '{raw_mention}' مش موجود في منيو MOKA حالياً. "
        "تحب أرشحلك حاجة قريبة من البيتزا أو الكريب أو البرجر؟"
    )


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
