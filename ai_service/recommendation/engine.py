"""Intelligent Context-Aware Recommendation & Combo Upsell Engine for Wasla AI (MOKA).

Strict Commercial Grounding Rules:
1. Recommend ONLY real, currently available products from MOKA's menu database.
2. NEVER invent discounts or combos that are not explicitly configured in `official_combos`.
3. When no official combo exists, recommend separate real items and compute their exact
   combined price using deterministic Decimal arithmetic.
4. Respect customer budget limits, never repeat rejected offers, and never upsell during
   complaints, cancellations, or human handoffs.
5. Recommended items are NEVER added to the cart without explicit customer acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ai_service.schemas.models import CartSnapshot, CustomerIntent
from ai_service.understanding.menu_index import MenuIndex

# Default category -> ordered real MOKA product IDs for context-aware upselling.
# Every ID below is verified in menu/menu.json:
# - item_065: باكت بطاطس (30 EGP)
# - item_066: بطاطس شيدر هالبينو (65 EGP)
# - item_081: ميكس صودا تروبيكال (70 EGP)
# - item_082: ميكس صودا مانجو لوكو (80 EGP)
# - item_096: ميلك شيك فراولة (80 EGP)
# - item_075: كابوتشينو (60 EGP)
DEFAULT_RECOMMENDATION_RULES: dict[str, list[str]] = {
    "cat_sandwich_pizza": ["item_065", "item_081", "item_096"],
    "cat_crepes": ["item_065", "item_081", "item_096"],
    "cat_special_crepes": ["item_065", "item_081", "item_096"],
    "cat_beef_burger": ["item_065", "item_081", "item_066"],
    "cat_chicken_burger": ["item_065", "item_081", "item_066"],
    "cat_pizza_cheese": ["item_081", "item_065", "item_082"],
    "cat_pizza_beef": ["item_081", "item_065", "item_082"],
    "cat_pizza_chicken": ["item_081", "item_065", "item_082"],
    "cat_meals": ["item_065", "item_082", "item_081"],
    "cat_pasta": ["item_065", "item_081", "item_082"],
    "cat_dessert": ["item_075", "item_100"],
    "cat_waffle": ["item_075", "item_100"],
    "cat_pancake": ["item_075", "item_100"],
    "cat_kashtota": ["item_075", "item_100"],
}


@dataclass
class OfficialComboConfig:
    """Explicitly configured restaurant combo (only offered when present in DB/config)."""

    combo_id: str
    name_ar: str
    trigger_categories: list[str]
    item_product_ids: list[str]
    combo_price: Decimal


@dataclass
class RecommendedItem:
    product_id: str
    backend_item_id: int | None
    name_ar: str
    category_id: str
    price: Decimal


@dataclass
class UpsellOffer:
    offer_type: str  # "separate_items" | "official_combo"
    items: list[RecommendedItem]
    combined_price: Decimal
    combo_id: str | None = None
    combo_name_ar: str | None = None
    prompt_ar: str = ""
    trigger_category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offer_type": self.offer_type,
            "combo_id": self.combo_id,
            "combo_name_ar": self.combo_name_ar,
            "combined_price": str(self.combined_price),
            "trigger_category": self.trigger_category,
            "prompt_ar": self.prompt_ar,
            "items": [
                {
                    "product_id": i.product_id,
                    "backend_item_id": i.backend_item_id,
                    "name_ar": i.name_ar,
                    "category_id": i.category_id,
                    "price": str(i.price),
                }
                for i in self.items
            ],
        }


class RecommendationEngine:
    """Configurable, strictly database-grounded recommendation & upsell engine."""

    def __init__(
        self,
        menu_index: MenuIndex,
        rules: dict[str, list[str]] | None = None,
        official_combos: list[OfficialComboConfig] | None = None,
        enabled: bool = True,
    ):
        self.menu_index = menu_index
        self.rules = dict(rules or DEFAULT_RECOMMENDATION_RULES)
        self.official_combos = list(official_combos or [])
        self.enabled = enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def recommend(
        self,
        *,
        cart: CartSnapshot,
        intent: CustomerIntent,
        unavailable_product_ids: set[str] | None = None,
        rejected_product_ids: set[str] | None = None,
        upsell_declined: bool = False,
        budget_limit: Decimal | None = None,
        max_suggestions: int = 2,
    ) -> UpsellOffer | None:
        """Return a grounded upsell offer or None when rules/guards disallow recommending."""
        if not self.enabled:
            return None
        if upsell_declined:
            return None
        if intent in (
            CustomerIntent.COMPLAINT,
            CustomerIntent.CANCEL_ORDER,
            CustomerIntent.REQUEST_HUMAN,
            CustomerIntent.REMOVE_ITEM,
        ):
            return None
        if not cart.items:
            return None

        unavailable = unavailable_product_ids or set()
        rejected = rejected_product_ids or set()
        in_cart_ids = {i.product_id for i in cart.items}
        remaining_budget: Decimal | None = None
        if budget_limit is not None:
            remaining_budget = budget_limit - cart.subtotal
            if remaining_budget <= Decimal("0.00"):
                return None

        # Determine most recent cart item's category
        latest_item = cart.items[-1]
        latest_prod = self.menu_index.get_product(latest_item.product_id)
        if not latest_prod:
            return None
        trigger_cat = latest_prod.category_id

        # 1. Check explicitly configured official combos (never invent combos!)
        for combo in self.official_combos:
            if trigger_cat not in combo.trigger_categories:
                continue
            if any(pid in rejected or pid in unavailable for pid in combo.item_product_ids):
                continue
            if all(pid in in_cart_ids for pid in combo.item_product_ids):
                continue
            if remaining_budget is not None and combo.combo_price > remaining_budget:
                continue
            combo_items: list[RecommendedItem] = []
            valid = True
            for pid in combo.item_product_ids:
                p = self.menu_index.get_product(pid)
                if not p or p.is_available is False or p.base_price is None:
                    valid = False
                    break
                combo_items.append(
                    RecommendedItem(
                        product_id=p.product_id,
                        backend_item_id=p.backend_item_id,
                        name_ar=p.name_ar,
                        category_id=p.category_id,
                        price=p.base_price,
                    )
                )
            if valid and combo_items:
                prompt = (
                    f"عندنا كومبو {combo.name_ar} بسعر {combo.combo_price:.2f} ج.م، "
                    "تحب أضيفهولك مع الأوردر؟"
                )
                return UpsellOffer(
                    offer_type="official_combo",
                    items=combo_items,
                    combined_price=combo.combo_price,
                    combo_id=combo.combo_id,
                    combo_name_ar=combo.name_ar,
                    prompt_ar=prompt,
                    trigger_category=trigger_cat,
                )

        # 2. Recommend separate available real menu items (exact combined price, zero fake discounts)
        candidate_pids = self.rules.get(trigger_cat, [])
        chosen: list[RecommendedItem] = []
        running_total = Decimal("0.00")

        for pid in candidate_pids:
            if len(chosen) >= max_suggestions:
                break
            if pid in in_cart_ids or pid in unavailable or pid in rejected:
                continue
            prod = self.menu_index.get_product(pid)
            if not prod or prod.is_available is False or prod.base_price is None:
                continue
            price = Decimal(str(prod.base_price))
            if remaining_budget is not None and (running_total + price) > remaining_budget:
                continue
            chosen.append(
                RecommendedItem(
                    product_id=prod.product_id,
                    backend_item_id=prod.backend_item_id,
                    name_ar=prod.name_ar,
                    category_id=prod.category_id,
                    price=price,
                )
            )
            running_total += price

        if not chosen:
            return None

        if len(chosen) == 1:
            one = chosen[0]
            prompt = f"تحب أضيفلك {one.name_ar} ({one.price:.0f} ج.م) مع الأوردر؟"
        else:
            parts = " أو ".join(f"{c.name_ar} ({c.price:.0f} ج.م)" for c in chosen)
            prompt = f"تحب أضيفلك {parts}؟ (الإجمالي مع بعض {running_total:.0f} ج.م)"

        return UpsellOffer(
            offer_type="separate_items",
            items=chosen,
            combined_price=running_total,
            prompt_ar=prompt,
            trigger_category=trigger_cat,
        )
