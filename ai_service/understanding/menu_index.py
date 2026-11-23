"""Canonical MOKA Menu Index, Arabic Normalization, and Disambiguation Engine."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_service.schemas.models import (
    MenuItemDTO,
    MenuModifierDTO,
    MenuOptionGroupDTO,
    MenuSizeDTO,
)


def normalize_ar(text: str) -> str:
    """Normalize Egyptian Arabic orthography for reliable menu matching."""
    if not text:
        return ""
    s = text.strip().lower()
    s = re.sub(r"[إأآا]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = re.sub(r"[\u064B-\u0652]", "", s)  # strip diacritics
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Common Egyptian colloquial aliases -> canonical MOKA product_ids
COLLOQUIAL_ALIASES: dict[str, str] = {
    "بطاطس محمره": "item_073",      # باكت بطاطس
    "بطاطس": "item_073",            # باكت بطاطس
    "باكت بطاطس": "item_073",
    "فرايز": "item_073",
    "كريب زنجر": "item_006",        # كريب زنجر حار
    "زنجر": "item_006",
    "كريب ستربس": "item_005",       # كريب كريسبي (ستربس فريش)
    "كريب شيش طاووق": "item_009",   # كريب شيش
    "فراخ مشويه": "item_024",       # وجبة تشكن جريل
    "وجبه فراخ مشويه": "item_024",
    "وجبه شيش": "item_022",         # وجبة شيش طاووق
    "وجبه كريسبي": "item_023",      # وجبة كريسبي تشيكن
    "نجرسكو": "item_025",           # نجرسكو فراخ
    "الفريدو": "item_026",          # باستا تشكن الفريدو
    "مارجريتا": "item_029",         # بيتزا مارجريتا
    "بيبروني": "item_033",          # بيتزا بيبروني
    "سوبر سوبريم": "item_034",      # بيتزا سوبر سوبريم
    "قهوه": "item_087",             # قهوه تركي
    "تركي": "item_087",             # قهوه تركي
    "كولسلو": "item_080",           # ambiguous small/large handled separately
}

# Short generic words that appear across multiple MOKA categories and MUST trigger clarification
AMBIGUOUS_SHORT_TERMS: dict[str, list[str]] = {
    "نوتيلا": ["ميلك شيك نوتيلا", "بان كيك نوتيلا", "قشطوطة نوتيلا", "طاجن نوتيلا", "وافل نوتيلا"],
    "لوتس": ["ميلك شيك لوتس", "بان كيك لوتس", "قشطوطة لوتس", "طاجن لوتس", "وافل لوتس"],
    "كيندر": ["ميلك شيك كيندر", "بان كيك كيندر", "قشطوطة كيندر", "طاجن كيندر", "وافل كيندر"],
    "بستاشيو": ["ميلك شيك بستاشيو", "بان كيك بستاشيو", "قشطوطة بستاشيو", "وافل بستاشيو"],
    "كلاسيك": ["كلاسيك برجر", "كلاسيك تشكن", "ميكس صودا كلاسيك"],
    "تروبيكال": ["ميكس صودا تروبيكال", "ريد بول ميكس تروبيكال"],
    "باشون فروت": ["ميكس صودا باشون فروت", "ميلك شيك باشون فروت", "ريد بول ميكس باشون فروت"],
    "مانجو لوكو": ["ميكس صودا مانجو لوكو", "ريد بول ميكس مانجو لوكو"],
    "برجر": ["كلاسيك برجر", "تشيز برجر", "سموكي برجر", "باربكيو برجر", "وايلد برجر", "كلاسيك تشكن"],
    "بيتزا": ["بيتزا مارجريتا", "بيتزا مكس جبن", "بيتزا سجق", "بيتزا بيبروني", "بيتزا تشكن رانش/باربكيو"],
    "كريب": ["كريب بطاطس", "كريب كريسبي", "كريب زنجر حار", "كريب شاورما", "كريب مشكل فراخ"],
    "وافل": ["وافل نوتيلا", "وافل لوتس", "وافل كيندر", "وافل بستاشيو"],
    "قشطوطه": ["قشطوطة نوتيلا", "قشطوطة لوتس", "قشطوطة كراميل", "قشطوطة كيندر", "قشطوطة بستاشيو"],
}


class MenuIndex:
    """In-memory indexed view of MOKA's canonical menu.json."""

    def __init__(self, menu_path: Path):
        self.menu_path = menu_path
        raw = json.loads(menu_path.read_text(encoding="utf-8"))
        self.restaurant_name: str = raw.get("restaurant_name", "MOKA")
        self.currency: str = raw.get("currency", "EGP")

        self.categories: dict[str, dict[str, Any]] = {
            c["category_id"]: c for c in raw.get("categories", [])
        }
        self.modifiers: dict[str, MenuModifierDTO] = {
            m["modifier_id"]: MenuModifierDTO(
                modifier_id=m["modifier_id"],
                name_ar=m["name_ar"],
                name_en=m["name_en"],
                price=Decimal(str(m["price"])),
                applies_to_categories=m.get("applies_to_categories", []),
            )
            for m in raw.get("modifiers", [])
        }

        self.products: dict[str, MenuItemDTO] = {}
        self._norm_name_to_id: dict[str, str] = {}

        for idx, p in enumerate(raw.get("products", []), start=1):
            cat = self.categories.get(p["category_id"], {})
            dto = MenuItemDTO(
                product_id=p["product_id"],
                backend_item_id=idx,
                name_ar=p["name_ar"],
                name_en=p["name_en"],
                category_id=p["category_id"],
                category_ar=cat.get("name_ar", ""),
                description_ar=p.get("description_ar"),
                base_price=Decimal(str(p["base_price"])) if p.get("base_price") is not None else None,
                sizes=[
                    MenuSizeDTO(
                        code=s["code"],
                        name_ar=s["name_ar"],
                        name_en=s["name_en"],
                        price=Decimal(str(s["price"])),
                    )
                    for s in p.get("sizes", [])
                ],
                modifiers=p.get("modifiers", []),
                options=[
                    MenuOptionGroupDTO(
                        option_group=o["option_group"],
                        required=o.get("required", True),
                        choices=o.get("choices", []),
                    )
                    for o in p.get("options", [])
                ],
                is_available=p.get("available"),
            )
            self.products[dto.product_id] = dto
            self._norm_name_to_id[normalize_ar(dto.name_ar)] = dto.product_id
            self._norm_name_to_id[normalize_ar(dto.name_en)] = dto.product_id

    def get_product(self, product_id: str) -> MenuItemDTO | None:
        return self.products.get(product_id)

    def get_modifier(self, modifier_id: str) -> MenuModifierDTO | None:
        return self.modifiers.get(modifier_id)

    def list_by_category(self, category_query: str | None = None) -> list[MenuItemDTO]:
        if not category_query:
            return list(self.products.values())
        q = normalize_ar(category_query)
        matched_cat_ids = {
            cid
            for cid, c in self.categories.items()
            if q in normalize_ar(c.get("name_ar", "")) or q in normalize_ar(c.get("name_en", ""))
        }
        return [p for p in self.products.values() if p.category_id in matched_cat_ids]

    def resolve_mention(self, mention: str) -> tuple[MenuItemDTO | None, list[MenuItemDTO]]:
        """Resolve a customer mention into (exact_match, candidate_matches).

        Returns (None, candidates) when the mention is ambiguous across categories.
        """
        norm = normalize_ar(mention)
        if not norm:
            return None, []

        # 1. Explicit ambiguity guard
        if norm in AMBIGUOUS_SHORT_TERMS:
            candidates = []
            for name in AMBIGUOUS_SHORT_TERMS[norm]:
                pid = self._norm_name_to_id.get(normalize_ar(name))
                if pid and pid in self.products:
                    candidates.append(self.products[pid])
            return None, candidates

        # 2. Exact normalized match
        if norm in self._norm_name_to_id:
            return self.products[self._norm_name_to_id[norm]], []

        # 3. Colloquial alias match
        if norm in COLLOQUIAL_ALIASES:
            pid = COLLOQUIAL_ALIASES[norm]
            if norm == "كولسلو":
                return None, [self.products["item_079"], self.products["item_080"]]
            return self.products.get(pid), []

        # 4. Substring / token overlap search
        tokens = set(norm.split())
        scored: list[tuple[int, MenuItemDTO]] = []
        for p in self.products.values():
            p_norm = normalize_ar(p.name_ar)
            if norm in p_norm or p_norm in norm:
                scored.append((100 + len(p_norm), p))
            else:
                p_tokens = set(p_norm.split())
                overlap = len(tokens & p_tokens)
                if overlap >= max(1, len(tokens) - 1) and overlap >= 2:
                    scored.append((overlap * 20, p))

        if not scored:
            return None, []

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]
        top_matches = [item for score, item in scored if score >= top_score - 10]
        if len(top_matches) == 1:
            return top_matches[0], []
        return None, top_matches[:5]

    def resolve_unit_price(self, product: MenuItemDTO, size_code: str | None) -> tuple[Decimal | None, str | None]:
        """Return (unit_price, error_reason) deterministically from canonical menu."""
        if product.sizes:
            if not size_code:
                size_labels = " أو ".join(f"{s.name_ar} ({s.code}: {s.price} ج)" for s in product.sizes)
                return None, f"محتاج نحدد الحجم لـ {product.name_ar}: {size_labels}"
            for s in product.sizes:
                if s.code.upper() == size_code.upper() or normalize_ar(s.name_ar) == normalize_ar(size_code):
                    return s.price, None
            return None, f"الحجم '{size_code}' غير متاح لـ {product.name_ar}."
        if product.base_price is None:
            return None, f"السعر غير محدد لـ {product.name_ar}."
        return product.base_price, None

    def build_compact_prompt_catalog(self) -> str:
        """Compact token-efficient representation of MOKA's menu for LLM grounding."""
        lines = [f"MENU ({self.restaurant_name} - {self.currency}):"]
        by_cat: dict[str, list[MenuItemDTO]] = {}
        for p in self.products.values():
            by_cat.setdefault(p.category_id, []).append(p)
        for cid, items in by_cat.items():
            cat_name = self.categories[cid]["name_ar"]
            lines.append(f"[{cat_name}]")
            for p in items:
                if p.sizes:
                    price_str = "/".join(f"{s.code}:{s.price}" for s in p.sizes)
                else:
                    price_str = f"{p.base_price}"
                opt_str = ""
                if p.options:
                    opt_str = " | req:" + ",".join(f"{o.option_group}({'/'.join(o.choices)})" for o in p.options)
                lines.append(f"- {p.product_id}: {p.name_ar} ({price_str}){opt_str}")
        lines.append("[EXTRAS]")
        for m in self.modifiers.values():
            lines.append(f"- {m.modifier_id}: {m.name_ar} (+{m.price} {self.currency})")
        return "\n".join(lines)
