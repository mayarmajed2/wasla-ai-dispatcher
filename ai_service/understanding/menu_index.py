"""Canonical MOKA Menu Index, Advanced Egyptian Arabic Normalization & Fuzzy Disambiguation Engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_service.schemas.models import (
    MenuItemDTO,
    MenuModifierDTO,
    MenuOptionGroupDTO,
    MenuSizeDTO,
)
from ai_service.understanding.arabic_nlp import (
    hybrid_similarity,
    normalize_arabic,
    phonetic_arabic,
    strip_fillers,
)

# Backwards-compatible export
normalize_ar = normalize_arabic

# Calibrated fuzzy matching thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.64
AMBIGUITY_MARGIN = 0.05

# Common Egyptian colloquial aliases -> canonical MOKA Arabic product names (never fragile numeric IDs)
COLLOQUIAL_ALIASES: dict[str, str] = {
    "بطاطس محمره": "باكت بطاطس",
    "بطاطس": "باكت بطاطس",
    "باكت بطاطس": "باكت بطاطس",
    "كريب زنجر": "كريب زنجر حار",
    "زنجر": "كريب زنجر حار",
    "كريب ستربس": "كريب كريسبي",
    "كريب شيش طاووق": "كريب شيش",
    "فراخ مشويه": "وجبة تشكن جريل",
    "وجبه فراخ مشويه": "وجبة تشكن جريل",
    "وجبه شيش": "وجبة شيش طاووق",
    "وجبه كريسبي": "وجبة كريسبي تشيكن",
    "نجرسكو": "نجرسكو فراخ",
    "الفريدو": "باستا تشكن الفريدو",
    "مارجريتا": "بيتزا مارجريتا",
    "بيبروني": "بيتزا بيبروني",
    "سوبر سوبريم": "بيتزا سوبر سوبريم",
    "سوبر بوسطن": "بيتزا سوبر بوسطن",
    "قهوه": "قهوه تركي",
    "تركي": "قهوه تركي",
    "بيتزا تشكن رانش": "بيتزا تشكن رانش/باربكيو",
    "بيتزا فراخ رانش": "بيتزا تشكن رانش/باربكيو",
    "بيتزا تشكن باربكيو": "بيتزا تشكن رانش/باربكيو",
    "بيتزا فراخ باربكيو": "بيتزا تشكن رانش/باربكيو",
    "بيتزا تشكن باربيكيو": "بيتزا تشكن رانش/باربكيو",
    "بيتزا فراخ باربيكيو": "بيتزا تشكن رانش/باربكيو",
}

# Category-level / overlapping short terms that MUST trigger polite disambiguation
AMBIGUOUS_SHORT_TERMS: dict[str, list[str]] = {
    "نوتيلا": ["ميلك شيك نوتيلا", "بان كيك نوتيلا", "قشطوطة نوتيلا", "طاجن نوتيلا", "وافل نوتيلا"],
    "لوتس": ["ميلك شيك لوتس", "بان كيك لوتس", "قشطوطة لوتس", "طاجن لوتس", "وافل لوتس"],
    "كيندر": ["ميلك شيك كيندر", "بان كيك كيندر", "قشطوطة كيندر", "طاجن كيندر", "وافل كيندر"],
    "بستاشيو": ["ميلك شيك بستاشيو", "بان كيك بستاشيو", "قشطوطة بستاشيو", "وافل بستاشيو"],
    "كلاسيك": ["كلاسيك برجر", "كلاسيك تشكن", "ميكس صودا كلاسيك"],
    "تروبيكال": ["ميكس صودا تروبيكال", "ريد بول ميكس تروبيكال"],
    "باشون فروت": ["ميكس صودا باشون فروت", "ميلك شيك باشون فروت", "ريد بول ميكس باشون فروت"],
    "مانجو لوكو": ["ميكس صودا مانجو لوكو", "ريد بول ميكس مانجو لوكو"],
    "برجر": ["كلاسيك برجر", "تشيز برجر", "سموكي برجر", "باربكيو برجر", "وايلد برجر"],
    "برجر لحم": ["كلاسيك برجر", "تشيز برجر", "سموكي برجر", "باربكيو برجر", "وايلد برجر"],
    "برجر فراخ": ["كلاسيك تشكن", "تشيز تشكين", "تشكين بومب", "سلو تشكين", "باربكيو تشكين"],
    "بيتزا": ["بيتزا مارجريتا", "بيتزا مكس جبن", "بيتزا سجق", "بيتزا بيبروني", "بيتزا تشكن رانش/باربكيو"],
    "كريب": ["كريب بطاطس", "كريب كريسبي", "كريب زنجر حار", "كريب شاورما", "كريب مشكل فراخ"],
    "وافل": ["وافل نوتيلا", "وافل لوتس", "وافل كيندر", "وافل بستاشيو"],
    "قشطوطه": ["قشطوطة نوتيلا", "قشطوطة لوتس", "قشطوطة كراميل", "قشطوطة كيندر", "قشطوطة بستاشيو"],
    "بان كيك": ["بان كيك نوتيلا", "بان كيك لوتس", "بان كيك كيندر", "بان كيك بستاشيو"],
    "طاجن": ["طاجن نوتيلا", "طاجن كيندر", "طاجن اوريو", "طاجن لوتس", "طاجن ميكس موكا"],
    "ميلك شيك": ["ميلك شيك اوريو", "ميلك شيك نوتيلا", "ميلك شيك لوتس", "ميلك شيك مانجو", "ميلك شيك فراولة"],
    "باستا": ["نجرسكو فراخ", "باستا تشكن الفريدو", "باستا وايت ستربس", "باستا نجرسكو كرانشي"],
    "وجبه": ["وجبة شيش طاووق", "وجبة كريسبي تشيكن", "وجبة تشكن جريل"],
    "كولسلو": ["كولسلو كبير", "كولسلو صغير"],
}


@dataclass
class MatchResult:
    product: MenuItemDTO | None = None
    candidates: list[MenuItemDTO] = field(default_factory=list)
    confidence: float = 0.0
    tier: str = "low"  # "high" | "medium" | "ambiguous" | "low"
    category_hint: str | None = None


class MenuIndex:
    """In-memory indexed view of MOKA's canonical 145-item menu.json with typo-tolerant matching."""

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
        self._phonetic_name_to_id: dict[str, str] = {}

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
            self._norm_name_to_id[normalize_arabic(dto.name_ar)] = dto.product_id
            self._norm_name_to_id[normalize_arabic(dto.name_en)] = dto.product_id
            self._phonetic_name_to_id[phonetic_arabic(dto.name_ar)] = dto.product_id

    def get_product(self, product_id: str) -> MenuItemDTO | None:
        return self.products.get(product_id)

    def get_modifier(self, modifier_id: str) -> MenuModifierDTO | None:
        return self.modifiers.get(modifier_id)

    def list_by_category(self, category_query: str | None = None) -> list[MenuItemDTO]:
        if not category_query:
            return list(self.products.values())
        q = normalize_arabic(category_query)
        matched_cat_ids = {
            cid
            for cid, c in self.categories.items()
            if q in normalize_arabic(c.get("name_ar", "")) or q in normalize_arabic(c.get("name_en", ""))
        }
        return [p for p in self.products.values() if p.category_id in matched_cat_ids]

    def resolve_mention_with_confidence(
        self,
        mention: str,
        category_hint: str | None = None,
    ) -> MatchResult:
        """Hybrid calibrated resolver: Exact -> Phonetic -> Alias -> Ambiguity -> Fuzzy."""
        norm = strip_fillers(mention) or normalize_arabic(mention)
        if not norm:
            return MatchResult(tier="low", confidence=0.0)

        # If a category hint exists from an earlier turn (e.g., "بيتزا" + "مرجريتا"), combine for matching
        combined_query = f"{category_hint} {norm}".strip() if (category_hint and category_hint not in norm) else norm

        # 1. Explicit ambiguity guard (unless category_hint narrows it)
        if norm in AMBIGUOUS_SHORT_TERMS and not category_hint:
            candidates = []
            for name in AMBIGUOUS_SHORT_TERMS[norm]:
                pid = self._norm_name_to_id.get(normalize_arabic(name))
                if pid and pid in self.products:
                    candidates.append(self.products[pid])
            return MatchResult(
                product=None,
                candidates=candidates,
                confidence=0.70,
                tier="ambiguous",
                category_hint=norm,
            )

        # 2. Exact normalized match
        for q in (combined_query, norm):
            if q in self._norm_name_to_id:
                return MatchResult(product=self.products[self._norm_name_to_id[q]], confidence=1.0, tier="high")

        # 3. Phonetic exact match (e.g., "بيتزا مرجريتا" -> "بيتزا مارجريتا", "كريب كرسبي" -> "كريب كريسبي")
        for q in (combined_query, norm):
            ph = phonetic_arabic(q)
            if ph in self._phonetic_name_to_id:
                return MatchResult(product=self.products[self._phonetic_name_to_id[ph]], confidence=0.96, tier="high")

        # 4. Colloquial alias match
        for q in (combined_query, norm):
            if q in COLLOQUIAL_ALIASES:
                pid = self._norm_name_to_id.get(normalize_arabic(COLLOQUIAL_ALIASES[q]))
                prod = self.products.get(pid) if pid else None
                if prod:
                    return MatchResult(product=prod, confidence=0.95, tier="high")
            ph = phonetic_arabic(q)
            for alias_k, alias_name in COLLOQUIAL_ALIASES.items():
                if phonetic_arabic(alias_k) == ph:
                    pid = self._norm_name_to_id.get(normalize_arabic(alias_name))
                    prod = self.products.get(pid) if pid else None
                    if prod:
                        return MatchResult(product=prod, confidence=0.94, tier="high")

        # 5. Hybrid fuzzy scoring across all 145 MOKA products
        pool = list(self.products.values())
        scored: list[tuple[float, MenuItemDTO]] = []
        for p in pool:
            s_ar = hybrid_similarity(combined_query, p.name_ar)
            s_ar_plain = hybrid_similarity(norm, p.name_ar)
            s_en = hybrid_similarity(norm, p.name_en)
            score = max(s_ar, s_ar_plain, s_en)
            # Boost items matching category_hint when resolving a multi-turn follow-up
            if category_hint and (
                normalize_arabic(category_hint) in normalize_arabic(p.name_ar)
                or normalize_arabic(category_hint) in normalize_arabic(p.category_ar)
            ):
                score = min(1.0, score + 0.06)
            if score >= MEDIUM_CONFIDENCE_THRESHOLD:
                scored.append((round(score, 4), p))

        if not scored:
            return MatchResult(product=None, candidates=[], confidence=0.0, tier="low")

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_item = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        # High confidence with clear separation over #2
        if best_score >= HIGH_CONFIDENCE_THRESHOLD and (best_score - second_score) >= AMBIGUITY_MARGIN:
            return MatchResult(product=best_item, candidates=[], confidence=best_score, tier="high")

        # Medium confidence or close top-2 candidates -> ask customer to choose safely
        top_candidates = [item for sc, item in scored[:4] if sc >= max(MEDIUM_CONFIDENCE_THRESHOLD, best_score - 0.12)]
        if len(top_candidates) == 1 and best_score >= HIGH_CONFIDENCE_THRESHOLD:
            return MatchResult(product=best_item, candidates=[], confidence=best_score, tier="high")

        return MatchResult(
            product=None,
            candidates=top_candidates,
            confidence=best_score,
            tier="medium",
        )

    def resolve_mention(self, mention: str) -> tuple[MenuItemDTO | None, list[MenuItemDTO]]:
        """Backwards-compatible tuple interface returning (exact_or_high_match, candidate_matches)."""
        res = self.resolve_mention_with_confidence(mention)
        if res.tier == "high" and res.product is not None:
            return res.product, []
        return None, res.candidates

    def resolve_unit_price(self, product: MenuItemDTO, size_code: str | None) -> tuple[Decimal | None, str | None]:
        """Return (unit_price, error_reason) deterministically from canonical menu."""
        if product.sizes:
            if not size_code:
                size_labels = " ولا ".join(f"{s.name_ar} ({s.price} ج.م)" for s in product.sizes)
                return None, f"تمام، {product.name_ar} 👌 تحبها {size_labels}؟"
            for s in product.sizes:
                if s.code.upper() == size_code.upper() or normalize_arabic(s.name_ar) == normalize_arabic(size_code):
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
