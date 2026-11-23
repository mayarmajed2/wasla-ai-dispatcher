"""Provider-agnostic LLM Client (Gemini 3.5 Flash-Lite primary + Gemini 3.8 Flash fallback)
plus a deterministic OfflineEgyptianNLU engine for zero-cost automated CI/unit testing.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ai_service.config import WaslaConfig
from ai_service.schemas.models import CustomerIntent, ExtractedOrderItem
from ai_service.understanding.menu_index import MenuIndex, normalize_ar

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

ARABIC_NUMBERS = {
    "واحد": 1, "واحدة": 1, "واحده": 1,
    "اتنين": 2, "اثنين": 2, "تنين": 2,
    "تلاته": 3, "تلاتة": 3, "ثلاثة": 3, "ثلاثه": 3,
    "اربعه": 4, "اربعة": 4, "أربعة": 4,
    "خمسه": 5, "خمسة": 5,
    "سته": 6, "ستة": 6,
}


class GeminiLLMClient:
    """Async REST client for active Gemini Flash models with structured JSON output."""

    def __init__(self, config: WaslaConfig):
        self.config = config

    @property
    def is_configured(self) -> bool:
        return bool(self.config.gemini_api_key and not self.config.gemini_api_key.startswith("your_"))

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        use_fallback_model: bool = False,
    ) -> tuple[dict[str, Any], str]:
        if not self.is_configured:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        model = self.config.ai_fallback_model if use_fallback_model else self.config.ai_primary_model
        url = f"{GEMINI_API_BASE}/{model}:generateContent?key={self.config.gemini_api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        return json.loads(raw_text), model


class OfflineEgyptianNLU:
    """Deterministic Egyptian Arabic NLU parser grounded strictly in MOKA's MenuIndex.

    Used for offline automated tests, fallback resilience, and verifying safety guardrails
    without requiring live paid API calls.
    """

    def __init__(self, menu_index: MenuIndex):
        self.menu_index = menu_index

    def parse(self, text: str) -> dict[str, Any]:
        norm = normalize_ar(text)

        # 1. Human handoff / complaint
        if any(k in norm for k in ["كلم موظف", "اكلم حد", "الكاشير", "مساعده بشريه", "شكوي", "شكوى", "الاكل بارد", "اتأخرتوا", "اتاخر"]):
            intent = CustomerIntent.COMPLAINT if any(k in norm for k in ["شكو", "بارد", "تاخر"]) else CustomerIntent.REQUEST_HUMAN
            return {"intent": intent.value, "confidence": 0.92, "items": [], "requires_clarification": False}

        # 2. Cancel order
        if any(k in norm for k in ["الغي", "الغاء", "مش عايز", "كنسل", "الغي الطلب", "الغي الاوردر"]):
            return {"intent": CustomerIntent.CANCEL_ORDER.value, "confidence": 0.93, "items": [], "requires_clarification": False}

        # 3. Confirm order
        if any(k in norm for k in ["اكد الطلب", "اكد الاوردر", "تمام اكد", "ايوه اكد", "موافق اكد", "ابعت الاوردر", "نفذ الطلب"]):
            return {"intent": CustomerIntent.CONFIRM_ORDER.value, "confidence": 0.94, "items": [], "requires_clarification": False}

        # 4. Delivery inquiry
        if any(k in norm for k in ["التوصيل بكام", "الدليفري بكام", "بتوصلوا", "مصاريف التوصيل", "توصيل"]):
            return {"intent": CustomerIntent.DELIVERY_INQUIRY.value, "confidence": 0.90, "items": [], "requires_clarification": False}

        # 5. Price inquiry
        if any(k in norm for k in ["بكام", "سعر", "سعره", "اسعار"]):
            items = self._extract_items(text)
            return {"intent": CustomerIntent.PRICE_INQUIRY.value, "confidence": 0.91, "items": items, "requires_clarification": False}

        # 6. Remove item
        if any(k in norm for k in ["شيل", "احذف", "امسح", "لغي ال"]):
            items = self._extract_items(text)
            return {"intent": CustomerIntent.REMOVE_ITEM.value, "confidence": 0.90, "items": items, "requires_clarification": False}

        # 7. Greeting only
        if norm in {"السلام عليكم", "مساء الخير", "صباح الخير", "اهلا", "ازيك", "الو", "هاي", "hello", "hi"}:
            return {"intent": CustomerIntent.GREETING.value, "confidence": 0.96, "items": [], "requires_clarification": False}

        # 8. Menu inquiry
        if any(k in norm for k in ["المنيو", "عندكم ايه", "ابعت المنيو", "القائمه"]):
            return {"intent": CustomerIntent.MENU_INQUIRY.value, "confidence": 0.92, "items": [], "requires_clarification": False}

        # 9. Order extraction
        items = self._extract_items(text)
        if items:
            return {"intent": CustomerIntent.ADD_TO_CART.value, "confidence": 0.88, "items": items, "requires_clarification": False}

        return {
            "intent": CustomerIntent.UNCLEAR_REQUEST.value,
            "confidence": 0.45,
            "items": [],
            "requires_clarification": True,
            "clarification_reason": "لم يتم التعرف على صنف محدد من منيو MOKA.",
        }

    def _extract_quantity(self, segment: str) -> int:
        digit_match = re.search(r"\b(\d+)\b", segment)
        if digit_match:
            val = int(digit_match.group(1))
            if 1 <= val <= 50 and val not in (10, 20):
                return val
        for word, num in ARABIC_NUMBERS.items():
            if re.search(rf"\b{word}\b", segment):
                return num
        return 1

    def _extract_size(self, norm: str) -> str | None:
        if any(w in norm for w in ["20 قطعه", "عشرين قطعه"]):
            return "20PCS"
        if any(w in norm for w in ["10 قطع", "عشر قطع"]):
            return "10PCS"
        if any(w in norm for w in ["دبل", "دابل", "double"]):
            return "D"
        if any(w in norm for w in ["سنجل", "single"]):
            return "S"
        if any(w in norm for w in ["كبير", "كبيره", "لارج", "large"]):
            return "L"
        if any(w in norm for w in ["وسط", "ميديم", "medium"]):
            return "M"
        return None

    def _extract_modifiers(self, norm: str) -> list[str]:
        mods: list[str] = []
        if any(w in norm for w in ["موتزريلا", "موزاريلا"]):
            mods.append("mod_crepe_mozzarella")
        if "هالبينو" in norm and "بيتزا تشكن هالبينو" not in norm and "بطاطس شيدر هالبينو" not in norm:
            mods.append("mod_crepe_jalapeno")
        if "بصل مكرمل" in norm:
            mods.append("mod_crepe_caramelized_onion")
        if "صوص زياده" in norm or "اكسترا صوص" in norm or "صوص من اختيارك" in norm:
            mods.append("mod_crepe_sauce_choice")
        if "حشو اطراف" in norm or "ستافد كراست" in norm:
            mods.append("mod_pizza_stuffed_crust")
        if "حبه كامله" in norm:
            mods.append("mod_pizza_whole_wheat_crust")
        if "اكسترا بسطرمه" in norm or "اكسترا بيبروني" in norm:
            mods.append("mod_pizza_extra_pastrami_pepperoni")
        return mods

    def _extract_notes(self, text: str) -> str:
        match = re.search(r"((?:من غير|بدون)\s+[\u0600-\u06FF\s]+)", text)
        return match.group(1).strip() if match else ""

    def _extract_items(self, text: str) -> list[dict[str, Any]]:
        norm_full = normalize_ar(text)
        segments = re.split(r"\bو\b|\+|,|،|\n", text)
        extracted: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Check longest product names first across full text or segments
        sorted_products = sorted(self.menu_index.products.values(), key=lambda p: len(p.name_ar), reverse=True)
        for seg in segments:
            seg_norm = normalize_ar(seg)
            if not seg_norm:
                continue
            matched = None
            for p in sorted_products:
                p_norm = normalize_ar(p.name_ar)
                if p_norm in seg_norm:
                    matched = p
                    break
            if matched and matched.product_id not in seen_ids:
                seen_ids.add(matched.product_id)
                opts: dict[str, str] = {}
                if matched.product_id == "item_037":  # بيتزا تشكن رانش/باربكيو
                    has_ranch = "رانش" in seg_norm
                    has_bbq = "باربكيو" in seg_norm or "باربيكيو" in seg_norm
                    if has_ranch and not has_bbq:
                        opts["sauce"] = "رانش"
                    elif has_bbq and not has_ranch:
                        opts["sauce"] = "باربيكيو"
                extracted.append(
                    ExtractedOrderItem(
                        product_id=matched.product_id,
                        raw_mention=seg.strip(),
                        quantity=self._extract_quantity(seg_norm),
                        size_code=self._extract_size(seg_norm),
                        modifier_ids=[m for m in self._extract_modifiers(seg_norm) if m in matched.modifiers],
                        selected_options=opts,
                        notes=self._extract_notes(seg),
                    ).model_dump()
                )
            else:
                # Check colloquial aliases or ambiguous short mentions
                cleaned = re.sub(r"\b(عايز|عاوز|محتاج|هات|ممكن|لو سمحت|بكام|سعر|شيل|احذف|واحد|واحده|اتنين|تلاته|\d+)\b", "", seg_norm).strip()
                cleaned = re.sub(r"\b(كبير|وسط|سنجل|دبل|من غير.*|بدون.*)\b", "", cleaned).strip()
                if cleaned:
                    exact, candidates = self.menu_index.resolve_mention(cleaned)
                    if exact and exact.product_id not in seen_ids:
                        seen_ids.add(exact.product_id)
                        extracted.append(
                            ExtractedOrderItem(
                                product_id=exact.product_id,
                                raw_mention=seg.strip(),
                                quantity=self._extract_quantity(seg_norm),
                                size_code=self._extract_size(seg_norm),
                                modifier_ids=[m for m in self._extract_modifiers(seg_norm) if m in exact.modifiers],
                                notes=self._extract_notes(seg),
                            ).model_dump()
                        )
                    elif candidates:
                        extracted.append(
                            ExtractedOrderItem(
                                product_id=None,
                                raw_mention=cleaned,
                                quantity=self._extract_quantity(seg_norm),
                                size_code=self._extract_size(seg_norm),
                                notes=self._extract_notes(seg),
                            ).model_dump()
                        )
        return extracted
