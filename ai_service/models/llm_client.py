"""Provider-agnostic LLM Client (Gemini 3.5 Flash-Lite primary + Gemini 3.8 Flash fallback)
plus a typo-tolerant OfflineEgyptianNLU engine grounded strictly in MOKA's 145-item MenuIndex.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ai_service.config import WaslaConfig
from ai_service.schemas.models import CustomerIntent, ExtractedOrderItem
from ai_service.understanding.arabic_nlp import normalize_arabic, phonetic_arabic, strip_fillers
from ai_service.understanding.menu_index import MenuIndex

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

ARABIC_NUMBERS = {
    "واحد": 1, "واحدة": 1, "واحده": 1,
    "اتنين": 2, "اثنين": 2, "تنين": 2,
    "تلاته": 3, "تلاتة": 3, "ثلاثة": 3, "ثلاثه": 3,
    "اربعه": 4, "اربعة": 4, "أربعة": 4,
    "خمسه": 5, "خمسة": 5,
    "سته": 6, "ستة": 6,
}

# Prefixes indicating a customer self-correction (e.g., "لأ اقصد مارجريتا", "خليها بيبروني")
CORRECTION_PREFIXES = ("لا ", "لأ ", "اقصد", "قصدي", "خليها", "خليه", "بدلها", "بدله", "غيرها")


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
        url = f"{GEMINI_API_BASE}/{model}:generateContent"
        headers = {"x-goog-api-key": self.config.gemini_api_key, "Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
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
    """Typo-tolerant Egyptian Arabic NLU parser grounded strictly in MOKA's MenuIndex.

    Clearly identified as offline deterministic/fuzzy NLU (never presented as live LLM inference).
    """

    def __init__(self, menu_index: MenuIndex):
        self.menu_index = menu_index

    def parse(self, text: str, category_hint: str | None = None) -> dict[str, Any]:
        norm = normalize_arabic(text)
        if not norm:
            return {
                "intent": CustomerIntent.UNCLEAR_REQUEST.value,
                "confidence": 0.0,
                "items": [],
                "requires_clarification": True,
                "clarification_reason": "empty_message",
            }

        # 1. Human handoff / complaint
        if any(k in norm for k in ["كلم موظف", "اكلم حد", "الكاشير", "مساعده بشريه", "شكوي", "الاكل بارد", "اتاخرتوا", "اتاخر"]):
            intent = CustomerIntent.COMPLAINT if any(k in norm for k in ["شكو", "بارد", "تاخر"]) else CustomerIntent.REQUEST_HUMAN
            return {"intent": intent.value, "confidence": 0.92, "items": [], "requires_clarification": False}

        # 2. Cancel order
        if any(k in norm for k in ["الغي الطلب", "الغي الاوردر", "الغاء الطلب", "كنسل", "الغي كل", "مش عايز حاجه"]):
            return {"intent": CustomerIntent.CANCEL_ORDER.value, "confidence": 0.93, "items": [], "requires_clarification": False}

        # 3. Confirm order
        if any(k in norm for k in ["اكد الطلب", "اكد الاوردر", "تمام اكد", "ايوه اكد", "موافق اكد", "ابعت الاوردر", "نفذ الطلب"]):
            return {"intent": CustomerIntent.CONFIRM_ORDER.value, "confidence": 0.94, "items": [], "requires_clarification": False}

        # 4. Delivery inquiry
        if any(k in norm for k in ["التوصيل بكام", "الدليفري بكام", "بتوصلوا", "مصاريف التوصيل", "توصيل"]):
            return {"intent": CustomerIntent.DELIVERY_INQUIRY.value, "confidence": 0.90, "items": [], "requires_clarification": False}

        # 5. Price inquiry
        if any(k in norm for k in ["بكام", "بكم", "سعر", "سعره", "سعرها", "اسعار"]):
            items, meta = self._extract_items(text, category_hint=category_hint)
            return {
                "intent": CustomerIntent.PRICE_INQUIRY.value,
                "confidence": 0.91,
                "items": items,
                "requires_clarification": False,
                "match_meta": meta,
            }

        # 6. Remove item
        if any(k in norm for k in ["شيل", "احذف", "امسح"]):
            items, meta = self._extract_items(text, category_hint=category_hint)
            return {
                "intent": CustomerIntent.REMOVE_ITEM.value,
                "confidence": 0.90,
                "items": items,
                "requires_clarification": False,
                "match_meta": meta,
            }

        # 7. Greeting only
        if norm in {"السلام عليكم", "مساء الخير", "صباح الخير", "اهلا", "ازيك", "الو", "هاي", "hello", "hi"}:
            return {"intent": CustomerIntent.GREETING.value, "confidence": 0.96, "items": [], "requires_clarification": False}

        # 8. Menu inquiry
        if any(k in norm for k in ["المنيو", "عندكم ايه", "ابعت المنيو", "القائمه"]):
            return {"intent": CustomerIntent.MENU_INQUIRY.value, "confidence": 0.92, "items": [], "requires_clarification": False}

        # 9. Customer self-correction or Order extraction
        is_correction = any(norm.startswith(p) for p in CORRECTION_PREFIXES)
        items, meta = self._extract_items(text, category_hint=category_hint)
        if items:
            best_conf = max((m.get("confidence", 0.8) for m in meta), default=0.85)
            return {
                "intent": CustomerIntent.MODIFY_ORDER.value if is_correction else CustomerIntent.ADD_TO_CART.value,
                "confidence": round(best_conf, 2),
                "items": items,
                "requires_clarification": False,
                "match_meta": meta,
                "is_correction": is_correction,
            }

        # If customer typed a food request with fillers (e.g. "عايزة سوشي") that matched NO MOKA item
        cleaned = strip_fillers(text)
        if cleaned:
            return {
                "intent": CustomerIntent.ADD_TO_CART.value,
                "confidence": 0.35,
                "items": [ExtractedOrderItem(product_id=None, raw_mention=cleaned, quantity=self._extract_quantity(norm)).model_dump()],
                "requires_clarification": True,
                "clarification_reason": "unknown_product",
                "match_meta": [{"tier": "low", "confidence": 0.0, "candidates": []}],
            }

        return {
            "intent": CustomerIntent.UNCLEAR_REQUEST.value,
            "confidence": 0.35,
            "items": [],
            "requires_clarification": True,
            "clarification_reason": "unclear_item",
        }

    def _split_segments(self, text: str) -> list[str]:
        """Split multi-item Arabic orders cleanly without breaking words that contain 'و'."""
        norm = normalize_arabic(text)
        # Split on explicit separators: standalone 'و', '+', ',', '،', newline
        parts = re.split(r"\s+و\s+|\+|,|،|\n", norm)
        final_parts: list[str] = []
        # Also split Egyptian attached 'و' before quantities or major category nouns (e.g. "بيتزا مارجريتا وكريب كريسبي")
        attach_re = re.compile(
            r"\s+و(?=(?:واحد|واحده|اتنين|اثنين|تلاته|اربعه|بيتزا|كريب|برجر|باستا|وجبه|بطاطس|وافل|بان|قشطوطه|طاجن|ميلك|ايس|هوت|قهوه|سندوتش|\d+))"
        )
        for p in parts:
            sub = attach_re.split(p.strip())
            final_parts.extend(s.strip() for s in sub if s.strip())
        return final_parts

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
        if any(w in norm for w in ["كبير", "كبيره", "لارج", "large"]) or re.search(r"\bl\b", norm):
            return "L"
        if any(w in norm for w in ["وسط", "ميديم", "medium"]) or re.search(r"\bm\b", norm):
            return "M"
        return None

    def _extract_modifiers(self, norm: str) -> list[str]:
        ph = phonetic_arabic(norm)
        mods: list[str] = []
        if "موتزريلا" in ph:
            mods.append("mod_crepe_mozzarella")
        if "هالبينو" in ph and "بيتزا تشكن هالبينو" not in ph and "بطاطس شيدر هالبينو" not in ph:
            mods.append("mod_crepe_jalapeno")
        if "بصل مكرمل" in ph:
            mods.append("mod_crepe_caramelized_onion")
        if "صوص زياده" in ph or "اكسترا صوص" in ph or "صوص من اختيارك" in ph:
            mods.append("mod_crepe_sauce_choice")
        if "حشو اطراف" in ph or "ستافد كراست" in ph:
            mods.append("mod_pizza_stuffed_crust")
        if "حبه كامله" in ph:
            mods.append("mod_pizza_whole_wheat_crust")
        if "اكسترا بسطرمه" in ph or "اكسترا بيبروني" in ph:
            mods.append("mod_pizza_extra_pastrami_pepperoni")
        return mods

    def _extract_notes(self, text: str) -> str:
        match = re.search(r"((?:من غير|بدون)\s+[\u0600-\u06FFa-zA-Z0-9\s]+)", text)
        return match.group(1).strip() if match else ""

    def _extract_options(self, product_id: str, seg_norm: str) -> dict[str, str]:
        opts: dict[str, str] = {}
        if product_id == "item_037":  # بيتزا تشكن رانش/باربكيو
            ph = phonetic_arabic(seg_norm)
            has_ranch = "رانش" in ph
            has_bbq = "باربكيو" in ph
            if has_ranch and not has_bbq:
                opts["sauce"] = "رانش"
            elif has_bbq and not has_ranch:
                opts["sauce"] = "باربيكيو"
        elif product_id == "item_092":  # ميكس صودا كلاسيك
            for flavor in ["فراوله", "صن شاين", "كيوي", "مانجو", "توت", "خوخ"]:
                if flavor in seg_norm:
                    opts["flavor"] = flavor
                    break
        return opts

    def _extract_items(
        self,
        text: str,
        category_hint: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        segments = self._split_segments(text)
        extracted: list[dict[str, Any]] = []
        match_meta: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for seg in segments:
            cleaned = strip_fillers(seg)
            if not cleaned:
                continue

            res = self.menu_index.resolve_mention_with_confidence(cleaned, category_hint=category_hint)
            qty = self._extract_quantity(seg)
            size = self._extract_size(seg)
            notes = self._extract_notes(seg)

            if res.tier == "high" and res.product is not None:
                if res.product.product_id in seen_ids:
                    continue
                seen_ids.add(res.product.product_id)
                mods = [m for m in self._extract_modifiers(seg) if m in res.product.modifiers]
                opts = self._extract_options(res.product.product_id, seg)
                extracted.append(
                    ExtractedOrderItem(
                        product_id=res.product.product_id,
                        raw_mention=cleaned,
                        quantity=qty,
                        size_code=size,
                        modifier_ids=mods,
                        selected_options=opts,
                        notes=notes,
                    ).model_dump()
                )
                match_meta.append({"tier": "high", "confidence": res.confidence, "candidates": []})
            elif res.tier in ("ambiguous", "medium") and res.candidates:
                extracted.append(
                    ExtractedOrderItem(
                        product_id=None,
                        raw_mention=cleaned,
                        quantity=qty,
                        size_code=size,
                        notes=notes,
                    ).model_dump()
                )
                match_meta.append(
                    {
                        "tier": res.tier,
                        "confidence": res.confidence,
                        "candidates": [c.product_id for c in res.candidates],
                        "category_hint": res.category_hint,
                    }
                )

        return extracted, match_meta
