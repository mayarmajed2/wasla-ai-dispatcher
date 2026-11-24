"""Gemini REST Client with Native Function Calling + Typo-Tolerant OfflineEgyptianNLU."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from typing import Any

import httpx

from ai_service.config import WaslaConfig
from ai_service.schemas.models import CustomerIntent, ExtractedOrderItem
from ai_service.tools.registry import ALLOWED_TOOL_NAMES, TOOL_DECLARATIONS
from ai_service.understanding.arabic_nlp import (
    is_gibberish_input,
    normalize_arabic,
    phonetic_arabic,
    strip_fillers,
)
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

# Explicit correction cues; a plain "لا" is not enough to replace an existing cart item.
CORRECTION_PREFIXES = ("اقصد", "قصدي", "خليها", "خليه", "خليهم", "بدلها", "بدله", "غيرها")


def _strip_unsupported_schema_fields(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_unsupported_schema_fields(v)
            for k, v in node.items()
            if k not in {"minimum", "maximum", "additionalProperties"}
        }
    if isinstance(node, list):
        return [_strip_unsupported_schema_fields(x) for x in node]
    return node


def _build_gemini_function_declarations() -> list[dict[str, Any]]:
    return [_strip_unsupported_schema_fields(copy.deepcopy(decl)) for decl in TOOL_DECLARATIONS]


def _extract_json_from_text(raw_text: str) -> dict[str, Any] | None:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _infer_payload_from_function_calls(function_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Synthesize dispatcher-compatible intent & items when Gemini responds via native functionCall."""
    if not function_calls:
        return {
            "intent": CustomerIntent.UNCLEAR_REQUEST.value,
            "confidence": 0.4,
            "items": [],
            "requires_clarification": True,
            "function_calls": [],
        }

    items: list[dict[str, Any]] = []
    intent = CustomerIntent.UNCLEAR_REQUEST.value
    for fc in function_calls:
        name = fc["name"]
        args = fc["args"]
        if name == "modify_cart":
            intent = CustomerIntent.ADD_TO_CART.value
            items.append(
                {
                    "product_id": args.get("product_id"),
                    "raw_mention": args.get("raw_mention") or str(args.get("product_id") or ""),
                    "quantity": args.get("quantity", 1),
                    "size_code": args.get("size_code"),
                    "modifier_ids": args.get("modifier_ids") or [],
                    "selected_options": args.get("selected_options") or {},
                    "notes": args.get("notes") or "",
                }
            )
        elif name == "check_item_availability":
            if intent == CustomerIntent.UNCLEAR_REQUEST.value:
                intent = CustomerIntent.PRICE_INQUIRY.value
            items.append(
                {
                    "product_id": args.get("product_id"),
                    "raw_mention": str(args.get("product_id") or ""),
                    "quantity": 1,
                }
            )
        elif name == "get_active_menu" and intent == CustomerIntent.UNCLEAR_REQUEST.value:
            intent = CustomerIntent.MENU_INQUIRY.value
        elif name == "remove_cart_item" and intent == CustomerIntent.UNCLEAR_REQUEST.value:
            intent = CustomerIntent.REMOVE_ITEM.value
            items.append({"product_id": args.get("cart_item_id"), "raw_mention": str(args.get("cart_item_id") or ""), "quantity": 1})
        elif name == "resolve_delivery" and intent == CustomerIntent.UNCLEAR_REQUEST.value:
            intent = CustomerIntent.DELIVERY_INQUIRY.value
        elif name == "confirm_and_lock_order":
            intent = CustomerIntent.CONFIRM_ORDER.value
        elif name == "request_human_assistance":
            intent = CustomerIntent.REQUEST_HUMAN.value
        elif name == "get_cart_summary" and intent == CustomerIntent.UNCLEAR_REQUEST.value:
            intent = CustomerIntent.PRICE_INQUIRY.value

    return {
        "intent": intent,
        "confidence": 0.90,
        "items": items,
        "requires_clarification": False,
        "function_calls": function_calls,
    }


class GeminiLLMClient:
    """Async REST client for active Gemini Flash models with Native Function Calling + structured JSON."""

    def __init__(self, config: WaslaConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self._transport = transport

    @property
    def is_configured(self) -> bool:
        key = (self.config.gemini_api_key or "").strip()
        return bool(key and not key.startswith("your_") and key.lower() not in {"disabled", "none", "false", "null"})

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
        effective_system_prompt = (
            system_prompt.rstrip()
            + "\nIMPORTANT: Unless emitting a declared tool functionCall, respond ONLY with a valid JSON object (never plain prose)."
        )
        payload = {
            "systemInstruction": {"parts": [{"text": effective_system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "tools": [{"functionDeclarations": _build_gemini_function_declarations()}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {
                "temperature": 0.1,
            },
        }

        data: dict[str, Any] | None = None
        last_exc: Exception | None = None
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                client_kwargs: dict[str, Any] = {"timeout": 20.0}
                if self._transport is not None:
                    client_kwargs["transport"] = self._transport
                async with httpx.AsyncClient(**client_kwargs) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                        await asyncio.sleep(0.25)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 < max_attempts:
                    await asyncio.sleep(0.25)
                    continue
                raise
            except Exception:
                raise

        if data is None:
            raise last_exc or RuntimeError("Gemini request failed without response")

        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini response missing candidates")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        function_calls: list[dict[str, Any]] = []
        text_chunks: list[str] = []

        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("Malformed Gemini response part")
            if "functionCall" in part:
                fc = part["functionCall"]
                if not isinstance(fc, dict):
                    raise ValueError("Malformed Gemini functionCall object")
                fc_name = fc.get("name")
                fc_args = fc.get("args", {})
                if not isinstance(fc_name, str) or fc_name not in ALLOWED_TOOL_NAMES:
                    raise ValueError(f"Invalid or disallowed Gemini functionCall tool: {fc_name}")
                if not isinstance(fc_args, dict):
                    raise ValueError(f"Malformed Gemini functionCall args for {fc_name}")
                function_calls.append({"name": fc_name, "args": fc_args})
            elif "text" in part and isinstance(part["text"], str):
                text_chunks.append(part["text"])

        combined_text = "\n".join(text_chunks).strip()
        parsed_json = _extract_json_from_text(combined_text) if combined_text else None

        if parsed_json is not None:
            if function_calls:
                parsed_json["function_calls"] = function_calls
                if not parsed_json.get("items") and any(fc["name"] in ("modify_cart", "check_item_availability") for fc in function_calls):
                    inferred = _infer_payload_from_function_calls(function_calls)
                    parsed_json["items"] = inferred["items"]
                    if not parsed_json.get("intent"):
                        parsed_json["intent"] = inferred["intent"]
            return parsed_json, model

        if function_calls:
            return _infer_payload_from_function_calls(function_calls), model

        raise ValueError(f"Gemini returned invalid/non-JSON response: {combined_text[:120]}")


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
        if any(
            k in norm
            for k in [
                "الغي الطلب",
                "الغي الاوردر",
                "الغاء الطلب",
                "كنسل",
                "الغي كل",
                "مش عايز حاجه",
                "مش عايز اطلب",
                "مش هطلب",
                "بلاش الاوردر",
                "بلاش الطلب",
                "امسح كل حاجه",
            ]
        ):
            return {"intent": CustomerIntent.CANCEL_ORDER.value, "confidence": 0.93, "items": [], "requires_clarification": False}

        # 2b. Tracking is a read-only order lookup, never a delivery-fee question.
        has_order_subject = any(term in norm for term in ("الاوردر", "اوردر", "الطلب", "طلب"))
        asks_order_number = any(
            phrase in norm
            for phrase in ("رقم الاوردر", "رقم اوردر", "رقم الطلب", "رقم طلب", "نمره الاوردر", "نمره الطلب")
        )
        order_id_match = re.search(r"(?:الاوردر|اوردر|الطلب|طلب)\s*(?:رقم\s*)?#?\s*(\d+)", norm)
        if has_order_subject and (
            asks_order_number
            or any(term in norm for term in ("اتحرك", "طلع", "وصل", "لفين", "فين", "حالته", "تتبع"))
        ):
            result = {
                "intent": CustomerIntent.ORDER_STATUS.value,
                "confidence": 0.98,
                "items": [],
                "requires_clarification": False,
            }
            if order_id_match:
                result["order_id"] = int(order_id_match.group(1))
            return result

        if any(term in norm for term in ("المجموع كام", "الاجمالي كام", "اجمالي الطلب كام", "حسابي كام")):
            return {"intent": CustomerIntent.GET_CART.value, "confidence": 0.95, "items": [], "requires_clarification": False}

        # Availability questions, product rejection, and customer-reported stock issues are distinct.
        unavailable_report = re.search(r"\s+مش\s+(?:موجود|متوفر|متاح)(?:\s|$)", norm)
        if unavailable_report:
            return {
                "intent": CustomerIntent.REPORT_UNAVAILABLE.value,
                "confidence": 0.95,
                "items": [],
                "requires_clarification": False,
            }

        if re.match(r"^مفيش\s+", norm):
            return {
                "intent": CustomerIntent.AVAILABILITY_INQUIRY.value,
                "confidence": 0.95,
                "items": [],
                "requires_clarification": False,
            }

        rejected_product = re.match(r"^مش\s+(?:عايزه?|عاوزه?|حابب|محتاجه?)\s+(.+)$", norm)
        if rejected_product:
            item_text = rejected_product.group(1).strip()
            items, meta = self._extract_items(item_text, category_hint=category_hint)
            return {
                "intent": CustomerIntent.REJECT_PRODUCT.value,
                "confidence": 0.95,
                "items": items,
                "requires_clarification": False,
                "match_meta": meta,
            }

        # 3. Confirm order (also preserve any items mentioned in the same burst before confirming)
        if any(k in norm for k in ["اكد الطلب", "اكد الاوردر", "تمام اكد", "ايوه اكد", "موافق اكد", "ابعت الاوردر", "نفذ الطلب"]):
            items, meta = self._extract_items(text, category_hint=category_hint)
            return {
                "intent": CustomerIntent.CONFIRM_ORDER.value,
                "confidence": 0.94,
                "items": items,
                "requires_clarification": False,
                "match_meta": meta,
            }

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

        # 6. Explicitly negated item mentions must never be treated as new additions.
        refusal = re.match(
            r"^(?:لا[،, ]+)?(?:مش(?:\s+(?:عايزه?|محتاجه?))?|بلاش|ما\s+تحطش|ماتحطش|متضيفش|ما\s+تضيفش|ما\s+تطلبش)\s+(.+)$",
            norm,
        )
        if refusal:
            item_text = re.split(
                r"\s+(?:خلاص|شيله|شيلها|شيلهم|طلعه|طلعيه|امسحه|احذفه|من\s+الطلب|من\s+السله|اللي\s+ضفته).*$",
                refusal.group(1),
                maxsplit=1,
            )[0].strip()
            items, meta = self._extract_items(item_text or refusal.group(1), category_hint=category_hint)
            if not items and item_text:
                items = [
                    ExtractedOrderItem(product_id=None, raw_mention=item_text, quantity=1).model_dump()
                ]
                meta = [{"tier": "low", "confidence": 0.0, "candidates": []}]
            if items:
                return {
                    "intent": CustomerIntent.REMOVE_ITEM.value,
                    "confidence": 0.90,
                    "items": items,
                    "requires_clarification": False,
                    "match_meta": meta,
                }

        # 7. Remove item
        if any(k in norm for k in ["شيل", "احذف", "امسح", "طلعه", "طلعيه", "بلاش"]):
            items, meta = self._extract_items(text, category_hint=category_hint)
            return {
                "intent": CustomerIntent.REMOVE_ITEM.value,
                "confidence": 0.90,
                "items": items,
                "requires_clarification": False,
                "match_meta": meta,
            }

        # 8. Greeting only
        if norm in {"السلام عليكم", "مساء الخير", "صباح الخير", "اهلا", "ازيك", "الو", "هاي", "hello", "hi"}:
            return {"intent": CustomerIntent.GREETING.value, "confidence": 0.96, "items": [], "requires_clarification": False}

        # 9. Category-specific menu inquiry. These questions must not fall through
        # to fuzzy add-to-cart extraction, even when the requested category is absent.
        category_terms = [
            normalize_arabic(c.get("name_ar", ""))
            for c in self.menu_index.categories.values()
        ] + ["عصير", "عصاير", "عصائر", "باستا", "برجر", "بيتزا", "ميلك شيك"]
        asks_for_category = any(term and term in norm for term in category_terms)
        asks_what_is_available = any(term in norm for term in ("ايه", "انواع", "المتاح", "موجود"))
        addresses_restaurant = any(term in norm for term in ("عندكم", "عندكو", "عندنا"))
        if any(k in norm for k in ["المنيو", "عندكم ايه", "عندكو ايه", "ابعت المنيو", "القائمه"]) or (
            addresses_restaurant and asks_for_category and asks_what_is_available
        ):
            return {"intent": CustomerIntent.MENU_INQUIRY.value, "confidence": 0.92, "items": [], "requires_clarification": False}

        # 10. Customer self-correction or Order extraction
        raw_trimmed = (text or "").strip().lower()
        is_correction = any(norm.startswith(p.strip()) for p in CORRECTION_PREFIXES) or any(
            raw_trimmed.startswith(p) for p in CORRECTION_PREFIXES
        ) or bool(re.match(r"^لا[،, ]+(?:اقصد|قصدي|خلي|بدل|غير)", norm))
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

        # 9b. Elliptical self-correction referencing previous cart item (e.g. "لأ، خليها كبيرة", "خليهم 2", "بدون بصل")
        if is_correction:
            corr_size = self._extract_size(norm)
            corr_qty = self._extract_explicit_quantity(norm)
            corr_mods = self._extract_modifiers(norm)
            corr_notes = self._extract_notes(text)
            if corr_size or corr_qty is not None or corr_mods or corr_notes:
                return {
                    "intent": CustomerIntent.MODIFY_ORDER.value,
                    "confidence": 0.90,
                    "items": [],
                    "requires_clarification": False,
                    "is_correction": True,
                    "correction_patch": {
                        "size_code": corr_size,
                        "quantity": corr_qty,
                        "modifier_ids": corr_mods,
                        "notes": corr_notes,
                    },
                }

        # 10. Reject random keysmash / gibberish cleanly instead of echoing it as a dish
        if is_gibberish_input(text):
            return {
                "intent": CustomerIntent.UNCLEAR_REQUEST.value,
                "confidence": 0.30,
                "items": [],
                "requires_clarification": True,
                "clarification_reason": "unclear_item",
            }

        # 11. Customer typed a food request with fillers (e.g. "عايزة سوشي") that matched NO MOKA item
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
        """Split multi-item Arabic orders cleanly across newlines & separators without breaking 'و' inside words."""
        raw_lines = [ln.strip() for ln in re.split(r"[\r\n]+", text or "") if ln.strip()]
        final_parts: list[str] = []
        attach_re = re.compile(
            r"\s+و(?=(?:واحد|واحده|اتنين|اثنين|تلاته|اربعه|بيتزا|كريب|برجر|باستا|وجبه|بطاطس|وافل|بان|قشطوطه|طاجن|ميلك|ايس|هوت|قهوه|سندوتش|طبق|\d+))"
        )
        for line in raw_lines:
            norm = normalize_arabic(line)
            parts = re.split(r"\s+و\s+|\+|,|،", norm)
            for p in parts:
                sub = attach_re.split(p.strip())
                final_parts.extend(s.strip() for s in sub if s.strip())
        return final_parts

    def _extract_explicit_quantity(self, segment: str) -> int | None:
        digit_match = re.search(r"\b(\d+)\b", segment)
        if digit_match:
            val = int(digit_match.group(1))
            if 1 <= val <= 50 and val not in (10, 20):
                return val
        for word, num in ARABIC_NUMBERS.items():
            if re.search(rf"\b{word}\b", segment):
                return num
        return None

    def _extract_quantity(self, segment: str) -> int:
        return self._extract_explicit_quantity(segment) or 1

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
        variant_index: dict[tuple[Any, ...], int] = {}

        for seg in segments:
            cleaned = strip_fillers(seg)
            qty = self._extract_quantity(seg)
            size = self._extract_size(seg)
            notes = self._extract_notes(seg)
            mods_raw = self._extract_modifiers(seg)

            # Elliptical follow-up segment referencing the previous item (e.g., "وواحدة كبير" or "\nمن غير بصل")
            if not cleaned:
                if extracted and extracted[-1].get("product_id"):
                    prev = extracted[-1]
                    prev_prod = self.menu_index.get_product(prev["product_id"])
                    if prev_prod and (size is not None or mods_raw):
                        mods = [m for m in mods_raw if m in prev_prod.modifiers]
                        opts = self._extract_options(prev_prod.product_id, seg)
                        v_key = (
                            prev_prod.product_id,
                            size,
                            tuple(sorted(mods)),
                            tuple(sorted(opts.items())),
                            notes,
                        )
                        if v_key in variant_index:
                            extracted[variant_index[v_key]]["quantity"] += qty
                        else:
                            idx = len(extracted)
                            extracted.append(
                                ExtractedOrderItem(
                                    product_id=prev_prod.product_id,
                                    raw_mention=prev["raw_mention"],
                                    quantity=qty,
                                    size_code=size,
                                    modifier_ids=mods,
                                    selected_options=opts,
                                    notes=notes,
                                ).model_dump()
                            )
                            variant_index[v_key] = idx
                            match_meta.append({"tier": "high", "confidence": 0.92, "candidates": []})
                    elif notes and not prev.get("notes"):
                        prev["notes"] = notes
                continue

            res = self.menu_index.resolve_mention_with_confidence(cleaned, category_hint=category_hint)
            if res.tier == "high" and res.product is not None:
                mods = [m for m in mods_raw if m in res.product.modifiers]
                opts = self._extract_options(res.product.product_id, seg)
                v_key = (
                    res.product.product_id,
                    size,
                    tuple(sorted(mods)),
                    tuple(sorted(opts.items())),
                    notes,
                )
                if v_key in variant_index:
                    extracted[variant_index[v_key]]["quantity"] += qty
                else:
                    idx = len(extracted)
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
                    variant_index[v_key] = idx
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
