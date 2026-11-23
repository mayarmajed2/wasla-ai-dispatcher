"""Core Wasla AI Dispatcher — Egyptian Arabic Ordering & Deterministic Safety Orchestrator."""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.audio.pipeline import HybridAudioPipeline
from ai_service.config import WaslaConfig
from ai_service.dispatcher.cashier_directives import CashierDirectiveInterpreter
from ai_service.models.finetuned_intent import FineTunedIntentClassifier
from ai_service.models.llm_client import GeminiLLMClient, OfflineEgyptianNLU
from ai_service.schemas.models import (
    AudioInput,
    CashierDirectiveRequest,
    CashierDirectiveResponse,
    ConversationMessage,
    CustomerIntent,
    DispatcherResponse,
    ExtractedOrderItem,
    SessionState,
    ToolCallRecord,
)
from ai_service.security.guardrails import detect_prompt_injection, redact_pii
from ai_service.tools.registry import ToolExecutor
from ai_service.understanding.menu_index import MenuIndex, normalize_ar

logger = logging.getLogger("wasla.dispatcher")


class WaslaAIDispatcher:
    """Stateless-core / injected-backend AI Dispatcher for Wasla."""

    def __init__(
        self,
        config: WaslaConfig,
        menu_index: MenuIndex,
        backend: Any,
        llm_client: GeminiLLMClient | None = None,
        audio_pipeline: HybridAudioPipeline | None = None,
    ):
        self.config = config
        self.menu_index = menu_index
        self.backend = backend
        self.llm_client = llm_client or GeminiLLMClient(config)
        self.offline_nlu = OfflineEgyptianNLU(menu_index)
        self.finetuned_intent = FineTunedIntentClassifier()
        self.audio_pipeline = audio_pipeline or HybridAudioPipeline(config)
        self.tools = ToolExecutor(backend)
        self.directive_interpreter = CashierDirectiveInterpreter(menu_index)
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, phone_number: str) -> SessionState:
        if phone_number not in self._sessions:
            self._sessions[phone_number] = SessionState(phone_number=phone_number)
        return self._sessions[phone_number]

    def interpret_cashier_directive(self, req: CashierDirectiveRequest) -> CashierDirectiveResponse:
        return self.directive_interpreter.interpret(req)

    async def handle_audio(self, phone_number: str, audio: AudioInput) -> DispatcherResponse:
        transcription = await self.audio_pipeline.process(audio)
        response = await self.handle_text(phone_number, transcription.transcript, input_type="audio")
        return response.model_copy(update={"transcript": transcription.transcript})

    async def handle_text(
        self,
        phone_number: str,
        text: str,
        input_type: str = "text",
        data_source: str = "api",
        is_test: bool = False,
    ) -> DispatcherResponse:
        session = self.get_session(phone_number)
        self._current_source = data_source
        self._current_is_test = is_test
        clean_text = text.strip()
        if self.config.redact_pii_in_logs:
            logger.info("Incoming turn [%s]: %s", redact_pii(phone_number), redact_pii(clean_text))

        session.history.append(ConversationMessage(role="customer", content=clean_text, input_type=input_type))
        tool_calls: list[ToolCallRecord] = []

        # 1. Security guardrail: prompt injection / fake discount rejection
        if detect_prompt_injection(clean_text):
            reply = "أهلاً بحضرتك في MOKA! الأسعار والخصومات معتمدة من سيستم المطعم فقط. تحب تطلب إيه من المنيو؟"
            session.history.append(ConversationMessage(role="assistant", content=reply))
            return DispatcherResponse(
                intent=CustomerIntent.UNCLEAR_REQUEST,
                confidence=1.0,
                requires_clarification=True,
                clarification_reason="blocked_prompt_injection",
                cart=session.cart,
                reply_ar=reply,
                model_used="security_guardrail",
            )

        # 2. Extract customer details if present (name, backup phone, address)
        self._capture_customer_details(session, clean_text)

        # 3. Check if answering a pending clarification (size / option / ambiguous item)
        if session.pending_clarification:
            resolved = await self._try_resolve_clarification(session, clean_text, tool_calls)
            if resolved is not None:
                return resolved

        # 4. Run NLU (Gemini 3.5 Flash-Lite / 3.8 Flash if configured, else deterministic typo-tolerant OfflineEgyptianNLU)
        category_hint = (session.pending_clarification or {}).get("category_hint")
        parsed, model_used = await self._run_nlu(clean_text, category_hint=category_hint)
        raw_intent_str = str(parsed.get("intent", CustomerIntent.UNCLEAR_REQUEST.value)).strip().lower()
        intent_alias_map = {
            "place_order": "add_to_cart",
            "order": "add_to_cart",
            "create_order": "add_to_cart",
            "ask_price": "price_inquiry",
            "ask_menu": "menu_inquiry",
            "confirm": "confirm_order",
            "cancel": "cancel_order",
            "human": "request_human",
        }
        raw_intent_str = intent_alias_map.get(raw_intent_str, raw_intent_str)
        try:
            intent = CustomerIntent(raw_intent_str)
        except ValueError:
            intent = CustomerIntent.ADD_TO_CART if parsed.get("items") else CustomerIntent.UNCLEAR_REQUEST
        confidence = float(parsed.get("confidence", 0.7))
        raw_items = []
        for item_dict in parsed.get("items") or []:
            if not isinstance(item_dict, dict):
                continue
            sanitized = dict(item_dict)
            if sanitized.get("notes") is None:
                sanitized["notes"] = ""
            if sanitized.get("modifier_ids") is None:
                sanitized["modifier_ids"] = []
            if sanitized.get("selected_options") is None:
                sanitized["selected_options"] = {}
            if sanitized.get("quantity") is None:
                sanitized["quantity"] = 1
            if sanitized.get("raw_mention") is None:
                sanitized["raw_mention"] = clean_text
            raw_items.append(ExtractedOrderItem(**sanitized))
        match_meta = parsed.get("match_meta", [])

        # If customer is correcting a previously added item (e.g., "لأ خليها بيبروني"), replace last cart item
        if parsed.get("is_correction") and raw_items and raw_items[0].product_id and session.cart.items:
            last_id = session.cart.items[-1].cart_item_id
            rem_rec, _ = await self.tools.execute(session, "remove_cart_item", {"cart_item_id": last_id})
            tool_calls.append(rem_rec)

        # 5. Dispatch by intent with strict deterministic business safety
        if intent == CustomerIntent.GREETING:
            reply = "أهلاً بحضرتك في مطعم MOKA! نورتنا، تحب تطلب إيه النهاردة؟ (كريب، بيتزا، برجر، باستا، وجبات، أو مشروبات وحلويات)"
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.MENU_INQUIRY:
            rec, items = await self.tools.execute(session, "get_active_menu", {})
            tool_calls.append(rec)
            reply = (
                f"عندنا {len(items)} صنف متاح في منيو MOKA: كريب وكريب سبيشال، وجبات، باستا، بيتزا وساندوتش بيتزا، "
                "برجر لحم وفراخ، مقبلات، ومشروبات وحلويات الكافيه. تحب تشوف أسعار قسم معين؟"
            )
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.PRICE_INQUIRY:
            return await self._handle_price_inquiry(session, raw_items, tool_calls, model_used)

        if intent == CustomerIntent.DELIVERY_INQUIRY:
            zone = self._extract_zone(clean_text) or "default"
            rec, res = await self.tools.execute(session, "resolve_delivery", {"zone_name": zone})
            tool_calls.append(rec)
            reply = res.message if not isinstance(res, BackendIntegrationError) else res.message
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent in (CustomerIntent.REQUEST_HUMAN, CustomerIntent.COMPLAINT):
            rec, _ = await self.tools.execute(
                session,
                "request_human_assistance",
                {"reason": clean_text, "urgency": "high" if intent == CustomerIntent.COMPLAINT else "normal"},
            )
            tool_calls.append(rec)
            reply = "حاضر يا فندم، تم تحويل المحادثة فوراً للكاشير وهيتواصل مع حضرتك حالاً."
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.CANCEL_ORDER:
            await self.backend.clear_cart(session.phone_number)
            session.cart = await self.backend.get_cart_summary(session.phone_number)
            session.pending_clarification = None
            session.awaiting_final_confirmation = False
            reply = "تم إلغاء السلة الحالية بنجاح. لو حبيت تطلب تاني في أي وقت إحنا موجودين!"
            return self._finish(session, intent, confidence, [], False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.REMOVE_ITEM:
            return await self._handle_remove_item(session, raw_items, tool_calls, model_used)

        if intent == CustomerIntent.CONFIRM_ORDER:
            return await self._handle_confirm_order(session, tool_calls, model_used)

        if intent in (CustomerIntent.ADD_TO_CART, CustomerIntent.MODIFY_ORDER) and raw_items:
            return await self._handle_add_items(session, intent, confidence, raw_items, tool_calls, model_used, match_meta=match_meta)

        # Fallback: if customer just provided details while awaiting confirmation
        if session.cart.items and (session.customer_name or session.delivery_address or session.backup_phone):
            return await self._handle_confirm_order(session, tool_calls, model_used, explicit_confirm=False)

        reply = "ممكن توضحيلي اسم الصنف اللي تحبيه أكتر؟ عندنا في MOKA كريب، بيتزا، برجر، باستا، وجبات، ومشروبات وحلويات."
        return self._finish(session, CustomerIntent.UNCLEAR_REQUEST, confidence, [], True, "unclear_item", tool_calls, reply, model_used)

    async def _run_nlu(self, text: str, category_hint: str | None = None) -> tuple[dict[str, Any], str]:
        if self.llm_client.is_configured:
            allowed_intents = ", ".join(i.value for i in CustomerIntent)
            system_prompt = (
                "You are Wasla AI NLU for MOKA Egyptian restaurant. Return ONLY valid JSON with keys: "
                f"intent (MUST be one of: {allowed_intents}), confidence (0..1), "
                "items (list of {product_id, raw_mention, quantity, size_code, modifier_ids, selected_options, notes}), "
                "requires_clarification, clarification_reason. Never invent product_ids or prices.\n"
                + self.menu_index.build_compact_prompt_catalog()
            )
            try:
                data, model = await self.llm_client.generate_json(system_prompt, text, use_fallback_model=False)
                if float(data.get("confidence", 0)) < 0.72:
                    data, model = await self.llm_client.generate_json(system_prompt, text, use_fallback_model=True)
                return data, model
            except Exception as exc:
                logger.warning("Live Gemini call failed (%s); falling back to OfflineEgyptianNLU.", exc)
        return self.offline_nlu.parse(text, category_hint=category_hint), "offline_egyptian_nlu"

    async def _handle_price_inquiry(
        self,
        session: SessionState,
        items: list[ExtractedOrderItem],
        tool_calls: list[ToolCallRecord],
        model_used: str,
    ) -> DispatcherResponse:
        if not items or not items[0].product_id:
            reply = "تحب تعرف سعر أنهي صنف بالظبط من منيو MOKA؟"
            return self._finish(session, CustomerIntent.PRICE_INQUIRY, 0.85, items, True, "missing_product", tool_calls, reply, model_used)

        prod = self.menu_index.get_product(items[0].product_id)
        if not prod:
            reply = "الصنف ده مش موجود في منيو MOKA حالياً."
            return self._finish(session, CustomerIntent.PRICE_INQUIRY, 0.85, items, True, "unknown_product", tool_calls, reply, model_used)

        rec, _ = await self.tools.execute(session, "check_item_availability", {"product_id": prod.product_id})
        tool_calls.append(rec)
        if prod.sizes:
            prices = " | ".join(f"{s.name_ar}: {s.price} ج.م" for s in prod.sizes)
            reply = f"سعر {prod.name_ar}: {prices}. تحب أضيفه للسلة؟"
        else:
            reply = f"سعر {prod.name_ar}: {prod.base_price} ج.م. تحب أضيفه للسلة؟"
        return self._finish(session, CustomerIntent.PRICE_INQUIRY, 0.95, items, False, None, tool_calls, reply, model_used)

    async def _handle_add_items(
        self,
        session: SessionState,
        intent: CustomerIntent,
        confidence: float,
        items: list[ExtractedOrderItem],
        tool_calls: list[ToolCallRecord],
        model_used: str,
        match_meta: list[dict[str, Any]] | None = None,
    ) -> DispatcherResponse:
        added_names: list[str] = []
        meta_list = match_meta or []
        for idx, item in enumerate(items):
            item_meta = meta_list[idx] if idx < len(meta_list) else {}
            if not item.product_id:
                res = self.menu_index.resolve_mention_with_confidence(item.raw_mention)
                candidates = res.candidates
                if candidates:
                    opts = " / ".join(c.name_ar for c in candidates[:5])
                    cat_hint = item_meta.get("category_hint") or res.category_hint
                    session.pending_clarification = {
                        "type": "ambiguous_product",
                        "item": item.model_dump(),
                        "candidates": [c.product_id for c in candidates[:5]],
                        "category_hint": cat_hint,
                    }
                    if res.tier == "medium":
                        reply = f"تقصد أي واحد من دول في منيو MOKA: {opts}؟"
                    else:
                        reply = f"عندنا أكتر من اختيار لـ '{item.raw_mention}': {opts}. تحب أنهي واحد فيهم؟"
                    return self._finish(session, intent, confidence, items, True, "ambiguous_product", tool_calls, reply, model_used)
                reply = f"للأسف '{item.raw_mention}' مش موجود في منيو MOKA حالياً. تحب تطلب حاجة تانية من البيتزا أو الكريب أو البرجر أو الحلويات؟"
                return self._finish(session, intent, confidence, items, True, "unknown_product", tool_calls, reply, model_used)

            rec, res = await self.tools.execute(
                session,
                "modify_cart",
                {
                    "product_id": item.product_id,
                    "quantity": item.quantity,
                    "size_code": item.size_code,
                    "modifier_ids": item.modifier_ids,
                    "selected_options": item.selected_options,
                    "notes": item.notes,
                },
            )
            tool_calls.append(rec)
            if isinstance(res, BackendIntegrationError):
                if res.code in ("MISSING_OR_INVALID_SIZE", "MISSING_REQUIRED_OPTION"):
                    session.pending_clarification = {"type": res.code, "item": item.model_dump()}
                    return self._finish(session, intent, confidence, items, True, res.code, tool_calls, res.message, model_used)
                return self._finish(session, intent, confidence, items, True, res.code, tool_calls, res.message, model_used)
            prod = self.menu_index.get_product(item.product_id)
            if prod:
                added_names.append(f"{item.quantity} × {prod.name_ar}")

        summary_rec, cart = await self.tools.execute(session, "get_cart_summary", {})
        tool_calls.append(summary_rec)
        reply = (
            f"تمام! تم إضافة ({'، '.join(added_names)}) للسلة. "
            f"إجمالي السلة حالياً {cart.subtotal} ج.م (قبل التوصيل). تحب تزود حاجة تانية زي بطاطس أو مشروب، ولا نأكد الأوردر؟"
        )
        return self._finish(session, intent, confidence, items, False, None, tool_calls, reply, model_used)

    async def _handle_remove_item(
        self,
        session: SessionState,
        items: list[ExtractedOrderItem],
        tool_calls: list[ToolCallRecord],
        model_used: str,
    ) -> DispatcherResponse:
        if not session.cart.items:
            return self._finish(session, CustomerIntent.REMOVE_ITEM, 0.9, items, False, None, tool_calls, "السلة فاضية حالياً يا فندم.", model_used)
        target_pid = items[0].product_id if items and items[0].product_id else session.cart.items[-1].product_id
        rec, res = await self.tools.execute(session, "remove_cart_item", {"cart_item_id": target_pid})
        tool_calls.append(rec)
        if isinstance(res, BackendIntegrationError):
            return self._finish(session, CustomerIntent.REMOVE_ITEM, 0.8, items, True, res.code, tool_calls, res.message, model_used)
        reply = f"تم حذف الصنف من السلة. إجمالي السلة دلوقتي {session.cart.subtotal} ج.م."
        return self._finish(session, CustomerIntent.REMOVE_ITEM, 0.92, items, False, None, tool_calls, reply, model_used)

    async def _handle_confirm_order(
        self,
        session: SessionState,
        tool_calls: list[ToolCallRecord],
        model_used: str,
        explicit_confirm: bool = True,
    ) -> DispatcherResponse:
        if not session.cart.items:
            return self._finish(session, CustomerIntent.CONFIRM_ORDER, 0.9, [], True, "empty_cart", tool_calls, "السلة فاضية يا فندم، اختار الأصناف الأول عشان نأكد الأوردر.", model_used)

        missing = []
        if not session.customer_name:
            missing.append("الاسم")
        if not session.delivery_address:
            missing.append("عنوان التوصيل بالتفصيل")
        if not session.backup_phone:
            missing.append("رقم تليفون بديل")

        if missing:
            session.awaiting_final_confirmation = True
            reply = (
                f"عشان نثبت الأوردر (إجمالي {session.cart.total_amount} ج.م)، محتاجين منك بس: {' و'.join(missing)}."
            )
            return self._finish(session, CustomerIntent.CONFIRM_ORDER, 0.92, [], True, "missing_order_details", tool_calls, reply, model_used)

        if not explicit_confirm:
            session.awaiting_final_confirmation = True
            reply = (
                f"ملخص طلبك: {len(session.cart.items)} صنف، الإجمالي شامل التوصيل {session.cart.total_amount} ج.م "
                f"على عنوان ({session.delivery_address}). أؤكد وأبعت الأوردر للمطبخ؟"
            )
            return self._finish(session, CustomerIntent.PROVIDE_DETAILS, 0.92, [], True, "awaiting_explicit_confirm", tool_calls, reply, model_used)

        rec, res = await self.tools.execute(
            session,
            "confirm_and_lock_order",
            {
                "customer_name": session.customer_name,
                "backup_phone": session.backup_phone,
                "delivery_address": session.delivery_address,
                "confirmed_by_customer": True,
                "data_source": getattr(self, "_current_source", "api"),
                "is_test": getattr(self, "_current_is_test", False),
            },
        )
        tool_calls.append(rec)
        if isinstance(res, BackendIntegrationError):
            return self._finish(session, CustomerIntent.CONFIRM_ORDER, 0.9, [], True, res.code, tool_calls, res.message, model_used)

        order_id = res.get("id")
        reply = f"تم تأكيد طلبك بنجاح! رقم الأوردر #{order_id} واتبعت فوراً للمطبخ والكاشير. بألف هنا!"
        return self._finish(session, CustomerIntent.CONFIRM_ORDER, 0.98, [], False, None, tool_calls, reply, model_used, order_id=order_id)

    async def _try_resolve_clarification(
        self,
        session: SessionState,
        text: str,
        tool_calls: list[ToolCallRecord],
    ) -> DispatcherResponse | None:
        pending = session.pending_clarification
        if not pending:
            return None
        ptype = pending["type"]
        item = ExtractedOrderItem(**pending["item"])
        norm = normalize_ar(text)

        if ptype == "MISSING_OR_INVALID_SIZE":
            size = self.offline_nlu._extract_size(norm)
            if size:
                item.size_code = size
                session.pending_clarification = None
                return await self._handle_add_items(session, CustomerIntent.ADD_TO_CART, 0.95, [item], tool_calls, "clarification_resolver")

        if ptype == "MISSING_REQUIRED_OPTION":
            if "رانش" in norm:
                item.selected_options["sauce"] = "رانش"
            elif "باربكيو" in norm or "باربيكيو" in norm:
                item.selected_options["sauce"] = "باربيكيو"
            for flavor in ["فراوله", "صن شاين", "كيوي", "مانجو", "توت", "خوخ"]:
                if flavor in norm:
                    item.selected_options["flavor"] = flavor
            if item.selected_options:
                session.pending_clarification = None
                return await self._handle_add_items(session, CustomerIntent.ADD_TO_CART, 0.95, [item], tool_calls, "clarification_resolver")

        if ptype == "ambiguous_product":
            cat_hint = pending.get("category_hint")
            res = self.menu_index.resolve_mention_with_confidence(text, category_hint=cat_hint)
            if res.tier == "high" and res.product is not None:
                item.product_id = res.product.product_id
                size_in_reply = self.offline_nlu._extract_size(norm)
                if size_in_reply and not item.size_code:
                    item.size_code = size_in_reply
                opts_in_reply = self.offline_nlu._extract_options(res.product.product_id, norm)
                if opts_in_reply:
                    item.selected_options.update(opts_in_reply)
                notes_in_reply = self.offline_nlu._extract_notes(text)
                if notes_in_reply:
                    item.notes = notes_in_reply
                session.pending_clarification = None
                return await self._handle_add_items(session, CustomerIntent.ADD_TO_CART, res.confidence, [item], tool_calls, "clarification_resolver")

            for cid in pending.get("candidates", []):
                prod = self.menu_index.get_product(cid)
                if prod and (normalize_ar(prod.name_ar) in norm or normalize_ar(prod.category_ar) in norm):
                    item.product_id = prod.product_id
                    session.pending_clarification = None
                    return await self._handle_add_items(session, CustomerIntent.ADD_TO_CART, 0.94, [item], tool_calls, "clarification_resolver")
        return None

    def _capture_customer_details(self, session: SessionState, text: str) -> None:
        name_match = re.search(r"(?:الاسم|اسمي)\s*[:\-]?\s*([^\n،,]+)", text)
        if name_match:
            session.customer_name = name_match.group(1).strip()
        addr_match = re.search(r"(?:العنوان|عنواني)\s*[:\-]?\s*([^\n،,]+)", text)
        if addr_match:
            session.delivery_address = addr_match.group(1).strip()
        phone_match = re.search(r"(?:رقم\s*بديل|البديل|تليفون\s*بديل)\s*[:\-]?\s*(01[0125]\d{8})", text)
        if phone_match:
            session.backup_phone = phone_match.group(1).strip()

    def _extract_zone(self, text: str) -> str | None:
        for z in ["المعادي", "مدينة نصر", "التجمع", "الدقي", "المهندسين", "free_zone"]:
            if z in text:
                return z
        return None

    def _finish(
        self,
        session: SessionState,
        intent: CustomerIntent,
        confidence: float,
        items: list[ExtractedOrderItem],
        requires_clarification: bool,
        clarification_reason: str | None,
        tool_calls: list[ToolCallRecord],
        reply: str,
        model_used: str,
        order_id: int | None = None,
    ) -> DispatcherResponse:
        session.history.append(ConversationMessage(role="assistant", content=reply))
        return DispatcherResponse(
            intent=intent,
            confidence=confidence,
            extracted_items=items,
            requires_clarification=requires_clarification,
            clarification_reason=clarification_reason,
            tool_calls=tool_calls,
            cart=session.cart,
            order_id=order_id or session.active_order_id,
            needs_human_takeover=session.needs_human_takeover,
            reply_ar=reply,
            model_used=model_used,
        )
