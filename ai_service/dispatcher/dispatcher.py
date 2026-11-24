"""Core Wasla AI Dispatcher — Hybrid CAMeLBERT LoRA + Gemini Function Calling + Deterministic Safety."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import logging
import re
from typing import Any
import uuid

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.audio.pipeline import HybridAudioPipeline
from ai_service.complaints.detector import ComplaintAssessment, assess_complaint
from ai_service.config import WaslaConfig
from ai_service.dispatcher import persona
from ai_service.dispatcher.cashier_directives import CashierDirectiveInterpreter
from ai_service.models.finetuned_intent import FineTunedIntentClassifier
from ai_service.models.llm_client import GeminiLLMClient, OfflineEgyptianNLU
from ai_service.recommendation.engine import RecommendationEngine
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
    TurnExecutionContext,
)
from ai_service.security.guardrails import detect_prompt_injection, redact_pii
from ai_service.tools.registry import ToolExecutor
from ai_service.understanding.arabic_nlp import is_gibberish_input, strip_fillers
from ai_service.understanding.menu_index import MenuIndex, normalize_ar

logger = logging.getLogger("wasla.dispatcher")

_UPSELL_DECLINE_PATTERNS = (
    "لا شكرا",
    "لأ شكرا",
    "لا مش عايز",
    "لأ مش عايز",
    "لا مش عايزه",
    "لأ مش عايزه",
    "مش عايز",
    "مش عايزه",
    "مش محتاج",
    "مش محتاجه",
    "كفايه",
    "كفاية",
    "بس كده",
    "لا بس",
    "لأ بس",
    "بلاش",
    "البرجر بس",
    "البيتزا بس",
    "الكريب بس",
    "السندوتش بس",
    "هو بس",
    "ده بس",
    "بلاش اضافات",
    "من غير اضافات",
    "بدون اضافات",
)

_UPSELL_ACCEPT_PATTERNS = (
    "اه ضيف",
    "أيوه ضيف",
    "ايوه ضيف",
    "تمام ضيف",
    "ماشي ضيف",
    "اه زود",
    "ايوه زود",
    "تمام زود",
    "ماشي زود",
    "اه هات",
    "ايوه هات",
    "ضيفهم",
    "زودهم",
    "اه موافق",
    "تمام موافق",
    "اه ماشي",
    "ايوه ماشي",
    "موافق ضيف",
)

_NON_DESTRUCTIVE_LORA_INTENTS = {
    "greeting",
    "menu_inquiry",
    "complaint",
    "request_human",
}
_HIGH_CONF_LORA_INTENTS = {
    "delivery_inquiry",
    "price_inquiry",
    "unclear_request",
}

_OFF_MENU_DISH_WORDS = (
    "سوشي",
    "كشري",
    "فته",
    "كوارع",
    "محشي",
    "ملوخيه",
    "حواوشي",
    "سمك",
    "جمبري",
    "فطير",
    "شوربه",
    "كباب",
    "كفته",
    "شاورما",
    "مشويات",
    "sushi",
    "koshary",
)

_CONFIRM_PHRASES = ("اكد", "تمام", "موافق", "نفذ", "اعتمد", "ابعت الاوردر")


class WaslaAIDispatcher:
    """Thread-safe, restart-persistent Hybrid AI Dispatcher for Wasla."""

    def __init__(
        self,
        config: WaslaConfig,
        menu_index: MenuIndex,
        backend: Any,
        llm_client: GeminiLLMClient | None = None,
        audio_pipeline: HybridAudioPipeline | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        finetuned_intent: FineTunedIntentClassifier | None = None,
    ):
        self.config = config
        self.menu_index = menu_index
        self.backend = backend
        self.llm_client = llm_client or GeminiLLMClient(config)
        self.offline_nlu = OfflineEgyptianNLU(menu_index)
        self.finetuned_intent = finetuned_intent or FineTunedIntentClassifier(
            checkpoint_dir=config.lora_checkpoint_dir,
            enabled=config.lora_enabled,
            prefer_gpu=config.lora_prefer_gpu,
        )
        self.audio_pipeline = audio_pipeline or HybridAudioPipeline(config)
        self.recommendation_engine = recommendation_engine or RecommendationEngine(
            menu_index, enabled=config.upsell_enabled
        )
        self.tools = ToolExecutor(backend)
        self.directive_interpreter = CashierDirectiveInterpreter(menu_index)
        self._sessions: dict[str, SessionState] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._last_active_route: str = "gemini+lora+menu_index" if self.llm_client.is_configured else "lora+egyptian_nlu"
        self._telemetry: dict[str, int] = {
            "turns_total": 0,
            "lora_calls": 0,
            "gemini_primary_calls": 0,
            "gemini_fallback_calls": 0,
            "local_hybrid_calls": 0,
        }

    def _get_lock(self, phone_number: str) -> asyncio.Lock:
        lock = self._session_locks.get(phone_number)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[phone_number] = lock
        return lock

    def get_session(self, phone_number: str) -> SessionState:
        is_mock = bool(getattr(self.backend, "IS_MOCK", True))
        if hasattr(self.backend, "load_session_state"):
            if (not is_mock) or (phone_number not in self._sessions):
                restored = self.backend.load_session_state(phone_number, ttl_seconds=self.config.session_ttl_seconds)
                if restored is not None:
                    self._sessions[phone_number] = restored
                    return restored

        existing = self._sessions.get(phone_number)
        if existing is not None:
            now = datetime.now(timezone.utc)
            updated = existing.updated_at if existing.updated_at.tzinfo else existing.updated_at.replace(tzinfo=timezone.utc)
            if (
                self.config.session_ttl_seconds > 0
                and (now - updated) > timedelta(seconds=self.config.session_ttl_seconds)
                and not existing.needs_human_takeover
            ):
                fresh = SessionState(
                    phone_number=phone_number,
                    customer_name=existing.customer_name,
                    backup_phone=existing.backup_phone,
                    delivery_address=existing.delivery_address,
                    session_epoch=existing.session_epoch + 1,
                    updated_at=now,
                )
                self._sessions[phone_number] = fresh
                if hasattr(self.backend, "save_session_state"):
                    self.backend.save_session_state(fresh)
                return fresh
            return existing

        fresh = SessionState(phone_number=phone_number)
        self._sessions[phone_number] = fresh
        return fresh

    def _persist_session(self, session: SessionState) -> None:
        session.updated_at = datetime.now(timezone.utc)
        self._sessions[session.phone_number] = session
        if hasattr(self.backend, "save_session_state"):
            try:
                self.backend.save_session_state(session)
            except Exception as exc:
                logger.warning("Failed to persist session state for %s: %s", redact_pii(session.phone_number), exc)

    def reset_session(self, phone_number: str) -> None:
        self._sessions.pop(phone_number, None)
        if hasattr(self.backend, "delete_session_state"):
            self.backend.delete_session_state(phone_number)

    def set_human_takeover(
        self,
        phone_number: str,
        active: bool,
        actor: str | None = None,
        *,
        persist: bool = True,
    ) -> SessionState:
        session = self.get_session(phone_number)
        session.needs_human_takeover = active
        session.human_takeover_by = actor if active else None
        if persist:
            self._persist_session(session)
        return session

    def get_runtime_status(self) -> dict[str, Any]:
        lora_status = self.finetuned_intent.get_status()
        return {
            **lora_status,
            "active_inference_route": self._last_active_route,
            "telemetry": dict(self._telemetry),
        }

    def interpret_cashier_directive(self, req: CashierDirectiveRequest) -> CashierDirectiveResponse:
        return self.directive_interpreter.interpret(req)

    async def handle_audio(
        self,
        phone_number: str,
        audio: AudioInput,
        data_source: str = "api",
        is_test: bool = False,
    ) -> DispatcherResponse:
        transcription = await self.audio_pipeline.process(audio)
        response = await self.handle_text(
            phone_number,
            transcription.transcript,
            input_type="audio",
            data_source=data_source,
            is_test=is_test,
        )
        return response.model_copy(update={"transcript": transcription.transcript})

    async def handle_text(
        self,
        phone_number: str,
        text: str,
        input_type: str = "text",
        data_source: str = "api",
        is_test: bool = False,
    ) -> DispatcherResponse:
        async with self._get_lock(phone_number):
            ctx = TurnExecutionContext(
                data_source=data_source,
                is_test=is_test,
                request_id=uuid.uuid4().hex[:12],
            )
            return await self._handle_text_locked(phone_number, text, input_type=input_type, ctx=ctx)

    async def _handle_text_locked(
        self,
        phone_number: str,
        text: str,
        *,
        input_type: str,
        ctx: TurnExecutionContext,
    ) -> DispatcherResponse:
        self._telemetry["turns_total"] += 1
        if hasattr(self.backend, "sync_menu_sync"):
            self.backend.sync_menu_sync()

        session = self.get_session(phone_number)
        clean_text = text.strip()
        if self.config.redact_pii_in_logs:
            logger.info("Incoming turn [%s]: %s", redact_pii(phone_number), redact_pii(clean_text))

        session.history.append(ConversationMessage(role="customer", content=clean_text, input_type=input_type))
        tool_calls: list[ToolCallRecord] = []
        executed_signatures: set[str] = set()

        # 0. Human Takeover Hard Gate: stop automated AI replies once staff/owner takes over!
        if session.needs_human_takeover:
            complaint_check = assess_complaint(clean_text)
            complaint_dict = (
                {
                    "category": complaint_check.category,
                    "severity": complaint_check.severity,
                    "requires_owner_escalation": complaint_check.requires_owner_escalation,
                    "summary_en": complaint_check.summary_en,
                    "matched_signals": complaint_check.matched_signals,
                    "critical_update": complaint_check.severity == "CRITICAL",
                }
                if complaint_check.is_complaint
                else None
            )
            self._persist_session(session)
            return DispatcherResponse(
                intent=CustomerIntent.COMPLAINT if complaint_check.is_complaint else CustomerIntent.REQUEST_HUMAN,
                confidence=1.0,
                cart=session.cart,
                order_id=session.active_order_id,
                needs_human_takeover=True,
                suppress_ai_reply=True,
                complaint_assessment=complaint_dict,
                reply_ar="",
                model_used="human_takeover_active",
            )

        # 1. Security guardrail: prompt injection / fake discount rejection
        if detect_prompt_injection(clean_text):
            reply = "أهلاً بيك في MOKA ❤️ الأسعار والخصومات معتمدة من سيستم المطعم بس. تحب تطلب إيه من المنيو؟"
            session.history.append(ConversationMessage(role="assistant", content=reply))
            session.last_intent = CustomerIntent.UNCLEAR_REQUEST.value
            self._persist_session(session)
            return DispatcherResponse(
                intent=CustomerIntent.UNCLEAR_REQUEST,
                confidence=1.0,
                requires_clarification=True,
                clarification_reason="blocked_prompt_injection",
                cart=session.cart,
                reply_ar=reply,
                model_used="security_guardrail",
            )

        # 2. Extract customer details & budget preference if present
        self._capture_customer_details(session, clean_text)
        detected_budget = persona.extract_budget_limit(clean_text)
        if detected_budget is not None:
            session.budget_limit = detected_budget

        # 3. High-recall safety & complaint pre-check (NEVER gated behind high confidence for food safety!)
        pre_complaint = assess_complaint(clean_text)
        if pre_complaint.is_complaint and pre_complaint.severity in ("CRITICAL", "HIGH"):
            return await self._handle_complaint(
                session, clean_text, pre_complaint, tool_calls, "complaint_detector", executed_signatures=executed_signatures
            )

        # 4. Check if responding to a pending upsell recommendation (accept or decline)
        if session.pending_upsell:
            upsell_resp = await self._try_handle_pending_upsell(
                session, clean_text, tool_calls, executed_signatures=executed_signatures
            )
            if upsell_resp is not None:
                return upsell_resp

        # 5. Check if answering a pending clarification (size / option / ambiguous item)
        if session.pending_clarification:
            resolved = await self._try_resolve_clarification(
                session, clean_text, tool_calls, executed_signatures=executed_signatures
            )
            if resolved is not None:
                return resolved

        # 6. Run Hybrid NLU (CAMeLBERT LoRA + Gemini Function Calling + Deterministic MenuIndex)
        category_hint = (session.pending_clarification or {}).get("category_hint")
        parsed, model_used = await self._run_nlu(clean_text, category_hint=category_hint, history=session.history)
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
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.7))))

        raw_items: list[ExtractedOrderItem] = []
        for item_dict in parsed.get("items") or []:
            if not isinstance(item_dict, dict):
                continue
            sanitized = self._sanitize_item_dict(item_dict, fallback_mention=clean_text)
            raw_items.append(ExtractedOrderItem(**sanitized))
        match_meta = parsed.get("match_meta", [])

        # Execute validated non-destructive Gemini function calls (menu / availability / delivery) cleanly via ToolExecutor
        gemini_fc_list = parsed.get("function_calls") or []
        pre_avail_entries: list[tuple[Any, bool, str | None]] = []
        for fc in gemini_fc_list:
            if not isinstance(fc, dict):
                continue
            fc_name = fc.get("name")
            fc_args = fc.get("args") or {}
            if fc_name == "check_item_availability" and fc_args.get("product_id"):
                pid = str(fc_args["product_id"])
                if self.menu_index.get_product(pid):
                    rec, avail_res = await self.tools.execute(
                        session, "check_item_availability", {"product_id": pid}, executed_signatures=executed_signatures
                    )
                    tool_calls.append(rec)
                    if isinstance(avail_res, dict) and avail_res.get("item") is not None:
                        pre_avail_entries.append((avail_res["item"], bool(avail_res["available"]), None))
            elif fc_name == "get_active_menu":
                rec, _ = await self.tools.execute(
                    session, "get_active_menu", {"category": fc_args.get("category")} if fc_args.get("category") else {}, executed_signatures=executed_signatures
                )
                tool_calls.append(rec)

        # Low-confidence safety gate: never allow low-confidence predictions (<0.55) to create or confirm orders
        if confidence < 0.55 and intent == CustomerIntent.CONFIRM_ORDER:
            intent = CustomerIntent.UNCLEAR_REQUEST

        # Elliptical self-correction (e.g., "لأ، خليها كبيرة" without repeating item name)
        if (
            parsed.get("is_correction")
            and parsed.get("correction_patch")
            and not raw_items
            and session.cart.items
            and not session.order_confirmed
        ):
            patch = parsed["correction_patch"]
            last_line = session.cart.items[-1]
            corrected_item = ExtractedOrderItem(
                product_id=last_line.product_id,
                raw_mention=last_line.name_ar,
                quantity=patch.get("quantity") or last_line.quantity,
                size_code=patch.get("size_code") or last_line.size_code,
                modifier_ids=patch.get("modifier_ids") or [m["modifier_id"] for m in last_line.modifiers if m.get("modifier_id")],
                selected_options=last_line.selected_options,
                notes=patch.get("notes") or last_line.notes,
            )
            return await self._handle_add_items(
                session,
                CustomerIntent.MODIFY_ORDER,
                confidence,
                [corrected_item],
                tool_calls,
                model_used,
                replace_cart_item_id=last_line.cart_item_id,
                executed_signatures=executed_signatures,
            )

        # 7. Dispatch by intent with strict deterministic business safety
        if intent == CustomerIntent.COMPLAINT or pre_complaint.is_complaint:
            assessment = pre_complaint if pre_complaint.is_complaint else assess_complaint(
                clean_text, predicted_intent="complaint"
            )
            return await self._handle_complaint(
                session, clean_text, assessment, tool_calls, model_used, executed_signatures=executed_signatures
            )

        if intent == CustomerIntent.GREETING:
            reply = persona.render_greeting(len(session.history))
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.MENU_INQUIRY:
            req_cat = self._extract_requested_category(clean_text)
            rec, items = await self.tools.execute(
                session, "get_active_menu", {"category": req_cat} if req_cat else {}, executed_signatures=executed_signatures
            )
            tool_calls.append(rec)
            reply = persona.render_menu_overview(items, requested_category=req_cat)
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.PRICE_INQUIRY:
            return await self._handle_price_inquiry(
                session,
                clean_text,
                raw_items,
                tool_calls,
                model_used,
                pre_avail_entries=pre_avail_entries,
                executed_signatures=executed_signatures,
            )

        if intent == CustomerIntent.DELIVERY_INQUIRY:
            zone = self._extract_zone(clean_text) or "default"
            rec, res = await self.tools.execute(
                session, "resolve_delivery", {"zone_name": zone}, executed_signatures=executed_signatures
            )
            tool_calls.append(rec)
            reply = res.message if not isinstance(res, BackendIntegrationError) else res.message
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.REQUEST_HUMAN:
            rec, _ = await self.tools.execute(
                session,
                "request_human_assistance",
                {"reason": clean_text, "urgency": "normal"},
                executed_signatures=executed_signatures,
            )
            tool_calls.append(rec)
            session.pending_upsell = None
            reply = persona.render_human_handoff()
            return self._finish(session, intent, confidence, raw_items, False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.CANCEL_ORDER:
            await self.backend.clear_cart(session.phone_number)
            session.cart = await self.backend.get_cart_summary(session.phone_number)
            session.pending_clarification = None
            session.pending_upsell = None
            session.awaiting_final_confirmation = False
            reply = persona.render_order_cancelled()
            return self._finish(session, intent, confidence, [], False, None, tool_calls, reply, model_used)

        if intent == CustomerIntent.REMOVE_ITEM:
            session.pending_upsell = None
            return await self._handle_remove_item(
                session, raw_items, tool_calls, model_used, executed_signatures=executed_signatures
            )

        if intent == CustomerIntent.CONFIRM_ORDER:
            session.pending_upsell = None
            if raw_items and any(i.product_id for i in raw_items):
                add_resp = await self._handle_add_items(
                    session,
                    CustomerIntent.ADD_TO_CART,
                    confidence,
                    [i for i in raw_items if i.product_id],
                    tool_calls,
                    model_used,
                    match_meta=match_meta,
                    executed_signatures=executed_signatures,
                )
                if add_resp.requires_clarification:
                    return add_resp
            return await self._handle_confirm_order(
                session, tool_calls, model_used, ctx=ctx, executed_signatures=executed_signatures
            )

        nat_reply = str(parsed.get("natural_reply_ar") or "").strip()
        last_bot_msg = next((m.content for m in reversed(session.history) if m.role == "assistant"), "")
        safe_nat_reply = (
            nat_reply
            if (len(nat_reply) >= 8 and not re.search(r"\d", nat_reply) and nat_reply != last_bot_msg)
            else ""
        )

        if intent in (CustomerIntent.ADD_TO_CART, CustomerIntent.MODIFY_ORDER) and raw_items:
            return await self._handle_add_items(
                session,
                intent,
                confidence,
                raw_items,
                tool_calls,
                model_used,
                match_meta=match_meta,
                natural_reply_ar=safe_nat_reply,
                replace_cart_item_id=(
                    session.cart.items[-1].cart_item_id
                    if parsed.get("is_correction") and session.cart.items and not session.order_confirmed
                    else None
                ),
                executed_signatures=executed_signatures,
            )

        if intent == CustomerIntent.MODIFY_ORDER and not raw_items:
            reply = safe_nat_reply or "تمام يا فندم، تحب تعدل إيه بالظبط في طلبك؟ 😊"
            return self._finish(
                session, CustomerIntent.MODIFY_ORDER, confidence, [], True, "clarify_order_action", tool_calls, reply, model_used
            )

        # Fallback: if customer just provided details while awaiting confirmation (remembering previous fields)
        if session.cart.items and (session.customer_name or session.delivery_address or session.backup_phone) and session.awaiting_final_confirmation:
            return await self._handle_confirm_order(
                session,
                tool_calls,
                model_used,
                ctx=ctx,
                explicit_confirm=False,
                executed_signatures=executed_signatures,
            )

        # Reject gibberish / unintelligible noise cleanly without pretending it's a dish
        if is_gibberish_input(clean_text):
            reply = persona.render_unclear_request(turn_count=len(session.history))
            return self._finish(
                session, CustomerIntent.UNCLEAR_REQUEST, confidence, [], True, "unclear_item", tool_calls, reply, model_used
            )

        # Detect off-menu dish requests ONLY when intent is unclear/add_to_cart and no matched items
        cleaned_mention = strip_fillers(clean_text)
        if intent == CustomerIntent.ADD_TO_CART and cleaned_mention and 1 <= len(cleaned_mention.split()) <= 3:
            check_res = self.menu_index.resolve_mention_with_confidence(cleaned_mention)
            if not check_res.candidates and not check_res.product:
                unknown_item = ExtractedOrderItem(product_id=None, raw_mention=cleaned_mention, quantity=1)
                reply = safe_nat_reply or persona.render_unknown_product(cleaned_mention, turn_count=len(session.history))
                return self._finish(
                    session,
                    CustomerIntent.ADD_TO_CART,
                    confidence,
                    [unknown_item],
                    True,
                    "unknown_product",
                    tool_calls,
                    reply,
                    model_used,
                )

        reply = safe_nat_reply or persona.render_unclear_request(turn_count=len(session.history))
        return self._finish(
            session, CustomerIntent.UNCLEAR_REQUEST, confidence, [], True, "unclear_item", tool_calls, reply, model_used
        )

    def _sanitize_item_dict(self, item_dict: dict[str, Any], *, fallback_mention: str) -> dict[str, Any]:
        sanitized = dict(item_dict)
        # Drop any unauthorized or price-related keys if hallucinated by LLM
        for forbidden in ("price", "unit_price", "line_total", "total", "discount"):
            sanitized.pop(forbidden, None)
        if not isinstance(sanitized.get("notes"), str):
            sanitized["notes"] = ""
        if not isinstance(sanitized.get("modifier_ids"), list):
            sanitized["modifier_ids"] = []
        else:
            sanitized["modifier_ids"] = [
                str(m) for m in sanitized["modifier_ids"] if m is not None and self.menu_index.get_modifier(str(m)) is not None
            ]
        if not isinstance(sanitized.get("selected_options"), dict):
            sanitized["selected_options"] = {}
        try:
            qty = int(sanitized.get("quantity", 1))
        except (TypeError, ValueError):
            qty = 1
        sanitized["quantity"] = max(1, min(50, qty))
        if not isinstance(sanitized.get("raw_mention"), str) or not sanitized["raw_mention"].strip():
            sanitized["raw_mention"] = fallback_mention

        # Validate product_id strictly against authoritative MenuIndex (never trust hallucinated IDs)
        pid = sanitized.get("product_id")
        if pid is not None:
            pid_str = str(pid).strip()
            if self.menu_index.get_product(pid_str) is None:
                res = self.menu_index.resolve_mention_with_confidence(sanitized["raw_mention"])
                sanitized["product_id"] = res.product.product_id if (res.tier == "high" and res.product) else None
            else:
                sanitized["product_id"] = pid_str
        return sanitized

    async def _handle_complaint(
        self,
        session: SessionState,
        clean_text: str,
        assessment: ComplaintAssessment,
        tool_calls: list[ToolCallRecord],
        model_used: str,
        *,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse:
        session.pending_upsell = None
        rec, _ = await self.tools.execute(
            session,
            "request_human_assistance",
            {"reason": f"[{assessment.severity}:{assessment.category}] {clean_text}", "urgency": "high"},
            executed_signatures=executed_signatures,
        )
        tool_calls.append(rec)
        complaint_payload = {
            "category": assessment.category,
            "severity": assessment.severity,
            "requires_owner_escalation": assessment.requires_owner_escalation,
            "summary_en": assessment.summary_en,
            "matched_signals": assessment.matched_signals,
            "excerpt": redact_pii(clean_text)[:300],
        }
        return self._finish(
            session,
            CustomerIntent.COMPLAINT,
            0.96,
            [],
            False,
            None,
            tool_calls,
            assessment.customer_reply_ar,
            model_used,
            complaint_assessment=complaint_payload,
        )

    async def _try_handle_pending_upsell(
        self,
        session: SessionState,
        text: str,
        tool_calls: list[ToolCallRecord],
        *,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse | None:
        pending = session.pending_upsell
        if not pending:
            return None
        norm = normalize_ar(text)
        offered_items = pending.get("items", [])
        offered_ids = [i["product_id"] for i in offered_items]
        explicit_item_match = self.menu_index.resolve_mention_with_confidence(text)
        if (
            explicit_item_match.product
            and explicit_item_match.product.product_id not in offered_ids
        ) or (
            explicit_item_match.candidates
            and not any(candidate.product_id in offered_ids for candidate in explicit_item_match.candidates)
        ):
            session.pending_upsell = None
            return None

        if any(normalize_ar(p) in norm for p in _UPSELL_DECLINE_PATTERNS) or norm in ("لا", "لأ", "لا لا", "شكرا", "بس"):
            if any(w in norm for w in ("اكد", "الغ", "كنسل")):
                session.pending_upsell = None
                return None
            session.upsell_declined = True
            for pid in offered_ids:
                if pid not in session.rejected_upsell_ids:
                    session.rejected_upsell_ids.append(pid)
            session.pending_upsell = None
            reply = persona.render_upsell_declined(session.cart, turn_count=len(session.history))
            return self._finish(
                session,
                CustomerIntent.MODIFY_ORDER,
                0.95,
                [],
                False,
                None,
                tool_calls,
                reply,
                "upsell_handler",
                upsell_event={
                    "action": "rejected",
                    "offer_type": pending.get("offer_type"),
                    "offered_product_ids": offered_ids,
                },
            )

        accepted_items: list[dict[str, Any]] = []
        is_generic_accept = any(normalize_ar(p) in norm for p in _UPSELL_ACCEPT_PATTERNS) or norm in (
            "اه",
            "ايوه",
            "أيوه",
            "تمام ضيفهم",
            "ماشي",
        )
        for item in offered_items:
            item_norm = normalize_ar(item["name_ar"])
            if item_norm in norm:
                accepted_items.append(item)
            elif "بطاطس" in norm and "بطاطس" in item_norm and "شيدر" not in norm:
                accepted_items.append(item)
            elif "صودا" in norm and "صودا" in item_norm:
                accepted_items.append(item)

        if not accepted_items and is_generic_accept:
            if any(w in norm for w in ("ضيفهم", "زودهم", "الاتنين", "الاثنين", "كلهم", "كومبو")):
                accepted_items = offered_items
            elif offered_items:
                accepted_items = [offered_items[0]]

        if accepted_items:
            session.pending_upsell = None
            added_names: list[str] = []
            added_revenue = Decimal("0.00")
            extracted: list[ExtractedOrderItem] = []
            for item in accepted_items:
                rec, res = await self.tools.execute(
                    session,
                    "modify_cart",
                    {
                        "product_id": item["product_id"],
                        "quantity": 1,
                        "size_code": None,
                        "modifier_ids": [],
                        "selected_options": {},
                        "notes": "",
                    },
                    executed_signatures=executed_signatures,
                )
                tool_calls.append(rec)
                if not isinstance(res, BackendIntegrationError):
                    added_names.append(f"1 × {item['name_ar']}")
                    prod = self.menu_index.get_product(item["product_id"])
                    live_price = prod.base_price if (prod and prod.base_price is not None) else Decimal(str(item["price"]))
                    added_revenue += Decimal(str(live_price))
                    extracted.append(ExtractedOrderItem(product_id=item["product_id"], raw_mention=item["name_ar"], quantity=1))

            if added_names:
                summary_rec, cart = await self.tools.execute(
                    session, "get_cart_summary", {}, executed_signatures=executed_signatures
                )
                tool_calls.append(summary_rec)
                reply = (
                    f"حاضر من عيوني ❤️ ضفتلك ({'، '.join(added_names)}). "
                    f"إجمالي السلة دلوقتي {cart.subtotal} ج.م (قبل التوصيل). تحب نأكد الأوردر؟"
                )
                return self._finish(
                    session,
                    CustomerIntent.ADD_TO_CART,
                    0.96,
                    extracted,
                    False,
                    None,
                    tool_calls,
                    reply,
                    "upsell_handler",
                    upsell_event={
                        "action": "accepted",
                        "offer_type": pending.get("offer_type"),
                        "accepted_product_ids": [i["product_id"] for i in accepted_items],
                        "added_revenue": str(added_revenue),
                    },
                )

        session.pending_upsell = None
        return None

    def _merge_nlu_with_deterministic(
        self,
        llm_data: dict[str, Any],
        det_parsed: dict[str, Any],
        lora_pred: tuple[str, float, float] | None,
        text: str = "",
    ) -> dict[str, Any]:
        """Ground LLM output safely in deterministic MenuIndex extraction & LoRA cross-check."""
        merged = dict(llm_data)
        det_items = det_parsed.get("items") or []
        det_has_candidates = any(
            isinstance(meta, dict) and meta.get("candidates")
            for meta in (det_parsed.get("match_meta") or [])
        )
        det_has_grounded_items = any(
            isinstance(item, dict) and item.get("product_id")
            for item in det_items
        )

        # Deterministic entities are authoritative; only admit model entities that resolve to the same menu item.
        if det_items and (det_has_grounded_items or det_has_candidates):
            merged["items"] = det_items
            merged["match_meta"] = det_parsed.get("match_meta", [])
        else:
            grounded_items: list[dict[str, Any]] = []
            normalized_text = normalize_ar(text)
            text_match = self.menu_index.resolve_mention_with_confidence(text) if text else None
            for item in llm_data.get("items") or []:
                if not isinstance(item, dict):
                    continue
                sanitized = self._sanitize_item_dict(item, fallback_mention=text)
                pid = sanitized.get("product_id")
                mention = sanitized.get("raw_mention", "").strip()
                if pid and mention == str(pid):
                    mention = text
                match = self.menu_index.resolve_mention_with_confidence(mention) if mention else None
                mention_in_text = bool(mention and normalize_ar(mention) in normalized_text)
                if pid:
                    product = self.menu_index.get_product(pid)
                    mention_matches = bool(
                        mention_in_text and match and match.product and match.product.product_id == pid
                    )
                    full_text_matches = bool(text_match and text_match.product and text_match.product.product_id == pid)
                    canonical_name_in_text = bool(
                        product
                        and normalized_text
                        and (normalize_ar(product.name_ar) in normalized_text or normalize_ar(product.name_en) in normalized_text)
                    )
                    if not (mention_matches or full_text_matches or canonical_name_in_text):
                        continue
                elif not mention_in_text:
                    continue
                elif match and match.tier == "high" and match.product:
                    sanitized["product_id"] = match.product.product_id
                grounded_items.append(sanitized)
            merged["items"] = grounded_items or det_items
            if det_items and not grounded_items:
                merged["match_meta"] = det_parsed.get("match_meta", [])
            elif "match_meta" in det_parsed:
                merged["match_meta"] = det_parsed.get("match_meta", [])

        det_intent = str(det_parsed.get("intent", CustomerIntent.UNCLEAR_REQUEST.value))
        det_confidence = float(det_parsed.get("confidence", 0.0))
        det_action_is_grounded = (
            det_intent in (CustomerIntent.ADD_TO_CART.value, CustomerIntent.MODIFY_ORDER.value)
            and (det_has_grounded_items or det_has_candidates)
        )
        if det_intent != CustomerIntent.UNCLEAR_REQUEST.value and (
            det_confidence >= 0.88
            or det_action_is_grounded
            or (det_intent == CustomerIntent.REMOVE_ITEM.value and det_items)
        ):
            merged["intent"] = det_intent
            merged["confidence"] = det_confidence
            if det_parsed.get("requires_clarification"):
                merged["requires_clarification"] = True
                merged["clarification_reason"] = det_parsed.get("clarification_reason")

        if det_parsed.get("is_correction"):
            merged["is_correction"] = True
            if det_parsed.get("correction_patch"):
                merged["correction_patch"] = det_parsed["correction_patch"]

        if lora_pred is not None and lora_pred[0] == str(merged.get("intent", "")):
            merged["confidence"] = round(min(1.0, max(float(merged.get("confidence", 0.7)), lora_pred[1])), 4)
        return merged

    async def _run_nlu(
        self,
        text: str,
        category_hint: str | None = None,
        history: list[ConversationMessage] | None = None,
    ) -> tuple[dict[str, Any], str]:
        # 1. Invoke FineTunedIntentClassifier (CAMeLBERT PEFT LoRA) cleanly in every live turn
        lora_pred: tuple[str, float, float] | None = None
        try:
            self._telemetry["lora_calls"] += 1
            lora_pred = self.finetuned_intent.predict(text)
        except Exception as exc:
            logger.warning("FineTunedIntentClassifier inference failed safely (%s); continuing.", exc)
            lora_pred = None

        # 2. Deterministic Egyptian NLU & MenuIndex extraction (source of truth for items/variants/ambiguity)
        det_parsed = self.offline_nlu.parse(text, category_hint=category_hint)
        det_conf = float(det_parsed.get("confidence", 0.0))
        det_intent = str(det_parsed.get("intent", CustomerIntent.UNCLEAR_REQUEST.value))

        # 3. Gemini Orchestration (Primary -> Fallback on low confidence OR API/timeout/parse failure)
        if self.llm_client.is_configured:
            allowed_intents = ", ".join(i.value for i in CustomerIntent)
            system_prompt = (
                "You are Wasla AI, a warm, natural Egyptian Arabic restaurant employee at MOKA. "
                "Use the declared tools when checking items, viewing the menu, or editing the cart, OR return valid JSON with keys: "
                f"intent (MUST be one of: {allowed_intents}), confidence (0..1), "
                "items (list of {product_id, raw_mention, quantity, size_code, modifier_ids, selected_options, notes}), "
                "requires_clarification, clarification_reason, natural_reply_ar.\n"
                "STRICT RULES:\n"
                "1. Never invent product_ids, menu items, prices, or discounts.\n"
                "2. Preserve EVERY item and EVERY size/variant if the customer orders multiple sizes of the same product.\n"
                "3. Never confirm an order without explicit customer approval and required delivery details.\n"
                "4. If the customer asks for ANY dish/drink NOT in MOKA's catalog below, "
                "set intent='add_to_cart' (or 'price_inquiry'), include {product_id: null, raw_mention: '<dish>', quantity: 1}, "
                "set clarification_reason='unknown_product', AND write a polite Egyptian Arabic `natural_reply_ar`.\n"
                "5. Never repeat a previous assistant reply verbatim; vary phrasing naturally.\n"
                + self.menu_index.build_compact_prompt_catalog()
            )
            recent_ctx = ""
            if history:
                recent_lines = [f"{m.role}: {m.content}" for m in history[-4:]]
                recent_ctx = "Recent conversation:\n" + "\n".join(recent_lines) + "\n\nCurrent customer message: "
            user_payload = recent_ctx + text

            try:
                self._telemetry["gemini_primary_calls"] += 1
                data, model = await self.llm_client.generate_json(system_prompt, user_payload, use_fallback_model=False)
                if not isinstance(data, dict) or float(data.get("confidence", 0)) < 0.72:
                    raise ValueError(
                        f"Primary Gemini returned low confidence ({data.get('confidence') if isinstance(data, dict) else 'non-dict'})"
                    )
                self._last_active_route = f"{model}+lora+menu_index" if lora_pred else f"{model}+menu_index"
                return self._merge_nlu_with_deterministic(data, det_parsed, lora_pred, text), model
            except Exception as primary_exc:
                logger.warning("Primary Gemini call failed (%s); invoking fallback model.", primary_exc)
                try:
                    self._telemetry["gemini_fallback_calls"] += 1
                    data, model = await self.llm_client.generate_json(system_prompt, user_payload, use_fallback_model=True)
                    if not isinstance(data, dict) or float(data.get("confidence", 0)) < 0.55:
                        raise ValueError("Fallback Gemini returned invalid or low-confidence JSON")
                    self._last_active_route = f"{model}(fallback)+lora+menu_index" if lora_pred else f"{model}(fallback)+menu_index"
                    return self._merge_nlu_with_deterministic(data, det_parsed, lora_pred, text), model
                except Exception as fallback_exc:
                    logger.warning("Fallback Gemini call failed (%s); falling back to Local Hybrid.", fallback_exc)

        # 4. Local Confidence-Routed Hybrid: CAMeLBERT LoRA + Deterministic Egyptian Arabic NLU
        self._telemetry["local_hybrid_calls"] += 1
        if lora_pred is not None:
            lora_intent, lora_conf, _ = lora_pred
            has_matched_item = any(isinstance(i, dict) and i.get("product_id") for i in (det_parsed.get("items") or []))
            has_candidates = any(isinstance(m, dict) and m.get("candidates") for m in (det_parsed.get("match_meta") or []))
            has_menu_signal = has_matched_item or has_candidates or bool(det_parsed.get("is_correction"))
            det_high_conf = det_conf >= 0.90
            norm_txt = normalize_ar(text)
            is_known_offmenu_food = any(w in norm_txt for w in _OFF_MENU_DISH_WORDS)

            if not has_menu_signal and not det_high_conf and not is_gibberish_input(text) and not is_known_offmenu_food:
                if lora_intent == "confirm_order":
                    if lora_conf >= 0.90 and any(p in norm_txt for p in _CONFIRM_PHRASES):
                        det_parsed = {**det_parsed, "intent": "confirm_order", "confidence": lora_conf, "requires_clarification": False}
                elif (
                    (lora_intent in _NON_DESTRUCTIVE_LORA_INTENTS and lora_conf >= 0.55)
                    or (lora_intent in _HIGH_CONF_LORA_INTENTS and lora_conf >= 0.72)
                ):
                    det_parsed = {
                        **det_parsed,
                        "intent": lora_intent,
                        "confidence": lora_conf,
                        "items": [],
                        "requires_clarification": (lora_intent == "unclear_request"),
                        "clarification_reason": "unclear_item" if lora_intent == "unclear_request" else None,
                    }
                elif lora_conf >= 0.55 and lora_intent in ("cancel_order", "remove_item", "modify_order"):
                    # Preserve non-destructive intent classification without guessing items
                    det_parsed = {
                        **det_parsed,
                        "intent": lora_intent,
                        "confidence": lora_conf,
                        "items": [],
                        "requires_clarification": True,
                        "clarification_reason": "clarify_order_action",
                    }
            elif lora_intent == det_intent:
                det_parsed = {**det_parsed, "confidence": round(min(1.0, max(det_conf, lora_conf)), 4)}

            route_tag = f"{self.finetuned_intent.model_version}+egyptian_nlu"
            self._last_active_route = route_tag
            return det_parsed, route_tag

        self._last_active_route = "offline_egyptian_nlu"
        return det_parsed, "offline_egyptian_nlu"

    async def _handle_price_inquiry(
        self,
        session: SessionState,
        clean_text: str,
        items: list[ExtractedOrderItem],
        tool_calls: list[ToolCallRecord],
        model_used: str,
        *,
        pre_avail_entries: list[tuple[Any, bool, str | None]] | None = None,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse:
        query_items = list(items)
        if not query_items:
            raw_ext, _ = self.offline_nlu._extract_items(clean_text)
            query_items = [ExtractedOrderItem(**self._sanitize_item_dict(x, fallback_mention=clean_text)) for x in raw_ext]

        resolved_entries: list[tuple[Any, bool, str | None]] = list(pre_avail_entries or [])
        seen_entry_keys: set[tuple[str, str | None]] = {(p.product_id, sc) for p, _, sc in resolved_entries}
        unknown_mentions: list[str] = []
        ambiguous_candidates: list[Any] = []
        ambiguous_mention = ""

        for item in query_items:
            if item.product_id:
                key = (item.product_id, item.size_code)
                if key in seen_entry_keys:
                    continue
                rec, avail_res = await self.tools.execute(
                    session,
                    "check_item_availability",
                    {"product_id": item.product_id},
                    executed_signatures=executed_signatures,
                )
                tool_calls.append(rec)
                if isinstance(avail_res, dict) and avail_res.get("item") is not None:
                    resolved_entries.append((avail_res["item"], bool(avail_res["available"]), item.size_code))
                    seen_entry_keys.add(key)
            elif item.raw_mention:
                res = self.menu_index.resolve_mention_with_confidence(item.raw_mention)
                if res.tier == "high" and res.product is not None:
                    key = (res.product.product_id, item.size_code)
                    if key not in seen_entry_keys:
                        rec, avail_res = await self.tools.execute(
                            session,
                            "check_item_availability",
                            {"product_id": res.product.product_id},
                            executed_signatures=executed_signatures,
                        )
                        tool_calls.append(rec)
                        if isinstance(avail_res, dict) and avail_res.get("item") is not None:
                            resolved_entries.append((avail_res["item"], bool(avail_res["available"]), item.size_code))
                            seen_entry_keys.add(key)
                elif res.candidates:
                    ambiguous_mention = item.raw_mention
                    ambiguous_candidates = res.candidates
                else:
                    unknown_mentions.append(item.raw_mention)

        if resolved_entries:
            reply = persona.render_multi_price_reply(resolved_entries)
            if unknown_mentions:
                reply += f"\n(ملاحظة: '{'، '.join(unknown_mentions)}' مش موجود في منيو MOKA)"
            return self._finish(
                session, CustomerIntent.PRICE_INQUIRY, 0.95, query_items, False, None, tool_calls, reply, model_used
            )

        if ambiguous_candidates:
            reply = persona.render_ambiguous_product(
                ambiguous_mention, [c.name_ar for c in ambiguous_candidates[:5]], is_medium=True
            )
            return self._finish(
                session, CustomerIntent.PRICE_INQUIRY, 0.85, query_items, True, "ambiguous_product", tool_calls, reply, model_used
            )

        if unknown_mentions:
            reply = persona.render_unknown_product(unknown_mentions[0], turn_count=len(session.history))
            return self._finish(
                session, CustomerIntent.PRICE_INQUIRY, 0.88, query_items, True, "unknown_product", tool_calls, reply, model_used
            )

        reply = "تحب تعرف سعر أنهي صنف بالظبط من منيو MOKA؟ 😊"
        return self._finish(
            session, CustomerIntent.PRICE_INQUIRY, 0.85, query_items, True, "missing_product", tool_calls, reply, model_used
        )

    async def _get_unavailable_product_ids(self) -> set[str]:
        unavailable: set[str] = set()
        if hasattr(self.backend, "_unavailable_ids"):
            unavailable.update(getattr(self.backend, "_unavailable_ids"))
        for p in self.menu_index.all_products():
            if p.is_available is False:
                unavailable.add(p.product_id)
        return unavailable

    async def _handle_add_items(
        self,
        session: SessionState,
        intent: CustomerIntent,
        confidence: float,
        items: list[ExtractedOrderItem],
        tool_calls: list[ToolCallRecord],
        model_used: str,
        match_meta: list[dict[str, Any]] | None = None,
        natural_reply_ar: str = "",
        replace_cart_item_id: str | None = None,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse:
        # If previous order was already confirmed and customer starts ordering again, begin a clean cart epoch
        if session.order_confirmed:
            await self.backend.clear_cart(session.phone_number)
            session.cart = await self.backend.get_cart_summary(session.phone_number)
            session.order_confirmed = False
            session.active_order_id = None
            session.confirmed_cart_sig = None
            session.upsell_declined = False
            session.session_epoch += 1

        # Low-confidence guard against hallucinated order additions
        if confidence < 0.55 and not any(
            isinstance(m, dict) and m.get("tier") == "high" for m in (match_meta or [])
        ) and any(i.product_id is not None for i in items):
            reply = persona.render_unclear_request(turn_count=len(session.history))
            return self._finish(
                session, CustomerIntent.UNCLEAR_REQUEST, confidence, items, True, "low_confidence", tool_calls, reply, model_used
            )

        added_names: list[str] = []
        meta_list = match_meta or []
        for idx, item in enumerate(items):
            item_meta = meta_list[idx] if idx < len(meta_list) else {}
            if not item.product_id:
                res = self.menu_index.resolve_mention_with_confidence(item.raw_mention)
                candidates = res.candidates
                if candidates:
                    cat_hint = item_meta.get("category_hint") or res.category_hint
                    session.pending_clarification = {
                        "type": "ambiguous_product",
                        "item": item.model_dump(),
                        "candidates": [c.product_id for c in candidates[:5]],
                        "category_hint": cat_hint,
                        "replace_cart_item_id": replace_cart_item_id,
                    }
                    reply = persona.render_ambiguous_product(
                        item.raw_mention, [c.name_ar for c in candidates[:5]], is_medium=(res.tier == "medium")
                    )
                    return self._finish(
                        session, intent, confidence, items, True, "ambiguous_product", tool_calls, reply, model_used
                    )
                reply = natural_reply_ar or persona.render_unknown_product(item.raw_mention, turn_count=len(session.history))
                return self._finish(
                    session, intent, confidence, items, True, "unknown_product", tool_calls, reply, model_used
                )

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
                executed_signatures=executed_signatures,
            )
            tool_calls.append(rec)
            if isinstance(res, BackendIntegrationError):
                if res.code in ("MISSING_OR_INVALID_SIZE", "MISSING_REQUIRED_OPTION"):
                    session.pending_clarification = {
                        "type": res.code,
                        "item": item.model_dump(),
                        "replace_cart_item_id": replace_cart_item_id,
                    }
                    return self._finish(session, intent, confidence, items, True, res.code, tool_calls, res.message, model_used)
                return self._finish(session, intent, confidence, items, True, res.code, tool_calls, res.message, model_used)
            prod = self.menu_index.get_product(item.product_id)
            if prod:
                size_suffix = f" ({item.size_code})" if item.size_code else ""
                added_names.append(f"{item.quantity} × {prod.name_ar}{size_suffix}")

        if replace_cart_item_id:
            rec, result = await self.tools.execute(
                session,
                "remove_cart_item",
                {"cart_item_id": replace_cart_item_id},
                executed_signatures=executed_signatures,
            )
            tool_calls.append(rec)
            if isinstance(result, BackendIntegrationError):
                return self._finish(
                    session,
                    intent,
                    confidence,
                    items,
                    True,
                    result.code,
                    tool_calls,
                    result.message,
                    model_used,
                )

        summary_rec, cart = await self.tools.execute(
            session, "get_cart_summary", {}, executed_signatures=executed_signatures
        )
        tool_calls.append(summary_rec)

        unavailable_ids = await self._get_unavailable_product_ids()
        upsell = self.recommendation_engine.recommend(
            cart=cart,
            intent=intent,
            unavailable_product_ids=unavailable_ids,
            rejected_product_ids=set(session.rejected_upsell_ids),
            upsell_declined=session.upsell_declined,
            budget_limit=session.budget_limit,
        )
        upsell_dict = upsell.to_dict() if upsell else None
        upsell_event = None
        if upsell_dict:
            session.pending_upsell = upsell_dict
            upsell_event = {
                "action": "offered",
                "offer_type": upsell_dict["offer_type"],
                "offered_product_ids": [i["product_id"] for i in upsell_dict["items"]],
                "combined_price": upsell_dict["combined_price"],
            }

        reply = persona.render_item_added(added_names, cart, upsell=upsell, turn_count=len(session.history))
        return self._finish(
            session,
            intent,
            confidence,
            items,
            False,
            None,
            tool_calls,
            reply,
            model_used,
            upsell_offer=upsell_dict,
            upsell_event=upsell_event,
        )

    async def _handle_remove_item(
        self,
        session: SessionState,
        items: list[ExtractedOrderItem],
        tool_calls: list[ToolCallRecord],
        model_used: str,
        *,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse:
        if not session.cart.items:
            return self._finish(
                session, CustomerIntent.REMOVE_ITEM, 0.9, items, False, None, tool_calls, "السلة فاضية حالياً يا فندم.", model_used
            )

        if not items:
            target_line = session.cart.items[-1]
        else:
            item = items[0]
            candidate_ids: set[str] = set()
            if item.product_id:
                candidate_ids.add(item.product_id)
            elif item.raw_mention:
                match = self.menu_index.resolve_mention_with_confidence(item.raw_mention)
                if match.product:
                    candidate_ids.add(match.product.product_id)
                candidate_ids.update(candidate.product_id for candidate in match.candidates)

            matching_lines = [line for line in session.cart.items if line.product_id in candidate_ids]
            if not matching_lines:
                reply = f"مش لاقي ({item.raw_mention}) في السلة حالياً. تحب أشيل صنف تاني؟"
                return self._finish(
                    session,
                    CustomerIntent.REMOVE_ITEM,
                    0.9,
                    items,
                    True,
                    "item_not_in_cart",
                    tool_calls,
                    reply,
                    model_used,
                )

            matching_product_ids = {line.product_id for line in matching_lines}
            if len(matching_product_ids) > 1:
                session.pending_clarification = {
                    "type": "ambiguous_remove_item",
                    "item": item.model_dump(),
                    "candidate_product_ids": sorted(matching_product_ids),
                }
                candidate_names = [
                    product.name_ar
                    for product_id in sorted(matching_product_ids)
                    if (product := self.menu_index.get_product(product_id)) is not None
                ]
                reply = persona.render_ambiguous_product(item.raw_mention, candidate_names, is_medium=False)
                return self._finish(
                    session,
                    CustomerIntent.REMOVE_ITEM,
                    0.9,
                    items,
                    True,
                    "ambiguous_remove_item",
                    tool_calls,
                    reply,
                    model_used,
                )
            target_line = matching_lines[-1]

        rec, res = await self.tools.execute(
            session,
            "remove_cart_item",
            {"cart_item_id": target_line.cart_item_id},
            executed_signatures=executed_signatures,
        )
        tool_calls.append(rec)
        if isinstance(res, BackendIntegrationError):
            return self._finish(session, CustomerIntent.REMOVE_ITEM, 0.8, items, True, res.code, tool_calls, res.message, model_used)
        reply = persona.render_item_removed(session.cart)
        return self._finish(session, CustomerIntent.REMOVE_ITEM, 0.92, items, False, None, tool_calls, reply, model_used)

    async def _handle_confirm_order(
        self,
        session: SessionState,
        tool_calls: list[ToolCallRecord],
        model_used: str,
        *,
        ctx: TurnExecutionContext | None = None,
        explicit_confirm: bool = True,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse:
        exec_ctx = ctx or TurnExecutionContext()
        if not session.cart.items:
            return self._finish(
                session,
                CustomerIntent.CONFIRM_ORDER,
                0.9,
                [],
                True,
                "empty_cart",
                tool_calls,
                "السلة فاضية يا فندم، اختار الأصناف الأول عشان نأكد الأوردر 😊",
                model_used,
            )

        # Prevent duplicate order creation if this exact cart epoch was already confirmed
        cart_sig = "|".join(
            f"{i.product_id}:{i.quantity}:{i.size_code}:{','.join(m.get('name', '') for m in i.modifiers)}:{i.notes}"
            for i in session.cart.items
        )
        expected_idem = hashlib.sha256(
            f"{session.phone_number}:{session.session_epoch}:{cart_sig}".encode("utf-8")
        ).hexdigest()[:24]
        if session.order_confirmed and session.confirmed_cart_sig == expected_idem and session.active_order_id:
            reply = f"طلبك رقم #{session.active_order_id} متأكد بالفعل واتبعت للمطبخ ❤️ تحب تبدأ طلب جديد؟"
            return self._finish(
                session, CustomerIntent.CONFIRM_ORDER, 0.98, [], False, None, tool_calls, reply, model_used, order_id=session.active_order_id
            )

        # Revalidate live DB/backend prices & availability before accepting confirmation
        if hasattr(self.backend, "revalidate_cart"):
            revalidated_cart, unavailable_names, price_changed = await self.backend.revalidate_cart(session.phone_number)
            session.cart = revalidated_cart
            if unavailable_names:
                unavail_str = "، ".join(unavailable_names)
                reply = f"نعتذر جداً 🙏 الصنف ({unavail_str}) خلص من المطبخ دلوقتي وتم شيله من السلة. تحب نبدله بصنف تاني؟"
                return self._finish(
                    session, CustomerIntent.CONFIRM_ORDER, 0.92, [], True, "ITEM_OUT_OF_STOCK", tool_calls, reply, model_used
                )
            if price_changed:
                session.awaiting_final_confirmation = True
                reply = (
                    f"تم تحديث أسعار المنيو الرسمية في المطعم — الإجمالي المحدّث لطلبك هو "
                    f"{session.cart.total_amount} ج.م. تحب نأكد الأوردر بالسعر الجديد؟"
                )
                return self._finish(
                    session,
                    CustomerIntent.CONFIRM_ORDER,
                    0.92,
                    [],
                    True,
                    "PRICE_UPDATED_RECONFIRM",
                    tool_calls,
                    reply,
                    model_used,
                )

        missing = []
        if not session.customer_name:
            missing.append("الاسم")
        if not session.delivery_address:
            missing.append("عنوان التوصيل بالتفصيل")
        if not session.backup_phone:
            missing.append("رقم تليفون بديل")

        if missing:
            session.awaiting_final_confirmation = True
            reply = persona.render_missing_details(session.cart.total_amount, missing)
            return self._finish(
                session, CustomerIntent.CONFIRM_ORDER, 0.92, [], True, "missing_order_details", tool_calls, reply, model_used
            )

        if not explicit_confirm:
            session.awaiting_final_confirmation = True
            reply = (
                f"ملخص طلبك: {len(session.cart.items)} صنف، الإجمالي شامل التوصيل {session.cart.total_amount} ج.م "
                f"على عنوان ({session.delivery_address}). أؤكد وأبعت الأوردر للمطبخ؟"
            )
            return self._finish(
                session,
                CustomerIntent.PROVIDE_DETAILS,
                0.92,
                [],
                True,
                "awaiting_explicit_confirm",
                tool_calls,
                reply,
                model_used,
            )

        # Ensure delivery fee is resolved deterministically before customer total confirmation
        if session.cart.delivery_fee == Decimal("0.00") and session.zone_name != "free_zone":
            deliv_rec, _ = await self.tools.execute(
                session,
                "resolve_delivery",
                {"zone_name": session.zone_name or "default"},
                executed_signatures=executed_signatures,
            )
            tool_calls.append(deliv_rec)

        rec, res = await self.tools.execute(
            session,
            "confirm_and_lock_order",
            {
                "customer_name": session.customer_name,
                "backup_phone": session.backup_phone,
                "delivery_address": session.delivery_address,
                "confirmed_by_customer": True,
                "data_source": exec_ctx.data_source,
                "is_test": exec_ctx.is_test,
            },
            executed_signatures=executed_signatures,
        )
        tool_calls.append(rec)
        if isinstance(res, BackendIntegrationError):
            return self._finish(session, CustomerIntent.CONFIRM_ORDER, 0.9, [], True, res.code, tool_calls, res.message, model_used)

        if isinstance(res.get("cart"), dict):
            c = res["cart"]
            session.cart = session.cart.model_copy(
                update={
                    "subtotal": Decimal(str(c.get("subtotal", session.cart.subtotal))),
                    "delivery_fee": Decimal(str(c.get("delivery_fee", session.cart.delivery_fee))),
                    "total_amount": Decimal(str(c.get("total_amount", session.cart.total_amount))),
                }
            )

        order_id = res.get("id")
        reply = persona.render_order_confirmed(order_id)
        return self._finish(
            session, CustomerIntent.CONFIRM_ORDER, 0.98, [], False, None, tool_calls, reply, model_used, order_id=order_id
        )

    async def _try_resolve_clarification(
        self,
        session: SessionState,
        text: str,
        tool_calls: list[ToolCallRecord],
        *,
        executed_signatures: set[str] | None = None,
    ) -> DispatcherResponse | None:
        pending = session.pending_clarification
        if not pending:
            return None
        ptype = pending["type"]
        item = ExtractedOrderItem(**pending["item"])
        norm = normalize_ar(text)

        if ptype == "ambiguous_remove_item":
            candidate_ids = set(pending.get("candidate_product_ids") or [])
            match = self.menu_index.resolve_mention_with_confidence(text)
            if match.product and match.product.product_id in candidate_ids:
                matching_lines = [
                    line for line in session.cart.items if line.product_id == match.product.product_id
                ]
                if matching_lines:
                    session.pending_clarification = None
                    selected_item = item.model_copy(
                        update={
                            "product_id": match.product.product_id,
                            "raw_mention": match.product.name_ar,
                        }
                    )
                    rec, result = await self.tools.execute(
                        session,
                        "remove_cart_item",
                        {"cart_item_id": matching_lines[-1].cart_item_id},
                        executed_signatures=executed_signatures,
                    )
                    tool_calls.append(rec)
                    if isinstance(result, BackendIntegrationError):
                        return self._finish(
                            session,
                            CustomerIntent.REMOVE_ITEM,
                            0.9,
                            [selected_item],
                            True,
                            result.code,
                            tool_calls,
                            result.message,
                            "clarification_resolver",
                        )
                    return self._finish(
                        session,
                        CustomerIntent.REMOVE_ITEM,
                        0.92,
                        [selected_item],
                        False,
                        None,
                        tool_calls,
                        persona.render_item_removed(session.cart),
                        "clarification_resolver",
                    )

            candidate_names = [
                product.name_ar
                for product_id in sorted(candidate_ids)
                if (product := self.menu_index.get_product(product_id)) is not None
            ]
            reply = persona.render_ambiguous_product(item.raw_mention, candidate_names, is_medium=False)
            return self._finish(
                session,
                CustomerIntent.REMOVE_ITEM,
                0.9,
                [item],
                True,
                "ambiguous_remove_item",
                tool_calls,
                reply,
                "clarification_resolver",
            )

        if ptype == "MISSING_OR_INVALID_SIZE":
            size = self.offline_nlu._extract_size(norm)
            if size:
                item.size_code = size
                session.pending_clarification = None
                return await self._handle_add_items(
                    session,
                    CustomerIntent.ADD_TO_CART,
                    0.95,
                    [item],
                    tool_calls,
                    "clarification_resolver",
                    replace_cart_item_id=pending.get("replace_cart_item_id"),
                    executed_signatures=executed_signatures,
                )

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
                return await self._handle_add_items(
                    session,
                    CustomerIntent.ADD_TO_CART,
                    0.95,
                    [item],
                    tool_calls,
                    "clarification_resolver",
                    replace_cart_item_id=pending.get("replace_cart_item_id"),
                    executed_signatures=executed_signatures,
                )

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
                return await self._handle_add_items(
                    session,
                    CustomerIntent.ADD_TO_CART,
                    res.confidence,
                    [item],
                    tool_calls,
                    "clarification_resolver",
                    replace_cart_item_id=pending.get("replace_cart_item_id"),
                    executed_signatures=executed_signatures,
                )

            for cid in pending.get("candidates", []):
                prod = self.menu_index.get_product(cid)
                if prod and (normalize_ar(prod.name_ar) in norm or normalize_ar(prod.category_ar) in norm):
                    item.product_id = prod.product_id
                    session.pending_clarification = None
                    return await self._handle_add_items(
                        session,
                        CustomerIntent.ADD_TO_CART,
                        0.94,
                        [item],
                        tool_calls,
                        "clarification_resolver",
                        replace_cart_item_id=pending.get("replace_cart_item_id"),
                        executed_signatures=executed_signatures,
                    )
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
        elif session.awaiting_final_confirmation and not session.backup_phone:
            bare_phone = re.search(r"\b(01[0125]\d{8})\b", text)
            if bare_phone:
                session.backup_phone = bare_phone.group(1).strip()

    def _extract_zone(self, text: str) -> str | None:
        for z in ["المعادي", "مدينة نصر", "التجمع", "الدقي", "المهندسين", "free_zone"]:
            if z in text:
                return z
        return None

    def _extract_requested_category(self, text: str) -> str | None:
        norm = normalize_ar(text)
        for cat in ["بيتزا", "كريب", "برجر", "باستا", "وجبات", "مشروبات", "وافل", "بان كيك", "قشطوطه", "طواجن", "سندوتش"]:
            if normalize_ar(cat) in norm:
                return cat
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
        upsell_offer: dict[str, Any] | None = None,
        upsell_event: dict[str, Any] | None = None,
        complaint_assessment: dict[str, Any] | None = None,
    ) -> DispatcherResponse:
        session.last_intent = intent.value
        session.history.append(ConversationMessage(role="assistant", content=reply))
        self._persist_session(session)
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
            upsell_offer=upsell_offer,
            upsell_event=upsell_event,
            complaint_assessment=complaint_assessment,
            reply_ar=reply,
            model_used=model_used,
        )
