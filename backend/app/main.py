"""Wasla — Integrated FastAPI Backend & AI Dispatcher Server.

Preserves all original Backend endpoints, SQLAlchemy models, 7-state OrderStatus machine,
and deterministic Decimal cart_calculator while integrating:
- Canonical MOKA 145-item menu auto-seeding from menu/menu.json
- WaslaAIDispatcher (Egyptian Arabic NLU + Hybrid Audio + Function Calling)
- CashierDirectiveInterpreter (Human-in-the-Loop validation on /api/orders/{id}/instruct)
- Meta WhatsApp Webhook GET verification + HMAC-SHA256 signature check + AI dispatch
- Live SQL-backed /api/metrics & /api/menu/{id}/availability endpoints for the Cashier Console
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Ensure workspace root is importable for ai_service
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    log_audit,
    require_admin,
    verify_password,
)
from app.cart_calculator import calculate_cart
from app.config import settings
from app.database import Base, SessionLocal, engine, get_db
from app.meta_service import MetaWhatsAppClient
from app.models import (
    AuditLog,
    ChatSession,
    ComplaintTicket,
    FeedbackRequest,
    InboundMessageBuffer,
    MenuItem,
    NotificationAttempt,
    Order,
    OrderEvent,
    OrderStatus,
    OrderStatusHistory,
    Outbox,
    ProcessedMessage,
    RecommendationEvent,
    RestaurantSettings,
    ScheduledJob,
    User,
)
from app.notifications import NotificationService, TelegramBotClient
from app.scheduler import DELIVERED_STATUSES, DurableJobScheduler
from app.ws_manager import ws_manager

from ai_service.adapters.interfaces import BackendIntegrationError
from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.complaints.detector import assess_complaint
from ai_service.config import WaslaConfig
from ai_service.dispatcher.debounce import MessageDebouncer
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.schemas.models import (
    AudioInput,
    BackendConfirmOrderPayload,
    CartSnapshot,
    CashierDirectiveRequest,
    DeliveryResolution,
    SessionState,
)
from ai_service.understanding.menu_index import MenuIndex


DASHBOARD_DIR = ROOT_DIR / "dashboard"
MENU_JSON_PATH = ROOT_DIR / "menu" / "menu.json"
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)


def _migrate_sqlite_columns() -> None:
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(orders)")).fetchall()
        }
        order_additions = [
            ("backup_phone", "VARCHAR(32) NOT NULL DEFAULT ''"),
            ("confirmed_by_customer", "BOOLEAN NOT NULL DEFAULT 0"),
            ("idempotency_key", "VARCHAR(128)"),
            ("data_source", "VARCHAR(32) NOT NULL DEFAULT 'api'"),
            ("is_test", "BOOLEAN NOT NULL DEFAULT 0"),
            ("order_type", "VARCHAR(32) NOT NULL DEFAULT 'Delivery'"),
            ("refunded", "BOOLEAN NOT NULL DEFAULT 0"),
            ("confirmed_at", "DATETIME"),
            ("accepted_at", "DATETIME"),
            ("preparing_at", "DATETIME"),
            ("ready_at", "DATETIME"),
            ("out_for_delivery_at", "DATETIME"),
            ("delivered_at", "DATETIME"),
            ("cancelled_at", "DATETIME"),
            ("failed_at", "DATETIME"),
            ("delivery_confirmed_by", "VARCHAR(64)"),
        ]
        for col_name, col_def in order_additions:
            if col_name not in columns:
                connection.execute(text(f"ALTER TABLE orders ADD COLUMN {col_name} {col_def}"))
        connection.execute(
            text("CREATE UNIQUE INDEX IF NOT EXISTS ux_orders_idempotency_key ON orders (idempotency_key)")
        )

        session_cols = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(chat_sessions)")).fetchall()
        }
        for col_name, col_def in [
            ("human_takeover_by", "VARCHAR(64)"),
            ("human_takeover_at", "DATETIME"),
            ("last_customer_message_at", "DATETIME"),
            ("opted_out_reviews", "BOOLEAN NOT NULL DEFAULT 0"),
        ]:
            if session_cols and col_name not in session_cols:
                connection.execute(text(f"ALTER TABLE chat_sessions ADD COLUMN {col_name} {col_def}"))

        fb_cols = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(feedback_requests)")).fetchall()
        }
        for col_name, col_def in [
            ("customer_phone", "VARCHAR(32) NOT NULL DEFAULT ''"),
            ("issue_reported", "TEXT"),
            ("channel", "VARCHAR(32) NOT NULL DEFAULT 'whatsapp'"),
            ("suppressed_reason", "VARCHAR(120)"),
            ("policy_blocked_reason", "VARCHAR(160)"),
        ]:
            if fb_cols and col_name not in fb_cols:
                connection.execute(text(f"ALTER TABLE feedback_requests ADD COLUMN {col_name} {col_def}"))

        job_cols = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(scheduled_jobs)")).fetchall()
        }
        for col_name, col_def in [
            ("locked_by", "VARCHAR(64)"),
            ("locked_until", "DATETIME"),
        ]:
            if job_cols and col_name not in job_cols:
                connection.execute(text(f"ALTER TABLE scheduled_jobs ADD COLUMN {col_name} {col_def}"))

        user_cols = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(users)")).fetchall()
        }
        if user_cols and "must_change_password" not in user_cols:
            connection.execute(
                text("ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT 0")
            )


def seed_moka_menu_if_empty() -> int:
    """Seed canonical 145 MOKA menu items when menu_items table is empty."""
    db = SessionLocal()
    try:
        if db.scalar(select(MenuItem).limit(1)) is not None:
            return 0
        if not MENU_JSON_PATH.exists():
            return 0
        menu_data = json.loads(MENU_JSON_PATH.read_text(encoding="utf-8"))
        categories = {c["category_id"]: c["name_ar"] for c in menu_data["categories"]}
        modifiers_map = {m["modifier_id"]: m for m in menu_data["modifiers"]}

        inserted = 0
        for p in menu_data["products"]:
            base_price = p["base_price"]
            if base_price is None and p.get("sizes"):
                base_price = p["sizes"][0]["price"]
            extras = [
                {
                    "modifier_id": mid,
                    "name": modifiers_map[mid]["name_ar"],
                    "price": str(modifiers_map[mid]["price"]),
                }
                for mid in p.get("modifiers", [])
                if mid in modifiers_map
            ]
            sizes = [
                {"code": s["code"], "name": s["name_ar"], "price": str(s["price"])}
                for s in p.get("sizes", [])
            ]
            db.add(
                MenuItem(
                    name=p["name_ar"],
                    category=categories.get(p["category_id"], p["category_id"]),
                    base_price=Decimal(str(base_price)),
                    is_available=True,
                    modifiers_json={
                        "product_id": p["product_id"],
                        "sizes": sizes,
                        "extras": extras,
                        "options": p.get("options", []),
                    },
                )
            )
            inserted += 1
        db.commit()
        return inserted
    finally:
        db.close()


def _delivery_quote(zone_name: str) -> dict:
    zone = zone_name.strip() or "default"
    fee = settings.delivery_free_zone_fee if zone == settings.delivery_free_zone else settings.delivery_default_fee
    return {
        "deliverable": True,
        "zone_name": zone,
        "delivery_fee": str(fee),
        "estimated_minutes": settings.delivery_estimated_minutes,
    }


def _associate_active_order(phone_number: str, order_id: int, db: Session) -> None:
    row = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
    if row is None:
        db.add(ChatSession(phone_number=phone_number, active_order_id=order_id))
    else:
        row.active_order_id = order_id


class _DirectDBOrderBridge(MockBackendAdapter):
    """Production DB-backed bridge syncing live PostgreSQL/SQLite menu, carts, sessions & orders."""

    IS_MOCK = False

    def sync_menu_sync(self) -> None:
        db = SessionLocal()
        try:
            rows = db.scalars(select(MenuItem).order_by(MenuItem.id.asc())).all()
            if rows:
                self._unavailable_ids = self.menu_index.apply_db_items(rows)
        finally:
            db.close()

    def _hydrate_cart_from_db(self, phone_number: str) -> None:
        db = SessionLocal()
        try:
            row = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
            if row is None or not isinstance(row.pending_cart, dict) or not row.pending_cart:
                self._carts[phone_number] = CartSnapshot()
            else:
                self._carts[phone_number] = CartSnapshot.model_validate(row.pending_cart)
        finally:
            db.close()

    def _persist_cart_to_db(self, phone_number: str, cart: CartSnapshot) -> None:
        db = SessionLocal()
        try:
            stmt = select(ChatSession).where(ChatSession.phone_number == phone_number)
            if engine.dialect.name == "postgresql":
                stmt = stmt.with_for_update()
            row = db.scalar(stmt)
            if row is None:
                row = ChatSession(phone_number=phone_number)
                db.add(row)
            row.pending_cart = cart.model_dump(mode="json")
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def load_session_state(self, phone_number: str, ttl_seconds: int = 3600) -> SessionState | None:
        db = SessionLocal()
        try:
            row = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
            if row is None:
                return None

            conv_state = dict(row.conversation_state or {})
            raw_v2 = conv_state.get("session_v2")
            if isinstance(raw_v2, dict):
                state = SessionState.model_validate(raw_v2)
            else:
                cart_obj = CartSnapshot.model_validate(row.pending_cart) if (isinstance(row.pending_cart, dict) and row.pending_cart) else CartSnapshot()
                state = SessionState(
                    phone_number=phone_number,
                    cart=cart_obj,
                    active_order_id=row.active_order_id,
                    needs_human_takeover=bool(row.needs_human_takeover),
                    human_takeover_by=row.human_takeover_by,
                    latest_directive=row.latest_directive,
                    last_intent=conv_state.get("last_intent"),
                )

            # Enforce DB column authority for takeover & active order & cart snapshot
            if isinstance(row.pending_cart, dict) and row.pending_cart:
                state = state.model_copy(update={"cart": CartSnapshot.model_validate(row.pending_cart)})
            state = state.model_copy(
                update={
                    "active_order_id": row.active_order_id if row.active_order_id is not None else state.active_order_id,
                    "needs_human_takeover": bool(row.needs_human_takeover),
                    "human_takeover_by": row.human_takeover_by,
                    "latest_directive": row.latest_directive or state.latest_directive,
                }
            )

            now = datetime.now(timezone.utc)
            last_seen = row.last_customer_message_at or state.updated_at
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            if ttl_seconds > 0 and (now - last_seen) > timedelta(seconds=ttl_seconds) and not state.needs_human_takeover:
                expired = SessionState(
                    phone_number=phone_number,
                    customer_name=state.customer_name,
                    backup_phone=state.backup_phone,
                    delivery_address=state.delivery_address,
                    session_epoch=state.session_epoch + 1,
                    updated_at=now,
                )
                self._carts[phone_number] = CartSnapshot()
                row.pending_cart = expired.cart.model_dump(mode="json")
                row.conversation_state = {**conv_state, "session_v2": expired.model_dump(mode="json")}
                db.commit()
                return expired

            self._carts[phone_number] = state.cart
            return state
        finally:
            db.close()

    def save_session_state(self, session: SessionState) -> None:
        trimmed_history = session.history[-30:] if len(session.history) > 30 else session.history
        to_save = session.model_copy(update={"history": trimmed_history, "updated_at": datetime.now(timezone.utc)})
        self._carts[session.phone_number] = to_save.cart

        db = SessionLocal()
        try:
            stmt = select(ChatSession).where(ChatSession.phone_number == session.phone_number)
            if engine.dialect.name == "postgresql":
                stmt = stmt.with_for_update()
            row = db.scalar(stmt)
            if row is None:
                row = ChatSession(phone_number=session.phone_number)
                db.add(row)

            conv_state = dict(row.conversation_state or {})
            conv_state["session_v2"] = to_save.model_dump(mode="json")
            if to_save.last_intent:
                conv_state["last_intent"] = to_save.last_intent

            row.pending_cart = to_save.cart.model_dump(mode="json")
            row.active_order_id = to_save.active_order_id
            row.needs_human_takeover = to_save.needs_human_takeover
            row.human_takeover_by = to_save.human_takeover_by
            if to_save.latest_directive is not None:
                row.latest_directive = to_save.latest_directive
            row.conversation_state = conv_state
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_session_state(self, phone_number: str) -> None:
        self._carts.pop(phone_number, None)
        db = SessionLocal()
        try:
            row = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
            if row is not None:
                row.pending_cart = {}
                row.needs_human_takeover = False
                row.human_takeover_by = None
                conv = dict(row.conversation_state or {})
                conv.pop("session_v2", None)
                row.conversation_state = conv
                db.commit()
        finally:
            db.close()

    async def modify_cart(self, phone_number: str, *args, **kwargs):
        self._hydrate_cart_from_db(phone_number)
        cart = await super().modify_cart(phone_number, *args, **kwargs)
        self._persist_cart_to_db(phone_number, cart)
        return cart

    async def remove_cart_item(self, phone_number: str, cart_item_id: str):
        self._hydrate_cart_from_db(phone_number)
        cart = await super().remove_cart_item(phone_number, cart_item_id)
        self._persist_cart_to_db(phone_number, cart)
        return cart

    async def get_cart_summary(self, phone_number: str):
        self._hydrate_cart_from_db(phone_number)
        cart = await super().get_cart_summary(phone_number)
        self._persist_cart_to_db(phone_number, cart)
        return cart

    async def clear_cart(self, phone_number: str):
        cart = await super().clear_cart(phone_number)
        self._persist_cart_to_db(phone_number, cart)
        return cart

    async def resolve_delivery(self, phone_number: str, *args, **kwargs):
        self._hydrate_cart_from_db(phone_number)
        zone_name = str(kwargs.get("zone_name") or (args[0] if args else "")).strip() or "default"
        quote = _delivery_quote(zone_name)
        cart = self._carts.get(phone_number, CartSnapshot())
        updated = self._recalculate(
            cart.model_copy(update={"delivery_fee": Decimal(quote["delivery_fee"]), "zone_name": zone_name})
        )
        self._carts[phone_number] = updated
        self._persist_cart_to_db(phone_number, updated)
        return DeliveryResolution(
            deliverable=True,
            zone_name=zone_name,
            delivery_fee=Decimal(quote["delivery_fee"]),
            estimated_minutes=quote["estimated_minutes"],
            message=f"رسوم التوصيل لمنطقة {zone_name}: {quote['delivery_fee']} ج.م.",
        )

    async def revalidate_cart(self, phone_number: str):
        self._hydrate_cart_from_db(phone_number)
        updated, unavailable_names, price_changed = await super().revalidate_cart(phone_number)
        self._persist_cart_to_db(phone_number, updated)
        return updated, unavailable_names, price_changed

    async def confirm_and_lock_order(self, payload: BackendConfirmOrderPayload) -> dict:
        self.sync_menu_sync()
        db = SessionLocal()
        try:
            order_payload = ConfirmOrder(
                customer_name=payload.customer_name,
                customer_phone=payload.customer_phone,
                backup_phone=payload.backup_phone,
                delivery_address=payload.delivery_address,
                zone_name=payload.zone_name,
                items=[
                    CartItem(
                        item_id=i.item_id,
                        quantity=i.quantity,
                        size_code=i.size_code,
                        modifiers=i.modifiers,
                        selected_options=i.selected_options,
                        notes=i.notes,
                    )
                    for i in payload.items
                ],
                idempotency_key=payload.idempotency_key,
                confirmed_by_customer=payload.confirmed_by_customer,
                data_source=payload.data_source,
                is_test=payload.is_test,
            )
            return await _create_order_in_db(order_payload, db)
        except HTTPException as exc:
            code = "ITEM_OUT_OF_STOCK" if exc.status_code == 409 else "ORDER_VALIDATION_ERROR"
            raise BackendIntegrationError(code, str(exc.detail), exc.status_code) from exc
        finally:
            db.close()

    async def get_customer_order_status(
        self, phone_number: str, order_id: int | None = None
    ) -> dict | None:
        db = SessionLocal()
        try:
            stmt = select(Order).where(Order.customer_phone == phone_number)
            if settings.app_env.lower() == "production":
                stmt = stmt.where(Order.is_test.is_(False))
            if order_id is not None:
                stmt = stmt.where(Order.id == order_id)
                order = db.scalar(stmt)
            else:
                session_row = db.scalar(
                    select(ChatSession).where(ChatSession.phone_number == phone_number)
                )
                if session_row is not None and session_row.active_order_id is not None:
                    order = db.scalar(stmt.where(Order.id == session_row.active_order_id))
                else:
                    order = db.scalar(stmt.order_by(Order.created_at.desc(), Order.id.desc()).limit(1))
            if order is None:
                return None
            status = getattr(order.status, "value", order.status)
            return {"id": order.id, "status": str(status)}
        finally:
            db.close()


wasla_config = WaslaConfig(menu_json_path=MENU_JSON_PATH)
menu_index = MenuIndex(MENU_JSON_PATH)
db_bridge = _DirectDBOrderBridge(menu_index)
ai_dispatcher = WaslaAIDispatcher(wasla_config, menu_index, db_bridge)
inbound_debouncer = MessageDebouncer(window_seconds=wasla_config.message_debounce_seconds)
meta_client = MetaWhatsAppClient(
    access_token=settings.meta_access_token,
    phone_number_id=settings.meta_phone_number_id,
    graph_version=settings.meta_graph_version,
)
telegram_client = TelegramBotClient(bot_token=settings.telegram_bot_token)
notification_service = NotificationService(settings, meta_client, telegram_client)
job_scheduler = DurableJobScheduler(
    notification_service=notification_service,
    review_delay_minutes=settings.review_delay_minutes,
    arrival_check_delay_minutes=settings.arrival_check_delay_minutes,
)


async def feedback_worker() -> None:
    """Durable restart-safe scheduler loop backed by `scheduled_jobs` table."""
    boot_db = SessionLocal()
    try:
        job_scheduler.recover_stale_jobs_on_startup(boot_db)
    finally:
        boot_db.close()

    while True:
        db = SessionLocal()
        try:
            await job_scheduler.process_due_jobs(db)
        finally:
            db.close()
        await asyncio.sleep(30)


def _validate_production_security() -> None:
    """Refuse to boot in production with insecure default secrets or SQLite fallback."""
    if settings.app_env.lower() != "production":
        return
    if "change-me" in settings.auth_secret_key or len(settings.auth_secret_key) < 32:
        raise RuntimeError("SECURITY FATAL: Set a strong 32+ char AUTH_SECRET_KEY in production.")
    if settings.bootstrap_admin_password == "WaslaAdmin2026!":
        raise RuntimeError("SECURITY FATAL: Change BOOTSTRAP_ADMIN_PASSWORD before running in production.")
    if settings.database_url.startswith("sqlite"):
        raise RuntimeError("PRODUCTION FATAL: SQLite is forbidden in production. Configure PostgreSQL DATABASE_URL.")
    if not settings.meta_app_secret:
        raise RuntimeError("SECURITY FATAL: Set META_APP_SECRET to authenticate production WhatsApp webhooks.")


def seed_initial_admin_if_empty() -> None:
    """Create initial admin account if users table is empty (flags must_change_password in prod)."""
    db = SessionLocal()
    try:
        if db.scalar(select(User).limit(1)) is not None:
            return
        db.add(
            User(
                username=settings.bootstrap_admin_username,
                full_name="مدير النظام (Admin)",
                password_hash=hash_password(settings.bootstrap_admin_password),
                role="admin",
                is_active=True,
                must_change_password=(settings.app_env.lower() == "production"),
            )
        )
        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _validate_production_security()
    _migrate_sqlite_columns()
    seed_moka_menu_if_empty()
    db_bridge.sync_menu_sync()
    seed_initial_admin_if_empty()
    await recover_unflushed_inbound_messages()
    worker_task = asyncio.create_task(feedback_worker())
    try:
        yield
    finally:
        worker_task.cancel()
        await inbound_debouncer.flush_all()


_space_host = os.getenv("SPACE_HOST", "").strip()
if _space_host and settings.dashboard_public_url == "http://127.0.0.1:8000":
    settings.dashboard_public_url = f"https://{_space_host}"

_cors_origins = list(
    dict.fromkeys(
        o
        for o in [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            f"https://{_space_host}" if _space_host else "",
            settings.dashboard_public_url.rstrip("/"),
        ]
        if o
    )
)
app = FastAPI(title="Wasla — Restaurant AI Dispatcher", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sliding-window rate limiter + security headers middleware
_RATE_BUCKETS: dict[str, list[float]] = {}


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    import time

    if request.url.path in ("/api/ai/chat", "/api/orders"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = [t for t in _RATE_BUCKETS.get(client_ip, []) if now - t < 60.0]
        if len(bucket) >= settings.rate_limit_per_minute:
            return PlainTextResponse("Rate limit exceeded", status_code=429)
        bucket.append(now)
        _RATE_BUCKETS[client_ip] = bucket

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    if os.getenv("SPACE_ID") and request.url.path == "/":
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
    else:
        response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

if (DASHBOARD_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_DIR / "assets"), name="assets")


class CartItem(BaseModel):
    item_id: int
    quantity: int = Field(ge=1, le=50)
    size_code: str | None = None
    modifiers: list[dict] = Field(default_factory=list)
    selected_options: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class ConfirmOrder(BaseModel):
    customer_name: str = Field(min_length=2)
    customer_phone: str = Field(min_length=8)
    backup_phone: str = Field(min_length=8)
    delivery_address: str = Field(min_length=5)
    zone_name: str = "default"
    items: list[CartItem] = Field(min_length=1)
    idempotency_key: str = Field(min_length=4, max_length=128)
    confirmed_by_customer: bool
    data_source: str = "api"
    is_test: bool = False


class Directive(BaseModel):
    directive: str
    directive_type: str = "free_text"
    extra_delay_minutes: int | None = None


class StatusUpdate(BaseModel):
    status: OrderStatus


class CustomerOrderStatusRequest(BaseModel):
    customer_phone: str = Field(min_length=8, max_length=32)
    order_id: int | None = Field(default=None, ge=1)


class OrderModification(BaseModel):
    message: str


class FeedbackInput(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    issue_reported: str = ""


class ComplaintResolveInput(BaseModel):
    resolution_note: str = ""
    release_human_takeover: bool = True


class TakeoverToggleInput(BaseModel):
    active: bool


class NotificationSettingsUpdate(BaseModel):
    upsell_enabled: bool | None = None
    complaint_escalation_enabled: bool | None = None
    timezone_name: str | None = None
    review_delay_minutes: int | None = Field(default=None, ge=5, le=1440)
    arrival_check_delay_minutes: int | None = Field(default=None, ge=15, le=1440)


class AvailabilityUpdate(BaseModel):
    is_available: bool


class MenuSizePriceUpdate(BaseModel):
    code: str = Field(min_length=1, max_length=16)
    name: str | None = Field(default=None, max_length=80)
    price: Decimal = Field(gt=0)


class PriceUpdate(BaseModel):
    base_price: Decimal = Field(gt=0)
    sizes: list[MenuSizePriceUpdate] | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=2)
    password: str = Field(min_length=4)


class CreateStaffRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8)
    role: str = Field(default="cashier", pattern="^(admin|cashier)$")


class AIChatRequest(BaseModel):
    phone_number: str = Field(min_length=8)
    message: str = Field(min_length=1)


ALLOWED_TRANSITIONS = {
    # Legacy / backward-compatible 7-state transitions + commercial lifecycle aliases
    OrderStatus.DRAFTING: {OrderStatus.PENDING_CONFIRMATION, OrderStatus.SUBMITTED, OrderStatus.CONFIRMED, OrderStatus.CANCELLED},
    OrderStatus.PENDING_CONFIRMATION: {OrderStatus.CONFIRMED, OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.SUBMITTED: {OrderStatus.ACCEPTED, OrderStatus.IN_KITCHEN, OrderStatus.PREPARING, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.CONFIRMED: {OrderStatus.ACCEPTED, OrderStatus.IN_KITCHEN, OrderStatus.PREPARING, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.ACCEPTED: {OrderStatus.PREPARING, OrderStatus.IN_KITCHEN, OrderStatus.READY, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.IN_KITCHEN: {OrderStatus.READY, OrderStatus.DISPATCHED, OrderStatus.OUT_FOR_DELIVERY, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.PREPARING: {OrderStatus.READY, OrderStatus.DISPATCHED, OrderStatus.OUT_FOR_DELIVERY, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.READY: {OrderStatus.OUT_FOR_DELIVERY, OrderStatus.DISPATCHED, OrderStatus.DELIVERED, OrderStatus.FULFILLED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.DISPATCHED: {OrderStatus.UNREACHABLE_ALERT, OrderStatus.FULFILLED, OrderStatus.DELIVERED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.OUT_FOR_DELIVERY: {OrderStatus.UNREACHABLE_ALERT, OrderStatus.DELIVERED, OrderStatus.FULFILLED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.UNREACHABLE_ALERT: {OrderStatus.DISPATCHED, OrderStatus.OUT_FOR_DELIVERY, OrderStatus.FULFILLED, OrderStatus.DELIVERED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.FULFILLED: set(),
    OrderStatus.DELIVERED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.FAILED: set(),
}


def _stamp_order_lifecycle(order: Order, new_status: OrderStatus, actor: str, now: datetime) -> None:
    if new_status in (OrderStatus.CONFIRMED, OrderStatus.SUBMITTED):
        order.confirmed_at = order.confirmed_at or now
    elif new_status == OrderStatus.ACCEPTED:
        order.accepted_at = now
    elif new_status in (OrderStatus.PREPARING, OrderStatus.IN_KITCHEN):
        order.preparing_at = now
    elif new_status == OrderStatus.READY:
        order.ready_at = now
    elif new_status in (OrderStatus.OUT_FOR_DELIVERY, OrderStatus.DISPATCHED):
        order.out_for_delivery_at = now
    elif new_status in (OrderStatus.DELIVERED, OrderStatus.FULFILLED):
        order.delivered_at = now
        order.delivery_confirmed_by = actor
    elif new_status == OrderStatus.CANCELLED:
        order.cancelled_at = now
    elif new_status == OrderStatus.FAILED:
        order.failed_at = now


async def _create_order_in_db(payload: ConfirmOrder, db: Session) -> dict:
    if not payload.confirmed_by_customer:
        raise HTTPException(
            status_code=400,
            detail="The customer must confirm the complete order before dispatch",
        )

    existing = db.scalar(select(Order).where(Order.idempotency_key == payload.idempotency_key))
    if existing is not None:
        if existing.customer_phone != payload.customer_phone:
            raise HTTPException(status_code=409, detail="Idempotency key belongs to a different customer.")
        _associate_active_order(payload.customer_phone, existing.id, db)
        db.commit()
        return {"id": existing.id, "status": existing.status, "duplicate": True}

    resolved_items = []
    for cart_item in payload.items:
        menu_item = db.get(MenuItem, cart_item.item_id)
        if not menu_item or not menu_item.is_available:
            raise HTTPException(status_code=409, detail=f"Item {cart_item.item_id} is unavailable")
        unit_price = str(menu_item.base_price)
        item_name = menu_item.name
        menu_config = menu_item.modifiers_json if isinstance(menu_item.modifiers_json, dict) else {}
        sizes = menu_config.get("sizes", [])
        if cart_item.size_code:
            size = next(
                (sz for sz in sizes if str(sz.get("code", "")).upper() == cart_item.size_code.upper()),
                None,
            )
            if size is None:
                raise HTTPException(status_code=400, detail=f"Invalid size for item {cart_item.item_id}")
            unit_price = str(size["price"])
            item_name = f"{menu_item.name} ({size['code']})"

        product_id = menu_config.get("product_id")
        product = db_bridge.menu_index.get_product(str(product_id)) if product_id else None
        allowed_modifier_ids = set(product.modifiers) if product else set()
        resolved_modifiers = []
        seen_modifier_ids = set()
        for selection in cart_item.modifiers:
            modifier_id = str(selection.get("modifier_id", "")).strip()
            modifier = db_bridge.menu_index.get_modifier(modifier_id)
            if (
                not modifier_id
                or modifier is None
                or modifier_id not in allowed_modifier_ids
                or modifier_id in seen_modifier_ids
            ):
                raise HTTPException(status_code=400, detail=f"Invalid modifier for item {cart_item.item_id}")
            seen_modifier_ids.add(modifier_id)
            stored_extra = next(
                (
                    extra
                    for extra in menu_config.get("extras", [])
                    if extra.get("modifier_id") == modifier_id
                    or (not extra.get("modifier_id") and extra.get("name") == modifier.name_ar)
                ),
                None,
            )
            if stored_extra is None:
                raise HTTPException(status_code=400, detail=f"Modifier unavailable for item {cart_item.item_id}")
            resolved_modifiers.append(
                {
                    "modifier_id": modifier_id,
                    "name": str(stored_extra["name"]),
                    "price": str(Decimal(str(stored_extra["price"]))),
                }
            )

        option_groups = {
            str(option.get("option_group")): option
            for option in menu_config.get("options", [])
            if isinstance(option, dict) and option.get("option_group")
        }
        selected_options = cart_item.selected_options
        if any(group not in option_groups for group in selected_options):
            raise HTTPException(status_code=400, detail=f"Invalid option for item {cart_item.item_id}")
        for group, option in option_groups.items():
            selected = selected_options.get(group)
            if option.get("required", True) and not selected:
                raise HTTPException(status_code=400, detail=f"Missing required option for item {cart_item.item_id}")
            if selected and selected not in option.get("choices", []):
                raise HTTPException(status_code=400, detail=f"Invalid option for item {cart_item.item_id}")

        resolved_items.append(
            {
                "item_id": menu_item.id,
                "name": item_name,
                "base_price": unit_price,
                "quantity": cart_item.quantity,
                "modifiers": resolved_modifiers,
                "selected_options": selected_options,
                "notes": cart_item.notes,
            }
        )

    delivery_fee = _delivery_quote(payload.zone_name)["delivery_fee"]
    cart = calculate_cart(resolved_items, Decimal(delivery_fee))
    now = datetime.now(timezone.utc)
    order = Order(
        customer_name=payload.customer_name,
        customer_phone=payload.customer_phone,
        idempotency_key=payload.idempotency_key,
        backup_phone=payload.backup_phone,
        confirmed_by_customer=True,
        delivery_address=payload.delivery_address,
        subtotal=Decimal(cart["subtotal"]),
        delivery_fee=Decimal(cart["delivery_fee"]),
        total_amount=Decimal(cart["total_amount"]),
        status=OrderStatus.SUBMITTED,
        items_snapshot=cart["items"],
        data_source=payload.data_source,
        is_test=payload.is_test,
        confirmed_at=now,
    )
    db.add(order)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(select(Order).where(Order.idempotency_key == payload.idempotency_key))
        if existing is None:
            raise
        if existing.customer_phone != payload.customer_phone:
            raise HTTPException(status_code=409, detail="Idempotency key belongs to a different customer.")
        _associate_active_order(payload.customer_phone, existing.id, db)
        db.commit()
        return {"id": existing.id, "status": existing.status, "duplicate": True}
    _associate_active_order(payload.customer_phone, order.id, db)
    db.add_all(
        [
            OrderStatusHistory(
                order_id=order.id,
                from_status=OrderStatus.PENDING_CONFIRMATION.value,
                to_status=OrderStatus.SUBMITTED.value,
                actor="customer_confirmed",
                note="Confirmed by customer",
                changed_at=now,
            ),
            OrderEvent(order_id=order.id, event_type="created", payload={"status": "SUBMITTED"}),
            OrderEvent(order_id=order.id, event_type="idempotency_key", payload={"key": payload.idempotency_key}),
            Outbox(
                event_id=str(uuid.uuid4()),
                event_type="order_created",
                payload={"order_id": order.id, "total_amount": cart["total_amount"], "status": "SUBMITTED"},
            ),
        ]
    )
    # Schedule 90-minute fallback arrival check durably in DB
    job_scheduler.schedule_post_confirmation_arrival_check(db, order, confirmed_at=now)

    # Persist the order and a notification fallback before making any external delivery attempt.
    retry_job = job_scheduler.schedule_cashier_retry(db, order, delay_seconds=30)
    db.commit()

    cashier_notify = await notification_service.notify_cashier_new_order(db, order)
    if cashier_notify.get("status") != "FAILED":
        retry_job.status = "COMPLETED"
        retry_job.completed_at = datetime.now(timezone.utc)
    db.commit()
    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "order_created",
            "order_id": order.id,
            "timestamp": now.isoformat(),
            "schema_version": 1,
            "payload": {
                "customer_name": order.customer_name,
                "total_amount": str(order.total_amount),
                "status": order.status,
                "cashier_notification": cashier_notify,
            },
        }
    )
    if cashier_notify.get("status") == "FAILED":
        await ws_manager.broadcast(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "cashier_notification_failed",
                "order_id": order.id,
                "timestamp": now.isoformat(),
                "schema_version": 1,
                "payload": {"error": cashier_notify.get("error"), "retry_scheduled": True},
            }
        )
    return {"id": order.id, "status": order.status, "cart": cart, "cashier_notification": cashier_notify}


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    menu_count = len(db.scalars(select(MenuItem)).all())
    return {
        "status": "ok",
        "service": "wasla-ai-dispatcher",
        "menu_items_loaded": menu_count,
        "ai_model": wasla_config.ai_primary_model,
        "audio_mode": wasla_config.ai_audio_mode,
    }


@app.get("/api/ready")
def readiness(db: Session = Depends(get_db)):
    """Production readiness probe reporting live DB health & honest external/local AI status."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    ai_status = ai_dispatcher.get_runtime_status()
    return {
        "ready": db_ok,
        "database_dialect": engine.dialect.name,
        "database_connected": db_ok,
        "providers": {
            "gemini_llm_configured": ai_dispatcher.llm_client.is_configured,
            "groq_whisper_configured": bool(wasla_config.groq_api_key and not wasla_config.groq_api_key.startswith("your_")),
            "meta_whatsapp_configured": meta_client.is_configured,
            "meta_webhook_hmac_enabled": bool(settings.meta_app_secret),
            "telegram_configured": telegram_client.is_configured,
            "scheduler_cron_configured": bool(settings.scheduler_cron_secret.strip()),
            "finetuned_lora_checkpoint_exists": ai_status["checkpoint_exists"],
            "finetuned_lora_loaded": ai_status["model_loaded"],
            "finetuned_lora_version": ai_status["model_version"],
            "finetuned_lora_taxonomy_compatible": ai_status["taxonomy_compatible"],
            "finetuned_lora_inference_available": ai_status["inference_available"],
            "finetuned_lora_routing_available": ai_status["routing_available"],
            "active_inference_route": ai_status["active_inference_route"],
        },
        "ai_models": ai_status,
    }


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == payload.username.strip(), User.is_active.is_(True)))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")
    token = create_access_token(user)
    log_audit(db, user.username, "login", target=user.username)
    db.commit()
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "full_name": user.full_name, "role": user.role},
    }


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=4)
    new_password: str = Field(min_length=8)


@app.post("/api/auth/change-password")
def change_password(
    payload: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db_user = db.get(User, user.id)
    if not db_user or not verify_password(payload.current_password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="كلمة المرور الحالية غير صحيحة")
    if payload.new_password == payload.current_password:
        raise HTTPException(status_code=400, detail="كلمة المرور الجديدة يجب أن تختلف عن الحالية")
    db_user.password_hash = hash_password(payload.new_password)
    db_user.must_change_password = False
    log_audit(db, db_user.username, "change_password", target=db_user.username)
    db.commit()
    return {"status": "password_updated", "must_change_password": False}


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "must_change_password": getattr(user, "must_change_password", False),
    }


@app.get("/api/staff")
def list_staff(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.scalars(select(User).order_by(User.id.asc())).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@app.post("/api/staff", status_code=201)
def create_staff(payload: CreateStaffRequest, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    existing = db.scalar(select(User).where(User.username == payload.username.strip()))
    if existing:
        raise HTTPException(status_code=409, detail="اسم المستخدم مسجل بالفعل")
    new_user = User(
        username=payload.username.strip(),
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(new_user)
    log_audit(db, admin.username, "create_staff", target=new_user.username, details={"role": new_user.role})
    db.commit()
    return {"id": new_user.id, "username": new_user.username, "full_name": new_user.full_name, "role": new_user.role}


@app.get("/api/audit-logs")
def list_audit_logs(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(100)).all()
    return [
        {
            "id": l.id,
            "actor_username": l.actor_username,
            "action": l.action,
            "target": l.target,
            "details": l.details,
            "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]


@app.get("/api/conversations")
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.scalars(select(ChatSession).order_by(ChatSession.id.desc())).all()
    return [
        {
            "id": s.id,
            "phone_number": s.phone_number,
            "active_order_id": s.active_order_id,
            "needs_human_takeover": s.needs_human_takeover,
            "conversation_state": s.conversation_state,
            "pending_cart": s.pending_cart,
            "latest_directive": s.latest_directive,
        }
        for s in sessions
    ]


@app.post("/api/conversations/{phone_number}/resolve-handoff")
def resolve_handoff(phone_number: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    session.needs_human_takeover = False
    log_audit(db, user.username, "resolve_handoff", target=phone_number)
    db.commit()
    return {"phone_number": phone_number, "needs_human_takeover": False}


@app.get("/api/metrics")
def get_metrics(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Real SQL-backed operational metrics — strictly excludes simulator/test orders from production totals."""
    all_orders = db.scalars(select(Order)).all()
    real_orders = [o for o in all_orders if not getattr(o, "is_test", False)]
    test_orders = [o for o in all_orders if getattr(o, "is_test", False)]

    menu_items = db.scalars(select(MenuItem)).all()
    sessions = db.scalars(select(ChatSession)).all()
    feedbacks = db.scalars(select(FeedbackRequest).where(FeedbackRequest.rating.is_not(None))).all()

    by_status = {s.value: 0 for s in OrderStatus}
    revenue = Decimal("0.00")
    for o in real_orders:
        st = o.status.value if isinstance(o.status, OrderStatus) else str(o.status)
        by_status[st] = by_status.get(st, 0) + 1
        if st != OrderStatus.CANCELLED.value:
            revenue += Decimal(str(o.total_amount))

    avg_rating = round(sum(f.rating for f in feedbacks) / len(feedbacks), 2) if feedbacks else None
    available_count = sum(1 for m in menu_items if m.is_available)
    handoff_count = sum(1 for s in sessions if s.needs_human_takeover)

    return {
        "total_orders": len(real_orders),
        "test_or_demo_orders": len(test_orders),
        "by_status": by_status,
        "total_revenue_egp": str(revenue.quantize(Decimal("0.01"))),
        "menu_total": len(menu_items),
        "menu_available": available_count,
        "menu_out_of_stock": len(menu_items) - available_count,
        "active_conversations": len(sessions),
        "human_handoffs": handoff_count,
        "average_rating": avg_rating,
        "ratings_count": len(feedbacks),
    }


@app.get("/api/menu")
def get_menu(db: Session = Depends(get_db)):
    items = db.scalars(select(MenuItem).where(MenuItem.is_available.is_(True))).all()
    return [
        {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "base_price": str(item.base_price),
            "modifiers": item.modifiers_json,
        }
        for item in items
    ]


@app.get("/api/menu/all")
def get_all_menu_items(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db.scalars(select(MenuItem).order_by(MenuItem.id.asc())).all()
    return [
        {
            "id": item.id,
            "product_id": (item.modifiers_json or {}).get("product_id"),
            "name": item.name,
            "category": item.category,
            "base_price": str(item.base_price),
            "is_available": item.is_available,
            "modifiers": item.modifiers_json,
        }
        for item in items
    ]


@app.patch("/api/menu/{item_id}/availability")
async def toggle_menu_availability(
    item_id: int,
    payload: AvailabilityUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = db.get(MenuItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    item.is_available = payload.is_available
    log_audit(db, user.username, "toggle_availability", target=str(item.id), details={"is_available": payload.is_available})
    db.commit()

    db_bridge.sync_menu_sync()

    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "menu_updated",
            "item_id": item.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {"name": item.name, "is_available": item.is_available},
        }
    )
    return {"id": item.id, "name": item.name, "is_available": item.is_available}


@app.patch("/api/menu/{item_id}/price")
async def update_menu_price(
    item_id: int,
    payload: PriceUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    item = db.get(MenuItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    old_price = str(item.base_price)
    item.base_price = payload.base_price

    mods = dict(item.modifiers_json or {})
    if payload.sizes is not None:
        normalized_sizes = []
        for sz in payload.sizes:
            normalized_sizes.append(
                {
                    "code": sz.code,
                    "name": sz.name or sz.code,
                    "price": str(sz.price.quantize(Decimal("0.01"))),
                }
            )
        mods["sizes"] = normalized_sizes
        item.modifiers_json = mods
    elif isinstance(mods.get("sizes"), list) and mods["sizes"]:
        updated_sizes = [dict(sz) for sz in mods["sizes"]]
        updated_sizes[0]["price"] = str(Decimal(str(payload.base_price)).quantize(Decimal("0.01")))
        mods["sizes"] = updated_sizes
        item.modifiers_json = mods

    log_audit(db, admin.username, "update_price", target=str(item.id), details={"old": old_price, "new": str(payload.base_price)})
    db.commit()
    db_bridge.sync_menu_sync()

    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "menu_updated",
            "item_id": item.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {"name": item.name, "base_price": str(item.base_price), "sizes": mods.get("sizes", [])},
        }
    )
    return {"id": item.id, "name": item.name, "base_price": str(item.base_price), "sizes": mods.get("sizes", [])}


async def _persist_ai_interaction(
    db: Session,
    phone_number: str,
    customer_text: str,
    resp: Any,
) -> dict[str, Any]:
    """Persist chat session, upsell analytics, and immediate complaint tickets + owner escalation."""
    now = datetime.now(timezone.utc)
    session_row = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
    if session_row is None:
        session_row = ChatSession(phone_number=phone_number)
        db.add(session_row)
    session_row.last_customer_message_at = now
    session_row.pending_cart = resp.cart.model_dump(mode="json")
    if resp.order_id:
        session_row.active_order_id = resp.order_id
    if resp.needs_human_takeover:
        session_row.needs_human_takeover = True
        session_row.human_takeover_at = session_row.human_takeover_at or now
    existing_conv = dict(session_row.conversation_state or {})
    existing_conv.update(
        {
            "last_intent": resp.intent.value,
            "requires_clarification": resp.requires_clarification,
            "model_used": resp.model_used,
            "suppress_ai_reply": resp.suppress_ai_reply,
        }
    )
    session_row.conversation_state = existing_conv

    # Record genuine recommendation/upsell analytics
    if resp.upsell_event:
        ev = resp.upsell_event
        pids = ev.get("accepted_product_ids") or ev.get("offered_product_ids") or []
        db.add(
            RecommendationEvent(
                customer_phone=phone_number,
                order_id=resp.order_id or session_row.active_order_id,
                offer_type=ev.get("offer_type") or "separate_items",
                action=ev.get("action") or "offered",
                product_ids=pids,
                added_revenue=Decimal(str(ev.get("added_revenue", "0.00"))),
            )
        )

    escalation_result: dict[str, Any] | None = None
    if resp.complaint_assessment:
        ca = resp.complaint_assessment
        # Check for existing open complaint ticket for this customer to avoid duplicate notifications
        existing_ticket = db.scalar(
            select(ComplaintTicket)
            .where(
                ComplaintTicket.customer_phone == phone_number,
                ComplaintTicket.status != "RESOLVED",
            )
            .order_by(ComplaintTicket.created_at.desc())
        )
        excerpt = ca.get("excerpt") or customer_text[:300]
        if existing_ticket is not None:
            excerpts = list(existing_ticket.sanitized_excerpts or [])
            excerpts.append(excerpt)
            existing_ticket.sanitized_excerpts = excerpts[-10:]
            is_critical_update = bool(ca.get("critical_update") or ca.get("severity") == "CRITICAL")
            if is_critical_update and existing_ticket.severity != "CRITICAL":
                existing_ticket.severity = "CRITICAL"
            existing_ticket.human_takeover_active = True
            ticket = existing_ticket
            escalation_result = await notification_service.escalate_complaint_to_owner(
                db, ticket, critical_update=is_critical_update
            )
        else:
            latest_order = db.scalar(
                select(Order)
                .where(Order.customer_phone == phone_number)
                .order_by(Order.created_at.desc())
            )
            ticket = ComplaintTicket(
                order_id=resp.order_id or session_row.active_order_id or (latest_order.id if latest_order else None),
                customer_phone=phone_number,
                customer_name=latest_order.customer_name if latest_order else "",
                category=ca.get("category", "service_issue"),
                severity=ca.get("severity", "HIGH"),
                summary=ca.get("summary_en", "Customer complaint escalated."),
                sanitized_excerpts=[excerpt],
                status="OPEN",
                human_takeover_active=True,
            )
            db.add(ticket)
            db.flush()
            escalation_result = await notification_service.escalate_complaint_to_owner(db, ticket)

        if escalation_result and escalation_result.get("status") == "FAILED":
            job_scheduler.schedule_owner_complaint_retry(db, ticket, delay_seconds=30)

        session_row.needs_human_takeover = True
        session_row.human_takeover_by = session_row.human_takeover_by or "complaint_auto_escalation"
        session_row.human_takeover_at = session_row.human_takeover_at or now
        conv_upd = dict(session_row.conversation_state or {})
        if isinstance(conv_upd.get("session_v2"), dict):
            sv2 = dict(conv_upd["session_v2"])
            sv2["needs_human_takeover"] = True
            sv2["human_takeover_by"] = session_row.human_takeover_by
            conv_upd["session_v2"] = sv2
            session_row.conversation_state = conv_upd
        ai_dispatcher.set_human_takeover(phone_number, True, actor="complaint_auto_escalation", persist=False)
        db.flush()
        await ws_manager.broadcast(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "complaint_escalated",
                "order_id": ticket.order_id,
                "timestamp": now.isoformat(),
                "schema_version": 1,
                "payload": {
                    "ticket_id": ticket.id,
                    "category": ticket.category,
                    "severity": ticket.severity,
                    "escalation": escalation_result,
                },
            }
        )

    db.commit()
    return {"escalation": escalation_result}


@app.post("/api/ai/chat")
async def ai_chat(payload: AIChatRequest, db: Session = Depends(get_db)):
    """Disabled in production (no simulator in production)."""
    if settings.app_env.lower() == "production":
        raise HTTPException(status_code=404, detail="Not found")
    existing_session = db.scalar(select(ChatSession).where(ChatSession.phone_number == payload.phone_number))
    if existing_session is not None:
        ai_dispatcher.set_human_takeover(
            payload.phone_number, bool(existing_session.needs_human_takeover), actor=existing_session.human_takeover_by
        )
    resp = await ai_dispatcher.handle_text(
        payload.phone_number,
        payload.message,
        data_source="simulator",
        is_test=True,
    )
    side = await _persist_ai_interaction(db, payload.phone_number, payload.message, resp)
    out = resp.model_dump(mode="json")
    if side.get("escalation"):
        out["complaint_escalation"] = side["escalation"]
    return out


@app.post("/api/orders", status_code=201)
async def confirm_order(payload: ConfirmOrder, db: Session = Depends(get_db)):
    return await _create_order_in_db(payload, db)


@app.get("/api/delivery/quote")
def delivery_quote(zone_name: str = Query(default="default")):
    return _delivery_quote(zone_name)


@app.post("/api/customer/order-status")
def customer_order_status(
    payload: CustomerOrderStatusRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    expected_key = settings.backend_service_api_key
    supplied_key = request.headers.get("X-Backend-Service-Key", "")
    if not expected_key:
        raise HTTPException(status_code=503, detail="Customer order tracking service authentication is not configured.")
    if not supplied_key or not hmac.compare_digest(supplied_key, expected_key):
        raise HTTPException(status_code=401, detail="Backend service authentication required.")

    stmt = select(Order).where(Order.customer_phone == payload.customer_phone)
    if settings.app_env.lower() == "production":
        stmt = stmt.where(Order.is_test.is_(False))
    if payload.order_id is not None:
        order = db.scalar(stmt.where(Order.id == payload.order_id))
    else:
        session_row = db.scalar(select(ChatSession).where(ChatSession.phone_number == payload.customer_phone))
        if session_row is not None and session_row.active_order_id is not None:
            order = db.scalar(stmt.where(Order.id == session_row.active_order_id))
        else:
            order = db.scalar(stmt.order_by(Order.created_at.desc(), Order.id.desc()).limit(1))
    if order is None:
        return {"id": None, "status": None}
    status = getattr(order.status, "value", order.status)
    return {"id": order.id, "status": str(status)}


@app.get("/api/orders")
def list_orders(
    include_test: bool = Query(default=False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = select(Order).order_by(Order.created_at.desc())
    if not include_test or settings.app_env.lower() == "production":
        stmt = stmt.where(Order.is_test.is_(False))
    orders = db.scalars(stmt).all()
    result = []
    for order in orders:
        feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.order_id == order.id))
        modification_events = db.scalars(
            select(OrderEvent)
            .where(OrderEvent.order_id == order.id, OrderEvent.event_type == "customer_modification")
            .order_by(OrderEvent.created_at.desc())
        ).all()
        latest_modification = modification_events[0].payload if modification_events else None
        result.append(
            {
                "id": order.id,
                "customer_name": order.customer_name,
                "customer_phone": order.customer_phone,
                "backup_phone": getattr(order, "backup_phone", ""),
                "confirmed_by_customer": getattr(order, "confirmed_by_customer", False),
                "delivery_address": order.delivery_address,
                "subtotal": str(order.subtotal),
                "delivery_fee": str(order.delivery_fee),
                "total_amount": str(order.total_amount),
                "status": order.status,
                "items": order.items_snapshot,
                "data_source": getattr(order, "data_source", "api"),
                "is_test": getattr(order, "is_test", False),
                "created_at": order.created_at.isoformat(),
                "feedback": None
                if feedback is None
                else {
                    "due_at": feedback.due_at.isoformat(),
                    "sent": feedback.sent_at is not None,
                    "rating": feedback.rating,
                    "comment": feedback.comment,
                },
                "latest_modification": latest_modification,
            }
        )
    return result


@app.patch("/api/orders/{order_id}/status")
async def update_status(
    order_id: int,
    payload: StatusUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    current = OrderStatus(order.status)
    if payload.status not in ALLOWED_TRANSITIONS[current]:
        raise HTTPException(status_code=409, detail=f"Cannot change {current.value} to {payload.status.value}")
    now = datetime.now(timezone.utc)
    order.status = payload.status
    _stamp_order_lifecycle(order, payload.status, actor=user.username, now=now)
    log_audit(db, user.username, "update_order_status", target=str(order.id), details={"status": payload.status.value})
    db.add(
        OrderStatusHistory(
            order_id=order.id,
            from_status=current.value,
            to_status=payload.status.value,
            actor=user.username,
            changed_at=now,
        )
    )
    db.add(OrderEvent(order_id=order.id, event_type="status_changed", payload={"status": payload.status.value}))
    db.add(
        Outbox(
            event_id=str(uuid.uuid4()),
            event_type="order_status_changed",
            payload={"order_id": order.id, "status": payload.status.value},
        )
    )
    if payload.status.value in DELIVERED_STATUSES:
        job_scheduler.schedule_post_delivery_review(db, order, delivered_at=now)
    db.commit()
    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "order_status_changed",
            "order_id": order.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {"status": payload.status.value},
        }
    )
    return {"id": order.id, "status": order.status}


@app.post("/api/orders/{order_id}/instruct")
async def instruct_customer(
    order_id: int,
    payload: Directive,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if not payload.directive.strip():
        raise HTTPException(status_code=400, detail="Directive cannot be empty")

    log_audit(db, user.username, "cashier_directive", target=str(order.id), details={"type": payload.directive_type})

    interpreted = ai_dispatcher.interpret_cashier_directive(
        CashierDirectiveRequest(
            order_id=order.id,
            directive_type=payload.directive_type,
            directive_text=payload.directive,
            extra_delay_minutes=payload.extra_delay_minutes,
            authorized_by_cashier=True,
        )
    )

    db.add(
        OrderEvent(
            order_id=order.id,
            event_type="cashier_directive",
            payload={
                "directive": payload.directive,
                "customer_message_ar": interpreted.customer_message_ar,
                "warnings": interpreted.warnings,
            },
        )
    )
    db.add(
        Outbox(
            event_id=str(uuid.uuid4()),
            event_type="cashier_directive_ready",
            payload={
                "order_id": order.id,
                "customer_phone": order.customer_phone,
                "customer_message_ar": interpreted.customer_message_ar,
            },
        )
    )
    db.commit()
    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "customer_alert",
            "order_id": order.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {
                "directive": payload.directive,
                "customer_message_ar": interpreted.customer_message_ar,
            },
        }
    )
    delivery_result = await meta_client.send_text_message(
        order.customer_phone,
        interpreted.customer_message_ar,
    )
    return {
        "order_id": order.id,
        "status": "directed",
        "message": payload.directive,
        "customer_message_ar": interpreted.customer_message_ar,
        "warnings": interpreted.warnings,
        "requires_cashier_approval": interpreted.requires_cashier_approval,
        "whatsapp_delivery": delivery_result,
    }


@app.post("/api/orders/{order_id}/modification")
async def notify_customer_modification(order_id: int, payload: OrderModification, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    message = payload.message.strip()
    if len(message) < 2:
        raise HTTPException(status_code=400, detail="Modification details are required")
    db.add(OrderEvent(order_id=order.id, event_type="customer_modification", payload={"message": message}))
    db.commit()
    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "order_modified",
            "order_id": order.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {"message": message},
        }
    )
    return {"order_id": order.id, "status": "modification_received", "message": message}


@app.post("/api/feedback/{token}")
async def submit_feedback(token: str, payload: FeedbackInput, db: Session = Depends(get_db)):
    feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.token == token))
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback link not found")
    if feedback.submitted_at is not None:
        raise HTTPException(status_code=409, detail="Feedback already submitted")
    if feedback.sent_at is None:
        raise HTTPException(status_code=409, detail="Feedback period has not started")
    feedback.rating = payload.rating
    feedback.comment = payload.comment.strip()
    feedback.issue_reported = payload.issue_reported.strip() or None
    feedback.submitted_at = datetime.now(timezone.utc)
    db.add(
        OrderEvent(
            order_id=feedback.order_id,
            event_type="feedback_submitted",
            payload={
                "rating": payload.rating,
                "comment": feedback.comment,
                "issue_reported": feedback.issue_reported,
            },
        )
    )
    # If customer reported an issue with a low rating, open a complaint ticket and escalate to owner
    if feedback.issue_reported or payload.rating <= 2:
        order = db.get(Order, feedback.order_id)
        if order:
            ticket = ComplaintTicket(
                order_id=order.id,
                customer_phone=order.customer_phone,
                customer_name=order.customer_name,
                category="post_order_review_issue",
                severity="HIGH" if payload.rating == 1 else "MEDIUM",
                summary=f"Low rating ({payload.rating}/5) review with issue reported.",
                sanitized_excerpts=[feedback.issue_reported or feedback.comment or f"Rating {payload.rating}/5"],
                status="OPEN",
                human_takeover_active=False,
            )
            db.add(ticket)
            db.flush()
            fb_esc = await notification_service.escalate_complaint_to_owner(db, ticket)
            if fb_esc.get("status") == "FAILED":
                job_scheduler.schedule_owner_complaint_retry(db, ticket, delay_seconds=30)
    db.commit()
    return {"status": "submitted", "rating": payload.rating, "issue_reported": feedback.issue_reported}


@app.get("/api/webhook")
def verify_whatsapp_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
):
    """Official Meta WhatsApp Cloud API webhook verification handshake."""
    if hub_mode == "subscribe" and hub_verify_token == settings.meta_verify_token and hub_challenge:
        return PlainTextResponse(hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Webhook verification failed")


def _insert_processed_message(db: Session, message_id: str) -> bool:
    """Atomically claim an inbound provider update by its unique database key."""
    values = {
        "meta_message_id": message_id,
        "processed_at": datetime.now(timezone.utc),
    }
    dialect = db.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert

        statement = insert(ProcessedMessage).values(**values).on_conflict_do_nothing(
            index_elements=[ProcessedMessage.meta_message_id]
        )
        return db.execute(statement).rowcount == 1
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        statement = insert(ProcessedMessage).values(**values).on_conflict_do_nothing(
            index_elements=[ProcessedMessage.meta_message_id]
        )
        return db.execute(statement).rowcount == 1

    if db.scalar(select(ProcessedMessage.id).where(ProcessedMessage.meta_message_id == message_id)):
        return False
    db.add(ProcessedMessage(**values))
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        if db.scalar(select(ProcessedMessage.id).where(ProcessedMessage.meta_message_id == message_id)):
            return False
        raise
    return True


def _parse_webhook_json(raw_body: bytes) -> dict:
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON webhook body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Webhook body must be a JSON object")
    return body


async def _process_debounced_whatsapp_turn(sender: str, combined_text: str) -> dict:
    """Execute one aggregated WhatsApp customer turn safely inside a fresh DB session."""
    db = SessionLocal()
    try:
        buffered_rows = db.scalars(
            select(InboundMessageBuffer)
            .where(
                InboundMessageBuffer.phone_number == sender,
                InboundMessageBuffer.status.in_(("BUFFERED", "PROCESSING")),
            )
            .order_by(InboundMessageBuffer.msg_timestamp.asc(), InboundMessageBuffer.id.asc())
        ).all()
        for row in buffered_rows:
            row.status = "PROCESSING"
        db.commit()

        final_text = combined_text.strip()
        if buffered_rows:
            db_combined = "\n".join(r.content.strip() for r in buffered_rows if r.content and r.content.strip())
            if len(db_combined) >= len(final_text):
                final_text = db_combined

        existing_sess = db.scalar(select(ChatSession).where(ChatSession.phone_number == sender))
        if existing_sess is not None:
            ai_dispatcher.set_human_takeover(
                sender,
                bool(existing_sess.needs_human_takeover),
                actor=existing_sess.human_takeover_by,
                persist=False,
            )

        ai_resp = await ai_dispatcher.handle_text(
            sender,
            final_text,
            data_source="whatsapp_webhook",
            is_test=False,
        )
        side = await _persist_ai_interaction(db, sender, final_text, ai_resp)
        if ai_resp.suppress_ai_reply:
            delivery = {"sent": False, "mode": "suppressed_human_takeover"}
        else:
            delivery = await meta_client.send_text_message(sender, ai_resp.reply_ar)

        now = datetime.now(timezone.utc)
        for row in buffered_rows:
            row.status = "PROCESSED"
            row.processed_at = now
        db.commit()

        return {
            "to": sender,
            "reply_ar": ai_resp.reply_ar,
            "intent": ai_resp.intent.value,
            "suppress_ai_reply": ai_resp.suppress_ai_reply,
            "whatsapp_delivery": delivery,
            "complaint_escalation": side.get("escalation"),
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def recover_unflushed_inbound_messages() -> int:
    """Replay any buffered WhatsApp inbound messages orphaned by a worker restart."""
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(InboundMessageBuffer)
            .where(InboundMessageBuffer.status.in_(("BUFFERED", "PROCESSING")))
            .order_by(InboundMessageBuffer.id.asc())
        ).all()
        grouped: dict[str, list[str]] = {}
        for r in rows:
            grouped.setdefault(r.phone_number, []).append(r.content)
    finally:
        db.close()

    recovered = 0
    for sender, msgs in grouped.items():
        combined = "\n".join(m.strip() for m in msgs if m and m.strip())
        if combined:
            await _process_debounced_whatsapp_turn(sender, combined)
            recovered += len(msgs)
    return recovered


@app.post("/api/webhook")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    raw_body = await request.body()

    # Enforce Meta HMAC-SHA256 signature verification whenever META_APP_SECRET is configured
    if settings.meta_app_secret:
        sig_header = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(
            settings.meta_app_secret.encode("utf-8"), raw_body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig_header, expected):
            raise HTTPException(status_code=401, detail="Invalid Meta webhook signature")

    body = _parse_webhook_json(raw_body)
    entries = body.get("entry", [])
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook entry list")
    processed = 0
    status_updates = 0
    text_dispatch_items: list[tuple[str, str, int | None, bool, str]] = []
    audio_items: list[tuple[str, str]] = []

    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("changes", []), list):
            raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook entry")
        for change in entry.get("changes", []):
            if not isinstance(change, dict):
                raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook change")
            value = change.get("value", {})
            if not isinstance(value, dict):
                raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook value")
            statuses = value.get("statuses", [])
            messages = value.get("messages", [])
            if not isinstance(statuses, list) or not isinstance(messages, list):
                raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook messages")
            for st in statuses:
                if not isinstance(st, dict):
                    raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook status")
                st_id = st.get("id")
                st_val = st.get("status", "")
                if not st_id or not st_val:
                    continue
                dedup_key = f"status:{st_id}:{st_val}"
                if not _insert_processed_message(db, dedup_key):
                    continue
                db.add(
                    Outbox(
                        event_id=str(uuid.uuid4()),
                        event_type="whatsapp_delivery_status",
                        payload={
                            "wamid": st_id,
                            "status": st_val,
                            "timestamp": st.get("timestamp"),
                            "errors": st.get("errors"),
                        },
                    )
                )
                status_updates += 1

            for message in messages:
                if not isinstance(message, dict):
                    raise HTTPException(status_code=400, detail="Invalid WhatsApp webhook message")
                meta_id = message.get("id")
                if not meta_id:
                    continue
                if not _insert_processed_message(db, meta_id):
                    continue
                processed += 1

                sender = message.get("from", "")
                msg_type = message.get("type", "")
                raw_ts = message.get("timestamp")
                try:
                    ts_int = int(raw_ts) if raw_ts is not None else None
                except (TypeError, ValueError):
                    ts_int = None

                if msg_type == "text" and sender:
                    text_body = (message.get("text") or {}).get("body", "").strip()
                    if text_body:
                        db.add(
                            InboundMessageBuffer(
                                message_uid=meta_id,
                                phone_number=sender,
                                content=text_body,
                                data_source="whatsapp_webhook",
                                msg_timestamp=ts_int,
                                status="BUFFERED",
                            )
                        )
                        pre_c = assess_complaint(text_body)
                        urgent = pre_c.is_complaint and pre_c.severity in ("CRITICAL", "HIGH")
                        text_dispatch_items.append((sender, text_body, ts_int, urgent, meta_id))
                elif msg_type == "audio" and sender:
                    media_id = (message.get("audio") or {}).get("id", "")
                    if media_id:
                        audio_items.append((sender, media_id))

    # Commit deduplication + durable inbound buffer BEFORE awaiting 3s debounce window
    db.commit()

    replies = []
    for sender, text_body, ts_int, urgent, _meta_id in text_dispatch_items:
        batch_reply, is_leader = await inbound_debouncer.enqueue_with_metadata(
            sender,
            text_body,
            _process_debounced_whatsapp_turn,
            urgent_bypass=urgent,
            msg_timestamp=ts_int,
        )
        if is_leader:
            replies.append(batch_reply)
        else:
            replies.append({"to": sender, "debounced_merged": True, "intent": batch_reply.get("intent")})

    for sender, media_id in audio_items:
        try:
            existing_sess = db.scalar(select(ChatSession).where(ChatSession.phone_number == sender))
            if existing_sess is not None:
                ai_dispatcher.set_human_takeover(
                    sender, bool(existing_sess.needs_human_takeover), actor=existing_sess.human_takeover_by
                )
            audio_bytes, mime_type = await meta_client.download_media(media_id)
            ai_resp = await ai_dispatcher.handle_audio(
                sender,
                AudioInput(audio_bytes=audio_bytes, mime_type=mime_type, filename=f"{media_id}.ogg"),
                data_source="whatsapp_webhook",
                is_test=False,
            )
            side = await _persist_ai_interaction(db, sender, ai_resp.transcript or "", ai_resp)
            if ai_resp.suppress_ai_reply:
                delivery = {"sent": False, "mode": "suppressed_human_takeover"}
            else:
                delivery = await meta_client.send_text_message(sender, ai_resp.reply_ar)
            replies.append(
                {
                    "to": sender,
                    "transcript": ai_resp.transcript,
                    "reply_ar": ai_resp.reply_ar,
                    "intent": ai_resp.intent.value,
                    "suppress_ai_reply": ai_resp.suppress_ai_reply,
                    "whatsapp_delivery": delivery,
                    "complaint_escalation": side.get("escalation"),
                }
            )
        except Exception:
            logger.exception("Failed to process inbound WhatsApp audio")
            replies.append({"to": sender, "audio_error": "Unable to process audio message"})
    db.commit()
    return {"accepted": True, "processed": processed, "status_updates": status_updates, "replies": replies}


# ==============================================================================
# PHASE 3, 5, 6, 7 — CUSTOMER EXPERIENCE, COMPLAINTS, TAKEOVER & SCHEDULER API
# ==============================================================================


@app.post("/api/orders/{order_id}/confirm-arrival")
async def confirm_order_arrival(
    order_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark order DELIVERED after verified arrival confirmation & schedule 1-hour review."""
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    current = OrderStatus(order.status)
    if current in (OrderStatus.CANCELLED, OrderStatus.FAILED):
        raise HTTPException(status_code=409, detail=f"Cannot confirm arrival for {current.value} order")
    now = datetime.now(timezone.utc)
    order.status = OrderStatus.DELIVERED
    _stamp_order_lifecycle(order, OrderStatus.DELIVERED, actor=user.username, now=now)
    db.add(
        OrderStatusHistory(
            order_id=order.id,
            from_status=current.value,
            to_status=OrderStatus.DELIVERED.value,
            actor=user.username,
            note="Arrival confirmed",
            changed_at=now,
        )
    )
    feedback, job = job_scheduler.schedule_post_delivery_review(db, order, delivered_at=now)
    log_audit(db, user.username, "confirm_order_arrival", target=str(order.id))
    db.commit()
    return {
        "order_id": order.id,
        "status": order.status,
        "delivered_at": order.delivered_at.isoformat(),
        "review_scheduled_at": feedback.due_at.isoformat(),
        "scheduler_job_id": job.id,
    }


@app.post("/api/conversations/{phone_number}/takeover")
def toggle_human_takeover(
    phone_number: str,
    payload: TakeoverToggleInput,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Activate or release Human Takeover — when active, AI immediately stops automated replies."""
    session = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_number))
    if session is None:
        session = ChatSession(phone_number=phone_number)
        db.add(session)
    now = datetime.now(timezone.utc)
    session.needs_human_takeover = payload.active
    session.human_takeover_by = user.username if payload.active else None
    session.human_takeover_at = now if payload.active else None
    ai_dispatcher.set_human_takeover(phone_number, payload.active, actor=user.username)
    log_audit(
        db,
        user.username,
        "activate_human_takeover" if payload.active else "release_human_takeover",
        target=phone_number,
    )
    db.commit()
    return {
        "phone_number": phone_number,
        "needs_human_takeover": session.needs_human_takeover,
        "human_takeover_by": session.human_takeover_by,
    }


@app.get("/api/complaints")
def list_complaints(
    status: str | None = Query(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = select(ComplaintTicket).order_by(ComplaintTicket.created_at.desc())
    if status and status != "ALL":
        stmt = stmt.where(ComplaintTicket.status == status)
    tickets = db.scalars(stmt).all()
    return [
        {
            "id": t.id,
            "order_id": t.order_id,
            "customer_phone": t.customer_phone,
            "customer_name": t.customer_name,
            "category": t.category,
            "severity": t.severity,
            "summary": t.summary,
            "sanitized_excerpts": t.sanitized_excerpts,
            "status": t.status,
            "owner_notified": t.owner_notified,
            "escalation_channel": t.escalation_channel,
            "escalation_count": t.escalation_count,
            "owner_acknowledged_at": t.owner_acknowledged_at.isoformat() if t.owner_acknowledged_at else None,
            "owner_acknowledged_by": t.owner_acknowledged_by,
            "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
            "resolved_by": t.resolved_by,
            "resolution_note": t.resolution_note,
            "human_takeover_active": t.human_takeover_active,
            "created_at": t.created_at.isoformat(),
        }
        for t in tickets
    ]


@app.post("/api/complaints/{ticket_id}/acknowledge")
def acknowledge_complaint(
    ticket_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.get(ComplaintTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Complaint ticket not found")
    now = datetime.now(timezone.utc)
    ticket.owner_acknowledged_at = ticket.owner_acknowledged_at or now
    ticket.owner_acknowledged_by = user.username
    if ticket.status in ("OPEN", "ESCALATED"):
        ticket.status = "ACKNOWLEDGED"
    log_audit(db, user.username, "acknowledge_complaint", target=str(ticket.id))
    db.commit()
    return {"id": ticket.id, "status": ticket.status, "owner_acknowledged_by": ticket.owner_acknowledged_by}


@app.post("/api/complaints/{ticket_id}/resolve")
def resolve_complaint(
    ticket_id: int,
    payload: ComplaintResolveInput,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.get(ComplaintTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Complaint ticket not found")
    now = datetime.now(timezone.utc)
    ticket.status = "RESOLVED"
    ticket.resolved_at = now
    ticket.resolved_by = user.username
    ticket.resolution_note = payload.resolution_note.strip()
    if payload.release_human_takeover:
        ticket.human_takeover_active = False
        sess = db.scalar(select(ChatSession).where(ChatSession.phone_number == ticket.customer_phone))
        if sess:
            sess.needs_human_takeover = False
            sess.human_takeover_by = None
        ai_dispatcher.set_human_takeover(ticket.customer_phone, False)
    log_audit(db, user.username, "resolve_complaint", target=str(ticket.id))
    db.commit()
    return {"id": ticket.id, "status": ticket.status, "resolved_by": ticket.resolved_by}


@app.get("/api/customer-experience/overview")
def customer_experience_overview(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Genuine SQL-backed Customer Experience, Reviews, Complaints, Upsell & Scheduler KPIs."""
    feedbacks = db.scalars(select(FeedbackRequest).order_by(FeedbackRequest.id.desc())).all()
    submitted = [f for f in feedbacks if f.rating is not None]
    distribution = {str(star): sum(1 for f in submitted if f.rating == star) for star in range(1, 6)}
    avg_rating = round(sum(f.rating for f in submitted) / len(submitted), 2) if submitted else None
    satisfied_pct = (
        round((sum(1 for f in submitted if f.rating >= 4) / len(submitted)) * 100.0, 1) if submitted else None
    )

    tickets = db.scalars(select(ComplaintTicket).order_by(ComplaintTicket.created_at.desc())).all()
    open_c = sum(1 for t in tickets if t.status == "OPEN")
    escalated_c = sum(1 for t in tickets if t.status == "ESCALATED")
    ack_c = sum(1 for t in tickets if t.status in ("ACKNOWLEDGED", "IN_PROGRESS"))
    resolved_c = sum(1 for t in tickets if t.status == "RESOLVED")
    takeover_c = sum(1 for t in tickets if t.human_takeover_active)

    resp_mins = []
    res_mins = []
    for t in tickets:
        c_at = t.created_at.replace(tzinfo=timezone.utc) if t.created_at.tzinfo is None else t.created_at
        if t.owner_acknowledged_at:
            a_at = t.owner_acknowledged_at.replace(tzinfo=timezone.utc) if t.owner_acknowledged_at.tzinfo is None else t.owner_acknowledged_at
            resp_mins.append(max(0.0, (a_at - c_at).total_seconds() / 60.0))
        if t.resolved_at:
            r_at = t.resolved_at.replace(tzinfo=timezone.utc) if t.resolved_at.tzinfo is None else t.resolved_at
            res_mins.append(max(0.0, (r_at - c_at).total_seconds() / 60.0))

    rec_events = db.scalars(select(RecommendationEvent)).all()
    offered_n = sum(1 for e in rec_events if e.action == "offered")
    accepted_n = sum(1 for e in rec_events if e.action == "accepted")
    rejected_n = sum(1 for e in rec_events if e.action == "rejected")
    upsell_rev = sum((Decimal(str(e.added_revenue)) for e in rec_events if e.action == "accepted"), Decimal("0.00"))

    jobs = db.scalars(select(ScheduledJob)).all()
    job_counts: dict[str, int] = {}
    for j in jobs:
        job_counts[j.status] = job_counts.get(j.status, 0) + 1

    notifs = db.scalars(select(NotificationAttempt).order_by(NotificationAttempt.created_at.desc()).limit(30)).all()

    return {
        "reviews": {
            "total_submitted": len(submitted),
            "average_rating": avg_rating,
            "satisfaction_rate_pct": satisfied_pct,
            "distribution": distribution,
            "recent": [
                {
                    "order_id": f.order_id,
                    "rating": f.rating,
                    "comment": f.comment,
                    "issue_reported": f.issue_reported,
                    "submitted_at": f.submitted_at.isoformat() if f.submitted_at else None,
                }
                for f in submitted[:20]
            ],
        },
        "complaints": {
            "total": len(tickets),
            "open": open_c,
            "escalated": escalated_c,
            "acknowledged": ack_c,
            "resolved": resolved_c,
            "human_takeover_active": takeover_c,
            "avg_response_minutes": round(sum(resp_mins) / len(resp_mins), 1) if resp_mins else None,
            "avg_resolution_minutes": round(sum(res_mins) / len(res_mins), 1) if res_mins else None,
        },
        "recommendations": {
            "upsell_enabled": ai_dispatcher.recommendation_engine.enabled,
            "offered": offered_n,
            "accepted": accepted_n,
            "rejected": rejected_n,
            "acceptance_rate_pct": round((accepted_n / offered_n) * 100.0, 1) if offered_n else None,
            "added_revenue_egp": str(upsell_rev.quantize(Decimal("0.01"))),
        },
        "scheduler": {
            "counts_by_status": job_counts,
            "review_delay_minutes": job_scheduler.review_delay_minutes,
            "arrival_check_delay_minutes": job_scheduler.arrival_check_delay_minutes,
        },
        "notifications": {
            "telegram_configured": telegram_client.is_configured and bool(settings.cashier_telegram_chat_id),
            "owner_telegram_configured": telegram_client.is_configured
            and bool(settings.restaurant_owner_telegram_chat_id),
            "groups_api_eligibility": notification_service.policy_guard.groups_api_eligibility(),
            "recent_attempts": [
                {
                    "id": n.id,
                    "type": n.notification_type,
                    "channel": n.channel,
                    "status": n.status,
                    "order_id": n.order_id,
                    "complaint_id": n.complaint_id,
                    "attempts": n.attempts,
                    "last_error": n.last_error,
                    "created_at": n.created_at.isoformat(),
                }
                for n in notifs
            ],
        },
    }


@app.post("/api/scheduler/run-due")
async def run_due_scheduler_jobs(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    results = await job_scheduler.process_due_jobs(db)
    log_audit(db, admin.username, "scheduler_run_due", target="scheduled_jobs", details={"processed": len(results)})
    db.commit()
    return {"processed": len(results), "results": results}


@app.post("/api/scheduler/cron-tick")
async def scheduler_cron_tick(
    request: Request,
    db: Session = Depends(get_db),
):
    """External zero-cost cron endpoint (Cloudflare Workers Cron / GitHub Actions Cron).

    Requires a configured `SCHEDULER_CRON_SECRET` passed in `X-Scheduler-Secret` header.
    Executes due and missed jobs with atomic worker lease locking.
    """
    expected = (settings.scheduler_cron_secret or "").strip()
    provided = request.headers.get("X-Scheduler-Secret", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="SCHEDULER_CRON_SECRET is not configured")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid scheduler secret")
    recovered = job_scheduler.recover_stale_jobs_on_startup(db)
    results = await job_scheduler.process_due_jobs(db)
    return {"ok": True, "recovered_stale": recovered, "processed": len(results), "results": results}


@app.get("/api/settings/notifications")
def get_notification_settings(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.scalar(select(RestaurantSettings).limit(1))
    return {
        "timezone_name": row.timezone_name if row else settings.restaurant_timezone,
        "upsell_enabled": row.upsell_enabled if row else ai_dispatcher.recommendation_engine.enabled,
        "complaint_escalation_enabled": row.complaint_escalation_enabled
        if row
        else settings.complaint_escalation_enabled,
        "review_delay_minutes": row.review_delay_minutes if row else job_scheduler.review_delay_minutes,
        "arrival_check_delay_minutes": row.arrival_check_delay_minutes
        if row
        else job_scheduler.arrival_check_delay_minutes,
        "meta_billable_authorized": settings.meta_billable_messages_authorized,
        "groups_api_eligibility": notification_service.policy_guard.groups_api_eligibility(),
    }


@app.patch("/api/settings/notifications")
def update_notification_settings(
    payload: NotificationSettingsUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    row = db.scalar(select(RestaurantSettings).limit(1))
    if row is None:
        row = RestaurantSettings(
            timezone_name=settings.restaurant_timezone,
            upsell_enabled=ai_dispatcher.recommendation_engine.enabled,
            complaint_escalation_enabled=settings.complaint_escalation_enabled,
            review_delay_minutes=job_scheduler.review_delay_minutes,
            arrival_check_delay_minutes=job_scheduler.arrival_check_delay_minutes,
        )
        db.add(row)
    if payload.upsell_enabled is not None:
        row.upsell_enabled = payload.upsell_enabled
        ai_dispatcher.recommendation_engine.set_enabled(payload.upsell_enabled)
    if payload.complaint_escalation_enabled is not None:
        row.complaint_escalation_enabled = payload.complaint_escalation_enabled
        settings.complaint_escalation_enabled = payload.complaint_escalation_enabled
    if payload.timezone_name is not None:
        row.timezone_name = payload.timezone_name
        settings.restaurant_timezone = payload.timezone_name
    if payload.review_delay_minutes is not None:
        row.review_delay_minutes = payload.review_delay_minutes
        job_scheduler.review_delay_minutes = payload.review_delay_minutes
    if payload.arrival_check_delay_minutes is not None:
        row.arrival_check_delay_minutes = payload.arrival_check_delay_minutes
        job_scheduler.arrival_check_delay_minutes = payload.arrival_check_delay_minutes
    log_audit(db, admin.username, "update_notification_settings", target="restaurant_settings")
    db.commit()
    return {
        "timezone_name": row.timezone_name,
        "upsell_enabled": row.upsell_enabled,
        "complaint_escalation_enabled": row.complaint_escalation_enabled,
        "review_delay_minutes": row.review_delay_minutes,
        "arrival_check_delay_minutes": row.arrival_check_delay_minutes,
    }


@app.post("/api/settings/telegram/verify")
async def verify_telegram_from_server(
    send_test: bool = Query(default=False),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Single-shot controlled Telegram check & chat discovery from the deployed server (admin-only, zero token leak)."""
    import tempfile
    from scripts.setup_telegram_verify import run_verification

    offset_path = Path(tempfile.gettempdir()) / "wasla_telegram_offset.json"
    result = await run_verification(
        send_test=send_test,
        settings=settings,
        client=telegram_client,
        offset_file=offset_path,
    )
    log_audit(db, admin.username, "telegram_server_verify", target="telegram_bot", details={"status": result.get("status")})
    db.commit()
    return result


def _telegram_webhook_secret() -> str:
    return hmac.new(
        settings.auth_secret_key.encode("utf-8"),
        b"wasla-tg-webhook-secret",
        hashlib.sha256,
    ).hexdigest()[:48]


@app.post("/api/settings/telegram/activate-chat-webhook")
async def activate_telegram_chat_webhook(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Register official Telegram HTTPS webhook for live customer text & voice ordering (private chats only)."""
    webhook_url = f"{settings.dashboard_public_url.rstrip('/')}/api/telegram/webhook"
    res = await telegram_client.set_webhook(webhook_url, _telegram_webhook_secret())
    log_audit(db, admin.username, "activate_telegram_webhook", target=webhook_url, details={"ok": res.get("ok")})
    db.commit()
    return {"ok": bool(res.get("ok")), "webhook_url": webhook_url, "description": res.get("description")}


@app.post("/api/telegram/webhook")
async def telegram_customer_webhook(request: Request, db: Session = Depends(get_db)):
    """Zero-cost live Telegram customer ordering & Egyptian voice-note channel (strictly private chats only)."""
    provided_sec = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not provided_sec or not hmac.compare_digest(provided_sec, _telegram_webhook_secret()):
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")

    body = _parse_webhook_json(await request.body())
    update_id = body.get("update_id")
    if isinstance(update_id, bool) or not isinstance(update_id, int):
        raise HTTPException(status_code=400, detail="Telegram update_id is required")
    dedup_key = f"tg_upd:{update_id}"
    if not _insert_processed_message(db, dedup_key):
        db.rollback()
        return {"ok": True, "duplicate": True}
    db.commit()

    msg = body.get("message") or {}
    chat = msg.get("chat") or {}
    if chat.get("type") != "private":
        db.commit()
        return {"ok": True, "ignored": "non_private_chat"}

    chat_id_str = str(chat.get("id", "")).strip()
    if not chat_id_str:
        db.commit()
        return {"ok": True}
    phone_key = chat_id_str if len(chat_id_str) >= 8 else chat_id_str.zfill(10)

    text_body = (msg.get("text") or "").strip()
    if text_body.startswith(("/start", "/reset")):
        ai_dispatcher.reset_session(phone_key)
        ai_dispatcher.set_human_takeover(phone_key, False)
        existing_sess = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_key))
        if existing_sess is not None:
            existing_sess.needs_human_takeover = False
            existing_sess.human_takeover_by = None
            existing_sess.pending_cart = {}
        db.commit()
        welcome = (
            "أهلاً بيك في مطعم MOKA ❤️🍕\n"
            "أنا وصلة (Wasla AI)، مساعدك الذكي!\n"
            "تقدر تطلب مني كتابة أو تبعتلي فويس نوت باللهجة المصرية وهحسبلك طلبك وأبعته للمطبخ فوراً 😊\n"
            "تحب تطلب إيه النهارده؟ (اكتب /reset في أي وقت لبدء طلب جديد)"
        )
        await telegram_client.send_message(chat_id_str, welcome)
        return {"ok": True, "action": "welcome_sent"}

    existing_sess = db.scalar(select(ChatSession).where(ChatSession.phone_number == phone_key))
    if existing_sess is not None:
        ai_dispatcher.set_human_takeover(
            phone_key, bool(existing_sess.needs_human_takeover), actor=existing_sess.human_takeover_by
        )

    if text_body:
        ai_resp = await ai_dispatcher.handle_text(
            phone_key,
            text_body,
            data_source="telegram_bot",
            is_test=False,
        )
        await _persist_ai_interaction(db, phone_key, text_body, ai_resp)
        db.commit()
        if not ai_resp.suppress_ai_reply and ai_resp.reply_ar:
            await telegram_client.send_message(chat_id_str, ai_resp.reply_ar)
        return {"ok": True, "intent": ai_resp.intent.value}

    voice_obj = msg.get("voice") or msg.get("audio")
    if voice_obj and voice_obj.get("file_id"):
        try:
            audio_bytes = await telegram_client.download_file(voice_obj["file_id"])
            mime = voice_obj.get("mime_type") or "audio/ogg"
            ai_resp = await ai_dispatcher.handle_audio(
                phone_key,
                AudioInput(audio_bytes=audio_bytes, mime_type=mime, filename="voice.ogg"),
                data_source="telegram_bot",
                is_test=False,
            )
            await _persist_ai_interaction(db, phone_key, ai_resp.transcript or "", ai_resp)
            db.commit()
            if not ai_resp.suppress_ai_reply and ai_resp.reply_ar:
                await telegram_client.send_message(chat_id_str, ai_resp.reply_ar)
            return {"ok": True, "intent": ai_resp.intent.value, "transcript": ai_resp.transcript}
        except Exception:
            logger.exception("Failed to process inbound Telegram audio update %s", update_id)
            db.commit()
            await telegram_client.send_message(chat_id_str, "معلش الصوت مكنش واضح، ممكن تبعته تاني أو تكتب طلبك؟ 😊")
            return {"ok": False, "audio_error": "Unable to process audio message"}

    db.commit()
    return {"ok": True}


@app.websocket("/ws/cashier")
async def cashier_socket(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
