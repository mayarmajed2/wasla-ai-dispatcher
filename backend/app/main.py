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

from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.schemas.models import AudioInput, BackendConfirmOrderPayload, CashierDirectiveRequest
from ai_service.understanding.menu_index import MenuIndex


DASHBOARD_DIR = ROOT_DIR / "dashboard"
MENU_JSON_PATH = ROOT_DIR / "menu" / "menu.json"

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
                {"name": modifiers_map[mid]["name_ar"], "price": str(modifiers_map[mid]["price"])}
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


class _DirectDBOrderBridge(MockBackendAdapter):
    """Bridges WaslaAIDispatcher cart & confirmation directly into the live SQLAlchemy DB."""

    IS_MOCK = False

    async def confirm_and_lock_order(self, payload: BackendConfirmOrderPayload) -> dict:
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
        finally:
            db.close()


wasla_config = WaslaConfig(menu_json_path=MENU_JSON_PATH)
menu_index = MenuIndex(MENU_JSON_PATH)
db_bridge = _DirectDBOrderBridge(menu_index)
ai_dispatcher = WaslaAIDispatcher(wasla_config, menu_index, db_bridge)
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
    seed_initial_admin_if_empty()
    worker_task = asyncio.create_task(feedback_worker())
    try:
        yield
    finally:
        worker_task.cancel()


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
    quantity: int = Field(ge=1)
    size_code: str | None = None
    modifiers: list[dict] = Field(default_factory=list)
    notes: str = ""


class ConfirmOrder(BaseModel):
    customer_name: str = Field(min_length=2)
    customer_phone: str = Field(min_length=8)
    backup_phone: str = Field(min_length=8)
    delivery_address: str = Field(min_length=5)
    zone_name: str = "default"
    items: list[CartItem] = Field(min_length=1)
    idempotency_key: str
    confirmed_by_customer: bool
    data_source: str = "api"
    is_test: bool = False


class Directive(BaseModel):
    directive: str
    directive_type: str = "free_text"
    extra_delay_minutes: int | None = None


class StatusUpdate(BaseModel):
    status: OrderStatus


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


class PriceUpdate(BaseModel):
    base_price: Decimal = Field(gt=0)


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

    events = db.scalars(select(OrderEvent).where(OrderEvent.event_type == "idempotency_key")).all()
    for event in events:
        if event.payload.get("key") == payload.idempotency_key:
            existing = db.get(Order, event.order_id)
            if existing:
                return {"id": existing.id, "status": existing.status, "duplicate": True}

    resolved_items = []
    for cart_item in payload.items:
        menu_item = db.get(MenuItem, cart_item.item_id)
        if not menu_item or not menu_item.is_available:
            raise HTTPException(status_code=409, detail=f"Item {cart_item.item_id} is unavailable")
        unit_price = str(menu_item.base_price)
        item_name = menu_item.name
        if cart_item.size_code and isinstance(menu_item.modifiers_json, dict):
            for sz in menu_item.modifiers_json.get("sizes", []):
                if str(sz.get("code", "")).upper() == cart_item.size_code.upper():
                    unit_price = str(sz["price"])
                    item_name = f"{menu_item.name} ({sz['code']})"
                    break
        resolved_items.append(
            {
                "item_id": menu_item.id,
                "name": item_name,
                "base_price": unit_price,
                "quantity": cart_item.quantity,
                "modifiers": cart_item.modifiers,
                "notes": cart_item.notes,
            }
        )

    delivery_fee = "0.00" if payload.zone_name == "free_zone" else "20.00"
    cart = calculate_cart(resolved_items, Decimal(delivery_fee))
    now = datetime.now(timezone.utc)
    order = Order(
        customer_name=payload.customer_name,
        customer_phone=payload.customer_phone,
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
    db.flush()
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

    # Immediately notify Cashier Group (Telegram zero-cost); never lose order if notification fails
    cashier_notify = await notification_service.notify_cashier_new_order(db, order)
    if cashier_notify.get("status") == "FAILED":
        job_scheduler.schedule_cashier_retry(db, order, delay_seconds=30)

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
    """Production readiness probe reporting live DB health & honest external provider status."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

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
            "finetuned_lora_checkpoint_exists": ai_dispatcher.finetuned_intent.checkpoint_exists,
            "finetuned_lora_loaded": ai_dispatcher.finetuned_intent.is_loaded,
        },
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

    pid = (item.modifiers_json or {}).get("product_id")
    if pid:
        db_bridge.set_item_unavailable(pid, unavailable=not payload.is_available)

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
def update_menu_price(
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
    log_audit(db, admin.username, "update_price", target=str(item.id), details={"old": old_price, "new": str(payload.base_price)})
    db.commit()
    return {"id": item.id, "name": item.name, "base_price": str(item.base_price)}


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
    session_row.conversation_state = {
        "last_intent": resp.intent.value,
        "requires_clarification": resp.requires_clarification,
        "model_used": resp.model_used,
        "suppress_ai_reply": resp.suppress_ai_reply,
    }

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
        session_row.human_takeover_at = session_row.human_takeover_at or now
        ai_dispatcher.set_human_takeover(phone_number, True, actor="complaint_auto_escalation")
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

    body = json.loads(raw_body.decode("utf-8"))
    processed = 0
    status_updates = 0
    replies = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for st in value.get("statuses", []):
                st_id = st.get("id")
                st_val = st.get("status", "")
                if not st_id or not st_val:
                    continue
                dedup_key = f"status:{st_id}:{st_val}"
                if db.scalar(select(ProcessedMessage).where(ProcessedMessage.meta_message_id == dedup_key)) is not None:
                    continue
                db.add(ProcessedMessage(meta_message_id=dedup_key))
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

            for message in value.get("messages", []):
                meta_id = message.get("id")
                if not meta_id:
                    continue
                existing = db.scalar(
                    select(ProcessedMessage).where(ProcessedMessage.meta_message_id == meta_id)
                )
                if existing is not None:
                    continue
                db.add(ProcessedMessage(meta_message_id=meta_id))
                processed += 1

                sender = message.get("from", "")
                msg_type = message.get("type", "")
                if sender:
                    existing_sess = db.scalar(select(ChatSession).where(ChatSession.phone_number == sender))
                    if existing_sess is not None:
                        ai_dispatcher.set_human_takeover(
                            sender, bool(existing_sess.needs_human_takeover), actor=existing_sess.human_takeover_by
                        )
                if msg_type == "text" and sender:
                    text_body = (message.get("text") or {}).get("body", "").strip()
                    if text_body:
                        ai_resp = await ai_dispatcher.handle_text(
                            sender,
                            text_body,
                            data_source="whatsapp_webhook",
                            is_test=False,
                        )
                        side = await _persist_ai_interaction(db, sender, text_body, ai_resp)
                        if ai_resp.suppress_ai_reply:
                            delivery = {"sent": False, "mode": "suppressed_human_takeover"}
                        else:
                            delivery = await meta_client.send_text_message(sender, ai_resp.reply_ar)
                        replies.append(
                            {
                                "to": sender,
                                "reply_ar": ai_resp.reply_ar,
                                "intent": ai_resp.intent.value,
                                "suppress_ai_reply": ai_resp.suppress_ai_reply,
                                "whatsapp_delivery": delivery,
                                "complaint_escalation": side.get("escalation"),
                            }
                        )
                elif msg_type == "audio" and sender:
                    media_id = (message.get("audio") or {}).get("id", "")
                    if media_id:
                        try:
                            audio_bytes, mime_type = await meta_client.download_media(media_id)
                            ai_resp = await ai_dispatcher.handle_audio(
                                sender,
                                AudioInput(audio_bytes=audio_bytes, mime_type=mime_type, filename=f"{media_id}.ogg"),
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
                        except Exception as exc:
                            replies.append({"to": sender, "audio_error": str(exc)})
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


@app.websocket("/ws/cashier")
async def cashier_socket(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
