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

from app.cart_calculator import calculate_cart
from app.config import settings
from app.database import Base, SessionLocal, engine, get_db
from app.meta_service import MetaWhatsAppClient
from app.models import (
    ChatSession,
    FeedbackRequest,
    MenuItem,
    Order,
    OrderEvent,
    OrderStatus,
    Outbox,
    ProcessedMessage,
)
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
        if "backup_phone" not in columns:
            connection.execute(
                text("ALTER TABLE orders ADD COLUMN backup_phone VARCHAR(32) NOT NULL DEFAULT ''")
            )
        if "confirmed_by_customer" not in columns:
            connection.execute(
                text("ALTER TABLE orders ADD COLUMN confirmed_by_customer BOOLEAN NOT NULL DEFAULT 0")
            )
        if "data_source" not in columns:
            connection.execute(
                text("ALTER TABLE orders ADD COLUMN data_source VARCHAR(32) NOT NULL DEFAULT 'api'")
            )
        if "is_test" not in columns:
            connection.execute(
                text("ALTER TABLE orders ADD COLUMN is_test BOOLEAN NOT NULL DEFAULT 0")
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


async def feedback_worker() -> None:
    while True:
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            requests = db.scalars(
                select(FeedbackRequest).where(FeedbackRequest.sent_at.is_(None))
            ).all()
            for feedback in requests:
                due_at = feedback.due_at
                if due_at.tzinfo is None:
                    due_at = due_at.replace(tzinfo=timezone.utc)
                if due_at <= now:
                    order = db.get(Order, feedback.order_id)
                    if not order:
                        continue
                    feedback.sent_at = now
                    db.add(
                        Outbox(
                            event_id=str(uuid.uuid4()),
                            event_type="feedback_request_due",
                            payload={
                                "order_id": order.id,
                                "customer_phone": order.customer_phone,
                                "feedback_token": feedback.token,
                            },
                        )
                    )
                    db.add(
                        OrderEvent(
                            order_id=order.id,
                            event_type="feedback_request_sent",
                            payload={"feedback_token": feedback.token},
                        )
                    )
            db.commit()
        finally:
            db.close()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _migrate_sqlite_columns()
    seed_moka_menu_if_empty()
    worker_task = asyncio.create_task(feedback_worker())
    try:
        yield
    finally:
        worker_task.cancel()


app = FastAPI(title="Wasla — Restaurant AI Dispatcher", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
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


class AvailabilityUpdate(BaseModel):
    is_available: bool


class AIChatRequest(BaseModel):
    phone_number: str = Field(min_length=8)
    message: str = Field(min_length=1)


ALLOWED_TRANSITIONS = {
    OrderStatus.DRAFTING: {OrderStatus.SUBMITTED, OrderStatus.CANCELLED},
    OrderStatus.SUBMITTED: {OrderStatus.IN_KITCHEN, OrderStatus.CANCELLED},
    OrderStatus.IN_KITCHEN: {OrderStatus.DISPATCHED, OrderStatus.CANCELLED},
    OrderStatus.DISPATCHED: {OrderStatus.UNREACHABLE_ALERT, OrderStatus.FULFILLED, OrderStatus.CANCELLED},
    OrderStatus.UNREACHABLE_ALERT: {OrderStatus.DISPATCHED, OrderStatus.FULFILLED, OrderStatus.CANCELLED},
    OrderStatus.FULFILLED: set(),
    OrderStatus.CANCELLED: set(),
}


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
    )
    db.add(order)
    db.flush()
    db.add_all(
        [
            OrderEvent(order_id=order.id, event_type="created", payload={"status": "SUBMITTED"}),
            OrderEvent(order_id=order.id, event_type="idempotency_key", payload={"key": payload.idempotency_key}),
            Outbox(
                event_id=str(uuid.uuid4()),
                event_type="order_created",
                payload={"order_id": order.id, "total_amount": cart["total_amount"], "status": "SUBMITTED"},
            ),
        ]
    )
    db.commit()
    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "order_created",
            "order_id": order.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {
                "customer_name": order.customer_name,
                "total_amount": str(order.total_amount),
                "status": order.status,
            },
        }
    )
    return {"id": order.id, "status": order.status, "cart": cart}


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
        },
    }


@app.get("/api/metrics")
def get_metrics(db: Session = Depends(get_db)):
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
def get_all_menu_items(db: Session = Depends(get_db)):
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
async def toggle_menu_availability(item_id: int, payload: AvailabilityUpdate, db: Session = Depends(get_db)):
    item = db.get(MenuItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    item.is_available = payload.is_available
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


@app.post("/api/ai/chat")
async def ai_chat(payload: AIChatRequest, db: Session = Depends(get_db)):
    """Interactive Wasla AI Simulator endpoint — explicitly tags generated orders as simulator/is_test=True."""
    resp = await ai_dispatcher.handle_text(
        payload.phone_number,
        payload.message,
        data_source="simulator",
        is_test=True,
    )

    session_row = db.scalar(select(ChatSession).where(ChatSession.phone_number == payload.phone_number))
    if session_row is None:
        session_row = ChatSession(phone_number=payload.phone_number)
        db.add(session_row)
    session_row.pending_cart = resp.cart.model_dump(mode="json")
    session_row.active_order_id = resp.order_id
    session_row.needs_human_takeover = resp.needs_human_takeover
    session_row.conversation_state = {
        "last_intent": resp.intent.value,
        "requires_clarification": resp.requires_clarification,
        "model_used": resp.model_used,
    }
    db.commit()
    return resp.model_dump(mode="json")


@app.post("/api/orders", status_code=201)
async def confirm_order(payload: ConfirmOrder, db: Session = Depends(get_db)):
    return await _create_order_in_db(payload, db)


@app.get("/api/orders")
def list_orders(db: Session = Depends(get_db)):
    orders = db.scalars(select(Order).order_by(Order.created_at.desc())).all()
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
async def update_status(order_id: int, payload: StatusUpdate, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    current = OrderStatus(order.status)
    if payload.status not in ALLOWED_TRANSITIONS[current]:
        raise HTTPException(status_code=409, detail=f"Cannot change {current.value} to {payload.status.value}")
    order.status = payload.status
    db.add(OrderEvent(order_id=order.id, event_type="status_changed", payload={"status": payload.status.value}))
    db.add(
        Outbox(
            event_id=str(uuid.uuid4()),
            event_type="order_status_changed",
            payload={"order_id": order.id, "status": payload.status.value},
        )
    )
    if payload.status == OrderStatus.FULFILLED:
        existing_feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.order_id == order.id))
        if existing_feedback is None:
            db.add(
                FeedbackRequest(
                    order_id=order.id,
                    token=str(uuid.uuid4()),
                    due_at=datetime.now(timezone.utc) + timedelta(minutes=30),
                )
            )
            db.add(OrderEvent(order_id=order.id, event_type="feedback_scheduled", payload={"delay_minutes": 30}))
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


@app.post("/api/orders/{order_id}/reset-demo")
async def reset_order_for_demo(order_id: int, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order.status = OrderStatus.IN_KITCHEN
    db.add(OrderEvent(order_id=order.id, event_type="demo_reset", payload={"status": OrderStatus.IN_KITCHEN.value}))
    feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.order_id == order.id))
    if feedback:
        db.delete(feedback)
    db.commit()
    await ws_manager.broadcast(
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "order_status_changed",
            "order_id": order.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "payload": {"status": OrderStatus.IN_KITCHEN.value, "demo_reset": True},
        }
    )
    return {"id": order.id, "status": order.status}


@app.post("/api/orders/{order_id}/instruct")
async def instruct_customer(order_id: int, payload: Directive, db: Session = Depends(get_db)):
    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if not payload.directive.strip():
        raise HTTPException(status_code=400, detail="Directive cannot be empty")

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
def submit_feedback(token: str, payload: FeedbackInput, db: Session = Depends(get_db)):
    feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.token == token))
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback link not found")
    if feedback.submitted_at is not None:
        raise HTTPException(status_code=409, detail="Feedback already submitted")
    if feedback.sent_at is None:
        raise HTTPException(status_code=409, detail="Feedback period has not started")
    feedback.rating = payload.rating
    feedback.comment = payload.comment.strip()
    feedback.submitted_at = datetime.now(timezone.utc)
    db.add(
        OrderEvent(
            order_id=feedback.order_id,
            event_type="feedback_submitted",
            payload={"rating": payload.rating, "comment": feedback.comment},
        )
    )
    db.commit()
    return {"status": "submitted", "rating": payload.rating}


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
    replies = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            for message in change.get("value", {}).get("messages", []):
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
                if msg_type == "text" and sender:
                    text_body = (message.get("text") or {}).get("body", "").strip()
                    if text_body:
                        ai_resp = await ai_dispatcher.handle_text(
                            sender,
                            text_body,
                            data_source="whatsapp_webhook",
                            is_test=False,
                        )
                        delivery = await meta_client.send_text_message(sender, ai_resp.reply_ar)
                        replies.append(
                            {
                                "to": sender,
                                "reply_ar": ai_resp.reply_ar,
                                "intent": ai_resp.intent.value,
                                "whatsapp_delivery": delivery,
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
                            delivery = await meta_client.send_text_message(sender, ai_resp.reply_ar)
                            replies.append(
                                {
                                    "to": sender,
                                    "transcript": ai_resp.transcript,
                                    "reply_ar": ai_resp.reply_ar,
                                    "intent": ai_resp.intent.value,
                                    "whatsapp_delivery": delivery,
                                }
                            )
                        except Exception as exc:
                            replies.append({"to": sender, "audio_error": str(exc)})
    db.commit()
    return {"accepted": True, "processed": processed, "replies": replies}


@app.websocket("/ws/cashier")
async def cashier_socket(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
