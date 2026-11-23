from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class OrderStatus(str, Enum):
    DRAFTING = "DRAFTING"
    SUBMITTED = "SUBMITTED"
    IN_KITCHEN = "IN_KITCHEN"
    DISPATCHED = "DISPATCHED"
    UNREACHABLE_ALERT = "UNREACHABLE_ALERT"
    FULFILLED = "FULFILLED"
    CANCELLED = "CANCELLED"


class MenuItem(Base):
    __tablename__ = "menu_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    base_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    modifiers_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phone_number: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    conversation_state: Mapped[dict] = mapped_column(JSON, default=dict)
    pending_cart: Mapped[dict] = mapped_column(JSON, default=dict)
    active_order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True)
    needs_human_takeover: Mapped[bool] = mapped_column(Boolean, default=False)
    latest_directive: Mapped[str | None] = mapped_column(Text, nullable=True)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(160), nullable=False)
    customer_phone: Mapped[str] = mapped_column(String(32), nullable=False)
    backup_phone: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    confirmed_by_customer: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    delivery_address: Mapped[str] = mapped_column(Text, nullable=False)
    subtotal: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    delivery_fee: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(String(32), default=OrderStatus.DRAFTING)
    items_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class OrderEvent(Base):
    __tablename__ = "order_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProcessedMessage(Base):
    __tablename__ = "processed_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meta_message_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Outbox(Base):
    __tablename__ = "outbox"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)


class FeedbackRequest(Base):
    __tablename__ = "feedback_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), unique=True, nullable=False)
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
