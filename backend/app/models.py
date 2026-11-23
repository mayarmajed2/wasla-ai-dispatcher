from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class OrderStatus(str, Enum):
    # Original 7-state values (preserved for backward compatibility)
    DRAFTING = "DRAFTING"
    SUBMITTED = "SUBMITTED"
    IN_KITCHEN = "IN_KITCHEN"
    DISPATCHED = "DISPATCHED"
    UNREACHABLE_ALERT = "UNREACHABLE_ALERT"
    FULFILLED = "FULFILLED"
    CANCELLED = "CANCELLED"
    # Explicit commercial lifecycle states (Phase 3)
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    ACCEPTED = "ACCEPTED"
    PREPARING = "PREPARING"
    READY = "READY"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


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
    human_takeover_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    human_takeover_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_customer_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    opted_out_reviews: Mapped[bool] = mapped_column(Boolean, default=False)
    latest_directive: Mapped[str | None] = mapped_column(Text, nullable=True)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(160), nullable=False)
    customer_phone: Mapped[str] = mapped_column(String(32), nullable=False)
    backup_phone: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    confirmed_by_customer: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    delivery_address: Mapped[str] = mapped_column(Text, nullable=False)
    order_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Delivery")
    subtotal: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    delivery_fee: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(String(32), default=OrderStatus.DRAFTING)
    items_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    data_source: Mapped[str] = mapped_column(String(32), nullable=False, default="api")
    is_test: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    refunded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    preparing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    out_for_delivery_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivery_confirmed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class OrderStatusHistory(Base):
    __tablename__ = "order_status_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False, index=True)
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


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
    customer_phone: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    issue_reported: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="whatsapp")
    suppressed_reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    policy_blocked_reason: Mapped[str | None] = mapped_column(String(160), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"
    __table_args__ = (Index("ix_scheduled_jobs_status_run_at", "status", "run_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    customer_phone: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class NotificationAttempt(Base):
    __tablename__ = "notification_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    notification_type: Mapped[str] = mapped_column(String(64), nullable=False)  # cashier_order | owner_complaint | customer_review | arrival_check
    channel: Mapped[str] = mapped_column(String(32), nullable=False)  # telegram | whatsapp | dashboard_only
    recipient_ref: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    complaint_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")  # SENT | FAILED | POLICY_BLOCKED | DRY_RUN | SUPPRESSED
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider_response: Mapped[dict] = mapped_column(JSON, default=dict)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ComplaintTicket(Base):
    __tablename__ = "complaint_tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    customer_phone: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    customer_name: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="HIGH")  # CRITICAL | HIGH | MEDIUM | LOW
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    sanitized_excerpts: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="OPEN")  # OPEN | ESCALATED | ACKNOWLEDGED | IN_PROGRESS | RESOLVED
    owner_notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    escalation_channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    escalation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_escalated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    owner_acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    owner_acknowledged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    human_takeover_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RecommendationEvent(Base):
    __tablename__ = "recommendation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_phone: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    offer_type: Mapped[str] = mapped_column(String(32), nullable=False, default="separate_items")
    action: Mapped[str] = mapped_column(String(24), nullable=False)  # offered | accepted | rejected
    product_ids: Mapped[list] = mapped_column(JSON, default=list)
    added_revenue: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=Decimal("0.00"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RestaurantSettings(Base):
    __tablename__ = "restaurant_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timezone_name: Mapped[str] = mapped_column(String(64), nullable=False, default="Africa/Cairo")
    upsell_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    complaint_escalation_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    meta_billable_authorized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    monthly_free_service_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=1000)
    cashier_group_channel: Mapped[str] = mapped_column(String(32), nullable=False, default="telegram")
    review_delay_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    arrival_check_delay_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=90)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="cashier")  # admin | cashier
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    target: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
