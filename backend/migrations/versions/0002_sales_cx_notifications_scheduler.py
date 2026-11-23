"""Add OrderStatusHistory, ScheduledJob, NotificationAttempt, ComplaintTicket, RecommendationEvent, RestaurantSettings & lifecycle columns.

Revision ID: 0002_sales_cx_notifications
Revises: 0001_initial_schema
Create Date: 2026-11-23
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_sales_cx_notifications"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Extend existing orders table with lifecycle timestamps & metadata
    with op.batch_alter_table("orders") as batch_op:
        batch_op.add_column(sa.Column("order_type", sa.String(32), nullable=False, server_default="Delivery"))
        batch_op.add_column(sa.Column("refunded", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("preparing_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("out_for_delivery_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("delivery_confirmed_by", sa.String(64), nullable=True))

    # 2. Extend chat_sessions for 24h Meta window tracking, opt-out & human takeover audit
    with op.batch_alter_table("chat_sessions") as batch_op:
        batch_op.add_column(sa.Column("human_takeover_by", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("human_takeover_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("last_customer_message_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("opted_out_reviews", sa.Boolean(), nullable=False, server_default=sa.false()))

    # 3. Extend feedback_requests for issue reports & Meta policy/suppression visibility
    with op.batch_alter_table("feedback_requests") as batch_op:
        batch_op.add_column(sa.Column("customer_phone", sa.String(32), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("issue_reported", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("channel", sa.String(32), nullable=False, server_default="whatsapp"))
        batch_op.add_column(sa.Column("suppressed_reason", sa.String(120), nullable=True))
        batch_op.add_column(sa.Column("policy_blocked_reason", sa.String(160), nullable=True))

    # 4. Order Status History table
    op.create_table(
        "order_status_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=False),
        sa.Column("from_status", sa.String(32), nullable=True),
        sa.Column("to_status", sa.String(32), nullable=False),
        sa.Column("actor", sa.String(64), nullable=False, server_default="system"),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_order_status_history_order_id", "order_status_history", ["order_id"])

    # 5. Durable restart-safe ScheduledJob table
    op.create_table(
        "scheduled_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False, unique=True),
        sa.Column("job_type", sa.String(64), nullable=False),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("customer_phone", sa.String(32), nullable=False, server_default=""),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_scheduled_jobs_status_run_at", "scheduled_jobs", ["status", "run_at"])
    op.create_index("ix_scheduled_jobs_order_id", "scheduled_jobs", ["order_id"])

    # 6. NotificationAttempt table (idempotent delivery log)
    op.create_table(
        "notification_attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False, unique=True),
        sa.Column("notification_type", sa.String(64), nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column("recipient_ref", sa.String(128), nullable=False, server_default=""),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("complaint_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider_response", sa.JSON(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_notification_attempts_order_id", "notification_attempts", ["order_id"])
    op.create_index("ix_notification_attempts_complaint_id", "notification_attempts", ["complaint_id"])

    # 7. ComplaintTicket table
    op.create_table(
        "complaint_tickets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("customer_phone", sa.String(32), nullable=False),
        sa.Column("customer_name", sa.String(160), nullable=False, server_default=""),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False, server_default="HIGH"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("sanitized_excerpts", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="OPEN"),
        sa.Column("owner_notified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("escalation_channel", sa.String(32), nullable=True),
        sa.Column("escalation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_escalated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner_acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner_acknowledged_by", sa.String(64), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(64), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=False, server_default=""),
        sa.Column("human_takeover_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_complaint_tickets_customer_phone", "complaint_tickets", ["customer_phone"])
    op.create_index("ix_complaint_tickets_order_id", "complaint_tickets", ["order_id"])

    # 8. RecommendationEvent table
    op.create_table(
        "recommendation_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("customer_phone", sa.String(32), nullable=False),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("offer_type", sa.String(32), nullable=False, server_default="separate_items"),
        sa.Column("action", sa.String(24), nullable=False),
        sa.Column("product_ids", sa.JSON(), nullable=False),
        sa.Column("added_revenue", sa.Numeric(10, 2), nullable=False, server_default="0.00"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_recommendation_events_customer_phone", "recommendation_events", ["customer_phone"])

    # 9. RestaurantSettings table
    op.create_table(
        "restaurant_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timezone_name", sa.String(64), nullable=False, server_default="Africa/Cairo"),
        sa.Column("upsell_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("complaint_escalation_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("meta_billable_authorized", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("monthly_free_service_limit", sa.Integer(), nullable=False, server_default="1000"),
        sa.Column("cashier_group_channel", sa.String(32), nullable=False, server_default="telegram"),
        sa.Column("review_delay_minutes", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("arrival_check_delay_minutes", sa.Integer(), nullable=False, server_default="90"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("restaurant_settings")
    op.drop_table("recommendation_events")
    op.drop_table("complaint_tickets")
    op.drop_table("notification_attempts")
    op.drop_table("scheduled_jobs")
    op.drop_table("order_status_history")
    with op.batch_alter_table("feedback_requests") as batch_op:
        batch_op.drop_column("policy_blocked_reason")
        batch_op.drop_column("suppressed_reason")
        batch_op.drop_column("channel")
        batch_op.drop_column("issue_reported")
        batch_op.drop_column("customer_phone")
    with op.batch_alter_table("chat_sessions") as batch_op:
        batch_op.drop_column("opted_out_reviews")
        batch_op.drop_column("last_customer_message_at")
        batch_op.drop_column("human_takeover_at")
        batch_op.drop_column("human_takeover_by")
    with op.batch_alter_table("orders") as batch_op:
        batch_op.drop_column("delivery_confirmed_by")
        batch_op.drop_column("failed_at")
        batch_op.drop_column("cancelled_at")
        batch_op.drop_column("delivered_at")
        batch_op.drop_column("out_for_delivery_at")
        batch_op.drop_column("ready_at")
        batch_op.drop_column("preparing_at")
        batch_op.drop_column("accepted_at")
        batch_op.drop_column("confirmed_at")
        batch_op.drop_column("refunded")
        batch_op.drop_column("order_type")
