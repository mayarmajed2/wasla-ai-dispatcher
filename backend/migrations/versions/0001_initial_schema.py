"""Initial production schema with indexes for PostgreSQL & SQLite.

Revision ID: 0001_initial_schema
Create Date: 2026-11-23
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "menu_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("category", sa.String(80), nullable=False),
        sa.Column("base_price", sa.Numeric(10, 2), nullable=False),
        sa.Column("is_available", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("modifiers_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_menu_items_category", "menu_items", ["category"])

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("customer_name", sa.String(160), nullable=False),
        sa.Column("customer_phone", sa.String(32), nullable=False),
        sa.Column("backup_phone", sa.String(32), nullable=False, server_default=""),
        sa.Column("confirmed_by_customer", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("delivery_address", sa.Text(), nullable=False),
        sa.Column("subtotal", sa.Numeric(10, 2), nullable=False),
        sa.Column("delivery_fee", sa.Numeric(10, 2), nullable=False),
        sa.Column("total_amount", sa.Numeric(10, 2), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="DRAFTING"),
        sa.Column("items_snapshot", sa.JSON(), nullable=False),
        sa.Column("data_source", sa.String(32), nullable=False, server_default="api"),
        sa.Column("is_test", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_is_test", "orders", ["is_test"])
    op.create_index("ix_orders_created_at", "orders", ["created_at"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("phone_number", sa.String(32), unique=True, nullable=False),
        sa.Column("conversation_state", sa.JSON(), nullable=False),
        sa.Column("pending_cart", sa.JSON(), nullable=False),
        sa.Column("active_order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("needs_human_takeover", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latest_directive", sa.Text(), nullable=True),
    )

    op.create_table(
        "order_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_order_events_order_id", "order_events", ["order_id"])

    op.create_table(
        "processed_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("meta_message_id", sa.String(128), unique=True, nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "outbox",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(64), unique=True, nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("delivered", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "feedback_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), unique=True, nullable=False),
        sa.Column("token", sa.String(64), unique=True, nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(64), unique=True, nullable=False),
        sa.Column("full_name", sa.String(120), nullable=False),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="cashier"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_username", sa.String(64), nullable=False),
        sa.Column("action", sa.String(80), nullable=False),
        sa.Column("target", sa.String(120), nullable=False, server_default=""),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in [
        "audit_logs",
        "users",
        "feedback_requests",
        "outbox",
        "processed_messages",
        "order_events",
        "chat_sessions",
        "orders",
        "menu_items",
    ]:
        op.drop_table(table)
