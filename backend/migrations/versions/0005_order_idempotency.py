"""Persist and enforce order idempotency keys.

Revision ID: 0005_order_idempotency
Revises: 0004_phase_b_durable_ai_state
Create Date: 2026-11-24
"""

import json

from alembic import op
import sqlalchemy as sa

revision = "0005_order_idempotency"
down_revision = "0004_phase_b_durable_ai_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orders") as batch_op:
        batch_op.add_column(sa.Column("idempotency_key", sa.String(length=128), nullable=True))

    bind = op.get_bind()
    orders = sa.table(
        "orders",
        sa.column("id", sa.Integer()),
        sa.column("idempotency_key", sa.String(length=128)),
    )
    events = sa.table(
        "order_events",
        sa.column("order_id", sa.Integer()),
        sa.column("event_type", sa.String(length=64)),
        sa.column("payload", sa.JSON()),
    )
    seen_keys: set[str] = set()
    rows = bind.execute(
        sa.select(events.c.order_id, events.c.payload)
        .where(events.c.event_type == "idempotency_key")
        .order_by(events.c.order_id)
    )
    for order_id, payload in rows:
        if isinstance(payload, str):
            payload = json.loads(payload)
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str) or not key or key in seen_keys:
            continue
        bind.execute(
            orders.update().where(orders.c.id == order_id).values(idempotency_key=key)
        )
        seen_keys.add(key)

    op.create_index("ux_orders_idempotency_key", "orders", ["idempotency_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ux_orders_idempotency_key", table_name="orders")
    with op.batch_alter_table("orders") as batch_op:
        batch_op.drop_column("idempotency_key")
