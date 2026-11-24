"""Add durable `inbound_message_buffers` table for restart-safe WhatsApp debouncing.

Revision ID: 0004_phase_b_durable_ai_state
Revises: 0003_scheduler_leases
Create Date: 2026-11-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_phase_b_durable_ai_state"
down_revision = "0003_scheduler_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inbound_message_buffers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message_uid", sa.String(length=128), nullable=False),
        sa.Column("phone_number", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("data_source", sa.String(length=32), nullable=False, server_default="whatsapp_webhook"),
        sa.Column("msg_timestamp", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="BUFFERED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_uid"),
    )
    op.create_index("ix_inbound_message_buffers_phone_number", "inbound_message_buffers", ["phone_number"])
    op.create_index("ix_inbound_message_buffers_status", "inbound_message_buffers", ["status"])
    op.create_index("ix_inbound_buf_phone_status", "inbound_message_buffers", ["phone_number", "status"])


def downgrade() -> None:
    op.drop_index("ix_inbound_buf_phone_status", table_name="inbound_message_buffers")
    op.drop_index("ix_inbound_message_buffers_status", table_name="inbound_message_buffers")
    op.drop_index("ix_inbound_message_buffers_phone_number", table_name="inbound_message_buffers")
    op.drop_table("inbound_message_buffers")
