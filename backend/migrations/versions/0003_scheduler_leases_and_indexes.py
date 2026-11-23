"""Add concurrent worker lease/locking columns (`locked_by`, `locked_until`) to `scheduled_jobs`.

Revision ID: 0003_scheduler_leases_and_indexes
Revises: 0002_sales_cx_notifications_scheduler
Create Date: 2026-11-23
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_scheduler_leases_and_indexes"
down_revision = "0002_sales_cx_notifications_scheduler"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("scheduled_jobs") as batch_op:
        batch_op.add_column(sa.Column("locked_by", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_scheduled_jobs_locked_until", "scheduled_jobs", ["locked_until"])


def downgrade() -> None:
    op.drop_index("ix_scheduled_jobs_locked_until", table_name="scheduled_jobs")
    with op.batch_alter_table("scheduled_jobs") as batch_op:
        batch_op.drop_column("locked_until")
        batch_op.drop_column("locked_by")
