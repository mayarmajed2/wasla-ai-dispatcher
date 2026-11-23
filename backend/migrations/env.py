"""Alembic environment configuration targeting Wasla SQLAlchemy metadata."""

from __future__ import annotations

import os
from alembic import context

from app.config import settings
from app.database import Base, build_sqlalchemy_engine
from app import models  # noqa: F401 — ensure all tables are registered

config = context.config
target_metadata = Base.metadata


def _resolve_url() -> str:
    configured = (config.get_main_option("sqlalchemy.url") or "").strip()
    if configured and not configured.startswith("driver://"):
        return configured
    env_url = (os.getenv("DATABASE_URL") or "").strip()
    return env_url or settings.database_url


def run_migrations_offline() -> None:
    url = _resolve_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _resolve_url()
    connectable = build_sqlalchemy_engine(url, pool_size=2, max_overflow=0)
    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
