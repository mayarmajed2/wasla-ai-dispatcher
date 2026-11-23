"""Database engine & session factory supporting SQLite (local dev) and PostgreSQL (production).

Automatically handles:
- Render/Neon `postgres://` scheme normalization to `postgresql://`
- Linux/Docker `psycopg` (v3) driver when available
- Pure-Python `pg8000` SSL fallback on Windows when `libpq` is unavailable
"""

from __future__ import annotations

import ssl
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


def build_sqlalchemy_engine(database_url: str, pool_size: int = 10, max_overflow: int = 20):
    url = (database_url or "").strip()
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})

    # Normalize postgres:// -> postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    connect_args: dict = {}
    has_psycopg = False
    try:
        import psycopg  # noqa: F401

        has_psycopg = True
    except Exception:
        has_psycopg = False

    if not has_psycopg and "+pg8000" not in url:
        # Pure-Python PostgreSQL driver fallback with strict TLS SSL
        if url.startswith("postgresql+psycopg://"):
            url = "postgresql+pg8000://" + url[len("postgresql+psycopg://") :]
        elif url.startswith("postgresql+psycopg2://"):
            url = "postgresql+pg8000://" + url[len("postgresql+psycopg2://") :]
        elif url.startswith("postgresql://"):
            url = "postgresql+pg8000://" + url[len("postgresql://") :]

        parts = urlsplit(url)
        query_params = dict(parse_qsl(parts.query))
        sslmode = query_params.pop("sslmode", "require")
        query_params.pop("channel_binding", None)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query_params), parts.fragment))
        if sslmode != "disable":
            connect_args["ssl_context"] = ssl.create_default_context()

    return create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=connect_args,
    )


engine = build_sqlalchemy_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
