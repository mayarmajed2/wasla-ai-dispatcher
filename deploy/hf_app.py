"""Hugging Face Spaces Free Gradio-SDK Entrypoint for Wasla AI.

Executes Alembic PostgreSQL migrations to head and starts Uvicorn on port 7860,
serving the FastAPI backend, MOKA Cashier Dashboard (/), WebSocket (/ws/cashier),
Meta WhatsApp Webhook (/api/webhook), and External Cron endpoint (/api/scheduler/cron-tick)
on Hugging Face's 100% free (0 EGP, no credit card) CPU basic tier.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "backend").is_dir() else _HERE.parent
BACKEND_DIR = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from alembic import command as alembic_command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402


def bootstrap_database() -> None:
    """Apply Alembic migrations cleanly before booting the ASGI server."""
    ini_path = BACKEND_DIR / "alembic.ini"
    cfg = AlembicConfig(str(ini_path))
    alembic_command.upgrade(cfg, "head")


def build_app():
    bootstrap_database()
    from app.main import app as fastapi_app

    return fastapi_app


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860")
    asgi_app = build_app()
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)
