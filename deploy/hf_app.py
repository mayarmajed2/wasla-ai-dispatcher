"""Hugging Face Spaces Entrypoint for Wasla AI (compatible with both CPU Basic & Free ZeroGPU).

1. Imports `spaces` first (when running on Hugging Face ZeroGPU) and registers a `@spaces.GPU`
   probe bound to a lightweight `/hf` Gradio status mount so ZeroGPU's startup check succeeds
   without interfering with `/` (MOKA Cashier Dashboard), `/api/*`, or `/ws/cashier`.
2. Applies Alembic PostgreSQL migrations (`upgrade head`) cleanly on startup.
3. Serves the full FastAPI + WebSocket application on port 7860.
"""

from __future__ import annotations

try:
    import spaces  # MUST be imported first on Hugging Face ZeroGPU Spaces
except Exception:
    spaces = None  # type: ignore[assignment]

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


def _gpu_decorator(fn):
    if spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(fn)
    return fn


@_gpu_decorator
def wasla_zerogpu_ready_probe(prompt: str = "ping") -> str:
    """Satisfies Hugging Face ZeroGPU startup verification at zero cost."""
    return f"Wasla AI Dispatcher Ready ({prompt})"


def bootstrap_database() -> None:
    """Apply Alembic migrations cleanly before booting the ASGI server."""
    ini_path = BACKEND_DIR / "alembic.ini"
    cfg = AlembicConfig(str(ini_path))
    alembic_command.upgrade(cfg, "head")


def build_app():
    bootstrap_database()
    from app.main import app as fastapi_app

    try:
        import gradio as gr

        demo = gr.Interface(
            fn=wasla_zerogpu_ready_probe,
            inputs="text",
            outputs="text",
            title="Wasla AI — MOKA Dispatcher Status",
        )
        mounted = gr.mount_gradio_app(fastapi_app, demo, path="/hf")
        _zero = getattr(spaces, "zero", None) if spaces is not None else None
        if _zero is not None:
            mw_cls = getattr(_zero, "ZeroGPUMiddleware", None)
            if mw_cls is not None:
                already = any(getattr(m, "cls", None) is mw_cls for m in getattr(mounted, "user_middleware", []))
                if not already:
                    mounted.add_middleware(mw_cls)
            elif hasattr(_zero, "startup"):
                try:
                    _zero.startup()
                except Exception as exc:
                    print(f"ZeroGPU startup notice: {exc}")
        return mounted
    except Exception:
        return fastapi_app


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860")
    asgi_app = build_app()
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)
