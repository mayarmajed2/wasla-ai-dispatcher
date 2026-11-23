"""Pytest path & isolation setup for Wasla AI."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Protect live PostgreSQL from accidental writes when running unit/mocked pytest suites locally
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(ROOT / '.pytest_isolated_default.db').as_posix()}")
os.environ.setdefault("APP_ENV", "development")
