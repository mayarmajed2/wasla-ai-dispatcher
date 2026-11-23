"""Production Authentication & Role-Based Access Control (Admin / Cashier).

Uses PBKDF2-HMAC-SHA256 password hashing (200,000 iterations + per-user random salt)
and HMAC-SHA256 signed bearer tokens with expiration. Pure Python standard library.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import AuditLog, User

_bearer = HTTPBearer(auto_error=False)
_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iters_s, salt, digest_hex = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iters_s))
        return hmac.compare_digest(dk.hex(), digest_hex)
    except Exception:
        return False


def create_access_token(user: User) -> str:
    payload = {
        "sub": user.username,
        "role": user.role,
        "name": user.full_name,
        "exp": int(time.time()) + (settings.auth_token_ttl_hours * 3600),
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    sig = hmac.new(settings.auth_secret_key.encode("utf-8"), raw.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        raw, sig = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid token format") from exc

    expected = hmac.new(settings.auth_secret_key.encode("utf-8"), raw.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    payload = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_access_token(creds.credentials)
    user = db.scalar(select(User).where(User.username == payload["sub"], User.is_active.is_(True)))
    if user is None:
        raise HTTPException(status_code=401, detail="User inactive or not found")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin permission required")
    return user


def log_audit(db: Session, actor: str, action: str, target: str = "", details: dict[str, Any] | None = None) -> None:
    db.add(AuditLog(actor_username=actor, action=action, target=target, details=details or {}))
