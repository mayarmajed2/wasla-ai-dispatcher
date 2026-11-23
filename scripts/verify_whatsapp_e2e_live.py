"""Live Public HTTPS, Dashboard, Auth/RBAC, WebSocket, Scheduler & Meta Webhook End-to-End Verification Script.

Verifies an actually deployed FastAPI / HTTPS backend:
1. Dashboard GET / (HTTP 200 HTML)
2. GET /api/health & GET /api/ready (PostgreSQL connectivity)
3. GET /api/menu (145 MOKA products loaded)
4. Authentication & RBAC (/api/orders 401 -> /api/auth/login -> /api/auth/me -> /api/staff 200)
5. Pure-Python RFC 6455 WebSocket handshake on /ws/cashier (HTTP 101 Switching Protocols)
6. External Scheduler Cron-Tick security & execution (POST /api/scheduler/cron-tick)
7. Meta Webhook GET verification handshake (`hub.mode=subscribe`, `hub.verify_token`, `hub.challenge`)
8. Invalid HMAC-SHA256 rejection (401) when META_APP_SECRET is set
9. Signed HMAC-SHA256 POST /api/webhook (message + delivery statuses + duplicate replay rejection)

Never prints secrets.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import ssl
from urllib.parse import urlsplit
import uuid

import httpx


def _check_websocket_handshake(base_url: str, timeout: float = 15.0) -> dict:
    """Perform a minimal RFC 6455 HTTP/1.1 Upgrade handshake on /ws/cashier without extra dependencies."""
    parts = urlsplit(base_url)
    is_wss = parts.scheme == "https"
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if is_wss else 80)
    base_path = parts.path.rstrip("/")
    ws_path = f"{base_path}/ws/cashier" if base_path else "/ws/cashier"

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    raw_sock = socket.create_connection((host, port), timeout=timeout)
    sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=host) if is_wss else raw_sock
    try:
        req = (
            f"GET {ws_path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock.sendall(req.encode("ascii"))
        resp_head = sock.recv(2048).decode("utf-8", errors="replace")
        first_line = resp_head.splitlines()[0] if resp_head else ""
        return {
            "upgraded_101": "101" in first_line,
            "status_line": first_line,
        }
    except Exception as exc:
        return {"upgraded_101": False, "error": str(exc)}
    finally:
        try:
            sock.close()
        except Exception:
            pass


def run_check(
    base_url: str,
    verify_token: str,
    app_secret: str,
    test_phone: str,
    output_path: Path | None,
    *,
    admin_user: str = "",
    admin_pass: str = "",
    cron_secret: str = "",
) -> dict:
    base = base_url.rstrip("/")
    report: dict = {"base_url": base, "checks": {}, "status": "FAILED"}

    with httpx.Client(timeout=30.0) as client:
        # 1. Dashboard + Health & Readiness + Menu
        dash = client.get(f"{base}/")
        h = client.get(f"{base}/api/health")
        r = client.get(f"{base}/api/ready")
        m = client.get(f"{base}/api/menu")
        ready_body = r.json() if r.status_code == 200 else {"raw": r.text}
        menu_body = m.json() if m.status_code == 200 else []
        report["checks"]["dashboard_html"] = {"status_code": dash.status_code, "is_html": "html" in dash.headers.get("content-type", "")}
        report["checks"]["health"] = {"status_code": h.status_code, "body": h.json() if h.status_code == 200 else h.text}
        report["checks"]["ready"] = {"status_code": r.status_code, "body": ready_body}
        report["checks"]["menu"] = {"status_code": m.status_code, "items_count": len(menu_body) if isinstance(menu_body, list) else 0}

        # 2. Authentication & RBAC verification (optional if credentials supplied)
        unauth_orders = client.get(f"{base}/api/orders")
        report["checks"]["unauth_orders_blocked"] = {"status_code": unauth_orders.status_code, "blocked_401": unauth_orders.status_code == 401}
        if admin_user and admin_pass:
            login = client.post(f"{base}/api/auth/login", json={"username": admin_user, "password": admin_pass})
            token = (login.json() or {}).get("access_token", "") if login.status_code == 200 else ""
            headers_auth = {"Authorization": f"Bearer {token}"} if token else {}
            me = client.get(f"{base}/api/auth/me", headers=headers_auth) if token else None
            staff = client.get(f"{base}/api/staff", headers=headers_auth) if token else None
            report["checks"]["auth_rbac"] = {
                "login_status": login.status_code,
                "me_status": me.status_code if me else None,
                "staff_admin_status": staff.status_code if staff else None,
                "passed": bool(login.status_code == 200 and me and me.status_code == 200 and staff and staff.status_code == 200),
            }

        # 3. WebSocket /ws/cashier handshake
        report["checks"]["websocket_cashier"] = _check_websocket_handshake(base)

        # 4. External Cron-Tick verification (optional if secret supplied)
        if cron_secret:
            bad_cron = client.post(f"{base}/api/scheduler/cron-tick", headers={"X-Scheduler-Secret": "invalid-secret"})
            good_cron = client.post(f"{base}/api/scheduler/cron-tick", headers={"X-Scheduler-Secret": cron_secret})
            report["checks"]["scheduler_cron_tick"] = {
                "bad_secret_401": bad_cron.status_code == 401,
                "valid_secret_200": good_cron.status_code == 200,
                "body": good_cron.json() if good_cron.status_code == 200 else good_cron.text,
            }

        # 5. Webhook GET Verification Challenge
        challenge = f"wasla-challenge-{uuid.uuid4().hex[:8]}"
        wh_get = client.get(
            f"{base}/api/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": verify_token, "hub.challenge": challenge},
        )
        report["checks"]["webhook_get_challenge"] = {
            "status_code": wh_get.status_code,
            "challenge_matched": wh_get.text.strip() == challenge,
        }

        # 6. Signed POST Webhook + Delivery Status Update + Duplicate Replay Check
        msg_id = f"wamid.verify.{uuid.uuid4().hex[:12]}"
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "verify_entry",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [
                                    {
                                        "from": test_phone,
                                        "id": msg_id,
                                        "timestamp": "1763900000",
                                        "type": "text",
                                        "text": {"body": "السلام عليكم"},
                                    }
                                ],
                                "statuses": [
                                    {
                                        "id": f"wamid.out.{uuid.uuid4().hex[:8]}",
                                        "status": "delivered",
                                        "timestamp": "1763900005",
                                        "recipient_id": test_phone,
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        if app_secret:
            bad = client.post(
                f"{base}/api/webhook",
                content=raw,
                headers={"Content-Type": "application/json", "X-Hub-Signature-256": "sha256=invalid"},
            )
            report["checks"]["invalid_hmac_rejected"] = {"status_code": bad.status_code, "rejected_401": bad.status_code == 401}
            sig = "sha256=" + hmac.new(app_secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
            headers = {"Content-Type": "application/json", "X-Hub-Signature-256": sig}
        else:
            headers = {"Content-Type": "application/json"}

        post1 = client.post(f"{base}/api/webhook", content=raw, headers=headers)
        post2 = client.post(f"{base}/api/webhook", content=raw, headers=headers)
        p1_json = post1.json() if post1.status_code == 200 else {"error": post1.text}
        p2_json = post2.json() if post2.status_code == 200 else {"error": post2.text}
        report["checks"]["webhook_post_signed"] = {
            "status_code": post1.status_code,
            "processed": p1_json.get("processed"),
            "status_updates": p1_json.get("status_updates"),
        }
        report["checks"]["webhook_duplicate_prevented"] = {
            "status_code": post2.status_code,
            "processed": p2_json.get("processed"),
            "deduplicated": p2_json.get("processed") == 0 and p2_json.get("status_updates") == 0,
        }

    ok = (
        report["checks"]["dashboard_html"]["status_code"] == 200
        and report["checks"]["health"]["status_code"] == 200
        and report["checks"]["ready"]["status_code"] == 200
        and bool(ready_body.get("database_connected"))
        and report["checks"]["menu"]["items_count"] == 145
        and report["checks"]["unauth_orders_blocked"]["blocked_401"]
        and report["checks"]["websocket_cashier"]["upgraded_101"]
        and report["checks"]["webhook_get_challenge"]["challenge_matched"]
        and report["checks"]["webhook_post_signed"]["status_code"] == 200
        and report["checks"]["webhook_duplicate_prevented"]["deduplicated"]
        and (not admin_user or report["checks"].get("auth_rbac", {}).get("passed", False))
        and (not cron_secret or (report["checks"].get("scheduler_cron_tick", {}).get("bad_secret_401") and report["checks"].get("scheduler_cron_tick", {}).get("valid_secret_200")))
        and (not app_secret or report["checks"].get("invalid_hmac_rejected", {}).get("rejected_401", False))
    )
    report["status"] = "VERIFIED" if ok else "PARTIAL_OR_FAILED"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify deployed HTTPS backend and Meta WhatsApp webhook.")
    parser.add_argument("--base-url", required=True, help="Deployed HTTPS URL (e.g. https://wasla-moka.onrender.com)")
    parser.add_argument("--verify-token", default=os.getenv("META_VERIFY_TOKEN", "restaurant-ai-dev"))
    parser.add_argument("--app-secret", default=os.getenv("META_APP_SECRET", ""))
    parser.add_argument("--test-phone", default="201099990001")
    parser.add_argument("--admin-user", default=os.getenv("BOOTSTRAP_ADMIN_USERNAME", ""))
    parser.add_argument("--admin-pass", default=os.getenv("BOOTSTRAP_ADMIN_PASSWORD", ""))
    parser.add_argument("--cron-secret", default=os.getenv("SCHEDULER_CRON_SECRET", ""))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None
    report = run_check(
        args.base_url,
        args.verify_token,
        args.app_secret,
        args.test_phone,
        out_path,
        admin_user=args.admin_user,
        admin_pass=args.admin_pass,
        cron_secret=args.cron_secret,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
