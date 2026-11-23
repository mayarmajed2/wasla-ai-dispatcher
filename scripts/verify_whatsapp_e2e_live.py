"""Live Public HTTPS & Meta WhatsApp Webhook End-to-End Verification Script.

Verifies an actually deployed HTTPS backend:
1. GET /api/health & GET /api/ready
2. Meta Webhook GET verification handshake (`hub.mode=subscribe`, `hub.verify_token`, `hub.challenge`)
3. Invalid HMAC-SHA256 signature rejection (401) when META_APP_SECRET is set
4. Signed HMAC-SHA256 POST /api/webhook callback with Egyptian Arabic test message
5. Duplicate `meta_message_id` prevention (processed == 0 on replay)

Never prints secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import uuid

import httpx


def run_check(base_url: str, verify_token: str, app_secret: str, test_phone: str, output_path: Path | None) -> dict:
    base = base_url.rstrip("/")
    report: dict = {"base_url": base, "checks": {}, "status": "FAILED"}

    with httpx.Client(timeout=30.0) as client:
        # 1. Health & Readiness
        h = client.get(f"{base}/api/health")
        r = client.get(f"{base}/api/ready")
        report["checks"]["health"] = {"status_code": h.status_code, "body": h.json() if h.status_code == 200 else h.text}
        report["checks"]["ready"] = {"status_code": r.status_code, "body": r.json() if r.status_code == 200 else r.text}

        # 2. Webhook GET Verification Challenge
        challenge = f"wasla-challenge-{uuid.uuid4().hex[:8]}"
        wh_get = client.get(
            f"{base}/api/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": verify_token, "hub.challenge": challenge},
        )
        report["checks"]["webhook_get_challenge"] = {
            "status_code": wh_get.status_code,
            "challenge_matched": wh_get.text.strip() == challenge,
        }

        # 3. Signed POST Webhook + Duplicate Replay Check
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
        report["checks"]["webhook_post_signed"] = {"status_code": post1.status_code, "processed": p1_json.get("processed")}
        report["checks"]["webhook_duplicate_prevented"] = {
            "status_code": post2.status_code,
            "processed": p2_json.get("processed"),
            "deduplicated": p2_json.get("processed") == 0,
        }

    ok = (
        report["checks"]["health"]["status_code"] == 200
        and report["checks"]["webhook_get_challenge"]["challenge_matched"]
        and report["checks"]["webhook_post_signed"]["status_code"] == 200
        and report["checks"]["webhook_duplicate_prevented"]["deduplicated"]
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
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None
    report = run_check(args.base_url, args.verify_token, args.app_secret, args.test_phone, out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
