"""Phase 3 Production Engineering Tests:
- Atomic DB Backup, Restore, SHA-256 Checksum & Tamper Detection
- Meta WhatsApp Cloud API v21.0 Outbound Send & 2-Step OGG Media Download (via MockTransport)
- Exact Egyptian Arabic WER & CER Calculation
- Readiness Probe (/api/ready) & Security Headers Verification
"""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.meta_service import MetaWhatsAppClient  # noqa: E402
from scripts.backup_restore import backup_sqlite, restore_sqlite  # noqa: E402
from scripts.eval_egyptian_audio_wer import compute_cer, compute_wer  # noqa: E402


def test_sqlite_backup_restore_and_checksum_tamper_detection(tmp_path):
    src_db = tmp_path / "source.db"
    con = sqlite3.connect(src_db)
    try:
        con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, total TEXT)")
        con.execute("INSERT INTO orders VALUES (1, 'مي طارق', '170.00')")
        con.commit()
    finally:
        con.close()

    backup_dir = tmp_path / "backups"
    backup_file, digest = backup_sqlite(src_db, backup_dir)
    assert backup_file.exists()
    assert len(digest) == 64

    # Delete original DB and restore from backup
    src_db.unlink()
    assert restore_sqlite(backup_file, src_db) is True

    con2 = sqlite3.connect(src_db)
    try:
        row = con2.execute("SELECT id, customer, total FROM orders WHERE id=1").fetchone()
        assert row == (1, "مي طارق", "170.00")
    finally:
        con2.close()

    # Tamper with backup file -> checksum verification MUST reject it
    backup_file.write_bytes(backup_file.read_bytes() + b"CORRUPT")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        restore_sqlite(backup_file, tmp_path / "tampered_target.db")


def test_meta_whatsapp_client_dry_run_and_graph_api_v21_contract():
    async def _run():
        # 1. Dry-run when unconfigured
        dry = MetaWhatsAppClient(access_token="", phone_number_id="")
        assert dry.is_configured is False
        res = await dry.send_text_message("201012345678", "أهلاً بك")
        assert res["sent"] is False
        assert res["mode"] == "dry_run_unconfigured"

        # 2. Live Graph API v21.0 contract via MockTransport
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured[request.url.path] = {"method": request.method, "auth": request.headers.get("Authorization")}
            if request.url.path.endswith("/messages"):
                body = json.loads(request.content.decode("utf-8"))
                assert body["messaging_product"] == "whatsapp"
                assert body["to"] == "201012345678"
                return httpx.Response(200, json={"messages": [{"id": "wamid.out_001"}]})
            if request.url.path.endswith("/media_123"):
                return httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/download/audio_1", "mime_type": "audio/ogg"})
            if request.url.path.endswith("/download/audio_1"):
                return httpx.Response(200, content=b"OggS\x00\x02MOCK_OPUS_BYTES")
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            live = MetaWhatsAppClient(
                access_token="test_token_123",
                phone_number_id="123456789",
                client=mock_http,
            )
            assert live.is_configured is True
            sent = await live.send_text_message("201012345678", "تم تأكيد طلبك")
            assert sent["sent"] is True
            assert sent["mode"] == "live"

            audio_bytes, mime = await live.download_media("media_123")
            assert audio_bytes.startswith(b"OggS")
            assert mime == "audio/ogg"

    asyncio.run(_run())


def test_egyptian_arabic_wer_and_cer_math():
    assert compute_wer("عايز بيتزا مارجريتا كبير", "عايز بيتزا مارجريتا كبير") == 0.0
    assert compute_cer("عايز بيتزا مارجريتا كبير", "عايز بيتزا مارجريتا كبير") == 0.0
    # 1 word substitution out of 4 = 0.25 WER
    assert compute_wer("عايز بيتزا مارجريتا كبير", "عايز بيتزا بيبروني كبير") == 0.25
    assert 0.0 < compute_cer("بيتزا مارجريتا", "بيتزا مرجريتا") < 0.20


def test_readiness_probe_and_security_headers(tmp_path):
    import os

    os.environ["DATABASE_URL"] = f"sqlite:///{(tmp_path / 'ready_test.db').as_posix()}"
    for mod in ["app.config", "app.database", "app.models", "app.main"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import app.main as main_mod  # noqa: E402

    async def _run():
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.get("/api/ready")
            assert r.status_code == 200
            data = r.json()
            assert data["ready"] is True
            assert data["database_connected"] is True
            assert "gemini_llm_configured" in data["providers"]
            assert "meta_whatsapp_configured" in data["providers"]
            assert r.headers.get("X-Content-Type-Options") == "nosniff"
            assert r.headers.get("X-Frame-Options") == "DENY"

    asyncio.run(_run())
