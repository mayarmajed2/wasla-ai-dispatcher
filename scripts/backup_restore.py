"""Database Backup & Restore Utility with SHA-256 Integrity Verification.

Supports:
- SQLite: Atomic online backup via sqlite3.Connection.backup() + restore + SHA-256 verification.
- PostgreSQL: pg_dump / psql wrapper when DATABASE_URL starts with postgresql.

Usage:
    python scripts/backup_restore.py backup --db restaurant_ai.db --out backups/
    python scripts/backup_restore.py restore --backup backups/<file>.db --target restaurant_ai.db
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_sqlite(source_db: Path, backup_dir: Path) -> tuple[Path, str]:
    if not source_db.exists():
        raise FileNotFoundError(f"Source SQLite database not found: {source_db}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"{source_db.stem}_{ts}.db"

    src = sqlite3.connect(source_db)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    checksum = sha256_file(dest)
    (dest.with_suffix(".db.sha256")).write_text(f"{checksum}  {dest.name}\n", encoding="utf-8")
    return dest, checksum


def restore_sqlite(backup_file: Path, target_db: Path) -> bool:
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    checksum_file = backup_file.with_suffix(".db.sha256")
    if checksum_file.exists():
        expected = checksum_file.read_text(encoding="utf-8").strip().split()[0]
        actual = sha256_file(backup_file)
        if expected != actual:
            raise ValueError(f"Checksum mismatch! expected={expected}, actual={actual}")

    target_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_file, target_db)
    # Verify restored SQLite integrity
    conn = sqlite3.connect(target_db)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError(f"Restored DB failed integrity check: {result}")
    finally:
        conn.close()
    return True


def backup_postgres(database_url: str, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"wasla_pg_{ts}.sql"
    subprocess.run(["pg_dump", database_url, "-f", str(dest)], check=True)
    checksum = sha256_file(dest)
    dest.with_suffix(".sql.sha256").write_text(f"{checksum}  {dest.name}\n", encoding="utf-8")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wasla DB Backup & Restore Utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backup")
    b.add_argument("--db", default="restaurant_ai.db")
    b.add_argument("--out", default="backups")

    r = sub.add_parser("restore")
    r.add_argument("--backup", required=True)
    r.add_argument("--target", default="restaurant_ai.db")

    args = parser.parse_args()
    if args.cmd == "backup":
        if args.db.startswith("postgresql"):
            out_path = backup_postgres(args.db, Path(args.out))
            print(f"PostgreSQL backup created: {out_path}")
        else:
            out_path, digest = backup_sqlite(Path(args.db), Path(args.out))
            print(f"SQLite backup created: {out_path} (sha256={digest})")
    elif args.cmd == "restore":
        restore_sqlite(Path(args.backup), Path(args.target))
        print(f"SQLite backup restored & integrity verified -> {args.target}")
