"""Seed MOKA's 145 confirmed menu items from menu/menu.json into the existing Backend DB.

Usage:
    python scripts/seed_moka_menu.py
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "existing_system" / "RESTURANT AI"
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import delete  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models import MenuItem  # noqa: E402


def seed_moka_menu(replace_existing: bool = False) -> int:
    menu_data = json.loads((ROOT / "menu" / "menu.json").read_text(encoding="utf-8"))
    categories = {c["category_id"]: c["name_ar"] for c in menu_data["categories"]}
    modifiers_map = {m["modifier_id"]: m for m in menu_data["modifiers"]}

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if replace_existing:
            db.execute(delete(MenuItem))
            db.commit()

        inserted = 0
        for p in menu_data["products"]:
            base_price = p["base_price"]
            if base_price is None and p.get("sizes"):
                base_price = p["sizes"][0]["price"]

            extras = [
                {"name": modifiers_map[mid]["name_ar"], "price": str(modifiers_map[mid]["price"])}
                for mid in p.get("modifiers", [])
                if mid in modifiers_map
            ]
            sizes = [
                {"code": s["code"], "name": s["name_ar"], "price": str(s["price"])}
                for s in p.get("sizes", [])
            ]

            db.add(
                MenuItem(
                    name=p["name_ar"],
                    category=categories.get(p["category_id"], p["category_id"]),
                    base_price=Decimal(str(base_price)),
                    is_available=True,
                    modifiers_json={
                        "product_id": p["product_id"],
                        "sizes": sizes,
                        "extras": extras,
                        "options": p.get("options", []),
                    },
                )
            )
            inserted += 1
        db.commit()
        return inserted
    finally:
        db.close()


if __name__ == "__main__":
    count = seed_moka_menu(replace_existing=True)
    print(f"Seeded {count} MOKA menu items into existing Backend database.")
