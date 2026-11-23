"""Automated Dataset Validation Pipeline (Schema, Duplicates, Leakage, Menu Grounding, PII)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "training" / "data"
MENU_PATH = ROOT / "menu" / "menu.json"
EVAL_150_PATH = ROOT / "datasets" / "evaluation" / "moka_typo_eval_150.json"

PHONE_RE = re.compile(r"\b01[0125]\d{8}\b")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate() -> dict:
    menu = json.loads(MENU_PATH.read_text(encoding="utf-8"))
    valid_pids = {p["product_id"] for p in menu["products"]}

    train = _load_jsonl(DATA_DIR / "train.jsonl")
    val = _load_jsonl(DATA_DIR / "val.jsonl")
    test = _load_jsonl(DATA_DIR / "test_unseen.jsonl")
    eval_150 = json.loads(EVAL_150_PATH.read_text(encoding="utf-8"))

    train_texts = {" ".join(r["text"].lower().split()) for r in train}
    val_texts = {" ".join(r["text"].lower().split()) for r in val}
    test_texts = {" ".join(r["text"].lower().split()) for r in test}
    eval_150_texts = {" ".join(r["utterance"].lower().split()) for r in eval_150}

    assert len(train_texts) == len(train), "Duplicate rows found in train.jsonl"
    assert len(val_texts) == len(val), "Duplicate rows found in val.jsonl"
    assert len(test_texts) == len(test), "Duplicate rows found in test_unseen.jsonl"

    # Zero leakage assertions
    assert not (train_texts & val_texts), "Data leakage between train and val!"
    assert not (train_texts & test_texts), "Data leakage between train and test_unseen!"
    assert not (val_texts & test_texts), "Data leakage between val and test_unseen!"
    assert not (train_texts & eval_150_texts), "Data leakage between train and moka_typo_eval_150!"

    # Validate menu IDs & zero PII
    for split_name, rows in [("train", train), ("val", val), ("test_unseen", test)]:
        for r in rows:
            assert r.get("text") and r.get("intent"), f"Missing text/intent in {split_name}"
            assert not PHONE_RE.search(r["text"]), f"PII phone number found in {split_name}: {r['text']}"
            for pid in r.get("expected_product_ids", []):
                assert pid in valid_pids, f"Invalid MOKA product_id {pid} in {split_name}"

    return {
        "valid": True,
        "train": len(train),
        "val": len(val),
        "test_unseen": len(test),
        "leakage_train_val": 0,
        "leakage_train_test": 0,
        "leakage_train_eval150": 0,
        "invalid_product_ids": 0,
        "pii_leaks": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
