"""Improved PEFT LoRA Fine-Tuning v2 (Class-Weighted Loss + Hard Negatives + Best Val Macro-F1 Selection).

Fixes the low Macro-F1 (0.5382) of v1 by:
1. Using inverse-frequency class-weighted CrossEntropyLoss to eliminate majority-class dominance.
2. Adding curated hard-negative contrastive examples for confused intent pairs
   (delivery_inquiry vs price_inquiry, cancel_order vs remove_item, modify_order vs add_to_cart).
3. Increasing LoRA capacity (r=16, alpha=32, target_modules=["query", "key", "value"]).
4. Selecting & saving the BEST checkpoint strictly by Validation Macro-F1 (not final epoch).
5. Preserving v1 checkpoint untouched at `training/checkpoints/camelbert-wasla-lora/` for rollback/comparison.

Base Model: CAMeL-Lab/bert-base-arabic-camelbert-mix (Verified Official License: Apache-2.0)
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "training" / "data"
IND_200_PATH = ROOT / "datasets" / "evaluation" / "independent_challenge_200.jsonl"
EVAL_150_PATH = ROOT / "datasets" / "evaluation" / "moka_typo_eval_150.json"
OUT_CKPT_DIR = ROOT / "training" / "checkpoints" / "camelbert-wasla-lora-v2"
REPORT_PATH = ROOT / "training" / "reports" / "training_v2_log.json"

BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
SEED = 42

# Hard-negative contrastive examples (strictly excluded from independent_challenge_200 & moka_typo_eval_150)
HARD_NEGATIVES: list[tuple[str, str]] = [
    # delivery_inquiry vs price_inquiry (explicitly contrast "بكام" with delivery words)
    ("خدمة التوصيل بكام؟", "delivery_inquiry"),
    ("رسوم الشحن والتوصيل كام؟", "delivery_inquiry"),
    ("التوصيل للمنطقة دي بيكلف كام؟", "delivery_inquiry"),
    ("هو الدليفري بفلوس ولا مجاني؟", "delivery_inquiry"),
    ("بتوصلوا لمناطق إيه بالظبط؟", "delivery_inquiry"),
    ("الطيار بيوصل في قد إيه؟", "delivery_inquiry"),
    ("في توصيل للزمالك والدقي؟", "delivery_inquiry"),
    ("مصاريف التوصيل كام للمقطم؟", "delivery_inquiry"),
    ("الدليفري شغال لحد الساعة كام؟", "delivery_inquiry"),
    ("بتوصلوا لحد باب الشقة؟", "delivery_inquiry"),
    ("كام سعر الدليفري عندكم؟", "delivery_inquiry"),
    ("الدليفري بياخد وقت كتير؟", "delivery_inquiry"),
    # cancel_order vs remove_item
    ("الغي الأوردر بالكامل خلاص مش عايز", "cancel_order"),
    ("كنسل الطلب كله لو سمحت", "cancel_order"),
    ("مش هنكمل الطلب الغيه كله", "cancel_order"),
    ("الغي الطلب اللي عملته دلوقتي", "cancel_order"),
    ("شيل البطاطس بس من الطلب", "remove_item"),
    ("احذف المشروب ده وسيب الباقي", "remove_item"),
    ("امسح الكريب من السلة وخلي البيتزا", "remove_item"),
    ("طلع الوافل من السلة لو سمحت", "remove_item"),
    # modify_order vs add_to_cart
    ("خلي العدد تلاتة بدل اتنين", "modify_order"),
    ("غير الحجم من وسط لكبير", "modify_order"),
    ("بدل اللحم بفراخ في البرجر", "modify_order"),
    ("شيل البصل والطماطم من الساندوتش", "modify_order"),
    ("قصدي صوص باربيكيو مش رانش", "modify_order"),
    ("عدّل الملاحظات واكتب بدون هالبينو", "modify_order"),
    # confirm_order vs greeting
    ("تمام اعتمد الطلب وابعته للمطبخ", "confirm_order"),
    ("ايوه موافق على السعر أكد الأوردر", "confirm_order"),
    ("نفذ الأوردر دلوقتي خلاص", "confirm_order"),
    # request_human vs complaint
    ("حولني للكاشير أكلمه بنفسي", "request_human"),
    ("عايز أتكلم مع موظف مش بوت", "request_human"),
    ("الأكل جالي بارد وناقص أصناف", "complaint"),
    ("الطلب اتأخر أكتر من ساعة ونص", "complaint"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class IntentDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, label2id: dict[str, int], max_length: int = 64):
        self.encodings = tokenizer(
            [r["text"] for r in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([label2id[r["intent"]] for r in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def eval_split(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = criterion(out.logits, labels)
            total_loss += float(loss.item()) * len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(out.logits.argmax(dim=-1).cpu().tolist())
    avg_loss = round(total_loss / max(1, len(y_true)), 4)
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    return avg_loss, acc, macro_f1


def train_v2() -> dict:
    set_seed(SEED)
    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    # Build forbidden set from independent_challenge_200 & moka_typo_eval_150 to guarantee ZERO leakage
    forbidden = set()
    for r in load_jsonl(IND_200_PATH):
        forbidden.add(" ".join(r["text"].lower().split()))
    for r in json.loads(EVAL_150_PATH.read_text(encoding="utf-8")):
        forbidden.add(" ".join(r["utterance"].lower().split()))

    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "val.jsonl")

    # Add verified non-leaking hard negatives to train split
    existing_keys = {" ".join(r["text"].lower().split()) for r in train_rows + val_rows}
    added_hn = 0
    for txt, intent in HARD_NEGATIVES:
        k = " ".join(txt.lower().split())
        if k not in forbidden and k not in existing_keys:
            train_rows.append({"text": txt, "intent": intent, "data_origin": "hard_negative_synthetic"})
            existing_keys.add(k)
            added_hn += 1

    labels = sorted({r["intent"] for r in train_rows})
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # Compute inverse-frequency class weights (capped for stability)
    counts = Counter(r["intent"] for r in train_rows)
    total_n = len(train_rows)
    num_classes = len(labels)
    weights = torch.tensor(
        [min(4.0, total_n / (num_classes * counts[lbl])) for lbl in labels],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_classes, id2label=id2label, label2id=label2id
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(IntentDataset(train_rows, tokenizer, label2id), batch_size=16, shuffle=True)
    val_loader = DataLoader(IntentDataset(val_rows, tokenizer, label2id), batch_size=16, shuffle=False)

    epochs = 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_macro_f1 = -1.0
    best_epoch = 0
    history = []
    started = time.perf_counter()

    OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            labels_t = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = criterion(out.logits, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item()) * len(labels_t)
            seen += len(labels_t)

        train_loss = round(running_loss / max(1, seen), 4)
        val_loss, val_acc, val_macro_f1 = eval_split(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1})
        print(
            f"[v2 Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2%} | val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )

        # Save BEST checkpoint by Validation Macro-F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            model.save_pretrained(OUT_CKPT_DIR)
            tokenizer.save_pretrained(OUT_CKPT_DIR)
            (OUT_CKPT_DIR / "label_map.json").write_text(
                json.dumps({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    duration_sec = round(time.perf_counter() - started, 2)
    report = {
        "experiment_name": "wasla-camelbert-mix-lora-v2",
        "base_model": BASE_MODEL,
        "verified_license": "Apache-2.0",
        "device": str(device),
        "lora": {"r": 16, "lora_alpha": 32, "target_modules": ["query", "key", "value"]},
        "class_weighted_loss": True,
        "hard_negatives_added": added_hn,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "duration_seconds": duration_sec,
        "epochs": history,
        "checkpoint_dir": str(OUT_CKPT_DIR.relative_to(ROOT)),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print("\nV2 TRAINING COMPLETE:", json.dumps(train_v2(), ensure_ascii=False, indent=2))
