"""Real PyTorch + Hugging Face Transformers + PEFT LoRA Fine-Tuning Script.

Trains a LoRA adapter on `CAMeL-Lab/bert-base-arabic-camelbert-mix` (MIT License)
for Egyptian Arabic MOKA Restaurant Intent Classification.

Saves:
- Real PEFT LoRA adapter checkpoint in `training/checkpoints/camelbert-wasla-lora/`
- Reproducible training metrics & loss curve in `training/reports/training_log.json`
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "training" / "configs" / "lora_intent_config.json"
DATA_DIR = ROOT / "training" / "data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class IntentDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, label2id: dict[str, int], max_length: int):
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


def evaluate_loop(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += float(out.loss.item()) * len(batch["labels"])
            preds = out.logits.argmax(dim=-1)
            correct += int((preds == batch["labels"]).sum().item())
            total += len(batch["labels"])
    return round(total_loss / max(1, total), 4), round(correct / max(1, total), 4)


def train() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "val.jsonl")

    labels = sorted({r["intent"] for r in train_rows})
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    base_model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
    )
    model = get_peft_model(base_model, lora_cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_ds = IntentDataset(train_rows, tokenizer, label2id, cfg["max_length"])
    val_ds = IntentDataset(val_rows, tokenizer, label2id, cfg["max_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    epochs = cfg["training"]["epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["training"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # Measure untrained baseline on val before epoch 1
    base_val_loss, base_val_acc = evaluate_loop(model, val_loader, device)
    print(f"[Epoch 0 / Untrained Base] val_loss={base_val_loss:.4f} | val_acc={base_val_acc:.2%}", flush=True)

    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(out.loss.item()) * len(batch["labels"])
            seen += len(batch["labels"])

        train_loss = round(running_loss / max(1, seen), 4)
        val_loss, val_acc = evaluate_loop(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2%}",
            flush=True,
        )

    duration_sec = round(time.perf_counter() - started, 2)

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "experiment_name": cfg["experiment_name"],
        "base_model": base_model_id,
        "license": cfg["base_model_license"],
        "device": str(device),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio_pct": round(100.0 * trainable_params / total_params, 4),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "duration_seconds": duration_sec,
        "epoch_0_base_val_loss": base_val_loss,
        "epoch_0_base_val_acc": base_val_acc,
        "epochs": history,
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
    }
    report_dir = ROOT / cfg["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_log.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = train()
    print("\nTRAINING COMPLETE:", json.dumps(res, ensure_ascii=False, indent=2))
