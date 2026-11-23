"""Scientific 3-Way Held-Out Evaluation on Untouched `training/data/test_unseen.jsonl`.

Compares:
- Model A: Untrained Base Model (`CAMeL-Lab/bert-base-arabic-camelbert-mix`)
- Model B: Fine-Tuned PEFT LoRA Model (`training/checkpoints/camelbert-wasla-lora`)
- Model C: Existing Wasla AI Hybrid System (`WaslaAIDispatcher` + `OfflineEgyptianNLU`)

Reports real measured Intent Accuracy, Macro-F1, Weighted-F1, Hallucinated Product Rate,
and average inference latency (ms). Never fabricates metrics.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai_service.adapters.mock_backend import MockBackendAdapter  # noqa: E402
from ai_service.config import WaslaConfig  # noqa: E402
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher  # noqa: E402
from ai_service.understanding.menu_index import MenuIndex  # noqa: E402

DATA_PATH = ROOT / "training" / "data" / "test_unseen.jsonl"
CKPT_DIR = ROOT / "training" / "checkpoints" / "camelbert-wasla-lora"
CONFIG_PATH = ROOT / "training" / "configs" / "lora_intent_config.json"
REPORT_PATH = ROOT / "training" / "reports" / "evaluation_comparison.json"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_hf_model(model, tokenizer, rows: list[dict], id2label: dict[int, str], max_length: int) -> dict:
    model.eval()
    y_true = [r["intent"] for r in rows]
    y_pred = []
    latencies_ms = []

    with torch.no_grad():
        for r in rows:
            t0 = time.perf_counter()
            enc = tokenizer(r["text"], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            logits = model(**enc).logits
            pred_idx = int(logits.argmax(dim=-1).item())
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            y_pred.append(id2label[pred_idx])

    return {
        "intent_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "avg_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
        "samples": len(rows),
    }


async def _eval_wasla_hybrid(rows: list[dict]) -> dict:
    cfg = WaslaConfig(gemini_api_key="")  # Deterministic hybrid NLU baseline
    idx = MenuIndex(ROOT / "menu" / "menu.json")
    valid_pids = set(idx.products.keys())

    y_true = [r["intent"] for r in rows]
    y_pred = []
    latencies_ms = []
    hallucinated = 0

    for i, r in enumerate(rows):
        dispatcher = WaslaAIDispatcher(cfg, idx, MockBackendAdapter(idx))
        t0 = time.perf_counter()
        resp = await dispatcher.handle_text(f"01066{i:05d}", r["text"])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        y_pred.append(resp.intent.value)
        for item in resp.extracted_items:
            if item.product_id and item.product_id not in valid_pids:
                hallucinated += 1

    return {
        "intent_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "hallucinated_product_count": hallucinated,
        "avg_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
        "samples": len(rows),
    }


def main() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = _load_jsonl(DATA_PATH)
    label_map = json.loads((CKPT_DIR / "label_map.json").read_text(encoding="utf-8"))
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    base_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)

    # Model A: Untrained base model with freshly initialized classification head
    torch.manual_seed(cfg["seed"])
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    res_a = _eval_hf_model(base_model, tokenizer, rows, id2label, cfg["max_length"])

    # Model B: Real Fine-Tuned PEFT LoRA checkpoint loaded from disk
    base_for_lora = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, CKPT_DIR)
    res_b = _eval_hf_model(lora_model, tokenizer, rows, id2label, cfg["max_length"])

    # Model C: Existing Wasla AI Hybrid NLU System
    res_c = asyncio.run(_eval_wasla_hybrid(rows))

    report = {
        "evaluation_dataset": "training/data/test_unseen.jsonl",
        "dataset_origin": "synthetic_moka_grounded (held-out unseen split, zero overlap with train & moka_typo_eval_150)",
        "samples": len(rows),
        "model_a_untrained_base": {"name": base_id, **res_a},
        "model_b_finetuned_lora": {"name": f"{base_id} + PEFT LoRA", "checkpoint": str(CKPT_DIR.relative_to(ROOT)), **res_b},
        "model_c_wasla_hybrid_nlu": {"name": "Wasla OfflineEgyptianNLU + Fuzzy MenuIndex", **res_c},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
