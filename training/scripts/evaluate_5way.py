"""5-Way Scientific Evaluation + 12x12 Confusion Matrix + Per-Class Precision/Recall/F1
on the Genuinely Independent 204-Example Challenge Benchmark (`independent_challenge_200.jsonl`).

Compares:
- Model A: Supervised TF-IDF (Char+Word N-gram) + Class-Balanced LogisticRegression Baseline
- Model B: Previous CAMeLBERT LoRA v1 (`training/checkpoints/camelbert-wasla-lora`)
- Model C: Improved Class-Weighted CAMeLBERT LoRA v2 (`training/checkpoints/camelbert-wasla-lora-v2`)
- Model D: Existing Wasla Rule/Fuzzy NLU (`OfflineEgyptianNLU`)
- Model E: Final Hybrid System (v2 LoRA + Deterministic MenuIndex + Safety Guardrails)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai_service.adapters.mock_backend import MockBackendAdapter  # noqa: E402
from ai_service.config import WaslaConfig  # noqa: E402
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher  # noqa: E402
from ai_service.understanding.menu_index import MenuIndex  # noqa: E402

TRAIN_PATH = ROOT / "training" / "data" / "train.jsonl"
IND_200_PATH = ROOT / "datasets" / "evaluation" / "independent_challenge_200.jsonl"
CKPT_V1 = ROOT / "training" / "checkpoints" / "camelbert-wasla-lora"
CKPT_V2 = ROOT / "training" / "checkpoints" / "camelbert-wasla-lora-v2"
BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
OUT_REPORT = ROOT / "training" / "reports" / "evaluation_5way_independent.json"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics_block(y_true: list[str], y_pred: list[str], labels: list[str], latencies: list[float]) -> dict:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        lbl: {
            "precision": round(rep[lbl]["precision"], 4),
            "recall": round(rep[lbl]["recall"], 4),
            "f1": round(rep[lbl]["f1-score"], 4),
            "support": int(rep[lbl]["support"]),
        }
        for lbl in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else 0.0
    return {
        "intent_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "p95_latency_ms": round(p95, 2),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def _eval_lora(ckpt_dir: Path, rows: list[dict], labels: list[str]) -> tuple[dict, list[tuple[str, float]]]:
    label_map = json.loads((ckpt_dir / "label_map.json").read_text(encoding="utf-8"))
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()

    y_true = [r["intent"] for r in rows]
    y_pred = []
    preds_with_conf = []
    lats = []

    with torch.no_grad():
        for r in rows:
            t0 = time.perf_counter()
            enc = tok(r["text"], truncation=True, padding=True, max_length=64, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            lats.append((time.perf_counter() - t0) * 1000.0)
            lbl = id2label[idx]
            y_pred.append(lbl)
            preds_with_conf.append((lbl, float(probs[idx].item())))

    return _metrics_block(y_true, y_pred, labels, lats), preds_with_conf


async def _eval_wasla_rule_nlu(rows: list[dict], labels: list[str]) -> tuple[dict, list[tuple[str, float]]]:
    cfg = WaslaConfig(gemini_api_key="")
    idx = MenuIndex(ROOT / "menu" / "menu.json")
    y_true = [r["intent"] for r in rows]
    y_pred = []
    rule_preds = []
    lats = []

    for i, r in enumerate(rows):
        dispatcher = WaslaAIDispatcher(cfg, idx, MockBackendAdapter(idx))
        t0 = time.perf_counter()
        resp = await dispatcher.handle_text(f"01077{i:05d}", r["text"])
        lats.append((time.perf_counter() - t0) * 1000.0)
        y_pred.append(resp.intent.value)
        rule_preds.append((resp.intent.value, float(resp.confidence)))

    return _metrics_block(y_true, y_pred, labels, lats), rule_preds


def main() -> dict:
    train_rows = _load_jsonl(TRAIN_PATH)
    ind_rows = _load_jsonl(IND_200_PATH)
    labels = sorted({r["intent"] for r in ind_rows})
    y_true = [r["intent"] for r in ind_rows]

    # Model A: Supervised TF-IDF + Balanced LogisticRegression
    pipe = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2))),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit([r["text"] for r in train_rows], [r["intent"] for r in train_rows])
    lats_a = []
    y_pred_a = []
    for r in ind_rows:
        t0 = time.perf_counter()
        y_pred_a.append(str(pipe.predict([r["text"]])[0]))
        lats_a.append((time.perf_counter() - t0) * 1000.0)
    res_a = _metrics_block(y_true, y_pred_a, labels, lats_a)

    # Model B: Previous CAMeLBERT LoRA v1
    res_b, _ = _eval_lora(CKPT_V1, ind_rows, labels)

    # Model C: Improved Class-Weighted CAMeLBERT LoRA v2
    res_c, v2_preds = _eval_lora(CKPT_V2, ind_rows, labels)

    # Model D: Existing Wasla Hybrid Rule/Fuzzy NLU
    res_d, d_preds = asyncio.run(_eval_wasla_rule_nlu(ind_rows, labels))

    # Model E: Final Combined Hybrid (Rule high-confidence safety + v2 LoRA when rule uncertain/ambiguous)
    y_pred_e = []
    for (v2_lbl, v2_conf), (d_lbl, d_conf) in zip(v2_preds, d_preds):
        if d_lbl == "unclear_request" and d_conf < 0.9:
            y_pred_e.append(v2_lbl if v2_conf >= 0.45 else d_lbl)
        elif d_conf >= 0.90:
            y_pred_e.append(d_lbl)
        else:
            y_pred_e.append(v2_lbl if v2_conf >= 0.55 else d_lbl)
    res_e = _metrics_block(y_true, y_pred_e, labels, [res_c["avg_latency_ms"]])

    report = {
        "benchmark": "datasets/evaluation/independent_challenge_200.jsonl",
        "samples": len(ind_rows),
        "model_a_tfidf_logreg_baseline": res_a,
        "model_b_camelbert_lora_v1": res_b,
        "model_c_camelbert_lora_v2": res_c,
        "model_d_wasla_rule_fuzzy_nlu": res_d,
        "model_e_final_hybrid_v2_plus_rules": res_e,
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    for k in [
        "model_a_tfidf_logreg_baseline",
        "model_b_camelbert_lora_v1",
        "model_c_camelbert_lora_v2",
        "model_d_wasla_rule_fuzzy_nlu",
        "model_e_final_hybrid_v2_plus_rules",
    ]:
        m = rep[k]
        print(f"{k:38s} | Acc={m['intent_accuracy']:.2%} | Macro-F1={m['macro_f1']:.4f} | Weighted-F1={m['weighted_f1']:.4f} | Avg={m['avg_latency_ms']}ms")
