"""Phase B Scientific 5-Way AI Evaluation on Held-Out Unseen Test, Independent Challenge & Typo Suites.

Compares without ever mixing training/validation data into test metrics:
1. CAMeLBERT Base (`CAMeL-Lab/bert-base-arabic-camelbert-mix`, un-finetuned sequence classifier, seed=42)
2. CAMeLBERT LoRA v1 (`training/checkpoints/camelbert-wasla-lora`)
3. CAMeLBERT LoRA v2 (`training/checkpoints/camelbert-wasla-lora-v2`)
4. Gemini (`GeminiLLMClient`, bounded live evaluation on stratified balanced subset if GEMINI_API_KEY is available)
5. Wasla Hybrid (Repaired `WaslaAIDispatcher` combining CAMeLBERT LoRA v2 + Deterministic MenuIndex + Safety)

Measures & reports:
- Intent Accuracy, Macro Precision, Macro Recall, Macro-F1, Weighted-F1, Per-Intent F1, 12x12 Confusion Matrix
- Latency P50 & P95 (ms), Model Loading Time (ms), Process RSS Memory Usage (MB via psutil)
- Fallback frequency, Incorrect Product Extraction Rate & False Positive Rate for Nonexistent Products
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_service.adapters.mock_backend import MockBackendAdapter  # noqa: E402
from ai_service.config import WaslaConfig  # noqa: E402
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher  # noqa: E402
from ai_service.models.finetuned_intent import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    V1_CHECKPOINT_DIR,
    FineTunedIntentClassifier,
)
from ai_service.models.llm_client import GeminiLLMClient  # noqa: E402
from ai_service.understanding.menu_index import MenuIndex  # noqa: E402

TEST_UNSEEN_PATH = ROOT / "training" / "data" / "test_unseen.jsonl"
IND_200_PATH = ROOT / "datasets" / "evaluation" / "independent_challenge_203_v2.jsonl"
TYPO_EVAL_PATH = ROOT / "datasets" / "evaluation" / "moka_typo_eval_150.json"
OUT_PATH = ROOT / "training" / "reports" / "phase_b_scientific_evaluation_v2.json"
BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)


def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    latencies: list[float],
    *,
    load_time_ms: float | None = None,
    mem_before_mb: float | None = None,
    mem_after_mb: float | None = None,
    fallback_count: int = 0,
) -> dict[str, Any]:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        lbl: {
            "precision": round(float(rep[lbl]["precision"]), 4),
            "recall": round(float(rep[lbl]["recall"]), 4),
            "f1": round(float(rep[lbl]["f1-score"]), 4),
            "support": int(rep[lbl]["support"]),
        }
        for lbl in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    sorted_lat = sorted(latencies)
    p50 = round(float(statistics.median(sorted_lat)), 2) if sorted_lat else 0.0
    p95_idx = min(len(sorted_lat) - 1, max(0, int(len(sorted_lat) * 0.95) - 1)) if sorted_lat else 0
    p95 = round(float(sorted_lat[p95_idx]), 2) if sorted_lat else 0.0

    return {
        "samples": len(y_true),
        "intent_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_avg_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "load_time_ms": load_time_ms,
        "memory_rss_before_mb": mem_before_mb,
        "memory_rss_after_mb": mem_after_mb,
        "memory_delta_mb": round(mem_after_mb - mem_before_mb, 2) if (mem_before_mb and mem_after_mb) else None,
        "fallback_frequency_pct": round((fallback_count / max(1, len(y_true))) * 100.0, 2),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def _eval_camelbert_base(datasets: dict[str, list[dict[str, Any]]], labels: list[str]) -> dict[str, Any]:
    torch.manual_seed(42)
    label_map = json.loads((DEFAULT_CHECKPOINT_DIR / "label_map.json").read_text(encoding="utf-8"))
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    mem0 = _rss_mb()
    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id, local_files_only=True
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )
    model.eval()
    load_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)
    mem1 = _rss_mb()

    results: dict[str, Any] = {}
    for split_name, rows in datasets.items():
        y_true = [r["intent"] for r in rows]
        y_pred = []
        lats = []
        with torch.no_grad():
            for r in rows:
                t0 = time.perf_counter()
                enc = tok(r["text"], truncation=True, padding=True, max_length=64, return_tensors="pt")
                logits = model(**enc).logits
                best_idx = int(torch.softmax(logits, dim=-1)[0].argmax().item())
                lats.append((time.perf_counter() - t0) * 1000.0)
                y_pred.append(id2label[best_idx])
        results[split_name] = _compute_metrics(
            y_true, y_pred, labels, lats, load_time_ms=load_ms, mem_before_mb=mem0, mem_after_mb=mem1
        )
    results["notes"] = "Un-finetuned CAMeLBERT-Mix base encoder with seeded classification head (seed=42)."
    return results


def _eval_lora_clf(ckpt_dir: Path, datasets: dict[str, list[dict[str, Any]]], labels: list[str]) -> dict[str, Any]:
    mem0 = _rss_mb()
    t_load0 = time.perf_counter()
    clf = FineTunedIntentClassifier(checkpoint_dir=ckpt_dir)
    ok = clf.load()
    load_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)
    mem1 = _rss_mb()
    if not ok:
        return {"error": clf.load_error}

    results: dict[str, Any] = {}
    for split_name, rows in datasets.items():
        y_true = [r["intent"] for r in rows]
        y_pred = []
        lats = []
        fallback_count = 0
        for r in rows:
            pred = clf.predict(r["text"])
            if pred is None:
                fallback_count += 1
                y_pred.append("unclear_request")
                lats.append(0.0)
            else:
                lbl, _, lat = pred
                y_pred.append(lbl)
                lats.append(lat)
        results[split_name] = _compute_metrics(
            y_true,
            y_pred,
            labels,
            lats,
            load_time_ms=load_ms,
            mem_before_mb=mem0,
            mem_after_mb=mem1,
            fallback_count=fallback_count,
        )
    return results


async def _eval_wasla_hybrid(datasets: dict[str, list[dict[str, Any]]], labels: list[str]) -> dict[str, Any]:
    cfg = WaslaConfig(gemini_api_key="disabled")
    idx = MenuIndex(ROOT / "menu" / "menu.json")
    mem0 = _rss_mb()
    t_load0 = time.perf_counter()
    clf = FineTunedIntentClassifier(checkpoint_dir=DEFAULT_CHECKPOINT_DIR)
    clf.load()
    load_ms = round((time.perf_counter() - t_load0) * 1000.0, 2)
    mem1 = _rss_mb()

    results: dict[str, Any] = {}
    for split_name, rows in datasets.items():
        y_true = [r["intent"] for r in rows]
        y_pred = []
        lats = []
        fallback_count = 0
        for i, r in enumerate(rows):
            disp = WaslaAIDispatcher(cfg, idx, MockBackendAdapter(idx), finetuned_intent=clf)
            t0 = time.perf_counter()
            resp = await disp.handle_text(f"01066{i:06d}", r["text"])
            lats.append((time.perf_counter() - t0) * 1000.0)
            y_pred.append(resp.intent.value)
            if "offline" in resp.model_used:
                fallback_count += 1
        results[split_name] = _compute_metrics(
            y_true,
            y_pred,
            labels,
            lats,
            load_time_ms=load_ms,
            mem_before_mb=mem0,
            mem_after_mb=mem1,
            fallback_count=fallback_count,
        )

    typo_cases = json.loads(TYPO_EVAL_PATH.read_text(encoding="utf-8"))
    intent_ok = 0
    item_total = 0
    item_exact = 0
    incorrect_extractions = 0
    nonexistent_total = 0
    nonexistent_fp = 0
    typo_lats = []
    for i, c in enumerate(typo_cases):
        disp = WaslaAIDispatcher(cfg, idx, MockBackendAdapter(idx), finetuned_intent=clf)
        t0 = time.perf_counter()
        resp = await disp.handle_text(f"01088{i:06d}", c["utterance"])
        typo_lats.append((time.perf_counter() - t0) * 1000.0)
        if resp.intent.value == c["expected_intent"]:
            intent_ok += 1
        pred_ids = {x.product_id for x in resp.extracted_items if x.product_id}
        exp_ids = set(c["expected_product_ids"])
        if exp_ids:
            item_total += 1
            if pred_ids == exp_ids:
                item_exact += 1
            else:
                incorrect_extractions += 1
        if c["category"] == "M_nonexistent_items":
            nonexistent_total += 1
            if pred_ids:
                nonexistent_fp += 1

    results["moka_typo_154"] = {
        "samples": len(typo_cases),
        "intent_accuracy": round(intent_ok / len(typo_cases), 4),
        "exact_product_extraction_acc": round(item_exact / max(1, item_total), 4),
        "incorrect_product_extraction_rate_pct": round((incorrect_extractions / max(1, item_total)) * 100.0, 2),
        "nonexistent_product_fp_rate_pct": round((nonexistent_fp / max(1, nonexistent_total)) * 100.0, 2),
        "latency_p50_ms": round(float(statistics.median(typo_lats)), 2),
        "latency_p95_ms": round(float(sorted(typo_lats)[int(len(typo_lats) * 0.95) - 1]), 2),
    }
    return results


async def _probe_and_eval_gemini(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    cfg = WaslaConfig()
    client = GeminiLLMClient(cfg)
    if not client.is_configured:
        return {
            "status": "SKIPPED_NO_API_KEY",
            "reason": "GEMINI_API_KEY is not configured in environment.",
        }

    # Stratified balanced 24-sample subset (2 per intent x 12 intents)
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    sample_rows = []
    for lbl in labels:
        sample_rows.extend(by_intent.get(lbl, [])[:2])

    idx = MenuIndex(ROOT / "menu" / "menu.json")
    clf = FineTunedIntentClassifier(checkpoint_dir=DEFAULT_CHECKPOINT_DIR)
    clf.load()

    y_true: list[str] = []
    y_pred_gemini: list[str] = []
    lats_gemini: list[float] = []
    fallback_count_gemini = 0
    errors: list[str] = []

    allowed = ", ".join(labels)
    sys_prompt = (
        f"Classify this Egyptian restaurant customer message into ONE intent ({allowed}). "
        'Return JSON: {"intent": "<label>", "confidence": 0.95, "items": [], "requires_clarification": false}.\n'
        + idx.build_compact_prompt_catalog()
    )

    for i, r in enumerate(sample_rows):
        t0 = time.perf_counter()
        pred_lbl = "unclear_request"
        used_fb = False
        try:
            data, _ = await asyncio.wait_for(client.generate_json(sys_prompt, r["text"], use_fallback_model=False), timeout=10.0)
            pred_lbl = str(data.get("intent", "unclear_request")).strip().lower()
            if pred_lbl not in labels or float(data.get("confidence", 0)) < 0.72:
                raise ValueError("low confidence or unknown intent")
        except Exception as primary_exc:
            used_fb = True
            try:
                data, _ = await asyncio.wait_for(client.generate_json(sys_prompt, r["text"], use_fallback_model=True), timeout=10.0)
                pred_lbl = str(data.get("intent", "unclear_request")).strip().lower()
                if pred_lbl not in labels:
                    pred_lbl = "unclear_request"
            except Exception as fb_exc:
                errors.append(f"sample_{i}: primary={type(primary_exc).__name__} fallback={type(fb_exc).__name__}")
                pred_lbl = "unclear_request"
        lats_gemini.append((time.perf_counter() - t0) * 1000.0)
        y_true.append(r["intent"])
        y_pred_gemini.append(pred_lbl)
        if used_fb:
            fallback_count_gemini += 1
        if len(errors) >= 6:
            break

    gemini_metrics = _compute_metrics(y_true, y_pred_gemini, labels, lats_gemini, fallback_count=fallback_count_gemini)
    gemini_metrics["status"] = "PARTIAL_OR_RATE_LIMITED" if errors else "LIVE_EVALUATED"
    gemini_metrics["evaluated_subset"] = (
        f"Stratified {len(y_true)} samples from independent_challenge_203_v2.jsonl"
    )
    if errors:
        gemini_metrics["errors"] = errors[:5]
    return gemini_metrics


def main() -> dict[str, Any]:
    test_unseen = _load_jsonl(TEST_UNSEEN_PATH)
    ind_200 = _load_jsonl(IND_200_PATH)
    labels = sorted({r["intent"] for r in ind_200})

    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    train_texts = {normalize(r["text"]) for r in _load_jsonl(ROOT / "training" / "data" / "train.jsonl")}
    val_texts = {normalize(r["text"]) for r in _load_jsonl(ROOT / "training" / "data" / "val.jsonl")}
    test_texts = {normalize(r["text"]) for r in test_unseen}
    ind_texts = {normalize(r["text"]) for r in ind_200}
    if train_texts & ind_texts or val_texts & ind_texts or test_texts & ind_texts:
        raise ValueError("Independent challenge set overlaps train, validation, or test data.")

    independent_key = f"independent_challenge_{len(ind_200)}"
    datasets = {
        "test_unseen_106": test_unseen,
        independent_key: ind_200,
    }

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_hygiene": {
            "train_count": len(train_texts),
            "val_count": len(val_texts),
            "test_unseen_count": len(test_unseen),
            "independent_challenge_count": len(ind_200),
            "overlap_train_test": len(train_texts & test_texts),
            "overlap_val_test": len(val_texts & test_texts),
            "overlap_train_independent": len(train_texts & ind_texts),
            "overlap_val_independent": len(val_texts & ind_texts),
        },
        "models": {},
    }

    report["models"]["1_camelbert_base"] = _eval_camelbert_base(datasets, labels)
    report["models"]["2_camelbert_lora_v1"] = _eval_lora_clf(V1_CHECKPOINT_DIR, datasets, labels)
    report["models"]["3_camelbert_lora_v2"] = _eval_lora_clf(DEFAULT_CHECKPOINT_DIR, datasets, labels)
    report["models"]["4_gemini_direct_probe"] = asyncio.run(_probe_and_eval_gemini(ind_200, labels))
    report["models"]["5_wasla_hybrid_repaired"] = asyncio.run(_eval_wasla_hybrid(datasets, labels))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    for name, block in rep["models"].items():
        independent_key = next(
            (key for key in block if key.startswith("independent_challenge_")),
            None,
        )
        if independent_key is not None:
            m = block[independent_key]
            t = block.get("test_unseen_106", {})
            print(
                f"{name:26s} | Unseen106 Acc={t.get('intent_accuracy', 0):.2%} F1={t.get('macro_f1', 0):.4f} "
                f"| Ind{m['samples']} Acc={m['intent_accuracy']:.2%} Macro-P={m['macro_precision']:.4f} Macro-R={m['macro_recall']:.4f} "
                f"Macro-F1={m['macro_f1']:.4f} Weighted-F1={m['weighted_f1']:.4f} P50={m['latency_p50_ms']}ms P95={m['latency_p95_ms']}ms "
                f"| RSS+={m['memory_delta_mb']}MB"
            )
            if "moka_typo_154" in block:
                tp = block["moka_typo_154"]
                print(
                    f"  -> MOKA Typo 154: IntentAcc={tp['intent_accuracy']:.2%} ExactItemAcc={tp['exact_product_extraction_acc']:.2%} "
                    f"IncorrectExtract={tp['incorrect_product_extraction_rate_pct']}% NonexistentFP={tp['nonexistent_product_fp_rate_pct']}%"
                )
        else:
            print(
                f"{name:26s} | status={block.get('status')} samples={block.get('samples')} "
                f"Acc={block.get('intent_accuracy')} Macro-F1={block.get('macro_f1')} fallback={block.get('fallback_frequency_pct')}%"
            )
