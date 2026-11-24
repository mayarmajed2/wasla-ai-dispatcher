# Wasla AI — Model & Inference Scientific Evaluation Report (Phase B)

**Report Timestamp:** `2026-11-24T19:21:18Z` (`training/reports/phase_b_scientific_evaluation.json`)  
**Evaluation Script:** `training/scripts/evaluate_phase_b_scientific.py`  
**No Retraining Performed:** Both preserved PEFT LoRA checkpoints (`v1` and `v2`) were evaluated as-shipped after integrating live `FineTunedIntentClassifier.predict()` into `WaslaAIDispatcher`.

---

## 1. Checkpoint Inspection & Runtime Engineering

| Artifact | Base Model | PEFT Rank (`r`) | LoRA Alpha | Target Modules | Labels (`label_map.json`) | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `training/checkpoints/camelbert-wasla-lora` (**v1**) | `CAMeL-Lab/bert-base-arabic-camelbert-mix` | `8` | `16` | `["value", "query"]` | 12 intents (`0..11`) | Apache-2.0 |
| `training/checkpoints/camelbert-wasla-lora-v2` (**v2**, default) | `CAMeL-Lab/bert-base-arabic-camelbert-mix` | `16` | `32` | `["query", "value", "key"]` | 12 intents (`0..11`) | Apache-2.0 |

### 12-Class Intent Label Space
`add_to_cart` (0), `cancel_order` (1), `complaint` (2), `confirm_order` (3), `delivery_inquiry` (4), `greeting` (5), `menu_inquiry` (6), `modify_order` (7), `price_inquiry` (8), `remove_item` (9), `request_human` (10), `unclear_request` (11).

### Runtime Integration (`ai_service/models/finetuned_intent.py` & `ai_service/dispatcher/dispatcher.py`)
- **Lazy Load + Thread-Safe Cache (`_MODEL_CACHE`):** `FineTunedIntentClassifier.__init__` initializes with `is_loaded = False` so backend startup and `/api/ready` remain sub-second and never misreport a checkpoint as loaded purely because its directory exists on disk. Upon the first `.load()` or `.predict()` call, weights load offline-first (`local_files_only=True`, Hub fallback if needed) and cache by `(checkpoint_dir, base_model, device)`. Subsequent classifier instances reuse the cached weights in **`1.69 ms`**.
- **Hardware Support:** Uses GPU when `torch.cuda.is_available()` and `WASLA_LORA_PREFER_GPU=true`; otherwise runs cleanly on CPU.
- **Confidence-Gated Hybrid Routing:**
  - `greeting`, `menu_inquiry`, `complaint`, `request_human`: routed when LoRA confidence `>= 0.55` and no contradictory menu item extraction exists.
  - `delivery_inquiry`, `price_inquiry`, `unclear_request`: routed when LoRA confidence `>= 0.72` (avoiding false positives on cart/checkout queries such as `"المجموع كام؟"`).
  - `confirm_order`: gated by **both** high confidence (`>= 0.90`) **and** explicit deterministic customer confirmation cues — never triggering an unintended order confirmation from low-confidence neural predictions alone.
  - Low-confidence neural outputs (`< 0.55`) without menu items or explicit commands never create or confirm orders.

---

## 2. Dataset Hygiene & Split Discipline

Validation metrics from the original training log (e.g., v2 best validation `Macro-F1 = 0.8858`) are strictly separated from held-out evaluation sets:

| Dataset Split | Path | Samples | Train Overlap | Val Overlap |
| :--- | :--- | :--- | :--- | :--- |
| Train | `training/data/train.jsonl` | 520 | — | — |
| Validation | `training/data/val.jsonl` | 106 | — | — |
| **Held-Out Unseen Test** | `training/data/test_unseen.jsonl` | **106** | **0** | **0** |
| **Independent Challenge (Balanced 17 × 12)** | `datasets/evaluation/independent_challenge_200.jsonl` | **204** | **1** | **0** |
| **MOKA Egyptian Typo & Nonexistent Suite** | `datasets/evaluation/moka_typo_eval_150.json` | **154** | N/A | N/A |

---

## 3. 5-Way Model Comparison

### A. Held-Out Unseen Test (`test_unseen_106`, 106 samples)

| Model | Intent Acc | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | P50 (ms) | P95 (ms) | Load Time (ms) | Process RSS Delta | Fallback % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CAMeLBERT Base** *(seeded head, un-finetuned)* | 39.62% | 0.0365 | 0.0814 | 0.0504 | 0.2451 | 78.61 | 128.92 | 6310.56 | +547.10 MB | 0.0% |
| **2. CAMeLBERT LoRA v1** *(r=8, alpha=16)* | 79.25% | 0.5785 | 0.5648 | 0.5382 | 0.7606 | 88.45 | 148.02 | 1971.67 | +65.14 MB* | 0.0% |
| **3. CAMeLBERT LoRA v2** *(r=16, alpha=32)* | 82.08% | 0.6094 | 0.6609 | 0.6197 | 0.7978 | 98.03 | 202.38 | 1317.55 | +430.23 MB* | 0.0% |
| **4. Wasla Hybrid (Repaired, Local + LoRA v2)** | **83.02%** | **0.9694** | **0.6630** | **0.7470** | **0.8127** | 141.51 | 1013.27 | 1.69 *(cached)* | +0.00 MB *(cached)* | 0.0% |

*\*In-process incremental RSS after base CAMeLBERT was already loaded (`353.88 MB -> 900.98 MB -> 945.07 MB -> 1379.85 MB`).*

### B. Independent Balanced Challenge (`independent_challenge_204`, 204 samples, 17 per intent) + Bounded Live Gemini Probe

| Model | Intent Acc | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | P50 (ms) | P95 (ms) | Fallback % | Scope Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. CAMeLBERT Base** | 8.33% | 0.0149 | 0.0833 | 0.0202 | 0.0202 | 84.14 | 180.43 | 0.0% | Collapses mainly to `add_to_cart` |
| **2. CAMeLBERT LoRA v1** | 30.88% | 0.3309 | 0.3088 | 0.2703 | 0.2703 | 88.97 | 218.46 | 0.0% | Weak OOD generalization |
| **3. CAMeLBERT LoRA v2** | **60.29%** | 0.6046 | **0.6029** | **0.5890** | **0.5890** | 92.40 | 199.87 | 0.0% | Best pure local dialect generalization |
| **4. Gemini Direct Probe** *(stratified 24-sample)* | **83.33%** | **0.8611** | **0.8333** | **0.8194** | **0.8194** | 1893.15 | 8857.08 | **12.5%** | 3/24 fallback events (timeout/HTTP/parse) |
| **5. Wasla Hybrid (Repaired, Offline Config)** | 48.53% | 0.8414 | 0.4853 | 0.5016 | 0.5016 | 732.44 | 1500.73 | 0.0% | High precision (`0.8414`); conservative confidence gating |

### C. Grounded MOKA Egyptian Typo & Safety Benchmark (`moka_typo_154`, Wasla Hybrid)

| Metric | Measured Score |
| :--- | :---: |
| **Samples** | 154 |
| **Intent Accuracy** | **100.00%** (`1.0000`) |
| **Exact Product Extraction Accuracy** | **97.52%** (`0.9752`) |
| **Incorrect Product Extraction Rate** | **2.48%** |
| **Nonexistent Product False Positive Rate** | **0.00%** (`0.0%`) |
| **Latency P50 / P95** | **99.03 ms** / **681.99 ms** |

---

## 4. Per-Intent Performance Breakdown (`independent_challenge_204`)

| Intent (17 support each) | LoRA v1 F1 | LoRA v2 F1 | Wasla Hybrid (Offline) P / R / F1 | Gemini Probe (n=2/cls) F1 |
| :--- | :---: | :---: | :---: | :---: |
| `add_to_cart` | 0.7368 | **0.6897** | 0.1524 / 0.9412 / 0.2623 | 0.0000 |
| `cancel_order` | 0.2727 | 0.5806 | 1.0000 / 0.5882 / **0.7407** | 1.0000 |
| `complaint` | 0.4000 | **0.7273** | 0.6842 / 0.7647 / 0.7222 | 1.0000 |
| `confirm_order` | 0.1379 | **0.8108** | 1.0000 / 0.5294 / 0.6923 | 1.0000 |
| `delivery_inquiry` | 0.0870 | 0.6531 | 0.8333 / 0.5882 / **0.6897** | 1.0000 |
| `greeting` | 0.1538 | **0.6061** | 1.0000 / 0.1176 / 0.2105 | 0.6667 |
| `menu_inquiry` | 0.1000 | 0.2500 | 1.0000 / 0.1765 / **0.3000** | 1.0000 |
| `modify_order` | 0.3200 | 0.5789 | 0.8182 / 0.5294 / **0.6429** | 0.6667 |
| `price_inquiry` | 0.3596 | 0.6154 | 0.7333 / 0.6471 / **0.6875** | 1.0000 |
| `remove_item` | 0.4286 | 0.7500 | 0.8750 / 0.8235 / **0.8485** | 1.0000 |
| `request_human` | 0.1600 | **0.6061** | 1.0000 / 0.0588 / 0.1111 | 1.0000 |
| `unclear_request` | 0.0870 | **0.2000** | 1.0000 / 0.0588 / 0.1111 | 0.5000 |

### Per-Intent F1 on `test_unseen_106` (Unbalanced Commercial Distribution, n=106)

| Intent (Support) | Base F1 | LoRA v1 F1 | LoRA v2 F1 | Wasla Hybrid (Offline) F1 |
| :--- | :---: | :---: | :---: | :---: |
| `add_to_cart` (43) | 0.6043 | **0.9885** | 0.9524 | 0.8350 |
| `cancel_order` (4) | 0.0000 | 0.3333 | 0.3333 | **0.6667** |
| `complaint` (4) | 0.0000 | 0.6000 | 0.6667 | **1.0000** |
| `confirm_order` (4) | 0.0000 | 0.6000 | **0.6667** | **0.6667** |
| `delivery_inquiry` (4) | 0.0000 | 0.4000 | **0.8571** | **0.8571** |
| `greeting` (4) | 0.0000 | 0.4000 | 0.5000 | **0.6667** |
| `menu_inquiry` (4) | 0.0000 | 0.0000 | 0.0000 | **0.4000** |
| `modify_order` (10) | 0.0000 | 0.8333 | 0.8333 | **0.9474** |
| `price_inquiry` (12) | 0.0000 | 0.8571 | **0.9600** | 0.9167 |
| `remove_item` (9) | 0.0000 | 0.8750 | **1.0000** | 0.9412 |
| `request_human` (4) | 0.0000 | 0.5714 | **0.6667** | **0.6667** |
| `unclear_request` (4) | 0.0000 | 0.0000 | 0.0000 | **0.4000** |

---

## 5. Confusion Matrices (`12 × 12`, Rows = True, Columns = Predicted)

Label order (`0..11`): `add_to_cart`, `cancel_order`, `complaint`, `confirm_order`, `delivery_inquiry`, `greeting`, `menu_inquiry`, `modify_order`, `price_inquiry`, `remove_item`, `request_human`, `unclear_request`.

### CAMeLBERT LoRA v2 — `test_unseen_106`
```text
[[40,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0],
 [ 0,  1,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  1,  0],
 [ 0,  0,  0,  0,  0,  2,  0,  1,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1],
 [ 0,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0],
 [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0],
 [ 0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  0,  0]]
```

### CAMeLBERT LoRA v2 — `independent_challenge_204`
```text
[[10,  0,  0,  0,  0,  2,  0,  5,  0,  0,  0,  0],
 [ 0,  9,  1,  3,  0,  0,  0,  0,  0,  2,  1,  1],
 [ 2,  0, 12,  0,  0,  0,  0,  2,  0,  0,  0,  1],
 [ 0,  1,  0, 15,  0,  0,  0,  0,  0,  0,  0,  1],
 [ 0,  0,  0,  0, 16,  0,  0,  0,  0,  0,  0,  1],
 [ 0,  0,  0,  0,  2, 10,  0,  0,  0,  1,  2,  2],
 [ 0,  0,  0,  0,  6,  0,  3,  0,  7,  0,  1,  0],
 [ 0,  0,  3,  0,  0,  0,  0, 11,  2,  0,  0,  1],
 [ 0,  0,  0,  0,  5,  0,  0,  0, 12,  0,  0,  0],
 [ 0,  2,  0,  0,  0,  0,  0,  2,  0, 12,  0,  1],
 [ 0,  1,  0,  0,  2,  0,  2,  0,  0,  0, 10,  2],
 [ 0,  1,  0,  2,  1,  4,  2,  1,  1,  0,  2,  3]]
```

### Wasla Hybrid (Repaired, Offline Config) — `test_unseen_106`
```text
[[43,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 2,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 1,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0],
 [ 2,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0],
 [ 2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0,  0],
 [ 1,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0,  0],
 [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0],
 [ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0],
 [ 3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]]
```

### Wasla Hybrid (Repaired, Offline Config) — `independent_challenge_204`
```text
[[16,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
 [ 5, 10,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0],
 [ 2,  0, 13,  0,  0,  0,  0,  1,  1,  0,  0,  0],
 [ 5,  0,  1,  9,  2,  0,  0,  0,  0,  0,  0,  0],
 [ 6,  0,  1,  0, 10,  0,  0,  0,  0,  0,  0,  0],
 [15,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0],
 [10,  0,  1,  0,  0,  0,  3,  0,  3,  0,  0,  0],
 [ 7,  0,  0,  0,  0,  0,  0,  9,  0,  1,  0,  0],
 [ 6,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0,  0],
 [ 3,  0,  0,  0,  0,  0,  0,  0,  0, 14,  0,  0],
 [15,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0],
 [15,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1]]
```

---

## 6. Scientific Conclusions & Interpretation
1. **Fine-Tuning Effectiveness:** Fine-tuning `CAMeL-Lab/bert-base-arabic-camelbert-mix` with LoRA v2 raises held-out `test_unseen_106` accuracy from **39.62%** (Macro-F1 **0.0504**) to **82.08%** (Macro-F1 **0.6197**), and independent balanced challenge accuracy from **8.33%** (Macro-F1 **0.0202**) to **60.29%** (Macro-F1 **0.5890**), strictly beating v1 (`79.25%` / `30.88%`).
2. **Hybrid Precision vs. Standalone Classifier:** Pairing LoRA v2 with deterministic rules and strict confidence gates in `WaslaAIDispatcher` reaches **83.02%** accuracy / **0.9694** Macro Precision / **0.7470** Macro-F1 on `test_unseen_106` and **100.00%** Intent Accuracy / **97.52%** Exact Product Extraction / **0.00%** Nonexistent FP Rate on `moka_typo_154`.
3. **Role of Live Gemini Function Calling:** On open-ended OOD Egyptian phrasings (`independent_challenge_204`), the stratified 24-sample Gemini probe achieved **83.33%** intent accuracy / **0.8194** Macro-F1 while hitting `12.5%` transient/fallback events — demonstrating why the production topology runs **Gemini primary -> Gemini fallback -> CAMeLBERT LoRA v2 + Deterministic Local Hybrid** rather than trusting any single model blindly.
