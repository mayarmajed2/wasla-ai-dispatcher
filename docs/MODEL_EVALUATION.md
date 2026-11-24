# Wasla AI — Independent Model Evaluation & Audio Benchmark Report

> **Validity notice (2026-11-25):** The historical benchmark included one exact overlap between training data and its 204-row independent challenge set. Those historical files and scores remain unchanged and do not describe the separate 203-row v2 set. The corrected v2 evaluation and leakage audit are recorded in `training/reports/evaluation_5way_independent_v2.json` and `training/reports/leakage_audit_independent_challenge_v2.json`; use those reports for the limited v2 benchmark results.

---

## 1. Base Model & Verified License Audit

- **Base Model:** `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- **Verified Official License (Hugging Face API):** **`Apache-2.0`** *(corrected from an earlier erroneous "MIT" mention; commercial use permitted with attribution and Apache-2.0 notice)*.
- **Audio Benchmark Dataset:** `MohamedRashad/fleurs-ar-eg` (`test` split) — Hugging Face metadata tag `license:cc` (Creative Commons).

---

## 2. Dataset Provenance & Template Leakage Audit

All intent training/evaluation text examples are **manually authored / synthetic Egyptian Arabic domain data** built around MOKA's real 145-item menu (0 real customer WhatsApp logs were available).

Historical leakage audit (`training/reports/leakage_audit.json`, preserved as originally recorded):

| Pair Compared | Exact Overlap | High Template Similarity (`cosine >= 0.70`) | Mean Max Cosine Sim | Interpretation |
|---|---:|---:|---:|---|
| `train_v1` (520) vs `test_unseen_v1` (106) | `0` | `26 / 106` (24.5%) | `0.5985` | Confirms structural template similarity in v1 split |
| `train_v1` (520) vs **`independent_challenge_200` (204)** | `1` | `11 / 204` (5.4%) | **`0.4183`** | **Genuinely harder, independently authored benchmark** |
| `train_v1` (520) vs `moka_typo_eval_150` (154) | `0` | — | `0.4693` | Clean separation |

---

## 3. CAMeLBERT-Mix PEFT LoRA v1 → v2 Optimization

To fix v1's low Macro-F1 (`0.5382` on `test_unseen_v1`, `0.2703` on independent challenge), **v2** (`training/scripts/train_lora_v2.py`) introduced:
1. **Class-weighted `CrossEntropyLoss`** (inverse class frequency balancing).
2. **33 curated Egyptian Arabic hard negatives** across rare/confusable intents (`complaint`, `request_human`, `modify_order`, `remove_item`, `cancel_order`, `unclear_request`).
3. **Expanded LoRA capacity:** `r=16`, `lora_alpha=32`, `target_modules=["query", "key", "value"]` (`893,964` trainable params / `109,984,536` total = `0.81%`).
4. **Best-Validation-Macro-F1 checkpoint selection** across 6 epochs (`1143.57s` on 12-core CPU).

### Real v2 Training Curve (`training/reports/training_v2_log.json`)

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 | Saved? |
|---:|---:|---:|---:|---:|---|
| 1 | `2.4261` | `2.1580` | `24.53%` | `0.1520` | — |
| 2 | `1.8874` | `1.3026` | `76.42%` | `0.5872` | — |
| **3** | **`1.1579`** | **`0.7036`** | **`94.34%`** | **`0.8858`** | **Best Checkpoint Saved (`camelbert-wasla-lora-v2`)** |
| 4 | `0.7506` | `0.4985` | `94.34%` | `0.8858` | — |
| 5 | `0.4640` | `0.3767` | `93.40%` | `0.8679` | — |
| 6 | `0.3689` | `0.3629` | `94.34%` | `0.8771` | — |

---

## 4. Independent 5-Way Scientific Comparison (`independent_challenge_200.jsonl`, n = 204)

The historical five-model report below evaluates the original **204-example balanced independent Egyptian Arabic benchmark** (17 examples × 12 intents) and is retained unchanged at `training/reports/evaluation_5way_independent.json`. It is not a result for the separate 203-row v2 set; that evaluation is stored at `training/reports/evaluation_5way_independent_v2.json`.

| Model | Intent Accuracy | Macro-F1 | Weighted-F1 | Avg Latency (CPU) | P95 Latency |
|---|---:|---:|---:|---:|---:|
| **Model A: Supervised TF-IDF (Char+Word) + Logistic Regression** | **`81.37%`** | **`0.8100`** | **`0.8100`** | **`1.62 ms`** | `2.35 ms` |
| **Model B: CAMeLBERT-Mix LoRA v1** *(baseline fine-tune)* | `30.88%` | `0.2703` | `0.2703` | `71.62 ms` | `85.00 ms` |
| **Model C: CAMeLBERT-Mix LoRA v2** *(class-weighted + hard negatives)* | **`60.29%`** *(+29.41% vs v1)* | **`0.5890`** *(+0.3187 vs v1)* | **`0.5890`** | `62.03 ms` | `74.97 ms` |
| **Model D: Wasla Rule / Fuzzy Egyptian NLU** | `38.73%` | `0.4031` | `0.4031` | `308.95 ms` | — |
| **Model E: Naive Hybrid (LoRA v2 + Rules)** | `49.02%` | `0.5036` | `0.5036` | `62.03 ms` | — |

### Honest Scientific Findings & Error Analysis
1. **v2 LoRA more than doubled v1** on the independent benchmark (`0.2703 → 0.5890` Macro-F1; `30.88% → 60.29%` Accuracy) and outperformed the rule engine by `+21.56%` Accuracy.
2. **Why TF-IDF + Logistic Regression (`81.37%` / `0.8100` Macro-F1, `1.62 ms`) beat 110M CAMeLBERT LoRA v2 on 553 training samples:** On small domain datasets (<1,000 samples) with strong Egyptian lexical markers (`كام`, `فين`, `الغي`, `شكوى`, `كاشير`), regularized character+word n-gram linear models generalize with lower variance than 894K transformer adapter weights. Collecting 2,000+ real anonymized MOKA WhatsApp utterances after pilot launch will allow CAMeLBERT LoRA to surpass linear models on paraphrased/unseen expressions.
3. **Per-Class Bottlenecks in LoRA v2 (12×12 Confusion Matrix):**
   - Strongest intents: `confirm_order` (`F1=0.8108`), `remove_item` (`F1=0.7500`), `complaint` (`F1=0.7273`), `add_to_cart` (`F1=0.6897`).
   - Weakest intents: `unclear_request` (`F1=0.2000`) and `menu_inquiry` (`F1=0.2500`), frequently confused with `delivery_inquiry` and `price_inquiry`.
4. **Why Naive Hybrid (`49.02%`) trailed pure LoRA v2 (`60.29%`):** Giving rule overrides priority on open-ended sentences dragged down intent accuracy when rules defaulted to `add_to_cart` or `clarify`. In production, **Wasla Rule/Fuzzy NLU** is best used for **exact 145-item menu entity extraction, typo tolerance, and deterministic safety guards**, while **Gemini / LoRA v2 / TF-IDF** handle intent classification.

---

## 5. Egyptian Arabic Typo-Tolerance & Entity Recognition (`moka_typo_eval_150.json`, n = 154)

- **Intent Accuracy:** `100.00%` (`154/154`)
- **Entity Precision:** `100.00%`
- **Entity Recall:** `97.71%`
- **Entity F1:** `98.84%`
- **Nonexistent Item False-Positive Rate:** **`0.00%`**
- **Ambiguous Item Clarification Accuracy:** **`100.00%`**

---

## 6. Real Egyptian Arabic Audio Benchmark (Groq Whisper `whisper-large-v3-turbo`)

Executed live via `scripts/eval_egyptian_audio_wer.py` against Hugging Face **`MohamedRashad/fleurs-ar-eg` (`test` split)** (`training/reports/groq_whisper_20_fleurs.json`):

| Run | Samples | Provider & Model | Measured Avg WER | Measured Avg CER | Avg Latency |
|---|---:|---|---:|---:|---:|
| Initial Pilot | `5` | `groq:whisper-large-v3-turbo` | `8.78%` | `2.57%` | `~1,260 ms` |
| **Expanded Benchmark** | **`20`** | **`groq:whisper-large-v3-turbo`** | **`10.11%`** | **`2.76%`** | **`~2,300 ms`** |

- Exact `0.0%` WER achieved on 3/20 clips; worst outlier (`41.18%` WER) involved foreign cricket proper nouns (*Sachin Tendulkar / Rahul Dravid*) rather than Egyptian Arabic vocabulary.

---

## 7. Live LLM & WhatsApp Provider Verification

- **Google Gemini (`gemini-3.5-flash-lite`):** Live `HTTP 200 OK`, `1341.1 ms`, `66 tokens` (`31` prompt / `35` output), valid structured JSON extraction on misspelled Egyptian Arabic.
- **Meta WhatsApp Cloud API (`v27.0`):** Live `HTTP 200 OK` on test phone ID `1515063625018594` (`+1 555-183-2445`, status `GREEN`, `VERIFIED`). End-to-end live webhook delivery remains blocked until hosted on a public HTTPS endpoint with `META_APP_SECRET`.
