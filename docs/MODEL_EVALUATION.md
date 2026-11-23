# Wasla AI — Scientific Model Training & Evaluation Report

**Date:** 2026-11-23  
**Hardware:** 12-Core AMD64 CPU (`PyTorch 2.15.0+cpu`, `transformers 5.24.0`, `peft 0.22.0`)

---

## 1. Real PEFT LoRA Fine-Tuning Execution Summary

- **Base Model:** `CAMeL-Lab/bert-base-arabic-camelbert-mix` (MIT License, 109,394,712 total params)
- **Method:** PEFT LoRA (`r=8`, `lora_alpha=16`, `lora_dropout=0.1`, `target_modules=["query", "value"]`)
- **Trainable Parameters:** `304,140` (`0.278%` of total parameters)
- **Dataset:** `training/data/` (`synthetic_moka_grounded`, derived strictly from verified 145 MOKA items, seed `42`, **0 leakage** across `train` / `val` / `test_unseen` / `moka_typo_eval_150`, **0 PII**)
  - `train.jsonl`: **520** examples
  - `val.jsonl`: **106** examples
  - `test_unseen.jsonl`: **106** genuinely unseen held-out examples
- **Training Duration:** `375.48 seconds` (4 epochs on 12-core CPU)
- **Saved Checkpoint:** `training/checkpoints/camelbert-wasla-lora/` (`adapter_model.safetensors`: 1.22 MB)

### Training & Validation Loss Curve (Actual Executed Run)
| Epoch | Train Loss | Val Loss | Val Accuracy |
|---|---|---|---|
| 0 (Untrained Base) | — | `2.2265` | `39.62%` |
| 1 | `2.0461` | `1.6425` | `49.06%` |
| 2 | `1.4877` | `1.1310` | `58.49%` |
| 3 | `1.1141` | `0.8812` | `73.58%` |
| 4 | `0.9331` | `0.8179` | `73.58%` |

---

## 2. Scientific 3-Way Held-Out Comparison (`training/data/test_unseen.jsonl`, n=106)

| Model | Intent Accuracy | Macro-F1 | Weighted-F1 | Avg CPU Latency | Hallucinated Products |
|---|---|---|---|---|---|
| **Model A: Untrained Base (`CAMeLBERT-Mix`)** | `39.62%` | `0.0504` | `0.2451` | `60.01 ms` | N/A |
| **Model B: Fine-Tuned PEFT LoRA (`camelbert-wasla-lora`)** | **`79.25%`** *(+39.63%)* | `0.5382` | **`0.7606`** | **`60.52 ms`** | N/A |
| **Model C: Wasla Hybrid NLU + Fuzzy MenuIndex** | `77.36%` | **`0.6043`** | `0.7311` | `101.90 ms` | **`0`** |

**Honest Analysis & Known Limitation:**
- Fine-tuning doubled held-out intent accuracy over the untrained base (`39.62%` → `79.25%`) and achieved lower CPU latency (`60.5ms`) than full fuzzy scan (`101.9ms`).
- Because `add_to_cart` (287) and `price_inquiry` (80) had more training rows than minority intents like `delivery_inquiry` (30), the 4-epoch LoRA adapter occasionally confuses delivery price questions (`"التوصيل بكام؟"`) with `price_inquiry`. Class-balanced oversampling + more epochs on a free Colab/Kaggle GPU will raise Macro-F1 further.
- Therefore, in production, the fine-tuned LoRA classifier is combined with deterministic MOKA menu grounding and `Decimal` backend pricing — never replacing financial safety rules.

---

## 3. Live External Provider & Real Egyptian Audio Benchmarks

- **Google Gemini (`gemini-3.5-flash-lite`):** Live `HTTP 200 OK`, `1341.1 ms` latency, `66 tokens` (`31 prompt + 35 output`), valid structured JSON extraction on misspelled Egyptian Arabic.
- **Groq Whisper (`whisper-large-v3-turbo`) on Real Hugging Face `MohamedRashad/fleurs-ar-eg` (`test` split, CC license, n=5):**
  - **Measured AVG WER:** **`8.78%`** | **Measured AVG CER:** **`2.57%`** (~`1.26s` avg latency).
- **Meta WhatsApp Cloud API (`v27.0`):** Live `HTTP 200 OK` on authorized test phone `1515063625018594` (`+1 555-183-2445`, `GREEN`, `VERIFIED`).
