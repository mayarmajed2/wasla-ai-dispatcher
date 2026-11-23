# Wasla AI — Egyptian Arabic Fine-Tuning Pipeline (`training/`)

## Model Selection Rationale (Phase 2)

| Candidate | Params | License | Arabic Dialect Strength | Hardware Needed | Decision |
|---|---|---|---|---|---|
| **`CAMeL-Lab/bert-base-arabic-camelbert-mix`** | 109M | **MIT** (Commercial OK) | Pretrained specifically on **Egyptian/Dialectal + MSA + Classical Arabic** | **CPU trainable & deployable (~15–25ms/query)** | **SELECTED** for real local PEFT LoRA intent fine-tuning |
| **`Qwen/Qwen2.5-3B-Instruct`** | 3B | **Apache-2.0** (Commercial OK) | Strong multilingual + generative JSON tool-calling | Requires 12–16 GB CUDA GPU (T4/L4) | Documented as Cloud GPU QLoRA alternative |

## Reproducible Pipeline Commands

```bash
# 1. Prepare deterministic train/val/test_unseen splits (seed=42, zero leakage)
python training/scripts/prepare_dataset.py

# 2. Validate schema, zero duplicates, zero train/test/eval150 leakage, zero PII
python training/scripts/validate_dataset.py

# 3. Execute real PEFT LoRA fine-tuning (r=8, alpha=16, target_modules=[query, value])
python training/scripts/train_lora_intent.py

# 4. Run 3-way scientific evaluation (Base vs Fine-Tuned LoRA vs Wasla Hybrid)
python training/scripts/evaluate_models.py
```

## Safety Guarantee
The fine-tuned LoRA model is used strictly for **intent understanding** alongside deterministic MOKA menu grounding. It **never** sets prices, invents menu items, bypasses stock checks, or locks orders without customer confirmation.
