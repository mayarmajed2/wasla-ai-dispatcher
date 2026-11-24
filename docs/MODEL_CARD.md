# Model Card — Wasla AI Egyptian Arabic Intent Classifier (`camelbert-wasla-lora-v2`)

## Model Details
- **Developed by:** Wasla AI Engineering Team (for MOKA Restaurant & Cafe, Egypt)
- **Model Type:** PEFT LoRA Sequence Classification Adapter (12 Egyptian Arabic restaurant intents)
- **Base Model:** [`CAMeL-Lab/bert-base-arabic-camelbert-mix`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix) (109.4M params)
- **Base Model License:** **`Apache-2.0`** (verified via Hugging Face Hub API; commercial use permitted with attribution)
- **Adapter Checkpoint:** `training/checkpoints/camelbert-wasla-lora-v2/` (`r=16`, `lora_alpha=32`, `target_modules=["query", "key", "value"]`, `893,964` trainable parameters)
- **Rollback Checkpoint:** `training/checkpoints/camelbert-wasla-lora/` (v1 preserved intact)

## Intended Use
- **Potential Use:** Research and evaluation of Egyptian Arabic restaurant intent classification for the 12 labels listed above. The checkpoints are not eligible for application routing because their label set does not match the current `CustomerIntent` taxonomy.
- **Out-of-Scope Use:** Must **never** compute prices, invent menu items, or auto-confirm orders. Pricing is strictly handled by deterministic `Decimal` backend arithmetic (`backend/app/cart_calculator.py`), and menu entity matching is grounded against `menu/menu.json` (145 confirmed items).

## Training Data & Limitations
- Trained on **553 synthetic/manually authored Egyptian Arabic restaurant utterances** (520 base + 33 curated hard negatives) with inverse-frequency class-weighted `CrossEntropyLoss`, selecting the best checkpoint by Validation Macro-F1 (Epoch 3: `val_macro_f1 = 0.8858`, `val_acc = 94.34%`).
- **Known Limitation:** Training data is synthetic/manually authored (0 real customer WhatsApp logs). The former independent benchmark had an exact training-text overlap; that row was removed, leaving 203 examples. Previously reported benchmark scores are invalid for the corrected dataset and require a fresh evaluation.
- **Runtime Status:** The local adapter is reported as inference-available only after genuine model loading, and as routable only when checkpoint labels match the application taxonomy. No compatible checkpoint is currently available.
