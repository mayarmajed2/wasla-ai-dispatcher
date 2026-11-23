# Model Card — Wasla AI Egyptian Arabic Intent Classifier (`camelbert-wasla-lora-v2`)

## Model Details
- **Developed by:** Wasla AI Engineering Team (for MOKA Restaurant & Cafe, Egypt)
- **Model Type:** PEFT LoRA Sequence Classification Adapter (12 Egyptian Arabic restaurant intents)
- **Base Model:** [`CAMeL-Lab/bert-base-arabic-camelbert-mix`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix) (109.4M params)
- **Base Model License:** **`Apache-2.0`** (verified via Hugging Face Hub API; commercial use permitted with attribution)
- **Adapter Checkpoint:** `training/checkpoints/camelbert-wasla-lora-v2/` (`r=16`, `lora_alpha=32`, `target_modules=["query", "key", "value"]`, `893,964` trainable parameters)
- **Rollback Checkpoint:** `training/checkpoints/camelbert-wasla-lora/` (v1 preserved intact)

## Intended Use
- **Primary Use:** Local CPU-friendly intent classification of Egyptian Arabic restaurant WhatsApp messages for MOKA (`add_to_cart`, `modify_order`, `remove_item`, `confirm_order`, `cancel_order`, `menu_inquiry`, `price_inquiry`, `delivery_inquiry`, `greeting`, `complaint`, `request_human`, `unclear_request`).
- **Out-of-Scope Use:** Must **never** compute prices, invent menu items, or auto-confirm orders. Pricing is strictly handled by deterministic `Decimal` backend arithmetic (`backend/app/cart_calculator.py`), and menu entity matching is grounded against `menu/menu.json` (145 confirmed items).

## Training Data & Limitations
- Trained on **553 synthetic/manually authored Egyptian Arabic restaurant utterances** (520 base + 33 curated hard negatives) with inverse-frequency class-weighted `CrossEntropyLoss`, selecting the best checkpoint by Validation Macro-F1 (Epoch 3: `val_macro_f1 = 0.8858`, `val_acc = 94.34%`).
- **Known Limitation:** Because training data is small and synthetic (0 real customer WhatsApp logs), on the independent 204-example challenge benchmark (`datasets/evaluation/independent_challenge_200.jsonl`), LoRA v2 achieves **`60.29%` Accuracy / `0.5890` Macro-F1** (`62.03 ms` CPU), whereas a regularized char+word TF-IDF + Logistic Regression baseline reaches **`81.37%` Accuracy / `0.8100` Macro-F1** (`1.62 ms` CPU).
- **Mitigation:** In production, use Gemini (`gemini-3.5-flash-lite`) as primary NLU, with local models + deterministic fuzzy menu matcher as fast fallback, and retrain LoRA on 2,000+ anonymized real MOKA WhatsApp utterances after pilot launch.
