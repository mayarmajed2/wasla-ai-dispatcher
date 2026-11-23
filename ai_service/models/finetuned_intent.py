"""Fine-Tuned CAMeLBERT-Mix PEFT LoRA Intent Classifier Adapter for Wasla AI.

Loads the actual saved PEFT LoRA checkpoint from disk (`training/checkpoints/camelbert-wasla-lora`).
Never claims to be active unless the real checkpoint loads cleanly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("wasla.finetuned_intent")

DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "training" / "checkpoints" / "camelbert-wasla-lora"
)


class FineTunedIntentClassifier:
    """Lazy-loading PEFT LoRA adapter for Egyptian Arabic restaurant intent classification."""

    def __init__(self, checkpoint_dir: Path | None = None, base_model_id: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix"):
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.base_model_id = base_model_id
        self._model = None
        self._tokenizer = None
        self._id2label: dict[int, str] = {}
        self._load_error: str | None = None

    @property
    def checkpoint_exists(self) -> bool:
        return (self.checkpoint_dir / "adapter_config.json").exists() and (self.checkpoint_dir / "label_map.json").exists()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        if self.is_loaded:
            return True
        if not self.checkpoint_exists:
            self._load_error = f"Checkpoint not found at {self.checkpoint_dir}"
            return False
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            label_map = json.loads((self.checkpoint_dir / "label_map.json").read_text(encoding="utf-8"))
            label2id = label_map["label2id"]
            self._id2label = {int(k): v for k, v in label_map["id2label"].items()}

            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
            base = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_id,
                num_labels=len(label2id),
                id2label=self._id2label,
                label2id=label2id,
            )
            self._model = PeftModel.from_pretrained(base, self.checkpoint_dir)
            self._model.eval()
            self._load_error = None
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Failed to load fine-tuned LoRA checkpoint: %s", exc)
            return False

    def predict(self, text: str) -> tuple[str, float, float] | None:
        """Return (intent_label, confidence_prob, latency_ms) or None if unavailable."""
        if not self.is_loaded and not self.load():
            return None
        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            enc = self._tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            best_idx = int(probs.argmax().item())
            conf = float(probs[best_idx].item())
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return self._id2label[best_idx], round(conf, 4), latency_ms
