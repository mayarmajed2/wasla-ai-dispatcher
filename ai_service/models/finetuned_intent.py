"""Fine-Tuned CAMeLBERT-Mix PEFT LoRA Intent Classifier Adapter for Wasla AI.

Loads the saved PEFT LoRA checkpoint from disk (`training/checkpoints/camelbert-wasla-lora-v2`
by default, with v1 at `training/checkpoints/camelbert-wasla-lora`).
Features:
- Lazy initialization (never blocks fast startup; never reports loaded before actual weight init)
- Thread-safe process-wide model/tokenizer cache to avoid reloading across requests
- Automatic GPU detection when available with clean CPU fallback
- Graceful failure capture when dependencies or base model cache are unavailable
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ai_service.schemas.models import CustomerIntent

logger = logging.getLogger("wasla.finetuned_intent")

DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "training" / "checkpoints" / "camelbert-wasla-lora-v2"
)
V1_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "training" / "checkpoints" / "camelbert-wasla-lora"
)

_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def _load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    label2id_raw = raw.get("label2id")
    id2label_raw = raw.get("id2label")
    if not isinstance(label2id_raw, dict) or not isinstance(id2label_raw, dict) or not label2id_raw:
        raise ValueError(f"Invalid LoRA label mapping in {path}")

    if any(not isinstance(label, str) or type(index) is not int for label, index in label2id_raw.items()):
        raise ValueError(f"Invalid label2id entries in {path}")
    try:
        id2label = {int(index): label for index, label in id2label_raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid id2label indices in {path}") from exc

    label2id = dict(label2id_raw)
    expected_id2label = {index: label for label, index in label2id.items()}
    if (
        set(label2id.values()) != set(range(len(label2id)))
        or id2label != expected_id2label
        or len(id2label) != len(id2label_raw)
    ):
        raise ValueError(f"label2id and id2label are inconsistent in {path}")
    return label2id, id2label


def _detect_target_device(prefer_gpu: bool = True) -> str:
    if not prefer_gpu:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class FineTunedIntentClassifier:
    """Lazy-loading, cached PEFT LoRA adapter for Egyptian Arabic restaurant intent classification."""

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        base_model_id: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        enabled: bool = True,
        prefer_gpu: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
        self.base_model_id = base_model_id
        self.enabled = enabled
        self.prefer_gpu = prefer_gpu
        self._model = None
        self._tokenizer = None
        self._id2label: dict[int, str] = {}
        self._device: str = "cpu"
        self._load_error: str | None = None
        self._load_time_ms: float | None = None
        self._adapter_config: dict[str, Any] | None = None

    @property
    def checkpoint_exists(self) -> bool:
        return (
            (self.checkpoint_dir / "adapter_config.json").exists()
            and (self.checkpoint_dir / "adapter_model.safetensors").exists()
            and (self.checkpoint_dir / "label_map.json").exists()
        )

    @property
    def model_version(self) -> str:
        name = self.checkpoint_dir.name
        if name.endswith("-v2"):
            return "camelbert-wasla-lora-v2"
        return name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def inference_available(self) -> bool:
        return bool(self.enabled and self.is_loaded and self._load_error is None)

    @property
    def taxonomy_compatible(self) -> bool:
        if not self.checkpoint_exists:
            return False
        try:
            label2id, _ = _load_label_mapping(self.checkpoint_dir / "label_map.json")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return set(label2id) == {intent.value for intent in CustomerIntent}

    @property
    def routing_available(self) -> bool:
        return self.inference_available and self.taxonomy_compatible

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str:
        return self._device

    def _read_adapter_config(self) -> dict[str, Any]:
        if self._adapter_config is not None:
            return self._adapter_config
        cfg_path = self.checkpoint_dir / "adapter_config.json"
        if cfg_path.exists():
            try:
                self._adapter_config = json.loads(cfg_path.read_text(encoding="utf-8"))
                return self._adapter_config
            except Exception:
                pass
        self._adapter_config = {}
        return self._adapter_config

    def _resolve_base_model_id(self) -> str:
        cfg = self._read_adapter_config()
        return str(cfg.get("base_model_name_or_path") or self.base_model_id)

    def load(self) -> bool:
        if not self.enabled:
            self._load_error = "LoRA classifier disabled by configuration"
            return False
        if self.is_loaded:
            return True
        if not self.checkpoint_exists:
            self._load_error = f"Checkpoint not found at {self.checkpoint_dir}"
            return False

        target_device = _detect_target_device(prefer_gpu=self.prefer_gpu)
        base_id = self._resolve_base_model_id()
        cache_key = (str(self.checkpoint_dir.resolve()), base_id, target_device)

        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                self._tokenizer = cached["tokenizer"]
                self._model = cached["model"]
                self._id2label = cached["id2label"]
                self._device = cached["device"]
                self._load_time_ms = cached["load_time_ms"]
                self._load_error = None
                return True

            t0 = time.perf_counter()
            try:
                label2id, id2label = _load_label_mapping(self.checkpoint_dir / "label_map.json")

                import torch
                from peft import PeftModel
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
                try:
                    base = AutoModelForSequenceClassification.from_pretrained(
                        base_id,
                        num_labels=len(label2id),
                        id2label=id2label,
                        label2id=label2id,
                        local_files_only=True,
                    )
                except Exception:
                    base = AutoModelForSequenceClassification.from_pretrained(
                        base_id,
                        num_labels=len(label2id),
                        id2label=id2label,
                        label2id=label2id,
                    )

                model = PeftModel.from_pretrained(base, self.checkpoint_dir)
                model.to(torch.device(target_device))
                model.eval()

                load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                _MODEL_CACHE[cache_key] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "id2label": id2label,
                    "device": target_device,
                    "load_time_ms": load_ms,
                }
                self._tokenizer = tokenizer
                self._model = model
                self._id2label = id2label
                self._device = target_device
                self._load_time_ms = load_ms
                self._load_error = None
                return True
            except Exception as exc:
                self._load_error = str(exc)
                logger.warning("Failed to load fine-tuned LoRA checkpoint (%s): %s", self.checkpoint_dir, exc)
                return False

    def predict(self, text: str) -> tuple[str, float, float] | None:
        """Return (intent_label, confidence_prob, latency_ms) or None if unavailable."""
        if not self.enabled:
            return None
        if not self.is_loaded and not self.load():
            return None
        if not (text or "").strip():
            return None
        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            enc = self._tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(self._device) for k, v in enc.items()}
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            best_idx = int(probs.argmax().item())
            conf = float(probs[best_idx].item())
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return self._id2label[best_idx], round(conf, 4), latency_ms

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checkpoint_exists": self.checkpoint_exists,
            "checkpoint_v1_exists": (V1_CHECKPOINT_DIR / "adapter_config.json").exists(),
            "checkpoint_v2_exists": (DEFAULT_CHECKPOINT_DIR / "adapter_config.json").exists(),
            "checkpoint_path": str(self.checkpoint_dir),
            "model_loaded": self.is_loaded,
            "model_version": self.model_version,
            "base_model_id": self._resolve_base_model_id(),
            "taxonomy_compatible": self.taxonomy_compatible,
            "device": self._device if self.is_loaded else _detect_target_device(prefer_gpu=self.prefer_gpu),
            "inference_available": self.inference_available,
            "routing_available": self.routing_available,
            "load_time_ms": self._load_time_ms,
            "load_error": self._load_error,
        }
