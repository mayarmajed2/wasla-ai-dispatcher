"""Tests verifying real PEFT LoRA checkpoint loading, inference, and safe fallback."""

from pathlib import Path

import pytest

from ai_service.models.finetuned_intent import (
    DEFAULT_CHECKPOINT_DIR,
    V1_CHECKPOINT_DIR,
    FineTunedIntentClassifier,
    _load_label_mapping,
)


def test_finetuned_lora_safe_fallback_when_checkpoint_missing(tmp_path):
    clf = FineTunedIntentClassifier(checkpoint_dir=tmp_path / "nonexistent_ckpt")
    assert clf.checkpoint_exists is False
    assert clf.load() is False
    assert clf.is_loaded is False
    assert clf.predict("عايز بيتزا مارجريتا كبير") is None


def test_real_checkpoint_label_maps_are_bijective_and_match_adapter_class_count():
    for checkpoint_dir in (DEFAULT_CHECKPOINT_DIR, V1_CHECKPOINT_DIR):
        label2id, id2label = _load_label_mapping(checkpoint_dir / "label_map.json")
        assert len(label2id) == 12
        assert all(id2label[index] == label for label, index in label2id.items())
        assert (checkpoint_dir / "adapter_model.safetensors").is_file()
        assert (checkpoint_dir / "adapter_config.json").is_file()


def test_real_finetuned_lora_checkpoint_loads_and_predicts():
    clf = FineTunedIntentClassifier()
    assert clf.checkpoint_exists is True, "Real trained LoRA checkpoint must exist at training/checkpoints/camelbert-wasla-lora"
    if not clf.load():
        if clf.load_error and "Application Control policy has blocked" in clf.load_error:
            pytest.skip("Host application-control policy blocks the native model dependency.")
        pytest.fail(f"Real LoRA checkpoint failed to load: {clf.load_error}")
    assert clf.is_loaded is True

    intent, conf, latency_ms = clf.predict("عاوز بيتزا مارجريتا كبير لو سمحت")
    assert intent == "add_to_cart"
    assert conf >= 0.50
    assert latency_ms > 0.0

    intent2, conf2, _ = clf.predict("بكام كريب كريسبي؟")
    assert intent2 == "price_inquiry"
    assert conf2 >= 0.40
