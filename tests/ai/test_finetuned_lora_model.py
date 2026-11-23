"""Tests verifying real PEFT LoRA checkpoint loading, inference, and safe fallback."""

from pathlib import Path

from ai_service.models.finetuned_intent import FineTunedIntentClassifier


def test_finetuned_lora_safe_fallback_when_checkpoint_missing(tmp_path):
    clf = FineTunedIntentClassifier(checkpoint_dir=tmp_path / "nonexistent_ckpt")
    assert clf.checkpoint_exists is False
    assert clf.load() is False
    assert clf.is_loaded is False
    assert clf.predict("عايز بيتزا مارجريتا كبير") is None


def test_real_finetuned_lora_checkpoint_loads_and_predicts():
    clf = FineTunedIntentClassifier()
    assert clf.checkpoint_exists is True, "Real trained LoRA checkpoint must exist at training/checkpoints/camelbert-wasla-lora"
    assert clf.load() is True
    assert clf.is_loaded is True

    intent, conf, latency_ms = clf.predict("عاوز بيتزا مارجريتا كبير لو سمحت")
    assert intent == "add_to_cart"
    assert conf >= 0.50
    assert latency_ms > 0.0

    intent2, conf2, _ = clf.predict("بكام كريب كريسبي؟")
    assert intent2 == "price_inquiry"
    assert conf2 >= 0.40
