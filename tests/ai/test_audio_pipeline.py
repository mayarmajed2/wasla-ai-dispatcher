from __future__ import annotations

from pathlib import Path

import pytest

from ai_service.audio.pipeline import HybridAudioPipeline
from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.schemas.models import AudioInput, AudioTranscriptionResult
from ai_service.understanding.menu_index import MenuIndex

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_hybrid_falls_back_after_empty_transcript_without_fabricating_confidence(monkeypatch):
    pipeline = HybridAudioPipeline(WaslaConfig(gemini_api_key="test", groq_api_key="test"))
    fallback_calls = 0

    async def empty_primary(_audio):
        return AudioTranscriptionResult(transcript="", provider="gemini:test", latency_ms=1)

    async def usable_fallback(_audio):
        nonlocal fallback_calls
        fallback_calls += 1
        return AudioTranscriptionResult(transcript="عايز بيتزا", provider="groq:test", latency_ms=2)

    monkeypatch.setattr(pipeline, "_transcribe_gemini_direct", empty_primary)
    monkeypatch.setattr(pipeline, "_transcribe_stt", usable_fallback)

    result = await pipeline.process(AudioInput(audio_bytes=b"OggS" + b"x" * 20))

    assert fallback_calls == 1
    assert result.transcript == "عايز بيتزا"
    assert result.confidence is None
    assert result.used_fallback is True


@pytest.mark.parametrize("uncertain_transcript", ["عايز", "اااااا", "بيتزا و"])
@pytest.mark.asyncio
async def test_hybrid_falls_back_for_incomplete_or_garbled_unscored_transcript(
    monkeypatch, uncertain_transcript
):
    pipeline = HybridAudioPipeline(WaslaConfig(gemini_api_key="test", groq_api_key="test"))
    fallback_calls = 0

    async def uncertain_primary(_audio):
        return AudioTranscriptionResult(
            transcript=uncertain_transcript,
            provider="gemini:test",
            latency_ms=1,
        )

    async def usable_fallback(_audio):
        nonlocal fallback_calls
        fallback_calls += 1
        return AudioTranscriptionResult(
            transcript="عايز بيتزا",
            provider="groq:test",
            latency_ms=2,
        )

    monkeypatch.setattr(pipeline, "_transcribe_gemini_direct", uncertain_primary)
    monkeypatch.setattr(pipeline, "_transcribe_stt", usable_fallback)

    result = await pipeline.process(AudioInput(audio_bytes=b"OggS" + b"x" * 20))

    assert fallback_calls == 1
    assert result.transcript == "عايز بيتزا"
    assert result.confidence is None
    assert result.used_fallback is True
    assert result.is_usable is True


@pytest.mark.parametrize("uncertain_transcript", ["عايز", "اااااا", "بيتزا و"])
@pytest.mark.asyncio
async def test_unscored_uncertain_transcript_is_rejected_without_fallback(
    monkeypatch, uncertain_transcript
):
    pipeline = HybridAudioPipeline(WaslaConfig(gemini_api_key="test"))

    async def uncertain_primary(_audio):
        return AudioTranscriptionResult(
            transcript=uncertain_transcript,
            provider="gemini:test",
            latency_ms=1,
        )

    monkeypatch.setattr(pipeline, "_transcribe_gemini_direct", uncertain_primary)

    result = await pipeline.process(AudioInput(audio_bytes=b"OggS" + b"x" * 20))

    assert result.transcript == uncertain_transcript
    assert result.confidence is None
    assert result.is_usable is False


@pytest.mark.asyncio
async def test_complete_unscored_transcript_passes_structural_gate(monkeypatch):
    pipeline = HybridAudioPipeline(WaslaConfig(gemini_api_key="test"))

    async def usable_primary(_audio):
        return AudioTranscriptionResult(
            transcript="عايز بيتزا",
            provider="gemini:test",
            latency_ms=1,
        )

    monkeypatch.setattr(pipeline, "_transcribe_gemini_direct", usable_primary)

    result = await pipeline.process(AudioInput(audio_bytes=b"OggS" + b"x" * 20))

    assert result.confidence is None
    assert result.is_usable is True


@pytest.mark.asyncio
async def test_uncertain_stt_fallback_is_not_marked_usable(monkeypatch):
    pipeline = HybridAudioPipeline(WaslaConfig(gemini_api_key="test", groq_api_key="test"))

    async def uncertain_primary(_audio):
        return AudioTranscriptionResult(
            transcript="عايز",
            provider="gemini:test",
            latency_ms=1,
        )

    async def uncertain_fallback(_audio):
        return AudioTranscriptionResult(
            transcript="بيتزا و",
            provider="groq:test",
            latency_ms=2,
        )

    monkeypatch.setattr(pipeline, "_transcribe_gemini_direct", uncertain_primary)
    monkeypatch.setattr(pipeline, "_transcribe_stt", uncertain_fallback)

    result = await pipeline.process(AudioInput(audio_bytes=b"OggS" + b"x" * 20))

    assert result.used_fallback is True
    assert result.is_usable is False


@pytest.mark.asyncio
async def test_provider_transcript_text_is_not_mislabeled_as_confidence(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "عايز بيتزا"}]}}]}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("ai_service.audio.pipeline.httpx.AsyncClient", lambda **_kwargs: Client())
    pipeline = HybridAudioPipeline(WaslaConfig(gemini_api_key="test"))

    result = await pipeline._transcribe_gemini_direct(
        AudioInput(audio_bytes=b"OggS" + b"x" * 20)
    )

    assert result.transcript == "عايز بيتزا"
    assert result.confidence is None


@pytest.mark.asyncio
async def test_empty_voice_result_is_a_safe_repeat_not_an_order_mutation():
    class EmptyTranscriptPipeline:
        async def process(self, _audio):
            return AudioTranscriptionResult(
                transcript="", provider="test", latency_ms=1, is_usable=False
            )

    menu = MenuIndex(ROOT / "menu" / "menu.json")
    backend = MockBackendAdapter(menu)
    dispatcher = WaslaAIDispatcher(
        WaslaConfig(gemini_api_key="", lora_enabled=False),
        menu,
        backend,
        audio_pipeline=EmptyTranscriptPipeline(),
    )

    response = await dispatcher.handle_audio(
        "01070000017",
        AudioInput(audio_bytes=b"OggS" + b"x" * 20),
    )

    assert response.intent.value == "unclear_request"
    assert response.requires_clarification is True
    assert response.transcript == ""
    assert response.clarification_reason == "unclear_audio"
    assert response.cart.items == []
    assert backend._orders == {}


@pytest.mark.asyncio
async def test_rejected_nonempty_voice_cannot_add_to_cart_or_confirm_order():
    class UncertainTranscriptPipeline:
        async def process(self, _audio):
            return AudioTranscriptionResult(
                transcript="عايز بيتزا",
                provider="test",
                latency_ms=1,
                is_usable=False,
            )

    menu = MenuIndex(ROOT / "menu" / "menu.json")
    backend = MockBackendAdapter(menu)
    dispatcher = WaslaAIDispatcher(
        WaslaConfig(gemini_api_key="", lora_enabled=False),
        menu,
        backend,
        audio_pipeline=UncertainTranscriptPipeline(),
    )

    response = await dispatcher.handle_audio(
        "01070000018",
        AudioInput(audio_bytes=b"OggS" + b"x" * 20),
    )

    assert response.intent.value == "unclear_request"
    assert response.requires_clarification is True
    assert response.clarification_reason == "unclear_audio"
    assert response.transcript == ""
    assert response.cart.items == []
    assert backend._orders == {}
