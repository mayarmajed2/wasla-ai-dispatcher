"""Real Egyptian Arabic Audio Evaluation (WER & CER) on Hugging Face FLEURS (ar-eg).

Loads real Egyptian Arabic test audio clips from `MohamedRashad/fleurs-ar-eg` (test split: 428 rows)
via the official Hugging Face Datasets Server API, sends each audio clip through Wasla's
HybridAudioPipeline (Option A Gemini Direct / Option B Groq Whisper / Option C Hybrid),
and computes mathematically exact WER & CER on normalized Egyptian Arabic transcripts.

Requires GEMINI_API_KEY and/or GROQ_API_KEY in local .env. Never fabricates WER/CER metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_service.audio.pipeline import HybridAudioPipeline  # noqa: E402
from ai_service.config import WaslaConfig  # noqa: E402
from ai_service.schemas.models import AudioInput  # noqa: E402
from ai_service.understanding.arabic_nlp import normalize_arabic  # noqa: E402


def _edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_arabic(reference).split()
    hyp_words = normalize_arabic(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return round(_edit_distance(ref_words, hyp_words) / len(ref_words), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_arabic(reference).replace(" ", ""))
    hyp_chars = list(normalize_arabic(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return round(_edit_distance(ref_chars, hyp_chars) / len(ref_chars), 4)


async def run_evaluation(dataset: str, split: str, limit: int, audio_mode: str) -> dict:
    cfg = WaslaConfig(ai_audio_mode=audio_mode)
    if not cfg.gemini_api_key and not cfg.groq_api_key and not cfg.openai_api_key:
        raise RuntimeError(
            "BLOCKED: No live audio API credentials configured. Set GEMINI_API_KEY and/or GROQ_API_KEY in .env."
        )

    pipeline = HybridAudioPipeline(cfg)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": dataset, "config": "default", "split": split, "offset": 0, "length": limit},
        )
        resp.raise_for_status()
        rows = resp.json().get("rows", [])

        results = []
        for entry in rows:
            row = entry.get("row", {})
            ref = (row.get("transcription") or row.get("sentence") or "").strip()
            audio_list = row.get("audio") or []
            if not ref or not audio_list:
                continue
            audio_src = audio_list[0].get("src")
            if not audio_src:
                continue

            audio_resp = await client.get(audio_src)
            audio_resp.raise_for_status()
            mime = audio_resp.headers.get("content-type", "audio/wav").split(";")[0].strip().lower()
            if not mime.startswith("audio/"):
                mime = "audio/wav"
            out = None
            last_err = None
            for attempt in range(3):
                try:
                    out = await pipeline.process(AudioInput(audio_bytes=audio_resp.content, mime_type=mime, filename="hf_clip.wav"))
                    break
                except Exception as exc:
                    last_err = exc
                    await asyncio.sleep(1.5 * (attempt + 1))
            if out is None:
                raise last_err
            results.append(
                {
                    "reference": ref,
                    "hypothesis": out.transcript,
                    "wer": compute_wer(ref, out.transcript),
                    "cer": compute_cer(ref, out.transcript),
                    "provider": out.provider,
                    "latency_ms": out.latency_ms,
                }
            )

    avg_wer = round(sum(r["wer"] for r in results) / len(results), 4) if results else None
    avg_cer = round(sum(r["cer"] for r in results) / len(results), 4) if results else None
    return {
        "dataset": dataset,
        "split": split,
        "evaluated_samples": len(results),
        "audio_mode": audio_mode,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "samples": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate real Egyptian Arabic audio WER/CER on Hugging Face.")
    parser.add_argument("--dataset", default="MohamedRashad/fleurs-ar-eg")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", default="hybrid", choices=["direct", "stt", "hybrid"])
    parser.add_argument("--output", default="", help="Optional path to save JSON report.")
    args = parser.parse_args()
    summary = asyncio.run(run_evaluation(args.dataset, args.split, args.limit, args.mode))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
