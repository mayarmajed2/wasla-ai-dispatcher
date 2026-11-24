"""Template-Level & Semantic Similarity Leakage Audit across all Wasla AI datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT / "training" / "data" / "train.jsonl"
TEST_V1_PATH = ROOT / "training" / "data" / "test_unseen.jsonl"
IND_200_PATH = ROOT / "datasets" / "evaluation" / "independent_challenge_203_v2.jsonl"
EVAL_150_PATH = ROOT / "datasets" / "evaluation" / "moka_typo_eval_150.json"
OUT_REPORT = ROOT / "training" / "reports" / "leakage_audit_independent_challenge_v2.json"


def _load_jsonl(path: Path) -> list[str]:
    return [json.loads(l)["text"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pair_leakage(train_texts: list[str], eval_texts: list[str], name: str) -> dict:
    train_set = {" ".join(t.lower().split()) for t in train_texts}
    eval_set = {" ".join(t.lower().split()) for t in eval_texts}
    exact_overlap = len(train_set & eval_set)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    all_mat = vec.fit_transform(train_texts + eval_texts)
    tr_mat = all_mat[: len(train_texts)]
    ev_mat = all_mat[len(train_texts) :]
    sims = cosine_similarity(ev_mat, tr_mat)
    max_sims = sims.max(axis=1)

    return {
        "pair": name,
        "eval_size": len(eval_texts),
        "exact_overlap": exact_overlap,
        "near_duplicate_ge_0_85": int((max_sims >= 0.85).sum()),
        "high_template_sim_ge_0_70": int((max_sims >= 0.70).sum()),
        "mean_max_cosine_sim": round(float(np.mean(max_sims)), 4),
        "p95_max_cosine_sim": round(float(np.percentile(max_sims, 95)), 4),
    }


def main() -> dict:
    train_t = _load_jsonl(TRAIN_PATH)
    test_v1 = _load_jsonl(TEST_V1_PATH)
    ind_200 = _load_jsonl(IND_200_PATH)
    eval_150 = [r["utterance"] for r in json.loads(EVAL_150_PATH.read_text(encoding="utf-8"))]

    report = {
        "train_size": len(train_t),
        "audits": [
            pair_leakage(train_t, test_v1, "train_v1 vs test_unseen_v1 (same generator)"),
            pair_leakage(train_t, ind_200, "train_v1 vs independent_challenge_203_v2 (separate author)"),
            pair_leakage(train_t, eval_150, "train_v1 vs moka_typo_eval_150 (separate typo suite)"),
        ],
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
