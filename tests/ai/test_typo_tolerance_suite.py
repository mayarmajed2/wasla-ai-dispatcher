"""Dedicated 154-Case Egyptian Arabic Typo-Tolerance, Fuzzy Matching & Multi-Turn Context Evaluation Suite.

All utterances in moka_typo_eval_150.json are explicitly labeled `synthetic_evaluation`
(manually authored against MOKA's real 145-item menu — NOT real customer logs).
"""

import asyncio
import json
from decimal import Decimal
from pathlib import Path

from ai_service.adapters.mock_backend import MockBackendAdapter
from ai_service.config import WaslaConfig
from ai_service.dispatcher.dispatcher import WaslaAIDispatcher
from ai_service.understanding.menu_index import MenuIndex

ROOT = Path(__file__).resolve().parent.parent.parent
MENU_PATH = ROOT / "menu" / "menu.json"
TYPO_EVAL_PATH = ROOT / "datasets" / "evaluation" / "moka_typo_eval_150.json"


def _make_dispatcher() -> WaslaAIDispatcher:
    cfg = WaslaConfig(gemini_api_key="")  # Explicit offline deterministic/fuzzy mode
    idx = MenuIndex(MENU_PATH)
    return WaslaAIDispatcher(cfg, idx, MockBackendAdapter(idx))


def test_154_case_egyptian_arabic_typo_and_colloquial_evaluation():
    cases = json.loads(TYPO_EVAL_PATH.read_text(encoding="utf-8"))
    assert len(cases) >= 150

    async def _run():
        intent_correct = 0
        exact_item_correct = 0
        exact_item_total = 0
        tp = fp = fn = 0
        nonexistent_fp = 0
        nonexistent_total = 0
        ambiguous_clar_ok = 0
        ambiguous_total = 0
        failures = []

        for case in cases:
            dispatcher = _make_dispatcher()
            resp = await dispatcher.handle_text(f"01088{case['id'][-3:]}", case["utterance"])

            if resp.intent.value == case["expected_intent"]:
                intent_correct += 1
            else:
                failures.append((case["id"], case["utterance"], f"intent {resp.intent.value} != {case['expected_intent']}"))

            pred_pids = {i.product_id for i in resp.extracted_items if i.product_id}
            exp_pids = set(case["expected_product_ids"])

            tp += len(pred_pids & exp_pids)
            fp += len(pred_pids - exp_pids)
            fn += len(exp_pids - pred_pids)

            if exp_pids:
                exact_item_total += 1
                if pred_pids == exp_pids:
                    exact_item_correct += 1
                else:
                    failures.append((case["id"], case["utterance"], f"items {pred_pids} != {exp_pids}"))

            if case["category"] == "M_nonexistent_items":
                nonexistent_total += 1
                if pred_pids:
                    nonexistent_fp += 1
                    failures.append((case["id"], case["utterance"], f"FALSE POSITIVE matched {pred_pids}"))

            if case["category"] == "L_ambiguous_names":
                ambiguous_total += 1
                if resp.requires_clarification and not pred_pids:
                    ambiguous_clar_ok += 1
                else:
                    failures.append((case["id"], case["utterance"], f"ambiguous did not clarify: {pred_pids}"))

        intent_acc = intent_correct / len(cases)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        exact_acc = exact_item_correct / exact_item_total if exact_item_total else 0.0
        fp_rate = nonexistent_fp / nonexistent_total if nonexistent_total else 0.0
        amb_acc = ambiguous_clar_ok / ambiguous_total if ambiguous_total else 0.0

        print(
            f"\n[TYPO EVAL RESULTS - {len(cases)} synthetic cases]\n"
            f"  Intent Accuracy:       {intent_acc:.2%} ({intent_correct}/{len(cases)})\n"
            f"  Entity Precision:      {precision:.2%}\n"
            f"  Entity Recall:         {recall:.2%}\n"
            f"  Entity F1:             {f1:.2%}\n"
            f"  Exact Menu Match Acc:  {exact_acc:.2%} ({exact_item_correct}/{exact_item_total})\n"
            f"  Nonexistent FP Rate:   {fp_rate:.2%} ({nonexistent_fp}/{nonexistent_total})\n"
            f"  Ambiguous Clarify Acc: {amb_acc:.2%} ({ambiguous_clar_ok}/{ambiguous_total})\n"
            f"  Failures ({len(failures)}): {failures[:10]}"
        )

        # Calibrated strict commercial acceptance thresholds
        assert intent_acc >= 0.95
        assert precision >= 0.95
        assert recall >= 0.92
        assert exact_acc >= 0.92
        assert fp_rate == 0.0, f"Nonexistent items matched falsely: {failures}"
        assert amb_acc == 1.0

    asyncio.run(_run())


def test_multi_turn_pizza_typo_and_self_correction_context():
    dispatcher = _make_dispatcher()
    phone = "01022223333"

    async def _run():
        # Exact user-reported failure scenario:
        # Turn 1: "عايزة بيتزا" -> asks which pizza (NOT generic loop!)
        r1 = await dispatcher.handle_text(phone, "عايزة بيتزا")
        assert r1.requires_clarification is True
        assert r1.clarification_reason == "ambiguous_product"
        assert "مارجريتا" in r1.reply_ar

        # Turn 2: Misspelled follow-up "مارجرينا" -> combines with "بيتزا", recognizes بيتزا مارجريتا, asks size!
        r2 = await dispatcher.handle_text(phone, "مارجرينا")
        assert r2.requires_clarification is True
        assert r2.clarification_reason == "MISSING_OR_INVALID_SIZE"
        assert "بيتزا مارجريتا" in r2.reply_ar

        # Turn 3: Size reply "وسط" -> adds بيتزا مارجريتا (M) = 125.00 EGP
        r3 = await dispatcher.handle_text(phone, "وسط")
        assert r3.requires_clarification is False
        assert r3.cart.subtotal == Decimal("125.00")

        # Turn 4: Customer corrects item -> "لأ خليها بيبروني وسط" -> replaces with بيتزا بيبروني (M) = 170.00 EGP
        r4 = await dispatcher.handle_text(phone, "لأ خليها بيبروني وسط")
        assert r4.requires_clarification is False
        assert len(r4.cart.items) == 1
        assert r4.cart.subtotal == Decimal("170.00")

    asyncio.run(_run())
