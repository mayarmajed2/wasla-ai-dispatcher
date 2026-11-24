# Wasla AI --- Phase A: AI Audit Report

**Audit date:** 24 November 2026\
**Source:** `wasla-ai-dispatcher-main.zip` (repository snapshot)\
**Audit mode:** Read-only source inspection + local test execution\
**Model changes/training:** None

## 1. Executive summary

The ZIP contains the actual application source, two CAMeLBERT PEFT LoRA
checkpoints, training/evaluation data, and tests. The audit found that
the project has a functioning deterministic restaurant-order dispatcher
and fuzzy menu matcher, but several documented AI capabilities are not
connected to the live request path.

The most important confirmed findings are:

1.  **CAMeLBERT LoRA is instantiated but never used for intent
    inference.** `WaslaAIDispatcher` creates
    `FineTunedIntentClassifier`, but no dispatcher path calls
    `.predict()`. The existing LoRA load test fails in this audit
    environment because `peft` is not installed; `transformers` is also
    absent, and neither `peft`, `transformers`, nor `torch` is declared
    in `requirements.txt`.
2.  **Gemini Function Calling is not implemented in the Gemini
    request.** The repository defines `TOOL_DECLARATIONS` and a local
    `ToolExecutor`, but `GeminiLLMClient.generate_json()` sends no
    `tools` / `functionDeclarations` field. The dispatcher calls local
    tools itself based on parsed intent.
3.  **Gemini fallback is skipped on primary API errors.** The fallback
    model is called only when a successful primary response has
    confidence below `0.72`. If the primary request raises an HTTP,
    timeout, or JSON parsing exception, the code goes directly to
    `OfflineEgyptianNLU`.
4.  **Conversation state is process-local and is not rehydrated from the
    database.** The dispatcher stores `SessionState` in
    `self._sessions`; the backend writes a cart snapshot to
    `ChatSession.pending_cart` but no request path restores that
    snapshot into the dispatcher. The production bridge inherits the
    in-memory mock cart implementation.
5.  **Menu behavior can contradict current backend data.** The AI loads
    a static `MenuIndex` once. Price edits update the database but do
    not refresh the index used for AI replies/cart calculation. The
    order-creation path later recalculates from the database, creating a
    possible displayed-total vs final-total mismatch. Availability is
    also ignored in price replies.
6.  **The menu parser drops repeated instances of the same product in
    one utterance**, even if the customer requested different sizes.
    Multi-product price inquiries also return only the first requested
    product.

These findings are based on source inspection and local reproductions.
No live Gemini request, live WhatsApp request, production database test,
or model training was performed.

## 2. Scope and evidence discipline

Inspected:

-   `ai_service/models/finetuned_intent.py`
-   `ai_service/models/llm_client.py`
-   `ai_service/dispatcher/dispatcher.py`
-   `ai_service/tools/registry.py`
-   `ai_service/understanding/menu_index.py`
-   `ai_service/understanding/arabic_nlp.py`
-   `ai_service/schemas/models.py`
-   `ai_service/adapters/mock_backend.py`
-   `backend/app/main.py`
-   `backend/app/models.py`
-   `requirements.txt`
-   LoRA checkpoint metadata and safetensors files
-   Training datasets, evaluation reports, and test suites

The archive does not include a `.git/` directory, so this is an audit of
the supplied source snapshot, not a commit-history or diff audit.

**Evidence labels used below:**

-   **Runtime reproduced:** executed locally against the supplied
    source.
-   **Source-traced:** follows directly from the code path, but not
    tested against a live production service.
-   **Not verified:** requires credentials, a live provider, or
    production infrastructure unavailable in this audit.

## 3. Confirmed findings

### A-01 --- CAMeLBERT LoRA is not in the dispatcher inference path

**Severity: Critical for the stated LoRA requirement**

**Evidence**

-   `ai_service/dispatcher/dispatcher.py:93-100` constructs
    `self.finetuned_intent`.
-   `ai_service/dispatcher/dispatcher.py:510-545` implements the actual
    NLU route: Gemini when configured, otherwise `OfflineEgyptianNLU`.
    It does not call `self.finetuned_intent.predict()`.
-   Repository-wide search found no production call to `.predict()`;
    references are limited to the classifier implementation,
    initialization/health reporting, and its test.
-   `backend/app/main.py:655-656` reports `checkpoint_exists` and
    `is_loaded`, but does not trigger a load.
-   `ai_service/models/finetuned_intent.py:50-74` lazily imports
    `torch`, `peft`, and `transformers`.
-   `requirements.txt` does not declare `torch`, `peft`, or
    `transformers`.

**Runtime reproduction**

Command:

``` bash
pytest -q tests/ai/test_finetuned_lora_model.py
```

Result in this audit environment: **1 passed, 1 failed**. The
real-checkpoint test fails at `clf.load()` with:

``` text
Failed to load fine-tuned LoRA checkpoint: No module named 'peft'
```

`transformers` is also absent in this environment. The failure occurs
before the adapter is loaded into a model; it is not evidence that the
safetensors weights are corrupt.

**Checkpoint inspection**

-   Both v1 and v2 adapter files exist in the ZIP.
-   The v2 adapter metadata identifies the CAMeL-BERT base model, PEFT
    LoRA, rank 16, alpha 32, and target modules `query`, `key`, and
    `value`.
-   The v2 safetensors file was structurally readable and exposed 74
    tensors.
-   The base model weights are not bundled;
    `AutoModelForSequenceClassification.from_pretrained()` must obtain
    the base model from the Hugging Face cache or network.

**Conclusion**

The checkpoint artifact exists, but the production dispatcher does not
use it. In the supplied dependency manifest, a clean installation also
lacks the packages required to load it. The README/model card should not
be read as proof that LoRA is active in live inference.

### A-02 --- Gemini Function Calling declarations are not sent to Gemini

**Severity: High if Gemini-native tool calling is a requirement;
otherwise an architecture/documentation mismatch**

**Evidence**

-   `ai_service/tools/registry.py:16-103` defines eight
    `TOOL_DECLARATIONS`.
-   `ai_service/tools/registry.py:106-226` implements local Python
    execution in `ToolExecutor`.
-   `ai_service/models/llm_client.py:43-74` constructs the Gemini
    `generateContent` payload with `systemInstruction`, `contents`, and
    `generationConfig`; it contains no `tools` or
    `functionDeclarations`.
-   `ai_service/dispatcher/dispatcher.py:251-304` and subsequent intent
    branches call `ToolExecutor` directly based on the parsed intent.

**Runtime/source reproduction**

The supplied regression test inspects `GeminiLLMClient.generate_json()`
and fails because the request payload does not contain a `tools` field.

**Conclusion**

There is a local deterministic tool-execution layer, which is useful for
transactional safety. But Gemini is currently a structured-JSON NLU
step, not a Gemini Function Calling agent. The distinction matters:
Gemini cannot select and call the declared tools through the provider
API in the current implementation.

### A-03 --- Primary Gemini failures bypass the configured fallback model

**Severity: High**

**Evidence**

-   `ai_service/dispatcher/dispatcher.py:539-542` calls the fallback
    model only after a successful primary response whose confidence is
    below `0.72`.
-   `ai_service/dispatcher/dispatcher.py:543-545` catches exceptions and
    immediately uses `OfflineEgyptianNLU`.

**Runtime reproduction**

A fake LLM was configured to raise a simulated primary HTTP 503 and
record each call. The dispatcher called it once with
`use_fallback_model=False`; it did not call `use_fallback_model=True`.
It returned `offline_egyptian_nlu`.

**Conclusion**

The configured fallback handles low-confidence output, not primary
transport/API/parse failures. This is narrower than the apparent
primary/fallback configuration suggests.

### A-04 --- Conversation state is not durable or rehydrated

**Severity: Critical for multi-worker/restart-safe conversations**

**Evidence**

-   `ai_service/dispatcher/dispatcher.py:102-107` stores sessions in an
    in-memory dictionary.
-   `ai_service/schemas/models.py:180-198` contains cart, customer
    details, pending clarification, pending upsell, rejected upsells,
    history, and other conversational state.
-   `backend/app/main.py:906-930` writes a cart snapshot and a small
    `conversation_state` summary to `ChatSession`.
-   `pending_cart` is written at line 919, but no request path reads it
    to reconstruct `SessionState`.
-   `backend/app/main.py:244-247` creates one global dispatcher and
    `_DirectDBOrderBridge`.
-   `_DirectDBOrderBridge` at `backend/app/main.py:210-213` inherits
    `MockBackendAdapter`.
-   `ai_service/adapters/mock_backend.py:41-45` stores carts in an
    in-memory dictionary; `modify_cart()` and `get_cart_summary()` use
    that dictionary.

**Runtime reproduction**

The supplied regression test:

1.  Adds a real menu item through one dispatcher.
2.  Creates a second dispatcher with the same backend adapter.
3.  Reads the new session for the same customer.

The first dispatcher has one cart item; the second dispatcher session
has zero items, even though the adapter's in-memory cart still contains
one. The test fails as expected.

**Conclusion**

A process restart loses dispatcher state. Multiple worker processes can
also hold divergent sessions/carts. The database snapshot is currently
write-only from the AI dispatcher's perspective; pending clarification,
pending upsell, history, customer details, budget, and related state are
not reconstructed.

`SESSION_TTL_SECONDS` is defined in `ai_service/config.py:53`, but no
expiry/cleanup path uses it.

### A-05 --- Mutable request metadata is stored on the shared dispatcher instance

**Severity: Medium; concurrency risk**

**Evidence**

-   `ai_service/dispatcher/dispatcher.py:132-133` assigns
    `_current_source` and `_current_is_test` on `self` for each incoming
    request.
-   `ai_service/dispatcher/dispatcher.py:772-773` reads those instance
    attributes later when confirming an order.
-   `backend/app/main.py:247` uses a shared global dispatcher.

**Conclusion**

Concurrent requests can overwrite each other's metadata between the
initial assignment and the later awaited confirmation path. This is a
source-traced race risk; no concurrent stress test was run for this
specific case.

### A-06 --- Same product requested in two sizes is silently dropped

**Severity: High for order extraction correctness**

**Evidence**

-   `ai_service/models/llm_client.py:264-268` initializes a `seen_ids`
    set.
-   `ai_service/models/llm_client.py:279-282` skips any later segment
    whose `product_id` is already in that set.

**Runtime reproduction**

Input:

``` text
عايز بيتزا مارجريتا وسط وبيتزا مارجريتا كبير
```

Observed parser output: one `item_029` line with size `M`; the requested
large variant is omitted.

The regression test expecting two line items fails. This can understate
the customer's order.

### A-07 --- Price inquiry ignores unavailable status

**Severity: High for customer-facing menu accuracy**

**Evidence**

-   `ai_service/dispatcher/dispatcher.py:574-576` calls
    `check_item_availability`, but discards its returned `available`
    flag and renders the normal price reply regardless.

**Runtime reproduction**

The mock backend was set to mark `item_029` unavailable. The dispatcher
returned:

``` text
سعر بيتزا مارجريتا: وسط: 125 ج.م | كبير: 150 ج.م ...
```

The tool-call record simultaneously reported `available=False`.

**Conclusion**

The dispatcher knows the item is unavailable but still presents it as a
normal purchasable item in the price response.

### A-08 --- Multi-product price inquiry answers only the first product

**Severity: Medium**

**Evidence**

-   `ai_service/dispatcher/dispatcher.py:554-577` handles only
    `items[0]`.

**Runtime reproduction**

Input:

``` text
بكام بيتزا مارجريتا وسط وكريب كوردن بلو
```

The parser extracts both `item_029` and `item_020`, but the final reply
contains only the MARGHERITA pizza price.

### A-09 --- AI menu prices can become stale after dashboard price edits

**Severity: Critical for price-consistency / customer-confirmation
safety**

**Evidence**

-   `backend/app/main.py:244-247` creates `MenuIndex` once at
    application startup.
-   `ai_service/understanding/menu_index.py:99-157` loads product prices
    from the static JSON file into memory.
-   `backend/app/main.py:889-903` updates `MenuItem.base_price` in the
    database but does not refresh `menu_index` or `db_bridge`.
-   `_DirectDBOrderBridge` inherits mock cart methods;
    `ai_service/adapters/mock_backend.py:112-158` calculates cart line
    prices from the in-memory `MenuIndex`.
-   `backend/app/main.py:516-541` resolves item prices again from the
    current database row when creating the final order.

**Conclusion**

After a dashboard price edit, the AI can show and calculate a cart using
the old JSON-loaded price, while order creation recalculates using the
new database price. The customer may therefore confirm a total different
from the amount stored on the order. This was source-traced, not
exercised against a live database.

### A-10 --- Availability toggles are not restored into the AI adapter after restart

**Severity: High**

**Evidence**

-   `backend/app/main.py:858-886` persists the availability flag to the
    database and also updates `db_bridge`'s in-memory unavailable set.
-   `_DirectDBOrderBridge` inherits `MockBackendAdapter`;
    `ai_service/adapters/mock_backend.py:38-45` initializes
    `_unavailable_ids` as an empty set.
-   `ai_service/adapters/mock_backend.py:64-77` uses that in-memory set
    for active menu and availability checks.
-   No startup path hydrates `_unavailable_ids` from persisted
    `MenuItem.is_available`.
-   `backend/app/main.py:516-520` later rejects unavailable items during
    final order creation.

**Conclusion**

An out-of-stock state can be lost by the AI adapter after restart even
though it remains stored in the database. The customer may be allowed to
add the item and only receive a rejection at final confirmation.

### A-11 --- The 3-second message debouncer exists but is not wired into inbound handling

**Severity: Medium; documented capability is not active in the inspected
paths**

**Evidence**

-   `ai_service/dispatcher/debounce.py:10-56` implements
    `MessageDebouncer`.
-   Repository search found its runtime use only in its tests; the
    inbound handlers in `backend/app/main.py` call
    `ai_dispatcher.handle_text()` directly (including lines 1030, 1372,
    and 1879).

**Conclusion**

The class is implemented and tested in isolation, but fragmented
incoming customer messages are not aggregated by this debouncer in the
inspected webhook/API paths.

## 4. Test execution

### Existing test suites

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Command                                                                                                                                                         Result
  ----------------------------------------------------------------------------------------------------------------------------- ----------------------------------------
  `pytest -q tests/ai/test_dispatcher_and_eval.py tests/ai/test_multi_turn_dialogues.py tests/unit/test_menu_and_security.py`                              **13 passed**

  `pytest -q tests/ backend/test_cart_calculator.py`                                                                                             **41 passed, 2 failed**

  `pytest -q tests/ai/test_finetuned_lora_model.py`                                                                                               **1 passed, 1 failed**
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

The two failures in the full suite were:

1.  LoRA load test: missing `peft`. This is also a dependency-manifest
    gap because the runtime code imports `peft`/`transformers`/`torch`,
    but the main `requirements.txt` does not declare them.
2.  PostgreSQL URL normalization test: missing `pg8000` in this audit
    runtime. `pg8000>=1.31.0` is present in the repository requirements,
    so this particular failure is an audit-environment provisioning gap,
    not proof of a source defect.

The repository's previous `docs/PROJECT_AUDIT_AND_TEST_RESULTS.md`
reports a different environment and 14 passing tests. The current ZIP
contains a broader test suite; results above are from this audit run,
under Python 3.13.5.

### New regression reproductions

A separate file, `wasla_phase_a_repro_tests.py`, contains seven tests
for the reproduced issues:

-   duplicate product variants are dropped;
-   unavailable item price is still presented normally;
-   only the first product in a multi-product price inquiry is answered;
-   Gemini primary failure bypasses fallback;
-   a new dispatcher instance does not restore the previous cart
    session;
-   LoRA prediction is never called by the dispatcher;
-   Gemini tool declarations are absent from the request payload.

**Current result: 7 failed**, which is expected for these regression
tests against the current implementation. The file is separate from the
supplied repository snapshot.

## 5. Existing model-evaluation evidence (not rerun)

The repository's saved training/evaluation artifacts state:

-   Dataset summary: 732 synthetic MOKA-grounded examples; 520 train,
    106 validation, 106 unseen test.
-   The v2 training log reports best validation Macro-F1 `0.8858`.
-   The independent challenge file contains 204 balanced examples (17
    for each of 12 intents).
-   The saved independent benchmark reports LoRA v2 accuracy `60.29%`,
    Macro-F1 `0.5890`, average latency `62.03 ms`.
-   The same saved report gives the TF-IDF + Logistic Regression
    baseline `81.37%` accuracy, Macro-F1 `0.8100`, average latency
    `1.62 ms`.

These are values recorded in the repository's reports; they were not
recomputed by training a model in this audit. The validation score
`0.8858` must not be presented as independent-test performance.

## 6. What was not verified

-   No live Gemini request was made. The ZIP contains `.env.example`,
    not `.env`; no Gemini API key was available in the audit
    environment.
-   No live WhatsApp/Meta or Telegram exchange was made.
-   No production PostgreSQL or multi-worker deployment was exercised.
-   Full CAMeLBERT inference was not run because the required
    PEFT/Transformers packages were unavailable. The adapter files were
    inspected structurally only.
-   No training, fine-tuning, checkpoint overwrite, source-code patch,
    or model change was performed.

## 7. Recommended Phase A priority order

1.  **Price integrity:** make the AI menu/cart use the same
    authoritative, current database prices as final order creation; add
    a test proving displayed and confirmed totals match after a price
    edit.
2.  **Durable session state:** restore cart, customer details, pending
    clarification/upsell, takeover state, and history from persistent
    storage; ensure worker-safe locking/coordination.
3.  **LoRA decision:** either wire the classifier into the NLU path with
    explicit confidence/fallback policy and declared dependencies, or
    clearly mark it as offline-only and remove claims that it is active
    in production.
4.  **Gemini orchestration decision:** either implement provider-native
    Function Calling and execute returned calls through validated
    `ToolExecutor`, or document the current approach accurately as
    structured-JSON NLU plus deterministic local tool routing.
5.  **Fallback reliability:** invoke the configured fallback model for
    eligible primary transport/provider/parse failures, with bounded
    retry and clear telemetry.
6.  **Menu parser correctness:** preserve repeated product
    lines/variants, handle every requested product in price inquiries,
    and respect availability in customer replies.
7.  **Operational wiring:** connect message debounce to actual inbound
    handlers and remove shared mutable per-request metadata from the
    dispatcher instance.

**Phase A conclusion:** The codebase has meaningful deterministic
ordering and safety components, and the focused offline dispatcher/menu
tests pass. However, the LoRA, Gemini Function Calling, durable
conversation-state, and live menu synchronization claims are not fully
supported by the inspected runtime path. The findings above should be
resolved and re-tested before any model retraining is considered.
