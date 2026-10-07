# AlpsBench Task 2 on CUPMEM

This package adapts AlpsBench's supplied memories and incoming dialogue to the
existing CUPMEM write pipeline. **No existing CUPMEM file is modified.** Imports
flow only from `alpsbench` to `cup_mem`. All parsing, initial schema assignment,
output formatting, evaluation, and benchmark logging are in this package.

## Reviewed benchmark contract

Inspected official AlpsBench revision:
`9930dc7c86aefc17b6833838b87453368807789c`.

- [Hugging Face release](https://huggingface.co/datasets/Cosineyx/Alpsbench)
- [Prediction contract](https://github.com/ThisIsCosine/AlpsBench/blob/9930dc7c86aefc17b6833838b87453368807789c/docs/prediction_contract.md)
- [Native public scorer](https://github.com/ThisIsCosine/AlpsBench/blob/9930dc7c86aefc17b6833838b87453368807789c/src/benchmark/metrics.py)
- [Native prediction validator](https://github.com/ThisIsCosine/AlpsBench/blob/9930dc7c86aefc17b6833838b87453368807789c/src/benchmark/runner.py)
- [Public input builder, including original source layout](https://github.com/ThisIsCosine/AlpsBench/blob/9930dc7c86aefc17b6833838b87453368807789c/src/benchmark/data.py)

The public Task-2 input uses `benchmark_id`, `session_id`, and
`input.memory` / `input.new_dialogue`. The prediction must contain **exactly**:

```json
{"benchmark_id": "example_id", "memory_items": []}
```

`memory_items` is the final updated memory state, **not action classes**.
Public references instead use `gold.answer`. The native local scorer compares
multisets of canonicalized `(label, value, type, preference_attitude, time_scope,
emotion)`, reporting precision, recall, F1, and exact match, then macro averages.
It ignores memory IDs and confidence for this comparison, and counts duplicates.
Its text normalization lowercases and normalizes whitespace; it is not semantic
similarity matching. The documented legacy matching/LLM judge is not the
released runnable scorer. Results here are explicitly labeled **public local
proxy**, not official hidden-judge or paper results.

Public `examples`, `dev`, and `validation` references can be scored. Public
`test` has no released gold and is prediction-only. The evaluator loads the
official `metrics.py` directly from a caller-provided checkout and records its
SHA-256; it does not copy or substitute scoring formulas. Pin the checkout above
for reproducibility. Loading another revision intentionally uses that revision's
native scorer; the trace distinguishes its hash from the inspected revision.

## Existing components reused

| CUPMEM concept | Existing component |
| --- | --- |
| Unchanged schema | `cup_mem.memory.schema.BUCKET_SCHEMAS`, `schema_prompt_text`, `is_valid_bucket_track` |
| Session write orchestration | `CupMemEngine.write_session` / `SessionWriterMixin.process_session` |
| Evidence and state extraction | `ChunkerMixin.chunk_session`, `DeltaExtractorMixin.extract_deltas` |
| Local updates | `UpdateResolverMixin.judge_update`, native same-session arbitration |
| Same-slot / same-domain lookup | `ProfileStore.retrieve_candidates_for_track`, temporal variants, profile subset methods |
| Direct and affected candidates | `InvalidationLaneMixin._collect_direct_bucket_tracks`, `propose_affected_buckets`, `_build_invalidation_lane_specs` |
| Global candidates | `InvalidationLaneMixin._build_global_recall_lane_spec` |
| Proposal merge and adjudication | `InvalidationResolverMixin.propose_invalidations`, `InvalidationMergeMixin`, `InvalidationJudgeMixin.judge_invalidation` |
| ACTIVE / STALE / UNKNOWN_CURRENT | `ProfileStore`, `StoreTransitionMixin.apply_update` / `apply_invalidation`, `ProfileItem`, `UnknownCurrent` |
| Stale evidence linkage | `StaleLinkerMixin.link_stale_support` |
| LLM calls and cache | `cup_mem.llm_layer.client.LLMClient` |

CUPMEM calls domains **buckets** and slots **local tracks**. Adapter assignment
records expose `domain` and `slot`; internal items use the existing field names.
No schema names, cardinalities, prompts, thresholds, retrieval, update rules,
adjudication, or query behavior are changed.

## Experimental boundary

1. Parse an allowlisted input projection. Ignore the dataset's updater `query`,
   `old_dialogue`, outer gold, `answer`, and all control metadata. The actual
   public inputs contain `metadata.controls.expectedAction` and
   `metadata.selected_memory_id`; **neither is model-visible here**.
2. Classify each initial memory's **value only** into one valid existing
   bucket/track pair. Invalid assignments fail rather than extending the schema.
   No gold labels, gold updated memories, or annotation controls are consulted.
   A poor schema fit is possible: the original CUPMEM schema remains unchanged.
3. Seed a `ProfileStore` directly with every supplied memory as ACTIVE. Preserve
   the value byte-for-byte, including whitespace, and keep the complete original
   object in a deep-copied sidecar. No extraction, ADD decision, merging, or
   deduplication is run at initialization: doing so would change the benchmark's
   starting state. Even multiple memories assigned to a single-cardinality slot
   are preserved initially; native cardinality rules govern subsequent writes.
4. Initial items have namespaced internal IDs and session index 0. Feed the
   **entire** `new_dialogue` into exactly one native write call, session index 1.
   Assistant turns are preserved at the boundary; CUPMEM's normal user-turn
   selection still controls evidence. Reset the engine between examples.
5. Convert the completed transition. Emit one memory per final ACTIVE item,
   including WEAK items. Preserve unchanged originals exactly. For new/refined/
   replaced values, keep the native text, confidence, and evidence; derive
   direct/indirect from native delta provenance. A separate LLM classifier fills
   only AlpsBench metadata such as labels, attitude, and time scope. It cannot
   change memory membership or values. Its response keys are validated.
6. Omit STALE items from asserted current memories. Keep UNKNOWN_CURRENT in the
   trace without inventing a replacement fact. If a track contains both a native
   ACTIVE item and UNKNOWN_CURRENT, retain the ACTIVE item as CUPMEM does.
7. Only **after predictions have been written**, load separate references, join
   gold into traces, and invoke the native public scorer.

Output classification is an additional measured adapter cost and can affect
proxy scores. CUPMEM-generated wording can also score poorly under exact field
matching despite semantic similarity. The adapter does not rewrite values to
optimize that metric or recreate benchmark gold.

No incoming timestamp is supplied in the inspected examples. The default
session time is empty; generated output uses `updated_at: null` in that case.
Use `--session-time` to record an explicitly chosen experiment timestamp. Native
session-order guards use indices, so no gold or fabricated wall-clock date is
needed. Original timestamps remain intact for unchanged memories.

## Installation and execution

Run from this repository's root, with Python 3.10+ and CUPMEM's dependencies:

```bash
python -m pip install -r cup_mem/requirements.txt
```

CUPMEM requires an existing local embedding model directory; this adapter does
not introduce a lexical fallback or silently download model weights. Set the
provider key in `OPENAI_API_KEY` (or select another variable with `--api-key-env`).
The CLI uses the existing Chat Completions `LLMClient`.

```bash
python -m alpsbench.task2.runner \
  --input /path/to/AlpsBench/benchmark_data/dev/task2/model_input.jsonl \
  --output-dir outputs/alpsbench/dev \
  --model YOUR_MODEL \
  --embedding-model-path /path/to/local/embedding-model \
  --split dev \
  --references /path/to/AlpsBench/benchmark_data/dev/task2/reference_output.jsonl \
  --benchmark-root /path/to/AlpsBench
```

For test inference, use `--split test` and omit references and benchmark root.
Use `--base-url` / `OPENAI_BASE_URL` for another compatible provider.
`--cache-dir` defaults to `outputs/alpsbench/cache`; it is namespaced by endpoint,
and CUPMEM's native cache hashes model plus full messages. Initial mapping
messages include the schema and versioned classification prompt. Thus repeated
values reuse mappings, while schema/prompt/model changes change cache keys.
The original CUPMEM handling of non-cacheable temperature requests is preserved.

The runner also accepts original Hugging Face source rows with
`entry.memory`, `entry.record.new_dialogue`, and `entry.metadata.session_id`
(or those same fields without the `entry` wrapper). Combined source gold is
ignored, not automatically converted into references. Malformed inputs fail
explicitly. Git LFS pointer files must first be hydrated to actual JSONL.

To use the official benchmark's prediction-program interface:

```bash
python /path/to/AlpsBench/scripts/evaluate.py \
  --task task2 --split examples \
  --predict-program python \
  --predict-arg=-m --predict-arg=alpsbench.task2.runner \
  --predict-arg=--stdin \
  --predict-arg=--output-dir --predict-arg=outputs/alpsbench/adapter \
  --predict-arg=--model --predict-arg=YOUR_MODEL \
  --predict-arg=--embedding-model-path --predict-arg=/path/to/local/embedding-model
```

In this mode stdout contains only the prediction object. Each example's traces
go into a hashed ID subdirectory, so separate subprocesses do not overwrite one
another. Batch execution amortizes embedding-model loading and is preferable
for larger runs. Batch reruns replace the named output artifacts; use a new
output directory to retain previous runs. Completed examples are flushed to disk
before the next example starts.

To score already-generated predictions without model calls:

```bash
python -m alpsbench.task2.evaluator \
  --benchmark-root /path/to/AlpsBench \
  --predictions outputs/alpsbench/dev/predictions.jsonl \
  --references /path/to/AlpsBench/benchmark_data/dev/task2/reference_output.jsonl \
  --split dev --output outputs/alpsbench/dev/evaluation.json
```

## Trace artifacts

- `predictions.jsonl`: native submission rows, with no additional top-level keys.
- `traces.jsonl`: one record per completed example, including original memories,
  domain/slot assignments, incoming dialogue, extracted evidence, update
  candidates and local decisions, direct/affected/global candidates, adjudication,
  before/after stores, output conversion, prediction, and `gold_result` (null
  without public references). Also includes the full native session trace,
  thresholds, schema hash, model configuration, cache usage, and LLM call records.
- `evaluation.json`: native row scores, summary, split, and scorer provenance,
  when public references are supplied.

Native lane logs contain candidate pairs and summaries. The trace adapter adds
full candidate memory views using native store lookup methods on the untouched
initial decision snapshot. Empty arrays mean a lane or stage had no candidates;
the native trace includes filter and arbitration details. Original benchmark
objects are nested beside `domain`/`slot` to avoid overwriting any original keys.

## Tests and review map

```bash
python -m pip install pytest
python -m pytest alpsbench/task2/tests -q
ALPSBENCH_ROOT=/path/to/AlpsBench python -m pytest alpsbench/task2/tests -q
```

Tests cover schema pairs, invalid assignments, lossless initialization, duplicate
IDs and single-cardinality initialization, exactly one session, original input
layout, gold/control exclusion, native writer execution, all three candidate
lanes, native update/invalidation/weak semantics, protected output values,
mapping cache reuse, and dependency direction. The optional native-scoring test
checks exact canonical fields, duplicate counting, invalid IDs, and test-split
rejection. Test doubles replace network responses and embedding ranking only;
production behavior is not replaced.

Every added source file is listed below. No pre-existing source files changed.

| File | Purpose |
| --- | --- |
| [`__init__.py`](__init__.py) | Top-level package boundary |
| [`task2/__init__.py`](task2/__init__.py) | Task package boundary |
| [`task2/models.py`](task2/models.py) | Inference-only types and client protocol |
| [`task2/dataset.py`](task2/dataset.py) | Input projection, validation, reference loading |
| [`task2/memory_adapter.py`](task2/memory_adapter.py) | Schema assignment and lossless initial store |
| [`task2/session_adapter.py`](task2/session_adapter.py) | One-session mapping |
| [`task2/output_adapter.py`](task2/output_adapter.py) | Final ACTIVE-state conversion and protected metadata classification |
| [`task2/evaluator.py`](task2/evaluator.py) | Native scorer loading and evaluation CLI |
| [`task2/logging_utils.py`](task2/logging_utils.py) | Structured trace construction and JSONL serialization |
| [`task2/runner.py`](task2/runner.py) | Prediction API, batch CLI, stdin adapter, late gold join |
| [`task2/tests/__init__.py`](task2/tests/__init__.py) | Test package |
| [`task2/tests/test_adapters.py`](task2/tests/test_adapters.py) | Offline and native scorer tests |
| [`README.md`](README.md) | Contract, rationale, usage, limitations, and review index |
