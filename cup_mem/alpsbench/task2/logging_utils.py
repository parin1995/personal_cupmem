"""Structured traces kept separate from contract-conforming submissions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .memory_adapter import InitialMemory
from .schema import JsonObject, Task2Input


def build_trace(
    example: Task2Input, initial: InitialMemory, session_trace: JsonObject,
    prediction: JsonObject, output_logs: list[JsonObject],
) -> JsonObject:
    """Expose requested phases without replacing any CUPMEM decision logic.

    Native lane logs retain pairs and summaries. Recover full candidate views
    using the very same ProfileStore lookup methods on the unmodified initial
    decision store. Preserve the complete native trace alongside this index.
    """
    lanes = []
    for lane in session_trace.get("invalidation_lane_logs", []):
        pairs = [(pair["bucket"], pair["local_track"]) for pair in lane["candidate_pairs"]]
        lookup = (initial.store.get_exact_profile_subset_before_session
                  if lane["lane_type"] == "global_recall"
                  else initial.store.get_profile_subset_before_session)
        lanes.append({**deepcopy(lane), "candidate_memory": lookup(pairs, session_index=1)})
    return {
        "trace_version": 1,
        "example_id": example.benchmark_id,
        "initial_alpsbench_memories": deepcopy(list(example.initial_memories)),
        "cupmem_domain_slot_assignments": initial.assignment_trace(),
        "initial_cupmem_memory": initial.store.to_snapshot(),
        "incoming_dialogue": deepcopy(list(example.new_dialogue)),
        "extracted_state_evidence": deepcopy(session_trace.get("chunks", [])),
        "state_update_candidates": deepcopy(session_trace.get("delta_filter_logs", [])),
        "local_update_decisions": deepcopy(session_trace.get("delta_logs", [])),
        "direct_revision_candidates": [lane for lane in lanes if lane["lane_type"] == "direct_bucket"],
        "affected_regions": deepcopy(session_trace.get("affected_buckets", [])),
        "affected_candidates": [lane for lane in lanes if lane["lane_type"] == "affected_bucket"],
        "global_candidates": [lane for lane in lanes if lane["lane_type"] == "global_recall"],
        "adjudication_decisions": deepcopy(session_trace.get("invalidation_logs", [])),
        "final_cupmem_memory": deepcopy(session_trace["profile_snapshot_after_session"]),
        "alpsbench_prediction": deepcopy(prediction),
        "output_conversion": deepcopy(output_logs),
        "gold_result": None,
        "cupmem_session_trace": deepcopy(session_trace),
    }


def write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    """Write UTF-8 JSONL without adding trace keys to prediction rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
