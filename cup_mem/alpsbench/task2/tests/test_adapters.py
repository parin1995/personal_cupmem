"""Test boundaries using native CUPMEM transitions, without paid model calls."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cup_mem import CupMemEngine, LLMClient
from cup_mem.memory.models import InvalidationProposal, SessionDelta, UpdateDecision
from cup_mem.memory.schema import BUCKET_SCHEMAS

from alpsbench.task2.dataset import parse_input
from alpsbench.task2.evaluator import NativeEvaluator
from alpsbench.task2.memory_adapter import SchemaClassifier, initialize_memory, validate_assignment
from alpsbench.task2.schema import SchemaAssignment
from alpsbench.task2.output_adapter import convert_output
from alpsbench.task2.runner import predict_example
from alpsbench.task2.session_adapter import build_session

PAIR = SchemaAssignment("location_and_living", "current_base_location")


def row() -> dict:
    """Synthetic public-shaped example, with deliberate annotation tripwires."""
    return {
        "benchmark_id": "example_1", "session_id": "session_1", "task": "task2",
        "gold": {"answer": [{"value": "GOLD_SENTINEL"}]},
        "input": {
            "memory": [{"memory_id": "m1", "value": "  Lives in London.\n",
                        "label": "Personal_Background/Location", "type": "direct",
                        "confidence": 0.8, "time_scope": "long_term", "emotion": None,
                        "preference_attitude": None, "extra_original_field": {"keep": True}}],
            "new_dialogue": [{"role": "user", "text": "I moved to Paris."},
                             {"role": "assistant", "text": "How is Paris?"},
                             {"role": "user", "text": "I live there now."}],
            "old_dialogue": [{"role": "user", "text": "OLD_DIALOGUE_SENTINEL"}],
            "query": "QUERY_SENTINEL",
            "metadata": {"selected_memory_id": "SELECTED_SENTINEL",
                         "controls": {"expectedAction": "ACTION_SENTINEL"}},
            "answer": [{"value": "ANSWER_SENTINEL"}],
        },
    }


def metadata(item: dict, evidence: list[dict]) -> dict:
    """A deterministic formatter standing in for boundary metadata labeling."""
    return {"label": "Personal_Background/Location", "label_suggestion": None,
            "time_scope": "long_term", "emotion": None, "preference_attitude": None,
            "reasoning": "Classified the accepted location value."}


def delta() -> SessionDelta:
    return SessionDelta("d_00001", "s_001", 1, "", PAIR.domain, PAIR.slot,
                        "Lives in Paris.", "explicit", ["c_00001"], 0.95)


def trace_for_delta(value: SessionDelta) -> dict:
    return {"delta_logs": [{"delta": value.to_dict()}],
            "chunks": [{"chunk_id": "c_00001", "source_turn_ids": ["t_0", "t_2"]}]}


@pytest.mark.parametrize("domain,bucket", BUCKET_SCHEMAS.items())
def test_all_native_schema_pairs_are_accepted(domain, bucket):
    for slot in bucket.tracks:
        assert validate_assignment(SchemaAssignment(domain, slot)).slot == slot


@pytest.mark.parametrize("assignment", [SchemaAssignment("invented", "slot"),
    SchemaAssignment("stable_preferences", "current_base_location")])
def test_invalid_schema_pairs_rejected(assignment):
    with pytest.raises(ValueError, match="Invalid CUPMEM"):
        validate_assignment(assignment)


def test_initial_memories_are_lossless_and_not_filtered_by_cardinality():
    source = row()
    source["input"]["memory"] *= 2  # repeated source IDs and a single-cardinality slot
    example = parse_input(source)
    seen = []

    def classify(value):
        seen.append(value)
        return PAIR

    initial = initialize_memory(example, classify)
    items = initial.store.get_active_for_track(PAIR.domain, PAIR.slot)
    assert len(items) == 2
    assert len({item.item_id for item in items}) == 2
    assert all(item.value == source["input"]["memory"][0]["value"] for item in items)
    assert all(item.created_session_index == 0 and item.status == "ACTIVE" for item in items)
    assert seen == [memory["value"] for memory in source["input"]["memory"]]
    assert list(initial.originals.values()) == source["input"]["memory"]
    initial.originals[items[0].item_id]["extra_original_field"]["keep"] = False
    assert example.initial_memories[0]["extra_original_field"]["keep"] is True
    assert initial.store.retrieve_candidates_for_track_before_session(
        PAIR.domain, PAIR.slot, session_index=1)["same_track_active"] == items


def test_session_preserves_all_turns_and_native_user_selection():
    example = parse_input(row())
    session = build_session(example)
    assert session == [{"role": turn["role"], "content": turn["text"]} for turn in example.new_dialogue]
    assert [turn["turn_id"] for turn in CupMemEngine.session_to_user_turns(session)] == ["t_0", "t_2"]


def test_original_huggingface_layout_and_public_projection_agree():
    public = row()
    original = {"entry": {"metadata": {"session_id": "example_1", "controls": {"gold": "secret"}},
                          "memory": public["input"]["memory"],
                          "record": {"new_dialogue": public["input"]["new_dialogue"]},
                          "answer": [{"value": "secret"}]}}
    parsed = parse_input(original)
    assert parsed.initial_memories == parse_input(public).initial_memories
    assert parsed.new_dialogue == parse_input(public).new_dialogue


def test_unchanged_output_is_the_original_object_and_never_calls_formatter():
    example = parse_input(row())
    initial = initialize_memory(example, lambda value: PAIR)

    def forbidden(*args):
        raise AssertionError("Unchanged memory must bypass output classification")

    prediction, _ = convert_output(example, initial, initial.store.to_snapshot(), {}, forbidden)
    assert prediction == {"benchmark_id": example.benchmark_id, "memory_items": list(example.initial_memories)}


@pytest.mark.parametrize("action", ["REFINE", "REPLACE", "INDIRECT_INVALIDATE"])
def test_output_uses_actual_native_transition(action):
    example = parse_input(row())
    initial = initialize_memory(example, lambda value: PAIR)
    store = initial.store.clone()
    change = delta()
    store.apply_update(change, UpdateDecision(action, "test", "alps_initial_00000"), timestamp="")
    snapshot = store.to_snapshot()
    prediction, _ = convert_output(example, initial, snapshot, trace_for_delta(change), metadata)
    if action == "INDIRECT_INVALIDATE":
        assert prediction["memory_items"] == []
        assert snapshot["unknown_current"][0]["status"] == "UNKNOWN_CURRENT"
    else:
        assert len(prediction["memory_items"]) == 1
        memory = prediction["memory_items"][0]
        assert memory["value"] == "Lives in Paris."
        assert memory["type"] == "direct"
        assert memory["evidence"]["turn_indices"] == [0, 2]
        assert memory["updated_at"] is None
    assert store.to_snapshot() == snapshot  # output conversion is read-only
    assert initial.originals["alps_initial_00000"]["value"] == example.initial_memories[0]["value"]


def test_addition_retains_original_and_weak_challenge_stays_active():
    example = parse_input(row())
    initial = initialize_memory(example, lambda value: PAIR)
    store = initial.store.clone()
    proposal = InvalidationProposal("lip_1", "s_001", PAIR.domain, PAIR.slot, [], [], "test", 0.8)
    store.apply_invalidation(proposal, UpdateDecision("WEAK_CHALLENGE", "test", "alps_initial_00000"), timestamp="")
    change = delta()
    change.bucket, change.local_track = "stable_preferences", "enduring_preference"
    store.apply_update(change, UpdateDecision("ADD", "test"), timestamp="")
    prediction, _ = convert_output(example, initial, store.to_snapshot(), trace_for_delta(change), metadata)
    assert prediction["memory_items"][0] == example.initial_memories[0]
    assert len(prediction["memory_items"]) == 2


def test_output_classifier_cannot_rewrite_values():
    example = parse_input(row())
    initial = initialize_memory(example, lambda value: PAIR)
    store = initial.store.clone()
    change = delta()
    store.apply_update(change, UpdateDecision("REPLACE", "test", "alps_initial_00000"), timestamp="")
    with pytest.raises(ValueError, match="exactly"):
        convert_output(example, initial, store.to_snapshot(), trace_for_delta(change),
                       lambda *args: {**metadata(*args), "value": "forbidden rewrite"})


class ScriptedLLM:
    """Phase-aware fake: the complete native write orchestration stays real."""

    def __init__(self, update=True):
        self.calls = []
        self.update = update

    def call_json(self, system_prompt, user_payload, **kwargs):
        payload = json.loads(user_payload)
        phase = kwargs["extra_meta"]["phase"]
        self.calls.append((phase, user_payload))
        if phase == "alpsbench_initial_schema_assignment":
            return {"domain": PAIR.domain, "slot": PAIR.slot}
        if phase == "session_chunker":
            if not self.update:
                return {"chunks": []}
            return {"chunks": [{"source_turn_ids": ["t_0", "t_2"], "text": "Lives in Paris.",
                                "chunk_type": "explicit_state", "temporal_scope": "current",
                                "state_salience": 1.0, "evidence_strength": "strong",
                                "candidate_bucket_tracks": [{"bucket": PAIR.domain, "local_track": PAIR.slot}]}]}
        if phase == "delta_extractor":
            return {"deltas": [delta().to_dict()]}
        if phase == "update_judge":
            return {"decision": "REPLACE", "target_item_id": "alps_initial_00000", "reason": "Moved"}
        if phase == "bucket_bridge_proposer":
            return {"affected_buckets": [{"bucket": "routine_and_transport", "reason": "Move affects commute", "confidence": 0.8}]}
        if phase == "latent_invalidation_proposer":
            return {"proposals": []}
        if phase == "stale_support_judge":
            return {"link_type": "NONE"}
        raise AssertionError(f"Unexpected native phase: {phase}; {payload}")


class FakeRetriever:
    """Offline ranking double; no production fallback retrieval is introduced."""

    def rank(self, *, query_text, candidates, text_getter, top_k):
        return [{"score": 1.0, "item": item} for item in list(candidates)[:top_k]]


@pytest.mark.parametrize("update", [False, True])
def test_real_writer_one_session_no_gold_leakage_and_full_trace(monkeypatch, update):
    monkeypatch.setattr("cup_mem.pipeline.build_retriever", lambda *args, **kwargs: FakeRetriever())
    llm = ScriptedLLM(update)
    engine = CupMemEngine(llm=llm, embedding_model_path="test-only")
    original_write = engine.write_session
    writes = []

    def spy(**kwargs):
        writes.append(deepcopy(kwargs))
        return original_write(**kwargs)

    monkeypatch.setattr(engine, "write_session", spy)
    result = predict_example(parse_input(row()), engine=engine, classifier=SchemaClassifier(llm), formatter=metadata)
    assert len(writes) == 1 and len(writes[0]["session"]) == 3
    assert writes[0]["session_index"] == 1
    all_payloads = json.dumps(llm.calls)
    assert "SENTINEL" not in all_payloads
    assert result.trace["gold_result"] is None
    if update:
        assert result.prediction["memory_items"][0]["value"] == "Lives in Paris."
        for key in ("direct_revision_candidates", "affected_candidates", "global_candidates"):
            assert result.trace[key]
        assert result.trace["local_update_decisions"][0]["apply_result"]["applied_action"] == "REPLACE"
        assert result.trace["final_cupmem_memory"]["stale_archive"][0]["status"] == "STALE"
    # Changing gold/control values leaves every inference payload and prediction identical.
    changed = row()
    changed["gold"] = {"answer": [{"value": "different gold"}]}
    changed["input"]["metadata"] = {"controls": {"expectedAction": "different action"}}
    calls_before = list(llm.calls)
    llm.calls.clear()
    again = predict_example(parse_input(changed), engine=engine, classifier=SchemaClassifier(llm), formatter=metadata)
    assert again.prediction == result.prediction
    # Lane calls may finish in a different order because native CUPMEM uses threads.
    assert sorted(llm.calls) == sorted(calls_before)


def test_schema_mapping_uses_existing_llm_cache(tmp_path):
    count = []

    def create(**kwargs):
        count.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"domain": PAIR.domain, "slot": PAIR.slot})))], usage=None)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    llm = LLMClient(model="test", api_key="fake", base_url="https://example.invalid", cache_dir=tmp_path, openai_cls=FakeOpenAI)
    classifier = SchemaClassifier(llm)
    assert classifier("Lives in London") == classifier("Lives in London") == PAIR
    assert len(count) == 1
    assert llm.get_call_records()[-1]["cache_hit"] is True


def test_one_way_dependency():
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "cup_mem").rglob("*.py"))
    assert paths
    for path in paths:
        assert "import alpsbench" not in path.read_text()
        assert "from alpsbench" not in path.read_text()


@pytest.mark.skipif(not os.environ.get("ALPSBENCH_ROOT"), reason="Set ALPSBENCH_ROOT to test the native scorer")
def test_official_scorer_exact_fields_duplicates_and_validation():
    evaluator = NativeEvaluator(Path(os.environ["ALPSBENCH_ROOT"]))
    memory = row()["input"]["memory"][0]
    reference = {"benchmark_id": "x", "gold": {"answer": [memory]}}
    prediction = {"benchmark_id": "x", "memory_items": [deepcopy(memory)]}
    assert evaluator.evaluate([prediction], {"x": reference}, split="dev")["summary"]["mean_f1"] == 1
    prediction["memory_items"].append(deepcopy(memory))
    assert evaluator.evaluate([prediction], {"x": reference}, split="dev")["scores"][0]["precision"] == 0.5
    prediction["memory_items"] = [{**memory, "emotion": "different"}]
    assert evaluator.evaluate([prediction], {"x": reference}, split="dev")["summary"]["mean_f1"] == 0
    with pytest.raises(ValueError, match="No public"):
        evaluator.evaluate([prediction], {"x": reference}, split="test")
    with pytest.raises(ValueError, match="IDs"):
        evaluator.evaluate([prediction, prediction], {"x": reference}, split="dev")
    with pytest.raises(ValueError, match="exactly"):
        evaluator.evaluate([{**prediction, "action": "Retention"}], {"x": reference}, split="dev")
