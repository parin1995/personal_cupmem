"""Read released public JSONL or original Hugging Face Task-2 rows.

The public input's metadata can include expectedAction and selected_memory_id.
An allowlisted projection, rather than removal of known gold key names, keeps
all such annotations out of initialization, CUPMEM, and output classification.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from .models import JsonObject, Task2Input


def read_jsonl(path: Path) -> Iterator[JsonObject]:
    """Read objects with useful line errors, including unhydrated LFS files."""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON (hydrate Git LFS files first)") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def _objects(value: Any, field: str) -> list[JsonObject]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be an array of objects")
    return value


def parse_input(row: JsonObject) -> Task2Input:
    """Project input data without forwarding gold, query prompts, or controls.

    Supports the public ``input.memory/new_dialogue`` contract and the original
    ``entry``/``record`` release layout used by upstream's data builder. Gold in
    a combined source row is ignored; references must be loaded separately.
    """
    if "input" in row:
        if row.get("task", "task2") != "task2":
            raise ValueError("Expected a Task-2 row")
        payload = row["input"]
        if not isinstance(payload, dict):
            raise ValueError("input must be an object")
        benchmark_id = row.get("benchmark_id")
        session_id = row.get("session_id", benchmark_id)
        memories = payload.get("memory")
        dialogue = payload.get("new_dialogue")
    else:
        source = row.get("entry", row)
        if not isinstance(source, dict) or not isinstance(source.get("record"), dict):
            raise ValueError("Expected public input or original Task-2 record")
        metadata = source.get("metadata", {})
        session_id = metadata.get("session_id") if isinstance(metadata, dict) else None
        benchmark_id = session_id
        memories = source.get("memory")
        dialogue = source["record"].get("new_dialogue")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("A nonempty benchmark_id/session_id is required")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A nonempty session_id is required")
    memories = _objects(memories, "memory")
    for memory in memories:
        if not isinstance(memory.get("value"), str) or not memory["value"].strip():
            raise ValueError("Each initial memory needs a nonempty string value")
    turns = []
    for turn in _objects(dialogue, "new_dialogue"):
        role = turn.get("role")
        text = turn.get("text", turn.get("content"))
        if role not in {"user", "assistant", "system"} or not isinstance(text, str):
            raise ValueError("Each dialogue turn needs a valid role and string text/content")
        if "text" in turn and "content" in turn and turn["text"] != turn["content"]:
            raise ValueError("Conflicting text/content in dialogue turn")
        # No turn-level annotations are allowed to travel with the conversation.
        turns.append({"role": role, "text": text})
    return Task2Input(benchmark_id, session_id, tuple(deepcopy(memories)), tuple(turns))


def load_inputs(path: Path) -> list[Task2Input]:
    """Load one independent initial store per unique example ID."""
    result = [parse_input(row) for row in read_jsonl(path)]
    if len({item.benchmark_id for item in result}) != len(result):
        raise ValueError("Duplicate benchmark_id in inputs")
    return result


def load_references(path: Path) -> dict[str, JsonObject]:
    """Load public reference rows for post-prediction evaluation only."""
    result = {}
    for row in read_jsonl(path):
        identifier = row.get("benchmark_id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("Missing or duplicate reference benchmark_id")
        if not isinstance(row.get("gold"), dict):
            raise ValueError("Reference requires gold.answer")
        _objects(row["gold"].get("answer"), "gold.answer")
        result[identifier] = row
    return result
