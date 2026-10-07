"""Explicit boundaries between model-visible inputs and evaluation references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

JsonObject = dict[str, Any]


class JsonClient(Protocol):
    """The small portion of CUPMEM's LLM client used by boundary adapters."""

    def call_json(self, system_prompt: str, user_payload: str, **kwargs: Any) -> JsonObject:
        """Return a JSON object, using the client's normal response cache."""
        ...


@dataclass(frozen=True)
class Task2Input:
    """Inference-only projection; deliberately has no gold or control fields.

    ``initial_memories`` retains the original objects for lossless output and
    audit. Only their values are sent to the initial schema classifier. No
    old dialogue is replayed: those observations already exist in memory.
    """

    benchmark_id: str
    session_id: str
    initial_memories: tuple[JsonObject, ...]
    new_dialogue: tuple[JsonObject, ...]


@dataclass(frozen=True)
class SchemaAssignment:
    """Adapter names for CUPMEM's unchanged bucket/local_track schema."""

    domain: str
    slot: str


@dataclass
class PredictionResult:
    """A submission row and a separate, richer diagnostic trace."""

    prediction: JsonObject
    trace: JsonObject
