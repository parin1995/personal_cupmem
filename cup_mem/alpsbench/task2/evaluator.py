"""Use the upstream public Task-2 scorer directly; do not approximate it.

The inspected release compares canonicalized memory multisets, not action
labels. Its local proxy must not be reported as the hidden benchmark judge.
An explicit checkout keeps the scorer auditable and avoids a copied fork.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from .dataset import load_references, read_jsonl
from .schema import JsonObject

INSPECTED_REVISION = "9930dc7c86aefc17b6833838b87453368807789c"


class NativeEvaluator:
    """Load metrics.py from an explicit official AlpsBench source checkout."""

    def __init__(self, benchmark_root: Path):
        path = benchmark_root.resolve() / "src" / "benchmark" / "metrics.py"
        if not path.is_file():
            raise ValueError(f"Missing native AlpsBench scorer: {path}")
        spec = importlib.util.spec_from_file_location("_alpsbench_native_metrics", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        self.metrics: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.metrics)
        self.provenance = {"kind": "upstream_public_local_proxy", "source": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "adapter_inspected_revision": INSPECTED_REVISION}

    def evaluate(
        self, predictions: list[JsonObject], references: dict[str, JsonObject],
        *, split: str,
    ) -> JsonObject:
        """Validate IDs and invoke native per-row scoring and aggregation.

        Test references are explicitly rejected. Missing or duplicate IDs fail
        rather than silently dropping examples or inflating a partial score.
        """
        if not self.metrics.supports_local_scoring(split):
            raise ValueError(f"No public local scoring for split {split!r}")
        identifiers = [row.get("benchmark_id") for row in predictions]
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError("Predictions require nonempty string benchmark_id")
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(references):
            raise ValueError("Prediction/reference IDs must match exactly and be unique")
        scores = []
        for prediction in predictions:
            if set(prediction) != {"benchmark_id", "memory_items"}:
                raise ValueError("Task-2 prediction requires exactly benchmark_id and memory_items")
            items = prediction["memory_items"]
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise ValueError("memory_items must be an array of objects")
            reference = references[prediction["benchmark_id"]]
            if reference.get("benchmark_id") != prediction["benchmark_id"]:
                raise ValueError("Reference object ID does not match its index")
            gold = reference.get("gold", {}).get("answer")
            if not isinstance(gold, list) or any(not isinstance(item, dict) for item in gold):
                raise ValueError("Reference requires gold.answer array of objects")
            score = self.metrics.score_prediction_row(task="task2", prediction=prediction, reference=reference)
            scores.append({"benchmark_id": prediction["benchmark_id"], **score})
        return {"provenance": self.provenance, "split": split, "scores": scores,
                "summary": self.metrics.summarize_score_rows(task="task2", score_rows=scores)}


def main() -> None:
    """Score existing predictions without initializing any LLM or CUPMEM engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--split", choices=("examples", "dev", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = NativeEvaluator(args.benchmark_root).evaluate(
        list(read_jsonl(args.predictions)), load_references(args.references), split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
