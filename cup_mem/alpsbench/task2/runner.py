"""Run the unchanged CUPMEM writer once per independent AlpsBench example."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from cup_mem import CupMemEngine, LLMClient, TraceConfig
from cup_mem.memory.schema import schema_prompt_text

from .dataset import load_inputs, load_references, parse_input
from .evaluator import NativeEvaluator
from .logging_utils import build_trace, write_jsonl
from .memory_adapter import SchemaClassifier, initialize_memory
from .schema import PredictionResult, SchemaAssignment, Task2Input
from .output_adapter import LLMMetadataFormatter, MetadataFormatter, convert_output
from .session_adapter import build_session


def predict_example(
    example: Task2Input, *, engine: CupMemEngine,
    classifier: Callable[[str], SchemaAssignment], formatter: MetadataFormatter,
    session_time: str = "",
) -> PredictionResult:
    """Initialize, write exactly once, and convert without access to references.

    An engine can be reused to amortize loading the embedding model, but every
    example resets all its memory, evidence banks, and counters. Initial items
    have session index 0 and new_dialogue has index 1, preserving native temporal
    guards. No synthetic date is inferred from gold or from the wall clock.
    """
    engine.reset()
    initial = initialize_memory(example, classifier)
    engine.store = initial.store.clone()
    session_trace = engine.write_session(
        session=build_session(example), session_index=1, session_time=session_time)
    prediction, output_logs = convert_output(
        example, initial, engine.store.to_snapshot(), session_trace, formatter)
    trace = build_trace(example, initial, session_trace, prediction, output_logs)
    trace["configuration"] = {
        "thresholds": asdict(engine.thresholds),
        "schema_sha256": hashlib.sha256(schema_prompt_text().encode()).hexdigest(),
        "session_time": session_time,
    }
    return PredictionResult(prediction, trace)


def main() -> None:
    """Batch CLI or the upstream stdin/stdout prediction-program protocol.

    References are opened only after every prediction has been finalized and
    written. They can enrich trace artifacts and drive scoring, but cannot be
    passed to either adapter or the writer. Stdout is reserved for predictions
    in --stdin mode; progress messages go to stderr.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Task-2 model_input.jsonl or original source JSONL")
    source.add_argument("--stdin", action="store_true", help="Read one public input row and emit one prediction")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/alpsbench/cache"))
    parser.add_argument("--session-time", default="", help="Optional experiment timestamp; absent dates stay unknown")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--benchmark-root", type=Path, help="Official AlpsBench checkout, required for scoring")
    parser.add_argument("--split", choices=("examples", "dev", "validation", "test"), default="test")
    args = parser.parse_args()
    if args.references and (args.split == "test" or args.stdin):
        parser.error("References are only allowed for batch examples/dev/validation runs")
    if args.references and not args.benchmark_root:
        parser.error("--references requires --benchmark-root for native scoring")
    if args.input and args.input.resolve() in {
        (args.output_dir / name).resolve() for name in ("predictions.jsonl", "traces.jsonl", "evaluation.json")
    }:
        parser.error("Output would overwrite the input")
    if args.references and args.references.resolve().parent == args.output_dir.resolve():
        parser.error("Keep reference files separate from the output directory")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Set {args.api_key_env} for the LLM client")
    examples = [parse_input(json.load(sys.stdin))] if args.stdin else load_inputs(args.input)
    if args.stdin:
        # Upstream starts one process per example; avoid overwriting the previous
        # example's audit files when all processes use the same output directory.
        example_key = hashlib.sha256(examples[0].benchmark_id.encode()).hexdigest()
        args.output_dir = args.output_dir / example_key
    # CUPMEM keys its cache by model/messages. Namespace by endpoint as well to
    # avoid cross-provider reuse of identically named models, without changing it.
    endpoint_key = hashlib.sha256(args.base_url.encode()).hexdigest()[:16]
    llm = LLMClient(model=args.model, api_key=api_key, base_url=args.base_url,
                    cache_dir=args.cache_dir / endpoint_key)
    engine = CupMemEngine(llm=llm, embedding_model_path=args.embedding_model_path,
                          embedding_device=args.embedding_device,
                          trace_config=TraceConfig(enable_debug_trace=True))
    classifier, formatter = SchemaClassifier(llm), LLMMetadataFormatter(llm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, traces = [], []
    # Stream completed examples to disk; a later failure must not erase them.
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as pred_stream, \
         (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as trace_stream:
        for example in examples:
            llm.reset_usage_tracking()
            result = predict_example(example, engine=engine, classifier=classifier,
                                     formatter=formatter, session_time=args.session_time)
            result.trace["configuration"].update({
                "model": args.model, "base_url": args.base_url,
                "embedding_model_path": args.embedding_model_path,
                "embedding_device": args.embedding_device,
            })
            result.trace["llm_calls"] = llm.get_call_records()
            result.trace["llm_usage"] = llm.get_usage_summary()
            predictions.append(result.prediction)
            traces.append(result.trace)
            pred_stream.write(json.dumps(result.prediction, ensure_ascii=False, allow_nan=False) + "\n")
            trace_stream.write(json.dumps(result.trace, ensure_ascii=False, allow_nan=False) + "\n")
            pred_stream.flush()
            trace_stream.flush()
            print(f"Completed {example.benchmark_id}", file=sys.stderr)
    if args.references:
        references = load_references(args.references)
        evaluation = NativeEvaluator(args.benchmark_root).evaluate(predictions, references, split=args.split)
        for trace in traces:
            trace["gold_result"] = deepcopy(references[trace["example_id"]]["gold"])
        write_jsonl(args.output_dir / "traces.jsonl", traces)
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.stdin:
        print(json.dumps(predictions[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
