#!/usr/bin/env python3
"""Compare dense and pruned causal LMs using the ShortGPT PG19 protocol.

The ShortGPT paper samples ten random 1,000-token segments per PG19
document and reports an ablation using 100 instances.  The defaults below use
ten documents, yielding 100 shared segments for a reproducible comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

# Llama2-7b
# DEFAULT_ORIGINAL_MODEL = "/acpl-ssd32/meta-llama/Llama-2-7b-hf"
# DEFAULT_PRUNED_MODEL = (
#     "/acpl-ssd32/shortgpt/Llama2-7b-9layers"
# )

# Gemma-7b
DEFAULT_ORIGINAL_MODEL = "/acpl-ssd32/google/gemma-7b"
DEFAULT_PRUNED_MODEL = (
    "/acpl-ssd30/shortgpt/gemma-7b-6layers"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PPL of the original and pruned models on shared PG19 segments."
    )
    parser.add_argument("--original-model", default=DEFAULT_ORIGINAL_MODEL)
    parser.add_argument("--pruned-model", default=DEFAULT_PRUNED_MODEL)
    parser.add_argument("--dataset", default="emozilla/pg19")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--num-documents",
        type=int,
        default=10,
        help="Number of PG19 documents to use (default: 10).",
    )
    parser.add_argument(
        "--segments-per-document",
        type=int,
        default=10,
        help="Random segments sampled from each document (default: 10).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1000,
        help="Tokens in each segment, including the first conditioning token.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device map, for example "auto" or "balanced".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ppl_results.json"),
    )
    args = parser.parse_args()

    for name in (
        "num_documents",
        "segments_per_document",
        "sequence_length",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2 for next-token loss")
    return args


def load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from datasets import load_dataset
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install torch, transformers, "
            "accelerate, datasets, safetensors, and tqdm."
        ) from exc
    return torch, load_dataset, tqdm, (AutoModelForCausalLM, AutoTokenizer)


def sample_segments(
    dataset: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    text_column: str,
    num_documents: int,
    segments_per_document: int,
    sequence_length: int,
    seed: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Tokenize PG19 documents and sample deterministic, non-repeated starts."""
    rng = random.Random(seed)
    segments: list[list[int]] = []
    sample_manifest: list[dict[str, Any]] = []
    used_documents = 0

    for source_index, example in enumerate(dataset):
        if text_column not in example:
            available = ", ".join(sorted(example))
            raise KeyError(
                f"Column {text_column!r} is absent from the dataset; found: {available}"
            )

        token_ids = tokenizer(
            example[text_column], add_special_tokens=False, truncation=False
        )["input_ids"]
        max_start = len(token_ids) - sequence_length
        if max_start < 0:
            continue

        population_size = max_start + 1
        if population_size < segments_per_document:
            starts = [
                rng.randrange(population_size)
                for _ in range(segments_per_document)
            ]
        else:
            starts = rng.sample(range(population_size), segments_per_document)

        for start in starts:
            segment = token_ids[start : start + sequence_length]
            if len(segment) != sequence_length:
                raise RuntimeError("Internal error: sampled a truncated segment")
            segments.append(segment)
            sample_manifest.append(
                {
                    "source_index": source_index,
                    "start_token": start,
                    "end_token": start + sequence_length,
                }
            )

        used_documents += 1
        print(
            f"Sampled document {used_documents}/{num_documents} "
            f"(dataset index {source_index}, {len(token_ids):,} tokens)"
        )
        if used_documents == num_documents:
            break

    if used_documents < num_documents:
        raise RuntimeError(
            f"Only {used_documents} documents contained at least {sequence_length} tokens; "
            f"requested {num_documents}."
        )
    return segments, sample_manifest


def batched(items: list[list[int]], batch_size: int) -> Iterable[list[list[int]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_model(
    model_path: str,
    segments: list[list[int]],
    *,
    batch_size: int,
    dtype_name: str,
    device_map: str,
    torch: Any,
    tqdm: Any,
    auto_model: Any,
) -> dict[str, Any]:
    dtype = getattr(torch, dtype_name)
    print(f"\nLoading {model_path}")
    model = auto_model.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    total_nll = 0.0
    total_predicted_tokens = 0
    started_at = time.perf_counter()
    batches = batched(segments, batch_size)
    num_batches = math.ceil(len(segments) / batch_size)

    with torch.inference_mode():
        for batch in tqdm(batches, total=num_batches, desc=Path(model_path).name):
            input_ids = torch.tensor(batch, dtype=torch.long, device=input_device)
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)

            # CausalLM loss predicts positions 1..L-1. All segments have equal
            # length and no padding, so every item contributes L-1 targets.
            predicted_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
            total_nll += outputs.loss.float().item() * predicted_tokens
            total_predicted_tokens += predicted_tokens

    mean_nll = total_nll / total_predicted_tokens
    result = {
        "model_path": str(Path(model_path).resolve()),
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "predicted_tokens": total_predicted_tokens,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    print(
        f"PPL={result['perplexity']:.4f}, NLL={mean_nll:.6f}, "
        f"tokens={total_predicted_tokens:,}, time={result['elapsed_seconds']:.1f}s"
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    torch, load_dataset, tqdm, auto_classes = load_dependencies()
    auto_model, auto_tokenizer = auto_classes

    tokenizer = auto_tokenizer.from_pretrained(args.original_model, use_fast=True)
    print(f"Loading {args.dataset} ({args.split}, streaming=True)")
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    segments, manifest = sample_segments(
        dataset,
        tokenizer,
        text_column=args.text_column,
        num_documents=args.num_documents,
        segments_per_document=args.segments_per_document,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    print(f"Prepared {len(segments)} shared segments")

    model_results = []
    for model_path in (args.original_model, args.pruned_model):
        model_results.append(
            evaluate_model(
                model_path,
                segments,
                batch_size=args.batch_size,
                dtype_name=args.dtype,
                device_map=args.device_map,
                torch=torch,
                tqdm=tqdm,
                auto_model=auto_model,
            )
        )

    dense_ppl = model_results[0]["perplexity"]
    pruned_ppl = model_results[1]["perplexity"]
    result = {
        "protocol": {
            "dataset": args.dataset,
            "split": args.split,
            "num_documents": args.num_documents,
            "segments_per_document": args.segments_per_document,
            "sequence_length": args.sequence_length,
            "num_segments": len(segments),
            "seed": args.seed,
            "add_special_tokens": False,
            "aggregation": "token-weighted_nll",
        },
        "models": model_results,
        "comparison": {
            "absolute_ppl_increase": pruned_ppl - dense_ppl,
            "relative_ppl_increase_percent": (pruned_ppl / dense_ppl - 1.0) * 100.0,
        },
        "sample_manifest": manifest,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved results to {args.output}")
    print(
        f"Dense {dense_ppl:.4f} -> pruned {pruned_ppl:.4f} "
        f"({result['comparison']['relative_ppl_increase_percent']:+.2f}%)"
    )


if __name__ == "__main__":
    main()
