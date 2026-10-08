#!/usr/bin/env python3
"""Evaluate dense and statically pruned Llama-2 models on GSM8K.

This script reproduces the GSM8K protocol used for the ShortGPT paper's
Table 3 as closely as possible without requiring OpenCompass:

* the complete GSM8K test split (1,319 examples),
* OpenCompass's fixed four-shot chain-of-thought prompt,
* greedy generation with at most 512 new tokens, and
* OpenCompass's numeric answer extraction and exact-match accuracy (%).

The requested 23-layer checkpoint is a static ShortGPT model.  It should be
compared with the paper's ``ShortGPT`` result, not ``ShortGPT-gen``: the latter
keeps all layers and only skips selected layers while processing prompt tokens.

Example:
    python evaluate/gsm8k.py

Quick smoke test:
    python evaluate/gsm8k.py --limit 8 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_ORIGINAL_MODEL = "/acpl-ssd32/meta-llama/Llama-2-7b-hf"
DEFAULT_PRUNED_MODEL = "/acpl-ssd32/shortgpt/Llama2-7b-9layers"
DEFAULT_OUTPUT = Path("outputs/gsm8k_results.json")

# These are the four demonstrations in OpenCompass's gsm8k_gen config.  Some
# wording and arithmetic typos are intentionally preserved for protocol parity.
FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Angelo and Melanie want to plan how many hours over the next week "
        "they should study together for their test next week. They have 2 "
        "chapters of their textbook to study and 4 worksheets to memorize. "
        "They figure out that they should dedicate 3 hours to each chapter of "
        "their textbook and 1.5 hours for each worksheet.\nIf they plan to "
        "study no more than 4 hours each day, how many days should they plan "
        "to study total over the next week if they take a 10-minute break "
        "every hour, include 3 10-minute snack breaks each day, and 30 minutes "
        "for lunch each day?",
        "Angelo and Melanie think they should dedicate 3 hours to each of the "
        "2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets "
        "they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 "
        "worksheets = 6 hours total.\nAngelo and Melanie need to start with "
        "planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\n"
        "However, they need to include time for breaks and lunch.\nEvery hour "
        "they want to include a 10-minute break, so 12 total hours x 10 minutes "
        "= 120 extra minutes for breaks.\nThey also want to include 3 10-minute "
        "snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include "
        "30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes "
        "for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 "
        "minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan "
        "12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to "
        "study no more than 4 hours each day, 15 hours / 4 hours each day = "
        "3.75\nThey will need to plan to study 4 days to allow for all the time "
        "they need.\nThe answer is 4\n",
    ),
    (
        "Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 "
        "free throws.  Their opponents score double the 2 pointers but half "
        "the 3 pointers and free throws.  What's the total number of points "
        "scored by both teams added together?",
        "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points "
        "in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored "
        "8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free "
        "throws count as one point so they scored 10*1=10 points in free "
        "throws.\nAll together his team scored 50+24+10= 84 points\nMark's "
        "opponents scored double his team's number of 2 pointers, meaning they "
        "scored 50*2=100 points in 2 pointers.\nHis opponents scored half his "
        "team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 "
        "pointers.\nThey also scored half Mark's team's points in free throws, "
        "meaning they scored 10/2=5 points in free throws.\nAll together Mark's "
        "opponents scored 100+12+5=117 points\nThe total score for the game is "
        "both team's scores added together, so it is 84+117=201 points\nThe "
        "answer is 201\n",
    ),
    (
        "Bella has two times as many marbles as frisbees. She also has 20 more "
        "frisbees than deck cards. If she buys 2/5 times more of each item, "
        "what would be the total number of the items she will have if she "
        "currently has 60 marbles?",
        "When Bella buys 2/5 times more marbles, she'll have increased the "
        "number of marbles by 2/5*60 = 24\nThe total number of marbles she'll "
        "have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has "
        "two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\n"
        "If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more "
        "frisbees.\nThe total number of frisbees she'll have will increase to "
        "30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning "
        "she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, "
        "she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards "
        "she'll have is 10+4 = 14\nTogether, Bella will have a total of "
        "14+42+84 = 140 items\nThe answer is 140\n",
    ),
    (
        "A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 "
        "bananas in the first three baskets and 2 less of each fruit in the "
        "fourth basket. How many fruits are there?",
        "For the first three baskets, the number of apples and oranges in one "
        "basket is 9+15=24\nIn total, together with bananas, the number of "
        "fruits in one basket is 24+14=38 for the first three baskets.\nSince "
        "there are three baskets each having 38 fruits, there are 3*38=114 "
        "fruits in the first three baskets.\nThe number of apples in the fourth "
        "basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\n"
        "The combined number of oranges and apples in the fourth basket is "
        "13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, "
        "the fourth basket has 20+12=32 fruits.\nThe four baskets together have "
        "32+114=146 fruits.\nThe answer is 146\n",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dense and static ShortGPT checkpoints using the paper's "
            "four-shot GSM8K exact-match protocol."
        )
    )
    parser.add_argument("--original-model", default=DEFAULT_ORIGINAL_MODEL)
    parser.add_argument("--pruned-model", default=DEFAULT_PRUNED_MODEL)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path (default: --original-model for both checkpoints).",
    )
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples; omit for the full 1,319-example test set.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Model context length including generated tokens (default: 4096).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="OpenCompass GSM8K default is 512.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device map, for example "auto" or "balanced".',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model/dataset code.",
    )
    args = parser.parse_args()

    for name in ("batch_size", "max_seq_length", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_new_tokens >= args.max_seq_length:
        parser.error("--max-new-tokens must be smaller than --max-seq-length")
    return args


def load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from datasets import load_dataset
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError) as exc:
        dependency = getattr(exc, "name", None) or type(exc).__name__
        raise SystemExit(
            f"Could not load dependency {dependency}: {exc}. Install a working "
            "torch, transformers, accelerate, datasets, safetensors, "
            "sentencepiece, and tqdm environment."
        ) from exc
    return torch, load_dataset, tqdm, (AutoModelForCausalLM, AutoTokenizer)


def build_prompt(question: str) -> str:
    """Render OpenCompass's role-less four-shot prompt exactly."""
    messages: list[str] = []
    for example_question, example_answer in FEW_SHOT_EXAMPLES:
        messages.append(
            f"Question: {example_question}\nLet's think step by step\nAnswer:"
        )
        messages.append(example_answer)
    messages.append(f"Question: {question}\nLet's think step by step\nAnswer:")

    # OpenCompass's LMTemplateParser inserts one newline between role messages.
    # Demonstration answers already end in a newline, yielding a blank line
    # between examples, just as in the original config.
    return "\n".join(messages)


def extract_prediction(text: str) -> str:
    """OpenCompass gsm8k_postprocess used for the paper-style score."""
    text = text.split("Question:", maxsplit=1)[0]
    numbers = re.findall(r"\-?\d+\.\d+|\-?\d+", text)
    return numbers[-1] if numbers else "NULL"


def extract_reference(answer: str) -> str:
    """OpenCompass gsm8k_dataset_postprocess for GSM8K reference answers."""
    marker = "#### "
    if marker not in answer:
        raise ValueError(f"GSM8K answer does not contain {marker!r}: {answer!r}")
    return answer.split(marker, maxsplit=1)[1].replace(",", "").strip()


def answers_equal(prediction: str, reference: str) -> bool:
    """Match OpenCompass's Gsm8kEvaluator numeric equality behavior."""
    try:
        return prediction == reference or abs(float(prediction) - int(reference)) < 1e-6
    except (TypeError, ValueError, OverflowError):
        return False


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.resolve()) if path.exists() else model_path


def load_gsm8k(
    *,
    load_dataset: Any,
    dataset_name: str,
    dataset_config: str,
    split: str,
    limit: int | None,
    trust_remote_code: bool,
) -> list[dict[str, str]]:
    print(f"Loading {dataset_name} ({dataset_config}, split={split})")
    dataset = load_dataset(
        dataset_name,
        dataset_config or None,
        split=split,
        trust_remote_code=trust_remote_code,
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples: list[dict[str, str]] = []
    for index, row in enumerate(dataset):
        if "question" not in row or "answer" not in row:
            raise KeyError(
                "GSM8K rows must contain 'question' and 'answer'; "
                f"row {index} has {sorted(row)}"
            )
        examples.append(
            {
                "index": index,
                "question": str(row["question"]),
                "reference": extract_reference(str(row["answer"])),
            }
        )
    if not examples:
        raise RuntimeError("The selected GSM8K split is empty")
    return examples


def evaluate_model(
    model_path: str,
    examples: list[dict[str, str]],
    *,
    tokenizer: Any,
    auto_model: Any,
    torch: Any,
    tqdm: Any,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    print(f"\nLoading model: {model_path}")
    dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    model = auto_model.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.generation_config.do_sample = False
    # Llama-2's bundled generation_config enables sampling.  Clear its sampling
    # knobs as well as disabling sampling so recent Transformers versions do not
    # emit misleading warnings during greedy generation.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    input_device = model.get_input_embeddings().weight.device
    if input_device.type == "meta":
        raise RuntimeError(
            "Could not determine the input device from the sharded model. "
            "Try a different --device-map."
        )

    predictions: list[dict[str, Any]] = []
    correct = 0
    started_at = time.perf_counter()
    max_input_tokens = max_seq_length - max_new_tokens
    batches = batched(examples, batch_size)
    num_batches = (len(examples) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch in tqdm(batches, total=num_batches, desc=Path(model_path).name):
            prompts = [build_prompt(item["question"]) for item in batch]
            encoded = tokenizer(
                prompts,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            too_long = [length for length in prompt_lengths if length > max_input_tokens]
            if too_long:
                raise RuntimeError(
                    f"Prompt has {max(too_long)} tokens, but only {max_input_tokens} "
                    "input tokens fit. Increase --max-seq-length or reduce "
                    "--max-new-tokens; automatic truncation is disabled to preserve "
                    "the paper's four-shot prompt."
                )

            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            input_width = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            continuations = tokenizer.batch_decode(
                generated[:, input_width:], skip_special_tokens=True
            )

            for item, prompt_tokens, text in zip(batch, prompt_lengths, continuations):
                extracted = extract_prediction(text)
                is_correct = answers_equal(extracted, item["reference"])
                correct += int(is_correct)
                predictions.append(
                    {
                        "index": item["index"],
                        "question": item["question"],
                        "reference": item["reference"],
                        "prediction": text,
                        "extracted_answer": extracted,
                        "correct": is_correct,
                        "prompt_tokens": int(prompt_tokens),
                    }
                )

    elapsed = time.perf_counter() - started_at
    accuracy = 100.0 * correct / len(examples)
    result = {
        "model_path": resolve_model_path(model_path),
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "elapsed_seconds": elapsed,
        "predictions": predictions,
    }
    print(
        f"Accuracy={accuracy:.2f}% ({correct}/{len(examples)}), "
        f"time={elapsed:.1f}s"
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer_path = args.tokenizer or args.original_model
    tokenizer = auto_tokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_gsm8k(
        load_dataset=load_dataset,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Prepared {len(examples)} GSM8K examples")

    model_results = []
    for model_path in (args.original_model, args.pruned_model):
        model_results.append(
            evaluate_model(
                model_path,
                examples,
                tokenizer=tokenizer,
                auto_model=auto_model,
                torch=torch,
                tqdm=tqdm,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens,
                dtype_name=args.dtype,
                device_map=args.device_map,
                trust_remote_code=args.trust_remote_code,
            )
        )

    dense_accuracy = model_results[0]["accuracy"]
    pruned_accuracy = model_results[1]["accuracy"]
    retention = (
        100.0 * pruned_accuracy / dense_accuracy if dense_accuracy > 0 else None
    )
    result = {
        "protocol": {
            "benchmark": "GSM8K",
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "num_examples": len(examples),
            "prompt": "OpenCompass fixed 4-shot chain-of-thought",
            "generation": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "seed": args.seed,
            },
            "metric": "OpenCompass numeric exact-match accuracy (%)",
            "tokenizer_path": resolve_model_path(tokenizer_path),
            "variant": "dense vs static ShortGPT checkpoint (not ShortGPT-gen)",
            "paper_table_3_reference": {
                "llama2_7b_dense": 15.69,
                "llama2_7b_shortgpt_static": 1.60,
                "llama2_7b_shortgpt_gen": 14.71,
            },
        },
        "models": model_results,
        "comparison": {
            "absolute_accuracy_change_points": pruned_accuracy - dense_accuracy,
            "relative_performance_retention_percent": retention,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved results to {args.output}")
    print(
        f"Dense {dense_accuracy:.2f}% -> pruned {pruned_accuracy:.2f}% "
        f"(change {pruned_accuracy - dense_accuracy:+.2f} points)"
    )
    if retention is not None:
        print(f"Relative performance retention: {retention:.2f}%")


if __name__ == "__main__":
    main()
