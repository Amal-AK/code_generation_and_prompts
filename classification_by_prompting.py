#!/usr/bin/env python3
"""
Zero-shot 4-class prompt classifier using GPT-5-mini via OpenAI Batch API.

Reproduces the exact test split from train_lora_classifier.py (seed=42, val_split=0.2)
and classifies each prompt as LV / SF / US / CLEAN.

Two modes:
  --mode batch   (default): submit all requests via OpenAI Batch API (async, ~50% cheaper)
  --mode online:            send requests in parallel via chat completions (faster turnaround)

Usage:
    export OPENAI_API_KEY=sk-...
    python classification_by_prompting.py --data_dir .
    python classification_by_prompting.py --data_dir . --mode online --max_workers 16
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import OpenAI
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm

# ── Labels (must match train_lora_classifier.py exactly) ─────────────────────

LABEL2ID = {"LV": 0, "SF": 1, "US": 2, "CLEAN": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 4

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code-benchmark quality auditor. Classify a coding problem prompt into one of:

  LV    – The prompt uses vague or imprecise wording.
  SF    – The prompt contains syntax or formatting errors.
  US    – The prompt is missing a constraint or condition.
  CLEAN – The prompt is complete and well-formed.

Reply with ONLY the label: LV, SF, US, or CLEAN. No explanation.
"""

USER_TEMPLATE = "Classify the following coding problem prompt:\n\n{prompt}"

# ── Data loading (identical to train_lora_classifier.py) ─────────────────────


def load_mutated(path: str, label: str) -> List[Tuple[str, int]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("applicable", True):
                continue
            text = obj.get("mutated_prompt", "").strip()
            if text:
                records.append((text, LABEL2ID[label]))
    return records


def load_clean(path: str, text_field: str) -> List[Tuple[str, int]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "").strip()
            if text:
                records.append((text, LABEL2ID["CLEAN"]))
    return records


def load_all_data(data_dir: str) -> List[Tuple[str, int]]:
    data_dir = Path(data_dir)
    all_records: List[Tuple[str, int]] = []

    mutation_files = [
        (data_dir / "mutations/humanEval_lv_with_tests.jsonl",     "LV"),
        (data_dir / "mutations/humanEval_SF_with_tests.jsonl",     "SF"),
        (data_dir / "mutations/HumanEval_US_with_tests.jsonl",     "US"),
        (data_dir / "mutations/mbpp_LV_with_tests.jsonl",          "LV"),
        (data_dir / "mutations/mbpp_SF_with_tests.jsonl",          "SF"),
        (data_dir / "mutations/mbpp_US_with_tests.jsonl",          "US"),
        (data_dir / "mutations/livecodebench_LV_with_tests.jsonl", "LV"),
        (data_dir / "mutations/livecodebench_SF_with_tests.jsonl", "SF"),
        (data_dir / "mutations/livecodebench_US_with_tests.jsonl", "US"),
    ]

    for path, label in mutation_files:
        if path.exists():
            records = load_mutated(str(path), label)
            print(f"  {path.name}: {len(records):5d} {label}")
            all_records.extend(records)
        else:
            print(f"  {path.name}: not found, skipping")

    clean_files = [
        (data_dir / "datasets/humanEval/HumanEval.jsonl",                "prompt"),
        (data_dir / "datasets/mbpp/mbpp.jsonl",                          "text"),
        (data_dir / "datasets/livecodebench/livecodebench_public.jsonl", "prompt"),
    ]

    for path, field in clean_files:
        if path.exists():
            records = load_clean(str(path), field)
            print(f"  {path.name}: {len(records):5d} CLEAN")
            all_records.extend(records)
        else:
            print(f"  {path.name}: not found, skipping")

    return all_records


# ── Request helpers ───────────────────────────────────────────────────────────

def make_messages(text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(prompt=text)},
    ]


def parse_label(raw: str) -> str | None:
    raw = raw.strip().upper()
    for label in ("LV", "SF", "US", "CLEAN"):
        if raw.startswith(label):
            return label
    return None


# ── Batch API mode ────────────────────────────────────────────────────────────

def run_batch(client: OpenAI, model: str, val_records: List[Tuple[str, int]],
              output_dir: Path, poll_interval: int = 30) -> dict[int, str]:
    """Submit via OpenAI Batch API, poll until done, return {index: pred_label}."""
    batch_input_path = output_dir / "batch_input.jsonl"
    batch_output_path = output_dir / "batch_output.jsonl"

    # Build batch input file
    print("Building batch input file...")
    with open(batch_input_path, "w") as f:
        for i, (text, _) in enumerate(val_records):
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": make_messages(text),
                },
            }
            f.write(json.dumps(request) + "\n")
    print(f"  Written {len(val_records)} requests to {batch_input_path}")

    # Upload input file
    print("Uploading batch input file...")
    with open(batch_input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {uploaded.id}")

    # Create batch
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch ID: {batch.id}  Status: {batch.status}")

    # Save batch ID for recovery
    meta_path = output_dir / "batch_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"batch_id": batch.id, "input_file_id": uploaded.id}, f, indent=2)
    print(f"  Batch metadata saved to {meta_path}")

    # Poll until complete
    print("Polling batch status...")
    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"  Status: {batch.status}  "
              f"completed={counts.completed}/{counts.total}  failed={counts.failed}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status: {batch.status}")

    if batch.request_counts.failed > 0 and batch.request_counts.completed == 0:
        error_file_id = getattr(batch, "error_file_id", None)
        sample = ""
        if error_file_id:
            import json as _json
            raw = client.files.content(error_file_id).content.decode()
            first = next((l for l in raw.splitlines() if l.strip()), "")
            if first:
                obj = _json.loads(first)
                sample = _json.dumps(obj.get("response", obj), indent=2)[:500]
        raise RuntimeError(
            f"All {batch.request_counts.failed} requests failed.\n"
            f"error_file_id: {error_file_id}\nSample error:\n{sample}"
        )

    # Download output
    print("Downloading batch output...")
    content = client.files.content(batch.output_file_id)
    with open(batch_output_path, "wb") as f:
        f.write(content.content)
    print(f"  Saved to {batch_output_path}")

    # Parse results
    results: dict[int, str] = {}
    with open(batch_output_path) as f:
        for line in f:
            obj = json.loads(line)
            idx = int(obj["custom_id"])
            raw = obj["response"]["body"]["choices"][0]["message"]["content"]
            label = parse_label(raw)
            results[idx] = label if label else raw  # keep raw for error reporting
    return results


# ── Online (parallel) mode ────────────────────────────────────────────────────

def classify_one(client: OpenAI, model: str, text: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=make_messages(text),
            )
            raw = resp.choices[0].message.content
            label = parse_label(raw)
            return label if label else raw.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def run_online(client: OpenAI, model: str, val_records: List[Tuple[str, int]],
               output_path: Path, max_workers: int) -> dict[int, str]:
    """Classify using parallel chat completions. Appends to output_path."""
    # Resume: load already-done
    results: dict[int, str] = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                results[obj["index"]] = obj["predicted"]
        print(f"Resuming: {len(results)} done, {len(val_records) - len(results)} remaining.")

    todo = [(i, text, true_label)
            for i, (text, true_label) in enumerate(val_records)
            if i not in results]

    def _classify(args_tuple):
        i, text, true_label = args_tuple
        pred = classify_one(client, model, text)
        return i, pred

    with open(output_path, "a") as out_f, \
         ThreadPoolExecutor(max_workers=max_workers) as pool, \
         tqdm(total=len(todo), desc="Classifying") as pbar:

        futures = {pool.submit(_classify, item): item for item in todo}
        for fut in as_completed(futures):
            i, pred = fut.result()
            out_f.write(json.dumps({"index": i, "predicted": pred}) + "\n")
            out_f.flush()
            results[i] = pred
            pbar.update(1)

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(val_records: List[Tuple[str, int]], results: dict[int, str], model: str):
    true_labels, pred_labels = [], []
    n_invalid = 0
    for i, (_, true_label) in enumerate(val_records):
        pred_str = results.get(i)
        if pred_str not in LABEL2ID:
            n_invalid += 1
            continue
        true_labels.append(true_label)
        pred_labels.append(LABEL2ID[pred_str])

    if n_invalid:
        print(f"\nWARNING: {n_invalid} unparseable predictions excluded from metrics.")

    true_np = np.array(true_labels)
    pred_np = np.array(pred_labels)

    mcc         = matthews_corrcoef(true_np, pred_np)
    f1_macro    = f1_score(true_np, pred_np, average="macro")
    f1_weighted = f1_score(true_np, pred_np, average="weighted")

    print("\n" + "=" * 60)
    print(f"Model: {model}")
    print(f"\nMCC:        {mcc:.4f}")
    print(f"F1 macro:   {f1_macro:.4f}")
    print(f"F1 weighted:{f1_weighted:.4f}")

    print("\nClassification report:")
    print(classification_report(true_np, pred_np,
                                target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)]))

    print("Confusion matrix:")
    cm = confusion_matrix(true_np, pred_np)
    header = "       " + "  ".join(f"{ID2LABEL[i]:>5}" for i in range(NUM_CLASSES))
    print(header)
    for i, row in enumerate(cm):
        print(f"  {ID2LABEL[i]:5} " + "  ".join(f"{v:5d}" for v in row))

    return {
        "model": model,
        "n_test": len(val_records),
        "n_invalid": n_invalid,
        "mcc": round(mcc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=".",                       help="Root dir containing JSONL files")
    parser.add_argument("--output_dir",  default="./gpt_classifier_output", help="Where to save results")
    parser.add_argument("--model",       default="gpt-5-mini",              help="OpenAI model name")
    parser.add_argument("--mode",        default="batch", choices=["batch", "online"],
                        help="'batch' uses OpenAI Batch API (cheap, async); 'online' uses parallel chat completions")
    parser.add_argument("--val_split",   type=float, default=0.2,           help="Must match train_lora_classifier.py")
    parser.add_argument("--seed",        type=int,   default=42,            help="Must match train_lora_classifier.py")
    parser.add_argument("--max_workers", type=int,   default=16,            help="Parallel workers (online mode only)")
    parser.add_argument("--poll_interval", type=int, default=30,            help="Seconds between batch status polls")
    parser.add_argument("--batch_id",   default=None,
                        help="Resume an existing batch job by ID (batch mode only; skips submission)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproduce exact test split ────────────────────────────────────────────
    print("\nLoading data...")
    all_records = load_all_data(args.data_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    random.shuffle(all_records)

    n_val = int(len(all_records) * args.val_split)
    val_records = all_records[:n_val]
    print(f"\nTotal: {len(all_records)}  |  Test split (first {args.val_split*100:.0f}%): {len(val_records)}")
    dist = Counter(ID2LABEL[label] for _, label in val_records)
    print("Test set distribution: " + "  ".join(f"{k}={dist[k]}" for k in ["LV", "SF", "US", "CLEAN"]))

    # ── Resume existing batch by ID ───────────────────────────────────────────
    if args.batch_id:
        print(f"\nResuming batch {args.batch_id}...")
        batch = client.batches.retrieve(args.batch_id)
        while batch.status not in ("completed", "failed", "cancelled", "expired"):
            time.sleep(args.poll_interval)
            batch = client.batches.retrieve(args.batch_id)
            counts = batch.request_counts
            print(f"  Status: {batch.status}  completed={counts.completed}/{counts.total}")
        if batch.status != "completed":
            raise RuntimeError(f"Batch ended with status: {batch.status}")
        content = client.files.content(batch.output_file_id)
        batch_output_path = output_dir / "batch_output.jsonl"
        with open(batch_output_path, "wb") as f:
            f.write(content.content)
        results: dict[int, str] = {}
        with open(batch_output_path) as f:
            for line in f:
                obj = json.loads(line)
                idx = int(obj["custom_id"])
                raw = obj["response"]["body"]["choices"][0]["message"]["content"]
                label = parse_label(raw)
                results[idx] = label if label else raw

    elif args.mode == "batch":
        results = run_batch(client, args.model, val_records, output_dir, args.poll_interval)

    else:  # online
        online_output = output_dir / "gpt_results.jsonl"
        results = run_online(client, args.model, val_records, online_output, args.max_workers)

    # ── Evaluate & save ───────────────────────────────────────────────────────
    summary = evaluate(val_records, results, args.model)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
