#!/usr/bin/env python3
"""
eval_lora_lcb.py
────────────────
Evaluate a LoRA-finetuned model on LiveCodeBench LV mutations to test
generalization beyond the HumanEval / MBPP training distribution.

Usage:
  python eval_lora_lcb.py \
      --baseModel  deepseek-ai/deepseek-coder-6.7b-instruct \
      --loraAdapter ./finetune_deepseek_lv_output/best_lora_sft \
      --inputFile   ./mutations/livecodebench_LV_with_tests.jsonl \
      --outputDir   ./results/lora_lcb_eval \
      --gpus        0,1,2,3 \
      --limit       200

Compares adapter vs baseline (no adapter) on the same samples.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import trange

# ── re-use evaluation utilities from main_inference.py ───────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from main_inference import (
    generate_response,
    extract_code,
    evaluate_lcb_with_timeout,
    load_records,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_lora_lcb")


# ── model loader ─────────────────────────────────────────────────────────────

def load_model(model_name: str, lora_adapter: str | None, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if lora_adapter:
        logger.info("Loading LoRA adapter: %s", lora_adapter)
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()
        logger.info("Adapter merged into base model")

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=False,
    )
    return model, tokenizer, gen


# ── single-pass eval loop ─────────────────────────────────────────────────────

def run_eval(records, gen, tokenizer, model_name: str, label: str, timeout: int):
    total = exec_ok = pass1 = 0
    results = []

    pbar = trange(len(records), desc=label, ncols=90)
    for idx in pbar:
        row = records[idx]
        task_id = row.get("task_id", str(idx))
        pbar.set_description(f"{label} | {task_id}")

        row_prompt = (
            row.get("mutated_prompt")
            or row.get("original_prompt")
            or row.get("prompt")
            or ""
        ).strip()

        raw_tests = row.get("test", "[]")
        try:
            test_cases = json.loads(raw_tests) if isinstance(raw_tests, str) else raw_tests
        except Exception:
            test_cases = []

        prompt = textwrap.dedent(f"""
            You are a senior competitive programmer.

            Task:
            {row_prompt}

            Write a **complete Python program** that reads all input from stdin and writes the answer to stdout.
            Use only the Python standard library. Place all `import` statements at the very top.

            ⚠️ Return *only* valid Python code in a single code block:
            ```python
            <your code here>
            ```
        """)

        try:
            response = generate_response(prompt, gen, model_name, tokenizer, max_tokens=2048)
            code = extract_code(response)
            passed, n_tests, status = evaluate_lcb_with_timeout(
                code, test_cases, timeout_seconds=timeout
            )
            pass_at_1 = passed == n_tests and n_tests > 0 and status == "OK"
        except Exception as exc:
            passed, n_tests, status, code, response, pass_at_1 = 0, len(test_cases), f"ERROR:{exc}", "", "", False

        total += 1
        if status == "OK":
            exec_ok += 1
        if pass_at_1:
            pass1 += 1

        results.append({
            **row,
            "GeneratedCode":     code,
            "GeneratedResponse": response,
            "PromptUsed":        row_prompt,
            "n_Tests":           n_tests,
            "Tests_Passed":      passed,
            "Pass@1":            pass_at_1,
            "Eval_Status":       status,
        })

    summary = {
        "label":           label,
        "samples":         total,
        "pass1_rate":      round(pass1 / total, 4) if total else 0.0,
        "exec_rate":       round(exec_ok / total, 4) if total else 0.0,
        "pass1_count":     pass1,
    }
    logger.info("%-20s  pass@1=%.3f  exec=%.3f  (%d/%d)",
                label, summary["pass1_rate"], summary["exec_rate"], pass1, total)
    return results, summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseModel",   default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--loraAdapter", default="./finetune_deepseek_lv_output/best_lora_sft")
    parser.add_argument("--inputFile",   default="./mutations/livecodebench_LV_with_tests.jsonl")
    parser.add_argument("--outputDir",   default="./results/lora_lcb_eval")
    parser.add_argument("--limit",       type=int, default=200,
                        help="number of LCB samples to evaluate")
    parser.add_argument("--timeout",     type=int, default=50)
    parser.add_argument("--gpus",        default=None)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skipBaseline", action="store_true",
                        help="skip baseline (no-adapter) run; only eval with adapter")
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("CUDA_VISIBLE_DEVICES=%s", args.gpus)

    set_seed(args.seed)
    out_dir = Path(args.outputDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.inputFile)[: args.limit]
    logger.info("Loaded %d LCB LV samples from %s", len(records), args.inputFile)

    summaries = []

    # ── 1. Baseline (no adapter) ──────────────────────────────────────────────
    if not args.skipBaseline:
        logger.info("=" * 60)
        logger.info("BASELINE (no LoRA adapter)")
        logger.info("=" * 60)
        _, tokenizer, gen = load_model(args.baseModel, lora_adapter=None)
        base_results, base_summary = run_eval(
            records, gen, tokenizer, args.baseModel, "baseline", args.timeout
        )
        (out_dir / "baseline_lcb_lv.json").write_text(json.dumps(base_results, indent=2))
        summaries.append(base_summary)
        # free memory
        import gc; del gen; gc.collect(); torch.cuda.empty_cache()

    # ── 2. LoRA adapter ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ADAPTER: %s", args.loraAdapter)
    logger.info("=" * 60)
    _, tokenizer, gen = load_model(args.baseModel, lora_adapter=args.loraAdapter)
    adapter_results, adapter_summary = run_eval(
        records, gen, tokenizer, args.baseModel, "lora_adapter", args.timeout
    )
    (out_dir / "adapter_lcb_lv.json").write_text(json.dumps(adapter_results, indent=2))
    summaries.append(adapter_summary)

    # ── 3. Summary ────────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("GENERALIZATION SUMMARY (LiveCodeBench LV)")
    logger.info("%s", "=" * 60)
    for s in summaries:
        logger.info("  %-20s  pass@1=%.3f  exec=%.3f", s["label"], s["pass1_rate"], s["exec_rate"])

    if len(summaries) == 2:
        delta = summaries[1]["pass1_rate"] - summaries[0]["pass1_rate"]
        logger.info("  Delta (adapter - baseline): %+.3f", delta)

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    logger.info("Results written to %s", out_dir)


if __name__ == "__main__":
    main()
