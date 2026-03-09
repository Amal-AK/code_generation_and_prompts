#!/usr/bin/env python3
"""
eval_clean_as_us_mbpp.py
─────────────────────────
For each MBPP CLEAN prompt that the classifier predicted as US,
generate code with Qwen2.5-Coder-7B-Instruct and run the MBPP test suite.

Reports pass@1 per problem and overall, to measure whether models
actually fail on these "clean" but potentially underspecified prompts.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_clean_as_us_mbpp.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Config ─────────────────────────────────────────────────────────────────────
GEN_MODEL      = "Qwen/Qwen2.5-Coder-7B-Instruct"
ANALYSIS_JSON  = "clean_as_us_analysis.json"
MBPP_PATH      = "datasets/mbpp/mbpp.jsonl"
OUTPUT_JSON    = "eval_clean_as_us_mbpp_results.json"
MAX_NEW_TOKENS = 512
TIMEOUT        = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_func_name(prompt: str) -> str | None:
    m = re.search(r"\bdef\s+(\w+)\s*\(", prompt)
    return m.group(1) if m else None


def build_instruction(prompt: str, func_name: str | None) -> str:
    func_line = (
        f"Write **one** function named `{func_name}` that solves the task."
        if func_name else
        "Write **one** Python function that solves the task."
    )
    return textwrap.dedent(f"""
        You are a senior Python developer.

        Task:
        {prompt}

        {func_line}
        If helpers are needed, define them above the main function.

        **Use only the Python standard library and place every required `import` at the very top.**

        Return *only* valid Python code in a single code block:
        ```python
        <your code here>
        ```
    """).strip()


def run_mbpp_tests(code: str, test_list: list[str], timeout: int = TIMEOUT) -> tuple[bool, str]:
    """Run MBPP assert-style tests. Returns (passed, error_msg)."""
    script = code + "\n\n" + "\n".join(test_list) + "\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr.decode(errors="replace") or result.stdout.decode(errors="replace"))[:300]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load the CLEAN→US misclassified cases
    cases = json.load(open(ANALYSIS_JSON))
    mbpp_cases = [c for c in cases if c["dataset"] == "mbpp"]
    print(f"CLEAN→US MBPP cases: {len(mbpp_cases)}")

    # Load MBPP ground-truth test cases indexed by task_id
    mbpp_db: dict[str, dict] = {}
    with open(MBPP_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mbpp_db[str(obj["task_id"])] = obj
    print(f"Loaded {len(mbpp_db)} MBPP records")

    # Match cases to their test lists
    to_eval = []
    for c in mbpp_cases:
        tid = str(c["task_id"])
        if tid not in mbpp_db:
            print(f"  WARNING: {tid} not found in MBPP db — skipping")
            continue
        rec = mbpp_db[tid]
        to_eval.append({
            "task_id":   tid,
            "prompt":    c["prompt"],
            "test_list": rec.get("test_list", []),
            "source_text": rec.get("text", ""),  # original MBPP prompt for reference
        })
    print(f"Matched {len(to_eval)} / {len(mbpp_cases)} cases\n")

    # Load generation model
    print(f"Loading model: {GEN_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    input_device = next(model.parameters()).device
    print(f"  Model on {input_device}\n")

    results = []
    passed_total = 0

    for item in tqdm(to_eval, desc="Evaluating", ncols=90):
        prompt    = item["prompt"]
        func_name = extract_func_name(prompt)
        instr     = build_instruction(prompt, func_name)

        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=1024).to(input_device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code_block(raw)

        passed, err = run_mbpp_tests(code, item["test_list"])
        if passed:
            passed_total += 1

        rec = {
            "task_id":       item["task_id"],
            "prompt":        prompt,
            "mbpp_text":     item["source_text"],
            "generated_code": code,
            "passed":        passed,
            "error":         err,
            "n_tests":       len(item["test_list"]),
        }
        results.append(rec)

        status = "✓" if passed else "✗"
        print(f"  [{status}] {item['task_id']:>6}  {prompt[:60].strip()!r}")
        if not passed and err:
            print(f"         → {err[:120]}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(results)
    pass1 = passed_total / n if n else 0

    print(f"\n{'='*60}")
    print(f"MBPP CLEAN→US prompts  n={n}  pass@1={pass1:.3f}  ({passed_total}/{n} passed)")
    print(f"{'='*60}")

    failed = [r for r in results if not r["passed"]]
    print(f"\nFailed ({len(failed)}):")
    for r in failed:
        print(f"  {r['task_id']:>6}: {r['prompt'][:70].strip()}")
        if r["error"]:
            print(f"           {r['error'][:100]}")

    summary = {
        "model":     GEN_MODEL,
        "n":         n,
        "passed":    passed_total,
        "pass1":     round(pass1, 4),
        "results":   results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
