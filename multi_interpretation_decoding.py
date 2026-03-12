#!/usr/bin/env python3
"""
self_consistency_lv.py
──────────────────────
Interpretation-aware self-consistency for LV-mutated HumanEval.

The model is told the prompt is vague and asked to generate up to N solutions,
each covering a different interpretation of the ambiguous terms and edge cases.
Solutions are tested one-by-one; we stop as soon as one passes all tests.

Prompt strategy:
  "This prompt is ambiguous. Generate {N} solutions, each interpreting the
   vague parts differently, to maximise the chance one satisfies hidden tests."

Output per problem:
  {
    "task_id":        "HumanEval/0",
    "first_pass_at":  2,          # null if none of the N solutions passed
    "attempts_made":  2,          # stopped early at 2
    "solutions": [
      {"index": 1, "code": "...", "passed_all": false, "tests_passed": 5, "n_tests": 7},
      {"index": 2, "code": "...", "passed_all": true,  "tests_passed": 7, "n_tests": 7},
    ]
  }

Usage:
    python self_consistency_lv.py \\
        --modelName   Qwen/Qwen2.5-Coder-7B-Instruct \\
        --inputFile   mutations/humanEval_lv_with_tests.jsonl \\
        --outputFile  results/self_consistency/Qwen7B__humanEval_lv_sc10.jsonl \\
        --maxSamples  10
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sc_lv")


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python developer specialising in under-specified requirements.
When a prompt contains vague or ambiguous language, you enumerate distinct \
interpretations and provide a correct implementation for each one.\
"""

def build_multi_solution_prompt(mutated_prompt: str, n: int, func_name: str) -> str:
    return textwrap.dedent(f"""
        The following programming prompt uses vague or ambiguous language — some \
terms, variable names, or constraints may have multiple valid interpretations.

        Your task is to generate exactly {n} different Python implementations, \
each based on a distinct interpretation of the ambiguous parts, so that at least \
one of your solutions satisfies the hidden test suite.

        Prompt:
        {mutated_prompt}

        Rules:
        - Each solution must define a function named `{func_name}`.
        - Each solution must be complete and self-contained.
        - Use only the Python standard library.
        - Number each solution exactly as shown below.
        - Do NOT add explanations between solutions — only code blocks.

        Format (repeat exactly {n} times):

        ### Solution 1
        ```python
        <your code here>
        ```

        ### Solution 2
        ```python
        <your code here>
        ```

        (continue up to Solution {n})
    """).strip()


# ── Parse N solutions from model output ───────────────────────────────────────

def parse_solutions(response: str, n: int) -> List[str]:
    """
    Extract up to n code blocks from a response that contains
      ### Solution k
      ```python ... ```
    Falls back to extracting any ```python ... ``` blocks if the numbered
    format is not found.
    """
    # Primary: numbered sections
    solutions: List[str] = []
    pattern = re.compile(
        r"###\s*Solution\s*\d+\s*\n```(?:python)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(response):
        solutions.append(textwrap.dedent(m.group(1)).strip())
        if len(solutions) == n:
            break

    # Fallback: any ```python ... ``` blocks
    if not solutions:
        for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL):
            solutions.append(textwrap.dedent(m.group(1)).strip())
            if len(solutions) == n:
                break

    # Last resort: whole response as one solution
    if not solutions:
        solutions.append(response.strip())

    return solutions


# ── Test harness ───────────────────────────────────────────────────────────────

def build_check_code(test_code: str) -> Tuple[str, int]:
    CHECK_RE = re.compile(r"^\s*def\s+check\s*\(\s*candidate\s*\)\s*:", re.M)
    if CHECK_RE.search(test_code):
        n_tests = len(re.findall(r"^\s*assert\b", test_code, re.M))
        wrapper = textwrap.dedent(f"""
            _original_check = check
            def check(candidate):
                try:
                    _original_check(candidate)
                    return {n_tests}, {n_tests}
                except AssertionError:
                    return 0, {n_tests}
        """)
        return test_code.rstrip() + "\n\n" + wrapper, n_tests

    lines = [ln.rstrip() for ln in test_code.splitlines() if ln.lstrip().startswith("assert")]
    n_tests = len(lines)
    body = ["def check(candidate):", "    passed = 0", f"    total = {n_tests}"]
    for ln in lines:
        body += [
            "    try:",
            f"        assert {ln.lstrip()[len('assert '):]}",
            "        passed += 1",
            "    except AssertionError:",
            "        pass",
        ]
    body.append("    return passed, total")
    return "\n".join(body), n_tests


# ── Sandbox ────────────────────────────────────────────────────────────────────

def _safe_exec(candidate_code: str, check_code: str, queue: mp.Queue,
               entry_point: Optional[str]) -> None:
    try:
        env: Dict[str, Any] = {}
        exec(candidate_code, env)

        fn = None
        if entry_point:
            fn = env.get(entry_point)
            if not callable(fn):
                for k, v in env.items():
                    if k.lower() == entry_point.lower() and callable(v):
                        fn = v
                        break
        if not callable(fn):
            fns = [v for v in env.values() if callable(v)]
            fn = fns[0] if fns else None
        if fn is None:
            queue.put((0, 0, "Function not found"))
            return

        env["candidate"] = fn
        exec(check_code + "\n_result = check(candidate)", env)
        passed, total = env["_result"]
        queue.put((passed, total, "OK"))
    except Exception as exc:
        queue.put((0, 0, f"ERROR: {type(exc).__name__}: {exc}"))


def run_tests(candidate_code: str, check_code: str,
              entry_candidates: List[Optional[str]], timeout: int = 20) -> Tuple[int, int, str]:
    for ep in entry_candidates:
        q: mp.Queue = mp.Queue()
        proc = mp.Process(target=_safe_exec, args=(candidate_code, check_code, q, ep))
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return 0, 0, "ERROR: Timeout"
        try:
            passed, total, status = q.get_nowait()
        except Exception:
            return 0, 0, "ERROR: Queue empty"

        if "not found" not in status.lower():
            return passed, total, status

    return 0, 0, "ERROR: Function not found"


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(model_name: str):
    logger.info("Loading %s …", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded.")
    return model, tok


def generate(model, tokenizer, user_prompt: str, model_name: str,
             max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    else:
        name = model_name.lower()
        if "llama" in name:
            text = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_prompt} [/INST]"
        elif "deepseek" in name:
            text = f"### Instruction:\n{SYSTEM_PROMPT}\n\n{user_prompt}\n### Response:\n"
        elif "qwen" in name:
            text = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        else:
            text = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — diversity comes from the prompt asking for N interpretations
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName",    required=True)
    parser.add_argument("--inputFile",    default="mutations/humanEval_lv_with_tests.jsonl")
    parser.add_argument("--outputFile",   required=True)
    parser.add_argument("--maxSamples",   type=int, default=10,
                        help="number of interpretations to ask for")
    parser.add_argument("--maxNewTokens", type=int, default=2048,
                        help="token budget — needs to be large to fit N solutions")
    parser.add_argument("--limit",        type=int, default=0,
                        help="truncate dataset for quick testing (0 = all)")
    args = parser.parse_args()

    records = [json.loads(l) for l in Path(args.inputFile).read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    model, tokenizer = load_model(args.modelName)

    out_path = Path(args.outputFile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    first_pass_counts = [0] * (args.maxSamples + 1)   # index k = solved on solution k
    never_solved = 0

    pbar = trange(len(records), desc="problems", ncols=90)
    for idx in pbar:
        row = records[idx]
        task_id = row.get("task_id", str(idx))
        pbar.set_description(task_id)

        mutated_prompt = (
            row.get("mutated_prompt") or row.get("prompt") or row.get("original_prompt") or ""
        )

        # Function name in mutated prompt (may differ from entry_point)
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", mutated_prompt)
        mutated_name  = m.group(1) if m else None
        original_entry = (row.get("entry_point") or "").strip()
        name_for_template = mutated_name or original_entry or "solution"

        entry_candidates: List[Optional[str]] = []
        if mutated_name:
            entry_candidates.append(mutated_name)
        if original_entry and original_entry not in entry_candidates:
            entry_candidates.append(original_entry)
        entry_candidates.append(None)

        check_code, n_tests = build_check_code(row.get("test", ""))
        user_prompt = build_multi_solution_prompt(mutated_prompt, args.maxSamples, name_for_template)

        # Single model call → N solutions in one response
        response = generate(model, tokenizer, user_prompt, args.modelName, args.maxNewTokens)
        solutions = parse_solutions(response, args.maxSamples)

        logger.info("%s — model returned %d solution(s)", task_id, len(solutions))

        tested: List[Dict] = []
        first_pass_at: Optional[int] = None

        for sol_idx, code in enumerate(solutions, start=1):
            passed, total, status = run_tests(code, check_code, entry_candidates)
            passed_all = (passed == n_tests and n_tests > 0 and status == "OK")

            tested.append({
                "index":        sol_idx,
                "code":         code,
                "tests_passed": passed,
                "n_tests":      n_tests,
                "status":       status,
                "passed_all":   passed_all,
            })

            logger.info("  %s  solution %d/%d  %d/%d  [%s]",
                        task_id, sol_idx, len(solutions), passed, n_tests, status)

            if passed_all:
                first_pass_at = sol_idx
                break   # stop — correct solution found

        if first_pass_at is None:
            never_solved += 1
        else:
            first_pass_counts[first_pass_at] += 1

        results.append({
            "task_id":       task_id,
            "mutated_prompt": mutated_prompt,
            "entry_point":   original_entry,
            "n_solutions_generated": len(solutions),
            "first_pass_at": first_pass_at,
            "attempts_made": len(tested),
            "solutions":     tested,
            "raw_response":  response,
        })

        out_path.write_text(
            "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(results)
    print("\n" + "=" * 60)
    print(f"Interpretation-aware Self-Consistency — LV")
    print(f"Model  : {args.modelName}")
    print(f"N      : {args.maxSamples} interpretations requested")
    print(f"Dataset: {args.inputFile}   problems={n}")
    print("=" * 60)
    print(f"\n{'Solution k':<13} {'First solve':<14} {'Cumulative pass@k'}")
    print("-" * 44)
    cumulative = 0
    for k in range(1, args.maxSamples + 1):
        cumulative += first_pass_counts[k]
        print(f"  k={k:<9}  {first_pass_counts[k]:<14}  {cumulative/n*100:.1f}%")
    print(f"\n  Never solved : {never_solved}  ({never_solved/n*100:.1f}%)")
    print(f"\n  pass@1 (sol 1 only) = {first_pass_counts[1]/n*100:.1f}%")
    print(f"  pass@{args.maxSamples} (any solution) = {sum(first_pass_counts[1:])/n*100:.1f}%")
    print("=" * 60)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
