from __future__ import annotations

"""
oracle_inference.py
───────────────────
Oracle-label study: does knowing the mutation type improve code generation?

Conditions:
  baseline  – standard prompt, no type hint (control)
  oracle    – type-specific hint injected using ground-truth label

For US oracle the model is asked to state the missing constraint *before*
generating code; this is captured in `predicted_constraint` per record and
lets us evaluate both whether it predicted correctly and whether it helped.

Usage:
  python oracle_inference.py \\
      --inputFiles mutations/humanEval_lv_with_tests.jsonl \\
                   mutations/humanEval_SF_with_tests.jsonl \\
                   mutations/humanEval_US_with_tests.jsonl \\
      --modelName  Qwen/Qwen2.5-Coder-7B-Instruct \\
      --conditions baseline oracle \\
      --outputDir  oracle_output/
"""

import argparse
import json
import logging
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from main_inference import (
    build_chat_prompt,
    cleanup_model,
    convert_general_check_code_HumanEval,
    convert_general_check_code_MBPP,
    evaluate_with_timeout,
    expected_name,
    extract_code,
    load_records,
    set_seed,
)

# ── env ───────────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("oracle_inference")


# ── oracle hints ──────────────────────────────────────────────────────────
ORACLE_HINTS: Dict[str, str] = {
    "LV": (
        "Note: This specification uses vague wording, function names, parameter names, "
        "and descriptions that may not fully convey the intended behavior. "
        "The input/output examples in the docstring reflect the precise intended behavior. "
        "Examine each example carefully to determine exactly what the function must compute, "
        "then implement it correctly."
    ),
    "SF": (
        "Note: This specification may have structural or formatting issues "
        "(e.g. inconsistent indentation, whitespace). "
        "Parse the intent carefully and implement the function correctly."
    ),
    "US": (
        "Note: This specification is missing a constraint. "
        "The input/output examples in the docstring reflect the complete intended behavior. "
        "Examine each example carefully to identify what constraint the description omits, "
        "state it in one sentence starting with exactly 'Missing constraint: ', "
        "then implement the function that satisfies that constraint."
    ),
}

# ── example-guided hint (US only) ─────────────────────────────────────────
# The mutated prompt already contains the docstring examples (>>> lines).
# This hint directs the model to use those examples as evidence to recover
# the missing constraint before writing code.
EXAMPLE_GUIDED_HINT_US = (
    "Note: The description above is underspecified — a key constraint has been omitted. "
    "However, the input/output examples in the docstring reflect the *complete* intended behavior.\n\n"
    "Follow these steps:\n"
    "Step 1 — Examine each example carefully: trace what relationship the inputs and output must satisfy.\n"
    "Step 2 — In one sentence starting with exactly 'Missing constraint: ', state the constraint "
    "the examples reveal but the description omits.\n"
    "Step 3 — Implement the function that satisfies that constraint."
)


# ── prompt builder ────────────────────────────────────────────────────────
def build_prompt(row_prompt: str, func_name: str, hint: Optional[str]) -> str:
    note_block = f"\n{hint}\n" if hint else ""
    return textwrap.dedent(f"""
        You are a senior Python developer.

        Task:
        {row_prompt}
        {note_block}
        Write **one** function named `{func_name}` that solves the task.
        If helpers are needed, define them above the main function.

        **Use only the Python standard library and place every required `import` at the very top.**

        Return *only* valid Python code in a single code block:
        ```python
        <your code here>
        ```
    """).strip()


# ── extract the predicted constraint from a US oracle response ───────────
def extract_predicted_constraint(response: str) -> str:
    m = re.search(r"Missing constraint:\s*(.+?)(?:\n|```|$)", response, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ── generate one response ─────────────────────────────────────────────────
def generate_response(prompt: str, gen, model_name: str, tokenizer) -> str:
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    return gen(
        formatted,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
    )[0]["generated_text"]


# ── run one (file × condition) pass ──────────────────────────────────────
def run_condition(
    records: List[Dict[str, Any]],
    condition: str,
    is_mbpp: bool,
    gen,
    model_name: str,
    tokenizer,
    timeout: int,
) -> List[Dict[str, Any]]:
    results = []

    for row in tqdm(records, desc=condition, ncols=80):
        row = dict(row)  # shallow copy

        mtype = row.get("mutation_type", "").upper()
        if condition == "oracle":
            hint = ORACLE_HINTS.get(mtype)
        elif condition == "example_guided":
            # For US: use example-driven CoT; for LV/SF: fall back to oracle hint
            hint = EXAMPLE_GUIDED_HINT_US if mtype == "US" else ORACLE_HINTS.get(mtype)
        else:
            hint = None

        row_prompt = (
            row.get("mutated_prompt")
            or row.get("prompt")
            or row.get("original_prompt")
            or ""
        )

        # ── function-name resolution (same logic as main_inference) ───────
        default_name   = expected_name(row)
        original_entry = (row.get("entry_point") or "").strip() or default_name
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", row_prompt)
        mutated_name   = m.group(1) if m else None
        func_name      = mutated_name or original_entry

        prompt = build_prompt(row_prompt, func_name, hint)

        try:
            response = generate_response(prompt, gen, model_name, tokenizer)
            code     = extract_code(response)

            if is_mbpp:
                test_src = "\n".join(row["test_list"])
                check_code, n_tests = convert_general_check_code_MBPP(test_src, func_name)
            else:
                check_code, n_tests = convert_general_check_code_HumanEval(row["test"], func_name)

            # try mutated name → original entry → auto-detect
            passed, status = 0, "ERROR: Not evaluated"
            for ep in ([mutated_name] if mutated_name else []) + [original_entry, None]:
                passed, status = evaluate_with_timeout(
                    code, check_code, timeout_seconds=timeout, entry_point=ep
                )
                if "not found" not in str(status) and not str(status).startswith("Function `"):
                    break

            pass_at_1 = (passed == n_tests and status == "OK")

        except Exception as exc:
            response = code = ""
            pass_at_1 = False
            passed = n_tests = 0
            status = f"ERROR: {type(exc).__name__}: {exc}"

        row.update(
            condition=condition,
            oracle_hint=hint or "",
            predicted_constraint=(
                extract_predicted_constraint(response)
                if condition in ("oracle", "example_guided") and mtype == "US"
                else ""
            ),
            GeneratedCode=code,
            GeneratedResponse=response,
            PromptUsed=row_prompt,
            n_Tests=n_tests,
            Tests_Passed=passed,
            **{"Pass@1": pass_at_1},
            Eval_Status=status,
        )
        results.append(row)

    return results


# ── summary table ─────────────────────────────────────────────────────────
def summarise(all_results: List[Dict]) -> None:
    tbl: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        tbl[r["condition"]][r.get("mutation_type", "?")].append(r["Pass@1"])

    ALL_CONDITIONS = ("baseline", "oracle", "example_guided")

    print("\n── Oracle Study Results ────────────────────────────────────────")
    print(f"{'Condition':<16} {'Type':<6} {'n':>5} {'Pass@1':>8}")
    print("─" * 39)
    for cond in ALL_CONDITIONS:
        if cond not in tbl:
            continue
        for mtype in ("LV", "SF", "US"):
            if mtype not in tbl[cond]:
                continue
            vals = tbl[cond][mtype]
            n    = len(vals)
            p1   = sum(vals) / n if n else 0.0
            print(f"{cond:<16} {mtype:<6} {n:>5} {p1:>8.3f}")
        print()

    # delta vs baseline
    print("── Δ vs Baseline ───────────────────────────────────────────────")
    print(f"{'Type':<6} {'baseline':>10} {'oracle':>10} {'ex_guided':>12} {'Δ_oracle':>10} {'Δ_ex':>10}")
    print("─" * 62)
    for mtype in ("LV", "SF", "US"):
        base_vals = tbl["baseline"].get(mtype, [])
        if not base_vals:
            continue
        b = sum(base_vals) / len(base_vals)
        oracle_vals = tbl["oracle"].get(mtype, [])
        eg_vals     = tbl["example_guided"].get(mtype, [])
        o  = sum(oracle_vals) / len(oracle_vals) if oracle_vals else float("nan")
        eg = sum(eg_vals)     / len(eg_vals)     if eg_vals     else float("nan")
        d_o  = (o  - b) if oracle_vals else float("nan")
        d_eg = (eg - b) if eg_vals     else float("nan")
        print(f"{mtype:<6} {b:>10.3f} {o:>10.3f} {eg:>12.3f} {d_o:>+10.3f} {d_eg:>+10.3f}")


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle label study")
    parser.add_argument("--inputFiles",  nargs="+", required=True,
                        help="mutation JSONL files to evaluate")
    parser.add_argument("--modelName",   default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--conditions",  nargs="+", default=["baseline", "oracle"],
                        choices=["baseline", "oracle", "example_guided"])
    parser.add_argument("--outputDir",   default="./oracle_output")
    parser.add_argument("--timeout",     type=int, default=50)
    parser.add_argument("--limit",       type=int, default=10_000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.outputDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", args.modelName)
    tokenizer = AutoTokenizer.from_pretrained(args.modelName, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.modelName,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )

    all_results: List[Dict] = []

    for input_file in args.inputFiles:
        records = load_records(input_file)[: args.limit]
        is_mbpp = "mbpp" in input_file.lower()
        file_stem = Path(input_file).stem

        for condition in args.conditions:
            out_path = out_dir / f"{condition}__{file_stem}.json"

            if out_path.exists():
                logger.info("Resuming — already done: %s", out_path.name)
                all_results.extend(json.loads(out_path.read_text()))
                continue

            logger.info("Running  condition=%s  file=%s", condition, file_stem)
            results = run_condition(
                records, condition, is_mbpp,
                gen, args.modelName, tokenizer, args.timeout,
            )
            out_path.write_text(json.dumps(results, indent=2))
            all_results.extend(results)
            logger.info("Saved → %s", out_path)

    cleanup_model(model, gen)

    summarise(all_results)

    all_path = out_dir / "all_results.json"
    all_path.write_text(json.dumps(all_results, indent=2))
    logger.info("All results → %s", all_path)


if __name__ == "__main__":
    main()
