#!/usr/bin/env python3
"""
api_inference.py

Evaluate OpenAI API models (e.g. gpt-4o, gpt-5-mini) on MBPP / HumanEval /
LiveCodeBench using the OpenAI **Batch API** (50% cheaper than individual calls).

Pipeline per dataset:
  1. Build one prompt per record.
  2. Upload all prompts as a single batch job.
  3. Poll until the batch is complete (up to 24 h).
  4. Download responses and run the same sandbox evaluation as main_inference.py.

Usage:
    export OPENAI_API_KEY=sk-...
    python api_inference.py --model gpt-5-mini --outputDir ./results/api
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import trange

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("api_inference")

client = OpenAI()

# ─────────────────────────── sandbox (copied from main_inference.py) ──────────────────

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Timeout!")

signal.signal(signal.SIGALRM, _timeout_handler)


def extract_code(txt: str) -> str:
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    code = m.group(1) if m else txt
    return textwrap.dedent(code).strip()


def expected_name(row: Dict, default: str = "solution") -> str:
    ep = (row.get("entry_point") or "").strip()
    if ep:
        return ep
    test_str = row.get("test") or (row.get("test_list") or [""])[0]
    m = re.match(r"\s*assert\s+(\w+)\s*\(", test_str)
    return m.group(1) if m else default


SAFE_BUILTINS = {"sorted", "len", "sum", "min", "max", "any", "all"}


def convert_general_check_code_MBPP(test_code: str, func_name: str) -> Tuple[str, int]:
    if func_name in SAFE_BUILTINS:
        return test_code, test_code.count("assert")
    lines = [ln.rstrip() for ln in test_code.splitlines() if ln.lstrip().startswith("assert")]
    total = len(lines)
    body = ["def check(candidate):", "    passed = 0", f"    total  = {total}"]
    for ln in lines:
        ln = ln.lstrip()
        ln = re.sub(rf"\b{re.escape(func_name)}\s*\(", "candidate(", ln, count=1)
        body += ["    try:", f"        assert {ln[len('assert '):]}", "        passed += 1",
                 "    except AssertionError:", "        pass"]
    body.append("    return passed, total")
    return "\n".join(body), total


def convert_general_check_code_HumanEval(test_code: str, func_name: str) -> Tuple[str, int]:
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
    total = len(lines)
    body = ["def check(candidate):", "    passed = 0", f"    total  = {total}"]
    for ln in lines:
        ln = re.sub(rf"\b{re.escape(func_name)}\s*\(", "candidate(", ln, count=1)
        body += ["    try:", f"        assert {ln.lstrip()[len('assert '):]}", "        passed += 1",
                 "    except AssertionError:", "        pass"]
    body.append("    return passed, total")
    return "\n".join(body), total


def _safe_exec(candidate_code, check_code, queue, entry_point=None):
    try:
        env: Dict[str, Any] = {}
        exec(candidate_code, env)
        if entry_point:
            fn = env.get(entry_point)
            if not callable(fn):
                for k, v in env.items():
                    if k.lower() == entry_point.lower() and callable(v):
                        fn = v
                        break
            if not callable(fn):
                queue.put((0, f"Function `{entry_point}` not found"))
                return
            fns = [fn]
        else:
            fns = [v for v in env.values() if callable(v)]
        if not fns:
            queue.put((0, "Function not found"))
            return
        def run(fn):
            env["candidate"] = fn
            exec(check_code + "\n_result = check(candidate)", env)
            passed, _ = env["_result"]
            return passed
        queue.put((max(run(f) for f in fns), "OK"))
    except Exception as exc:
        queue.put((0, f"ERROR: {type(exc).__name__}: {exc}"))


def evaluate_with_timeout(candidate_code, check_code, *, timeout_seconds=20, entry_point=None):
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_safe_exec, args=(candidate_code, check_code, queue, entry_point))
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0, "ERROR: Timeout/Killed"
    try:
        return queue.get_nowait()
    except Exception:
        return 0, "ERROR: Unknown"


def _safe_exec_lcb(candidate_code, test_cases, queue, per_test_timeout=10):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(candidate_code)
        tmp_path = f.name
    passed = 0
    total = len(test_cases)
    first_error = "OK"
    try:
        for tc in test_cases:
            expected = (tc.get("output") or "").rstrip()
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=tc.get("input", ""),
                    capture_output=True, text=True, timeout=per_test_timeout,
                )
                actual = proc.stdout.rstrip()
                if actual == expected:
                    passed += 1
                elif first_error == "OK":
                    first_error = f"WrongAnswer: expected {expected!r}, got {actual!r}"
                    if not actual and proc.stderr.strip():
                        first_error += f" [stderr: {proc.stderr.strip()[:200]!r}]"
            except subprocess.TimeoutExpired:
                if first_error == "OK":
                    first_error = "Timeout"
            except Exception as exc:
                if first_error == "OK":
                    first_error = f"ERROR: {exc}"
    finally:
        os.unlink(tmp_path)
    queue.put((passed, total, "OK" if passed == total else first_error))


def evaluate_lcb_with_timeout(candidate_code, test_cases, *, timeout_seconds=60, per_test_timeout=10):
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_safe_exec_lcb, args=(candidate_code, test_cases, queue, per_test_timeout))
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0, len(test_cases), "ERROR: Timeout/Killed"
    try:
        return queue.get_nowait()
    except Exception:
        return 0, len(test_cases), "ERROR: Unknown"


# ─────────────────────────── data loading ─────────────────────────────────────────────

def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text("utf-8").lstrip("\ufeff").strip()
    if p.suffix == ".json":
        return json.loads(text)
    if p.suffix in (".jsonl", ".ndjson"):
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return pd.read_csv(p).to_dict(orient="records")


# ─────────────────────────── prompt builders ──────────────────────────────────────────

def build_he_mbpp_prompt(row: Dict) -> Tuple[str, str, str, str | None]:
    """Returns (row_prompt, llm_prompt, name_for_template, mutated_name)."""
    row_prompt = (
        row.get("mutated_prompt") or row.get("prompt") or row.get("original_prompt")
        or row.get("prompt_text") or row.get("text") or ""
    )
    default_name   = expected_name(row)
    original_entry = (row.get("entry_point") or "").strip() or default_name
    m              = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", row_prompt)
    mutated_name   = m.group(1) if m else None
    name_for_template = mutated_name or original_entry or default_name
    llm_prompt = textwrap.dedent(f"""
        You are a senior Python developer.

        Task:
        {row_prompt}

        Write **one** function named `{name_for_template}` that solves the task.
        If helpers are needed, define them above the main function.

        **Use only the Python standard library and place every required `import` statement at the very top of the code block.**

        ⚠️ Return *only* valid Python code, wrapped in a single code block like this:
        ```python
        <your code here>
        ```
    """)
    return row_prompt, llm_prompt, name_for_template, original_entry, mutated_name


def build_lcb_prompt(row: Dict) -> Tuple[str, str, List]:
    """Returns (row_prompt, llm_prompt, test_cases)."""
    row_prompt = (
        row.get("mutated_prompt") or row.get("original_prompt") or row.get("prompt") or ""
    ).strip()
    raw_tests = row.get("test", "[]")
    try:
        test_cases = json.loads(raw_tests) if isinstance(raw_tests, str) else raw_tests
    except Exception:
        test_cases = []
    llm_prompt = textwrap.dedent(f"""
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
    return row_prompt, llm_prompt, test_cases


# ─────────────────────────── OpenAI Batch API ─────────────────────────────────────────

def run_openai_batch(prompts_by_id: Dict[str, str], model_name: str, max_tokens: int) -> Dict[str, str]:
    """Submit prompts via Batch API; returns {custom_id: response_text}."""
    lines = [
        json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
            },
        })
        for cid, prompt in prompts_by_id.items()
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        tmp_path = f.name

    with open(tmp_path, "rb") as fh:
        upload = client.files.create(file=fh, purpose="batch")
    os.unlink(tmp_path)
    logger.info("Uploaded batch input: %s  (%d requests)", upload.id, len(lines))

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info("Batch job created: %s", batch.id)

    while True:
        batch = client.batches.retrieve(batch.id)
        rc    = getattr(batch, "request_counts", None)
        done  = getattr(rc, "completed", "?") if rc else "?"
        total = getattr(rc, "total",     "?") if rc else "?"
        logger.info("Batch %s: %s  (%s/%s done)", batch.id, batch.status, done, total)
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired"):
            raise RuntimeError(f"Batch {batch.id} ended with status {batch.status}")
        time.sleep(30)

    raw = client.files.content(batch.output_file_id)
    raw = raw.content if hasattr(raw, "content") else raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    results: Dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cid   = entry.get("custom_id", "")
        try:
            content = entry["response"]["body"]["choices"][0]["message"]["content"] or ""
        except Exception:
            content = ""
        results[cid] = content

    logger.info("Batch complete: %d/%d responses", len(results), len(lines))
    return results


# ─────────────────────────── direct (non-batch) calls ────────────────────────────────

def run_direct(prompts_by_id: Dict[str, str], model_name: str, max_tokens: int) -> Dict[str, str]:
    """Call the API one request at a time; returns {custom_id: response_text}."""
    results: Dict[str, str] = {}
    ids = list(prompts_by_id.keys())
    for i, cid in enumerate(ids):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompts_by_id[cid]}],
                max_completion_tokens=max_tokens,
            )
            results[cid] = resp.choices[0].message.content or ""
        except Exception as exc:
            err = str(exc)
            if "401" in err or "invalid_api_key" in err or "Unauthorized" in err:
                raise RuntimeError(f"Authentication failed — check your OPENAI_API_KEY.\n{exc}") from exc
            logger.warning("Request %s failed: %s", cid, exc)
            results[cid] = ""
        if (i + 1) % 10 == 0:
            logger.info("Direct calls: %d/%d done", i + 1, len(ids))
    return results


# ─────────────────────────── download-only save ──────────────────────────────────────

def save_responses_only(records: List[Dict], responses: Dict[str, str], args) -> None:
    """Save records with GeneratedResponse + GeneratedCode extracted; skip sandbox eval."""
    is_lcb = "livecodebench" in Path(args.inputFile).stem.lower()
    for idx, row in enumerate(records):
        response = responses.get(str(idx), "")
        code     = extract_code(response)
        if is_lcb:
            row_prompt, _, _ = build_lcb_prompt(row)
        else:
            row_prompt, _, _, _, _ = build_he_mbpp_prompt(row)
        row.update({
            "GeneratedCode":     code,
            "GeneratedResponse": response,
            "PromptUsed":        row_prompt,
            "Eval_Status":       "pending",
        })
    Path(args.outputFile).write_text(json.dumps(records, indent=2), "utf-8")
    logger.info("Saved %d responses (no eval) → %s", len(records), args.outputFile)


# ─────────────────────────── evaluation loops ─────────────────────────────────────────

def evaluate_he_mbpp(records: List[Dict], responses: Dict[str, str], args) -> Dict:
    total = success_exec = pass1_true = false_pass = 0
    pbar  = trange(len(records), desc="evaluating", ncols=80)

    for idx in pbar:
        row = records[idx]
        pbar.set_description(str(row.get("task_id", idx)))

        row_prompt, _, name_for_template, original_entry, mutated_name = build_he_mbpp_prompt(row)
        response = responses.get(str(idx), "")
        code     = extract_code(response)

        try:
            if "mbpp" in args.inputFile.lower():
                check_code, n_tests = convert_general_check_code_MBPP(
                    "\n".join(row["test_list"]), name_for_template)
            else:
                check_code, n_tests = convert_general_check_code_HumanEval(
                    row["test"], name_for_template)

            entry_candidates = []
            if mutated_name:
                entry_candidates.append(mutated_name)
            if original_entry and original_entry not in entry_candidates:
                entry_candidates.append(original_entry)
            entry_candidates.append(None)

            passed = 0
            status = "ERROR: Not evaluated"
            for ep in entry_candidates:
                passed, status = evaluate_with_timeout(
                    code, check_code, timeout_seconds=args.timeout, entry_point=ep)
                if isinstance(status, str) and ("not found" in status or status.startswith("Function `")):
                    continue
                break

            pass_at_1 = passed == n_tests and status == "OK"

        except Exception as exc:
            passed, status, code, response, n_tests, pass_at_1 = 0, f"ERROR: {exc}", "", "", 0, False
            logger.exception("Task %s failed", row.get("task_id", idx))

        logger.info("Task %-30s  %d/%d  [%s]", row.get("task_id", idx), passed, n_tests, status)

        total += 1
        if status == "OK":
            success_exec += 1
        if pass_at_1:
            pass1_true += 1
        if status == "OK" and not pass_at_1:
            false_pass += 1

        row.update({
            "GeneratedCode":     code,
            "GeneratedResponse": response,
            "PromptUsed":        row_prompt,
            "TestCases":         row.get("test", ""),
            "n_Tests":           n_tests,
            "Tests_Passed":      passed,
            "Pass@1":            pass_at_1,
            "Eval_Status":       status,
        })

    Path(args.outputFile).write_text(json.dumps(records, indent=2), "utf-8")
    return {
        "Model":               args.model,
        "Dataset":             args.inputFile,
        "Samples":             total,
        "SuccessExec":         success_exec,
        "Pass@1_TRUE":         pass1_true,
        "Runnable_FALSE":      false_pass,
        "SuccessExecRate":     round(success_exec / total, 3) if total else 0.0,
        "Pass@1_Rate":         round(pass1_true   / total, 3) if total else 0.0,
        "Runnable_FALSE_Rate": round(false_pass   / total, 3) if total else 0.0,
    }


def evaluate_lcb(records: List[Dict], responses: Dict[str, str], args) -> Dict:
    total = success_exec = pass1_true = 0
    pbar  = trange(len(records), desc="evaluating LCB", ncols=80)

    for idx in pbar:
        row = records[idx]
        pbar.set_description(str(row.get("task_id", idx)))

        row_prompt, _, test_cases = build_lcb_prompt(row)
        response  = responses.get(str(idx), "")
        code      = extract_code(response)
        n_tests   = len(test_cases)

        try:
            passed, n_tests, status = evaluate_lcb_with_timeout(
                code, test_cases, timeout_seconds=args.timeout)
            pass_at_1 = passed == n_tests and n_tests > 0 and status == "OK"
        except Exception as exc:
            passed, status, code, response, pass_at_1 = 0, f"ERROR: {exc}", "", "", False
            logger.exception("Task %s failed", row.get("task_id", idx))

        logger.info("Task %-30s  %d/%d  [%s]", row.get("task_id", idx), passed, n_tests, status)

        total += 1
        if status == "OK":
            success_exec += 1
        if pass_at_1:
            pass1_true += 1

        row.update({
            "GeneratedCode":     code,
            "GeneratedResponse": response,
            "PromptUsed":        row_prompt,
            "n_Tests":           n_tests,
            "Tests_Passed":      passed,
            "Pass@1":            pass_at_1,
            "Eval_Status":       status,
        })

    Path(args.outputFile).write_text(json.dumps(records, indent=2), "utf-8")
    return {
        "Model":           args.model,
        "Dataset":         args.inputFile,
        "Samples":         total,
        "SuccessExec":     success_exec,
        "Pass@1_TRUE":     pass1_true,
        "SuccessExecRate": round(success_exec / total, 3) if total else 0.0,
        "Pass@1_Rate":     round(pass1_true   / total, 3) if total else 0.0,
    }


# ─────────────────────────── eval-only (from saved response files) ───────────────────

def evaluate_from_file(response_file: str, args) -> Dict:
    """Run sandbox eval on a JSON file that already has GeneratedCode populated."""
    records = json.loads(Path(response_file).read_text("utf-8"))
    is_lcb  = "livecodebench" in Path(response_file).stem.lower()

    args.inputFile  = response_file
    args.outputFile = response_file   # overwrite in place

    # build a fake responses dict from what's already saved
    responses = {str(idx): row.get("GeneratedResponse", "") for idx, row in enumerate(records)}

    if is_lcb:
        return evaluate_lcb(records, responses, args)
    else:
        return evaluate_he_mbpp(records, responses, args)


# ─────────────────────────── main ─────────────────────────────────────────────────────

_HUMANEVAL_MUTATION_FILES = [
    "./datasets/humanEval/HumanEval.jsonl",
    "./mutations/HumanEval_US_with_tests.jsonl",
    "./mutations/humanEval_lv_with_tests.jsonl",
    "./mutations/humanEval_SF_with_tests.jsonl",
]

_MBPP_FILES = [
    "./datasets/mbpp/mbpp.jsonl",
    "./mutations/mbpp_US_with_tests.jsonl",
    "./mutations/mbpp_LV_with_tests.jsonl",
    "./mutations/mbpp_SF_with_tests.jsonl",
]

_LCB_FILES = [
    "./datasets/livecodebench/livecodebench_public.jsonl",
    "./mutations/livecodebench_US_with_tests.jsonl",
    "./mutations/livecodebench_LV_with_tests.jsonl",
    "./mutations/livecodebench_SF_with_tests.jsonl",
]

_ALL_FILES = _HUMANEVAL_MUTATION_FILES + _MBPP_FILES + _LCB_FILES


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenAI API models via Batch API on MBPP / HumanEval / LCB"
    )
    parser.add_argument("--model",      default="gpt-5-mini",
                        help="OpenAI model name (e.g. gpt-5-mini, gpt-4o-mini)")
    parser.add_argument("--inputFiles", nargs="+", default=_HUMANEVAL_MUTATION_FILES,
                        help="dataset paths to evaluate (default: HumanEval + mutations)")
    parser.add_argument("--outputDir",  default="./results/api")
    parser.add_argument("--maxTokens",  type=int, default=2048)
    parser.add_argument("--timeout",    type=int, default=50,
                        help="sandbox timeout per problem in seconds (ignored with --skip-eval)")
    parser.add_argument("--limit",      type=int, default=1000,
                        help="evaluate on first N samples per file")
    parser.add_argument("--skip-eval",  action="store_true",
                        help="download generated code only; skip sandbox evaluation")
    parser.add_argument("--no-batch",   action="store_true",
                        help="use direct API calls instead of the Batch API (faster start, no 50%% discount)")
    parser.add_argument("--eval-only",  nargs="+", metavar="JSON_FILE",
                        help="skip API calls; run sandbox eval on already-saved JSON response files")
    args = parser.parse_args()

    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── eval-only mode: just run sandbox on saved files ───────────────────
    if args.eval_only:
        all_summaries = []
        for response_file in args.eval_only:
            logger.info("=" * 60)
            logger.info("Eval-only: %s", response_file)
            logger.info("=" * 60)
            args.inputFile  = response_file
            args.outputFile = response_file
            summary = evaluate_from_file(response_file, args)
            all_summaries.append(summary)
            logger.info("Summary: %s", summary)
        if all_summaries:
            df = pd.DataFrame(all_summaries)
            print("\n" + df.to_string(index=False))
            df.to_csv(output_dir / "all_results_summary.csv", index=False)
            logger.info("Combined summary saved.")
        return

    all_summaries = []

    for input_file in args.inputFiles:
        logger.info("=" * 60)
        logger.info("Dataset: %s", input_file)
        logger.info("=" * 60)

        records      = load_records(input_file)[: args.limit]
        is_lcb       = "livecodebench" in Path(input_file).stem.lower()
        model_slug   = args.model.replace("/", "_")
        dataset_slug = Path(input_file).stem

        args.inputFile  = input_file
        args.outputFile = str(output_dir / f"{model_slug}__{dataset_slug}.json")

        # ── 1. build prompts ──────────────────────────────────────────────
        prompts_by_id: Dict[str, str] = {}
        for idx, row in enumerate(records):
            if is_lcb:
                _, llm_prompt, _ = build_lcb_prompt(row)
            else:
                _, llm_prompt, _, _, _ = build_he_mbpp_prompt(row)
            prompts_by_id[str(idx)] = llm_prompt

        # ── 2. submit batch & download (or direct calls) ─────────────────
        logger.info("Submitting %d prompts to %s ...", len(prompts_by_id), args.model)
        if args.no_batch:
            responses = run_direct(prompts_by_id, args.model, args.maxTokens)
        else:
            responses = run_openai_batch(prompts_by_id, args.model, args.maxTokens)

        # ── 3. save responses or run full eval ────────────────────────────
        if args.skip_eval:
            save_responses_only(records, responses, args)
        else:
            summary = (evaluate_lcb if is_lcb else evaluate_he_mbpp)(records, responses, args)
            all_summaries.append(summary)
            logger.info("Summary: %s", summary)

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        print("\n" + df.to_string(index=False))
        df.to_csv(output_dir / "all_results_summary.csv", index=False)
        logger.info("Combined summary saved.")


if __name__ == "__main__":
    main()
