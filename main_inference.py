from __future__ import annotations

# ─────────────────────────────────────── imports ───────────────────────────────────────
import argparse
import logging
import multiprocessing as mp
import os
import random
import re
import signal
import textwrap
from typing import Any, Tuple , List, Dict
import ast
import json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Mistral3 (used by Devstral) ships only ForConditionalGeneration in some
# transformers versions; register it so AutoModelForCausalLM resolves it.
try:
    from transformers.models.mistral3 import Mistral3Config, Mistral3ForConditionalGeneration
    AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)
except (ImportError, Exception):
    pass

# ─────────────────────────────── global settings & logging ─────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mbpp_humaneval_evaluator")


# ───────────────────────────────── reproducibility helper ──────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]


# ─────────────────────────────── generation prompt helpers ─────────────────────────────
def build_chat_prompt(prompt: str, model_name: str, tokenizer=None) -> str:
    """
    Return a correctly formatted *chat* prompt for almost any HF-style model.
    1. If the `tokenizer` exposes `.apply_chat_template(...)`, use that.
    2. Otherwise fall back to a small set of hard-coded templates.
    3. Fallback-to-fallback: return the raw prompt.
    """
    # -------- 1 : let the tokenizer do the job whenever it can ------------
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            # the new HF API (> 4.38) – safest option
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # some models ship an old or buggy template, drop to rule-table
            pass

    # -------- 2 : minimal rule-table for popular models -------------------
    name = model_name.lower()

    # llama-family (CodeLlama, Llama-2-chat, etc.)
    if "llama" in name:
        return f"<s>[INST] {prompt.strip()} [/INST]"

    # DeepSeek-Coder Instruct
    if "deepseek" in name:
        return f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"

    # StarCoder-family (BigCode, SantaCoder) – they expect plain text + <|endoftext|>
    if "starcoder" in name or "santacoder" in name:
        return prompt.strip() + tokenizer.eos_token

    # Qwen chat / code
    if "qwen" in name:
        return f"<|im_start|>user\n{prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    # Mistral-Instruct / Zephyr / Phi-2-chat – use the same style as Llama-chat
    if any(k in name for k in ("mistral", "zephyr", "phi")):
        return f"<s>[INST] {prompt.strip()} [/INST]"

    # OpenAI ChatCompletion models (if you ever wrap them here)
    if "gpt-3" in name or "gpt-4" in name:
        # with OpenAI you usually _don't_ build the prompt yourself,
        # but return something sensible anyway:
        return prompt.strip()

    # -------- 3 : last resort --------------------------------------------
    return prompt.strip()

# ───────────────────────── generator wrapper ────────────────────────────
def generate_response(prompt: str,
                      generator,
                      model_name: str,
                      tokenizer,
                      max_tokens: int = 512) -> str:
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    return generator(
        formatted,
        max_new_tokens=max_tokens,
        do_sample=False,
        return_full_text=False,
    )[0]["generated_text"]

# ───────────────────────────────── code‑extraction utility ─────────────────────────────



def extract_code(txt: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    code = m.group(1) if m else txt
    return textwrap.dedent(code).strip()  


# ─────────────────────────────── test‑code converters ─────────────────────────────────

SAFE_BUILTINS = {"sorted", "len", "sum", "min", "max", "any", "all"}


def convert_general_check_code_MBPP(test_code: str,
                                    func_name: str) -> tuple[str, int]:
    if func_name in SAFE_BUILTINS:                       # rare clash
        return test_code, test_code.count("assert")

    lines = [ln.rstrip() for ln in test_code.splitlines() if ln.lstrip().startswith("assert")]
    total = len(lines)

    body = ["def check(candidate):",
            "    passed = 0",
            f"    total  = {total}"]
    for ln in lines:
        ln = ln.lstrip()                                # fix indentation
        ln = re.sub(rf"\b{re.escape(func_name)}\s*\(",
                    "candidate(", ln, count=1)
        body += ["    try:",
                 f"        assert {ln[len('assert '):]}",  # keep assert body
                 "        passed += 1",
                 "    except AssertionError:",
                 "        pass"]
    body.append("    return passed, total")
    return "\n".join(body), total



# ─────────────────────────── HumanEval converter ──────────────────────────

def convert_general_check_code_HumanEval(test_code: str,
                                         func_name: str) -> tuple[str, int]:
    
    CHECK_RE = re.compile(r"^\s*def\s+check\s*\(\s*candidate\s*\)\s*:", re.M)

    # ── Case A – full harness present ────────────────────────────────────
    if CHECK_RE.search(test_code):
        n_tests = len(re.findall(r"^\s*assert\b", test_code, re.M))

        wrapper = textwrap.dedent(f"""
            # ------------- harness wrapper (auto-added) -------------
            _original_check = check
            def check(candidate):
                try:
                    _original_check(candidate)     # run original test suite
                    return {n_tests}, {n_tests}    # all passed
                except AssertionError:
                    return 0, {n_tests}            # at least one failed
        """)
        return test_code.rstrip() + "\n\n" + wrapper, n_tests

    # ── Case B – only plain asserts (very uncommon in HumanEval) ────────
    lines = [ln.rstrip() for ln in test_code.splitlines()
             if ln.lstrip().startswith("assert")]
    total = len(lines)

    body = ["def check(candidate):",
            "    passed = 0",
            f"    total  = {total}"]
    for ln in lines:
        ln = re.sub(rf"\b{re.escape(func_name)}\s*\(", "candidate(", ln, count=1)
        body += ["    try:",
                 f"        {ln.lstrip()[len('assert '):]}",
                 "        passed += 1",
                 "    except AssertionError:",
                 "        pass"]
    body.append("    return passed, total")
    return "\n".join(body), total



# ─────────────────────────────── sandbox execution utilities ──────────────────────────

class TimeoutException(Exception):
    """Raised by the signal handler when wall‑clock time is exceeded."""


def _timeout_handler(signum, frame):  # noqa: D401, pylint: disable=unused-argument
    raise TimeoutException("Timeout!")


signal.signal(signal.SIGALRM, _timeout_handler)


def expected_name(row, default: str = "solution") -> str:

    ep = (row.get("entry_point") or "").strip()   
    if ep:                                       
        return ep
    # MBPP: grab the first identifier right after "assert "
    m = re.match(r"\s*assert\s+(\w+)\s*\(", row["test"])
    if m:
        return m.group(1)

    # nothing found
    return default


# ───────────────────────────── sandbox runner patch ──────────────────────
def _safe_exec(candidate_code: str,
               check_code: str,
               queue: mp.Queue,
               entry_point: str | None = None) -> None:
    try:
        env: dict[str, Any] = {}
        exec(candidate_code, env)

        # --- entry-point ------------------------------------
        
        if entry_point:
            fn = env.get(entry_point)
            if not callable(fn):
                # try a case-insensitive match (model sometimes changes case)
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
            

        def run(fn) -> int:
            env['candidate'] = fn
            exec(check_code + "\n_result = check(candidate)", env)
            passed, _ = env["_result"]
            return passed

        queue.put((max(run(f) for f in fns), "OK"))

    except Exception as exc:
        queue.put((0, f"ERROR: {type(exc).__name__}: {exc}"))



def evaluate_with_timeout(
    candidate_code: str,
    check_code: str,
    *,
    timeout_seconds: int = 20,
    entry_point: str | None = None,
) -> Tuple[int, str]:
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


# ─────────────────────────────── dataset loop & evaluation ────────────────────────────


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    ext = p.suffix.lower()

    text = p.read_text("utf-8").lstrip("\ufeff").strip()   # BOM-safe

    if ext in {".json"}:                                   # tableau JSON
        return json.loads(text)

    if ext in {".jsonl", ".ndjson"}:                       # JSON-lines
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    if ext in {".csv"}:                                    # CSV classique
        # utilise pandas pour gérer les quotes / newlines
        return pd.read_csv(p).to_dict(orient="records")

    if ext in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t").to_dict(orient="records")

    raise ValueError(f"Inconnu : extension {ext} pour {path}")


def generate_from_dataset(args, gen, tokenizer):
    records = load_records(args.inputFile)[: args.limit]

    total = success_exec = pass1_true = false_pass = 0
    pbar = trange(len(records), desc="evaluating", ncols=80)

    for idx in pbar:
        row = records[idx]
        pbar.set_description(f"{row.get('task_id', idx)}")

        # ---------- choose the prompt text (prefer mutated_prompt) ----------
        row_prompt = (
            row.get("mutated_prompt")
            or row.get("prompt")
            or row.get("original_prompt")
            or row.get("prompt_text")
            or ""
        )
        if not row_prompt:
            logger.warning("No prompt found for %s", row.get("task_id", idx))

        # ---------- expected name (from entry_point / test) ----------
        default_name = expected_name(row)
        original_entry = (row.get("entry_point") or "").strip() or default_name

        # ---------- try to detect a function name inside the mutated prompt ----------
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", row_prompt)
        mutated_name = m.group(1) if m else None

        # Use mutated_name in the template if available else use original entry
        name_for_template = mutated_name or original_entry or default_name

        prompt = textwrap.dedent(f"""
            You are a senior Python developer.

            Task:
            {row_prompt}

            Write **one** function named `{name_for_template}` that solves the task.
            If helpers are needed, define them above the main function.

            **Use only the Python standard library and place every required `import …` statement at the very top of the code block.**

                    ⚠️ Return *only* valid Python code, wrapped in a single code block like this:
                    ```python
                    <your code here>
                    ```
                    """)

        try:
            # Generate and extract code
            response = generate_response(prompt, gen, args.modelName, tokenizer)
            code = extract_code(response)

            # Prepare the check harness
            if "mbpp" in args.inputFile.lower():
                check_code, n_tests = convert_general_check_code_MBPP(row["test"], name_for_template)
            else:   # HumanEval
                check_code, n_tests = convert_general_check_code_HumanEval(row["test"], name_for_template)

            # ---------- Robust evaluation: try multiple entry_point candidates ----------
            entry_candidates = []
            if mutated_name:
                entry_candidates.append(mutated_name)
            if original_entry and original_entry not in entry_candidates:
                entry_candidates.append(original_entry)
            entry_candidates.append(None)  # last resort: auto-detect

            passed = 0
            status = "ERROR: Not evaluated"
            for entry_candidate in entry_candidates:
                passed, status = evaluate_with_timeout(
                    code,
                    check_code,
                    timeout_seconds=args.timeout,
                    entry_point=entry_candidate,
                )

                if isinstance(status, str) and ("not found" in status or status.startswith("Function `")):
                    continue
                break

            # Logging / pretty output
            logger.info(
                "\n"
                "Task  : %s\n"
                "PromptUsed:\n%s\n"
                "Code  :\n%s\n"
                "Checks:\n%s\n"
                "Result: %d / %d tests passed   [%s]\n"
                "───────────────────────────────────────────────",
                row.get("task_id", idx),
                row_prompt,
                code,
                check_code,
                passed,
                n_tests,
                status
            )

            pass_at_1 = passed == n_tests and status == "OK"

        except Exception as exc:
            passed = 0
            status = f"ERROR: {type(exc).__name__}: {exc}"
            code = ""
            response = ""   # ← ADD THIS LINE
            n_tests = 0
            pass_at_1 = False
            logger.exception("Task %s failed during evaluation", row.get("task_id", idx))

        # Metrics bookkeeping
        total += 1
        if status == "OK":
            success_exec += 1
        if pass_at_1:
            pass1_true += 1
        if status == "OK" and not pass_at_1:
            false_pass += 1

        # Store results back into record
        row.update({
            "GeneratedCode": code,
            "GeneratedResponse": response,
            "PromptUsed": row_prompt,
            "TestCases": row.get("test", ""),
            "n_Tests": n_tests,
            "Tests_Passed": passed,
            "Pass@1": pass_at_1,
            "Eval_Status": status,
        })

    # Write detailed output
    output_path = Path(args.outputFile)
    output_path.write_text(json.dumps(records, indent=2), "utf-8")

    # Summary
    summary = {
        "Model": args.modelName,
        "Dataset": args.inputFile,
        "Samples": total,
        "SuccessExec": success_exec,
        "Pass@1_TRUE": pass1_true,
        "Runnable_FALSE": false_pass,
        "SuccessExecRate": round(success_exec / total, 3) if total else 0.0,
        "Pass@1_Rate": round(pass1_true / total, 3) if total else 0.0,
        "Runnable_FALSE_Rate": round(false_pass / total, 3) if total else 0.0,
    }

    return summary


# ─────────────────────────── NEW: memory cleanup helper ───────────────────────────────

def cleanup_model(model, generator) -> None:
    """Delete model and generator, free GPU and CPU memory."""
    import gc
    del generator
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Model unloaded and memory freed.")


# ───────────────────────────────────────────── main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM code generation on MBPP / HumanEval")
    parser.add_argument("--inputFiles",  nargs="+", default=["./mutations/humanEval_US_with_tests.jsonl", "./mutations/humanEval_lv_with_tests.jsonl", "./mutations/humanEval_SF_with_tests.jsonl"], help="one or more dataset paths")
    parser.add_argument("--outputDir",   default="./results", help="directory to write output files")
    parser.add_argument("--modelNames",  nargs="+", default=["mistralai/Devstral-Small-2-24B-Instruct-2512"], help="one or more HF model ids")
    parser.add_argument("--timeout",     type=int, default=50,   help="execution timeout")
    parser.add_argument("--limit",       type=int, default=1000, help="eval on first N samples")
    parser.add_argument("--seed",        type=int, default=42,   help="random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s n_gpu=%s", device, torch.cuda.device_count())

    all_summaries = []

    # ── outer loop: models ────────────────────────────────────────────────
    for model_name in args.modelNames:
        logger.info("=" * 60)
        logger.info("Loading model: %s", model_name)
        logger.info("=" * 60)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False,
            )
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            continue

        # ── inner loop: datasets ──────────────────────────────────────────
        for input_file in args.inputFiles:
            logger.info("-" * 60)
            logger.info("Dataset: %s", input_file)
            logger.info("-" * 60)

            # Build per-(model, dataset) output filename
            model_slug   = model_name.replace("/", "_")
            dataset_slug = Path(input_file).stem
            output_file  = output_dir / f"{model_slug}__{dataset_slug}.json"

            # Patch args for the existing generate_from_dataset function
            args.modelName  = model_name
            args.inputFile  = input_file
            args.outputFile = str(output_file)

            try:
                summary = generate_from_dataset(args, generator, tokenizer)
                all_summaries.append(summary)
                logger.info("Finished %s on %s → %s", model_name, input_file, output_file)
                logger.info("Summary: %s", summary)
            except Exception as exc:
                logger.error("Error evaluating %s on %s: %s", model_name, input_file, exc)

        # ── free memory before loading the next model ─────────────────────
        cleanup_model(model, generator)

    # ── final combined summary ────────────────────────────────────────────
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        print("\n" + df.to_string(index=False))

        summary_csv = output_dir / "all_results_summary.csv"
        df.to_csv(summary_csv, index=False)
        logger.info("Combined summary → %s", summary_csv)


if __name__ == "__main__":
    main()