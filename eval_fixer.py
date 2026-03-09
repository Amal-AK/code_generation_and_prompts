"""
eval_fixer.py
─────────────
Evaluate Pass@1 for the LoRA fixer on LV and SF mutations.

Three conditions per sample:
  baseline  – base model, LoRA OFF, mutated_prompt
  pipeline  – classifier → LoRA ON/OFF, mutated_prompt
  oracle    – base model, LoRA OFF, original_prompt  (upper bound)

Datasets:
  humaneval LV / SF
  mbpp      LV / SF
  lcb       LV / SF   ← out-of-distribution (not in training)

Usage:
    python eval_fixer.py \
        --classifierCkpt  lora_1b_output/best_lora_classifier.pt \
        --loraAdapterDir  finetune_lv_sf_output/best_lora_sft \
        --outputDir       eval_fixer_output \
        --device          cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from inference_pipeline import FixingPipeline, extract_code_block
from main_inference import (
    convert_general_check_code_HumanEval,
    convert_general_check_code_MBPP,
    evaluate_lcb_with_timeout,
    evaluate_with_timeout,
)

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASETS = [
    # (name,        mutation_file,                                  kind)
    ("humaneval_LV", "mutations/humanEval_lv_with_tests.jsonl",    "humaneval"),
    ("humaneval_SF", "mutations/humanEval_SF_with_tests.jsonl",    "humaneval"),
    ("humaneval_US", "mutations/HumanEval_US_with_tests.jsonl",    "humaneval"),
    ("mbpp_LV",      "mutations/mbpp_LV_with_tests.jsonl",         "mbpp"),
    ("mbpp_SF",      "mutations/mbpp_SF_with_tests.jsonl",         "mbpp"),
    ("mbpp_US",      "mutations/mbpp_US_with_tests.jsonl",         "mbpp"),
    ("lcb_LV",       "mutations/livecodebench_LV_with_tests.jsonl","lcb"),
    ("lcb_SF",       "mutations/livecodebench_SF_with_tests.jsonl","lcb"),
    ("lcb_US",       "mutations/livecodebench_US_with_tests.jsonl","lcb"),
]

LCB_ORIGINAL = "datasets/livecodebench/livecodebench_public.jsonl"

TIMEOUT = 20


def load_lcb_difficulty(data_dir: str) -> dict:
    """Returns {task_id: difficulty} from the original LCB dataset."""
    path = os.path.join(data_dir, LCB_ORIGINAL)
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            result[r["task_id"]] = r.get("difficulty", "unknown")
    return result


# ── Test runner ────────────────────────────────────────────────────────────────
def run_eval(code: str, row: dict, kind: str) -> bool:
    if kind == "humaneval":
        ep         = row["entry_point"]
        check_code, _ = convert_general_check_code_HumanEval(row["test"], ep)
        passed, _  = evaluate_with_timeout(code, check_code,
                                           timeout_seconds=TIMEOUT, entry_point=ep)
        return passed > 0

    if kind == "mbpp":
        test_str   = "\n".join(row["test_list"])
        # extract func name from first assertion
        import re
        m          = re.search(r"assert\s+(\w+)\s*\(", test_str)
        ep         = m.group(1) if m else "func"
        check_code, _ = convert_general_check_code_MBPP(test_str, ep)
        passed, _  = evaluate_with_timeout(code, check_code,
                                           timeout_seconds=TIMEOUT, entry_point=ep)
        return passed > 0

    if kind == "lcb":
        test_cases = json.loads(row["test"]) if isinstance(row["test"], str) else row["test"]
        passed, total, _ = evaluate_lcb_with_timeout(code, test_cases,
                                                     timeout_seconds=TIMEOUT)
        return passed == total and total > 0

    return False


# ── Single-condition generation ────────────────────────────────────────────────
def generate_baseline(pipeline: FixingPipeline, prompt: str) -> str:
    """Base model, LoRA OFF, raw prompt."""
    raw = pipeline._generate(prompt, use_lora=False)
    return extract_code_block(raw)


def generate_oracle(pipeline: FixingPipeline, original_prompt: str) -> str:
    """Base model, LoRA OFF, clean original prompt."""
    raw = pipeline._generate(original_prompt, use_lora=False)
    return extract_code_block(raw)


# ── Main eval loop ─────────────────────────────────────────────────────────────
def evaluate_dataset(
    pipeline: FixingPipeline,
    name: str,
    path: str,
    kind: str,
    output_dir: str,
    difficulty_filter: Optional[dict] = None,   # {task_id: difficulty}
    difficulty: Optional[str] = None,            # e.g. "easy"
    us_agent=None,                               # USRecoveryAgent or None
) -> dict:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("applicable", True):
                continue
            if difficulty and difficulty_filter:
                if difficulty_filter.get(r.get("task_id")) != difficulty:
                    continue
            records.append(r)

    is_us = name.endswith("_US")

    results = []
    for row in tqdm(records, desc=name, ncols=90):
        mutated   = row["mutated_prompt"]
        original  = row["original_prompt"]
        ep        = row.get("entry_point")

        # ── baseline ──────────────────────────────────────────────────────────
        code_base = generate_baseline(pipeline, mutated)
        pass_base = run_eval(code_base, row, kind)

        # ── pipeline ──────────────────────────────────────────────────────────
        pipe_out  = pipeline(mutated, entry_point=ep)
        code_pipe = pipe_out["code"]
        pass_pipe = run_eval(code_pipe, row, kind)

        # ── oracle ────────────────────────────────────────────────────────────
        code_ora  = generate_oracle(pipeline, original)
        pass_ora  = run_eval(code_ora, row, kind)

        # ── recovery agent (GPT-4o + code_interpreter) ────────────────────────
        pass_agent   = None
        fixed_prompt = None
        if us_agent is not None:
            agent_out    = us_agent.recover(mutated, row=row, kind=kind, entry_point=ep)
            pass_agent   = agent_out["passed"]
            fixed_prompt = agent_out["fixed_prompt"]

        rec = {
            "task_id":       row.get("task_id"),
            "mutation_type": row.get("mutation_type"),
            "clf_pred":      pipe_out["mutation_type"],
            "clf_conf":      round(pipe_out["confidence"], 4),
            "lora_used":     pipe_out["lora_used"],
            "pass_baseline": pass_base,
            "pass_pipeline": pass_pipe,
            "pass_oracle":   pass_ora,
        }
        if us_agent is not None:
            rec["pass_agent"]   = pass_agent
            rec["fixed_prompt"] = fixed_prompt

        results.append(rec)

    # ── summary ───────────────────────────────────────────────────────────────
    n          = len(results)
    base_p1    = sum(r["pass_baseline"] for r in results) / n if n else 0
    pipe_p1    = sum(r["pass_pipeline"] for r in results) / n if n else 0
    ora_p1     = sum(r["pass_oracle"]   for r in results) / n if n else 0
    lora_used  = sum(r["lora_used"]     for r in results)

    summary = {
        "dataset":          name,
        "n":                n,
        "lora_activated":   lora_used,
        "pass1_baseline":   round(base_p1, 4),
        "pass1_pipeline":   round(pipe_p1, 4),
        "pass1_oracle":     round(ora_p1,  4),
        "delta_vs_base":    round(pipe_p1 - base_p1, 4),
        "delta_vs_oracle":  round(pipe_p1 - ora_p1,  4),
    }

    if us_agent is not None:
        agent_p1 = sum(r["pass_agent"] for r in results) / n if n else 0
        summary["pass1_agent"]           = round(agent_p1, 4)
        summary["delta_agent_vs_base"]   = round(agent_p1 - base_p1, 4)
        summary["delta_agent_vs_oracle"] = round(agent_p1 - ora_p1,  4)

    # save per-record results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierCkpt",  required=True)
    parser.add_argument("--loraAdapterDir",  required=True)
    parser.add_argument("--genModel",        default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataDir",         default=".")
    parser.add_argument("--outputDir",       default="eval_fixer_output")
    parser.add_argument("--threshold",       type=float, default=0.75)
    parser.add_argument("--device",          default="cuda:0",
                        help="Device for classifier (gen model uses device_map=auto)")
    parser.add_argument("--datasets",        default="all",
                        help="Comma-separated subset, e.g. humaneval_LV,lcb_SF, or 'all'")
    parser.add_argument("--difficulty",      default=None,
                        help="Filter LCB by difficulty: easy | medium | hard")
    parser.add_argument("--gpt4Model",       default="gpt-4o",
                        help="OpenAI model for US recovery agent (default: gpt-4o)")
    parser.add_argument("--noAgent",         action="store_true",
                        help="Disable the GPT-4 agent for US datasets")
    args = parser.parse_args()

    pipeline = FixingPipeline.from_checkpoints(
        classifier_ckpt      = args.classifierCkpt,
        lora_adapter_dir     = args.loraAdapterDir,
        gen_model_name       = args.genModel,
        clf_device           = args.device,
        confidence_threshold = args.threshold,
    )

    us_agent = None
    if not args.noAgent:
        from recovery_agent import RecoveryAgent
        us_agent = RecoveryAgent(pipeline=pipeline, model=args.gpt4Model)
        print(f"Recovery agent (GPT-4o + code_interpreter): {args.gpt4Model}")

    requested = set(args.datasets.split(",")) if args.datasets != "all" else None
    diff_map  = load_lcb_difficulty(args.dataDir) if args.difficulty else None

    all_summaries = []
    for name, rel_path, kind in DATASETS:
        if requested and name not in requested:
            continue
        path = os.path.join(args.dataDir, rel_path)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        diff_filter = diff_map if kind == "lcb" else None
        summary = evaluate_dataset(pipeline, name, path, kind, args.outputDir,
                                   difficulty_filter=diff_filter,
                                   difficulty=args.difficulty if kind == "lcb" else None,
                                   us_agent=us_agent)
        all_summaries.append(summary)

        agent_str = ""
        if "pass1_agent" in summary:
            agent_str = (f"  agent={summary['pass1_agent']:.3f}"
                         f"  Δagent={summary['delta_agent_vs_base']:+.3f}"
                         f"  Δoracle={summary['delta_agent_vs_oracle']:+.3f}")
        print(
            f"\n{name:20s}  n={summary['n']:>4}  "
            f"baseline={summary['pass1_baseline']:.3f}  "
            f"pipeline={summary['pass1_pipeline']:.3f}  "
            f"oracle={summary['pass1_oracle']:.3f}  "
            f"Δbase={summary['delta_vs_base']:+.3f}"
            f"{agent_str}"
        )

    # save summary table
    summary_path = os.path.join(args.outputDir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSummary saved → {summary_path}")

    # print aggregate across all datasets
    if all_summaries:
        total_n    = sum(s["n"] for s in all_summaries)
        total_base = sum(s["pass1_baseline"] * s["n"] for s in all_summaries) / total_n
        total_pipe = sum(s["pass1_pipeline"] * s["n"] for s in all_summaries) / total_n
        total_ora  = sum(s["pass1_oracle"]   * s["n"] for s in all_summaries) / total_n
        print(
            f"\n{'OVERALL':20s}  n={total_n:>4}  "
            f"baseline={total_base:.3f}  "
            f"pipeline={total_pipe:.3f}  "
            f"oracle={total_ora:.3f}  "
            f"Δbase={total_pipe - total_base:+.3f}"
        )


if __name__ == "__main__":
    main()
