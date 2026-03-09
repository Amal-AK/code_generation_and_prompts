"""
eval_agent_only.py
──────────────────
Standalone evaluation of the RecoveryAgent on a single mutation dataset.

Flow per sample:
  mutated prompt
      → GPT-4o + code_interpreter (recovers missing constraint → fixed prompt)
      → local model (generates code from fixed prompt)
      → real hidden tests → pass/fail

No classifier, no LoRA routing, no baseline/oracle conditions.

Usage:
    python eval_agent_only.py \
        --mutationFile  mutations/HumanEval_US_with_tests.jsonl \
        --kind          humaneval \
        --genModel      Qwen/Qwen2.5-Coder-7B-Instruct \
        --outputDir     eval_agent_output \
        --gpt4Model     gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from eval_fixer import run_eval


# ── Minimal generator (no LoRA, no classifier) ────────────────────────────────

class SimpleGenerator:
    """Wraps a causal LM for code generation. Matches the pipeline._generate interface."""

    def __init__(self, model_name: str, max_new_tokens: int = 512):
        print(f"Loading generation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype  = torch.float16,
            device_map   = "auto",
            trust_remote_code = True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print("  Model loaded.")

    def _generate(self, instruction: str, use_lora: bool = False) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                pad_token_id   = self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


# ── Helpers (mirrors inference_pipeline) ──────────────────────────────────────

def extract_code_block(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_func_name(prompt: str):
    m = re.search(r"\bdef\s+(\w+)\s*\(", prompt)
    return m.group(1) if m else None


def build_instruction(prompt: str, func_name) -> str:
    from inference_pipeline import build_lora_instruction
    return build_lora_instruction(prompt, func_name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutationFile", default="mutations/HumanEval_US_with_tests.jsonl")
    parser.add_argument("--kind",         default="humaneval",
                        choices=["humaneval", "mbpp", "lcb"])
    parser.add_argument("--genModel",     default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--outputDir",    default="eval_agent_output")
    parser.add_argument("--gpt4Model",    default="gpt-4o")
    parser.add_argument("--maxNewTokens", type=int, default=512)
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    records = []
    with open(args.mutationFile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("applicable", True):
                records.append(r)
    print(f"Loaded {len(records)} samples from {args.mutationFile}")

    # ── Load local model ──────────────────────────────────────────────────────
    generator = SimpleGenerator(args.genModel, max_new_tokens=args.maxNewTokens)

    # ── Load agent ────────────────────────────────────────────────────────────
    from recovery_agent import RecoveryAgent
    agent = RecoveryAgent(pipeline=generator, model=args.gpt4Model)
    print(f"Recovery agent: {args.gpt4Model}\n")

    # ── Eval loop ─────────────────────────────────────────────────────────────
    os.makedirs(args.outputDir, exist_ok=True)
    dataset_name = Path(args.mutationFile).stem
    out_path = os.path.join(args.outputDir, f"{dataset_name}_agent.jsonl")

    # Resume from existing output
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["task_id"])
        print(f"Resuming — {len(done_ids)} already done")

    results = []
    passed_total = 0

    with open(out_path, "a") as fout:
        for row in tqdm(records, desc=dataset_name, ncols=90):
            task_id = row.get("task_id")
            if task_id in done_ids:
                continue

            mutated = row["mutated_prompt"]
            ep      = row.get("entry_point")

            try:
                agent_out    = agent.recover(mutated, row=row, kind=args.kind, entry_point=ep)
                pass_agent   = agent_out["passed"]
                fixed_prompt = agent_out["fixed_prompt"]
            except Exception as e:
                print(f"\n  [ERROR] {task_id}: {e}")
                pass_agent   = False
                fixed_prompt = mutated

            if pass_agent:
                passed_total += 1

            rec = {
                "task_id":      task_id,
                "mutation_type": row.get("mutation_type"),
                "pass_agent":   pass_agent,
                "fixed_prompt": fixed_prompt,
            }
            results.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    pass1 = passed_total / n if n else 0
    summary = {
        "dataset":     dataset_name,
        "n":           n,
        "pass1_agent": round(pass1, 4),
    }
    summary_path = os.path.join(args.outputDir, f"{dataset_name}_agent_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{dataset_name}  n={n}  pass@1={pass1:.3f}")
    print(f"Results → {out_path}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
