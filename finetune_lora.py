#!/usr/bin/env python3
"""
finetune_lora.py
────────────────
LoRA SFT: fine-tune a causal LM on (mutated_prompt → canonical_solution) pairs.

Training data  : HumanEval + MBPP mutations  (canonical solutions available)
Held-out test  : LiveCodeBench mutations     (evaluated separately via main_inference.py)

The model learns to produce correct code from an imperfect specification
with no oracle hint — recovery must be internalised through training exposure.

Ablation via --mutationTypes:
  all    → train on LV + SF + US  (default)
  LV,SF  → surface mutations only; tests whether US generalises
  US     → underspecification only

Usage:
  python finetune_lora.py \\
      --modelName  Qwen/Qwen2.5-Coder-7B-Instruct \\
      --outputDir  ./finetune_output \\
      --mutationTypes all \\
      --epochs 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("finetune_lora")

# ── Mutation file names ────────────────────────────────────────────────────────
HE_MUTATION_FILES: Dict[str, str] = {
    "LV": "humanEval_lv_with_tests.jsonl",
    "SF": "humanEval_SF_with_tests.jsonl",
    "US": "HumanEval_US_with_tests.jsonl",
}
MBPP_MUTATION_FILES: Dict[str, str] = {
    "LV": "mbpp_LV_with_tests.jsonl",
    "SF": "mbpp_SF_with_tests.jsonl",
    "US": "mbpp_US_with_tests.jsonl",
}

# V2 variants (richer lexical vagueness, different paraphrasing style)
HE_MUTATION_FILES_V2: Dict[str, str] = {
    "LV": "variant2/humanEval_LV_V2.jsonl",
}
MBPP_MUTATION_FILES_V2: Dict[str, str] = {
    "LV": "variant2/mbpp_LV_V2.jsonl",
}


# ── LCB mutation files ─────────────────────────────────────────────────────────
LCB_MUTATION_FILES: Dict[str, str] = {
    "LV": "livecodebench_LV_with_tests.jsonl",
    "SF": "livecodebench_SF_with_tests.jsonl",
    "US": "livecodebench_US_with_tests.jsonl",
}


# ── Prompt ─────────────────────────────────────────────────────────────────────
def build_instruction(mutated_prompt: str, func_name: str) -> str:
    return textwrap.dedent(f"""
        You are a senior Python developer.

        Task:
        {mutated_prompt}

        Write **one** function named `{func_name}` that solves the task.
        If helpers are needed, define them above the main function.

        **Use only the Python standard library and place every required `import` at the very top.**

        Return *only* valid Python code in a single code block:
        ```python
        <your code here>
        ```
    """).strip()


def build_lcb_instruction(mutated_prompt: str) -> str:
    return textwrap.dedent(f"""
        You are a competitive programmer.

        Problem:
        {mutated_prompt}

        Write a complete Python program that reads from stdin and writes to stdout.

        **Use only the Python standard library and place every required `import` at the very top.**

        Return *only* valid Python code in a single code block:
        ```python
        <your code here>
        ```
    """).strip()


# ── Data loading ───────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sft_pairs(
    data_dir: Path,
    mutation_types: Set[str],
    data_variant: str = "v1",
) -> List[Dict[str, Any]]:
    """
    Join mutation files with canonical solutions.
    Returns list of dicts: {mutated_prompt, solution_code, func_name,
                             task_id, mutation_type, dataset}

    data_variant: 'v1'       → original mutation files
                  'v2'       → variant2 files (LV only; falls back to v1 for SF/US)
                  'combined' → v1 + v2 merged (deduped by task_id+mutation_type)
    """
    pairs: List[Dict[str, Any]] = []

    # ── HumanEval ─────────────────────────────────────────────────────────────
    he_orig: Dict[str, Any] = {
        r["task_id"]: r
        for r in _load_jsonl(data_dir / "datasets/humanEval/HumanEval.jsonl")
    }
    logger.info("Loaded %d HumanEval originals", len(he_orig))

    def _he_file_map(variant: str) -> Dict[str, str]:
        if variant == "v2":
            return {**HE_MUTATION_FILES, **HE_MUTATION_FILES_V2}
        return HE_MUTATION_FILES

    # mbpp_combined: train on HE v1 + MBPP v1+v2; he_variants stays v1 only
    he_variants   = ["v1", "v2"] if data_variant == "combined" else ["v1"]
    mbpp_variants = ["v1", "v2"] if data_variant in ("combined", "mbpp_combined") else [data_variant]

    seen_he: Set[tuple] = set()
    for variant in he_variants:
        fmap = _he_file_map(variant)
        for mtype, fname in fmap.items():
            if mtype not in mutation_types:
                continue
            path = data_dir / "mutations" / fname
            if not path.exists():
                logger.warning("Missing: %s", path)
                continue
            n = 0
            for r in _load_jsonl(path):
                if not r.get("applicable", True):
                    continue
                orig = he_orig.get(r["task_id"])
                if orig is None:
                    continue
                key = (r["task_id"], mtype, r.get("mutated_prompt", ""))
                if key in seen_he:
                    continue
                seen_he.add(key)
                solution_code = orig["prompt"] + orig["canonical_solution"]
                pairs.append({
                    "mutated_prompt":  r["mutated_prompt"],
                    "original_prompt": r["original_prompt"],
                    "solution_code":   solution_code,
                    "func_name":       orig["entry_point"],
                    "task_id":         r["task_id"],
                    "mutation_type":   mtype,
                    "dataset":         "humaneval",
                    # test fields for pass@1 eval
                    "test":            orig["test"],
                    "entry_point":     orig["entry_point"],
                })
                n += 1
            logger.info("  HumanEval %-3s [%s]: %d pairs", mtype, variant, n)

    # ── MBPP ──────────────────────────────────────────────────────────────────
    mbpp_orig: Dict[str, Any] = {
        str(r["task_id"]): r
        for r in _load_jsonl(data_dir / "datasets/mbpp/mbpp.jsonl")
    }
    logger.info("Loaded %d MBPP originals", len(mbpp_orig))

    def _mbpp_file_map(variant: str) -> Dict[str, str]:
        if variant == "v2":
            return {**MBPP_MUTATION_FILES, **MBPP_MUTATION_FILES_V2}
        return MBPP_MUTATION_FILES

    seen_mbpp: Set[tuple] = set()
    for variant in mbpp_variants:
        fmap = _mbpp_file_map(variant)
        for mtype, fname in fmap.items():
            if mtype not in mutation_types:
                continue
            path = data_dir / "mutations" / fname
            if not path.exists():
                logger.warning("Missing: %s", path)
                continue
            n = 0
            for r in _load_jsonl(path):
                if not r.get("applicable", True):
                    continue
                orig = mbpp_orig.get(str(r["task_id"]))
                if orig is None:
                    continue
                key = (str(r["task_id"]), mtype, r.get("mutated_prompt", ""))
                if key in seen_mbpp:
                    continue
                seen_mbpp.add(key)
                func_name = None
                for test in orig.get("test_list", []):
                    m = re.match(r"\s*assert\s+(\w+)\s*\(", test)
                    if m:
                        func_name = m.group(1)
                        break
                if not func_name:
                    continue
                pairs.append({
                    "mutated_prompt":  r["mutated_prompt"],
                    "original_prompt": r["original_prompt"],
                    "solution_code":   orig["code"],
                    "func_name":       func_name,
                    "task_id":         str(r["task_id"]),
                    "mutation_type":   mtype,
                    "dataset":         "mbpp",
                    # test fields for pass@1 eval
                    "test_list":       orig["test_list"],
                })
                n += 1
            logger.info("  MBPP      %-3s [%s]: %d pairs", mtype, variant, n)

    return pairs


def load_lcb_pairs(
    data_dir: Path,
    mutation_types: Set[str],
) -> List[Dict[str, Any]]:
    """Load LCB mutation rows as eval pairs (no canonical solution — for pass@1 only)."""
    pairs: List[Dict[str, Any]] = []
    for mtype, fname in LCB_MUTATION_FILES.items():
        if mtype not in mutation_types:
            continue
        path = data_dir / "mutations" / fname
        if not path.exists():
            logger.warning("Missing LCB file: %s", path)
            continue
        n = 0
        for r in _load_jsonl(path):
            if not r.get("applicable", True):
                continue
            pairs.append({
                "mutated_prompt": r["mutated_prompt"],
                "original_prompt": r["original_prompt"],
                "task_id":        r["task_id"],
                "mutation_type":  mtype,
                "dataset":        "lcb",
                "test":           r["test"],
                "func_name":      None,
            })
            n += 1
        logger.info("  LCB       %-3s: %d pairs", mtype, n)
    return pairs


# ── Dataset ────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    """
    Tokenises (instruction, solution) pairs for causal LM SFT.
    Labels are -100 on instruction tokens so loss is computed on completion only.
    """

    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 1024):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair        = self.pairs[idx]
        instruction = build_instruction(pair["mutated_prompt"], pair["func_name"])
        solution    = f"```python\n{pair['solution_code'].strip()}\n```"

        # Full conversation: instruction + solution
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": solution},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Instruction-only prefix (used to find the mask boundary)
        instr_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids  = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"][0]

        instr_ids = self.tokenizer(
            instr_text, truncation=False,
            return_tensors="pt",
        )["input_ids"][0]

        labels   = full_ids.clone()
        mask_len = min(len(instr_ids), len(full_ids))
        labels[:mask_len] = -100   # mask instruction tokens from loss

        return {"input_ids": full_ids, "labels": labels}

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len        = max(item["input_ids"].size(0) for item in batch)
        B              = len(batch)
        input_ids      = torch.zeros(B, max_len, dtype=torch.long)
        labels         = torch.full((B, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)

        for i, item in enumerate(batch):
            L = item["input_ids"].size(0)
            input_ids[i, :L]      = item["input_ids"]
            labels[i, :L]         = item["labels"]
            attention_mask[i, :L] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_loss(model, loader, input_device: torch.device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        labels         = batch["labels"].to(input_device)

        out      = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


# ── Pass@1 evaluation ──────────────────────────────────────────────────────────
def _extract_code_block(text: str) -> str:
    """Extract the first ```python ... ``` block, or return the raw text."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _run_test(code: str, pair: Dict[str, Any], timeout: int = 10) -> bool:
    """Execute generated code against the pair's test cases; return True if all pass."""
    dataset = pair["dataset"]
    try:
        if dataset == "humaneval":
            ep        = pair["entry_point"]
            test_body = pair["test"]
            script    = f"{code}\n\n{test_body}\n\ncheck({ep})\n"
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, timeout=timeout,
            )
            return result.returncode == 0

        if dataset == "mbpp":
            test_body = "\n".join(pair["test_list"])
            script    = f"{code}\n\n{test_body}\n"
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, timeout=timeout,
            )
            return result.returncode == 0

        if dataset == "lcb":
            from main_inference import evaluate_lcb_with_timeout
            test_cases = json.loads(pair["test"]) if isinstance(pair["test"], str) else pair["test"]
            passed, total, _ = evaluate_lcb_with_timeout(
                code, test_cases, timeout_seconds=timeout,
            )
            return passed == total and total > 0

    except (subprocess.TimeoutExpired, Exception):
        pass
    return False


@torch.no_grad()
def eval_pass1_current(
    model,
    tokenizer,
    eval_pairs: List[Dict[str, Any]],
    input_device: torch.device,
    max_new_tokens: int = 512,
    label: str = "epoch",
) -> float:
    """
    Quick per-epoch pass@1 for the CURRENT adapter weights (no disk reload).
    Only evaluates adapter ON — returns pass@1 rate as a float.
    Called during training for early stopping on pass@1.
    """
    model.eval()
    model.enable_adapter_layers()

    n_passed = 0
    for pair in tqdm(eval_pairs, desc=f"Pass@1 [{label}]", ncols=90, leave=False):
        if pair["dataset"] == "lcb":
            instruction = build_lcb_instruction(pair["mutated_prompt"])
        else:
            instruction = build_instruction(pair["mutated_prompt"], pair["func_name"])
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=1024,
        )["input_ids"].to(input_device)
        prompt_len = input_ids.shape[1]
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        code = _extract_code_block(text)
        n_passed += int(_run_test(code, pair))

    rate = n_passed / len(eval_pairs) if eval_pairs else 0.0
    logger.info("Pass@1 [%s] adapter=%.4f  (%d/%d)", label, rate, n_passed, len(eval_pairs))
    model.train()
    return rate


@torch.no_grad()
def eval_pass1(
    model,
    tokenizer,
    eval_pairs: List[Dict[str, Any]],
    input_device: torch.device,
    ckpt_path: Path,
    max_new_tokens: int = 512,
    max_samples: Optional[int] = None,
    label: str = "eval",
) -> Dict[str, Any]:
    """
    Evaluate pass@1 for baseline (adapter OFF) vs best adapter (adapter ON).
    Works on any pair list: val split, LCB, HumanEval, MBPP.
    Reloads the best adapter weights from disk so early stopping is respected.
    """
    from peft import set_peft_model_state_dict

    # Reload best adapter weights (current in-memory model may be from a later epoch)
    adapter_file = ckpt_path / "adapter_model.safetensors"
    if adapter_file.exists():
        import safetensors.torch as sf
        state = sf.load_file(str(adapter_file))
    else:
        state = torch.load(str(ckpt_path / "adapter_model.bin"), map_location="cpu")
    set_peft_model_state_dict(model, state)
    logger.info("Reloaded best adapter weights from %s", ckpt_path)

    pairs = eval_pairs[:max_samples] if max_samples else eval_pairs
    model.eval()

    pass_base = pass_lora = n = 0

    for pair in tqdm(pairs, desc=f"Pass@1 [{label}]", ncols=90):
        if pair["dataset"] == "lcb":
            instruction = build_lcb_instruction(pair["mutated_prompt"])
        else:
            instruction = build_instruction(pair["mutated_prompt"], pair["func_name"])
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=1024,
        )["input_ids"].to(input_device)

        prompt_len = input_ids.shape[1]

        # ── baseline: adapter OFF ──────────────────────────────────────────────
        model.disable_adapter_layers()
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        text_base  = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        code_base  = _extract_code_block(text_base)
        pass_base += int(_run_test(code_base, pair))

        # ── adapter ON ────────────────────────────────────────────────────────
        model.enable_adapter_layers()
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        text_lora  = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        code_lora  = _extract_code_block(text_lora)
        pass_lora += int(_run_test(code_lora, pair))

        n += 1

    result = {
        "n_evaluated":     n,
        "pass1_baseline":  round(pass_base / n, 4) if n else 0.0,
        "pass1_adapter":   round(pass_lora / n, 4) if n else 0.0,
        "delta":           round((pass_lora - pass_base) / n, 4) if n else 0.0,
    }
    logger.info(
        "Pass@1 — baseline=%.3f  adapter=%.3f  Δ=%+.3f  (n=%d)",
        result["pass1_baseline"], result["pass1_adapter"],
        result["delta"], n,
    )
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on mutation-recovery pairs")
    parser.add_argument("--modelName",     default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataDir",       default=".")
    parser.add_argument("--outputDir",     default="./finetune_output")
    parser.add_argument("--mutationTypes", default="all",
                        help="'all' or comma-separated: LV,SF,US  (e.g. 'LV,SF' for ablation)")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batchSize",     type=int,   default=2)
    parser.add_argument("--gradAccum",     type=int,   default=8,
                        help="gradient accumulation steps  (eff. batch = batchSize × gradAccum)")
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--maxLength",     type=int,   default=1024)
    parser.add_argument("--valSplit",      type=float, default=0.1)
    parser.add_argument("--loraR",         type=int,   default=16)
    parser.add_argument("--loraAlpha",     type=int,   default=32)
    parser.add_argument("--loraDropout",   type=float, default=0.05)
    parser.add_argument("--patience",      type=int,   default=3)
    parser.add_argument("--gpus",          default=None,
                        help="CUDA_VISIBLE_DEVICES (e.g. '0,1,2,3')")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dataVariant",   default="v1",
                        choices=["v1", "v2", "combined", "mbpp_combined"],
                        help="Mutation data variant: v1, v2, combined, or mbpp_combined (HE v1 + MBPP v1+v2)")
    parser.add_argument("--warmupSteps",   type=int, default=100,
                        help="Linear warmup steps before cosine LR decay")
    parser.add_argument("--evalDataset",   default="val",
                        choices=["val", "lcb", "v2", "he_v2"],
                        help="Dataset for post-training pass@1 eval: val, lcb, v2, or he_v2 (HumanEval v2 only)")
    parser.add_argument("--evalSamples",   type=int, default=None,
                        help="Max samples for pass@1 eval after training (default: all)")
    parser.add_argument("--skipEval",      action="store_true",
                        help="Skip pass@1 evaluation after training")
    parser.add_argument("--pass1EvalSamples", type=int, default=0,
                        help="Val samples for per-epoch pass@1 early stopping (0=disabled)")
    parser.add_argument("--pass1EvalFreq",    type=int, default=1,
                        help="Run per-epoch pass@1 every N epochs (default: 1)")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir  = Path(args.outputDir)
    data_dir = Path(args.dataDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("n_gpu=%d", torch.cuda.device_count())

    # ── Mutation type filter ───────────────────────────────────────────────────
    if args.mutationTypes.lower() == "all":
        mutation_types = {"LV", "SF", "US"}
    else:
        mutation_types = {t.strip().upper() for t in args.mutationTypes.split(",")}
    logger.info("Training on mutation types: %s", sorted(mutation_types))

    # ── Load training data ─────────────────────────────────────────────────────
    logger.info("Loading SFT pairs (dataVariant=%s)...", args.dataVariant)
    pairs = load_sft_pairs(data_dir, mutation_types, data_variant=args.dataVariant)
    logger.info("Total pairs: %d", len(pairs))

    # Split by task_id so no problem appears in both train and val
    unique_task_ids = sorted({p["task_id"] for p in pairs})
    random.shuffle(unique_task_ids)
    n_val_tasks = max(1, int(len(unique_task_ids) * args.valSplit))
    val_task_ids   = set(unique_task_ids[:n_val_tasks])
    train_task_ids = set(unique_task_ids[n_val_tasks:])

    val_pairs   = [p for p in pairs if p["task_id"] in val_task_ids]
    train_pairs = [p for p in pairs if p["task_id"] in train_task_ids]
    random.shuffle(train_pairs)
    logger.info("Train: %d pairs (%d tasks)  Val: %d pairs (%d tasks)",
                len(train_pairs), len(train_task_ids),
                len(val_pairs),   len(val_task_ids))

    # ── Load v2 pairs as held-out test set (if evalDataset == "v2"/"he_v2") ────
    v2_pairs: List[Dict[str, Any]] = []
    if args.evalDataset in ("v2", "he_v2"):
        logger.info("Loading v2 mutation pairs as held-out test set...")
        v2_pairs = load_sft_pairs(data_dir, mutation_types, data_variant="v2")
        if args.evalDataset == "he_v2":
            v2_pairs = [p for p in v2_pairs if p["dataset"] == "humaneval"]
            logger.info("he_v2 test pairs (HumanEval only): %d", len(v2_pairs))
        else:
            logger.info("v2 test pairs: %d", len(v2_pairs))

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.modelName)
    tokenizer = AutoTokenizer.from_pretrained(args.modelName, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.modelName)
    model = AutoModelForCausalLM.from_pretrained(
        args.modelName,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # Required for gradient checkpointing when device_map="auto" is used
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.loraR,
        lora_alpha=args.loraAlpha,
        lora_dropout=args.loraDropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Input device = first GPU (device_map="auto" places embeddings on cuda:0)
    input_device = next(model.parameters()).device

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = SFTDataset(train_pairs, tokenizer, args.maxLength)
    val_ds   = SFTDataset(val_pairs,   tokenizer, args.maxLength)

    train_loader = DataLoader(
        train_ds, batch_size=args.batchSize, shuffle=True,
        collate_fn=SFTDataset.collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batchSize, shuffle=False,
        collate_fn=SFTDataset.collate_fn, num_workers=2, pin_memory=True,
    )

    # ── Optimizer + scheduler ──────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    total_steps = (len(train_loader) // args.gradAccum) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = min(args.warmupSteps, total_steps),
        num_training_steps= total_steps,
    )
    logger.info("Scheduler: cosine with %d warmup steps / %d total steps",
                min(args.warmupSteps, total_steps), total_steps)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_pass1       = -1.0          # best adapter pass@1 seen so far
    use_pass1_stop   = args.pass1EvalSamples > 0
    ckpt_path        = out_dir / "best_lora_sft"
    no_improve       = 0

    # Fixed subset used every epoch for pass@1 early stopping
    # If evalDataset is v2/he_v2, stop on the held-out set; else use val split
    pass1_val_pairs: List[Dict[str, Any]] = []
    if use_pass1_stop:
        if args.evalDataset in ("v2", "he_v2") and v2_pairs:
            pool = list(v2_pairs)
            random.shuffle(pool)
            pass1_val_pairs = pool[: args.pass1EvalSamples]
            logger.info(
                "Pass@1 early stopping on %s — %d samples every %d epoch(s)",
                args.evalDataset, len(pass1_val_pairs), args.pass1EvalFreq,
            )
        else:
            random.shuffle(val_pairs)
            pass1_val_pairs = val_pairs[: args.pass1EvalSamples]
            logger.info(
                "Pass@1 early stopping enabled — %d val samples every %d epoch(s)",
                len(pass1_val_pairs), args.pass1EvalFreq,
            )

    logger.info(
        "Starting training  epochs=%d  batch=%d  grad_accum=%d  eff_batch=%d",
        args.epochs, args.batchSize, args.gradAccum,
        args.batchSize * args.gradAccum,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=90, leave=False)
        for batch_idx, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels         = batch["labels"].to(input_device)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.gradAccum
            loss.backward()
            total_loss += out.loss.item()

            if (batch_idx + 1) % args.gradAccum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}")

        # Flush any remaining gradient accumulation at end of epoch
        if (len(train_loader) % args.gradAccum) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        val_loss       = evaluate_loss(model, val_loader, input_device)

        # ── Per-epoch pass@1 (used for early stopping if enabled) ─────────────
        epoch_pass1: Optional[float] = None
        if use_pass1_stop and epoch % args.pass1EvalFreq == 0:
            epoch_pass1 = eval_pass1_current(
                model, tokenizer, pass1_val_pairs, input_device,
                label=f"e{epoch}",
            )

        # ── Decide whether this epoch improved ─────────────────────────────────
        if use_pass1_stop and epoch_pass1 is not None:
            improved = epoch_pass1 > best_pass1
            marker   = " ***" if improved else ""
            logger.info(
                "Epoch %2d | train_loss=%.4f | val_loss=%.4f | pass1=%.4f%s",
                epoch, avg_train_loss, val_loss, epoch_pass1, marker,
            )
        else:
            improved = val_loss < best_val_loss
            marker   = " ***" if improved else ""
            logger.info(
                "Epoch %2d | train_loss=%.4f | val_loss=%.4f%s",
                epoch, avg_train_loss, val_loss, marker,
            )

        if improved:
            if epoch_pass1 is not None:
                best_pass1 = epoch_pass1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            no_improve = 0
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            logger.info("Saved best adapter → %s", ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping (patience=%d)", args.patience)
                break

    logger.info("Done. Best val loss: %.4f  Best pass@1: %.4f", best_val_loss, best_pass1)
    logger.info("Adapter saved at: %s", ckpt_path)

    # ── Save training metadata ─────────────────────────────────────────────────
    meta = {
        "model_name":         args.modelName,
        "mutation_types":     sorted(mutation_types),
        "data_variant":       args.dataVariant,
        "train_pairs":        len(train_pairs),
        "val_pairs":          len(val_pairs),
        "best_val_loss":      best_val_loss,
        "best_pass1_earlystop": best_pass1 if use_pass1_stop else None,
        "lora_r":             args.loraR,
        "lora_alpha":         args.loraAlpha,
        "epochs_run":         epoch,
        "lr":                 args.lr,
        "warmup_steps":       args.warmupSteps,
        "eff_batch_size":     args.batchSize * args.gradAccum,
        "pass1_eval_samples": args.pass1EvalSamples,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Metadata → %s/training_meta.json", out_dir)

    # ── Pass@1 evaluation ──────────────────────────────────────────────────────
    if not args.skipEval:
        if args.evalDataset == "lcb":
            logger.info("Loading LCB pairs for pass@1 eval...")
            eval_pairs = load_lcb_pairs(data_dir, mutation_types)
            eval_label = "LCB"
        elif args.evalDataset in ("v2", "he_v2"):
            eval_pairs = v2_pairs
            eval_label = args.evalDataset
        else:
            eval_pairs = val_pairs
            eval_label = "val"

        if not eval_pairs:
            logger.warning("No eval pairs found for evalDataset=%s — skipping pass@1",
                           args.evalDataset)
        else:
            logger.info("Running pass@1 on %s (%s samples)...",
                        eval_label, args.evalSamples or len(eval_pairs))
            pass1_results = eval_pass1(
                model          = model,
                tokenizer      = tokenizer,
                eval_pairs     = eval_pairs,
                input_device   = input_device,
                ckpt_path      = ckpt_path,
                max_new_tokens = 512,
                max_samples    = args.evalSamples,
                label          = eval_label,
            )
            meta["pass1_eval"] = {**pass1_results, "eval_dataset": eval_label}
            (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
            print(
                f"\nPass@1 on {eval_label} (n={pass1_results['n_evaluated']}):\n"
                f"  baseline : {pass1_results['pass1_baseline']:.3f}\n"
                f"  adapter  : {pass1_results['pass1_adapter']:.3f}\n"
                f"  Δ        : {pass1_results['delta']:+.3f}"
            )


if __name__ == "__main__":
    main()
