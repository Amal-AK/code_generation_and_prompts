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
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    variants_to_load = ["v1", "v2"] if data_variant == "combined" else [data_variant]

    seen_he: Set[tuple] = set()
    for variant in variants_to_load:
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
    for variant in variants_to_load:
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
        solution    = (
            f"{pair['original_prompt'].strip()}\n\n"
            f"```python\n{pair['solution_code'].strip()}\n```"
        )

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
            test_body = pair["test"]          # contains def check(candidate): ...
            script    = f"{code}\n\n{test_body}\n\ncheck({ep})\n"
        else:  # mbpp
            test_body = "\n".join(pair["test_list"])
            script    = f"{code}\n\n{test_body}\n"

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


@torch.no_grad()
def eval_pass1_on_val(
    model,
    tokenizer,
    val_pairs: List[Dict[str, Any]],
    input_device: torch.device,
    ckpt_path: Path,
    max_new_tokens: int = 512,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate pass@1 on the val split for:
      - baseline : adapter disabled  (base model)
      - adapter  : best saved adapter loaded from ckpt_path

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

    pairs = val_pairs[:max_samples] if max_samples else val_pairs
    model.eval()

    pass_base = pass_lora = n = 0

    for pair in tqdm(pairs, desc="Pass@1 eval", ncols=90):
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
    parser.add_argument("--lr",            type=float, default=2e-4)
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
                        choices=["v1", "v2", "combined"],
                        help="Mutation data variant: v1 (original), v2 (variant2/), combined (both)")
    parser.add_argument("--evalSamples",   type=int, default=None,
                        help="Max val samples for pass@1 eval after training (default: all)")
    parser.add_argument("--skipEval",      action="store_true",
                        help="Skip pass@1 evaluation after training")
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

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading SFT pairs (dataVariant=%s)...", args.dataVariant)
    pairs = load_sft_pairs(data_dir, mutation_types, data_variant=args.dataVariant)
    logger.info("Total pairs: %d", len(pairs))

    # Split by task_id so no problem appears in both train and val
    # (prevents data leakage when multiple variants of the same task exist)
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
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    ckpt_path     = out_dir / "best_lora_sft"
    no_improve    = 0

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
        marker         = " ***" if val_loss < best_val_loss else ""

        logger.info(
            "Epoch %2d | train_loss=%.4f | val_loss=%.4f%s",
            epoch, avg_train_loss, val_loss, marker,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            logger.info("Saved best adapter → %s", ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping (patience=%d)", args.patience)
                break

    logger.info("Done. Best val loss: %.4f", best_val_loss)
    logger.info("Adapter saved at: %s", ckpt_path)

    # ── Save training metadata ─────────────────────────────────────────────────
    meta = {
        "model_name":     args.modelName,
        "mutation_types": sorted(mutation_types),
        "data_variant":   args.dataVariant,
        "train_pairs":    len(train_pairs),
        "val_pairs":      len(val_pairs),
        "best_val_loss":  best_val_loss,
        "lora_r":         args.loraR,
        "lora_alpha":     args.loraAlpha,
        "epochs_run":     epoch,
        "lr":             args.lr,
        "eff_batch_size": args.batchSize * args.gradAccum,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Metadata → %s/training_meta.json", out_dir)

    # ── Pass@1 evaluation on val set ───────────────────────────────────────────
    if not args.skipEval:
        logger.info("Running pass@1 evaluation on val set (%s samples)...",
                    args.evalSamples or "all")
        pass1_results = eval_pass1_on_val(
            model          = model,
            tokenizer      = tokenizer,
            val_pairs      = val_pairs,
            input_device   = input_device,
            ckpt_path      = ckpt_path,
            max_new_tokens = 512,
            max_samples    = args.evalSamples,
        )
        meta["pass1_eval"] = pass1_results
        (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"\nPass@1 on val set (n={pass1_results['n_evaluated']}):\n"
            f"  baseline : {pass1_results['pass1_baseline']:.3f}\n"
            f"  adapter  : {pass1_results['pass1_adapter']:.3f}\n"
            f"  Δ        : {pass1_results['delta']:+.3f}"
        )


if __name__ == "__main__":
    main()
