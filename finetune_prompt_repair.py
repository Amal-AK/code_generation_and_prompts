#!/usr/bin/env python3
"""
finetune_prompt_repair.py
─────────────────────────
LoRA SFT: fine-tune a small model on (mutated_prompt → original_prompt).

The model learns to REPAIR a defective specification — not to generate code.
At inference time the repaired prompt is fed into any off-the-shelf code model.

Training data (--trainOn): lcb, apps, mbpp, he  (comma-separated, default: lcb,apps)
Held-out test (--evalHeldOut): he_v1, lcb, val

Mutation types (--mutationTypes):
  all   → LV + SF + US
  LV    → lexical vagueness only  (default)
  LV,SF → surface defects only

Evaluation (after training):
  1. Repair quality : BLEU-4 between repaired and original prompt
  2. End-to-end     : repaired prompts saved as JSONL → run through
                      main_inference.py with any code model

Usage:
  python finetune_prompt_repair.py \\
      --modelName   Qwen/Qwen2.5-Coder-7B-Instruct \\
      --outputDir   ./finetune_repair_output \\
      --mutationTypes all \\
      --gpus 0,1 \\
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
logger = logging.getLogger("prompt_repair")

# ── Re-use data loading from finetune_lora ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from finetune_lora import (
    load_sft_pairs, _load_jsonl,
    LCB_MUTATION_FILES, APPS_MUTATION_FILES,
    MBPP_MUTATION_FILES, MBPP_MUTATION_FILES_V2,
    HE_MUTATION_FILES,
)


def _to_repair(r: dict, mtype: str, dataset: str) -> dict:
    return {
        "task_id":         r["task_id"],
        "mutated_prompt":  r["mutated_prompt"],
        "original_prompt": r["original_prompt"],
        "mutation_type":   mtype,
        "dataset":         dataset,
        "test":            r.get("test", ""),
        "entry_point":     r.get("entry_point", ""),
    }


def load_train_repair_pairs(
    data_dir: Path,
    mutation_types: Set[str],
) -> List[Dict[str, Any]]:
    """MBPP v1+v2 + LCB + APPS — HumanEval held out for eval."""
    pairs: List[Dict[str, Any]] = []

    # MBPP v1 + v2
    for fmap in (MBPP_MUTATION_FILES, MBPP_MUTATION_FILES_V2):
        for mtype, fname in fmap.items():
            if mtype not in mutation_types:
                continue
            path = data_dir / "mutations" / fname
            if not path.exists():
                continue
            n = 0
            for r in _load_jsonl(path):
                if not r.get("applicable", True): continue
                if not r.get("original_prompt") or not r.get("mutated_prompt"): continue
                pairs.append(_to_repair(r, mtype, "mbpp"))
                n += 1
            logger.info("  MBPP [%s] %-3s: %d", fname, mtype, n)

    # LCB
    for mtype, fname in LCB_MUTATION_FILES.items():
        if mtype not in mutation_types:
            continue
        path = data_dir / "mutations" / fname
        if not path.exists():
            continue
        n = 0
        for r in _load_jsonl(path):
            if not r.get("applicable", True): continue
            if not r.get("original_prompt") or not r.get("mutated_prompt"): continue
            pairs.append(_to_repair(r, mtype, "lcb"))
            n += 1
        logger.info("  LCB  %-3s: %d", mtype, n)

    # APPS
    for mtype, fname in APPS_MUTATION_FILES.items():
        if mtype not in mutation_types:
            continue
        path = data_dir / "mutations" / fname
        if not path.exists():
            continue
        n = 0
        for r in _load_jsonl(path):
            if not r.get("applicable", True): continue
            if not r.get("original_prompt") or not r.get("mutated_prompt"): continue
            pairs.append(_to_repair(r, mtype, "apps"))
            n += 1
        logger.info("  APPS %-3s: %d", mtype, n)

    return pairs


def load_he_repair_pairs(
    data_dir: Path,
    mutation_types: Set[str],
) -> List[Dict[str, Any]]:
    """HumanEval v1 — used as held-out eval set."""
    pairs: List[Dict[str, Any]] = []
    for mtype, fname in HE_MUTATION_FILES.items():
        if mtype not in mutation_types:
            continue
        path = data_dir / "mutations" / fname
        if not path.exists():
            continue
        n = 0
        for r in _load_jsonl(path):
            if not r.get("applicable", True): continue
            if not r.get("original_prompt") or not r.get("mutated_prompt"): continue
            pairs.append(_to_repair(r, mtype, "humaneval"))
            n += 1
        logger.info("  HumanEval (eval) %-3s: %d", mtype, n)
    return pairs


# ── Prompts ────────────────────────────────────────────────────────────────────

REPAIR_INSTRUCTION = """\
The following programming prompt has been rewritten using vague or ambiguous \
language — some terms, variable names, or constraints may have been softened \
or removed.

Restore it to its precise, original specification. Keep the function signature \
and docstring examples exactly as they appear. Do NOT add code or explanations.

Vague prompt:
{mutated_prompt}

Return ONLY the restored prompt.\
"""


def build_repair_instruction(mutated_prompt: str) -> str:
    return REPAIR_INSTRUCTION.format(mutated_prompt=mutated_prompt.strip())


# ── Dataset ────────────────────────────────────────────────────────────────────

class RepairDataset(Dataset):
    """
    Tokenises (mutated_prompt → original_prompt) pairs for causal LM SFT.
    Loss is computed on the target (original_prompt) tokens only.
    """

    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 1024):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        instruction = build_repair_instruction(pair["mutated_prompt"])
        target      = pair["original_prompt"].strip()

        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": target},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
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
            instr_text, truncation=False, return_tensors="pt",
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


# ── BLEU-4 (token-level, no external deps) ────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i: i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def sentence_bleu4(hypothesis: str, reference: str) -> float:
    """Compute BLEU-4 between a single hypothesis and reference (word-level)."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp:
        return 0.0

    import math
    bp = min(1.0, math.exp(1 - len(ref) / len(hyp))) if len(hyp) < len(ref) else 1.0

    score = 1.0
    for n in range(1, 5):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        clipped = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
        total   = max(len(hyp) - n + 1, 0)
        if total == 0:
            return 0.0
        score *= (clipped / total) if clipped > 0 else 1e-9

    return bp * (score ** 0.25)


# ── Repair quality evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def eval_repair_quality(
    model,
    tokenizer,
    pairs: List[Dict],
    input_device: torch.device,
    max_new_tokens: int = 512,
    label: str = "eval",
) -> Dict[str, float]:
    """
    Generate repaired prompts for `pairs` and compute:
      - mean BLEU-4 vs original_prompt
      - exact-match rate (after strip/lower)
    """
    model.eval()
    model.enable_adapter_layers()

    bleu_scores, exact = [], 0

    for pair in tqdm(pairs, desc=f"Repair eval [{label}]", ncols=90, leave=False):
        instruction = build_repair_instruction(pair["mutated_prompt"])
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=1024,
        )["input_ids"].to(input_device)
        prompt_len = input_ids.shape[1]

        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        repaired = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        original = pair["original_prompt"].strip()

        bleu_scores.append(sentence_bleu4(repaired, original))
        if repaired.lower() == original.lower():
            exact += 1

    n = len(pairs)
    result = {
        "n":         n,
        "bleu4":     round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
        "exact_match": round(exact / n, 4) if n else 0.0,
    }
    logger.info("Repair quality [%s]  BLEU-4=%.4f  ExactMatch=%.4f  (n=%d)",
                label, result["bleu4"], result["exact_match"], n)
    model.train()
    return result


# ── Generate + save repaired prompts (for downstream code eval) ────────────────

@torch.no_grad()
def generate_repaired_prompts(
    model,
    tokenizer,
    pairs: List[Dict],
    input_device: torch.device,
    out_path: Path,
    max_new_tokens: int = 512,
) -> None:
    """
    Writes a JSONL file with repaired prompts that can be directly fed into
    main_inference.py (same format as mutation files, with mutated_prompt
    replaced by the repaired text).
    """
    model.eval()
    model.enable_adapter_layers()
    rows = []

    for pair in tqdm(pairs, desc="Generating repaired prompts", ncols=90):
        instruction = build_repair_instruction(pair["mutated_prompt"])
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=1024,
        )["input_ids"].to(input_device)
        prompt_len = input_ids.shape[1]

        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        repaired = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        rows.append({
            "task_id":          pair["task_id"],
            "original_prompt":  pair.get("original_prompt", ""),
            "mutated_prompt":   pair["mutated_prompt"],
            "repaired_prompt":  repaired,
            "mutation_type":    pair["mutation_type"],
            "dataset":          pair["dataset"],
            # keep test fields so main_inference.py can eval directly
            "test":             pair.get("test", ""),
            "entry_point":      pair.get("entry_point", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    logger.info("Repaired prompts saved → %s  (%d rows)", out_path, len(rows))


# ── End-to-end pass@1 evaluation ──────────────────────────────────────────────

@torch.no_grad()
def eval_e2e_pass1(
    model,
    tokenizer,
    pairs: List[Dict],
    input_device: torch.device,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    For each pair, generate code from BOTH the mutated and repaired prompt
    using the same base model (adapter disabled). Compare pass@1.

    Returns:
      pass1_mutated  : baseline (mutated prompt, no repair)
      pass1_repaired : after repair (repaired prompt)
      delta          : pass1_repaired - pass1_mutated
    """
    from main_inference import (
        evaluate_with_timeout,
        convert_general_check_code_HumanEval,
        extract_code,
    )
    import re, textwrap

    def _build_codegen_prompt(prompt_text: str) -> str:
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", prompt_text)
        func_name = m.group(1) if m else "solution"
        return textwrap.dedent(f"""
            You are a senior Python developer.

            Task:
            {prompt_text}

            Write **one** function named `{func_name}` that solves the task.
            Use only the Python standard library.

            Return *only* valid Python code in a single code block:
            ```python
            <your code here>
            ```
        """).strip()

    def _generate(prompt_text: str) -> str:
        instr = _build_codegen_prompt(prompt_text)
        chat  = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(chat, return_tensors="pt", truncation=True,
                        max_length=1024)["input_ids"].to(input_device)
        out = model.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def _run(code: str, pair: Dict) -> bool:
        check_code, n_tests = convert_general_check_code_HumanEval(
            pair["test"], pair.get("entry_point", "solution")
        )
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", pair.get("repaired_prompt",
                                                               pair["mutated_prompt"]))
        mutated_name = m.group(1) if m else None
        original_ep  = pair.get("entry_point", "")
        candidates   = list(dict.fromkeys(
            [x for x in [mutated_name, original_ep, None] if x is not None] + [None]
        ))
        for ep in candidates:
            passed, status = evaluate_with_timeout(code, check_code,
                                                   timeout_seconds=20, entry_point=ep)
            if "not found" not in str(status).lower():
                return passed == n_tests and status == "OK"
        return False

    model.eval()
    pass_mutated = pass_repaired = 0
    rows = []

    for pair in tqdm(pairs, desc="E2E pass@1 eval", ncols=90):
        # ── baseline: mutated prompt, adapter OFF ──────────────────────────────
        model.disable_adapter_layers()
        code_mut = extract_code(_generate(pair["mutated_prompt"]))
        ok_mut   = _run(code_mut, pair)

        # ── repaired: repaired prompt, adapter OFF ─────────────────────────────
        repaired = pair.get("repaired_prompt", pair["mutated_prompt"])
        code_rep = extract_code(_generate(repaired))
        ok_rep   = _run(code_rep, pair)

        pass_mutated  += int(ok_mut)
        pass_repaired += int(ok_rep)
        rows.append({
            "task_id":        pair["task_id"],
            "pass_mutated":   ok_mut,
            "pass_repaired":  ok_rep,
            "code_mutated":   code_mut,
            "code_repaired":  code_rep,
            "repaired_prompt": repaired,
        })

    n = len(pairs)
    result = {
        "n":              n,
        "pass1_mutated":  round(pass_mutated  / n, 4),
        "pass1_repaired": round(pass_repaired / n, 4),
        "delta":          round((pass_repaired - pass_mutated) / n, 4),
        "rows":           rows,
    }
    logger.info(
        "E2E pass@1 — mutated=%.3f  repaired=%.3f  Δ=%+.3f  (n=%d)",
        result["pass1_mutated"], result["pass1_repaired"], result["delta"], n,
    )
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT: mutated_prompt → original_prompt")
    parser.add_argument("--modelName",       default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataDir",         default=".")
    parser.add_argument("--outputDir",       default="./finetune_repair_output")
    parser.add_argument("--mutationTypes",   default="LV",
                        help="'all' or comma-separated: LV,SF,US")
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batchSize",       type=int,   default=4)
    parser.add_argument("--gradAccum",       type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=5e-5)
    parser.add_argument("--maxLength",       type=int,   default=1024)
    parser.add_argument("--valSplit",        type=float, default=0.1)
    parser.add_argument("--loraR",           type=int,   default=16)
    parser.add_argument("--loraAlpha",       type=int,   default=32)
    parser.add_argument("--loraDropout",     type=float, default=0.05)
    parser.add_argument("--patience",        type=int,   default=3)
    parser.add_argument("--gpus",            default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--warmupSteps",     type=int,   default=100)
    parser.add_argument("--skipEval",        action="store_true")
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

    mutation_types = (
        {"LV", "SF", "US"}
        if args.mutationTypes.lower() == "all"
        else {t.strip().upper() for t in args.mutationTypes.split(",")}
    )
    logger.info("Mutation types: %s", sorted(mutation_types))

    # ── Data ───────────────────────────────────────────────────────────────────
    logger.info("Loading train pairs (MBPP+LCB+APPS)  mutations=%s", sorted(mutation_types))
    train_pairs = load_train_repair_pairs(data_dir, mutation_types)
    random.shuffle(train_pairs)
    logger.info("Train total: %d pairs", len(train_pairs))

    # 10% of train as val (task-id level to avoid leakage within a dataset)
    unique_task_ids = sorted({p["task_id"] for p in train_pairs})
    random.shuffle(unique_task_ids)
    n_val     = max(1, int(len(unique_task_ids) * args.valSplit))
    val_ids   = set(unique_task_ids[:n_val])
    train_ids = set(unique_task_ids[n_val:])
    val_pairs   = [p for p in train_pairs if p["task_id"] in val_ids]
    train_pairs = [p for p in train_pairs if p["task_id"] in train_ids]
    logger.info("Train: %d  Val: %d", len(train_pairs), len(val_pairs))

    logger.info("Loading eval pairs (HumanEval — held out)  mutations=%s", sorted(mutation_types))
    eval_pairs = load_he_repair_pairs(data_dir, mutation_types)
    logger.info("Eval (HumanEval): %d pairs", len(eval_pairs))

    # ── Tokenizer + model ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.modelName)
    tokenizer = AutoTokenizer.from_pretrained(args.modelName, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model: %s", args.modelName)
    model = AutoModelForCausalLM.from_pretrained(
        args.modelName,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

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
    input_device = next(model.parameters()).device

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = RepairDataset(train_pairs, tokenizer, args.maxLength)
    val_ds   = RepairDataset(val_pairs,   tokenizer, args.maxLength)
    train_loader = DataLoader(
        train_ds, batch_size=args.batchSize, shuffle=True,
        collate_fn=RepairDataset.collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batchSize, shuffle=False,
        collate_fn=RepairDataset.collate_fn, num_workers=2, pin_memory=True,
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

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0
    ckpt_path     = out_dir / "best_lora_repair"
    epoch         = 0

    logger.info("Training: epochs=%d  batch=%d  gradAccum=%d  eff_batch=%d",
                args.epochs, args.batchSize, args.gradAccum,
                args.batchSize * args.gradAccum)

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

        if len(train_loader) % args.gradAccum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train = total_loss / len(train_loader)
        val_loss  = evaluate_loss(model, val_loader, input_device)

        improved = val_loss < best_val_loss
        marker   = " ***" if improved else ""
        logger.info("Epoch %2d | train=%.4f | val=%.4f%s", epoch, avg_train, val_loss, marker)

        if improved:
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

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "model_name":      args.modelName,
        "task":            "prompt_repair",
        "mutation_types":  sorted(mutation_types),
        "train_datasets":  "MBPP(v1+v2) + LCB + APPS",
        "eval_dataset":    "HumanEval LV v1 (held out)",
        "train_pairs":     len(train_pairs),
        "val_pairs":       len(val_pairs),
        "eval_pairs":      len(eval_pairs),
        "best_val_loss":   best_val_loss,
        "lora_r":          args.loraR,
        "lora_alpha":      args.loraAlpha,
        "epochs_run":      epoch,
        "lr":              args.lr,
        "eff_batch_size":  args.batchSize * args.gradAccum,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    # ── Post-training eval ─────────────────────────────────────────────────────
    if not args.skipEval and eval_pairs:
        from peft import set_peft_model_state_dict
        import safetensors.torch as sf
        adapter_file = ckpt_path / "adapter_model.safetensors"
        if adapter_file.exists():
            state = sf.load_file(str(adapter_file))
        else:
            state = torch.load(str(ckpt_path / "adapter_model.bin"), map_location="cpu")
        set_peft_model_state_dict(model, state)
        logger.info("Reloaded best adapter from %s", ckpt_path)

        repair_metrics = eval_repair_quality(
            model, tokenizer, eval_pairs, input_device,
            max_new_tokens=512, label="HumanEval_LV",
        )
        meta["repair_eval"] = {**repair_metrics, "eval_dataset": "HumanEval LV v1"}
        (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

        print(f"\nRepair quality on HumanEval LV (held-out, n={repair_metrics['n']}):")
        print(f"  BLEU-4      : {repair_metrics['bleu4']:.4f}")
        print(f"  Exact match : {repair_metrics['exact_match']:.4f}")

        # ── Generate repaired prompts then run end-to-end pass@1 ──────────────
        repaired_path = out_dir / "repaired_prompts_humaneval_lv.jsonl"
        generate_repaired_prompts(
            model, tokenizer, eval_pairs, input_device,
            out_path=repaired_path, max_new_tokens=512,
        )

        # Load repaired prompts back so e2e eval has the repaired_prompt field
        repaired_pairs = [json.loads(l) for l in repaired_path.read_text().splitlines() if l.strip()]
        # Attach test/entry_point from eval_pairs (not stored in repaired file fully)
        ep_map   = {p["task_id"]: p for p in eval_pairs}
        for rp in repaired_pairs:
            orig = ep_map.get(rp["task_id"], {})
            rp.setdefault("test",        orig.get("test", ""))
            rp.setdefault("entry_point", orig.get("entry_point", ""))

        e2e = eval_e2e_pass1(model, tokenizer, repaired_pairs, input_device)

        meta["e2e_eval"] = {k: v for k, v in e2e.items() if k != "rows"}
        (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
        (out_dir / "e2e_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in e2e["rows"]) + "\n"
        )

        print(f"\nEnd-to-end pass@1 on HumanEval LV (n={e2e['n']}):")
        print(f"  Mutated prompt  (baseline) : {e2e['pass1_mutated']:.3f}")
        print(f"  Repaired prompt (ours)     : {e2e['pass1_repaired']:.3f}")
        print(f"  Δ                          : {e2e['delta']:+.3f}")
        print(f"\nDetailed results → {out_dir}/e2e_results.jsonl")


if __name__ == "__main__":
    main()
