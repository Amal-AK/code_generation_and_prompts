from __future__ import annotations

# ────────────────────────────────────── imports ────────────────────────────────────────
import argparse
import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ──────────────────────────────────── logging ──────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mutant_judge")


# ──────────────────────────────── GPU selection ───────────────────────────────────────

def set_visible_gpus(gpus: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info("CUDA_VISIBLE_DEVICES=%s", gpus)


# ──────────────────────────────────── data loading ────────────────────────────────────

def load_records(path: str | Path) -> List[Dict[str, Any]]:
    p   = Path(path)
    ext = p.suffix.lower()
    txt = p.read_text("utf-8").lstrip("\ufeff").strip()
    if ext == ".json":
        return json.loads(txt)
    if ext in {".jsonl", ".ndjson"}:
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
    if ext == ".csv":
        return pd.read_csv(p).to_dict(orient="records")
    raise ValueError(f"Unknown extension: {ext}")


# ──────────────────────────────── Criteria Definition ─────────────────────────────────
#
# Two layers:
#   GLOBAL_CRITERIA — asked for every mutant regardless of type
#   TYPE_CRITERIA   — asked only for the matching mutation_type
#
# Each criterion:
#   key        — output column name
#   scale      — "binary" (0/1) or "1-5"
#   is_filter  — if True, score=0 short-circuits remaining criteria
#   question   — exact text sent to the judge

GLOBAL_CRITERIA = [
    {
        "key":       "semantic_preservation",
        "scale":     "binary",
        "is_filter": True,
        "question": (
            "Does the mutated prompt still describe the same coding task as the original? "
            "Focus on the described task, not on syntactic correctness or formatting. "
            "Answer 1 for yes, 0 for no."
        ),
    },
    {
        "key":       "naturalness",
        "scale":     "binary",
        "is_filter": False,
        "question": (
            "Does the mutated prompt read like something a real developer could write, "
            "even if it has typos or formatting errors? "
            "Answer 1 for yes, 0 for no."
        ),
    },
    {
        "key":       "recoverability",
        "scale":     "binary",
        "is_filter": False,
        "question": (
            "Can a skilled programmer still implement the correct solution from the mutated prompt? "
            "Answer 1 for yes, 0 for no."
        ),
    },
]

TYPE_CRITERIA: Dict[str, List[Dict]] = {

    "LV": [
        {
            "key":       "lexical_compliance",
            "scale":     "binary",
            "is_filter": True,
            "question": (
                "Did the mutation replace specific terms, variable names, or descriptions "
                "with vaguer equivalents, without changing the overall task? "
                "Answer 1 for yes, 0 for no."
            ),
        },
    ],

    "SF": [
        {
            "key":       "formatting_compliance",
            "scale":     "binary",
            "is_filter": True,
            "question": (
                "Does the mutated prompt contain typos or formatting errors compared to the original? "
                "Answer 1 for yes, 0 for no."
            ),
        },
    ],

    "US": [
        {
            "key":       "underspec_compliance",
            "scale":     "binary",
            "is_filter": True,
            "question": (
                "Is any detail, condition, or requirement missing from the mutated prose "
                "compared to the original? Ignore code example lines (>>>). "
                "Answer 1 for yes, 0 for no."
            ),
        },
    ],

    "unknown": [],
}


# ──────────────────────────────── Local Judge Model ───────────────────────────────────
# Qwen2.5-Coder-7B-Instruct at float16 ≈ 14 GB — fits on a single V100-32GB.
# We load once and keep in memory for all evaluations.

class LocalJudge:
    """
    Wraps a local HuggingFace causal LM for single-integer scoring.
    Uses greedy decoding — model is expected to output a bare 0 or 1.
    """

    def __init__(self, model_name: str):
        logger.info("Loading judge model: %s  (device_map=auto, all GPUs)", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Judge model loaded across %d GPUs", torch.cuda.device_count())

    def _format_prompt(self, system: str, user: str) -> str:
        """
        Use chat_template when available, fall back to plain system+user block.
        gpt-oss-20b follows the ChatML format.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        if getattr(self.tokenizer, "chat_template", None) is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Plain fallback
        return f"### System\n{system}\n\n### User\n{user}\n\n### Assistant\n"

    @staticmethod
    def _parse_score(text: str) -> Optional[int]:
        """Extract 0 or 1 from the model output.

        Tries JSON first (expected format: {"score": 0}), then falls back
        to a bare-digit regex scan on the last token in the output.
        """
        # JSON path — covers {"score": 0}, {"score":1}, partial like `0}`
        try:
            # The prompt primes the model with '{"score": ' so the completion
            # may be just `0}` — prepend the prefix to recover valid JSON.
            candidate = text if text.startswith("{") else '{"score": ' + text
            data = json.loads(candidate)
            val = int(data["score"])
            if val in (0, 1):
                return val
        except (ValueError, KeyError, TypeError):
            pass
        # Regex fallback — take the last 0 or 1 in the output
        matches = re.findall(r"\b([01])\b", text)
        return int(matches[-1]) if matches else None

    @torch.no_grad()
    def score_batch(
        self,
        prompts: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[Optional[int]]:
        """
        Score a list of (system, user) pairs in mini-batches.
        Returns a list of integers (0 or 1) or None when extraction fails.
        On CUDA OOM the batch size is halved and the chunk is retried automatically.
        Falls back to size-1 batches; if a single item still OOMs it is skipped (None).
        """
        all_results: List[Optional[int]] = []
        dev = next(self.model.parameters()).device
        current_batch_size = batch_size

        pbar = tqdm(total=len(prompts), desc="Scoring batches", unit="prompt")
        i = 0
        while i < len(prompts):
            chunk = prompts[i : i + current_batch_size]
            formatted = [self._format_prompt(sys_p, usr_p) for sys_p, usr_p in chunk]

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(dev)
            attn_mask = inputs["attention_mask"].to(dev)
            prefix_len = input_ids.shape[1]

            try:
                out = self.model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                for out_ids in out:
                    generated = self.tokenizer.decode(
                        out_ids[prefix_len:], skip_special_tokens=True
                    ).strip()
                    score = self._parse_score(generated)
                    if score is None:
                        logger.warning("No binary score in judge output: %r", generated[:100])
                    all_results.append(score)
                pbar.update(len(chunk))
                i += current_batch_size

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if current_batch_size == 1:
                    logger.warning("OOM on single item at index %d — skipping", i)
                    all_results.append(None)
                    pbar.update(1)
                    i += 1
                else:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning("CUDA OOM — reducing batch size to %d and retrying", current_batch_size)

        pbar.close()
        return all_results

    @torch.no_grad()
    def score(self, system: str, user: str) -> Optional[int]:
        return self.score_batch([(system, user)], batch_size=1)[0]

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ──────────────────────────────── Prompt Builder ──────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert evaluator of code generation benchmark mutations. "
    "You will be shown an original coding prompt, a mutation type, and the mutated prompt. "
    "Evaluate the quality of this mutation by answering a specific question with a numeric score. "
    'Return ONLY a JSON object in this exact format: {"score": 0} or {"score": 1}. '
    "No reasoning, no explanation, no extra text. Just the JSON object. "
    "Do NOT reproduce or quote any part of the input prompts."
)


def build_user_prompt(
    original:      str,
    mutated:       str,
    mutation_type: str,
    question:      str,
) -> str:
    return (
        f"## Original prompt\n{original}\n\n"
        f"## Mutation type\n{mutation_type}\n\n"
        f"## Mutated prompt\n{mutated}\n\n"
        f"## Question\n{question}\n\n"
        'Your answer (JSON only, no reasoning): {"score": '
    )


# ──────────────────────────────── Criterion Evaluation ────────────────────────────────

def judge_criterion(
    judge:         LocalJudge,
    original:      str,
    mutated:       str,
    mutation_type: str,
    criterion:     Dict,
) -> Optional[int]:
    user_msg = build_user_prompt(original, mutated, mutation_type, criterion["question"])
    score    = judge.score(SYSTEM_PROMPT, user_msg)

    if score is None:
        return None

    # Validate range
    if score not in (0, 1):
        logger.warning("Out-of-range binary score %d for %s — treating as None",
                       score, criterion["key"])
        return None

    return score


# ──────────────────────────────── Score Aggregation ───────────────────────────────────

def get_all_criteria(mutation_type: str) -> List[Dict]:
    specific = TYPE_CRITERIA.get(mutation_type, TYPE_CRITERIA["unknown"])
    return GLOBAL_CRITERIA + specific


def compute_scores(row: Dict, mutation_type: str) -> Dict:
    # No aggregation — individual criterion scores are the output
    return {}


# ──────────────────────────────── Main Evaluation Loop ────────────────────────────────

def _score_and_save(judge, pending: List[Dict], out_path: Path, batch_size: int):
    """Score pending items and append results to out_path immediately after scoring."""
    if not pending:
        return []

    # Build (system, user) prompt pairs for every item × criterion
    all_prompts: List[Tuple[str, str]] = []
    prompt_meta: List[Tuple[int, str]] = []

    for i, item in enumerate(pending):
        criteria = get_all_criteria(item["mutation_type"])
        for criterion in criteria:
            user_msg = build_user_prompt(
                item["original"], item["mutated"],
                item["mutation_type"], criterion["question"],
            )
            all_prompts.append((SYSTEM_PROMPT, user_msg))
            prompt_meta.append((i, criterion["key"]))

    logger.info("  Judge calls: %d  (batch_size=%d)", len(all_prompts), batch_size)

    # Sort by prompt length → minimal padding waste per batch
    sorted_indices = sorted(range(len(all_prompts)), key=lambda i: len(all_prompts[i][0]) + len(all_prompts[i][1]))
    sorted_prompts = [all_prompts[i] for i in sorted_indices]
    sorted_meta    = [prompt_meta[i] for i in sorted_indices]

    all_scores_sorted = judge.score_batch(sorted_prompts, batch_size=batch_size)

    for (item_idx, criterion_key), score in zip(sorted_meta, all_scores_sorted):
        pending[item_idx]["criterion_scores"][criterion_key] = score

    # Write results immediately after this file finishes
    results = []
    with open(out_path, "a", encoding="utf-8") as fout:
        for item in pending:
            row: Dict[str, Any] = {
                "_judge_id":     item["judge_id"],
                "task_id":       item["task_id"],
                "mutation_type": item["mutation_type"],
                "dataset":       item["dataset"],
            }
            row.update(item["criterion_scores"])
            row.update(compute_scores(item["criterion_scores"], item["mutation_type"]))
            results.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    return results


def evaluate_mutants(args):

    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support — load all already-evaluated judge_ids from the combined output
    out_path = output_dir / "judge_results.jsonl"
    done_ids: set = set()
    if out_path.exists() and not args.overwrite:
        for ln in out_path.read_text("utf-8").splitlines():
            if ln.strip():
                done_ids.add(json.loads(ln).get("_judge_id", ""))
        logger.info("Resuming — %d already evaluated", len(done_ids))

    # Load judge model once
    judge = LocalJudge(args.judgeModel)

    all_results: List[Dict] = []
    total_limit = args.limit

    # ── Process one input file at a time → save results after each file ────────
    for input_file in args.inputFiles:
        recs = load_records(input_file)
        file_label = Path(input_file).stem
        logger.info("── File: %s  (%d records)", file_label, len(recs))

        pending: List[Dict[str, Any]] = []
        skipped = 0
        for r in recs:
            if total_limit is not None and (len(all_results) + len(pending)) >= total_limit:
                break
            original      = (r.get("original_prompt") or r.get("prompt") or "").strip()
            mutated       = (r.get("mutated_prompt") or "").strip()
            mutation_type = str(r.get("mutation_type", "unknown"))
            task_id       = str(r.get("task_id", ""))
            judge_id      = f"{task_id}__{mutation_type}"

            if not original or not mutated or original == mutated:
                skipped += 1
                continue
            if judge_id in done_ids:
                skipped += 1
                continue

            pending.append({
                "judge_id":         judge_id,
                "task_id":          task_id,
                "mutation_type":    mutation_type,
                "dataset":          r.get("dataset") or ("humaneval" if task_id.startswith("HumanEval") else "livecodebench" if task_id.startswith("LiveCodeBench") else "mbpp"),
                "original":         original,
                "mutated":          mutated,
                "criterion_scores": {},
            })

        logger.info("  Pending: %d  skipped/resumed: %d", len(pending), skipped)

        if not pending:
            logger.info("  All records already evaluated, skipping.")
            continue

        # Score and save this file's results immediately
        file_results = _score_and_save(judge, pending, out_path, args.batchSize)
        all_results.extend(file_results)

        # Mark newly evaluated ids as done for subsequent files
        for item in pending:
            done_ids.add(item["judge_id"])

        logger.info("  Saved %d results for %s", len(file_results), file_label)

    judge.cleanup()

    if not all_results:
        logger.warning("No results to summarize")
        return

    df = pd.DataFrame(all_results)
    n = len(df)

    # Per dataset × mutation_type × criterion
    detail_rows = []
    for dataset, dgrp in df.groupby("dataset"):
        for mut, grp in dgrp.groupby("mutation_type"):
            d = {
                "dataset":       dataset,
                "mutation_type": mut,
                "n":             len(grp),
            }
            for c in get_all_criteria(mut):
                vals = grp[c["key"]].dropna()
                d[c["key"]] = round(vals.mean(), 3) if len(vals) else None
            detail_rows.append(d)
    df_detail = pd.DataFrame(detail_rows)

    # Per mutation type only (aggregated across datasets)
    mut_rows = []
    for mut, grp in df.groupby("mutation_type"):
        d = {
            "mutation_type": mut,
            "n":             len(grp),
        }
        for c in get_all_criteria(mut):
            vals = grp[c["key"]].dropna()
            d[c["key"]] = round(vals.mean(), 3) if len(vals) else None
        mut_rows.append(d)
    df_mut = pd.DataFrame(mut_rows)

    summary = {
        "judge_model":              args.judgeModel,
        "n_mutants":                n,
        "per_dataset_mutation_type": detail_rows,
        "per_mutation_type":        mut_rows,
    }

    print("\n" + "=" * 75)
    print("  LLM-as-Judge  ·  Mutant Quality Evaluation")
    print("=" * 75)
    print(f"  Judge model      : {args.judgeModel}")
    print(f"  Mutants evaluated: {n}")
    print()
    print("  Per dataset × mutation type × criterion:")
    print("-" * 75)
    for row in detail_rows:
        ds  = row["dataset"]
        mut = row["mutation_type"]
        print(f"  [{ds}] [{mut}]  n={row['n']}")
        for c in get_all_criteria(mut):
            val = row.get(c["key"])
            val_str = f"{val:.3f}" if val is not None else "N/A"
            print(f"    {c['key']:<30} {val_str}")
        print()
    print("=" * 75 + "\n")

    df.to_csv(output_dir / "judge_results.csv",               index=False)
    df_mut.to_csv(output_dir / "judge_by_mutation.csv",       index=False)
    df_detail.to_csv(output_dir / "judge_by_dataset_mut.csv", index=False)
    (output_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved → %s", output_dir)


# ──────────────────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge: evaluate mutant quality using openai/gpt-oss-20b locally"
    )
    parser.add_argument("--inputFiles",  nargs="+", required=True,
                        help="Mutated JSONL files — mutation_type field required")
    parser.add_argument("--outputDir",   default="./judge_output")
    parser.add_argument("--limit",       type=int, default=10_000)

    # Judge model — local HuggingFace
    parser.add_argument("--judgeModel",  default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="HuggingFace model id for the judge")
    parser.add_argument("--gpus",        type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES, e.g. '0,1,2,4'")

    parser.add_argument("--batchSize",   type=int, default=16,
                        help="Batch size for judge inference (default 16)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-evaluate already-judged records")

    args = parser.parse_args()

    if args.gpus is not None:
        set_visible_gpus(args.gpus)

    logger.info("Judge=%s", args.judgeModel)
    evaluate_mutants(args)


if __name__ == "__main__":
    main()