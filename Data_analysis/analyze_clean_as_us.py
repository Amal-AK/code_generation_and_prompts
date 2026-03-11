#!/usr/bin/env python3
"""
analyze_clean_as_us.py
──────────────────────
Reproduce the classifier val split (seed=42), run the best LoRA classifier,
and analyse cases where a CLEAN prompt is predicted as US.

For each misclassified CLEAN→US sample:
  - Shows the prompt text
  - Looks up pass@1 for that task_id across all oracle model outputs
    (oracle ran on US *mutations* of the same task, which is a useful proxy
     for how hard / ambiguous the underlying problem is)

Output:
  clean_as_us_analysis.json   — full data per sample
  clean_as_us_summary.txt     — human-readable summary
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(".")
CKPT_PATH      = Path("lora_classifier_outputs/seed42/best_lora_classifier.pt")
MODEL_NAME     = "Qwen/Qwen2.5-Coder-1.5B"
ORACLE_DIR     = Path("oracle_output")
SEED           = 42
VAL_SPLIT      = 0.2
BATCH_SIZE     = 8
MAX_LENGTH     = 512
OUTPUT_JSON    = "clean_as_us_analysis.json"
OUTPUT_TXT     = "clean_as_us_summary.txt"

LABEL2ID = {"LV": 0, "SF": 1, "US": 2, "CLEAN": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Model (same architecture as train_lora_classifier.py) ─────────────────────

class LoRAPromptClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim: int, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        emb    = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.dropout(emb))


# ── Data loading (mirrors train_lora_classifier.py, but keeps metadata) ────────

def load_mutated_with_meta(path: Path, label: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("applicable", True):
                continue
            text = obj.get("mutated_prompt", "").strip()
            if text:
                records.append({
                    "text":          text,
                    "label":         LABEL2ID[label],
                    "label_name":    label,
                    "task_id":       obj.get("task_id", ""),
                    "source_file":   path.name,
                    "dataset":       _infer_dataset(path.name),
                })
    return records


def load_clean_with_meta(path: Path, text_field: str, dataset: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "").strip()
            if text:
                records.append({
                    "text":          text,
                    "label":         LABEL2ID["CLEAN"],
                    "label_name":    "CLEAN",
                    "task_id":       str(obj.get("task_id", obj.get("problem_id", ""))),
                    "source_file":   path.name,
                    "dataset":       dataset,
                })
    return records


def _infer_dataset(fname: str) -> str:
    if "humanEval" in fname or "HumanEval" in fname:
        return "humaneval"
    if "mbpp" in fname:
        return "mbpp"
    if "livecodebench" in fname:
        return "lcb"
    return "unknown"


def load_all_data_with_meta() -> List[Dict]:
    all_records: List[Dict] = []

    mutation_files = [
        (DATA_DIR / "mutations/humanEval_lv_with_tests.jsonl",     "LV"),
        (DATA_DIR / "mutations/humanEval_SF_with_tests.jsonl",     "SF"),
        (DATA_DIR / "mutations/HumanEval_US_with_tests.jsonl",     "US"),
        (DATA_DIR / "mutations/mbpp_LV_with_tests.jsonl",          "LV"),
        (DATA_DIR / "mutations/mbpp_SF_with_tests.jsonl",          "SF"),
        (DATA_DIR / "mutations/mbpp_US_with_tests.jsonl",          "US"),
        (DATA_DIR / "mutations/livecodebench_LV_with_tests.jsonl", "LV"),
        (DATA_DIR / "mutations/livecodebench_SF_with_tests.jsonl", "SF"),
        (DATA_DIR / "mutations/livecodebench_US_with_tests.jsonl", "US"),
    ]
    for path, label in mutation_files:
        if path.exists():
            all_records.extend(load_mutated_with_meta(path, label))

    clean_files = [
        (DATA_DIR / "datasets/humanEval/HumanEval.jsonl",                "prompt",  "humaneval"),
        (DATA_DIR / "datasets/mbpp/mbpp.jsonl",                          "text",    "mbpp"),
        (DATA_DIR / "datasets/livecodebench/livecodebench_public.jsonl", "prompt",  "lcb"),
    ]
    for path, field, ds in clean_files:
        if path.exists():
            all_records.extend(load_clean_with_meta(path, field, ds))

    return all_records


# ── Oracle pass@1 lookup ───────────────────────────────────────────────────────

def load_oracle_pass1() -> Dict[str, Dict[str, bool]]:
    """
    Returns {task_id: {model_name: pass@1_bool}} from oracle_output directories.
    Only uses all_results.json from each model folder.
    """
    result: Dict[str, Dict[str, bool]] = defaultdict(dict)

    for entry in ORACLE_DIR.iterdir():
        if not entry.is_dir():
            continue
        model_name = entry.name
        all_res_path = entry / "all_results.json"
        if not all_res_path.exists():
            continue
        try:
            data = json.load(open(all_res_path))
            for r in data:
                tid = str(r.get("task_id", ""))
                p1  = r.get("Pass@1", None)
                if tid and p1 is not None:
                    result[tid][model_name] = bool(p1)
        except Exception:
            pass

    return result


# ── Collate ───────────────────────────────────────────────────────────────────

def make_collate(tokenizer):
    def collate(batch):
        texts  = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         torch.tensor(labels, dtype=torch.long),
            "meta":           batch,
        }
    return collate


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("Loading data …")
    all_data = load_all_data_with_meta()

    # Reproduce val split: stratify by (label, dataset) same as training
    # train_lora_classifier.py just does random shuffle + slice
    random.shuffle(all_data)
    n_val    = int(len(all_data) * VAL_SPLIT)
    val_data = all_data[:n_val]   # matches train_lora_classifier.py: val=[:n_val], train=[n_val:]

    clean_val = [d for d in val_data if d["label_name"] == "CLEAN"]
    print(f"Val set: {len(val_data)} total, {len(clean_val)} CLEAN")

    print("Loading oracle pass@1 data …")
    oracle = load_oracle_pass1()
    oracle_models = sorted({m for v in oracle.values() for m in v})
    print(f"  Oracle models: {oracle_models}")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {MODEL_NAME}")
    base_encoder = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    base_encoder = get_peft_model(base_encoder, lora_cfg)
    hidden_dim   = base_encoder.config.hidden_size
    model        = LoRAPromptClassifier(base_encoder, hidden_dim)

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    input_device = next(model.encoder.parameters()).device
    model.head    = model.head.to(input_device)
    model.dropout = model.dropout.to(input_device)
    model.eval()

    # ── Run inference on val CLEAN samples only ────────────────────────────────
    loader = DataLoader(
        clean_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate(tokenizer),
    )

    clean_as_us: List[Dict[str, Any]] = []
    clean_correct = 0

    print("Running classifier inference on CLEAN val samples …")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=90):
            iids  = batch["input_ids"].to(input_device)
            amask = batch["attention_mask"].to(input_device)
            logits = model(iids, amask)
            preds  = logits.argmax(dim=-1).cpu().tolist()
            for meta, pred in zip(batch["meta"], preds):
                pred_name = ID2LABEL[pred]
                if pred_name == "CLEAN":
                    clean_correct += 1
                elif pred_name == "US":
                    # Misclassified CLEAN → US
                    tid = meta["task_id"]
                    oracle_row = oracle.get(tid, {})
                    entry = {
                        "task_id":    tid,
                        "dataset":    meta["dataset"],
                        "pred":       pred_name,
                        "true_label": "CLEAN",
                        "prompt":     meta["text"],
                        "oracle_pass1_by_model": oracle_row,
                        # Overall oracle pass rate across models (on US mutation of same task)
                        "oracle_n_models":    len(oracle_row),
                        "oracle_n_pass":      sum(oracle_row.values()),
                        "oracle_pass_rate":   round(sum(oracle_row.values()) / len(oracle_row), 3)
                                              if oracle_row else None,
                    }
                    clean_as_us.append(entry)

    total_clean = len(clean_val)
    clean_acc   = clean_correct / total_clean if total_clean else 0

    print(f"\nCLEAN samples in val: {total_clean}")
    print(f"  Predicted CLEAN correctly: {clean_correct} ({clean_acc:.1%})")
    print(f"  Predicted as US (errors):  {len(clean_as_us)}")

    # ── Dataset breakdown ──────────────────────────────────────────────────────
    by_ds: Dict[str, int] = defaultdict(int)
    for e in clean_as_us:
        by_ds[e["dataset"]] += 1
    print("\nCLEAN→US breakdown by dataset:")
    for ds, cnt in sorted(by_ds.items()):
        print(f"  {ds}: {cnt}")

    # ── Oracle summary for CLEAN→US cases ─────────────────────────────────────
    with_oracle = [e for e in clean_as_us if e["oracle_n_models"] > 0]
    print(f"\nOf {len(clean_as_us)} CLEAN→US cases, {len(with_oracle)} have oracle data "
          "(US mutation evaluated on same task_id)")
    if with_oracle:
        avg_pass = sum(e["oracle_pass_rate"] for e in with_oracle) / len(with_oracle)
        print(f"  Avg oracle pass@1 rate (US mutation, across models): {avg_pass:.3f}")
        # Compare to overall oracle pass rate for all US tasks
        all_task_rates = []
        for tid, mdict in oracle.items():
            if mdict:
                all_task_rates.append(sum(mdict.values()) / len(mdict))
        overall_avg = sum(all_task_rates) / len(all_task_rates) if all_task_rates else 0
        print(f"  Overall avg oracle pass@1 rate (all US tasks):        {overall_avg:.3f}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(clean_as_us, f, indent=2)
    print(f"\nSaved {len(clean_as_us)} cases → {OUTPUT_JSON}")

    # ── Save readable summary ──────────────────────────────────────────────────
    lines = []
    lines.append(f"CLEAN→US misclassifications ({len(clean_as_us)} cases)\n")
    lines.append("=" * 70 + "\n\n")

    for i, e in enumerate(clean_as_us, 1):
        lines.append(f"[{i}] task_id={e['task_id']}  dataset={e['dataset']}\n")
        if e["oracle_pass1_by_model"]:
            lines.append("    Oracle pass@1 (US mutation, per model):\n")
            for model_name, passed in sorted(e["oracle_pass1_by_model"].items()):
                lines.append(f"      {model_name}: {'✓' if passed else '✗'}\n")
            lines.append(f"    Pass rate: {e['oracle_pass_rate']:.0%} ({e['oracle_n_pass']}/{e['oracle_n_models']})\n")
        else:
            lines.append("    No oracle data for this task_id\n")
        lines.append("    Prompt:\n")
        # Indent and truncate at 600 chars
        prompt = e["prompt"][:600] + ("…" if len(e["prompt"]) > 600 else "")
        for pline in prompt.split("\n"):
            lines.append(f"      {pline}\n")
        lines.append("\n" + "-" * 70 + "\n\n")

    with open(OUTPUT_TXT, "w") as f:
        f.writelines(lines)
    print(f"Saved readable summary → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
