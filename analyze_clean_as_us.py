#!/usr/bin/env python3
"""
Find CLEAN prompts misclassified as US by the LoRA classifier (seed=42),
then check Pass@1 for those prompts across all original model runs.
"""
import json, random, os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SEED       = 42
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
CKPT       = "lora_classifier_outputs/seed42/best_lora_classifier.pt"
VAL_SPLIT  = 0.2
MAX_LEN    = 512
BATCH      = 32
LABEL2ID   = {"LV": 0, "SF": 1, "US": 2, "CLEAN": 3}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}

# ── reproduce data loading with metadata ──────────────────────────────────────
def load_mutated(path, label):
    recs = []
    for line in open(path):
        obj = json.loads(line.strip())
        if not obj.get("applicable", True): continue
        text = obj.get("mutated_prompt", "").strip()
        tid  = obj.get("task_id", obj.get("problem_id", ""))
        if text:
            recs.append({"text": text, "label": LABEL2ID[label],
                         "source": Path(path).name, "task_id": tid})
    return recs

def load_clean(path, field):
    recs = []
    for line in open(path):
        obj = json.loads(line.strip())
        text = obj.get(field, "").strip()
        tid  = obj.get("task_id", obj.get("problem_id", ""))
        if text:
            recs.append({"text": text, "label": LABEL2ID["CLEAN"],
                         "source": Path(path).name, "task_id": tid})
    return recs

# Exact record counts from training log (seed42 run) — required to reproduce val split
TRAINING_CAPS = {
    "humanEval_lv_with_tests.jsonl":     163,
    "humanEval_SF_with_tests.jsonl":     164,
    "HumanEval_US_with_tests.jsonl":     160,
    "mbpp_LV_with_tests.jsonl":          974,
    "mbpp_SF_with_tests.jsonl":          974,
    "mbpp_US_with_tests.jsonl":          974,
    "livecodebench_LV_with_tests.jsonl": 1054,
    "livecodebench_SF_with_tests.jsonl": 1055,
    "livecodebench_US_with_tests.jsonl": 1055,
    "HumanEval.jsonl":                   164,
    "mbpp.jsonl":                        974,
    "livecodebench_public.jsonl":        1055,
}

all_records = []
for path, lbl in [
    ("mutations/humanEval_lv_with_tests.jsonl",     "LV"),
    ("mutations/humanEval_SF_with_tests.jsonl",     "SF"),
    ("mutations/HumanEval_US_with_tests.jsonl",     "US"),
    ("mutations/mbpp_LV_with_tests.jsonl",          "LV"),
    ("mutations/mbpp_SF_with_tests.jsonl",          "SF"),
    ("mutations/mbpp_US_with_tests.jsonl",          "US"),
    ("mutations/livecodebench_LV_with_tests.jsonl", "LV"),
    ("mutations/livecodebench_SF_with_tests.jsonl", "SF"),
    ("mutations/livecodebench_US_with_tests.jsonl", "US"),
]:
    if Path(path).exists():
        recs = load_mutated(path, lbl)
        cap  = TRAINING_CAPS.get(Path(path).name)
        if cap is not None:
            recs = recs[:cap]
        all_records.extend(recs)

for path, field in [
    ("datasets/humanEval/HumanEval.jsonl",                "prompt"),
    ("datasets/mbpp/mbpp.jsonl",                          "text"),
    ("datasets/livecodebench/livecodebench_public.jsonl", "prompt"),
]:
    if Path(path).exists():
        recs = load_clean(path, field)
        cap  = TRAINING_CAPS.get(Path(path).name)
        if cap is not None:
            recs = recs[:cap]
        all_records.extend(recs)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
random.shuffle(all_records)

n_val      = int(len(all_records) * VAL_SPLIT)
val_records = all_records[:n_val]
print(f"Val set: {len(val_records)} records")
clean_val = [(i, r) for i, r in enumerate(val_records) if r["label"] == LABEL2ID["CLEAN"]]
print(f"CLEAN in val: {len(clean_val)}")

# ── load model + checkpoint ───────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                  torch_dtype=torch.float16)
lora_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      lora_dropout=0.05, bias="none")
encoder = get_peft_model(base, lora_cfg)

ckpt    = torch.load(CKPT, map_location="cpu", weights_only=False)
sd      = ckpt["model_state_dict"]
# checkpoint keys are prefixed with "encoder." (from LoRAPromptClassifier wrapper)
# strip that prefix so they match the bare encoder's key namespace
lora_sd = {k.removeprefix("encoder."): v for k, v in sd.items() if "lora_" in k}
head_w  = sd["head.weight"]
head_b  = sd["head.bias"]
encoder.load_state_dict(lora_sd, strict=False)

head = nn.Linear(1536, 4)
head.weight = nn.Parameter(head_w)
head.bias   = nn.Parameter(head_b)

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device).eval()
head.to(device).eval()

# ── run inference on val set ──────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, records): self.r = records
    def __len__(self): return len(self.r)
    def __getitem__(self, i): return self.r[i]["text"], self.r[i]["label"]

def collate(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

loader = DataLoader(DS(val_records), batch_size=BATCH,
                    shuffle=False, collate_fn=collate)

all_preds = []
with torch.no_grad():
    for ids, mask, _ in tqdm(loader, desc="Inference"):
        ids, mask = ids.to(device), mask.to(device)
        out  = encoder(input_ids=ids, attention_mask=mask)
        emb  = (out.last_hidden_state.float() * mask.unsqueeze(-1).float()).sum(1) \
               / mask.float().sum(1, keepdim=True).clamp(min=1)
        logits = head(emb)
        all_preds.extend(logits.argmax(-1).cpu().tolist())

# ── find CLEAN predicted as US ────────────────────────────────────────────────
clean_as_us = [(i, val_records[i]) for i, p in enumerate(all_preds)
               if val_records[i]["label"] == LABEL2ID["CLEAN"] and p == LABEL2ID["US"]]
print(f"\nCLEAN predicted as US: {len(clean_as_us)}")

# ── load all original result files ───────────────────────────────────────────
# Build a lookup: task_id -> {model -> Pass@1}
print("\nLoading result files...")
orig_files = list(Path("results").rglob("*.json"))
# keep only "Orig" / "public" / base runs (not mutations)
orig_files = [f for f in orig_files if not any(x in f.name for x in
              ["_LV_", "_SF_", "_US_", "lv_", "sf_", "us_", "_lv.", "_sf.", "_us."])]

task_pass = {}   # task_id -> {model_key: bool}
for fpath in orig_files:
    try:
        data = json.load(open(fpath))
    except: continue
    if not isinstance(data, list) or not data: continue
    model_key = fpath.parts[1] + "/" + fpath.stem
    for r in data:
        tid = r.get("task_id", "")
        if not tid: continue
        p1  = r.get("Pass@1")
        if tid not in task_pass:
            task_pass[tid] = {}
        if p1 is not None:
            task_pass[tid][model_key] = bool(p1)

# ── report Pass@1 for misclassified CLEAN prompts ────────────────────────────
print(f"\n{'task_id':<30} {'source':<40} {'models_found':<6} {'pass_rate'}")
print("-" * 100)

model_totals = {}   # model_key -> [pass_count, total]
task_results = []

for idx, rec in clean_as_us:
    tid = rec["task_id"]
    src = rec["source"]
    passes = task_pass.get(tid, {})
    n = len(passes)
    p = sum(passes.values())
    rate = p/n if n else float("nan")
    task_results.append((tid, src, passes, rate))
    for mk, v in passes.items():
        if mk not in model_totals: model_totals[mk] = [0, 0]
        model_totals[mk][0] += int(v)
        model_totals[mk][1] += 1
    print(f"{tid:<30} {src:<40} {n:<6} {rate:.2f}" if n else
          f"{tid:<30} {src:<40} {'?':<6} n/a")

print("\n\n=== Pass@1 across models for CLEAN-predicted-as-US examples ===")
print(f"{'Model':<60} {'Pass':<6} {'Total':<6} {'Rate'}")
print("-" * 85)
for mk in sorted(model_totals):
    p, t = model_totals[mk]
    print(f"{mk:<60} {p:<6} {t:<6} {p/t:.3f}")

# save
out = [{"task_id": r["task_id"], "source": r["source"],
        "pass_results": {m: bool(v) for m, v in task_pass.get(r["task_id"],{}).items()}}
       for _, r in clean_as_us]
json.dump(out, open("clean_as_us_analysis.json", "w"), indent=2)
print(f"\nSaved {len(out)} records → clean_as_us_analysis.json")
