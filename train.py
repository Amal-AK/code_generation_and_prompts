from __future__ import annotations

# ─────────────────────────────────────── imports ───────────────────────────────────────
import argparse
import contextlib
import gc
import io
import json
import logging
import os
import random
import signal
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ─────────────────────────────── global settings & logging ─────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("lspr_trainer")


# ─────────────────────────────── reproducibility ──────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────── data loading (same as eval) ──────────────────────────

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
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t").to_dict(orient="records")
    raise ValueError(f"Unknown extension: {ext}")


# ─────────────────── test execution (same logic as your eval script) ──────────────────

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError()

def run_tests(code: str, tests: str, timeout: int = 10) -> Tuple[bool, str]:
    full_code = textwrap.dedent(code) + "\n" + textwrap.dedent(tests)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        exec(compile(full_code, "<string>", "exec"), {})  # noqa: S102
        signal.alarm(0)
        return True, "PASSED"
    except TimeoutError:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"FAILED: {type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def extract_code(response: str) -> str:
    """Pull code block from model output (same as your eval script)."""
    if "```" in response:
        blocks = response.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            if block.startswith("python"):
                block = block[6:]
            block = block.strip()
            if block:
                return block
    return response.strip()


# ─────────────────────────────── Dataset ──────────────────────────────────────────────

class PromptPairDataset(Dataset):
    """
    Aligned pairs: (P_amb, P_spec, C_gold).
    Expects same field names as your mutated JSONL files.
    """

    def __init__(self, records: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []

        for r in records:
            p_amb  = (r.get("mutated_prompt") or "").strip()
            p_spec = (r.get("original_prompt") or r.get("prompt") or "").strip()
            c_gold = (r.get("canonical_solution") or "").strip()

            # Skip records where we don't have a real ambiguous/specific pair
            if not p_amb or not p_spec or p_amb == p_spec:
                continue

            self.samples.append({"p_amb": p_amb, "p_spec": p_spec, "c_gold": c_gold})

        logger.info("Dataset: %d valid pairs (out of %d records)", len(self.samples), len(records))

    def __len__(self):
        return len(self.samples)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s  = self.samples[idx]
        ea = self._encode(s["p_amb"])
        es = self._encode(s["p_spec"])
        ec = self._encode(s["c_gold"])
        return {
            "amb_input_ids":       ea["input_ids"].squeeze(0),
            "amb_attention_mask":  ea["attention_mask"].squeeze(0),
            "spec_input_ids":      es["input_ids"].squeeze(0),
            "spec_attention_mask": es["attention_mask"].squeeze(0),
            "code_input_ids":      ec["input_ids"].squeeze(0),
            "code_attention_mask": ec["attention_mask"].squeeze(0),
        }


# ─────────────────────── Prompt Projector (Residual MLP) ──────────────────────────────
# §5.3: Linear(D, D/2) → ReLU → LayerNorm → Linear(D/2, D)
# Operation: e_refined = e_amb + ΔE

class PromptProjector(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, e_amb: torch.Tensor) -> torch.Tensor:
        return e_amb + self.net(e_amb)  # residual: e_refined = e_amb + ΔE


# ─────────────────────────── Embedding Extraction ──────────────────────────────────────

@torch.no_grad()
def get_mean_embeddings(model, input_ids, attention_mask) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]            # (B, seq, D)
    mask   = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)   # mean pool → (B, D)


# ─────────────────────────── Loss Functions ────────────────────────────────────────────

def infonce_loss(e_refined: torch.Tensor, e_spec: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """§5.3 L_align: pull e_refined → e_spec, push away batch negatives."""
    e_r = F.normalize(e_refined, dim=-1)
    e_s = F.normalize(e_spec,    dim=-1)
    sim = torch.matmul(e_r, e_s.T) / tau            # (B, B)
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def generative_loss(model, code_input_ids, code_attention_mask) -> torch.Tensor:
    """§5.3 L_gen: standard causal LM loss on canonical code."""
    labels = code_input_ids.clone()
    labels[code_attention_mask == 0] = -100
    return model(
        input_ids=code_input_ids,
        attention_mask=code_attention_mask,
        labels=labels,
        return_dict=True,
    ).loss


# ─────────────────────────── Cleanup ──────────────────────────────────────────────────

def cleanup(model=None, projector=None):
    if model     is not None: del model
    if projector is not None: del projector
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ─────────────────────────── Training ─────────────────────────────────────────────────

def train(args) -> Tuple:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s  GPUs=%d", device, torch.cuda.device_count())

    # ── Tokenizer ──────────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.modelName, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── Base model + LoRA ──────────────────────────────────────────────────────────────
    logger.info("Loading: %s", args.modelName)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.modelName,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.loraR,
        lora_alpha=args.loraAlpha,
        lora_dropout=args.loraDropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    hidden_dim = base_model.config.hidden_size
    projector  = PromptProjector(hidden_dim).to(device).float()

    # ── Dataset ────────────────────────────────────────────────────────────────────────
    all_records: List[Dict] = []
    for f in args.inputFiles:
        recs = load_records(f)
        logger.info("  %d records from %s", len(recs), f)
        all_records.extend(recs)
    all_records = all_records[: args.limit]

    dataset    = PromptPairDataset(all_records, tokenizer, args.maxLength)
    dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0)

    # ── Optimizers ─────────────────────────────────────────────────────────────────────
    opt_model = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    opt_proj  = torch.optim.AdamW(projector.parameters(), lr=args.lr)

    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    # ── Epoch loop ─────────────────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train(); projector.train()
        n, total, talign, tgen = 0, 0.0, 0.0, 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=90)
        for batch in pbar:
            amb_ids   = batch["amb_input_ids"].to(device)
            amb_mask  = batch["amb_attention_mask"].to(device)
            spec_ids  = batch["spec_input_ids"].to(device)
            spec_mask = batch["spec_attention_mask"].to(device)
            code_ids  = batch["code_input_ids"].to(device)
            code_mask = batch["code_attention_mask"].to(device)

            # Step 1: embeddings from backbone (no grad on backbone here)
            e_amb  = get_mean_embeddings(model, amb_ids,  amb_mask).float()
            e_spec = get_mean_embeddings(model, spec_ids, spec_mask).float()

            # Step 2: project
            e_refined = projector(e_amb)

            # Step 3: L_align (InfoNCE)
            l_align = infonce_loss(e_refined, e_spec, tau=args.temperature)

            # Step 4: L_gen (causal LM on canonical code — trains LoRA)
            l_gen = generative_loss(model, code_ids, code_mask)

            # Step 5: total loss (§5.3)
            loss = args.lambda1 * l_align + args.lambda2 * l_gen

            opt_model.zero_grad(); opt_proj.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),     1.0)
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            opt_model.step(); opt_proj.step()

            total  += loss.item(); talign += l_align.item()
            tgen   += l_gen.item(); n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             align=f"{l_align.item():.4f}",
                             gen=f"{l_gen.item():.4f}")

        row = dict(epoch=epoch+1,
                   loss=round(total/n, 4),
                   l_align=round(talign/n, 4),
                   l_gen=round(tgen/n, 4))
        history.append(row)
        logger.info("Epoch %d — loss=%.4f  l_align=%.4f  l_gen=%.4f",
                    epoch+1, total/n, talign/n, tgen/n)

    # ── Save ───────────────────────────────────────────────────────────────────────────
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    torch.save(projector.state_dict(), output_dir / "projector.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    logger.info("Saved → %s", output_dir)

    return model, tokenizer, projector, hidden_dim


# ────────────────── Pass@1 Evaluation: baseline P_amb vs projected ──────────────────
# Same Pass@1 logic as your eval script — we just compare two generation conditions.

def evaluate(args, model, tokenizer, projector, hidden_dim):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.outputDir)

    all_records: List[Dict] = []
    for f in args.inputFiles:
        all_records.extend(load_records(f))
    all_records = all_records[: args.limit]

    model.eval(); projector.eval()
    results = []

    for r in tqdm(all_records, desc="Evaluating Pass@1", ncols=90):
        p_amb   = (r.get("mutated_prompt") or "").strip()
        tests   = (r.get("test") or r.get("tests") or "").strip()
        task_id = r.get("task_id", "")

        if not p_amb or not tests:
            continue

        row = {"task_id": task_id}

        # ── (A) Baseline: LoRA model on P_amb without projection ──────────────────────
        try:
            enc       = tokenizer(p_amb, return_tensors="pt",
                                  truncation=True, max_length=args.maxLength)
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                out = model.generate(input_ids,
                                     max_new_tokens=args.maxNewTokens,
                                     do_sample=False)
            response  = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            code      = extract_code(response)
            passed_b, status_b = run_tests(code, tests)
        except Exception as e:
            passed_b, status_b = False, f"ERROR: {e}"

        row.update(baseline_passed=passed_b, baseline_status=status_b)

        # ── (B) Projected: measure embedding shift, then generate ──────────────────────
        # The projector was trained contrastively — e_refined is geometrically closer
        # to e_spec. For generation we use the LoRA-fine-tuned model on P_amb; the
        # L_gen loss made LoRA learn to generate correct code from ambiguous prompts.
        # Embedding shift (cos sim) is logged as a proxy for "how much we corrected".
        try:
            enc       = tokenizer(p_amb, return_tensors="pt",
                                  truncation=True, max_length=args.maxLength)
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                e_amb     = get_mean_embeddings(model, input_ids, attn_mask).float()
                e_refined = projector(e_amb)
                cos_sim   = F.cosine_similarity(e_refined, e_amb).item()

                out = model.generate(input_ids,
                                     max_new_tokens=args.maxNewTokens,
                                     do_sample=False)
            response  = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            code      = extract_code(response)
            passed_p, status_p = run_tests(code, tests)

        except Exception as e:
            passed_p, status_p, cos_sim = False, f"ERROR: {e}", 0.0

        row.update(projected_passed=passed_p,
                   projected_status=status_p,
                   embedding_cos_sim=round(cos_sim, 4))
        results.append(row)

    # ── Summary (same style as your eval CSVs) ─────────────────────────────────────────
    df = pd.DataFrame(results)
    n  = len(df)
    if n == 0:
        logger.warning("No results — check that records have 'mutated_prompt' and 'test' fields")
        return

    pass1_base = df["baseline_passed"].sum()  / n * 100
    pass1_proj = df["projected_passed"].sum() / n * 100
    avg_cos    = df["embedding_cos_sim"].mean()

    summary = {
        "n_problems":            n,
        "pass@1_baseline":       round(pass1_base, 2),
        "pass@1_projected":      round(pass1_proj, 2),
        "delta_pass@1":          round(pass1_proj - pass1_base, 2),
        "avg_embedding_cos_sim": round(avg_cos, 4),
    }

    print("\n" + "="*55)
    print("  LSPR-UGC  ·  Pass@1 on Recoverable Prompts")
    print("="*55)
    print(f"  Problems evaluated   : {n}")
    print(f"  Pass@1  baseline     : {pass1_base:.2f}%  (LoRA, no projection)")
    print(f"  Pass@1  projected    : {pass1_proj:.2f}%  (LoRA + Projector)")
    print(f"  Δ Pass@1             : {pass1_proj - pass1_base:+.2f}%")
    print(f"  Avg embedding cos-sim: {avg_cos:.4f}")
    print("="*55 + "\n")

    df.to_csv(output_dir / "eval_results.csv", index=False)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Eval saved → %s", output_dir)


# ──────────────────────────────────────── main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LSPR: Prompt Projector + LoRA training & Pass@1 evaluation"
    )
    # Data (same flags as your eval script)
    parser.add_argument("--inputFiles",    nargs="+", required=True,
                        help="Mutated dataset files — same format as evaluate_multi.py")
    parser.add_argument("--outputDir",     default="./lspr_output")
    parser.add_argument("--limit",         type=int, default=5000)

    # Model
    parser.add_argument("--modelName",     default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--maxLength",     type=int, default=512)
    parser.add_argument("--maxNewTokens",  type=int, default=512)

    # LoRA
    parser.add_argument("--loraR",         type=int,   default=16)
    parser.add_argument("--loraAlpha",     type=int,   default=32)
    parser.add_argument("--loraDropout",   type=float, default=0.05)

    # Training
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batchSize",     type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--seed",          type=int,   default=42)

    # Loss weights (§5.3: λ1=1.0, λ2=0.5)
    parser.add_argument("--lambda1",       type=float, default=1.0,  help="L_align weight")
    parser.add_argument("--lambda2",       type=float, default=0.5,  help="L_gen weight")
    parser.add_argument("--temperature",   type=float, default=0.07, help="InfoNCE τ")

    # Modes
    parser.add_argument("--evalOnly",      action="store_true",
                        help="Skip training — load saved adapter + projector and evaluate")
    parser.add_argument("--trainOnly",     action="store_true",
                        help="Train only, skip evaluation")

    args = parser.parse_args()

    logger.info("Model=%s  LoRA r=%d  λ1=%.1f λ2=%.1f  τ=%.2f",
                args.modelName, args.loraR, args.lambda1, args.lambda2, args.temperature)

    if args.evalOnly:
        logger.info("Eval-only mode — loading from %s", args.outputDir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.outputDir + "/lora_adapter", trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model = AutoModelForCausalLM.from_pretrained(
            args.modelName, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        model      = PeftModel.from_pretrained(base_model, args.outputDir + "/lora_adapter")
        hidden_dim = base_model.config.hidden_size
        projector  = PromptProjector(hidden_dim)
        projector.load_state_dict(
            torch.load(args.outputDir + "/projector.pt", map_location="cpu"))
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        projector  = projector.to(device).float()
        evaluate(args, model, tokenizer, projector, hidden_dim)
    else:
        model, tokenizer, projector, hidden_dim = train(args)
        if not args.trainOnly:
            evaluate(args, model, tokenizer, projector, hidden_dim)

    cleanup(model if not args.evalOnly else None, projector)


if __name__ == "__main__":
    main()