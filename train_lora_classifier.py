#!/usr/bin/env python3
"""
4-class prompt mutation classifier — LoRA or full fine-tuned Qwen2.5-Coder-1.5B.

Two modes (--mode):
  lora  (default): LoRA adapters on attention layers + Linear(1536, 4) head
  full:            all encoder params unfrozen + Linear(1536, 4) head

Classes: LV=0, SF=1, US=2, CLEAN=3

Usage:
    python train_lora_classifier.py --data_dir . --output_dir ./lora_output --mode lora --tsne
    python train_lora_classifier.py --data_dir . --output_dir ./full_output --mode full --tsne
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, f1_score
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ── Labels ────────────────────────────────────────────────────────────────────

LABEL2ID = {"LV": 0, "SF": 1, "US": 2, "CLEAN": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 4

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
HIDDEN_DIM = 1536  # Qwen2.5-Coder-1.5B hidden size


# ── Model ─────────────────────────────────────────────────────────────────────

class LoRAPromptClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder  # LoRA-wrapped or fully unfrozen depending on --mode
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # No torch.no_grad() here — gradients must flow through LoRA adapters
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, D)
        mask = attention_mask.unsqueeze(-1).float()
        # mean pool over non-padding tokens
        embeddings = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return embeddings  # (B, D)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.get_embeddings(input_ids, attention_mask)
        return self.head(self.dropout(embeddings))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mutated(path: str, label: str) -> List[Tuple[str, int]]:
    """Load a *_with_tests.jsonl mutation file, filtering applicable=False."""
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
                records.append((text, LABEL2ID[label]))
    return records


def load_clean(path: str, text_field: str) -> List[Tuple[str, int]]:
    """Load original prompts from a dataset file as CLEAN class."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "").strip()
            if text:
                records.append((text, LABEL2ID["CLEAN"]))
    return records


def load_all_data(data_dir: str) -> List[Tuple[str, int]]:
    data_dir = Path(data_dir)
    all_records: List[Tuple[str, int]] = []

    mutation_files = [
        (data_dir / "mutations/humanEval_lv_with_tests.jsonl",     "LV"),
        (data_dir / "mutations/humanEval_SF_with_tests.jsonl",     "SF"),
        (data_dir / "mutations/HumanEval_US_with_tests.jsonl",     "US"),
        (data_dir / "mutations/mbpp_LV_with_tests.jsonl",          "LV"),
        (data_dir / "mutations/mbpp_SF_with_tests.jsonl",          "SF"),
        (data_dir / "mutations/mbpp_US_with_tests.jsonl",          "US"),
        (data_dir / "mutations/livecodebench_LV_with_tests.jsonl", "LV"),
        (data_dir / "mutations/livecodebench_SF_with_tests.jsonl", "SF"),
        (data_dir / "mutations/livecodebench_US_with_tests.jsonl", "US"),
    ]

    for path, label in mutation_files:
        if path.exists():
            records = load_mutated(str(path), label)
            print(f"  {path.name}: {len(records):5d} {label}")
            all_records.extend(records)
        else:
            print(f"  {path.name}: not found, skipping")

    clean_files = [
        (data_dir / "datasets/humanEval/HumanEval.jsonl",                "prompt"),
        (data_dir / "datasets/mbpp/mbpp.jsonl",                          "text"),
        (data_dir / "datasets/livecodebench/livecodebench_public.jsonl", "prompt"),
    ]

    for path, field in clean_files:
        if path.exists():
            records = load_clean(str(path), field)
            print(f"  {path.name}: {len(records):5d} CLEAN")
            all_records.extend(records)
        else:
            print(f"  {path.name}: not found, skipping")

    return all_records


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, records: List[Tuple[str, int]], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.records[idx]

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_class_weights(records: List[Tuple[str, int]]) -> torch.Tensor:
    counts = [0] * NUM_CLASSES
    for _, label in records:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in counts]
    print("  Class counts:  " + "  ".join(f"{ID2LABEL[i]}={counts[i]}" for i in range(NUM_CLASSES)))
    print("  Class weights: " + "  ".join(f"{ID2LABEL[i]}={weights[i]:.3f}" for i in range(NUM_CLASSES)))
    return torch.tensor(weights, dtype=torch.float)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        if criterion is not None:
            total_loss += criterion(logits, labels).item() * len(labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    avg_loss = total_loss / len(all_labels) if criterion is not None else None
    return acc, avg_loss, all_preds, all_labels


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_emb, all_labels = [], []
    for input_ids, attention_mask, labels in tqdm(loader, desc="Extracting embeddings"):
        emb = model.get_embeddings(input_ids.to(device), attention_mask.to(device))
        all_emb.append(emb.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.vstack(all_emb), np.array(all_labels)


def plot_pca(embeddings: np.ndarray, labels: np.ndarray, output_path: str, mode: str = "lora"):
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_id in range(NUM_CLASSES):
        mask = labels == cls_id
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=colors[cls_id], label=ID2LABEL[cls_id], alpha=0.6, s=15)
    ax.legend(fontsize=12)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=11)
    label = "LoRA" if mode == "lora" else "Full Fine-tune"
    ax.set_title(f"PCA — {label} Qwen2.5-Coder-1.5B embeddings", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, output_path: str, mode: str = "lora"):
    print("Running t-SNE (this may take a minute)...")
    projected = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_id in range(NUM_CLASSES):
        mask = labels == cls_id
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=colors[cls_id], label=ID2LABEL[cls_id], alpha=0.6, s=15)
    ax.legend(fontsize=12)
    label = "LoRA" if mode == "lora" else "Full Fine-tune"
    ax.set_title(f"t-SNE — {label} Qwen2.5-Coder-1.5B embeddings", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=".",                 help="Root dir containing JSONL files")
    parser.add_argument("--output_dir",  default="./lora_output",     help="Where to save checkpoints and plots")
    parser.add_argument("--model_name",  default=MODEL_NAME)
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--patience",    type=int,   default=5,    help="Early stopping patience (epochs)")
    parser.add_argument("--mode",        default="lora", choices=["lora", "full"],
                        help="'lora' = LoRA adapters only (default); 'full' = unfreeze all encoder params")
    # LoRA hyperparameters (ignored when --mode full)
    parser.add_argument("--lora_r",      type=int,   default=8,    help="LoRA rank")
    parser.add_argument("--lora_alpha",  type=int,   default=16,   help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout",type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--tsne",        action="store_true", help="Save t-SNE and PCA plots after training")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    all_records = load_all_data(args.data_dir)
    random.shuffle(all_records)
    print(f"Total: {len(all_records)} records\n")

    n_val = int(len(all_records) * args.val_split)
    val_records   = all_records[:n_val]
    train_records = all_records[n_val:]
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # ── Tokenizer + base encoder ───────────────────────────────────────────────
    print(f"\nLoading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_encoder = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    # Enable gradient checkpointing to trade compute for memory:
    # activations are recomputed during backward instead of stored → ~60% less activation memory
    base_encoder.gradient_checkpointing_enable()

    # ── Apply LoRA or full fine-tuning ────────────────────────────────────────
    lora_config = None
    if args.mode == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        encoder = get_peft_model(base_encoder, lora_config)
        encoder.print_trainable_parameters()
    else:  # full fine-tuning
        for p in base_encoder.parameters():
            p.requires_grad = True
        encoder = base_encoder
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"Mode: full fine-tuning — all {total_params:,} encoder params trainable")
    encoder = encoder.to(device)

    # ── Classifier ────────────────────────────────────────────────────────────
    model = LoRAPromptClassifier(encoder, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=args.dropout)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.4f}%)")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = PromptDataset(train_records, tokenizer, args.max_length)
    val_ds   = PromptDataset(val_records,   tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=val_ds.collate_fn,   num_workers=4, pin_memory=True)

    # ── Loss & optimizer ──────────────────────────────────────────────────────
    print("\nClass distribution (train):")
    class_weights = compute_class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimize all trainable params: LoRA adapters + classification head
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    ckpt_name     = "best_lora_classifier.pt" if args.mode == "lora" else "best_full_classifier.pt"
    best_ckpt     = os.path.join(args.output_dir, ckpt_name)
    epochs_no_imp = 0
    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()
        avg_train_loss = total_loss / len(train_records)
        val_acc, val_loss, _, _ = evaluate(model, val_loader, device, criterion)

        marker = " ***" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch:2d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v for k, v in model.state_dict().items() if k in trainable_keys},
                "val_acc": val_acc,
                "label2id": LABEL2ID,
                "lora_config": lora_config.__dict__ if lora_config is not None else None,
                "args": vars(args),
            }, best_ckpt)
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val_acc={best_val_acc:.4f})...")
    ckpt = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    val_acc, val_loss, val_preds, val_labels = evaluate(model, val_loader, device, criterion)

    mcc = matthews_corrcoef(val_labels, val_preds)
    f1_macro    = f1_score(val_labels, val_preds, average="macro")
    f1_weighted = f1_score(val_labels, val_preds, average="weighted")
    print(f"\nMCC:        {mcc:.4f}")
    print(f"F1 macro:   {f1_macro:.4f}")
    print(f"F1 weighted:{f1_weighted:.4f}")

    print("\nClassification report:")
    print(classification_report(val_labels, val_preds, target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)]))
    print("Confusion matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    header = "       " + "  ".join(f"{ID2LABEL[i]:>5}" for i in range(NUM_CLASSES))
    print(header)
    for i, row in enumerate(cm):
        print(f"  {ID2LABEL[i]:5} " + "  ".join(f"{v:5d}" for v in row))

    # ── t-SNE / PCA ───────────────────────────────────────────────────────────
    if args.tsne:
        print("\nPreparing t-SNE/PCA on full dataset...")
        all_ds     = PromptDataset(all_records, tokenizer, args.max_length)
        all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=all_ds.collate_fn, num_workers=4)
        embeddings, labels_np = extract_embeddings(model, all_loader, device)
        np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
        np.save(os.path.join(args.output_dir, "labels.npy"),     labels_np)
        plot_pca(embeddings,  labels_np, os.path.join(args.output_dir, "pca.png"),  mode=args.mode)
        plot_tsne(embeddings, labels_np, os.path.join(args.output_dir, "tsne.png"), mode=args.mode)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
