#!/usr/bin/env python3
"""
plot_analysis.py  —  Four publication-quality figures for the mutation robustness paper.

  Fig 1 – By mutation type   : mean Pass@1 drop per mutation (US / LV / SF)
  Fig 2 – By benchmark       : Pass@1 under each mutation, per benchmark
  Fig 3 – By model size      : robustness vs model size tier
  Fig 4 – By model type      : reasoning-API vs open-source models

Output: figures/fig{1..4}_*.pdf + .png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    print(f"  saved → {OUT}/{name}.{{pdf,png}}")

# ── palette ────────────────────────────────────────────────────────────────────
MUT_COLORS = {"US": "#d62728", "LV": "#ff7f0e", "SF": "#2ca02c"}
MUT_ORDER  = ["US", "LV", "SF"]
DS_COLORS  = {"HumanEval": "#1f77b4", "MBPP": "#9467bd", "LCB": "#8c564b"}
SIZE_COLORS= {"Small (≤7B)": "#4878d0", "Large (15–34B)": "#ee854a", "API": "#6acc65"}

# ── load & enrich data ─────────────────────────────────────────────────────────
df = pd.read_csv("results/all_metrics.csv")

MODEL_SIZE = {
    "codellama_CodeLlama-7b-Instruct-hf":          "Small (≤7B)",
    "deepseek-ai_deepseek-coder-6.7b-instruct":    "Small (≤7B)",
    "Qwen_Qwen2.5-Coder-7B-Instruct":             "Small (≤7B)",
    "bigcode_starcoder2-15b-instruct-v0.1":        "Large (15–34B)",
    "mistralai_Codestral-22B-v0.1":               "Large (15–34B)",
    "codellama_CodeLlama-34b-Instruct-hf":         "Large (15–34B)",
    "deepseek-ai_deepseek-coder-33b-instruct":     "Large (15–34B)",
    "Qwen_Qwen2.5-Coder-32B-Instruct":            "Large (15–34B)",
    "gpt-5-mini":                                  "API",
    "claude-sonnet-4-20250514":                    "API",
}
MODEL_TYPE = {m: ("Reasoning (API)" if s == "API" else "Open-source")
              for m, s in MODEL_SIZE.items()}

df["SizeTier"] = df["Model"].map(MODEL_SIZE)
df["ModelType"] = df["Model"].map(MODEL_TYPE)

# pivot: one row per (Model, Dataset), columns = Pass@1 per mutation
pivot = df.pivot_table(index=["Model","Dataset","SizeTier","ModelType"],
                       columns="Mutation", values="Pass@1").reset_index()

# compute absolute drops relative to Orig
for m in MUT_ORDER:
    if m in pivot.columns:
        pivot[f"drop_{m}"] = pivot["Orig"] - pivot[m]

# ── restrict to HumanEval + MBPP for meaningful analysis ──────────────────────
he_mbpp = pivot[pivot["Dataset"].isin(["HumanEval", "MBPP"])].copy()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — By mutation type
# Mean ± std Pass@1 drop across all models, per dataset
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

for ax, dataset in zip(axes, ["HumanEval", "MBPP"]):
    sub = he_mbpp[he_mbpp["Dataset"] == dataset]
    means = [sub[f"drop_{m}"].mean() for m in MUT_ORDER]
    stds  = [sub[f"drop_{m}"].std()  for m in MUT_ORDER]
    bars = ax.bar(MUT_ORDER, means, yerr=stds, capsize=4,
                  color=[MUT_COLORS[m] for m in MUT_ORDER],
                  edgecolor="white", linewidth=0.5, error_kw={"linewidth":1.2})
    ax.set_title(dataset)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Pass@1 drop (pp)" if ax == axes[0] else "")
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Pass@1 drop by mutation type (mean ± std across models)", y=1.01)
fig.tight_layout()
save(fig, "fig1_by_mutation_type")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — By benchmark
# Mean Pass@1 across models, for Orig → US → LV → SF, one line per benchmark
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4))

x_labels = ["Orig", "US", "LV", "SF"]
x_pos    = np.arange(len(x_labels))

for dataset, color in DS_COLORS.items():
    sub = pivot[pivot["Dataset"] == dataset]
    means = []
    for col in x_labels:
        if col in sub.columns:
            val = sub[col].dropna().mean()
        else:
            val = np.nan
        means.append(val)
    # mask missing
    ys = np.array(means, dtype=float)
    mask = ~np.isnan(ys)
    ax.plot(x_pos[mask], ys[mask], marker="o", color=color,
            label=dataset, linewidth=2, markersize=6)
    for xi, yi in zip(x_pos[mask], ys[mask]):
        ax.text(xi, yi + 0.5, f"{yi:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=color)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Mutation")
ax.set_ylabel("Mean Pass@1 (%) across models")
ax.set_title("Benchmark sensitivity to mutations")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig2_by_benchmark")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — By model size
# Grouped bar: mean Pass@1 drop per size tier, for each mutation, HE+MBPP pooled
# Inset scatter: Orig Pass@1 vs US drop (one point per model×dataset)
# ══════════════════════════════════════════════════════════════════════════════
size_order = ["Small (≤7B)", "Large (15–34B)", "API"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# left: grouped bar
ax = axes[0]
n_sizes = len(size_order)
n_muts  = len(MUT_ORDER)
width   = 0.22
offsets = np.linspace(-(n_muts-1)*width/2, (n_muts-1)*width/2, n_muts)
x = np.arange(n_sizes)

for i, mut in enumerate(MUT_ORDER):
    means, stds = [], []
    for tier in size_order:
        sub = he_mbpp[he_mbpp["SizeTier"] == tier][f"drop_{mut}"].dropna()
        means.append(sub.mean() if len(sub) else 0)
        stds.append(sub.std()  if len(sub) else 0)
    ax.bar(x + offsets[i], means, width, yerr=stds, capsize=3,
           color=MUT_COLORS[mut], label=mut, edgecolor="white",
           linewidth=0.4, error_kw={"linewidth":1.0})

ax.set_xticks(x)
ax.set_xticklabels(size_order)
ax.set_ylabel("Mean Pass@1 drop (pp)")
ax.set_title("Drop by model size tier")
ax.legend(frameon=False, title="Mutation")
ax.set_ylim(bottom=0)

# right: scatter Orig Pass@1 vs US drop (one dot per model×dataset)
ax = axes[1]
for tier in size_order:
    sub = he_mbpp[he_mbpp["SizeTier"] == tier].dropna(subset=["Orig","drop_US"])
    ax.scatter(sub["Orig"], sub["drop_US"], color=SIZE_COLORS[tier],
               label=tier, s=55, alpha=0.85, edgecolors="white", linewidth=0.4)

# regression line
all_valid = he_mbpp.dropna(subset=["Orig","drop_US"])
m, b = np.polyfit(all_valid["Orig"], all_valid["drop_US"], 1)
xs = np.linspace(all_valid["Orig"].min(), all_valid["Orig"].max(), 100)
ax.plot(xs, m*xs + b, "k--", linewidth=1, alpha=0.5)

ax.set_xlabel("Orig Pass@1 (%)")
ax.set_ylabel("US mutation drop (pp)")
ax.set_title("Capability vs. robustness under US")
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Effect of model size on mutation robustness", y=1.01)
fig.tight_layout()
save(fig, "fig3_by_model_size")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Reasoning vs open-source
# For each dataset × mutation, compare mean Pass@1 drop for the two groups
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
TYPE_COLORS = {"Reasoning (API)": "#e377c2", "Open-source": "#7f7f7f"}
type_order  = ["Open-source", "Reasoning (API)"]

for ax, dataset in zip(axes, ["HumanEval", "MBPP"]):
    sub = he_mbpp[he_mbpp["Dataset"] == dataset]
    n_muts  = len(MUT_ORDER)
    n_types = len(type_order)
    width   = 0.30
    x       = np.arange(n_muts)
    offsets = [-width/2, width/2]

    for j, mtype in enumerate(type_order):
        means, stds = [], []
        for mut in MUT_ORDER:
            s = sub[sub["ModelType"] == mtype][f"drop_{mut}"].dropna()
            means.append(s.mean() if len(s) else 0)
            stds.append(s.std()  if len(s) else 0)
        ax.bar(x + offsets[j], means, width, yerr=stds, capsize=3,
               color=TYPE_COLORS[mtype], label=mtype, edgecolor="white",
               linewidth=0.4, error_kw={"linewidth":1.0})
        for xi, mean in zip(x + offsets[j], means):
            ax.text(xi, mean + 0.2, f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MUT_ORDER)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Mean Pass@1 drop (pp)" if ax == axes[0] else "")
    ax.set_title(dataset)
    ax.set_ylim(bottom=0)
    if ax == axes[0]:
        ax.legend(frameon=False)

fig.suptitle("Reasoning (API) vs open-source models — mutation sensitivity", y=1.01)
fig.tight_layout()
save(fig, "fig4_by_model_type")
plt.close()

print("\nDone. All figures saved to figures/")
