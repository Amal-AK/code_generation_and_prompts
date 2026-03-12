#!/usr/bin/env python3
"""
plot_rq1.py — RQ1: Model robustness to prompt mutations.

Four comparison dimensions, each as a separate figure:
  Fig 1 – By mutation type   : mean Pass@1 drop per mutation, per dataset
  Fig 2 – By dataset         : which benchmark is most sensitive?
  Fig 3 – By model size      : Small / Large / API robustness profiles
  Fig 4 – By model type      : Reasoning (API) vs Open-source
  Fig 5 – Full heatmap       : all models × all (dataset × mutation) conditions

Output: figures/rq1_fig{1..5}_*.pdf + .png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    print(f"  saved → {OUT}/{name}.{{pdf,png}}")


# ── palettes ───────────────────────────────────────────────────────────────────
MUT_COLORS  = {"US": "#d62728", "LV": "#ff7f0e", "SF": "#2ca02c"}
MUT_ORDER   = ["US", "LV", "SF"]
DS_COLORS   = {"HumanEval": "#1f77b4", "MBPP": "#9467bd", "LCB": "#8c564b"}
DS_ORDER    = ["HumanEval", "MBPP", "LCB"]
SIZE_COLORS = {"Small (≤7B)": "#4878d0", "Large (15–34B)": "#ee854a", "API": "#6acc65"}
SIZE_ORDER  = ["Small (≤7B)", "Large (15–34B)", "API"]
TYPE_COLORS = {"Open-source": "#7f7f7f", "Reasoning (API)": "#e377c2"}
TYPE_ORDER  = ["Open-source", "Reasoning (API)"]

MODEL_LABEL = {
    "codellama_CodeLlama-7b-Instruct-hf":       "CodeLlama-7B",
    "deepseek-ai_deepseek-coder-6.7b-instruct": "DeepSeek-6.7B",
    "Qwen_Qwen2.5-Coder-7B-Instruct":           "Qwen2.5-7B",
    "bigcode_starcoder2-15b-instruct-v0.1":     "StarCoder2-15B",
    "mistralai_Codestral-22B-v0.1":             "Codestral-22B",
    "codellama_CodeLlama-34b-Instruct-hf":      "CodeLlama-34B",
    "deepseek-ai_deepseek-coder-33b-instruct":  "DeepSeek-33B",
    "Qwen_Qwen2.5-Coder-32B-Instruct":          "Qwen2.5-32B",
    "gpt-5-mini":                               "GPT-5-mini",
    "claude-sonnet-4-20250514":                 "Claude Sonnet",
}
MODEL_SIZE = {
    "codellama_CodeLlama-7b-Instruct-hf":       "Small (≤7B)",
    "deepseek-ai_deepseek-coder-6.7b-instruct": "Small (≤7B)",
    "Qwen_Qwen2.5-Coder-7B-Instruct":           "Small (≤7B)",
    "bigcode_starcoder2-15b-instruct-v0.1":     "Large (15–34B)",
    "mistralai_Codestral-22B-v0.1":             "Large (15–34B)",
    "codellama_CodeLlama-34b-Instruct-hf":      "Large (15–34B)",
    "deepseek-ai_deepseek-coder-33b-instruct":  "Large (15–34B)",
    "Qwen_Qwen2.5-Coder-32B-Instruct":          "Large (15–34B)",
    "gpt-5-mini":                               "API",
    "claude-sonnet-4-20250514":                 "API",
}
MODEL_TYPE = {m: ("Reasoning (API)" if s == "API" else "Open-source")
              for m, s in MODEL_SIZE.items()}

# ── load & enrich ──────────────────────────────────────────────────────────────
df = pd.read_csv("results/all_metrics.csv")
df["SizeTier"]  = df["Model"].map(MODEL_SIZE)
df["ModelType"] = df["Model"].map(MODEL_TYPE)
df["ShortName"] = df["Model"].map(MODEL_LABEL)

pivot = df.pivot_table(
    index=["Model", "Dataset", "SizeTier", "ModelType", "ShortName"],
    columns="Mutation", values="Pass@1"
).reset_index()

for m in MUT_ORDER:
    if m in pivot.columns:
        pivot[f"drop_{m}"] = pivot["Orig"] - pivot[m]

# all datasets including LCB
all_ds = pivot.copy()
# HE + MBPP only (LCB drops are near-zero and compress the scale)
he_mbpp = pivot[pivot["Dataset"].isin(["HumanEval", "MBPP"])].copy()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — By mutation type (all 3 datasets, 1 subplot each)
# Mean ± std Pass@1 drop across all models, per mutation, per dataset
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for ax, dataset in zip(axes, DS_ORDER):
    sub = all_ds[all_ds["Dataset"] == dataset]
    means = [sub[f"drop_{m}"].mean() for m in MUT_ORDER]
    stds  = [sub[f"drop_{m}"].std()  for m in MUT_ORDER]
    bars  = ax.bar(MUT_ORDER, means, yerr=stds, capsize=4,
                   color=[MUT_COLORS[m] for m in MUT_ORDER],
                   edgecolor="white", linewidth=0.5,
                   error_kw={"linewidth": 1.2})
    ax.set_title(dataset)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Pass@1 drop (pp)" if ax == axes[0] else "")
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Fig 1 — Pass@1 drop by mutation type (mean ± std across all models)", y=1.01)
fig.tight_layout()
save(fig, "rq1_fig1_by_mutation_type")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — By dataset
# Grouped bars: x = dataset, bars = mutation type, showing mean drop
# All 3 datasets side-by-side — immediately shows LCB immunity
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

n_ds   = len(DS_ORDER)
n_muts = len(MUT_ORDER)
width  = 0.22
x      = np.arange(n_ds)
offsets = np.linspace(-(n_muts - 1) * width / 2, (n_muts - 1) * width / 2, n_muts)

for i, mut in enumerate(MUT_ORDER):
    means, stds = [], []
    for ds in DS_ORDER:
        sub = all_ds[all_ds["Dataset"] == ds][f"drop_{mut}"].dropna()
        means.append(sub.mean() if len(sub) else 0)
        stds.append(sub.std()  if len(sub) else 0)
    bars = ax.bar(x + offsets[i], means, width, yerr=stds, capsize=3,
                  color=MUT_COLORS[mut], label=mut, edgecolor="white",
                  linewidth=0.4, error_kw={"linewidth": 1.0})
    for xi, mean in zip(x + offsets[i], means):
        ax.text(xi, max(mean + 0.2, 0.3), f"{mean:.1f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(DS_ORDER)
ax.set_xlabel("Dataset")
ax.set_ylabel("Mean Pass@1 drop (pp) across all models")
ax.set_title("Fig 2 — Benchmark sensitivity to mutations")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, title="Mutation")
fig.tight_layout()
save(fig, "rq1_fig2_by_dataset")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — By model size  (3 subplots: HumanEval | MBPP | LCB)
# Grouped bars: x = size tier, bars = mutation, one subplot per dataset
# + scatter inset on right: Orig vs US drop (HE+MBPP only, scale matters there)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})

for ax, dataset in zip(axes[:3], DS_ORDER):
    sub_ds = all_ds[all_ds["Dataset"] == dataset]
    n_s    = len(SIZE_ORDER)
    n_m    = len(MUT_ORDER)
    width  = 0.22
    x      = np.arange(n_s)
    offs   = np.linspace(-(n_m - 1) * width / 2, (n_m - 1) * width / 2, n_m)

    for i, mut in enumerate(MUT_ORDER):
        means, stds = [], []
        for tier in SIZE_ORDER:
            sub = sub_ds[sub_ds["SizeTier"] == tier][f"drop_{mut}"].dropna()
            means.append(sub.mean() if len(sub) else 0)
            stds.append(sub.std()  if len(sub) else 0)
        ax.bar(x + offs[i], means, width, yerr=stds, capsize=3,
               color=MUT_COLORS[mut], label=mut, edgecolor="white",
               linewidth=0.4, error_kw={"linewidth": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in SIZE_ORDER], fontsize=9)
    ax.set_title(dataset)
    ax.set_ylabel("Mean Pass@1 drop (pp)" if ax == axes[0] else "")
    ax.set_ylim(bottom=0)
    if ax == axes[0]:
        ax.legend(frameon=False, title="Mutation", fontsize=9)

# scatter: Orig Pass@1 vs US drop (HE+MBPP, meaningful drops only)
ax = axes[3]
for tier in SIZE_ORDER:
    sub = he_mbpp[he_mbpp["SizeTier"] == tier].dropna(subset=["Orig", "drop_US"])
    ax.scatter(sub["Orig"], sub["drop_US"],
               color=SIZE_COLORS[tier], label=tier,
               s=60, alpha=0.85, edgecolors="white", linewidth=0.4)

valid = he_mbpp.dropna(subset=["Orig", "drop_US"])
m_fit, b_fit = np.polyfit(valid["Orig"], valid["drop_US"], 1)
xs = np.linspace(valid["Orig"].min(), valid["Orig"].max(), 100)
ax.plot(xs, m_fit * xs + b_fit, "k--", linewidth=1, alpha=0.5, label="Trend")
ax.set_xlabel("Orig Pass@1 (%)")
ax.set_ylabel("US drop (pp)")
ax.set_title("Capability vs. robustness")
ax.legend(frameon=False, fontsize=8)

fig.suptitle("Fig 3 — Robustness by model size tier", y=1.01)
fig.tight_layout()
save(fig, "rq1_fig3_by_model_size")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — By model type: Reasoning (API) vs Open-source
# 3 subplots (one per dataset), grouped bars: x = mutation, bars = model type
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)

for ax, dataset in zip(axes, DS_ORDER):
    sub_ds = all_ds[all_ds["Dataset"] == dataset]
    n_m    = len(MUT_ORDER)
    n_t    = len(TYPE_ORDER)
    width  = 0.30
    x      = np.arange(n_m)
    offs   = [-width / 2, width / 2]

    for j, mtype in enumerate(TYPE_ORDER):
        means, stds = [], []
        for mut in MUT_ORDER:
            s = sub_ds[sub_ds["ModelType"] == mtype][f"drop_{mut}"].dropna()
            means.append(s.mean() if len(s) else 0)
            stds.append(s.std()  if len(s) else 0)
        bars = ax.bar(x + offs[j], means, width, yerr=stds, capsize=3,
                      color=TYPE_COLORS[mtype], label=mtype, edgecolor="white",
                      linewidth=0.4, error_kw={"linewidth": 1.0})
        for xi, mean in zip(x + offs[j], means):
            ax.text(xi, max(mean + 0.15, 0.2), f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MUT_ORDER)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Mean Pass@1 drop (pp)" if ax == axes[0] else "")
    ax.set_title(dataset)
    ax.set_ylim(bottom=0)
    if ax == axes[0]:
        ax.legend(frameon=False)

fig.suptitle("Fig 4 — Reasoning (API) vs open-source: mutation sensitivity", y=1.01)
fig.tight_layout()
save(fig, "rq1_fig4_by_model_type")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Full heatmap: all models × all (dataset × mutation) conditions
# Rows  = models sorted by size tier then Orig HumanEval (best → worst)
# Cols  = 9 cells: HumanEval×{US,LV,SF} | MBPP×{US,LV,SF} | LCB×{US,LV,SF}
# Color = diverging: red = drop, white = 0, blue = slight gain
# ══════════════════════════════════════════════════════════════════════════════
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

col_order = [(ds, mut) for ds in DS_ORDER for mut in MUT_ORDER]

# sort models: API (best) → Large → Small; within tier by Orig HE descending
tier_rank = {"API": 0, "Large (15–34B)": 1, "Small (≤7B)": 2}
model_order_df = (
    pivot[pivot["Dataset"] == "HumanEval"][["Model", "ShortName", "SizeTier", "Orig"]]
    .drop_duplicates()
    .assign(tier_rank=lambda d: d["SizeTier"].map(tier_rank))
    .sort_values(["tier_rank", "Orig"], ascending=[True, False])
)
model_order = model_order_df["Model"].tolist()
row_labels  = model_order_df["ShortName"].tolist()
row_tiers   = model_order_df["SizeTier"].tolist()

# build matrix — absolute Pass@1 drop in pp: Orig − Mutated
matrix = np.full((len(model_order), len(col_order)), np.nan)
for i, model in enumerate(model_order):
    for j, (ds, mut) in enumerate(col_order):
        row = pivot[(pivot["Model"] == model) & (pivot["Dataset"] == ds)]
        if not row.empty and f"drop_{mut}" in row.columns:
            val = row[f"drop_{mut}"].values[0]
            if not np.isnan(val):
                matrix[i, j] = val

# find tier group boundaries
tier_boundaries = []
prev = row_tiers[0]
for i, t in enumerate(row_tiers[1:], 1):
    if t != prev:
        tier_boundaries.append(i)
        prev = t
tier_label_rows = {}
for i, tier in enumerate(row_tiers):
    tier_label_rows.setdefault(tier, []).append(i)

# ── figure layout ──────────────────────────────────────────────────────────────
n_rows, n_cols = len(model_order), len(col_order)
fig, ax = plt.subplots(figsize=(13, 7))
fig.subplots_adjust(left=0.25, right=0.88, top=0.78, bottom=0.10)

# diverging colormap: blue (gain) ← white (0) → red (drop)
vmax = max(15.0, float(np.nanpercentile(matrix, 97)))
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=vmax)
im   = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)

# ── cell text annotations ──────────────────────────────────────────────────────
for i in range(n_rows):
    for j in range(n_cols):
        val = matrix[i, j]
        if np.isnan(val):
            continue
        norm_val = norm(val)          # 0–1 in colormap space
        # white text on saturated cells, black on pale cells
        txt_color = "white" if (norm_val > 0.72 or norm_val < 0.18) else "#222222"
        weight    = "bold" if j % 3 == 0 else "normal"   # bold for US columns
        ax.text(j, i, f"{val:+.1f}" if val < 0 else f"{val:.1f}",
                ha="center", va="center", fontsize=8.5,
                color=txt_color, fontweight=weight)

# ── column x-tick labels (just mutation abbreviation) ─────────────────────────
ax.set_xticks(range(n_cols))
ax.set_xticklabels(["US", "LV", "SF"] * 3, fontsize=10)
for i, lbl in enumerate(ax.get_xticklabels()):
    if i % 3 == 0:          # US columns → bold
        lbl.set_fontweight("bold")

# ── dataset group headers above the heatmap ───────────────────────────────────
ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(n_rows - 0.5, -0.5)
for ds_i, ds in enumerate(DS_ORDER):
    center   = ds_i * 3 + 1          # column 1, 4, 7
    left_col = ds_i * 3 - 0.45
    right_col= ds_i * 3 + 2.45
    # colored underline bracket
    ax.annotate("", xy=(right_col, -0.5), xytext=(left_col, -0.5),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-", color=DS_COLORS[ds], lw=2),
                annotation_clip=False)
    ax.text(center, -0.9, ds, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=DS_COLORS[ds],
            clip_on=False,
            transform=ax.transData)

# ── vertical separator lines between dataset groups ────────────────────────────
for j in [3, 6]:
    ax.axvline(j - 0.5, color="#333333", linewidth=1.8, zorder=5)

# ── horizontal separator lines between size tier groups ───────────────────────
for i in tier_boundaries:
    ax.axhline(i - 0.5, color="#555555", linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=5)

# ── row y-tick labels ──────────────────────────────────────────────────────────
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=10)
for lbl, tier in zip(ax.get_yticklabels(), row_tiers):
    lbl.set_color(SIZE_COLORS[tier])

# ── size-tier bracket annotations on the left ─────────────────────────────────
BRACKET_X = -2.8    # data-coord left of row labels (clip_on=False)
TEXT_X     = -3.2
for tier, rows in tier_label_rows.items():
    top = min(rows) - 0.38
    bot = max(rows) + 0.38
    mid = np.mean(rows)
    # vertical bar
    ax.plot([BRACKET_X, BRACKET_X], [top, bot],
            color=SIZE_COLORS[tier], linewidth=2.5, clip_on=False, zorder=6)
    # horizontal ticks at ends
    ax.plot([BRACKET_X, BRACKET_X + 0.15], [top, top],
            color=SIZE_COLORS[tier], linewidth=2.0, clip_on=False, zorder=6)
    ax.plot([BRACKET_X, BRACKET_X + 0.15], [bot, bot],
            color=SIZE_COLORS[tier], linewidth=2.0, clip_on=False, zorder=6)
    # tier label text
    short_tier = {"API": "API", "Large (15–34B)": "Large\n(15–34B)",
                  "Small (≤7B)": "Small\n(≤7B)"}[tier]
    ax.text(TEXT_X, mid, short_tier,
            ha="right", va="center", fontsize=9,
            color=SIZE_COLORS[tier], clip_on=False,
            fontweight="bold")

# ── colorbar ───────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.025)
cbar.set_label("Pass@1 drop (pp)\n← gain  |  loss →", fontsize=9, labelpad=8)
cbar.ax.axhline(y=norm(0), color="black", linewidth=0.8, linestyle="--")

# ── title ──────────────────────────────────────────────────────────────────────
ax.set_title("Absolute Pass@1 drop (pp) under each mutation — all models & benchmarks",
             fontsize=12, pad=40, loc="center")

save(fig, "rq1_fig5_heatmap")
plt.close()

print("\nDone. All RQ1 figures saved to figures/")
