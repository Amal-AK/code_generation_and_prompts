#!/usr/bin/env python3
"""
Bar chart: judge general-question scores per benchmark and mutation type.
General questions:
  - Recoverability   : can an expert recover the original intent?
  - Naturalness      : does the mutated prompt read naturally?
  - Sem. Preservation: is the core task semantically preserved?

Output: figures/fig_judge_general.{pdf,png}
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── data (from judge_by_dataset_mut.csv) ──────────────────────────────────────
# {benchmark: {mutation: {metric: value}}}
data = {
    "HumanEval": {
        "US": {"Recoverability": 0.888, "Naturalness": 0.806, "Semantic Preservation": 0.662},
        "LV": {"Recoverability": 0.994, "Naturalness": 0.988, "Semantic Preservation": 0.957},
        "SF": {"Recoverability": 0.512, "Naturalness": 0.994, "Semantic Preservation": 0.530},
    },
    "MBPP": {
        "US": {"Recoverability": 0.704, "Naturalness": 0.821, "Semantic Preservation": 0.378},
        "LV": {"Recoverability": 0.979, "Naturalness": 0.989, "Semantic Preservation": 0.917},
        "SF": {"Recoverability": 0.958, "Naturalness": 0.997, "Semantic Preservation": 0.909},
    },
    "LCB": {
        "US": {"Recoverability": 0.869, "Naturalness": 0.505, "Semantic Preservation": 0.600},
        "LV": {"Recoverability": 0.980, "Naturalness": 0.961, "Semantic Preservation": 0.949},
        "SF": {"Recoverability": 0.794, "Naturalness": 0.992, "Semantic Preservation": 0.740},
    },
}

metrics    = ["Recoverability", "Naturalness", "Semantic Preservation"]
mutations  = ["US", "LV", "SF"]
benchmarks = ["HumanEval", "MBPP", "LCB"]

MUT_COLORS = {
    "US": "#C47A3A",   # muted terracotta
    "LV": "#5B8DB8",   # muted steel blue
    "SF": "#5BA07A",   # muted sage green
}
MUT_HATCHES = {"US": "//", "LV": "", "SF": ".."}

# ── layout: one subplot per benchmark ─────────────────────────────────────────
bar_w     = 0.14
group_gap = 0.28
x = np.arange(len(metrics)) * (len(mutations) * bar_w + group_gap)
metric_labels = ["Recoverability", "Naturalness", "Semantic\nPreservation"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

for ax, bench in zip(axes, benchmarks):
    for i, mut in enumerate(mutations):
        vals   = [data[bench][mut][m] for m in metrics]
        offset = (i - (len(mutations) - 1) / 2) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w,
                        color=MUT_COLORS[mut], edgecolor="#444444", linewidth=0.7,
                        hatch=MUT_HATCHES[mut], label=mut, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v*100:.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#111111")

    ax.set_title(bench, fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="y", labelsize=11)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.5, color="#dddddd", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Score", fontsize=13)

# shared legend
handles = [plt.Rectangle((0, 0), 1, 1,
                          facecolor=MUT_COLORS[m], edgecolor="#444444",
                          hatch=MUT_HATCHES[m], linewidth=0.7)
           for m in mutations]
fig.legend(handles, mutations, title="Mutation", fontsize=11, title_fontsize=11,
           loc="lower right", bbox_to_anchor=(1.0, 0.08), framealpha=0.85)

fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_judge_general.{ext}",
                dpi=300, bbox_inches="tight")
    print(f"Saved figures/fig_judge_general.{ext}")
