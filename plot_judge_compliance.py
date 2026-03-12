#!/usr/bin/env python3
"""
Bar chart: mutation rule compliance per benchmark and mutation type.
Compliance metric used:
  LV  -> lexical_compliance
  SF  -> formatting_compliance
  US  -> underspec_compliance

Output: figures/fig_judge_compliance.{pdf,png}
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
# rows: HumanEval, MBPP, LCB
# cols: US, LV, SF  (order chosen for visual grouping)
data = {
    "HumanEval": {"US": 0.894, "LV": 1.000, "SF": 1.000},
    "MBPP":      {"US": 0.882, "LV": 1.000, "SF": 1.000},
    "LCB":       {"US": 0.853, "LV": 0.999, "SF": 0.997},
}

mutations  = ["US", "LV", "SF"]
benchmarks = ["HumanEval", "MBPP", "LCB"]

# muted colorblind-safe palette
MUT_COLORS = {
    "US": "#C47A3A",   # muted terracotta
    "LV": "#5B8DB8",   # muted steel blue
    "SF": "#5BA07A",   # muted sage green
}

# ── layout ────────────────────────────────────────────────────────────────────
n_benchmarks = len(benchmarks)
n_mutations  = len(mutations)
bar_w  = 0.18
group_gap = 0.22
x = np.arange(n_benchmarks) * (n_mutations * bar_w + group_gap)

fig, ax = plt.subplots(figsize=(7, 4))

for i, mut in enumerate(mutations):
    vals   = [data[b][mut] for b in benchmarks]
    offset = (i - (n_mutations - 1) / 2) * bar_w
    bars   = ax.bar(x + offset, vals, bar_w,
                    color=MUT_COLORS[mut], edgecolor="#444444", linewidth=0.7,
                    label=mut, zorder=3)
    # value labels on top — skip near-100 to avoid overlap
    for bar, v in zip(bars, vals):
        if v < 0.999:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{v*100:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#111111")

# ── axes styling ──────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12, fontweight="bold")
ax.set_ylabel("Compliance rate", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylim(0.0, 1.06)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=2)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linewidth=0.5, color="#dddddd", zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(title="Mutation", fontsize=9, title_fontsize=9,
          loc="lower right", framealpha=0.85)

fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_judge_compliance.{ext}",
                dpi=300, bbox_inches="tight")
    print(f"Saved figures/fig_judge_compliance.{ext}")
