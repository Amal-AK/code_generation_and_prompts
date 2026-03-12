import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats

df = pd.read_csv("results/all_metrics.csv")

# Clean up model names
model_name_map = {
    "Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B",
    "Qwen_Qwen2.5-Coder-7B-Instruct":  "Qwen2.5-Coder-7B",
    "deepseek-ai_deepseek-coder-33b-instruct": "DeepSeek-Coder-33B",
    "deepseek-ai_deepseek-coder-6.7b-instruct": "DeepSeek-Coder-6.7B",
    "codellama_CodeLlama-7b-Instruct-hf":  "CodeLlama-7B",
    "codellama_CodeLlama-34b-Instruct-hf": "CodeLlama-34B",
    "bigcode_starcoder2-15b-instruct-v0.1": "StarCoder2-15B",
    "mistralai_Codestral-22B-v0.1": "Codestral-22B",
    "gpt-4o-mini": "GPT-5-mini",
    "claude-sonnet-4": "Claude Sonnet 4",
}
df["Model"] = df["Model"].map(model_name_map).fillna(df["Model"])

orig = df[df["Mutation"] == "Orig"][["Model", "Dataset", "Pass@1"]].rename(columns={"Pass@1": "Orig_Pass1"})
muts = df[df["Mutation"] != "Orig"][["Model", "Dataset", "Mutation", "Pass@1"]].rename(columns={"Pass@1": "Mut_Pass1"})

merged = muts.merge(orig, on=["Model", "Dataset"])
merged["Drop"] = merged["Orig_Pass1"] - merged["Mut_Pass1"]

# ── colours & markers ────────────────────────────────────────────────
mut_colors  = {"US": "#e63946", "LV": "#2a9d8f", "SF": "#f4a261"}
bench_marks = {"HumanEval": "o", "MBPP": "s", "livecodebench": "^"}

fig, ax = plt.subplots(figsize=(7, 5))

for (mut, bench), grp in merged.groupby(["Mutation", "Dataset"]):
    ax.scatter(
        grp["Orig_Pass1"], grp["Drop"],
        color=mut_colors[mut],
        marker=bench_marks.get(bench, "o"),
        s=60, alpha=0.82, linewidths=0.4, edgecolors="white",
        zorder=3,
    )

# overall regression line across all mutations / benchmarks
x_all = merged["Orig_Pass1"].values
y_all = merged["Drop"].values
slope, intercept, r, p, _ = stats.linregress(x_all, y_all)
x_line = np.linspace(x_all.min(), x_all.max(), 200)
ax.plot(x_line, intercept + slope * x_line, color="black", lw=1.5,
        linestyle="--", zorder=4, label=f"OLS  r={r:.2f}, p={p:.3f}")

ax.axhline(0, color="gray", lw=0.8, linestyle=":")

ax.set_xlabel("Original Pass@1 (%)", fontsize=12)
ax.set_ylabel("Drop in Pass@1 (pp)", fontsize=12)
ax.set_title("Do stronger models suffer larger drops under mutation?", fontsize=12)

# ── legend: mutations (colour) ────────────────────────────────────────
mut_handles = [
    mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                  markersize=7, label=m)
    for m, c in mut_colors.items()
]
# legend: benchmarks (shape)
bench_handles = [
    mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                  markersize=7, label=b.replace("livecodebench", "LCB"))
    for b, mk in bench_marks.items()
]
reg_handle = mlines.Line2D([], [], color="black", linestyle="--",
                            linewidth=1.5, label=f"OLS  r={r:.2f}, p={p:.3f}")

leg1 = ax.legend(handles=mut_handles,    title="Mutation", loc="upper left",  fontsize=9)
leg2 = ax.legend(handles=bench_handles,  title="Benchmark", loc="upper center", fontsize=9)
leg3 = ax.legend(handles=[reg_handle],   loc="lower right", fontsize=9)
ax.add_artist(leg1)
ax.add_artist(leg2)

plt.tight_layout()
plt.savefig("figures/corr_orig_vs_drop.pdf", bbox_inches="tight")
plt.savefig("figures/corr_orig_vs_drop.png", dpi=150, bbox_inches="tight")
print(f"Saved. OLS: slope={slope:.3f}, r={r:.2f}, p={p:.4f}")
