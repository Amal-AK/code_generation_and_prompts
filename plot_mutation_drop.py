import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("results/small_models/all_results_summary.csv")

# Classify dataset and variant
def classify(row):
    d = row["Dataset"]
    if "humanEval/HumanEval" in d or "HumanEval.jsonl" in d:
        return "HumanEval", "Original"
    elif "mbpp/mbpp" in d:
        return "MBPP", "Original"
    elif "humanEval_lv" in d or "humanEval_LV" in d:
        return "HumanEval", "LV"
    elif "humanEval_SF" in d:
        return "HumanEval", "SF"
    elif "HumanEval_US" in d:
        return "HumanEval", "US"
    elif "mbpp_LV" in d:
        return "MBPP", "LV"
    elif "mbpp_SF" in d:
        return "MBPP", "SF"
    elif "mbpp_US" in d:
        return "MBPP", "US"
    return "Unknown", "Unknown"

df[["BenchDataset", "Variant"]] = df.apply(classify, axis=1, result_type="expand")

model_map = {
    "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama-7B",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek-6.7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-7B",
}
df["ModelShort"] = df["Model"].map(model_map)

mut_variants = ["LV", "SF", "US"]
datasets = ["HumanEval", "MBPP"]
models = ["CodeLlama-7B", "DeepSeek-6.7B", "Qwen2.5-7B"]
colors = {"CodeLlama-7B": "#4C72B0", "DeepSeek-6.7B": "#DD8452", "Qwen2.5-7B": "#55A868"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig.suptitle("Pass@1 Drop vs Original — Small Models", fontsize=14, fontweight="bold", y=1.01)

for ax, dataset in zip(axes, datasets):
    sub = df[df["BenchDataset"] == dataset]
    pivot = sub.pivot_table(index="Variant", columns="ModelShort", values="Pass@1_Rate")

    # compute drop relative to original
    orig = pivot.loc["Original"]
    drop = pivot.loc[mut_variants].subtract(orig)  # negative = drop

    x = np.arange(len(mut_variants))
    width = 0.22
    offsets = [-width, 0, width]

    for i, model in enumerate(models):
        vals = drop[model].values * 100
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=colors[model], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            va = "top" if v < 0 else "bottom"
            offset = -0.3 if v < 0 else 0.3
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:+.1f}%", ha="center", va=va, fontsize=8, color="#222")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mut_variants, fontsize=11)
    ax.set_ylabel("Pass@1 Change vs Original (pp)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles = [mpatches.Patch(color=colors[m], label=m) for m in models]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig("results/small_models/mutation_drop.png", dpi=150, bbox_inches="tight")
print("Saved to results/small_models/mutation_drop.png")
