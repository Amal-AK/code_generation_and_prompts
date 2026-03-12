import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.read_csv("results/all_metrics.csv")
orig = df[df["Mutation"]=="Orig"][["Model","Dataset","Pass@1"]].rename(columns={"Pass@1":"Orig"})
muts = df[df["Mutation"]!="Orig"][["Model","Dataset","Mutation","Pass@1"]].rename(columns={"Pass@1":"Mut"})
m = muts.merge(orig, on=["Model","Dataset"])
m["Drop"] = m["Orig"] - m["Mut"]

model_name_map = {
    "Qwen_Qwen2.5-Coder-32B-Instruct":          "Qwen2.5-32B",
    "Qwen_Qwen2.5-Coder-7B-Instruct":           "Qwen2.5-7B",
    "deepseek-ai_deepseek-coder-33b-instruct":   "DS-Coder-33B",
    "deepseek-ai_deepseek-coder-6.7b-instruct":  "DS-Coder-6.7B",
    "codellama_CodeLlama-7b-Instruct-hf":        "CodeLlama-7B",
    "codellama_CodeLlama-34b-Instruct-hf":       "CodeLlama-34B",
    "bigcode_starcoder2-15b-instruct-v0.1":      "SC2-15B",
    "mistralai_Codestral-22B-v0.1":              "Codestral",
    "gpt-5-mini":                                "GPT-5-mini",
    "claude-sonnet-4-20250514":                  "Claude-S4",
}
m["Model"] = m["Model"].map(model_name_map).fillna(m["Model"])

from scipy import stats

mutations = ["US", "LV", "SF"]
mut_colors = {"US": "#e63946", "LV": "#2a9d8f", "SF": "#f4a261"}

# average drop and orig across benchmarks per model
avg_orig = m.groupby("Model")["Orig"].mean()
avg_drop = m.groupby(["Model", "Mutation"])["Drop"].mean().reset_index()

results = {}
for mut in mutations:
    sub = avg_drop[avg_drop["Mutation"] == mut].set_index("Model")
    combined = pd.DataFrame({"orig": avg_orig, "drop": sub["Drop"]}).dropna()
    r, p = stats.pearsonr(combined["orig"], combined["drop"])
    results[mut] = (r, p, combined)

# r matrix: rows = benchmarks + avg-across-benchmarks, cols = mutations
r_vals = np.array([[results[mut][0] for mut in mutations]])
p_vals = np.array([[results[mut][1] for mut in mutations]])

fig, ax = plt.subplots(figsize=(4.5, 1.6))
im = ax.imshow(r_vals, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)

ax.set_xticks(range(len(mutations)))
ax.set_xticklabels(mutations, fontsize=12, fontweight="bold")
ax.set_yticks([0])
ax.set_yticklabels(["Avg across\nbenchmarks"], fontsize=10)

for j in range(len(mutations)):
    r, p = r_vals[0, j], p_vals[0, j]
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "\n(n.s.)"
    ax.text(j, 0, f"{r:.2f}{stars}", ha="center", va="center",
            fontsize=11, fontweight="bold" if "\n" not in stars else "normal")

cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Pearson r", fontsize=9)

ax.set_title("Pearson r: model strength (orig Pass@1)\nvs drop under mutation  (* p<.05  ** p<.01  *** p<.001)",
             fontsize=9.5)
plt.tight_layout()
plt.savefig("figures/corr_by_mutation.pdf", bbox_inches="tight")
plt.savefig("figures/corr_by_mutation.png", dpi=150, bbox_inches="tight")
print("Saved.")
