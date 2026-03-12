#!/usr/bin/env python3
"""
Publication-quality t-SNE and PCA figures for the three classifiers.
Uses the original seed-42 images (correct cluster structure) but crops
out the embedded titles and replaces them with large, readable labels.

Outputs:
  figures/fig_tsne_classifiers.{pdf,png}
  figures/fig_pca_classifiers.{pdf,png}
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── source images ──────────────────────────────────────────────────────────────
PANELS = [
    ("Linear Probe\n(frozen backbone)",
     "linear_classifier_outputs/seed42/tsne.png",
     "linear_classifier_outputs/seed42/pca.png"),
    ("LoRA Fine-tune",
     "lora_classifier_outputs/seed42/tsne.png",
     "lora_classifier_outputs/seed42/pca.png"),
    ("Full Fine-tune",
     "full_classifier_outputs/seed123456/tsne.png",
     "full_classifier_outputs/seed123456/pca.png"),
]

# title text occupies rows 0–54; plot content starts at row 55
CROP_TOP = 55

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})


def make_figure(img_key, suptitle, outname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    for ax, (panel_title, tsne_path, pca_path) in zip(axes, PANELS):
        path = tsne_path if img_key == "tsne" else pca_path
        img  = np.array(Image.open(path))[CROP_TOP:, :]   # crop title
        ax.imshow(img, aspect="auto")
        ax.set_title(panel_title, fontsize=15, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=15, y=1.01)
    fig.tight_layout(w_pad=1.5)

    fig.savefig(OUT / f"{outname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{outname}.png", bbox_inches="tight", dpi=200)
    print(f"Saved → figures/{outname}.{{pdf,png}}")
    plt.close()


make_figure(
    "tsne",
    "t-SNE of classifier embeddings  —  Qwen2.5-Coder-1.5B  (perplexity=30,  seed=42)",
    "fig_tsne_classifiers",
)

make_figure(
    "pca",
    "PCA of classifier embeddings  —  Qwen2.5-Coder-1.5B  (PC1 vs PC2)",
    "fig_pca_classifiers",
)
