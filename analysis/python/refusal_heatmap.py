"""
Guardrailing Asymmetry Analysis (Python port)

Computes and visualises image generation refusal rates by model,
prompt category, and language-context combination.

Input (in DATA_DIR):
  df_unified_ALL.csv — columns: model_type, category, lang_country

Output:
  refusal_heatmap.pdf, refusal_heatmap_filtered.pdf

Requirements:
  pip install pandas matplotlib seaborn
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expected image counts per (model, category, lang_country) cell
EXPECTED = {"imagen": 20, "gptimage": 20, "dalle": 100}

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(os.path.join(DATA_DIR, "df_unified_ALL.csv"))

# =============================================================================
# COMPUTE REFUSAL RATES
# =============================================================================

counts = (
    df.groupby(["model_type", "category", "lang_country"])
    .size()
    .reset_index(name="n")
)
counts["expected_n"]   = counts["model_type"].map(EXPECTED)
counts["n_refusals"]   = counts["expected_n"] - counts["n"]
counts["pct_refusals"] = (counts["n_refusals"] / counts["expected_n"]) * 100

# Sanity
over = counts[counts["n"] > counts["expected_n"]]
if len(over) > 0:
    print("WARNING: some combinations exceed expected count:")
    print(over)

# Summary
summary = (
    counts.groupby(["model_type", "lang_country"])
    .agg(
        mean_pct_refusal=("pct_refusals", "mean"),
        total_generated=("n", "sum"),
        total_expected=("expected_n", "sum"),
        total_refusals=("n_refusals", "sum"),
    )
    .reset_index()
)
summary["overall_pct_refusal"] = (summary["total_refusals"] / summary["total_expected"]) * 100
print("Refusal summary:")
print(summary.to_string(index=False))


# =============================================================================
# HEATMAP HELPER
# =============================================================================

def refusal_heatmap(data, title, subtitle, filename, figsize=(16, 12)):
    """Faceted heatmap: category × lang_country, one panel per model."""
    models = sorted(data["model_type"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, squeeze=False)

    for ax, mdl in zip(axes.ravel(), models):
        sub = data[data["model_type"] == mdl].copy()
        pivot = sub.pivot_table(
            index="category", columns="lang_country",
            values="pct_refusals", aggfunc="first", fill_value=0,
        )
        sns.heatmap(
            pivot, ax=ax, cmap="Reds", vmin=0,
            annot=True, fmt=".0f", annot_kws={"size": 7},
            linewidths=0.3, linecolor="white", cbar_kws={"label": "% Refused"},
        )
        ax.set_title(mdl, fontweight="bold", fontsize=12)
        ax.set_xlabel("Language / Context")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    if subtitle:
        fig.text(0.5, 0.995, subtitle, ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FULL HEATMAP
# =============================================================================

refusal_heatmap(
    counts,
    "Image Generation Refusal Rates",
    "By model, category, and language-context",
    "refusal_heatmap.pdf",
)

# =============================================================================
# FILTERED HEATMAP (non-zero, excluding Imagen)
# =============================================================================

cats_with_refusals = counts[counts["pct_refusals"] > 0]["category"].unique()
filtered = counts[
    (counts["model_type"] != "imagen") &
    (counts["category"].isin(cats_with_refusals))
]

refusal_heatmap(
    filtered,
    "Image Generation Refusal Rates",
    "Categories with ≥1 refusal | DALL·E 3 & GPT-Image (Imagen: 0%)",
    "refusal_heatmap_filtered.pdf",
    figsize=(16, 8),
)

print("\nRefusal analysis complete.")
