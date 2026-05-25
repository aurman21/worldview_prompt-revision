"""
GPT-Image Non-English Revised Prompts Analysis (Python port)

Analyses the share of GPT-Image revised prompts returned in a
non-English language, by language-country and category.

Inputs (in DATA_DIR):
  openai-revised-apr26-en-fin.csv — GPT-Image revised prompts with
    detected response language (ID, revised_prompt, language, revised_prompt_en)
  ID_List.csv — prompt metadata (ID, prompt, category, type, collected)

Outputs:
  gptimage_nonenglish_by_context.pdf
  gptimage_nonenglish_heatmap.pdf (if category variation is notable)

Requirements:
  pip install pandas matplotlib seaborn
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

dfopen = pd.read_csv(os.path.join(DATA_DIR, "openai-revised-apr26-en-fin.csv"))
idlist = pd.read_csv(os.path.join(DATA_DIR, "ID_List.csv"))

# =============================================================================
# PARSE TYPE → language + country
# =============================================================================

def parse_type(df):
    df = df.copy()
    df["type_norm"] = (
        df["type"]
        .str.replace("English_CleanUp", "English", regex=False)
        .str.replace("Chinese_Simplified", "ChineseS", regex=False)
        .str.replace("Chinese_Traditional", "ChineseT", regex=False)
    )
    split = df["type_norm"].str.split("_", n=1, expand=True)
    df["prompt_language"] = split[0]
    df["country"] = split[1].str.strip().replace("", None) if 1 in split.columns else None
    df["lang_country"] = df.apply(
        lambda r: r["prompt_language"] if pd.isna(r["country"]) or r["country"] == ""
        else f"{r['prompt_language']}_{r['country']}", axis=1
    )
    return df.drop(columns=["type_norm"], errors="ignore")


# =============================================================================
# PREPARE DATA
# =============================================================================

df = dfopen.rename(columns={"language": "response_language"})
df = df.merge(
    idlist[["ID", "prompt", "category", "type", "collected"]],
    on="ID", how="left",
)
df = parse_type(df)
df = df[df["revised_prompt_en"].notna()].copy()
df["is_nonenglish"] = df["response_language"] != "en"


# =============================================================================
# 1. BY LANGUAGE-COUNTRY
# =============================================================================

by_context = (
    df.groupby("lang_country")
    .agg(n=("is_nonenglish", "size"),
         n_nonenglish=("is_nonenglish", "sum"))
    .reset_index()
)
by_context["pct_nonenglish"] = (by_context["n_nonenglish"] / by_context["n"] * 100).round(1)
by_context = by_context.sort_values("pct_nonenglish", ascending=False)

print("=== Non-English response rates by language-country ===")
print(by_context.to_string(index=False))


# =============================================================================
# 2. BY CATEGORY
# =============================================================================

by_category = (
    df.groupby("category")
    .agg(n=("is_nonenglish", "size"),
         n_nonenglish=("is_nonenglish", "sum"))
    .reset_index()
)
by_category["pct_nonenglish"] = (by_category["n_nonenglish"] / by_category["n"] * 100).round(1)


# =============================================================================
# 3. BY LANGUAGE-COUNTRY × CATEGORY
# =============================================================================

by_both = (
    df.groupby(["lang_country", "category"])
    .agg(n=("is_nonenglish", "size"),
         n_nonenglish=("is_nonenglish", "sum"))
    .reset_index()
)
by_both["pct_nonenglish"] = (by_both["n_nonenglish"] / by_both["n"] * 100).round(1)


# =============================================================================
# 4. BAR CHART
# =============================================================================

plot_data = by_context[by_context["pct_nonenglish"] > 0].sort_values("pct_nonenglish")

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(plot_data["lang_country"], plot_data["pct_nonenglish"], color="steelblue")
for i, (_, row) in enumerate(plot_data.iterrows()):
    ax.text(row["pct_nonenglish"] + 0.5, i, f"{row['pct_nonenglish']:.0f}%", va="center", fontsize=8)
ax.set_xlabel("% non-English responses")
ax.set_title("GPT Image: Share of revised prompts returned in non-English",
             fontweight="bold")
ax.set_xlim(0, plot_data["pct_nonenglish"].max() * 1.15)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "gptimage_nonenglish_by_context.pdf"))
plt.close(fig)


# =============================================================================
# 5. HEATMAP (if category variation is notable)
# =============================================================================

if by_category["pct_nonenglish"].std() > 3:
    print("\nCategory variation is notable — generating heatmap.")

    heat_data = by_both[by_both["pct_nonenglish"] > 0].copy()
    pivot = heat_data.pivot_table(
        index="category", columns="lang_country",
        values="pct_nonenglish", aggfunc="first", fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot, ax=ax, cmap="YlGnBu", vmin=0,
        annot=True, fmt=".0f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "% Non-English"},
    )
    ax.set_title("GPT Image: Non-English revised prompts\n"
                 "By language-context × category", fontweight="bold")
    ax.set_xlabel("Language / Context")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "gptimage_nonenglish_heatmap.pdf"))
    plt.close(fig)
else:
    print("\nCategory variation is small — bar chart sufficient.")

print("\nGPT-Image non-English analysis complete.")
