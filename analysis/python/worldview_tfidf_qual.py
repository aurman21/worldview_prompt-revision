"""
Step 3: Qualitative TF-IDF Exploration (Python port)

Generates LaTeX tables and category annotations from TF-IDF terms
computed by worldview_full_pipeline.py.

Inputs (in OUTPUT_DIR, produced by the main pipeline):
  tfidf_top20_terms.csv
  cms_cfs_scores.csv
  pipeline_decomposition_detail.csv (optional)

Outputs:
  tfidf_main_table.tex, tfidf_appendix_table.tex,
  tfidf_main_compact.csv, tfidf_appendix_full.csv,
  tfidf_appendix_condensed.csv, tfidf_category_counts.csv

Requirements:
  pip install pandas
"""

import os
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "output"

# =============================================================================
# 1. LOAD DATA
# =============================================================================

tfidf   = pd.read_csv(os.path.join(OUTPUT_DIR, "tfidf_top20_terms.csv"))
results = pd.read_csv(os.path.join(OUTPUT_DIR, "cms_cfs_scores.csv"))

# =============================================================================
# 2. SELECT CONTEXTS
# =============================================================================

context_ranks = (
    results.groupby("context")["cfs"].mean()
    .sort_values(ascending=False)
    .reset_index()
)

top_contexts    = context_ranks.head(3)["context"].tolist()
bottom_contexts = context_ranks.tail(3)["context"].tolist()
selected = top_contexts + bottom_contexts

print("Selected contexts for main text:")
print(f"  High CFS: {', '.join(top_contexts)}")
print(f"  Low CFS:  {', '.join(bottom_contexts)}")

# =============================================================================
# 3. MAIN TEXT TABLE (top-5 per selected context)
# =============================================================================

main_table = (
    tfidf[tfidf["context"].isin(selected)]
    .sort_values("tfidf_score", ascending=False)
    .groupby(["context", "model"])
    .head(5)
)
main_table["rank"] = main_table.groupby(["context", "model"]).cumcount() + 1

main_compact = (
    main_table[["context", "model", "rank", "term"]]
    .pivot_table(index=["context", "rank"], columns="model",
                 values="term", aggfunc="first")
    .reset_index()
    .sort_values(["context", "rank"])
)
main_compact.to_csv(os.path.join(OUTPUT_DIR, "tfidf_main_compact.csv"), index=False)

print("\n=== Main Text Table (compact) ===")
for ctx in selected:
    sub = main_compact[main_compact["context"] == ctx]
    print(f"\n{ctx}:")
    for _, row in sub.iterrows():
        vals = [str(row.get(m, "--") or "--")
                for m in ["dalle", "gptimage", "imagen"]]
        print(f"  {int(row['rank'])}. {' | '.join(vals)}")


# =============================================================================
# 4. LATEX — MAIN TEXT TABLE
# =============================================================================

def escape_latex(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Context & \# & DALL\textperiodcentered E~3 & GPT-Image & Imagen \\",
    r"\midrule",
]

for ctx in selected:
    rows = main_compact[main_compact["context"] == ctx]
    for i, (_, row) in enumerate(rows.iterrows()):
        prefix = escape_latex(ctx) if i == 0 else ""
        d = row.get("dalle", "--") or "--"
        g = row.get("gptimage", "--") or "--"
        im = row.get("imagen", "--") or "--"
        lines.append(f"{prefix} & {int(row['rank'])} & {d} & {g} & {im} \\\\")
    if ctx != selected[-1]:
        lines.append(r"\midrule")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Top-5 TF-IDF distinctive terms for the three highest-"
    r" and three lowest-CFS contexts.}",
    r"\label{tab:tfidf_main}",
    r"\end{table*}",
]

with open(os.path.join(OUTPUT_DIR, "tfidf_main_table.tex"), "w") as f:
    f.write("\n".join(lines))
print("\nLaTeX table saved to tfidf_main_table.tex")


# =============================================================================
# 5. APPENDIX: FULL TOP-10
# =============================================================================

appendix_table = (
    tfidf
    .sort_values("tfidf_score", ascending=False)
    .groupby(["context", "model"])
    .head(10)
)
appendix_table["rank"] = appendix_table.groupby(["context", "model"]).cumcount() + 1
appendix_table = appendix_table.sort_values(["context", "model", "rank"])
appendix_table.to_csv(os.path.join(OUTPUT_DIR, "tfidf_appendix_full.csv"), index=False)

appendix_condensed = (
    appendix_table.groupby(["context", "model"])["term"]
    .apply(lambda x: ", ".join(x))
    .reset_index()
    .pivot_table(index="context", columns="model", values="term", aggfunc="first")
    .reset_index()
)
appendix_condensed.to_csv(os.path.join(OUTPUT_DIR, "tfidf_appendix_condensed.csv"), index=False)

# LaTeX for appendix
app_lines = [
    r"\begin{table*}[h]",
    r"\centering",
    r"\scriptsize",
    r"\begin{tabular}{lp{4cm}p{4cm}p{4cm}}",
    r"\toprule",
    r"Context & DALL\textperiodcentered E~3 & GPT-Image & Imagen \\",
    r"\midrule",
]

for ctx in sorted(appendix_condensed["context"].unique()):
    row = appendix_condensed[appendix_condensed["context"] == ctx].iloc[0]
    d  = row.get("dalle", "--") or "--"
    g  = row.get("gptimage", "--") or "--"
    im = row.get("imagen", "--") or "--"
    app_lines.append(f"{escape_latex(ctx)} & {d} & {g} & {im} \\\\")

app_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Top-10 TF-IDF distinctive terms for all contexts across"
    r" three models.}",
    r"\label{tab:tfidf_appendix}",
    r"\end{table*}",
]

with open(os.path.join(OUTPUT_DIR, "tfidf_appendix_table.tex"), "w") as f:
    f.write("\n".join(app_lines))
print("Appendix LaTeX table saved to tfidf_appendix_table.tex")


# =============================================================================
# 6. CATEGORY ANALYSIS
# =============================================================================

CATEGORIES = {
    "demographic": {"woman", "women", "man", "men", "girl", "boy",
                    "child", "children", "elder", "elderly", "young",
                    "diverse", "multicultural", "caucasian", "hispanic",
                    "asian", "african", "black", "white", "skin"},
    "temporal":    {"ancient", "traditional", "historic", "historical",
                    "colonial", "old", "heritage", "modern", "contemporary",
                    "urban", "futuristic"},
    "exotic":      {"vibrant", "colorful", "colourful", "exotic", "mystical",
                    "magical", "bustling", "lively", "rustic", "quaint",
                    "serene", "tranquil", "picturesque"},
    "nature":      {"snow", "snowy", "winter", "ice", "frozen", "freeze",
                    "pine", "forest", "lake", "mountain", "desert", "sand",
                    "tropical", "palm", "jungle", "ocean", "river"},
    "religion":    {"mosque", "minaret", "church", "temple", "pagoda",
                    "cathedral", "buddhist", "islamic", "christian",
                    "hindu", "prayer", "religious", "spiritual", "hijab"},
    "architecture": {"pyramid", "sphinx", "tower", "castle", "palace",
                     "cathedral", "dome", "minaret", "pagoda", "shrine",
                     "cobblestone", "medieval", "baroque", "gothic"},
}

cat_rows = []
for (ctx, mdl), grp in appendix_table.groupby(["context", "model"]):
    row = {"context": ctx, "model": mdl}
    for cat_name, cat_terms in CATEGORIES.items():
        row[f"n_{cat_name}"] = int(grp["term"].isin(cat_terms).sum())
    cat_rows.append(row)

category_counts = pd.DataFrame(cat_rows)
category_counts.to_csv(os.path.join(OUTPUT_DIR, "tfidf_category_counts.csv"), index=False)


# =============================================================================
# 7. PIPELINE DECOMPOSITION TABLE (if available)
# =============================================================================

decomp_path = os.path.join(OUTPUT_DIR, "pipeline_decomposition_detail.csv")

if os.path.exists(decomp_path):
    print("\n=== Pipeline Decomposition (LaTeX) ===")
    decomp = pd.read_csv(decomp_path)
    decomp_sel = decomp[decomp["context"].isin(top_contexts)]

    dl = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lp{3.5cm}p{3.5cm}p{3.5cm}}",
        r"\toprule",
        r"Context & Propagated & Text-only & Visual-only \\",
        r"\midrule",
    ]
    for ctx in top_contexts:
        row = decomp_sel[decomp_sel["context"] == ctx]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        dl.append(f"{escape_latex(ctx)} & {r['propagated']} & "
                  f"{r['text_only']} & {r['visual_only']} \\\\")
    dl += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Pipeline decomposition for the three highest-CFS contexts.}",
        r"\label{tab:pipeline_decomp}",
        r"\end{table*}",
    ]
    with open(os.path.join(OUTPUT_DIR, "pipeline_decomposition_table.tex"), "w") as f:
        f.write("\n".join(dl))
    print("Pipeline decomposition LaTeX saved.")
else:
    print("\nNo pipeline decomposition found (run visual analysis first).")

print(f"\nAll outputs in {OUTPUT_DIR}/")
