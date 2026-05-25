"""
ABLATION: Isolating the Revision Layer Effect (Python port)

Compares VQA descriptions of images generated from ORIGINAL prompts vs
REVISED prompts using open-source image models (SDXL, Flux 2 Dev).

Inputs (in DATA_DIR):
  vqa_descriptions_ablation.csv, prompts_sdrun.csv, geo_stopwords.txt

Analyses per image model:
  0. VQA description length diagnostics
  1. Paired CMS on VQA descriptions (TF-IDF cosine distance from baseline)
  2. TF-IDF distinctive terms per context x prompt_type
  3. CFS on VQA descriptions
  4. Term-level 2x2 decomposition
  5. Per-term McNemar tests

Requirements:
  pip install pandas numpy scikit-learn scipy matplotlib nltk
"""

import os
import re
import warnings
from itertools import product as iter_product

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "output/ablation"

VQA_CSV        = os.path.join(DATA_DIR, "vqa_descriptions_ablation.csv")
PROMPTS_CSV    = os.path.join(DATA_DIR, "prompts_sdrun.csv")
STOPWORDS_PATH = os.path.join(DATA_DIR, "geo_stopwords.txt")

TAU_QUANTILE = 0.75
K_TOP        = 10
K_TFIDF_SHOW = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

_lem = WordNetLemmatizer()
_en_stops = set(nltk_stopwords.words("english"))


def load_geo_stopwords(path):
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for m in re.findall(r'"([^"]+)"', line):
                terms.add(m.strip().lower())
    return terms


def lemmatize_clean(text):
    """Lowercase, remove non-alpha, lemmatize, remove short words."""
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return [_lem.lemmatize(w) for w in text.split() if len(w) > 2]


def tokenize_remove_stops(text, geo_stops):
    """Full pipeline: lemmatize then remove English + geo stops."""
    tokens = lemmatize_clean(text)
    return [t for t in tokens if t not in _en_stops and t not in geo_stops]


geo_stops = load_geo_stopwords(STOPWORDS_PATH)
print(f"Loaded {len(geo_stops)} geographic/artifact stopwords")

# =============================================================================
# LOAD & JOIN DATA
# =============================================================================

print(f"Loading VQA data from: {VQA_CSV}")
vqa_raw = pd.read_csv(VQA_CSV)
print(f"  VQA rows: {len(vqa_raw)}")
print(f"  Models: {', '.join(vqa_raw['model'].unique())}")
print(f"  Prompt types: {', '.join(vqa_raw['prompt_type'].unique())}")

print(f"\nLoading prompts from: {PROMPTS_CSV}")
prompts = pd.read_csv(PROMPTS_CSV)

# Build metadata
keep_cols = ["prompt_id", "context", "type", "is_baseline", "base_prompt_id"]
for c in ["original_clean", "revised_clean"]:
    if c in prompts.columns:
        keep_cols.append(c)
prompts_meta = prompts[keep_cols].drop_duplicates()
prompts_meta["context"] = prompts_meta.apply(
    lambda r: "baseline" if (pd.isna(r["context"]) and r.get("is_baseline", False))
    else r["context"], axis=1
)

vqa = vqa_raw.merge(
    prompts_meta[["prompt_id", "context", "is_baseline", "base_prompt_id"]],
    on="prompt_id", how="left",
)

n_missing = vqa["context"].isna().sum()
if n_missing > 0:
    print(f"WARNING: {n_missing} VQA rows without context")
else:
    print("All VQA rows joined to context.")

# Detect baseline
bl_vals = vqa.loc[vqa["is_baseline"] == True, "context"].unique()
if len(bl_vals) == 0:
    bl_vals = vqa.loc[vqa["context"].str.contains("baseline", case=False, na=False), "context"].unique()
BASELINE_CONTEXT = bl_vals[0] if len(bl_vals) > 0 else "baseline"
print(f"Baseline context: {BASELINE_CONTEXT}")


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================

def run_ablation_analysis(vqa_model, model_name, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"ANALYSING MODEL: {model_name}")
    print(f"{'='*60}")

    # --- 0. Description length ---
    vqa_model = vqa_model.copy()
    vqa_model["desc_nwords"] = vqa_model["description"].str.split().str.len()

    length_stats = (
        vqa_model.groupby(["context", "prompt_type"])["desc_nwords"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    length_stats.columns = ["context", "prompt_type", "mean_words",
                            "sd_words", "median_words", "n"]
    length_stats.to_csv(os.path.join(output_subdir, "desc_length_stats.csv"), index=False)

    # --- Tokenize ---
    vqa_model["tokens"] = vqa_model["description"].apply(
        lambda d: tokenize_remove_stops(str(d), geo_stops)
    )

    # --- 1. Paired CMS ---
    print("\n  Section 1: Paired CMS")

    # Build per-document texts
    doc_data = vqa_model.copy()
    doc_data["doc_text"] = doc_data["tokens"].apply(lambda t: " ".join(t))
    doc_data["doc_id"] = (
        doc_data["prompt_id"].astype(str) + "__" +
        doc_data["context"].astype(str) + "__" +
        doc_data["prompt_type"].astype(str)
    )

    # TF-IDF DTM
    texts = doc_data["doc_text"].tolist()
    doc_ids = doc_data["doc_id"].tolist()
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
    try:
        dtm = vec.fit_transform(texts)
    except ValueError:
        print("    Not enough terms for TF-IDF; skipping CMS.")
        return

    # Map doc_id → index
    id2idx = {d: i for i, d in enumerate(doc_ids)}

    baseline_docs = doc_data[doc_data["context"] == BASELINE_CONTEXT]
    context_docs  = doc_data[doc_data["context"] != BASELINE_CONTEXT]

    # Join via base_prompt_id
    bp_map = vqa_model[["prompt_id", "base_prompt_id"]].drop_duplicates()

    cms_rows = []
    for _, cr in context_docs.iterrows():
        bp = bp_map[bp_map["prompt_id"] == cr["prompt_id"]]
        if len(bp) == 0:
            continue
        bpid = bp.iloc[0]["base_prompt_id"]
        bl = baseline_docs[
            (baseline_docs["prompt_id"].isin(
                bp_map[bp_map["base_prompt_id"] == bpid]["prompt_id"]
            )) &
            (baseline_docs["prompt_type"] == cr["prompt_type"])
        ]
        if len(bl) == 0:
            continue
        bl_doc = bl.iloc[0]["doc_id"]
        if cr["doc_id"] not in id2idx or bl_doc not in id2idx:
            continue
        v1 = dtm[id2idx[cr["doc_id"]]]
        v2 = dtm[id2idx[bl_doc]]
        sim = cosine_similarity(v1, v2)[0, 0]
        cms_rows.append({
            "prompt_id": cr["prompt_id"],
            "context": cr["context"],
            "prompt_type": cr["prompt_type"],
            "cms": 1.0 - sim,
        })

    cms_df = pd.DataFrame(cms_rows)
    if len(cms_df) == 0:
        print("    No CMS pairs found.")
    else:
        cms_summary = (
            cms_df.groupby(["context", "prompt_type"])["cms"]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
        )
        cms_summary.columns = ["context", "prompt_type", "mean_cms",
                                "sd_cms", "median_cms", "n"]
        cms_summary.to_csv(os.path.join(output_subdir, "cms_vqa_summary.csv"), index=False)

        # Paired Wilcoxon
        cms_wide = cms_df.pivot_table(
            index=["prompt_id", "context"], columns="prompt_type",
            values="cms", aggfunc="first",
        ).dropna(subset=["original", "revised"]).reset_index()

        paired_rows = []
        for ctx, grp in cms_wide.groupby("context"):
            if len(grp) < 5:
                continue
            try:
                stat, p = wilcoxon(grp["revised"], grp["original"],
                                   alternative="greater")
            except Exception:
                stat, p = np.nan, np.nan
            paired_rows.append({
                "context": ctx, "n_pairs": len(grp),
                "median_original": grp["original"].median(),
                "median_revised": grp["revised"].median(),
                "median_diff": (grp["revised"] - grp["original"]).median(),
                "wilcox_V": stat, "wilcox_p": p,
            })

        if paired_rows:
            paired_tests = pd.DataFrame(paired_rows)
            from statsmodels.stats.multitest import multipletests
            _, pvals_adj, _, _ = multipletests(
                paired_tests["wilcox_p"].fillna(1), method="holm"
            )
            paired_tests["p_adj"] = pvals_adj
            paired_tests["sig"] = paired_tests["p_adj"].apply(
                lambda p: "***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "ns"
            )
            paired_tests.to_csv(
                os.path.join(output_subdir, "cms_paired_tests.csv"), index=False
            )
            print(f"    Paired tests saved ({len(paired_tests)} contexts)")

    # --- 2. TF-IDF distinctive terms ---
    print("\n  Section 2: TF-IDF Terms")

    ctx_tokens = vqa_model[vqa_model["context"] != BASELINE_CONTEXT].copy()
    ctx_tokens["condition"] = ctx_tokens["context"] + "::" + ctx_tokens["prompt_type"]

    # Count words per condition
    word_counts = []
    for cond, grp in ctx_tokens.groupby("condition"):
        all_toks = [t for toks in grp["tokens"] for t in toks]
        for w in set(all_toks):
            word_counts.append({"condition": cond, "word": w, "n": all_toks.count(w)})
    wc_df = pd.DataFrame(word_counts)

    # TF-IDF via sklearn on condition documents
    cond_docs = ctx_tokens.groupby("condition")["tokens"].apply(
        lambda x: " ".join(t for toks in x for t in toks)
    ).reset_index()
    cond_docs.columns = ["condition", "doc"]

    cond_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
    cond_mat = cond_vec.fit_transform(cond_docs["doc"])
    cond_terms = cond_vec.get_feature_names_out()

    tfidf_rows = []
    for i, cond in enumerate(cond_docs["condition"]):
        scores = cond_mat[i].toarray().ravel()
        top_idx = scores.argsort()[::-1][:K_TFIDF_SHOW]
        for j in top_idx:
            if scores[j] > 0:
                tfidf_rows.append({
                    "condition": cond, "word": cond_terms[j], "tf_idf": scores[j],
                })

    top_terms = pd.DataFrame(tfidf_rows)
    top_terms[["context", "prompt_type"]] = top_terms["condition"].str.split("::", n=1, expand=True)
    top_terms.to_csv(os.path.join(output_subdir, "tfidf_terms_by_condition.csv"), index=False)

    # Term comparison
    rev_terms = set(
        top_terms[top_terms["prompt_type"] == "revised"]
        .apply(lambda r: (r["context"], r["word"]), axis=1)
    )
    orig_terms = set(
        top_terms[top_terms["prompt_type"] == "original"]
        .apply(lambda r: (r["context"], r["word"]), axis=1)
    )

    all_term_pairs = rev_terms | orig_terms
    comparison_rows = []
    for ctx, word in all_term_pairs:
        comparison_rows.append({
            "context": ctx, "word": word,
            "in_revised": (ctx, word) in rev_terms,
            "in_original": (ctx, word) in orig_terms,
        })
    term_comparison = pd.DataFrame(comparison_rows)

    revised_only  = term_comparison[term_comparison["in_revised"] & ~term_comparison["in_original"]]
    original_only = term_comparison[~term_comparison["in_revised"] & term_comparison["in_original"]]
    revised_only.to_csv(os.path.join(output_subdir, "terms_revised_only.csv"), index=False)
    original_only.to_csv(os.path.join(output_subdir, "terms_original_only.csv"), index=False)

    # --- 3. CFS ---
    print("\n  Section 3: CFS")

    def compute_ablation_cfs(tokens_df, tau_q=TAU_QUANTILE, k=K_TOP):
        """CFS computed from tokenised VQA descriptions."""
        cfs_rows = []
        for ctx, grp in tokens_df.groupby("context"):
            all_toks = [t for toks in grp["tokens"] for t in toks]
            if len(all_toks) == 0:
                continue
            # TF-IDF within this context
            doc = " ".join(all_toks)
            # Simple term frequency as proxy
            from collections import Counter
            counts = Counter(all_toks)
            if len(counts) == 0:
                continue
            scores = pd.Series(counts)
            tau = scores.quantile(tau_q)
            top_k = scores.nlargest(k).index.tolist()
            prevalence = (scores > tau).mean()
            # Spread: fraction of prompts with any top-k term
            n_total = len(grp)
            n_hit = grp["tokens"].apply(lambda t: any(w in top_k for w in t)).sum()
            spread = n_hit / n_total if n_total > 0 else 0
            cfs_rows.append({
                "context": ctx, "prevalence": prevalence,
                "spread": spread, "cfs": (prevalence + spread) / 2,
            })
        return pd.DataFrame(cfs_rows)

    cfs_parts = []
    for pt in ["original", "revised"]:
        sub = ctx_tokens[ctx_tokens["prompt_type"] == pt]
        c = compute_ablation_cfs(sub)
        c["prompt_type"] = pt
        cfs_parts.append(c)
    cfs_by_type = pd.concat(cfs_parts, ignore_index=True)
    cfs_by_type.to_csv(os.path.join(output_subdir, "cfs_vqa_by_type.csv"), index=False)

    cfs_wide = cfs_by_type.pivot_table(
        index="context", columns="prompt_type", values="cfs", aggfunc="first"
    ).reset_index()
    if "original" in cfs_wide.columns and "revised" in cfs_wide.columns:
        cfs_wide["diff"] = cfs_wide["revised"] - cfs_wide["original"]
        cfs_wide.to_csv(os.path.join(output_subdir, "cfs_paired_diff.csv"), index=False)
        print(f"    Mean CFS original: {cfs_wide['original'].mean():.3f}  "
              f"revised: {cfs_wide['revised'].mean():.3f}")

    # --- 4. Term decomposition ---
    print("\n  Section 4: Term Decomposition")

    term_comparison["category"] = term_comparison.apply(
        lambda r: "both (image-model bias)" if r["in_revised"] and r["in_original"]
        else "revised-only (revision-layer-driven)" if r["in_revised"]
        else "original-only (image-model specific)" if r["in_original"]
        else "neither", axis=1
    )

    decomp_summary = (
        term_comparison.groupby(["context", "category"]).size()
        .unstack(fill_value=0).reset_index()
    )
    decomp_summary.to_csv(
        os.path.join(output_subdir, "term_decomposition_summary.csv"), index=False
    )
    term_comparison.to_csv(
        os.path.join(output_subdir, "term_decomposition_detail.csv"), index=False
    )

    # --- 5. McNemar tests ---
    print("\n  Section 5: McNemar Tests")

    # Binary presence per (prompt_id, context, prompt_type, word)
    presence_rows = []
    for _, row in ctx_tokens.iterrows():
        for w in set(row["tokens"]):
            if (row["context"], w) in all_term_pairs:
                presence_rows.append({
                    "prompt_id": row["prompt_id"],
                    "context": row["context"],
                    "prompt_type": row["prompt_type"],
                    "word": w,
                })
    presence_df = pd.DataFrame(presence_rows).drop_duplicates()

    if len(presence_df) > 0:
        presence_wide = presence_df.assign(present=1).pivot_table(
            index=["prompt_id", "context", "word"],
            columns="prompt_type", values="present",
            aggfunc="first", fill_value=0,
        ).reset_index()

        # Ensure both columns exist
        for col in ["original", "revised"]:
            if col not in presence_wide.columns:
                presence_wide[col] = 0

        mcnemar_rows = []
        for (ctx, word), grp in presence_wide.groupby(["context", "word"]):
            a = int(((grp["original"] == 1) & (grp["revised"] == 1)).sum())
            b = int(((grp["original"] == 0) & (grp["revised"] == 1)).sum())
            c = int(((grp["original"] == 1) & (grp["revised"] == 0)).sum())
            d = int(((grp["original"] == 0) & (grp["revised"] == 0)).sum())
            n = len(grp)

            if (b + c) < 5:
                p = np.nan
            else:
                # McNemar with continuity correction
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                from scipy.stats import chi2 as chi2_dist
                p = 1 - chi2_dist.cdf(chi2, df=1)

            pct_o = (grp["original"] == 1).mean() * 100
            pct_r = (grp["revised"] == 1).mean() * 100
            direction = ("revision amplifies" if pct_r > pct_o
                        else "revision suppresses" if pct_r < pct_o
                        else "no difference")
            mcnemar_rows.append({
                "context": ctx, "word": word,
                "n_both": a, "n_rev_only": b, "n_orig_only": c, "n_neither": d,
                "n_total": n, "pct_original": pct_o, "pct_revised": pct_r,
                "mcnemar_p": p, "direction": direction,
            })

        mcnemar_df = pd.DataFrame(mcnemar_rows)
        valid_p = mcnemar_df["mcnemar_p"].notna()
        if valid_p.sum() > 0:
            from statsmodels.stats.multitest import multipletests
            _, padj, _, _ = multipletests(
                mcnemar_df.loc[valid_p, "mcnemar_p"], method="holm"
            )
            mcnemar_df.loc[valid_p, "mcnemar_p_adj"] = padj
        else:
            mcnemar_df["mcnemar_p_adj"] = np.nan

        mcnemar_df.to_csv(os.path.join(output_subdir, "mcnemar_term_tests.csv"), index=False)
        sig = mcnemar_df[mcnemar_df["mcnemar_p_adj"] < 0.05]
        print(f"    Significant terms: {len(sig)}")

    print(f"\n  {model_name} analysis complete. Outputs in: {output_subdir}")


# =============================================================================
# RUN FOR EACH IMAGE MODEL
# =============================================================================

models = vqa["model"].unique()
print(f"\nImage models found: {', '.join(models)}")

for m in models:
    vqa_sub = vqa[vqa["model"] == m].copy()
    if "flux" in m.lower():
        label = "Flux 2 Dev"
    elif any(x in m.lower() for x in ["stable", "sdxl", "sd"]):
        label = "SDXL"
    else:
        label = m
    subdir = os.path.join(OUTPUT_DIR, re.sub(r"[^a-zA-Z0-9]", "_", m).lower())
    run_ablation_analysis(vqa_sub, label, subdir)


# =============================================================================
# CROSS-MODEL COMPARISON
# =============================================================================

print("\n=== Cross-model comparison ===")

for pattern in ["cms_paired_tests.csv", "cfs_paired_diff.csv",
                "term_decomposition_summary.csv"]:
    import glob
    files = glob.glob(os.path.join(OUTPUT_DIR, "**", pattern), recursive=True)
    if len(files) > 1:
        parts = []
        for f in files:
            df = pd.read_csv(f)
            df["model"] = os.path.basename(os.path.dirname(f))
            parts.append(df)
        combined = pd.concat(parts, ignore_index=True)
        out_name = pattern.replace(".csv", "_all_models.csv")
        combined.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
        print(f"  Saved: {out_name}")

print("\n=== All ablation analyses complete ===")
print(f"Outputs in: {OUTPUT_DIR}")
