"""
WORLDVIEW Analysis Pipeline (Python port)

Main analysis script accompanying:
  "Prompt Revision as a Source of Cultural Bias in Commercial
   Text-to-Image Systems"

Implements:
  Step 1: CMS  = Contextual Markedness Score (semantic distance from baseline)
  Step 2: CFS  = Cultural Flattening Score = (prevalence + spread) / 2
  Visual: VQA-based visual TF-IDF + pipeline decomposition
  Robustness checks and cross-model consistency

Inputs (in DATA_DIR):
  worldview_prompts.csv, sbert_embeddings.csv, clip_image_embeddings.csv,
  vqa_descriptions.csv, geo_stopwords.txt

Outputs (in OUTPUT_DIR):
  cms_cfs_scores.csv, tfidf_top20_terms.csv, text_image_correlations.csv,
  robustness_tau_k.csv, cross_model_consistency.csv,
  visual_cfs.csv, pipeline_decomposition_summary.csv,
  and corresponding PDF figures.

Requirements:
  pip install pandas numpy scikit-learn scipy matplotlib seaborn
              adjustText nltk
"""

import os
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from adjustText import adjust_text

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR       = "data"
OUTPUT_DIR     = "output"
N_BASE_PROMPTS = 280

PROMPTS_PATH   = os.path.join(DATA_DIR, "worldview_prompts.csv")
SBERT_EMB_PATH = os.path.join(DATA_DIR, "sbert_embeddings.csv")
CLIP_EMB_PATH  = os.path.join(DATA_DIR, "clip_image_embeddings.csv")
VQA_PATH       = os.path.join(DATA_DIR, "vqa_descriptions.csv")
STOPWORDS_PATH = os.path.join(DATA_DIR, "geo_stopwords.txt")

TAU_QUANTILES = [0.50, 0.75, 0.90]
K_VALUES      = [5, 10, 15, 20]
TAU_DEFAULT   = 0.75
K_DEFAULT     = 10

MODEL_COLS = {"dalle": "#E41A1C", "imagen": "#377EB8", "gptimage": "#4DAF4A"}
MODEL_LABS = {"dalle": "DALL·E 3", "imagen": "Imagen", "gptimage": "GPT-Image"}
MODEL_MARKERS = {"dalle": "o", "imagen": "^", "gptimage": "s"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})

# =============================================================================
# HELPERS
# =============================================================================

_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip non-alpha, collapse whitespace."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", str(text).lower())).strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; returns NaN if either vector is zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def load_geo_stopwords(path: str) -> set:
    """Parse the geo_stopwords.txt file → set of lowercase terms."""
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for m in re.findall(r'"([^"]+)"', line):
                terms.add(m.strip().lower())
    return terms


def extract_insertions(original: str, revised: str, filter_set: set) -> list:
    """Content-word insertions: lemmatised tokens in revised but not original,
    minus stopwords/geo terms, min length 3."""
    orig_toks = {_lemmatizer.lemmatize(w) for w in original.split() if len(w) > 2}
    rev_toks  = [_lemmatizer.lemmatize(w) for w in revised.split() if len(w) > 2]
    return [t for t in rev_toks if t not in orig_toks and t not in filter_set]


def tfidf_per_model(context_docs: pd.DataFrame, min_df: int = 3,
                    max_df: float = 0.8) -> pd.DataFrame:
    """Compute TF-IDF scores per (model, context) document.

    Parameters
    ----------
    context_docs : DataFrame with columns [context, model, doc]
    min_df : minimum document frequency for vocabulary pruning
    max_df : maximum document proportion for vocabulary pruning

    Returns a long DataFrame: [model, context, term, tfidf_score].
    """
    rows = []
    for model, grp in context_docs.groupby("model"):
        texts = grp["doc"].tolist()
        contexts = grp["context"].tolist()
        n_docs = len(texts)
        # min_df as int if fewer docs than the proportion threshold
        effective_min_df = min(min_df, n_docs)
        effective_max_df = max_df if isinstance(max_df, float) else max_df
        vec = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            min_df=effective_min_df,
            max_df=effective_max_df,
        )
        try:
            mat = vec.fit_transform(texts)
        except ValueError:
            continue
        terms = vec.get_feature_names_out()
        for i in range(n_docs):
            scores = mat[i].toarray().ravel()
            nz = np.nonzero(scores)[0]
            for j in nz:
                rows.append({
                    "model": model,
                    "context": contexts[i],
                    "term": terms[j],
                    "tfidf_score": float(scores[j]),
                })
    return pd.DataFrame(rows)


def compute_cfs(tfidf_terms: pd.DataFrame, prompt_tokens: pd.DataFrame,
                tau_quantile: float, k: int) -> pd.DataFrame:
    """Compute Cultural Flattening Score = (prevalence + spread) / 2.

    Parameters
    ----------
    tfidf_terms : long DataFrame [model, context, term, tfidf_score]
    prompt_tokens : DataFrame with columns [prompt_id, context, model, inserted_tokens]
    tau_quantile : quantile for the tau threshold
    k : number of top terms for spread computation
    """
    results = []
    for model, mterms in tfidf_terms.groupby("model"):
        tau = mterms["tfidf_score"].quantile(tau_quantile)

        for ctx, cterms in mterms.groupby("context"):
            n_vocab = len(cterms)
            n_above = int((cterms["tfidf_score"] > tau).sum())
            prevalence = n_above / n_vocab if n_vocab > 0 else 0.0

            top_k = cterms.nlargest(k, "tfidf_score")["term"].tolist()

            ctx_prompts = prompt_tokens[
                (prompt_tokens["context"] == ctx) & (prompt_tokens["model"] == model)
            ]
            n_total = len(ctx_prompts)
            if n_total == 0:
                spread = np.nan
            else:
                n_hit = ctx_prompts["inserted_tokens"].apply(
                    lambda toks: any(t in top_k for t in toks)
                ).sum()
                spread = n_hit / n_total

            cfs = (prevalence + spread) / 2
            results.append({
                "context": ctx, "model": model,
                "prevalence": prevalence, "spread": spread, "cfs": cfs,
            })
    return pd.DataFrame(results)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=== Loading data ===")

prompts = pd.read_csv(PROMPTS_PATH)
prompts["prompt_id"] = prompts["prompt_id"].astype(str)
prompts["is_baseline"] = prompts["country"].isna()
prompts["row_idx"] = range(len(prompts))
prompts["base_prompt_id"] = (prompts["prompt_id"].astype(int) - 1) % N_BASE_PROMPTS + 1
prompts["context"] = prompts.apply(
    lambda r: None if r["is_baseline"] else f"{r['country']}+{r['language']}", axis=1
)
prompts["revised_clean"]  = prompts["revised_prompt"].apply(clean_text)
prompts["original_clean"] = prompts["original_prompt"].apply(clean_text)

print(f"  {len(prompts)} rows | {prompts['model'].nunique()} models | "
      f"{prompts['country'].nunique()} countries | {prompts['language'].nunique()} languages")

# SBERT
print("Loading SBERT embeddings...")
sbert_df = pd.read_csv(SBERT_EMB_PATH)
sbert_dim_cols = [c for c in sbert_df.columns if c.startswith("dim_")]
sbert_mat = sbert_df[sbert_dim_cols].values
assert len(sbert_mat) == len(prompts), "SBERT row count mismatch"

# CLIP
print("Loading CLIP embeddings...")
clip_raw = pd.read_csv(CLIP_EMB_PATH)
clip_dim_cols = [c for c in clip_raw.columns if c.startswith("dim_")]
clip_mat = clip_raw[clip_dim_cols].values

clip_meta = clip_raw[["prompt_id", "model"]].copy()
clip_meta["prompt_id"] = clip_meta["prompt_id"].astype(str)
clip_meta["clip_row"] = range(len(clip_meta))
clip_meta = clip_meta.merge(
    prompts[["prompt_id", "model", "country", "language",
             "is_baseline", "base_prompt_id", "context"]],
    on=["prompt_id", "model"], how="left",
)
clip_meta = clip_meta[
    (~clip_meta["is_baseline"]) |
    (clip_meta["is_baseline"] & clip_meta["language"].str.lower().eq("english"))
].copy()
print(f"  CLIP: {len(clip_meta)} images after filtering")


# =============================================================================
# 2. LOAD GEO STOPLIST
# =============================================================================

geo_terms = load_geo_stopwords(STOPWORDS_PATH)
en_stops = set(nltk_stopwords.words("english"))
filter_set = en_stops | geo_terms
print(f"  Loaded {len(geo_terms)} geographic/artifact stopwords")


# =============================================================================
# 3. INSERTION EXTRACTION
# =============================================================================

print("\n=== Extracting insertions ===")

country_rows  = prompts[~prompts["is_baseline"]].copy()
baseline_rows = prompts[
    prompts["is_baseline"] & prompts["language"].str.lower().eq("english")
].copy()

country_rows["inserted_tokens"] = country_rows.apply(
    lambda r: extract_insertions(r["original_clean"], r["revised_clean"], filter_set),
    axis=1,
)

mean_ins = country_rows["inserted_tokens"].apply(len).mean()
print(f"  Mean insertions per revised prompt: {mean_ins:.1f} tokens")


# =============================================================================
# 4. STEP 1: CMS — TEXT (SBERT)
# =============================================================================

print("\n=== Step 1: Computing CMS (text) ===")

bl_lookup = baseline_rows[["base_prompt_id", "model", "row_idx"]]
cms_text = np.full(len(country_rows), np.nan)

for i, (idx, cr) in enumerate(country_rows.iterrows()):
    bl = bl_lookup[
        (bl_lookup["base_prompt_id"] == cr["base_prompt_id"]) &
        (bl_lookup["model"] == cr["model"])
    ]
    if len(bl) == 0:
        continue
    bl_vec = sbert_mat[bl["row_idx"].values].mean(axis=0)
    cr_vec = sbert_mat[cr["row_idx"]]
    cms_text[i] = 1.0 - cosine_sim(bl_vec, cr_vec)

country_rows["cms_text"] = cms_text

cms_by_context = (
    country_rows.groupby(["context", "model"])["cms_text"]
    .mean()
    .reset_index()
    .rename(columns={"cms_text": "cms"})
)
print(f"  {len(cms_by_context)} (context, model) pairs")


# =============================================================================
# 5. STEP 1 (cont.): CMS — IMAGE (CLIP)
# =============================================================================

print("\n=== Step 1: Computing CMS (image) ===")

bl_img = clip_meta[clip_meta["is_baseline"]].copy()
ct_img = clip_meta[~clip_meta["is_baseline"]].copy()

bl_groups = bl_img.groupby(["base_prompt_id", "model"])["clip_row"].apply(list).reset_index()
bl_groups.columns = ["base_prompt_id", "model", "clip_rows"]

cms_img = np.full(len(ct_img), np.nan)
ct_img_reset = ct_img.reset_index(drop=True)
for i, cr in ct_img_reset.iterrows():
    bl = bl_groups[
        (bl_groups["base_prompt_id"] == cr["base_prompt_id"]) &
        (bl_groups["model"] == cr["model"])
    ]
    if len(bl) == 0:
        continue
    bl_rows = bl.iloc[0]["clip_rows"]
    bl_vec = clip_mat[bl_rows].mean(axis=0)
    cr_vec = clip_mat[cr["clip_row"]]
    cms_img[i] = 1.0 - cosine_sim(bl_vec, cr_vec)

ct_img = ct_img_reset.copy()
ct_img["cms_image"] = cms_img

img_displacement = ct_img[["prompt_id", "base_prompt_id", "model",
                            "country", "language", "context", "cms_image"]]


# =============================================================================
# 6. TEXT-IMAGE ALIGNMENT (Spearman)
# =============================================================================

print("\n=== Text-image markedness alignment ===")

text_disp = country_rows[["prompt_id", "base_prompt_id", "country",
                           "language", "model", "context", "cms_text"]]

alignment = text_disp.merge(
    img_displacement[["base_prompt_id", "model", "country", "language", "cms_image"]],
    on=["base_prompt_id", "model", "country", "language"],
    how="inner",
)
alignment = alignment.dropna(subset=["cms_text", "cms_image"])
print(f"  Matched: {len(alignment)} rows")

corr_rows = []
for (ctx, mdl), grp in alignment.groupby(["context", "model"]):
    if len(grp) < 5:
        continue
    rho, p = spearmanr(grp["cms_text"], grp["cms_image"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    corr_rows.append({
        "context": ctx, "model": mdl,
        "spearman_rho": rho, "p_value": p, "n": len(grp), "sig": sig,
    })
alignment_corr = pd.DataFrame(corr_rows)
alignment_corr.to_csv(os.path.join(OUTPUT_DIR, "text_image_correlations.csv"), index=False)

print("\n  Pooled correlations:")
for mdl, grp in alignment.groupby("model"):
    rho, p = spearmanr(grp["cms_text"], grp["cms_image"])
    print(f"    {mdl}: rho={rho:.3f}, p={p:.2e}, n={len(grp)}")


# =============================================================================
# 7. STEP 2: CFS — TF-IDF
# =============================================================================

print("\n=== Step 2: Computing CFS ===")

# Build per-(context, model) documents from inserted tokens
context_docs = (
    country_rows.groupby(["context", "model"])["inserted_tokens"]
    .apply(lambda x: " ".join(t for toks in x for t in toks))
    .reset_index()
    .rename(columns={"inserted_tokens": "doc"})
)

tfidf_all_terms = tfidf_per_model(context_docs, min_df=3, max_df=0.8)

# Save top-20
top20 = (
    tfidf_all_terms
    .sort_values("tfidf_score", ascending=False)
    .groupby(["model", "context"])
    .head(20)
)
top20.to_csv(os.path.join(OUTPUT_DIR, "tfidf_top20_terms.csv"), index=False)

# Compute default CFS
print(f"  Default: tau={TAU_DEFAULT*100:.0f}th percentile, k={K_DEFAULT}")

prompt_tokens = country_rows[["prompt_id", "context", "model", "inserted_tokens"]]
cfs_default = compute_cfs(tfidf_all_terms, prompt_tokens, TAU_DEFAULT, K_DEFAULT)


# =============================================================================
# 8. COMBINE CMS + CFS
# =============================================================================

print("\n=== Combining CMS and CFS ===")

results = cms_by_context.merge(cfs_default, on=["context", "model"], how="inner")
results.to_csv(os.path.join(OUTPUT_DIR, "cms_cfs_scores.csv"), index=False)
print(f"  {len(results)} (context, model) scores")


# =============================================================================
# 9. PLOTS
# =============================================================================

print("\n=== Generating plots ===")


def dot_plot(df, x_col, x_label, filename, sort_col=None):
    """Dot plot of contexts × models."""
    if sort_col is None:
        sort_col = x_col
    order = df.groupby("context")[sort_col].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 10))
    for mdl in df["model"].unique():
        sub = df[df["model"] == mdl]
        y = [order.index(c) for c in sub["context"]]
        ax.scatter(sub[x_col], y, c=MODEL_COLS.get(mdl, "grey"),
                   marker=MODEL_MARKERS.get(mdl, "o"), s=40, alpha=0.85,
                   label=MODEL_LABS.get(mdl, mdl), zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel(x_label)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)


# 9a. CMS dot plot
dot_plot(results, "cms", "Contextual Markedness Score (CMS)", "cms_by_context.pdf")

# 9b. CFS dot plot
dot_plot(results, "cfs", "Cultural Flattening Score (CFS)", "cfs_by_context.pdf")

# 9c. Prevalence vs spread scatter
fig, axes = plt.subplots(1, len(results["model"].unique()), figsize=(15, 6), sharey=True)
if not hasattr(axes, "__len__"):
    axes = [axes]
for ax, (mdl, grp) in zip(axes, results.groupby("model")):
    ax.scatter(grp["prevalence"], grp["spread"], c="#b2182b", s=30, alpha=0.7)
    texts = []
    for _, row in grp.iterrows():
        texts.append(ax.text(row["prevalence"], row["spread"], row["context"],
                             fontsize=6, color="grey30"))
    adjust_text(texts, ax=ax)
    ax.axvline(0.25, ls="--", c="grey60", lw=0.7)
    ax.axhline(0.50, ls="--", c="grey60", lw=0.7)
    ax.set_title(MODEL_LABS.get(mdl, mdl), fontweight="bold")
    ax.set_xlabel("Term prevalence")
    if ax == axes[0]:
        ax.set_ylabel("Term spread")
    ax.set_ylim(0, 1)
fig.savefig(os.path.join(OUTPUT_DIR, "scatter_prevalence_vs_spread.pdf"))
plt.close(fig)

# 9d. Text vs image CMS scatter
models_in_align = alignment["model"].unique()
fig, axes = plt.subplots(1, len(models_in_align), figsize=(12, 5), sharey=True)
if not hasattr(axes, "__len__"):
    axes = [axes]
for ax, (mdl, grp) in zip(axes, alignment.groupby("model")):
    ax.scatter(grp["cms_text"], grp["cms_image"], s=3, alpha=0.1, c="grey30")
    z = np.polyfit(grp["cms_text"], grp["cms_image"], 1)
    xs = np.linspace(grp["cms_text"].min(), grp["cms_text"].max(), 100)
    ax.plot(xs, np.polyval(z, xs), c="#d7191c", lw=1.2)
    ax.set_title(MODEL_LABS.get(mdl, mdl), fontweight="bold")
    ax.set_xlabel("Text CMS (SBERT)")
    if ax == axes[0]:
        ax.set_ylabel("Image CMS (CLIP)")
fig.savefig(os.path.join(OUTPUT_DIR, "scatter_text_vs_image_cms.pdf"))
plt.close(fig)


# =============================================================================
# 10. ROBUSTNESS CHECKS
# =============================================================================

print("\n=== Robustness checks ===")

robust_rows = []
for tau_q in TAU_QUANTILES:
    for k in K_VALUES:
        cfs_alt = compute_cfs(tfidf_all_terms, prompt_tokens, tau_q, k)
        merged = cfs_default.merge(
            cfs_alt[["context", "model", "cfs"]].rename(columns={"cfs": "cfs_alt"}),
            on=["context", "model"],
        )
        kt, _ = kendalltau(merged["cfs"], merged["cfs_alt"])
        sr, _ = spearmanr(merged["cfs"], merged["cfs_alt"])
        robust_rows.append({
            "tau_q": tau_q, "k": k,
            "kendall_tau": kt, "spearman_rho": sr, "n": len(merged),
        })

robustness = pd.DataFrame(robust_rows)
robustness.to_csv(os.path.join(OUTPUT_DIR, "robustness_tau_k.csv"), index=False)
print(robustness.to_string(index=False))


# =============================================================================
# 11. CROSS-MODEL CONSISTENCY
# =============================================================================

print("\n=== Cross-model consistency ===")

models = results["model"].unique().tolist()
xmodel_rows = []
for m1, m2 in combinations(models, 2):
    d1 = results[results["model"] == m1].set_index("context")
    d2 = results[results["model"] == m2].set_index("context")
    shared = d1.index.intersection(d2.index)
    if len(shared) < 3:
        continue
    xmodel_rows.append({
        "model_1": m1, "model_2": m2,
        "tau_cms": kendalltau(d1.loc[shared, "cms"], d2.loc[shared, "cms"])[0],
        "tau_cfs": kendalltau(d1.loc[shared, "cfs"], d2.loc[shared, "cfs"])[0],
        "tau_prevalence": kendalltau(d1.loc[shared, "prevalence"], d2.loc[shared, "prevalence"])[0],
        "tau_spread": kendalltau(d1.loc[shared, "spread"], d2.loc[shared, "spread"])[0],
        "n": len(shared),
    })

cross_model = pd.DataFrame(xmodel_rows)
cross_model.to_csv(os.path.join(OUTPUT_DIR, "cross_model_consistency.csv"), index=False)
print(cross_model.to_string(index=False))


# =============================================================================
# 12. VISUAL-LEVEL ANALYSIS (VQA)
# =============================================================================

print("\n=== Visual-level analysis (VQA) ===")

VQA_MODELS = ["dalle", "imagen", "gptimage"]

if not os.path.exists(VQA_PATH):
    print(f"  VQA file not found at: {VQA_PATH}")
    print("  Skipping visual analysis.")
else:
    vqa_raw = pd.read_csv(VQA_PATH)
    vqa_raw["prompt_id"] = vqa_raw["prompt_id"].astype(str)
    print(f"  Loaded {len(vqa_raw)} VQA descriptions")

    vqa = vqa_raw.merge(
        prompts[prompts["model"].isin(VQA_MODELS)][
            ["prompt_id", "model", "country", "language", "is_baseline",
             "base_prompt_id", "context", "original_clean"]
        ],
        on=["prompt_id", "model"], how="inner",
    )
    print(f"  Matched: {len(vqa)} rows "
          f"({vqa['is_baseline'].sum()} baseline, {(~vqa['is_baseline']).sum()} context)")

    vqa["desc_clean"] = vqa["description"].apply(clean_text)

    # 12a. Visual inserted tokens
    vqa_ctx = vqa[~vqa["is_baseline"]].copy()
    vqa_ctx["visual_tokens"] = vqa_ctx.apply(
        lambda r: extract_insertions(r["original_clean"], r["desc_clean"], filter_set),
        axis=1,
    )
    print(f"  Mean visual tokens per image: {vqa_ctx['visual_tokens'].apply(len).mean():.1f}")

    # 12b. Visual TF-IDF
    vis_docs = (
        vqa_ctx.groupby(["model", "context"])["visual_tokens"]
        .apply(lambda x: " ".join(t for toks in x for t in toks))
        .reset_index()
        .rename(columns={"visual_tokens": "doc"})
    )
    visual_tfidf = tfidf_per_model(vis_docs, min_df=3, max_df=0.8)

    vis_top20 = (
        visual_tfidf.sort_values("tfidf_score", ascending=False)
        .groupby(["model", "context"]).head(20)
    )
    vis_top20.to_csv(os.path.join(OUTPUT_DIR, "visual_tfidf_top20.csv"), index=False)

    # 12c. Visual CFS
    vis_prompt_tokens = vqa_ctx[["prompt_id", "context", "model", "visual_tokens"]].rename(
        columns={"visual_tokens": "inserted_tokens"}
    )
    visual_cfs_parts = []
    for m in VQA_MODELS:
        vt = visual_tfidf[visual_tfidf["model"] == m]
        vp = vis_prompt_tokens[vis_prompt_tokens["model"] == m]
        if len(vt) == 0:
            continue
        c = compute_cfs(vt, vp, TAU_DEFAULT, K_DEFAULT)
        c["model"] = m
        visual_cfs_parts.append(c)
    visual_cfs = pd.concat(visual_cfs_parts, ignore_index=True)
    visual_cfs.to_csv(os.path.join(OUTPUT_DIR, "visual_cfs.csv"), index=False)

    # 12d. Pipeline decomposition
    print("\n  Computing pipeline decomposition...")
    K_DECOMP = 20

    decomp_rows = []
    for m in VQA_MODELS:
        text_top = (
            tfidf_all_terms[tfidf_all_terms["model"] == m]
            .sort_values("tfidf_score", ascending=False)
            .groupby("context").head(K_DECOMP)
            .groupby("context")["term"].apply(set).to_dict()
        )
        vis_top = (
            visual_tfidf[visual_tfidf["model"] == m]
            .sort_values("tfidf_score", ascending=False)
            .groupby("context").head(K_DECOMP)
            .groupby("context")["term"].apply(set).to_dict()
        )
        for ctx in set(text_top) & set(vis_top):
            tt = text_top[ctx]
            vt = vis_top[ctx]
            prop = tt & vt
            to = tt - vt
            vo = vt - tt
            n_total = len(prop) + len(to) + len(vo)
            decomp_rows.append({
                "model": m, "context": ctx,
                "propagated": ", ".join(sorted(prop)),
                "text_only": ", ".join(sorted(to)),
                "visual_only": ", ".join(sorted(vo)),
                "n_propagated": len(prop),
                "n_text_only": len(to),
                "n_visual_only": len(vo),
                "n_total": n_total,
                "pct_propagated": len(prop) / n_total if n_total > 0 else 0,
            })

    decomp = pd.DataFrame(decomp_rows)
    decomp_summary = decomp[["model", "context", "n_propagated", "n_text_only",
                              "n_visual_only", "n_total", "pct_propagated"]]
    decomp_summary.to_csv(
        os.path.join(OUTPUT_DIR, "pipeline_decomposition_summary.csv"), index=False
    )
    decomp[["model", "context", "propagated", "text_only", "visual_only"]].to_csv(
        os.path.join(OUTPUT_DIR, "pipeline_decomposition_detail.csv"), index=False
    )

    for m in VQA_MODELS:
        ds = decomp_summary[decomp_summary["model"] == m]
        if len(ds) > 0:
            print(f"    [{m}] Mean propagated: {ds['n_propagated'].mean():.1f} "
                  f"/ {K_DECOMP} terms ({ds['pct_propagated'].mean()*100:.0f}%)")

    # 12e. Propagation stacked bar
    prop_rates = decomp_summary.copy()
    prop_rates["n_visual_total"] = prop_rates["n_propagated"] + prop_rates["n_visual_only"]
    prop_rates["propagation_rate"] = np.where(
        prop_rates["n_visual_total"] > 0,
        prop_rates["n_propagated"] / prop_rates["n_visual_total"], 0
    )
    prop_rates.to_csv(os.path.join(OUTPUT_DIR, "propagation_rates.csv"), index=False)

    # Propagation summary
    prop_summary_rows = []
    for m in VQA_MODELS:
        pr = prop_rates[prop_rates["model"] == m]["propagation_rate"]
        if len(pr) == 0:
            continue
        prop_summary_rows.append({
            "model": MODEL_LABS.get(m, m),
            "median_prop": pr.median(), "mean_prop": pr.mean(),
            "min_prop": pr.min(), "max_prop": pr.max(), "sd_prop": pr.std(),
            "n_contexts": len(pr),
            "n_majority_rewriter": int((pr > 0.5).sum()),
        })
    pd.DataFrame(prop_summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "propagation_summary_by_model.csv"), index=False
    )

    print("\n  Visual analysis complete.")


print(f"\n=== Done. All outputs in {OUTPUT_DIR}/ ===")
