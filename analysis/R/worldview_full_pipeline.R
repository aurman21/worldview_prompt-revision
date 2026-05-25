# =============================================================================
# WORLDVIEW Analysis Pipeline
#
# Main analysis script accompanying:
#   "Prompt Revision as a Source of Cultural Bias in Commercial
#    Text-to-Image Systems"
#
# Implements:
#   Step 1: CMS  = Contextual Markedness Score (semantic distance from baseline)
#   Step 2: CFS  = Cultural Flattening Score = (prevalence + spread) / 2
#   Step 3: Qualitative TF-IDF (see worldview_tfidf_qual.R)
#   Visual: VQA-based visual TF-IDF + pipeline decomposition
#   Robustness checks and cross-model consistency
#
# Inputs (in DATA_DIR):
#   worldview_prompts.csv   — master prompt dataset
#   sbert_embeddings.csv    — SBERT embeddings (generate_sbert_embeddings.py)
#   clip_image_embeddings.csv — CLIP image embeddings (generate_clip_embeddings.py)
#   vqa_descriptions.csv    — VQA image descriptions (generate_vqa_descriptions.py)
#   geo_stopwords.txt       — geographic/artifact stopwords
#
# Outputs (in OUTPUT_DIR):
#   cms_cfs_scores.csv, tfidf_top20_terms.csv, text_image_correlations.csv,
#   robustness_tau_k.csv, cross_model_consistency.csv,
#   visual_cfs.csv, pipeline_decomposition_summary.csv,
#   and corresponding PDF figures.
#
# Requirements:
#   install.packages(c("tidyverse", "textstem", "text2vec",
#                      "ggrepel", "stopwords"))
# =============================================================================

library(tidyverse)
library(textstem)
library(text2vec)
library(ggrepel)

# =============================================================================
# CONFIG — adjust paths to match your directory layout
# =============================================================================

DATA_DIR       <- "data"
OUTPUT_DIR     <- "output"
N_BASE_PROMPTS <- 280

PROMPTS_PATH   <- file.path(DATA_DIR, "worldview_prompts.csv")
SBERT_EMB_PATH <- file.path(DATA_DIR, "sbert_embeddings.csv")
CLIP_EMB_PATH  <- file.path(DATA_DIR, "clip_image_embeddings.csv")
VQA_PATH       <- file.path(DATA_DIR, "vqa_descriptions.csv")
STOPWORDS_PATH <- file.path(DATA_DIR, "geo_stopwords.txt")

# Robustness grid
TAU_QUANTILES <- c(0.50, 0.75, 0.90)
K_VALUES      <- c(5, 10, 15, 20)

# Defaults for main results
TAU_DEFAULT <- 0.75
K_DEFAULT   <- 10

# Plot theme
PLOT_THEME <- theme_minimal(base_size = 14) +
  theme(
    strip.text       = element_text(face = "bold", size = 14),
    axis.title       = element_text(size = 13),
    axis.text        = element_text(size = 11),
    legend.text      = element_text(size = 11),
    legend.title     = element_text(size = 12),
    plot.title       = element_text(size = 15, face = "bold"),
    plot.subtitle    = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

MODEL_COLS   <- c(dalle = "#E41A1C", imagen = "#377EB8", gptimage = "#4DAF4A")
MODEL_LABS   <- c(dalle = "DALL\u00b7E 3", imagen = "Imagen", gptimage = "GPT-Image")
MODEL_SHAPES <- c(dalle = 16, imagen = 17, gptimage = 15)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

cat("=== Loading data ===\n")

prompts <- read_csv(PROMPTS_PATH, show_col_types = FALSE) %>%
  mutate(
    prompt_id      = as.character(prompt_id),
    is_baseline    = is.na(country),
    row_idx        = row_number(),
    base_prompt_id = ((as.numeric(prompt_id) - 1) %% N_BASE_PROMPTS) + 1,
    context        = if_else(is_baseline, NA_character_, paste0(country, "+", language)),
    revised_clean  = revised_prompt %>%
      tolower() %>%
      str_replace_all("[^a-z\\s]", " ") %>%
      str_squish(),
    original_clean = original_prompt %>%
      tolower() %>%
      str_replace_all("[^a-z\\s]", " ") %>%
      str_squish()
  )

cat(sprintf("  %d rows | %d models | %d countries | %d languages\n",
            nrow(prompts), n_distinct(prompts$model),
            n_distinct(prompts$country, na.rm = TRUE),
            n_distinct(prompts$language)))

# SBERT
cat("Loading SBERT embeddings...\n")
sbert_mat <- as.matrix(
  read_csv(SBERT_EMB_PATH, show_col_types = FALSE) %>% select(starts_with("dim_"))
)
stopifnot(nrow(sbert_mat) == nrow(prompts))

# CLIP
cat("Loading CLIP embeddings...\n")
clip_raw <- read_csv(CLIP_EMB_PATH, show_col_types = FALSE)
clip_mat <- as.matrix(clip_raw %>% select(starts_with("dim_")))

clip_meta <- clip_raw %>%
  select(prompt_id, model) %>%
  mutate(prompt_id = as.character(prompt_id), clip_row = row_number()) %>%
  left_join(
    prompts %>% select(prompt_id, model, country, language, is_baseline, base_prompt_id, context),
    by = c("prompt_id", "model")
  ) %>%
  filter(!is_baseline | (is_baseline & tolower(language) == "english"))

cat(sprintf("  CLIP: %d images after filtering\n", nrow(clip_meta)))


# =============================================================================
# 2. LOAD GEO STOPLIST FROM EXTERNAL FILE
# =============================================================================

cosine_sim <- function(a, b) {
  d <- sum(a * b)
  n <- sqrt(sum(a^2)) * sqrt(sum(b^2))
  if (n < 1e-10) return(NA_real_)
  d / n
}

# Parse stopwords file: extract quoted terms, skip comments and blanks
geo_terms <- readLines(STOPWORDS_PATH) %>%
  str_extract_all('"([^"]+)"') %>%
  unlist() %>%
  str_remove_all('"') %>%
  str_trim() %>%
  tolower() %>%
  unique()

cat(sprintf("  Loaded %d geographic/artifact stopwords\n", length(geo_terms)))

filter_set <- c(stopwords::stopwords("en", source = "snowball"), geo_terms) %>%
  unique() %>%
  tolower()


# =============================================================================
# 3. INSERTION EXTRACTION
# =============================================================================

cat("\n=== Extracting insertions ===\n")

country_rows  <- prompts %>% filter(!is_baseline)
baseline_rows <- prompts %>% filter(is_baseline & tolower(language) == "english")

extract_insertions <- function(original, revised, fs) {
  orig <- original %>% str_split("\\s+") %>% unlist() %>%
    lemmatize_words() %>% .[nchar(.) > 2]
  rev <- revised %>% str_split("\\s+") %>% unlist() %>%
    lemmatize_words() %>% .[nchar(.) > 2]
  ins <- setdiff(rev, orig)
  ins[!ins %in% fs]
}

country_rows <- country_rows %>%
  rowwise() %>%
  mutate(inserted_tokens = list(extract_insertions(original_clean, revised_clean, filter_set))) %>%
  ungroup()

cat(sprintf("  Mean insertions per revised prompt: %.1f tokens\n",
            mean(lengths(country_rows$inserted_tokens))))


# =============================================================================
# 4. STEP 1: CONTEXTUAL MARKEDNESS SCORE (CMS) — TEXT
# =============================================================================

cat("\n=== Step 1: Computing CMS (text) ===\n")

bl_text_lookup <- baseline_rows %>% select(base_prompt_id, model, row_idx)

cms_text_vec <- rep(NA_real_, nrow(country_rows))
for (i in seq_len(nrow(country_rows))) {
  cr <- country_rows[i, ]
  bl <- bl_text_lookup %>%
    filter(base_prompt_id == cr$base_prompt_id, model == cr$model)
  if (nrow(bl) == 0) next
  bl_vec <- colMeans(sbert_mat[bl$row_idx, , drop = FALSE])
  cr_vec <- sbert_mat[cr$row_idx, ]
  cms_text_vec[i] <- 1 - cosine_sim(bl_vec, cr_vec)
}
country_rows$cms_text <- cms_text_vec

cms_by_context <- country_rows %>%
  group_by(context, model) %>%
  summarise(cms = mean(cms_text, na.rm = TRUE), .groups = "drop")

cat(sprintf("  %d (context, model) pairs\n", nrow(cms_by_context)))


# =============================================================================
# 5. STEP 1 (cont.): CMS — IMAGE (CLIP)
# =============================================================================

cat("\n=== Step 1: Computing CMS (image) ===\n")

bl_img <- clip_meta %>% filter(is_baseline)
ct_img <- clip_meta %>% filter(!is_baseline)

bl_img_groups <- bl_img %>%
  group_by(base_prompt_id, model) %>%
  summarise(clip_rows = list(clip_row), .groups = "drop")

cms_img_vec <- rep(NA_real_, nrow(ct_img))
for (i in seq_len(nrow(ct_img))) {
  cr <- ct_img[i, ]
  bl <- bl_img_groups %>%
    filter(base_prompt_id == cr$base_prompt_id, model == cr$model)
  if (nrow(bl) == 0) next
  bl_vec <- colMeans(clip_mat[unlist(bl$clip_rows), , drop = FALSE])
  cr_vec <- clip_mat[cr$clip_row, ]
  cms_img_vec[i] <- 1 - cosine_sim(bl_vec, cr_vec)
}
ct_img$cms_image <- cms_img_vec

img_displacement <- ct_img %>%
  select(prompt_id, base_prompt_id, model, country, language, context, cms_image)


# =============================================================================
# 6. TEXT-IMAGE MARKEDNESS ALIGNMENT (Spearman correlations)
# =============================================================================

cat("\n=== Text-image markedness alignment ===\n")

text_disp <- country_rows %>%
  select(prompt_id, base_prompt_id, country, language, model, context, cms_text)

alignment_joined <- text_disp %>%
  inner_join(
    img_displacement %>% select(base_prompt_id, model, country, language, cms_image),
    by = c("base_prompt_id", "model", "country", "language")
  ) %>%
  filter(!is.na(cms_text) & !is.na(cms_image))

cat(sprintf("  Matched: %d rows\n", nrow(alignment_joined)))

alignment_corr <- alignment_joined %>%
  group_by(context, model) %>%
  filter(n() >= 5) %>%
  summarise(
    spearman_rho = cor(cms_text, cms_image, method = "spearman", use = "complete.obs"),
    p_value = tryCatch(
      cor.test(cms_text, cms_image, method = "spearman", exact = FALSE)$p.value,
      error = function(e) NA_real_
    ),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(sig = case_when(p_value < 0.001 ~ "***", p_value < 0.01 ~ "**",
                         p_value < 0.05 ~ "*", TRUE ~ ""))

write_csv(alignment_corr, file.path(OUTPUT_DIR, "text_image_correlations.csv"))

# Pooled
cat("\n  Pooled correlations:\n")
alignment_joined %>%
  group_by(model) %>%
  summarise(
    rho = cor(cms_text, cms_image, method = "spearman", use = "complete.obs"),
    p = cor.test(cms_text, cms_image, method = "spearman", exact = FALSE)$p.value,
    n = n(), .groups = "drop"
  ) %>% print()


# =============================================================================
# 7. STEP 2: CULTURAL FLATTENING SCORE (CFS) — TF-IDF
# =============================================================================

cat("\n=== Step 2: Computing CFS ===\n")

# --- 7a. TF-IDF per (context, model) ---

context_docs <- country_rows %>%
  group_by(context, model) %>%
  summarise(doc = paste(unlist(inserted_tokens), collapse = " "), .groups = "drop")

tfidf_all_terms <- context_docs %>%
  group_by(model) %>%
  group_modify(function(data, grp) {
    it <- itoken(data$doc, progressbar = FALSE)
    vocab <- create_vocabulary(it) %>%
      prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.8)
    vectorizer <- vocab_vectorizer(vocab)
    dtm <- create_dtm(it, vectorizer)
    tfidf_model <- TfIdf$new()
    tfidf_mat <- fit_transform(dtm, tfidf_model)
    
    map_dfr(seq_len(nrow(data)), function(i) {
      scores <- as.numeric(tfidf_mat[i, ])
      names(scores) <- colnames(tfidf_mat)
      scores <- scores[scores > 0]
      tibble(
        context = data$context[i],
        term = names(scores),
        tfidf_score = as.numeric(scores)
      )
    })
  }) %>%
  ungroup()

write_csv(
  tfidf_all_terms %>%
    group_by(model, context) %>%
    slice_max(tfidf_score, n = 20) %>%
    ungroup(),
  file.path(OUTPUT_DIR, "tfidf_top20_terms.csv")
)


# --- 7b. Function to compute CFS for given tau_q and k ---

compute_cfs <- function(tfidf_all_terms, country_rows, tau_quantile, k) {
  
  tau_vals <- tfidf_all_terms %>%
    group_by(model) %>%
    summarise(tau = quantile(tfidf_score, tau_quantile), .groups = "drop")
  
  prevalence <- tfidf_all_terms %>%
    left_join(tau_vals, by = "model") %>%
    group_by(context, model) %>%
    summarise(
      n_above_tau = sum(tfidf_score > tau),
      n_vocab     = n(),
      prevalence  = n_above_tau / n_vocab,
      .groups = "drop"
    ) %>%
    select(context, model, prevalence)
  
  top_k_terms <- tfidf_all_terms %>%
    group_by(context, model) %>%
    slice_max(tfidf_score, n = k, with_ties = FALSE) %>%
    summarise(top_terms = list(term), .groups = "drop")
  
  prompt_insertions <- country_rows %>%
    select(prompt_id, context, model, inserted_tokens)
  
  spread <- top_k_terms %>%
    left_join(
      prompt_insertions %>%
        group_by(context, model) %>%
        summarise(
          prompt_data = list(tibble(prompt_id = prompt_id, tokens = inserted_tokens)),
          .groups = "drop"
        ),
      by = c("context", "model")
    ) %>%
    rowwise() %>%
    mutate(
      spread = {
        pd <- prompt_data
        tt <- top_terms
        n_total <- nrow(pd)
        if (n_total == 0) {
          NA_real_
        } else {
          n_hit <- sum(sapply(pd$tokens, function(toks) any(toks %in% tt)))
          n_hit / n_total
        }
      }
    ) %>%
    ungroup() %>%
    select(context, model, spread)
  
  # Arithmetic mean
  cfs <- prevalence %>%
    inner_join(spread, by = c("context", "model")) %>%
    mutate(cfs = (prevalence + spread) / 2)
  
  return(cfs)
}


# --- 7c. Compute default CFS ---

cat(sprintf("  Default: tau=%.0fth percentile, k=%d\n",
            TAU_DEFAULT * 100, K_DEFAULT))

cfs_default <- compute_cfs(tfidf_all_terms, country_rows, TAU_DEFAULT, K_DEFAULT)


# =============================================================================
# 8. COMBINE CMS + CFS
# =============================================================================

cat("\n=== Combining CMS and CFS ===\n")

results <- cms_by_context %>%
  inner_join(cfs_default, by = c("context", "model"))

write_csv(results, file.path(OUTPUT_DIR, "cms_cfs_scores.csv"))
cat(sprintf("  %d (context, model) scores\n", nrow(results)))


# =============================================================================
# 9. PLOTS
# =============================================================================

cat("\n=== Generating plots ===\n")

# --- 9a. CMS dot plot (Figure 1) ---

p_cms <- results %>%
  mutate(context = fct_reorder(context, cms, .fun = mean)) %>%
  ggplot(aes(x = cms, y = context, colour = model, shape = model)) +
  geom_point(size = 2.5, alpha = 0.85) +
  scale_colour_manual(values = MODEL_COLS, labels = MODEL_LABS) +
  scale_shape_manual(values = MODEL_SHAPES, labels = MODEL_LABS) +
  labs(x = "Contextual Markedness Score (CMS)",
       y = NULL, colour = NULL, shape = NULL,
       caption = "(Sorted by mean CMS across models)") +
  PLOT_THEME +
  theme(legend.position = "top")

ggsave(file.path(OUTPUT_DIR, "cms_by_context.pdf"),
       p_cms, width = 8, height = 10)


# --- 9b. CFS dot plot (Figure 2) ---

p_cfs <- results %>%
  mutate(context = fct_reorder(context, cfs, .fun = mean)) %>%
  ggplot(aes(x = cfs, y = context, colour = model, shape = model)) +
  geom_point(size = 2.5, alpha = 0.85) +
  scale_colour_manual(values = MODEL_COLS, labels = MODEL_LABS) +
  scale_shape_manual(values = MODEL_SHAPES, labels = MODEL_LABS) +
  labs(x = "Cultural Flattening Score (CFS)",
       y = NULL, colour = NULL, shape = NULL,
       caption = "(Sorted by mean CFS across models)") +
  PLOT_THEME +
  theme(legend.position = "top")

ggsave(file.path(OUTPUT_DIR, "cfs_by_context.pdf"),
       p_cfs, width = 8, height = 10)


# --- 9c. Prevalence vs spread scatter (Figure 3) ---

p_prev_spread <- results %>%
  ggplot(aes(x = prevalence, y = spread)) +
  geom_vline(xintercept = 0.25, linetype = "dashed", colour = "grey60") +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_point(alpha = 0.7, size = 2.5, colour = "#b2182b") +
  geom_text_repel(aes(label = context), size = 3, max.overlaps = 25, colour = "grey30") +
  facet_wrap(~model, labeller = labeller(model = MODEL_LABS)) +
  scale_x_continuous(limits = c(0, max(results$prevalence) * 1.05)) +
  scale_y_continuous(limits = c(0, 1)) +
  annotate("text", x = Inf, y = Inf, label = "Pervasive\nflattening",
           hjust = 1.1, vjust = 1.3, size = 3.5, colour = "#b2182b", fontface = "bold.italic") +
  annotate("text", x = -Inf, y = -Inf, label = "Minimal\nmarking",
           hjust = -0.05, vjust = -0.5, size = 3.2, colour = "grey50", fontface = "italic") +
  annotate("text", x = Inf, y = -Inf, label = "Concentrated but\ntopically constrained",
           hjust = 1.1, vjust = -0.5, size = 3.2, colour = "grey50", fontface = "italic") +
  annotate("text", x = -Inf, y = Inf, label = "Light but\nubiquitous",
           hjust = -0.05, vjust = 1.3, size = 3.2, colour = "grey50", fontface = "italic") +
  labs(x = "Term prevalence (share of vocab above \u03c4)",
       y = "Term spread (fraction of prompts with top-k terms)",
       caption = "(Top-right = distinctive terms are both concentrated and appear everywhere)") +
  PLOT_THEME +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(OUTPUT_DIR, "scatter_prevalence_vs_spread.pdf"),
       p_prev_spread, width = 15, height = 6)


# --- 9d. Text vs image CMS scatter (Figure 7 / Appendix I) ---

p_scatter <- alignment_joined %>%
  ggplot(aes(x = cms_text, y = cms_image)) +
  geom_point(alpha = 0.1, size = 0.8) +
  geom_smooth(method = "lm", colour = "#d7191c", se = TRUE, linewidth = 0.9) +
  facet_wrap(~model, labeller = labeller(model = MODEL_LABS)) +
  labs(x = "Text-level CMS (SBERT cosine distance)",
       y = "Image-level CMS (CLIP cosine distance)",
       caption = "(Each point = one prompt\u2013context pair)") +
  PLOT_THEME

ggsave(file.path(OUTPUT_DIR, "scatter_text_vs_image_cms.pdf"),
       p_scatter, width = 12, height = 5)


# =============================================================================
# 10. ROBUSTNESS CHECKS (Appendix F)
# =============================================================================

cat("\n=== Robustness checks ===\n")

robustness_results <- expand_grid(
  tau_q = TAU_QUANTILES,
  k     = K_VALUES
) %>%
  rowwise() %>%
  mutate(
    cfs_data = list(compute_cfs(tfidf_all_terms, country_rows, tau_q, k))
  ) %>%
  ungroup()

robustness_corr <- robustness_results %>%
  rowwise() %>%
  mutate(
    comparison = {
      cfs_alt <- cfs_data
      cfs_joined <- cfs_default %>%
        select(context, model, cfs) %>%
        inner_join(cfs_alt %>% select(context, model, cfs_alt = cfs),
                   by = c("context", "model"))
      list(tibble(
        tau_q = tau_q,
        k = k,
        kendall_tau = cor(cfs_joined$cfs, cfs_joined$cfs_alt,
                          method = "kendall", use = "complete.obs"),
        spearman_rho = cor(cfs_joined$cfs, cfs_joined$cfs_alt,
                           method = "spearman", use = "complete.obs"),
        n = nrow(cfs_joined)
      ))
    }
  ) %>%
  ungroup() %>%
  mutate(comparison = map(comparison, ~.x)) %>%
  select(comparison) %>%
  unnest(comparison)

cat("\n  Rank correlation with default (tau=75th, k=10):\n")
print(robustness_corr)

write_csv(robustness_corr, file.path(OUTPUT_DIR, "robustness_tau_k.csv"))


# =============================================================================
# 11. CROSS-MODEL CONSISTENCY (Appendix G)
# =============================================================================

cat("\n=== Cross-model consistency ===\n")

models <- unique(results$model)
model_pairs <- combn(models, 2, simplify = FALSE)

cross_model <- map_dfr(model_pairs, function(pair) {
  d1 <- results %>% filter(model == pair[1]) %>% select(context, cms, cfs, prevalence, spread)
  d2 <- results %>% filter(model == pair[2]) %>% select(context, cms, cfs, prevalence, spread)
  joined <- inner_join(d1, d2, by = "context", suffix = c("_1", "_2"))
  
  tibble(
    model_1 = pair[1],
    model_2 = pair[2],
    tau_cms       = cor(joined$cms_1, joined$cms_2, method = "kendall"),
    tau_cfs       = cor(joined$cfs_1, joined$cfs_2, method = "kendall"),
    tau_prevalence = cor(joined$prevalence_1, joined$prevalence_2, method = "kendall"),
    tau_spread    = cor(joined$spread_1, joined$spread_2, method = "kendall"),
    n = nrow(joined)
  )
})

cat("\n  Cross-model Kendall's tau:\n")
print(cross_model)

write_csv(cross_model, file.path(OUTPUT_DIR, "cross_model_consistency.csv"))


# =============================================================================
# 12. VISUAL-LEVEL ANALYSIS (VQA — multi-model)
# =============================================================================

cat("\n=== Visual-level analysis (VQA) ===\n")

VQA_MODELS <- c("dalle", "imagen", "gptimage")

if (!file.exists(VQA_PATH)) {
  cat("  VQA file not found at:", VQA_PATH, "\n")
  cat("  Skipping visual analysis. Run generate_vqa_descriptions.py first.\n")
} else {
  
  vqa_raw <- read_csv(VQA_PATH, show_col_types = FALSE) %>%
    mutate(prompt_id = as.character(prompt_id))
  
  cat(sprintf("  Loaded %d VQA descriptions\n", nrow(vqa_raw)))
  
  vqa <- vqa_raw %>%
    inner_join(
      prompts %>%
        filter(model %in% VQA_MODELS) %>%
        select(prompt_id, model, country, language, is_baseline,
               base_prompt_id, context, original_clean),
      by = c("prompt_id", "model")
    )
  
  if ("model.x" %in% names(vqa)) {
    vqa <- vqa %>%
      mutate(model = coalesce(model.x, model.y)) %>%
      select(-any_of(c("model.x", "model.y")))
  }
  
  cat(sprintf("  Matched to metadata: %d rows (%d baseline, %d context-specified)\n",
              nrow(vqa), sum(vqa$is_baseline), sum(!vqa$is_baseline)))
  
  for (m in VQA_MODELS) {
    cat(sprintf("    %s: %d rows\n", m, sum(vqa$model == m)))
  }
  
  vqa <- vqa %>%
    mutate(
      desc_clean = description %>%
        tolower() %>%
        str_replace_all("[^a-z\\s]", " ") %>%
        str_squish()
    )
  
  # --- 12a. Extract visual inserted tokens ---
  
  vqa_context <- vqa %>% filter(!is_baseline)
  
  vqa_context <- vqa_context %>%
    rowwise() %>%
    mutate(visual_tokens = list(extract_insertions(original_clean, desc_clean, filter_set))) %>%
    ungroup()
  
  cat(sprintf("  Mean visual tokens per image: %.1f\n",
              mean(lengths(vqa_context$visual_tokens))))
  
  # --- 12b. Visual TF-IDF ---
  
  visual_docs <- vqa_context %>%
    group_by(model, context) %>%
    summarise(doc = paste(unlist(visual_tokens), collapse = " "), .groups = "drop")
  
  visual_tfidf <- visual_docs %>%
    group_by(model) %>%
    group_modify(function(data, grp) {
      it <- itoken(data$doc, progressbar = FALSE)
      vocab <- create_vocabulary(it) %>%
        prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.8)
      vectorizer <- vocab_vectorizer(vocab)
      dtm <- create_dtm(it, vectorizer)
      tfidf_model <- TfIdf$new()
      tfidf_mat <- fit_transform(dtm, tfidf_model)
      
      map_dfr(seq_len(nrow(data)), function(i) {
        scores <- as.numeric(tfidf_mat[i, ])
        names(scores) <- colnames(tfidf_mat)
        scores <- scores[scores > 0]
        tibble(
          context = data$context[i],
          term = names(scores),
          tfidf_score = as.numeric(scores)
        )
      })
    }) %>%
    ungroup()
  
  write_csv(
    visual_tfidf %>%
      group_by(model, context) %>%
      slice_max(tfidf_score, n = 20) %>%
      ungroup(),
    file.path(OUTPUT_DIR, "visual_tfidf_top20.csv")
  )
  
  # --- 12c. Visual CFS ---
  
  visual_cfs <- map_dfr(VQA_MODELS, function(m) {
    vt <- visual_tfidf %>% filter(model == m)
    vc <- vqa_context %>% filter(model == m) %>% rename(inserted_tokens = visual_tokens)
    compute_cfs(vt, vc, TAU_DEFAULT, K_DEFAULT) %>%
      mutate(model = m)
  })
  
  write_csv(visual_cfs, file.path(OUTPUT_DIR, "visual_cfs.csv"))
  
  # --- 12d. Pipeline decomposition ---
  
  cat("\n  Computing pipeline decomposition...\n")
  
  K_DECOMP <- 20
  
  decomposition <- map_dfr(VQA_MODELS, function(m) {
    text_top <- tfidf_all_terms %>%
      filter(model == m) %>%
      group_by(context) %>%
      slice_max(tfidf_score, n = K_DECOMP, with_ties = FALSE) %>%
      summarise(text_terms = list(term), .groups = "drop")
    
    visual_top <- visual_tfidf %>%
      filter(model == m) %>%
      group_by(context) %>%
      slice_max(tfidf_score, n = K_DECOMP, with_ties = FALSE) %>%
      summarise(visual_terms = list(term), .groups = "drop")
    
    text_top %>%
      inner_join(visual_top, by = "context") %>%
      rowwise() %>%
      mutate(
        propagated  = list(intersect(text_terms, visual_terms)),
        text_only   = list(setdiff(text_terms, visual_terms)),
        visual_only = list(setdiff(visual_terms, text_terms)),
        n_propagated  = length(propagated),
        n_text_only   = length(text_only),
        n_visual_only = length(visual_only),
        n_total       = n_propagated + n_text_only + n_visual_only,
        pct_propagated = n_propagated / n_total,
        model = m
      ) %>%
      ungroup()
  })
  
  decomp_summary <- decomposition %>%
    select(model, context, n_propagated, n_text_only, n_visual_only, n_total, pct_propagated)
  
  write_csv(decomp_summary, file.path(OUTPUT_DIR, "pipeline_decomposition_summary.csv"))
  
  decomp_detail <- decomposition %>%
    select(model, context, propagated, text_only, visual_only) %>%
    mutate(
      propagated  = map_chr(propagated, ~paste(.x, collapse = ", ")),
      text_only   = map_chr(text_only, ~paste(.x, collapse = ", ")),
      visual_only = map_chr(visual_only, ~paste(.x, collapse = ", "))
    )
  
  write_csv(decomp_detail, file.path(OUTPUT_DIR, "pipeline_decomposition_detail.csv"))
  
  cat("\n  Pipeline decomposition summary:\n")
  for (m in VQA_MODELS) {
    ds <- decomp_summary %>% filter(model == m)
    cat(sprintf("    [%s] Mean propagated: %.1f / %d terms (%.0f%%)\n",
                m, mean(ds$n_propagated), K_DECOMP, mean(ds$pct_propagated) * 100))
  }
  
  # --- 12e. Propagation rate plot (Figure 9 / Appendix J) ---
  
  propagation_rates <- decomp_summary %>%
    mutate(
      n_visual_total = n_propagated + n_visual_only,
      propagation_rate = ifelse(n_visual_total > 0, n_propagated / n_visual_total, 0)
    )
  
  write_csv(propagation_rates, file.path(OUTPUT_DIR, "propagation_rates.csv"))
  
  model_labels <- c("dalle" = "DALL\u00b7E 3", "imagen" = "Imagen",
                     "gptimage" = "GPT-Image")
  
  prop_stacked <- propagation_rates %>%
    mutate(
      pct_propagated = ifelse(n_visual_total > 0, n_propagated / n_visual_total, 0),
      pct_visual_only = ifelse(n_visual_total > 0, n_visual_only / n_visual_total, 0),
      model_label = factor(model_labels[model], levels = model_labels)
    )
  
  context_order <- prop_stacked %>%
    group_by(context) %>%
    summarise(mean_prop = mean(pct_propagated), .groups = "drop") %>%
    arrange(mean_prop) %>%
    pull(context)
  
  prop_long <- prop_stacked %>%
    select(model_label, context, pct_propagated, pct_visual_only) %>%
    pivot_longer(cols = starts_with("pct_"), names_to = "source", values_to = "fraction") %>%
    mutate(
      source = factor(source,
                      levels = c("pct_propagated", "pct_visual_only"),
                      labels = c("Propagated from rewriter", "Image model only")),
      context = factor(context, levels = context_order)
    )
  
  p_propagation_stacked <- prop_long %>%
    ggplot(aes(x = fraction, y = context, fill = source)) +
    geom_col(width = 0.75, alpha = 0.85) +
    scale_x_continuous(labels = scales::percent_format(), expand = c(0, 0)) +
    scale_fill_manual(values = c("Propagated from rewriter" = "#b2182b",
                                 "Image model only" = "#2166ac")) +
    facet_wrap(~model_label, ncol = 2) +
    labs(x = "Share of distinctive visual terms", y = NULL, fill = NULL) +
    PLOT_THEME +
    theme(legend.position = "top", panel.grid.major.y = element_blank())
  
  ggsave(file.path(OUTPUT_DIR, "propagation_stacked_by_context.pdf"),
         p_propagation_stacked, width = 14, height = 10)
  
  # --- 12f. Text vs visual CFS scatter (Appendix J) ---
  
  text_cfs_models <- results %>%
    filter(model %in% VQA_MODELS) %>%
    select(model, context, text_cfs = cfs)
  
  visual_cfs_plot <- visual_cfs %>%
    select(model, context, visual_cfs = cfs) %>%
    inner_join(text_cfs_models, by = c("model", "context"))
  
  p_text_visual_cfs <- visual_cfs_plot %>%
    ggplot(aes(x = text_cfs, y = visual_cfs)) +
    geom_point(alpha = 0.7, size = 2.5, colour = "#b2182b") +
    geom_text_repel(aes(label = context), size = 3, max.overlaps = 20, colour = "grey30") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
    facet_wrap(~model, ncol = 2) +
    labs(x = "Text-level CFS (from revised prompts)",
         y = "Visual-level CFS (from VQA descriptions)") +
    PLOT_THEME
  
  ggsave(file.path(OUTPUT_DIR, "scatter_text_vs_visual_cfs.pdf"),
         p_text_visual_cfs, width = 14, height = 7)
  
  # Propagation summary stats
  prop_summary <- prop_stacked %>%
    group_by(model_label) %>%
    summarise(
      median_prop = median(pct_propagated),
      mean_prop   = mean(pct_propagated),
      min_prop    = min(pct_propagated),
      max_prop    = max(pct_propagated),
      sd_prop     = sd(pct_propagated),
      n_contexts  = n(),
      n_majority_rewriter = sum(pct_propagated > 0.5),
      .groups = "drop"
    )
  
  write_csv(prop_summary, file.path(OUTPUT_DIR, "propagation_summary_by_model.csv"))
  
  cat("\n  Visual analysis complete.\n")
}


cat(sprintf("\n=== Done. All outputs in %s/ ===\n", OUTPUT_DIR))
