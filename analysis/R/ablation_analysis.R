# =============================================================================
# ABLATION: Isolating the Revision Layer Effect
#
# Compares VQA descriptions of images generated from ORIGINAL prompts vs
# REVISED prompts (from GPT-Image's revision layer), using open-source
# image models without built-in revision layers (SDXL, Flux 2 Dev).
#
# Inputs (in DATA_DIR):
#   vqa_descriptions_ablation.csv — prompt_id, prompt_type, prompt, model, description
#   prompts_sdrun.csv             — prompt_id + context metadata
#   geo_stopwords.txt             — geographic/artifact stopwords
#
# Analyses per image model (flux, stablediffusion):
#   0. VQA description length diagnostics
#   1. Paired CMS on VQA descriptions (cosine distance from baseline)
#   2. TF-IDF distinctive terms per context × prompt_type
#   3. CFS (cultural flattening score) on VQA descriptions
#   4. Term-level 2×2 decomposition
#   5. Per-term McNemar tests
#
# Requirements:
#   install.packages(c("tidyverse", "textstem", "tidytext",
#                      "text2vec", "patchwork", "stopwords"))
# =============================================================================

library(tidyverse)
library(textstem)
library(tidytext)
library(text2vec)

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   <- "data"
OUTPUT_DIR <- "output/ablation"

VQA_CSV     <- file.path(DATA_DIR, "vqa_descriptions_ablation.csv")
PROMPTS_CSV <- file.path(DATA_DIR, "prompts_sdrun.csv")
STOPWORDS_PATH <- file.path(DATA_DIR, "geo_stopwords.txt")

TAU_QUANTILE <- 0.75
K_TOP        <- 10
K_TFIDF_SHOW <- 20

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

PLOT_THEME <- theme_minimal(base_size = 13) +
  theme(
    strip.text       = element_text(face = "bold"),
    legend.position  = "top",
    panel.grid.minor = element_blank()
  )

PROMPT_TYPE_COLS <- c(original = "#377EB8", revised = "#E41A1C")


# =============================================================================
# LOAD GEO STOPLIST FROM EXTERNAL FILE
# =============================================================================

geo_stops <- readLines(STOPWORDS_PATH) %>%
  str_extract_all('"([^"]+)"') %>%
  unlist() %>%
  str_remove_all('"') %>%
  str_trim() %>%
  tolower() %>%
  unique()

cat(sprintf("Loaded %d geographic/artifact stopwords\n", length(geo_stops)))


# =============================================================================
# LOAD & JOIN DATA
# =============================================================================

cat("Loading VQA data from:", VQA_CSV, "\n")
vqa_raw <- read_csv(VQA_CSV, show_col_types = FALSE)
cat("  VQA rows:", nrow(vqa_raw), "\n")
cat("  Models:", paste(unique(vqa_raw$model), collapse = ", "), "\n")
cat("  Prompt types:", paste(unique(vqa_raw$prompt_type), collapse = ", "), "\n")

cat("\nLoading prompts metadata from:", PROMPTS_CSV, "\n")
prompts <- read_csv(PROMPTS_CSV, show_col_types = FALSE)

prompts_meta <- prompts %>%
  select(prompt_id, context, type, is_baseline, base_prompt_id,
         original_prompt = original_clean, revised_prompt = revised_clean) %>%
  distinct() %>%
  mutate(context = if_else(is.na(context) & is_baseline == TRUE, "baseline", context))

# Join context onto VQA data
vqa <- vqa_raw %>%
  left_join(prompts_meta %>% select(prompt_id, context, is_baseline, base_prompt_id),
            by = "prompt_id")

n_missing <- sum(is.na(vqa$context))
if (n_missing > 0) {
  cat("WARNING:", n_missing, "VQA rows could not be joined to a context\n")
} else {
  cat("All VQA rows successfully joined to context.\n")
}

# Identify baseline context
baseline_vals <- unique(vqa$context[vqa$is_baseline == TRUE | vqa$is_baseline == 1])
if (length(baseline_vals) == 0) {
  baseline_vals <- unique(vqa$context[grepl("baseline|cleanup|clean.?up", vqa$context, ignore.case = TRUE)])
}
BASELINE_CONTEXT <- baseline_vals[1]
cat("Baseline context:", BASELINE_CONTEXT, "\n")


# =============================================================================
# ANALYSIS FUNCTION (runs for each image model)
# =============================================================================

run_ablation_analysis <- function(vqa_model, model_name, output_subdir) {
  
  dir.create(output_subdir, showWarnings = FALSE, recursive = TRUE)
  cat("\n\n================================================================\n")
  cat("ANALYSING MODEL:", model_name, "\n")
  cat("================================================================\n")
  
  # --- 0. Description length diagnostics ---
  
  cat("\n=== Section 0: Description Length ===\n")
  
  vqa_model <- vqa_model %>%
    mutate(desc_nwords = str_count(description, "\\S+"))
  
  length_stats <- vqa_model %>%
    group_by(context, prompt_type) %>%
    summarise(mean_words = mean(desc_nwords, na.rm = TRUE),
              sd_words   = sd(desc_nwords, na.rm = TRUE),
              median_words = median(desc_nwords, na.rm = TRUE),
              n = n(), .groups = "drop")
  write_csv(length_stats, file.path(output_subdir, "desc_length_stats.csv"))
  
  length_paired <- vqa_model %>%
    select(prompt_id, context, prompt_type, desc_nwords) %>%
    pivot_wider(names_from = prompt_type, values_from = desc_nwords) %>%
    filter(!is.na(original), !is.na(revised))
  
  if (nrow(length_paired) > 0) {
    length_test <- wilcox.test(length_paired$revised, length_paired$original,
                               paired = TRUE, alternative = "two.sided", exact = FALSE)
    cat("  Wilcoxon: V =", length_test$statistic,
        " p =", format.pval(length_test$p.value),
        " median diff =", median(length_paired$revised - length_paired$original, na.rm = TRUE), "\n")
  }
  
  
  # --- Tokenize ---
  
  vqa_tokens <- vqa_model %>%
    mutate(desc_clean = description %>%
             str_to_lower() %>%
             lemmatize_strings() %>%
             str_remove_all("[^a-z0-9\\s]") %>%
             str_squish()) %>%
    unnest_tokens(word, desc_clean) %>%
    anti_join(stop_words, by = "word") %>%
    filter(!word %in% geo_stops, nchar(word) > 2)
  
  
  # --- 1. Paired CMS ---
  
  cat("\n=== Section 1: Paired CMS ===\n")
  
  doc_texts <- vqa_tokens %>%
    group_by(prompt_id, context, prompt_type) %>%
    summarise(text = paste(word, collapse = " "), .groups = "drop") %>%
    mutate(doc_id = paste(prompt_id, context, prompt_type, sep = "__"))
  
  it <- itoken(doc_texts$text, ids = doc_texts$doc_id, progressbar = FALSE)
  vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min = 3)
  vectorizer <- vocab_vectorizer(vocab)
  it2 <- itoken(doc_texts$text, ids = doc_texts$doc_id, progressbar = FALSE)
  dtm <- create_dtm(it2, vectorizer)
  tfidf_model <- TfIdf$new()
  dtm_tfidf <- fit_transform(dtm, tfidf_model)
  
  doc_meta <- tibble(doc_id = rownames(dtm_tfidf)) %>%
    separate(doc_id, into = c("prompt_id", "context", "prompt_type"),
             sep = "__", remove = FALSE) %>%
    mutate(prompt_id = as.numeric(prompt_id))
  
  baseline_docs <- doc_meta %>% filter(context == BASELINE_CONTEXT)
  context_docs  <- doc_meta %>% filter(context != BASELINE_CONTEXT)
  
  vqa_with_base <- vqa_model %>%
    select(prompt_id, base_prompt_id) %>% distinct()
  
  cms_results <- context_docs %>%
    left_join(vqa_with_base, by = "prompt_id") %>%
    inner_join(
      baseline_docs %>%
        left_join(vqa_with_base, by = "prompt_id") %>%
        select(base_prompt_id, prompt_type, baseline_doc_id = doc_id),
      by = c("base_prompt_id", "prompt_type")
    ) %>%
    filter(doc_id %in% rownames(dtm_tfidf),
           baseline_doc_id %in% rownames(dtm_tfidf)) %>%
    rowwise() %>%
    mutate(
      cos_sim = {
        v1 <- dtm_tfidf[doc_id, , drop = FALSE]
        v2 <- dtm_tfidf[baseline_doc_id, , drop = FALSE]
        n1 <- sqrt(sum(v1^2)); n2 <- sqrt(sum(v2^2))
        if (n1 == 0 | n2 == 0) NA_real_ else as.numeric(sum(v1 * v2) / (n1 * n2))
      },
      cms = 1 - cos_sim
    ) %>%
    ungroup()
  
  if (nrow(cms_results) == 0) {
    cat("  No CMS pairs found via base_prompt_id; trying prompt_id fallback...\n")
    cms_results <- context_docs %>%
      inner_join(
        baseline_docs %>% select(prompt_id, prompt_type, baseline_doc_id = doc_id),
        by = c("prompt_id", "prompt_type")
      ) %>%
      filter(doc_id %in% rownames(dtm_tfidf),
             baseline_doc_id %in% rownames(dtm_tfidf)) %>%
      rowwise() %>%
      mutate(
        cos_sim = {
          v1 <- dtm_tfidf[doc_id, , drop = FALSE]
          v2 <- dtm_tfidf[baseline_doc_id, , drop = FALSE]
          n1 <- sqrt(sum(v1^2)); n2 <- sqrt(sum(v2^2))
          if (n1 == 0 | n2 == 0) NA_real_ else as.numeric(sum(v1 * v2) / (n1 * n2))
        },
        cms = 1 - cos_sim
      ) %>%
      ungroup()
  }
  
  cms_summary <- cms_results %>%
    group_by(context, prompt_type) %>%
    summarise(mean_cms = mean(cms, na.rm = TRUE),
              sd_cms = sd(cms, na.rm = TRUE),
              median_cms = median(cms, na.rm = TRUE),
              n = n(), .groups = "drop")
  write_csv(cms_summary, file.path(output_subdir, "cms_vqa_summary.csv"))
  
  # Paired Wilcoxon per context
  cms_paired <- cms_results %>%
    select(prompt_id, context, prompt_type, cms) %>%
    pivot_wider(names_from = prompt_type, values_from = cms) %>%
    filter(!is.na(original), !is.na(revised))
  
  if (nrow(cms_paired) > 0) {
    paired_tests <- cms_paired %>%
      group_by(context) %>%
      summarise(
        n_pairs = n(),
        median_original = median(original, na.rm = TRUE),
        median_revised  = median(revised, na.rm = TRUE),
        median_diff     = median(revised - original, na.rm = TRUE),
        wilcox_V = tryCatch(
          wilcox.test(revised, original, paired = TRUE, alternative = "greater", exact = FALSE)$statistic,
          error = function(e) NA_real_
        ),
        wilcox_p = tryCatch(
          wilcox.test(revised, original, paired = TRUE, alternative = "greater", exact = FALSE)$p.value,
          error = function(e) NA_real_
        ),
        .groups = "drop"
      ) %>%
      mutate(p_adj = p.adjust(wilcox_p, method = "holm"),
             sig = case_when(p_adj < 0.001 ~ "***", p_adj < 0.01 ~ "**",
                             p_adj < 0.05 ~ "*", TRUE ~ "ns"))
    
    cat("\nPaired Wilcoxon tests (revised > original?):\n")
    print(paired_tests)
    write_csv(paired_tests, file.path(output_subdir, "cms_paired_tests.csv"))
    
    global_test <- wilcox.test(cms_paired$revised, cms_paired$original,
                               paired = TRUE, alternative = "greater", exact = FALSE)
    cat("  Global: V =", global_test$statistic,
        " p =", format.pval(global_test$p.value), "\n")
  }
  
  
  # --- 2. TF-IDF distinctive terms ---
  
  cat("\n=== Section 2: TF-IDF Terms ===\n")
  
  tfidf_docs <- vqa_tokens %>%
    filter(context != BASELINE_CONTEXT) %>%
    mutate(condition = paste(context, prompt_type, sep = "::")) %>%
    count(condition, word) %>%
    bind_tf_idf(word, condition, n)
  
  top_terms <- tfidf_docs %>%
    group_by(condition) %>%
    slice_max(tf_idf, n = K_TFIDF_SHOW, with_ties = FALSE) %>%
    ungroup() %>%
    separate(condition, into = c("context", "prompt_type"), sep = "::", remove = FALSE)
  
  write_csv(top_terms, file.path(output_subdir, "tfidf_terms_by_condition.csv"))
  
  revised_terms  <- top_terms %>% filter(prompt_type == "revised") %>%
    select(context, word) %>% mutate(in_revised = TRUE)
  original_terms <- top_terms %>% filter(prompt_type == "original") %>%
    select(context, word) %>% mutate(in_original = TRUE)
  
  term_comparison <- full_join(revised_terms, original_terms, by = c("context", "word")) %>%
    replace_na(list(in_revised = FALSE, in_original = FALSE))
  
  revised_only  <- term_comparison %>% filter(in_revised & !in_original)
  original_only <- term_comparison %>% filter(!in_revised & in_original)
  
  write_csv(revised_only, file.path(output_subdir, "terms_revised_only.csv"))
  write_csv(original_only, file.path(output_subdir, "terms_original_only.csv"))
  
  
  # --- 3. CFS ---
  
  cat("\n=== Section 3: CFS ===\n")
  
  compute_cfs <- function(tokens_df, tau_q = TAU_QUANTILE, k = K_TOP) {
    context_tfidf <- tokens_df %>%
      count(context, word) %>%
      bind_tf_idf(word, context, n)
    
    map_dfr(unique(tokens_df$context), function(ctx) {
      ctx_data <- context_tfidf %>% filter(context == ctx)
      if (nrow(ctx_data) == 0) return(NULL)
      tau <- quantile(ctx_data$tf_idf, tau_q, na.rm = TRUE)
      top_k <- ctx_data %>% slice_max(tf_idf, n = k, with_ties = FALSE)
      prevalence <- mean(top_k$tf_idf >= tau)
      top_k_terms <- top_k$word
      prompts_with_term <- tokens_df %>%
        filter(context == ctx, word %in% top_k_terms) %>%
        n_distinct(.$prompt_id)
      total_prompts <- n_distinct(tokens_df$prompt_id[tokens_df$context == ctx])
      spread <- prompts_with_term / max(total_prompts, 1)
      tibble(context = ctx, prevalence = prevalence, spread = spread,
             cfs = (prevalence + spread) / 2)
    })
  }
  
  cfs_by_type <- vqa_tokens %>%
    filter(context != BASELINE_CONTEXT) %>%
    group_split(prompt_type) %>%
    map_dfr(function(df) {
      compute_cfs(df) %>% mutate(prompt_type = unique(df$prompt_type))
    })
  
  write_csv(cfs_by_type, file.path(output_subdir, "cfs_vqa_by_type.csv"))
  
  cfs_paired <- cfs_by_type %>%
    select(context, prompt_type, cfs) %>%
    pivot_wider(names_from = prompt_type, values_from = cfs)
  
  if (all(c("original", "revised") %in% names(cfs_paired))) {
    cfs_diff <- cfs_paired %>%
      mutate(diff = revised - original,
             pct_change = (revised - original) / pmax(original, 0.001) * 100)
    write_csv(cfs_diff, file.path(output_subdir, "cfs_paired_diff.csv"))
    cat("  Mean CFS original:", mean(cfs_paired$original, na.rm = TRUE),
        "  Mean CFS revised:", mean(cfs_paired$revised, na.rm = TRUE), "\n")
  }
  
  
  # --- 4. Term decomposition ---
  
  cat("\n=== Section 4: Term Decomposition ===\n")
  
  decomp <- term_comparison %>%
    mutate(category = case_when(
      in_revised & in_original  ~ "both (image-model bias)",
      in_revised & !in_original ~ "revised-only (revision-layer-driven)",
      !in_revised & in_original ~ "original-only (image-model specific)",
      TRUE ~ "neither"
    ))
  
  decomp_summary <- decomp %>%
    count(context, category) %>%
    pivot_wider(names_from = category, values_from = n, values_fill = 0)
  write_csv(decomp_summary, file.path(output_subdir, "term_decomposition_summary.csv"))
  write_csv(decomp, file.path(output_subdir, "term_decomposition_detail.csv"))
  
  # Plot
  p_decomp <- decomp %>%
    count(context, category) %>%
    mutate(category = factor(category,
                             levels = c("revised-only (revision-layer-driven)",
                                        "both (image-model bias)",
                                        "original-only (image-model specific)"))) %>%
    filter(!is.na(category)) %>%
    ggplot(aes(x = n, y = reorder(context, n, sum), fill = category)) +
    geom_col(alpha = 0.85) +
    scale_fill_manual(values = c(
      "revised-only (revision-layer-driven)" = "#E41A1C",
      "both (image-model bias)"              = "#984EA3",
      "original-only (image-model specific)" = "#377EB8"
    )) +
    labs(x = paste0("Number of distinctive terms (top-", K_TFIDF_SHOW, ")"),
         y = NULL, fill = NULL, caption = model_name) +
    PLOT_THEME +
    theme(legend.direction = "vertical")
  ggsave(file.path(output_subdir, "term_decomposition.pdf"), p_decomp, width = 9, height = 5)
  
  
  # --- 5. McNemar tests ---
  
  cat("\n=== Section 5: McNemar Tests ===\n")
  
  all_top_words <- term_comparison %>%
    filter(context != BASELINE_CONTEXT) %>%
    select(context, word)
  
  presence_wide <- vqa_tokens %>%
    filter(context != BASELINE_CONTEXT) %>%
    semi_join(all_top_words, by = c("context", "word")) %>%
    distinct(prompt_id, context, prompt_type, word) %>%
    mutate(present = 1L) %>%
    pivot_wider(names_from = prompt_type, values_from = present, values_fill = 0L)
  
  all_combos <- expand_grid(
    vqa_model %>% filter(context != BASELINE_CONTEXT) %>%
      distinct(prompt_id, context),
    word = unique(all_top_words$word)
  ) %>%
    semi_join(all_top_words, by = c("context", "word"))
  
  presence_full <- all_combos %>%
    left_join(presence_wide, by = c("prompt_id", "context", "word")) %>%
    replace_na(list(original = 0L, revised = 0L))
  
  mcnemar_results <- presence_full %>%
    group_by(context, word) %>%
    summarise(
      n_both      = sum(original == 1 & revised == 1),
      n_rev_only  = sum(original == 0 & revised == 1),
      n_orig_only = sum(original == 1 & revised == 0),
      n_neither   = sum(original == 0 & revised == 0),
      n_total     = n(),
      pct_original = mean(original) * 100,
      pct_revised  = mean(revised) * 100,
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      mcnemar_p = {
        b <- n_rev_only; c_val <- n_orig_only
        if ((b + c_val) < 5) NA_real_
        else tryCatch(
          mcnemar.test(matrix(c(n_both, b, c_val, n_neither), nrow = 2),
                       correct = TRUE)$p.value,
          error = function(e) NA_real_
        )
      },
      direction = case_when(
        pct_revised > pct_original ~ "revision amplifies",
        pct_revised < pct_original ~ "revision suppresses",
        TRUE ~ "no difference"
      )
    ) %>%
    ungroup() %>%
    mutate(mcnemar_p_adj = p.adjust(mcnemar_p, method = "holm"))
  
  write_csv(mcnemar_results, file.path(output_subdir, "mcnemar_term_tests.csv"))
  
  sig_terms <- mcnemar_results %>% filter(mcnemar_p_adj < 0.05)
  cat("  Significant terms:", nrow(sig_terms), "\n")
  cat("  Amplified:", sum(sig_terms$direction == "revision amplifies"),
      " Suppressed:", sum(sig_terms$direction == "revision suppresses"), "\n")
  
  cat("\n  ", model_name, "analysis complete. Outputs in:", output_subdir, "\n")
}


# =============================================================================
# RUN FOR EACH IMAGE MODEL
# =============================================================================

models <- unique(vqa$model)
cat("\nImage models found:", paste(models, collapse = ", "), "\n")

for (m in models) {
  vqa_subset <- vqa %>% filter(model == m)
  model_label <- case_when(
    grepl("flux", m, ignore.case = TRUE) ~ "Flux 2 Dev",
    grepl("stable|sdxl|sd", m, ignore.case = TRUE) ~ "SDXL",
    TRUE ~ m
  )
  run_ablation_analysis(vqa_subset, model_label,
                        file.path(OUTPUT_DIR, tolower(gsub("[^a-zA-Z0-9]", "_", m))))
}


# =============================================================================
# CROSS-MODEL COMPARISON
# =============================================================================

cat("\n=== Cross-model comparison ===\n")

for (pattern in c("cms_paired_tests.csv", "cfs_paired_diff.csv", "term_decomposition_summary.csv")) {
  files <- list.files(OUTPUT_DIR, pattern = pattern, recursive = TRUE, full.names = TRUE)
  if (length(files) > 1) {
    combined <- map_dfr(files, function(f) {
      read_csv(f, show_col_types = FALSE) %>%
        mutate(model = basename(dirname(f)))
    })
    out_name <- paste0(tools::file_path_sans_ext(pattern), "_all_models.csv")
    write_csv(combined, file.path(OUTPUT_DIR, out_name))
    cat("  Saved:", out_name, "\n")
  }
}


# =============================================================================
# COMBINED FIGURE (Figure 4)
# =============================================================================

library(patchwork)

CATEGORY_COLS <- c(
  "revised-only (revision-layer-driven)" = "#E41A1C",
  "both (image-model bias)"              = "#984EA3",
  "original-only (image-model specific)" = "#377EB8"
)
CATEGORY_LEVELS <- names(CATEGORY_COLS)

decomp_files <- list.files(OUTPUT_DIR, pattern = "term_decomposition_detail.csv",
                           recursive = TRUE, full.names = TRUE)

if (length(decomp_files) >= 2) {
  decomp_all <- map_dfr(decomp_files, function(f) {
    read_csv(f, show_col_types = FALSE) %>%
      mutate(model = case_when(
        grepl("flux", dirname(f), ignore.case = TRUE) ~ "Flux 2 Dev",
        grepl("stable|sdxl", dirname(f), ignore.case = TRUE) ~ "SDXL",
        TRUE ~ basename(dirname(f))
      ))
  }) %>%
    mutate(category = factor(category, levels = CATEGORY_LEVELS))
  
  make_panel <- function(df, title) {
    df %>%
      count(context, category) %>%
      filter(!is.na(category)) %>%
      ggplot(aes(x = n, y = reorder(context, n, sum), fill = category)) +
      geom_col(alpha = 0.85) +
      scale_fill_manual(values = CATEGORY_COLS) +
      labs(x = "Number of distinctive terms (top-20)",
           y = NULL, fill = NULL, title = title) +
      PLOT_THEME
  }
  
  p_sdxl <- make_panel(decomp_all %>% filter(model == "SDXL"), "SDXL")
  p_flux <- make_panel(decomp_all %>% filter(model == "Flux 2 Dev"), "Flux 2 Dev")
  
  p_combined <- p_sdxl + p_flux +
    plot_layout(ncol = 2, guides = "collect") &
    theme(legend.position = "bottom", legend.direction = "vertical")
  
  ggsave(file.path(OUTPUT_DIR, "term_decomposition_combined.pdf"),
         p_combined, width = 12, height = 5)
  cat("  Combined decomposition figure saved.\n")
}

cat("\n=== All ablation analyses complete ===\n")
cat("Outputs in:", OUTPUT_DIR, "\n")
