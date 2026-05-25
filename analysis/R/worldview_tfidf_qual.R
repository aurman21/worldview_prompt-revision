# =============================================================================
# Step 3: Qualitative TF-IDF Exploration (Stereotypical Content Analysis)
#
# Generates LaTeX tables and category annotations from TF-IDF terms
# computed by worldview_full_pipeline.R.
#
# Inputs (in OUTPUT_DIR, produced by the main pipeline):
#   tfidf_top20_terms.csv
#   cms_cfs_scores.csv
#   pipeline_decomposition_detail.csv (optional)
#
# Outputs:
#   tfidf_main_table.tex         — main text table (top-5, high/low CFS)
#   tfidf_appendix_table.tex     — appendix table (top-10, all contexts)
#   tfidf_main_compact.csv
#   tfidf_appendix_full.csv
#   tfidf_appendix_condensed.csv
#   tfidf_category_counts.csv
#   pipeline_decomposition_table.tex (if visual analysis was run)
#
# Requirements:
#   install.packages("tidyverse")
# =============================================================================

library(tidyverse)

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR <- "output"

# =============================================================================
# 1. LOAD DATA
# =============================================================================

tfidf   <- read_csv(file.path(OUTPUT_DIR, "tfidf_top20_terms.csv"), show_col_types = FALSE)
results <- read_csv(file.path(OUTPUT_DIR, "cms_cfs_scores.csv"), show_col_types = FALSE)

# =============================================================================
# 2. SELECT CONTEXTS FOR MAIN TEXT TABLE
# =============================================================================

context_ranks <- results %>%
  group_by(context) %>%
  summarise(mean_cfs = mean(cfs, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_cfs))

top_contexts    <- context_ranks %>% slice_head(n = 3) %>% pull(context)
bottom_contexts <- context_ranks %>% slice_tail(n = 3) %>% pull(context)
selected <- c(top_contexts, bottom_contexts)

cat("Selected contexts for main text:\n")
cat("  High CFS:", paste(top_contexts, collapse = ", "), "\n")
cat("  Low CFS:", paste(bottom_contexts, collapse = ", "), "\n")

# =============================================================================
# 3. MAIN TEXT TABLE: top-5 per selected context
# =============================================================================

main_table <- tfidf %>%
  filter(context %in% selected) %>%
  group_by(context, model) %>%
  slice_max(tfidf_score, n = 5, with_ties = FALSE) %>%
  mutate(rank = row_number()) %>%
  ungroup() %>%
  select(context, model, rank, term, tfidf_score)

main_compact <- main_table %>%
  select(context, model, rank, term) %>%
  pivot_wider(
    id_cols = c(context, rank),
    names_from = model,
    values_from = term
  ) %>%
  arrange(context, rank)

write_csv(main_compact, file.path(OUTPUT_DIR, "tfidf_main_compact.csv"))

cat("\n=== Main Text Table (compact) ===\n")
main_compact %>%
  group_by(context) %>%
  group_walk(function(data, key) {
    cat(sprintf("\n%s:\n", key$context))
    for (i in seq_len(nrow(data))) {
      cols <- data[i, ] %>% select(-context, -rank) %>% as.character()
      cat(sprintf("  %d. %s\n", data$rank[i], paste(cols, collapse = " | ")))
    }
  })


# =============================================================================
# 4. GENERATE LATEX FOR MAIN TEXT TABLE
# =============================================================================

latex_lines <- c(
  "\\begin{table*}[t]",
  "\\centering",
  "\\small",
  "\\begin{tabular}{llccc}",
  "\\toprule",
  "Context & \\# & DALL\\textperiodcentered E~3 & GPT-Image & Imagen \\\\",
  "\\midrule"
)

for (ctx in selected) {
  rows <- main_compact %>% filter(context == ctx)
  for (i in seq_len(nrow(rows))) {
    prefix <- if (i == 1) str_replace_all(ctx, "_", "\\\\_") else ""
    dalle_val   <- if ("dalle" %in% names(rows))    rows$dalle[i]    else "--"
    gptimg_val  <- if ("gptimage" %in% names(rows)) rows$gptimage[i] else "--"
    imagen_val  <- if ("imagen" %in% names(rows))   rows$imagen[i]   else "--"
    line <- sprintf("%s & %d & %s & %s & %s \\\\",
                    prefix, rows$rank[i],
                    dalle_val %||% "--",
                    gptimg_val %||% "--",
                    imagen_val %||% "--")
    latex_lines <- c(latex_lines, line)
  }
  if (ctx != tail(selected, 1)) {
    latex_lines <- c(latex_lines, "\\midrule")
  }
}

latex_lines <- c(latex_lines,
                 "\\bottomrule",
                 "\\end{tabular}",
                 "\\caption{Top-5 TF-IDF distinctive terms for the three highest-",
                 "and three lowest-CFS contexts.}",
                 "\\label{tab:tfidf_main}",
                 "\\end{table*}")

writeLines(latex_lines, file.path(OUTPUT_DIR, "tfidf_main_table.tex"))
cat("\nLaTeX table saved to tfidf_main_table.tex\n")


# =============================================================================
# 5. APPENDIX: FULL TOP-10 FOR ALL CONTEXTS
# =============================================================================

appendix_table <- tfidf %>%
  group_by(context, model) %>%
  slice_max(tfidf_score, n = 10, with_ties = FALSE) %>%
  mutate(rank = row_number()) %>%
  ungroup() %>%
  select(context, model, rank, term, tfidf_score) %>%
  arrange(context, model, rank)

write_csv(appendix_table, file.path(OUTPUT_DIR, "tfidf_appendix_full.csv"))

appendix_condensed <- appendix_table %>%
  group_by(context, model) %>%
  summarise(top_10_terms = paste(term, collapse = ", "), .groups = "drop") %>%
  pivot_wider(id_cols = context, names_from = model, values_from = top_10_terms)

write_csv(appendix_condensed, file.path(OUTPUT_DIR, "tfidf_appendix_condensed.csv"))

# LaTeX for appendix
app_latex <- c(
  "\\begin{table*}[h]",
  "\\centering",
  "\\scriptsize",
  "\\begin{tabular}{lp{4cm}p{4cm}p{4cm}}",
  "\\toprule",
  "Context & DALL\\textperiodcentered E~3 & GPT-Image & Imagen \\\\",
  "\\midrule"
)

for (ctx in sort(unique(appendix_condensed$context))) {
  row <- appendix_condensed %>% filter(context == ctx)
  dalle_val   <- if ("dalle" %in% names(row))    row$dalle    else "--"
  gptimg_val  <- if ("gptimage" %in% names(row)) row$gptimage else "--"
  imagen_val  <- if ("imagen" %in% names(row))   row$imagen   else "--"
  line <- sprintf("%s & %s & %s & %s \\\\",
                  str_replace_all(ctx, "_", "\\\\_"),
                  dalle_val %||% "--",
                  gptimg_val %||% "--",
                  imagen_val %||% "--")
  app_latex <- c(app_latex, line)
}

app_latex <- c(app_latex,
               "\\bottomrule",
               "\\end{tabular}",
               "\\caption{Top-10 TF-IDF distinctive terms for all contexts across",
               "three models. Terms are ordered by TF-IDF score (highest first).}",
               "\\label{tab:tfidf_appendix}",
               "\\end{table*}")

writeLines(app_latex, file.path(OUTPUT_DIR, "tfidf_appendix_table.tex"))
cat("Appendix LaTeX table saved to tfidf_appendix_table.tex\n")


# =============================================================================
# 6. CATEGORY ANALYSIS
# =============================================================================

demographic_terms  <- c("woman", "women", "man", "men", "girl", "boy",
                        "child", "children", "elder", "elderly", "young",
                        "diverse", "multicultural", "caucasian", "hispanic",
                        "asian", "african", "black", "white", "skin")
temporal_terms     <- c("ancient", "traditional", "historic", "historical",
                        "colonial", "old", "heritage", "modern", "contemporary",
                        "urban", "futuristic")
exotic_terms       <- c("vibrant", "colorful", "colourful", "exotic", "mystical",
                        "magical", "bustling", "lively", "rustic", "quaint",
                        "serene", "tranquil", "picturesque")
nature_terms       <- c("snow", "snowy", "winter", "ice", "frozen", "freeze",
                        "pine", "forest", "lake", "mountain", "desert", "sand",
                        "tropical", "palm", "jungle", "ocean", "river")
religion_terms     <- c("mosque", "minaret", "church", "temple", "pagoda",
                        "cathedral", "buddhist", "islamic", "christian",
                        "hindu", "prayer", "religious", "spiritual", "hijab")
architecture_terms <- c("pyramid", "sphinx", "tower", "castle", "palace",
                        "cathedral", "dome", "minaret", "pagoda", "shrine",
                        "cobblestone", "medieval", "baroque", "gothic")

category_counts <- appendix_table %>%
  mutate(
    is_demographic  = term %in% demographic_terms,
    is_temporal     = term %in% temporal_terms,
    is_exotic       = term %in% exotic_terms,
    is_nature       = term %in% nature_terms,
    is_religion     = term %in% religion_terms,
    is_architecture = term %in% architecture_terms
  ) %>%
  group_by(context, model) %>%
  summarise(
    n_demographic  = sum(is_demographic),
    n_temporal     = sum(is_temporal),
    n_exotic       = sum(is_exotic),
    n_nature       = sum(is_nature),
    n_religion     = sum(is_religion),
    n_architecture = sum(is_architecture),
    .groups = "drop"
  )

write_csv(category_counts, file.path(OUTPUT_DIR, "tfidf_category_counts.csv"))


# =============================================================================
# 7. PIPELINE DECOMPOSITION TABLE (if visual analysis was run)
# =============================================================================

decomp_path <- file.path(OUTPUT_DIR, "pipeline_decomposition_detail.csv")

if (file.exists(decomp_path)) {
  cat("\n=== Pipeline Decomposition (LaTeX) ===\n")
  
  decomp <- read_csv(decomp_path, show_col_types = FALSE)
  
  decomp_selected <- decomp %>% filter(context %in% top_contexts)
  
  decomp_latex <- c(
    "\\begin{table*}[t]",
    "\\centering",
    "\\small",
    "\\begin{tabular}{lp{3.5cm}p{3.5cm}p{3.5cm}}",
    "\\toprule",
    "Context & Propagated & Text-only & Visual-only \\\\",
    "\\midrule"
  )
  
  for (ctx in top_contexts) {
    row <- decomp_selected %>% filter(context == ctx)
    if (nrow(row) == 0) next
    line <- sprintf("%s & %s & %s & %s \\\\",
                    str_replace_all(ctx, "_", "\\\\_"),
                    row$propagated[1], row$text_only[1], row$visual_only[1])
    decomp_latex <- c(decomp_latex, line)
  }
  
  decomp_latex <- c(decomp_latex,
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\caption{Pipeline decomposition for the three highest-CFS contexts.}",
                    "\\label{tab:pipeline_decomp}",
                    "\\end{table*}")
  
  writeLines(decomp_latex, file.path(OUTPUT_DIR, "pipeline_decomposition_table.tex"))
  cat("Pipeline decomposition LaTeX saved.\n")
} else {
  cat("\nNo pipeline decomposition found (run visual analysis first).\n")
}

cat(sprintf("\nAll outputs in %s/\n", OUTPUT_DIR))
