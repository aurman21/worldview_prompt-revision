# =============================================================================
# Guardrailing Asymmetry Analysis (Appendix C, Figure 5)
#
# Computes and visualises image generation refusal rates by model,
# prompt category, and language-context combination.
#
# Input (in DATA_DIR):
#   df_unified_ALL.csv — unified dataset with columns:
#     model_type (dalle/gptimage/imagen), category, lang_country
#
# Output:
#   refusal_heatmap.pdf — full heatmap (all models)
#   refusal_heatmap_filtered.pdf — filtered (non-zero categories, excl. Imagen)
#
# Requirements:
#   install.packages(c("tidyverse"))
# =============================================================================

library(tidyverse)

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   <- "data"
OUTPUT_DIR <- "output"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# LOAD DATA
# =============================================================================

df_unified <- read_csv(file.path(DATA_DIR, "df_unified_ALL.csv"), show_col_types = FALSE)

# Expected image counts per prompt per model
# (DALL-E 3: up to 5 images × 20 prompts = 100; Imagen/GPT-Image: 1 × 20 = 20)
expected <- c(imagen = 20, gptimage = 20, dalle = 100)

# =============================================================================
# COMPUTE REFUSAL RATES
# =============================================================================

df_counts <- df_unified %>%
  count(model_type, category, lang_country) %>%
  mutate(
    expected_n   = expected[model_type],
    n_refusals   = expected_n - n,
    pct_refusals = (n_refusals / expected_n) * 100
  )

# Sanity check
overcount <- df_counts %>% filter(n > expected_n)
if (nrow(overcount) > 0) {
  cat("WARNING: some combinations exceed expected count:\n")
  print(overcount)
}

# Summary
df_refusal_summary <- df_counts %>%
  group_by(model_type, lang_country) %>%
  summarise(
    mean_pct_refusal = mean(pct_refusals),
    total_generated  = sum(n),
    total_expected   = sum(expected_n),
    total_refusals   = sum(n_refusals),
    .groups = "drop"
  ) %>%
  mutate(overall_pct_refusal = (total_refusals / total_expected) * 100)

cat("Refusal summary:\n")
print(df_refusal_summary, n = 50)


# =============================================================================
# FULL HEATMAP (Figure 5)
# =============================================================================

ggplot(df_counts, aes(x = lang_country, y = category, fill = pct_refusals)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.0f%%", pct_refusals)), size = 2.5) +
  facet_wrap(~ model_type, ncol = 1) +
  scale_fill_gradient(low = "white", high = "firebrick", name = "% Refused") +
  labs(
    title = "Image Generation Refusal Rates",
    subtitle = "By model, category, and language-context",
    x = "Language / Context", y = "Category"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text  = element_text(face = "bold")
  )

ggsave(file.path(OUTPUT_DIR, "refusal_heatmap.pdf"), width = 16, height = 12)


# =============================================================================
# FILTERED HEATMAP (non-zero categories, excluding Imagen)
# =============================================================================

cats_with_refusals <- df_counts %>%
  filter(pct_refusals > 0) %>%
  pull(category) %>%
  unique()

df_counts_filtered <- df_counts %>%
  filter(model_type != "imagen", category %in% cats_with_refusals)

ggplot(df_counts_filtered, aes(x = lang_country, y = category, fill = pct_refusals)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(pct_refusals > 0, sprintf("%.0f%%", pct_refusals), "")),
            size = 2.8) +
  facet_wrap(~ model_type, ncol = 1) +
  scale_fill_gradient(low = "white", high = "firebrick", name = "% Refused") +
  labs(
    title = "Image Generation Refusal Rates",
    subtitle = "Categories with at least 1 refusal | DALL-E 3 & GPT-Image (Imagen: 0% throughout)",
    x = "Language / Country", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text  = element_text(face = "bold", size = 12),
    panel.grid  = element_blank()
  )

ggsave(file.path(OUTPUT_DIR, "refusal_heatmap_filtered.pdf"), width = 16, height = 8)

cat("\nRefusal analysis complete.\n")
