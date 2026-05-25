# =============================================================================
# GPT-Image Non-English Revised Prompts Analysis (Appendix D, Figure 6)
#
# Analyses the share of GPT-Image revised prompts returned in a non-English
# language, by language-country combination and prompt category.
#
# Inputs (in DATA_DIR):
#   openai-revised-apr26-en-fin.csv — GPT-Image revised prompts with
#     detected response language (ID, revised_prompt, language, revised_prompt_en)
#   ID_List.csv — prompt metadata (ID, prompt, category, type, collected)
#
# Outputs:
#   gptimage_nonenglish_by_context.pdf — bar chart by context
#   gptimage_nonenglish_heatmap.pdf    — heatmap by context × category
#
# Requirements:
#   install.packages("tidyverse")
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

dfopen26 <- read_csv(file.path(DATA_DIR, "openai-revised-apr26-en-fin.csv"),
                     show_col_types = FALSE)
idlist   <- read_csv(file.path(DATA_DIR, "ID_List.csv"), show_col_types = FALSE)

# =============================================================================
# PARSE TYPE → language + country
# =============================================================================

parse_type <- function(df) {
  df %>%
    mutate(
      type_norm = type %>%
        str_replace("English_CleanUp", "English") %>%
        str_replace("Chinese_Simplified", "ChineseS") %>%
        str_replace("Chinese_Traditional", "ChineseT")
    ) %>%
    separate(type_norm, into = c("prompt_language", "country"), sep = "_",
             extra = "merge", fill = "right") %>%
    mutate(
      country      = na_if(str_trim(country), ""),
      lang_country = if_else(is.na(country), prompt_language,
                             paste(prompt_language, country, sep = "_"))
    ) %>%
    select(-any_of("type_norm"))
}

# =============================================================================
# PREPARE DATA
# =============================================================================

df <- dfopen26 %>%
  rename(response_language = language) %>%
  left_join(idlist %>% select(ID, prompt, category, type, collected), by = "ID") %>%
  parse_type() %>%
  filter(!is.na(revised_prompt_en)) %>%
  mutate(is_nonenglish = response_language != "en")


# =============================================================================
# 1. BY LANGUAGE-COUNTRY
# =============================================================================

by_context <- df %>%
  group_by(lang_country) %>%
  summarise(
    n = n(),
    n_nonenglish = sum(is_nonenglish),
    pct_nonenglish = round((n_nonenglish / n) * 100, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(pct_nonenglish))

cat("=== Non-English response rates by language-country ===\n")
print(by_context, n = 50)


# =============================================================================
# 2. BY CATEGORY
# =============================================================================

by_category <- df %>%
  group_by(category) %>%
  summarise(
    n = n(),
    n_nonenglish = sum(is_nonenglish),
    pct_nonenglish = round((n_nonenglish / n) * 100, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(pct_nonenglish))


# =============================================================================
# 3. BY LANGUAGE-COUNTRY × CATEGORY
# =============================================================================

by_both <- df %>%
  group_by(lang_country, category) %>%
  summarise(
    n = n(),
    n_nonenglish = sum(is_nonenglish),
    pct_nonenglish = round((n_nonenglish / n) * 100, 1),
    .groups = "drop"
  )


# =============================================================================
# 4. BAR CHART: by language-country
# =============================================================================

ggplot(by_context %>% filter(pct_nonenglish > 0),
       aes(x = reorder(lang_country, pct_nonenglish), y = pct_nonenglish)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = sprintf("%.0f%%", pct_nonenglish)),
            hjust = -0.2, size = 3) +
  coord_flip() +
  labs(
    title = "GPT Image: Share of revised prompts returned in non-English",
    subtitle = "By language-country combination (contexts with 0% omitted)",
    x = NULL, y = "% non-English responses"
  ) +
  theme_minimal(base_size = 11) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

ggsave(file.path(OUTPUT_DIR, "gptimage_nonenglish_by_context.pdf"),
       width = 10, height = 8)


# =============================================================================
# 5. HEATMAP: language-country × category (Figure 6)
# =============================================================================

if (sd(by_category$pct_nonenglish) > 3) {
  cat("\nCategory variation is notable \u2014 generating heatmap.\n")
  
  ggplot(by_both %>% filter(pct_nonenglish > 0),
         aes(x = lang_country, y = category, fill = pct_nonenglish)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.0f%%", pct_nonenglish)), size = 2.5) +
    scale_fill_gradient(low = "lightyellow", high = "steelblue", name = "% Non-English") +
    labs(
      title = "GPT Image: Non-English revised prompts",
      subtitle = "By language-context \u00d7 category",
      x = "Language / Context", y = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid  = element_blank()
    )
  
  ggsave(file.path(OUTPUT_DIR, "gptimage_nonenglish_heatmap.pdf"),
         width = 16, height = 8)
} else {
  cat("\nCategory variation is small \u2014 bar chart by context is sufficient.\n")
}

cat("\nGPT-Image non-English analysis complete.\n")
