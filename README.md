# WORLDVIEW: Auditing Cultural Bias in T2I Prompt Revision Layers

Code and data accompanying:

> **Prompt Revision as a Source of Cultural Bias in Commercial Text-to-Image Systems**


## Overview

WORLDVIEW is a benchmark for measuring cultural bias introduced by the **prompt revision layers** in commercial text-to-image (T2I) systems. It contains 8,960 prompts across 15 languages and 31 language–country contexts, with revised prompts collected from DALL-E 3, Imagen, and GPT-Image.

We introduce two metrics:

- **CMS** (Contextual Markedness Score) — semantic distance between a context-specific revised prompt and its culturally unmarked baseline, measuring how much the revision layer "marks" non-default contexts.
- **CFS** (Cultural Flattening Score) — the degree to which a small set of stereotypical terms dominates the revision layer's vocabulary for a given context, combining term *prevalence* and *spread*.

An ablation study using SDXL and Flux 2 Dev (models without built-in revision layers) confirms that the revision layer is causally responsible for the observed stereotyping.

## Repository Structure

```
worldview/
├── README.md
├── requirements.txt
├── data/
│   ├── worldview_prompts.csv            # Core benchmark (8960 × 3 models)
│   ├── sbert_embeddings.csv             # SBERT embeddings (precomputed)
│   ├── clip_image_embeddings.csv        # CLIP image embeddings (precomputed)
│   ├── vqa_descriptions.csv             # VQA descriptions (precomputed)
│   ├── prompts_sdrun.csv                # Ablation prompt subset
│   ├── vqa_descriptions_ablation.csv    # Ablation VQA descriptions
│   ├── openai-revised-apr26-en-fin.csv  # GPT-Image language detection
│   ├── ID_List.csv                      # Original prompt metadata
│   ├── df_unified_ALL.csv               # Unified refusal analysis data
│   └── geo_stopwords.txt                # Geographic/artifact stopword list
├── analysis/
│   ├── R/
│   │   ├── worldview_full_pipeline.R    # Main analysis (§4–5, App F–J)
│   │   ├── worldview_tfidf_qual.R       # Qualitative TF-IDF (§5.3, App H)
│   │   ├── ablation_analysis.R          # Ablation study (§5.4, App K)
│   │   ├── refusal_heatmap.R            # Guardrailing analysis (§5.5, App C)
│   │   └── gptimage_nonenglish_heatmap.R  # Non-English prompts (App D)
│   └── python/
│       ├── worldview_full_pipeline.py   # Python port of main analysis
│       ├── worldview_tfidf_qual.py      # Python port of TF-IDF tables
│       ├── ablation_analysis.py         # Python port of ablation
│       ├── refusal_heatmap.py           # Python port of refusal analysis
│       └── gptimage_nonenglish_heatmap.py  # Python port of language analysis
└── preprocessing/
    ├── generate_sbert_embeddings.py     # SBERT embedding generation
    ├── generate_clip_embeddings.py      # CLIP embedding generation
    ├── generate_vqa_descriptions.py     # VQA descriptions (Ollama)
    ├── generate_vqa_ablation.py         # Ablation VQA descriptions
    └── generate_ablation_images.py      # SDXL/Flux image generation (Replicate)
```

## Quick Start

### Running the analysis (Python)

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4 stopwords

# Main pipeline — produces all CMS, CFS, TF-IDF, and visual analysis
python analysis/python/worldview_full_pipeline.py

# Qualitative TF-IDF tables (runs after main pipeline)
python analysis/python/worldview_tfidf_qual.py

# Ablation analysis
python analysis/python/ablation_analysis.py

# Appendix heatmaps
python analysis/python/refusal_heatmap.py
python analysis/python/gptimage_nonenglish_heatmap.py
```

### Running the analysis (R)

```r
install.packages(c("tidyverse", "textstem", "text2vec", "tidytext",
                    "ggrepel", "stopwords", "patchwork", "scales"))

source("analysis/R/worldview_full_pipeline.R")
source("analysis/R/worldview_tfidf_qual.R")
source("analysis/R/ablation_analysis.R")
source("analysis/R/refusal_heatmap.R")
source("analysis/R/gptimage_nonenglish_heatmap.R")
```

All scripts read from `data/` and write to `output/`. Adjust `DATA_DIR` and `OUTPUT_DIR` at the top of each script if your layout differs.

## Data

### `worldview_prompts.csv` (core benchmark)

| Column | Description |
|--------|-------------|
| `prompt_id` | Unique identifier |
| `original_prompt` | Original English prompt |
| `revised_prompt` | System-revised prompt from the T2I model |
| `model` | `dalle`, `imagen`, or `gptimage` |
| `country` | Target country (empty for baseline) |
| `language` | Prompt language |
| `context` | Language–country pairing |
| `category` | Prompt category (e.g. architecture, food, daily life) |
| `is_baseline` | Whether this is a culturally unmarked baseline prompt |
| `base_prompt_id` | Links context-specific prompts to their baseline |

### Precomputed embeddings and descriptions

The `sbert_embeddings.csv`, `clip_image_embeddings.csv`, and `vqa_descriptions.csv` files are precomputed outputs from the preprocessing scripts. They are included so that the analysis scripts can be run without access to the original images or GPU resources.

### Stopword list

`geo_stopwords.txt` contains geographic names, foreign-language fragments, translation artifacts, and generic evaluative terms that are filtered out during insertion extraction. The list is shared across all analysis scripts and is documented inline with category headers.

## Preprocessing

The preprocessing scripts are provided for full reproducibility but require additional resources:

| Script | Requires |
|--------|----------|
| `generate_sbert_embeddings.py` | CPU or GPU, ~10 min |
| `generate_clip_embeddings.py` | GPU recommended, original images (not released within this repository!) |
| `generate_vqa_descriptions.py` | Local Ollama with Qwen2.5-VL-7B, original images |
| `generate_vqa_ablation.py` | Local Ollama, ablation images |
| `generate_ablation_images.py` | Replicate API token |

## Notes

- The **R and Python analysis scripts produce equivalent outputs** but may differ slightly in numerical values due to differences in lemmatisation (R: textstem/Hunspell vs Python: NLTK/WordNet) and TF-IDF implementations (R: text2vec vs Python: scikit-learn). The main analysis in the paper was done in R, Python ports are released for convenience.
- Images are not included in this release due to size and licensing. The precomputed embeddings and VQA descriptions are sufficient to reproduce all analyses.
- The ablation study uses a fixed random seed (21) for reproducibility.
- We used Claude to add proper comments to the original R scripts used for analysis and to port those analysis scripts into Python. We have checked that no errors were introduced at this stage.



## License

The WORLDVIEW benchmark data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The analysis code is released under the MIT License.
