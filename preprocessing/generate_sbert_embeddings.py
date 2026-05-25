"""
Generate SBERT sentence embeddings for the WORLDVIEW pipeline.

Embeds all revised prompts from worldview_prompts.csv using
all-MiniLM-L6-v2 (384-dimensional, English, L2-normalised).

Input:  data/worldview_prompts.csv
Output: data/sbert_embeddings.csv  (row_id + 384 dim columns)

Takes ~5-10 min on CPU for ~60K rows, ~1 min on GPU.

Usage:
    pip install sentence-transformers pandas
    python generate_sbert_embeddings.py
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = "data"
INPUT_PATH  = os.path.join(DATA_DIR, "worldview_prompts.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "sbert_embeddings.csv")

REWRITE_COLUMN = "revised_prompt"
MODEL_NAME     = "all-MiniLM-L6-v2"
BATCH_SIZE     = 256

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    if REWRITE_COLUMN not in df.columns:
        print(f"\nERROR: Column '{REWRITE_COLUMN}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return

    texts = df[REWRITE_COLUMN].fillna("").astype(str).tolist()
    n_empty = sum(1 for t in texts if t.strip() == "")
    print(f"  {len(texts)} rows, {n_empty} empty/missing (embedded as empty string)")

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} prompts (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    dim_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=dim_cols)
    emb_df.insert(0, "row_id", range(len(emb_df)))

    emb_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. Saved to {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
