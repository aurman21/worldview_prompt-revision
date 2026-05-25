"""
Generate CLIP image embeddings for the WORLDVIEW pipeline.

Embeds images from three T2I models using CLIP ViT-B/32.
Only the first image per prompt is used (DALL-E suffix _1-0).

Input:
  data/worldview_prompts.csv         — metadata (prompt_id → country mapping)
  images/dalle/{ID}_1-0.png          — DALL-E 3 images
  images/imagen/{ID}.jpg             — Imagen images
  images/gptimage/{ID}.png           — GPT-Image images

Output:
  data/clip_image_embeddings.csv     — prompt_id, model, country, dim_0…dim_511

Usage:
    pip install torch torchvision open-clip-torch pandas pillow tqdm
    python generate_clip_embeddings.py
"""

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIG — adjust image directories to match your layout
# =============================================================================

DATA_DIR     = "data"
METADATA_PATH = os.path.join(DATA_DIR, "worldview_prompts.csv")
OUTPUT_PATH   = os.path.join(DATA_DIR, "clip_image_embeddings.csv")

IMAGE_CONFIGS = {
    "dalle": {
        "dir": "images/dalle",
        "pattern": "*_1-0.png",
        "id_extractor": lambda fname: fname.rsplit("_1-0", 1)[0],
    },
    "imagen": {
        "dir": "images/imagen",
        "pattern": "*.jpg",
        "id_extractor": lambda fname: os.path.splitext(fname)[0],
    },
    "gptimage": {
        "dir": "images/gptimage",
        "pattern": "*.png",
        "id_extractor": lambda fname: os.path.splitext(fname)[0],
    },
}

CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE      = 64


# =============================================================================
# HELPERS
# =============================================================================

def discover_images(config, model_name):
    """Find image files and extract prompt IDs from filenames."""
    img_dir = config["dir"]
    if not os.path.isdir(img_dir):
        print(f"  WARNING: Directory not found: {img_dir}")
        return []

    files = glob.glob(os.path.join(img_dir, config["pattern"]))
    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            pid = config["id_extractor"](fname)
            records.append({"filepath": fpath, "prompt_id": pid, "model": model_name})
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    return records


def embed_images(filepaths, model, preprocess, device, batch_size=64):
    """Embed images with CLIP. Returns L2-normalised embeddings."""
    import torch

    all_embeddings = []
    for i in tqdm(range(0, len(filepaths), batch_size), desc="Embedding"):
        batch_paths = filepaths[i:i + batch_size]
        images = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"  Error loading {fp}: {e}")
                images.append(preprocess(Image.new("RGB", (224, 224))))

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading CLIP model: {CLIP_MODEL} ({CLIP_PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device).eval()

    # Metadata for prompt_id → country mapping
    print(f"Loading metadata from {METADATA_PATH}...")
    meta = pd.read_csv(METADATA_PATH)
    id_to_country = (
        meta[["prompt_id", "country"]]
        .drop_duplicates(subset=["prompt_id"])
        .set_index("prompt_id")["country"]
        .to_dict()
    )

    # Discover images
    all_records = []
    for model_name, config in IMAGE_CONFIGS.items():
        print(f"\nDiscovering {model_name} in {config['dir']}...")
        records = discover_images(config, model_name)
        print(f"  Found {len(records)} images")
        all_records.extend(records)

    if not all_records:
        print("ERROR: No images found. Check IMAGE_CONFIGS paths.")
        return

    df = pd.DataFrame(all_records)
    print(f"\nTotal images to embed: {len(df)}")

    # Embed
    print(f"\nEmbedding (batch_size={BATCH_SIZE})...")
    embeddings = embed_images(df["filepath"].tolist(), model, preprocess, device, BATCH_SIZE)
    print(f"  Embedding shape: {embeddings.shape}")

    # Build output
    dim_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=dim_cols)
    emb_df["prompt_id"] = df["prompt_id"].values
    emb_df["model"] = df["model"].values
    emb_df["country"] = emb_df["prompt_id"].map(id_to_country)
    emb_df = emb_df[["prompt_id", "model", "country"] + dim_cols]

    emb_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")

    # Summary
    print("\n=== Summary ===")
    for m in df["model"].unique():
        n = (emb_df["model"] == m).sum()
        n_bl = ((emb_df["model"] == m) & emb_df["country"].isna()).sum()
        print(f"  {m}: {n} images ({n_bl} baseline, {n - n_bl} country-specified)")


if __name__ == "__main__":
    main()
