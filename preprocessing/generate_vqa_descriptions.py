"""
Generate VQA image descriptions for visual-level bias analysis.

Uses Ollama (local) with Qwen2.5-VL-7B to describe generated images.
Filters to prompt IDs present in worldview_prompts.csv.
Supports resumption from existing output CSV.

Input:
  data/worldview_prompts.csv              — metadata filter
  images/dalle/{ID}_1-0.png               — DALL-E 3 images
  images/imagen/{ID}.jpg                  — Imagen images
  images/gptimage/{ID}.png                — GPT-Image images

Output:
  data/vqa_descriptions.csv               — prompt_id, model, description

Setup:
    ollama pull qwen2.5vl:7b
    pip install requests pandas tqdm

Usage:
    python generate_vqa_descriptions.py
"""

import os
import glob
import base64
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR      = "data"
METADATA_PATH = os.path.join(DATA_DIR, "worldview_prompts.csv")
OUTPUT_CSV    = os.path.join(DATA_DIR, "vqa_descriptions.csv")

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

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "qwen2.5vl:7b"
PROMPT      = "Describe this image in detail"
FLUSH_EVERY = 50
NUM_WORKERS = 1  # increase to 2-3 if GPU handles it


# =============================================================================
# IMAGE DISCOVERY
# =============================================================================

def discover_images(config, model_name):
    img_dir = config["dir"]
    if not os.path.isdir(img_dir):
        print(f"  WARNING: {img_dir} not found")
        return []

    files = glob.glob(os.path.join(img_dir, config["pattern"]))
    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            pid = config["id_extractor"](fname)
            records.append({"filepath": fpath, "prompt_id": pid, "model": model_name})
        except Exception:
            pass
    return records


def filter_to_metadata(records, meta_path):
    meta = pd.read_csv(meta_path)
    # Drop unnamed index columns
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]
    meta["prompt_id"] = meta["prompt_id"].astype(str)
    valid = set(zip(meta["prompt_id"], meta["model"]))
    filtered = [r for r in records if (r["prompt_id"], r["model"]) in valid]
    print(f"  Filtered: {len(records)} -> {len(filtered)}")
    return filtered


# =============================================================================
# VQA
# =============================================================================

def describe_one(filepath):
    with open(filepath, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 256},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        else:
            print(f"  Status {resp.status_code}: {filepath}")
            return None
    except Exception as e:
        print(f"  Error {filepath}: {e}")
        return None


def flush_batch(batch, output_csv):
    if not batch:
        return
    new_df = pd.DataFrame(batch)
    if os.path.exists(output_csv):
        new_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        new_df.to_csv(output_csv, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Discover images
    all_records = []
    for model_name, config in IMAGE_CONFIGS.items():
        print(f"Discovering {model_name} in {config['dir']}...")
        recs = discover_images(config, model_name)
        print(f"  Found {len(recs)}")
        all_records.extend(recs)

    # Filter to metadata
    print(f"\nFiltering to {METADATA_PATH}...")
    all_records = filter_to_metadata(all_records, METADATA_PATH)
    print(f"Total to annotate: {len(all_records)}")

    # Resume from existing
    done_keys = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        existing["prompt_id"] = existing["prompt_id"].astype(str)
        done_keys = set(zip(existing["prompt_id"], existing["model"]))
        print(f"Resuming: {len(done_keys)} already done")

    todo = [r for r in all_records if (r["prompt_id"], r["model"]) not in done_keys]
    print(f"Remaining: {len(todo)}")

    if not todo:
        print("All done!")
        return

    batch = []
    total_new = 0

    if NUM_WORKERS == 1:
        for rec in tqdm(todo, desc="Annotating"):
            desc = describe_one(rec["filepath"])
            batch.append({
                "prompt_id": rec["prompt_id"],
                "model": rec["model"],
                "description": desc,
            })
            if len(batch) >= FLUSH_EVERY:
                flush_batch(batch, OUTPUT_CSV)
                total_new += len(batch)
                batch = []
    else:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {pool.submit(describe_one, rec["filepath"]): rec for rec in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
                rec = futures[future]
                try:
                    desc = future.result()
                except Exception as e:
                    print(f"  Failed {rec['filepath']}: {e}")
                    desc = None
                batch.append({
                    "prompt_id": rec["prompt_id"],
                    "model": rec["model"],
                    "description": desc,
                })
                if len(batch) >= FLUSH_EVERY:
                    flush_batch(batch, OUTPUT_CSV)
                    total_new += len(batch)
                    batch = []

    if batch:
        flush_batch(batch, OUTPUT_CSV)
        total_new += len(batch)
    print(f"\nDone! {total_new} new descriptions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
