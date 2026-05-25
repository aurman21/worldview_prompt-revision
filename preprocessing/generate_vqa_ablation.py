"""
Generate VQA image descriptions for ablation study images.

Uses Ollama (local) with Qwen2.5-VL-7B to describe images generated
by SDXL and Flux 2 Dev from both original and revised prompts.

Input:
  data/prompts_sdrun.csv                              — prompt metadata
  images/ablation/sdxl/original/{prompt_id}_o.png
  images/ablation/sdxl/revised/{prompt_id}_r.png
  images/ablation/flux2dev/original/{prompt_id}_o.png
  images/ablation/flux2dev/revised/{prompt_id}_r.png

Output:
  data/vqa_descriptions_ablation.csv  — prompt_id, prompt_type, prompt, model, description

Setup:
    ollama pull qwen2.5vl:7b
    pip install requests pandas tqdm

Usage:
    python generate_vqa_ablation.py
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

DATA_DIR     = "data"
ABLATION_DIR = "images/ablation"
PROMPTS_CSV  = os.path.join(DATA_DIR, "prompts_sdrun.csv")
OUTPUT_CSV   = os.path.join(DATA_DIR, "vqa_descriptions_ablation.csv")

IMAGE_CONFIGS = [
    {"model": "stablediffusion", "folder": "sdxl",     "prompt_types": ["original", "revised"]},
    {"model": "flux",            "folder": "flux2dev",  "prompt_types": ["original", "revised"]},
]

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "qwen2.5vl:7b"
PROMPT      = "Describe this image in detail"
FLUSH_EVERY = 50
NUM_WORKERS = 1


# =============================================================================
# IMAGE DISCOVERY
# =============================================================================

def load_prompts(csv_path):
    """Load prompt texts keyed by (prompt_id, prompt_type)."""
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["prompt_id"] = df["prompt_id"].astype(str)

    prompt_map = {}
    for _, row in df.iterrows():
        pid = row["prompt_id"]
        prompt_map[(pid, "original")] = row["original_prompt"]
        revised = row.get("revised_prompt", "")
        if pd.notna(revised) and str(revised).strip() and str(revised).strip() != "NA":
            prompt_map[(pid, "revised")] = str(revised)
    return prompt_map


def discover_images(prompt_map):
    """Find all ablation images and pair with prompt text."""
    records = []
    for cfg in IMAGE_CONFIGS:
        model_name = cfg["model"]
        folder = cfg["folder"]
        for ptype in cfg["prompt_types"]:
            suffix = "_o.png" if ptype == "original" else "_r.png"
            img_dir = os.path.join(ABLATION_DIR, folder, ptype)
            if not os.path.isdir(img_dir):
                print(f"  WARNING: {img_dir} not found, skipping")
                continue
            files = glob.glob(os.path.join(img_dir, f"*{suffix}"))
            for fpath in files:
                fname = os.path.basename(fpath)
                pid = fname.replace(suffix, "")
                key = (pid, ptype)
                if key not in prompt_map:
                    continue
                records.append({
                    "filepath": fpath,
                    "prompt_id": pid,
                    "prompt_type": ptype,
                    "prompt": prompt_map[key],
                    "model": model_name,
                })
        print(f"  {model_name}: {sum(1 for r in records if r['model'] == model_name)} images")
    return records


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
    print(f"Loading prompts from {PROMPTS_CSV}...")
    prompt_map = load_prompts(PROMPTS_CSV)
    print(f"  {len(prompt_map)} prompt entries loaded")

    print(f"\nDiscovering images in {ABLATION_DIR}/...")
    all_records = discover_images(prompt_map)
    print(f"Total images found: {len(all_records)}")

    # Resume
    done_keys = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        existing["prompt_id"] = existing["prompt_id"].astype(str)
        done_keys = set(zip(existing["prompt_id"], existing["prompt_type"], existing["model"]))
        print(f"Resuming: {len(done_keys)} already done")

    todo = [r for r in all_records
            if (r["prompt_id"], r["prompt_type"], r["model"]) not in done_keys]
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
                "prompt_type": rec["prompt_type"],
                "prompt": rec["prompt"],
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
                    "prompt_type": rec["prompt_type"],
                    "prompt": rec["prompt"],
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
